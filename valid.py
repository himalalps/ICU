from Datasets_valid import Valid_Dataset

import re
import os
import string
import logging
import argparse
import random
import numpy as np
import pandas as pd

from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoForCausalLM,
    OPTForCausalLM,
    GPT2LMHeadModel,
)


from utils import (
    DIALOG_DATASETS,
    CLASSIFICATION_DATASETS,
    PPL_DATASETS,
    COMPLETION_DATASETS,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def valid(valid_data, model, tokenizer, args):
    result_dict = {}
    for data in valid_data:
        result = valid_step(data[0], data[1], data[2], model, tokenizer, args)
        result_dict[data[0] + data[1]] = result
    result_list = [result_dict]
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(f"{args.dir}/valid_result.csv")


def valid_step(dataset_name, valid_subset_path, type_path, model, tokenizer, args):
    dataset, dataloader = get_loader(
        dataset_name, valid_subset_path, type_path, tokenizer, args
    )
    if valid_subset_path:
        task = f"{dataset_name}_{valid_subset_path}"
    else:
        task = dataset_name
    logging.info("{} {}".format(task, len(dataloader)))
    if any(name in dataset_name for name in COMPLETION_DATASETS):
        return lambada_evaluation(dataloader, args.prefix_length, task, model)
    elif any(name in dataset_name for name in CLASSIFICATION_DATASETS):
        return classification_verbalizer(
            dataloader, args.prefix_length, task, model, args
        )
    elif any(name in dataset_name for name in PPL_DATASETS):
        return validation_ppl(dataset, task, model, tokenizer, args)
    elif any(name in dataset_name for name in DIALOG_DATASETS):
        return dialog_evaluation(dataloader, args.prefix_length, task, model, tokenizer)
    else:
        raise Exception("dataset_name not supported")


def get_loader(dataset_name, valid_subset_path, type_path, tokenizer, args):
    valid_dataset = Valid_Dataset(
        dataset_name,
        valid_subset_path,
        type_path,
        tokenizer,
        args.prefix_length,
        args.suffix_length,
        args.cache
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return valid_dataset, valid_dataloader


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def rid_of_specials(text):
        text = text.replace("<extra_id_0>", "")
        text = text.replace("<extra_id_1>", "")
        return text

    return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))


def get_rid_of_pad(tokens):
    while tokens[-1] == -100 or tokens[-1] == tokenizer.pad_token_id:
        tokens.pop()
    return tokens


def ids_to_clean_text(generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return list(map(str.strip, gen_text))


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def classification_verbalizer(dataloader, padding_length, task, model, args):
    total_acc = 0
    len_data = 0

    for batch_idx, batch in enumerate(dataloader):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"]
        choices = batch["choices"]
        answer_index = batch["answer_index"]
        cur_batch_size = len(source_ids)
        len_data += cur_batch_size

        answer_idx = [-1] * cur_batch_size
        for i in range(cur_batch_size):
            answer_idx[i] = answer_index[i]

        inps = []
        cont_toks_list = []
        inplens = []

        answers = torch.zeros(cur_batch_size, len(choices), device=device)

        for c_idx in range(len(choices)):
            choice_ids = tokenizer.batch_encode_plus(
                list(choices[c_idx]),
                max_length=args.prefix_length,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].tolist()
            for i in range(cur_batch_size):
                context_enc = get_rid_of_pad(source_ids[i])
                continuation_enc = get_rid_of_pad(choice_ids[i])

                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= max_length

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(padding_length):][:-1],
                    dtype=torch.long,
                ).to(device)
                (inplen,) = inp.shape
                cont = continuation_enc

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        # [padding_length - seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        )
                        + tokenizer.pad_token_id,
                    ],
                    dim=0,
                )
                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
            multi_logits = F.log_softmax(
                model(batched_inps)[0][:, :, :], dim=-1
            )  # [batch, padding_length, vocab]

            cnt = 0

            for logits, inp, inplen, cont_toks in zip(
                multi_logits, inps, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                original_logits = logits

                # [1, seq, vocab]
                logits = logits[inplen - contlen : inplen].unsqueeze(0)

                # Check if per-token argmax is exactly equal to continuation
                cont_toks = (
                    torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0).to(device)
                )  # [1, seq]
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                loss = -float(logits.sum())
                answers[cnt][c_idx] = loss
                cnt += 1

            inps = []
            cont_toks_list = []
            inplens = []

        answer_idx = torch.Tensor(answer_idx).to(device)
        answers = torch.argmin(answers, dim=1)
        total_acc += int(torch.where(answers == answer_idx, 1, 0).sum())

    acc_avg = total_acc / len_data

    logging.info("{}/acc {}".format(task, acc_avg))
    return acc_avg


def lambada_evaluation(dataloader, padding_length, task, model):
    total_loss = 0
    total_acc = 0
    total_f1 = 0
    len_data = 0
    for batch_idx, batch in enumerate(dataloader):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"].tolist()
        cur_batch_size = len(source_ids)
        len_data += cur_batch_size

        inps = []
        cont_toks_list = []
        inplens = []
        for i in range(cur_batch_size):
            if source_ids[i] == target_ids[i]:
                context_enc = source_ids[i][: padding_length - 10]
                continuation_enc = target_ids[i][padding_length - 10 :]
            else:
                context_enc = get_rid_of_pad(source_ids[i])
                continuation_enc = get_rid_of_pad(target_ids[i])

            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= max_length

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(padding_length):][:-1],
                dtype=torch.long,
            ).to(device)
            (inplen,) = inp.shape
            cont = continuation_enc

            # pad length from seq to padding_length
            inp = torch.cat(
                [
                    inp,  # [seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(
                        inp.device
                    ),  # [padding_length - seq]
                ],
                dim=0,
            )
            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
        multi_logits = F.log_softmax(
            model(batched_inps)[0][:, :, :], dim=-1
        ).cpu()  # [batch, padding_length, vocab]
        for logits, inp, inplen, cont_toks in zip(
            multi_logits, inps, inplens, cont_toks_list
        ):
            # Slice to original seq length
            contlen = len(cont_toks)
            original_logits = logits
            # [1, seq, vocab]
            logits = logits[inplen - contlen : inplen].unsqueeze(0)
            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                0
            )  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            predicted = ids_to_clean_text(greedy_tokens)
            ground_truth = ids_to_clean_text(cont_toks)
            em = exact_match_score(predicted[0], ground_truth[0])
            f1 = _f1_score(predicted[0], ground_truth[0])

            logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                -1
            )  # [1, seq]
            # Answer: (log prob, is-exact-match)
            loss = -float(logits.sum())
            if bool(max_equal) or em == 1:
                total_acc += 1

            total_loss += loss
            total_f1 += f1

    total_loss_avg = total_loss / len_data
    total_acc_avg = total_acc / len_data
    total_f1_avg = total_f1 / len_data
    logging.info("{}/loss {}".format(task, total_loss_avg))
    logging.info("{}/acc {}".format(task, total_acc_avg))
    logging.info("{}/f1 {}".format(task, total_f1_avg))
    return total_acc_avg


def dialog_evaluation(dataloader, padding_length, task, model, tokenizer):
    total_loss = 0
    total_f1 = 0
    len_data = 0
    for batch_idx, batch in enumerate(dataloader):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"].tolist()
        cur_batch_size = len(source_ids)
        len_data += cur_batch_size

        inps, cont_toks_list, inplens = [], [], []
        for i in range(cur_batch_size):
            context_enc = get_rid_of_pad(source_ids[i])
            continuation_enc = get_rid_of_pad(target_ids[i])

            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= max_length

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(padding_length):], dtype=torch.long
            ).to(device)
            (inplen,) = inp.shape
            cont = continuation_enc

            # pad length from seq to padding_length
            inp = torch.cat(
                [
                    inp,  # [seq]
                    # [padding_length - seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(
                        inp.device
                    )
                    + tokenizer.pad_token_id,
                ],
                dim=0,
            )
        inps.append(inp.unsqueeze(0))  # [1, padding_length]
        cont_toks_list.append(cont)
        inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
        multi_logits = model(batched_inps)[0][:, :, :]  # [batch, padding_length, vocab]

        full_logits, full_cont_toks = [], []
        for logits, inp, inplen, cont_toks in zip(
            multi_logits, inps, inplens, cont_toks_list
        ):

            # Slice to original seq length
            contlen = len(cont_toks)

            if contlen >= padding_length:
                cont_toks = cont_toks[: int(padding_length / 2)]
                contlen = len(cont_toks)

            # [seq, vocab]
            logits = logits[inplen - contlen - 1 : inplen - 1]
            # Check if per-token argmax is exactly equal to continuation
            cont_toks = torch.tensor(cont_toks, dtype=torch.long).to(device)  # [seq]

            assert logits.shape[0] == cont_toks.shape[0]

            full_logits.append(logits)
            full_cont_toks.append(cont_toks)

        full_logits = torch.cat(full_logits)
        full_cont_toks = torch.cat(full_cont_toks)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(full_logits, full_cont_toks)
        total_loss += loss

        generate_input = []
        for source_id in source_ids:
            inplen = len(source_id)
            inp = torch.tensor(source_id, dtype=torch.long).to(device)
            inp = torch.cat(
                [
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(
                        inp.device
                    )
                    + tokenizer.pad_token_id,
                    inp,
                ],
                dim=0,
            )
            generate_input.append(inp.unsqueeze(0))  # [1, padding_length]

        inputs = torch.cat(generate_input, dim=0)
        attention_masks = inputs.ne(tokenizer.pad_token_id).long()
        generated_ids = model.generate(
            inputs,
            attention_mask=attention_masks,
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
        )[:, padding_length:]
        generated_text = tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
        generated_text = [t.split("\nUser ")[0] for t in generated_text]
        target_text = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        # # Debugging
        # source_text = tokenizer.batch_decode(source_ids, skip_special_tokens=True)
        # for s, g, t in zip(source_text, generated_text, target_text):
        #     logging.debug("---------------------")
        #     logging.debug(f"[Prefix] {s}")
        #     logging.debug(f"[Ground Truth] {t}")
        #     logging.debug(f"[Generated] {g}")
        #     logging.debug("---------------------")

        for g, t in zip(generated_text, target_text):
            total_f1 += _f1_score(g, t)

    total_loss_avg = total_loss / len_data
    total_f1_avg = total_f1 / len_data

    logging.info("{}/loss {}".format(task, total_loss_avg))

    logging.info("{}/f1 {}".format(task, total_f1_avg))

    return total_f1_avg


def validation_ppl(dataset, task, model, tokenizer, args):
    dataset_df = dataset.dataset
    encoding = tokenizer("\n\n".join(dataset_df["text"]), return_tensors="pt")

    seq_len = encoding.input_ids.size(1)

    log_prob = []
    for begin_loc in range(0, seq_len, args.suffix_length):
        end_loc = min(begin_loc + args.suffix_length, seq_len)

        input_ids = encoding.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        outputs = model(input_ids, labels=target_ids)

        log_prob.append(outputs[0])

        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(log_prob).mean())
    logger.info("{}/loss {}".format(task, ppl))
    return ppl.item()


if __name__ == "__main__":
    seed = 42
    seed_everything(seed)

    valid_data = [
        ["datasets/lambada.csv", "", "test"],
        ["datasets/hellaswag", "", "validation"],
        ["datasets/winogrande", "winogrande_s", "validation"],
        ["datasets/super_glue", "copa", "validation"],
        ["datasets/ai2_arc", "ARC-Easy", "validation"],
        ["datasets/ai2_arc", "ARC-Challenge", "validation"],
        ["datasets/piqa", "", "validation"],
        ["datasets/math_qa", "", "validation"],
        ["datasets/pubmed_qa.csv", "", ""],
        ["datasets/validation_data/wizard_of_wikipedia.json", "", ""],
        ["datasets/validation_data/empathetic_dialogues.json", "", ""],
        ["datasets/validation_data/blended_skill_talk.json", "", ""],
        ["datasets/validation_data/wizard_of_internet.json", "", ""],
        ["datasets/validation_data/pile.csv", "", ""],
        ["datasets/validation_data/wikitext.csv", "", ""],
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-neo-125m",
        help="model name or path",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="EleutherAI/gpt-neo-125m",
        help="tokenizer name or path",
    )

    parser.add_argument(
        "--prefix_length", type=int, default=512, help="prefix length of input"
    )

    parser.add_argument(
        "--suffix_length", type=int, default=512, help="suffix length of input"
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="pytorch device")

    parser.add_argument("--batch_size", type=int, default=32, help="train batch size")

    parser.add_argument("--num_workers", type=int, default=48, help="train num workers")

    parser.add_argument(
        "--dir", type=str, default="result", help="directory to store the results"
    )

    parser.add_argument(
        "--cache", type=str, default="./cache", help="dataset cache directory"
    )

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=f"{args.dir}/valid.log",
        filemode="w",
    )

    logger = logging.getLogger()

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    tokenizer.pad_token = tokenizer.eos_token
    
    # Different models have different kwargs
    if "gpt-neo" in args.model_name:
        model: GPTNeoForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            resid_dropout=0,
            embed_dropout=0,
            attention_dropout=0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elif "opt" in args.model_name:
        model: OPTForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.model_name, dropout=0, attention_dropout=0, activation_dropout=0
        )
    else:  # GPT2
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            pad_token_id=tokenizer.eos_token_id,
        )
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    try:
        max_length = model.config.n_ctx
    except AttributeError:
        max_length = model.config.max_position_embeddings

    model.eval()
    with torch.no_grad():
        valid(valid_data, model, tokenizer, args)
