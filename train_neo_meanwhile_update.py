from utils import batch_compute_bert_score
from Datasets_mean import Custom_Dataset


import os
import logging
import random
import argparse
import numpy as np
import pandas as pd


import torch

from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import (
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoForCausalLM,
    OPTForCausalLM,
    GPT2LMHeadModel,
)
from torchmetrics.functional import accuracy
import nltk
from nltk.translate.bleu_score import sentence_bleu

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loader(
    unlearn_data_path, learn_data_path, gpt2tokenizer, tokenizer, model, device, args
):
    dataset = Custom_Dataset(
        unlearn_data_path,
        learn_data_path,
        gpt2tokenizer,
        tokenizer,
        model,
        device,
        args.prefix_length,
        args.suffix_length,
        args.batch_size,
    )
    candidate = [i for i in range(len(dataset))]

    train_dataloader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(candidate),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    valid_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    return train_dataloader, valid_dataloader, dataset, candidate


def predict(message, tokenizer, model, device):
    model_inputs = tokenizer.encode(message, return_tensors="pt").to(device)
    model_outputs = model.generate(
        model_inputs,
        max_new_tokens=100,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    model_outputs = model_outputs[0, len(model_inputs[0]) :]
    model_output_text = tokenizer.decode(model_outputs, skip_special_tokens=True)
    return model_output_text


def val(
    epoch,
    model,
    val_loader,
    dataset,
    gpt2tokenizer,
    tokenizer,
    valid_result,
    scores,
    el_n,
    device,
    args,
):
    logger.info("Validating epoch {}".format(epoch))
    with torch.no_grad():
        unlearn_cnt = val_score(
            epoch,
            model,
            val_loader,
            dataset,
            gpt2tokenizer,
            tokenizer,
            scores,
            device,
            args,
        )
        val_loss = validation_forget(epoch, model, val_loader, tokenizer, device)
        acc = validation_ma(epoch, model, val_loader, tokenizer, device, args)
        els = validation_el(epoch, model, val_loader, tokenizer, device, el_n, args)
        valid_dict = {
            "epoch": epoch,
            "val_loss": val_loss,
            "acc": acc,
        }
        for n, el in zip(el_n, els):
            valid_dict[f"el_{n}"] = el
        valid_result.append(valid_dict)
        if acc < args.acc and any([el < args.el for el in els]):
            logger.info("Epoch {} should stop, acc {}, el {}".format(epoch, acc, els))
            return True
        if unlearn_cnt < 1:
            logger.info("Epoch {} should stop, left {}".format(epoch, unlearn_cnt))
            return True

        torch.cuda.empty_cache()
        return False


def val_score(
    epoch, model, val_loader, dataset, gpt2tokenizer, tokenizer, scores, device, args
):
    def entropy(tokens):
        d = {}
        p = 0
        for token in tokens:
            try:
                d[token] += 1
            except:
                d[token] = 1
        for token in d:
            p -= d[token] / len(tokens) * np.log2(d[token] / len(tokens))
        return p

    data = []
    model.eval()
    unlearn_cnt = 0
    for batch_idx, batch in enumerate(val_loader):
        reference_list = gpt2tokenizer.batch_decode(batch["unlearn_gpt2_suffix"])

        message = gpt2tokenizer.batch_decode(batch["unlearn_gpt2_prefix"])
        source = tokenizer(
            message,
            max_length=50,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = source["input_ids"].to(device)
        candidate_list = model.generate(
            input_ids,
            max_new_tokens=50,
            num_beams=2,
            pad_token_id=tokenizer.eos_token_id,
        )
        candidate_list = [candidate[50:] for candidate in candidate_list]
        diff = [
            len(set(candidate.tolist())) / len(candidate)
            for candidate in candidate_list
        ]
        ens = [entropy(candidate.tolist()) for candidate in candidate_list]
        candidate_list = tokenizer.batch_decode(candidate_list)

        P, R, F1 = batch_compute_bert_score(
            candidate_list, reference_list, args.bert_name, device
        )

        unlearn_flag = []
        learn_flag = []
        for i in range(len(P)):
            reference = nltk.word_tokenize(reference_list[i].lower())
            candidate = nltk.word_tokenize(candidate_list[i].lower())
            bleu_score = sentence_bleu([reference], candidate)
            if F1[i].item() < args.f1 or bleu_score < args.bleu:
                unlearn_flag.append(0)
                learn_flag.append(0)
            else:
                unlearn_flag.append(1)
                learn_flag.append(1)
            data.append(
                {
                    "id": batch["id"][i].item(),
                    "diff": diff[i],
                    "entropy": ens[i],
                    "bleu": bleu_score,
                    "P": P[i].item(),
                    "R": R[i].item(),
                    "F1": F1[i].item(),
                    "candidate": candidate_list[i],
                    "reference": reference_list[i],
                }
            )
            if batch_idx % 50 == 0:
                logger.debug("candidate {}".format(candidate_list[i]))
                logger.debug("reference {}".format(reference_list[i]))
                logger.debug(
                    "id {} diff {} entropy {} bleu {} P {} R {} F1 {}".format(
                        batch["id"][i].item(),
                        diff[i],
                        ens[i],
                        bleu_score,
                        P[i].item(),
                        R[i].item(),
                        F1[i].item(),
                    )
                )
        if batch_idx % 50 == 0:
            logger.debug(" validating.. {}/{}".format(batch_idx, len(val_loader)))

        dataset.update(batch["id"], unlearn_flag, learn_flag)
        unlearn_cnt += unlearn_flag.count(1)
    logging.debug(unlearn_cnt)

    df = pd.DataFrame(data)
    df.sort_values(by="id", ascending=True, inplace=True)
    df.to_csv(f"{args.dir}/epoch{epoch}.csv", index=False)

    scores.append(
        {
            "epoch": epoch,
            "diff": df["diff"].mean(),
            "entropy": df["entropy"].mean(),
            "bleu": df["bleu"].mean(),
            "P": df["P"].mean(),
            "R": df["R"].mean(),
            "F1": df["F1"].mean(),
        }
    )

    logger.debug(predict("Who is Harry Potter?", tokenizer, model, device))

    logger.info("diff {}".format(df["diff"].mean()))
    logger.info("entropy {}".format(df["entropy"].mean()))
    logger.info("bleu {}".format(df["bleu"].mean()))
    logger.info(
        "bert_score [epoch {}] P {} R {} F1 {}".format(
            epoch, df["P"].mean(), df["R"].mean(), df["F1"].mean()
        )
    )

    return unlearn_cnt


def validation_forget(epoch, model, val_loader, tokenizer, device):
    model.eval()
    epoch_loss = 0
    for batch_idx, batch in enumerate(val_loader):
        input_ids = batch["unlearn_prefix_ids"].to(device)
        attention_mask = batch["unlearn_prefix_mask"].to(device)
        target_ids = batch["unlearn_suffix_ids"]
        target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
        target_ids = target_ids.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        epoch_loss += outputs[0].item()

    logger.info("val_loss [epoch {}] {}".format(epoch, epoch_loss / len(val_loader)))
    return epoch_loss / len(val_loader)


def validation_ma(epoch, model, val_loader, tokenizer, device, args):
    model.eval()
    epoch_acc = 0

    for batch_idx, batch in enumerate(val_loader):
        input_ids = batch["unlearn_prefix_ids"].to(device)

        labels, preds = [], []
        for i in range(1, args.target_length):
            label = input_ids[..., i]
            prompt = input_ids[..., :i]

            try:
                pred = model.generate(
                    prompt, max_length=i + 1, pad_token_id=tokenizer.eos_token_id
                )[:, -1]
            except:
                pred = model.generate(
                    torch.squeeze(prompt),
                    max_length=i + 1,
                    pad_token_id=tokenizer.eos_token_id,
                ).squeeze()[-1]
            labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))

        preds = torch.stack(preds)
        labels = torch.stack(labels)

        try:
            score = accuracy(
                preds,
                labels,
                ignore_index=-100,
            )
        except:
            score = accuracy(
                preds,
                labels,
                task="multiclass",
                num_classes=tokenizer.vocab_size,
                ignore_index=-100,
            )
        epoch_acc += score.item()

    logger.info("acc [epoch {}] {}".format(epoch, epoch_acc / len(val_loader)))
    return epoch_acc / len(val_loader)


def validation_el(epoch, model, val_loader, tokenizer, device, el_n, args):
    def ngram_of_1D_tensor(X, n):
        grams = [tuple(X[i : i + n].tolist()) for i in range(X.shape[0] - n + 1)]
        return grams

    model.eval()
    el_score = {n: [] for n in el_n}

    for batch_idx, batch in enumerate(val_loader):
        input_ids = batch["unlearn_prefix_ids"].to(device)

        cur_batch_size = input_ids.shape[0]
        numerator = {n: [0] * cur_batch_size for n in el_n}
        for i in reversed(range(150, args.target_length)):
            label = input_ids[..., i : args.target_length]
            prompt = input_ids[..., :i]
            pred = model.generate(
                prompt,
                max_length=args.target_length,
                pad_token_id=tokenizer.eos_token_id,
            )[..., i:]

            for example_idx in range(cur_batch_size):
                p, l = pred[example_idx], label[example_idx]
                for n in el_n:
                    p_ngram = ngram_of_1D_tensor(p, n)
                    l_ngram = ngram_of_1D_tensor(l, n)
                    l_unique = set(l_ngram)
                    p_tp = [i for i in p_ngram if i in l_unique]

                    try:
                        p_acc = len(p_tp) / len(l_ngram)
                        numerator[n][example_idx] += p_acc
                    except ZeroDivisionError:
                        pass

        for n in el_n:
            el_score[n] += [
                s / (args.target_length - 150 - (n - 1)) for s in numerator[n]
            ]

    els = []
    for n in el_n:
        logger.info(
            "el_{}-gram [epoch {}] {}".format(
                n, epoch, sum(el_score[n]) / len(el_score[n])
            )
        )
        els.append(sum(el_score[n]) / len(el_score[n]))
    return els


def main(args):
    unlearn_data_path = f"./datasets/exp/{args.exp}/unlearn/_dataset.npy"
    learn_data_path = f"./datasets/exp/{args.exp}/learn/_dataset.npy"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    el_n = [10]
    valid_result = []
    scores = []

    gpt2tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_name)
    gpt2tokenizer.padding_side = "left"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.padding_side = "left"
    if "gpt" in args.tokenizer_name:
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
        model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            pad_token_id=tokenizer.eos_token_id,
        )

    model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    unlearn_loader, val_loader, dataset, candidate = get_loader(
        unlearn_data_path,
        learn_data_path,
        gpt2tokenizer,
        tokenizer,
        model,
        device,
        args,
    )

    epoch = 0
    val(
        0,
        model,
        val_loader,
        dataset,
        gpt2tokenizer,
        tokenizer,
        valid_result,
        scores,
        el_n,
        device,
        args,
    )
    while True:
        torch.cuda.empty_cache()
        model.train()

        logger.info("Epoch {}: training".format(epoch + 1))

        for batch_idx, batch in enumerate(unlearn_loader):
            optimizer.zero_grad()

            input_ids = batch["unlearn_prefix_ids"][batch["unlearn_flag"] != 0].to(
                device
            )
            attention_mask = batch["unlearn_prefix_mask"][
                batch["unlearn_flag"] != 0
            ].to(device)
            target_ids = batch["unlearn_suffix_ids"][batch["unlearn_flag"] != 0]
            target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
            target_ids = target_ids.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            unlearn_loss = outputs[0]

            input_ids = batch["learn_prefix_ids"][batch["learn_flag"] != 0].to(device)
            attention_mask = batch["learn_prefix_mask"][batch["learn_flag"] != 0].to(
                device
            )
            target_ids = batch["learn_suffix_ids"][batch["learn_flag"] != 0]
            target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
            target_ids = target_ids.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            learn_loss = outputs[0]

            prob_p = batch["learn_prob"][batch["learn_flag"] != 0].to(device)
            prob_q = torch.nn.functional.softmax(outputs.logits, -1)

            kl_loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

            loss = -args.uw * unlearn_loss + args.lw * learn_loss + args.kl * kl_loss

            loss.backward()
            optimizer.step()

            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tunlearn_loss: {:.6f}\tlearn_loss: {:.6f}\tkl_loss: {:.6f}".format(
                    epoch + 1,
                    batch_idx,
                    len(unlearn_loader),
                    100.0 * batch_idx / len(unlearn_loader),
                    unlearn_loss.item(),
                    learn_loss.item(),
                    kl_loss.item(),
                )
            )

        if val(
            epoch + 1,
            model,
            val_loader,
            dataset,
            gpt2tokenizer,
            tokenizer,
            valid_result,
            scores,
            el_n,
            device,
            args,
        ):
            break
        epoch += 1

    valid_df = pd.DataFrame(valid_result)
    valid_df.to_csv(f"{args.dir}/valid.csv", index=False)
    bert_df = pd.DataFrame(scores)
    bert_df.to_csv(f"{args.dir}/score.csv", index=False)
    model.save_pretrained(
        f"{args.dir}/{args.model_name}_{args.exp}_lr{args.lr}_uw{args.uw}_lw{args.lw}_kl{args.kl}_epoch{epoch+1}_updateboth"
    )


if __name__ == "__main__":
    seed = 42
    seed_everything(seed)

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", type=str, default="exp0", help="path to train data")

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
        "--gpt2_name",
        type=str,
        default="openai-community/gpt2",
        help="gpt2 name or path",
    )

    parser.add_argument(
        "--bert_name",
        type=str,
        default="google-bert/bert-base-uncased",
        help="bert model name or path",
    )

    parser.add_argument(
        "--prefix_length", type=int, default=200, help="prefix length of input"
    )

    parser.add_argument(
        "--suffix_length", type=int, default=200, help="suffix length of input"
    )

    parser.add_argument(
        "--target_length", type=int, default=200, help="length of target sequence"
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="pytorch device")

    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")

    parser.add_argument("--num_workers", type=int, default=8, help="train num workers")

    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate")

    parser.add_argument(
        "--uw", type=float, default=1.0, help="weight of unlearning loss"
    )

    parser.add_argument("--lw", type=float, default=0.5, help="weight of learning loss")

    parser.add_argument("--kl", type=float, default=1.0, help="weight of kl loss")

    parser.add_argument("--f1", type=float, default=0.3, help="f1 threshold")

    parser.add_argument("--bleu", type=float, default=0.01, help="bleu threshold")

    parser.add_argument("--acc", type=float, default=0.5994, help="ma threshold")

    parser.add_argument("--el", type=float, default=0.0499, help="el threshold")

    parser.add_argument(
        "--dir", type=str, default="result", help="directory to store the results"
    )

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=f"{args.dir}/res.log",
        filemode="w",
    )

    logger = logging.getLogger()

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    main(args)
