from utils import batch_compute_bert_score
from Datasets_neo import Custom_Dataset


import os
import logging
import random
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt

import torch

# import torch.nn as nn
# import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    GPT2Tokenizer,
    AutoModelForCausalLM,
    GPTNeoForCausalLM,
    OPTForCausalLM,
    GPT2LMHeadModel,
)
from torchmetrics.functional import accuracy

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="result/app.log",
    filemode="w",
)

logger = logging.getLogger()


tokenizer_name_or_path = "EleutherAI/gpt-neo-125m"
model_name_or_path = "EleutherAI/gpt-neo-125m"
bert_name_or_path = "bert-base-uncased"
gpt2_name_or_path = "gpt2"
unlearn_data_path = "./datasets/test/_dataset.npy"
learn_data_path = "./datasets/learn/_dataset.npy"
prefix_length = 200
suffix_length = 200
target_length = 200
epochs = 30
batch_size = 8
num_workers = 8

truncate_len = 128
self_predict = True
lambda_val = 0.5

el_n = [10]

valid_result = []

num_epoch = 30


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
seed_everything(seed)


def get_loader(data_path, gpt2tokenizer, tokenizer):
    dataset = Custom_Dataset(
        data_path,
        gpt2tokenizer,
        tokenizer,
        prefix_length,
        suffix_length,
    )

    sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers
    )
    valid_dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return train_dataloader, valid_dataloader


def predict(message):
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


def val(epoch):
    logger.info("Validating epoch {}".format(epoch))
    with torch.no_grad():
        val_score(epoch)
        val_loss = validation_forget(epoch)
        acc = validation_ma(epoch)
        els = validation_el(epoch)
        valid_dict = {
            "epoch": epoch,
            "val_loss": val_loss,
            "acc": acc,
        }
        for n, el in zip(el_n, els):
            valid_dict[f"el_{n}"] = el
        valid_result.append(valid_dict)
    if acc < 0.2994 and any([el < 0.0499 for el in els]):
        logger.info("Epoch {} should stop, acc {}, el {}".format(epoch, acc, els))
        return True
    else:
        torch.cuda.empty_cache()
        return False


def val_score(epoch):
    data = []
    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        reference_list = gpt2tokenizer.batch_decode(batch["gpt2_suffix"])

        message = gpt2tokenizer.batch_decode(batch["gpt2_prefix"])
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
        candidate_list = tokenizer.batch_decode(candidate_list)

        P, R, F1 = batch_compute_bert_score(
            candidate_list, reference_list, bert_name_or_path, device
        )
        for i in range(len(P)):
            data.append(
                {
                    "id": batch["id"][i].item(),
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
                    "id {} P {} R {} F1 {}".format(
                        batch["id"][i].item(), P[i].item(), R[i].item(), F1[i].item()
                    )
                )
        if batch_idx % 50 == 0:
            logger.debug(" validating.. {}/{}".format(batch_idx, len(val_loader)))

    df = pd.DataFrame(data)
    df.sort_values(by="id", ascending=True, inplace=True)
    df.to_csv(f"./result/epoch{epoch}.csv", index=False)

    logger.debug(predict("This is a test for the model."))


def validation_forget(epoch):
    model.eval()
    epoch_loss = 0
    for batch_idx, batch in enumerate(val_loader):
        input_ids = batch["prefix_ids"].to(device)
        attention_mask = batch["prefix_mask"].to(device)
        target_ids = batch["suffix_ids"]
        target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
        target_ids = target_ids.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        epoch_loss += outputs[0].item()

    logger.info("val_loss [epoch {}] {}".format(epoch, epoch_loss / len(val_loader)))
    return epoch_loss / len(val_loader)


def validation_ma(epoch):
    model.eval()
    epoch_acc = 0

    for batch_idx, batch in enumerate(val_loader):
        input_ids = batch["prefix_ids"].to(device)

        labels, preds = [], []
        for i in range(1, target_length):
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
                num_classes=tokenizer.pad_token_id,
                ignore_index=-100,
            )
        epoch_acc += score.item()

    logger.info("acc [epoch {}] {}".format(epoch, epoch_acc / len(val_loader)))
    return epoch_acc / len(val_loader)


def validation_el(epoch):
    def ngram_of_1D_tensor(X, n):
        grams = [tuple(X[i : i + n].tolist()) for i in range(X.shape[0] - n + 1)]
        return grams

    model.eval()
    el_score = {n: [] for n in el_n}

    for batch_idx, batch in enumerate(val_loader):
        input_ids = batch["prefix_ids"].to(device)

        cur_batch_size = input_ids.shape[0]
        numerator = {n: [0] * cur_batch_size for n in el_n}
        for i in reversed(range(150, target_length)):
            label = input_ids[..., i:target_length]
            prompt = input_ids[..., :i]
            pred = model.generate(
                prompt, max_length=target_length, pad_token_id=tokenizer.eos_token_id
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
            el_score[n] += [s / (target_length - 150 - (n - 1)) for s in numerator[n]]

    els = []
    for n in el_n:
        logger.info(
            "el_{}-gram [epoch {}] {}".format(
                n, epoch, sum(el_score[n]) / len(el_score[n])
            )
        )
        els.append(sum(el_score[n]) / len(el_score[n]))
    return els


gpt2tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name_or_path)
gpt2tokenizer.padding_side = "left"
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
tokenizer.padding_side = "left"
if "gpt" in tokenizer_name_or_path:
    tokenizer.pad_token = tokenizer.eos_token
# Different models have different kwargs
if "gpt-neo" in model_name_or_path:
    model: GPTNeoForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        resid_dropout=0,
        embed_dropout=0,
        attention_dropout=0,
        pad_token_id=tokenizer.eos_token_id,
    )
elif "opt" in model_name_or_path:
    model: OPTForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dropout=0, attention_dropout=0, activation_dropout=0
    )
else:  # GPT2
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        pad_token_id=tokenizer.eos_token_id,
    )
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.98))

# scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

unlearn_loader, val_loader = get_loader(unlearn_data_path, gpt2tokenizer, tokenizer)
learn_loader, _ = get_loader(learn_data_path, gpt2tokenizer, tokenizer)


val(0)
for epoch in range(num_epoch):
    torch.cuda.empty_cache()
    model.train()
    epoch_loss = 0

    logger.info("Epoch {}: training".format(epoch + 1))

    for batch_idx, batch in enumerate(unlearn_loader):
        optimizer.zero_grad()
        input_ids = batch["prefix_ids"].to(device)
        attention_mask = batch["prefix_mask"].to(device)
        target_ids = batch["suffix_ids"]
        target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
        target_ids = target_ids.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        current_loss = outputs[0]

        loss = -lambda_val * current_loss
        epoch_loss += current_loss

        loss.backward()
        optimizer.step()

        logger.info(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch + 1,
                batch_idx,
                len(unlearn_loader) + len(learn_loader),
                100.0 * batch_idx / (len(unlearn_loader) + len(learn_loader)),
                current_loss.item(),
            )
        )

    for batch_idx, batch in enumerate(learn_loader):
        optimizer.zero_grad()
        input_ids = batch["prefix_ids"].to(device)
        attention_mask = batch["prefix_mask"].to(device)
        target_ids = batch["suffix_ids"]
        target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
        target_ids = target_ids.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        current_loss = outputs[0]

        loss = lambda_val * current_loss
        epoch_loss += current_loss

        loss.backward()
        optimizer.step()

        logger.info(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch + 1,
                batch_idx + len(unlearn_loader),
                len(unlearn_loader) + len(learn_loader),
                100.0 * batch_idx / (len(learn_loader) + len(unlearn_loader)),
                current_loss.item(),
            )
        )

    logger.info(
        "Epoch {} finished, mean loss: {:.6f}".format(
            epoch + 1, epoch_loss.item() / (len(learn_loader) + len(unlearn_loader))
        )
    )
    if val(epoch + 1):
        break
    # scheduler.step()

valid_df = pd.DataFrame(valid_result)
valid_df.to_csv("./result/valid.csv", index=False)
