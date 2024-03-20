from utils import batch_compute_bert_score
from Datasets import Custom_Dataset

import pandas as pd

import logging

# import numpy as np
# import matplotlib.pyplot as plt

import torch

# import torch.nn as nn
# import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="result/app.log",
    filemode="w",
)

logger = logging.getLogger()


tokenizer_name_or_path = "EleutherAI/gpt-neo-1.3B"
model_name_or_path = "EleutherAI/gpt-neo-1.3B"
bert_name_or_path = "bert-base-uncased"
gpt2_name_or_path = "gpt2"
prefix_path = "./datasets/target/train_prefix.npy"
suffix_path = "./datasets/target/train_suffix.npy"
target_length = 200
epochs = 10
batch_size = 8
num_workers = 8

truncate_len = 1000
self_predict = True
chunk_len = 32


def get_train_loader(prefix_path, suffix_path, gpt2tokenizer, tokenizer):
    train_dataset = Custom_Dataset(
        prefix_path, suffix_path, gpt2tokenizer, tokenizer, truncate_len, self_predict
    )

    # sampler = RandomSampler(train_dataset)
    sampler = SequentialSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers
    )
    return dataloader


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


def val_score(epoch):
    logger.info("Epoch {}: validating".format(epoch))
    data = []
    model.eval()
    for batch_idx, batch in enumerate(train_loader):
        reference_list = gpt2tokenizer.batch_decode(batch["gpt2_suffix"])

        input_ids = batch["prefix_ids"].to(device)

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
            logger.debug(" validating.. {}/{}".format(batch_idx, len(train_loader)))

    df = pd.DataFrame(data)
    df.sort_values(by="id", ascending=True, inplace=True)
    df.to_csv(f"./result/epoch{epoch}.csv", index=False)

    logger.debug(predict("This is a test for the model."))


gpt2tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name_or_path)
gpt2tokenizer.padding_side = "left"
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
tokenizer.padding_side = "left"
if "gpt" in tokenizer_name_or_path:
    tokenizer.pad_token = tokenizer.eos_token
# Different models have different kwargs
if "gpt-neo" in model_name_or_path:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        resid_dropout=0,
        embed_dropout=0,
        attention_dropout=0,
        pad_token_id=tokenizer.eos_token_id,
    )
elif "opt" in model_name_or_path:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dropout=0, attention_dropout=0, activation_dropout=0
    )
else:  # GPT2
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        pad_token_id=tokenizer.eos_token_id,
    )
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.98))

scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

train_loader = get_train_loader(prefix_path, suffix_path, gpt2tokenizer, tokenizer)

val_score(0)

for i in range(truncate_len // chunk_len):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        logger.info("Epoch {}: training".format(epoch))

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx not in range(
                i * chunk_len // batch_size, (i + 1) * chunk_len // batch_size
            ):
                continue
            input_ids = batch["prefix_ids"].to(device)
            attention_mask = batch["prefix_mask"].to(device)
            target_ids = batch["suffix_ids"]
            target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = -outputs[0]
            epoch_loss += outputs[0]

            loss.backward()
            optimizer.step()

            if batch_idx % chunk_len == 0:
                logger.info(
                    "Train Chunk: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        100.0 * batch_idx / len(train_loader),
                        outputs[0].item(),
                    )
                )

        logger.info(
            "Epoch {} finished, mean loss: {:.6f}".format(
                epoch, epoch_loss.item() / chunk_len * batch_size
            )
        )
    val_score(i + 1)
    scheduler.step()
