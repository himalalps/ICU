import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPTNeoForCausalLM


class Custom_Dataset(Dataset):
    def __init__(
        self,
        unlearn_data_path,
        learn_data_path,
        gpt2tokenizer,
        tokenizer,
        model=None,
        prefix_length=50,
        suffix_length=50,
        batch_size=8,
        **kwargs,
    ):
        super(Custom_Dataset, self).__init__(**kwargs)
        self.unlearn_data_path: str = unlearn_data_path
        self.learn_data_path: str = learn_data_path
        self.gpt2tokenizer: GPT2Tokenizer = gpt2tokenizer
        self.tokenizer: GPT2Tokenizer = tokenizer
        self.model: GPTNeoForCausalLM = model
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.batch_size = batch_size
        self._getdata()
        self._getpretrain()

    def _getdata(self):
        unlearn_ds = np.load(self.unlearn_data_path)
        unlearn_df = pd.DataFrame(data=unlearn_ds)

        learn_ds = np.load(self.learn_data_path)
        learn_df = pd.DataFrame(data=learn_ds)

        assert len(unlearn_df) == len(learn_df)

        self.unlearn_gpt2_prefix = np.array(unlearn_df.iloc[:, 100:150]).astype(
            np.int64
        )
        self.unlearn_gpt2_suffix = np.array(unlearn_df.iloc[:, 150:200]).astype(
            np.int64
        )

        self.learn_gpt2_prefix = np.array(learn_df.iloc[:, 100:150]).astype(np.int64)
        self.learn_gpt2_suffix = np.array(learn_df.iloc[:, 150:200]).astype(np.int64)

        unlearn_prefix_df = unlearn_df.apply(
            self._convert, axis=1, args=(self.prefix_length, True)
        )
        self.unlearn_prefix_ids = np.array(unlearn_prefix_df["input_ids"])
        self.unlearn_prefix_mask = np.array(unlearn_prefix_df["attention_mask"])
        unlearn_suffix_df = unlearn_df.apply(
            self._convert, axis=1, args=(self.suffix_length, False)
        )
        self.unlearn_suffix_ids = np.array(unlearn_suffix_df["input_ids"])

        learn_prefix_df = learn_df.apply(
            self._convert, axis=1, args=(self.prefix_length, True)
        )
        self.learn_prefix_ids = np.array(learn_prefix_df["input_ids"])
        self.learn_prefix_mask = np.array(learn_prefix_df["attention_mask"])
        learn_suffix_df = learn_df.apply(
            self._convert, axis=1, args=(self.suffix_length, False)
        )
        self.learn_suffix_ids = np.array(learn_suffix_df["input_ids"])

        self.unlearn_flag = np.ones(len(self.unlearn_gpt2_suffix))
        self.learn_flag = np.ones(len(self.learn_gpt2_suffix))

    def _getpretrain(self):
        assert len(self.learn_prefix_ids) == len(self.learn_suffix_ids)
        assert len(self.learn_prefix_ids) == len(self.learn_prefix_mask)
        probs = []
        self.model.eval()
        for i in range(len(self.learn_prefix_ids) // self.batch_size):
            input_ids = torch.cat(
                [
                    self.learn_prefix_ids[id]
                    for id in range(i * self.batch_size, (i + 1) * self.batch_size)
                ]
            ).to(self.model.device)
            attention_mask = torch.cat(
                [
                    self.learn_prefix_mask[id]
                    for id in range(i * self.batch_size, (i + 1) * self.batch_size)
                ]
            ).to(self.model.device)
            target_ids = torch.cat(
                [
                    self.learn_suffix_ids[id]
                    for id in range(i * self.batch_size, (i + 1) * self.batch_size)
                ]
            )
            target_ids[target_ids[:, :] == self.tokenizer.pad_token_id] = -100
            target_ids = target_ids.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=target_ids
                )

            prob_p = torch.nn.functional.softmax(outputs.logits, -1)
            probs.append(prob_p)

        self.probs = torch.cat(probs).to("cpu")

    def _convert(self, row, length, flag):
        message = self.gpt2tokenizer.decode(row)
        source = self.tokenizer(
            message,
            max_length=length,
            add_special_tokens=flag,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return pd.Series(source)

    def __len__(self):
        return len(self.unlearn_gpt2_suffix)

    def __getitem__(self, idx):
        return {
            "id": idx,
            "unlearn_flag": self.unlearn_flag[idx],
            "unlearn_gpt2_prefix": self.unlearn_gpt2_prefix[idx],
            "unlearn_gpt2_suffix": self.unlearn_gpt2_suffix[idx],
            "unlearn_prefix_ids": self.unlearn_prefix_ids[idx].squeeze(),
            "unlearn_prefix_mask": self.unlearn_prefix_mask[idx].squeeze(),
            "unlearn_suffix_ids": self.unlearn_suffix_ids[idx].squeeze(),
            "learn_flag": self.learn_flag[idx],
            "learn_gpt2_prefix": self.learn_gpt2_prefix[idx],
            "learn_gpt2_suffix": self.learn_gpt2_suffix[idx],
            "learn_prefix_ids": self.learn_prefix_ids[idx].squeeze(),
            "learn_prefix_mask": self.learn_prefix_mask[idx].squeeze(),
            "learn_suffix_ids": self.learn_suffix_ids[idx].squeeze(),
            "learn_prob": self.probs[idx],
        }

    def update(self, ids, unlearn_flag, learn_flag):
        for i in range(len(ids)):
            self.unlearn_flag[ids[i]] = unlearn_flag[i]
            self.learn_flag[ids[i]] = learn_flag[i]


if __name__ == "__main__":
    tokenizer_name_or_path = "EleutherAI/gpt-neo-125m"
    gpt2_name_or_path = "gpt2"
    unlearn_data_path = "./datasets/exp/exp0/unlearn/_dataset.npy"
    learn_data_path = "./datasets/exp/exp0/learn/_dataset.npy"

    gpt2tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
    model = GPTNeoForCausalLM.from_pretrained(
        tokenizer_name_or_path,
        resid_dropout=0,
        embed_dropout=0,
        attention_dropout=0,
        pad_token_id=tokenizer.eos_token_id,
    )
    gpt2tokenizer.padding_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model.to(device)

    dataset = Custom_Dataset(
        unlearn_data_path, learn_data_path, gpt2tokenizer, tokenizer, model
    )
    # print(len(dataset))
    # print(dataset[10])
