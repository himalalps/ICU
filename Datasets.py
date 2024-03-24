import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer


class Custom_Dataset(Dataset):
    def __init__(
        self,
        prefix_path,
        suffix_path,
        gpt2tokenizer,
        tokenizer,
        truncate_len,
        self_predict=False,
        prefix_length=50,
        suffix_length=50,
        **kwargs,
    ):
        super(Custom_Dataset, self).__init__(**kwargs)
        self.prefix_path: str = prefix_path
        self.suffix_path: str = suffix_path
        self.gpt2tokenizer: GPT2Tokenizer = gpt2tokenizer
        self.tokenizer: GPT2Tokenizer = tokenizer
        self.self_predict = self_predict
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self._getdata(truncate_len)

    def _getdata(self, truncate_len):
        if self.self_predict:
            prefix_ds = np.load(self.prefix_path)
            prefix_df = pd.DataFrame(data=prefix_ds)
            suffix_ds = np.load(self.suffix_path)
            suffix_df = pd.DataFrame(data=suffix_ds)
            suffix = suffix_df.head(truncate_len)

            self.gpt2_suffix = np.array(suffix).astype(np.int64)

            suffix_df.rename(columns=lambda x: x + 50, inplace=True)
            df = pd.concat([prefix_df, suffix_df], axis=1)
            df = df.head(truncate_len)

            prefix_df = df.apply(self._convert, axis=1, args=(self.prefix_length, True))
            self.prefix_ids = np.array(prefix_df["input_ids"])
            self.prefix_mask = np.array(prefix_df["attention_mask"])
            suffix_df = df.apply(
                self._convert, axis=1, args=(self.suffix_length, False)
            )
            self.suffix_ids = np.array(suffix_df["input_ids"])
        else:
            prefix_ds = np.load(self.prefix_path)
            prefix_df = pd.DataFrame(data=prefix_ds)
            prefix_df = prefix_df.head(truncate_len)

            prefix_df = prefix_df.apply(
                self._convert, axis=1, args=(self.prefix_length, True)
            )
            self.prefix_ids = np.array(prefix_df["input_ids"])
            self.prefix_mask = np.array(prefix_df["attention_mask"])

            suffix_ds = np.load(self.suffix_path)
            suffix_df = pd.DataFrame(data=suffix_ds)
            suffix_df = suffix_df.head(truncate_len)

            self.gpt2_suffix = np.array(suffix_df).astype(np.int64)

            suffix_df = suffix_df.apply(
                self._convert, axis=1, args=(self.suffix_length, False)
            )
            self.suffix_ids = np.array(suffix_df["input_ids"])
            # self.suffix_mask = np.array(suffix_df["attention_mask"])

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
        return len(self.gpt2_suffix)

    def __getitem__(self, idx):
        return {
            "id": idx,
            "gpt2_suffix": self.gpt2_suffix[idx],
            "prefix_ids": self.prefix_ids[idx].squeeze(),
            "prefix_mask": self.prefix_mask[idx].squeeze(),
            "suffix_ids": self.suffix_ids[idx].squeeze(),
            # "suffix_mask": self.suffix_mask[idx],
        }


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    tokenizer_name_or_path = "EleutherAI/gpt-neo-1.3B"
    gpt2_name_or_path = "gpt2"
    prefix_path = "./datasets/target/train_prefix.npy"
    suffix_path = "./datasets/target/train_suffix.npy"
    gpt2tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = Custom_Dataset(prefix_path, suffix_path, gpt2tokenizer, tokenizer)
    print(len(dataset))
    print(dataset[10])
