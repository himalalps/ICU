import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class Custom_Dataset(Dataset):
    def __init__(
        self,
        prefix_path,
        suffix_path,
        gpt2tokenizer,
        tokenizer,
        prefix_length=50,
        suffix_length=50,
        **kwargs,
    ):
        super(Custom_Dataset, self).__init__(**kwargs)
        self.prefix_path = prefix_path
        self.suffix_path = suffix_path
        self.gpt2tokenizer = gpt2tokenizer
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self._getdata()

    def _getdata(self):
        prefix_ds = np.load(self.prefix_path)
        prefix_df = pd.DataFrame(data=prefix_ds)

        prefix_df = prefix_df.apply(self._convert, axis=1, args=(self.prefix_length,))
        self.prefix_ids = np.array(prefix_df["input_ids"])
        self.prefix_mask = np.array(prefix_df["attention_mask"])

        suffix_ds = np.load(self.suffix_path)
        suffix_df = pd.DataFrame(data=suffix_ds)

        self.gpt2_suffix = np.array(suffix_df).astype(np.int64)

        suffix_df = suffix_df.apply(self._convert, axis=1, args=(self.suffix_length,))
        self.suffix_ids = np.array(suffix_df["input_ids"])
        # self.suffix_mask = np.array(suffix_df["attention_mask"])

    def _convert(self, row, length):
        message = self.gpt2tokenizer.decode(row)
        source = self.tokenizer(
            message,
            max_length=length,
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
