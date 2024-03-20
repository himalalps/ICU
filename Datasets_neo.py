import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer


class Custom_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        gpt2tokenizer,
        tokenizer,
        prefix_length=50,
        suffix_length=50,
        **kwargs,
    ):
        super(Custom_Dataset, self).__init__(**kwargs)
        self.data_path: str = data_path
        self.gpt2tokenizer: GPT2Tokenizer = gpt2tokenizer
        self.tokenizer: GPT2Tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self._getdata()

    def _getdata(self):
        data_df = pd.read_csv(self.data_path)

        self.df = data_df.drop(columns=["corpus"])
        self.ids = np.array(self.df["doc_id"]).astype(np.int64)
        suffix = self.df["text"].apply(self._convert)
        self.gpt2_suffix = np.array(suffix).astype(np.int64)

        df = self.df["text"].apply(self._convert_id, args=(self.prefix_length,))
        self.prefix_ids = np.array(df["input_ids"])
        self.prefix_mask = np.array(df["attention_mask"])
        self.suffix_ids = np.array(df["input_ids"])

    def _convert(self, row):
        message = self.gpt2tokenizer.encode(row)
        message = message[-50:]
        return pd.Series(message)

    def _convert_id(self, row, length):
        message = self.gpt2tokenizer.encode(row)
        message = message[-100:]
        source = self.tokenizer(
            row,
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
            "id": self.ids[idx],
            "gpt2_suffix": self.gpt2_suffix[idx],
            "prefix_ids": self.prefix_ids[idx].squeeze(),
            "prefix_mask": self.prefix_mask[idx].squeeze(),
            "suffix_ids": self.suffix_ids[idx].squeeze(),
            # "suffix_mask": self.suffix_mask[idx],
        }


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    # tokenizer_name_or_path = "EleutherAI/gpt-neo-1.3B"
    # gpt2_name_or_path = "gpt2"
    # prefix_path = "./datasets/target/train_prefix.npy"
    # suffix_path = "./datasets/target/train_suffix.npy"
    # gpt2tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name_or_path)
    # tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
    # tokenizer.pad_token = tokenizer.eos_token
    # dataset = Custom_Dataset(prefix_path, suffix_path, gpt2tokenizer, tokenizer)
    # print(len(dataset))
    # print(dataset[10])
