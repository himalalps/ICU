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
        ds = np.load(self.data_path)
        df = pd.DataFrame(data=ds)

        self.gpt2_prefix = np.array(df.iloc[:, 100:150]).astype(np.int64)
        self.gpt2_suffix = np.array(df.iloc[:, 150:200]).astype(np.int64)

        prefix_df = df.apply(self._convert, axis=1, args=(self.prefix_length, True))
        self.prefix_ids = np.array(prefix_df["input_ids"])
        self.prefix_mask = np.array(prefix_df["attention_mask"])
        suffix_df = df.apply(self._convert, axis=1, args=(self.suffix_length, False))
        self.suffix_ids = np.array(suffix_df["input_ids"])

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
            "gpt2_prefix": self.gpt2_prefix[idx],
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
    data_path = "./datasets/target/train_dataset.npy"
    gpt2tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = Custom_Dataset(data_path, gpt2tokenizer, tokenizer)
    print(len(dataset))
    print(dataset[10])
