import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer


class Custom_Dataset(Dataset):
    def __init__(
        self,
        unlearn_data_path,
        learn_data_path,
        gpt2tokenizer,
        tokenizer,
        prefix_length=50,
        suffix_length=50,
        **kwargs,
    ):
        super(Custom_Dataset, self).__init__(**kwargs)
        self.unlearn_data_path: str = unlearn_data_path
        self.learn_data_path: str = learn_data_path
        self.gpt2tokenizer: GPT2Tokenizer = gpt2tokenizer
        self.tokenizer: GPT2Tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self._getdata()

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
        }

    def update(self, ids, unlearn_flag, learn_flag):
        for i in range(len(ids)):
            self.unlearn_flag[ids[i]] = unlearn_flag[i]
            self.learn_flag[ids[i]] = learn_flag[i]


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
