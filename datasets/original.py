import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch

train_ds = np.load("./datasets/target/train_dataset.npy")
train_df = pd.DataFrame(data=train_ds)

train_prefix_ds = np.load("./datasets/target/train_prefix.npy")
train_prefix_df = pd.DataFrame(data=train_prefix_ds)

train_suffix_ds = np.load("./datasets/target/train_suffix.npy")
train_suffix_df = pd.DataFrame(data=train_suffix_ds)

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


with open("original.txt", "w") as f:
    for i in range(len(train_df)):
        f.write(str(i))
        f.write("\ninstance:\n")
        f.write(gpt2tokenizer.decode(train_df.loc[i]))
        f.write("\n---\nprefix:\n")
        f.write(gpt2tokenizer.decode(train_prefix_df.loc[i]))
        f.write("\n---\nsuffix:\n")
        f.write(gpt2tokenizer.decode(train_suffix_df.loc[i]))
        f.write("\n---------------------------------------------\n")
