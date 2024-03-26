import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

df = pd.read_csv("./datasets/lm_extraction_128_0.csv")

ids = list(df["doc_id"])

train_ds = np.load("./datasets/target/train_dataset.npy")
train_df = pd.DataFrame(data=train_ds)

train_df = train_df.drop(index=ids)
train_ds = np.array(train_df)
np.save("_dataset.npy", train_ds)

train_preprefix_ds = np.load("./datasets/target/train_preprefix.npy")
train_preprefix_df = pd.DataFrame(data=train_preprefix_ds)

train_preprefix_df = train_preprefix_df.drop(index=ids)
train_preprefix_ds = np.array(train_preprefix_df)
np.save("_preprefix.npy", train_preprefix_ds)

train_prefix_ds = np.load("./datasets/target/train_prefix.npy")
train_prefix_df = pd.DataFrame(data=train_prefix_ds)

train_prefix_df = train_prefix_df.drop(index=ids)
train_prefix_ds = np.array(train_prefix_df)
np.save("_prefix.npy", train_prefix_ds)

train_suffix_ds = np.load("./datasets/target/train_suffix.npy")
train_suffix_df = pd.DataFrame(data=train_suffix_ds)

train_suffix_df = train_suffix_df.drop(index=ids)
train_suffix_ds = np.array(train_suffix_df)
np.save("_suffix.npy", train_suffix_ds)

# result = []


# for i in range(len(df)):
#     result.append(gpt2tokenizer.encode(df["text"][i]))
#     result[i] = result[i][:200]
#     while len(result[i]) < 200:
#         result[i].append(gpt2tokenizer.eos_token_id)
# prompts = np.array(result, dtype=np.uint16)

# np.save("_dataset.npy", prompts)

# np.save("_preprefix.npy", prompts[:, :100])
# np.save("_prefix.npy", prompts[:, 100:150])
# np.save("_suffix.npy", prompts[:, 150:200])
