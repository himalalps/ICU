import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

doc_ids= []

for i in range(5):
    df = pd.read_csv(f"./datasets/lm_extraction_128_{i}.csv")
    doc_ids += df["doc_id"].tolist()

doc_ids = list(set(doc_ids))
doc_ids.sort()

train_ds = np.load("./datasets/target/train_dataset.npy")
# train_df = pd.DataFrame(data=train_ds)

# train_df = train_df.apply(gpt2tokenizer.decode, axis=1)

# train_df.rename("text")
# sub_df = train_df[doc_ids]
# sub_df.to_csv("./datasets/exp/all/lm_extraction_631.csv")

prompts = train_ds[doc_ids]
print(prompts.dtype)
print(prompts.shape)

# result = []

# for i in range(len(df)):
#     result.append(gpt2tokenizer.encode(df["text"][i]))
#     result[i] = result[i][:200]
#     while len(result[i]) < 200:
#         result[i].append(gpt2tokenizer.eos_token_id)
# prompts = np.array(result, dtype=np.uint16)

np.save("_dataset.npy", prompts)

np.save("_preprefix.npy", prompts[:, :100])
np.save("_prefix.npy", prompts[:, 100:150])
np.save("_suffix.npy", prompts[:, 150:200])

