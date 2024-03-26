import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

df = pd.read_csv("./datasets/lm_extraction_128_0.csv")

result = []

for i in range(len(df)):
    result.append(gpt2tokenizer.encode(df["text"][i]))
    result[i] = result[i][:200]
    while len(result[i]) < 200:
        result[i].append(gpt2tokenizer.eos_token_id)
prompts = np.array(result, dtype=np.uint16)

# np.save("_dataset.npy", prompts)

# np.save("_preprefix.npy", prompts[:, :100])
# np.save("_prefix.npy", prompts[:, 100:150])
# np.save("_suffix.npy", prompts[:, 150:200])
print(prompts)