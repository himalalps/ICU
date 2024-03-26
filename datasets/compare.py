import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch

train_prefix_ds = np.load("./datasets/target/train_prefix.npy")
train_prefix_df = pd.DataFrame(data=train_prefix_ds)

train_suffix_ds = np.load("./datasets/target/train_suffix.npy")
train_suffix_df = pd.DataFrame(data=train_suffix_ds)

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_neo_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-1.3B", resid_dropout=0, embed_dropout=0, attention_dropout=0
).to(device)


def predict(message):
    model_inputs = gpt_neo_tokenizer.encode(message, return_tensors="pt").to(device)
    model_outputs = model.generate(
        model_inputs,
        max_new_tokens=50,
        num_beams=1,
        pad_token_id=gpt_neo_tokenizer.eos_token_id,
    )
    model_outputs = model_outputs[0, len(model_inputs[0]) :]
    model_output_text = gpt_neo_tokenizer.decode(
        model_outputs.cpu(), skip_special_tokens=True
    )
    return model_output_text


with open("data-original.txt", "w") as f:
    for i in range(len(train_prefix_df)):
        print(i)
        f.write("prefix:\n")
        message = ""
        for item in train_prefix_df.loc[i]:
            message += gpt2tokenizer.decode(item)
        f.write(message)
        f.write("\n\n\nsuffix:\n")
        for item in train_suffix_df.loc[i]:
            f.write(gpt2tokenizer.decode(item))
        f.write("\n\n\npredict:\n")
        f.write(predict(message))
        f.write("\n---------------------------------------------\n")
