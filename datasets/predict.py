import numpy as np
import pandas as pd

from transformers import GPT2Tokenizer, AutoModelForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-1.3B", resid_dropout=0, embed_dropout=0, attention_dropout=0
)


def predict(message):
    model_inputs = tokenizer.encode(message, return_tensors="pt")
    model_outputs = model.generate(
        model_inputs,
        max_new_tokens=100,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    model_outputs = model_outputs[0, len(model_inputs[0]) :]
    model_output_text = tokenizer.decode(model_outputs, skip_special_tokens=True)
    return model_output_text


print(
    predict(
        "F91E030938500209393\n:103E6000840096BBB09BFECF1F9AA8954091C000DE\n:103E700047FD02C0815089F793D0813479F490D006\n:103E8000182FA0D0123811F480E004C088E0113857\n:103E900009F083E07ED080"
    )
)
