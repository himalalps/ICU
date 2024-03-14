from torch import nn
from transformers import GPT2Tokenizer, AutoModelForCausalLM


class Unlearn(nn.Module):
    def __init__(
        self, tokenizer_name_or_path, model_name_or_path, target_length=200, **kwargs
    ):
        super(Unlearn, self).__init__(**kwargs)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
        if "gpt" in tokenizer_name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Different models have different kwargs
        if "gpt-neo" in model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                resid_dropout=0,
                embed_dropout=0,
                attention_dropout=0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elif "opt" in model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, dropout=0, attention_dropout=0, activation_dropout=0
            )
        else:  # GPT2
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.target_length = target_length

    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=lm_labels)
