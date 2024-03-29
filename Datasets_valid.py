import re
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from datasets import load_dataset
from utils import normalize_reply, DIALOG_DATASETS
from transformers import GPT2Tokenizer


class Valid_Dataset(Dataset):
    def __init__(
        self,
        dataset_name,
        valid_subset_path,
        type_path,
        tokenizer,
        prefix_length=200,
        suffix_length=200,
        cache_dir=".cache/",
        **kwargs,
    ):
        super(Valid_Dataset, self).__init__(**kwargs)
        self.dataset_name: str = dataset_name
        self.valid_subset_path = valid_subset_path
        self.type_path = type_path
        self.tokenizer: GPT2Tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.cache_dir = cache_dir
        self._getdata()

    def _getdata(self):
        if ".csv" in self.dataset_name:
            dataset = pd.read_csv(self.dataset_name)
        elif ".json" in self.dataset_name:
            dataset = pd.read_json(self.dataset_name)
        else:
            if self.valid_subset_path:
                dataset = load_dataset(
                    self.dataset_name,
                    self.valid_subset_path,
                    split=self.type_path,
                    verification_mode="no_checks",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.type_path,
                    verification_mode="no_checks",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )
            dataset = dataset.to_pandas()

        if self.dataset_name == "ai2_arc":
            dataset["length"] = dataset["choices"].apply(lambda x: len(x["text"]))
            dataset = dataset[dataset["length"] == 4]
        self.dataset = dataset.dropna()

    def input_to_target(self, input):
        input_s = input.split(" ")
        input_ = " ".join(input_s[: len(input_s) - 1])
        target = " " + input_s[len(input_s) - 1]
        return input_, target

    def create_dialogue_prompt(self, turns):
        # prompt = 'A converstaion between two Users:\n'
        prompt = ""
        for i, turn in enumerate(turns):
            turn = normalize_reply(turn)

            if i % 2 == 0:
                prompt += f"User 1: {turn}\n"
            else:
                prompt += f"User 2: {turn}\n"

        if i % 2:
            prompt += f"User 1:"
        else:
            prompt += f"User 2:"
        return prompt

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch):
        choices = []
        answer_index = 0

        if "lambada" in self.dataset_name:
            input_, target_ = self.input_to_target(example_batch["text"])
        elif "piqa" in self.dataset_name:
            input_ = example_batch["goal"]
            choices = [" " + example_batch["sol1"], " " + example_batch["sol2"]]
            target_ = choices[int(example_batch["label"])]
            answer_index = int(example_batch["label"])
        elif "hellaswag" in self.dataset_name:
            input_ = example_batch["ctx"]
            choices = []
            choices = [" " + c for c in example_batch["endings"]]
            target_ = choices[int(example_batch["label"])]
            answer_index = int(example_batch["label"])
        elif "ai2_arc" in self.dataset_name:
            input_ = example_batch["question"]
            choices = [" " + c for c in example_batch["choices"]["text"]]
            if len(choices) < 4:
                choices.append("just for work")
            elif len(choices) > 4:
                if example_batch["answerKey"] == "E":
                    choices = choices[-4:]
                    example_batch["choices"]["label"] = example_batch["choices"]["label"][-4:]
                else:
                    choices = choices[:4]
            answer_index = (
                example_batch["choices"]["label"]
                .tolist()
                .index(example_batch["answerKey"])
            )
            target_ = choices[answer_index]
        elif "winogrande" in self.dataset_name:
            input_, rest = example_batch["sentence"].split(" _")
            choices = [
                " " + example_batch["option1"] + rest,
                " " + example_batch["option2"] + rest,
            ]
            answer_index = int(example_batch["answer"]) - 1  # Label are '1' or '2'
            target_ = choices[answer_index]
        elif "math_qa" in self.dataset_name:
            input_ = example_batch["Problem"]
            choices = [
                c[4:].rstrip(" ,")
                for c in re.findall(
                    r"[abcd] \) .*?, |e \) .*?$", example_batch["options"]
                )
            ]
            answer_index = ["a", "b", "c", "d", "e"].index(example_batch["correct"])
            target_ = choices[answer_index]
        elif "pubmed_qa" in self.dataset_name:
            input_ = f"Context: {example_batch['abstract']}\nQuestion: {example_batch['question']}\nAnswer:"
            choices = [" yes", " maybe", " no"]
            answer_index = ["yes", "maybe", "no"].index(example_batch["final_decision"])
            target_ = choices[answer_index]
        elif "super_glue" in self.dataset_name and self.valid_subset_path == "copa":
            input_ = example_batch["premise"]
            choices = [
                " " + example_batch["choice1"],
                " " + example_batch["choice2"],
            ]
            answer_index = int(example_batch["label"])
            target_ = choices[answer_index]
        elif any(d in self.dataset_name for d in DIALOG_DATASETS):
            input_ = self.create_dialogue_prompt(example_batch["text"][:-1])
            target_ = normalize_reply(example_batch["text"][-1])
        elif "pile" in self.dataset_name:
            input_, target_ = example_batch["text"], example_batch["text"]
        elif "wikitext" in self.dataset_name:
            input_, target_ = example_batch["text"], example_batch["text"]
        else:
            input_, target_ = example_batch["text"], example_batch["text"]

        source = self.tokenizer(
            input_,
            max_length=self.prefix_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer(
            target_,
            max_length=self.suffix_length,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return source, targets, choices, answer_index

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        try:
            source, targets, choices, answer_index = self.convert_to_features(data)
        except:
            print(data)

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "choices": choices,
            "answer_index": answer_index,
        }


if __name__ == "__main__":
    tokenizer_name_or_path = "EleutherAI/gpt-neo-1.3B"
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    valid_data = [
        # ["datasets/lambada.csv", "", "test"],
        # ["datasets/piqa", "", "validation"],
        # ["datasets/hellaswag", "", "validation"],
        # ["datasets/ai2_arc", "ARC-Easy", "validation"],
        ["datasets/ai2_arc", "ARC-Challenge", "validation"],
        # ["datasets/super_glue", "copa", "validation"],
        # ["datasets/winogrande", "winogrande_s", "validation"],
        # ["datasets/math_qa", "", "validation"],
    ]
    for data in valid_data:
        valid_dataset = Valid_Dataset(data[0], data[1], data[2], tokenizer)
        print(len(valid_dataset))
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=8, num_workers=48, shuffle=False
        )

        for batch_idx, batch in enumerate(valid_dataloader):
            print(batch_idx)

