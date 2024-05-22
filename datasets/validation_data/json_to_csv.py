import csv
import json
import pandas as pd


json_list = [
    "validation_data/wizard_of_wikipedia.json",
    "validation_data/empathetic_dialogues.json",
    "validation_data/blended_skill_talk.json",
    "validation_data/wizard_of_internet.json",
]

for file in json_list:
    df = pd.DataFrame(columns=["doc_id", "corpus", "text"])
    with open(file) as f:
        content = json.load(f)
        for idx, item in content["text"].items():
            df.loc[len(df)] = [idx, file.split(".")[0].split("/")[1], item]
    print(df.head(10))
    print(len(df))
    df.to_csv(file.split(".")[0] + ".csv", index=False, quoting=csv.QUOTE_NONE, escapechar='|')
