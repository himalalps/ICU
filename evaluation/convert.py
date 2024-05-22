import numpy as np
import pandas as pd
import json

doc_ids = []

for i in range(5):
    df = pd.read_csv(f"./lm_extraction_128_{i}.csv")
    doc_ids += df["doc_id"].tolist()

doc_ids = list(set(doc_ids))
doc_ids.sort()

# for i in range(5):
#     df = pd.read_csv(f"./lm_extraction_128_{i}.csv")
#     ind = [doc_ids.index(df["doc_id"][i]) for i in range(len(df))]
#     with open(f"./{i}.txt", "w") as f:
#         for i in ind:
#             f.write(f"{i}\n")


policies = ["and", "ours"]

for j in range(5):
    df = pd.read_csv(f"./lm_extraction_128_{j}.csv")
    ind = [doc_ids.index(df["doc_id"][i]) for i in range(len(df))]

    # print(ind)

    before_df = pd.read_csv("2_7Bneo.csv")
    opt_df = pd.read_csv("2_7Bopt.csv")

    before_df = before_df.iloc[ind].reset_index(drop=True)
    before_df.id = before_df.index
    opt_df = opt_df.iloc[ind].reset_index(drop=True)
    opt_df.id = opt_df.index

    before_df.to_csv(f"2_7B-{j}/neo.csv", index=False)
    opt_df.to_csv(f"2_7B-{j}/opt.csv", index=False)

    dfs = []
    for policy in policies:
        df = pd.read_csv(f"./2_7B-{j}/{policy}.csv")
        assert len(df) == len(ind), f"{policy}, {len(df)} != {len(ind)}"
        dfs.append(df)

    results = []

    for i in range(len(ind)):
        prefix = before_df["prefix"][i].replace("<|endoftext|>", "")
        reference = before_df["reference"][i].replace("<|endoftext|>", "")
        before = before_df["candidate"][i].replace("<|endoftext|>", "")
        opt = opt_df["candidate"][i].replace("<|endoftext|>", "")
        results.append(
            {"prefix": prefix, "reference": reference, "original": before, "opt": opt}
        )
        for k in range(len(policies)):
            results[-1][f"{policies[k]}"] = dfs[k]["candidate"][i].replace(
                "<|endoftext|>", ""
            )

    json.dump(results, open(f"2_7B-{j}/results.json", "w"))
