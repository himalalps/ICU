import os
import faiss
import numpy as np
import pandas as pd

from transformers import GPT2Tokenizer
from sentence_transformers import SentenceTransformer


def get_embeddings(df, gpt2tokenizer, model):
    sentence = []

    for i in range(len(df)):
        sentence.append(gpt2tokenizer.decode(df.loc[i, 100:]))

    embeddings = model.encode(sentence)
    return sentence, embeddings


def prep(i, exp_path, train_path, gpt2tokenizer, model):
    file_path = exp_path + f"exp{i}/"
    if not os.path.exists(file_path + "unlearn"):
        os.makedirs(file_path + "unlearn")
    if not os.path.exists(file_path + "learn"):
        os.makedirs(file_path + "learn")

    df = pd.read_csv(file_path + f"lm_extraction_128_{i}.csv")
    doc_ids = list(df["doc_id"])

    train_ds = np.load(train_path)
    train_df = pd.DataFrame(data=train_ds)
    df = train_df[train_df.index.isin(doc_ids)]
    df = df.dropna().reset_index(drop=True)
    prompts = np.array(df, dtype=np.uint16)
    np.save(file_path + "unlearn/_dataset.npy", prompts)
    np.save(file_path + "unlearn/_preprefix.npy", prompts[:, :100])
    np.save(file_path + "unlearn/_prefix.npy", prompts[:, 100:150])
    np.save(file_path + "unlearn/_suffix.npy", prompts[:, 150:200])

    train_df = train_df.drop(index=doc_ids)
    train_df = train_df.dropna().reset_index(drop=True)

    _, target_embeddings = get_embeddings(df, gpt2tokenizer, model)

    _, data_embeddings = get_embeddings(train_df, gpt2tokenizer, model)

    n = data_embeddings.shape[0]

    index = faiss.IndexFlatL2(d)

    index.add(data_embeddings)

    D, I = index.search(target_embeddings, k)

    index_df = pd.DataFrame(I)
    index_df.to_csv(file_path + "nearest10index.csv")

    ids = list(index_df[0])
    train_df = train_df.loc[ids]

    prompts = np.array(train_df, dtype=np.uint16)
    np.save(file_path + "learn/_dataset.npy", prompts)
    np.save(file_path + "learn/_preprefix.npy", prompts[:, :100])
    np.save(file_path + "learn/_prefix.npy", prompts[:, 100:150])
    np.save(file_path + "learn/_suffix.npy", prompts[:, 150:200])
    return doc_ids


if __name__ == "__main__":
    exp_path = "./datasets/exp/"
    train_path = "./datasets/train_dataset.npy"

    d = 384  # embedding size
    k = 10  # KNN

    gpt2tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:1")
    gpt2tokenizer.padding_side = "left"

    doc_ids = []

    for i in range(5):
        doc_ids += prep(i, exp_path, train_path, gpt2tokenizer, model)

    doc_ids = list(set(doc_ids))

    if not os.path.exists(exp_path + "all/unlearn"):
        os.makedirs(exp_path + "all/unlearn")

    train_ds = np.load(train_path)
    train_df = pd.DataFrame(data=train_ds)
    df = train_df[train_df.index.isin(doc_ids)]
    df = df.dropna().reset_index(drop=True)
    df.to_csv(exp_path + f"all/lm_extraction_{len(doc_ids)}.csv")
    prompts = np.array(df, dtype=np.uint16)
    np.save(exp_path + "all/unlearn/_dataset.npy", prompts)
    np.save(exp_path + "all/unlearn/_preprefix.npy", prompts[:, :100])
    np.save(exp_path + "all/unlearn/_prefix.npy", prompts[:, 100:150])
    np.save(exp_path + "all/unlearn/_suffix.npy", prompts[:, 150:200])
