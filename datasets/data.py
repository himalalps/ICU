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


if __name__ == "__main__":
    file_path = "datasets/exp/exp1/"
    df_path = "lm_extraction_128_1.csv"
    train_path = "./datasets/target/train_dataset.npy"
    if not os.path.exists(file_path + "unlearn"):
        os.makedirs(file_path + "unlearn")
    if not os.path.exists(file_path + "learn"):
        os.makedirs(file_path + "learn")

    d = 384
    k = 10
    gpt2tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gpt2tokenizer.padding_side = "left"

    df = pd.read_csv(file_path + df_path)
    ids = list(df["doc_id"])

    train_ds = np.load(train_path)
    train_df = pd.DataFrame(data=train_ds)
    df = train_df[train_df.index.isin(ids)]
    df = df.dropna().reset_index(drop=True)
    prompts = np.array(df, dtype=np.uint16)
    np.save(file_path + "unlearn/_dataset.npy", prompts)
    np.save(file_path + "unlearn/_preprefix.npy", prompts[:, :100])
    np.save(file_path + "unlearn/_prefix.npy", prompts[:, 100:150])
    np.save(file_path + "unlearn/_suffix.npy", prompts[:, 150:200])

    train_df = train_df.drop(index=ids)
    train_df = train_df.dropna().reset_index(drop=True)

    target_sentence, target_embeddings = get_embeddings(df, gpt2tokenizer, model)

    data_sentence, data_embeddings = get_embeddings(train_df, gpt2tokenizer, model)

    n = data_embeddings.shape[0]

    print("learn len:", n)

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
