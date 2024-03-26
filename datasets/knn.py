import faiss
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer
from sentence_transformers import SentenceTransformer


def get_embeddings(data_path):
    ds = np.load(data_path)
    df = pd.DataFrame(data=ds)
    sentence = []

    for i in range(len(df)):
        sentence.append(gpt2tokenizer.decode(df.loc[i, 100:]))

    embeddings = model.encode(sentence)
    return sentence, embeddings


d = 384
k = 10

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = SentenceTransformer("all-MiniLM-L6-v2")

target_sentence, target_embeddings = get_embeddings("./datasets/test/_dataset.npy")

data_sentence, data_embeddings = get_embeddings("./datasets/learn/_dataset.npy")

n = data_embeddings.shape[0]

index = faiss.IndexFlatL2(d)

index.add(data_embeddings)

D, I = index.search(target_embeddings, k)


index_df = pd.DataFrame(I)
index_df.to_csv("nearest10index.csv")
