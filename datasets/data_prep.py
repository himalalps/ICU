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
    file_path = ""
