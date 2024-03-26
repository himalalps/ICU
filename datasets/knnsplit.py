import numpy as np
import pandas as pd

index_df = pd.read_csv("./datasets/exp/nearest10index.csv", index_col=0)

ids = list(index_df["0"])


def transform(data_path, file_path):
    train_ds = np.load(file_path + data_path)
    train_df = pd.DataFrame(data=train_ds)
    train_df = train_df.loc[ids]
    train_ds = np.array(train_df)
    print(list(train_ds[0]))
    print(len(train_ds[0]))
    np.save(data_path, train_ds)


transform("_dataset.npy", "./datasets/learn/")
transform("_prefix.npy", "./datasets/learn/")
transform("_preprefix.npy", "./datasets/learn/")
transform("_suffix.npy", "./datasets/learn/")
