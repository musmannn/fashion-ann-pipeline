# normalize pixels and split off a validation set
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW = "data/raw"
OUT = "data/processed"


def main():
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    os.makedirs(OUT, exist_ok=True)

    train = np.load(f"{RAW}/train.npz")
    test = np.load(f"{RAW}/test.npz")

    # standardize with the training mean/std instead of plain /255
    x_train = train["x"].astype("float32") / 255.0
    mean, std = x_train.mean(), x_train.std()
    x_train = (x_train - mean) / std
    x_test = (test["x"].astype("float32") / 255.0 - mean) / std

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, train["y"],
        test_size=params["val_size"],
        random_state=params["seed"],
        stratify=train["y"],
    )

    np.savez_compressed(f"{OUT}/train.npz", x=x_tr, y=y_tr)
    np.savez_compressed(f"{OUT}/val.npz", x=x_val, y=y_val)
    np.savez_compressed(f"{OUT}/test.npz", x=x_test, y=test["y"])
    print("train/val/test:", x_tr.shape, x_val.shape, x_test.shape)


if __name__ == "__main__":
    main()
