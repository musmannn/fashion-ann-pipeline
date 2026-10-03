# normalize pixels and split off a validation set
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW = "data/raw"
OUT = "data/processed"


def minmax(x):
    lo = x.min(axis=(1, 2), keepdims=True)
    hi = x.max(axis=(1, 2), keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-7)


def main():
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    os.makedirs(OUT, exist_ok=True)

    train = np.load(f"{RAW}/train.npz")
    test = np.load(f"{RAW}/test.npz")

    # scale every image to [0, 1] on its own (per-image min-max)
    x_train = minmax(train["x"].astype("float32"))
    x_test = minmax(test["x"].astype("float32"))

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
