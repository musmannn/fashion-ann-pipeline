# loads the model + test set, writes metrics.json and a confusion matrix png
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    model = keras.models.load_model("models/model.h5")
    test = np.load("data/processed/test.npz")
    x, y = test["x"], test["y"]

    loss, acc = model.evaluate(x, y, verbose=0)
    pred = np.argmax(model.predict(x, verbose=0), axis=1)

    os.makedirs("reports", exist_ok=True)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=100)

    metrics = {"test_loss": round(float(loss), 4), "test_accuracy": round(float(acc), 4)}
    json.dump(metrics, open("metrics.json", "w"), indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
