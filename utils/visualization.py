import matplotlib.pyplot as plt
import numpy as np

def save_train_graph(train_log, path):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_log["loss"], label="Train Loss")
    plt.legend()
    plt.title("Train Loss Curve")

    plt.subplot(2, 1, 2)
    for k in ["OKS", "PCK", "PCP", "PDJ"]:
        plt.plot(train_log[k], label=f"Train {k}")
    plt.legend()
    plt.title("Train Metrics Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_val_graph(val_log, path):
    if len(val_log["loss"]) == 0:
        return  # Nothing to save yet
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(2, len(val_log["loss"]) * 3 + 1, 3), val_log["loss"], label="Val Loss")
    plt.legend()
    plt.title("Validation Loss Curve")

    plt.subplot(2, 1, 2)
    for k in ["OKS", "PCK", "PCP", "PDJ"]:
        plt.plot(np.arange(2, len(val_log[k]) * 3 + 1, 3), val_log[k], label=f"Val {k}")
    plt.legend()
    plt.title("Validation Metrics Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
