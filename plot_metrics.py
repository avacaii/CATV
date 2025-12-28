import ast
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """
    Parse a training log inside the lambda file system, saves time
    """
    metrics: Dict[str, List[float]] = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            # Quick filter: skip lines that clearly aren't dicts
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                record = ast.literal_eval(line)
            except Exception:
                continue

            # Only care about records that have an epoch + loss (typical training step logs)
            if "epoch" not in record or "loss" not in record:
                continue

            for key, value in record.items():
                # Only keep numeric values (float/int)
                if isinstance(value, (float, int)):
                    metrics.setdefault(key, []).append(float(value))

    if not metrics:
        raise ValueError(f"No metric dicts found in log file: {log_path}")

    return metrics


def plot_all_metrics(
    metrics: Dict[str, List[float]],
    output_path: str = "training_metrics_4in1.png",
) -> None:
    """
    Create a 2x2 subplot figure:
      - loss vs epoch
      - learning_rate vs epoch (if available)
      - mean_token_accuracy vs epoch (if available)
      - grad_norm vs epoch (if available)

    Saves figure to output_path.
    """
    if "epoch" not in metrics:
        raise ValueError("Metrics dict must contain 'epoch' key.")

    epoch = metrics["epoch"]

    # Prepare figure
    plt.figure(figsize=(12, 8))

    # 1. Loss vs Epoch
    plt.subplot(2, 2, 1)
    if "loss" in metrics:
        plt.plot(epoch, metrics["loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
    else:
        plt.text(0.5, 0.5, "No 'loss' data", ha="center", va="center")
        plt.axis("off")

    # 2. Learning Rate vs Epoch
    plt.subplot(2, 2, 2)
    if "learning_rate" in metrics:
        plt.plot(epoch, metrics["learning_rate"])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate vs Epoch")
    else:
        plt.text(0.5, 0.5, "No 'learning_rate' data", ha="center", va="center")
        plt.axis("off")

    # 3. Mean Token Accuracy vs Epoch
    plt.subplot(2, 2, 3)
    if "mean_token_accuracy" in metrics:
        plt.plot(epoch, metrics["mean_token_accuracy"])
        plt.xlabel("Epoch")
        plt.ylabel("Mean Token Accuracy")
        plt.title("Mean Token Accuracy vs Epoch")
    else:
        plt.text(0.5, 0.5, "No 'mean_token_accuracy' data", ha="center", va="center")
        plt.axis("off")

    # 4. Grad Norm vs Epoch
    plt.subplot(2, 2, 4)
    if "grad_norm" in metrics:
        plt.plot(epoch, metrics["grad_norm"])
        plt.xlabel("Epoch")
        plt.ylabel("Grad Norm")
        plt.title("Grad Norm vs Epoch")
    else:
        plt.text(0.5, 0.5, "No 'grad_norm' data", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved training metrics figure to: {os.path.abspath(output_path)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse training log and generate metric plots."
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to training log file (e.g., stage2_harmful_finetune.log)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="training_metrics_4in1.png",
        help="Output PNG path for combined 2x2 figure",
    )

    args = parser.parse_args()

    metrics = parse_training_log(args.log)
    plot_all_metrics(metrics, args.out)


if __name__ == "__main__":
    main()
