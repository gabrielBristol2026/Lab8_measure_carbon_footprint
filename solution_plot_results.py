import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob


def plot_results(log_dir):
    """Find all CSV log files and generate comparison plots."""
    csv_files = glob.glob(os.path.join(log_dir, "log_lr_*.csv"))

    if not csv_files:
        print(f"No log files found in {log_dir}. Exiting.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Plot 1: Test Loss vs. Epoch ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title("Test Loss vs. Epoch for Different Learning Rates")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Test Loss")

    # --- Plot 2: Accuracy vs. Epoch ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title("Test Accuracy vs. Epoch for Different Learning Rates")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")

    for file in csv_files:
        try:
            # Extract learning rate from filename
            lr_str = os.path.basename(file).replace("log_lr_", "").replace(".csv", "")
            lr = float(lr_str)

            df = pd.read_csv(file)

            if "epoch" not in df.columns or "test_loss" not in df.columns or "accuracy" not in df.columns:
                print(f"Warning: Skipping {file} due to missing columns.")
                continue

            # Plot loss
            ax1.plot(df["epoch"], df["test_loss"], marker=".", linestyle="-", label=f"LR = {lr}")

            # Plot accuracy
            ax2.plot(df["epoch"], df["accuracy"], marker=".", linestyle="-", label=f"LR = {lr}")

        except Exception as e:
            print(f"Could not process file {file}: {e}")

    # Finalize and save Loss plot
    ax1.legend()
    ax1.grid(True)
    loss_plot_path = os.path.join(log_dir, "loss_comparison.png")
    fig1.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    print(f"Loss comparison plot saved to {loss_plot_path}")

    # Finalize and save Accuracy plot
    ax2.legend()
    ax2.grid(True)
    acc_plot_path = os.path.join(log_dir, "accuracy_comparison.png")
    fig2.savefig(acc_plot_path, dpi=150, bbox_inches="tight")
    print(f"Accuracy comparison plot saved to {acc_plot_path}")

    plt.close(fig1)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Plot results from training logs.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the log files.")
    args = parser.parse_args()
    plot_results(args.dir)


if __name__ == "__main__":
    main()
