"""Plot main-training losses by stage."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    path = Path(args.run_directory)
    data = np.genfromtxt(path / "history.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for phase in (1, 2, 3, 4):
        subset = data[data["stage"] == phase]
        for ax, key, label in zip(
            axes,
            ("loss", "pde", "boundary"),
            ("Weighted objective", "PDE MSE", "Boundary MSE"),
        ):
            ax.semilogy(subset["updates"], subset[key], lw=0.65, label=f"Stage {phase}")
            ax.set(xlabel="Main Adam updates", ylabel=label)
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.savefig(path / "loss_history.svg")
    fig.savefig(path / "loss_history.png", dpi=180)


if __name__ == "__main__":
    main()
