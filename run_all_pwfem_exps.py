#!/usr/bin/env python3
"""
Run all BayesPWFEM experiments across PDE datasets.
- Supports optional training & testing
- Integrates BayesDL initialization automatically
- Collects metrics into results_summary.csv
"""

import os
import subprocess
import csv
import argparse

# -------------------------------
# CONFIG
# -------------------------------
PDE_LIST = ["poisson", "diffusion", "shock", "frac", "crack"]
BASE_DATA = "data"
RESULTS_FILE = "results_summary.csv"


def run_command(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_last_metrics_npz(path):
    """Read final metrics from npz file"""
    if not os.path.exists(path):
        print(f"[Warning] {path} not found")
        return None
    import numpy as np
    try:
        data = np.load(path, allow_pickle=True)
        metrics = data["metrics"]
        if metrics.ndim == 2:
            mean_metrics = metrics.mean(axis=0)
            return dict(
                mse=mean_metrics[0],
                psnr=mean_metrics[1],
                ssim=mean_metrics[2],
                calib=mean_metrics[3] if metrics.shape[1] > 3 else 0.0,
            )
    except Exception as e:
        print(f"[Error reading {path}]: {e}")
    return None


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; only run testing.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs for training.")
    parser.add_argument("--particles", type=int, default=50,
                        help="Number of particles for SVGD.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # prepare results CSV
    header = ["PDE", "MSE", "PSNR", "SSIM", "Calibration"]
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # loop over PDE datasets
    for pde in PDE_LIST:
        print(f"\n==== Processing {pde.upper()} ====\n")

        checkpoint_dir = f"checkpoints/bayespwfem/{pde}"
        out_dir = f"outputs/{pde}"
        log_dir = f"logs/{pde}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        train_dir = f"{BASE_DATA}/{pde}/train"
        test_dir = f"{BASE_DATA}/{pde}/test"

        if not args.skip_train:
            run_command([
                "python", "BayesPWFEM_train.py",
                "--train_dir", train_dir,
                "--test_dir", test_dir,
                "--epochs", str(args.epochs),
                "--particles", str(args.particles),
                "--checkpoint_dir", checkpoint_dir,
                "--out_dir", out_dir,
                "--log_dir", log_dir,
                "--gpu", str(args.gpu),
                "--batch_size", str(args.batch_size),
                "--prior_weight", "1e-4",
                "--svgd_step", "1e-4",
            ])

        run_command([
            "python", "BayesPWFEM_test.py",
            "--dataset", pde,
        ])

        metrics_npz = os.path.join("outputs", f"metrics_{pde}_PWFEM.npz")
        last_metrics = read_last_metrics_npz(metrics_npz)

        if last_metrics:
            row = [
                pde,
                f"{last_metrics['mse']:.6f}",
                f"{last_metrics['psnr']:.2f}",
                f"{last_metrics['ssim']:.3f}",
                f"{last_metrics['calib']:.4f}",
            ]
            with open(RESULTS_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        else:
            print(f"[Warning] No metrics found for {pde}")

    print(f"\n✅ Summary saved in {RESULTS_FILE}")


if __name__ == "__main__":
    main()
