from typing import *
"""
dataset_synthesis.py: module for the Bayesian PWFEM / ProbSR project.
Auto-documented for open-source release.
"""

#!/usr/bin/env python3
"""
Run experiments on multiple PDE benchmarks:
- Poisson
- Multiscale diffusion
- Shock waves
- Fractional Laplacian
- Allen–Cahn phase field

Workflow:
1. (optional) Generate datasets (train/test) via run_pde_exp.py
2. (optional) Train model via run_exp.py
3. Collect last epoch metrics from metrics.csv
4. Save summary into results_summary.csv
"""

import os, subprocess, csv, argparse


PDE_LIST = ["poisson", "diffusion", "shock", "frac", "crack"]

def run_command(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def read_last_line_csv(path):
    """Read the last line of a CSV file as dict"""
    if not os.path.exists(path):
        print(f"[Warning] metrics.csv not found: {path}")
        return None
    with open(path, "r") as f:
        reader = list(csv.DictReader(f))
        if len(reader) == 0:
            return None
        return reader[-1]  # last epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip data generation & training, only collect metrics")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train if not skipping")
    parser.add_argument("--particles", type=int, default=50,
                        help="Number of particles for SVGD if not skipping")
    parser.add_argument("--skip_datagen", type=bool, default=False,
                        help="Skip data generation & training, only collect metrics")
    args = parser.parse_args()
    base_data = "data"
    results_file = "results_summary.csv"

    # prepare results CSV
    header = ["PDE","epoch","mse_bic","mse_prob","psnr_bic","psnr_prob","ssim_bic","ssim_prob"]
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for pde in PDE_LIST:
        print(f"\n==== Processing {pde.upper()} ====\n")

        checkpoint_dir = f"checkpoints/{pde}"
        out_dir = f"outputs/{pde}"
        log_dir = f"logs/{pde}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        if not args.skip_train:
            # 1. 数据生成
            if not args.skip_datagen:
                run_command([
                    "python", "pde_data_gen.py",
                    "--pde", pde,
                    "--out_dir", base_data,
                    "--hr_side", "160",
                    "--lr_side", "40",
                    "--up", "4",
                    "--train_n", "500",
                    "--test_n", "100",
                    "--seed", "42"
                ])

if __name__ == "__main__":
    main()
