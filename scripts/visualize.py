from typing import *

#!/usr/bin/env python3
"""
Visualize a saved npz or outputs.
Usage:
  python scripts/visualize.py --file data/train/sample_000000.npz --out out.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--out", default="viz.png")
    args = p.parse_args()
    npz = np.load(args.file)
    u_hr = npz['u_hr']
    u_lr = npz['u_lr']
    bic = zoom(u_lr, zoom=int(u_hr.shape[0]/u_lr.shape[0]), order=3)
    fig, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(u_lr, cmap='viridis'); axs[0].set_title('LR')
    axs[1].imshow(bic, cmap='viridis'); axs[1].set_title('bicubic')
    axs[2].imshow(u_hr, cmap='viridis'); axs[2].set_title('HR truth')
    for ax in axs: ax.axis('off')
    plt.savefig(args.out, dpi=200)
    print("Saved", args.out)

if __name__=='__main__':
    main()

if __name__ == "__main__":
    main()
