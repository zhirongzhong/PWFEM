from typing import *
"""
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec


def save_grid_imgs(save_dir, prefix, u_lr, u_hr, bicubic, rec_mean, rec_std=None):
    """
    Save visualization grid for SR results with independent colorbars per row.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Prepare data dimensions ---
    def ensure_3d(x):
        """Ensure input has shape (C, H, W)"""
        if x is None:
            return None
        if x.ndim == 2:
            return x[None, ...]  # (1, H, W)
        elif x.ndim == 3:
            return x  # (C, H, W)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

    # Convert all inputs to consistent 3D format
    u_lr_3d = ensure_3d(u_lr)
    u_hr_3d = ensure_3d(u_hr)
    bicubic_3d = ensure_3d(bicubic)
    rec_mean_3d = ensure_3d(rec_mean)
    rec_std_3d = ensure_3d(rec_std) if rec_std is not None else None

    # Determine number of channels from HR image
    C = u_hr_3d.shape[0]  # number of channels

    # Determine number of columns
    ncols = 5 if rec_std_3d is not None else 4

    fig_width = 4 * ncols + 1.5  # Extra space for colorbars
    fig_height = 4 * C
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(C, ncols + 1,
                  width_ratios=[1] * ncols + [0.08],  # Main images + colorbar column
                  wspace=0.1, hspace=0.1)

    # Titles for each column
    titles = ["LR input", "HR truth", "Bicubic", "Reconstructed mean"]
    if rec_std_3d is not None:
        titles.append("Reconstructed std")

    # Channel labels
    ch_labels = ["Channel 1", "Channel 2", "Channel 3"]
    ch_labels = ch_labels[:C]  # Trim to actual number of channels

    # --- Process each channel (row) independently ---
    for ci in range(C):
        # Extract data for this channel
        channel_data = []
        for data_3d in [u_lr_3d, u_hr_3d, bicubic_3d, rec_mean_3d]:
            if data_3d is not None and ci < data_3d.shape[0]:
                channel_data.append(data_3d[ci])
            else:
                # If data doesn't have this channel, use zeros (shouldn't happen in normal case)
                dummy_shape = u_hr_3d[0].shape if u_hr_3d is not None else (100, 100)
                channel_data.append(np.zeros(dummy_shape))

        if rec_std_3d is not None and ci < rec_std_3d.shape[0]:
            channel_data.append(rec_std_3d[ci])

        # Calculate color range for this row (excluding rec_std if present)
        main_images = channel_data[:4]  # First 4 images share color scale
        vmin_shared = min(img.min() for img in main_images)
        vmax_shared = max(img.max() for img in main_images)

        # Add margin to avoid singular values
        data_range = vmax_shared - vmin_shared
        if data_range == 0:
            vmin_shared -= 0.1
            vmax_shared += 0.1
        else:
            margin = data_range * 0.05
            vmin_shared -= margin
            vmax_shared += margin

        # For rec_std, calculate separate color range
        if rec_std_3d is not None and len(channel_data) > 4:
            vmin_std = channel_data[4].min()
            vmax_std = channel_data[4].max()
            std_range = vmax_std - vmin_std
            if std_range == 0:
                vmin_std -= 0.1
                vmax_std += 0.1
            else:
                margin_std = std_range * 0.05
                vmin_std -= margin_std
                vmax_std += margin_std

        # --- Create subplots for this row ---
        row_images = []  # To store imshow objects for colorbar

        for j in range(ncols):
            ax = fig.add_subplot(gs[ci, j])

            # Use shared color scale for first 4 images, separate for rec_std
            if j < 4:  # LR, HR, bicubic, rec_mean
                im = ax.imshow(channel_data[j], cmap="viridis",
                               vmin=vmin_shared, vmax=vmax_shared)
                if j == 0:  # Store first image for shared colorbar
                    row_images.append(im)
            else:  # rec_std (if present)
                im = ax.imshow(channel_data[j], cmap="viridis",
                               vmin=vmin_std, vmax=vmax_std)
                # Store rec_std image for its own colorbar
                row_images.append(im)

            # Set titles only for top row
            if ci == 0:
                ax.set_title(titles[j], fontsize=11, pad=10)

            ax.axis("off")

        # Add row label on the left
        label_ax = fig.add_subplot(gs[ci, 0])
        label_ax.set_ylabel(ch_labels[ci], fontsize=12, rotation=90, labelpad=20)
        label_ax.axis('off')

        # --- Add colorbar for this row ---
        cax_shared = fig.add_subplot(gs[ci, ncols])  # Last column for colorbar

        if ncols > 4:  # Has rec_std, need two colorbars
            # We'll put both colorbars in the same column, stacked
            # Adjust the grid spec for this row to accommodate two colorbars
            pass

        # Create colorbar for shared images
        if row_images:  # If we have at least one image
            cbar_shared = fig.colorbar(row_images[0], cax=cax_shared, orientation='vertical')
            cbar_shared.set_label("Value", fontsize=9)
            cbar_shared.ax.tick_params(labelsize=8)

    # --- Save figure ---
    fname = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return fname


# Example usage and test function
def test_save_grid_imgs():
    """Test function to demonstrate the modified save_grid_imgs"""
    import numpy as np

    # Create test data
    np.random.seed(42)

    # Test case 1: Single channel (2D input)
    print("Testing single channel (2D) visualization...")
    u_lr_2d = np.random.rand(40, 40)
    u_hr_2d = np.random.rand(160, 160)
    bicubic_2d = np.random.rand(160, 160)
    rec_mean_2d = np.random.rand(160, 160)
    rec_std_2d = np.random.rand(160, 160) * 0.1

    save_grid_imgs("./test_output", "test_single_channel",
                   u_lr_2d, u_hr_2d, bicubic_2d, rec_mean_2d, rec_std_2d)

    # Test case 2: Multi-channel (3D input)
    print("Testing multi-channel (3D) visualization...")
    u_lr_3d = np.random.rand(2, 40, 40)  # 2 channels
    u_hr_3d = np.random.rand(2, 160, 160)
    bicubic_3d = np.random.rand(2, 160, 160)
    rec_mean_3d = np.random.rand(2, 160, 160)
    rec_std_3d = np.random.rand(2, 160, 160) * 0.1

    save_grid_imgs("./test_output", "test_multi_channel",
                   u_lr_3d, u_hr_3d, bicubic_3d, rec_mean_3d, rec_std_3d)


if __name__ == "__main__":
    # Run test
    test_save_grid_imgs()