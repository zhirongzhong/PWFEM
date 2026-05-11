from typing import *
"""
"""

import os, glob
import numpy as np
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, data_dir, transform=None, limit=None):
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if limit:
            files = files[:limit]
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        u_hr = data['u_hr'].astype(np.float32)  # (H,H)
        u_lr = data['u_lr'].astype(np.float32)  # (h,h)
        params = data.get('params', None)
        sample = {'u_hr': u_hr, 'u_lr': u_lr}
        if params is not None:
            sample['params'] = params.astype(np.float32)
        if self.transform:
            sample = self.transform(sample)
        return sample
