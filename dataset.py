import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset

class AugmentedTensorDataset(TensorDataset):
    def __init__(self, tensors, noise_vector=None, nudge=0.1, augment_prob=1.0, clip_range=None, seed=None, device=torch.device('cpu')):
        super().__init__(*tensors)
        assert len(tensors) >= 1, "Expect at least (features, targets)"
        self.x = tensors[0]
        self.y = tensors[1] if len(tensors) > 1 else None

        self.device = device
        self.noise_vector = None
        if noise_vector is not None:
            nv = np.asarray(noise_vector, dtype=np.float32)
            if nv.ndim != 1:
                raise ValueError("noise_vector must be 1D")
            if nv.shape[0] != self.x.shape[1]:
                raise ValueError(f"noise_vector length {nv.shape[0]} != num_features {self.x.shape[1]}")
            self.noise_vector = torch.tensor(nv, dtype=torch.float32, device=device)

        self.nudge = float(nudge)
        self.augment_prob = float(augment_prob)
        self.clip_range = clip_range
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx] if self.y is not None else None

        if (self.noise_vector is not None) and (self.augment_prob > 0.0):
            if self.rng is not None:
                do_aug = self.rng.rand() < self.augment_prob
            else:
                do_aug = (np.random.rand() < self.augment_prob)

            if do_aug:
                std = (self.noise_vector * self.nudge).to(x.device)
                noise = torch.randn_like(x) * std
                x = x + noise
                if self.clip_range is not None:
                    lo, hi = self.clip_range
                    x = torch.clamp(x, min=lo, max=hi)

        return (x, y) if y is not None else (x,)

def load_noise_vector_from_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Noise JSON not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    arr = np.asarray(data, dtype=np.float32)
    return arr
