import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl


class MyDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        transform: Optional[Callable] = None,
        patch_size: int = 64,
        label_csv: str = None,
    ):
        self.data_dir = Path(data_path)
        self.transform = transform
        self.patch_size = patch_size
        self.split = split

        self.imgs: List[Path] = sorted([f for f in self.data_dir.iterdir() if f.suffix.lower() == ".tif"])

        self.labels = {}
        if label_csv:
            label_df = pd.read_csv(label_csv)
            for _, row in label_df.iterrows():
                xid = str(row["xid"])
                self.labels[xid] = (float(row["Z"]), float(row["Z_ERR"]))

        # 划分训练/验证
        train_split = 0.9
        split_idx = int(len(self.imgs) * train_split)
        if split == "train":
            self.imgs = self.imgs[:split_idx]
        elif split == "val":
            self.imgs = self.imgs[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        # 收集红移（仅 Z）用于统计 & 过采样
        self.z_values = [self._get_z_from_filename(p)[0] for p in self.imgs]
        self.z_mean = float(np.mean(self.z_values)) if len(self.z_values) else 0.0
        self.z_std = float(np.std(self.z_values)) if len(self.z_values) else 1.0

        print(f"{split} dataset: {len(self.imgs)} images")
        print(f"Redshift stats: mean={self.z_mean:.4f}, std={self.z_std:.4f}")

    def _get_z_from_filename(self, img_path: Path) -> Tuple[float, float]:
        # 000123_proc.tif -> xid = 123 -> 映射到 (Z, Z_ERR)
        filename = img_path.stem
        xid = filename.split("_")[0].lstrip("0")
        if xid == "":
            xid = "0"
        return self.labels.get(xid, (0.0, 0.0))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img_path = self.imgs[idx]
        img = tifffile.imread(img_path).astype(np.float32)

        # 如果是单通道 (H, W)，让它有一个 channel 维度 (H, W, 1)
        if img.ndim == 2:
            img = img[:, :, None]

        z, z_err = self._get_z_from_filename(img_path)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]  # (C, H, W) torch.float32

        # 预训练阶段不需要 z，但我们仍返回（便于之后微调/监控）
        return img, torch.tensor(z, dtype=torch.float32), torch.tensor(z_err, dtype=torch.float32)


class VAEDataset(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        patch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
        label_csv: str = None,
        # ------- 过采样超参 -------
        oversample_power: float = 0.5,
        oversample_max_ratio: float = 25.0,
        epoch_len_factor: float = 1.0,
        use_oversample: bool = True,
        # >>> 新增：兼容 configs/vae.yaml 里的旧键 <<<
        use_balanced_sampler: Optional[bool] = None,
        # >>> 新增：吞掉以后可能出现的其它键，避免再报 unexpected kwarg <<<
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.label_csv = label_csv
        self.use_oversample = use_oversample

        self.oversample_power = oversample_power
        self.oversample_max_ratio = oversample_max_ratio
        self.epoch_len_factor = epoch_len_factor
        self.use_oversample = use_oversample

        self.train_dataset: Optional[MyDataset] = None
        self.val_dataset: Optional[MyDataset] = None
        
    def setup(self, stage: Optional[str] = None):

        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.CenterCrop(height=self.patch_size, width=self.patch_size),
            A.Resize(height=self.patch_size, width=self.patch_size),
            ToTensorV2(),
        ])

        val_transforms = A.Compose([
            A.CenterCrop(height=self.patch_size, width=self.patch_size),
            A.Resize(height=self.patch_size, width=self.patch_size),
            ToTensorV2(),
        ])

        self.train_dataset = MyDataset(
            data_path=self.data_path,
            split="train",
            transform=train_transforms,
            patch_size=self.patch_size,
            label_csv=self.label_csv,
        )
        self.val_dataset = MyDataset(
            data_path=self.data_path,
            split="val",
            transform=val_transforms,
            patch_size=self.patch_size,
            label_csv=self.label_csv,
        )


    def _build_weights(self) -> torch.Tensor:
        z = torch.tensor(self.train_dataset.z_values, dtype=torch.float32)
        # 你的全量分箱：0.0-0.1, ..., 0.6-0.7
        edges = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32)

        bins = torch.bucketize(z, edges, right=False) - 1
        bins = bins.clamp(0, len(edges) - 2)

        counts = torch.bincount(bins, minlength=len(edges) - 1).float()
        freq = counts / counts.sum()

        # bin 权重 ∝ 1 / freq^α
        w_bin = torch.pow(1.0 / torch.clamp(freq, 1e-12), self.oversample_power)

        # 限制最大/最小比，避免极端过采样
        ratio = w_bin / w_bin.min()
        w_bin = torch.where(ratio > self.oversample_max_ratio, w_bin.min() * self.oversample_max_ratio, w_bin)

        weights = w_bin[bins].double().clamp(min=1e-6)
        return weights

    def train_dataloader(self):
        if self.use_oversample:
            weights = self._build_weights()
            num_samples = int(self.epoch_len_factor * len(self.train_dataset))
            sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                sampler=sampler,       # 用 sampler 时不要 shuffle
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
