# data_loader.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import config
from typing import Tuple, List
import random

class AWBDataset(Dataset):
    """白平衡强化学习数据集"""
    
    def __init__(self, mode: str = 'train'):
        super().__init__()
        self.mode = mode
        self.pairs = self._load_pairs()
        
        # 数据增强参数
        self.augment_prob = 0.7 if mode == 'train' else 0.0
        self.flip_augment = False
        self.color_augment = False

    def _load_pairs(self) -> List[Tuple[str, str]]:
        """加载图像路径对"""
        base_dir = config.TRAIN_LABEL_DIR if self.mode == 'train' else config.TEST_LABEL_DIR
        
        src_list = []
        with open(os.path.join(base_dir, "cast.txt"), 'r') as f:
            for line in f:
                src_list.append(line.strip())
                
        dst_list = []
        with open(os.path.join(base_dir, "awb.txt"), 'r') as f:
            for line in f:
                dst_list.append(line.strip())
        
        pairs = []
        for src_name, dst_name in zip(src_list, dst_list):
            src_path = os.path.join(config.SRC_IMAGE_DIR, src_name)
            dst_path = os.path.join(config.DST_IMAGE_DIR, dst_name)
            if os.path.exists(src_path) and os.path.exists(dst_path):
                pairs.append((src_path, dst_path))
            else:
                print(f"Missing pair: SRC-{src_path} | DST-{dst_path}")
        return pairs

    def _load_image(self, path: str) -> np.ndarray:
        """直接加载图像"""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        return img

    def _preprocess(self, src_img: np.ndarray, dst_img: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """核心预处理流程"""
        # 尺寸验证
        if src_img.shape != dst_img.shape:
            src_img = cv2.resize(src_img, (dst_img.shape[1], dst_img.shape[0]))
            
        # 数据增强
        # if self.mode == 'train' and random.random() < self.augment_prob:
        #     src_img, dst_img = self._apply_augmentations(src_img, dst_img)
        
        # 转换张量
        src_tensor = self._img2tensor(src_img)
        dst_tensor = self._img2tensor(dst_img)
        
        return src_tensor, dst_tensor

    def _apply_augmentations(self, src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用数据增强"""
        # 随机水平翻转
        if self.flip_augment and random.random() > 0.5:
            src = cv2.flip(src, 1)
            dst = cv2.flip(dst, 1)
            
        # 颜色扰动
        if self.color_augment:
            src = self._color_jitter(src)
            
        return src, dst

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """颜色抖动增强"""
        # 亮度调整
        if random.random() > 0.8:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = np.clip(v * random.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
            img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
            
        # 对比度调整
        if random.random() > 0.8:
            img = np.clip(img * random.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
            
        return img

    def _img2tensor(self, img: np.ndarray) -> torch.Tensor:
        """图像转张量标准化"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """直接返回原始路径信息"""
        src_path, dst_path = self.pairs[idx]
        src_img = self._load_image(src_path)
        dst_img = self._load_image(dst_path)
        src_tensor, dst_tensor = self._preprocess(src_img, dst_img)
        return src_tensor, dst_tensor, src_path, dst_path

def get_dataloader(batch_size: int = None, mode: str = 'train') -> DataLoader:
    """获取数据加载器"""
    dataset = AWBDataset(mode=mode)
    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=(mode == 'train'),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

# 示例用法
if __name__ == "__main__":
    train_loader = get_dataloader(mode='train')
    for batch_idx, (src, dst, src_paths, dst_paths) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  SRC shape: {src.shape} | DST shape: {dst.shape}")
        print(f"  Sample paths: {src_paths[0]} -> {dst_paths[0]}")
        if batch_idx >= 2:
            break