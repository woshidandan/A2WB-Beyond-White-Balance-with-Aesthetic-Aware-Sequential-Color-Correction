# common.py
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from typing import Union, Tuple, List
from datetime import datetime
from config import config
from torchmetrics.image import StructuralSimilarityIndexMeasure

def setup_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_device_memory(device: str = "cuda:0") -> dict:
    """检查指定GPU设备的显存使用情况"""
    if "cuda" in device:
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        used = torch.cuda.memory_allocated(device) / 1e9
        free = total - used
        return {
            'total (GB)': round(total, 2),
            'used (GB)': round(used, 2),
            'free (GB)': round(free, 2)
        }
    return {'error': '仅支持CUDA设备'}

def to_device(data: Union[torch.Tensor, dict], device: str):
    """将数据转移到指定设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data

def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    """将张量转换为Numpy数组（自动处理梯度和设备）"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(tensor)

def tensor2pil(tensor: torch.Tensor) -> list:
    """
    将 BCHW 格式的 RGB 张量转换为 PIL 图像列表
    输入张量范围: [0, 1]，形状: [Batch, Channel, Height, Width]
    返回值: 包含 PIL 图像的列表（长度为 Batch）
    """
    if tensor.dim() != 4:
        raise ValueError("输入张量必须是四维的 BCHW 格式")
    
    # 将张量从 GPU 移到 CPU 并解绑梯度
    tensor = tensor.cpu().detach()
    
    # 转换为 0-255 范围的 uint8 类型
    tensor = tensor.mul(255).byte()
    
    # 调整维度顺序为 [Batch, Height, Width, Channel]
    tensor = tensor.permute(0, 2, 3, 1).numpy()
    
    # 生成 PIL 图像列表
    return [Image.fromarray(image) for image in tensor]

def pil2tensor(pil_list: list) -> torch.Tensor:
    """
    将 PIL 图像列表转换为 BCHW 格式的 RGB 张量
    输入要求: PIL 图像列表 (所有图像必须为 RGB 模式且尺寸相同)
    返回值: [0, 1] 范围的 float32 张量，形状为 [Batch, Channel, Height, Width]
    """
    # 转换为 numpy 数组并预处理
    tensor_list = []
    for pil_img in pil_list:
        # 转换为 HWC 格式的 numpy 数组 (范围 0-255)
        np_img = np.array(pil_img.convert("RGB"))  # 强制转换为 RGB 模式
        
        # 转换为 CHW 格式的 float32 张量 (范围 0-1)
        tensor = torch.from_numpy(np_img).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC → CHW
        
        tensor_list.append(tensor.to(config.DEVICE))
    
    # 合并批次维度
    return torch.stack(tensor_list, dim=0)  # CHW → BCHW

# L通道 [0-1] AB通道 [-1, 0.992]
def tensor2normalizedLab(tensor: torch.Tensor) -> torch.Tensor:
    """
    修正后的RGB转LAB归一化函数
    输出范围:
        L: [0, 1]    (对应OpenCV的0~255)
        a: [-1, 1]   (对应-128~127)
        b: [-1, 1]
    """
    return tensor
    tensor = torch.clamp(tensor, 0.0, 1.0)
    device = tensor.device
    
    # 使用浮点计算避免8位精度损失
    np_images = tensor.permute(0, 2, 3, 1).cpu().numpy()  # BxHxWxC
    lab_images = []
    
    for img in np_images:
        # 使用32位浮点转换
        bgr_float = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2Lab)
        
        # 分离并归一化通道
        L = lab[:,:,0] / 100.0  # [0,100]
        a = lab[:,:,1] / 127.0  # [-127, 128]
        b = lab[:,:,2] / 127.0  # [-127, 128]
        
        normalized_lab = np.stack([L, a, b], axis=-1)
        lab_images.append(normalized_lab)
    
    lab_tensor = torch.from_numpy(np.array(lab_images)).permute(0, 3, 1, 2).to(device)
    return lab_tensor.float()

def to_cpu(tensor):
    return tensor.detach().cpu()

def exist_value(tensor, value):
    num_elements = tensor.shape[0]
    for i in range(0, num_elements):
        sum_values = torch.sum(tensor[i] == value)
        if sum_values > 0:
            return True
    return False

def calculate_psnr(img1: Union[torch.Tensor, np.ndarray], 
                  img2: Union[torch.Tensor, np.ndarray],
                  max_val: float = 1.0) -> float:
    """计算PSNR指标，支持Tensor和Numpy输入"""
    if isinstance(img1, torch.Tensor):
        mse = torch.mean((img1 - img2) ** 2).item()
    else:
        mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(max_val) - 10 * np.log10(mse + 1e-8)

def save_images(images: Union[torch.Tensor, np.ndarray], 
               save_dir: str,
               prefix: str = "img",
               max_save: int = 4) -> None:
    """保存图像批次到指定目录"""
    os.makedirs(save_dir, exist_ok=True)
    # 输入 BCHW RGB 0-1
    if isinstance(images, torch.Tensor):
        images = tensor2numpy(images)
    
    for i, img in enumerate(images[:max_save]):
        # print(f"save - img.shape:{img.shape}")
        # print(f"Value range before scaling: min={img.min()}, max={img.max()}")  # 应为 [0,1]
        if img.shape[0] == 3:  # CHW -> HWC
            img = img.transpose(1,2,0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        # cv2.imwrite(
        #     os.path.join(save_dir, f"{prefix}_{i}_{datetime.now().strftime('%H%M%S')}.png"),
        #     cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # )

        cv2.imwrite(
            os.path.join(save_dir, f"{prefix}_{i}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )

def print_action_probabilities(actions):
    import numpy as np
    actions_np = actions.cpu().numpy() if hasattr(actions, 'cpu') else np.array(actions)
    unique, counts = np.unique(actions_np, return_counts=True)
    prob = counts / counts.sum()
    # ret = ('  ' + ',\t'.join([f"{int(u)} / {p:.2f}" for u, p in zip(sorted(unique), prob[unique.argsort()])]))
    ret = (' \t' + ' \t,'.join([f"{int(u)} / {int(p*100)}%" for u, p in zip(sorted(unique), prob[unique.argsort()])]))
    return ret


# 输入 src/dst (rgb, BCHW, [0-1]) 计算PSNR，返回数据格式 tensor 长度B
def compute_psnr(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    计算逐样本PSNR (BCHW格式输入)
    
    参数:
        src (torch.Tensor): 输入张量，形状为[B, C, H, W]，范围[0, 1]
        dst (torch.Tensor): 目标张量，形状与src相同
    
    返回:
        torch.Tensor: PSNR值，形状为[B]
    """
    assert src.shape == dst.shape, "src和dst形状必须相同"
    
    # 计算逐样本MSE
    mse = torch.mean((src - dst) ** 2, dim=(1, 2, 3))  # 沿CHW维度求平均
    
    # 处理极小值保证数值稳定
    eps = 1e-10  # 避免log10(0)
    
    # 计算PSNR (基于MAX=1的公式推导)
    psnr = -10 * torch.log10(mse + eps)
    
    return psnr

# 输入 src/dst (rgb, BCHW, [0-1]) 计算SSIM，返回数据格式 tensor 长度B
def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    简化版SSIM计算 (输入形状[B,C,H,W], 返回形状[B])
    """
    # 初始化SSIM计算器（推荐在设备初始化时创建）
    ssim_calc = StructuralSimilarityIndexMeasure(
        data_range=1.0,       # 输入范围为[0,1]
        gaussian_kernel=True, # 使用高斯滤波
        kernel_size=11,       # 与原实现参数一致
        reduction='none'      # 关闭自动reduce
    ).to(config.DEVICE)
    return ssim_calc(pred, target)











def normalize_lab(lab: np.ndarray) -> np.ndarray:
    """标准化LAB颜色空间到[-1,1]范围"""
    lab = lab.astype(np.float32)
    lab[:, :, 0] = (lab[:, :, 0] / 100.0) * 2 - 1  # L通道 [0,100] -> [-1,1]
    lab[:, :, 1:] = (lab[:, :, 1:] / 127.0)        # A/B通道 [-127,127] -> [-1,1]
    return lab

def denormalize_lab(lab: np.ndarray) -> np.ndarray:
    """反标准化LAB颜色空间到原始范围"""
    lab = lab.copy().astype(np.float32)
    lab[:, :, 0] = (lab[:, :, 0] + 1) * 50         # L通道 [-1,1] -> [0,100]
    lab[:, :, 1:] = lab[:, :, 1:] * 127            # A/B通道 [-1,1] -> [-127,127]
    return lab.astype(np.uint8)

def bgr2lab_tensor(bgr: torch.Tensor) -> torch.Tensor:
    """将BGR张量转换为LAB张量 (范围归一化)"""
    if bgr.dim() == 4:  # Batch处理
        return torch.stack([bgr2lab_tensor(img) for img in bgr])
    
    bgr_np = tensor2numpy(bgr.permute(1,2,0)) * 255
    lab = cv2.cvtColor(bgr_np.astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab_tensor = torch.from_numpy(normalize_lab(lab)).permute(2,0,1)
    return lab_tensor.float()

def lab2bgr_tensor(lab: torch.Tensor) -> torch.Tensor:
    """将LAB张量转换为BGR张量 (范围反归一化)"""
    if lab.dim() == 4:  # Batch处理
        return torch.stack([lab2bgr_tensor(img) for img in lab])
    
    lab_np = tensor2numpy(lab.permute(1,2,0))
    denorm_lab = denormalize_lab(lab_np)
    bgr = cv2.cvtColor(denorm_lab, cv2.COLOR_LAB2BGR)
    return torch.from_numpy(bgr/255.0).permute(2,0,1).float()

def gradient_clip(model: nn.Module, max_norm: float = 1.0):
    """梯度裁剪"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def check_gradients(model: nn.Module) -> dict:
    """检查模型梯度状态"""
    grads = {
        'max': -np.inf,
        'min': np.inf,
        'mean': 0,
        'zero_grads': 0
    }
    total = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.abs()
            grads['max'] = max(grads['max'], grad.max().item())
            grads['min'] = min(grads['min'], grad.min().item())
            grads['mean'] += grad.mean().item()
            total += 1
        else:
            grads['zero_grads'] += 1
    if total > 0:
        grads['mean'] /= total
    return grads

def generate_heatmap(size: Tuple[int, int], 
                    center: Tuple[int, int], 
                    sigma: float = 5.0) -> np.ndarray:
    """生成高斯热力图"""
    xx, yy = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    dist = (xx - center[0])**2 + (yy - center[1])**2
    heatmap = np.exp(-dist / (2 * sigma**2))
    return heatmap / heatmap.max()

def image_normalize(img: np.ndarray) -> np.ndarray:
    """图像归一化到[-1,1]范围"""
    return (img.astype(np.float32) / 127.5) - 1.0

def image_denormalize(img: np.ndarray) -> np.ndarray:
    """图像反归一化到[0,255]范围"""
    return ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

# 测试代码
if __name__ == "__main__":
    # 测试设备内存检查
    print("GPU内存状态:", check_device_memory())
    
    # 测试颜色空间转换
    dummy_img = torch.rand(3, 256, 256)
    lab = bgr2lab_tensor(dummy_img)
    print("LAB张量范围:", lab.min().item(), lab.max().item())
    
    # 测试PSNR计算
    img1 = torch.rand(1, 3, 256, 256)
    img2 = img1 + 0.1*torch.randn_like(img1)
    print("PSNR值:", calculate_psnr(img1, img2))
    
    # 测试梯度检查
    model = nn.Linear(10, 10)
    loss = model(torch.rand(5,10)).sum()
    loss.backward()
    print("梯度统计:", check_gradients(model))