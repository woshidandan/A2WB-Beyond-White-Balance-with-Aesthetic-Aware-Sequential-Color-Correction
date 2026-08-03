# config.py
import os
import torch
from datetime import datetime

class Config:
    """Central configuration for PixelRL project"""
    
    # ==================== Hardware ====================
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    MULTI_GPU = False  # 是否启用多GPU训练
    
    # ==================== Path Settings ====================
    # 原始数据路径
    BASE_DATA_DIR = "/home/yzf/2025/enhance_dataset"
    SRC_IMAGE_DIR = os.path.join(BASE_DATA_DIR, "cast")    # 色偏图像
    DST_IMAGE_DIR = os.path.join(BASE_DATA_DIR, "awb")     # 目标图像
    
    # 训练/测试标签文件
    TRAIN_LABEL_DIR = os.path.join(BASE_DATA_DIR, "train")
    TEST_LABEL_DIR = os.path.join(BASE_DATA_DIR, "test")
    
    # 模型存储路径
    SAVE_ROOT = "./checkpoints"
    MODEL_PREFIX = "PixelRL_awb"
    
    @property
    def model_save_path(self):
        """动态生成模型保存路径"""
        os.makedirs(self.SAVE_ROOT, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            self.SAVE_ROOT, 
            f"{self.MODEL_PREFIX}_{timestamp}.pt"
        )
    
    # ==================== Training Hyperparams ====================
    BATCH_SIZE = 1           # 输入批次大小 
    TRAIN_STEPS = 100000      # 总训练步数
    SAVE_INTERVAL = 100      # 模型保存间隔
    VAL_INTERVAL = 1000        # 验证集评估间隔
    IMAGE_SAVE_INTERVAL = 50  # 定期查看校正效果
    
    # 学习率配置（分层）
    LEARNING_RATES = {
        'backbone': 1e-4,     # 主干网络学习率
        'value_head': 1e-5,   # 价值头学习率
        'policy_head': 1e-5   # 策略头学习率
    }
    
    # 优化器参数
    OPTIMIZER_PARAMS = {
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999)
    }
    
    # ==================== RL Parameters ====================
    GAMMA = 0.9               # 奖励折扣因子
    T_MAX = 5                # 多步回报步长
    N_ACTIONS = 6             # 可选动作数量
    
    # ==================== Image Processing ====================
    IMG_MEAN = [0.485, 0.456, 0.406]  # BGR均值
    IMG_STD = [0.229, 0.224, 0.225]   # BGR标准差



    # ==================== 奖励固定权重配置 ====================
    REWARD_AES_COEF = 0.1


    # ==================== RL Loss ====================
    POLICY_LOSS_COEF = 1.5
    VALUE_LOSS_COEF = 1
    ENTROPY_COEF = 1
    GRAD_CLIP = 0.95
    
    # ==================== Logging ====================
    LOG_DIR = "./logs"
    TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")
    CSV_LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")
    
    def __init__(self):
        """自动创建必要目录"""
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.TENSORBOARD_DIR, exist_ok=True)

# 单例配置实例
config = Config()

# 示例用法：
if __name__ == "__main__":
    print("Device:", config.DEVICE)
    print("Model save path:", config.model_save_path)
    print("Learning rates:", config.LEARNING_RATES)