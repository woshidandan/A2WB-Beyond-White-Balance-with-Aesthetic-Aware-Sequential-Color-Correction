# main.py
import os
import sys
import signal
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from config import config
from data_loader import get_dataloader
from net import PixelRL_model
from trainer import PixelRLTrainer
from common import setup_seed, check_device_memory
from config import config

# os.chdir('/home/yzf/2025/p5-不用cache') # debug时使用

class TrainingSession:
    """封装训练会话状态"""
    def __init__(self, args):
        self.args = args
        self.model = None
        self.optimizer = None
        self.trainer = None
        self.current_step = 0
        self._init_environment()
        
    def _init_environment(self):
        """初始化信号处理和随机种子"""
        def save_on_exit(signum, frame):
            print("\n捕获中断信号，正在保存模型...")
            self._save_checkpoint(emergency=True)
            sys.exit(0)
            
        signal.signal(signal.SIGINT, save_on_exit)
        setup_seed(42)
    
    def _build_model(self):
        """模型构建逻辑"""
        model = PixelRL_model(config.N_ACTIONS)
        return model.to(config.DEVICE)
    
    def _save_checkpoint(self, emergency=False):
        os.makedirs(config.SAVE_ROOT, exist_ok=True)
        """保存检查点"""
        if self.model is None:
            return
            
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.current_step
        }
        
        path = config.model_save_path.replace(".pt", "_emergency.pt") if emergency else \
             os.path.join(config.SAVE_ROOT, f"checkpoint_step_{self.cur_index}.pt")
        
        torch.save(checkpoint, path)
        print(f"检查点已保存至: {path}")
    
    def run(self):
        """主训练流程"""
        # 初始化组件
        self.model = self._build_model()
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.module.feature_extractor.parameters() if config.MULTI_GPU else self.model.feature_extractor.parameters(), 
             'lr': config.LEARNING_RATES['backbone']},
            {'params': self.model.module.value_head.parameters() if config.MULTI_GPU else self.model.value_head.parameters(),
             'lr': config.LEARNING_RATES['value_head']},
            {'params': self.model.module.policy_head.parameters() if config.MULTI_GPU else self.model.policy_head.parameters(),
             'lr': config.LEARNING_RATES['policy_head']}
        ], ** config.OPTIMIZER_PARAMS)
        
        # 初始化训练器
        print(f"准备初始化： {config.DEVICE}")
        self.trainer = PixelRLTrainer(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=get_dataloader(mode='train'),
            val_loader=get_dataloader(mode='test'),
            writer=SummaryWriter(log_dir=config.TENSORBOARD_DIR),
            device=config.DEVICE
        )
        
        # 恢复检查点
        if self.args.resume:
            self._load_checkpoint()
        
        print("数据准备完成, 开始训练")
        # 主训练循环
        try:
            self.cur_index = 0
            iter_size = len(self.trainer.train_loader)
            for step in range(self.current_step, config.TRAIN_STEPS):    
                self.trainer.train_iter = iter(self.trainer.train_loader)
                for iterration in range(iter_size):
                    self.cur_index = step * iter_size + iterration
                    self.current_step = step
                    metrics = self.trainer.train_step(self.cur_index)

                    # 防止因图像尺寸过大跳过训练，返回0报错
                    if isinstance(metrics, dict):
                        self.trainer.writer.add_scalars('train_metrics', metrics, self.cur_index)

                    # 定期保存和验证
                    if (self.cur_index + 1) % config.SAVE_INTERVAL == 0:
                        self._save_checkpoint()
                    # if (self.cur_index + 1) % config.VAL_INTERVAL == 0:
                    #     val_metrics = self.trainer.validate()
                    #     self.trainer.writer.add_scalars('val_metrics', val_metrics, self.cur_index)
                    
        except Exception as e:
            print(f"训练异常终止: {str(e)}")
            self._save_checkpoint(emergency=True)
            raise e
    
    def _load_checkpoint(self):
        """加载检查点"""
        checkpoint = torch.load(self.args.resume, map_location=config.DEVICE)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_step = checkpoint['step']
        print(f"成功从检查点恢复: {self.args.resume}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PixelRL 训练主程序')
    parser.add_argument('--resume', type=str, default=None,
                       help='从指定检查点恢复训练')
    return parser.parse_args()

def main():
    args = parse_args()
    session = TrainingSession(args)
    session.run()

if __name__ == "__main__":
    main()