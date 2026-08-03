# trainer.py
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import config
from train_state import StateManager
from net import PixelRL_model
from common import *
from action.aesReward import *
from action.gradLoss import *

class PixelRLTrainer:
    """强化学习训练器，实现A3C算法"""
    
    def __init__(self, 
                 model: PixelRL_model,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 writer: SummaryWriter,
                 device: str = "cuda:0"):
        """
        Args:
            model: 待训练的强化学习模型
            optimizer: 优化器实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            writer: TensorBoard日志写入器
            device: 训练设备
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.device = device
        
        # 初始化状态管理器
        self.state_mgr = StateManager(self.device)

        # 设置美学评估模型
        self.aes_processor = AestheticProcessor()

        # 设置训练迭代器
        self.train_iter = iter(self.train_loader)  # 持久化迭代器
        
        # 训练状态跟踪
        self.current_step = 0
        self.best_metric = -np.inf
        
        # 经验回放缓存
        self.buffer = {
            'log_probs': [],
            'values': [],
            'entropies': [],
            'rewards': [],
            'masks': []
        }

        # loss权重
        self.policy_loss_weight = config.POLICY_LOSS_COEF
        self.value_loss_weight = config.VALUE_LOSS_COEF
        
        self.entropy_loss_weight = config.ENTROPY_COEF
        print(f"本次训练使用:{self.device}")

    def train_step(self, global_step: int) -> Dict[str, float]:
        """执行单步训练"""
        self.model.train()
        batch = next(self.train_iter)
        src_tensor, dst_tensor = self._preprocess_batch(batch) # BCHW RGB [0-1]

        # 更新超参数 熵的权重除10
        if (global_step) % 1000 == 0:
            self.update_super_para()

        # 过滤过大尺寸的图像
        if src_tensor.shape[2] * src_tensor.shape[3] > 534 * 356:
            print(f"当前图像尺寸太大(大于 534 * 356)，跳过")
            return 0
        
        # 初始化环境状态
        state = self.state_mgr.reset(src_tensor, dst_tensor)
        # input(f"dst_tensor:{self.state_mgr.dst_tensor.shape} {self.state_mgr.dst_tensor}")
        dst_lab = tensor2normalizedLab(self.state_mgr.dst_tensor)
        # input(f"more lab:{dst_lab.shape} {dst_lab}")
        # 多步经验收集
        print(f"epo:{global_step} shape:{dst_tensor.shape}")
        for t in range(config.T_MAX):
            pi, value, hidden = self.model.pi_and_v(state)
            actions = self._select_actions(pi)
            
            # 执行动作并获取奖励
            next_state = self.state_mgr.step(actions, hidden)

            reward = self._calculate_reward(state, next_state, dst_lab)
            print(f"\t\ttrain - t:{t} reward:{torch.mean(reward).item():.4f}\t\t {print_action_probabilities(actions)}")
            
            # 存储经验
            self._store_experience(pi, value, reward, done=(t == config.T_MAX-1))
            state = next_state
        
        
        # 获取最后一个状态的value
        with torch.no_grad():
            _, next_value, _ = self.model.pi_and_v(state)
        
        # 计算并应用梯度
        loss = self._compute_a3c_loss(next_value)
        self.writer.add_scalar('Loss/train', loss.item(), global_step)
        self._update_model(loss)
        
        # 记录指标
        metrics = {
            'total_loss': loss.item(),
            'avg_reward': torch.mean(torch.stack(self.buffer['rewards'])).item(),
            'psnr': compute_psnr(self.state_mgr.image_state[:, 0:3], dst_tensor).mean().item()
        }
        
        # 定期保存样本
        if global_step % config.IMAGE_SAVE_INTERVAL == 0:
            self._save_debug_images(global_step, self.state_mgr.image_state, self.state_mgr.dst_tensor)
        
        self._clear_buffer()
        return metrics

    def _preprocess_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """预处理数据批次"""
        src, dst, src_path, dst_path = batch
        save_path = os.path.join(config.LOG_DIR, "test")

        save_images(to_cpu(src), save_path, prefix="src")
        save_images(to_cpu(dst), save_path, prefix="dst")
        print(f"处理:{src_path}")
        # input(f"数据:{src}")
        return src.to(self.device), dst.to(self.device)

    def _select_actions(self, pi: torch.Tensor) -> torch.Tensor:
        """根据策略分布选择动作
        Args:
            pi: 策略头输出张量，形状为 (B, A, H, W)
                B: 批次大小 | A: 动作数 | H: 图像高度 | W: 图像宽度
        Returns:
            actions: 采样后的动作索引，形状为 (B, H, W)
        """
        # 确保输入维度正确
        assert pi.dim() == 4, f"输入张量应为4D (B, A, H, W)，当前维度为 {pi.dim()}"
        
        # 计算动作概率分布
        probs = torch.softmax(pi, dim=1)  # 沿动作维度归一化，形状保持 (B, A, H, W)
        
        # 调整维度顺序以适应Categorical分布要求
        # 转换为 (B, H, W, A)，使得每个空间位置独立采样
        probs_permuted = probs.permute(0, 2, 3, 1)
        
        # 创建分类分布
        action_dist = torch.distributions.Categorical(probs_permuted)
        
        # 采样动作 (自动处理为torch.int64类型)
        actions = action_dist.sample()  # 形状 (B, H, W)
        
        # 确保动作张量在CPU（如需存储）且数据类型正确
        actions = actions.to(device=pi.device, dtype=torch.long)
        
        # 存储对数概率和熵（保持原始空间结构）
        # input(f"lp:{action_dist.log_prob(actions).shape}")
        # input(f"en:{action_dist.entropy().shape}")
        self.buffer['log_probs'].append(action_dist.log_prob(actions))  # (B, H, W)
        self.buffer['entropies'].append(action_dist.entropy())         # (B, H, W)
        
        return actions

    def _calculate_reward(self, 
                         state: torch.Tensor, 
                         next_state: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
        """
        # 各自不同
        rebuild_reward:torch.Size([1, 1, 250, 376]) mean:0.1968

        # 全部统一
        aes_shape:torch.Size([1, 1, 250, 376]) mean:1.6477
        arti_loss:torch.Size([1, 1, 250, 376]) mean:0.1483
        psnr_reward:torch.Size([1]) mean:1.7771
        ssim_reward:torch.Size([]) mean:-0.2582
        """
        

        current_lab = state[:, :3]    # (B,3,H,W) RGB [0-1] 
        next_lab = next_state[:, :3]  # (B,3,H,W) RGB [0-1]
        target_lab = target           # (B,3,H,W) RGB [0-1]
        # input(f"target:{target_lab.shape} ,min:{torch.mean(target_lab)}, max:{torch.max(target_lab)}")

        # 重建奖励
        curr_distance = torch.square(current_lab - target_lab)
        next_distance = torch.square(next_lab - target_lab)
        rebuild_reward = torch.mean(curr_distance - next_distance, dim=1, keepdim=True) * 5 # B 1 H W
        # input(f"rebuild_reward:{rebuild_reward.shape} mean:{torch.mean(rebuild_reward).item():.4f}")
        
        # 美学奖励
        aes_reward = self.aes_processor.getAesImg2SubImg1(current_lab, next_lab).to(config.DEVICE)
        # input(f"aes_shape:{aes_reward.shape} mean:{torch.mean(aes_reward).item():.4f}")
        
        # 伪影损失
        arti_loss = pixel_grad_loss(next_lab, target_lab).to(config.DEVICE)
        # input(f"arti_loss:{arti_loss.shape} mean:{torch.mean(arti_loss).item():.4f}")
        
        # PSNR奖励
        curr_psnr = compute_psnr(current_lab, target_lab)
        next_psnr = compute_psnr(next_lab, target_lab)
        psnr_reward = next_psnr - curr_psnr
        # input(f"psnr_reward:{psnr_reward.shape} mean:{torch.mean(psnr_reward).item():.4f}")
        
        # SSIM奖励
        curr_ssim = compute_ssim(current_lab, target_lab)
        next_ssim = compute_ssim(next_lab, target_lab)
        ssim_reward = next_ssim - curr_ssim
        # input(f"ssim_reward:{ssim_reward.shape} mean:{torch.mean(ssim_reward).item():.4f}")
        
        # 扩展到空间维度 (B,1,H,W)
        # reward = reward.view(-1,1,1,1).expand(-1,1,*current_lab.shape[-2:])
        

        reward = rebuild_reward + 0.1 * aes_reward
        self.buffer['rewards'].append(reward)
        return reward

    def _store_experience(self, 
                         pi: torch.Tensor,
                         value: torch.Tensor,
                         reward: torch.Tensor,
                         done: bool):
        """存储经验到缓冲区"""
        self.buffer['values'].append(value)
        self.buffer['masks'].append(1.0 - done)

    def _compute_a3c_loss(self, next_value: torch.Tensor) -> torch.Tensor:
        """计算A3C算法的总损失（整合优势函数改进）"""
        returns = next_value  # 初始化为最后一个状态的value估计
        advantages = torch.zeros_like(returns)
        policy_loss = 0.0
        policy_entropy_loss = 0.0
        value_loss = 0.0

        # 反向遍历所有时间步（从最后一步到第一步）
        print(f"计算loss len(rewards):{len(self.buffer['rewards'])} len(masks):{len(self.buffer['masks'])}")
        for t in reversed(range(len(self.buffer['rewards']))):
            # ================= 计算回报 =================
            # 公式：R_t = r_t + γ * R_{t+1}
            returns = self.buffer['rewards'][t] + config.GAMMA * returns
            # ================= 计算TD误差 =================
            td_error = returns - self.buffer['values'][t]
            # ================= 计算优势函数 =================
            advantages = td_error.detach()
            # ================= 策略损失计算 =================
            # input(f"adv:{advantages.shape}")                    # B1HW
            # input(f"prob:{self.buffer['log_probs'][t].shape}")  # BHW
            # input(f"ent:{self.buffer['entropies'][t].shape}")   # BHW
            policy_loss -= self.buffer['log_probs'][t] * advantages.view_as(self.buffer['log_probs'][t])  # 保持维度一致后取平均
            policy_entropy_loss -= self.buffer['entropies'][t]
            # ================= 价值损失计算 =================
            value_loss += torch.square(td_error) / 2

        # ================= 总损失合成 =================
        # 调试打印（建议保留）
        total_loss = torch.nanmean(
            policy_loss * self.policy_loss_weight +
            value_loss * self.value_loss_weight +
            policy_entropy_loss * self.entropy_loss_weight
        )
        print(
            f"[Loss Components]\n"
            f"policy: {torch.mean(policy_loss).item():.4f} (x {self.policy_loss_weight})\n"
            f"value: {torch.mean(value_loss).item():.4f} (x {self.value_loss_weight})\n"
            f"entropy: {torch.mean(policy_entropy_loss).item():.4f} (x {self.entropy_loss_weight})\n"
            f"Total: {total_loss.item():.4f}"
        )
        return total_loss
    
    def _update_model(self, loss: torch.Tensor):
        """执行反向传播和参数更新"""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)  # 使用新配置

        # 打印所有梯度信息
        if False:
            print("\n=== 梯度信息 ===")
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print(f"参数: {name} 无梯度")
                    continue
                    
                grad = param.grad.data
                shape = tuple(grad.shape)
                norm = grad.norm().item()
                mean = grad.mean().item()
                
                # 只打印关键指标
                print(f"‖grad‖: {norm:.4f} | μ: {mean:.6f}\t\t" f"{name.ljust(25)} shape: {str(shape).ljust(15)} " )
            print("===============\n")

        self.optimizer.step()

    def validate(self) -> Dict[str, float]:
        print(f"eval:{len(self.val_loader)}个数据")
        """在验证集上评估模型"""
        self.model.eval()
        total_psnr = 0.0
        total_reward = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中"):
                src, dst = self._preprocess_batch(batch)

                state = self.state_mgr.reset(src, dst)
                for t in range(config.T_MAX):
                    pi, value, hidden = self.model.pi_and_v(state)
                    actions = torch.argmax(pi, dim=1)
                    next_state = self.state_mgr.step(actions, hidden)
                    reward = self._calculate_reward(state, next_state, dst)
                    
                    state = next_state
                    total_reward += reward.mean().item()
                
                total_psnr += compute_psnr(self.state_mgr.image_state[:, :3], dst).mean().item()
        
        metrics = {
            'val_psnr': total_psnr / len(self.val_loader),
            'val_reward': total_reward / len(self.val_loader)
        }
        return metrics

    def save_checkpoint(self, 
                       step: int, 
                       is_best: bool = False, 
                       emergency: bool = False):
        """保存训练状态检查点"""
        state = {
            'step': step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric
        }
        
        if emergency:
            path = os.path.join(config.SAVE_ROOT, "emergency.pt")
        else:
            path = config.model_save_path.replace(".pt", f"_step{step}.pt")
        
        torch.save(state, path)
        print(f"检查点已保存至：{path}")
        
        if is_best:
            best_path = os.path.join(config.SAVE_ROOT, "best_model.pt")
            torch.save(state, best_path)

    def _save_debug_images(self, 
                          step: int, 
                          state: torch.Tensor,
                          target: torch.Tensor):
        """保存调试图像到TensorBoard"""
        with torch.no_grad():
            # 生成样本图像
            gen_imgs = state.cpu()
            target_imgs = target.cpu()
            
            # 保存到日志目录
            save_path = os.path.join(config.LOG_DIR, "samples", f"step_{step}")
            save_images(gen_imgs, save_path, prefix="generated")
            save_images(target_imgs, save_path, prefix="target")
            
            # 写入TensorBoard
            # self.writer.add_images("Generated", gen_imgs, step)
            # self.writer.add_images("Target", target_imgs, step)

    def _clear_buffer(self):
        """清空经验回放缓存"""
        for key in self.buffer:
            self.buffer[key].clear()
    
    def update_super_para(self):
        self.entropy_loss_weight *= 0.1
        print(f"当前config.ENTROPY_COE:{self.entropy_loss_weight}")