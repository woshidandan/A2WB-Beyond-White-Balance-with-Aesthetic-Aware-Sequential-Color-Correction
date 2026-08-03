# net.py（核心网络结构改进）
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    """改进的卷积GRU单元，增强门控机制初始化"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_reset = nn.Conv2d(input_dim+hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_update = nn.Conv2d(input_dim+hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_new = nn.Conv2d(input_dim+hidden_dim, hidden_dim, kernel_size, padding=padding)
        
        # 正交初始化增强门控机制
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 1.0)  # 偏置初始化为1促进门控激活

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        reset_gate = torch.sigmoid(self.conv_reset(combined))
        update_gate = torch.sigmoid(self.conv_update(combined))
        combined_new = torch.cat([x, reset_gate * hidden], dim=1)
        new_state = torch.tanh(self.conv_new(combined_new))
        return (1 - update_gate) * hidden + update_gate * new_state

class FeatureExtractor(nn.Module):
    """增强梯度传播的特征提取器"""
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = self._make_dilated_block(base_dim, base_dim, dilation=2)
        self.conv3 = self._make_dilated_block(base_dim, base_dim, dilation=3)
        self.conv4 = self._make_dilated_block(base_dim, base_dim, dilation=4)
        
        self._init_weights()
    
    def _make_dilated_block(self, in_dim, out_dim, dilation):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_dim, out_dim, 1)  # 1x1卷积调整通道
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        x4 = self.conv4(x3) + x3
        return x1 + x2 + x3 + x4  # 跨层残差连接

class PolicyHead(nn.Module):
    """改进的策略头，增强梯度流动"""
    def __init__(self, feat_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.gru_cell = ConvGRUCell(feat_dim, hidden_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=3, dilation=3),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.action_head = nn.Conv2d(hidden_dim, num_actions, 3, padding=1)
        self._init_conv_weights()

    def _init_conv_weights(self):
        # 仅初始化非GRU的卷积层
        for m in [self.conv1, self.conv2, self.action_head]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden=None):
        if hidden is None:
            b, c, h, w = x.shape
            hidden = torch.zeros(b, x.size(1), h, w).to(x.device)
        
        hidden = self.gru_cell(x, hidden)
        x = self.conv1(hidden) + hidden
        x = self.conv2(x) + x
        return self.action_head(x), hidden

class ValueHead(nn.Module):
    """稳定训练的价值头"""
    def __init__(self, feat_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, 3, padding=3, dilation=3),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            # nn.InstanceNorm2d(1),  # 实例归一化
            # nn.Sigmoid()  # 输出范围[0,1]
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)  # 适合Sigmoid输出的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # 小偏置避免死区
         # 最后一层特殊初始化
        nn.init.xavier_normal_(self.value_head[0].weight, gain=0.1)  # 小增益防止初始输出过大


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        # return self.value_head(x) * 2 - 1  # 缩放至[-1,1]
        return self.value_head(x)

class PixelRL_model(nn.Module):
    """完整模型架构"""
    def __init__(self, num_actions):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channels=3)
        self.policy_head = PolicyHead(64, num_actions)
        self.value_head = ValueHead(64)
    
    def forward(self, x):
        
        lab = x[:, :3]
        prev_hidden = x[:, 3:] if x.size(1) > 3 else None
        
        base_feat = self.feature_extractor(lab)
        pi, new_hidden = self.policy_head(base_feat, prev_hidden)
        v = self.value_head(base_feat)
        return pi, v, new_hidden
    
    def pi_and_v(self, x):
        """用于训练时获取策略和价值"""
        return self.forward(x)
    
    def choose_best_actions(self, state):
        """用于推理时选择最优动作"""
        pi, v, hidden = self.forward(state)
        actions = torch.argmax(pi, dim=1)
        return actions, v, hidden

# 示例用法
if __name__ == "__main__":
    model = PixelRL_model(num_actions=6)
    dummy_input = torch.randn(2, 67, 64, 64)  # 包含隐藏状态的输入
    pi, v, hidden = model(dummy_input)
    print(f"策略输出形状: {pi.shape}")  # 预期: [2, 6, 64, 64]
    print(f"价值输出形状: {v.shape}")    # 预期: [2, 1, 64, 64]