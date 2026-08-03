"""
引入像素梯度差距
比对前后两张图片的梯度图，计算Loss
实际使用函数: pixel _grad_loss

"""
import torch
import torch.nn.functional as F

def gradient_magnitude(V):
    V = torch.as_tensor(V)  # 将输入转换为PyTorch张量
    B, T, H, W = V.shape

    # 计算x方向的梯度
    V_x = V[:, :, :, :-1] - V[:, :, :, 1:]
    V_x = F.pad(V_x, (0, 1), mode='constant', value=0)

    # 计算y方向的梯度
    V_y = V[:, :, :-1, :] - V[:, :, 1:, :]
    V_y = F.pad(V_y, (0, 0, 0, 1), mode='constant', value=0)

    # 计算梯度的模
    gradient_magnitude_V = torch.sqrt(V_x ** 2 + V_y ** 2 + 1e-6)
    return gradient_magnitude_V


"""
V 和 Vgt 分别是输入(B, C, H, W)的图像
我们经过gradient_magnitude函数计算的梯度分布gradient_magnitude_V和gradient_magnitude_Vgt
计算二者的差的平方, 需要保持形状不变并返回numpy数组
"""
def pixel_grad_loss(V, Vgt):
    B, T, H, W = V.shape

    gradient_magnitude_V = gradient_magnitude(V)
    gradient_magnitude_Vgt = gradient_magnitude(Vgt)

    # 计算梯度差的平方
    loss = (gradient_magnitude_V - gradient_magnitude_Vgt) ** 2

    # 对C通道求和，将形状从(B, T, H, W)转变为(B, H, W)
    loss_summed = torch.sum(loss, dim=1, keepdim=True)  # 在第二维(T通道)上求和
    

    # 将结果转换为NumPy数组
    return loss_summed

def balance_loss(V, Vgt):
    B, T, H, W = V.shape

    gradient_magnitude_V = gradient_magnitude(V)
    gradient_magnitude_Vgt = gradient_magnitude(Vgt)

    # 计算MSE损失
    L_balance = F.mse_loss(gradient_magnitude_V, gradient_magnitude_Vgt)

    return L_balance