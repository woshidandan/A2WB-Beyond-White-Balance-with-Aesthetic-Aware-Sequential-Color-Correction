<div align="center">

# A²WB

### Aesthetic-Aware Sequential Color Correction

**匿名作者**

匿名机构

[![Paper](https://img.shields.io/badge/Paper-ACM%20MM%202026-8A2BE2.svg)](ACM_MM_2026_A2WB_camera_ready.pdf)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

[[English](README.md)] · **简体中文**

</div>

A²WB 将自动白平衡重新建模为序列化的美学感知色彩校正问题。一个轻量级策略网络在联合的「保真度–美学」目标下，以迭代方式为每个像素选择 6 类细化动作之一（No-op、Denoise、Enhance、Colder、Warmer、Low-light）。本仓库提供第二阶段训练、单张图像推理以及环境/模型冒烟测试的代码。

<p align="center">
  <img src="assets/a2wb-overview.png" width="96%" alt="A²WB overview">
</p>
<p align="center"><em>A²WB 通过迭代式白平衡细化将偏色图像转换为感知上更受偏好的渲染结果，平衡色彩保真度与美学表现。</em></p>

## 方法

发布代码默认执行 **t** 步（默认 10 步）的策略网络推理，每一步为每个像素分配 6 个动作之一。

<p align="center">
  <img src="assets/a2wb-architecture.png" width="96%" alt="A²WB architecture">
</p>
<p align="center"><em>A²WB 网络结构。</em></p>

- **状态编码器（State Encoder）**：在当前 LAB 图像与上一步 ConvGRU 隐藏状态上使用膨胀卷积提取特征。
- **策略网络（Policy Network）**：ConvGRU + 膨胀卷积头，输出 6 通道动作概率图。
- **价值头（Value Head）**：并行膨胀卷积，预测每个像素的 state value，用于 A3C。
- **动作库（Action Bank）**：
  - 0 — No-op
  - 1 — Denoise（CBDNet）
  - 2 — Enhance（DeepLPF）
  - 3 — AWB Colder（DeepWB − t 与输入混合）
  - 4 — AWB Warmer（DeepWB − s 与输入混合）
  - 5 — Low-light Enhancement（SCINet）
- **联合保真度–美学目标（Joint Fidelity–Aesthetic Objective）**：
  R_all = α · R_a + β · R_f + θ · R_r
  - R_a — FMMNet 美学提升奖励。
  - R_f — 参考一致性奖励（与目标 AWB 的 MSE 下降量）。
  - R_r — 梯度幅度伪影正则化。

## 环境安装

```bash
conda create -n a2wb python=3.9 -y
conda activate a2wb
pip install -r requirements.txt
```

## 数据集与权重

下载 A²WB 训练数据对、FMMNet 美学打分权重以及发布的策略网络权重：

- [Google Drive](https://drive.google.com/file/d/1R_ubgxpWC0KA8THDXdcf5FMqD2Ngm6id/view?usp=sharing)

下载后的目录结构：

```text
weights/
├── pixelrl_seg/best_model.pt        # 发布的 A²WB 策略网络
├── deepwb/net_awb.pth               # DeepWB 白平衡模型
├── deepwb/net_t.pth                 # DeepWB 冷色模型
├── deepwb/net_s.pth                 # DeepWB 暖色模型
├── deeplpf/deeplpf_adobe_dpe.pt     # DeepLPF 图像增强模型
└── cbdnet/checkpoint.pth.tar        # CBDNet 去噪权重

action/SCINet/weights/medium.pt      # SCINet 暗光增强模型
action/FMMNet/*.pth                  # FMMNet 美学打分模型
```

下载后按以下目录组织文件：

```text
A2WB-Aesthetic-Aware-Sequential-Color-Correction/
├── weights/
│   ├── pixelrl_seg/best_model.pt
│   ├── deepwb/
│   ├── deeplpf/
│   └── cbdnet/
├── src/action/SCINet/weights/medium.pt
└── src/
```

## 推理

对单张图像执行推理（不使用语义分割）：

```bash
python src/infer_cli.py \
  --input path/to/input.png \
  --output result.png
```

使用语义分割掩码（以 pickle 文件存储，每个语义类对应一个 numpy 数组）：

```bash
python src/infer_cli.py \
  --input path/to/input.png \
  --seg path/to/mask_list.pkl \
  --output result_seg.png
```

完整参数：

```bash
python src/infer_cli.py \
  --input path/to/input.png \
  --seg path/to/mask_list.pkl \
  --output result.png \
  --steps 10 \
  --device cuda:0 \
  --weights weights/pixelrl_seg/best_model.pt
```

| 参数 | 说明 | 默认 |
|-----|------|-----|
| `--input, -i` | 输入图像路径 | （必填） |
| `--output, -o` | 输出图像路径 | `result.png` |
| `--seg, -s` | 语义分割 `.pkl` 文件 | None |
| `--steps` | 推理步数 | 10 |
| `--device, -d` | 推理设备 | 自动 |
| `--weights, -w` | 策略网络权重 | `weights/pixelrl_seg/best_model.pt` |

## 训练

发布代码的训练入口加载成对的偏色/AWB 图像，运行 A3C 训练器共 `--steps` 步，并将权重写入 `src/config.py` 中配置的目录。

```bash
python src/train_cli.py \
  --resume checkpoints/last_epoch.pt
```

训练超参数（路径、学习率、奖励权重、批大小等）位于 `src/config.py`。发布版本默认按以下结构组织数据：

```text
DATA_ROOT/cast/    # 偏色源图像
DATA_ROOT/awb/     # 目标 AWB 真值
DATA_ROOT/train/cast.txt
DATA_ROOT/train/awb.txt
DATA_ROOT/test/cast.txt
DATA_ROOT/test/awb.txt
```

训练目标：

```text
total_loss = policy_loss_weight * policy_loss
           + value_loss_weight   * value_loss
           + entropy_loss_weight * policy_entropy
reward     = rebuild_reward + 0.1 * aes_reward
```

`REWARD_AES_COEF`、`POLICY_LOSS_COEF`、`VALUE_LOSS_COEF`、`ENTROPY_COEF`、`GRAD_CLIP`、`BATCH_SIZE` 和 `TRAIN_STEPS` 均在 `src/config.py` 中定义。

### 训练冒烟测试

只检查环境、模型实例化与前向/反向传播（不加载数据与子模型）：

```bash
python src/smoke_test.py --train-step
```

仅检查前向：

```bash
python src/smoke_test.py
```

## 项目结构

```text
src/
├── neuralnet.py             # PixelRL 策略/价值网络（推理）
├── State.py                 # 推理状态管理器（支持分割引导）
├── infer_cli.py             # 单张图像推理入口
├── net.py                   # 训练用 PixelRL 网络（FeatureExtractor/PolicyHead/ValueHead）
├── train_state.py           # 训练 StateManager
├── trainer.py               # A3C 训练器与奖励合成
├── train_cli.py             # 第二阶段训练入口
├── config.py                # 超参数配置
├── common.py                # 训练工具函数（PSNR、SSIM、张量<->PIL）
├── data_loader.py           # AWB 配对数据集
├── smoke_test.py            # 环境和模型检查
├── utils/
│   └── common.py            # 推理工具函数（BGR<->LAB、动作日志）
└── action/
    ├── func_processor.py    # DeepWB / CBDNet / DeepLPF / SCINet 封装
    ├── aesReward.py         # FMMNet 美学奖励（仅训练）
    ├── gradLoss.py          # 梯度幅度伪影正则化
    ├── CBDNet/              # CBDNet 去噪
    ├── DeepLPF/             # DeepLPF 图像增强
    ├── DeepWB/              # DeepWB 白平衡
    ├── SCINet/              # SCINet 暗光增强
    └── FMMNet/              # FMMNet 美学打分（仅训练）
```

## 说明

- 推荐使用 GPU 进行推理以获得更快速度，CPU 也可运行。
- 语义分割掩码以 pickle 文件存储，内容为每类一个 numpy 数组（形状 `(H, W)`）。
- 首次推理会因子模型加载而较慢，后续推理将复用已加载的模型。
- `src/` 下的推理代码为发布的测试代码；`src/config.py`、`src/trainer.py`、`src/train_cli.py`、`src/data_loader.py`、`src/net.py`、`src/train_state.py` 以及 `src/action/aesReward.py`、`src/action/gradLoss.py`、`src/action/FMMNet/` 等训练相关代码从原始训练仓库抽离而来，仅供本地参考，未在本机重新跑通。

## 引用

```bibtex
@inproceedings{a2wb2026,
  title     = {A²WB: Aesthetic-Aware Sequential Color Correction},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the 35th ACM International Conference on Multimedia},
  year      = {2026},
}
```