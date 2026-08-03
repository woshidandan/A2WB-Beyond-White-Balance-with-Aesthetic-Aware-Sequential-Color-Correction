import os
import cv2
import copy
import numpy as np
import torch
from PIL import Image
from typing import Dict, Tuple, Optional
from common import tensor2pil, pil2tensor, tensor2normalizedLab, to_cpu, exist_value, save_images
from action.func_processor import *
from config import config

class StateManager:
    def __init__(self, device):
        self.device = torch.device(device)
        self.move_range = 3

        # 初始化动作处理器
        self.action_processors = {
            "awb": DeepWBProcessor(),
            "awb_all": DeepWBAllProcessor(),
            "denoise": CBDNetProcessor(),
            "enhance": DeepLPFProcessor(),
            "lowlight": SCILowLightProcessor()
        }

        # 全局状态
        self.tensor = None
        self.image_state = None

    def reset(self, src_tensor, dst_tensor):
        # 设置初始状态 BCHW RGB [0-1]
        self.src_tensor = src_tensor
        self.dst_tensor = dst_tensor

        # 当前图像状态, 需要维持为张量类型
        self.image_state = src_tensor.clone().to(config.DEVICE)

        # 首先进行白平衡处理
        self.image_state = self._process_batch_images(self.action_processors["awb"], self.image_state.clone())


        # 基于白平衡图计算其他状态
        if False:
            save_images(to_cpu(self.image_state), os.path.join(config.LOG_DIR, "test"), prefix="bgr_awb")
            bgr_denoise = self._process_batch_images(self.action_processors["denoise"], self.image_state.clone())
            save_images(to_cpu(bgr_denoise), os.path.join(config.LOG_DIR, "test"), prefix="denoise")
            bgr_enhance = self._process_batch_images(self.action_processors["enhance"], self.image_state.clone())
            save_images(to_cpu(bgr_enhance), os.path.join(config.LOG_DIR, "test"), prefix="bgr_enhance")
            
            # 测试目标图为增强图，看收敛情况
            # self.dst_tensor = bgr_enhance


            bgr_colder, bgr_warmer = self._process_awb_batch_images(self.action_processors["awb_all"], self.image_state.clone())
            save_images(to_cpu(bgr_colder), os.path.join(config.LOG_DIR, "test"), prefix="bgr_awb_colder")
            save_images(to_cpu(bgr_warmer), os.path.join(config.LOG_DIR, "test"), prefix="bgr_awb_warmer")
            bgr_low_light = self._process_batch_images(self.action_processors["lowlight"], self.image_state.clone())
            save_images(to_cpu(bgr_low_light), os.path.join(config.LOG_DIR, "test"), prefix="bgr_low_light")
        
        
        
        # 设置整体特征, 张量类型
        b, _, h, w = self.image_state.shape
        previous_state = torch.zeros(size=(b, 64, h, w), dtype=self.image_state.dtype, device=self.device)

        self.tensor = torch.cat([tensor2normalizedLab(self.image_state.clone()), previous_state], dim=1)
        print(f"预处理完成")
        return self.tensor

    def step(self, act, inner_state):
        # 测试某个动作的效果
        act = to_cpu(act)

        ACTION_DENOISE = 1
        ACTION_ENHANCE = ACTION_DENOISE + 1
        ACTION_AWB_COLDER = ACTION_ENHANCE + 1
        ACTION_AWB_WARMER = ACTION_AWB_COLDER + 1
        ACTION_LOW_LIGHT = ACTION_AWB_WARMER + 1

        # 备用
        bgr_denoise = self.image_state.clone()
        bgr_enhance = self.image_state.clone()
        bgr_colder = self.image_state.clone()
        bgr_warmer = self.image_state.clone()
        bgr_low_light = self.image_state.clone()

        # 动作执行, 默认生成图都在CPU上 降噪/增强/色温冷暖/暗光增强
        with torch.no_grad():
            if exist_value(act, ACTION_DENOISE):
                bgr_denoise = self._process_batch_images(self.action_processors["denoise"], self.image_state.clone())
            if exist_value(act, ACTION_ENHANCE):
                bgr_enhance = self._process_batch_images(self.action_processors["enhance"], self.image_state.clone())
            if exist_value(act, ACTION_AWB_COLDER) or exist_value(act, ACTION_AWB_WARMER):
                bgr_colder, bgr_warmer = self._process_awb_batch_images(self.action_processors["awb_all"], self.image_state.clone())
            if exist_value(act, ACTION_LOW_LIGHT):
                bgr_low_light = self._process_batch_images(self.action_processors["lowlight"], self.image_state.clone())

        # 复制通道 b1hw
        act = act.unsqueeze(1)
        act_3channel = torch.concat([act, act, act], 1)
        act_3channel = act_3channel.to(config.DEVICE)

        # 状态更新 降噪/增强/冷色/暖色/暗光增强
        # input(f"img:{self.image_state.device} bgr:{bgr_denoise.device}")
        self.image_state = torch.where(act_3channel == ACTION_DENOISE, bgr_denoise, self.image_state)
        self.image_state = torch.where(act_3channel == ACTION_ENHANCE, bgr_enhance, self.image_state)
        self.image_state = torch.where(act_3channel == ACTION_AWB_COLDER, bgr_colder, self.image_state)
        self.image_state = torch.where(act_3channel == ACTION_AWB_WARMER, bgr_warmer, self.image_state)
        self.image_state = torch.where(act_3channel == ACTION_LOW_LIGHT, bgr_low_light, self.image_state)


        # 生成新的状态张量
        new_lab = tensor2normalizedLab(self.image_state.clone())
        new_tensor = torch.cat([new_lab, inner_state], dim=1)  # 创建新对象
        self.tensor = new_tensor
        
        # 拼接整体状态
        # self.tensor[:, 0:3, :, :] = tensor2normalizedLab(self.image_state)
        # self.tensor[:, -64:, :, :] = inner_state

        return self.tensor

    def _process_awb_batch_images(self, processor, batch_images):
        """
        batch_images: B; PIL
        PIL: B; HWC; RGB; [0-255]
        """
        size = batch_images.shape
        batch_images = tensor2pil(batch_images)
        batch_images_colder = batch_images.copy()
        batch_images_warmer = batch_images.copy()
        for i in range(size[0]):
            img_pil = batch_images[i]  # 创建PIL图片
            # processor处理
            ratio = 0.05
            result_awb, result_t, result_s = processor.get_all_images(img_pil)
            batch_images_colder[i] = Image.blend(img_pil, result_t, ratio)
            batch_images_warmer[i] = Image.blend(img_pil, result_s, ratio)

        return pil2tensor(batch_images_colder), pil2tensor(batch_images_warmer)

    def _process_batch_images(self, processor, batch_images):
        """
        batch_images: B; PIL
        PIL: B; HWC; RGB; [0-255]
        """
        size = batch_images.shape
        batch_images = tensor2pil(batch_images)
        for i in range(size[0]):
            img_pil = batch_images[i]
            batch_images[i] = processor.get_process_image(img_pil)  # 使用processor处理
        return pil2tensor(batch_images)
