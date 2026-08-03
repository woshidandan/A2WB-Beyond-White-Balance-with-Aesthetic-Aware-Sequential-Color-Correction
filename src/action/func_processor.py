import os
import torch
from PIL import Image
import cv2
import numpy as np
from action.DeepWB.DeepWB_arch import deep_wb_model
import action.DeepWB.DeepWB_utilities.utils as utls
from action.DeepWB.DeepWB_utilities.deepWB import deep_wb as dwb_deep_wb
import action.DeepWB.DeepWB_arch.splitNetworks as splitter
from action.DeepWB.DeepWB_arch import deep_wb_single_task
import action.DeepLPF.model as model
import torchvision.transforms.functional as TF

import torch.nn as nn
from action.CBDNet.model.cbdnet import Network
from action.CBDNet.utils import read_img, chw_to_hwc, hwc_to_chw

import torch.utils
import torchvision.transforms as transforms
from action.SCINet.model import Finetunemodel

_ACTION_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_ACTION_DIR)
_WEIGHTS_DIR = os.path.join(_CODE_DIR, "weights")


class DeepWBProcessor:
    def __init__(self, model_path=None, device=None):
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        if model_path is None:
            model_path = os.path.join(_WEIGHTS_DIR, "deepwb", "net_awb.pth")
        self.net_awb = deep_wb_single_task.deepWBnet()
        self.net_awb.to(device=self.device)
        self.net_awb.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.net_awb.eval()

    def get_process_image(self, image):
        out_awb = dwb_deep_wb(image, task='awb', net_awb=self.net_awb, device=self.device, s=656)
        result_awb = utls.to_image(out_awb)
        return result_awb


class DeepWBAllProcessor:
    def __init__(self, model_path=None, device=None):
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        if model_path is None:
            model_path = os.path.join(_WEIGHTS_DIR, "deepwb") + "/"
        self.net_awb = deep_wb_single_task.deepWBnet()
        self.net_awb.to(device=self.device)
        self.net_awb.load_state_dict(torch.load(model_path + "net_awb.pth", map_location=self.device, weights_only=False))
        self.net_awb.eval()

        self.net_t = deep_wb_single_task.deepWBnet()
        self.net_t.to(device=self.device)
        self.net_t.load_state_dict(torch.load(model_path + "net_t.pth", map_location=self.device, weights_only=False))
        self.net_t.eval()

        self.net_s = deep_wb_single_task.deepWBnet()
        self.net_s.to(device=self.device)
        self.net_s.load_state_dict(torch.load(model_path + "net_s.pth", map_location=self.device, weights_only=False))
        self.net_s.eval()

    def get_process_image(self, image):
        out_awb = dwb_deep_wb(image, task='awb', net_awb=self.net_awb, device=self.device, s=656)
        result_awb = utls.to_image(out_awb)
        return result_awb

    def get_all_images(self, image):
        out_awb, out_t, out_s = dwb_deep_wb(image, task='all', net_awb=self.net_awb, net_t=self.net_t, net_s=self.net_s, device=self.device, s=656)
        result_awb = utls.to_image(out_awb)
        result_t = utls.to_image(out_t)
        result_s = utls.to_image(out_s)
        return result_awb, result_t, result_s


class DeepLPFProcessor:
    def __init__(self, model_path=None, device=None):
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        if model_path is None:
            model_path = os.path.join(_WEIGHTS_DIR, "deeplpf", "deeplpf_adobe_dpe.pt")
        self.net = model.DeepLPFNet()
        self.net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.net.to(self.device)
        if self.device.type == 'cuda':
            self.net = nn.DataParallel(self.net, device_ids=[0])
        self.net.eval()

    def get_process_image(self, image):
        input_img_tensor = TF.to_tensor(image)
        input_img_tensor = input_img_tensor.unsqueeze(0)
        input_img_tensor = input_img_tensor.to(self.device)

        with torch.no_grad():
            out_img_tensor = self.net(input_img_tensor)
            out_img_tensor = out_img_tensor.squeeze(0)
            out_img_tensor = out_img_tensor.cpu()
            result_img = TF.to_pil_image(out_img_tensor)
        return result_img


class CBDNetProcessor:
    def __init__(self, model_path=None, device=None):
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        if model_path is None:
            model_path = os.path.join(_WEIGHTS_DIR, "cbdnet", "checkpoint.pth.tar")
        self.model = Network().to(self.device)
        if self.device.type == 'cuda':
            self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.eval()
        model_info = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(model_info['state_dict'])

    def get_process_image(self, input_image_pil):
        input_image = (np.array(input_image_pil) / 255.0).astype(np.float32)
        input_var = torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, output = self.model(input_var)

        output_image = chw_to_hwc(output[0,...].cpu().numpy())
        output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))
        output_image_pil = Image.fromarray(output_image)

        return output_image_pil


class SCILowLightProcessor:
    def __init__(self, model_path=None, seed=2, device=None):
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        if model_path is None:
            model_path = os.path.join(_ACTION_DIR, "SCINet", "weights", "medium.pt")
        self.model_path = model_path
        self.seed = seed

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

        self.model = Finetunemodel(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_process_image(self, input_image_pil):
        input_rgb = input_image_pil.convert('RGB')
        img_norm = self.transform(input_rgb).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        low = img_norm
        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))
        low_tensor = torch.from_numpy(low).unsqueeze(0).to(self.device)

        with torch.no_grad():
            i, out_tensor = self.model(low_tensor)
            image_numpy = out_tensor[0].cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
            output_image_pil = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
        return output_image_pil
