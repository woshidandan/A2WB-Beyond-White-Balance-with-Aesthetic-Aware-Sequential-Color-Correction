# coding:utf-8
import torch.nn as nn
# from models.mv2 import mobile_net_v2
from action.FMMNet.mv2 import mobile_net_v2

import torch
import os, shutil
# import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import numpy as np

# 害我不浅, 找了这么久才找到你, 害我不能多卡训练, 天天只能用0卡
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

def TransformPicture(x):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
    return transform(x)

def get_score(y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to('cuda')
    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

def pre_probability(score, acc_0_3=1, acc_3_4=0.98, acc_4_5=0.80, acc_5_6=0.66, acc_6_7=0.94, acc_7_10=1.0):
    if score < 3:
        probability1 = acc_0_3 / 2
    elif 3 <= score < 4:
        probability1 = acc_3_4 / 2
    elif 4 <= score < 5:
        probability1 = acc_4_5 / 2
    elif 5 <= score < 6:
        probability1 = acc_5_6 / 2
    elif 6 <= score < 7:
        probability1 = acc_6_7 / 2
    elif 7 <= score <= 10:
        probability1 = acc_7_10 / 2
    return probability1

def re_score(score, low_flag=1.8, high_flag=8.6):
    if score < 5:
        if score <= low_flag:
            score = 0.1
        else:
            score = (score - low_flag) / (5 - low_flag) * 5
    else:
        if score >= high_flag:
            score = 9.8
        else:
            score = ((score - 5) / (high_flag - 5) * 5) + 5
    return score


class M_MNet(nn.Module):
    def __init__(self, pretrained_base_model=False):
        super(M_MNet, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)

        # print("x.shape: ", x.shape)

        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


