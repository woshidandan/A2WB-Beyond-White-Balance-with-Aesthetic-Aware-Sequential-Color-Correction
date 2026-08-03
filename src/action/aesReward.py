import torch
from PIL import Image
import numpy as np

from action.FMMNet.M_MNet import *


# 美学评分模型
class AestheticProcessor:
    def __init__(self, device=None, model_path='./action/FMMNet/1_srcc_best_balance_data_distort_goodaddscore_back_AVA_balance_remove_unusual_1_resocre-3-7-2_vacc0.8283796740172579_srcc0.8122795303100221.pth'):
        self.device = torch.device(device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model = M_MNet()
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    """
    输入: PIL图像
    """
    def getAesScore(self, image):
        image = TransformPicture(image)
        image = torch.unsqueeze(image, 0).to('cuda:0')
        out = self.model(image)
        score, _ = get_score(out)

        probability2 = torch.max(out, dim=1)[0]
        if probability2 >= 0.48:
            probability2 = 0.48
        
        score_p = pre_probability(score) + probability2
        score = re_score(score)

        if (isinstance(score_p, float)):
            score_p = torch.tensor([score_p], dtype=torch.float)
        
        return score.item(), score_p.item(), score.item() * score_p.item()
    
    """
    输入当前训练状态
    """
    def getBatchAesScores(self, batch_images):
        size = batch_images.shape
        batch_scores = np.zeros((size[0], 1, size[2], size[3]), dtype=float)
        for i in range(size[0]):
            cur_cv_img = batch_images[i].detach().cpu().numpy()
            # 第一步转换
            img_data = np.transpose(cur_cv_img, (1, 2, 0))  # CHW 转 HWC
            img_data = (img_data * 255).astype(np.uint8)  # 将范围从0-1缩放到0-255，并转换为uint8
            img_pil = Image.fromarray(img_data) # 使用PIL创建图片
            # processor处理
            _, _, img_score = self.getAesScore(img_pil)
            batch_scores[i, :, :, :] = img_score
        return torch.from_numpy(batch_scores)

    "比对两个训练状态美学得分"
    def getAesImg2SubImg1(self, img1, img2):
        score1 = self.getBatchAesScores(img1)
        score2 = self.getBatchAesScores(img2)
        return score2 - score1
        
