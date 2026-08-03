import matplotlib
matplotlib.use('agg')
from torch.autograd import Variable
import numpy as np
import torch
import sys
from PIL import Image
from skimage.metrics import structural_similarity as ssim

np.set_printoptions(threshold=sys.maxsize)


class ImageProcessing(object):

    @staticmethod
    def rgb_to_lab(img, is_training=True):
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)

        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.000001) + 0.055) / 1.055) ** 2.4) * img.gt(0.04045).float()

        rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
                                                [0.412453, 0.212671,
                                                    0.019334],  # R
                                                [0.357580, 0.715160,
                                                    0.119193],  # G
                                                [0.180423, 0.072169,
                                                    0.950227],  # B
                                                ]), requires_grad=False).cuda()

        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(img, Variable(torch.FloatTensor(
            [1/0.950456, 1.0, 1/1.088754]), requires_grad=False).cuda())

        epsilon = 6/29

        img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
            (torch.clamp(img, min=0.0001)**(1.0/3.0) * img.gt(epsilon**3).float())

        fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0,  500.0,    0.0],  # fx
                                                    [116.0, -500.0,  200.0],  # fy
                                                    [0.0,    0.0, -200.0],  # fz
                                                    ]), requires_grad=False).cuda()

        img = torch.matmul(img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).cuda()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        img[0, :, :] = img[0, :, :]/100
        img[1, :, :] = (img[1, :, :]/110 + 1)/2
        img[2, :, :] = (img[2, :, :]/110 + 1)/2

        img[(img != img).detach()] = 0

        img = img.contiguous()

        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def swapimdims_HW3_3HW(img):
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    @staticmethod
    def load_image(img_filepath, normaliser):
        img = ImageProcessing.normalise_image(np.array(Image.open(img_filepath)), normaliser)
        return img

    @staticmethod
    def normalise_image(img, normaliser):
        img = img.astype('float32') / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity):
        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for i in range(0, num_images):
            imageA = image_batchA[i, 0:3, :, :]
            imageB = image_batchB[i, 0:3, :, :]
            imageB = np.maximum(0, np.minimum(imageB, max_intensity))
            psnr_val += 10 * \
                np.log10(max_intensity ** 2 /
                         ImageProcessing.compute_mse(imageA, imageB))

        return psnr_val / num_images

    @staticmethod
    def compute_ssim(image_batchA, image_batchB):
        num_images = image_batchA.shape[0]
        ssim_val = 0.0

        for i in range(0, num_images):
            imageA = ImageProcessing.swapimdims_3HW_HW3(
                image_batchA[i, 0:3, :, :])
            imageB = ImageProcessing.swapimdims_3HW_HW3(
                image_batchB[i, 0:3, :, :])
            ssim_val += ssim(imageA, imageB, data_range=imageA.max() - imageA.min(), multichannel=True,
                             gaussian_weights=True, win_size=11)

        return ssim_val / num_images
