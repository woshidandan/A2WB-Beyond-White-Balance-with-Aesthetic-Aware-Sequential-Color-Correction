import matplotlib
matplotlib.use('agg')
import torch
import torch.nn as nn
import action.DeepLPF.unet as unet
from math import exp
import math
import torch.nn.functional as F
from action.DeepLPF.util import ImageProcessing
from torch.autograd import Variable


class DeepLPFLoss(nn.Module):

    def __init__(self, ssim_window_size=5, alpha=0.5):
        super(DeepLPFLoss, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size

    def create_window(self, window_size, num_channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous())
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.cuda() + mu2_sq.cuda() + C1) *
                     (sigma1_sq.cuda() + sigma2_sq.cuda() + C2))
        ssim_map = ssim_map1.cuda() / ssim_map2.cuda()

        v1 = 2.0 * sigma12.cuda() + C2
        v2 = sigma1_sq.cuda() + sigma2_sq.cuda() + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs


    def compute_msssim(self, img1, img2):
        if img1.shape[2]!=img2.shape[2]:
                img1=img1.transpose(2,3)

        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            ssim, cs = self.compute_ssim(img1, img2)

            # Relu normalize (not compliant with original definition)
            ssims.append(ssim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        # Simple normalize (not compliant with original definition)
        # TODO: remove support for normalize == True (kept for backward support)
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return output

    def forward(self, predicted_img_batch, target_img_batch):
        if predicted_img_batch.shape[2]!=target_img_batch.shape[2]:
                target_img_batch=target_img_batch.transpose(2,3)

        num_images = target_img_batch.shape[0]
        target_img_batch = target_img_batch

        ssim_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        l1_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))

        for i in range(0, num_images):

            target_img = target_img_batch[i, :, :, :].cuda()
            predicted_img = predicted_img_batch[i, :, :, :].cuda()

            predicted_img_lab = ImageProcessing.rgb_to_lab(
                predicted_img.squeeze(0))
            target_img_lab = ImageProcessing.rgb_to_lab(target_img.squeeze(0))

            target_img_L_ssim = target_img_lab[0, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
            target_img_L_ssim = target_img_L_ssim.unsqueeze(0)
            predicted_img_L_ssim = predicted_img_L_ssim.unsqueeze(0)

            ssim_value = self.compute_msssim(
                predicted_img_L_ssim, target_img_L_ssim)
            ssim_loss_value += (1.0 - ssim_value)

            l1_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)

        l1_loss_value = l1_loss_value/num_images
        ssim_loss_value = ssim_loss_value/num_images
        deeplpf_loss = l1_loss_value + 1e-3*ssim_loss_value
        return deeplpf_loss


class BinaryLayer(nn.Module):

    def forward(self, input):
        return torch.sign(input)

    def backward(self, grad_output):
        """Straight through estimator"""
        input = self.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


class CubicFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64, batch_size=1):
        super(CubicFilter, self).__init__()

        self.cubic_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.cubic_layer2 = MaxPoolBlock()
        self.cubic_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer4 = MaxPoolBlock()
        self.cubic_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer6 = MaxPoolBlock()
        self.cubic_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer8 = GlobalPoolingBlock(2)
        self.fc_cubic = torch.nn.Linear(
            num_out_channels, 60)  # cubic
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)

    def get_cubic_mask(self, feat, img):
        #######################################################
        ####################### Cubic #########################
        feat_cubic = torch.cat((feat, img), 1)
        feat_cubic = self.upsample(feat_cubic)

        x = self.cubic_layer1(feat_cubic)
        x = self.cubic_layer2(x)
        x = self.cubic_layer3(x)
        x = self.cubic_layer4(x)
        x = self.cubic_layer5(x)
        x = self.cubic_layer6(x)
        x = self.cubic_layer7(x)
        x = self.cubic_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        R = self.fc_cubic(x)

        cubic_mask = torch.zeros_like(img)

        x_axis = torch.div(torch.arange(img.shape[2]).view(-1, 1).repeat(1, img.shape[3]).cuda(), img.shape[2], rounding_mode='floor')
        y_axis = torch.div(torch.arange(img.shape[3]).repeat(img.shape[2], 1).cuda(), img.shape[3], rounding_mode='floor')

        # Cubic for R channel
        cubic_mask[0, 0, :, :] = R[0, 0] * (x_axis ** 3) + R[0, 1] * (x_axis ** 2) * y_axis + R[0, 2] * (
            x_axis ** 2) * img[0, 0, :, :] + R[0, 3] * (x_axis ** 2) + R[0, 4] * x_axis * (y_axis ** 2) + R[
            0, 5] * x_axis * y_axis * img[0, 0, :, :] \
            + R[0, 6] * x_axis * y_axis + R[0, 7] * x_axis * (img[0, 0, :, :] ** 2) + R[
            0, 8] * x_axis * img[0, 0, :, :] + R[0, 9] * x_axis + R[0, 10] * (
            y_axis ** 3) + R[0, 11] * (y_axis ** 2) * img[0, 0, :, :] \
            + R[0, 12] * (y_axis ** 2) + R[0, 13] * y_axis * (img[0, 0, :, :] ** 2) + R[
            0, 14] * y_axis * img[0, 0, :, :] + R[0, 15] * y_axis + R[0, 16] * (
            img[0, 0, :, :] ** 3) + R[0, 17] * (img[0, 0, :, :] ** 2) \
            + R[0, 18] * \
            img[0, 0, :, :] + R[0, 19]

        # Cubic for G channel
        cubic_mask[0, 1, :, :] = R[0, 20] * (x_axis ** 3) + R[0, 21] * (x_axis ** 2) * y_axis + R[0, 22] * (
            x_axis ** 2) * img[0, 1, :, :] + R[0, 23] * (x_axis ** 2) + R[0, 24] * x_axis * (y_axis ** 2) + R[
            0, 25] * x_axis * y_axis * img[0, 1, :, :] \
            + R[0, 26] * x_axis * y_axis + R[0, 27] * x_axis * (img[0, 1, :, :] ** 2) + R[
            0, 28] * x_axis * img[0, 1, :, :] + R[0, 29] * x_axis + R[0, 30] * (
            y_axis ** 3) + R[0, 31] * (y_axis ** 2) * img[0, 1, :, :] \
            + R[0, 32] * (y_axis ** 2) + R[0, 33] * y_axis * (img[0, 1, :, :] ** 2) + R[
            0, 34] * y_axis * img[0, 1, :, :] + R[0, 35] * y_axis + R[0, 36] * (
            img[0, 1, :, :] ** 3) + R[0, 37] * (img[0, 1, :, :] ** 2) \
            + R[0, 38] * \
            img[0, 1, :, :] + R[0, 39]

        # Cubic for B channel
        cubic_mask[0, 2, :, :] = R[0, 40] * (x_axis ** 3) + R[0, 41] * (x_axis ** 2) * y_axis + R[0, 42] * (
            x_axis ** 2) * img[0, 2, :, :] + R[0, 43] * (x_axis ** 2) + R[0, 44] * x_axis * (y_axis ** 2) + R[
            0, 45] * x_axis * y_axis * img[0, 2, :, :] \
            + R[0, 46] * x_axis * y_axis + R[0, 47] * x_axis * (img[0, 2, :, :] ** 2) + R[
            0, 48] * x_axis * img[0, 2, :, :] + R[0, 49] * x_axis + R[0, 50] * (
            y_axis ** 3) + R[0, 51] * (y_axis ** 2) * img[0, 2, :, :] \
            + R[0, 52] * (y_axis ** 2) + R[0, 53] * y_axis * (img[0, 2, :, :] ** 2) + R[
            0, 54] * y_axis * img[0, 2, :, :] + R[0, 55] * y_axis + R[0, 56] * (
            img[0, 2, :, :] ** 3) + R[0, 57] * (img[0, 2, :, :] ** 2) \
            + R[0, 58] * \
            img[0, 2, :, :] + R[0, 59]

        img_cubic = torch.clamp(img + cubic_mask, 0, 1)
        return img_cubic


class GraduatedFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        super(GraduatedFilter, self).__init__()

        self.graduated_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.graduated_layer2 = MaxPoolBlock()
        self.graduated_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer4 = MaxPoolBlock()
        self.graduated_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer6 = MaxPoolBlock()
        self.graduated_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer8 = GlobalPoolingBlock(2)
        self.fc_graduated = torch.nn.Linear(
            num_out_channels, 24)
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)
        self.bin_layer = BinaryLayer()

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1"""
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors"""
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_inverted_mask(self, factor, invert, d1, d2, max_scale, top_line):
        """Inverts the graduated filter based on a learnt binary variable"""
        if (invert == 1).all():

            if (factor >= 1).all():
                diff = ((factor-1))/2 + 1
                grad1 = (diff-factor)/d1
                grad2 = (1-diff)/d2
                mask_scale = torch.clamp(
                    factor+grad1*top_line+grad2*top_line, min=1, max=max_scale)
            else:
                diff = ((1-factor))/2 + factor
                grad1 = (diff-factor)/d1
                grad2 = (1-diff)/d2
                mask_scale = torch.clamp(
                    factor+grad1*top_line+grad2*top_line, min=0, max=1)
        else:

            if (factor >= 1).all():
                diff = ((factor-1))/2 + 1
                grad1 = (diff-factor)/d1
                grad2 = (factor-diff)/d2
                mask_scale = torch.clamp(
                    1+grad1*top_line+grad2*top_line, min=1, max=max_scale)
            else:
                diff = ((1-factor))/2 + factor
                grad1 = (diff-1)/d1
                grad2 = (factor-diff)/d2
                mask_scale = torch.clamp(
                    1+grad1*top_line+grad2*top_line, min=0, max=1)

        mask_scale = torch.clamp(mask_scale.unsqueeze(0), 0, max_scale)
        return mask_scale

    def get_graduated_mask(self, feat, img):
        #######################################################
        ####################### Graduated #####################
        eps = 1e-10

        x_axis = torch.div(torch.arange(img.shape[2], device='cuda').view(-1, 1).repeat(1, img.shape[3]), img.shape[2], rounding_mode='floor')
        y_axis = torch.div(torch.arange(img.shape[3], device='cuda').repeat(img.shape[2], 1), img.shape[3], rounding_mode='floor')

        feat_graduated = torch.cat((feat, img), 1)
        feat_graduated = self.upsample(feat_graduated)

        x = self.graduated_layer1(feat_graduated)
        x = self.graduated_layer2(x)
        x = self.graduated_layer3(x)
        x = self.graduated_layer4(x)
        x = self.graduated_layer5(x)
        x = self.graduated_layer6(x)
        x = self.graduated_layer7(x)
        x = self.graduated_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_graduated(x)

        above_or_below_line1 = ((self.bin_layer(G[0, 0]))+1)/2
        above_or_below_line2 = ((self.bin_layer(G[0, 1]))+1)/2
        above_or_below_line3 = ((self.bin_layer(G[0, 2]))+1)/2

        slope1 = G[0, 3].clone()
        slope2 = G[0, 4].clone()
        slope3 = G[0, 5].clone()

        y_axis_dist1 = self.tanh01(G[0, 6]) + eps
        y_axis_dist2 = self.tanh01(G[0, 7]) + eps
        y_axis_dist3 = self.tanh01(G[0, 8]) + eps

        y_axis_dist1 = torch.clamp(self.tanh01(G[0, 9]), y_axis_dist1.data, 1.0)
        y_axis_dist2 = torch.clamp(self.tanh01(G[0, 10]), y_axis_dist2.data, 1.0)
        y_axis_dist3 = torch.clamp(self.tanh01(G[0, 11]), y_axis_dist3.data, 1.0)

        y_axis_dist4= torch.clamp(self.tanh01(G[0, 12]), 0, y_axis_dist1.data)
        y_axis_dist5 = torch.clamp(self.tanh01(G[0, 13]), 0, y_axis_dist2.data)
        y_axis_dist6 = torch.clamp(self.tanh01(G[0, 14]), 0, y_axis_dist3.data)

        # Scales
        max_scale = 2
        min_scale = 0

        scale_factor1 = self.tanh01(G[0, 15]) * max_scale
        scale_factor2 = self.tanh01(G[0, 16]) * max_scale
        scale_factor3 = self.tanh01(G[0, 17]) * max_scale

        scale_factor4 = self.tanh01(G[0, 18]) * max_scale
        scale_factor5 = self.tanh01(G[0, 19]) * max_scale
        scale_factor6 = self.tanh01(G[0, 20]) * max_scale

        scale_factor7 = self.tanh01(G[0, 21]) * max_scale
        scale_factor8 = self.tanh01(G[0, 22]) * max_scale
        scale_factor9= self.tanh01(G[0, 23]) * max_scale

        slope1_angle = torch.atan(slope1)
        slope2_angle = torch.atan(slope2)
        slope3_angle = torch.atan(slope3)

        # Distances between central line and two outer lines
        d1 = self.tanh01(y_axis_dist1*torch.cos(slope1_angle))
        d2 = self.tanh01(y_axis_dist4*torch.cos(slope1_angle))
        d3 = self.tanh01(y_axis_dist2*torch.cos(slope2_angle))
        d4 = self.tanh01(y_axis_dist5*torch.cos(slope2_angle))
        d5 = self.tanh01(y_axis_dist3*torch.cos(slope3_angle))
        d6 = self.tanh01(y_axis_dist6*torch.cos(slope3_angle))

        top_line1 = self.tanh01(y_axis - (slope1 * x_axis + y_axis_dist1 + d1))
        top_line2 = self.tanh01(y_axis - (slope2 * x_axis + y_axis_dist2 + d3))
        top_line3 = self.tanh01(y_axis - (slope3 * x_axis + y_axis_dist3 + d5))

        mask_scale1 = self.get_inverted_mask(
            scale_factor1, above_or_below_line1, d1, d2, max_scale, top_line1)
        mask_scale2 = self.get_inverted_mask(
            scale_factor2, above_or_below_line1, d1, d2, max_scale, top_line1)
        mask_scale3 = self.get_inverted_mask(
            scale_factor3, above_or_below_line1, d1, d2, max_scale, top_line1)

        mask_scale_1 = torch.cat(
            (mask_scale1, mask_scale2, mask_scale3), dim=0)
        mask_scale_1 = torch.clamp(mask_scale_1.unsqueeze(0), 0, max_scale)

        mask_scale4 = self.get_inverted_mask(
            scale_factor4, above_or_below_line2, d3, d4, max_scale, top_line2)
        mask_scale5 = self.get_inverted_mask(
            scale_factor5, above_or_below_line2, d3, d4, max_scale, top_line2)
        mask_scale6 = self.get_inverted_mask(
            scale_factor6, above_or_below_line2, d3, d4, max_scale, top_line2)

        mask_scale_4 = torch.cat(
            (mask_scale4, mask_scale5, mask_scale6), dim=0)
        mask_scale_4 = torch.clamp(mask_scale_4.unsqueeze(0), 0, max_scale)

        mask_scale7 = self.get_inverted_mask(
            scale_factor7, above_or_below_line3, d5, d6, max_scale, top_line3)
        mask_scale8 = self.get_inverted_mask(
            scale_factor8, above_or_below_line3, d5, d6, max_scale, top_line3)
        mask_scale9 = self.get_inverted_mask(
            scale_factor9, above_or_below_line3, d5, d6, max_scale, top_line3)

        mask_scale_7 = torch.cat(
            (mask_scale7, mask_scale8, mask_scale9), dim=0)
        mask_scale_7 = torch.clamp(mask_scale_7.unsqueeze(0), 0, max_scale)

        mask_scale = torch.clamp(
            mask_scale_1*mask_scale_4*mask_scale_7, 0, max_scale)

        return mask_scale


class EllipticalFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        super(EllipticalFilter, self).__init__()

        self.elliptical_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.elliptical_layer2 = MaxPoolBlock()
        self.elliptical_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer4 = MaxPoolBlock()
        self.elliptical_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer6 = MaxPoolBlock()
        self.elliptical_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer8 = GlobalPoolingBlock(2)
        self.fc_elliptical = torch.nn.Linear(
            num_out_channels, 24)  # elliptical
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1"""
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors"""
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_mask(self, x_axis, y_axis, shift_x=0, shift_y=0, semi_axis_x=1, semi_axis_y=1, alpha=0,
                 scale_factor=2, max_scale=2, eps=1e-7, radius=1):
        """Gets the elliptical scaling mask according to the equation of a rotated ellipse"""
        ellipse_equation_part1 = (((x_axis - shift_x)*torch.cos(alpha) + (y_axis - shift_y)*torch.sin(alpha)) ** 2) / ((semi_axis_x)**2)
        ellipse_equation_part2 = (((x_axis - shift_x)*torch.sin(alpha) - (y_axis - shift_y)*torch.cos(alpha)) ** 2) / ((semi_axis_y)**2)

        # Set the scaling factors to decay with radius inside the ellipse
        mask_scale = self.where(ellipse_equation_part1+ellipse_equation_part2 < 1,
                                (torch.sqrt((x_axis - shift_x) ** 2 + (y_axis - shift_y) ** 2 + eps) * (1 - scale_factor)) / radius + scale_factor, 1)

        mask_scale = torch.clamp(mask_scale.unsqueeze(0), 0, max_scale)

        return mask_scale

    def get_elliptical_mask(self, feat, img):
        eps2 = 1e-7
        eps1 = 1e-10

        max_scale = 2
        min_scale = 0

        feat_elliptical = torch.cat((feat, img), 1)
        feat_elliptical = self.upsample(feat_elliptical)

        x = self.elliptical_layer1(feat_elliptical)
        x = self.elliptical_layer2(x)
        x = self.elliptical_layer3(x)
        x = self.elliptical_layer4(x)
        x = self.elliptical_layer5(x)
        x = self.elliptical_layer6(x)
        x = self.elliptical_layer7(x)
        x = self.elliptical_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_elliptical(x)

        # Normalised coordinates for x and y-axes
        x_axis = torch.div(torch.arange(img.shape[2], device=img.device).view(-1, 1).repeat(1, img.shape[3]), img.shape[2], rounding_mode='floor')
        y_axis = torch.div(torch.arange(img.shape[3], device=img.device).repeat(img.shape[2], 1), img.shape[3], rounding_mode='floor')

        right_x = torch.div(img.shape[2] - 1, img.shape[2], rounding_mode='floor')
        left_x = 0

        # Centre of ellipse, x-coordinate
        x_coord1 = self.tanh01(G[0, 0]) + eps1
        x_coord2 = self.tanh01(G[0, 1]) + eps1
        x_coord3 = self.tanh01(G[0, 2]) + eps1

        right_y = torch.div(img.shape[3] - 1, img.shape[3], rounding_mode='floor')
        left_y = 0

        # Centre of ellipse, y-coordinate
        y_coord1 = self.tanh01(G[0, 3]) + eps1
        y_coord2 = self.tanh01(G[0, 4]) + eps1
        y_coord3 = self.tanh01(G[0, 5]) + eps1

        # Semi-axis a
        a1 = self.tanh01(G[0, 6]) + eps1
        a2 = self.tanh01(G[0, 7]) + eps1
        a3 = self.tanh01(G[0, 8]) + eps1

        # Semi-axis b
        b1 = self.tanh01(G[0, 9]) + eps1
        b2 = self.tanh01(G[0, 10]) + eps1
        b3 = self.tanh01(G[0, 11]) + eps1

        # Angle to the x-axis
        A1 = self.tanh01(G[0, 12]) * math.pi + eps1
        A2 = self.tanh01(G[0, 13]) * math.pi + eps1
        A3 = self.tanh01(G[0, 14]) * math.pi + eps1

        scale1 = self.tanh01(G[0, 15]) * max_scale + eps1
        scale2 = self.tanh01(G[0, 16]) * max_scale + eps1
        scale3 = self.tanh01(G[0, 17]) * max_scale + eps1

        scale4  = self.tanh01(G[0, 18]) * max_scale + eps1
        scale5 = self.tanh01(G[0, 19]) * max_scale + eps1
        scale6 = self.tanh01(G[0, 20]) * max_scale + eps1

        scale7 = self.tanh01(G[0, 21]) * max_scale + eps1
        scale8 = self.tanh01(G[0, 22]) * max_scale + eps1
        scale9 = self.tanh01(G[0, 23]) * max_scale + eps1

        ############ Angle of orientation of the ellipses with respect to the y semi-axis
        angle_1 = torch.acos(torch.clamp((y_axis-y_coord1) /
            (torch.sqrt((x_axis-x_coord1)**2 + (y_axis-y_coord1)**2 + eps1)), -1+eps2, 1-eps2))-A1

        angle_2 = torch.acos(torch.clamp((y_axis-y_coord2) /
            (torch.sqrt((x_axis-x_coord2) ** 2 + (y_axis-y_coord2)**2 + eps1)), -1+eps2, 1-eps2))-A2

        angle_3 = torch.acos(torch.clamp((y_axis-y_coord3) /
            (torch.sqrt((x_axis-x_coord3) ** 2 + (y_axis-y_coord3)**2 + eps1)), -1+eps2, 1-eps2))-A3

        ############ Radius of the ellipses
        radius_1 = (a1*b1)/torch.sqrt((a1**2)*(torch.sin(angle_1)**2)+(b1**2)*(torch.cos(angle_1)**2) + eps1)

        radius_2 = (a2*b2)/torch.sqrt((a2**2)*(torch.sin(angle_2)**2)+(b2**2)*(torch.cos(angle_2)**2) + eps1)

        radius_3 = (a3*b3)/torch.sqrt((a3**2)*(torch.sin(angle_3)**2)+(b3**2)*(torch.cos(angle_3)**2) + eps1)


        ############ Scaling factors for the R,G,B channels, here we learn three ellipses
        mask_scale1 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord1, shift_y=y_coord1, semi_axis_x=a1, semi_axis_y=b1, alpha=angle_1, scale_factor=scale1,
                                    radius=radius_1)

        mask_scale2 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord1, shift_y=y_coord1, semi_axis_x=a1, semi_axis_y=b1, alpha=angle_1, scale_factor=scale2,
                                    radius=radius_1)

        mask_scale3 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord1, shift_y=y_coord1, semi_axis_x=a1, semi_axis_y=b1, alpha=angle_1, scale_factor=scale3,
                                    radius=radius_1)

        mask_scale_1 = torch.cat(
            (mask_scale1, mask_scale2, mask_scale3), dim=0)
        mask_scale_1_rad = torch.clamp(mask_scale_1.unsqueeze(0), 0, max_scale)

        ############ Scaling factors for the R,G,B channels, here we learn three ellipses
        mask_scale4 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord2, shift_y=y_coord2, semi_axis_x=a2, semi_axis_y=b2, alpha=angle_2, scale_factor=scale4,
                                    radius=radius_2)

        mask_scale5 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord2, shift_y=y_coord2, semi_axis_x=a2, semi_axis_y=b2, alpha=angle_2, scale_factor=scale5,
                                    radius=radius_2)

        mask_scale6 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord2, shift_y=y_coord2, semi_axis_x=a2, semi_axis_y=b3, alpha=angle_2, scale_factor=scale6,
                                    radius=radius_2)

        mask_scale_4 = torch.cat(
            (mask_scale4, mask_scale5, mask_scale6), dim=0)
        mask_scale_4_rad = torch.clamp(mask_scale_4.unsqueeze(0), 0, max_scale)

        ############ Scaling factors for the R,G,B channels, here we learn three ellipses
        mask_scale7 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord3, shift_y=y_coord3, semi_axis_x=a3, semi_axis_y=b3, alpha=angle_3, scale_factor=scale7,
                                    radius=radius_3)

        mask_scale8 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord3, shift_y=y_coord3, semi_axis_x=a3, semi_axis_y=b3, alpha=angle_3, scale_factor=scale8,
                                    radius=radius_3)

        mask_scale9 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord3, shift_y=y_coord3, semi_axis_x=a3, semi_axis_y=b3, alpha=angle_3, scale_factor=scale9,
                                    radius=radius_3)

        mask_scale_7 = torch.cat(
            (mask_scale7, mask_scale8, mask_scale9), dim=0)
        mask_scale_7_rad = torch.clamp(mask_scale_7.unsqueeze(0), 0, max_scale)

        ############ Mix the ellipses together by multiplication
        mask_scale_elliptical = torch.clamp(
            mask_scale_1_rad * mask_scale_4_rad * mask_scale_7_rad, 0, max_scale)

        return mask_scale_elliptical


class Block(nn.Module):

    def __init__(self):
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.avg_pool(x)
        return out


class DeepLPFParameterPrediction(nn.Module):
    import torch.nn.functional as F

    def __init__(self, num_in_channels=64, num_out_channels=64, batch_size=1):
        super(DeepLPFParameterPrediction, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.cubic_filter = CubicFilter()
        self.graduated_filter = GraduatedFilter()
        self.elliptical_filter = EllipticalFilter()


    def forward(self, x):
        x.contiguous()  # remove memory holes
        x.cuda()

        feat = x[:, 3:64, :, :]
        img = x[:, 0:3, :, :]

        torch.cuda.empty_cache()
        shape = x.shape

        img_cubic = self.cubic_filter.get_cubic_mask(feat, img)

        mask_scale_graduated = self.graduated_filter.get_graduated_mask(
            feat, img_cubic)
        mask_scale_elliptical = self.elliptical_filter.get_elliptical_mask(
            feat, img_cubic)

        mask_scale_fuse = torch.clamp(
            mask_scale_graduated+mask_scale_elliptical, 0, 2)

        img_fuse = torch.clamp(img_cubic*mask_scale_fuse, 0, 1)

        img = torch.clamp(img_fuse+img, 0, 1)

        return img


class DeepLPFNet(nn.Module):

    def __init__(self):
        super(DeepLPFNet, self).__init__()
        self.backbonenet = unet.UNetModel()
        self.deeplpfnet = DeepLPFParameterPrediction()

    def forward(self, img):
        feat = self.backbonenet(img)
        img = self.deeplpfnet(feat)

        return img
