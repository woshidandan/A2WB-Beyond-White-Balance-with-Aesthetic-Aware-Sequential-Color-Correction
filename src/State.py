import torch
from utils.common import *

from action.func_processor import *

class State:
    def __init__(self, device):
        self.device = device
        self.tensor = None
        self.move_range = 3

        self.device = torch.device(device)

        self.awb_processor = DeepWBProcessor()
        self.awb_all_processor = DeepWBAllProcessor()
        self.denoise_processor = CBDNetProcessor()
        self.enhance_processor = DeepLPFProcessor()
        self.lowlight_processor = SCILowLightProcessor()

    def reset(self, raw_dst, raw_src, mask_arr, text_arr):
        self.raw_dst = raw_dst
        self.raw_src = raw_src
        self.mask_arr = mask_arr
        self.text_arr = text_arr

        self.image_state = torch.from_numpy(self.raw_src)

        # Initial white balance
        self.image_state = self.process_batch_images(self.awb_processor, tensor2numpy(self.image_state))

        b, _, h, w = self.image_state.shape
        previous_state = torch.zeros(size=(b, 64, h, w), dtype=self.image_state.dtype)
        self.tensor = torch.concat([self.image_state, previous_state], dim=1)

    def set(self, lab_cur):
        temp = lab_cur.clone()
        temp[:, 0, :, :] /= 100.0
        temp[:, 1, :, :] /= 127.0
        temp[:, 2, :, :] /= 127.0
        self.tensor[:, :3, :, :] = temp

    def step(self, act, inner_state):
        act = to_cpu(act)
        inner_state = to_cpu(inner_state)

        ACTION_DENOISE = 1
        ACTION_ENHANCE = ACTION_DENOISE + 1
        ACTION_AWB_COLDER = ACTION_ENHANCE + 1
        ACTION_AWB_WARMER = ACTION_AWB_COLDER + 1
        ACTION_LOW_LIGHT = ACTION_AWB_WARMER + 1

        bgr_denoise = self.image_state.clone()
        bgr_enhance = self.image_state.clone()
        bgr_colder = self.image_state.clone()
        bgr_warmer = self.image_state.clone()
        bgr_low_light = self.image_state.clone()

        with torch.no_grad():
            if exist_value(act, ACTION_DENOISE):
                bgr_denoise = self.process_batch_images(self.denoise_processor, tensor2numpy(self.image_state))
            if exist_value(act, ACTION_ENHANCE):
                bgr_enhance = self.process_batch_images(self.enhance_processor, tensor2numpy(self.image_state))
            if exist_value(act, ACTION_AWB_COLDER) or exist_value(act, ACTION_AWB_WARMER):
                bgr_colder, bgr_warmer, bgr_awb = self.process_awb_batch_images(self.awb_all_processor, tensor2numpy(self.image_state))
            if exist_value(act, ACTION_LOW_LIGHT):
                bgr_low_light = self.process_batch_images(self.lowlight_processor, tensor2numpy(self.image_state))

        act = act.unsqueeze(1)
        act_3channel = torch.concat([act, act, act], 1)

        self.image_state = torch.where(act_3channel==ACTION_DENOISE, bgr_denoise, self.image_state)
        self.image_state = torch.where(act_3channel==ACTION_ENHANCE, bgr_enhance, self.image_state)
        self.image_state = torch.where(act_3channel==ACTION_AWB_COLDER, bgr_colder, self.image_state)
        self.image_state = torch.where(act_3channel==ACTION_AWB_WARMER, bgr_warmer, self.image_state)
        self.image_state = torch.where(act_3channel==ACTION_LOW_LIGHT, bgr_low_light, self.image_state)

        self.tensor[:,0:3,:,:] = self.image_state
        self.tensor[:,-64:,:,:] = inner_state

    def step_seg(self, act, inner_state):
        act = to_cpu(act)
        inner_state = to_cpu(inner_state)

        ACTION_DENOISE = 1
        ACTION_ENHANCE = ACTION_DENOISE + 1
        ACTION_AWB_COLDER = ACTION_ENHANCE + 1
        ACTION_AWB_WARMER = ACTION_AWB_COLDER + 1
        ACTION_LOW_LIGHT = ACTION_AWB_WARMER + 1

        bgr_denoise = self.image_state.clone()
        bgr_enhance = self.image_state.clone()
        bgr_colder = self.image_state.clone()
        bgr_warmer = self.image_state.clone()
        bgr_low_light = self.image_state.clone()

        with torch.no_grad():
            if exist_value(act, ACTION_DENOISE):
                bgr_denoise = self.process_batch_images(self.denoise_processor, tensor2numpy(self.image_state))
            if exist_value(act, ACTION_ENHANCE):
                bgr_enhance = self.process_batch_images(self.enhance_processor, tensor2numpy(self.image_state))
            if exist_value(act, ACTION_AWB_COLDER) or exist_value(act, ACTION_AWB_WARMER):
                bgr_colder, bgr_warmer, bgr_awb = self.process_awb_batch_images(self.awb_all_processor, tensor2numpy(self.image_state))
            if exist_value(act, ACTION_LOW_LIGHT):
                bgr_low_light = self.process_batch_images(self.lowlight_processor, tensor2numpy(self.image_state))

        act = act.unsqueeze(1)
        act_3channel = torch.cat([act, act, act], dim=1)

        self.image_state = torch.where(act_3channel==ACTION_DENOISE, bgr_denoise, self.image_state)
        self.image_state = torch.where(act_3channel==ACTION_ENHANCE, bgr_enhance, self.image_state)
        self.image_state = torch.where(act_3channel==ACTION_AWB_COLDER, bgr_colder, self.image_state)
        self.image_state = torch.where(act_3channel==ACTION_AWB_WARMER, bgr_warmer, self.image_state)
        self.image_state = torch.where(act_3channel==ACTION_LOW_LIGHT, bgr_low_light, self.image_state)

        # Semantic segmentation guided correction
        seg_act_3channel = act_3channel.clone()

        for i in range(self.mask_arr.shape[1]):
            mask_cur = self.mask_arr[:, i, :, :]  # (1, H, W)
            mask_cur = torch.from_numpy(mask_cur)
            non_mask_count_zeros = (mask_cur == 0).sum().item()
            mask_expanded = mask_cur.unsqueeze(1)  # (1, 1, H, W)
            act_3channel_type = act_3channel * mask_expanded  # (1, 3, H, W)

            for channel in range(3):
                channel_actions = act_3channel_type[0, channel, :, :]  # (H, W)
                unique_actions, counts = torch.unique(channel_actions, return_counts=True)

                # Exclude zero-action counts from outside the mask region
                if 0 in unique_actions:
                    zero_idx = (unique_actions == 0).nonzero(as_tuple=True)[0]
                    counts[zero_idx] -= non_mask_count_zeros

                if counts.sum() > 0:
                    max_count_idx = torch.argmax(counts)
                    most_common_action = unique_actions[max_count_idx]
                else:
                    most_common_action = 0

                seg_act_3channel[0, channel, :, :] = torch.where(mask_cur[0] == 1, most_common_action, seg_act_3channel[0, channel, :, :])

            mask_expanded_3channel = mask_expanded.repeat(1, 3, 1, 1)  # (1, 3, H, W)

            if torch.any(seg_act_3channel == ACTION_DENOISE):
                self.image_state = torch.where((seg_act_3channel==ACTION_DENOISE) & (mask_expanded_3channel == 1), bgr_denoise, self.image_state)
            if torch.any(seg_act_3channel == ACTION_ENHANCE):
                self.image_state = torch.where((seg_act_3channel==ACTION_ENHANCE) & (mask_expanded_3channel == 1), bgr_enhance, self.image_state)
            if torch.any(seg_act_3channel == ACTION_AWB_COLDER):
                self.image_state = torch.where((seg_act_3channel==ACTION_AWB_COLDER) & (mask_expanded_3channel == 1), bgr_colder, self.image_state)
            if torch.any(seg_act_3channel == ACTION_AWB_WARMER):
                self.image_state = torch.where((seg_act_3channel==ACTION_AWB_WARMER) & (mask_expanded_3channel == 1), bgr_warmer, self.image_state)
            if torch.any(seg_act_3channel == ACTION_LOW_LIGHT):
                self.image_state = torch.where((seg_act_3channel==ACTION_LOW_LIGHT) & (mask_expanded_3channel == 1), bgr_low_light, self.image_state)

        self.tensor[:,0:3,:,:] = self.image_state
        self.tensor[:,-64:,:,:] = inner_state

    def process_awb_batch_images(self, processor, batch_images):
        size = batch_images.shape
        batch_images_colder = batch_images.copy()
        batch_images_warmer = batch_images.copy()
        batch_images_awb = batch_images.copy()
        for i in range(size[0]):
            cur_cv_img = batch_images[i]
            img_pil = self.State2PIL(cur_cv_img)

            ratio = 0.05
            result_awb, result_t, result_s = processor.get_all_images(img_pil)
            img_colder = Image.blend(img_pil, result_t, ratio)
            img_warmer = Image.blend(img_pil, result_s, ratio)

            img_array_colder = self.PIL2State(img_colder)
            img_array_warmer = self.PIL2State(img_warmer)
            img_array_awb = self.PIL2State(result_awb)

            batch_images_colder[i] = img_array_colder
            batch_images_warmer[i] = img_array_warmer
            batch_images_awb[i] = img_array_awb
        return torch.from_numpy(batch_images_colder), torch.from_numpy(batch_images_warmer), torch.from_numpy(batch_images_awb)

    def process_batch_images(self, processor, batch_images):
        size = batch_images.shape
        for i in range(size[0]):
            cur_cv_img = batch_images[i]
            img_pil = self.State2PIL(cur_cv_img)
            img_process_pil = processor.get_process_image(img_pil)
            img_array = self.PIL2State(img_process_pil)
            batch_images[i] = img_array
        return torch.from_numpy(batch_images)

    def PIL2State(self, img_pil):
        img_array = np.array(img_pil)
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        img_array = (img_array / 255).astype(np.float32)  # Scale to [0, 1]
        return img_array

    def State2PIL(self, img_state):
        cur_cv_img = img_state
        img_data = np.transpose(cur_cv_img, (1, 2, 0))  # CHW to HWC
        img_data = img_data[:, :, ::-1]  # BGR to RGB
        img_data = (img_data * 255).astype(np.uint8)  # Scale to [0, 255]
        img_pil = Image.fromarray(img_data)
        return img_pil
