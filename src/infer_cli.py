#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True

import os
import argparse
import pickle
import numpy as np
import torch
from PIL import Image

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

from neuralnet import PixelRL_model
from State import State
from utils.common import bgr2lab_tensor_converter

torch.manual_seed(1)

DEFAULT_WEIGHTS = os.path.join(_script_dir, "weights", "pixelrl_seg", "best_model.pt")


def load_mask(seg_path):
    if seg_path and os.path.exists(seg_path):
        with open(seg_path, 'rb') as f:
            mask_list = pickle.load(f)
        if isinstance(mask_list, list):
            return np.stack(mask_list, axis=0)[np.newaxis, ...].astype(int)
        return mask_list[np.newaxis, ...].astype(int)
    return None


def process_image(image, weights_path=None, device=None, t_max=10, seg_path=None):
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    weights_path = weights_path or DEFAULT_WEIGHTS

    model = PixelRL_model(6).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()

    state_manager = State(str(device))

    # PIL -> tensor (1, 3, H, W) BGR [0-1]
    img_np = np.array(image.convert('RGB'))
    img_bgr = img_np[:, :, ::-1].copy()
    img_norm = (img_bgr / 255.0).astype(np.float32)
    src_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
    h, w = src_tensor.shape[2], src_tensor.shape[3]

    mask_arr = load_mask(seg_path)
    use_seg = mask_arr is not None
    if mask_arr is None:
        mask_arr = np.zeros((1, 1, h, w), dtype=int)
    text_arr = np.array([[]])

    dummy_dst = src_tensor.copy()
    state_manager.reset(dummy_dst, src_tensor, mask_arr, text_arr)

    lab_cur = torch.from_numpy(bgr2lab_tensor_converter(state_manager.image_state)).to(device)

    with torch.no_grad():
        for t in range(t_max):
            state_manager.set(lab_cur)
            state_var = state_manager.tensor.to(device)
            actions, _, inner_state = model.choose_best_actions(state_var)

            actions_cpu = actions.cpu()
            inner_state_cpu = inner_state.cpu()

            if use_seg:
                state_manager.step_seg(actions_cpu, inner_state_cpu)
            else:
                state_manager.step(actions_cpu, inner_state_cpu)

            lab_cur = torch.from_numpy(bgr2lab_tensor_converter(state_manager.image_state)).to(device)

    # tensor -> PIL
    img_data = state_manager.image_state[0].numpy()
    img_data = np.transpose(img_data, (1, 2, 0))[:, :, ::-1]  # CHW BGR -> HWC RGB
    img_data = (img_data * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output', '-o', type=str, default='result.png',
                        help='Output image path (default: result.png)')
    parser.add_argument('--seg', '-s', type=str, default=None,
                        help='Segmentation mask file path (.pkl)')
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Model weights path')
    parser.add_argument('--device', '-d', type=str, default=None,
                        help='Inference device (e.g. cuda:0, cpu)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of inference steps (default: 10)')

    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.join(_script_dir, args.input)
    if not os.path.exists(input_path):
        print(f"Error: input not found: {input_path}")
        sys.exit(1)

    image = Image.open(input_path).convert('RGB')
    print(f"Input: {input_path} ({image.size[0]}x{image.size[1]})")

    if args.seg:
        seg_path = args.seg if os.path.isabs(args.seg) else os.path.join(_script_dir, args.seg)
        print(f"Seg: {seg_path}")
    else:
        seg_path = None
        print("Seg: None (no semantic segmentation)")

    print(f"Processing (steps={args.steps})...")
    result = process_image(image, args.weights, args.device, args.steps, seg_path)

    output_path = args.output if os.path.isabs(args.output) else os.path.join(_script_dir, args.output)
    result.save(output_path)
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
