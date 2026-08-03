import torch
import torchvision.io as io
import torchvision.transforms as transforms
import numpy as np
import os
import cv2

def read_image(filepath):
    image = io.read_image(filepath, io.ImageReadMode.RGB)
    return image

def write_image(filepath, src):
    io.write_png(src, filepath)

def rgb2ycbcr(src):
    R = src[0]
    G = src[1]
    B = src[2]

    ycbcr = torch.zeros(size=src.shape)
    ycbcr[0] =  0.299 * R + 0.587 * G + 0.114 * B
    ycbcr[1] =  -0.16874 * R - 0.33126 * G + 0.5 * B + 128
    ycbcr[2] =  0.5 * R - 0.41869 * G - 0.08131 * B + 128

    # Y in range [16, 235]
    ycbcr[0] = torch.clip(ycbcr[0], 16, 235)
    # Cb, Cr in range [16, 240]
    ycbcr[[1, 2]] = torch.clip(ycbcr[[1, 2]], 16, 240)
    ycbcr = ycbcr.type(torch.uint8)
    return ycbcr

def ycbcr2rgb(src):
    Y = src[0]
    Cb = src[1]
    Cr = src[2]

    rgb = torch.zeros(size=src.shape)
    rgb[0] = Y + 1.402 * Cr - 179.456
    rgb[1] = Y - 0.34414 * Cb - 0.71414 * Cr + 135.45984
    rgb[2] = Y + 1.772 * Cb - 226.816

    rgb = torch.clip(rgb, 0, 255)
    rgb = rgb.type(torch.uint8)
    return rgb

# list all file in dir and sort
def sorted_list(dir):
    ls = os.listdir(dir)
    ls.sort()
    for i in range(0, len(ls)):
        ls[i] = os.path.join(dir, ls[i])
    return ls

def resize_bicubic(src, h, w):
    image = transforms.Resize((h, w), transforms.InterpolationMode.BICUBIC)(src)
    return image

def gaussian_blur(src, ksize=3, sigma=0.5):
    blur_image = transforms.GaussianBlur(kernel_size=ksize, sigma=sigma)(src)
    return blur_image

def upscale(src, scale):
    h = int(src.shape[1] * scale)
    w = int(src.shape[2] * scale)
    image = resize_bicubic(src, h, w)
    return image

def downscale(src, scale):
    h = int(src.shape[1] / scale)
    w = int(src.shape[2] / scale)
    image = resize_bicubic(src, h, w)
    return image

def make_lr(src, scale=3):
    h = src.shape[1]
    w = src.shape[2]
    lr_image = downscale(src, scale)
    lr_image = resize_bicubic(lr_image, h, w)
    return lr_image

def norm01(src):
    return src / 255

def denorm01(src):
    return src * 255

def exists(path):
    return os.path.exists(path)

def exist_value(tensor, value):
    num_elements = tensor.shape[0]
    for i in range(0, num_elements):
        sum_values = torch.sum(tensor[i] == value)
        if sum_values > 0:
            return True
    return False

def PSNR(y_true, y_pred, max_val=1.0):
    return 0

def random_crop(src, h, w):
    crop = transforms.RandomCrop([h, w])(src)
    return crop

def random_transform(src):
    _90_left, _90_right, _180 = 1, 3, 2
    operations = {
        0 : (lambda x : x                                       ),
        1 : (lambda x : torch.rot90(x, k=_90_left,  dims=(1, 2))),
        2 : (lambda x : torch.rot90(x, k=_90_right, dims=(1, 2))),
        3 : (lambda x : torch.rot90(x, k=_180,      dims=(1, 2))),
        4 : (lambda x : torch.fliplr(x)                         ),
        5 : (lambda x : torch.flipud(x)                         ),
    }
    idx = np.random.choice([0, 1, 2, 3, 4, 5])
    image_transform = operations[idx](src)
    return image_transform

def shuffle(X, Y):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of elements")
    indices = np.arange(0, X.shape[0])
    np.random.shuffle(indices)
    X = torch.index_select(X, dim=0, index=torch.as_tensor(indices))
    Y = torch.index_select(Y, dim=0, index=torch.as_tensor(indices))
    return X, Y

# todo: draw action map based on 2d action matrix
def draw_action_map(actions, color_table):
    h = actions.shape[0]
    w = actions.shape[1]
    action_map = torch.zeros((3, h, w), dtype=torch.uint8)
    for i in range(0, h):
        for j in range(0, w):
            action_map[:, i, j] = color_table[actions[i, j]]
    return action_map

def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy().copy()

def to_cpu(tensor):
    return tensor.detach().cpu()


def bgr2lab_tensor_converter(src):
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu().numpy()
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0,2,3,1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0,b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_BGR2Lab)
    return np.transpose(dst, (0,3,1,2))

def log_action_matrix(file, t, action, N_ACTIONS):
    ACTION_DENOISE = 1
    ACTION_ENHANCE = ACTION_DENOISE + 1
    ACTION_AWB_COLDER = ACTION_ENHANCE + 1
    ACTION_AWB_WARMER = ACTION_AWB_COLDER + 1
    ACTION_LOW_LIGHT = ACTION_AWB_WARMER + 1

    file.write(f'\tStep {t}:\n')

    for x in range(0, N_ACTIONS):
        cur_action = x
        per = np.round(np.sum(action == cur_action) / action.size * 100, 2)

        if cur_action == 0:
            action_name = "No Action"
        elif cur_action == ACTION_DENOISE:
            action_name = "Denoise"
        elif cur_action == ACTION_ENHANCE:
            action_name = "Enhance"
        elif cur_action == ACTION_AWB_COLDER:
            action_name = "AWB Colder"
        elif cur_action == ACTION_AWB_WARMER:
            action_name = "AWB Warmer"
        elif cur_action == ACTION_LOW_LIGHT:
            action_name = "Low Light Enhancement"
        else:
            action_name = "Unknown Action"

        file.write(f'\t\taction: {cur_action} ({action_name}) \t ratio {per}%\n')

    file.write(f'\n')

def print_action_matrix(t, action):
    N_ACTIONS=6

    ACTION_DENOISE = 1
    ACTION_ENHANCE = ACTION_DENOISE + 1
    ACTION_AWB_COLDER = ACTION_ENHANCE + 1
    ACTION_AWB_WARMER = ACTION_AWB_COLDER + 1
    ACTION_LOW_LIGHT = ACTION_AWB_WARMER + 1

    print(f'\tStep {t}:\n')

    for x in range(0, N_ACTIONS):
        cur_action = x
        per = np.round(np.sum(action == cur_action) / action.size * 100, 2)

        if cur_action == 0:
            action_name = "No Action"
        elif cur_action == ACTION_DENOISE:
            action_name = "Denoise"
        elif cur_action == ACTION_ENHANCE:
            action_name = "Enhance"
        elif cur_action == ACTION_AWB_COLDER:
            action_name = "AWB Colder"
        elif cur_action == ACTION_AWB_WARMER:
            action_name = "AWB Warmer"
        elif cur_action == ACTION_LOW_LIGHT:
            action_name = "Low Light Enhancement"
        else:
            action_name = "Unknown Action"

        print(f'\t\taction: {cur_action} ({action_name}) \t ratio {per}%\n')

    print(f'\n')


def writer_action_matrix(episode, t, action, writer, if_seg):
    N_ACTIONS = 6

    ACTION_DENOISE = 1
    ACTION_ENHANCE = ACTION_DENOISE + 1
    ACTION_AWB_COLDER = ACTION_ENHANCE + 1
    ACTION_AWB_WARMER = ACTION_AWB_COLDER + 1
    ACTION_LOW_LIGHT = ACTION_AWB_WARMER + 1

    action_names = {
        0: "No Action",
        ACTION_DENOISE: "Denoise",
        ACTION_ENHANCE: "Enhance",
        ACTION_AWB_COLDER: "AWB Colder",
        ACTION_AWB_WARMER: "AWB Warmer",
        ACTION_LOW_LIGHT: "Low Light Enhancement"
    }

    suffix = "_no_seg"
    if if_seg == "True":
        suffix = "_seg"
    else:
        suffix = "_no_seg"

    action_ratios = {action_name: 0 for action_name in action_names.values()}

    for cur_action in range(N_ACTIONS):
        action_name = action_names.get(cur_action, "Unknown Action")
        per = np.round(np.sum(action == cur_action) / action.size * 100, 2)
        action_ratios[action_name] = per
        writer.add_scalars(f'{t}_step{suffix}', {f'{action_names[cur_action]}': per}, global_step=episode)

    print(f"{suffix} Epi {episode} - Step {t} - ratios: {action_ratios}")

def log_action_ratios(actions_cpu, file, indices_index, t, N_ACTIONS=6):
    action_names = {
        0: "No Action",
        1: "Denoise",
        2: "Enhance",
        3: "AWB Colder",
        4: "AWB Warmer",
        5: "Low Light"
    }

    unique_actions, counts = np.unique(actions_cpu, return_counts=True)

    total_actions = np.sum(counts)

    file.write(f"\nImage {indices_index}, correction round: {t+1}\n")

    for action, count in zip(unique_actions, counts):
        action_name = action_names.get(action, f"Action {action}")
        action_ratio = count / total_actions
        file.write(f"Action: {action_name} (id: {action}), count: {count}, ratio: {action_ratio:.2f}\n")
