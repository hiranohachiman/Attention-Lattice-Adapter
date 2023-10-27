# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Mar  8 2023, 16:27:05) 
# [GCC 9.4.0]
# Embedded file name: /home/initial/workspace/smilab23/graduation_research/SAN/san/model/visualize.py
# Compiled at: 2023-10-25 18:19:35
# Size of source mod 2**32: 4109 bytes
from typing import List
import open_clip, torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Tuple
from PIL import Image
import os, datetime, numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2

def save_overlay_image_with_matplotlib(tensor, base_image, dir_path='output/test/matplt', file_name_pattern='{time}.png'):
    os.makedirs(dir_path, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = file_name_pattern.format(time=current_time)
    save_path = os.path.join(dir_path, file_name)
    img_np = tensor[0].cpu().detach().numpy()
    cmap = plt.get_cmap('RdBu_r')
    overlay = cmap(img_np)
    overlay = (overlay[:, :, :3] * 255).astype(np.uint8)
    overlay_resized = np.transpose(np.array(Image.fromarray(overlay).resize((base_image.shape[2], base_image.shape[1]))), (2,
                                                                                                                           0,
                                                                                                                           1))
    base_img_np = (base_image.cpu().numpy() * 255).astype(np.uint8)
    combined_image = 0.5 * base_img_np + 0.5 * overlay_resized
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    plt.imsave(save_path, np.transpose(combined_image, (1, 2, 0)))
    return     


def save_side_by_side_image(tensor, base_image, dir_path='output/test/horimg', file_name_pattern='{time}.png'):
    os.makedirs(dir_path, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = file_name_pattern.format(time=current_time)
    save_path = os.path.join(dir_path, file_name)
    tensor_np = tensor.cpu().detach().numpy()
    base_image_np = base_image.cpu().detach().numpy().transpose(1, 2, 0)
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())
    tensor_colored = plt.cm.jet(tensor_np[0])
    tensor_rgb = tensor_colored[:, :, :3]
    tensor_resized = cv2.resize(tensor_rgb, (base_image_np.shape[1], base_image_np.shape[0]))
    combined_image = np.hstack((tensor_resized, base_image_np))
    cv2.imwrite(save_path, (combined_image * 255).astype(np.uint8))


def save_image_to_directory(tensor, dir_path='output/test/tmp', file_name_pattern='{time}.png'):
    os.makedirs(dir_path, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = file_name_pattern.format(time=current_time)
    save_path = os.path.join(dir_path, file_name)
    np_img = tensor.cpu().detach().numpy().transpose(1, 2, 0) * 255
    np_img = np_img.astype('uint8')
    img = Image.fromarray(np_img)
    img.save(save_path)
    return     