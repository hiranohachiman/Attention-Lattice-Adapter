import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torchvision.transforms.functional as TF

def save_overlay_image_with_matplotlib(tensor, base_image, dir_path='output/test/matplt', file_name_pattern='{time}.png'):
    os.makedirs(dir_path, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = file_name_pattern.format(time=current_time)
    save_path = os.path.join(dir_path, file_name)

    img_np = tensor[0].cpu().detach().numpy()
    cmap = plt.get_cmap('RdBu_r')
    overlay = cmap(img_np)
    overlay = (overlay[:, :, :3] * 255).astype(np.uint8)
    overlay_resized = np.transpose(np.array(Image.fromarray(overlay).resize((base_image.shape[2], base_image.shape[1]))), (2, 0, 1))

    base_img_np = (base_image.cpu().numpy() * 255).astype(np.uint8)
    combined_image = 0.5 * base_img_np + 0.5 * overlay_resized
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    plt.imsave(save_path, np.transpose(combined_image, (1, 2, 0)))
    return save_path


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

    return save_path

def attn2binary_mask(attn_map, w, h, output_file, threshold="mean"):
    # アテンションマップを正規化
    tensor_np = attn_map.cpu().detach().numpy()
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())
    if threshold == "mean":
        threshold = tensor_np.mean()
    # アテンションマップをimg_dataのサイズにリサイズ
    attn_map_resized = torch.tensor(tensor_np)
    attn_map_resized = TF.resize(attn_map_resized, (h, w))

    # アテンションマップを2値化
    attn_map_binary = torch.where(attn_map_resized > threshold, 1.0, 0.0)

    # 2値化されたアテンションマップを1チャネルの画像として保存
    img_pil = TF.to_pil_image(attn_map_binary.squeeze(0))  # squeezeで1チャンネルにする
    img_pil = img_pil.convert('L')  # グレースケールに変換
    img_pil.save(output_file)

def save_img_data(img_data, output_file):
    # img_dataをPILイメージに変換して保存
    img_pil = TF.to_pil_image(img_data)
    img_data_output_file = f"{output_file}_image.png"
    img_pil.save(img_data_output_file)
    return img_data_output_file

def save_attn_map(attn_map, w, h, output_file):
    # アテンションマップを正規化
    tensor_np = attn_map.cpu().detach().numpy()
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())

    # アテンションマップをリサイズ
    attn_map_resized = Image.fromarray((tensor_np[0] * 255).astype(np.uint8)).resize((h, w), Image.BILINEAR)

    # リサイズしたアテンションマップを保存
    attn_map_output_file = f"{output_file}"
    attn_map_resized.save(attn_map_output_file)
    return attn_map_output_file
