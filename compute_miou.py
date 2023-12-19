import os
import cv2
import numpy as np
from PIL import Image

def resize_and_crop(image_array):
    # 画像を510x510にリサイズ
    resized_image = Image.fromarray(image_array).resize((510, 510))

    # NumPy配列に変換
    resized_array = np.array(resized_image)

    # 384x384の中央領域をクロップ
    start = (510 - 384) // 2  # 中央の開始位置を計算
    cropped_array = resized_array[start:start+384, start:start+384]

    return cropped_array

def comput_iou(gt, pred):

    pred = read_image(pred)
    gt = read_image(gt)
    gt = resize_and_crop(gt)
    threshold = np.mean(pred)
    std = np.std(pred)
    threshold = threshold + std
    pred = np.where(pred >= threshold, 1, 0)
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def read_image(file_path):
    image = Image.open(file_path).convert("L")
    image_array = np.array(image)
    return image_array



attn_dir = "output/2023-12-18-11:10:19/rise"
image_dir = "datasets/CUB/test"
def get_filenames(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

attn_files = get_filenames(attn_dir)
image_files = get_filenames(image_dir)
print(attn_files)
print(image_files)
ious = []
for attn_file in attn_files:
    attn_base_name = os.path.basename(attn_file)
    for image_file in image_files:
        image_base_name = os.path.basename(image_file)
        if attn_base_name == image_base_name:
            print(attn_base_name, image_base_name)
            ious.append(comput_iou(image_file, attn_file))
            break
print(np.mean(ious))
