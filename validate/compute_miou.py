import numpy as np
import cv2
import os
from tqdm import tqdm

gt_dir = "../datasets/CUB/extracted_test_segmentation_old"
attn_dir = "../output/2023-12-18-11:10:19/test_attn_maps"


def get_filenames(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


def process_files(file1, file2):
    # Load RGB image
    rgb_image = cv2.imread(file1)

    # Resize RGB image to 510x510
    rgb_image = cv2.resize(rgb_image, (510, 510))

    # Crop the center of the image to 384x384
    height, width = rgb_image.shape[:2]
    start_x = (width - 384) // 2
    start_y = (height - 384) // 2
    cropped_image = rgb_image[start_y:start_y+384, start_x:start_x+384]

    # Load grayscale image
    grayscale_image = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)

    # Convert images to numpy arrays
    cropped_image = np.array(cropped_image)
    grayscale_image = np.array(grayscale_image)

    return cropped_image, grayscale_image

def make_mask(attn_map, threshold):
    if threshold == "mean":
        threshold = np.mean(attn_map)
    mask = np.where(attn_map >= threshold, 1, 0)
    return mask

def calculate_iou(gt_img, pred_img):
    # ガウシアンフィルタを適用
    pred_img = gaussian_filter(pred_img)

    # 二値化
    gt_img = np.where(gt_img >= 128, 1, 0)
    pred_img = np.where(pred_img >= 128, 1, 0)

    # 論理積
    intersection = np.logical_and(gt_img, pred_img)
    # 論理和
    union = np.logical_or(gt_img, pred_img)
    # IOUスコア
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def compute_mean_iou(directory_gt, directory_pred):
    attn_files = get_filenames(attn_dir)
    gt_files = get_filenames(gt_dir)
    assert attn_files == gt_files, "The two directories do not contain the same set of image filenames."

    ious = []

    for file_name in tqdm(gt_files):
        gt_path = os.path.join(directory_gt, file_name)
        pred_path = os.path.join(directory_pred, file_name).replace("png","jpg")

        # ファイルが画像であるかの簡易的なチェック
        if gt_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred_img = cv2.imread(pred_path)
            ious.append(calculate_iou(gt_img, pred_img))

    return np.mean(ious)

def gaussian_filter(img):
    # ガウシアンフィルタを適用
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def main(args):
    print("computing iou...")
    mean_iou = compute_mean_iou(gt_dir, args.img_dir)
    print("mean_iou: ", mean_iou, f"predict_mode: {args.predict_mode}")
