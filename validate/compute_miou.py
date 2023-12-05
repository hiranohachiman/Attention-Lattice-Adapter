import numpy as np
import cv2
import os
from tqdm import tqdm

gt_dir = "datasets/CUB/extracted_test_segmentation_old"


def calculate_iou(gt_img, pred_img):
    # gt_imgを2値画像として扱い、0と1のみの値を持つように変換
    gt_img = (gt_img > 127).astype(np.uint8)
    if pred_img.shape[2] == 3:
        # pred_imgをグレースケールに変換
        pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_RGB2GRAY)
    # 2値画像に変換
    _, pred_binary = cv2.threshold(pred_gray, 127, 255, cv2.THRESH_BINARY)

    # 交差領域の計算
    intersection = np.logical_and(gt_img, pred_binary)
    # 統合領域の計算
    union = np.logical_or(gt_img, pred_binary)

    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_mean_iou(directory_gt, directory_pred):
    gt_files = sorted(os.listdir(directory_gt))
    pred_files = sorted(os.listdir(directory_pred))
    # 同じ名前のファイルを持つことを確認
    gt_names = [os.path.splitext(f)[0] for f in gt_files]
    pred_names = [os.path.splitext(f)[0] for f in pred_files]
    assert gt_names == pred_names, "The two directories do not contain the same set of image filenames."

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


def main(args):
    print("computing iou...")
    mean_iou = compute_mean_iou(gt_dir, args.img_dir)
    print("mean_iou: ", mean_iou, f"predict_mode: {args.predict_mode}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--img_dir", type=str, required=True
    )
    parser.add_argument(
        "--predict_mode", type=str, required=True
    )

    args = parser.parse_args()
    main(args)
