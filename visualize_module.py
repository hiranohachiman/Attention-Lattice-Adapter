import cv2
import numpy as np
import os

def get_color_image(path):
    """
    path からカラー画像をcv2で読み込んで返す
    """
    return cv2.imread(path, cv2.IMREAD_COLOR)


def get_grayscale_image(path):
    """
    path から白黒画像をcv2で読み込んで返す
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_directory_path(dir_path):
    """
    ディレクトリ内のファイルパスをリストに格納して返す
    """
    return [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]

def get_all_file_in_dir(dir_path):
    """
    dir_path 内のディレクトリを再帰的に探索してすべてのファイル名をリストにして返す
    """
    file_list = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def grayscale2binary(image, threshold="mean+std"):
    """
    imageを thresholdで2値化する
    """
    if threshold == "mean+std":
        thresh_val = image.mean() + image.std()
    elif type(threshold) in [int, float]:
        thresh_val = threshold
    else:
        raise ValueError("Invalid threshold value")
    _, binary_image = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)
    return binary_image


def compute_iou(image1, image2):
    """
    2つのimageのiouを計算する
    """
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def reshape(image, h, w):
    """
    imageを(h, w)にreshapeする
    """
    return cv2.resize(image, (w, h))


def overlay_image(image1, image2):
    """
    2つのimageを0.5:0.5の比率で重ね合わせる
    """
    return cv2.addWeighted(image1, 0.5, image2, 0.5, 0)


def save_images(image, path):
    """
    imageをpathに保存する
    """
    print(path)
    cv2.imwrite(path, image)


def gray_image2jet(image):
    """
    グレースケール画像をカラーマップ'jet'に変換する
    """
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def apply_gaussian(image):
    """
    imageにガウシアンカーネルを適用する
    """
    return cv2.GaussianBlur(image, (5, 5), 0)