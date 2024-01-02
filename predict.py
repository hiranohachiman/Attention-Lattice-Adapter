from typing import List, Union

import numpy as np

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import os

import huggingface_hub
import torch
import numpy as np
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, random_color
from huggingface_hub import hf_hub_download
from PIL import Image
import cv2
from torchsummary import summary
from torchvision.transforms import functional as TF
from san.data.dataloader import train_dataset, valid_dataset, test_dataset, _preprocess, _make_transform, _make_transform_for_mask

from san import add_san_config
from san.data.datasets.register_cub import CLASS_NAMES
from san.model.visualize import attn2binary_mask, save_img_data, save_attn_map
from san.model.san import SAN
from LambdaAttentionBranchNetworks.metrics.patch_insdel import PatchInsertionDeletion
from tqdm import tqdm


model_cfg = {
    "san_vit_b_16": {
        "config_file": "configs/san_clip_vit_res4_coco.yaml",
        "model_path": "huggingface:san_vit_b_16.pth",
    },
    "san_vit_large_16": {
        "config_file": "configs/san_clip_vit_large_res4_coco.yaml",
        "model_path": "huggingface:san_vit_large_14.pth",
    },
}

label_file = "datasets/CUB/id_score_sample.txt"
config_file = "configs/san_clip_vit_res4_coco.yaml"
mask_dir = "datasets/CUB/extracted_test_segmentation_old"


def download_model(model_path: str):
    """
    Download the model from huggingface hub.
    Args:
        model_path (str): the model path
    Returns:
        str: the downloaded model path
    """
    if "HF_TOKEN" in os.environ:
        huggingface_hub.login(token=os.environ["HF_TOKEN"])
    model_path = model_path.split(":")[1]
    model_path = hf_hub_download("Mendel192/san", filename=model_path)
    return model_path


def setup(config_file: str, device=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device or "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg

def my_load_model(config_file: str, model_path: str):
    cfg = setup(config_file)
    model = SAN(**SAN.from_config(cfg))
    print('Loading model from: ', model_path)
    model = torch.load(model_path)
    print('Loaded model from: ', model_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    return model

def get_attn_dir(args):
    attn_dir = args.output_dir.replace("id", "attn_map")
    return attn_dir

def get_image_data_details(line, args):
    img_path = line.split(',')[0]
    label = int(line.split(',')[1].replace('\n', '').replace(' ', ''))
    output_file = os.path.join(args.output_dir, img_path.replace("test/","",).replace("/","_").replace(" ",""))
    attn_path = os.path.join(get_attn_dir(args), img_path.replace("test/","",).replace("/","_").replace(" ",""))
    img_path = os.path.join('datasets/CUB/', img_path.replace(' ', ''))

    return (img_path, label, attn_path, output_file)

def normalize_batch(batch):
    # バッチごとにループ
    for i in range(batch.size(0)):
        # i番目のバッチを取得
        batch_i = batch[i]
        # バッチ内の最小値と最大値を取得
        min_val = torch.min(batch_i)
        max_val = torch.max(batch_i)
        # 0-1の範囲に正規化
        batch[i] = (batch_i - min_val) / (max_val - min_val)
    return batch

def save_attn_map(attn_map, path):
    # アテンションマップを正規化
    attn_map = normalize_batch(attn_map)
    attn_map = F.interpolate(attn_map, size=(384, 384), mode='bilinear', align_corners=False)
    # バッチの最初の要素を選択し、チャンネルの次元を削除
    attn_map = attn_map[0].squeeze()
    # PyTorch TensorをNumPy配列に変換
    attn_map = attn_map.cpu().detach().numpy()
    # attn_mapを0から1の範囲に正規化
    # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    # 値を0から255の範囲にスケーリング
    attn_map = attn_map * 255
    # 整数型にキャスト
    attn_map = attn_map.astype(np.uint8)
    # PIL Imageに変換
    attn_map = Image.fromarray(attn_map)
    # 画像を保存
    attn_map.save(path)

def predict_one_shot(model, image_path, output_path, mode="iou",device="cuda"):
    model = model.to(device)
    model.eval()
    img = cv2.imread(image_path)
    img = img[:, :, ::-1] # BGR to RGB.

    # to PIL.Image
    img = Image.fromarray(img)
    transform = _make_transform(istrain=False)
    img = transform(img)
    img = img.unsqueeze(0)
    logits, _, attn_map = model(img)
    # save attn_map
    save_attn_map(attn_map, os.path.join(output_path, f"{os.path.basename(image_path)}_attn_map.png"))
    _, predicted = torch.max(logits.data, 1)
    return predicted

def remove_other_components(mask, threshold=0.3):
    # Binarize the mask
    binary_mask = (mask > threshold).astype(np.uint8)
    # Detect connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    # Calculate the maximum value in the original mask for each component
    max_values = [np.max(mask[labels == i]) for i in range(num_labels)]
    # Find the component with the largest maximum value
    largest_max_value_component = np.argmax(max_values)
    # second_index = np.argsort(max_values)[-2]
    # third_index = np.argsort(max_values)[-3]
    # print(max_values)
    # print(largest_max_value_component)
    # print(second_index)
    # Create a new mask where all components other than the one with the largest max value are removed
    first_mask = np.where(labels == largest_max_value_component, mask*3, 0)
    # second_mask = np.where(labels == second_index, mask, 0)
    # third_mask = np.where(labels == third_index, mask / 3, 0)
    new_mask = first_mask
    # new_mask = first_mask + second_mask + third_mask
    return new_mask

def apply_heat_quantization(attention, q_level: int = 3):
    max_ = attention.max()
    min_ = attention.min()

    # quantization
    bin = np.linspace(min_, max_, q_level)
    # apply quantization
    for i in range(q_level - 1):
        attention[(attention >= bin[i]) & (attention < bin[i + 1])] = bin[i]

    return attention


def main(args):
    model_paths = [file for file in os.listdir(args.model_dir) if 'epoch_' in file]
    assert len(model_paths) > 0, "No model found in the model directory."
    assert len(model_paths) == 1, "More than one model found in the model directory."
    model = my_load_model(config_file, os.path.join(args.output_dir,model_paths[0]))
    pred_dir = os.path.join(args.output_dir, "test_attn_maps")
    print(compute_mean_iou(mask_dir, pred_dir))
    compute_id(args, model)

def compute_mean_iou(directory_gt, directory_pred):
    gt_files = sorted(os.listdir(directory_gt))
    pred_files = sorted(os.listdir(directory_pred))
    # 同じ名前のファイルを持つことを確認
    gt_names = [os.path.splitext(f)[0] for f in gt_files]
    pred_names = [os.path.splitext(f)[0] for f in pred_files]
    assert len(gt_names) == len(pred_names), "The two directories do not contain the same set of image filenames."

    ious = []

    for file_name in tqdm(gt_files):
        gt_path = os.path.join(directory_gt, file_name).replace("png","jpg")
        # print(pred_names)
        for pred_name in pred_files:
            # print(pred_name, file_name)
            if pred_name in file_name.replace("png","jpg"):
                pred_path = os.path.join(directory_pred, pred_name)
                # print(pred_path, gt_path)
                break

        # ファイルが画像であるかの簡易的なチェック
        if gt_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            gt_img = cv2.imread(gt_path.replace("jpg","png"), cv2.IMREAD_GRAYSCALE)
            gt_img = resize_image(gt_img)
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            # gt_img = apply_gaussian_filter(gt_img)
            ious.append(calculate_iou(gt_img, pred_img))

            # # Save gt_img as attn_gt_mask.png
            # cv2.imwrite("attn_gt_mask.png", gt_img)

            # # Save pred_img as attn_map.png
            # cv2.imwrite("attn_map.png", pred_img)

    return np.mean(ious)

def resize_image(image, size=(384, 384)):
    """
    Resize the image to the specified size.
    """
    resized_image = cv2.resize(image, size)
    return resized_image

def remove_other_components(mask, threshold=0.3):
    # Calculate the average pixel value
    avg_pixel = np.mean(mask)

    # Calculate the standard deviation
    std_dev = np.std(mask)

    # Calculate the threshold by adding the standard deviation to the average pixel value
    threshold = avg_pixel + std_dev

    # Binarize the mask using the threshold
    binary_mask = np.where(mask > threshold, 255, 0).astype(np.uint8)

    return binary_mask


def calculate_iou(gt_image, pred_image):
    """
    Calculate the Intersection over Union (IoU) between the ground truth and predicted images.
    """
    pred_image = remove_other_components(pred_image)
    intersection = np.logical_and(gt_image, pred_image)
    union = np.logical_or(gt_image, pred_image)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def compute_id(args, model):
    metrics = PatchInsertionDeletion(
        model=model,
        batch_size=8,
        patch_size=1,
        step=3072,
        dataset="str",
        device="cuda",)

    x = 0
    with open(label_file) as (f):
        lines = f.readlines()
        for line in tqdm(lines):
            img_path, _, _, _ = get_image_data_details(line, args)
            with torch.no_grad():
                label = predict_one_shot(model, img_path, args.output_dir, mode="id")
            img = cv2.imread(img_path)
            img = img[:, :, ::-1] # BGR to RGB.
            img = Image.fromarray(img)
            transform = _make_transform(istrain=False)
            single_image = transform(img).cpu().numpy()
            single_target = label

            single_attn = cv2.imread(os.path.join(args.output_dir, f"{os.path.basename(img_path)}_attn_map.png"), cv2.IMREAD_GRAYSCALE)
            single_attn = Image.fromarray(single_attn)
            transform_for_mask = _make_transform_for_mask(istrain=False)
            single_attn = transform_for_mask(single_attn).cpu().numpy()
            single_attn = apply_heat_quantization(single_attn)
            metrics.evaluate(single_image, single_attn, single_target)
            metrics.save_roc_curve(os.path.join(args.output_dir, "attn_app"))

            x += 1
            if x % 50 == 0:
                print("total_insertion:", metrics.total_insertion)
                print("total_deletion:", metrics.total_deletion)
                print("average", (metrics.total_insertion - metrics.total_deletion) / x)
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print("average_insertion:", metrics.total_insertion / x)
    print("average_deletion:", metrics.total_deletion / x)
    print("average", (metrics.total_insertion - metrics.total_deletion) / x)
    print("!!!!!!!!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser.add_argument(
    #     "--predict_mode", type=str, required=True, help="select from ['bird', 'name', 'class', 'none']"
    # )

    parser.add_argument(
        "--model_dir", type=str, required=True, help="path to model file"
    )


    parser.add_argument(
        "--output_dir", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    main(args)
