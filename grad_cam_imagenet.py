from typing import List, Union

import numpy as np
import gc

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import os

# from captum.attr import LRP, IntegratedGradients, GuidedBackprop
from pytorch_grad_cam import GradCAM, ScoreCAM

import huggingface_hub
import torch
import numpy as np
from torchinfo import summary
from torchvision import transforms
from captum.attr import LRP, IntegratedGradients, GuidedBackprop

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
from san.data.dataloader import _preprocess
from torchvision.transforms import InterpolationMode

from san import add_san_config
from san.data.datasets.register_cub import CLASS_NAMES
from san.model.visualize import attn2binary_mask, save_img_data, save_attn_map
from san.model.san import SAN

# from LambdaAttentionBranchNetworks.metrics.patch_insdel import PatchInsertionDeletion
from functools import partial
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

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
# model_path = "output/2023-12-13-23:34:53/epoch_48.pth"
# lora_path = "output/2023-12-13-23:34:53/lora_epoch_48.pth"


def my_load_model(config_file: str, model_path: str):
    print("Loading model from: ", model_path)
    model = torch.load(model_path)
    print("Loaded model from: ", model_path)
    model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
    return model


def get_attn_dir(args):
    attn_dir = args.output_dir.replace("id", "attn_map")
    return attn_dir


def get_image_data_details_imagenet(line, args):
    img_path = line.split(",")[0].strip().replace("ImageNetS919", "ImageNet")
    label = int(line.split(",")[1].replace("\n", "").replace(" ", "")) - 1
    output_file = os.path.join(args.output_dir, img_path).strip()
    gt_path = line.split(",")[2].strip().replace("JPEG", "png")

    return (img_path, label, gt_path, output_file)


def get_image_data_details(line, args):
    img_path = line.split(",")[0]
    label = int(line.split(",")[1].replace("\n", "").replace(" ", ""))
    output_file = os.path.join(
        args.output_dir,
        img_path.replace(
            "test/",
            "",
        )
        .replace("/", "_")
        .replace(" ", ""),
    )
    attn_path = os.path.join(
        "datasets/CUB/",
        img_path.replace("/", "_")
        .replace(
            "test_",
            "extracted_test_segmentation_old/",
        )
        .replace(" ", "")
        .replace("jpg", "png"),
    )
    img_path = os.path.join("datasets/CUB/", img_path.replace(" ", ""))

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
    attn_map = F.interpolate(
        attn_map, size=(384, 384), mode="bilinear", align_corners=False
    )
    # バッチの最初の要素を選択し、チャンネルの次元を削除
    attn_map = attn_map[0].squeeze()
    # PyTorch TensorをNumPy配列に変換
    attn_map = attn_map.cpu().detach().numpy()
    # attn_mapを0から1の範囲に正規化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    # 値を0から255の範囲にスケーリング
    attn_map = attn_map * 255
    # 整数型にキャスト
    attn_map = attn_map.astype(np.uint8)
    # PIL Imageに変換
    attn_map = Image.fromarray(attn_map)
    # 画像を保存
    attn_map.save(path)


def predict_one_shot(model, image_path, output_path, device="cuda"):
    model = model.to(device)
    model.eval()
    img = Image.open(image_path)
    img = _preprocess(img)

    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        logits, _, attn_map = model(img)
    # save attn_map
    # save_attn_map(attn_map, os.path.join(output_path, f"{os.path.basename(image_path)}_attn_map.png"))
    _, predicted = torch.max(logits.data, 1)
    return predicted, attn_map


def lrp(model, image, label, output_path, device="cuda"):
    # image.requires_grad_(True)
    with torch.no_grad():
        lrp = LRP(model)
        attribution = lrp.attribute(image, target=label)
        attribution = attribution.mean(dim=1)
        # print("ig.shape", attribution.shape) # [1, 384, 384]
    return attribution


def ig(model, image, label, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    ig = IntegratedGradients(model.forward)
    attribution = ig.attribute(image, target=label, n_steps=5)
    attribution = attribution.mean(dim=1)
    # print("ig.shape", attribution.shape) # [1, 384, 384]
    return attribution


def gradient_backprop(model, image, label, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    guided_backprop = GuidedBackprop(model)
    attribution = guided_backprop.attribute(image, target=label)
    attribution = attribution.mean(dim=1)
    # print("gradient_backprop.shape", attribution.shape) [1, 384, 384]
    return attribution


def score_cam(model, image, label, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    score_cam = ScoreCAM(model, target_layers=[model.clipfeatureclassifier.conv2])
    attribution = score_cam(image, targets=None)
    # print("score_cam.shape", attribution.shape) (1, 384, 384)
    attribution = torch.tensor(attribution)
    return attribution


def grad_cam(model, image, label, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    grad_cam = GradCAM(model, target_layers=[model.clipfeatureclassifier.conv2])
    attribution = grad_cam(image, targets=None)
    # print("grad_cam.shape", attribution.shape) # (1, 384, 384)
    attribution = torch.tensor(attribution)
    return attribution


def set_gradient_true(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_iou(preds, masks, threshold="mean", true_value=1, false_value=0):
    preds = F.interpolate(preds, size=(384, 384), mode="bilinear", align_corners=False)
    # save_attn_map(preds[0], "attn_map.png") # This function call is commented out as it's not defined in this snippet.
    ious = []
    # Convert preds to numpy array
    preds = preds.squeeze(1).cpu().detach().numpy()
    masks = masks.cpu().numpy()
    assert preds.shape == masks.shape
    for i in range(len(preds)):
        std = np.std(preds[i])
        # Determine the threshold value
        if threshold == "mean":
            threshold = np.mean(preds[i])

        # Apply Gaussian filter to preds
        # preds[i] = gaussian_filter(preds[i], sigma=1)

        # Apply the threshold with custom true and false values
        preds[i] = np.where(preds[i] >= threshold, true_value, false_value)
        # Calculate Intersection over Union (IoU)
        intersection = np.logical_and(preds[i], masks[i])
        union = np.logical_or(preds[i], masks[i])
        assert np.sum(union) != 0
        iou_score = np.sum(intersection) / np.sum(union)
        ious.append(iou_score)
    iou = sum(ious) / len(ious)

    return iou


def for_lrp(args):
    model_paths = [file for file in os.listdir(args.model_dir) if "epoch_" in file]
    # assert len(model_paths) == 1
    lrp_ious = []
    sample_count = 0
    correct = 0
    x = 0
    model_paths.sort()
    model_path = os.path.join(args.model_dir, model_paths[-1])
    model = my_load_model(config_file, model_path)
    with open(label_file) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            img_path, label, gt_path, output_file = get_image_data_details(line, args)
            gt_mask = Image.open(gt_path)
            gt_mask = _preprocess(gt_mask, color="L")
            img = Image.open(img_path)
            img = _preprocess(img)
            img = img.unsqueeze(0)
            img = img.to("cuda")

            with torch.no_grad():
                pred_class, pred_attn = predict_one_shot(
                    model, img_path, args.output_dir
                )

            if int(label) == int(pred_class[0]):
                correct += 1
            sample_count += 1

            model = model.eval()
            model = set_gradient_true(model)
            lrp_attn = lrp(model, img, pred_class, args.output_dir)
            lrp_iou = get_iou(lrp_attn.unsqueeze(0), gt_mask.unsqueeze(0))
            lrp_ious.append(lrp_iou)
            save_attn_map(
                lrp_attn.unsqueeze(0),
                os.path.join(
                    os.path.join(args.output_dir, "lrp_attn"),
                    os.path.basename(img_path),
                ),
            )
            x += 1
            if x % 10 == 0:
                print("")
                print("acc:", correct / sample_count)
                print("lrp_attn", sum(lrp_ious) / len(lrp_ious))
    print("")
    print("lrp_attn", sum(lrp_ious) / len(lrp_ious))


import os


def main(args):
    model_paths = [file for file in os.listdir(args.model_dir) if "epoch_" in file]
    # assert len(model_paths) == 1
    model_paths.sort()
    model_path = os.path.join(args.model_dir, model_paths[-1])
    model = my_load_model(config_file, model_path)
    pred_ious = []
    grad_cam_ious = []
    score_cam_ious = []
    ig_ious = []
    gradient_backprop_ious = []
    correct = 0
    sample_count = 0
    x = 0
    with open(label_file) as (f):
        lines = f.readlines()
        for line in tqdm(lines):
            img_path, label, gt_path, output_file = get_image_data_details(line, args)
            gt_mask = Image.open(gt_path)
            gt_mask = _preprocess(gt_mask, color="L")
            img = Image.open(img_path)
            img = _preprocess(img)

            img = img.unsqueeze(0)
            with torch.no_grad():
                pred_class, pred_attn = predict_one_shot(
                    model, img_path, args.output_dir
                )
            if int(label) - 1 == int(pred_class[0]):
                correct += 1
            sample_count += 1

            model = model.eval()
            model = set_gradient_true(model)

            grad_cam_attn = grad_cam(model, img, pred_class, args.output_dir)
            score_cam_attn = score_cam(model, img, pred_class, args.output_dir)
            ig_attn = ig(model, img, pred_class, args.output_dir)
            gradient_backprop_attn = gradient_backprop(
                model, img, pred_class, args.output_dir
            )
            # lrp_attn = lrp(model, img, pred_class, args.output_dir)

            pred_iou = get_iou(pred_attn, gt_mask.unsqueeze(0))
            grad_cam_iou = get_iou(grad_cam_attn.unsqueeze(0), gt_mask.unsqueeze(0))
            score_cam_iou = get_iou(score_cam_attn.unsqueeze(0), gt_mask.unsqueeze(0))
            ig_iou = get_iou(ig_attn.unsqueeze(0), gt_mask.unsqueeze(0))
            gradient_backprop_iou = get_iou(
                gradient_backprop_attn.unsqueeze(0), gt_mask.unsqueeze(0)
            )
            # lrp_iou = get_iou(lrp_attn.unsqueeze(0), gt_mask.unsqueeze(0))

            pred_ious.append(pred_iou)
            grad_cam_ious.append(grad_cam_iou)
            score_cam_ious.append(score_cam_iou)
            ig_ious.append(ig_iou)
            gradient_backprop_ious.append(gradient_backprop_iou)
            # lrp_ious.append(lrp_iou)

            gt_mask_dir = os.path.join(args.output_dir, "gt_mask")
            test_attn_dir = os.path.join(args.output_dir, "test_attn")
            grad_cam_attn_dir = os.path.join(args.output_dir, "grad_cam_attn")
            score_cam_attn_dir = os.path.join(args.output_dir, "score_cam_attn")
            ig_attn_dir = os.path.join(args.output_dir, "ig_attn")
            gradient_backprop_attn_dir = os.path.join(
                args.output_dir, "gradient_backprop_attn"
            )
            # lrp_attn_dir = os.path.join(args.output_dir, "lrp_attn")
            os.makedirs(gt_mask_dir, exist_ok=True)
            os.makedirs(test_attn_dir, exist_ok=True)
            os.makedirs(grad_cam_attn_dir, exist_ok=True)
            os.makedirs(score_cam_attn_dir, exist_ok=True)
            os.makedirs(ig_attn_dir, exist_ok=True)
            os.makedirs(gradient_backprop_attn_dir, exist_ok=True)
            # os.makedirs(lrp_attn_dir, exist_ok=True)

            save_attn_map(
                gt_mask.unsqueeze(0),
                os.path.join(gt_mask_dir, os.path.basename(img_path)),
            )
            save_attn_map(
                pred_attn,
                os.path.join(
                    test_attn_dir,
                    os.path.basename(img_path),
                ),
            )
            save_attn_map(
                grad_cam_attn.unsqueeze(0),
                os.path.join(
                    grad_cam_attn_dir,
                    os.path.basename(img_path),
                ),
            )
            save_attn_map(
                score_cam_attn.unsqueeze(0),
                os.path.join(
                    score_cam_attn_dir,
                    os.path.basename(img_path),
                ),
            )
            save_attn_map(
                ig_attn.unsqueeze(0),
                os.path.join(ig_attn_dir, os.path.basename(img_path)),
            )
            save_attn_map(
                gradient_backprop_attn.unsqueeze(0),
                os.path.join(
                    gradient_backprop_attn_dir,
                    os.path.basename(img_path),
                ),
            )
            # save_attn_map(lrp_attn.unsqueeze(0), os.path.join(os.path.join(args.output_dir, "lrp_attn"),os.path.basename(img_path)))
            x += 1
            if x % 10 == 0:
                for i, sample in enumerate(pred_ious):
                    if type(sample) != np.float64:
                        pred_ious[i] = 0
                for i, sample in enumerate(grad_cam_ious):
                    if type(sample) != np.float64:
                        grad_cam_ious[i] = 0
                print("")
                print("acc:", correct / sample_count)
                print("pred_iou:", sum(pred_ious) / len(pred_ious))
                print("grad_cam_iou:", sum(grad_cam_ious) / len(grad_cam_ious))
                print("score_cam_iou:", sum(score_cam_ious) / len(score_cam_ious))
                print("ig_iou:", sum(ig_ious) / len(ig_ious))
                print(
                    "gradient_backprop_iou",
                    sum(gradient_backprop_ious) / len(gradient_backprop_ious),
                )
                # print("lrp_attn", sum(lrp_ious) / len(lrp_ious))
    print("")
    print("acc:", correct / sample_count)
    print("pred_iou:", sum(pred_ious) / len(pred_ious))
    print("grad_cam_iou", sum(grad_cam_ious) / len(grad_cam_ious))
    print("score_cam_iou", sum(score_cam_ious) / len(score_cam_ious))
    print("ig_iou:", sum(ig_ious) / len(ig_ious))
    print(
        "gradient_backprop_iou",
        sum(gradient_backprop_ious) / len(gradient_backprop_ious),
    )


def _preprocess(image: Image.Image, color="RGB") -> torch.Tensor:
    """
    Preprocess the input image.
    Args:
        image (Image.Image): the input image
    Returns:
        torch.Tensor: the preprocessed image
    """
    if color == "L":
        array = np.asarray(image).copy()
        # 0以外の要素を255にする
        array[array != 0] = 255
        # PIL画像に戻す
        image = Image.fromarray(array)
        # グレースケールに変換
        image = image.convert("L")
        image = image.resize((384, 384))
        image = torch.from_numpy(np.asarray(image).copy()).float()
        return image

    else:
        image = image.convert("RGB")
        image = image.resize((384, 384))
        image = torch.from_numpy(np.asarray(image).copy()).float()
        image = image.permute(2, 0, 1)
        crop_size = 384
        x = int((image.shape[1] - crop_size) / 2)
        y = int((image.shape[0] - crop_size) / 2)
        image = image[y : y + crop_size, x : x + crop_size]
        return image


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--method_name",
        type=str,
        required=True,
        help="select from ['Grad-CAM', 'LRP', 'IG', 'GBP', 'score-CAM', 'all']",
    )

    parser.add_argument(
        "--model_dir", type=str, required=True, help="path to model file"
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    main(args)

# for test
# python grad_cam_imagenet.py --method_name LRP --model_dir output/test --output_dir output/test2
