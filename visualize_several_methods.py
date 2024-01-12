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

from captum.attr import LRP, IntegratedGradients, GuidedBackprop
from pytorch_grad_cam import GradCAM, ScoreCAM

import huggingface_hub
import torch
import numpy as np
from torchinfo import summary

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
# model_path = "output/2023-12-13-23:34:53/epoch_48.pth"
# lora_path = "output/2023-12-13-23:34:53/lora_epoch_48.pth"

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
    # model = SAN(**SAN.from_config(cfg))
    print('Loading model from: ', model_path)
    model = torch.load(model_path)
    print('Loaded model from: ', model_path)
    model.eval()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.cuda()
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


def predict_one_shot(model, image_path, output_path, device="cuda"):
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


def lrp(model, image, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    lrp = LRP(model)
    attribution = lrp.attribute(image, target=3)
    print("attribution.shape", attribution.shape)


def ig(model, image, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    ig = IntegratedGradients(model.forward)
    attribution = ig.attribute(image, target=3, n_steps=5)
    print("attribution.shape", attribution.shape)


def gradient_backprop(model, image, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    guided_backprop = GuidedBackprop(model)
    attribution = guided_backprop.attribute(image, target=3)
    print("attribution.shape", attribution.shape)


def score_cam(model, image, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    score_cam = ScoreCAM(model, target_layers=[model.clipfeatureclassifier.conv2])
    attribution = score_cam(image, targets=None)
    print("attribution.shape", attribution.shape)

def grad_cam(model, image, output_path, device="cuda"):
    model = model.to(device)
    image = image.to(device)
    grad_cam = GradCAM(model, target_layers=[model.clipfeatureclassifier.conv2])
    attribution = grad_cam(image, targets=None)
    print("attribution.shape", attribution.shape)

def set_gradient_true(model):
    for param in model.parameters():
        param.requires_grad = True
    return model



def main(args):
    model_paths = [file for file in os.listdir(args.model_dir) if 'epoch_' in file]
    assert len(model_paths) == 1
    model_path = os.path.join(args.model_dir, model_paths[0])
    model = my_load_model(config_file, model_path)

    image_path = "datasets/CUB/extracted_test/001.Black_footed_Albatross_Black_Footed_Albatross_0001_796111.jpg"
    image_path2 = "datasets/CUB/extracted_test/001.Black_footed_Albatross_Black_Footed_Albatross_0001_796111.jpg"
    img = cv2.imread(image_path)
    img2 = cv2.imread(image_path2)
    img = img[:, :, ::-1] # BGR to RGB.
    img2 = img2[:, :, ::-1] # BGR to RGB.
    # to PIL.Image
    img = Image.fromarray(img)
    img2 = Image.fromarray(img2)
    transform = _make_transform(istrain=False)
    img = transform(img)
    img2 = transform(img2)
    img = img.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    # img = torch.cat([img, img2], dim=0)
    model = model.eval()
    model = set_gradient_true(model)
    summary(model, input_size=(2,3,384,384))
    # model = lrp(model, img, args.output_dir)
    # lrp(model, img2, args.output_dir)
    for i in range(3):
        ig(model, img, args.output_dir)
        gradient_backprop(model, img, args.output_dir)
        score_cam(model, img, args.output_dir)
        grad_cam(model, img, args.output_dir)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--method_name", type=str, required=True, help="select from ['Grad-CAM', 'LRP', 'IG', 'GBP', 'score-CAM', 'all']"
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
# python visualize_several_methods.py --method_name LRP --model_dir output/test --output_dir output/test/test
