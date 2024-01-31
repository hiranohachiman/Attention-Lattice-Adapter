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
from san.data.dataloader import (
    _preprocess,
    # _make_transform,
    # _make_transform_for_mask,
)

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

label_file = "datasets/ImageNetS919/test_small.txt"
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


def my_load_model(config_file: str, model_path: str, lora_path: str = None):
    cfg = setup(config_file)
    model = SAN(**SAN.from_config(cfg))
    # model.load_state_dict(torch.load(model_path), strict=False)
    # if lora_path is not None:
    #     model.load_state_dict(torch.load(lora_path), strict=False)
    print(model_path)
    model = torch.load(model_path)
    print("Loading model from: ", model_path)
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(model_path)
    print("Loaded model from: ", model_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
    return model


def apply_heat_quantization(attention, q_level: int = 2):
    max_ = attention.max()
    min_ = attention.min()

    # quantization
    bin = np.linspace(min_, max_, q_level)
    # apply quantization
    for i in range(q_level - 1):
        attention[(attention >= bin[i]) & (attention < bin[i + 1])] = bin[i]

    return attention


def predict_one_shot(model, image_path, output_path, device="cuda"):
    model = model.to(device)
    model.eval()

    # to PIL.Image
    img = Image.open(image_path)
    img = _preprocess(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    logits, _, attn_map = model(img)
    attn_map = attn_map.squeeze(0)
    # save attn_map
    save_attn_map(
        attn_map,
        384,
        384,
        os.path.join(output_path, f"{os.path.basename(image_path)}"),
    )
    _, predicted = torch.max(logits.data, 1)
    return predicted


def get_image_data_details(line, args):
    img_path = line.split(",")[0]
    label = int(line.split(",")[1]) - 1
    return img_path, label


def main(args):
    model_paths = [file for file in os.listdir(args.model_dir) if "epoch_" in file]
    model_path = None
    lora_model_path = None
    for file_name in model_paths:
        if file_name.startswith("epoch_") and not file_name.startswith("lora_epoch_"):
            model_path = os.path.join(args.model_dir, file_name)
        elif file_name.startswith("lora_epoch_"):
            lora_model_path = os.path.join(args.model_dir, file_name)
    model = my_load_model(config_file, model_path, lora_model_path)
    metrics = PatchInsertionDeletion(
        model=model,
        batch_size=8,
        patch_size=1,
        step=3072,
        dataset="str",
        device="cuda",
    )

    x = 0
    with open(label_file) as (f):
        lines = f.readlines()
        for line in tqdm(lines):
            img_path, gt = get_image_data_details(line, args)
            print(img_path)
            with torch.no_grad():
                label = predict_one_shot(model, img_path, args.output_dir)
            single_image = Image.open(img_path)
            single_image = _preprocess(single_image)
            single_target = torch.tensor(gt)
            single_attn_path = os.path.join(args.attn_dir, os.path.basename(img_path))
            single_attn = Image.open(single_attn_path)
            single_attn = _preprocess(single_attn, color="L")
            # single_attn = apply_heat_quantization(single_attn)
            single_image = single_image.cpu().numpy()
            single_attn = single_attn.cpu().numpy()
            metrics.evaluate(single_image, single_attn, single_target)
            metrics.save_roc_curve(args.output_dir)

            x += 1
            if x % 50 == 0:
                print("total_insertion:", metrics.total_insertion)
                print("total_deletion:", metrics.total_deletion)
                print("average", (metrics.total_insertion - metrics.total_deletion) / x)
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print("total_insertion:", metrics.total_insertion)
    print("total_deletion:", metrics.total_deletion)
    print("ins-del score:", metrics.total_insertion - metrics.total_deletion)
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

    parser.add_argument(
        "--attn_dir", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    main(args)
