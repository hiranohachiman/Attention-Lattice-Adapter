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
from torchsummary import summary
from torchvision.transforms import functional as TF
from san.data.dataloader import train_dataset, valid_dataset, test_dataset, _preprocess

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

label_file = "datasets/CUB/test_label.txt"
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

def my_load_model(config_file: str, model_path: str, lora_path: str=None):
    cfg = setup(config_file)
    model = SAN(**SAN.from_config(cfg))
    model.load_state_dict(torch.load(model_path), strict=False)
    if lora_path is not None:
        model.load_state_dict(torch.load(lora_path), strict=False)

    print('Loading model from: ', model_path)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(model_path)
    print('Loaded model from: ', model_path)

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
    attn_map = F.interpolate(attn_map, size=(640, 640), mode='bilinear', align_corners=False)
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
    image = Image.open(image_path)
    image = _preprocess(image)
    image = image.unsqueeze(0).to(device)
    logits, _, attn_map = model(image)
    # save attn_map
    save_attn_map(attn_map, os.path.join(output_path, f"{os.path.basename(image_path)}_attn_map.png"))
    _, predicted = torch.max(logits.data, 1)
    return predicted

def main(args):
    model_paths = [file for file in os.listdir(args.model_dir) if 'epoch_' in file]
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
        step=8192,
        dataset="str",
        device="cuda",)

    x = 0
    with open(label_file) as (f):
        lines = f.readlines()
        for line in tqdm(lines):
            img_path, _, _, _ = get_image_data_details(line, args)
            label = predict_one_shot(model, img_path, args.output_dir)
            single_image = Image.open(img_path)
            single_image = _preprocess(single_image)
            single_image = single_image.cpu().detach().numpy()
            single_target = label
            single_attn = Image.open(os.path.join(args.output_dir, f"{os.path.basename(img_path)}_attn_map.png"))
            single_attn = _preprocess(single_attn, color="L")
            single_attn = single_attn.cpu().detach().numpy()
            metrics.evaluate(single_image, single_attn, single_target)
            metrics.save_roc_curve(args.output_dir)
            metrics.log()
            x += 1
            if x % 50 == 0:
                print("total_insertion:", metrics.total_insertion)
                print("total_deletion:", metrics.total_deletion)
                print("average", (metrics.total_insertion - metrics.total_deletion) / x)
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print("total_insertion:", metrics.total_insertion)
    print("total_deletion:", metrics.total_deletion)
    print("ins-del score:",metrics.total_insertion - metrics.total_deletion)
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
