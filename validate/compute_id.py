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
import json

import huggingface_hub
import torch
import numpy as np
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, random_color
from huggingface_hub import hf_hub_download
from PIL import Image
from torchsummary import summary
from torchvision.transforms import functional as TF

from san import add_san_config
from san.data.datasets.register_cub import CLASS_NAMES
from san.model.visualize import attn2binary_mask, save_img_data, save_attn_map

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

label_file = "datasets/CUB/id_score_sample.jsonl"
config_file = "configs/san_clip_vit_res4_coco.yaml"


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

def load_model(model_path: str):
    model = torch.load(model_path)
    print('Loading model from: ', model_path)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)

    print('Loaded model on ', device)
    return model

def get_attn_dir(args):
    attn_dir = args.output_dir.replace("id", "")
    return attn_dir

def get_image_data_details(line, args):
    line = json.loads(line)
    img_path = line["image_path"]
    label = line["label"]
    caption = line["caption"]
    output_file = os.path.join(args.output_dir, img_path.replace("test/","",).replace("/","_").replace(" ",""))
    attn_path = os.path.join(get_attn_dir(args), line["attn_map"])
    img_path = os.path.join('datasets/CUB/', img_path.replace(' ', ''))

    return (img_path, caption, label, attn_path, output_file)


def main(args):
    model = load_model(config_file, args.model_path)
    metrics = PatchInsertionDeletion(
        model=model,
        batch_size=1,
        patch_size=1,
        step=4096,
        dataset="str",
        device="cuda",)

    x = 0
    with open(label_file) as (f):
        lines = f.readlines()
        for line in tqdm(lines):
            img_path, caption, label, attn_path, output_file = get_image_data_details(line, args)
            single_image = Image.open(img_path)
            single_image = single_image.resize((640, 640), Image.ANTIALIAS)
            single_image = np.array(single_image).transpose(2, 0, 1)
            single_target = torch.tensor(label)
            single_attn = Image.open(attn_path)
            single_attn = single_attn.resize((640, 640), Image.ANTIALIAS)
            single_attn = np.array(single_attn)

            metrics.evaluate(single_image, single_attn, caption, single_target)
            metrics.save_roc_curve(args.output_dir)
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
    parser.add_argument(
        "--predict_mode", type=str, required=True, help="select from ['bird', 'name', 'class', 'none']"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="path to model file"
    )
    # parser.add_argument(
    #     '--img_dir', type=str, required=True, help='path to image dir.'
    # )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    main(args)
