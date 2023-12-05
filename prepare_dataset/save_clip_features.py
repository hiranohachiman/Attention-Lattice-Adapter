import json
import os
import torch
import open_clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from ..san.model.clip_utils import FeatureExtractor

input_dir = "../datasets/CUB/train_label.jsonl"

def get_clip_features(image: torch.tensor):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B/16",
                                                                         pretrained="openai")
    clip_visual_extractor = FeatureExtractor(model.visual,
                                                     last_layer_idx=9,
                                                     frozen_exclude=["positional_embedding"])
    clip_image_features = clip_visual_extractor(image)
    return clip_image_features

def get_image_data_details(line):
    line = json.loads(line)
    img_path = line["image_path"]
    label = line["label"]
    caption = line["caption"]
    img_path = os.path.join('datasets/CUB/', img_path.replace(' ', ''))
    return img_path, caption, label


def main():
    transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 必要に応じてリサイズ
    transforms.ToTensor()
    ])
    with open(input_dir) as (f):
        lines = f.readlines()
        for line in lines:
            img_path, caption, label = get_image_data_details(line)
            image = Image.open(img_path)
            image = transform(image)
            image = F.interpolate(image, scale_factor=0.5, mode='bilinear')
            clip_image_features = get_clip_features(img_path)
            print(clip_image_features.shape)

main()
