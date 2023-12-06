import json
import os
import torch
import open_clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/home/initial/workspace/smilab23/graduation_research/SAN')
from san.model.clip_utils import FeatureExtractor
import h5py
from tqdm import tqdm

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
    img_path = os.path.join('../datasets/CUB/', img_path.replace(' ', ''))
    return img_path, caption, label

def _preprocess(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the input image.
    Args:
        image (Image.Image): the input image
    Returns:
        torch.Tensor: the preprocessed image
    """
    image = image.convert('RGB')
    w, h = image.size

    # if w < h:
    #     image = image.resize((640, int(h * 640 / w)))
    # else:
    #     image = image.resize((int(w * 640 / h), 640))

    image = image.resize((640, 640))
    image = torch.from_numpy(np.asarray(image)).float()
    image = image.permute(2, 0, 1)
    return image


def save_features_to_hdf5(clip_image_features, file_path, img_index):
    with h5py.File(file_path, 'a') as h5f:
        for i in range(10):
            feature = clip_image_features[i]
            # データセットの形状は [画像数, 10, 768, 20, 20]
            dset_name = f'features_{i}'
            if dset_name not in h5f:
                # 初めての画像の場合、データセットを作成
                dset = h5f.create_dataset(dset_name, (1, 768, 20, 20), maxshape=(None, 768, 20, 20), chunks=True, compression="gzip")
                dset[0, ...] = feature.detach().numpy()
            else:
                # 既存のデータセットに追加
                dset = h5f[dset_name]
                dset.resize((dset.shape[0] + 1, 768, 20, 20))
                dset[img_index, ...] = feature.detach().numpy()

def main():
    with open(input_dir) as f:
        lines = f.readlines()
        for img_index, line in enumerate(tqdm(lines)):
            img_path, caption, label = get_image_data_details(line)
            image = Image.open(img_path)
            image = _preprocess(image)
            image = image.unsqueeze(0)
            image = F.interpolate(image, scale_factor=0.5, mode='bilinear')
            clip_image_features = get_clip_features(image)

            # HDF5ファイルに保存
            save_features_to_hdf5(clip_image_features, '../datasets/CUB/train_features.h5', img_index)


if __name__ == "__main__":
    main()
