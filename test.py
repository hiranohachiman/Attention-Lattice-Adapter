import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import sys
import open_clip
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


class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as h5f:
            self.length = h5f['cls_token_0'].shape[0]  # サンプル数を取得

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        features = []
        with h5py.File(self.file_path, 'r') as h5f:
            for i in range(10):  # 各特徴量を読み出す
                dset_name = f'cls_token_{i}'
                feature = h5f[dset_name][idx, ...]
                features.append(torch.from_numpy(feature))
        return features

# 使用例
dataset = HDF5Dataset('datasets/CUB/valid_cls_tokens.h5')
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# データローダーからデータを取得する例

input_dir = "datasets/CUB/valid_label.jsonl"

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


# def save_features_to_hdf5(clip_image_features, file_path, img_index):
#     with h5py.File(file_path, 'a') as h5f:
#         for i in range(10):
#             feature = clip_image_features[i]
#             # データセットの形状は [画像数, 10, 768, 20, 20]
#             dset_name = f'features_{i}'
#             if dset_name not in h5f:
#                 # 初めての画像の場合、データセットを作成
#                 dset = h5f.create_dataset(dset_name, (1, 768, 20, 20), maxshape=(None, 768, 20, 20), chunks=True, compression="gzip")
#                 dset[0, ...] = feature.detach().numpy()
#             else:
#                 # 既存のデータセットに追加
#                 dset = h5f[dset_name]
#                 dset.resize((dset.shape[0] + 1, 768, 20, 20))
#                 dset[img_index, ...] = feature.detach().numpy()

def get_features_from_index(index):
    with open(input_dir) as f:
        lines = f.readlines()
        img_path, caption, label = get_image_data_details(lines[index])
        image = Image.open(img_path)
        image = _preprocess(image)
        image = image.unsqueeze(0)
        image = F.interpolate(image, scale_factor=0.5, mode='bilinear')
        clip_image_features = get_clip_features(image)
        return clip_image_features


for index, batch_features in enumerate(tqdm(data_loader)):
    raw_features = get_features_from_index(index)
    for i in range(10):
        assert torch.equal(raw_features[f"{i}_cls_token"], batch_features[i]), "Tensors are not equal"
    # batch_features は、各サンプルに対する10個の特徴量のリスト
