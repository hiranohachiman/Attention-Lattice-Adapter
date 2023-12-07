from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import json
from PIL import Image
from torchvision import transforms
import numpy as np
import h5py

class CUBDataset(Dataset):
    def __init__(self, json_file, clip_features_file, cls_token_file, root_dir):
        """
        Args:
            json_file (string): JSONファイルへのパス。
            root_dir (string): 画像が格納されているディレクトリへのパス。
            transform (callable, optional): 画像に適用する変換。
        """
        self.data = []
        self.root_dir = root_dir
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.clip_features_file = os.path.join(root_dir, clip_features_file)
        print(cls_token_file)
        self.cls_token_file = os.path.join(root_dir, cls_token_file)

    def __len__(self):
        return len(self.data)
        # return len(self.data) // 60

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['image_path'])
        image = Image.open(img_name)
        image = _preprocess(image)
        label = torch.tensor(int(self.data[idx]['label']))
        caption = self.data[idx]['caption']
        features = []
        cls_tokens = []
        with h5py.File(self.clip_features_file, "r") as features_f:
            for i in range(10):
                dset_name = f"features_{i}"
                feature = features_f[dset_name][idx, ...]
                features.append(torch.from_numpy(feature))
        with h5py.File(self.cls_token_file, "r") as cls_tokens_f:
            for i in range(10):
                dset_name = f"cls_token_{i}"
                cls_token = cls_tokens_f[dset_name][idx, ...]
                cls_tokens.append(torch.from_numpy(cls_token))

        return image, features, cls_tokens, caption , label

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
    image = torch.from_numpy(np.asarray(image).copy()).float()
    image = image.permute(2, 0, 1)
    return image

train_dataset = CUBDataset(json_file='datasets/CUB/train_label.jsonl', clip_features_file="train_features.h5", cls_token_file="train_cls_tokens.h5", root_dir='datasets/CUB')
valid_dataset = CUBDataset(json_file='datasets/CUB/valid_label.jsonl', clip_features_file="valid_features.h5", cls_token_file="valid_cls_tokens.h5", root_dir='datasets/CUB')
test_dataset = CUBDataset(json_file='datasets/CUB/test_label.jsonl', clip_features_file="test_features.h5", cls_token_file="test_cls_tokens.h5", root_dir='datasets/CUB')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
