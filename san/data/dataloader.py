from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import json
from PIL import Image
from torchvision import transforms
import numpy as np

class CUBDataset(Dataset):
    def __init__(self, json_file, root_dir, istrain=False, transform=None):
        """
        Args:
            json_file (string): JSONファイルへのパス。
            root_dir (string): 画像が格納されているディレクトリへのパス。
            transform (callable, optional): 画像に適用する変換。
        """
        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
        # return len(self.data) // 60

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx]['image_path'])
        mask_path = os.path.join(self.root_dir, self.data[idx]['mask_path'])
        image = Image.open(img_path)
        image = _preprocess(image)
        mask = Image.open(mask_path)
        mask = _preprocess(mask, color="L")
        label = torch.tensor(int(self.data[idx]['label']))
        caption = self.data[idx]['caption']
        if self.transform:
            image = self.transform(image)

        return image, mask, caption, label



def _preprocess(image: Image.Image, color="RGB") -> torch.Tensor:
    """
    Preprocess the input image.
    Args:
        image (Image.Image): the input image
    Returns:
        torch.Tensor: the preprocessed image
    """
    if color == "L":
        image = image.convert('L')
        image = image.resize((640, 640))
        image = torch.from_numpy(np.asarray(image).copy()).float()

    else:
        image = image.convert('RGB')
        image = image.resize((640, 640))
        image = torch.from_numpy(np.asarray(image).copy()).float()
        image = image.permute(2, 0, 1)


    # if w < h:
    #     image = image.resize((640, int(h * 640 / w)))
    # else:
    #     image = image.resize((int(w * 640 / h), 640))

    return image

train_dataset = CUBDataset(json_file='datasets/CUB/new_train_label.jsonl', root_dir='datasets/CUB', istrain=True)
valid_dataset = CUBDataset(json_file='datasets/CUB/new_valid_label.jsonl', root_dir='datasets/CUB', istrain=False)
test_dataset = CUBDataset(json_file='datasets/CUB/new_test_label.jsonl', root_dir='datasets/CUB', istrain=False)
