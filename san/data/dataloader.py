from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import json
from PIL import Image
from torchvision import transforms
import numpy as np

class CUBDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
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

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['image_path'])
        image = Image.open(img_name)
        image = _preprocess(image)
        label = torch.tensor(int(self.data[idx]['label']))
        caption = self.data[idx]['caption']
        if self.transform:
            image = self.transform(image)

        return image, caption , label

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

train_dataset = CUBDataset(json_file='datasets/CUB/train_label.jsonl', root_dir='datasets/CUB')
valid_dataset = CUBDataset(json_file='datasets/CUB/valid_label.jsonl', root_dir='datasets/CUB')
test_dataset = CUBDataset(json_file='datasets/CUB/test_label.jsonl', root_dir='datasets/CUB')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
