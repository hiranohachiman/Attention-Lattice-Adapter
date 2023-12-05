from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import json
from PIL import Image
from torchvision import transforms

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
        label = torch.tensor(int(self.data[idx]['label']))
        caption = self.data[idx]['caption']
        if self.transform:
            image = self.transform(image)

        return image, caption , label

transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 必要に応じてリサイズ
    transforms.ToTensor()
])

train_dataset = CUBDataset(json_file='datasets/CUB/train_label.jsonl', root_dir='datasets/CUB', transform=transform)
valid_dataset = CUBDataset(json_file='datasets/CUB/valid_label.jsonl', root_dir='datasets/CUB', transform=transform)
test_dataset = CUBDataset(json_file='datasets/CUB/test_label.jsonl', root_dir='datasets/CUB', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
