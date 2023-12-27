from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import json
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

class CUBDataset(Dataset):
    def __init__(self, json_file, root_dir, split="train"):
        """
        Args:
            json_file (string): JSONファイルへのパス。
            root_dir (string): 画像が格納されているディレクトリへのパス。
            transform (callable, optional): 画像に適用する変換。
        """
        self.split = split
        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.root_dir = root_dir
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
        if split == "train":
            self.transforms = self._make_transform(istrain=True)
            self.transforms_for_mask = self._make_transform_for_mask(istrain=True)
        else:
            self.transforms = self._make_transform(istrain=False)
            self.transforms_for_mask = self._make_transform_for_mask(istrain=False)

    def __len__(self):
        return len(self.data)
        # return len(self.data) // 60

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx]['image_path'])
        mask_path = os.path.join(self.root_dir, self.data[idx]['mask_path'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = Image.fromarray(mask)
        mask = self.transforms_for_mask(mask)
        label = torch.tensor(int(self.data[idx]['label']))
        caption = self.data[idx]['caption']
        img = cv2.imread(img_path)
        img = img[:, :, ::-1] # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        if self.split == "train":
            return img, mask, caption, label
        else:
            return img, mask, caption, label, img_path

    def _make_transform(self, istrain: bool):
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        if istrain:
            transform = transforms.Compose([
                                transforms.Resize((510, 510), Image.BILINEAR),
                                transforms.RandomCrop((384, 384)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                                transforms.ToTensor(),
                                normalize
                        ])
        else:
            transform = transforms.Compose([
                                transforms.Resize((384, 384), Image.BILINEAR),
                                # transforms.CenterCrop((384, 384)),
                                transforms.ToTensor(),
                                normalize
                        ])
        return transform

    def _make_transform_for_mask(self, istrain: bool):
        normalize = transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        if istrain:
            transform_for_mask = transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),  # カラー画像をグレースケールに変換
                            transforms.Resize((510, 510), Image.BILINEAR),
                            transforms.RandomCrop((384, 384)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                            transforms.ToTensor()
                        ])
        else:
            transform_for_mask = transforms.Compose([
                                    transforms.Grayscale(num_output_channels=1),  # カラー画像をグレースケールに変換
                                    transforms.Resize((384, 384), Image.BILINEAR),
                                    # transforms.CenterCrop((384, 384)),
                                    transforms.ToTensor()
                                ])
        return transform_for_mask


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
        image = image.resize((384, 384))
        image = torch.from_numpy(np.asarray(image).copy()).float()
        crop_size = 384
        x = int((image.shape[1] - crop_size) / 2)
        y = int((image.shape[0] - crop_size) / 2)
        image = image[y:y+crop_size, x:x+crop_size]

    else:
        image = image.convert('RGB')
        image = image.resize((384, 384))
        image = torch.from_numpy(np.asarray(image).copy()).float()
        image = image.permute(2, 0, 1)
        crop_size = 384
        x = int((image.shape[1] - crop_size) / 2)
        y = int((image.shape[0] - crop_size) / 2)
        image = image[y:y+crop_size, x:x+crop_size]

    # if w < h:
    #     image = image.resize((640, int(h * 640 / w)))
    # else:
    #     image = image.resize((int(w * 640 / h), 640))

    return image

class ImageNetSDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        if split == "train":
            self.split = "train"
            label_file = os.path.join(root_dir, "labels.txt")
            with open(label_file, 'r') as f:
                label_names = f.readlines()
                label_names = [label_name.strip() for label_name in label_names]
            # print(label_names)
            root_dir = os.path.join(root_dir, "train")
            img_dirs = os.listdir(root_dir)
            # print(img_dirs)
            self.img_paths = []
            count = 0
            for img_dir in img_dirs:
                if "tar" in img_dir:
                    continue
                if img_dir.split("/")[-1] not in label_names:
                    continue

                label = label_names.index(img_dir.split("/")[-1])
                img_paths = os.listdir(os.path.join(root_dir, img_dir))
                for img_path in img_paths:
                    self.img_paths.append([os.path.join(root_dir, img_dir, img_path), label])
            # print(len(self.img_paths))
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx][0]
        label = self.img_paths[idx][1]
        img = Image.open(img_path)
        img = _preprocess(img)
        return img, 0, 0, label


image_net_s_dataset = ImageNetSDataset(root_dir="datasets/ImageNetS919", split="train")


train_dataset = CUBDataset(json_file='datasets/CUB/new_train_label.jsonl', root_dir='datasets/CUB', split="train")
valid_dataset = CUBDataset(json_file='datasets/CUB/new_valid_label.jsonl', root_dir='datasets/CUB', split="val")
test_dataset = CUBDataset(json_file='datasets/CUB/new_test_label.jsonl', root_dir='datasets/CUB', split="test")
