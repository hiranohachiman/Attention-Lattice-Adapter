# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Mar  8 2023, 16:27:05) 
# [GCC 9.4.0]
# Embedded file name: /home/initial/workspace/smilab23/graduation_research/SAN/san/model/layers.py
# Compiled at: 2023-10-26 09:42:22
# Size of source mod 2**32: 6685 bytes
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import CNNBlockBase, Conv2d
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    __doc__ = '\n    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and\n    variance normalization over the channel dimension for inputs that have shape\n    (batch_size, channels, height, width).\n    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950\n    '

    def __init__(self, normalized_shape, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape)).to('cuda:0')
        self.bias = nn.Parameter(torch.zeros(normalized_shape)).to('cuda:0')
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvReducer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvReducer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)


def patch_based_importance_avg(importance_map, patch_size=2):
    B, C, H, W = importance_map.shape
    new_H, new_W = H // patch_size, W // patch_size
    patches = importance_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    avg_patches = patches.mean(dim=4).mean(dim=4)
    return avg_patches.reshape(B, C, new_H, new_W)


class MLP(nn.Module):
    __doc__ = 'Very simple multi-layer perceptron (also called FFN)'

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList((affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])))

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        else:
            return x


class AddFusion(CNNBlockBase):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1)
        self.input_proj = nn.Sequential(LayerNorm(in_channels), Conv2d(in_channels,
          out_channels,
          kernel_size=1))
        weight_init.c2_xavier_fill(self.input_proj[-1])

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        y = F.interpolate((self.input_proj(y.contiguous())),
          size=spatial_shape,
          mode='bilinear',
          align_corners=False).permute(0, 2, 3, 1).reshape(x.shape)
        x = x + y
        return x


class MultipleFusion(CNNBlockBase):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1)
        self.input_proj = nn.Sequential(LayerNorm(in_channels), Conv2d(in_channels,
          out_channels,
          kernel_size=1))
        weight_init.c2_xavier_fill(self.input_proj[-1])

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        y = F.interpolate((self.input_proj(y.contiguous())),
          size=spatial_shape,
          mode='bilinear',
          align_corners=False).permute(0, 2, 3, 1).reshape(x.shape)
        x = x * y
        return x


def build_fusion_layer(fusion_type: str, in_channels: int, out_channels: int):
    if fusion_type == 'add':
        return AddFusion(in_channels, out_channels)
    if fusion_type == 'multiple':
        return MultipleFusion(in_channels, out_channels)
    raise ValueError('Unknown fusion type: {}'.format(fusion_type))


class ClassifierHead(nn.Module):

    def __init__(self, num_input_channels, num_classes):
        super(ClassifierHead, self).__init__()
        self.conv = nn.Conv2d(num_input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ClassificationCNN(nn.Module):

    def __init__(self, num_classes=200):
        super(ClassificationCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(-1, 1600)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FlattenLayer(nn.Module):

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class ModifiedModel(nn.Module):

    def __init__(self):
        super(ModifiedModel, self).__init__()
        self.flatten = FlattenLayer()
        self.fc = nn.Linear(6144, 1600)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x.view(8, 200)