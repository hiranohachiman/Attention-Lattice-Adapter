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

def zero_below_average(tensor):
    # テンソルの平均値を計算
    mean_val = tensor.mean()
    # 平均値以下の値を0に置換
    tensor[tensor <= mean_val] = 0
    return tensor

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

def normalize_per_batch(tensor):
    # 正規化されたテンソルを保存するための新しいテンソルを作成
    normalized_tensor = torch.zeros_like(tensor)
    epsilon = 10e-8
    # バッチの次元（0次元目）でループ
    for i in range(tensor.size(0)):
        batch_min = tensor[i].min()
        batch_max = tensor[i].max()
        normalized_tensor[i] = (tensor[i] - batch_min) / (batch_max - batch_min + epsilon)

    return normalized_tensor

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(512, 200)

    def forward(self, x):
        x = self.fc(x)  # [8, 100, 200]
        x = x.mean(dim=1, keepdim=True)  # [8, 1, 200]
        return x

class ModifiedModel(nn.Module):

    def __init__(self):
        super(ModifiedModel, self).__init__()
        self.flatten = FlattenLayer()
        self.fc = nn.Linear(6144, 1600)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x.view(8, 200)



class LinearLayer(nn.Module):
    # def __init__(self, in_dim, out_dim):
    #     super(LinearLayer, self).__init__()
    #     self.linear1 = nn.Linear(in_dim, 256)
    #     self.linear2 = nn.Linear(256, 1024)
    #     self.linear3 = nn.Linear(1024, 4096)
    #     self.linear4 = nn.Linear(4096, 512)
    #     self.linear5 = nn.Linear(512, out_dim)

    # def forward(self, x):
    #     x = self.linear1(x)
    #     x = self.linear2(x)
    #     x = self.linear3(x)
    #     x = self.linear4(x)
    #     x = self.linear5(x)
    #     return x

    def __init__(self, in_dim, out_dim):
        super(LinearLayer, self).__init__()
        self.linear1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.25)
        self.linear4 = nn.Linear(256, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)
        x = self.linear4(x)
        return x

class ABNClassifier(nn.Module):
    def __init__(self):
        super(ABNClassifier, self).__init__()
        self.bn = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 32, 1)  # 1x1 Convolution, 入力チャンネル: 100, 出力チャンネル: 16
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 200, 1)  # 1x1 Convolution, 入力チャンネル: 16, 出力チャンネル: 200

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling (GAP)
        x = torch.flatten(x, 1)  # バッチサイズとチャンネル数以外の次元を削除
        return x

class ClipFeatureClassifier(nn.Module):
    # def __init__(self, num_classes=200):
    #     super(ClipFeatureClassifier, self).__init__()
    #     self.conv1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     # 二つのプーリング層を通過した後のサイズを計算
    #     self.flattened_size = 128 * 5 * 5
    #     self.fc1 = nn.Linear(self.flattened_size, 1024)
    #     self.fc2 = nn.Linear(1024, num_classes)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    def __init__(self, num_classes=200):
        super(ClipFeatureClassifier, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TensorTransformation(nn.Module):
    def __init__(self):
        super(TensorTransformation, self).__init__()
        self.linear = nn.Linear(512, 768)
        self.conv = nn.Conv2d(1536, 768, kernel_size=3, padding=1)

    def forward(self, embedded_caption, clip_feature):
        # Tensor1を線形変換し、適切な形状に変換
        transformed_tensor1 = embedded_caption.unsqueeze(-1).unsqueeze(-1)  # [8, 768, 1, 1]

        # Tensor1をTensor2と同じサイズに拡大
        expanded_tensor1 = nn.functional.interpolate(transformed_tensor1, size=clip_feature.shape[2:], mode='nearest')  # [8, 768, 20, 20]

        # 2つのテンソルを結合
        concatenated_tensor = torch.cat([expanded_tensor1, clip_feature], dim=1)  # [8, 1536, 20, 20]

        # 結合したテンソルに畳み込みを適用
        output_tensor = self.conv(concatenated_tensor)  # [8, 768, 20, 20]

        return output_tensor

class DoubleTransposedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DoubleTransposedConv, self).__init__()
        # 第一層の逆畳み込み
        self.transposed_conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        # 第二層の逆畳み込み
        self.transposed_conv2 = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = self.transposed_conv1(x)
        x = self.transposed_conv2(x)
        return x
