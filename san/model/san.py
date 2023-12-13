from typing import List
import open_clip
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
# from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
# from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F
import numpy as np
import wandb
import math
import loralib as lora

from .layers import ClassifierHead, patch_based_importance_avg, ConvReducer, ClassificationCNN, ModifiedModel, normalize_per_batch, SimpleClassifier, LinearLayer, ABNClassifier, ClipFeatureClassifier, zero_below_average, TensorTransformation
from .clip_utils import FeatureExtractor, LearnableBgOvClassifier, PredefinedOvClassifier, RecWithAttnbiasHead, get_predefined_templates
from .criterion import SetCriterion, cross_entropy_loss, info_nce
from .matcher import HungarianMatcher
from .side_adapter import build_side_adapter_network
from .visualize import save_overlay_image_with_matplotlib, save_side_by_side_image, save_image_to_directory
from .attention import TransformerDecoder, ViTClassifier

@META_ARCH_REGISTRY.register()
class SAN(nn.Module):

    @configurable
    def __init__(self, *, clip_visual_extractor, clip_rec_head, side_adapter_network,
                 ov_classifier, caption_embedder, criterion, size_divisibility, asymetric_input=True,
                 clip_resolution=0.5, pixel_mean=[0.48683309, 0.50015243, 0.43198669],
                 pixel_std=[0.53930313, 0.54964845, 0.50728528],
                 sem_seg_postprocess_before_inference=False):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility
        self.criterion = criterion
        self.side_adapter_network = side_adapter_network
        self.clip_visual_extractor = clip_visual_extractor
        # self.clip_rec_head = clip_rec_head
        # self.ov_classifier = ov_classifier
        # self.caption_embedder = caption_embedder
        # self.linear = nn.Linear(100, 1)
        # self.linear2 = nn.Linear(768, 4096)
        # self.linear3 = nn.Linear(4096, 1024)
        # self.linear4 = nn.Linear(1024, 200)
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.conv1 = ConvReducer(100, 1)
        # self.conv2 = nn.Conv2d(100, 1)
        # self.simplecnn = ClassificationCNN()
        # self.modi = ModifiedModel()
        # self.mask_embs_classifier = SimpleClassifier()
        # self.attention = nn.MultiheadAttention(768, 8)
        # self.transformer = TransformerDecoder(200)
        self.linear5 = LinearLayer(200, 200)
        # self.linear6 = LinearLayer(512, 200)
        self.abnclassifier = ABNClassifier()
        self.clipfeatureclassifier = ClipFeatureClassifier()
        # self.tensortrainformer = TensorTransformation()
        # self.linear = nn.Linear(512, 768)
        self.transformer = ViTClassifier(input_channels=768, num_classes=200, dim=512, depth=2, heads=8, mlp_dim=1024, dropout=0.25)

    @classmethod
    def from_config(cls, cfg):
        no_object_weight = cfg.MODEL.SAN.NO_OBJECT_WEIGHT
        class_weight = cfg.MODEL.SAN.CLASS_WEIGHT
        dice_weight = cfg.MODEL.SAN.DICE_WEIGHT
        mask_weight = cfg.MODEL.SAN.MASK_WEIGHT
        matcher = HungarianMatcher(cost_class=class_weight,
                                   cost_mask=mask_weight,
                                   cost_dice=dice_weight,
                                   num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS)
        weight_dict = {'loss_ce': class_weight,
                       'loss_mask': mask_weight,
                       'loss_dice': dice_weight}
        aux_weight_dict = {}
        for i in range(len(cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS) - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        else:
            weight_dict.update(aux_weight_dict)
            losses = ['labels']
            criterion = SetCriterion(num_classes=cfg.MODEL.SAN.NUM_CLASSES,
                                     matcher=matcher,
                                     weight_dict=weight_dict,
                                     eos_coef=no_object_weight,
                                     losses=losses,
                                     num_points=cfg.MODEL.SAN.TRAIN_NUM_POINTS,
                                     oversample_ratio=cfg.MODEL.SAN.OVERSAMPLE_RATIO,
                                     importance_sample_ratio=cfg.MODEL.SAN.IMPORTANCE_SAMPLE_RATIO)
            model, _, preprocess = open_clip.create_model_and_transforms(cfg.MODEL.SAN.CLIP_MODEL_NAME,
                                                                         pretrained=cfg.MODEL.SAN.CLIP_PRETRAINED_NAME)
            ov_classifier = LearnableBgOvClassifier(model,
                                                    templates=get_predefined_templates(cfg.MODEL.SAN.CLIP_TEMPLATE_SET))
            caption_embedder = PredefinedOvClassifier(model,
                                                    templates=get_predefined_templates(cfg.MODEL.SAN.CLIP_TEMPLATE_SET))
            def integrate_lora_to_vit(vit_model, rank=16):
                # 変更するモジュールの名前と新しいモジュールを保持するリスト
                to_replace = []

                # モジュールを反復処理して変更すべきものを特定
                for name, module in vit_model.named_modules():
                    if isinstance(module, nn.Linear):
                        # LoRA層への置き換えを予約
                        in_features = module.in_features
                        out_features = module.out_features
                        new_module = lora.Linear(in_features, out_features, r=rank)
                        to_replace.append((name, new_module))

                # 実際にモジュールを置き換え
                for name, new_module in to_replace:
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = dict(vit_model.named_modules())[parent_name]
                    setattr(parent, child_name, new_module)
            def integrate_lora_to_specific_layers(vit_model, rank=16):
                layer_names = [
                    "transformer.resblocks.9.mlp.c_fc",
                    "transformer.resblocks.9.mlp.c_proj",
                    "transformer.resblocks.9.attn.in_proj_weight",
                    "transformer.resblocks.9.attn.in_proj_bias",
                    "transformer.resblocks.9.attn.out_proj",
                    "transformer.resblocks.9.ln_2",
                    # "transformer.resblocks.8.mlp.c_fc",
                    # "transformer.resblocks.8.mlp.c_proj",
                    # "transformer.resblocks.8.attn.in_proj_weight",
                    # "transformer.resblocks.8.attn.in_proj_bias",
                    # "transformer.resblocks.8.attn.out_proj",
                    # "transformer.resblocks.8.ln_2",
                    # "transformer.resblocks.7.mlp.c_fc",
                    # "transformer.resblocks.7.mlp.c_proj",
                    # "transformer.resblocks.7.attn.in_proj_weight",
                    # "transformer.resblocks.7.attn.in_proj_bias",
                    # "transformer.resblocks.7.attn.out_proj",
                    # "transformer.resblocks.7.ln_2",

                ]

                for name, module in vit_model.named_modules():
                    if name in layer_names and isinstance(module, nn.Linear):
                        in_features = module.in_features
                        out_features = module.out_features
                        new_module = lora.Linear(in_features, out_features, r=rank)
                        parent_name, child_name = name.rsplit('.', 1)
                        parent = dict(vit_model.named_modules())[parent_name]
                        setattr(parent, child_name, new_module)
            # CLIPのVision Transformer部分にLoRAを統合
            integrate_lora_to_specific_layers(model.visual)
            lora.mark_only_lora_as_trainable(model.visual)
            for name, param in model.visual.named_parameters():
                print(name, param.requires_grad)
            clip_visual_extractor = FeatureExtractor(model.visual,
                                                     last_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
                                                     frozen_exclude=cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE)
            clip_rec_head = RecWithAttnbiasHead(model.visual,
                                                first_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
                                                frozen_exclude=cfg.MODEL.SAN.CLIP_DEEPER_FROZEN_EXCLUDE,
                                                cross_attn=cfg.MODEL.SAN.REC_CROSS_ATTN,
                                                sos_token_format=cfg.MODEL.SAN.SOS_TOKEN_FORMAT,
                                                sos_token_num=cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES,
                                                downsample_method=cfg.MODEL.SAN.REC_DOWNSAMPLE_METHOD)
            pixel_mean, pixel_std = preprocess.transforms[-1].mean, preprocess.transforms[-1].std
            pixel_mean = [255.0 * x for x in pixel_mean]
            pixel_std = [255.0 * x for x in pixel_std]
            return {'clip_visual_extractor': clip_visual_extractor,
                    'clip_rec_head': clip_rec_head,
                    'side_adapter_network': build_side_adapter_network(cfg, clip_visual_extractor.output_shapes),
                    'ov_classifier': ov_classifier,
                    'caption_embedder': caption_embedder,
                    'criterion': criterion,
                    'size_divisibility': cfg.MODEL.SAN.SIZE_DIVISIBILITY,
                    'asymetric_input': cfg.MODEL.SAN.ASYMETRIC_INPUT,
                    'clip_resolution': cfg.MODEL.SAN.CLIP_RESOLUTION,
                    'sem_seg_postprocess_before_inference': cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
                    'pixel_mean': pixel_mean,
                    'pixel_std': pixel_std}

    def forward(self, images):
        images = [x.to(self.device) for x in images]
        # captions = [x for x in captions]
        # embedded_caption = self.caption_embedder(captions)
        # print(embedded_caption.shape) # [8, 512]
        # embedded_caption = self.linear(embedded_caption)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        clip_input = images.tensor
        # print(clip_input.shape) # [8, 3, 640, 640]

        if self.asymetric_input:
            clip_input = F.interpolate(clip_input, scale_factor=self.clip_resolution, mode='bilinear')
        # print(clip_input.shape) # [8, 3, 320, 320]
        clip_image_features = self.clip_visual_extractor(clip_input)
        # [8, 768, 20, 20], [1, 8, 768]
        mask_preds, attn_biases = self.side_adapter_network(images.tensor, clip_image_features)
        # reshaped_mask_preds = patch_based_importance_avg(mask_preds[-1])
        reshaped_mask_preds = self.conv1(mask_preds[-1])
        # reshaped_mask_preds = zero_below_average(reshaped_mask_preds)
        # reshaped_mask_preds = reshaped_mask_preds.repeat(1, 768, 1, 1)
        # clip_image_features[9] *= normalize_per_batch(reshaped_mask_preds)
        clip_image_features[9] *= reshaped_mask_preds
        # print(reshaped_mask_preds.shape)
        logits = self.clipfeatureclassifier(clip_image_features[9])
        # global average pooling
        # clip_image_features[9] += reshaped_mask_preds
        # multimodal_features = self.tensortrainformer(embedded_caption, clip_image_features[9])
        # info_loss = info_nce(multipled_clip_image_features, embedded_caption)
        # logits = self.clipfeatureclassifier(multipled_clip_image_features)
        logits = self.linear5(logits)

        attn_class_preds = self.abnclassifier(mask_preds[-1])
        return logits, attn_class_preds, reshaped_mask_preds

    @property
    def device(self):
        return self.pixel_mean.device
