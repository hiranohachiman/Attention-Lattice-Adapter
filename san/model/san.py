from typing import List
import open_clip
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F

from .layers import ClassifierHead, patch_based_importance_avg, ConvReducer, ClassificationCNN, ModifiedModel
from .clip_utils import FeatureExtractor, LearnableBgOvClassifier, PredefinedOvClassifier, RecWithAttnbiasHead, get_predefined_templates
from .criterion import SetCriterion, cross_entropy_loss
from .matcher import HungarianMatcher
from .side_adapter import build_side_adapter_network
from .visualize import save_overlay_image_with_matplotlib, save_side_by_side_image, save_image_to_directory


@META_ARCH_REGISTRY.register()
class SAN(nn.Module):

    @configurable
    def __init__(self, *, clip_visual_extractor, clip_rec_head, side_adapter_network, 
                 ov_classifier, criterion, size_divisibility, asymetric_input=True, 
                 clip_resolution=0.5, pixel_mean=[0.48145466, 0.4578275, 0.40821073], 
                 pixel_std=[0.26862954, 0.26130258, 0.27577711], 
                 sem_seg_postprocess_before_inference=False):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility
        self.criterion = criterion
        self.side_adapter_network = side_adapter_network
        self.clip_visual_extractor = clip_visual_extractor
        self.clip_rec_head = clip_rec_head
        self.ov_classifier = ov_classifier
        self.linear = nn.Linear(100, 1)
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.conv1 = ConvReducer(100, 1)
        self.simplecnn = ClassificationCNN()
        self.modi = ModifiedModel()

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
                    'criterion': criterion, 
                    'size_divisibility': cfg.MODEL.SAN.SIZE_DIVISIBILITY, 
                    'asymetric_input': cfg.MODEL.SAN.ASYMETRIC_INPUT, 
                    'clip_resolution': cfg.MODEL.SAN.CLIP_RESOLUTION, 
                    'sem_seg_postprocess_before_inference': cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE, 
                    'pixel_mean': pixel_mean, 
                    'pixel_std': pixel_std}

    def forward(self, batched_inputs):
        labels = []
        if 'vocabulary' in batched_inputs[0]:
            ov_classifier_weight = self.ov_classifier.logit_scale.exp() * \
                                   self.ov_classifier.get_classifier_by_vocabulary(batched_inputs[0]['vocabulary'])
        else:
            dataset_names = [x['meta']['dataset_name'] for x in batched_inputs]
            assert len(list(set(dataset_names))) == 1, 'All images in a batch must be from the same dataset.'
            ov_classifier_weight = self.ov_classifier.logit_scale.exp() * \
                                   self.ov_classifier.get_classifier_by_dataset_name(dataset_names[0])

        if self.training:
            labels = [x['label'].to(self.device) for x in batched_inputs]
            labels = torch.stack(labels)
        else:
            labels.append(torch.tensor(batched_inputs[0]['label']).to(self.device))
            labels = torch.stack(labels)

        images = [x['image'].to(self.device) for x in batched_inputs]
        for_saving_images = ImageList.from_tensors(images, self.size_divisibility)
        for_saving_images = for_saving_images.tensor
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        clip_input = images.tensor
        save_image_to_directory(for_saving_images[0])

        if self.asymetric_input:
            clip_input = F.interpolate(clip_input, scale_factor=self.clip_resolution, mode='bilinear')
        clip_image_features = self.clip_visual_extractor(clip_input)
        mask_preds, attn_biases = self.side_adapter_network(images.tensor, clip_image_features)
        reshaped_mask_preds = patch_based_importance_avg(mask_preds[-1])
        reshaped_mask_preds = self.conv1(reshaped_mask_preds)
        reshaped_mask_preds = reshaped_mask_preds.repeat(1, 768, 1, 1)
        clip_image_features[9] += reshaped_mask_preds
        mask_preds_for_output = self.conv1(mask_preds[-1])
        save_side_by_side_image(mask_preds_for_output[0], for_saving_images[0])

        mask_embs = [self.clip_rec_head(clip_image_features, attn_bias, normalize=True) for attn_bias in attn_biases]
        mask_logits = [torch.einsum('bqc,nc->bqn', mask_emb, ov_classifier_weight) for mask_emb in mask_embs]
        logits = mask_logits[-1][:, :, :200]
        logits = torch.mean(logits, dim=1)

        if self.training:
            loss = cross_entropy_loss(logits, labels)
            mask_preds = F.interpolate(mask_preds[-1],
                                       size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                                       mode='bilinear',
                                       align_corners=False)
            num_input_channels = mask_preds.size(1)
            num_classes = 200
            attn_classifier = ClassifierHead(num_input_channels, num_classes).cuda()
            attn_class_preds = attn_classifier(mask_preds)
            attn_loss = cross_entropy_loss(attn_class_preds, labels)
            losses = {'normal_loss': loss, 'attn_loss': attn_loss}
            return losses

        mask_preds = mask_preds[-1]
        mask_logits = mask_logits[-1]
        mask_preds = F.interpolate(mask_preds,
                                   size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                                   mode='bilinear',
                                   align_corners=False)
        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size, logit, label in zip(mask_logits, mask_preds, batched_inputs, images.image_sizes, logits, labels):
            height = input_per_image.get('height', image_size[0])
            width = input_per_image.get('width', image_size[1])
            processed_results.append({})
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_result, image_size, height, width)
                mask_cls_result = mask_cls_result.to(mask_pred_result)
            r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
            if not self.sem_seg_postprocess_before_inference:
                r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
            processed_results[-1]['sem_seg'] = r
            processed_results[-1]['pred_class'] = logit.argmax()
            processed_results[-1]['gt_class'] = label
        return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad),
                                       dtype=gt_masks.dtype,
                                       device=gt_masks.device)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            new_targets.append({'labels': targets_per_image.gt_classes, 'masks': padded_masks})
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
        return semseg

    @property
    def device(self):
        return self.pixel_mean.device
