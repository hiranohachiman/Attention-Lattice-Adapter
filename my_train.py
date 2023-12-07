try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import copy
import itertools
import logging
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch
from torch.nn import init
import torch.nn as nn
from tqdm import tqdm
import wandb
from torch import optim

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesSemSegEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab.lr_scheduler import WarmupPolyLR

from tabulate import tabulate

from san import (
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_san_config,
)
from san.model.san import SAN
from san.data import build_detection_test_loader, build_detection_train_loader
from san.utils import WandbWriter, setup_wandb
from san.data.dataloader import train_loader, valid_loader, test_loader

def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        #init.constant_(m.bias.data, 0.0)

def apply_weight_init(m):
    # モジュールが nn.Linear のインスタンスであるかどうかをチェックします。
    if isinstance(m, nn.Linear):
        weight_init_kaiming(m)

# class Trainer(DefaultTrainer):
#     def build_writers(self):
#         writers = super().build_writers()
#         # use wandb writer instead.
#         writers[-1] = WandbWriter()
#         return writers

#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         evaluator_list = []
#         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#         # semantic segmentation
#         if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
#             evaluator_list.append(
#                 SemSegEvaluator(
#                     dataset_name,
#                     distributed=True,
#                     output_dir=output_folder,
#                 )
#             )

#         if evaluator_type == "cityscapes_sem_seg":
#             assert (
#                 torch.cuda.device_count() > comm.get_rank()
#             ), "CityscapesEvaluator currently do not work with multiple machines."
#             return CityscapesSemSegEvaluator(dataset_name)

#         if len(evaluator_list) == 0:
#             raise NotImplementedError(
#                 "no Evaluator for the dataset {} with the type {}".format(
#                     dataset_name, evaluator_type
#                 )
#             )
#         elif len(evaluator_list) == 1:
#             return evaluator_list[0]
#         return DatasetEvaluators(evaluator_list)

#     @classmethod
#     def build_train_loader(cls, cfg):
#         # resue maskformer dataset mapper
#         if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
#             mapper = MaskFormerSemanticDatasetMapper(cfg, True)
#             return build_detection_train_loader(cfg, mapper=mapper)
#         else:
#             mapper = None
#             return build_detection_train_loader(cfg, mapper=mapper)

#     @classmethod
#     def build_test_loader(cls, cfg, dataset_name):
#         # Add dataset meta info.
#         return build_detection_test_loader(cfg, dataset_name)

#     @classmethod
#     def build_lr_scheduler(cls, cfg, optimizer):
#         # use poly scheduler
#         return build_lr_scheduler(cfg, optimizer)

#     @classmethod
#     def build_optimizer(cls, cfg, model):
#         # model.apply(apply_weight_init)
#         # # print("!!!!!!!!!!!!!!!!!!!")
#         # # print(model)
#         # # print("!!!!!!!!!!!!!!!!!!!!")
#         weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
#         weight_decay_embed_group = cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP
#         weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

#         defaults = {}
#         defaults["lr"] = cfg.SOLVER.BASE_LR
#         defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

#         norm_module_types = (
#             torch.nn.BatchNorm1d,
#             torch.nn.BatchNorm2d,
#             torch.nn.BatchNorm3d,
#             torch.nn.SyncBatchNorm,
#             # NaiveSyncBatchNorm inherits from BatchNorm2d
#             torch.nn.GroupNorm,
#             torch.nn.InstanceNorm1d,
#             torch.nn.InstanceNorm2d,
#             torch.nn.InstanceNorm3d,
#             torch.nn.LayerNorm,
#             torch.nn.LocalResponseNorm,
#         )

#         params: List[Dict[str, Any]] = []
#         memo: Set[torch.nn.parameter.Parameter] = set()
#         for module_name, module in model.named_modules():
#             for module_param_name, value in module.named_parameters(recurse=False):
#                 if not value.requires_grad:
#                     continue
#                 # Avoid duplicating parameters
#                 if value in memo:
#                     continue
#                 memo.add(value)

#                 hyperparams = copy.copy(defaults)
#                 hyperparams["param_name"] = ".".join([module_name, module_param_name])
#                 if "side_adapter_network" in module_name:
#                     hyperparams["lr"] = (
#                         hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
#                     )
#                 # scale clip lr
#                 if "clip" in module_name:
#                     hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
#                 if any([x in module_param_name for x in weight_decay_embed_group]):
#                     hyperparams["weight_decay"] = weight_decay_embed
#                 if isinstance(module, norm_module_types):
#                     hyperparams["weight_decay"] = weight_decay_norm
#                 if isinstance(module, torch.nn.Embedding):
#                     hyperparams["weight_decay"] = weight_decay_embed
#                 params.append({"params": [value], **hyperparams})

#         def maybe_add_full_model_gradient_clipping(optim):
#             # detectron2 doesn't have full model gradient clipping now
#             clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
#             enable = (
#                 cfg.SOLVER.CLIP_GRADIENTS.ENABLED
#                 and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
#                 and clip_norm_val > 0.0
#             )

#             class FullModelGradientClippingOptimizer(optim):
#                 def step(self, closure=None):
#                     all_params = itertools.chain(
#                         *[x["params"] for x in self.param_groups]
#                     )
#                     torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
#                     super().step(closure=closure)

#             return FullModelGradientClippingOptimizer if enable else optim

#         optimizer_type = cfg.SOLVER.OPTIMIZER
#         if optimizer_type == "SGD":
#             optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
#                 params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
#             )
#         elif optimizer_type == "ADAMW":
#             optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
#                 params, cfg.SOLVER.BASE_LR
#             )
#         else:
#             raise NotImplementedError(f"no optimizer type {optimizer_type}")
#         if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
#             optimizer = maybe_add_gradient_clipping(cfg, optimizer)
#         # display the lr and wd of each param group in a table
#         optim_info = defaultdict(list)
#         total_params_size = 0
#         for group in optimizer.param_groups:
#             optim_info["Param Name"].append(group["param_name"])
#             optim_info["Param Shape"].append(
#                 "X".join([str(x) for x in list(group["params"][0].shape)])
#             )
#             total_params_size += group["params"][0].numel()
#             optim_info["Lr"].append(group["lr"])
#             optim_info["Wd"].append(group["weight_decay"])
#         # Counting the number of parameters
#         optim_info["Param Name"].append("Total")
#         optim_info["Param Shape"].append("{:.2f}M".format(total_params_size / 1e6))
#         optim_info["Lr"].append("-")
#         optim_info["Wd"].append("-")
#         table = tabulate(
#             list(zip(*optim_info.values())),
#             headers=optim_info.keys(),
#             tablefmt="grid",
#             floatfmt=".2e",
#             stralign="center",
#             numalign="center",
#         )
#         logger = logging.getLogger("san")
#         logger.setLevel(logging.ERROR)
#         logger.info("Optimizer Info:\n{}\n".format(table))
#         return optimizer

#     @classmethod
#     def test_with_TTA(cls, cfg, model):
#         logger = logging.getLogger("detectron2.trainer")
#         # In the end of training, run an evaluation with TTA.
#         logger.info("Running inference with test-time augmentation ...")
#         model = SemanticSegmentorWithTTA(cfg, model)
#         evaluators = [
#             cls.build_evaluator(
#                 cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
#             )
#             for name in cfg.DATASETS.TEST
#         ]
#         res = cls.test(cfg, model, evaluators)
#         res = OrderedDict({k + "_TTA": v for k, v in res.items()})
#         return res

def my_train(model, train_loader, optimizer, scheduler, criterion, epoch, num_epochs, device="cuda"):
    model.train()
    model = model.to(device)
    criterion = criterion.to(device)
    main_losses = []
    attn_losses = []
    for i, (images, features, cls_tokens, captions, labels) in enumerate(tqdm(train_loader)):
        clip_image_features = {}
        for i in range(len(features)):
            clip_image_features[i] = features[i].to(device)
            clip_image_features[f"{i}_cls_token"] = cls_tokens[i].to(device)
        labels = labels.to(device)
        logits, attn_class_preds = model(images, clip_image_features, captions)
        main_loss = criterion(logits, labels)
        attn_loss = criterion(attn_class_preds, labels)
        loss = main_loss + attn_loss
        main_losses.append(main_loss.item())
        attn_losses.append(attn_loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_main_loss = sum(main_losses) / len(main_losses)
    avg_attn_loss = sum(attn_losses) / len(attn_losses)
    wandb.log({"main_loss": avg_main_loss, "attn_loss": avg_attn_loss})
    print('Epoch [{}/{}], main_loss:{:.3f}, attn_loss:{:.3f}, total_loss: {:.3f}, lr = {}'.format(epoch + 1, num_epochs, avg_main_loss, avg_attn_loss, avg_main_loss + avg_attn_loss, optimizer.param_groups[0]['lr']))
    scheduler.step()
    return model

def eval(model, valid_loader, epoch, split="val", device="cuda"):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        total = 0
        correct = 0
        for i, (images, features, cls_tokens, captions, labels) in enumerate(tqdm(valid_loader)):
            clip_image_features = {}
            for i in range(len(features)):
                clip_image_features[i] = features[i].to(device)
                clip_image_features[f"{i}_cls_token"] = cls_tokens[i].to(device)
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images, clip_image_features, captions)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Accuracy of the model on the {} images: {:.3f} %'.format(split, 100 * correct / total))
        wandb.log({f"{split}_accuracy": accuracy})
    return accuracy

def build_lr_scheduler(
    cfg, optimizer
):
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            power=cfg.SOLVER.POLY_LR_POWER,
            constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
        )

def build_optimizer(cfg, model):
    # model.apply(apply_weight_init)
    # # print("!!!!!!!!!!!!!!!!!!!")
    # # print(model)
    # # print("!!!!!!!!!!!!!!!!!!!!")
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed_group = cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            hyperparams["param_name"] = ".".join([module_name, module_param_name])
            if "side_adapter_network" in module_name:
                hyperparams["lr"] = (
                    hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                )
            # scale clip lr
            if "clip" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
            if any([x in module_param_name for x in weight_decay_embed_group]):
                hyperparams["weight_decay"] = weight_decay_embed
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(
                    *[x["params"] for x in self.param_groups]
                )
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    # display the lr and wd of each param group in a table
    optim_info = defaultdict(list)
    total_params_size = 0
    for group in optimizer.param_groups:
        optim_info["Param Name"].append(group["param_name"])
        optim_info["Param Shape"].append(
            "X".join([str(x) for x in list(group["params"][0].shape)])
        )
        total_params_size += group["params"][0].numel()
        optim_info["Lr"].append(group["lr"])
        optim_info["Wd"].append(group["weight_decay"])
    # Counting the number of parameters
    optim_info["Param Name"].append("Total")
    optim_info["Param Shape"].append("{:.2f}M".format(total_params_size / 1e6))
    optim_info["Lr"].append("-")
    optim_info["Wd"].append("-")
    table = tabulate(
        list(zip(*optim_info.values())),
        headers=optim_info.keys(),
        tablefmt="grid",
        floatfmt=".2e",
        stralign="center",
        numalign="center",
    )
    logger = logging.getLogger("san")
    logger.setLevel(logging.ERROR)
    logger.info("Optimizer Info:\n{}\n".format(table))
    return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    if not args.eval_only:
        setup_wandb(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="san")
    return cfg


def main(args):
    cfg = setup(args)
    model = SAN(**SAN.from_config(cfg))
    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    #     if cfg.TEST.AUG.ENABLED:
    #         res.update(Trainer.test_with_TTA(cfg, model))
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     return res

    # trainer = Trainer(cfg)
    optimizer = build_optimizer(cfg, model)
    # trainer.resume_or_load(resume=args.resume)
    criterion = nn.CrossEntropyLoss()
    num_epochs = cfg.SOLVER.MAX_ITER
    scheduler = build_lr_scheduler(cfg, optimizer)
    for epoch in range(num_epochs):
        model = my_train(model, train_loader, optimizer, scheduler, criterion, epoch, num_epochs)
        if epoch % 1 == 0:
            arrucacy = eval(model, valid_loader, epoch, split="val")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch}.pth"))
    return

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
