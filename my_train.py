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
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import random

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
from tabulate import tabulate

from san import (
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_san_config,
)
from san.model.san import SAN
from san.data import build_detection_test_loader, build_detection_train_loader
from san.utils import WandbWriter, setup_wandb
from san.data.dataloader import train_dataset, valid_dataset, test_dataset, _preprocess
from torchinfo import summary
import loralib as lora

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

class Trainer(DefaultTrainer):
    def build_writers(self):
        writers = super().build_writers()
        # use wandb writer instead.
        writers[-1] = WandbWriter()
        return writers

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # resue maskformer dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Add dataset meta info.
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        # use poly scheduler
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
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

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
def save_attn_map(attn_map, path):
    # バッチの最初の要素を選択し、チャンネルの次元を削除
    attn_map = attn_map[0].squeeze()
    # PyTorch TensorをNumPy配列に変換
    attn_map = attn_map.cpu().numpy()
    # attn_mapを0から1の範囲に正規化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    # 値を0から255の範囲にスケーリング
    attn_map = attn_map * 255
    # 整数型にキャスト
    attn_map = attn_map.astype(np.uint8)
    # PIL Imageに変換
    attn_map = Image.fromarray(attn_map)
    # 画像を保存
    attn_map.save(path)

def normalize_batch(batch):
    # バッチごとにループ
    for i in range(batch.size(0)):
        # i番目のバッチを取得
        batch_i = batch[i]
        # バッチ内の最小値と最大値を取得
        min_val = torch.min(batch_i)
        max_val = torch.max(batch_i)
        # 0-1の範囲に正規化
        batch[i] = (batch_i - min_val) / (max_val - min_val)
    return batch

def get_iou(preds, masks, threshhold="mean"):
    preds = F.interpolate(preds, size=(640, 640), mode='bilinear', align_corners=False)
    save_attn_map(preds[0], "attn_map.png")
    # ここでIoUを計算
    # preds = normalize_batch(preds)
    preds = preds.squeeze(1)
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    if threshhold == "mean":
        threshhold = np.mean(preds)
    preds = preds > threshhold
    # save_attn_map(torch.tensor(preds[0], "attn_map_threshhold.png"))
    intersection = np.logical_and(preds, masks)
    union = np.logical_or(preds, masks)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def my_train(model, train_loader, optimizer, scheduler, criterion, epoch, num_epochs, device="cuda"):
    model.train()
    model = model.to(device)
    criterion = criterion.to(device)
    main_losses = []
    attn_losses = []
    for i, (images, _, captions, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)  # deviceは 'cuda' または 'cuda:0' など
        labels = labels.to(device)
        logits, attn_class_preds, _ = model(images)
        main_loss = criterion(logits, labels)
        attn_loss = criterion(attn_class_preds, labels)
        main_losses.append(main_loss.item())
        attn_losses.append(attn_loss.item())
        loss = main_loss + attn_loss * 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_main_loss = sum(main_losses) / len(main_losses)
        avg_attn_loss = sum(attn_losses) / len(attn_losses)
    print('Epoch [{}/{}], main_loss:{:.3f}, attn_loss:{:.3f}, total_loss: {:.3f}, lr = {}'.format(epoch + 1, num_epochs, avg_main_loss, avg_attn_loss, avg_main_loss + avg_attn_loss, optimizer.param_groups[0]['lr']))
    scheduler.step()
    wandb.log({"main_loss": avg_main_loss, "attn_loss": avg_attn_loss})
    return model

def eval(model, valid_loader, criterion, split="val", device="cuda"):
    model.eval()
    model = model.to(device)
    losses = []
    ious = []
    with torch.no_grad():
        total = 0
        correct = 0
        for i, (images, masks, captions, labels) in enumerate(tqdm(valid_loader)):
            images = images.to(device)
            labels = labels.to(device)
            logits, _, attn_maps = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ious.append(get_iou(attn_maps, masks))
            losses.append(criterion(logits, labels).item())
        loss = sum(losses) / len(losses)
        accuracy = 100 * correct / total
        iou = sum(ious) / len(ious)
        print('Accuracy of the model on the {} images: {:.3f} %, loss: {:.3f}, iou: {:.3f}'.format(split, 100 * correct / total, loss, sum(ious) / len(ious)))
        wandb.log({f"{split}_accuracy": accuracy, f"{split}_loss:": loss})
    return accuracy, loss, iou

def predict_one_shot(model_path, image_path, caption, device="cuda"):
    model.load_state_dict(torch.load("output/2023-12-13-23:34:53/epoch_48.pth"), strict=False)
    model.load_state_dict(torch.load("output/2023-12-13-23:34:53/lora_epoch_48.pth"), strict=False)
    model = model.to(device)
    model.eval()
    image = Image.open(image_path)
    image = _preprocess(image)
    image = image.to(device)
    logits, _, attn_map = model(image)
    # save attn_map
    attn_map = attn_map.squeeze(1)
    attn_map = attn_map.cpu().numpy()
    attn_map = attn_map * 255
    attn_map = attn_map.astype(np.uint8)
    attn_map = Image.fromarray(attn_map)
    attn_map.save(f"{image_path}_attn_map.png")
    _, predicted = torch.max(logits.data, 1)
    return predicted

class EarlyStopping():
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float('inf')
        self.best_epoch = None
        self.no_improvement_count = 0

    def early_stop(self, loss, epoch):
        if loss < self.best_loss:
            self.best_loss = loss
            self.no_improvement_count = 0
            self.best_epoch = epoch
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience:
            return True, self.best_epoch
        else:
            return False, self.best_epoch

def delete_non_best_epoch_weights(directory, best_epoch):
    """
    指定されたディレクトリから、'best epoch'ではないモデルの重みを削除します。

    :param directory: モデルの重みが保存されているディレクトリ
    :param best_epoch: 保持したいベストエポックの番号
    """
    print(f"Deleting non-best epoch weights from {directory}...")
    for filename in os.listdir(directory):
        if not filename.endswith(".pth"):  # 重みファイルの拡張子に合わせてください
            continue

        # ファイル名からエポック番号を抽出 (ファイル名の形式に合わせて調整が必要)
        epoch_num = int(filename.split('_')[-1].split('.')[0])

        # ベストエポック以外のファイルを削除
        if epoch_num != best_epoch:
            os.remove(os.path.join(directory, filename))


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
    summary(model, input_size=(8,3,640,640))
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

    trainer = Trainer(cfg)
    optimizer = trainer.build_optimizer(cfg, model)
    scheduler = trainer.build_lr_scheduler(cfg, optimizer)
    trainer.resume_or_load(resume=args.resume)
    criterion = nn.CrossEntropyLoss()
    num_epochs = cfg.SOLVER.MAX_ITER
    early_stopper = EarlyStopping()
    train_loader = DataLoader(train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH , shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=4)
    best_epoch = 0
    for epoch in range(num_epochs):
        model = my_train(model, train_loader, optimizer, scheduler, criterion, epoch, num_epochs)
        arrucacy, loss, iou = eval(model, valid_loader, criterion, split="val")
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch}.pth"))
        torch.save(lora.lora_state_dict(model), os.path.join(cfg.OUTPUT_DIR, f"lora_epoch_{epoch}.pth"))
        early_stop, best_epoch = early_stopper.early_stop(loss, epoch)
        if early_stop:
            print("early stopped...")
            break
    delete_non_best_epoch_weights(cfg.OUTPUT_DIR, best_epoch)
    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, f"epoch_{best_epoch}.pth")), strict=False)
    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, f"lora_epoch_{best_epoch}.pth")), strict=False)
    early_stop, best_epoch = early_stopper.early_stop(loss, model)
    arrucacy, loss, iou = eval(model, test_loader, criterion, split="test")
    return

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

if __name__ == "__main__":
    torch_fix_seed(seed=42)
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
