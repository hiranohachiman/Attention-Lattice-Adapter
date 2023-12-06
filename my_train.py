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
from torchinfo import summary
from torch import optim
from tqdm import tqdm
from san import add_san_config
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)

import torchvision.models as models

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger


from san.data import build_detection_test_loader, build_detection_train_loader
from san.utils import WandbWriter, setup_wandb
from san.data.dataloader import train_loader, valid_loader, test_loader
from san.model.san import SAN


def my_train(model, train_loader, optimizer, criterion1, criterion2, epoch, num_epochs, device="cuda"):
    model.train()
    model = model.to(device)
    criterion1 = criterion1.to(device)
    criterion2 = criterion2.to(device)

    main_losses = []
    # attn_losses = []
    for i, (images, captions, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)  # deviceは 'cuda' または 'cuda:0' など
        labels = labels.to(device)
        # logits, attn_class_preds = model(images)
        logits = model(images)
        main_loss = criterion1(logits, labels)
        # attn_loss = criterion2(attn_class_preds, labels)
        main_losses.append(main_loss.item())
        # attn_losses.append(attn_loss.item())
        # loss = main_loss + attn_loss
        loss = main_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
    # print('Epoch [{}/{}], main_loss:{:.3f}, attn_loss:{:.3f}, total_loss: {:.3f}, lr = {}'.format(epoch + 1, num_epochs, sum(main_losses) / len(main_losses), sum(attn_losses) / len(attn_losses), sum(main_losses) / len(main_losses) + sum(attn_losses) / len(attn_losses), optimizer.param_groups[0]['lr']))
    print('Epoch [{}/{}], main_loss:{:.3f}, lr = {}'.format(epoch + 1, num_epochs, sum(main_losses) / len(main_losses), optimizer.param_groups[0]['lr']))
    return model

def eval(model, valid_loader, epoch, split="val", device="cuda"):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        total = 0
        correct = 0
        for i, (images, captions, labels) in enumerate(tqdm(valid_loader)):
            images = images.to(device)
            labels = labels.to(device)
            # logits, _ = model(images)
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Accuracy of the model on the {} images: {:.3f} %'.format(split, 100 * correct / total))
    return accuracy



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
    # model = Trainer.build_model(cfg)
    model = SAN(**SAN.from_config(cfg))
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 200)
    print(model)
    # summary(
    #     model,
    #     input_size=(8, 3, 640, 640),
    # )

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # trainer.resume_or_load(resume=args.resume)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    num_epochs = cfg.SOLVER.MAX_ITER//625
    for epoch in range(num_epochs):
        model = my_train(model, train_loader, optimizer, criterion1, criterion2, epoch, num_epochs)
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
