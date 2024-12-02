import os, sys
from rich.progress import track
import random
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

from dataset import get_ds
from utils import folder_setup, save_cfg, Logging, save_json
from mapping import mapping

def trainer(args):
    # Seed setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=args.idx)    

    # Folder setup and save setting
    args.exp_dir = folder_setup(args)
    save_cfg(args, args.exp_dir)

    # Dataset setup
    data, args = get_ds(args)
    _, _, _, train_dl, valid_dl, _ = data

    # Logging setup
    log_interface = Logging(args)

    # Task mapping
    if args.task not in mapping[args.ds]:
        raise ValueError(f"Currently, task {args.task} is not supported")
    task_dict = mapping[args.ds][args.task]

    # Metrics
    metric_dict = task_dict["metrics"]

    # Loss
    loss_fn = task_dict["loss"][args.loss]

    # Model
    model = task_dict["model"][args.model](args=args).to(device)

    # Optimizer, Scheduler
    optimizer = Adam(model.parameters(), lr = args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max= len(train_dl)*args.epochs)

    if args.wandb or args.neptune:
        log_interface.watch(model)

    # Training
    old_valid_loss = 1e26

    # Training loop
    for epoch in track(range(args.epochs)):
        args.epoch = epoch
        model.train()
        
        for _, (img, target) in enumerate(train_dl):
            img = img.to(device)
            for task_key in target:
                target[task_key] = target[task_key].to(device)
            # target = target.to(device)
            
            pred = model(img)
            seg_target = target['semantic']
            loss = loss_fn(pred["semantic"], seg_target)
            
            log_interface(key="train/loss", value=loss.item())
            
            for metric_key in metric_dict:
                metric_value = metric_dict[metric_key](pred["semantic"], seg_target)  
                log_interface(key=f"train/{metric_key}", value=metric_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            for _, (img, target) in enumerate(valid_dl):
                img = img.to(device)
                for task_key in target:
                    target[task_key] = target[task_key].to(device)
                # target = target.to(device)

                pred = model(img)
                seg_target = target['semantic']
                loss = loss_fn(pred["semantic"], seg_target)

                log_interface(key="valid/loss", value=loss.item())

                for metric_key in metric_dict:
                    metric_value = metric_dict[metric_key](pred["semantic"], seg_target)
                    log_interface(key=f"valid/{metric_key}", value=metric_value)

        # Logging can averaging
        log_interface.step(epoch=epoch)

        # Save best and last model
        mean_valid_loss = log_interface.log_avg["valid/loss"]
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_valid_loss
        }
        if  mean_valid_loss <= old_valid_loss:
            old_valid_loss = mean_valid_loss

            save_path = args.exp_dir + f"/best.pt"
            torch.save(save_dict, save_path)
        
        save_path = args.exp_dir + f"/last.pt"
        torch.save(save_dict, save_path)
    
    # Save model
    log_interface.log_model()        
