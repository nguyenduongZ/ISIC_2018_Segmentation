from trainer import trainer

import argparse
import torch
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ISIC2018 Segmentation")

    # Dataset
    parser.add_argument('--ds', type=str, required=True, choices=['isic2018'], help='dataset used in training')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--wk', type=int, default=1, help='number of workers')
    parser.add_argument('--pm', action='store_true', help='pin memory')
    parser.add_argument('--sz', type=int, default=256, help='size of processed image')
    parser.add_argument('--aug', action='store_true', default=False, help='augmentation')

    # TRAINING
    parser.add_argument('--idx', type=int, default=0, help='device index used in training')
    parser.add_argument('--seed', type=int, default=0, help='seed used in training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs used in training')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'segnet'], help='backbone used in training')
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'dice_loss', 'focal_loss', 'combine_loss'], help='loss function used in training')
    parser.add_argument('--task', type=str, default='seg', required=True,
        choices=['clf', 'seg'],
        help='training task')
    parser.add_argument('--test', action='store_true', help='toggle to say that this experiment is just flow testing')

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="ISIC2018 Segmentation",
        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
        help='toggle to use wandb for online saving')
    parser.add_argument('--neptune', action='store_true', 
        help='toggle to use Neptune.ai for logging')
    parser.add_argument('--neptune_prj', type=str, default="ISIC2018 Segmentation", 
        help='Neptune.ai project name')
    parser.add_argument('--neptune_api_token', type=str, required=False, 
        help='API token for Neptune.ai')


    # MODEL
    parser.add_argument('--init_ch', type=int, default=32, help='number of kernel in the first')
    parser.add_argument('--clf_n_classes', type=int, default=37, help='channels of output')
    parser.add_argument('--seg_n_classes', type=int, default=3, help='channels of output')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    # FOCAL - DICELOSS - CELOSS
    parser.add_argument('--alpha', type=float, default=0, 
        help="alpha hyperparameter used in focal loss")
    parser.add_argument('--gamma', type=float, default=0, 
        help="gamma hyperparameter used in focal loss")
    parser.add_argument('--smooth', type=float, default=1.0, 
        help="smooth hyperparameter used in dice loss")
    parser.add_argument('--epsilon', type=float, default=1e-7, 
        help="epsilon hyperparameter used in dice loss")
    parser.add_argument('--weight', type=float, default=None, 
        help="weight for each class in CrossEntropy loss")
    parser.add_argument('--ignore_index', type=int, default=-100, 
        help="index to ignore in CrossEntropy loss")
    parser.add_argument('--reduction', type=str, default='mean', choices=['none', 'mean', 'sum'], 
        help="reduction method for CrossEntropy loss")
    
    args = parser.parse_args()

    trainer(args)