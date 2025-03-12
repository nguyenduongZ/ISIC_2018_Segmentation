import os, sys
import argparse
from typing import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ISIC_2018_Segmentation')

    # Dataset
    parser.add_argument('--ds', type=str, required=True, choices = ['lesion'],
        help='dataset used in training')
    parser.add_argument('--bs', type=int, required=True, default=16,
        help='batch size used for data set')
    parser.add_argument('--pinmem', action='store_true',
        help='toggle to pin memory in data loader')
    parser.add_argument('--wk', type=int, default=12,  
        help='number of worker processor contributing to data preprocessing')
    parser.add_argument('--polar', action='store_true',
        help='use polar coordinates')
    parser.add_argument('--percent', type=float, default=None,
        help='percent of the training dataset to use')
    
    # TRAINING GENERAL SETTINGS
    parser.add_argument('--idx', type=int, default=0,
        help='device index used in training')
    parser.add_argument('--seed', type=int, default=42,
        help='seed used in training')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resunetpp', 'deeplab'],
        help='backbone used in training')
    parser.add_argument('--loss', type=str, default='loss', choices=['loss'],
        help='loss function used in training')
    parser.add_argument('--epochs', type=int, default=100,
        help='number of epochs used in training')
    parser.add_argument('--test', action='store_true',
        help='toggle to say that this experiment is just flow testing')

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="ISIC_2018_Segmentation",
        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
        help='toggle to use wandb for online saving')
    
    # MODEL
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate')
    # parser.add_argument('--init_features', type=int, default=32,
    #     help='number of kernel in the first')
    # parser.add_argument('--in_channels', type=int, default=3,
    #     help='number of in_channels')
    # parser.add_argument('--out_channels', type=int, default=1,
    #     help='number of out_channels')
    
    args = parser.parse_args()

    from train import train
    train(args)