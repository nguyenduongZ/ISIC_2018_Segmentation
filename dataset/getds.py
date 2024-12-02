import os, sys

from .isic2018 import CustomISIC2018
from torch.utils.data import DataLoader

def get_ds_isic2018(args):
    train_ds = CustomISIC2018(args=args, split="train")
    valid_ds = CustomISIC2018(args=args, split="valid")
    test_ds = CustomISIC2018(args=args, split="test")

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds(args):
    ds_mapping = {
        "isic2018" : get_ds_isic2018
    }

    data, args = ds_mapping[args.ds](args)

    return data, args