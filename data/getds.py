import os, sys
import numpy as np

from .lesion_dataset import LesionDataset
from torch.utils.data import DataLoader

def get_lesion_ds(args):

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    train_ds = LesionDataset(directory="train", polar=args.polar, percent=args.percent, center_augmentation=args.polar)
    valid_ds = LesionDataset(directory="valid", polar=args.polar)
    test_ds = LesionDataset(directory="valid", polar=args.polar)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.wk, worker_init_fn=worker_init)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, drop_last=False, num_workers=args.wk, worker_init_fn=worker_init)
    test_dl = DataLoader(test_ds, batch_size=args.bs, drop_last=False, num_workers=args.wk, worker_init_fn=worker_init)

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds(args):
    ds_mapping ={
        "lesion": get_lesion_ds
    }

    data, args = ds_mapping[args.ds](args)

    return data, args