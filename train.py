import argparse
import json
import os, sys
import datetime
from pprint import pprint
import torch
import random
import torch.optim as optim
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.utils import setup_logger
from ignite.metrics import Loss
from mapping import mapping
from utils import *
from data import get_ds
from metrics import DiceMetric

def train(args):
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

    # Data setup 
    data, args = get_ds(args)
    _, _, _, train_dl, valid_dl, _ = data

    # Logging setup
    logger = Logging(args)

    # Mapping
    try:
        model_class = mapping[args.ds]["model"][args.model]
        # metric_class = mapping[args.ds]["metrics"]["dsc"]
        loss_class = mapping[args.ds]["loss"][args.loss]
    except KeyError as e:
        raise ValueError(f"Invalid key in mapping: {e}")
    
    # Model setup
    if args.model == "unet":
        model = model_class(device=device)  
    else:
        model = model_class() 
    model.to(device)

    #
    criterion = loss_class()
    metrics = {
      'dsc': DiceMetric(device=device),
      'loss': Loss(criterion)
    }   
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Trainer and Evaluator
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger('Trainer')

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    best_dsc = 0

    # Event
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        nonlocal best_dsc
        train_evaluator.run(train_dl)
        validation_evaluator.run(valid_dl)
        curr_dsc = validation_evaluator.state.metrics["dsc"]
        if curr_dsc > best_dsc:
            best_dsc = curr_dsc
        logger.step(engine.state.epoch)

    def score_function(engine):
        return engine.state.metrics["dsc"]

    model_checkpoint = ModelCheckpoint(
        args.exp_dir,
        n_saved=2,
        filename_prefix='best',
        score_function=score_function,
        score_name='dsc',
        global_step_transform=None, 
        require_empty=False
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {'model': model})

    trainer.run(train_dl, max_epochs=args.epochs)