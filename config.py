# -*- coding: utf-8 -*-
"""Configurations."""

import torch
import random
import argparse
import logging
from pathlib import Path
from utils import set_logger, save_dict


def build_parser():
    """Get arguments from cmd."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed',
                        type=int,
                        default=1)

    parser.add_argument('--mode',
                        type=str,
                        choices=['train', 'eval'],
                        default='train')

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help="device")

    parser.add_argument('--init',
                        type=str,
                        default='xavier',
                        choices=['kaiming', 'xavier'])

    parser.add_argument('--ema_flag',
                        action='store_true',
                        default=False,
                        help="Use ema")

    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help="ema decay")

    parser.add_argument('--data',
                        type=str,
                        default='multi30k',
                        choices=['multi30k'],
                        help="Dataset")

    parser.add_argument('--task',
                        type=str,
                        default='en2de',
                        choices=['en2de', 'de2en'],
                        help="Translation Task")

    parser.add_argument('--max_len',
                        type=int,
                        default=50,
                        help="Maximum number of tokens")

    parser.add_argument('--label_smoothing',
                        type=float,
                        default=0.1)

    parser.add_argument('--batch_size',
                        type=int,
                        default=128)

    parser.add_argument('--epoch',
                        type=int,
                        default=50)

    parser.add_argument('--d_model',
                        type=int,
                        default=256,
                        help="Used 512 in the paper.")

    parser.add_argument('--d_ff',
                        type=int,
                        default=512,
                        help="2048 used in the paper.")

    parser.add_argument('--n_layers',
                        type=int,
                        default=3,
                        help="6 used in the paper.")

    parser.add_argument('--n_heads',
                        type=int,
                        default=8,
                        help="8 used in the paper.")

    parser.add_argument('--attention_drop',
                        type=float,
                        default=0.0,)

    parser.add_argument('--residual_drop',
                        type=float,
                        default=0.1)

    parser.add_argument('--embedding_drop',
                        type=float,
                        default=0.1)

    parser.add_argument('--pos_encoding',
                        type=str,
                        choices=['embedding', 'sinusoid'],
                        default='embedding')

    parser.add_argument('--optimization',
                        type=str,
                        choices=['Adam', 'AdamW'],
                        default='AdamW')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4)

    parser.add_argument('--lr',
                        type=float,
                        default=0.0005)

    parser.add_argument('--lr_betas',
                        type=float,
                        nargs='+',
                        default=[0.9, 0.98])

    parser.add_argument('--scheduler_policy',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="lr scheduler policy, 0: don't adjust lr, 1: warmup")

    parser.add_argument('--lr_warmup',
                        type=int,
                        default=4000,
                        help="lr scheduler warm up steps")

    parser.add_argument('--pad_index',
                        type=int,
                        default=1)

    parser.add_argument('--min_freq',
                        type=int,
                        default=2,
                        help="Minimum frequency to build vocab")

    parser.add_argument('--print_interval',
                        type=int,
                        default=100,
                        help="print interval")

    parser.add_argument('--save_path',
                        type=Path,
                        default='./model-store/ex01/')

    parser.add_argument('--wb_flag',
                        action='store_true',
                        default=False,
                        help="Use wandb")

    parser.add_argument('--wb_project',
                        type=str,
                        default='transformer')

    parser.add_argument('--wb_name',
                        type=str,
                        default=None)

    parser.add_argument('--wb_notes',
                        type=str,
                        default=None)

    parser.add_argument('--wb_tags',
                        type=str,
                        nargs='+',
                        default=None)

    return parser.parse_args()


def get_arguments():
    """Get arguments."""
    # Get arguments
    args = build_parser()

    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.random_seed)

    # Make a directory to save model weights
    args.save_path.mkdir(exist_ok=True, parents=True)

    # Calculate warmup steps for learning rate scheduler on multi30k dataset.
    if args.data == 'multi30k':
        # paramters in the paper.
        _warmup = 4_000  # warmup steps
        _num_iter = 100_000  # number of total steps

        train_size = 29_000  # multi30k dataset size
        num_iter = (train_size / args.batch_size) * args.epoch
        args.lr_warmup = int((_warmup * num_iter) / _num_iter)

    # Set logger
    set_logger(args.save_path)

    # Log arguments
    logging.debug("[Arguments]")
    for k, v in vars(args).items():
        logging.debug(f"{k}: {v}")

    # Save arguments to load the trained model.
    save_dict(args.save_path / 'args.pkl', args)

    return args
