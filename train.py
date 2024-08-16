# -*- coding: utf-8 -*-
"""Training Code."""

import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List
from collections import defaultdict
from models import get_transformer
from data import get_text_data
from utils import AverageMeter
from evaluate import evaluate_step, calc_bleu_score
from ema import EMA


def get_optim(model: nn.Module,
              optimization: str,
              weight_decay: float,
              lr: float,
              lr_betas: List[float],
              label_smoothing: float,
              scheduler_policy: int = 0,
              pad_index: int = 1,
              d_model: int = 512,
              warmup: int = 90,
              eps: float = 1e-9):
    """Get optimizer and criterion."""
    if optimization == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=lr,
                                betas=lr_betas,
                                weight_decay=weight_decay,
                                eps=eps)

    elif optimization == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               betas=lr_betas,
                               weight_decay=weight_decay,
                               eps=eps)

    else:
        raise NotImplementedError("Not expected optimization algorithm")

    if scheduler_policy == 0:
        scheduler = None

    elif scheduler_policy == 1:
        def lr_lambda(step):
            step = step + 1
            return d_model**(-0.5) * min(step**(-0.5), step * warmup**(-1.5))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise NotImplementedError("Not expected scheduler policy")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_index,
                                    label_smoothing=label_smoothing)
    return optimizer, criterion, scheduler


def train_step(model,
               dataloader,
               criterion,
               optimizer,
               scheduler,
               d_out,
               device,
               print_interval,
               ema,
               clip=1):
    """Train one epoch."""
    global global_step
    model.train()

    time_logger = defaultdict(AverageMeter)
    loss_logger = AverageMeter()

    tictoc = time.time()
    for i, (src, dst) in enumerate(dataloader, 1):
        time_logger['data'].update(time.time() - tictoc)
        optimizer.zero_grad()
        src = src.to(device)
        dst = dst.to(device)

        tictoc = time.time()
        output, _ = model(src, dst[:, :-1])
        time_logger['forward'].update(time.time() - tictoc)

        tictoc = time.time()
        output = output.contiguous().view(-1, d_out)
        dst = dst[:, 1:].contiguous().view(-1)

        loss = criterion(output, dst)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        time_logger['backward'].update(time.time() - tictoc)

        if ema is not None:
            ema.update()

        loss_logger.update(loss.item(), n=src.shape[0])

        global_step += 1
        if i % print_interval == 0:
            logging.info((f"Step: {global_step}, "
                          f"Time(ms): [data: {time_logger['data'].avg * 1000:1.4f}, "
                          f"forward: {time_logger['forward'].avg * 1000:1.4f}, "
                          f"backward: {time_logger['backward'].avg * 1000:1.4f}], "
                          f"Loss: {loss_logger.avg:1.4f}"))

    return loss_logger.avg


def train_model(args):
    """Train a model with arguments."""
    if args.wb_flag:
        import wandb
    global global_step
    global_step = 0

    device = torch.device(args.device)
    tokenizers, vocabs, datasets = get_text_data(data=args.data,
                                                 task=args.task,
                                                 pad_index=args.pad_index,
                                                 min_freq=args.min_freq)
    d_inp = len(vocabs['src'])
    d_out = len(vocabs['dst'])
    train_dataloader = DataLoader(datasets['train'],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=datasets['train'].collate_fn)

    valid_dataloader = DataLoader(datasets['valid'],
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=datasets['valid'].collate_fn)

    logging.info("Make a Transformer model.")
    transformer = get_transformer(args, d_inp, d_out)
    transformer.to(device)

    ema = None
    if args.ema_flag:
        ema = EMA(transformer, args.ema_decay, device)

    optimizer, criterion, scheduler = get_optim(model=transformer,
                                                optimization=args.optimization,
                                                weight_decay=args.weight_decay,
                                                lr=args.lr,
                                                lr_betas=args.lr_betas,
                                                label_smoothing=args.label_smoothing,
                                                scheduler_policy=args.scheduler_policy,
                                                d_model=args.d_model, warmup=args.lr_warmup,
                                                pad_index=args.pad_index)

    best_val_loss = float('inf')
    logging.info("Start training...")
    for epoch in range(args.epoch):
        train_loss = train_step(model=transformer,
                                dataloader=train_dataloader,
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                d_out=d_out,
                                print_interval=args.print_interval,
                                ema=ema,
                                device=device)

        valid_loss = evaluate_step(model=transformer,
                                   dataloader=valid_dataloader,
                                   criterion=criterion,
                                   d_out=d_out,
                                   device=device)

        if args.ema_flag:
            ema_valid_loss = evaluate_step(model=ema.shadow,
                                           dataloader=valid_dataloader,
                                           criterion=criterion,
                                           d_out=d_out,
                                           device=device)

        logging.info((f"Epoch: {epoch + 1}/{args.epoch}, "
                      f"Train Loss: {train_loss:1.4f}, Valid Loss: {valid_loss:1.4f}"))
        if args.ema_flag:
            logging.info(f"EMA Valid Loss: {ema_valid_loss:1.4f}")

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            ckpt = {
                'step': global_step,
                'state_dict': transformer.state_dict()
            }
            torch.save(ckpt, args.save_path / 'best_val.pt')

        if args.wb_flag:
            wb_log = {
                'train loss': train_loss,
                'valid loss': valid_loss,
            }
            if args.ema_flag:
                wb_log['ema valid loss'] = ema_valid_loss
            wandb.log(wb_log, step=global_step)

    # save trained models
    ckpt = {
        'step': global_step,
        'state_dict': transformer.state_dict(),
        'ema': ema.shadow.state_dict() if args.ema_flag else None
    }
    torch.save(ckpt, args.save_path / 'trained_model.pt')

    # check model performances
    best_val_ckpt = torch.load(args.save_path / 'best_val.pt', map_location=device)
    transformer.load_state_dict(best_val_ckpt['state_dict'])
    bleu_score, sample = calc_bleu_score(sentence_iter=datasets['test'],
                                         model=transformer,
                                         src_tokenizer=tokenizers['src'],
                                         dst_tokenizer=tokenizers['dst'],
                                         src_vocab=vocabs['src'],
                                         dst_vocab=vocabs['dst'],
                                         device=device,
                                         pad_index=args.pad_index,
                                         max_len=args.max_len)
    logging.info((f"BLEU Score(best val): {bleu_score:1.4f}"
                  f"\n\t\t\t - Ground Truth: {' '.join(sample[1])}"
                  f"\n\t\t\t - Prediction: {' '.join(sample[0])}"))
    if args.wb_flag:
        wandb.log({'BLEU': bleu_score})

    if args.ema_flag:
        ema_bleu_score, ema_sample = calc_bleu_score(sentence_iter=datasets['test'],
                                                     model=ema.shadow,
                                                     src_tokenizer=tokenizers['src'],
                                                     dst_tokenizer=tokenizers['dst'],
                                                     src_vocab=vocabs['src'],
                                                     dst_vocab=vocabs['dst'],
                                                     device=device,
                                                     pad_index=args.pad_index,
                                                     max_len=args.max_len)
        logging.info((f"BLEU Score(ema): {ema_bleu_score:1.4f}"
                      f"\n\t\t\t - Ground Truth: {' '.join(ema_sample[1])}"
                      f"\n\t\t\t - Prediction: {' '.join(ema_sample[0])}"))
        if args.wb_flag:
            wandb.log({'BLEU(ema)': ema_bleu_score})

    return transformer
