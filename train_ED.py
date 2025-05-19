#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import os,sys
import json
import numpy as np
import pandas as pd
import argparse
import random
import time
from tqdm import tqdm
from itertools import chain

import torch
import torch.optim as optim
import torch.nn.functional as F
from dataload import load_train_val
from utils import ssim, psnr
from model.SpatiotemporalAttention import AttentiveEncoder, AttentiveDecoder
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
log_dir = os.path.join("results/", "%s" % (
    str(str(now_time))
))
os.makedirs(log_dir, exist_ok=False)


def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()


def quantize(frames):
    # [-1.0, 1.0] -> {0, 255} -> [-1.0, 1.0]
    return ((frames + 1.0) * 127.5).int().float() / 127.5 - 1.0


def make_pair(frames, args):
    frames = torch.cat([frames] * args.multiplicity, dim=0).cuda()
    data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()
    return frames, data


def run(args):

    train, val, _, _ = load_train_val(args.seq_len, args.batch_size, args.dataset)


    if args.recover ==1:
        encoder, decoder = torch.load(args.model_path)
    else:
        encoder = AttentiveEncoder(data_dim=args.data_dim, linf_max=args.linf_max).cuda()
        decoder = AttentiveDecoder(encoder).cuda()


    optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)



    lambda1 = 1.0  # Weight for visual loss
    lambda2 = 1.0  # Weight for decoding loss
    # Set up the log directory
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    log_dir = os.path.join("results/", "%s" % (
        str(str(now_time))
    ))
    os.makedirs(log_dir, exist_ok=False)
    with open(os.path.join(log_dir, "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))
    writer = SummaryWriter(os.path.join(log_dir, "logs"))


    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = {
            "train.loss": [],
            "train.visual_loss": [],
            "train.decoding_loss": [],
            "train.raw_acc": [],
            "val.ssim": [],
            "val.psnr": [],
            "val.identity_acc": [],
        }

        # Train
        gc.collect()
        encoder.train()
        decoder.train()

        # Optimize Encoder-Decoder
        iterator = tqdm(train, ncols=0)
        for frames in iterator:
            frames, data = make_pair(frames, args)

            wm_frames = encoder(frames, data)
            wm_raw_data = decoder(wm_frames)

            # Visual loss (MSE)
            visual_loss = F.mse_loss(wm_frames, frames)
            # Decoding loss (BCE)
            decoding_loss = F.binary_cross_entropy_with_logits(wm_raw_data, data)
            # Total loss
            loss = lambda1 * visual_loss + lambda2 * decoding_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["train.loss"].append(loss.item())
            metrics["train.visual_loss"].append(visual_loss.item())
            metrics["train.decoding_loss"].append(decoding_loss.item())
            metrics["train.raw_acc"].append(get_acc(data, wm_raw_data))
            iterator.set_description(
                "%s | Loss %.3f | Visual %.3f | Decoding %.3f | Acc %.3f" % (
                    epoch,
                    np.mean(metrics["train.loss"]),
                    np.mean(metrics["train.visual_loss"]),
                    np.mean(metrics["train.decoding_loss"]),
                    np.mean(metrics["train.raw_acc"]),
                ))
        writer.add_scalar("train_loss", np.mean(metrics["train.loss"]), epoch)
        writer.close()

        # Validate
        gc.collect()
        encoder.eval()
        decoder.eval()
        iterator = tqdm(val, ncols=0)
        with torch.no_grad():
            for frames in iterator:
                frames = frames.cuda()
                data = torch.zeros((frames.size(0), args.data_dim)).random_(0, 2).cuda()

                wm_frames = encoder(frames, data)
                wm_identity_data = decoder(wm_frames)


                metrics["val.ssim"].append(ssim(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                metrics["val.psnr"].append(psnr(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                metrics["val.identity_acc"].append(get_acc(data, wm_identity_data))


                iterator.set_description("%s | SSIM %.3f | PSNR %.3f |Identity %.3f " % (
                    epoch,
                    np.mean(metrics["val.ssim"]),
                    np.mean(metrics["val.psnr"]),
                    np.mean(metrics["val.identity_acc"]),
                ))

        metrics = {k: round(np.mean(v), 3) if len(v) > 0 else "NaN" for k, v in metrics.items()}
        metrics["epoch"] = epoch
        history.append(metrics)
        pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
        with open(os.path.join(log_dir, "metrics.json"), "wt") as fout:
            fout.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))
        torch.save((encoder, decoder), os.path.join(log_dir, "model.pt"))
        scheduler.step(metrics["train.loss"])

class Logger(object):
    def __init__(self, stream=sys.stdout):
        output_dir = os.path.join(log_dir, "draft_logpool/")  # folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        log_name = log_name_time + ".txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == "__main__":
    sys.stdout = Logger(sys.stdout) #将输出记录到log
    sys.stderr = Logger(sys.stderr) #将错误信息记录到log
    parser = argparse.ArgumentParser()

    parser.add_argument('--recover',type=int, default=0)
    parser.add_argument('--model_path',type=str,default='/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/results/2024-05-22-15-16-55/model.pt')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--linf_max', type=float, default=0.016)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--dataset', type=str, default="../data/Kinetics-600")
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--data_dim', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--multiplicity', type=int, default=1)

    run(parser.parse_args())

