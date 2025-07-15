"""
Testing script for GUNet model evaluation.
"""

from __future__ import print_function, division
import os
import argparse
import sys
import subprocess
import time
import copy
import random
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from skimage import io, transform

from equivariant_lib.models.G_UNet_model import G_RadioWNet as G_UNet
from equivariant_lib.models.train import train_model
from lib import loaders, modules, func

warnings.filterwarnings("ignore")

# Set default tensor types for CUDA
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def calc_loss_test(pred, target, metrics):
    """
    Given one UNet output, compute MSE ("RMSE" in your original sense)
    and NMSE, and accumulate sum(loss*batch_size) into metrics.
    """
    criterion = nn.MSELoss()
    mse = criterion(pred, target)
    # "RMSE" step 
    rmse = torch.sqrt(mse)
    # "NMSE" step: divide by MSE(target, 0)
    denom = criterion(target, torch.zeros_like(target))
    nmse = mse / denom

    bs = target.size(0)
    metrics['RMSE U'] += rmse.item() * bs
    metrics['NMSE U'] += nmse.item() * bs

    return rmse, nmse


def print_metrics_test(metrics, epoch_samples):
    """
    Print average RMSE and NMSE over the test set.
    """
    avg_rmse  = metrics['RMSE U']  / epoch_samples
    avg_nmse = metrics['NMSE U'] / epoch_samples
    print(f"Test RMSE: {avg_rmse:.6f}")
    print(f"Test NMSE: {avg_nmse:.6f}")


def test_loss(model, dataset="coarse"):
    """
    Run one evaluation pass over Radio_test, reporting both RMSE and NMSE.
    dataset: "coarse" (yields inputs, targets) or
             "fine"   (yields inputs, targets, samples — samples are ignored).
    """
    since = time.time()
    model.eval()
    metrics = defaultdict(float)
    epoch_samples = 0

    generator = torch.Generator(device=device)
    loader = DataLoader(Radio_test,
                        batch_size=15,
                        generator=generator,
                        shuffle=True,
                        num_workers=16)

    with torch.no_grad():
        if dataset == "coarse":
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs)
                calc_loss_test(out, targets, metrics)
                epoch_samples += inputs.size(0)

        elif dataset == "fine":
            for inputs, targets, samples in loader:  # ignore samples
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs)
                calc_loss_test(out, targets, metrics)
                epoch_samples += inputs.size(0)

    print_metrics_test(metrics, epoch_samples)
    elapsed_time = time.time() - since
    print(f"Test time: {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s")

def create_parser():
    parser = argparse.ArgumentParser(description="Test GUNet model on RadioMapSeer")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to RadioMapSeer dataset')
    parser.add_argument('--experiment_type', type=str, choices=['DPM_cars', 'DPM_no_car', 'IRT_cars', 'IRT_no_car'], required=True, help='Experiment type')
    parser.add_argument('--symmetry_group', type=str, choices=['C2', 'D2', 'C4', 'D4', 'C8', 'D8'], default='D8', help='Equivariance symmetry group')
    return parser

def test(args):
    model_path = args.model_path
    dataset_path = args.dataset_path

    log_filename = os.path.join(model_path, "test_loss.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    sys.stdout = func.Logger(log_filename)

    global device, Radio_test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configure loader arguments based on experiment_type
    loader_kwargs = {
        "phase": "test",
        "dir_dataset": dataset_path,
        "thresh": 0.2
    }
    if 'cars' in args.experiment_type:
        loader_kwargs["carsSimul"] = "yes"
        loader_kwargs["carsInput"] = "yes"
    if args.experiment_type.startswith("IRT"):
        loader_kwargs["simulation"] = "IRT2"

    Radio_test = loaders.RadioUNet_c(**loader_kwargs)

    # Configure model
    in_channels = 3 if 'cars' in args.experiment_type else 2
    config = {
        'in_channels': in_channels,
        'out_channels': 1,
        'channels': [6, 50, 100, 100, 170],
        'image_size': 256,
        'n_conv': 2,
        'batch_norm': True,
        'dropout': 0,
        'kernel_size': 3,
        'padding': 1,
        'equvariant_mask': False,
        'group': args.symmetry_group
    }

    # Initialize and load model
    model = G_UNet(config, model_path=model_path).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pt')))
    print('Model weights loaded successfully!')

    # Run test evaluation
    test_loss(model, dataset="coarse")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    test(args)

