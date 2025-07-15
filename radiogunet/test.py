"""
Testing script for GUNet model evaluation.
"""

from __future__ import print_function, division
import os
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


def main():
    # Setup paths and logging
    model_path = 'path/to/model/'
    dataset_path = 'path/to/RadioMapSeer/'
    
    log_filename = os.path.join(model_path, "test_loss.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    sys.stdout = func.Logger(log_filename)
    
    global device, Radio_test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load test dataset - choose one option:
    # Option 1: DPM no Cars (comment out to use other options)
    Radio_test = loaders.RadioUNet_c(phase="test", dir_dataset=dataset_path, thresh=0.2)
    
    # # Option 2: IRT2 no Cars (comment out to use other options)
    # Radio_test = loaders.RadioUNet_c(phase="test", dir_dataset=dataset_path, thresh=0.2, simulation="IRT2")

    # # Option 3: DPM with Cars (comment out to use other options)
    # Radio_test = loaders.RadioUNet_c(phase="test",dir_dataset=dataset_path, thresh=0.2, carsSimul="yes", carsInput="yes")

    # # Option 4: IRT2 with Cars (comment out to use other options)
    # Radio_test = loaders.RadioUNet_c(phase="test",dir_dataset=dataset_path, thresh=0.2, carsSimul="yes", carsInput="yes", simulation="IRT2")
    
    # Model configuration
    config = {
        'in_channels': 3,           # Use 2 for standard/IRT2, 3 for car simulation
        'out_channels': 1,
        'channels': [6, 50, 100, 100, 170],
        'image_size': 256,
        'n_conv': 2,
        'batch_norm': True,
        'dropout': 0,
        'kernel_size': 3,
        'padding': 1,
        'equvariant_mask': False,
        'group': "D2"               # Rotation group for equivariance
    }
    
    # Initialize and load model
    model = G_UNet(config, model_path=model_path).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pt')))
    print('Model weights loaded successfully!')
    
    # Run test evaluation
    test_loss(model, dataset="coarse")


if __name__ == "__main__":
    main()
