"""
Training script for GUNet model on RadioMapSeer.
"""

from __future__ import print_function, division
import os
import sys
import subprocess
import random
import warnings
import argparse
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


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_parser():
    parser = argparse.ArgumentParser(description="Train GUNet on RadioMapSeer")
    parser.add_argument('--model_path', type=str, required=True, help='Path to save model and logs')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to RadioMapSeer dataset')
    parser.add_argument('--experiment_type', type=str, choices=['DPM_cars', 'DPM_no_car', 'IRT_cars', 'IRT_no_car'], required=True, help='Experiment type')
    parser.add_argument('--symmetry_group', type=str, choices=['C2', 'D2', 'C4', 'D4', 'C8', 'D8'], default='D8', help='Equivariance symmetry group')
    return parser


def train(args):
    set_seed(516)
    batch_size = 15
    num_epochs = 50
    lr = 1e-4
    schedule_step_size = 30

    model_path = args.model_path
    dataset_path = args.dataset_path

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    log_filename = os.path.join(model_path, "training_log.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    sys.stdout = func.Logger(log_filename)

    output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(output.stdout)
    print(f'Dataset path: {dataset_path}')

    # Configure loader arguments based on experiment_type
    loader_kwargs = {
        "phase": None,
        "dir_dataset": dataset_path,
        "thresh": 0.2
    }
    if 'cars' in args.experiment_type:
        loader_kwargs["carsSimul"] = "yes"
        loader_kwargs["carsInput"] = "yes"
    if args.experiment_type.startswith("IRT"):
        loader_kwargs["simulation"] = "IRT2"

    def make_loader(phase):
        kw = loader_kwargs.copy()
        kw["phase"] = phase
        return loaders.RadioUNet_c(**kw)
    Radio_train = make_loader("train")
    Radio_val = make_loader("val")
    Radio_test = make_loader("test")

    image_datasets = {'train': Radio_train, 'val': Radio_val}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator(device=device)

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, generator=generator, shuffle=False, num_workers=32),
        'val': DataLoader(Radio_val, batch_size=batch_size, generator=generator, shuffle=True, num_workers=32)
    }

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

    model = G_UNet(config, model_path=model_path).to(device)
    pytorch_total_params = sum([np.prod(p.shape) for p in model.parameters()])
    print(f"Total parameters: {pytorch_total_params}")

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=schedule_step_size, gamma=0.1)

    print('Starting training...')
    train_loss, val_loss = train_model(
        model=model,
        optimizer=optimizer_ft,
        dataloaders=dataloaders,
        scheduler=exp_lr_scheduler,
        num_epochs=num_epochs,
        device=device,
        model_path=model_path
    )

    loss_df = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss})
    loss_df.to_csv(os.path.join(model_path, 'loss.csv'), index=False)
    print('Training completed!')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    train(args)

