"""
Training script for GUNet model on RadioMapSeer.
"""

from __future__ import print_function, division
import os
import sys
import subprocess
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


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    # Configuration
    set_seed(516)
    batch_size = 15
    num_epochs = 50
    lr = 1e-4
    schedule_step_size = 30
    
    # Setup paths
    model_path = 'path/to/model/'
    dataset_path = 'path/to/RadioMapSeer/'
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Setup logging
    log_filename = os.path.join(model_path, "training_log.txt")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    sys.stdout = func.Logger(log_filename)
    
    # Display GPU info
    output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(output.stdout)
    print(f'Dataset path: {dataset_path}')
    
    # Load datasets - choose one option:
    # Option 1: DPM no Cars (comment out to use other options)
    Radio_train = loaders.RadioUNet_c(phase="train", dir_dataset=dataset_path, thresh=0.2)
    Radio_val = loaders.RadioUNet_c(phase="val", dir_dataset=dataset_path, thresh=0.2)
    Radio_test = loaders.RadioUNet_c(phase="test", dir_dataset=dataset_path, thresh=0.2)
    
    # # Option 2: IRT2 no Cars (comment out to use other options)
    # Radio_train = loaders.RadioUNet_c(phase="train", dir_dataset=dataset_path, thresh=0.2, simulation="IRT2")
    # Radio_val = loaders.RadioUNet_c(phase="val", dir_dataset=dataset_path, thresh=0.2, simulation="IRT2")
    # Radio_test = loaders.RadioUNet_c(phase="test", dir_dataset=dataset_path, thresh=0.2, simulation="IRT2")

    # # Option 3: DPM with Cars (comment out to use other options)
    # Radio_train = loaders.RadioUNet_c(phase="train",dir_dataset=dataset_path, thresh=0.2, carsSimul="yes", carsInput="yes")
    # Radio_val = loaders.RadioUNet_c(phase="val",dir_dataset=dataset_path, thresh=0.2, carsSimul="yes", carsInput="yes")
    # Radio_test = loaders.RadioUNet_c(phase="test",dir_dataset=dataset_path, thresh=0.2, carsSimul="yes", carsInput="yes")

    # # Option 4: IRT2 with Cars (comment out to use other options)
    # Radio_train = loaders.RadioUNet_c(phase="train",dir_dataset=dataset_path, thresh=0.2, carsSimul="yes", carsInput="yes", simulation="IRT2")
    # Radio_val = loaders.RadioUNet_c(phase="val",dir_dataset=dataset_path, thresh=0.2, carsSimul="yes", carsInput="yes", simulation="IRT2")
    # Radio_test = loaders.RadioUNet_c(phase="test",dir_dataset=dataset_path, thresh=0.2, carsSimul="yes", carsInput="yes", simulation="IRT2")
    
    image_datasets = {'train': Radio_train, 'val': Radio_val}
    
    # Setup data loaders
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator(device=device)
    
    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, generator=generator, 
                          shuffle=False, num_workers=32),
        'val': DataLoader(Radio_val, batch_size=batch_size, generator=generator, 
                        shuffle=True, num_workers=32)
    }
    
    # Model configuration
    config = {
        'in_channels': 2,
        'out_channels': 1,
        'channels': [6, 50, 100, 100, 170],
        'image_size': 256,
        'n_conv': 2,
        'batch_norm': True,
        'dropout': 0,
        'kernel_size': 3,
        'padding': 1,
        'equvariant_mask': False,
        'group': "D8"
    }
    
    # Initialize model
    model = G_UNet(config, model_path=model_path).to(device)
    pytorch_total_params = sum([np.prod(p.shape) for p in model.parameters()])
    print(f"Total parameters: {pytorch_total_params}")
    
    # Setup optimizer and scheduler
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=schedule_step_size, gamma=0.1)
    
    # Train model
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
    
    # Save loss history
    loss_df = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss})
    loss_df.to_csv(os.path.join(model_path, 'loss.csv'), index=False)
    print('Training completed!')


if __name__ == "__main__":
    main()
