import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import os
import time

def calc_loss_dense(pred, target, metrics, criterion):
    loss = criterion(pred, target)
    metrics['loss'] += loss.item() * target.size(0)
    return loss

# def calc_loss_sparse(pred, target, samples, metrics, num_samples):
#     criterion = nn.MSELoss()
#     loss = criterion(samples*pred, samples*target)*(256**2)/num_samples
#     metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

#     return loss

def print_metrics(metrics, epoch_samples, phase):
    stats = ", ".join(f"{k}: {metrics[k]/epoch_samples:4f}" for k in metrics)
    print(f"{phase}: {stats}")

def train_model(model, dataloaders, optimizer, scheduler, device,
                num_epochs=50, model_path='./data/'):
    criterion = nn.MSELoss()
    best_loss = float('inf')
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, targets in dataloaders[phase]:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)

                loss = calc_loss_dense(outputs, targets, metrics, criterion)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                train_loss_list.append(epoch_loss)
            else:
                val_loss_list.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    print("saving best model")
                    torch.save(model.state_dict(),
                               os.path.join(model_path, 'best_model.pt'))

        elapsed = time.time() - since
        print(f'{elapsed//60:.0f}m {elapsed%60:.0f}s')

    print(f'Best val loss: {best_loss:4f}')
    return train_loss_list, val_loss_list
