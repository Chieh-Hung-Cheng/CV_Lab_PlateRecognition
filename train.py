import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.transforms import functional as tf

# For reproducibility
# Set before loading model and dataset UNCHANGED
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Load data UNCHANGED
train_set = CCPD5000('./ccpd5000/train/')
valid_set = CCPD5000('./ccpd5000/valid/')
visul_set = ConcatDataset([
    Subset(train_set, random.sample(range(len(train_set)), 32)),
    Subset(valid_set, random.sample(range(len(valid_set)), 32)),
])
train_loader = DataLoader(train_set, 32, shuffle=True, num_workers=3)
valid_loader = DataLoader(valid_set, 32, shuffle=False, num_workers=1)
visul_loader = DataLoader(visul_set, 32, shuffle=False, num_workers=1)

device = 'cuda'
model = CCPDRegressor()
model = model.to(device)
criterion = nn.MSELoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
#device = 'cuda'
#model = CCPDRegressor().to(device)
#criterion = nn.MSELoss().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Log record UNCHANGED
log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)
print(log_dir)
history = {
    'train_mae': [],
    'valid_mae': [],
    'train_mse': [],
    'valid_mse': [],
}


# train
def train(pbar):
    model.train()  # train mode
    mae_steps = []
    mse_steps = []

    for image, ground_truth in iter(train_loader):
        image = image.to(device)
        ground_truth = ground_truth.to(device)

        optimizer.zero_grad()
        predict = model(image)
        loss = criterion(predict, ground_truth)
        loss.backward()
        optimizer.step()

        mae = F.l1_loss(predict, ground_truth).item()
        mse = F.mse_loss(predict, ground_truth).item()

        # UNCHANGED
        mae_steps.append(mae)
        mse_steps.append(mse)

        pbar.set_postfix(mae=mae, mse=mse)
        pbar.update(image.size(0))
    avg_mae = sum(mae_steps) / len(mae_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['train_mae'].append(avg_mae)
    history['train_mse'].append(avg_mse)

'''def train(pbar):
    model.train()
    mae_steps = []
    mse_steps = []

    for img_b, kpt_b in iter(train_loader):
        img_b = img_b.to(device)
        kpt_b = kpt_b.to(device)

        optimizer.zero_grad()
        pred_b = model(img_b)
        loss = criterion(pred_b, kpt_b)
        loss.backward()
        optimizer.step()

        mae = loss.detach().item()
        mse = F.mse_loss(pred_b.detach(), kpt_b.detach()).item()
        mae_steps.append(mae)
        mse_steps.append(mse)

        pbar.set_postfix(mae=mae, mse=mse)
        pbar.update(img_b.size(0))

    avg_mae = sum(mae_steps) / len(mae_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['train_mae'].append(avg_mae)
    history['train_mse'].append(avg_mse)
'''

def valid(pbar):
    model.eval()  # evaluation mode
    mae_steps = []
    mse_steps = []

    for image, ground_truth in iter(valid_loader):
        image = image.to(device)
        ground_truth = ground_truth.to(device)
        predict = model(image)
        loss = criterion(predict, ground_truth)

        mae = F.l1_loss(predict, ground_truth).item()
        mse = F.mse_loss(predict, ground_truth).item()
        # UNCHANGED
        mae_steps.append(mae)
        mse_steps.append(mse)

        pbar.set_postfix(mae=mae, mse=mse)
        pbar.update(image.size(0))

    avg_mae = sum(mae_steps) / len(mae_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['valid_mae'].append(avg_mae)
    history['valid_mse'].append(avg_mse)
'''
def valid(pbar):
    model.eval()
    mae_steps = []
    mse_steps = []

    for img_b, kpt_b in iter(valid_loader):
        img_b = img_b.to(device)
        kpt_b = kpt_b.to(device)
        pred_b = model(img_b)
        loss = criterion(pred_b, kpt_b)
        mae = loss.detach().item()

        mse = F.mse_loss(pred_b.detach(), kpt_b.detach()).item()
        mae_steps.append(mae)
        mse_steps.append(mse)

        pbar.set_postfix(mae=mae, mse=mse)
        pbar.update(img_b.size(0))

    avg_mae = sum(mae_steps) / len(mae_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['valid_mae'].append(avg_mae)
    history['valid_mse'].append(avg_mse)
'''

# Visualization UNCHANGED
def visul(pbar, epoch):
    model.eval()
    epoch_dir = log_dir / f'{epoch:03d}'
    epoch_dir.mkdir()
    for img_b, kpt_b in iter(visul_loader):
        pred_b = model(img_b.to(device)).cpu()
        for img, pred_kpt, true_kpt in zip(img_b, pred_b, kpt_b):
            img = tf.to_pil_image(img)
            vis = draw_plate(img, pred_kpt)
            vis = draw_kpts(vis, true_kpt, c='orange')
            vis = draw_kpts(vis, pred_kpt, c='red')
            vis.save(epoch_dir / f'{pbar.n:03d}.jpg')
            pbar.update()

# log record UNCHANGED
def log(epoch):
    with (log_dir / 'metrics.json').open('w') as f:
        json.dump(history, f)

    fig, ax = plt.subplots(2, 1, figsize=(6, 6), dpi=100)
    ax[0].set_title('MAE')
    ax[0].plot(range(epoch + 1), history['train_mae'], label='Train')
    ax[0].plot(range(epoch + 1), history['valid_mae'], label='Valid')
    ax[0].legend()
    ax[1].set_title('MSE')
    ax[1].plot(range(epoch + 1), history['train_mse'], label='Train')
    ax[1].plot(range(epoch + 1), history['valid_mse'], label='Valid')
    ax[1].legend()
    fig.savefig(str(log_dir / 'metrics.jpg'))
    plt.close()


# train epoch setting UNCHANGED
for epoch in range(30):
    print('Epoch', epoch, flush=True)
    with tqdm(total=len(train_set), desc='  Train') as pbar:
        train(pbar)

    with torch.no_grad():
        with tqdm(total=len(valid_set), desc='  Valid') as pbar:
            valid(pbar)
        with tqdm(total=len(visul_set), desc='  Visul') as pbar:
            visul(pbar, epoch)
        log(epoch)