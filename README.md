# CV-lab-project-1
## Path.py
----------------
- 使用自己的方式取出圖片的 ground truth <br>
  逐一拆解 "_" "&" "-" (利用join spilt處理字串)
- 此為之後讀資料的先行嘗試 <br>
- 註解者為範例code與notes
```python
from pathlib import Path

img_dir = Path('./ccpd5000/train/')
img_paths_all = img_dir.glob('*.jpg')
img_paths_list = list(img_paths_all)
img_paths = sorted(img_paths_list) #sorted image paths
#img_paths = img_dir.glob('*.jpg')
#img_paths = sorted(list(img_paths))


print('data size: '+str(len(img_paths)))
print('sample image name: ' + str(img_paths[8]))
#print(len(img_paths))
#print(img_paths[:5])

testname = img_paths[8].name
print(testname)

split_component = testname.split('-')
information = split_component[3]
## = name.split('-')[3]
##print(token)

information_split1 = "&".join(information.split('_'))
information_split2 = information_split1.split('&')
#token = token.replace('&', '_')
#print(token)

#values = token.split('_')
#print(values)

values = list()
for i in range(len(information_split2)):
    values.append(float(information_split2[i]))
print(values)

#values = [float(val) for val in values]
#print(values)

```
## util.py
- 繪圖與轉換的程式碼 <br> 有點高深似乎無從改起(保持與範例原樣)
```python
# UNCHANGED

import warnings

import torch
import numpy as np
from PIL import Image, ImageDraw
from skimage import util
from skimage.transform import ProjectiveTransform, warp

def draw_kpts(img, kpts, c='red', r=2.0):
    '''Draw keypoints on image.
    Args:
        img: (PIL.Image) will be modified
        kpts: (FloatTensor) keypoints in xy format, sized [8,]
        c: (PIL.Color) color of keypoints, default to 'red'
        r: (float) radius of keypoints, default to 2.0
    Return:
        img: (PIL.Image) modified image
    '''
    draw = ImageDraw.Draw(img)
    kpts = kpts.view(4, 2)
    kpts = kpts * torch.FloatTensor(img.size)
    kpts = kpts.numpy().tolist()
    for (x, y) in kpts:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
    return img


def draw_plate(img, kpts):
    '''Perspective tranform and draw the plate indicated by kpts to a 96x30 rectangle.
    Args:
        img: (PIL.Image) will be modified
        kpts: (FloatTensor) keypoints in xy format, sized [8,]
    Return:
        img: (PIL.Image) modified image
    Reference: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_geometric.html
    '''
    src = np.float32([[96, 30], [0, 30], [0, 0], [96, 0]])
    dst = kpts.view(4, 2).numpy()
    dst = dst * np.float32(img.size)

    transform = ProjectiveTransform()
    transform.estimate(src, dst)
    with warnings.catch_warnings(): # surpress skimage warning
        warnings.simplefilter("ignore")
        warped = warp(np.array(img), transform, output_shape=(30, 96))
        warped = util.img_as_ubyte(warped)
    plate = Image.fromarray(warped)
    img.paste(plate)
    return img
```

## data.py
- 與 path.py 之延伸，用以讀出data <br>
- 覆寫 __init__() __len__() __getitem__()函數
  - 以 for 迴圈將 grond truth 的長寬 normalize
  - 讀出 image 那段似乎是特定函數用法，保持不動

```python
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf



class CCPD5000:
    def __init__(self, img_dir):  # use CCPD5000(path) initiate
        self.img_dir = Path(img_dir)

        self.img_paths_all = self.img_dir.glob("*.jpg")
        self.img_paths_list = list(self.img_paths_all)
        self.img_paths = sorted(self.img_paths_list)  # sorted image path

    #    def __init__(self, img_dir):
    #        self.img_dir = Path(img_dir)
    #        self.img_paths = self.img_dir.glob('*.jpg')
    #        self.img_paths = sorted(list(self.img_paths))

    def __len__(self):  # return size
        return len(self.img_paths)

#    def __len__(self):
#        return len(self.img_paths)

    def __getitem__(self, index):  # call with CCPD5000((int)index) return [img , [8, ](tensor)ground_truth]
        target_img_path = self.img_paths[index]
        target_img = Image.open(target_img_path)
        (W, H) = target_img.size

        # get [8, ] ground truth
        img_name = target_img_path.name
        spilt_component = img_name.split("-")
        information = spilt_component[3]
        information_split1 = "&".join(information.split("_"))
        information_split2 = information_split1.split("&")

        ground_truth = list()
        for i in range(len(information_split2)):
            ground_truth.append(float(information_split2[i]))

        for idx in range(len(ground_truth)):
            if(idx%2==0):
                ground_truth[idx] = ground_truth[idx]/W
            else:
                ground_truth[idx] = ground_truth[idx]/H

        ground_truth = torch.tensor(ground_truth)

        # adjust image UNCHANGED
        target_img = target_img.convert("RGB")
        target_img = target_img.resize((192, 320))
        target_img = tf.to_tensor(target_img)

        return (target_img, ground_truth)



#    def __getitem__(self, idx):
#        img_path = self.img_paths[idx]

# load image
#        img = Image.open(img_path)
#        W, H = img.size
#        img = img.convert('RGB')
#        img = img.resize((192, 320))
#        img = tf.to_tensor(img)

# parse annotation
#        name = img_path.name
#        token = name.split('-')[3]
#        token = token.replace('&', '_')
#        kpt = [float(val) for val in token.split('_')]
#        kpt = torch.tensor(kpt)  # [8,]
#        kpt = kpt.view(4, 2)  # [4, 2]
#        kpt = kpt / torch.FloatTensor([W, H])
#        kpt = kpt.view(-1)  # [8,]

#        return img, kpt


train_set = CCPD5000('./ccpd5000/train')
print(len(train_set))

img, kpt = train_set[90]
print(img.size())
print(kpt.size())

print(kpt)

# image display for colab (disable for pycharm)

#img = tf.to_pil_image(img)
#vis = draw_kpts(img, kpt, c='orange')
#vis = draw_plate(vis, kpt)
#vis.save('./check.jpg')

#from IPython import display
#display.Image('./check.jpg')
```

## convolution_block.py
- 由於對 nn.Sequential() 用法不甚熟悉，所以用比較熟的 net() 用法慢慢架
- 重新設計 forward() 方法
  - convolution2D ->batch_norm2d -> leakyRelu -> maxPooling2d 為一週期
    - 4 cycles
    - 每次 maxpooling 的長寬更小(比起原先8*8)
  - 擬合函數
    - 三層 linear function 讓 tensor size reduce from 128 -> 8
```python
import torch
from torch import nn
from torch.nn import functional as F

class CCPDRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(4, 8, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(8, 16, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(64, 64, (3, 3), padding=1)

        self.batchN2_1 = nn.BatchNorm2d(4)
        self.batchN2_2 = nn.BatchNorm2d(8)
        self.batchN2_3 = nn.BatchNorm2d(16)
        self.batchN2_4 = nn.BatchNorm2d(32)
        self.batchN2_5 = nn.BatchNorm2d(64)
        self.batchN2_6 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

        self.leakyR = nn.LeakyReLU()
        self.sigmo = nn.Sigmoid()

    def forward(self, x):
        N = x.size(0)

        x = self.leakyR(self.batchN2_1(self.conv1(x)))
        x = F.max_pool2d(x, (4, 4))
        x = self.leakyR(self.batchN2_2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.leakyR(self.batchN2_3(self.conv3(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.leakyR(self.batchN2_4(self.conv4(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.leakyR(self.batchN2_5(self.conv5(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.leakyR(self.batchN2_6(self.conv6(x)))
        x = F.max_pool2d(x, (2, 2))

        #print(x.size())
        x = x.view(N, -1)

        x = self.leakyR(self.fc1(x))
        x = self.leakyR(self.fc2(x))
        x = self.sigmo(self.fc3(x))

        #print(x.size())
        return x



'''class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(cout, cout, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act1(self.bn2(self.conv2(x)))
        return x


class CCPDRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            nn.MaxPool2d((4, 4)),
            ConvBlock(16, 32),
            nn.MaxPool2d((4, 4)),
            ConvBlock(32, 64),
            nn.MaxPool2d(4, 4),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = x.view(N, -1)  # i.e. Flatten
        x = self.regressor(x)
        return x

'''
# Check UNCHANGED
device = 'cuda'
model = CCPDRegressor().to(device)
img_b = torch.rand(16, 3, 192, 320).to(device)
out_b = model(img_b)
print(out_b.size())  # expected [16, 8]
```
## train.py
  - 似乎許多都是固定架構與函式用法，重新寫了一次當練習(當作熟悉架構(?)
  - loss 改成以 MSE 評估
  - learning rate 改成2e-4(原先的兩倍(不然實測有些慢))
  - epoch 數改為(30)
    - train 與 vaild 的誤差(mae,mse)皆有持續下降趨勢(epoch = 22 後趨於不明顯)
    - 最低值恰好在 epoch = 29 時
      - TRAIN: avg_mae=0.00791, avg_mse=0.00011
      - VAILD: avg_mae=0.01147, avg_mse=0.00029
  - 顯示進度條與顯示圖片的code皆不動
```python
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
```

## RESULT

