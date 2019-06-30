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