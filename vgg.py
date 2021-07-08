import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# VGG16 architecture
VGG_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# NETWORK SHAPES (b- batch size)
# input:            b x 1 x 1499
# first conv seq:   b x 64 x 1499
# first MaxPool:    b x 64 x 749
# second conv seq:  b x 128 x 749
# second MaxPool:   b x 128 x 374
# third conv seq:   b x 256 x 374
# third MaxPool:    b x 256 x 187
# fourth conv seq:  b x 512 x 187
# fourth MaxPool:   b x 512 x 93
# fifth conv seq:   b x 512 x 93
# fifth MaxPool:    b x 512 x 46
# flatten:          b x 23552
# first Linear:     b x 4096
# second Linear:    b x 4096
# third Linear:     b x 16

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_16)
        self.fc = nn.Sequential(
            nn.Linear(512*46, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def create_conv_layers(self, net_arch):
        layers = []
        in_channels = self.in_channels

        for x in net_arch:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm1d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG_net(in_channels=1, num_classes=16).to(device)
x = torch.randn(10, 1, 1499).to(device)
print(model(x).shape)