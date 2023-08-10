import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class conv_block(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)         
        self.relu = nn.ReLU()     
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, smaller=False):
        super().__init__()
        self.smaller = smaller
        self.e1 = conv_block(1, 64)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        if not smaller:
            self.e4 = conv_block(256, 512)     
            self.b = conv_block(512, 1024)
        else:
            self.b = conv_block(256, 512)    
        
        if not smaller:
            self.ct1 = nn.ConvTranspose2d(1024, 512, 2, 2)
            self.d1 = conv_block(1024, 512)
        self.ct2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d2 = conv_block(512, 256)
        self.ct3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = conv_block(256, 128)
        self.ct4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d4 = conv_block(128, 64)
        self.last1 = torch.nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.last2 = torch.nn.Tanh()

    def forward(self, inputs):
        p1 = self.e1(inputs)
        p2 = self.e2(self.mp(p1))
        p3 = self.e3(self.mp(p2))
        if not self.smaller:
            p4 = self.e4(self.mp(p3))         
            b = self.b(self.mp(p4))
            ct1 = self.ct1(b)
            d1 = self.d1(torch.cat([ct1, p4], 1))
        else:
            b = self.b(self.mp(p3))
            d1 = b

        ct2 = self.ct2(d1)
        d2 = self.d2(torch.cat([ct2, p3], 1))
        ct3 = self.ct3(d2)
        d3 = self.d3(torch.cat([ct3, p2], 1))
        ct4 = self.ct4(d3)
        d4 = self.d4(torch.cat([ct4, p1], 1))
        outputs = self.last1(d4)         
        outputs = self.last2(outputs)
        return outputs