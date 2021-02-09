import torch
from torch import nn

class SRGAN(nn.Module):
    def __init__(self):

        super(SRGAN, self).__init__()
        self.block1 = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(1, 64, kernel_size=9),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64)
        )
        self.block8 = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(64, 1, kernel_size=9)
        )


    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        return (torch.tanh(block8) + 1) / 2

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pad1 = nn.ReplicationPad2d(1)

    def forward(self, x):
        residual = self.pad1(x)
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.pad1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual