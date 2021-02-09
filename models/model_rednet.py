from torch import nn
import torch.nn.functional as fn


class REDNet(nn.Module):
    def __init__(self, num_layers=5, num_features=64):
        super(REDNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_1 = x
        x = fn.relu(self.conv1(x))
        x_2 = x
        x = fn.relu(self.conv2(x))
        x_3 = x
        x = fn.relu(self.conv3(x))
        x = fn.relu(self.conv4(x))
        x = fn.relu(self.conv5(x))

        x = fn.relu(self.deconv1(x))
        x = fn.relu(self.deconv2(x))
        x = self.deconv3(x)
        x = fn.relu(x + x_3)
        x = self.deconv4(x)
        x = fn.relu(x + x_2)
        x = self.deconv5(x)
        x = fn.relu(x + x_1)
        return x