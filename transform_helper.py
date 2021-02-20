import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps


class PadWhite(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, tuple):
            self.height, self.width = size
        elif isinstance(size, int):
            self.height = self.width = size

    def __call__(self, img):
        if img.size[0] > self.width or img.size[1] > self.height:
            img.thumbnail((self.width, self.height))
        delta_width = self.width - img.size[0]
        delta_height = self.height - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width -
                   pad_width, delta_height-pad_height)
        return ImageOps.expand(img, padding, fill=255)


class AddGaussianNoice(object):
    def __init__(self, std=5, mean=0, is_stochastic=False):
        self.std = std
        self.mean = mean
        self.is_stochastic = is_stochastic

    def __call__(self, image):
        if self.is_stochastic:
            r_std = torch.randint(low=0, high=self.std+1, size=(1,)).item()/100
        else:
            r_std = self.std/100
        noise = torch.normal(self.mean, r_std, image.shape)
        out_img = image + noise
        out_img.data.clamp_(0, 1)
        return out_img
