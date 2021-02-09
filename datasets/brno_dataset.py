import os
import torch
from torch.utils.tensorboard.summary import image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import random
from utils import get_ununicode
import properties as properties


class BrnoDataset(Dataset):

    def __init__(self, data_list, image_dir, transform=None, include_name=False, eval=True):
        self.transform = transform
        self.include_name = include_name
        self.image_dir = image_dir
        self.eval = eval
        self.files = []
        with open(data_list, 'r') as file:
            for line in file:
                line = line.strip().replace(' ', ':', 1)
                line = line.split(':')
                if len(line[1]) <= properties.max_char_len_brno:
                    self.files.append((line[0], line[1]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name, label = self.files[idx]
        if not self.eval:
            label = get_ununicode(label)
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("L")
        if self.transform != None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        file_name = os.path.basename(img_name)
        if self.include_name:
            sample = (image, label, file_name)
        else:
            sample = (image, label)
        return sample

    def worker_init(self, pid):
        return np.random.seed(torch.initial_seed() % (2**32 - 1))
