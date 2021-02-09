import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import tesserocr
import numpy as np
import random
from utils import get_files
import properties as properties


class ImgDataset(Dataset):

    def __init__(self, data_dir, transform=None, include_name=False):
        self.transform = transform
        self.include_name = include_name
        self.files = []
        unprocessed = get_files(data_dir, ['png', 'jpg'])
        for img in unprocessed:
            if len(os.path.basename(img).split('_')[1]) <= properties.max_char_len:
                self.files.append(img)
        # with open('../sim_model/google_test_samples.txt', 'r') as file:
        #     self.files = [line.strip() for line in file]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name).convert("L")
        if self.transform != None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        file_name = os.path.basename(img_name)
        label = file_name.split('_')[1]
        if self.include_name:
            sample = (image, label, file_name)
        else:
            sample = (image, label)
        return sample

    def worker_init(self, pid):
        return np.random.seed(torch.initial_seed() % (2**32 - 1))
