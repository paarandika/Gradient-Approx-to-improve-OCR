import numpy as np
import tesserocr
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import properties as properties
from utils import get_ununicode, get_files, get_noisy_image


class OCRDataset(Dataset):

    def __init__(self, data_dir, ocr_helper, transform=None, include_name=False):
        self.include_name = include_name
        self.transform = transform
        self.ocr_helper = ocr_helper
        self.files = []

        unprocessed = get_files(data_dir, ['png', 'jpg'])
        for img in unprocessed:
            if len(os.path.basename(img).split('_')[1]) <= properties.max_char_len:
                self.files.append(img)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(img_name).convert("L")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        file_name = os.path.basename(img_name)
        label = file_name.split('_')[1]
        ocr_label = self.ocr_helper.get_labels(image)
        if self.include_name:
            sample = (image, ocr_label[0], img_name)
        else:
            sample = (image, ocr_label[0])
        return sample
