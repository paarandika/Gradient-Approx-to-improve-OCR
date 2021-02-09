import torch
import tesserocr
import numpy as np
from torchvision.transforms import ToPILImage
import easyocr

import properties as properties
import utils


class TessHelper():
    def __init__(self, empty_char=properties.empty_char, is_eval=False):
        self.empty_char = empty_char
        self.is_eval = is_eval
        self.api_single_line = tesserocr.PyTessBaseAPI(
            lang='eng', psm=tesserocr.PSM.SINGLE_LINE, path=properties.tesseract_path, oem=tesserocr.OEM.LSTM_ONLY)
        self.api_single_block = tesserocr.PyTessBaseAPI(
            lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK, path=properties.tesseract_path)

    def get_labels(self, imgs):
        labels = []
        for i in range(imgs.shape[0]):
            img = ToPILImage()(imgs[i])
            self.api_single_line.SetImage(img)
            label = self.api_single_line.GetUTF8Text().strip()
            if label == "":
                label = self.empty_char
            if self.is_eval:
                labels.append(label)
                continue

            label = utils.get_ununicode(label)
            if len(label) > properties.max_char_len:
                label = self.empty_char
            labels.append(label)
        return labels

    def get_string(self, img):
        img = ToPILImage()(img)
        self.api_single_block.SetImage(img)
        string = self.api_single_block.GetUTF8Text().strip()
        string = utils.get_ununicode(string)
        return string.split()
