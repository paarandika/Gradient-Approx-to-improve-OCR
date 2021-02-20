import torch
import tesserocr
import numpy as np
from torchvision.transforms import ToPILImage
import easyocr

import properties as properties
import utils


class EocrHelper():
    def __init__(self, empty_char=properties.empty_char, is_eval=False):
        self.empty_char = empty_char
        self.is_eval = is_eval
        # this works only becuase of a hack done in library. If fails remove gpu='cuda:1'
        # self.reader = easyocr.Reader(['en'], gpu='cuda:1')
        self.reader = easyocr.Reader(['en'], gpu=True)

    def get_labels(self, imgs):
        labels = []
        for i in range(imgs.shape[0]):
            img = ToPILImage()(imgs[i])
            img = np.asarray(img)
            label = self.reader.readtext(
                img, detail=0, width_ths=35.0, height_ths=35.0, ycenter_ths=35.0, paragraph=True)
            if len(label) == 0:
                label = ""
            else:
                label = label[0]
            
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
        img = np.asarray(img)
        string = self.reader.readtext(img, detail=0)
        for i in range(len(string)):
            string[i] = utils.get_ununicode(string[i])
        return string
