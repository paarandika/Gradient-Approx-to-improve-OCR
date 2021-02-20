import torch
import numpy as np
import argparse

from torch.nn import CTCLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as utils
import torchvision.transforms as transforms

from models.model_crnn import CRNN
from datasets.ocr_dataset import OCRDataset
from datasets.img_dataset import ImgDataset
from utils import get_char_maps, get_ocr_helper
from transform_helper import PadWhite, AddGaussianNoice
import properties as properties


class TrainCRNN():

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.random_seed = args.random_seed
        self.lr = args.lr
        self.max_epochs = args.epoch
        self.ocr = args.ocr
        self.std = args.std
        self.is_random_std = args.random_std
        self.dataset_name = args.dataset

        self.decay = 0.8
        self.decay_step = 10
        torch.manual_seed(self.random_seed)
        np.random.seed(torch.initial_seed())

        if self.dataset_name == 'pos':
            self.train_set = properties.pos_text_dataset_train
            self.validation_set = properties.pos_text_dataset_dev
        elif self.dataset_name == 'vgg':
            self.train_set = properties.vgg_text_dataset_train
            self.validation_set = properties.vgg_text_dataset_dev

        self.input_size = properties.input_size

        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = CRNN(self.vocab_size, False).to(self.device)
        self.model.register_backward_hook(self.model.backward_hook)

        self.ocr = get_ocr_helper(self.ocr)

        transform = transforms.Compose([
            PadWhite(self.input_size),
            transforms.ToTensor(),
        ])
        if self.ocr is not None:
            noisy_transform = transforms.Compose([
                PadWhite(self.input_size),
                transforms.ToTensor(),
                AddGaussianNoice(
                    std=self.std, is_stochastic=self.is_random_std)
            ])
            dataset = OCRDataset(
                self.train_set, transform=noisy_transform, ocr_helper=self.ocr)
            self.loader_train = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

            validation_set = OCRDataset(
                self.validation_set, transform=transform, ocr_helper=self.ocr)
            self.loader_validation = torch.utils.data.DataLoader(
                validation_set, batch_size=self.batch_size, drop_last=True)

        self.train_set_size = len(dataset)
        self.val_set_size = len(validation_set)

        self.loss_function = CTCLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.decay_step, gamma=self.decay)

    def _call_model(self, images, labels):
        X_var = images.to(self.device)
        scores = self.model(X_var)
        out_size = torch.tensor(
            [scores.shape[0]] * images.shape[0], dtype=torch.int)
        y_size = torch.tensor([len(l) for l in labels], dtype=torch.int)
        conc_label = ''.join(labels)
        y = [self.char_to_index[c] for c in conc_label]
        y_var = torch.tensor(y, dtype=torch.int)
        return scores, y_var, out_size, y_size

    def train(self):
        writer = SummaryWriter(properties.crnn_tensor_board)

        step = 0
        validation_step = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            training_loss = 0
            for images, labels in self.loader_train:
                self.model.zero_grad()
                scores, y, pred_size, y_size = self._call_model(images, labels)
                loss = self.loss_function(scores, y, pred_size, y_size)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                if step % 100 == 0:
                    print("Iteration: %d => %f" % (step, loss.item()))
                step += 1

            writer.add_scalar('Training Loss', training_loss /
                              (self.train_set_size//self.batch_size), epoch + 1)

            self.model.eval()
            validation_loss = 0
            with torch.no_grad():
                for images, labels in self.loader_validation:
                    scores, y, pred_size, y_size = self._call_model(
                        images, labels)
                    loss = self.loss_function(scores, y, pred_size, y_size)
                    validation_loss += loss.item()
                    validation_step += 1
            writer.add_scalar('Validation Loss', validation_loss /
                              (self.val_set_size//self.batch_size), epoch + 1)
            print("Epoch: %d/%d => Training loss: %f | Validation loss: %f" % ((epoch + 1),
                                                                               self.max_epochs, training_loss /
                                                                               (self.train_set_size //
                                                                                self.batch_size),
                                                                               validation_loss/(self.val_set_size//self.batch_size)))

            self.scheduler.step()
            torch.save(self.model, properties.crnn_model_path)
        writer.flush()
        writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains the CRNN model')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate, not used by adadealta')
    parser.add_argument('--epoch', type=int,
                        default=50, help='number of epochs')
    parser.add_argument('--std', type=int,
                        default=5, help='standard deviation of Gussian noice added to images (this value devided by 100)')
    parser.add_argument('--random_seed', type=int,
                        default=42, help='random seed for shuffles')
    parser.add_argument('--ocr', default="Tesseract",
                        help="performs training lebels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument('--dataset', default='pos',
                        help="performs training with given dataset [pos, vgg]")
    parser.add_argument('--random_std', action='store_false',
                        help='randomly selected integers from 0 upto given std value (devided by 100) will be used', default=True)

    args = parser.parse_args()
    print(args)
    trainer = TrainCRNN(args)
    trainer.train()
