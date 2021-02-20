import datetime
import torch
import argparse

from torch.nn import CTCLoss, MSELoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms

from models.model_crnn import CRNN
from models.model_unet import UNet
from datasets.img_dataset import ImgDataset
from utils import get_char_maps, set_bn_eval, pred_to_string
from utils import get_ocr_helper, compare_labels, save_img
from transform_helper import PadWhite, AddGaussianNoice
import properties as properties


class TrainNNPrep():

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.lr_crnn = args.lr_crnn
        self.lr_prep = args.lr_prep
        self.max_epochs = args.epoch
        self.inner_limit = args.inner_limit
        self.crnn_model_path = args.crnn_model
        self.sec_loss_scalar = args.scalar
        self.ocr_name = args.ocr
        self.std = args.std
        self.is_random_std = args.random_std
        torch.manual_seed(42)

        self.train_set = properties.vgg_text_dataset_train
        self.validation_set = properties.vgg_text_dataset_dev
        self.input_size = properties.input_size

        self.ocr = get_ocr_helper(self.ocr_name)

        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if self.crnn_model_path == '':
            self.crnn_model = CRNN(self.vocab_size, False).to(self.device)
        else:
            self.crnn_model = torch.load(
                properties.crnn_model_path).to(self.device)
        self.crnn_model.register_backward_hook(self.crnn_model.backward_hook)

        self.prep_model = UNet().to(self.device)

        transform = transforms.Compose([
            PadWhite(self.input_size),
            transforms.ToTensor(),
        ])
        self.dataset = ImgDataset(
            self.train_set, transform=transform, include_name=True)
        self.validation_set = ImgDataset(
            self.validation_set, transform=transform, include_name=True)

        self.loader_train = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.loader_validation = torch.utils.data.DataLoader(
            self.validation_set, batch_size=self.batch_size, drop_last=True)

        self.train_set_size = len(self.dataset)
        self.val_set_size = len(self.validation_set)

        self.primary_loss_fn = CTCLoss().to(self.device)
        self.secondary_loss_fn = MSELoss().to(self.device)
        self.optimizer_crnn = optim.Adam(
            self.crnn_model.parameters(), lr=self.lr_crnn, weight_decay=0)
        self.optimizer_prep = optim.Adam(
            self.prep_model.parameters(), lr=self.lr_prep, weight_decay=0)

    def _call_model(self, images, labels):
        X_var = images.to(self.device)
        scores = self.crnn_model(X_var)
        out_size = torch.tensor(
            [scores.shape[0]] * images.shape[0], dtype=torch.int)
        y_size = torch.tensor([len(l) for l in labels], dtype=torch.int)
        conc_label = ''.join(labels)
        y = [self.char_to_index[c] for c in conc_label]
        y_var = torch.tensor(y, dtype=torch.int)
        return scores, y_var, out_size, y_size

    def _get_loss(self, scores, y, pred_size, y_size, img_preds):
        pri_loss = self.primary_loss_fn(scores, y, pred_size, y_size)
        sec_loss = self.secondary_loss_fn(img_preds, torch.ones(
            img_preds.shape).to(self.device))*self.sec_loss_scalar
        loss = pri_loss + sec_loss
        return loss

    def add_noise(self, imgs, noiser):
        noisy_imgs = []
        for img in imgs:
            noisy_imgs.append(noiser(img))
        return torch.stack(noisy_imgs)

    def train(self):
        noiser = AddGaussianNoice(
            std=self.std, is_stochastic=self.is_random_std)
        writer = SummaryWriter(properties.prep_tensor_board)

        step = 0
        validation_step = 0
        self.crnn_model.zero_grad()
        for epoch in range(self.max_epochs):
            training_loss = 0
            for images, labels, names in self.loader_train:
                self.crnn_model.train()
                self.prep_model.eval()
                self.prep_model.zero_grad()
                self.crnn_model.zero_grad()

                X_var = images.to(self.device)
                img_preds = self.prep_model(X_var)
                img_preds = img_preds.detach().cpu()
                temp_loss = 0

                for i in range(self.inner_limit):
                    self.prep_model.zero_grad()
                    noisy_imgs = self.add_noise(img_preds, noiser)
                    noisy_labels = self.ocr.get_labels(noisy_imgs)
                    scores, y, pred_size, y_size = self._call_model(
                        noisy_imgs, noisy_labels)
                    loss = self.primary_loss_fn(
                        scores, y, pred_size, y_size)
                    temp_loss += loss.item()
                    loss.backward()

                CRNN_training_loss = temp_loss/self.inner_limit
                self.optimizer_crnn.step()
                writer.add_scalar('CRNN Training Loss',
                                  CRNN_training_loss, step)

                self.prep_model.train()
                self.crnn_model.train()
                self.crnn_model.apply(set_bn_eval)
                self.prep_model.zero_grad()
                self.crnn_model.zero_grad()

                img_preds = self.prep_model(X_var)
                scores, y, pred_size, y_size = self._call_model(
                    img_preds, labels)
                loss = self._get_loss(scores, y, pred_size, y_size, img_preds)
                loss.backward()
                self.optimizer_prep.step()

                training_loss += loss.item()
                if step % 100 == 0:
                    print("Iteration: %d => %f" % (step, loss.item()))
                step += 1

            writer.add_scalar('Training Loss', training_loss /
                              (self.train_set_size//self.batch_size), epoch + 1)

            self.prep_model.eval()
            self.crnn_model.eval()
            pred_correct_count = 0
            pred_CER = 0
            validation_loss = 0
            tess_accuracy = 0
            tess_CER = 0
            with torch.no_grad():
                for images, labels, names in self.loader_validation:
                    X_var = images.to(self.device)
                    img_preds = self.prep_model(X_var)
                    scores, y, pred_size, y_size = self._call_model(
                        img_preds, labels)
                    loss = self._get_loss(
                        scores, y, pred_size, y_size, img_preds)
                    validation_loss += loss.item()
                    preds = pred_to_string(scores, labels, self.index_to_char)
                    ocr_labels = self.ocr.get_labels(img_preds.cpu())
                    crt, cer = compare_labels(preds, labels)
                    tess_crt, tess_cer = compare_labels(
                        ocr_labels, labels)
                    pred_correct_count += crt
                    tess_accuracy += tess_crt
                    pred_CER += cer
                    tess_CER += tess_cer
                    validation_step += 1
            writer.add_scalar('Accuracy/CRNN_output',
                              pred_correct_count/self.val_set_size, epoch + 1)
            writer.add_scalar('Accuracy/'+self.ocr_name+'_output',
                              tess_accuracy/self.val_set_size, epoch + 1)
            writer.add_scalar('WER and CER/CRNN_CER',
                              pred_CER/self.val_set_size, epoch + 1)
            writer.add_scalar('WER and CER/'+self.ocr_name+'_CER',
                              tess_CER/self.val_set_size, epoch + 1)
            writer.add_scalar('Validation Loss', validation_loss /
                              (self.val_set_size//self.batch_size), epoch + 1)

            save_img(img_preds.cpu(), 'out_' +
                     str(epoch), properties.img_out_path, 8)
            if epoch == 0:
                save_img(images.cpu(), 'out_original',
                         properties.img_out_path, 8)

            print("CRNN correct count: %d; %s correct count: %d; (validation set size:%d)" % (
                pred_correct_count, self.ocr_name, tess_accuracy, self.val_set_size))
            print("CRNN CER:%d; %s CER: %d;" %
                  (pred_CER, self.ocr_name, tess_CER))
            print("Epoch: %d/%d => Training loss: %f | Validation loss: %f" % ((epoch + 1),
                                                                               self.max_epochs, training_loss /
                                                                               (self.train_set_size //
                                                                                self.batch_size),
                                                                               validation_loss/(self.val_set_size//self.batch_size)))
            torch.save(self.prep_model,
                       properties.prep_model_path + "Prep_model_"+str(epoch))
            torch.save(self.crnn_model, properties.prep_model_path +
                       "CRNN_model_" + str(epoch))
        writer.flush()
        writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains the Prep with VGG dataset')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--lr_crnn', type=float, default=0.0001,
                        help='CRNN learning rate, not used by adadealta')
    parser.add_argument('--scalar', type=float, default=1,
                        help='scalar in which the secondary loss is multiplied')
    parser.add_argument('--lr_prep', type=float, default=0.00005,
                        help='prep model learning rate, not used by adadealta')
    parser.add_argument('--epoch', type=int,
                        default=50, help='number of epochs')
    parser.add_argument('--std', type=int,
                        default=5, help='standard deviation of Gussian noice added to images (this value devided by 100)')
    parser.add_argument('--inner_limit', type=int,
                        default=2, help='number of inner loop iterations in Alogorithm 1')
    parser.add_argument('--crnn_model', default=properties.crnn_model_path,
                        help="specify non-default CRNN model location. If given empty, a new CRNN model will be used")
    parser.add_argument('--ocr', default='Tesseract',
                        help="performs training labels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument('--random_std', action='store_false',
                        help='randomly selected integers from 0 upto given std value (devided by 100) will be used', default=True)
    args = parser.parse_args()
    print(args)

    trainer = TrainNNPrep(args)

    start = datetime.datetime.now()
    trainer.train()
    end = datetime.datetime.now()

    with open(properties.param_path, 'w') as filetowrite:
        filetowrite.write(str(start) + '\n')
        filetowrite.write(str(args) + '\n')
        filetowrite.write(str(end) + '\n')
