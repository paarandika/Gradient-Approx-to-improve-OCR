import datetime
import torch
import argparse
import Levenshtein

import torchvision.transforms as transforms
from torch.nn import CTCLoss, MSELoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from transform_helper import PadWhite
from models.model_unet import UNet
from utils import get_ocr_helper, get_char_maps, save_img, compare_labels
from datasets.img_dataset import ImgDataset
import properties as properties


class TrainSFEPrep:
    def __init__(self, args):
        self.ocr_name = args.ocr
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epoch
        self.std = args.std
        self.ocr = args.ocr
        self.p_samples = args.p
        self.sec_loss_scalar = args.scalar

        self.train_set = properties.vgg_text_dataset_train
        self.validation_set = properties.vgg_text_dataset_dev
        self.input_size = properties.input_size

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.prep_model = UNet().to(self.device)
        self.ocr = get_ocr_helper(self.ocr)

        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)

        self.loss_fn = CTCLoss(reduction='none').to(self.device)

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

        self.val_set_size = len(self.validation_set)
        self.train_set_size = len(self.dataset)

        self.optimizer = optim.Adam(
            self.prep_model.parameters(), lr=self.lr, weight_decay=0)
        self.secondary_loss_fn = MSELoss().to(self.device)

    def _get_cer(self, preds, labels):
        cers = []
        for i in range(len(labels)):
            distance = Levenshtein.distance(labels[i], preds[i])
            cers.append(distance)
        return torch.tensor(cers, dtype=float)

    def train(self):
        step = 0
        validation_step = 0
        writer = SummaryWriter(properties.prep_tensor_board)
        temp_optimizer = optim.Adam(
            self.prep_model.parameters(), lr=0.01, weight_decay=0)
        temp_loss_fn = MSELoss().to(self.device)
        self.prep_model.train()
        for images, labels, names in self.loader_train:
            self.prep_model.zero_grad()
            X_var = images.to(self.device)
            preds = self.prep_model(X_var)
            loss = temp_loss_fn(preds, X_var)
            loss.backward()
            temp_optimizer.step()

        for epoch in range(self.epochs):
            training_loss = 0
            self.prep_model.train()
            for images, labels, names in self.loader_train:
                self.prep_model.zero_grad()

                X_var = images.to(self.device)
                preds = self.prep_model(X_var)
                _, c, h, w = images.shape
                grads = torch.zeros_like(images).to(self.device)
                batch_loss = 0
                for i in range(self.batch_size):
                    noise = torch.randn(
                        size=(self.p_samples, c, h, w)).to(self.device)
                    noise = torch.cat((noise, -noise), dim=0)
                    noisy_imgs = preds[i] + (noise*self.std)
                    noisy_imgs = noisy_imgs.view(2*self.p_samples, c, -1)
                    noisy_imgs -= noisy_imgs.min(2, keepdim=True)[0]
                    noisy_imgs /= noisy_imgs.max(2, keepdim=True)[0]
                    noisy_imgs = noisy_imgs.view(2*self.p_samples, c, h, w)
                    noisy_labels = self.ocr.get_labels(
                        noisy_imgs.detach().cpu())
                    loss = self._get_cer(
                        noisy_labels, [labels[i]]*2*self.p_samples)
                    mean_loss = loss.mean(dim=0)
                    batch_loss += mean_loss.item()
                    loss = loss.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    loss = noise*loss.to(self.device)
                    loss = torch.div(loss.mean(dim=0), self.std)
                    grads[i] += loss
                training_loss += (batch_loss/self.batch_size)

                sec_loss = self.secondary_loss_fn(preds, torch.ones(
                    preds.shape).to(self.device))*self.sec_loss_scalar
                sec_loss.backward(retain_graph=True)
                preds.backward(grads)
                self.optimizer.step()

                if step % 500 == 0:
                    print("Iteration: %d => %f" %
                          (step, batch_loss/self.batch_size))
                step += 1

            writer.add_scalar('Training Loss', training_loss /
                              (self.train_set_size//self.batch_size), epoch + 1)

            self.prep_model.eval()
            validation_loss = 0
            tess_crt_count = 0
            tess_CER = 0
            with torch.no_grad():
                for images, labels, names in self.loader_validation:
                    X_var = images.to(self.device)
                    img_preds = self.prep_model(X_var)
                    ocr_labels = self.ocr.get_labels(img_preds.detach().cpu())
                    loss = self._get_cer(ocr_labels, labels)
                    mean_loss = loss.mean(dim=0)
                    validation_loss += mean_loss.item()

                    tess_crt, tess_cer = compare_labels(ocr_labels, labels)
                    tess_crt_count += tess_crt
                    tess_CER += tess_cer
                    validation_step += 1
            writer.add_scalar('Accuracy/'+self.ocr_name+'_output',
                              tess_crt_count/self.val_set_size, epoch + 1)
            writer.add_scalar('WER and CER/'+self.ocr_name+'_CER',
                              tess_CER/self.val_set_size, epoch + 1)
            writer.add_scalar('Validation Loss', validation_loss /
                              (self.val_set_size//self.batch_size), epoch + 1)

            save_img(img_preds.cpu(), 'out_' +
                     str(epoch), properties.img_out_path)
            if epoch == 0:
                save_img(images.cpu(), 'out_original', properties.img_out_path)

            print("%s correct count: %d; (validation set size:%d)" % (
                self.ocr_name, tess_crt_count, self.val_set_size))
            print("%s CER: %d;" % (self.ocr_name, tess_CER))
            print("Epoch: %d/%d => Training loss: %f | Validation loss: %f" % ((epoch + 1),
                                                                               self.epochs, training_loss /
                                                                               (self.train_set_size //
                                                                                self.batch_size),
                                                                               validation_loss/(self.val_set_size//self.batch_size)))
            torch.save(self.prep_model,
                       properties.prep_model_path + "Prep_model_"+str(epoch))
            torch.save({
                'epoch': epoch,
                'prep_model_state_dict': self.prep_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, properties.prep_model_path + "state_"+str(epoch))
        writer.flush()
        writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains the CRNN model')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='prep model learning rate, not used by adadealta')
    parser.add_argument('--epoch', type=int,
                        default=50, help='number of epochs')
    parser.add_argument('--p', type=int,
                        default=5, help='number of purturbation samples')
    parser.add_argument('--std', type=int,
                        default=0.05, help='standard deviation of Gussian noice added to images')
    parser.add_argument('--prep_model', default='',
                        help="specify a trained prep model location. By default a new model will be used")
    parser.add_argument('--ocr', default='EasyOCR',
                        help="performs training labels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument('--scalar', type=float, default=1,
                        help='scalar in which the secondary loss is multiplied')
    args = parser.parse_args()
    print(args)

    start = datetime.datetime.now()
    TrainSFEPrep(args).train()
    end = datetime.datetime.now()

    with open(properties.param_path, 'w') as filetowrite:
        filetowrite.write(str(start) + '\n')
        filetowrite.write(str(args) + '\n')
        filetowrite.write(str(end) + '\n')
