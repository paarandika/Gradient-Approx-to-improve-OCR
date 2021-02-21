import torch
import argparse
import os

import torchvision.transforms as transforms

from datasets.patch_dataset import PatchDataset
from datasets.img_dataset import ImgDataset
from utils import show_img, compare_labels, get_text_stack, get_ocr_helper
from transform_helper import PadWhite
import properties as properties


class EvalPrep():

    def __init__(self, args):
        self.batch_size = 1
        self.show_txt = args.show_txt
        self.show_img = args.show_img
        self.prep_model_name = args.prep_model_name
        self.prep_model_path = args.prep_path
        self.ocr_name = args.ocr
        self.dataset_name = args.dataset

        if self.dataset_name == 'vgg':
            self.test_set = properties.vgg_text_dataset_test
            self.input_size = properties.input_size
        elif self.dataset_name == 'pos':
            self.test_set = properties.patch_dataset_test
            self.input_size = properties.input_size

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.prep_model = torch.load(os.path.join(
            self.prep_model_path, self.prep_model_name)).to(self.device)

        self.ocr = get_ocr_helper(self.ocr_name, is_eval=True)

        if self.dataset_name == 'pos':
            self.dataset = PatchDataset(self.test_set, pad=True)
        else:
            transform = transforms.Compose([
                PadWhite(self.input_size),
                transforms.ToTensor(),
            ])
            self.dataset = ImgDataset(
                self.test_set, transform=transform, include_name=True)
            self.loader_eval = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, num_workers=properties.num_workers)

    def _print_labels(self, labels, pred, ori):
        print()
        print('{:<25}{:<25}{:<25}'.format("GT Label",
                                          "Label for pred", "Label for original"))
        for i in range(len(labels)):
            try:
                print('{:<25}{:<25}{:<25}'.format(labels[i], pred[i], ori[i]))
            except:
                try:
                    print('{:<25}{:<25}{:<25}'.format(
                        labels[i], "*******", ori[i]))
                except:
                    continue

    def eval_area(self):
        print("Eval with ", self.ocr_name)
        self.prep_model.eval()
        pred_correct_count = 0
        ori_correct_count = 0
        ori_cer = 0
        pred_cer = 0
        counter = 0

        for images, labels, names in self.loader_eval:
            X_var = images.to(self.device)
            img_preds = self.prep_model(X_var)

            ocr_lbl_pred = self.ocr.get_labels(img_preds.cpu())
            ocr_lbl_ori = self.ocr.get_labels(images.cpu())

            if self.show_txt:
                self._print_labels(labels, ocr_lbl_pred, ocr_lbl_ori)

            prd_crt_count, prd_cer = compare_labels(
                ocr_lbl_pred, labels)
            ori_crt_count, o_cer = compare_labels(ocr_lbl_ori, labels)
            pred_correct_count += prd_crt_count
            ori_correct_count += ori_crt_count
            ori_cer += o_cer
            pred_cer += prd_cer

            if self.show_img:
                show_img(img_preds.detach().cpu(), "Processed images")
            counter += 1
        print()
        print('Correct count from predicted images: {:d}/{:d} ({:.5f})'.format(
            pred_correct_count, len(self.dataset), pred_correct_count/len(self.dataset)))
        print('Correct count from original images: {:d}/{:d} ({:.5f})'.format(
            ori_correct_count, len(self.dataset), ori_correct_count/len(self.dataset)))
        print('Average CER from original images: {:.5f}'.format(
            ori_cer/len(self.dataset)))
        print('Average CER from predicted images: {:.5f}'.format(
            pred_cer/len(self.dataset)))

    def eval_patch(self):
        print("Eval with ", self.ocr_name)
        self.prep_model.eval()
        ori_lbl_crt_count = 0
        ori_lbl_cer = 0
        prd_lbl_crt_count = 0
        prd_lbl_cer = 0
        lbl_count = 0
        counter = 0

        for image, labels_dict in self.dataset:
            text_crops, labels = get_text_stack(
                image.detach(), labels_dict, self.input_size)
            lbl_count += len(labels)
            ocr_labels = self.ocr.get_labels(text_crops)

            ori_crt_count, ori_cer = compare_labels(
                ocr_labels, labels)
            ori_lbl_crt_count += ori_crt_count
            ori_lbl_cer += ori_cer

            image = image.unsqueeze(0)
            X_var = image.to(self.device)
            pred = self.prep_model(X_var)
            pred = pred.detach().cpu()[0]

            pred_crops, labels = get_text_stack(
                pred, labels_dict, self.input_size)
            pred_labels = self.ocr.get_labels(pred_crops)
            prd_crt_count, prd_cer = compare_labels(
                pred_labels, labels)
            prd_lbl_crt_count += prd_crt_count
            prd_lbl_cer += prd_cer

            ori_cer = round(ori_cer/len(labels), 2)
            prd_cer = round(prd_cer/len(labels), 2)

            if self.show_img:
                show_img(image.cpu())
            if self.show_txt:
                self._print_labels(labels, pred_labels, ocr_labels)
            counter += 1

        print()
        print('Correct count from predicted images: {:d}/{:d} ({:.5f})'.format(
            prd_lbl_crt_count, lbl_count, prd_lbl_crt_count/lbl_count))
        print('Correct count from original images: {:d}/{:d} ({:.5f})'.format(
            ori_lbl_crt_count, lbl_count, ori_lbl_crt_count/lbl_count))
        print('Average CER from original images: ({:.5f})'.format(
            ori_lbl_cer/lbl_count))
        print('Average CER from predicted images: ({:.5f})'.format(
            prd_lbl_cer/lbl_count))

    def eval(self):
        if self.dataset_name == 'pos':
            self.eval_patch()
        else:
            self.eval_area()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains the CRNN model')
    parser.add_argument('--show_txt', action='store_true',
                        help='prints predictions and groud truth')
    parser.add_argument('--show_img', action='store_true',
                        help='shows each batch of images')
    parser.add_argument('--prep_path', default=properties.prep_model_path,
                        help="specify non-default prep model location")
    parser.add_argument('--dataset', default='pos',
                        help="performs training with given dataset [pos, vgg]")
    parser.add_argument('--ocr', default="Tesseract",
                        help="performs training lebels from given OCR [Tesseract,EasyOCR]")
    parser.add_argument("--prep_model_name",
                        default='prep_tesseract_pos', help='Prep model name')
    args = parser.parse_args()
    args.prep_path = "./trained_models/NN-based-prep/"
    # args.ocr = "EasyOCR"
    print(args)
    evaluator = EvalPrep(args)
    evaluator.eval()
