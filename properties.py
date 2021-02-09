# POS
pos_text_dataset_train = "../data/textarea_dataset_train"
pos_text_dataset_test = "../data/textarea_dataset_test"
pos_text_dataset_dev = "../data/textarea_dataset_dev"

# VGG
vgg_text_dataset_train = "../data/vgg_train"
# vgg_text_dataset_train = "../data/vgg_train_sizes/vgg_train_10k"
vgg_text_dataset_test = "../data/vgg_test"
vgg_text_dataset_dev = "../data/vgg_dev"

# Brno
data_img_dir = "/mnt/dataset/Datasets/b-mod_lines/lines"
brno_text_dataset_train = "/mnt/dataset/Datasets/b-mod_lines/m_train.medium"
brno_text_dataset_test = "/mnt/dataset/Datasets/b-mod_lines/m_test.medium"
brno_text_dataset_dev = "/mnt/dataset/Datasets/b-mod_lines/m_valid.medium"

# POS
patch_dataset_train = "../data/patch_dataset_train"
patch_dataset_test = "../data/patch_dataset_test"
patch_dataset_dev = "../data/patch_dataset_dev"
patch_dataset_train_mini = "../data/patch_dataset_train_mini"

test_dummy = "../data/test_dummy"

crnn_model_path = "./outputs/crnn_trained_model/model"
crnn_tensor_board = "./outputs/crnn_runs/"
prep_model_path = "./outputs/prep_trained_model/"
prep_tensor_board = "./outputs/prep_runs/"
img_out_path = "./outputs/img_out/"
param_path = "./outputs/params.txt"

input_size = (32, 128)
num_workers = 4
char_set = ['`', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '~', '€', '}', '\\', '/']

tesseract_path = "/usr/share/tesseract-ocr/4.00/tessdata"
empty_char = ' '
max_char_len = 25
max_char_len_brno = 50
brno_input_size = (100, 800)
