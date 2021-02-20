import os
import properties
from pathlib import Path

crnn_path = Path(properties.crnn_model_path)
if not os.path.exists(crnn_path.parent):
    os.makedirs(crnn_path.parent)
if not os.path.exists(properties.crnn_tensor_board):
    os.mkdir(properties.crnn_tensor_board)
if not os.path.exists(properties.prep_model_path):
    os.mkdir(properties.prep_model_path)
if not os.path.exists(properties.prep_tensor_board):
    os.mkdir(properties.prep_tensor_board)
if not os.path.exists(properties.img_out_path):
    os.mkdir(properties.img_out_path)
if not os.path.exists("./data"):
    os.mkdir("./data")
