# Unknown-box Approximation to Improve Optical Character Recognition Performance
This code repository contains the scripts needs to recreate the results mentioned in the manuscript "Unknown-box Approximation to Improve Optical Character Recognition Performance" ([Springer publication of ICDAR 2021](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_31) | [arxiv](https://arxiv.org/abs/2105.07983))
## Contents of the repo
### Scripts
* train_crnn.py - Script to train the CRNN model to avoid cold start problem
* train_nn_area.py - Script to train the NN-based preprocessor with VGG dataset
* train_nn_patch.py - Script to train the NN-based preprocessor with POS patch dataset
* train_sfe_area.py - Script to train the SFE-based preprocessor with VGG dataset
* train_sfe_patch.py - Script to train the SFE-based preprocessor with POS patch dataset
* eval_prep.py - Evaluate the preprocessor with two datasets and the two OCR engines
* properties.py - Contains global properties used by the scripts
### Directories
* trained_models - Pretrained preprocessor models and CRNN models
* datasets - Contains data loader scripts
* ocr_helper - Contains codes to connect with OCR engines
* models - Contains the two models
## Steps to run 
1. We have used Anaconda package manager in the Linux environment (Ubuntu 18.04.5 LTS) and recommends to use the same. Use the following command to create a conda environment named `ocr-test` with all the dependencies.
```bash
conda env create -f environment.yml
```
2.  Run `init_workspace.py` to create necessary directories. Download the three dataset zip files from "http://bit.ly/approx-ocr-grad" and put them in a directory named `data` and unzip them.
