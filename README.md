# Black-box Approximation to Improve OpticalCharacter Recognition Performance
This code repository contains the scripts needs to recreate the results mentioned in the manuscript "Unknown-box Approximation to Improve Optical Character Recognition Performance"
## Contents of the repo
## Steps to run 
1. Run `init_workspace.py` to create necessary directories.
Download the three dataset zip files from "http://bit.ly/ocr-grad-approx" and put them in a directory named data and unzip them.
2. Install the python packages listed in requirements.txt. We have used Anaconda package manager in Linux environment (Ubuntu 18.04.5 LTS) and recommends to use the same. You will have to install EasyOCR separately using Pip since there is no anaconda release for that. Use the following command to install it in your conda environment.
<your_home>/anaconda3/envs/<env_name>/bin/pip install easyocr==1.2.1