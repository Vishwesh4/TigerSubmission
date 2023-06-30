# Tiger Submission For Leaderboard 2
This repository contains the algorithm used to submit to tigerchallenge leaderboard 2 under the team name [SRI](https://tiger.grand-challenge.org/teams/t/2150/).  
The algorithm can be found [here](https://grand-challenge.org/algorithms/til-test6-2/).  
The algorithm explaination can be found [here](https://rumc-gcorg-p-public.s3.amazonaws.com/evaluation-supplementary/636/062f1d55-09c0-455b-ae42-72035e8c5013/TIGER_L2_Submission.pdf)
The model weights for the algorithm can be found [here](https://zenodo.org/record/8102199) shared under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license
<img src="https://github.com/Vishwesh4/TigerSubmission/blob/master/img.png?raw=true" width="500" height="500">

## Requirements

- Ubuntu software
  - Ubuntu20.04
  - ASAP 2.0

- Python packages
  - numpy==1.21.5
  - tqdm==4.62.3
  - torch==1.10.2
  - torchvision==0.11.3
  - scikit-image==0.19.2
  - opencv-python==4.5.5.64
  - albumentations==1.1.0

## Summary of the files in package
The packages consist of the following python files.

### \_\_init\_\_
This is an empty file used for the initialization of the package directory.

### \_\_main\_\_
Contains code for calling the package as a module. Runs the process function from the processing file.

### gcio
Contains code that deals with grand challenge input and output. It includes predefined input and output paths. 

### rw
Contains code for reading and writing. Includes function for reading a multi resolution image. Furthermore, it includes classes for writing required files for the challenge, namely: segmentation mask file, detection JSON file, and TILS score file.

### processing
Main processing file. Includes code for processing a slide and applies process functions to generate TILS score. The segmentation and detection JSON files generated are not valid and generated only for the purpose of submission.

### saved_models
`bihead_cell_tissue\resnet34` contains the saved model for the TILS regressor which is a bihead network with Resnet34 as its backbone. `tumorbed` contains the saved model for tumorbed binary classifier with Resnet18 as its backbone.

### utils
This directory contains helper functions for post processing and test time augmentations and TILS regressor model

### TIL_score
This file contains the main code where TIL score is calculated based on the patches collected while processing a whole slide image. It is assumed all the patches at a uniform stride can be stored in RAM for a single slide. For some slides there may be out of memory issues. Please contact the author/ raise an issue if that happens

## trainingcodes
Please read the `README.md` specified in the file

## Setup
A simple and minimal setup file is included to install the package via pip. Note that the package is not in the PyPI repository.

## Dockerfile
Dockerfile to be build and uploaded to grand-challenge. It installs 
 - nvidia/cuda:11.1-devel-ubuntu20.04 
 - python3.8-venv, 
 - ASAP2.0, 
 - tigeralgorithmexample + requirements

As an entry point, the \_\_main\_\_ file will be run; hence process function from the processing file will be called.


## Test and Export
To test if your algorithm works and (still) produces the correct outputs you add an image to ./testinput/ and a corresponding tissue mask in ./testinput/images/

After the image and the tissue background are present in the test and test/images folder, you can run the following command to build and test the docker:

```bash
./test.sh
```

This will build the docker, run the docker and check if the required output is present. Furthermore, it will check if the detected_lymphocytes.json is in valid json format. When there are no complaints in the output you can export the algorithm to an .tar.xz file with the following command:

```bash
./export.sh
```

The resulting .tar.xz file can be uploaded to the <a href="https://grand-challenge.org/">grand-challenge</a> platform