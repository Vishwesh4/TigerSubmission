# Tiger Submission For Leaderboard 2
This repository contains the algorithm used to submit to tigerchallenge leaderboard 2 under the team name SRI

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

## Setup
A simple and minimal setup file is included to install the package via pip. Note that the package is not in the PyPI repository.

## Dockerfile
Dockerfile to be build and uploaded to grand-challenge. It installs 
 - Ubuntu20.04, 
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