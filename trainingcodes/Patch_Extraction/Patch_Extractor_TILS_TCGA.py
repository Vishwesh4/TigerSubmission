"""
TCGA has small annotations making it difficult to sample from WSI
"""
from multiprocessing.sharedctypes import Value
from wholeslidedata.iterators import create_batch_iterator
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import json
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--job", help="job id, its one of {1,2,3,4,5,6,inference}",default=2)
args = parser.parse_args()

job = args.job

PATCH_SAVE = Path("/localdisk3/ramanav/TIL_Patches_v4")
# FOLDER_NAME = Path("patch_{}".format(job))
# PATCH_SAVE = PARENT_NAME / FOLDER_NAME


try:
    os.mkdir(PATCH_SAVE)
except:
    pass

if job!="inference":
    IMAGE_SAVE = PATCH_SAVE/Path("images")
    LABEL_SAVE = PATCH_SAVE/Path("labels")

else:
    IMAGE_SAVE = PATCH_SAVE/Path("test_images")
    LABEL_SAVE = PATCH_SAVE/Path("test_labels")

try:
    os.mkdir(IMAGE_SAVE)
    os.mkdir(LABEL_SAVE)
except:
    print("folders already exist")

print("Analyzing the dataset...")


data = []

roi_path = Path("/labs3/amartel_data3/tiger_dataset/tiger-training-data/wsirois/roi-level-annotations/tissue-cells/")

json_path = roi_path / Path("tiger-coco.json")
#Form data dictionary
with open(json_path,"r") as f: 
    data_hyp=json.load(f)

#Form look up table between file name and annotations
Dict_BB = {}
for items in data_hyp["images"]:
    img_name = Path(items["file_name"]).name
    if img_name[:4]=="TCGA":
        Dict_BB[img_name] = {}
        Dict_BB[img_name]["image_id"] = items["id"]
        Dict_BB[img_name]["boxes"] = []
        for annotations in data_hyp["annotations"]:
            if annotations["image_id"] == items["id"]:
                Dict_BB[img_name]["boxes"].append(annotations["bbox"])

img_names = list(Dict_BB.keys())

#Extraction of patches
for idx in range(len(img_names)):
    file_path = roi_path / Path("images") / Path(img_names[idx])
    mask_path = roi_path / Path("masks") / Path(img_names[idx])

    #Get the image and tissue mask
    img = cv2.imread(str(file_path))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    #Get the cell mask
    img_test = img.copy()
    #Pad image so as to avoid out of problems
    img_test = np.pad(img_test,20)
    mask_cell = np.zeros(np.shape(img_test)[:2])
    for boxes in Dict_BB[img_names[idx]]["boxes"]:
        mask_cell[boxes[1]+20:boxes[1]+20+boxes[3],boxes[0]+20:boxes[0]+20+boxes[2]] = 1
    mask_cell = mask_cell[20:-20,20:-20]

    labels = np.stack((mask,mask_cell))
    with open(str(IMAGE_SAVE/Path("{}_{}.npy".format(job,idx))), 'wb') as f:
        np.save(f, img)
    with open(str(LABEL_SAVE/Path("{}_{}.npy".format(job,idx))), 'wb') as f:
        np.save(f, labels)
    #Saving pixel count of patches for later operations
    # pixel_count = np.sum(np.reshape(labels[j,0,:,:],(-1,7)),axis=0)
    meta = {"Name":"{}_{}.npy".format(job,idx)}
    tils = labels[1,:,:]
    for k in range(8):
        meta["class_{}".format(k+1)] = np.sum(np.where(labels[0,:,:]==k,1,0))
    #Invasive tumour, tumour associated stroma, inflammed stroma
    imp_tissues = [1,2,6]
    for k in range(3):
        meta["TILS_{}".format(k+1)] = np.sum(np.logical_and(tils,np.where(labels[0,:,:]==imp_tissues[k],1,0)))
    data.append(meta)

print("Saving pixel counts...")
df = pd.DataFrame(data)
# df.to_csv(str(PATCH_SAVE/Path("{}_tilcount.csv".format(job))),index=False)
df.to_csv(str(PATCH_SAVE/Path("{}_tilcount.csv".format(job))),index=False)
