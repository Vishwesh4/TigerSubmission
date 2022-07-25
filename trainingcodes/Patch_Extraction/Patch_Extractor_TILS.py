"""
For formation of TILS dataset
Modified on 17th May 2022 -> Changed config file to tils_full_v3 and folder name to v3, accepting job to take training/inference
"""
from multiprocessing.sharedctypes import Value
from wholeslidedata.iterators import create_batch_iterator
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--job", help="job id, its one of {training,inference}",required=True)
parser.add_argument("--nreps", help="number of datapoints, total is 64(mentioned in config file, keep it to that number)*nreps",required=True)
args = parser.parse_args()

job = args.job
n_reps = int(args.nreps)

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
# config_file = "/home/ramanav/tiger_project/tigeralgorithmexample/modules/Dataset_Parsing/Data_files/tils_patch_extraction/tils_pe{}.yml".format(job)
config_file = "/home/ramanav/tiger_project/tigeralgorithmexample/modules/Dataset_Parsing/Data_files/tils_patch_extraction/tils_full_v4.yml"

# trainloader = create_batch_iterator(mode="training", user_config=config_file, cpus=16)
# print(trainloader.dataset.annotations_per_label_per_key)

# N_REPEATS = 3000
data = []
#Extraction of patches
with create_batch_iterator(user_config=config_file, 
                            mode=job, 
                            cpus=16) as trainloader:
    print(trainloader.dataset.annotations_per_label_per_key)
    for i in tqdm(range(n_reps)):
        images,labels,_ = next(trainloader)
        BATCH_SIZE = images.shape[0]
        for j in range(BATCH_SIZE):
            with open(str(IMAGE_SAVE/Path("{}_{}.npy".format(job,BATCH_SIZE*i+j))), 'wb') as f:
                np.save(f, images[j])
            with open(str(LABEL_SAVE/Path("{}_{}.npy".format(job,BATCH_SIZE*i+j))), 'wb') as f:
                np.save(f, labels[j])
            #Saving pixel count of patches for later operations
            # pixel_count = np.sum(np.reshape(labels[j,0,:,:],(-1,7)),axis=0)
            meta = {"Name":"{}_{}.npy".format(job,BATCH_SIZE*i+j)}
            tils = labels[j,1,:,:]
            for k in range(8):
                meta["class_{}".format(k+1)] = np.sum(np.where(labels[j,0,:,:]==k,1,0))
            #Invasive tumour, tumour associated stroma, inflammed stroma
            imp_tissues = [1,2,6]
            for k in range(3):
                meta["TILS_{}".format(k+1)] = np.sum(np.logical_and(tils,np.where(labels[j,0,:,:]==imp_tissues[k],1,0)))
            data.append(meta)

print("Saving pixel counts...")
df = pd.DataFrame(data)
# df.to_csv(str(PATCH_SAVE/Path("{}_tilcount.csv".format(job))),index=False)
df.to_csv(str(PATCH_SAVE/Path("{}_tilcount.csv".format(job))),index=False)
