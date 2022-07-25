import numpy as np
from pathlib import Path
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from tiatoolbox.tools import patchextraction
from shapely.geometry import Point, MultiPolygon
from skimage import io
import pandas as pd
import os
import time
from tqdm import tqdm
import argparse
from shapely.validation import make_valid


parser = argparse.ArgumentParser()
parser.add_argument("--job", help="job id, its one of {1,2,3,4}",required=True)
args = parser.parse_args()

job = int(args.job)
# job=3


def is_negative_sample(point,tissue_mask,tumour_bed,size=(256,256)):
    #Check if point is inside the tissue and if its inside any of the tumour bed
    temp = tissue_mask[int(point.y-size[0]//2):int(point.y+size[0]//2),int(point.x-size[1]//2):int(point.x+size[1]//2)]
    if np.sum(temp)/float(size[0]*size[1])<1.0:
        return False
    else:
        try:
            inside_tumour = tumour_bed.contains(point)
        except:
            valid_shape = make_valid(tumour_bed)
            inside_tumour = valid_shape.contains(point)
        return not inside_tumour

def point_sampler(N,tissue_mask,tumour_bed):
    point_list = []
    counter = 0
    prev = len(point_list)
    while(len(point_list) <= N):
        #For preventing error when there are no tissues in a particular x coordinate
        y = np.random.randint(tissue_mask.shape[0])
        total_sum = float(tissue_mask[y,:].sum())
        if total_sum!=0 and total_sum>=5: #No: of non zero points shouldnt be less than what we want to sample
            calc_pyx = tissue_mask[y,:]/total_sum
            index = np.random.choice(
                                    tissue_mask.shape[1],
                                    size=5,
                                    replace=False,
                                    p=calc_pyx
                                    )
            points = [Point(x,y) for x in index if is_negative_sample(Point(x,y),tissue_mask,tumour_bed)]
            point_list = point_list + points 
        else:
            pass
        curr = len(point_list)
        if (curr-prev)==0: 
            counter+=1
        else:
            counter=0
        if counter>=50: #No change continously for 50 iterations breaks the loop, this happens if its heavily tumourous
            break
    return point_list

PATCH_SAVE = Path("/localdisk3/ramanav/Tumourbed_patches")

try:
    os.mkdir(PATCH_SAVE)
except:
    pass

IMAGE_SAVE = PATCH_SAVE/Path(f"negetive_images_{job}")
# LABEL_SAVE = PATCH_SAVE/Path("labels")

try:
    os.mkdir(IMAGE_SAVE)
    # os.mkdir(LABEL_SAVE)
except:
    print("folders already exist")

parent = Path("/labs3/amartel_data3/tiger_dataset/tiger-training-data/wsibulk")

all_files = list((parent/Path("images")).glob("*.tif"))
if job==1:
    img_id = 33
    file_save = all_files[img_id:img_id+10]     
elif job==2:
    img_id = 46
    file_save = all_files[img_id:img_id+17]
elif job==3:
    img_id = 67
    file_save = all_files[img_id:img_id+16]
elif job==4:
    img_id = 83
    file_save = all_files[img_id:]
else:
    raise ValueError("Wrong job id")

data = []
for count,file_name in tqdm(enumerate(file_save)):
    #Path loading
    # img_id = 0
    # file_name = Path("235B.tif")
    file_name = Path(file_name.name)
    wsi_path = parent / Path("images") / file_name 
    tissue_mask_path =  parent/Path("tissue-masks") / Path(str(file_name.with_suffix(""))+"_tissue.tif")
    tumour_bed_path = parent / Path("annotations-tumor-bulk/xmls") / Path(file_name.with_suffix(".xml"))

    #Tissue mask loading
    tissue_mask = io.imread(str(tissue_mask_path))

    #Tumour bed loading
    wsa_tumour = WholeSlideAnnotation(tumour_bed_path)
    anot = wsa_tumour.annotations
    annotations = MultiPolygon(anot)

    #Extract points of patches
    point_list = point_sampler(10000,tissue_mask,annotations)
    if len(point_list)==0:
        continue
    df = pd.DataFrame(columns=["x","y"])
    df["x"] = np.array([points.x for points in point_list])
    df["y"] = np.array([points.y for points in point_list])
    print("Sampling done... Saving Patches")
    df1 = df.copy()
    patch_extractor = patchextraction.get_patch_extractor(
            input_img=OpenSlideWSIReader(wsi_path), # input image path, numpy array, or WSI object
            locations_list=df1, # path to list of points (csv, json), numpy list, panda DF
            method_name="point", # also supports "slidingwindow"
            patch_size=(256, 256), # size of the patch to extract around the centroids from centroids_list
            resolution=0,
            units="level",
        )
    for i,patches in enumerate(patch_extractor):
        with open(str(IMAGE_SAVE/Path("{}_{}.npy".format(img_id+count,i))), 'wb') as f:
            np.save(f,patches)
        meta = {"Name":"{}_{}.npy".format(img_id+count,i)}
        meta["x"] = df.iloc[i].x
        meta["y"] = df.iloc[i].y
        meta["file"] = str(file_name)
        data.append(meta)

print("Saving coordinates csv...")
df = pd.DataFrame(data)
# df.to_csv(str(PATCH_SAVE/Path("{}_tilcount.csv".format(job))),index=False)
df.to_csv(str(PATCH_SAVE/Path(f"{job}_tumourbedcoords.csv")),index=False)