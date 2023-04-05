import numpy as np
from tqdm import tqdm
import torchvision
import torch
import albumentations
from pathlib import Path
import shutil

from .TIL_score import TILEstimator_areaindv

from .gcio import (
    TMP_DETECTION_OUTPUT_PATH,
    TMP_SEGMENTATION_OUTPUT_PATH,
    TMP_TILS_SCORE_PATH,
    copy_data_to_output_folders,
    get_image_path_from_input_folder,
    get_tissue_mask_path_from_input_folder,
    initialize_output_folders,
)
from .rw import (
    READING_LEVEL,
    WRITING_TILE_SIZE,
    DetectionWriter,
    SegmentationWriter,
    TilsScoreWriter,
    open_multiresolutionimage_image,
)


print(f"Pytorch GPU available: {torch.cuda.is_available()}")

def isforeground(patch,threshold=0.95):
        assert len(patch.shape)==2
        h,w = patch.shape
        return np.sum(patch)/float(h*w)>threshold

def process():
    """Proceses a test slide"""
    
    level = READING_LEVEL
    tile_size = WRITING_TILE_SIZE # should be a power of 2

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    print(f'Processing image: {image_path}')
    print(f'Processing with mask: {tissue_mask_path}')

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()
    spacing_mean = (spacing[0]+spacing[1])/2
    # print(spacing_mean)

    # create writers
    print(f"Setting up writers")
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)

    transform_tta = [albumentations.VerticalFlip(p=1),
                     albumentations.HorizontalFlip(p=1),
                     albumentations.Rotate(limit=90,p=1),
                     albumentations.GaussianBlur(p=1)
                     ]
    #Set TIL scorer
    # BIOPSY = True
    BIOPSY = False
    PATH_SAVEDIR = Path("/home/user/saved_models")
    TUM_PATH = str(list((PATH_SAVEDIR/Path("tumorbed")).glob("*"))[0])
    TIL_PATH34 = str(list((PATH_SAVEDIR/Path("bihead_cell_tissue/res34")).glob("*"))[0])
    tilscorer = TILEstimator_areaindv(TUM_PATH,
                                      TIL_PATH34,
                                      transform=torchvision.transforms.ToTensor(),
                                      spacing=spacing_mean,
                                      biopsy=BIOPSY,
                                      is_tta=True,
                                      transform_tta=transform_tta,
                                      device=torch.device("cuda"))
    CHUNK_LIMIT = 15000
    count = 0
    chunk_counter = 0
    # loop over image and get tiles
    A = []
    for y in tqdm(range(0, dimensions[1], tile_size),desc="Gathering patches"):
        B = []
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()

            if isforeground(tissue_mask_tile):
                count = count + 1
                fill = count
                image_tile = image.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
                )
                tilscorer.collect(image_tile)
                chunk_counter+=1
            else:
                fill = 0
            B.extend([fill])
            #To prevent out of memory issue
            if chunk_counter>=CHUNK_LIMIT:
                #Reset chunk counter
                chunk_counter = 0
                tilscorer.calculate_chunk()
        B = np.array(B)
        A.append(B)
    #In cases of incomplete chunk computation
    if chunk_counter<CHUNK_LIMIT:
        tilscorer.calculate_chunk()
    A = np.array(A).astype(np.float32)
    print("Computing tils score...")
    tilscorer.get_template(A)
    del A
    # raise ValueError("bhak")
    tils_score = tilscorer.compute_til()
    tils_score_writer.set_tils_score(tils_score=tils_score)

    print("Saving...")
    # segmentation_writer.save()
    
    #Save fake segmentation file for submission
    output_path = TMP_SEGMENTATION_OUTPUT_PATH
    if output_path.suffix != '.tif':
        output_path = output_path / '.tif' 
    shutil.copyfile(tissue_mask_path,output_path)

    #Save fake detections for submission
    detections = [(1,1,1)]
    detection_writer.write_detections(
        detections=detections, spacing=spacing, x_offset=x, y_offset=y
        )
    detection_writer.save()
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
