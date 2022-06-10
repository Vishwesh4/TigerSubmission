"""
This script is test to see what types of preprocessing works on the output of tumorbed
"""
from typing import Union
import numpy as np
from skimage import morphology
import cv2
from .aniostropic_diffusion import anisodiff

def construct_image(template,X):
    """
    Based on the given tissuemasks template, puts the given array in the correct grid
    """ 
    fill_tumour = template.flatten().copy()
    fill_tumour[np.where(fill_tumour>=1)[0]] = 255*X
    tumour_heatmap = np.reshape(fill_tumour,np.shape(template))
    return tumour_heatmap,template

def resize(img,scale_percent=500,dim=None):
    if dim is None:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
    else:
        dim = dim
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def dist2px(dist,spacing):
    dist_px = int(round(dist/(256*spacing)))
    return dist_px

def area2px(areainmm,spacing):
    #64 pixels is 1mm^2 , 1*10^6(um to mm)/(0.5*0.5) px= 4*10^6px, 1 px^2 is 256*256 => 4*10^6/(256*256) = 61 
    px_area = int(round(areainmm*1000000/((256*spacing)**2)))
    return px_area

def convert2list_tumor(tumorbed,template):
    """
    Converts the image into list for computation purposes
    """
    processed_map = template*tumorbed
    #Subtracting 1 because indices start from 0
    indices = processed_map[np.where(processed_map>0)].astype(int)-1
    tem_flat = template.flatten()
    #To get the template in flattened form
    tem_flat = 0*tem_flat[np.where(tem_flat>0)]
    tem_flat[indices] = 1
    return tem_flat.astype(int)

def convert2list_tissue(tissuebed,template):
    tissue_position = (tissuebed>0)*1
    processed_map = template*tissue_position
    #Subtracting 1 because indices start from 0
    indices = processed_map[np.where(processed_map>0)].astype(int)-1
    tem_flat = template.flatten()
    tem_flat = 0*tem_flat[np.where(tem_flat>0)]
    tem_flat[indices] = tissuebed[np.where(processed_map>0)]/255.0
    return tem_flat.astype(np.float32)

def apply_filter(loc:np.array,Image:np.array)->np.array:
    """
    Applies filter to specific pixels and replaces the value to that pixel
    Parameters:
        loc: list - list of points to apply filter on
        Iamge: np.array Image from where pixels value will be refered
    """
    Image_filter = Image.copy()
    for i in zip(loc[0],loc[1]):
        x,y = i
        img_loc = Image_filter[np.clip(x-1,a_min=0,a_max=None):x+2,np.clip(y-1,a_min=0,a_max=None):y+2]
        total_elements = np.shape(img_loc)[0]*np.shape(img_loc)[1]
        #To exclude the value of itself
        average_value = (np.sum(img_loc)-Image_filter[x,y])/(total_elements-1)
        Image_filter[x,y] = average_value
    return Image_filter

def postprocess_tissue(template:np.array,Tissue_calculation:np.array)-> np.array:
    """
    Performs postprocessing on a given slide on tissue calculation, by filling small holes,
    removing isolated pixels and performing anisotropic diffusion
    """
    tumourbed,template = construct_image(template,Tissue_calculation)
    tumourbed = tumourbed.astype(np.uint8)

    #Processing
    ret,thresh1 = cv2.threshold(tumourbed,int(0.15*255),255,cv2.THRESH_BINARY)

    #Flood Fill
    filled = morphology.remove_small_holes(thresh1,area_threshold=3,connectivity=2)
    removesmallitems = morphology.remove_small_objects(filled,min_size=3,connectivity=2)
    
    # #Applying processed information to the image
    filled_pixels = np.where(filled-(thresh1/255)==1)
    Image_filled = apply_filter(filled_pixels,tumourbed)
    
    #Apply mask to the image_filled
    Image_filter = np.multiply(Image_filled,removesmallitems)
    
    #Anisotropic diffusion
    test_anisotropic = anisodiff(Image_filter,3,20,0.075,(1,1),2.5,1)
    difference = Image_filter-test_anisotropic
    final_anisotropic = test_anisotropic + np.clip(difference,0,255)
    final_anisotropic = np.multiply(final_anisotropic,removesmallitems)

    processed_tissue = convert2list_tissue(final_anisotropic,template)
    
    return processed_tissue

def postprocess_tumorbed(template:np.array,spacing:float,Tumorbed_calculation:np.array) -> np.array:
    """
    Performs post processing on a given slide, by resizing, closing/opening, filling and removing isolated pixels,
    to give a processed tumorbed
    """
    tissue_shape = np.shape(template)
    #Small buffer added, if ratio<0.5 then we dont perform any operation
    temp_prop = 0.02 + np.sum(template>0)/(tissue_shape[0]*tissue_shape[1])

    resize_dim = (500,500)
    kernel_size = (3,3)
    fill_area_thresh_mm = 0.05
    remove_area_thresh_mm = 1.5
    
    if temp_prop<0.5:
        #For biopsies with very less tissue area
        resize_dim = (700,700)
        kernel_size = (2,2)
        fill_area_thresh_mm = 0.05
        remove_area_thresh_mm = 0.5

    # Read the mask at a reasonable level for fast processing.

    tumourbed,template = construct_image(template,Tumorbed_calculation)
    tumourbed = tumourbed.astype(np.uint8)
    
    original_shape = np.shape(tumourbed.T)
    #Processing
    tumourbed = resize(tumourbed,dim=resize_dim)
    ret,thresh1 = cv2.threshold(tumourbed,int(0.5*255),255,cv2.THRESH_BINARY)
    #Opening and closing operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #Reshape
    tumourbed_reshape = resize(opening,dim=original_shape)

    #Flood Fill
    filled = morphology.remove_small_holes(tumourbed_reshape,area_threshold=area2px(fill_area_thresh_mm,spacing),connectivity=2)
    removesmallitems = morphology.remove_small_objects(filled,min_size=area2px(remove_area_thresh_mm,spacing),connectivity=2)
    processed_tumorbed = convert2list_tumor(removesmallitems,template)

    return processed_tumorbed

def postprocess(template:np.array,spacing:float,Tumorbed_calculation:np.array,Tissuebed_calculation:np.array) -> Union[np.array,np.array]:
    """
    Performs postprocessing on a given slide with calculated template, slide spacing, tumorbed calculation and tissuebed calculation
    Parameters:
        template: Tissue position based on given slide stride
        spacing: Slide spacing in units micrometer/pixel
        Tumorbed_calculation: Tumorbed probability maps for all the patches in a slide
        Tissuebed_calculation: Tissuebed density map for all the patches in a slide
    Returns:
        processed_tumorbed: (np.array) Processed tumorbed probability map, consists of {0,1}
        processed_tissue: (np.array) Processed tissue density map
    """
    processed_tumorbed = postprocess_tumorbed(template,spacing,Tumorbed_calculation)
    processed_tissue = postprocess_tissue(template,Tissuebed_calculation)

    return processed_tumorbed,processed_tissue
