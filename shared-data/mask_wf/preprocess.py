#!/usr/bin/env python3
import os
import numpy as np 
import pandas as pd
import os
from cv2 import imread, createCLAHE 
import cv2
import glob

image_path = os.path.join("./train_images")
mask_path = os.path.join("./train_masks")

images = os.listdir(image_path)
mask = os.listdir(mask_path)
mask = [fName.split(".png")[0] for fName in mask]
image_file_name = [fName.split("_mask")[0] for fName in mask]

check = [i for i in mask if "mask" in i]

testing_files = set(os.listdir(image_path)) & set(os.listdir(mask_path))
training_files = check

X_shape = 256*2
im_array = []
mask_array = []
norm_img = np.zeros((800,800))
for i in training_files: 
    im = cv2.resize(cv2.imread(os.path.join(image_path,i.split("_mask")[0]+".png")),(X_shape,X_shape))[:,:,0]
    mask = cv2.resize(cv2.imread(os.path.join(mask_path,i+".png")),(X_shape,X_shape))[:,:,0]    
    final_img = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
    final_mask = cv2.normalize(mask, norm_img, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(image_path,i.split("_mask")[0]+"_norm.png"), final_img)
    cv2.imwrite(os.path.join(mask_path,i+"_norm.png"), final_mask)
    im_array.append(final_img)
    mask_array.append(final_mask)