#!/usr/bin/env python3
#TRAIN = 70%
#Validation = 20%
#TEST = 10%

import sys
import argparse
from pathlib import Path
import os
import cv2
import numpy as np
from cv2 import imread 
from collections import defaultdict
import pickle

def parse_args(args):
    parser = argparse.ArgumentParser(description="Enter description here")
    parser.add_argument(
                "-i",
                "--input_dir",
                default=".",
                help="directory where input files will be read from"
            )

    parser.add_argument(
                "-o",
                "--output_dir",
                default=".",
                help="directory where output files will be written to"
            )

    return parser.parse_args(args)

if __name__=="__main__":
    args = parse_args(sys.argv[1:])    

    # do your computation, processing, data cleaning, etc
    DIR = args.input_dir
    filename = os.path.join(args.output_dir, "data_split.pkl")
    data = defaultdict(list)
    valid_data = list()
    mask_valid = list()
    files = os.listdir(DIR)
    all_images = [i for i in files if ".png" in i]
    masks = [i for i in all_images if "mask" in i]
    images = [i.split("_mask_")[0]+"_norm.png" for i in masks]
    test = [i for i in all_images if i not in masks and i not in images]
    for i in range(len(images)-1, int(0.7*(len(images))), -1):
        valid_data.append(images[i])
    for i in valid_data:
        mask_valid.append(i.split("_norm")[0]+"_mask_norm.png")
    for i in mask_valid:
        valid_data.append(i)
    images = [i for i in images if i not in valid_data]
    masks = [i for i in masks if i not in valid_data]
    data["train"] = images + masks
    data["valid"] = valid_data
    data["test"] = test
    output_file = open(filename, 'wb')
    pickle.dump(data, output_file)
    output_file.close()

