#!/usr/bin/env python3
import sys
import argparse
import os
import cv2
import numpy as np
from cv2 import imread 
from pathlib import Path

class DataPreprocessing:    
    
    @staticmethod
    def normalize(i):
        
        im = cv2.resize(cv2.imread(os.path.join(DIR, i)),(X_shape,X_shape))[:,:,0]
        final_img = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
        
        return final_img
    
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
    print("reading files from: {}".format(Path(args.input_dir).resolve()))

    # collect all the files you need (i.e. all filenames that match "*.jpg")
    for f in Path(args.input_dir).iterdir():
        print(f.resolve())

    # do your computation, processing, data cleaning, etc
    DIR = args.input_dir
    X_shape = 256
    norm_img = np.zeros((800,800))
    files = os.listdir(DIR)
    images = [i for i in files if ".png" in i]

    dp = DataPreprocessing()

    for i in images:        
        normalized_image = dp.normalize(i)
        cv2.imwrite(i.split(".png")[0]+"_norm.png", normalized_image)

    print("writing output files to: {}".format(Path(args.output_dir).resolve()))
