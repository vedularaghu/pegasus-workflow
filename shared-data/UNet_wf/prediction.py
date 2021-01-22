#!/usr/bin/env python3
import sys
import argparse
import h5py
import pickle
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

from pathlib import Path

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
    
    CURR_PATH = args.input_dir

    infile = open(CURR_PATH + "/data_split.pkl",'rb')

    new_dict = pickle.load(infile)
    
    print("_____here", new_dict)

    infile.close()

    path = CURR_PATH

    test_data = new_dict['test']

    X_test = [cv2.imread(os.path.join(path,i))[:,:,0] for i in test_data]

    model = load_model(CURR_PATH+"/model.h5", compile=False)

    test_vol = np.array(X_test, dtype=np.float32)
    
    print(test_vol)

    preds = model.predict(test_vol)

    pred_candidates = np.random.randint(1,test_vol.shape[0],len(preds))

    for i in range(len(preds)):
        img = np.squeeze(preds[pred_candidates[i]])
        cv2.imwrite(str(test_data[i].split('.png')[0]+'_mask.png'), img)

    print("writing output files to: {}".format(Path(args.output_dir).resolve()))
    
    


