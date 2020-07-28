#! /usr/bin/env python3
import os
os.environ['KAGGLE_USERNAME'] = "vedula"
os.environ['KAGGLE_KEY'] = "482a5c14ced45f63f3698eacb8fa0c62"

import kaggle
kaggle.api.dataset_download_files('nikhilpandey360/chest-xray-masks-and-labels/download', path='.', unzip=True)
