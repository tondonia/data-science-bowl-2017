import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import sys, getopt, argparse
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cPickle as pickle
import gzip
import time
import unicodecsv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create_replay')
    parser.add_argument('--input-folder', help='output folder', required=True)
    parser.add_argument('--output-folder', help='test folder', required=True);
    parser.set_defaults(overwrite_files=False)

    args = parser.parse_args()
    opts = vars(args)

    # Some constants
    patients = os.listdir(args.input_folder)
    patients.sort()

    fout = open(args.output_folder+"dsb_image_depths.csv","w")
    writer = unicodecsv.DictWriter(fout,encoding='utf-8',fieldnames=["key","depth"])
    writer.writeheader()

    for patient in patients:
        print "processing ",patient
        with gzip.open(args.input_folder + patient,'rb') as f:
            lungs, outsides = pickle.load(f)
            key = patient.split(".")[0]
            print key,lungs.shape[0]
            writer.writerow({"key":key,"depth":lungs.shape[0]});

            

