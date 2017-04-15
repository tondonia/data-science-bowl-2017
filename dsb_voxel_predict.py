from __future__ import division
import os.path
import numpy as np
import sys, getopt, argparse
import gzip
import cPickle as pickle
import skimage.morphology
import time
import re
import glob
import datetime
from random import shuffle
import json
import unicodecsv
from random import randint
import ntpath

from convnet_model import *


def read_labels(filename):
    labels = {}
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        #if not j['id'] == 'b8bb02d229361a623a4dc57aa0e5c485':
        labels[j['id']] = int(j['cancer'])
    return labels

def read_test_labels(filename):
    labels = []
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        #if not j['id'] == 'b8bb02d229361a623a4dc57aa0e5c485':
        labels.append(j['id'])
    return labels

def read_depths(filename):
    depths = {}
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        depths[j['key']] = int(j['depth'])
    return depths
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='dsb_voxel_predict')
    parser.add_argument('--input-folder', help='input folder', required=True)
    parser.add_argument('--output-folder', help='output folder', default="../stage1_false_positive/");
    parser.add_argument('--labels', help='labels file', default='../stage1_labels.csv')
    parser.add_argument('--stage1-labels', help='stage1 validation labels file', default='../stage1_sample_submission.csv')
    parser.add_argument('--model-name', help='model name', required=True)
    parser.add_argument('--model-file', help='model file', required=True)

    args = parser.parse_args()
    opts = vars(args)

    
    input_folder = args.input_folder
    print "input folder -> ",input_folder
        
    #load model
    model = create_model(args.model_name)
    print "loading existing weights into model from ",args.model_name
    model.load_weights(args.model_file)

    #load labels
    lmap = read_labels(args.labels)
    test_labels = read_test_labels(args.stage1_labels)

    keys = lmap.keys() + test_labels
    
    #for each label load voxels
    fout = open(args.output_folder+args.model_name+".csv","w")
    writer = unicodecsv.DictWriter(fout,encoding='utf-8',fieldnames=["key","voxel_id","proba","label","bbox"])
    writer.writeheader()
    for label in keys:
        file_glob = input_folder + label+"*.pkl.gz"
        voxel_files = glob.glob(file_glob)

        for filename in voxel_files:
            print "processing ",filename

            basename = ntpath.basename(filename)
            voxel_id = basename.split("_")[2].split(".")[0]
            
            with gzip.open(filename,'rb') as f:
                voxel = pickle.load(f)
                b = pickle.load(f)
                _label = pickle.load(f)
                if not _label == "ukn":
                    assert(int(_label) == int(lmap[label]))

                voxel = np.expand_dims(voxel,axis=3)
                voxel = np.expand_dims(voxel,axis=0)

                p = model.predict(voxel)
                print p
                proba = p[0][0]
                if _label == "ukn":
                    out_label = -1
                else:
                    out_label = int(lmap[label])
                writer.writerow({"key" : label,"voxel_id" : voxel_id,"proba" : proba,"label":out_label,"bbox":list(b)})

