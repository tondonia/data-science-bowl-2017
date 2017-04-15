from __future__ import division
import os.path
import numpy as np
import sys, getopt, argparse
import gzip
import cPickle as pickle
import time
import glob
import datetime
from random import shuffle
import json
import unicodecsv
from random import randint
import ntpath
from collections import defaultdict
import random
import ast
from sklearn.externals import joblib

def read_test_labels(filename):
    labels = []
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        labels.append(j['id'])
    return labels

def create_feature_row(key,num_voxels,fps,depths):
    vals = []
    for fp in fps:
        if not key in fp:
            print "failed to find ",key
            fp[key] = []
        for idx in range(0,min(num_voxels,len(fp[key]))):
            vals.append(0.0)
            vals.append(fp[key][idx]["proba"])
            depth = float(depths[key])
            b = ast.literal_eval(fp[key][idx]["bbox"])
            b = [int(v) for v in b]
            b = [b[0]/depth,b[1]/wh,b[2]/wh,b[3]/depth,b[4]/wh,b[5]/wh]
            for bv in b:
                assert(bv>=0 and bv <= 1.0)
            vals = vals + b
        for idx in range(len(fp[key]),num_voxels):
            vals = vals + [1.0,-1,-1,-1,-1,-1,-1,-1]
            #vals.append(0.0)
    return vals

def read_depths(filename):
    depths = {}
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        depths[j['key']] = int(j['depth'])
    return depths
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create_replay')
    parser.add_argument('--stage2-labels', help='stage2 validation labels file', default='../stage2_sample_submission.csv')
    parser.add_argument('--fp', help='voxel false positive predictions', action='append')
    parser.add_argument('--depths', help='image depths', default="..//stage2_false_positive/dsb_image_depths.csv");
    parser.add_argument('--model', help='model file', default='../stage1_false_positive/rf.model.pkl')
    parser.add_argument('--prediction-file', help='predictions file', required=True)
    parser.set_defaults(predict=False)
    
    args = parser.parse_args()
    opts = vars(args)

    keys = read_test_labels(args.stage2_labels)
    depths = read_depths(args.depths);

    
    fps = []
    #load false positive results
    for filename in args.fp:
        print "loading",filename
        fp = {}
        csvFile = open(filename)
        reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
        for j in reader:
            key = j["key"]
            vid = j["voxel_id"]
            proba = float(j["proba"])
            j["proba"] = proba
            if not key in fp:
                fp[key] = []
            fp[key].append(j)
        for key in fp:
            fp[key] = sorted(fp[key], key=lambda k: k['proba'], reverse=True)
        fps.append(fp)

    num_voxels = 2
    data = []
    wh = 512.0
    for key in keys:
        vals = create_feature_row(key,num_voxels,fps,depths)
        d = np.array(vals)
        d = np.expand_dims(d,axis=0)
        data.append(d)
    X = np.concatenate(data)
    print X
    model = joblib.load(args.model)


    proba = model.predict_proba(X)
    print proba

    scores = zip(keys,proba[:,1])
    fout = open(args.prediction_file,"w")
    writer = unicodecsv.DictWriter(fout,encoding='utf-8',fieldnames=["id","cancer"])
    writer.writeheader()
    for (key,score) in scores:
        writer.writerow({"id":key,"cancer":score})

