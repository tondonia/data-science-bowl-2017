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
from random import randint
import unicodecsv

from keras.layers.core import Reshape, Lambda
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, MaxPooling2D, Input, merge, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.layers.convolutional import Convolution2D, SeparableConvolution2D, AtrousConvolution2D
from keras.layers.local import LocallyConnected2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.optimizers import SGD , Adam, RMSprop
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from dsb_image import *
from unet_model import *
from voxel_extract import *

def read_labels(filename):
    labels = {}
    csvFile = open(filename)
    reader = unicodecsv.DictReader(csvFile,encoding='utf-8')
    for j in reader:
        #if not j['id'] == 'b8bb02d229361a623a4dc57aa0e5c485':
        labels[j['id']] = int(j['cancer'])
    return labels

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create_replay')
    parser.add_argument('--input-folder', help='train folder', required=True)
    parser.add_argument('--output-folder', help='output folder', required=True)
    parser.add_argument('--labels', help='labels file', default='../stage1_labels.csv')
    parser.add_argument('--model', help='model', required=True)
    parser.add_argument('--augment', help='augment images',dest='deterministic', action='store_false')
    parser.set_defaults(deterministic=True)
    
    args = parser.parse_args()
    opts = vars(args)

    lmap = read_labels(args.labels)

    model = unet_model(True,False)
    print "loading existing weights into model from ",args.model
    model.load_weights(args.model)

    patients = os.listdir(args.input_folder)
    patients.sort()

    print "deterministic -> ",args.deterministic
    
    t1 = int(round(time.time() * 1000))
    fnum = 0
    for filename in patients:
        fnum += 1
        print "processing file ",filename,fnum,"/",len(patients)

        key = filename.split(".")[0]

        file_glob = args.output_folder + key+"*"
        ofiles = glob.glob(file_glob)
        if len(ofiles)>0:
            print "skipping existing file ",filename
            continue
        
        (l,p) = predict_unet(model,args.input_folder,filename,args.deterministic,0.95)

        print l.shape,p.shape
        p = p.reshape(p.shape[0],512,512)
        l = l.reshape(l.shape[0],512,512)

        voxels = extract_voxels(l,p)
        print "Saving ",len(voxels)," voxel files"
        sys.stdout.flush()
        
        label = "ukn"
        if key in lmap:
            label = str(lmap[key])
        
        key = filename.split(".")[0]
        idx = 0
        if len(voxels) > 0:
            for (voxel,_,bbox) in voxels:
                file = gzip.open(args.output_folder+key+"_"+label+"_"+str(idx)+".pkl.gz",'wb')
                pickle.dump(voxel,file,protocol=-1)
                pickle.dump(bbox,file,protocol=-1)
                pickle.dump(label,file,protocol=-1)
                file.close()
                idx += 1
        else:
            print "creating empty touch file"
            touch(args.output_folder + key+".touch" )
        sys.stdout.flush()            

    
    
