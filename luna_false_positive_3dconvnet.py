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
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

from sklearn.metrics import accuracy_score

from convnet_model import *


def load_images(filenames,bboxes=False,debug=False):
    data = []
    data2 = []
    truths = []
    idx = 0
    for filename in filenames:
        idx += 1
        if debug:
            print "loading ",filename,idx,"/",len(filenames)
        with gzip.open(filename,'rb') as f:
            basename = ntpath.basename(filename)
            if not basename.startswith("dsb_"):
                vals = pickle.load(f)
                if len(vals) == 4:
                    (d,b,depth,t) = vals
                else:
                    (d,t) = vals
            else:
                d = pickle.load(f)
                b = pickle.load(f)
                t = pickle.load(f)
                if t == "ukn":
                    t = 0
                else:
                    t = int(t)
            d = np.expand_dims(d,axis=0)
            d = np.expand_dims(d,axis=4)
            data.append(d)
            if bboxes:
                d2 = np.array([b[0]/depth,b[3]/depth,b[1]/512,b[4]/512,b[2]/512,b[5]/512])
                assert(d2[0]<=1.0)
                assert(d2[1]<=1.0)
                assert(d2[2]<=1.0)
                assert(d2[3]<=1.0)
                assert(d2[4]<=1.0)
                assert(d2[5]<=1.0)                        
                d2 = np.expand_dims(d2,axis=0)
                data2.append(d2)
            truths.append(t)
    data = np.concatenate(data,axis=0)
    if bboxes:
        data2 = np.concatenate(data2,axis=0)
    truths = np.array(truths)
    return (data,data2,truths)


def create_truth(filenames,labels):
    vals = []
    for filename in filenames:
        vals.append(labels[filename])
    return np.array(vals)

def keras_generator(filenames,bboxes,batch_size):
    batch_files = []
    while True:
        for filename in filenames:
            batch_files.append(filename)
            if len(batch_files) == batch_size:
                l,b,t = load_images(batch_files)
                if bboxes:
                    yield [l,b],t
                else:
                    yield l,t
                batch_files = []

    

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='luna_false_positive_3dconvnet')
    parser.add_argument('--input-folder', help='input folder', required=True)
    parser.add_argument('--checkpoint', help='checkpoint model file')
    parser.add_argument('--bbox', help='use bbox in model',dest='has_bbox', action='store_true')
    parser.add_argument('--model-name', help='model name', required=True)
    parser.add_argument('--model-folder', help='model output folder', required=True)
    parser.set_defaults(has_bbox=False)

    args = parser.parse_args()
    opts = vars(args)

    file_glob = args.input_folder + "/*.pkl.gz"
    print file_glob
    filenames = glob.glob(file_glob)
    print len(filenames),"files to process"

    shuffle(filenames)
    
    split_idx = int(len(filenames)*0.8)

    train_filenames = filenames[0:split_idx]
    test_filenames = filenames[split_idx:]


    (d1,d2,t) = load_images(test_filenames,bboxes=args.has_bbox,debug=True)
    if args.has_bbox:
        print d1.shape,d2.shape,t.shape
    else:
        print d1.shape,t.shape

    baseline = np.zeros(t.shape)
    print accuracy_score(t, baseline)
    
    model = create_model(args.model_name)

    if not args.checkpoint is None:
        print "loading existing weights into model from ",args.checkpoint
        model.load_weights(args.checkpoint,by_name=True)

        
    now = datetime.datetime.now()
    tensorboard_logname = './logs_3d/{}'.format(now.strftime('%Y.%m.%d %H.%M'))
    tensorboard = TensorBoard(log_dir=tensorboard_logname)

    model_format = args.model_folder+".{epoch:03d}-{val_loss:.6f}.hdf5"
    model_best_format = args.model_folder+".{epoch:03d}-{val_loss:.6f}.best.hdf5"
    class_weight = {0:1,1:2.5}
    batch_size = 16
    model.fit_generator(keras_generator(train_filenames,args.has_bbox,batch_size),validation_data=(d1,t),
                        class_weight=class_weight,
                        samples_per_epoch=len(train_filenames),
                        callbacks=[
                            EarlyStopping(patience=50),                            
                            ModelCheckpoint(model_best_format,verbose=1,save_best_only=True,save_weights_only=True),
                            ModelCheckpoint(model_format,verbose=1,save_best_only=False,save_weights_only=True,mode='auto', period=1),
                        tensorboard],
                        nb_epoch=1500)

    
    
