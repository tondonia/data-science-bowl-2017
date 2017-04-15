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


from luna_image import *
from unet_model import *

def predict_images(model,filenames,output_folder=""):
    fnum = 0
    slices = []
    predicted_sum = 0
    for filename in filenames:
        t1 = int(round(time.time() * 1000))
        fnum += 1
        print "processing file ",filename,fnum,"/",len(filenames)

        if os.path.isfile(output_folder+filename):
            print "skipping existing file ",filename
            continue

        l, t, _ , _ = load_images([filename],True)
        
        preds = model.predict(l)
        p = np.where(preds>=0.95,1,0)
        #print l.shape,t.shape,p.shape
        correct = set(np.argwhere(t==1)[:,1])
        predicted = set (np.argwhere(p==1)[:,1])
        valid = predicted.intersection(correct)
        print "Correct:",len(correct),"Predicted",len(predicted),"overlap",len(valid),"percent correct",len(valid)/float(len(correct)+0.00001)
        print np.unique(t,return_counts=True)
        print np.unique(p,return_counts=True)
        predicted_sum += len(predicted)
    print "total predicted",predicted_sum,"avg per image ",predicted_sum/float(len(filenames))

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='luna_unet_predict')
    parser.add_argument('--luna16-folder', help='train folder', default="../luna16/gzuidhof/")
    parser.add_argument('--output-folder', help='test folder', default="../luna16/unet_predictions/")
    parser.add_argument('--model', help='model', default='models/unet/best.hdf5')
    parser.set_defaults(embed=False)

    args = parser.parse_args()
    opts = vars(args)

    input_folder = args.luna16_folder + "/1_1_1mm_slices_lung_ALL/subset[0-9]"
    
    file_glob = input_folder + "/1.3.6.1.4.1.14519.5.2.1.6279.6001.130438550890816550994739120843_*.pkl.gz"
    print file_glob
    filenames = glob.glob(file_glob)
    print len(filenames)

    model = unet_model(True,False)
    print "loading existing weights into model from ",args.model
    model.load_weights(args.model)

    predict_images(model,filenames,args.output_folder)
