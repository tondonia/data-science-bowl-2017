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
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

from luna_image import *
from unet_model import *

def keras_generator(filenames,batch_size):
    batch_files = []
    while True:
        for filename in filenames:
            batch_files.append(filename)
            if len(batch_files) == batch_size:
                l, t, w, _ = load_images(batch_files,True)
                yield l,t,w
                batch_files = []

class AccuracyCallback(Callback):

    def __init__(self,lungs,truth):
        self.truth = truth
        self.nb_samples = truth.shape[0]
        self.lungs = lungs
        self.masks = []
        self.names = ['acc_lung','acc_nodules']
        self.classes = np.unique(truth)
        for c in self.classes:
            mask = np.where(truth==c,1,0).astype(bool)
            print "accuracy callback class",c,np.count_nonzero(mask)
            self.masks.append(mask)


    def on_epoch_end(self, epoch, logs={}):
        print ""
        preds = self.model.predict(self.lungs,batch_size=8)
        sum_total = 0.0
        sum_correct = 0.0
        sum_correct_percent = 0.0

        pred_0 = np.where(preds<0.5,1,0)
        pred_0_tot = np.sum(pred_0)
        pred_0_0 = np.sum(pred_0[self.masks[0]])
        pred_0_1 = pred_0_tot - pred_0_0

        pred_1 = np.where(preds>=0.5,1,0)
        pred_1_tot = np.sum(pred_1)
        pred_1_1 = np.sum(pred_1[self.masks[1]])
        pred_1_0 = pred_1_tot - pred_1_1

        tot = pred_0_tot + pred_1_tot
        pred_0_percent = pred_0_tot/float(tot)
        pred_1_percent = pred_1_tot/float(tot)

        logs["pred_0_percent"] = np.float32(pred_0_percent)
        logs["pred_1_percent"] = np.float32(pred_1_percent)
        logs["correct_0"] = np.float32(pred_0_0)
        logs["correct_1"] = np.float32(pred_1_1)
        logs["false_positives"] = np.float32(pred_1_0)
        logs["false_negatives"] = np.float32(pred_0_1)

        print ""
        print logs
        print ""
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create_replay')
    parser.add_argument('--luna16-folder', help='train folder', default="../luna16/gzuidhof/")
    parser.add_argument('--checkpoint', help='checkpoint model file')
    parser.add_argument('--model', help='model prefix', default='models/unet/model')
    parser.add_argument('--embedding', help='train embedding',dest='embed', action='store_true')
    parser.set_defaults(embed=False)

    args = parser.parse_args()
    opts = vars(args)

    train_folder = args.luna16_folder+"/1_1_1mm_slices_lung/subset[0-8]"
    test_folder = args.luna16_folder+"/1_1_1mm_slices_lung/subset9"
    
    # embedding params
    if args.embed:
        print "Training an embedding model"
        model = unet_model(False,True)
    else:
        print "Training a unet model from scratch"
        model = unet_model(True,False)

    if not args.checkpoint is None:
        print "loading existing weights into model from ",args.checkpoint
        model.load_weights(args.checkpoint,by_name=True)

    
    file_glob = train_folder + "/*.pkl.gz"
    print file_glob
    train_filenames = glob.glob(file_glob)
    shuffle(train_filenames)
    
    file_glob = test_folder + "/*.pkl.gz"
    print file_glob
    test_filenames = glob.glob(file_glob)
    #test_filenames = test_filenames[0:5]
    l, t, _, _ = load_images(test_filenames,False)

    #print np.unique(t)
    
    now = datetime.datetime.now()
    tensorboard_logname = './logs_unet/{}'.format(now.strftime('%Y.%m.%d %H.%M'))
    tensorboard = TensorBoard(log_dir=tensorboard_logname)
    acc_callback = AccuracyCallback(l,t)

    model_format = args.model+".{epoch:03d}-{val_loss:.6f}.hdf5"
    model_best_format = args.model+".{epoch:03d}-{val_loss:.6f}.best.hdf5"
    
    model.fit_generator(keras_generator(train_filenames,2),validation_data=(l,t),
                        samples_per_epoch=1000,
                        callbacks=[
                            EarlyStopping(patience=50),
                            acc_callback,
                            ModelCheckpoint(model_best_format,verbose=1,save_best_only=True,save_weights_only=True),
                            ModelCheckpoint(model_format,verbose=1,save_best_only=False,save_weights_only=True,mode='auto', period=1),
                        tensorboard],
                        nb_epoch=1500)

    
    
