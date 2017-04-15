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

_EPSILON = 1e-8
INPUT_SIZE = 512
OUTPUT_SIZE = 512
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
#MEAN_PIXEL = 0.25
#MEAN_PIXEL =  0.66200809792889126
#MEAN_PIXEL = 0.00747357978813

FLIP_X = True
FLIP_Y = False
TRANS_RANGE = (-3,3)
ROT_RANGE = (-20,20)
ZOOM_RANGE = (0.9,1.1)

from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
from skimage.transform import warp, AffineTransform
CV2_AVAILABLE=False
print "OpenCV 2 NOT AVAILABLE, using skimage/scipy.ndimage instead"
                    

def augment_multi(images,random_flip_x,shift_x,shift_y,rotation_degrees,zoom_factor):
    pixels = images[0].shape[2]
    center = pixels/2.-0.5

    for i in range(len(images)):
        image = images[i]

        if random_flip_x:
            print "flipping image"
            #image[:,:] = image[:,::-1]
            image[:,:,:] = image[:,:,::-1] #original
        print image.shape
        if rotation_degrees != 0:
            print "rotating images"
            if i == 0: #lung
                image = rotate(image, rotation_degrees, axes=(1,2), reshape=False,cval=-3000,order=0)
            else:
                image = rotate(image, rotation_degrees, axes=(1,2), reshape=False,order=0)
            print "post rotate ",image.shape
            image = crop_or_pad_multi(image, pixels, -3000)
        print image.shape
        if shift_x != 0 or shift_y != 0:
            print "shifting images by ",shift_x,shift_y, image.shape
            if i == 0:
                image = shift(image, [0,shift_x,shift_y],order=0,cval=-3000)
            else:
                image = shift(image, [0,shift_x,shift_y],order=0)
        images[i] = image

    return images

def crop_or_pad_multi(image, desired_size, pad_value):
    if image.shape[2] < desired_size:
        offset = int(np.ceil((desired_size-image.shape[2])/2))
        image = np.pad(image, ((0,0),(offset,offset),(offset,offset)), 'constant', constant_values=pad_value)

    if image.shape[2] > desired_size:
        offset = (image.shape[2]-desired_size)//2
        image = image[:,offset:offset+desired_size,offset:offset+desired_size]

    return image


def normalize_multi(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def get_image_multi_nooutside(lungs,deterministic=True,random_flip_x=False,shift_x=0.0,shift_y=0.0,rotation_degrees=0.0,zoom_factor=0.0):
    if not deterministic:
        lungs = augment_multi([lungs],random_flip_x,shift_x,shift_y,rotation_degrees,zoom_factor)
    lungs = normalize_multi(lungs)
    #lungs = lungs - MEAN_PIXEL
    return lungs
    

def get_image_multi(lungs, deterministic,outsides,random_flip_x,shift_x,shift_y,rotation_degrees,zoom_factor):

    outsides = np.where(outsides>0,0,1)
    outsides = np.array(outsides, dtype=np.float32)

    if not deterministic:
        lungs, outsides = augment_multi([lungs, outsides],random_flip_x,shift_x,shift_y,rotation_degrees,zoom_factor)

    outsides = np.array(np.round(outsides),dtype=np.int64)

    lungs = lungs*(1-outsides)
    lungs = lungs-outsides*3000

    lungs = crop_or_pad_multi(lungs, INPUT_SIZE, -3000)

    lungs = normalize_multi(lungs)
    #lung = np.expand_dims(np.expand_dims(lung, axis=0),axis=0)

    #lungs = lungs - MEAN_PIXEL
    return lungs


def even_spacing(m,n):
    return [i*n//m + n//(2*m) for i in range(m)]

def predict_unet(model,folder,filename,deterministic=False,unet_threshold=0.95):
    with gzip.open(folder + filename,'rb') as f:
        lungs, outsides = pickle.load(f)
        #augmentations
        random_flip_x = FLIP_X and np.random.randint(2) == 1
        shift_x = int(np.random.uniform(*TRANS_RANGE))
        shift_y = int(np.random.uniform(*TRANS_RANGE))
        rotation_degrees = np.random.uniform(*ROT_RANGE)
        zoom_factor = np.random.uniform(*ZOOM_RANGE)

        t1 = int(round(time.time() * 1000))
        lungs = get_image_multi(lungs,deterministic,outsides,random_flip_x,shift_x,shift_y,rotation_degrees,zoom_factor)
        t2 = int(round(time.time() * 1000)) - t1
        print "processed file in ",t2
        sys.stdout.flush()            
            
        t1 = int(round(time.time() * 1000))
        #l = np.array(np.concatenate(slices,axis=0), dtype=np.float32)
        print lungs.shape
        lungs = np.expand_dims(lungs,axis=0)
        lungs = lungs.reshape(lungs.shape[1],lungs.shape[2],lungs.shape[3],lungs.shape[0])
        t2 = int(round(time.time() * 1000)) - t1
        print "data prep in",t2
        sys.stdout.flush()
        
        t1 = int(round(time.time() * 1000))
        preds = model.predict(lungs,batch_size=8)
        t2 = int(round(time.time() * 1000)) - t1
        print "predict in",t2
        sys.stdout.flush()            
        print preds.shape
        p = np.where(preds>=unet_threshold,1,0)
        return (lungs,p)




    
    
