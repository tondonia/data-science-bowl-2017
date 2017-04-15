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


_EPSILON = 1e-8
INPUT_SIZE = 512
OUTPUT_SIZE = 512
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

#MEAN_PIXEL = 0.25
#MEAN_PIXEL =  0.66200809792889126
#MEAN_PIXEL = 0.00872719537085

FLIP_X = True
FLIP_Y = False
TRANS_RANGE = (-3,3)
ROT_RANGE = (-20,20)
ZOOM_RANGE = (0.9,1.1)

from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
from skimage.transform import warp, AffineTransform
CV2_AVAILABLE=False

def augment(images):
    pixels = images[0].shape[1]
    center = pixels/2.-0.5

    random_flip_x = FLIP_X and np.random.randint(2) == 1
    random_flip_y = FLIP_Y and np.random.randint(2) == 1

    # Translation shift
    shift_x = np.random.uniform(*TRANS_RANGE)
    shift_y = np.random.uniform(*TRANS_RANGE)
    rotation_degrees = np.random.uniform(*ROT_RANGE)
    zoom_factor = np.random.uniform(*ZOOM_RANGE)
    #zoom_factor = 1 + (zoom_f/2-zoom_f*np.random.random())
    if CV2_AVAILABLE:
        M = cv2.getRotationMatrix2D((center, center), rotation_degrees, zoom_factor)
        M[0, 2] += shift_x
        M[1, 2] += shift_y

    for i in range(len(images)):
        image = images[i]

        if random_flip_x:
            image[:,:] = image[:,::-1,]
        if random_flip_y:
            image = image.transpose(1,0)
            image[:,:] = image[::-1,:]
            image = image.transpose(1,0)

        if i==0: # lung
            rotate(image, rotation_degrees, reshape=False, output=image, cval=-3000)
        else:# truth and outside
            rotate(image, rotation_degrees, reshape=False, output=image)
        #image2 = zoom(image, [zoom_factor,zoom_factor])
        image2 = crop_or_pad(image, pixels, -3000)
        shift(image2, [shift_x,shift_y], output=image)
        images[i] = image

    return images


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def get_image(filename, deterministic):
    with gzip.open(filename,'rb') as f:
        lung = pickle.load(f)
        lung = lung.transpose(1,0)
        
    truth_filename = filename.replace('lung','nodule')
    segmentation_filename = filename.replace('lung','lung_masks')
    #segmentation_filename = re.sub(r'subset[0-9]','',segmentation_filename)

    if os.path.isfile(truth_filename):
        with gzip.open(truth_filename,'rb') as f:
            truth = np.array(pickle.load(f),dtype=np.float32)
            truth = truth.transpose(1,0)
    else:
        print "WARN FAILED TO FIND TRUTHS"
        truth = np.zeros_like(lung)

    if os.path.isfile(segmentation_filename):
        with gzip.open(segmentation_filename,'rb') as f:
            outside = np.where(pickle.load(f)>0,0,1)
            outside = outside.transpose(1,0)            
    else:
        print "WARN FAILED TO FIND OUTSIDES"        
        outside = np.where(lung==0,1,0)
        print 'lung not found'

    #if P.ERODE_SEGMENTATION > 0:
    #    kernel = skimage.morphology.disk(P.ERODE_SEGMENTATION)
    #    outside = skimage.morphology.binary_erosion(outside, kernel)

    outside = np.array(outside, dtype=np.float32)

    if not deterministic:
        lung, truth, outside = augment([lung, truth, outside])

    truth = np.array(np.round(truth),dtype=np.int64)
    outside = np.array(np.round(outside),dtype=np.int64)

    #Set label of outside pixels to -10
    truth = truth - (outside*10)

    lung = lung*(1-outside)
    lung = lung-outside*3000

    lung = crop_or_pad(lung, INPUT_SIZE, -3000)
    truth = crop_or_pad(truth, OUTPUT_SIZE, 0)
    outside = crop_or_pad(outside, OUTPUT_SIZE, 1)

    lung = normalize(lung)
    lung = np.expand_dims(np.expand_dims(lung, axis=0),axis=0)

    #lung = lung - MEAN_PIXEL

    truth = np.array(np.expand_dims(np.expand_dims(truth, axis=0),axis=0),dtype=np.int64)
    return lung, truth

def crop_or_pad(image, desired_size, pad_value):
    if image.shape[0] < desired_size:
        offset = int(np.ceil((desired_size-image.shape[0])/2))
        image = np.pad(image, offset, 'constant', constant_values=pad_value)

    if image.shape[0] > desired_size:
        offset = (image.shape[0]-desired_size)//2
        image = image[offset:offset+desired_size,offset:offset+desired_size]

    return image

def weight_by_class_balance(truth, classes=None):
    """
    Determines a loss weight map given the truth by balancing the classes from the classes argument.
    The classes argument can be used to only include certain classes (you may for instance want to exclude the background).
    """

    if classes is None:
        # Include all classes
        classes = np.unique(truth)

    weight_map = np.zeros_like(truth, dtype=np.float32)
    total_amount = np.product(truth.shape)

    min_weight = sys.maxint
    for c in classes:
        class_mask = np.where(truth==c,1,0)
        class_weight = 1/((np.sum(class_mask)+1e-8)/total_amount)
        if class_weight < min_weight:
            min_weight = class_weight
        weight_map += (class_mask*class_weight)#/total_amount
    weight_map /= min_weight
    return weight_map


def load_images(filenames, deterministic=False):
    slices = [get_image(filename, deterministic) for filename in filenames]
    lungs, truths = zip(*slices)

    l = np.array(np.concatenate(lungs,axis=0), dtype=np.float32)
    t = np.concatenate(truths,axis=0)

    # Weight the loss by class balancing, classes other than 0 and 1
    # get set to 0 (the background is -10)
    w = weight_by_class_balance(t, classes=[0,1])

    #Set -1 labels back to label 0
    t = np.clip(t, 0, 100000)

    l = l.reshape(l.shape[0],l.shape[2],l.shape[3],l.shape[1])
    t = t.reshape(t.shape[0],t.shape[2]*t.shape[3]*t.shape[1],1)
    w = w.reshape(w.shape[0],w.shape[2]*w.shape[3]*w.shape[1])
    
    return l, t, w, filenames

