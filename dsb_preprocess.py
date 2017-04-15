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

#
# Modified code from https://github.com/gzuidhof/luna16
#


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)



def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


def pad_images(lung_slices,lung_mask_slices):
    original_shape = lung_slices.shape
    lung_slice_512 = np.zeros((original_shape[0],512,512),np.int16) - 3000
    lung_mask_slice_512 = np.zeros((original_shape[0],512,512),np.int16)

    offset = (512 - original_shape[1])
    upper_offset = np.round(offset/2)
    lower_offset = offset - upper_offset

    lung_slice_512[:,upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_slices
    lung_mask_slice_512[:,upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask_slices

    return lung_slice_512,lung_mask_slice_512


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create_replay')
    parser.add_argument('--input-folder', help='input folder', default='../stage1/')
    parser.add_argument('--output-folder', help='output folder', default='../stage1_processed/')
    parser.add_argument('--overwrite', help='overwrite existing files',dest='overwrite_files', action='store_true')
    parser.set_defaults(overwrite_files=False)

    args = parser.parse_args()
    opts = vars(args)

    # Some constants
    patients = os.listdir(args.input_folder)
    patients.sort()

    idx = 0
    for patient_filename in patients:
        idx += 1
        if patient_filename.endswith("_bad"):
            print "skipping as bad file"
            continue
        print "processing ",patient_filename,idx,"/",len(patients)
        lung_filename = args.output_folder+patient_filename+".pkl.gz"

        if os.path.isfile(lung_filename) and not args.overwrite_files:
            print "skipping existing file ",lung_filename
            continue

        t_start = int(round(time.time() * 1000))
        t1 = int(round(time.time() * 1000))
        patient = load_scan(args.input_folder + patient_filename)
        patient_pixels = get_pixels_hu(patient)
        print patient_pixels.dtype
        pix_resampled, spacing = resample(patient_pixels, patient, [1,1,1])
        print pix_resampled.dtype
        print pix_resampled.shape
        print spacing
        t2 = int(round(time.time() * 1000)) - t1
        print "resampling took ",t2
        t1 = int(round(time.time() * 1000))
        segmented_lungs_fill = segment_lung_mask(pix_resampled, True)    
        print segmented_lungs_fill.shape
        print np.unique(segmented_lungs_fill)
        t2 = int(round(time.time() * 1000)) - t1
        print "segmenting took ",t2
        t1 = int(round(time.time() * 1000))
        l,m = pad_images(pix_resampled,segmented_lungs_fill)
        t2 = int(round(time.time() * 1000)) - t1
        print "padding took ",t2

        print l.dtype
        
        t1 = int(round(time.time() * 1000))
        file = gzip.open(lung_filename,'wb')
        pickle.dump((l,m),file,protocol=-1)
        file.close()
        t2 = int(round(time.time() * 1000)) - t1
        print "saving took ",t2
        t2 = int(round(time.time() * 1000)) - t_start
        print "whole operation took ",t2

