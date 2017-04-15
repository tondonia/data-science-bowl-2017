import numpy as np
from skimage import measure, morphology
import sys


VOXEL_DEPTH=32
def get_bounding_box(b,shape):
    b = [b[0],b[1],b[2],b[3],b[4],b[5]]
    for i in range(3):
        width = abs(b[i]-b[i+3])
        if width<VOXEL_DEPTH:
            extra1 = (VOXEL_DEPTH-width)//2
            if width + (extra1 *2) < VOXEL_DEPTH:
                extra2 = extra1+1
            else:
                extra2 = extra1
            b[i] = b[i]-extra1
            b[i+3] = b[i+3] + extra2
            if b[i] < 0:
                b[i+3] += abs(b[i])
                b[i] = 0
            if b[i+3]>shape[i]:
                b[i] = b[i] - (b[i+3]-shape[i])
                b[i+3] = shape[i]
        else:
            middle= b[i] + (width//2)
            b[i] = middle - VOXEL_DEPTH//2
            b[i+3] = middle + VOXEL_DEPTH//2
    return b

def extract_voxels(images,pred_1,truths=None):
    eroded = morphology.erosion(pred_1,np.ones([3,3,3]))
    dilation = morphology.dilation(eroded,np.ones([3,3,3]))
    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    kept = 0
    removed = 0
    data = []
    for idx in range(len(regions)):
        b = regions[idx].bbox
        if regions[idx].area < 50:
            removed += 1
            continue
        kept += 1
        print "before->",b
        b = get_bounding_box(b,images.shape)
        print "after->",b
        image_voxel = images[b[0]:b[3],b[1]:b[4],b[2]:b[5]]
        label = 0
        if not truths is None:
            print "finding region in truths"
            truth_voxel = truths[b[0]:b[3],b[1]:b[4],b[2]:b[5]]
            nonzeros = np.count_nonzero(truth_voxel)
            if nonzeros > 0:
                label = 1
        assert(image_voxel.size==(VOXEL_DEPTH*VOXEL_DEPTH*VOXEL_DEPTH))
        print "Appending voxel with label ",label
        data.append((image_voxel,label,b))
    print "kept",kept,"removed",removed
    sys.stdout.flush()            
    return data

