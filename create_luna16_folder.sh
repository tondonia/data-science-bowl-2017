#!/bin/bash

BASE_FOLDER=$1

mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_lung_ALL
mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_lung_masks_ALL
mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_nodule_ALL
mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_lung
mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_lung_masks
mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_nodule

END=9
for i in $(seq 0 $END); do
    echo $i;
    mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_lung_ALL/subset${i}
    mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_lung_masks_ALL/subset${i}
    mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_nodule_ALL/subset${i}
    mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_lung/subset${i}
    mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_lung_masks/subset${i}
    mkdir -p ${BASE_FOLDER}/1_1_1mm_slices_nodule/subset${i}
done

	 
