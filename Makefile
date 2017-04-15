SHELL=/bin/bash


STAGE1_DATA=../stage1/
STAGE2_DATA=../stage2/
STAGE1_PROCESSED=../stage1_processed/
STAGE2_PROCESSED=../stage2_processed/
STAGE1_UNET_VOXELS=../stage1_unet/
STAGE2_UNET_VOXELS=../stage2_unet/
STAGE1_FALSE_POSITIVE_PREDICTIONS=../stage1_false_positive/
STAGE2_FALSE_POSITIVE_PREDICTIONS=../stage2_false_positive/
STAGE2_PREDICTIONS=../stage2_predictions/

LUNA16_FOLDER=../luna16
LUNA16_FOLDER_GZ=${LUNA16_FOLDER}/gzuidhof/
LUNA16_VOXELS=${LUNA16_FOLDER}/unet_predictions/

# create luna16 data
# requires you git clone https://github.com/gzuidhof/luna16
# luna16 data needs downloading and folders for subsets created
task.luna16_folders:
	./create_luna16_folder.sh ${LUNA16_FOLDER_GZ}
	touch task.luna16_folders

#segment nodules for unet training
task.luna16_nodule_segment:
	cd  ../luna16/gzuidhof ; python src/data_processing/create_same_spacing_data_NODULE.py
	touch task.luna16_nodule_segment

#segment all slices for unet prediction and voxel extraction 
task.luna16_all_segment:
	cd  ../luna16/gzuidhof ; python src/data_processing/create_same_spacing_data_ALL.py 
	touch task.luna16_all_segment

# create unet model from luna16 data
task.luna16_train_unet:
	mkdir -p logs_unet
	mkdir -p models/unet
	python luna16-unet.py --luna16-folder ${LUNA16_FOLDER_GZ}
	touch task.luna16_train_unet

# segment dsb lungs
task.dsb_stage1_preprocess:
	mkdir -p ${STAGE1_PROCESSED}
	python dsb_preprocess.py --input-folder ${STAGE1_DATA} --output-folder ${STAGE1_PROCESSED}
	touch task.dsb_stage1_preprocess

# segment dsb lungs
task.dsb_stage2_preprocess:
	mkdir -p ${STAGE2_PROCESSED}
	python dsb_preprocess.py --input-folder ${STAGE2_DATA} --output-folder ${STAGE2_PROCESSED}
	touch task.dsb_stage2_preprocess

# enter below the best unet model after training
UNET_BEST_MODEL=`ls -1 models/unet/ | grep best | tail -1`

# predict using unet model over all luna16 data and output voxels
task.luna16_unet_predict_voxels:
	mkdir -p ${LUNA16_VOXELS}
	python luna_unet_predict.py --luna16-folder ${LUNA16_FOLDER_GZ} --output-folder ${LUNA16_VOXELS} --model models/unet/${UNET_BEST_MODEL}
	touch task.luna16_unet_predict_voxels

#predict on dsb using unet model and output voxels for stage1 
task.dsb_unet_predict_voxels_stage1:
	mkdir -p ${STAGE1_UNET_VOXELS}
	python dsb_unet_predict.py --input-folder ${STAGE1_PROCESSED} --output-folder ${STAGE1_UNET_VOXELS} --model models/unet/${UNET_BEST_MODEL}
	touch task.dsb_unet_predict_voxels_stage1

#predict on dsb using unet model and output voxels for stage2
task.dsb_unet_predict_voxels_stage2:
	mkdir -p ${STAGE2_UNET_VOXELS}
	python dsb_unet_predict.py --input-folder ${STAGE2_PROCESSED} --output-folder ${STAGE2_UNET_VOXELS} --model models/unet/${UNET_BEST_MODEL}
	touch task.dsb_unet_predict_voxels_stage2

# get image depths from images
task.stage1_image_depths:
	python dsb_get_image_depths.py --input-folder ${STAGE1_PROCESSED} --output-folder ${STAGE1_FALSE_POSITIVE_PREDICTIONS}
task.stage2_image_depths:
	python dsb_get_image_depths.py --input-folder ${STAGE2_PROCESSED} --output-folder ${STAGE2_FALSE_POSITIVE_PREDICTIONS}

#create a 3D Convnets for false positive reduction
task.luna16_3dconvnet_model3:
	mkdir -p logs_3d
	mkdir models/models_3d_model3
	python luna_false_positive_3dconvnet.py --input-folder ${LUNA16_VOXELS} --model-name model3 --model-folder models/models_3d_model3
	touch task.luna16_3dconvnet_model3

task.luna16_3dconvnet_model3_noise:
	mkdir -p logs_3d
	mkdir models/models_3d_model3_noise
	python luna_false_positive_3dconvnet.py --input-folder ${LUNA16_VOXELS} --model-name model3_noise --model-folder models/models_3d_model3_noise
	touch task.luna16_3dconvnet_model3_noise

task.luna16_3dconvnet_model3_noise2:
	mkdir -p logs_3d
	mkdir models/models_3d_model3_noise2
	python luna_false_positive_3dconvnet.py --input-folder ${LUNA16_VOXELS} --model-name model3_noise2 --model-folder models/models_3d_model3_noise2
	touch task.luna16_3dconvnet_model3_noise2

FP_MODEL3=`ls -1 models/models_3d_model3/ | grep best | tail -1`
FP_MODEL3_NOISE=`ls -1 models/models_3d_model3_noise/ | grep best | tail -1`
FP_MODEL3_NOISE2=`ls -1 models/models_3d_model3_noise2/ | grep best | tail -1`

#create stage1 predictions from 3dcovnet models
task.dsb_voxel_predict_stage1_model3:
	mkdir -p ${STAGE1_FALSE_POSITIVE_PREDICTIONS}
	python dsb_voxel_predict.py --input-folder ${STAGE1_UNET_VOXELS} --output-folder ${STAGE1_FALSE_POSITIVE_PREDICTIONS}  --model-name model3 --model-file models/models_3d_model3/${FP_MODEL3}
	touch task.dsb_voxel_predict_stage1_model3

task.dsb_voxel_predict_stage1_model3_noise:
	mkdir -p ${STAGE1_FALSE_POSITIVE_PREDICTIONS}
	python dsb_voxel_predict.py --input-folder ${STAGE1_UNET_VOXELS} --output-folder ${STAGE1_FALSE_POSITIVE_PREDICTIONS}  --model-name model3_noise --model-file models/models_3d_model3_noise/${FP_MODEL3_NOISE}
	touch task.dsb_voxel_predict_stage1_model3_noise

task.dsb_voxel_predict_stage1_model3_noise2:
	mkdir -p ${STAGE1_FALSE_POSITIVE_PREDICTIONS}
	python dsb_voxel_predict.py --input-folder ${STAGE1_UNET_VOXELS} --output-folder ${STAGE1_FALSE_POSITIVE_PREDICTIONS}  --model-name model3_noise2 --model-file models/models_3d_model3_noise2/${FP_MODEL3_NOISE2}
	touch task.dsb_voxel_predict_stage1_model3_noise2

#create stage2 predictions from 3d convnet model
task.dsb_voxel_predict_stage2_model3:
	python dsb_voxel_predict_stage2.py --input-folder ${STAGE2_UNET_VOXELS} --output-folder ${STAGE2_FALSE_POSITIVE_PREDICTIONS}  --model-name model3 --model-file models/models_3d_model3/${FP_MODEL3}
task.dsb_voxel_predict_stage2_model3_noise:
	python dsb_voxel_predict_stage2.py --input-folder ${STAGE2_UNET_VOXELS} --output-folder ${STAGE2_FALSE_POSITIVE_PREDICTIONS}  --model-name model3_noise --model-file models/models_3d_model3_noise/${FP_MODEL3_NOISE}
task.dsb_voxel_predict_stage2_model3_noise2:
	python dsb_voxel_predict_stage2.py --input-folder ${STAGE2_UNET_VOXELS} --output-folder ${STAGE2_FALSE_POSITIVE_PREDICTIONS}  --model-name model3_noise2 --model-file models/models_3d_model3_noise2/${FP_MODEL3_NOISE2}

task.dsb_create_final_model:
	python dsb_create_voxel_model_predictions.py --output-folder ${STAGE1_FALSE_POSITIVE_PREDICTIONS}/ --fp ${STAGE1_FALSE_POSITIVE_PREDICTIONS}/model3.csv --fp ${STAGE1_FALSE_POSITIVE_PREDICTIONS}/model3_noise.csv --fp ${STAGE1_FALSE_POSITIVE_PREDICTIONS}/model3_noise2.csv

task.stage2_submission:
	mkdir -p ${STAGE2_PREDICTIONS}
	python dsb_stage2_submission.py --fp ${STAGE2_FALSE_POSITIVE_PREDICTIONS}/model3.csv --fp ${STAGE2_FALSE_POSITIVE_PREDICTIONS}/model3_noise.csv --fp ${STAGE2_FALSE_POSITIVE_PREDICTIONS}/model3_noise2.csv --prediction-file ${STAGE2_PREDICTIONS}/submission.csv


everything: task.luna16_folders task.luna16_nodule_segment task.luna16_all_segment task.luna16_train_unet task.dsb_stage1_preprocess task.dsb_stage2_preprocess task.luna16_unet_predict_voxels task.dsb_unet_predict_voxels_stage1 task.dsb_unet_predict_voxels_stage2 task.stage1_image_depths task.stage2_image_depths task.luna16_3dconvnet_model3 task.luna16_3dconvnet_model3_noise task.luna16_3dconvnet_model3_noise2 task.dsb_voxel_predict_stage1_model3 task.dsb_voxel_predict_stage1_model3_noise task.dsb_voxel_predict_stage1_model3_noise2 task.dsb_voxel_predict_stage2_model3 task.dsb_voxel_predict_stage2_model3_noise task.dsb_voxel_predict_stage2_model3_noise2 task.dsb_create_final_model task.stage2_submission

clean:
	rm -f task.*
