#!/bin/bash

GPU=0
BATCH_SIZE=2
BASE_DIR='../train_1_level_0421'
DATA='/media/sc/BackupDesk/TrainingData_0421_TSDF/SceneNetRGBD_1_level_test/*.tfrecords'
NUMBER_OF_STEPS=500000

# coarse level
IS_BASE_LEVEL=1
HIERARCHY_LEVEL=1
STORED_BLOCK_DIM=64
STORED_BLOCK_HEIGHT=64
BLOCK_DIM=64
BLOCK_HEIGHT=64
TRAIN_SAMPLES=0
VERSION=001


PREDICT_SEMANTICS=1  # set to 1 to predict semantics
WEIGHT_SEM=0.5

python train.py \
	  --gpu="${GPU}" \
	  --train_dir=${BASE_DIR}/train_v${VERSION} \
	  --batch_size="${BATCH_SIZE}" \
	  --data_filepattern="${DATA}" \
	  --stored_dim_block="${STORED_BLOCK_DIM}" \
	  --stored_height_block="${STORED_BLOCK_HEIGHT}" \
	  --dim_block="${BLOCK_DIM}" \
	  --height_block="${BLOCK_HEIGHT}" \
	  --hierarchy_level="${HIERARCHY_LEVEL}" \
	  --is_base_level="${IS_BASE_LEVEL}" \
	  --predict_semantics="${PREDICT_SEMANTICS}" \
	  --weight_semantic="${WEIGHT_SEM}" \
	  --number_of_steps="${NUMBER_OF_STEPS}"
