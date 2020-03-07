#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
NUMBER_OF_STEPS=100000
PREDICT_SEMANTICS=1  # set to 1 to predict semantics
WEIGHT_SEM=0.5
GPU=1
BATCH_SIZE=2
BASE_DIR='../train_0220'
# Fill in training data filepattern here.
PATTEN='train_*.tfrecords'
INPUT_DIR='/media/sc/BackupDesk/TrainingData_TSDF_0220/train_SceneNetRGBD_3_level_0220'
DATA='/media/sc/BackupDesk/TrainingData_TSDF_0220/train_SceneNetRGBD_3_level_0220/train_*.tfrecords'
DATA_PRED_3='/media/sc/BackupDesk/TrainingData_TSDF_0220/train_SceneNetRGBD_3_level_0220_pred_3/train_*.tfrecords'
DATA_PRED_2='/media/sc/BackupDesk/TrainingData_TSDF_0220/train_SceneNetRGBD_3_level_0220_pred_2/train_*.tfrecords'


NUMBER_OF_STEPS=40000
BASE_DIR='../train_test'
INPUT_DIR='/home/sc/research/scslam/cmake-build-debug/App/TrainingDataGenerator/tmp/SceneNetRGBD_3_level'
DATA=$INPUT_DIR'/'$PATTEN
DATA_PRED_3=$INPUT_DIR'_pred_3/'$PATTEN
DATA_PRED_2=$INPUT_DIR'_pred_2/'$PATTEN

#DATA='data/vox5-9-19_dim32/train_*.tfrecords'


# coarse level
if false; then 
IS_BASE_LEVEL=1
HIERARCHY_LEVEL=3
STORED_BLOCK_DIM=16
STORED_BLOCK_HEIGHT=16
BLOCK_DIM=16
BLOCK_HEIGHT=16
TRAIN_SAMPLES=0
VERSION=003
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
. ./run_train_data_gen_3_level.sh $INPUT_DIR ${BASE_DIR}/train_v${VERSION} $HIERARCHY_LEVEL $IS_BASE_LEVEL
fi

if false;then
# mid level
DATA=$DATA_PRED_3
IS_BASE_LEVEL=0
HIERARCHY_LEVEL=2
STORED_BLOCK_DIM=32
STORED_BLOCK_HEIGHT=32
BLOCK_DIM=32
BLOCK_HEIGHT=32
TRAIN_SAMPLES=1
VERSION=002

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
	  --number_of_steps="${NUMBER_OF_STEPS}" \
	  --train_samples
. ./run_train_data_gen_3_level.sh $INPUT_DIR ${BASE_DIR}/train_v${VERSION} $HIERARCHY_LEVEL $IS_BASE_LEVEL
fi

if true;then
## hi level
DATA=$DATA_PRED_2
IS_BASE_LEVEL=0
HIERARCHY_LEVEL=1
STORED_BLOCK_DIM=64
STORED_BLOCK_HEIGHT=64
BLOCK_DIM=64
BLOCK_HEIGHT=64
TRAIN_SAMPLES=1
VERSION=001

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
	  --number_of_steps="${NUMBER_OF_STEPS}" \
	  --train_samples
fi