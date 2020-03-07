#!/bin/bash
#trap "exit" INT TERM ERR
trap "kill 0" EXIT

#echo $# 
if [ $# -lt 3 ]; then
    echo "Must give three inputs: input_dir model_path level is_base_level"
    exit
fi

exe='/home/sc/research/ScanComplete/src/train_data_gen.py'

INPUT_DIR=$1
MODEL_PATH=$2
HIERARCHY_LEVEL=$3
IS_BASE_LEVEL=$(($4))
PREDICT_DIR=$INPUT_DIR"_pred_"$(($HIERARCHY_LEVEL+1))
OUTPUT_DIR=$INPUT_DIR"_pred_"$HIERARCHY_LEVEL
thread=4


## level 3 ##
if false;then
INPUT_DIR='/media/sc/BackupDesk/TrainingData_TSDF_0220/train_SceneNetRGBD_3_level_0220'
PREDICT_DIR=$INPUT_DIR"_pred"
OUTPUT_DIR=$PREDICT_DIR
MODEL_PATH='/home/sc/research/ScanComplete/train_0220/train_v003'
HIERARCHY_LEVEL=3
IS_BASE_LEVEL=1
thread=8
fi

## level 2 ##
if false;then
INPUT_DIR='/media/sc/BackupDesk/TrainingData_TSDF_0220/train_SceneNetRGBD_3_level_0220'
OUTPUT_DIR=$PREDICT_DIR'_2'
MODEL_PATH='/home/sc/research/ScanComplete/train_0220/train_v002'
HIERARCHY_LEVEL=2
IS_BASE_LEVEL=0
thread=4
fi

macro_CheckAndCreate() {
 if [ $# -lt 1 ]
  then
   echo "Must give one input path to check and create"
   exit
 fi
 if [ ! -d $1 ] 
  then
   echo "Create folder at " $1
   mkdir -p $1
 fi
}

macro_CheckAndCreate $OUTPUT_DIR
count=1
for path in "$INPUT_DIR"/*
do
  
  filename=$(basename -- "$path")
  extension="${filename##*.}"
  #filename="${filename%.*}"

  in_path=$INPUT_DIR'/'$filename
  out_path=$OUTPUT_DIR'/'$filename
  in_path_pred=$PREDICT_DIR'/'$filename
  echo "in_path: $in_path"
  echo "out_path: $out_path"
  echo "in_path_pred: $in_path_pred"

  if (( ${IS_BASE_LEVEL} > 0 )) ; then
    echo "1"
    python $exe \
    --input_dir=$in_path \
    --predict_dir=$in_path_pred \
    --output_dir=$out_path \
    --model_path=$MODEL_PATH \
    --hierarchy_level=$HIERARCHY_LEVEL \
    --is_base_level &
  else
    echo "0"
    python $exe \
    --input_dir=$in_path \
    --predict_dir=$in_path_pred \
    --output_dir=$out_path \
    --model_path=$MODEL_PATH \
    --hierarchy_level=$HIERARCHY_LEVEL &
  fi

  if [ $(($count % $thread)) -eq 0 ]; then
        echo "WAIT"
        wait
  fi

  count=$((count+1))
  
done

wait
echo "[BASH] run_train_dat_gen FINISHED !"