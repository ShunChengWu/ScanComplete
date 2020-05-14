#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

exe='/home/sc/research/ScanComplete/src/eval.py'

# Test set
INPUT_DIR='/media/sc/BackupDesk/TrainingData_TSDF_0311/SceneNetRGBD_1_level'
MODEL_BASE='/home/sc/research/ScanComplete/train_1_level'

INPUT_DIR='/media/sc/BackupDesk/TrainingData_TSDF_0311/SceneNetRGBD_3_level'
#INPUT_DIR='/media/sc/BackupDesk/TrainingData_TSDF_0311/SceneNetRGBD_3_level_test'
MODEL_BASE='../train_0311_1_level'

MODEL_BASE='../train_0324_1_level'
INPUT_DIR='/media/sc/BackupDesk/TrainingData_TSDF_0311/SceneNetRGBD_3_level_train'


MODEL_PATH_1=$MODEL_BASE'/train_v001'
PREDICT_DIR=$INPUT_DIR"_1_eval"
OUTPUT_DIR=$PREDICT_DIR
run_level_1=true
height_level_1=64

thread_level_1=1
mesh=0
write_output=1
debug=0
target_num=-1
start_from=-1



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

########################### LEVEL 1 #############################
HIERARCHY_LEVEL=1
HIERARCHY_LEVEL_PREV=1
IS_BASE_LEVEL=1
height=$height_level_1
thread=$thread_level_1
macro_CheckAndCreate $OUTPUT_DIR
count=1
if $run_level_1; then
    for path in "$INPUT_DIR"/*
    do
      
      filename=$(basename -- "$path")
      extension="${filename##*.}"
      filename="${filename%.*}"
      in_path=$INPUT_DIR'/'$filename'.'$extension
      out_path=$OUTPUT_DIR'/'$filename'_eval_'$HIERARCHY_LEVEL'.'$extension
      in_path_pred=$PREDICT_DIR'/'$filename'_eval_'$HIERARCHY_LEVEL_PREV'.'$extension
      eva_path=$PREDICT_DIR'/'$filename'_eval_'$HIERARCHY_LEVEL
      echo "in_path: $in_path"
      echo "out_path: $out_path"
      echo "in_path_pred: $in_path_pred"
    
      python $exe \
        --input_dir=$in_path \
        --predict_dir=$in_path_pred \
        --output_dir=$out_path \
        --model_path=$MODEL_PATH_1 \
        --hierarchy_level=$HIERARCHY_LEVEL \
        --height_input=$height \
        --output_eva=$eva_path \
        --write_output=$write_output \
        --debug=$debug \
        --target_num=$target_num \
        --start_from=$start_from \
        --is_base_level \
        --mesh=$mesh &
    
      if [ $(($count % $thread)) -eq 0 ]; then
            echo "WAIT"
            wait
      fi
      count=$((count+1))
    done
    wait
fi

echo "[BASH] run_train_dat_gen FINISHED !"
