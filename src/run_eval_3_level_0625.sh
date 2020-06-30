#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT

exe='/home/sc/research/ScanComplete/src/eval.py'

#Test Test
INPUT_DIR='/media/sc/SSD1TB/ReconstructionFromGT_TSDF_s200/ScanNet_3_level_test_simple' #test set
INPUT_DIR='/media/sc/SSD1TB/ReconstructionFromGT_TSDF_whole_s200/ScanNet_3_level_test' #whole scene
MODEL_BASE='../train_0625_sample'
MODEL_PATH_3=$MODEL_BASE'/train_v003'
MODEL_PATH_2=$MODEL_BASE'/train_v002'
MODEL_PATH_1=$MODEL_BASE'/train_v001'

CLASS_NUM=12

PREDICT_DIR=$INPUT_DIR"_eval"
OUTPUT_DIR=$PREDICT_DIR
run_level_3=true
run_level_2=true
run_level_1=true
height_level_3=16
height_level_2=32
height_level_1=64
thread_level_3=8
thread_level_2=4
thread_level_1=2
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


########################### LEVEL 1 ############################# target sequence
if false;then
    write_output=1
    mesh=1
    HIERARCHY_LEVEL=1
    HIERARCHY_LEVEL_PREV=2
    IS_BASE_LEVEL=0
    height=$height_level_1
    thread=$thread_level_1
    macro_CheckAndCreate $OUTPUT_DIR
    count=1
    #!/bin/bash
    for i in {475..999}
    do
       echo "Welcome $i times"
       target_num=$i
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
            --class_num=$CLASS_NUM \
            --output_eva=$eva_path \
            --write_output=$write_output \
            --debug=$debug \
            --target_num=$target_num \
            --start_from=$start_from \
            --mesh=$mesh &
        
          if [ $(($count % $thread)) -eq 0 ]; then
                echo "WAIT"
                wait
          fi
          count=$((count+1))
        done
        wait
        break
    fi
    done
    fi
################################################


########################### LEVEL 3 #############################
HIERARCHY_LEVEL=3
HIERARCHY_LEVEL_PREV=3
IS_BASE_LEVEL=1
height=$height_level_3
thread=$thread_level_3
macro_CheckAndCreate $OUTPUT_DIR
count=1
if $run_level_3; then
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
        --model_path=$MODEL_PATH_3 \
        --hierarchy_level=$HIERARCHY_LEVEL \
        --height_input=$height \
        --class_num=$CLASS_NUM \
        --is_base_level \
        --output_eva=$eva_path \
        --write_output=$write_output \
        --debug=$debug \
        --target_num=$target_num \
        --start_from=$start_from \
        --mesh=$mesh &
    # break
    
      if [ $(($count % $thread)) -eq 0 ]; then
            echo "WAIT"
            wait
      fi
      count=$((count+1))
    done
    wait
fi
########################### LEVEL 2 #############################
HIERARCHY_LEVEL=2
HIERARCHY_LEVEL_PREV=3
IS_BASE_LEVEL=0
height=$height_level_2
thread=$thread_level_2
macro_CheckAndCreate $OUTPUT_DIR
count=1
if $run_level_2; then
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
        --model_path=$MODEL_PATH_2 \
        --hierarchy_level=$HIERARCHY_LEVEL \
        --height_input=$height \
        --class_num=$CLASS_NUM \
        --output_eva=$eva_path \
        --write_output=$write_output \
        --debug=$debug \
        --target_num=$target_num \
        --start_from=$start_from \
        --mesh=$mesh &
  #  break
      if [ $(($count % $thread)) -eq 0 ]; then
            echo "WAIT"
            wait
      fi
      count=$((count+1))
    done
    wait
fi
###############################################

########################### LEVEL 1 #############################
HIERARCHY_LEVEL=1
HIERARCHY_LEVEL_PREV=2
IS_BASE_LEVEL=0
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
        --class_num=$CLASS_NUM \
        --output_eva=$eva_path \
        --write_output=$write_output \
        --debug=$debug \
        --target_num=$target_num \
        --start_from=$start_from \
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
