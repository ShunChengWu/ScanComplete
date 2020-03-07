exe='/home/sc/research/ScanComplete/src/eval.py'
debug=0

INPUT_DIR='/media/sc/SSD1TB/test_SceneNetRGBD_3_level'
PREDICT_DIR='/media/sc/SSD1TB/test_SceneNetRGBD_1_level_pred'
OUTPUT_DIR='/media/sc/SSD1TB/test_SceneNetRGBD_1_level_pred'
MODEL_PATH='/home/sc/research/ScanComplete/train_1_level/train_v001'
HIERARCHY_LEVEL=1
HIERARCHY_LEVEL_PREV=1
IS_BASE_LEVEL=1
thread=1
height=64


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
  filename="${filename%.*}"
  in_path=$INPUT_DIR'/'$filename'.'$extension
  out_path=$OUTPUT_DIR'/'$filename'_pred_'$HIERARCHY_LEVEL'.'$extension
  in_path_pred=$PREDICT_DIR'/'$filename'_pred_'$HIERARCHY_LEVEL_PREV'.'$extension
  eva_path=$PREDICT_DIR'/'$filename'_pred_'$HIERARCHY_LEVEL'.txt'
  echo "in_path: $in_path"
  echo "out_path: $out_path"
  echo "in_path_pred: $in_path_pred"
 
  python $exe \
    --input_dir=$in_path \
    --predict_dir=$in_path_pred \
    --output_dir=$out_path \
    --model_path=$MODEL_PATH \
    --hierarchy_level=$HIERARCHY_LEVEL \
    --height_input=$height \
    --is_base_level \
    --output_eva=$eva_path \
    --debug $debug&
 

  if [ $(($count % $thread)) -eq 0 ]; then
        echo "WAIT"
        wait
  fi
  count=$((count+1))
done
wait

echo "[BASH] run_train_dat_gen FINISHED !"
