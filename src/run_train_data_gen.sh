exe='/home/sc/research/ScanComplete/src/train_data_gen.py'
INPUT_DIR='/media/sc/SSD1TB/train_SceneNetRGBD_3_level'
PREDICT_DIR='/media/sc/SSD1TB/train_SceneNetRGBD_3_level_pred'
OUTPUT_DIR='/media/sc/SSD1TB/train_SceneNetRGBD_3_level_pred'
MODEL_PATH='/home/sc/research/ScanComplete/train/train_v003'
HIERARCHY_LEVEL=3
IS_BASE_LEVEL=1
thread=9

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
  sequenceName=$filename

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
