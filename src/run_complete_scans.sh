#!/bin/bash

# Runs scan completion with hierarchical model over a set of scenes.

# Parameter section begins here. Edit to change number of test scenes, which model to use, output path.
MAX_NUM_TEST_SCENES=1
NUM_HIERARCHY_LEVELS=3
BASE_OUTPUT_DIR=/home/sc/research/ScanComplete/vis

# Fill in path to test scenes
TEST_SCENES_PATH_3='/home/sc/research/ScanComplete/train_SceneNetRGBD_188'
TEST_SCENES_PATH_2='/home/sc/research/ScanComplete/train_SceneNetRGBD_094'
TEST_SCENES_PATH_1='/home/sc/research/ScanComplete/train_SceneNetRGBD_047'

# Fill in model to use here
PREDICT_SEMANTICS=1
HIERARCHY_LEVEL_3_MODEL='/home/sc/research/ScanComplete/train_0220/train_v003'
HIERARCHY_LEVEL_2_MODEL='/home/sc/research/ScanComplete/train_0220/train_v002'
HIERARCHY_LEVEL_1_MODEL='/home/sc/research/ScanComplete/train_0220/train_v001'

# Specify output folders for each hierarchy level.
OUTPUT_FOLDER_3=${BASE_OUTPUT_DIR}/vis_level3
OUTPUT_FOLDER_2=${BASE_OUTPUT_DIR}/vis_level2
OUTPUT_FOLDER_1=${BASE_OUTPUT_DIR}/vis_level1

# End parameter section.

--alsologtostderr \
--base_dir="/home/sc/research/ScanComplete/train/train_v003" \
--height_input="16" \
--hierarchy_level="3" \
--num_total_hierarchy_levels=3 \
--is_base_level=1 \
--predict_semantics=1 \
--output_folder=/home/sc/research/ScanComplete/vis/vis_level3 \
--input_scene="/home/sc/research/ScanComplete/train_SceneNetRGBD_188/train_0.tfrecords


# Run hierarchy.

# ------- hierarchy level 1 ------- #

IS_BASE_LEVEL=1
HIERARCHY_LEVEL=3
HEIGHT_INPUT=16

# Go through all test scenes.
count=1
for scene in $TEST_SCENES_PATH_3/*.tfrecords; do
  echo "Processing hierarchy level 3, scene $count of $MAX_NUM_TEST_SCENES: $scene".
  python complete_scan.py \
    --alsologtostderr \
    --base_dir="${HIERARCHY_LEVEL_3_MODEL}" \
    --height_input="${HEIGHT_INPUT}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER_3}" \
    --input_scene="${scene}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done

