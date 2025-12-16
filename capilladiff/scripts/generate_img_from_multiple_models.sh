#!/bin/bash

##################################################################################
############ NEEDED VARIABLES TO SET BEFORE RUNNING THE SCRIPT  ##################
##################################################################################

GEN_IMGS_FROM_ALL_MODELS_IN_FOLDER=True  # set to True to generate images from multiple models in MODEL_PATH_LIST

# 1. OPTION: If GEN_IMGS_FROM_ALL_MODELS_IN_FOLDER is True, specify the folder containing multiple model checkpoints
MODEL_FOLDER="ADD_YOUR_MODEL_FOLDER_PATH_HERE"  # path to folder containing multiple model checkpoints

# 2. OPTION: If GEN_IMGS_FROM_ALL_MODELS_IN_FOLDER is False, specify the list of model checkpoints to generate images from
MODEL_PATH_LIST=(
    "ADD_YOUR_MODEL_CHECKPOINT_PATH_HERE_1"
    "ADD_YOUR_MODEL_CHECKPOINT_PATH_HERE_2"
    "....."
)

##################################################################################
######################### END OF VARIABLE SETTINGS ###############################
##################################################################################

if [ "$GEN_IMGS_FROM_ALL_MODELS_IN_FOLDER" == "True" ]; then
    # find all subdirectories in MODEL_FOLDER and add them to MODEL_PATH_LIST
    for dir in "$MODEL_FOLDER"/*/; do
        MODEL_PATH_LIST+=("$dir")
    done
fi

# Ensure script uses its own directory as working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if all folders in the list exist
for MODEL_PATH in "${MODEL_PATH_LIST[@]}"; do
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model path $MODEL_PATH does not exist. Please check the paths in the MODEL_PATH_LIST variable."
        exit 1
    fi
done

# iterate over all models and generate images
for MODEL_PATH in "${MODEL_PATH_LIST[@]}"; do
    echo "y" | bash "$SCRIPT_DIR/generate_img.sh" "$MODEL_PATH"
done

echo "==============================================================="
echo "======= Image generation from multiple models completed ======="
echo "==============================================================="
