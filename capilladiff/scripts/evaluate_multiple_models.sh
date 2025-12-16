#!/bin/bash

##################################################################################
############ NEEDED VARIABLES TO SET BEFORE RUNNING THE SCRIPT  ##################
##################################################################################

EVAL_ALL_MODELS_IN_FOLDER=True  # set to True to evaluate multiple models in GEN_IMG_PATH_LIST

# 1. OPTION: If EVAL_ALL_MODELS_IN_FOLDER is True, specify the folder containing multiple generated image folders
EVAL_FOLDER="/cluster/work/medinfmk/capillaroscopy/CapillaDiff/generated_imgs/evaluation"  # path to folder containing generated images from multiple models


# 2. OPTION: If EVAL_ALL_MODELS_IN_FOLDER is False, specify the list of generated image folders to evaluate
GEN_IMG_PATH_LIST=(
    "ADD_YOUR_FIRST_GENERATED_IMAGES_PATH_HERE"  # path to the first generated images folder to evaluate
    "ADD_YOUR_SECOND_GENERATED_IMAGES_PATH_HERE" # path to the second generated images folder to evaluate
    "...."
)

##################################################################################
######################### END OF VARIABLE SETTINGS ###############################
##################################################################################

if [ "$EVAL_ALL_MODELS_IN_FOLDER" == "True" ]; then
    # find all subdirectories in EVAL_FOLDER and add them to GEN_IMG_PATH_LIST
    for dir in "$EVAL_FOLDER"/*/; do
        GEN_IMG_PATH_LIST+=("$dir")
    done
fi

# Ensure script uses its own directory as working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if all folders in the list exist
for GEN_IMG_PATH in "${GEN_IMG_PATH_LIST[@]}"; do
    if [ ! -d "$GEN_IMG_PATH" ]; then
        echo "Model path $GEN_IMG_PATH does not exist. Please check the paths in the GEN_IMG_PATH_LIST variable."
        exit 1
    fi
done

# iterate over all models and generate images
for GEN_IMG_PATH in "${GEN_IMG_PATH_LIST[@]}"; do
    echo "y" | bash "$SCRIPT_DIR/evaluate_model.sh" "$GEN_IMG_PATH"
done

echo "==============================================================="
echo "======== Image Evaluation of multiple models completed ========"
echo "==============================================================="
