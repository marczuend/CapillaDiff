#!/bin/bash

##################################################################################
############ NEEDED VARIABLES TO SET BEFORE RUNNING THE SCRIPT  ##################
##################################################################################

EVAL_ALL_MODELS=True  # set to True to evaluate multiple models in GEN_IMG_PATH_LIST
EVAL_FOLDER="/cluster/work/medinfmk/capillaroscopy/CapillaDiff/generated_imgs/evaluation"  # path to folder containing generated images from multiple models

# Paths and names
GEN_IMG_PATH_LIST=()

if [ "$EVAL_ALL_MODELS" == "True" ]; then
    # find all subdirectories in EVAL_FOLDER and add them to GEN_IMG_PATH_LIST
    for dir in "$EVAL_FOLDER"/*/; do
        GEN_IMG_PATH_LIST+=("$dir")
    done
fi

##################################################################################
######################### END OF VARIABLE SETTINGS ###############################
##################################################################################

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
    #bash "$SCRIPT_DIR/generate_img.sh" "$GEN_IMG_PATH"
done

echo "==============================================================="
echo "======== Image Evaluation of multiple models completed ========"
echo "==============================================================="
