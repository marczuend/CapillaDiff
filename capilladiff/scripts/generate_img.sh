#!/bin/bash

# Ensure script uses its own directory as working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

##################################################################################
############ NEEDED VARIABLES TO SET BEFORE RUNNING THE SCRIPT  ##################
##################################################################################

# Paths and names
MODEL_PATH="/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/bool_encoding_cfg_off_scalemin_0_7/checkpoints/checkpoint-10005"
EXPERIMENT="None"      # if "None", the name of the used model checkpoint will be used
METADATA_FILE="None"  # if "None", the metadata file used during training will be used

GEN_IMG_PATH="/cluster/work/medinfmk/capillaroscopy/CapillaDiff/generated_imgs/augmentation"  # path to save generated images
OVERWRITE_EXISTING=True  # set to True to overwrite existing images in the output directory, False otherwise

CONDITION_LIST_PATH="None"  # if "None", no special conditions will be used for image generation
CONDITIONS="None"           # if "None", all conditions in CONDITION_LIST_PATH will be used
SEED=random                 # set to "random" for random seed, or an integer for a fixed seed

BATCH_SIZE=8               # set the batch size for image generation
NUM_GEN_IMG=10 # Set the number of images you want to generate for each condition
MAX_NUM_GEN_IMG=22255  # Maximum number of images to generate in total, if [>0], NUM_GEN_IMG will be adjusted accordingly
IMG_DISTRIBUTION="inverse_proportional"     # set to "uniform", "proportional" or "inverse_proportional" for different image distribution strategies
                                            # only used if MAX_NUM_GEN_IMG > 0
MODEL_TYPE="conditional" # set "conditional" for Conditional SD, and "naive" for unconditional SD

##################################################################################
######################### END OF VARIABLE SETTINGS ###############################
##################################################################################

# If a first argument is provided, overwrite the default
if [ $# -ge 1 ]; then
    MODEL_PATH=$1
fi

if [ "$EXPERIMENT" == "None" ]; then
    # extract experiment name from model path
    MODEL_NAME=$(echo "$MODEL_PATH" | sed -n 's|.*/experiments/\([^/]*\)/.*|\1|p')
    CHECKPOINT_NUMBER=$(basename "$MODEL_PATH" | sed 's|checkpoint-||')
    EXPERIMENT="${MODEL_NAME}_ckpt_${CHECKPOINT_NUMBER}"
fi

GEN_IMG_PATH="${GEN_IMG_PATH}/${EXPERIMENT}"

# check if the output directory exists
if find "$GEN_IMG_PATH" -type f -name "*.png" 2>/dev/null | grep -q . && [ "$OVERWRITE_EXISTING" == "True" ]; then
    echo "==============================================================="
    echo -e "\n\033[1;41mWARNING\033[0m Output directory '$GEN_IMG_PATH' already exists and is not empty!"
    # ask if wanted to overwrite, create new directory, or abort
    echo "1) Continue and overwrite existing files"
    echo "2) Create a new directory with timestamp"
    echo "3) Abort process"
    read -p "Choose an option (1-3): " choice

    if [[ $choice == "1" ]]; then
        echo "Overwriting existing files in $GEN_IMG_PATH"
        rm -rf "$GEN_IMG_PATH"/*
    elif [[ $choice == "2" ]]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        GEN_IMG_PATH="${GEN_IMG_PATH}_$TIMESTAMP"
        mkdir -p "$GEN_IMG_PATH"
        echo "Created new directory: $GEN_IMG_PATH"
    else
        echo "Process aborted by user."
        exit 1
    fi
    echo "==============================================================="
fi

### Print settings
echo "================= Image Generation Settings ==================="
printf "%-30s : %s\n" "Experiment name" "$EXPERIMENT"
printf "%-30s : %s\n" "Model path" "$MODEL_PATH"
if [ "$METADATA_FILE" == "None" ]; then
    printf "%-30s : %s\n" "Metadata file path" "Using metadata file from training"
else
    printf "%-30s : %s\n" "Metadata file path" "$METADATA_FILE"
fi
if [ "$CONDITION_LIST_PATH" == "None" ]; then
    printf "%-30s : %s\n" "Condition list path" "Using all conditions from metadata file"
else
    printf "%-30s : %s\n" "Condition list path" "$CONDITION_LIST_PATH"
fi
printf "%-30s : %s\n" "Generated images path" "$GEN_IMG_PATH"
printf "%-30s : %s\n" "Condition list path" "$CONDITION_LIST_PATH"
# if MAX_NUM_GEN_IMG > 0, print MAX_NUM_GEN_IMG else print NUM_GEN_IMG
if [ "$MAX_NUM_GEN_IMG" -gt 0 ]; then
    printf "%-30s : %s\n" "Number of generated images" "$MAX_NUM_GEN_IMG"
else
    printf "%-30s : %s\n" "Number of generated images" "$NUM_GEN_IMG"
    printf "%-30s : %s\n" "Image distribution strategy" "$IMG_DISTRIBUTION"
fi
printf "%-30s : %s\n" "Model type" "$MODEL_TYPE"
printf "%-30s : %s\n" "Batch size" "$BATCH_SIZE"
printf "%-30s : %s\n" "Seed" "$SEED"
echo "==============================================================="

# Ask user if we should proceed
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo    # move to a new line
echo "==============================================================="

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Process aborted by user."
    exit 1
fi

## Generate images
python ../evaluation/generate_img.py \
--experiment $EXPERIMENT \
--model_checkpoint $MODEL_PATH \
--model_type $MODEL_TYPE \
--condition_list_address $CONDITION_LIST_PATH \
--gen_img_path $GEN_IMG_PATH \
--num_imgs $NUM_GEN_IMG \
--total_num_imgs $MAX_NUM_GEN_IMG \
--metadata_file_path $METADATA_FILE \
--overwrite_existing $OVERWRITE_EXISTING \
--seed $SEED \
--img_distribution $IMG_DISTRIBUTION \
--batch_size $BATCH_SIZE