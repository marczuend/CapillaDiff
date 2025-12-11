#!/bin/bash

# Ensure script uses its own directory as working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

##################################################################################
############ NEEDED VARIABLES TO SET BEFORE RUNNING THE SCRIPT  ##################
##################################################################################

# Paths and names
GEN_IMG_PATH="/cluster/work/medinfmk/capillaroscopy/synthetic_single/evaluation_fm_sz128"  # path to generated images to evaluate
REF_IMG_PATH="/cluster/work/medinfmk/capillaroscopy/content/images"  # path to reference images
EVAL_MODEL_PATH="/cluster/customapps/medinfmk/mazuend/CapillaDiff/models/inception_v3"
SUB_SET_SIZE=0        # set to >0 to evaluate only on a subset of the images
BATCH_SIZE=256               # set the batch size for evaluation
KID_SUBSET_SIZE=1000         # subset size for KID calculation
SEED=42                     # set to an integer for a fixed seed, or "random" for random seed

##################################################################################
######################### END OF VARIABLE SETTINGS ###############################
##################################################################################

# If a first argument is provided, overwrite the GEN_IMG_PATH default
if [ $# -ge 1 ]; then
    GEN_IMG_PATH=$1
fi

### Print settings
echo "================= Model Evaluation Settings ==================="
printf "Generated Images Path: %s\n" "$GEN_IMG_PATH"
printf "Reference Images Path: %s\n" "$REF_IMG_PATH"
printf "Evaluation Model Path: %s\n" "$EVAL_MODEL_PATH"
printf "KID Subset Size: %d\n" "$KID_SUBSET_SIZE"
printf "Batch Size: %d\n" "$BATCH_SIZE"
if [ "$SUB_SET_SIZE" -gt 0 ]; then
    printf "Subset Size: %d\n" "$SUB_SET_SIZE"
else
    printf "Subset Size: Maximum\n"
fi

if [ "$SEED" == "random" ]; then
    printf "Seed: random\n"
else
    printf "Seed: %d\n" "$SEED"
fi
echo "==============================================================="


# Ask user if we should proceed
#read -p "Do you want to proceed? (y/n) " -n 1 -r
echo    # move to a new line
echo "==============================================================="

#if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#    echo "Process aborted by user."
#    exit 1
#fi

## Generate images
python ../evaluation/calculate_image_quality_metrics.py \
--ref_img_path $REF_IMG_PATH \
--gen_img_path $GEN_IMG_PATH \
--eval_model_path $EVAL_MODEL_PATH \
--batch_size $BATCH_SIZE \
--sub_set_size $SUB_SET_SIZE \
--kid_subset_size $KID_SUBSET_SIZE \
--seed $SEED