#!/bin/bash

##################################################################################
############ NEEDED VARIABLES TO SET BEFORE RUNNING THE SCRIPT  ##################
##################################################################################

# Paths and names
MODEL_PATH_LIST=(
    # BOOL ENCODING MODELS
    "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/bool_encoding_cfg_off_run_1/checkpoints/checkpoint-12000"
    "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/bool_encoding_cfg_on_run_1/checkpoints/checkpoint-10005"

    "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/bool_encoding_cfg_off_scalemin_0_7/checkpoints/checkpoint-10005"
    "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/bool_encoding_cfg_off_scalemin_1_0/checkpoints/checkpoint-10005"

    # LEVEL ENCODING MODELS
    "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/level_encoding_cfg_off_run_1/checkpoints/checkpoint-1880"
    "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/level_encoding_cfg_on_run_1/checkpoints/checkpoint-1880"
    
    # TEXTMODE ENCODING MODELS
    "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/textmode_simple_bool_encoding_cfg_on_run_1/checkpoints/checkpoint-10005"
    "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/experiments/textmode_simple_level_encoding_cfg_on_run_1/checkpoints/checkpoint-10005"
)

##################################################################################
######################### END OF VARIABLE SETTINGS ###############################
##################################################################################

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
