#!/bin/bash

# Start job or create JSON file
# 1: start training job
# 0: print JSON file for debugging in VSCode
START_OR_PRINT_JSON=1

##################################################################################
# TRAINING PARAMETERS #

BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-05
MAX_TRAIN_STEPS=50         # 500
VALIDATION_EPOCHS=25        # 100
CHECKPOINTING_STEPS=25      # 100
MIXED_PRECISION="fp16"  # set to "fp16" or "bf16" for mixed precision training. Set to "no" for full precision training

##################################################################################
############ NEEDED VARIABLES TO SET BEFORE RUNNING THE SCRIPT  ##################
##################################################################################

## set the experiment name
export EXPERIMENT="3d_test_run"

## set "conditional" for training CapillaDiff, and "naive" for training Stable Diffuison
export SD_TYPE="naive"

## set the path to the training data directory. Folder contents must follow the structure described in
## https://github.com/marczuend/CapillaDiff/blob/main/README.md#data-preparation
export IMG_DIR="/cluster/customapps/medinfmk/mazuend/CapillaDiff/pseudo_data/images"
export METADATA_DIR="/cluster/customapps/medinfmk/mazuend/CapillaDiff/pseudo_data/metadata.csv"

## set the path to the pretrained model, which could be either pretrained Stable Diffusion, or a pretrained CapillaDiff model
export MODEL_NAME="/cluster/customapps/medinfmk/mazuend/CapillaDiff/models/bbbc021_morphodiff_ckpt/checkpoint"

## set the path to the pretrained VAE model. Downloaded from: https://huggingface.co/CompVis/stable-diffusion-v1-4
export VAE_DIR="$MODEL_NAME/vae"

## Fixed parameters ##
export CKPT_NUMBER=0
export TRAINED_STEPS=0

##################################################################################
######################### END OF VARIABLE SETTINGS ###############################
##################################################################################
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Go to home directory
cd ~

# CapillaDiff main directory
if [ ! -d "CapillaDiff" ]; then
    mkdir CapillaDiff
    echo "Created CapillaDiff directory"
fi
export CAPILLADIFF_DIR="$(realpath ./CapillaDiff)"
cd "$CAPILLADIFF_DIR"

# Temporary/cache directory
if [ ! -d "$CAPILLADIFF_DIR/capilladiff/tmp" ]; then
    mkdir -p "$CAPILLADIFF_DIR/capilladiff/tmp"
    echo "Created tmp directory for training/cache"
fi
export TMPDIR="$(realpath "$CAPILLADIFF_DIR/capilladiff/tmp")"

# Checkpoint directory
if [ ! -d "$CAPILLADIFF_DIR/checkpoint" ]; then
    mkdir -p "$CAPILLADIFF_DIR/checkpoint"
    echo "Created checkpoint directory"
fi
export CHECKPOINT_DIR="$(realpath "$CAPILLADIFF_DIR/checkpoint")"
export OUTPUT_DIR="$CHECKPOINT_DIR/${EXPERIMENT}-CapillaDiff"


## set the path to the log directory
export LOG_DIR="$TMPDIR/log/"
## check if LOG_DIR exists, if not create it
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p $LOG_DIR
fi


## set the path to the checkpointing log file in .csv format. Should change the CapillaDiff to SD if training unconditional Stable Diffusion 
export CKPT_LOG_FILE="${LOG_DIR}${EXPERIMENT}_log/${EXPERIMENT}_CapillaDiff_checkpoints.csv"

## the header for the checkpointing log file
export HEADER="dataset_id,log_dir,pretrained_model_dir,checkpoint_dir,seed,trained_steps,checkpoint_number"
mkdir -p ${LOG_DIR}${EXPERIMENT}_log

## Function to get column index by header name
get_column_index() {
    local header_line=$1
    local column_name=$2
    echo $(echo "$header_line" | tr ',' '\n' | nl -v 0 | grep "$column_name" | awk '{print $1}')
}

# Check if the checkpointing log CSV file exists
if [ ! -f "$CKPT_LOG_FILE" ]; then
    # If the file does not exist, create it and add the header
    echo "$HEADER" > "$CKPT_LOG_FILE"
    echo "CSV checkpointing log file created with header: $HEADER"

elif [ $(wc -l < "$CKPT_LOG_FILE") -eq 1 ]; then
    # overwrite the header line
    echo "$HEADER" > "$CKPT_LOG_FILE"
    echo "CSV checkpointing log file header overwritten with: $HEADER"

else
    echo "CSV checkpointing log file exists in $CKPT_LOG_FILE"
    echo "Reading the last line of the log file to resume training"
    # If the file exists, read the last line
    LAST_LINE=$(tail -n 1 "$CKPT_LOG_FILE")
    
    # Extract the header line to determine the index of "checkpoint_dir" column
    HEADER_LINE=$(head -n 1 "$CKPT_LOG_FILE")
    CHECKPOINT_DIR_INDEX=$(get_column_index "$HEADER_LINE" "checkpoint_dir")

    # Extract the checkpoint_dir value from the last line
    MODEL_NAME=$(echo "$LAST_LINE" | cut -d',' -f$(($CHECKPOINT_DIR_INDEX + 1)))

    # Extract the last column from the last line
    LAST_COLUMN=$(echo "$LAST_LINE" | awk -F',' '{print $NF}')
    # Convert the last column to an integer
    CKPT_NUMBER=$((LAST_COLUMN))

    # get the number of trained steps so far
    TRAINED_STEPS_INDEX=$(get_column_index "$HEADER_LINE" "trained_steps")
    TRAINED_STEPS=$(echo "$LAST_LINE" | cut -d',' -f$(($TRAINED_STEPS_INDEX + 1)))

fi

## check if CUDA is available
is_CUDA_AVAILABLE=$(python -c 'import torch; print(torch.cuda.is_available())')

echo "================= Setup Info =================================="
printf "%-20s : %s\n" "Experiment name" "$EXPERIMENT"
printf "%-20s : %s\n" "SD Type" "$SD_TYPE"
printf "%-20s : %s\n" "Image dir" "$IMG_DIR"
printf "%-20s : %s\n" "Metadata file" "$METADATA_DIR"
printf "%-20s : %s\n" "CUDA available" "$is_CUDA_AVAILABLE"
printf "%-20s : %s\n" "Python version" "$(python -V 2>&1)"
echo "================= Model Info ==================================="
printf "%-20s : %s\n" "Model directory" "$MODEL_NAME"
#printf "%-20s : %s\n" "Checkpoint number" "$CKPT_NUMBER"
#printf "%-20s : %s\n" "Trained steps" "$TRAINED_STEPS"
printf "%-20s : %s\n" "VAE model dir" "$VAE_DIR"
printf "%-20s : %s\n" "Checkpoint directory" "$OUTPUT_DIR"
printf "%-20s : %s\n" "Log directory" "$LOG_DIR"
echo "================= TRAINING INFO ==============================="
printf "%-30s : %s\n" "Batch size" "$BATCH_SIZE"
printf "%-30s : %s\n" "Learning rate" "$LEARNING_RATE"
printf "%-30s : %s\n" "Max training steps" "$MAX_TRAIN_STEPS"
printf "%-30s : %s\n" "Validation epochs" "$VALIDATION_EPOCHS"
printf "%-30s : %s\n" "Checkpointing steps" "$CHECKPOINTING_STEPS"
echo "=============================================================="

# add 1 to the value of CKPT_NUMBER
export CKPT_NUMBER=$((${CKPT_NUMBER}+1))

#################### START OLD MORPHODIFF ARGUMENTS ####################

## set the validation prompts/perturbation ids, separated by ,
#export VALID_PROMPT="cytochalasin-d,docetaxel,epothilone-b"

#################### END OLD MORPHODIFF ARGUMENTS ######################

if [ $START_OR_PRINT_JSON -eq 1 ]; then

  echo "Starting training... train.py script launched via accelerate"
  echo "=============================================================="
    
  accelerate launch --mixed_precision=$MIXED_PRECISION $SCRIPT_DIR/../train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --naive_conditional=$SD_TYPE \
    --img_data_dir=$IMG_DIR \
    --metadata_file_path=$METADATA_DIR \
    --dataset_id=$EXPERIMENT \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 \
    --use_ema \
    --train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --validation_epochs=$VALIDATION_EPOCHS \
    --validation_prompts=$VALID_PROMPT  \
    --checkpointing_steps=$CHECKPOINTING_STEPS \
    --output_dir=$OUTPUT_DIR \
    --image_column="image" \
    --pretrained_vae_path=$VAE_DIR \
    --cache_dir=$TMPDIR \
    --report_to="wandb" \
    --seed=42 \
    --checkpointing_log_file=$CKPT_LOG_FILE \
    --checkpoint_number=$CKPT_NUMBER \
    --trained_steps=$TRAINED_STEPS

  exit

      # --random_flip \
      #--caption_column='additional_feature' \
      #--logging_dir="${LOG_DIR}${EXPERIMENT}_log" \
      
fi  

cat <<EOF
############## for launch.json ##############
            "args": [
                "--pretrained_model_name_or_path", "$MODEL_NAME",
                "--naive_conditional", "$SD_TYPE",
                // "--train_data_dir", "/cluster/customapps/medinfmk/mazuend/CapillaDiff/pseudo_data/images",
                "--img_data_dir", "$IMG_DIR",
                "--metadata_file_path", "$METADATA_DIR",
                "--dataset_id", "$EXPERIMENT",
                // "--enable_xformers_memory_efficient_attention", // enable for gpu training
                "--resolution", "512",
                "--random_flip",
                "--use_ema",
                "--train_batch_size", "$BATCH_SIZE",                 // tiny batch for debugging
                "--gradient_accumulation_steps", "$GRADIENT_ACCUMULATION_STEPS",
                "--gradient_checkpointing",
                "--max_train_steps", "$MAX_TRAIN_STEPS",             // small number of steps for debug
                "--learning_rate", "$LEARNING_RATE",
                "--lr_scheduler", "constant",
                "--lr_warmup_steps", "0",
                "--validation_epochs", "$VALIDATION_EPOCHS",
                "--validation_prompts", "$VALID_PROMPT",
                "--checkpointing_steps", "$CHECKPOINTING_STEPS",
                "--output_dir", "$OUTPUT_DIR",
                "--image_column", "image",
                // "--caption_column", "additional_feature",
                "--pretrained_vae_path", "$VAE_DIR",
                "--cache_dir", "$TMPDIR",
                "--report_to", "wandb",
                // "--logging_dir", "${LOG_DIR}${EXPERIMENT}_log",
                "--seed", "42",
                "--checkpointing_log_file", "$CKPT_LOG_FILE",
                "--checkpoint_number", "$CKPT_NUMBER",
                "--trained_steps", "$TRAINED_STEPS"
            ]
EOF