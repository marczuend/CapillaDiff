#!/bin/bash
##################################################################################
############ NEEDED VARIABLES TO SET BEFORE RUNNING THE SCRIPT  ##################
##################################################################################
export EXPERIMENT="auto"   # set the experiment name or "auto" for automatic naming
export SD_TYPE="conditional"     # set "conditional" for using embedding based on metadata, "naive" for no conditioning

# set the relevant columns to use from the metadata file. If "all", all columns will be used
export RELEVANT_COLUMNS='["dysangiogenesis","enlarged_capillaries",
                          "giant_capillaries","capillary_loss",
                          "microhemorrhages","scleroderma_pattern"]'

export CONVERT_TO_BOOLEAN=0  # set to 1 to convert conditions to boolean embeddings, 0 otherwise
export USE_TEXT_MODE=None  # set to "simple" to use simple text encoding or "None" for no text encoding

export NUM_TRAIN_EPOCHS=5
export MAX_TRAIN_STEPS=None # "None" if provided MAX_TRAIN_STEPS, this will override NUM_TRAIN_EPOCHS
export USE_CFG=False               # set to 1 to use classifier-free guidance during training
export CFG_TRAINING_PROB=0.1      # set the probability of using classifier-free guidance

export BATCH_SIZE=6
export GRADIENT_ACCUMULATION_STEPS=1
export LEARNING_RATE=1e-05
export MIXED_PRECISION="fp16"      # set to "fp16" or "bf16" for mixed precision training. 
export SCALE_MIN=0.9               # set the minimum scale for random resized cropping during training (between 0 and 1)

## set path to save the experiment outputs (models, logs, tmp files, etc.)
export SAVE_DIR="/cluster/work/medinfmk/capillaroscopy/CapillaDiff"   # if "None", the home directory will be used

## set the path to the training data directory. Folder contents must follow the structure described in
## https://github.com/marczuend/CapillaDiff/blob/main/README.md#data-preparation
export IMG_DIR="/cluster/work/medinfmk/capillaroscopy/content/images"
export METADATA_FILE="/cluster/customapps/medinfmk/mazuend/CapillaDiff/metadata/metadata_CapillaDiff_training.csv"


## set the path to the pretrained model, which could be either pretrained Stable Diffusion, or a pretrained CapillaDiff model
export MODEL_PATH="/cluster/customapps/medinfmk/mazuend/CapillaDiff/models/CapillaDiff_base"
export CLIP_PATH="/cluster/customapps/medinfmk/mazuend/CapillaDiff/models/clip-vit-large-patch14"

##################################################################################
######################### END OF VARIABLE SETTINGS ###############################
##################################################################################

# currently, checkpoint loading and validation is not working yet and are relicts from MorphoDiff
export CHECKPOINTING_STEPS=999999999 #high number so it will not be activated 
export VALIDATION_EPOCHS=$CHECKPOINTING_STEPS

# auto-name generation experiment if "auto" is set
if [ "$EXPERIMENT" == "auto" ]; then
    EXPERIMENT=""
    if [ "$USE_TEXT_MODE" == "None" ] || [ "$USE_TEXT_MODE" == "simple" ]; then
        if [ "$USE_TEXT_MODE" == "simple" ]; then
            EXPERIMENT="${EXPERIMENT}textmode_simple_"
        fi
        if [ $CONVERT_TO_BOOLEAN -eq 1 ]; then
            EXPERIMENT="${EXPERIMENT}bool_"
        else
            EXPERIMENT="${EXPERIMENT}level_"
        fi
    else
        EXPERIMENT="${EXPERIMENT}textmode_${USE_TEXT_MODE}_"
    fi

    EXPERIMENT="${EXPERIMENT}encoding_"

    EXPERIMENT="${EXPERIMENT}cfg_${USE_CFG}_"

    EXPERIMENT="${EXPERIMENT}run_1"

echo "Auto-generated experiment name: $EXPERIMENT"
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# If SAVE_DIR is set, override the default CapillaDiff directory
if [ "$SAVE_DIR" != "None" ]; then

    # check if SAVE_DIR contains "CapillaDiff" in its path
    if [[ "$SAVE_DIR" != *"CapillaDiff"* ]]; then
        # append "CapillaDiff" to SAVE_DIR
        SAVE_DIR="$SAVE_DIR/CapillaDiff"
    fi
else
    # go to home directory
    cd ~ || { echo "Could not change to home directory."; exit 1; }
    export SAVE_DIR="$(realpath ./CapillaDiff)"
fi

# CapillaDiff main directory
if [ ! -d "$SAVE_DIR" ]; then
    mkdir -p "$SAVE_DIR"
    echo "Created CapillaDiff directory"
fi

cd "$SAVE_DIR"

# Output directory
export OUTPUT_DIR="$SAVE_DIR/experiments/${EXPERIMENT}"

if [[ -d "$OUTPUT_DIR" ]] && [[ "$(ls -A "$OUTPUT_DIR")" ]]; then
    echo -e "\n\033[1;41mWARNING\033[0m Output directory '$OUTPUT_DIR' already exists and is not empty!"
    echo -e "\033[1;33mContinuing will overwrite existing files in this directory.\033[0m"
    echo
    echo "Options:"
    echo "  y = continue and overwrite"
    echo "  c = create a new folder"
    echo "  a = abort"
    read -p "Choose (y/c/a): " -n 1 -r
    echo    # new line

    if [[ $REPLY =~ ^[Aa]$ ]]; then
        echo "Process aborted by user."
        exit 1
    fi

    if [[ $REPLY =~ ^[Cc]$ ]]; then
        
        # Extract prefix (everything before _run_<number>)
        PREFIX="${OUTPUT_DIR%_run_*}"

        # Extract current number
        current_num="${OUTPUT_DIR##*_run_}"

        # Search all existing folders with same prefix
        last_num=$(ls -d ${PREFIX}_run_* 2>/dev/null | \
                grep -oP "${PREFIX}_run_\K[0-9]+" | \
                sort -n | \
                tail -1)

        # If none exist -> start at 1
        if [[ -z "$last_num" ]]; then
            next_num=1
        else
            next_num=$((last_num + 1))
        fi

        # Build next directory name
        NEXT_OUTPUT_DIR="${PREFIX}_run_${next_num}"

        OUTPUT_DIR="$NEXT_OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
        echo "New output directory created: $OUTPUT_DIR"
        echo "==============================================================="
    fi

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # remove existing files in the output directory
        rm -rf "$OUTPUT_DIR"/*
        echo "Continuing; existing folder will be overwritten."
        echo "==============================================================="
    fi
fi

# Checkpoint directory
export CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    mkdir -p "$CHECKPOINT_DIR"
    echo "Created checkpoints directory"
fi

# Temporary/cache directory
export TMPDIR="$OUTPUT_DIR/tmp"
if [ ! -d "$TMPDIR" ]; then
    mkdir -p "$TMPDIR"
    echo "Created tmp directory for training/cache"
fi

## Function to get column index by header name
get_column_index() {
    local header_line=$1
    local column_name=$2
    echo $(echo "$header_line" | tr ',' '\n' | nl -v 0 | grep "$column_name" | awk '{print $1}')
}

## check if CUDA is available
is_CUDA_AVAILABLE=$(python -c 'import torch; print(torch.cuda.is_available())')

echo "================= Setup Info =================================="
printf "%-20s : %s\n" "Experiment name" "$EXPERIMENT"
printf "%-20s : %s\n" "Relevant columns" "$RELEVANT_COLUMNS"
printf "%-20s : %s\n" "Model directory" "$MODEL_PATH"
echo "================= Data Info ==================================="
printf "%-20s : %s\n" "Image dir" "$IMG_DIR"
printf "%-20s : %s\n" "Metadata file" "$METADATA_FILE"
printf "%-20s : %s\n" "Output directory" "$OUTPUT_DIR"
printf "%-20s : %s\n" "Checkpoint directory" "$CHECKPOINT_DIR"
printf "%-20s : %s\n" "Temporary directory" "$TMPDIR"
echo "================= System Info ================================="
printf "%-20s : %s\n" "CUDA available" "$is_CUDA_AVAILABLE"
printf "%-20s : %s\n" "Python version" "$(python -V 2>&1)"
echo "================= Training Info ==============================="
printf "%-30s : %s\n" "Batch size" "$BATCH_SIZE"
printf "%-30s : %s\n" "Learning rate" "$LEARNING_RATE"
# if MAX_TRAIN_STEPS is not set, show NUM_TRAIN_EPOCHS
if [ "$MAX_TRAIN_STEPS" = "None" ]; then
    printf "%-30s : %s\n" "Num training epochs" "$NUM_TRAIN_EPOCHS"
else
    printf "%-30s : %s\n" "Max training steps" "$MAX_TRAIN_STEPS"
fi
#printf "%-30s : %s\n" "Validation epochs" "$VALIDATION_EPOCHS"
#printf "%-30s : %s\n" "Checkpointing steps" "$CHECKPOINTING_STEPS"
if [ "$USE_CFG" -eq 1 ]; then
    printf "%-30s : %s\n" "Using CFG during training" "Yes"
    printf "%-30s : %s\n" "CFG training probability" "$CFG_TRAINING_PROB"
else
    printf "%-30s : %s\n" "Using CFG during training" "No"
fi
echo "================= Label Augmentation ==========================="
if [ "$USE_TEXT_MODE" == "None" ] || [ "$USE_TEXT_MODE" == "simple" ]; then
    if [ "$USE_TEXT_MODE" == "simple" ]; then
        printf "%-30s : %s\n" "Use text mode encoding" "$USE_TEXT_MODE"
    fi
    if [ "$CONVERT_TO_BOOLEAN" -eq 1 ]; then
        printf "%-30s : %s\n" "Converting to boolean encoding" "Yes"
    else
        printf "%-30s : %s\n" "Converting to level encoding" "Yes"
    fi
else
    printf "%-30s : %s\n" "Use text mode encoding" "$USE_TEXT_MODE"
fi
printf "%-30s : %s\n" "Minimum scale for random resized crop" "$SCALE_MIN"
echo "==============================================================="

# Ask user if we should proceed
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo    # move to a new line
echo "==============================================================="

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Process aborted by user."
    exit 1
fi

echo "Starting training... train.py script launched via accelerate"
echo "==============================================================="

accelerate launch --mixed_precision=$MIXED_PRECISION $SCRIPT_DIR/../train.py \
--pretrained_model_path=$MODEL_PATH \
--naive_conditional=$SD_TYPE \
--img_data_dir=$IMG_DIR \
--metadata_file_path=$METADATA_FILE \
--output_dir=$OUTPUT_DIR \
--checkpointing_steps=$CHECKPOINTING_STEPS \
--report_to_wandb=True \
--train_batch_size=$BATCH_SIZE \
--num_train_epochs=$NUM_TRAIN_EPOCHS \
--max_train_steps=$MAX_TRAIN_STEPS \
--learning_rate=$LEARNING_RATE \
--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--seed=42 \
--cache_dir=$TMPDIR \
--enable_xformers_memory_efficient_attention \
--resolution=512 \
--use_ema=True \
--convert_to_boolean=$CONVERT_TO_BOOLEAN \
--use_cfg=$USE_CFG \
--cfg_training_prob=$CFG_TRAINING_PROB \
--text_mode=$USE_TEXT_MODE \
--clip_path=$CLIP_PATH \
--scale_min=$SCALE_MIN \
--relevant_columns="$RELEVANT_COLUMNS"