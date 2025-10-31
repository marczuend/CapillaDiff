#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --qos=m2
#SBATCH --time=8:00:00
#SBATCH --job-name=cp_analysis
#SBATCH --error=out_dir/%x-%j.err
#SBATCH --output=out_dir/%x-%j.out

# get input arguments
preprocessing_type=$1
feature_type=$2
experiment=$3
dataset=$4

python analyze_processed_CP_features.py \
    --preprocessing_type $preprocessing_type \
    --feature_type $feature_type \
    --experiment $experiment \
    --dataset $dataset
