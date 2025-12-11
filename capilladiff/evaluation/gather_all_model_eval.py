#!/usr/bin/env python
# coding=utf-8

"""Gather all model evaluation results into a single CSV file."""

import os
import glob
import pandas as pd
import argparse

def find_all_model_eval(root):
    """
    Recursively find all model evaluation results in `root` and its subfolders.
    Supports common formats: csv
    """
    extensions = ("image_quality_metrics.csv",)
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return files


def gather_all_model_eval(eval_results_dir, file_name="image_quality_metrics.csv"):
    """Gathers all model evaluation results from CSV files in a directory into a single CSV file.
    
    Args:
        eval_results_dir (str): Directory containing individual model evaluation CSV files.
    
    Returns:
        pd.DataFrame: DataFrame containing all gathered evaluation results.
    """

    all_files = find_all_model_eval(eval_results_dir)
    df_list = []

    for file in all_files:
        experiment_name = os.path.basename(os.path.dirname(file))

        df = pd.read_csv(file)
        df['Experiment'] = experiment_name
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gather all model evaluation results into a single CSV file.")
    parser.add_argument("--eval_results_dir", type=str, default=None,
                        help="Directory containing individual model evaluation CSV files.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output CSV file to save the gathered evaluation results.")
    args = parser.parse_args()

    if args.eval_results_dir is None:
        eval_results_dir = "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/generated_imgs/evaluation"
    else:
        eval_results_dir = args.eval_results_dir

    if args.output_file is None:
        output_file = "/cluster/work/medinfmk/capillaroscopy/CapillaDiff/generated_imgs/evaluation/all_model_evaluation_results.csv"
    else:
        output_file = args.output_file

    combined_df = gather_all_model_eval(eval_results_dir)

    combined_df.to_csv(output_file, index=False)
    print(f"Gathered evaluation results saved to {output_file}")
