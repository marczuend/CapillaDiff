#!/usr/bin/env python3

import argparse
import os
import re
import csv
from tqdm import tqdm

# Regex pattern:
# - steps captured from "32/10000"
# - lr captured from "lr=1e-5" or even malformed ones like "lr=1e"
# - step_loss captured from "step_loss=0.122"
PATTERN = re.compile(
    r"Steps:.*?(\d+)/\d+.*?lr=([0-9.eE+-]+).*?step_loss=([0-9.eE+-]+)"
)

def safe_float(val):
    """Safely convert strings like 1e-5, 1e, .001, etc. Returns None if invalid."""
    try:
        return float(val)
    except ValueError:
        return None


def extract_progress_info(filepath):
    results = []

    # Count lines first for progress bar
    with open(filepath, "r", errors="ignore") as f:
        total_lines = sum(1 for _ in f)

    with open(filepath, "r", errors="ignore") as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            match = PATTERN.search(line)
            if match:
                step = int(match.group(1))
                lr_raw = match.group(2)
                step_loss_raw = match.group(3)

                lr = safe_float(lr_raw)
                step_loss = safe_float(step_loss_raw)

                # Skip broken lines (e.g., lr=1e)
                if lr is None or step_loss is None:
                    continue

                results.append({
                    "step": step,
                    "lr": lr,
                    "step_loss": step_loss
                })

    return results

def cleanup_doubled_loss(data):
    """Remove entries where step_loss is doubled consecutively."""
    cleaned_data = []
    prev_loss = 0.0
    prev_step = 1
    for entry in data:
        #print("prev_step:", prev_step, "current_step:", entry["step"], "prev_loss:", prev_loss, "current_loss:", entry["step_loss"])
        if not((prev_step == (entry["step"] - 1)) and (entry["step_loss"] == prev_loss)):
            cleaned_data.append(entry)
        prev_loss = entry["step_loss"]
        prev_step = entry["step"]
    return cleaned_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract step, lr, and step_loss from W&B tqdm logs."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the log file (.txt, .log, etc.)")
    parser.add_argument("--outdir", "-o", required=False,
                        help="Directory to save output CSV")

    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}")
        return

    # Determine output directory
    outdir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(input_path))
    os.makedirs(outdir, exist_ok=True)

    # Output file path
    output_path = os.path.join(outdir, "wandb_loss_log.csv")

    print("Extracting Loss from WandB logs...")
    data = extract_progress_info(input_path)

    if not data:
        print("No matching or valid lines found.")
        return

    data = cleanup_doubled_loss(data)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "lr", "step_loss"])
        writer.writeheader()
        writer.writerows(data)

    print(f"Loss-Log file saved to: {output_path}")


if __name__ == "__main__":
    main()
