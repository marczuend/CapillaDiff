import argparse
import os
from pathlib import Path
import shutil

import torch
from torchvision import models
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from tqdm import tqdm

def find_capilladiff(start_path):
    """Recursively search upward for a parent folder named 'CapillaDiff'."""
    target = "CapillaDiff"
    current = Path(start_path).resolve()

    for parent in [current] + list(current.parents):
        if parent.name == target:
            return parent

    return None

def check_internet_connection(save_dir):
    """
    Tests if HuggingFace Hub is reachable by attempting to download
    a tiny known file. Does NOT store the file permanently.
    """
    try:
        tmp_path = os.path.join(save_dir, "temp_hf_check")
        print("Checking internet connection to HuggingFace Hub...")

        hf_hub_download(
            repo_id="bert-base-uncased",
            filename="README.md",
            local_dir=tmp_path
        )

        shutil.rmtree(tmp_path, ignore_errors=True)
        print("Internet connection is available.")
        return True

    except Exception as e:
        print(f"Internet check failed: {e}")
        print("To run this script, an internet connection is required.")
        exit(1)

def download_inception(save_dir, folder_name="inception_v3"):
    """
    Downloads the official torchvision Inception-v3 weights
    and saves them to the specified directory.
    """

    inception = models.inception_v3(pretrained=True)
    inception.fc = torch.nn.Identity()  # pool3 features
    inception.eval()

    weights_enum = models.Inception_V3_Weights.DEFAULT
    # Download the weights for custom FID/KID calculation
    state_dict = torch.hub.load_state_dict_from_url(weights_enum.url, progress=True)

    save_dir = os.path.join(save_dir, folder_name)
    file_name = "inception_v3.pth"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(save_dir, file_name))

    # Download the model for official FID/KID calculation
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    os.environ["TORCH_HOME"] = save_dir + "/torchmetrics"
    fid = FrechetInceptionDistance(feature=2048)
    kid = KernelInceptionDistance(subset_size=1000, feature=2048)

    print(f"Inception-v3 weights downloaded and saved to {save_dir}")

def download_CapillaDiff_base(save_dir, folder_name="CapillaDiff_base"):
    """
    Downloads the current MorphoDiff model from HuggingFace Hub, saves it to the specified directory
    and restructures the files so it works as the CapillaDiff base model.
    """

    MorphoDiff_checkpoint_name = "bbbc021_14_compounds_morphodiff_ckpt"
    base_model_save_path = os.path.join(save_dir, folder_name)
    print("Downloading CapillaDiff_base...")

    hf_hub_download(
        repo_id="navidi/MorphoDiff_checkpoints",
        filename=MorphoDiff_checkpoint_name + ".zip",
        local_dir=save_dir
    )

    # Unzip the downloaded file
    print("Unzipping CapillaDiff_base...")
    shutil.unpack_archive(
        os.path.join(save_dir, MorphoDiff_checkpoint_name + ".zip"),
        base_model_save_path
    )

    # Remove the zip file
    os.remove(os.path.join(save_dir, MorphoDiff_checkpoint_name + ".zip"))

    ##### Restucture the folder order
    old_folders = os.listdir(base_model_save_path)

    # copy all files from subfolder "checkpoint" to base_model_save_path
    checkpoint_subfolder = os.path.join(base_model_save_path, MorphoDiff_checkpoint_name, "checkpoint")
    for item in tqdm(os.listdir(checkpoint_subfolder), desc="Restructuring CapillaDiff_base"):
        source = os.path.join(checkpoint_subfolder, item)
        destination = os.path.join(base_model_save_path, item)
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    # remove the now empty and unncessary folders
    for folder in old_folders:
        shutil.rmtree(os.path.join(base_model_save_path, folder), ignore_errors=True)
    
    # copy folder "unet_ema" and name it "unet"
    print("Restructuring UNet files...")
    source_unet_ema = os.path.join(base_model_save_path, "unet_ema")
    destination_unet = os.path.join(base_model_save_path, "unet")
    shutil.copytree(source_unet_ema, destination_unet, dirs_exist_ok=True)
    shutil.rmtree(source_unet_ema, ignore_errors=True)

    print("Restructuring complete.")
    print(f"CapillaDiff_base downloaded and saved to {base_model_save_path}")

def main():
    parser = argparse.ArgumentParser(description="Download needed models and weights.")
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default=None,
        help="Directory where the downloaded weights will be stored."
    )

    args = parser.parse_args()

    if args.save_dir is None:
        # get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        capilladiff_main_dir = find_capilladiff(script_dir)

        # Go one level up to CapillaDiff main directory
        parent_dir = os.path.dirname(capilladiff_main_dir)
        args.save_dir = os.path.join(parent_dir, "models")

    # ask user if saving directory is correct
    print(f"Models will be saved to: {args.save_dir}")
    confirm = input("Do you want to continue (y/n): ")
    if confirm.lower() != 'y':
        print("Exiting. Please rerun the script with the correct --save-dir argument.")
        exit(0)

    # Ensure save directory exists, if not, create it
    os.makedirs(args.save_dir, exist_ok=True)

    # Check internet connectivity
    check_internet_connection(args.save_dir)



    folder_name="CapillaDiff_base"
    if os.path.exists(os.path.join(args.save_dir, folder_name)):
        print("CapillaDiff_base already exist. Skipping download.")
    else:
        download_CapillaDiff_base(args.save_dir, folder_name)

    folder_name="inception_v3"
    if os.path.exists(os.path.join(args.save_dir, folder_name)):
        print("Inception-v3 weights already exist. Skipping download.")
    else:
        download_inception(args.save_dir, folder_name)

    # Get clip-vit-large-patch14 model from HuggingFace Hub
    folder_name="clip-vit-large-patch14"
    if os.path.exists(os.path.join(args.save_dir, folder_name)):
        print("CLIP model already exist. Skipping download.")
    else:
        print("Downloading CLIP model...")
        clip_model_save_path = os.path.join(args.save_dir, folder_name)
        snapshot_download(
            repo_id="openai/" + folder_name,
            local_dir=clip_model_save_path
        )
        print(" /n")
        print(f"CLIP model downloaded and saved to {clip_model_save_path}")

    # Get Medical BERT model from HuggingFace Hub
    folder_name="ClinicalBERT"
    if os.path.exists(os.path.join(args.save_dir, folder_name)):
        print("Medical BERT model already exist. Skipping download.")
    else:
        print("Downloading Medical BERT model...")
        medical_bert_model_save_path = os.path.join(args.save_dir, folder_name)
        snapshot_download(
            repo_id="medicalai/" + folder_name,
            local_dir=medical_bert_model_save_path
        )
        print(" /n")
        print(f"Medical BERT model downloaded and saved to {medical_bert_model_save_path}")

    # delete ALL cache folders
    print("Cleaning up cache folders...")
    for root, dirs, files in os.walk(args.save_dir):
        for dir_name in dirs:
            if dir_name == ".cache":
                cache_path = os.path.join(root, dir_name)
                shutil.rmtree(cache_path, ignore_errors=True)

    print("==============================================================")
    print("============ All downloads completed successfully ============")
    print("==============================================================")


if __name__ == "__main__":
    main()