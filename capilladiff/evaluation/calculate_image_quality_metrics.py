#!/usr/bin/env python
# coding=utf-8

import os
import glob
import pandas as pd
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm

from torch import nn
from torchvision import models
from scipy import linalg
from PIL import Image
from torchvision import transforms


def str2bool(v):
    """Convert string to boolean.

    Args:
        v (str): string to convert to boolean
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'TRUE', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'FALSE', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    return

def set_seed(seed):
    """Set seed for reproducibility.

    Args:
        seed (int): seed for reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return

def find_images(root):
    """
    Recursively find all images in `root` and its subfolders.
    Supports common formats: PNG, JPG, JPEG.
    """
    extensions = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return files


def calculate_metrics(ref_imgs, gen_imgs, eval_model, batch_size, use_official=True, kid_subset_size=1000):
    """Calculate FID and KID for the generated images.

    Args:
        ref_imgs (list): list of paths to reference images
        gen_imgs (list): list of paths to generated images
        eval_model (torch.nn.Module, optional): evaluation model
        batch_size (int): batch size for evaluation
        use_official (bool): whether to use official torchmetrics implementation
        kid_subset_size (int): subset size for KID calculation
    """

    fid_score, kid_mean, kid_std = None, None, None
    if use_official:
        print("Calculating FID and KID using official torchmetrics...")
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance

        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception-v3 input size
            transforms.PILToTensor()        # produces uint8, NOT normalized
        ])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fid = FrechetInceptionDistance(feature=2048).to(device)
        kid = KernelInceptionDistance(subset_size=kid_subset_size, feature=2048).to(device)

        # After having the models loaded, we change the TORCH_HOME back to original if it was set to ensure no conflicts
        global ORG_TORCH_HOME
        if ORG_TORCH_HOME is not None:
            os.environ["TORCH_HOME"] = ORG_TORCH_HOME

        # Process reference images
        for i in tqdm(range(0, len(ref_imgs), batch_size), desc="Processing Reference Images", unit="batch"):
            images = [transform(Image.open(p).convert('RGB')) for p in ref_imgs[i:i + batch_size]]
            batch = torch.stack(images).to(device)
            fid.update(batch, real=True)
            kid.update(batch, real=True)
        
        # Process generated images
        for i in tqdm(range(0, len(gen_imgs), batch_size), desc="Processing Generated Images", unit="batch"):
            images = [transform(Image.open(p).convert('RGB')) for p in gen_imgs[i:i + batch_size]]
            batch = torch.stack(images).to(device)
            fid.update(batch, real=False)
            kid.update(batch, real=False)

        fid_score = fid.compute().item()
        kid_mean_value, kid_std_value = kid.compute()
        kid_mean = kid_mean_value.item()
        kid_std = kid_std_value.item()

        # cleanup of torchmetrics cache
        fid.reset()
        kid.reset()

        print(f"Calculated Metrics: FID={fid_score}, KID_mean={kid_mean}, KID_std={kid_std}")

    else:
        print("Calculating FID using custom implementation...")
        # load images

        def get_features(images_path, model, batch_size):
            """Extract features from images using the evaluation model."""

            transform = transforms.Compose([
                transforms.Resize((348, 348)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.67212146, 0.63119322, 0.62765121],  # mean
                    [0.08998639, 0.11100586, 0.12605950]   # std
                )
            ])


            device = next(model.parameters()).device
            features = []

            for i in tqdm(range(0, len(images_path), batch_size), desc="Extracting Features", unit="batch"):
                # Load batch in a list comprehension (faster)
                images = [transform(Image.open(p).convert('RGB')) for p in images_path[i:i + batch_size]]
                batch = torch.stack(images).to(device)

                with torch.no_grad():
                    batch_features = model(batch).cpu().numpy()

                features.append(batch_features)

            return np.concatenate(features)

        def calculate_fid(real_features, generated_features):
            mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
            diff = mu1 - mu2
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return fid

        def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1.0):
            if gamma is None:
                gamma = 1.0 / x.size(1)
            return (gamma * x @ y.t() + coef0) ** degree

        def unbiased_mmd2(kernel_xx, kernel_xy, kernel_yy):
            m = kernel_xx.size(0)

            # Remove diagonals for unbiased version
            diag_x = torch.diag(kernel_xx)
            diag_y = torch.diag(kernel_yy)

            k_xx_sum = kernel_xx.sum() - diag_x.sum()
            k_yy_sum = kernel_yy.sum() - diag_y.sum()
            k_xy_sum = kernel_xy.sum()

            return (k_xx_sum + k_yy_sum - 2 * k_xy_sum) / (m * (m - 1))

        def kid(real_features, gen_features, subset_size=kid_subset_size):
            real_features = torch.tensor(real_features)
            gen_features = torch.tensor(gen_features)

            m = real_features.size(0)
            n = gen_features.size(0)

            results = []

            for _ in range(0, len(real_features), subset_size):
                idx_r = torch.randint(0, m, (subset_size,))
                idx_g = torch.randint(0, n, (subset_size,))

                r = real_features[idx_r]
                g = gen_features[idx_g]

                k_xx = polynomial_kernel(r, r)
                k_yy = polynomial_kernel(g, g)
                k_xy = polynomial_kernel(r, g)

                results.append(unbiased_mmd2(k_xx, k_xy, k_yy))

            return torch.stack(results).mean().item(), torch.stack(results).std().item()

        ref_features = get_features(ref_imgs, eval_model, batch_size)
        gen_features = get_features(gen_imgs, eval_model, batch_size)

        fid_score = calculate_fid(ref_features, gen_features)
        kid_mean, kid_std = kid(ref_features, gen_features, subset_size=kid_subset_size)

        print(f"Calculated Metrics: FID={fid_score}, KID_mean={kid_mean}, KID_std={kid_std}")

    return fid_score, kid_mean, kid_std

def evaluate_model(ref_img_path, gen_img_path, eval_model, batch_size, sub_set_size=None, kid_subset_size=1000,
                   use_official=True, use_custom=False):
    """Evaluate a single model by calculating FID and KID.
    Args:
        ref_img_path (str): path to the reference images
        gen_img_path (str): path to the generated images
        eval_model (torch.nn.Module): evaluation model
        batch_size (int): batch size for evaluation
        sub_set_size (int, optional): set to >0 to evaluate only on a subset of the images
        kid_subset_size (int): subset size for KID calculation
        use_official (bool): whether to use official torchmetrics implementation
        use_custom (bool): whether to use custom implementation
    Returns:
        official_metrics (dict): dictionary containing official FID and KID metrics
        custom_metrics (dict): dictionary containing custom FID and KID metrics
    """

    # load paths to images
    ref_imgs = find_images(ref_img_path)
    model_imgs = find_images(gen_img_path)

    # compare number of images
    if sub_set_size is not None and sub_set_size > 0:
        ref_imgs = random.sample(ref_imgs, min(sub_set_size, len(ref_imgs)))
        model_imgs = random.sample(model_imgs, min(sub_set_size, len(model_imgs)))
    else:
        if len(ref_imgs) > len(model_imgs):
            ref_imgs = random.sample(ref_imgs, len(model_imgs))
        else: # len(ref_imgs) < len(model_imgs)
            model_imgs = random.sample(model_imgs, len(ref_imgs))

    # calculate official FID, KID
    metrics = {}
    if use_official is True:
        fid_score, kid_mean, kid_std = calculate_metrics(ref_imgs, model_imgs, eval_model, batch_size, use_official=True, kid_subset_size=kid_subset_size)

        # add to metrics dictionary
        metrics['Official_FID'] = fid_score
        metrics['Official_KID_mean'] = kid_mean
        metrics['Official_KID_std'] = kid_std

    # calculate custom FID, KID
    if use_custom is True:
        fid_score, kid_mean, kid_std = calculate_metrics(ref_imgs, model_imgs, eval_model, batch_size, use_official=False, kid_subset_size=kid_subset_size)

        metrics['Custom_FID'] = fid_score
        metrics['Custom_KID_mean'] = kid_mean
        metrics['Custom_KID_std'] = kid_std

    return metrics

def parse_args():

    """Parse input arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--ref_img_path',
                        required=True,
                        help="directory address of real images")
    parser.add_argument('--gen_img_path',
                        required=True,
                        help="directory address of the generated images")
    parser.add_argument('--result_path',
                        default=None,
                        help="path to save test results")
    parser.add_argument('--experiment',
                        default=None,
                        help="experiment name")
    parser.add_argument('--seed',
                        default=42,
                        help="random seed for reproducibility")
    parser.add_argument('--eval_model_path',
                        required=True,
                        help="path to the evaluation model weights or 'inception' for default Inception-v3")
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help="batch size for evaluation")
    parser.add_argument('--sub_set_size',
                        type=int,
                        default=None,
                        help="set to >0 to evaluate only on a subset of the images")
    parser.add_argument('--kid_subset_size',
                        type=int,
                        default=1000,
                        help="subset size for KID calculation")
    parser.add_argument('--use_official',
                        type=str2bool,
                        default=True,
                        help="whether to use official torchmetrics implementation for FID and KID calculation")
    parser.add_argument('--use_custom',
                        type=str2bool,
                        default=True,
                        help="whether to use custom implementation for FID and KID calculation")
    
    return parser.parse_args()


def main():

    args = parse_args()
    if args.seed == 'random':
        args.seed = random.randint(1, 10000)
    else:
        args.seed = int(args.seed)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_official is True:
        # Set TORCH_HOME to the directory containing the Inception-v3 weights for official FID/KID calculation
        torch_temp_dir = os.path.dirname(os.path.abspath(args.eval_model_path))
        torch_temp_dir = os.path.join(torch_temp_dir, "torchmetrics")

        global ORG_TORCH_HOME
        ORG_TORCH_HOME = os.environ.get("TORCH_HOME", None)
        os.environ["TORCH_HOME"] = torch_temp_dir

    # loading custom evaluation model
    custom_model_name = None
    model = None
    if args.use_custom:
        if 'inception' in args.eval_model_path.lower() and 'v3' in args.eval_model_path.lower():
            inception_path = os.path.join(args.eval_model_path, "inception_v3.pth")
            state_dict = torch.load(inception_path, map_location="cpu")
            inception = models.inception_v3(weights=None, init_weights=False)
            inception.load_state_dict(state_dict)
            inception.fc = torch.nn.Identity() # Remove last layer to get features
            model = inception.eval().to(device)

            custom_model_name = "Inception-v3"
            print(f"Using Inception-v3 model for custom evaluation.")
        else:
            # Get file in eval_model_path
            if os.path.isfile(args.eval_model_path):
                weight_file = args.eval_model_path
            else:
                # List all files (ignore subdirectories)
                files = [f for f in glob.glob(os.path.join(args.eval_model_path, "*")) if os.path.isfile(f)]
                if len(files) == 0:
                    raise FileNotFoundError(f"No files found in {args.eval_model_path}")
                weight_file = files[0]  # pick the first file
            raise NotImplementedError(f"Evaluation model {weight_file} not implemented.")

    metrics = evaluate_model(
        args.ref_img_path,
        args.gen_img_path,
        model,
        args.batch_size,
        args.sub_set_size,
        args.kid_subset_size,
        args.use_official,
        args.use_custom
    )

    # save results to a csv file in gen_img_path
    if args.result_path is None:
        args.result_path = args.gen_img_path

    # save all results to a csv file
    metrics['Seed'] = args.seed
    if custom_model_name is not None:
        metrics['Custom_Evaluation_Model'] = custom_model_name
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(os.path.join(args.result_path, "image_quality_metrics.csv"), index=False)

    print("Evaluation completed. Results saved to:", args.result_path)

if __name__ == "__main__":
    main()
