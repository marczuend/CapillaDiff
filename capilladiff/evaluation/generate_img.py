#!/usr/bin/env python
# coding=utf-8

from diffusers import StableDiffusionPipeline, DDPMScheduler
import os
import torch
import datetime
import argparse
from transformers import CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel
from typing import Optional
import pandas as pd
import random
import numpy as np
from tqdm import tqdm

# Local imports
import sys
# Add parent folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CapillaDiff_encoder import ConditionEncoderInference, ConditionEncoder, DictToListEncoder
from CapillaDiff_dataloader import DatasetLoader

def str2bool(v):
    """Convert string to boolean.

    Args:
        v (str): string to convert to boolean"""

    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'TRUE', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'FALSE', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    return


class CapillaDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae,
                 text_encoder,
                 unet,
                 scheduler,
                 feature_extractor,
                 tokenizer=None,
                 safety_checker=None,
                 image_encoder=None,
                 requires_safety_checker=False):
        super().__init__(vae=vae,
                         text_encoder=text_encoder,
                         tokenizer=tokenizer,
                         unet=unet,
                         scheduler=scheduler,
                         safety_checker=safety_checker,
                         feature_extractor=feature_extractor,
                         image_encoder=image_encoder,
                         requires_safety_checker=requires_safety_checker)
        self.custom_text_encoder = text_encoder

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        # decode prompt using custom text encoder
        condition = DictToListEncoder().decode(prompt)
        embeddings = self.custom_text_encoder(condition)

        if num_images_per_prompt > 1:
            embeddings = embeddings.repeat(num_images_per_prompt, 1, 1)  # (num_images_per_prompt, seq_len, embed_dim)

        embeddings = embeddings.to(device)
        return embeddings, None


def set_seed(seed):
    """Set seed for reproducibility.

    Args:
        seed (int): seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return


def load_model_and_generate_images(pipeline, conditions,
                                   gen_img_path, overwrite_existing=False, batch_size=1):
    """Load the model and generate images for the given prompts.

    Args:
        pipeline: CapillaDiffusionPipeline
        conditions (list): contains condition name, number of images per condition, condition values
        gen_img_path (str): path to save generated images
    """

    # check batch size
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    total_num_imgs = sum([cond['num_imgs'] for cond in conditions])

    print(f"Total number of images to generate: {total_num_imgs}")

    # disable progress bar of pipeline
    from tqdm.auto import tqdm
    pipeline.progress_bar = lambda *args, **kwargs: tqdm(disable=True)

    for condition in tqdm(conditions, desc="Over all Conditions", unit="cond"):

        condition_name = condition.pop('condition_name')
        num_imgs = condition.pop('num_imgs')
        if num_imgs <= 0: continue

        img_series_dir = gen_img_path + "/" + condition_name

        # turn dictionary into list, so it gets accepted by pipeline
        prompt = DictToListEncoder().encode(condition)

        if not os.path.exists(img_series_dir):
            os.makedirs(img_series_dir)

        start = datetime.datetime.now()

        if overwrite_existing:
            for img in os.listdir(img_series_dir):
                os.remove(os.path.join(img_series_dir, img))

        imgs = os.listdir(img_series_dir)

        with tqdm(total=num_imgs, desc=f"{condition_name}", unit="img") as pbar:
            pbar.update(len(imgs))

            while len(imgs) < num_imgs:

                batch_size_eff = min(batch_size, num_imgs - len(imgs))
                output = pipeline(prompt=prompt, num_images_per_prompt=batch_size_eff)

                 # save all images in batch
                for i in range(batch_size_eff):
                    image_name = f'{condition_name}--{len(imgs) + i}.png'
                    image_path = os.path.join(img_series_dir, image_name)
                    output.images[i].save(image_path)

                pbar.update(batch_size_eff)
                imgs = os.listdir(img_series_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="CapillaDiff Image Generation")
    
    parser.add_argument('--model_checkpoint',
                        required=True,
                        help="a name for identifying the model")
    parser.add_argument('--metadata_file_path',
                        default=None, 
                        help="path to the metadata csv file on which the model was trained")
    parser.add_argument('--convert_to_boolean',
                        type=str2bool, nargs='?',
                        default=None,
                        help="whether to convert condition to boolean embedding. is given in training_config.json of trained model")
    parser.add_argument('--text_mode',
                        default="",
                        help="text mode used during training. is given in training_config.json of trained model")
    parser.add_argument('--clip_path',
                        default=None,
                        help="path to the CLIP model used for text encoding.")
    parser.add_argument('--condition_list_address',
                        required=True,
                        help="a file with a list of all condition combinations")
    parser.add_argument('--gen_img_path', 
                        default=None,
                        help="path to save generated images")
    parser.add_argument('--num_imgs', default=3,
                        type=int,
                        help="number of images to generate")
    parser.add_argument('--total_num_imgs',
                        default=None,
                        type=int,
                        help="total number of images to generate. if set, overrides --num_imgs")
    parser.add_argument('--img_distribution',
                        default='uniform',
                        help="image distribution per condition strategy: 'uniform', 'proportional' or 'inverse_proportional'")
    parser.add_argument('--experiment',
                        required=True,
                        help="experiment name")
    parser.add_argument('--model_type', default='conditional',
                        help="model type: conditional or naive")
    parser.add_argument('--overwrite_existing',
                        type=str2bool, nargs='?',
                        default=False,
                        help="whether to overwrite existing images in the output directory")
    parser.add_argument('--seed', 
                        type=str,
                        default='42',
                        help="random seed for reproducibility. Set to 'random' for a random seed.")
    parser.add_argument('--batch_size', 
                        type=int,
                        default=1,
                        help="batch size for image generation.")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # enable reproducibility
    if args.seed == 'random':
        args.seed = random.randint(0, 1000000)
        #print(f"Using random seed: {args.seed}")
    else:
        args.seed = int(args.seed)
    set_seed(args.seed)

    # read training_config for model
    try:
        import json
        model_config_path = os.path.join(
            args.model_checkpoint,
            'training_config.json')
        with open(model_config_path, 'r') as f:
            training_config = json.load(f)
        

        if args.convert_to_boolean is None:
            args.convert_to_boolean = training_config.get('convert_to_boolean', False)
        if args.metadata_file_path == "None" or args.metadata_file_path is None:
            args.metadata_file_path = training_config.get('metadata_file_path', None)
        if  args.text_mode == "":
            args.text_mode = training_config.get('text_mode', None)
        if args.text_mode == "None" or args.text_mode is None:
            args.text_mode = None
        if args.text_mode is not None:
            args.clip_path = training_config.get('clip_path', None)

    except Exception as e:
        raise Exception(f"Error reading training_config.json: {e}")

    if args.gen_img_path is None:
        args.gen_img_path = os.path.join(
            os.path.expanduser('~'),
            'CapillaDiff',
            'generated_images',
            args.experiment)

    if not os.path.exists(args.gen_img_path):
        os.makedirs(args.gen_img_path)
        print(f"Created directory for generated images: {args.gen_img_path}")

    # initialize SD model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.model_checkpoint+'/feature_extractor')
    print('Loaded feature_extractor')

    vae = AutoencoderKL.from_pretrained(
        args.model_checkpoint, subfolder="vae")
    print('Loaded vae model')

    unet = UNet2DConditionModel.from_pretrained(
        args.model_checkpoint, subfolder="unet_ema", use_auth_token=True, ignore_mismatched_sizes=True)
    print('Loaded EMA unet model')

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_checkpoint, subfolder="scheduler")
    print('Loaded noise_scheduler')

    # check if text mode is used
    clip = None
    if args.text_mode is not None:
        # load clip model
        from transformers import CLIPTokenizer, CLIPTextModel
        clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_path, local_files_only=True)
        clip_text_encoder = (CLIPTextModel.from_pretrained(args.clip_path, local_files_only=True)).to(device)
        clip = (clip_tokenizer, clip_text_encoder, device)
        print('Loaded CLIP model for text encoding')

    # dataset_id, cluster, model
    custom_encoder = ConditionEncoderInference(
        model_type=args.model_type,
        convert_to_boolean=args.convert_to_boolean,
        text_mode=args.text_mode,
        clip=clip
    )
    print('Loaded custom_text_encoder')

    # Initialize your custom pipeline
    pipeline = CapillaDiffusionPipeline(
        vae=vae,
        unet=unet,
        text_encoder=custom_encoder,
        feature_extractor=feature_extractor,
        scheduler=noise_scheduler)
    print('Initialized pipeline')

    pipeline.to(device)

    # Get images with labels
    dataset = DatasetLoader(
        img_folder_path = "", # not needed for image generation
        metadata_csv_path = args.metadata_file_path,
        #relevant_columns: list = None
        )

    dataset_info = dataset.get_dataset_info(custom_encoder)

    # decide on how mayny images to generate per condition
    if args.total_num_imgs is None or args.total_num_imgs <= 0:
        # add column with num_imgs per condition
        for cond in dataset_info['all_conditions']:
            cond['num_imgs'] = int(args.num_imgs)

    else:

        def allocate_counts(fractional_values, total_count):
            """
            Convert fractional allocation values into integers whose sum equals total_count.
            Uses the 'largest remainder method'.
            
            fractional_values: list of floats (weights * target_total)
            total_count: final number of items to allocate

            Returns: list of integers
            """

            # Step 1 — floor values
            base = [int(x) for x in fractional_values]

            # Step 2 — compute remaining images to distribute
            remainder = total_count - sum(base)
            if remainder < 0:
                raise ValueError("Total of floored values exceeds total_count, cannot allocate.")

            # Step 3 — compute fractional remainders
            fracs = [(x - int(x), idx) for idx, x in enumerate(fractional_values)]

            # Step 4 — give leftover counts to highest fractional parts
            fracs.sort(reverse=True)  # highest remainder first
            for i in range(remainder):
                idx = fracs[i][1]
                base[idx] += 1

            return base
        
        num_conditions = len(dataset_info['all_conditions'])
        if args.img_distribution == 'uniform':
            
            # ensure at least one image per condition
            if num_conditions > args.total_num_imgs:
                imgs_per_condition = 1
            else:
                imgs_per_condition = args.total_num_imgs / num_conditions
            for cond in dataset_info['all_conditions']:
                cond['num_imgs'] = int(imgs_per_condition)

        elif args.img_distribution == 'proportional':
            fractional = [cond['relative_frequency'] * args.total_num_imgs for cond in dataset_info['all_conditions']]
            normalized_fractional = [x / sum(fractional) * args.total_num_imgs for x in fractional]

            allocations = allocate_counts(normalized_fractional, args.total_num_imgs)

            for cond, count in zip(dataset_info['all_conditions'], allocations):
                cond['num_imgs'] = count

        elif args.img_distribution == 'inverse_proportional':
            # calculate inverse proportional frequencies
            total_inverse_freq = [1.0 / cond['relative_frequency'] for cond in dataset_info['all_conditions']]
            fractional = [(freq / num_conditions) * args.total_num_imgs for freq in total_inverse_freq]
            normalized_fractional = [x / sum(fractional) * args.total_num_imgs for x in fractional]

            allocations = allocate_counts(normalized_fractional, args.total_num_imgs)

            for cond, count in zip(dataset_info['all_conditions'], allocations):
                cond['num_imgs'] = count

    columns_to_keep = dataset.get_relevant_columns() + ['condition_name', 'num_imgs']
    conditions = [
        {col: cond[col] for col in columns_to_keep}
        for cond in dataset_info['all_conditions']
    ]

    # filter conditions based on model type
    if args.model_type == 'naive':
        conditions = conditions[:1]
        for col in dataset.get_relevant_columns():
            conditions[0][col] = 1.0
        conditions[0]['condition_name'] = 'naive'
    
    elif args.model_type != 'conditional':
        raise Exception("Model type not recognized.")

    # keep only conditions with num_imgs > 0
    conditions = [cond for cond in conditions if cond['num_imgs'] > 0]

    # copy conditions for logging
    conditions_log_df = pd.DataFrame(conditions)

    print("==========================================================")
    print("============== Starting image generation ================")
    print("==========================================================")

    load_model_and_generate_images(
        pipeline,
        conditions,
        args.gen_img_path,
        overwrite_existing=args.overwrite_existing,
        batch_size=args.batch_size
    )

    # save a config file with generation settings
    generation_config = {
        'model_checkpoint': args.model_checkpoint,
        'metadata_file_path': args.metadata_file_path,
        'convert_to_boolean': args.convert_to_boolean,
        'text_mode': args.text_mode,
        'condition_list_address': args.condition_list_address,
        'gen_img_path': args.gen_img_path,
        'num_imgs': args.num_imgs,
        'total_num_imgs': args.total_num_imgs,
        'img_distribution': args.img_distribution,
        'experiment': args.experiment,
        'model_type': args.model_type,
        'overwrite_existing': args.overwrite_existing,
        'seed': args.seed
    }
    config_save_path = os.path.join(args.gen_img_path, 'generation_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(generation_config, f, indent=4)

    # save a csv file with actual number of generated images per condition
    conditions_save_path = os.path.join(args.gen_img_path, 'generated_conditions.csv')
    conditions_log_df.to_csv(conditions_save_path, index=False)

    print("==========================================================")
    print("============== Image generation completed ================")
    print("==========================================================")
