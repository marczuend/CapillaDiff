#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import glob
import subprocess
import json
from typing import Optional
import shutil
from transformers import CLIPImageProcessor

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


# Local imports
from CapillaDiff_encoder import ConditionEncoderInference, ConditionEncoder
from CapillaDiff_dataloader import DatasetLoader

# Logger setup
logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# enforce minimal diffusers
check_min_version("0.26.0.dev0")

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="CapillaDiff training script.")

    parser.add_argument("--naive_conditional",
        type=str,
        default="conditional",
        help=(
        "Whether the model is trained in 'naive' or 'conditional' mode:\n"
        "- 'naive': no prompt/condition encoding is used; the model learns to generate images without any conditioning.\n"
        "- 'conditional': the model uses the input captions/conditions to guide generation (default)."
        ),
    )

    # ---------------------------------------
    # Model paths
    # ---------------------------------------
    parser.add_argument("--pretrained_model_path",
        type=str,
        required=True,
        help="Path to pretrained model",
    )
    parser.add_argument("--use_ema",
        action="store_true",
        help="Whether to use EMA model."
    )
    parser.add_argument("--clip_path",
        type=str,
        default=None,
        help="Path to the CLIP model."
    )

    # ---------------------------------------
    # Data
    # ---------------------------------------
    parser.add_argument("--img_data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the training images.",
    )
    parser.add_argument("--metadata_file_path",
        type=str,
        required=True,
        help="Path to the metadata file (must be a .csv file) containing the names and captions of images.",
    )
    parser.add_argument("--image_column",
        type=str,
        default="FileName",
        help="Name of the column of the dataset containing the path to the image.",
    )
    parser.add_argument("--caption_columns",
        type=str,
        default="all",
        help=(
            "The columns of the dataset containing a caption or a list of captions.\n"
            "The should look like this: 'col1,col2,...,colN' or use 'all' to use all columns."
        ),
    )
    parser.add_argument("--convert_to_boolean",
        type=int,
        default=1,
        help="Whether to convert the conditions to boolean embeddings. 1 for True, 0 for False."
    )
    parser.add_argument("--text_mode",
        type=str,
        default=None,
        help="The text mode to use for encoding conditions."
    )
    # ---------------------------------------
    # Output / checkpoints
    # ---------------------------------------
    parser.add_argument("--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--logging_dir",
        type=str,
        default=None,
        help="The logging directory where logs will be written.",
    )
    parser.add_argument("--checkpointing_dir",
        type=str,
        default=None,
        help="The directory where checkpoints will be saved.",
    )
    parser.add_argument("--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for "
            "resuming training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--trained_steps",
        type=int,
        default=0,
        help=(
            "The number of trained steps so far."
        ),
    )
    parser.add_argument("--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument("--report_to_wandb",
        type=bool,
        default=False,
        help=(
            'The integration to report the results and logs to weight and biases. Offline mode is used.'
        ),
    )
    parser.add_argument("--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of condition ids evaluated every `--validation_epochs` and logged to `--report_to`."),
    )

    # ---------------------------------------
    # Training hyperparameters
    # ---------------------------------------
    parser.add_argument("--train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs",
        type=int,
        default=500,
        help="Number of epochs to train for.",
    )
    parser.add_argument("--max_train_steps",
        #type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--validation_epochs",
        type=int,
        default=100,
        help="Run validation every X epochs.",
    )
    parser.add_argument("--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", '
            '"polynomial", "constant", "constant_with_warmup"].'
        ),
    )
    parser.add_argument("--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= "
            "1.10 and an Nvidia Ampere GPU. Default: value from accelerate config."
        ),
    )
    parser.add_argument("--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.",
    )
    parser.add_argument("--input_perturbation",
        type=float,
        default=0.0,
        help="The scale of input perturbation. Recommended 0.1.",
    )
    parser.add_argument("--noise_offset",
        type=float,
        default=0.0,
        help="The scale of noise offset.",
    )
    parser.add_argument("--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        ),
    )
    parser.add_argument("--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.\n"
            "1 CPU core: 0 or 1\n"
            "2 CPU cores: 0, 1 or 2\n"
            "4 CPU cores: 0, 1, 2, 3 or 4\n"
            "etc..."
        ),
    )
    parser.add_argument("--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--use_cfg",
        type=int,
        default=0,
        help="Whether to use classifier-free guidance."
    )
    parser.add_argument("--cfg_training_prob",
        type=float,
        default=0.1,
        help="The probability of using classifier-free guidance during training. Default is 0.1."
    )

    # ---------------------------------------
    # Optimizer parameters
    # ---------------------------------------
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--prediction_type",
        type=str,
        default=None,
        help=(
            "Type of prediction used for training. Options:\n"
            "  - 'epsilon': predict noise.\n"
            "  - 'v_prediction': predict velocity.\n"
            "  - None (default): uses the scheduler's default "
            "prediction type (`noise_scheduler.config.prediction_type`)."
        ),
    )
    parser.add_argument("--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--scale_min",
        type=float,
        default=0.9,
        help="Minimum scale for random resized crop during training.",
    )

    # ---------------------------------------
    # Execution environment
    # ---------------------------------------
    parser.add_argument("--cache_dir",
        type=str,
        default="./tmp",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument("--resolution",
        type=int,
        default=512,
        help="The resolution for input images. Images will be resized to this resolution.",
    )
    parser.add_argument("--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally.",
    )

    return parser.parse_args()


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
        self.custom_encoder = text_encoder

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
        embeddings = self.custom_encoder(prompt)
        embeddings = embeddings.to(device)
        return embeddings, None


# ---------------------------
# Utilities
# ---------------------------

def check_directory(directory: str) -> bool:
    """Check if a directory exists and is writable.

    Args:
        directory (str): The directory to check.
    Returns:
        bool: True if the directory exists and is writable, False otherwise.
    """
    
    test_file_path = os.path.join(directory, "test_write_permission.txt")
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(test_file_path, "w") as f:
            f.write("This is a test file to check write permissions.\n")
        os.remove(test_file_path)
        return True
    except IOError:
        return False

# TODO: Adjust if validation needed
def log_validation(args, accelerator, weight_dtype, step, ckpt_path):
    """ Log validation images to wandb.
    
    Args:
        args (argparse.Namespace): The parsed arguments.
        accelerator (Accelerator): The Accelerator object.
        weight_dtype (str): The weight dtype used for training.
        step (int): The current training step.
        ckpt_path (str): The path to the checkpoint.
        
    Returns:
        List[torch.Tensor]: The validation images.
    """
    raise ImportError(
        "Validation is not implemented yet for CapillaDiff.\n"
        "To avoid this error, do not pass --validation_prompts."
        )

    logger.info("Running validation... ")

    if args.pretrained_model_path != ckpt_path:
        if not os.path.exists(ckpt_path+'/feature_extractor'):
            os.makedirs(ckpt_path+'/feature_extractor')
        shutil.copyfile(
            args.pretrained_model_path+'/feature_extractor/preprocessor_config.json',
            ckpt_path+'/feature_extractor/preprocessor_config.json')
        unet = UNet2DConditionModel.from_pretrained(
            ckpt_path, subfolder="unet_ema", use_auth_token=True)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            ckpt_path, subfolder="unet", use_auth_token=True)

    feature_extractor = CLIPImageProcessor.from_pretrained(
        ckpt_path+'/feature_extractor')

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_path,
        subfolder="vae")

    noise_scheduler = DDPMScheduler.from_pretrained(
        ckpt_path, subfolder="scheduler")

    # check if text mode is used
    clip = None
    if args.text_mode != "None":
        # load clip model
        from transformers import CLIPTokenizer, CLIPTextModel
        clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_path)
        clip_text_encoder = (CLIPTextModel.from_pretrained(args.clip_path)).to(accelerator.device)
        clip = (clip_tokenizer, clip_text_encoder, accelerator.device)

    custom_encoder = ConditionEncoderInference(
        args.naive_conditional, convert_to_boolean=args.convert_to_boolean,
        text_mode=args.text_mode, clip=clip)

    pipeline = CapillaDiffusionPipeline(
        vae=vae,
        unet=unet,
        text_encoder=custom_encoder,
        feature_extractor=feature_extractor,
        scheduler=noise_scheduler)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_path = args.checkpointing_dir+"/checkpoint-"+str(step) + "/validation/"
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    images = []
    updated_validation_prompts = []

    for i in range(len(args.validation_prompts)):
        for j in range(4):
            with torch.autocast("cuda"):
                image = pipeline(
                    args.validation_prompts[i],
                    generator=generator).images[0]
            images.append(image)
            updated_validation_prompts.append(args.validation_prompts[i]+'-'+str(j))

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(
                            image,
                            caption=f"{i}: {updated_validation_prompts[i]} - step {step}",)
                        for i, image in enumerate(images)
                    ]
                }
            )

        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images

def encode_prompt(identifier, clip=None):
    """Get gene embedding generated by scGPT based on input identifier.

    Args:
        identifier (str): condition identifier

    Returns:
        prompt_embeds (torch.Tensor): gene embedding"""

    prompt_embeds = None
    global naive_conditional
    global convert_to_boolean
    global text_mode

    encoder = ConditionEncoder(naive_conditional, convert_to_boolean=convert_to_boolean, text_mode=text_mode, clip=clip)
    prompt_embeds = encoder.get_condition_embedding(identifier)

    return prompt_embeds

# ---------------------------
# Main training function
# ---------------------------

def main():

    args = parse_args()

    # Set global variables
    global naive_conditional
    naive_conditional = args.naive_conditional

    if args.use_cfg == 1:
        args.use_cfg = True
    else:
        args.use_cfg = False

    if args.convert_to_boolean == 1:
        args.convert_to_boolean = True
    else:
        args.convert_to_boolean = False

    global convert_to_boolean
    convert_to_boolean = args.convert_to_boolean

    if args.text_mode == "None":
        args.text_mode = None
    global text_mode
    text_mode = args.text_mode    

    # either use max_train_steps or num_train_epochs to determine total training steps
    if args.max_train_steps == "None":
        args.max_train_steps = None
    else:
        args.max_train_steps = int(args.max_train_steps)

    if args.checkpointing_dir is None:
        args.checkpointing_dir = os.path.join(args.output_dir, "checkpoints")

    # check if all directories exist and are writable
    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.output_dir, "logs")
    dirs_to_check = [args.output_dir, args.cache_dir, args.logging_dir, args.checkpointing_dir]
    for directory in dirs_to_check:
        if not check_directory(directory):
            raise IOError(f"Directory {directory} is not writable. Please check permissions.")

    # Set temporary directories
    os.environ["TMPDIR"] = args.cache_dir
    os.environ["TEMP"]   = args.cache_dir
    os.environ["TMP"]    = args.cache_dir

    # WandB logging
    accelerate_log_with = None
    if args.report_to_wandb:
        if not is_wandb_available():
            raise ImportError("Please install wandb to use the wandb logging functionality.")
        else:
            import wandb
            os.environ['WANDB_DIR'] = args.logging_dir
            os.environ["WANDB_MODE"] = "offline"
            # print("WandB logging data to directory: " + os.environ['WANDB_DIR'] + "/wandb")
            accelerate_log_with = "wandb"

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                      logging_dir=args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=accelerate_log_with,
        project_config=accelerator_project_config,
    )

    # prints information like number of processes, mixed precision, cpu/gpu used etc.
    # logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_path,
        subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_path,
            subfolder="vae"
        )

    unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path,
            subfolder="unet"
        )

    # Freeze vae and set unet to trainable
    vae.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path,
            subfolder="unet"
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
 
    # 6. Get images with labels
    dataset = DatasetLoader(
        img_folder_path = args.img_data_dir,
        metadata_csv_path = args.metadata_file_path,
        #relevant_columns: list = None
        )
    
    dataset = dataset.get_dataset_dict()

    scale_min = args.scale_min
    # check if scale_min is between 0 and 1
    if scale_min < 0.0 or scale_min > 1.0:
        raise ValueError("scale_min must be between 0.0 and 1.0")

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=512, scale=(scale_min,1.0)),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.67212146, 0.63119322, 0.62765121], # from Adriana, 08.12.2025
                                 [0.08998639, 0.11100586, 0.12605950])
            #transforms.Normalize([0.5, 0.5, 0.5],
            #                     [0.5, 0.5, 0.5])
        ]
    )

    def preprocess_train(examples, image_column="image"):
        from PIL import Image

        images = [Image.open(image).convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(
                seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples, caption_column='caption', clip=None):

        pixel_values = torch.stack(
            [example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()

        prompt_embeds = [encode_prompt(
            example[caption_column], clip=clip) for example in examples]
        input_ids = torch.stack([example for example in prompt_embeds])
        input_ids = input_ids.squeeze(1)

        return {"pixel_values": pixel_values,
                "input_ids": input_ids}

    # check if text mode is used
    clip = None
    if args.text_mode != "None":
        # load clip model
        from transformers import CLIPTokenizer, CLIPTextModel
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_path)
        text_encoder = (CLIPTextModel.from_pretrained(args.clip_path)).to(accelerator.device)
        clip = (tokenizer, text_encoder, accelerator.device)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, clip=clip),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(int(args.max_train_steps) / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info(
        f"""
    ***** Running training *****
    Num examples                             {len(train_dataset)}
    Num Epochs                               {args.num_train_epochs}
    Instantaneous batch size per device      {args.train_batch_size}
    Total train batch size                   {total_batch_size}
    (parallel × distributed × accumulation)
    Gradient Accumulation steps              {args.gradient_accumulation_steps}
    Total optimization steps                 {args.max_train_steps}
    """
    )
    global_step = 0
    first_epoch = 0
    

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.checkpointing_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.checkpointing_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    generate_img_step0_sign = True
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    total_trained_steps = args.trained_steps

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if generate_img_step0_sign and args.validation_prompts is not None:
                    log_validation(
                        args,
                        accelerator,
                        weight_dtype,
                        args.trained_steps,
                        args.pretrained_model_path
                    )
                    generate_img_step0_sign = False
                    
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(
                        latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(
                        latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = batch["input_ids"].float()

                if args.naive_conditional == 'naive':
                    assert torch.all(encoder_hidden_states == 1.), \
                        "encoder_hidden_states should be all ones for naive SD"
                elif args.naive_conditional == 'conditional':
                    # check that the encoder_hidden_states are not all ones
                    assert not torch.all(encoder_hidden_states == 1.), \
                        "encoder_hidden_states should not be all ones for CapillaDiff"

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # optional classifier-free guidance dropout
                if args.use_cfg:
                    drop_mask = torch.rand(bsz, device=latents.device) < args.cfg_training_prob
                    encoder_hidden_states = encoder_hidden_states.clone()
                    encoder_hidden_states[drop_mask] = torch.zeros((1, 77, 768)).to(latents.device)

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states,
                    return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                total_trained_steps = global_step + args.trained_steps
                #print('global_step:', global_step, 'total_trained_steps:', total_trained_steps)

                if (global_step % args.checkpointing_steps == 0) | (global_step == args.max_train_steps):
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.checkpointing_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.checkpointing_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        # remove old checkpoints if there are more than one saved checkpoints. Keep the latest one.
                        ckpt_files = os.listdir(args.checkpointing_dir)
                        ckpt_files = [f for f in ckpt_files if f.startswith("checkpoint-")]
                        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("-")[1]))

                        # remove the folder with smaller number
                        if len(ckpt_files) > 1:
                            old_ckpt_path = os.path.join(
                                args.checkpointing_dir, ckpt_files[0])

                            # check if there is any folder in old_ckpt_path
                            if os.path.exists(old_ckpt_path):
                                shutil.rmtree(old_ckpt_path)
                                logger.info(f"Removed state from {old_ckpt_path}")
                        save_path = os.path.join(
                            args.checkpointing_dir, f"checkpoint-{total_trained_steps}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # save checkpoint
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:

                            unet_ckpt = unwrap_model(unet)
                            if args.use_ema:
                                ema_unet.copy_to(unet_ckpt.parameters())
                            
                            feature_extractor = CLIPImageProcessor.from_pretrained(
                                args.pretrained_model_path+'/feature_extractor')


                            pipeline = StableDiffusionPipeline(
                                vae=accelerator.unwrap_model(vae),
                                text_encoder=None,
                                tokenizer=None,
                                unet=unet_ckpt,
                                scheduler=noise_scheduler,
                                feature_extractor=feature_extractor,
                                safety_checker=None,
                            )
                            pipeline.save_pretrained(save_path)
                            with open(os.path.join(save_path, 'training_config.json'), 'w') as f:
                                f.write(json.dumps(vars(args), indent=2))

                            if (args.validation_prompts is not None) and \
                                ((global_step % args.validation_epochs == 0) | (global_step == args.max_train_steps)):

                                log_validation(
                                    args,
                                    accelerator,
                                    weight_dtype,
                                    total_trained_steps,
                                    save_path
                                )
                                
                                # write in the args.checkpointing_log_file file
                                with open(args.checkpointing_log_file, "a") as f:
                                    f.write(args.logging_dir+','+args.pretrained_model_path+','+save_path+',' +
                                            str(args.seed)+','+str(total_trained_steps)+','+str(args.checkpoint_number)+"\n")

            logs = {"step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step == args.max_train_steps:
                break

    print("==========================================================")
    print("================== Training completed ====================")
    print("==========================================================")

    
    # cleanup temporary directories
    shutil.rmtree(args.cache_dir)

    accelerator.end_training()

    # extract loss from wandb logs and save to a csv file
    if args.report_to_wandb:
        def find_wandb_logs(root):
            """
            Find all wandb log files in the given root directory.
            """
            extensions = ("*.wandb",)
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
            return files
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "evaluation", "extract_wandb_loss_to_csv.py")

        # get newest wandb log file in args.logging_dir/wandb
        wandb_log_dir = os.path.join(args.logging_dir, "wandb")

        log_files = find_wandb_logs(wandb_log_dir)
        if not log_files:
            print("No wandb log files found. Could not extract loss log.")
            return
        log_files = sorted(log_files, key=lambda x: os.path.getctime(x), reverse=True)
        newest_log_file = log_files[0]
        input_file = os.path.join(wandb_log_dir, newest_log_file)
        outdir = args.logging_dir

        cmd = [
            "python3",
            script_path,
            "--input", input_file,
            "--outdir", outdir
        ]

        subprocess.run(cmd, check=True)
    

if __name__ == "__main__":
    main()