# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, DDPMScheduler
import os
import torch
import datetime
import argparse
from transformers import AutoFeatureExtractor
from diffusers import AutoencoderKL, UNet2DConditionModel
from perturbation_encoder import PerturbationEncoderInference
from typing import Optional
import pandas as pd
import random
import numpy as np


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


class CustomStableDiffusionPipeline(StableDiffusionPipeline):
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
        #embeddings = self.custom_text_encoder(prompt)
        embeddings = torch.ones(
                    (1, 77, 768))  # dummy embedding

        if False:
            from transformers import CLIPTokenizer, CLIPTextModel

            print("Using CLIP text encoder for embeddings...")
            path_clip ="/cluster/customapps/medinfmk/mazuend/CapillaDiff/clip_model"
            print("path: "+path_clip)

            tokenizer = CLIPTokenizer.from_pretrained(path_clip+"/clip_tokenizer", local_files_only=True)
            text_encoder = CLIPTextModel.from_pretrained(path_clip+"/clip_text_encoder", local_files_only=True)

            text = "a microscope image of cells"
            tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            embeddings = text_encoder(**tokens).last_hidden_state  # shape (1, 77, 768)

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


def load_model_and_generate_images(pipeline, model_checkpoint, prompts_df,
                                   gen_img_path, num_imgs=500):
    """Load the model and generate images for the given prompts.

    Args:
        model_checkpoint (str): The address of the model checkpoint.
        prompts (list): A list of prompts to generate images for.
        gen_img_path (str): The address of the directory to save the generated
        images."""

    model_name = model_checkpoint.split('/')[-2]+'_' +\
        model_checkpoint.split('/')[-1]

    for idx, row in prompts_df.iterrows():
        prompt = row['perturbation']
        ood = row['ood']

        model_dir = gen_img_path+prompt+'/'+model_name
        if 'naive' in model_checkpoint.lower():
            model_dir = gen_img_path+model_name
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        set_seed(42)
        start = datetime.datetime.now()
        imgs = os.listdir(model_dir)

        if len(imgs) >= num_imgs:
            print(str(len(imgs))+' already generated.')
            continue

        print('Number of images generated for '+prompt+': '+str(len(imgs)) +
              ', and '+str(num_imgs-len(imgs))+' more will be generated')
        while len(imgs) < num_imgs:
            image = pipeline(
                prompt=prompt)
            imgs = os.listdir(model_dir)
            image_name = f'{prompt}-generated-{len(imgs)}.png'
            image_path = model_dir+"/"+image_name
            image.images[0].save(image_path)
            imgs = os.listdir(model_dir)

        img_len = len(os.listdir(model_dir))
        assert img_len >= num_imgs

        generated_pert_dir = 'result/generated_perturbation_list/'
        if ood:
            ood_file = open(
                generated_pert_dir+model_name+'/ood_pert_generated.txt', 'a')
            ood_file.write('\n'+prompt)
            ood_file.close()
        else:
            in_dist_file = open(
                generated_pert_dir+model_name+'/in_dist_pert_generated.txt', 'a')
            in_dist_file.write('\n'+prompt)
            in_dist_file.close()

        print('Total images generated for '+prompt+': '+str(img_len))
        print("Generating time: ", datetime.datetime.now()-start)
        print()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default='',
                        help="a name for identifying the model")
    parser.add_argument('--perturbation_list_address',
                        default='',
                        help="a file with a list of all perturbations")
    parser.add_argument('--gen_img_path', default='',
                        help="a name for identifying the model")
    parser.add_argument('--num_imgs', default=3,
                        help="a name for identifying the model")
    parser.add_argument('--vae_path', default='',
                        help="a name for identifying the model")
    parser.add_argument('--experiment', default='HUVEC-01',
                        help="a name for identifying the model")
    parser.add_argument('--model_type', default='conditional',
                        help="a name for identifying the model")
    parser.add_argument("--ood", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="indicating if generate images for " +
                        "ood perturbations or in-dist perturbations")
    parser.add_argument('--cluster', default='-',
                        help="cluster")
    parser.add_argument('--model_name', default='SD',
                        help="model type")
    parser.add_argument('--clip_path', default='',
                        help="path to CLIP model")
    args = parser.parse_args()
    args.num_imgs = int(args.num_imgs)

    model_name = args.model_checkpoint.split('/')[-2]+'_' +\
            args.model_checkpoint.split('/')[-1]

    gen_pert_dir = 'result/generated_perturbation_list/'
    if not os.path.exists(gen_pert_dir+model_name):
        os.makedirs(gen_pert_dir+model_name)
        with open(gen_pert_dir+model_name+'/in_dist_pert_generated.txt', 'w') as f:
            f.write('in_dist')
        with open(gen_pert_dir+model_name+'/ood_pert_generated.txt', 'w') as f:
            f.write('ood')

    if not os.path.exists(args.gen_img_path):
        os.makedirs(args.gen_img_path)

    # read perturbations
    prompts = []
    perturbation_file = args.perturbation_list_address

    prompt_df = pd.read_csv(perturbation_file)

    if args.ood:
        prompt_df = prompt_df[prompt_df['ood']==True]
    else:
        prompt_df = prompt_df[prompt_df['ood']==False]

    if args.model_name == 'SD':
        # initialize SD model
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.model_checkpoint+'/feature_extractor')
        print('Loaded feature_extractor')

        vae = AutoencoderKL.from_pretrained(
            args.vae_path, subfolder="vae")
        print('Loaded vae model')

        unet = UNet2DConditionModel.from_pretrained(
            args.model_checkpoint, subfolder="unet_ema", use_auth_token=True)
        print('Loaded EMA unet model')

        noise_scheduler = DDPMScheduler.from_pretrained(
            args.model_checkpoint, subfolder="scheduler")
        print('Loaded noise_scheduler')

        # dataset_id, cluster, model
        custom_gene_encoder = PerturbationEncoderInference(
            args.experiment,
            args.model_type,
            args.model_name)
        print('Loaded custom_text_encoder')

        # Initialize your custom pipeline
        pipeline = CustomStableDiffusionPipeline(
            vae=vae,
            unet=unet,
            text_encoder=custom_gene_encoder,
            feature_extractor=feature_extractor,
            scheduler=noise_scheduler)
        print('Initialized pipeline')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline.to(device)

        load_model_and_generate_images(
            pipeline,
            args.model_checkpoint,
            prompt_df,
            args.gen_img_path,
            args.num_imgs)
