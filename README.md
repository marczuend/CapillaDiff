![Project Logo](morphodiff_logo.png) 


This is the official repository of CapillaDiff, a diffusion based generative pipeline to predict high-resolution capillaroscopy images with different conditions.

---

## Environmental Setting

Create a new virtual environment (python 3.10.12 was used in our experiments), and install the diffusers package provided in this repository using the following commands:

```bash
# clone github repository
git clone git@github.com:marczuend/CapillaDiff.git

# install diffusers
cd CapillaDiff/capilladiff/diffusers

pip install .

cd examples/text_to_image

pip install -r requirements.txt

# configure accelerator
accelerate config
```

The xformers and wandb (if needed) packages are also required for training and need to be installed in the morphodiff environment. The requirements.txt provides a list of packages and their versions used to run the scripts.

## Needed Models

CapillaDiff requires several pretrained components to function correctly. You must provide paths to these models when running the pipeline.

### 1. Stable Diffusion Base Model
You need a pretrained **Stable Diffusion checkpoint**, such as [**stable-diffusion-v1-4**](https://huggingface.co/CompVis/stable-diffusion-v1-4), which was originally used for training CapillaDiff.  

You can provide:
- A local folder containing the checkpoint, e.g.: /path/to/stable-diffusion-v1-4

### 2. Text Encoder and Tokenizer
CapillaDiff uses **text embeddings** to condition image generation. If you want to use this feature, you must provide local paths to a **CLIPTokenizer** and a **CLIPTextModel**, for example from [**openai/clip-vit-large-patch14**](https://huggingface.co/openai/clip-vit-large-patch14).

## Recommended folder structure

```bash
CapillaDiff/
├── capilladiff/
├── LICENSE
├── requirments.txt
└── README.md

models/
├── stable-diffusion-v1-4/
├── capilladiff_checkpoint/
└── clip-vit-large-patch14/
```
## Codebase overview [TO BE DONE]

```bash
├── code
│   ├── bash # Containing (Slurm) bash scripts for training, image generation, and testing generated images
│   ├── cellprofiler # Containing CellProfiler pipelines used for feature extraction, and Python scripts used for feature preprocessing and analysis
│   ├── evaluation # Containing Python scripts for image generation and distance metric calculation
│   ├── preprocessing # Contatinig Python scripts used for data pre-processing
│   ├── cell_cropped_benchmarking_code # Contatinig scripts used for cell-cropped image analysis 
│   ├── required_file # Contatining files required for perturbation encoding of all datasets (perturbation encoded vectors)
│   ├── perturbation_encoder.py # Implementation of perturbation encoding class as part of the MorphoDiff pipeline
│   └── train_text_to_image_cell_painting.py # The modified training script of Stable Diffusion
```
## Training/fine-tuning  (TO BE DONE)

The `scripts/train.sh` provides commands for defining parameters required for training CapillaDiff and Stable Diffusion, with description of each parameter provided in the bash script. After defining the parameters of the training script, run the following for submitting the training job using Slurm.

```bash
sbatch scripts/train.sh
```

After the training is completed for the specified number of steps, the `scripts/train.sh` automatically resubmits the job and resumes training from the last checkpoint. You can set the total_steps parameter to not train more than a specific number of steps, or comment the `scontrol requeue $SLURM_JOB_ID` line that resubmits the job once it is finished.

## Data Preparation  (TO BE DONE)

Data folder contents must follow the structure described in [https://huggingface.co/docs/datasets/image_dataset#imagefolder](https://huggingface.co/docs/datasets/image_dataset#imagefolder). In particular, a `metadata.jsonl` file must exist to provide the perturbation id for the images. The perturbation ids used in CapillaDiff analysis are provided in the required_file/ folder.

### Download Dataset  (TO BE DONE)

The raw datasets used to train CapillaDiff can be found on the LeoMed Cluster (/cluster/work/medinfmk/capillaroscopy/)

## Condition Encoding  (TO BE DONE)

To be determined
1. Features like number of bloodcells, etc...
2. Clinical Text encoding using CLIP to encode

## Image Generation (TO BE DONE)

The `scripts/generate_img.sh` script is a Slurm based bash script that takes the path to the pretrained checkpoint, a file of condition list, the number of images to generate per condition, and an address to save the images of each condition configuration in a separate folder. You should set the parameters described and documented in the `scripts/generate_img.sh` and run it as follow

```bash
sh scripts/generate_img.sh
```
}
```
