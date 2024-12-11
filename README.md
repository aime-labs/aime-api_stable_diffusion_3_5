# Stable Diffusion 3.5 - AIME Demo

[Stable Diffusion 3.5 Large](https://stability.ai/news/introducing-stable-diffusion-3-5) is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that features improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.

This is an inference-only reference implementation of Stable Diffusion 3.5. This version is optimized for integration with the AIME API Server for easy deployment and scaling of image generation tasks.

- Ready to use as a worker for the [AIME API Server](https://github.com/aime-team/aime-api-server)
- Added possibility to generate multiple images at once
- Preview images while processing
- Text to Image and Image to Image

## Download

Download the following models from HuggingFace into `models` directory:
1. [Stability AI SD3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors) or [Stability AI SD3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/blob/main/sd3.5_large_turbo.safetensors) or [Stability AI SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/sd3.5_medium.safetensors)
2. [OpenAI CLIP-L](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/clip_l.safetensors)
3. [OpenCLIP bigG](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/clip_g.safetensors)
4. [Google T5-XXL](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/t5xxl_fp16.safetensors)

This code also works for [Stability AI SD3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium.safetensors).

### ControlNets

Optionally, download [SD3.5 Large ControlNets](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets):
- [Blur ControlNet](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/blob/main/sd3.5_large_controlnet_blur.safetensors)
- [Canny ControlNet](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/blob/main/sd3.5_large_controlnet_canny.safetensors)
- [Depth ControlNet](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/blob/main/sd3.5_large_controlnet_depth.safetensors)

```py
from huggingface_hub import hf_hub_download
hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_blur.safetensors", local_dir="models")
hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_canny.safetensors", local_dir="models")
hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_depth.safetensors", local_dir="models")
```

### Or

```sh
sudo apt-get install git-lfs
git lfs install
mkdir /destination/to/checkpoints
cd /destination/to/checkpoints
git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-large
```

## Clone this repo
```sh
cd /destination/to/repo
git clone https://github.com/aime-labs/aime-api_stable_diffusion_3_5.git
```

## Setting up AIME MLC
```sh

mlc-create sd3-5 Pytorch 2.3.1-aime -d="/destination/to/checkpoints" -w="/destination/to/repo"
```
The -d flag will mount /destination/to/checkpoints to /data in the container. 

The -w flag will mount /destination/to/repo to /workspace in the container.


## Install requirements in AIME MLC
```sh
mlc-open sd3-5

pip install -r /workspace/aime-api_stable_diffusion_3_5/requirements.txt
```

## Run SD3.5 inference as HTTP/HTTPS API with AIME API Server

To run Stable Diffusion 3.5 as HTTP/HTTPS API with [AIME API Server](https://github.com/aime-team/aime-api-server) start the chat command with following command line:

```sh
mlc-open sd3.5

python3 /workspace/aime-api_stable_diffusion_3_5/main.py --api_server <url to API server>
```

It will start Stable Diffusion 3 as worker, waiting for job request through the AIME API Server.


## File Guide

- `sd3_infer.py` - entry point, review this for basic usage of diffusion model
- `sd3_impls.py` - contains the wrapper around the MMDiTX and the VAE
- `other_impls.py` - contains the CLIP models, the T5 model, and some utilities
- `mmditx.py` - contains the core of the MMDiT-X itself
- `main.py`
- folder `models` with the following files (download separately):
    - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL/SD3, can grab a public copy)
    - `clip_g.safetensors` (openclip bigG, same as SDXL/SD3, can grab a public copy)
    - `t5xxl.safetensors` (google T5-v1.1-XXL, can grab a public copy)
    - `sd3.5_large.safetensors` or `sd3.5_large_turbo.safetensors` or `sd3.5_medium.safetensors` (or `sd3_medium.safetensors`)

## Code Origin

The code included here originates from:
- Stability AI internal research code repository (MM-DiT)
- Public Stability AI repositories (eg VAE)
- Some unique code for this reference repo written by Alex Goodwin and Vikram Voleti for Stability AI
- Some code from ComfyUI internal Stability implementation of SD3 (for some code corrections and handlers)
- HuggingFace and upstream providers (for sections of CLIP/T5 code)

## License

This model is available under the [Stability AI Community License](https://stability.ai/community-license-agreement):
- Non-commercial Use: Free for non-commercial projects and research.
- Commercial Use: Free if your company’s annual revenue is less than $1 million.
- Ownership of Outputs: You own the images you generate.

Please review the [full license terms](https://stability.ai/community-license-agreement) for more information.

### Note

Some code in `other_impls` originates from HuggingFace and is subject to [the HuggingFace Transformers Apache2 License](https://github.com/huggingface/transformers/blob/main/LICENSE)
