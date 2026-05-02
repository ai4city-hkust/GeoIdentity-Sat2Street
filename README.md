<div align="center">

# GeoIdentity-Sat2Street

### Bridging Street View Coverage Disparities through Geographic Identity Preserving Generation from Satellite View

[![Paper](https://img.shields.io/badge/Paper-ISPRS%20JPRS-blue)](https://doi.org/10.1016/j.isprsjprs.2026.03.049)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.isprsjprs.2026.03.049-green)](https://doi.org/10.1016/j.isprsjprs.2026.03.049)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)
[![GitHub stars](https://img.shields.io/github/stars/ai4city-hkust/GeoIdentity-Sat2Street?style=social)](https://github.com/ai4city-hkust/GeoIdentity-Sat2Street)

**Official implementation for generating realistic and geographically faithful street-view imagery from satellite-view imagery.**

[Paper](https://doi.org/10.1016/j.isprsjprs.2026.03.049) |
[Installation](#installation) |
[Inference](#inference) |
[Training](#training) |
[Citation](#citation)

</div>

![GeoIdentity-Sat2Street pipeline](pipeline_overview.png)

## News

- **2026-03**: Paper published in *ISPRS Journal of Photogrammetry and Remote Sensing*.
- **2026-05**: Code for end-to-end inference, GAN training, and diffusion UNet fine-tuning released.

## Highlights

- **Satellite-to-street generation**: Synthesizes street-view imagery from widely available satellite imagery.
- **Geographic identity preservation**: Uses location-aware text prompts and structural priors to retain region-specific urban appearance.
- **Two-stage pipeline**: Combines polar-view conversion, Pix2PixHD coarse synthesis, caption generation, and ControlNet-guided diffusion refinement.
- **Research-oriented release**: Includes scripts for inference, stage-wise execution, GAN training, and UNet fine-tuning.

## Overview

Street View Imagery (SVI) is a critical data source for urban research, but its global coverage is highly uneven. Many cities in the Global South have sparse or incomplete street-view coverage, which limits downstream studies and can reinforce geographic data inequality.

**GeoIdentity-Sat2Street** addresses this gap by generating realistic, geographically faithful street-view images directly from satellite imagery. The framework is organized into two stages:

1. **Viewpoint Conversion**: A polar coordinate transformation maps the satellite image into a panoramic-like layout, then a Pix2PixHD conditional GAN generates a coarse street-level view.
2. **Geographic Identity Refinement**: A diffusion model refines the coarse view using semantic captions, explicit location metadata, and structural priors from depth and Canny ControlNet guidance.

## Pipeline

```text
Satellite image
  |
  |-- Step 1a: Polar transformation       step1_polar_transform.py
  |
  |-- Step 1b: Pix2PixHD GAN              step1_gan_inference.py
  |       -> coarse street view
  |
  |-- Step 2a: Caption generation         step2_caption_generation.py
  |       InstructBLIP + LaMini-T5 + optional location tag
  |
  |-- Step 2b: Diffusion refinement       step2_diffusion_refiner.py
          ControlNet depth/canny + fine-tuned UNet
          -> final street view
```

## Repository Structure

```text
GeoIdentity-Sat2Street/
|-- step1_polar_transform.py      # Satellite image -> polar-transformed image
|-- step1_gan_inference.py        # Polar image -> coarse street view with Pix2PixHD
|-- step2_caption_generation.py   # Coarse street view -> semantic caption JSONL
|-- step2_diffusion_refiner.py    # Caption + priors -> refined street view
|-- run_pipeline.py               # End-to-end inference pipeline
|-- train_gan.py                  # Pix2PixHD GAN training wrapper
|-- train_unet.py                 # Stable Diffusion UNet fine-tuning
|-- requirements.txt
`-- pipeline_overview.png
```

## Installation

```bash
git clone https://github.com/ai4city-hkust/GeoIdentity-Sat2Street
cd GeoIdentity-Sat2Street

conda create -n geoidentity python=3.9 -y
conda activate geoidentity

pip install -r requirements.txt

# Required for Stage 1b GAN inference/training
git clone https://github.com/NVIDIA/pix2pixHD
pip install dominate tensorboardX
```

**Requirements**

- Python 3.9+
- CUDA-capable GPU recommended
- Pix2PixHD checkpoint for Stage 1b
- Optional fine-tuned UNet checkpoint for Stage 2 refinement

## Inference

### Option A: End-to-end pipeline

```bash
python run_pipeline.py \
  --satellite_dir /path/to/satellite_images \
  --output_dir /path/to/output \
  --city Kathmandu \
  --country Nepal \
  --gan_ckpt /path/to/latest_net_G.pth \
  --unet_ckpt /path/to/unet_epoch_14.pt \
  --pix2pix_root ./pix2pixHD
```

The command creates the following output structure:

```text
output/
|-- 1_polar/          # Polar-transformed satellite images
|-- 2_coarse_gan/     # GAN-synthesized coarse street views
|-- 3_captions.jsonl  # Generated prompts and metadata
`-- 4_refined/        # Final refined street-view images
```

### Option B: Run each stage separately

```bash
# Step 1a: Polar transformation
python step1_polar_transform.py \
  --input_dir /path/to/satellite \
  --output_dir /path/to/polar

# Step 1b: GAN coarse synthesis
python step1_gan_inference.py \
  --polar_dir /path/to/polar \
  --output_dir /path/to/coarse \
  --pix2pix_root ./pix2pixHD \
  --checkpoint_path /path/to/latest_net_G.pth

# Step 2a: Caption generation
python step2_caption_generation.py \
  --image_dir /path/to/coarse \
  --output_jsonl /path/to/captions.jsonl \
  --city Kathmandu \
  --country Nepal

# Step 2b: Diffusion refinement
python step2_diffusion_refiner.py \
  --captions_jsonl /path/to/captions.jsonl \
  --output_dir /path/to/refined \
  --unet_ckpt /path/to/unet_epoch_14.pt
```

Notes:

- `--city` and `--country` are optional. When omitted, no location tag is appended to captions.
- `--unet_ckpt` is optional. When omitted, the vanilla Stable Diffusion v1.5 UNet is used.
- Stage 2 can be memory intensive; reduce batch size or image resolution if GPU memory is limited.

## Training

### Stage 1: Train the Pix2PixHD GAN

Prepare paired data in the Pix2PixHD-style structure:

```text
data/SAT2SVI/
|-- train_A/    # Polar-transformed satellite images
|-- train_B/    # Ground-truth street-view panoramas
|-- test_A/
`-- test_B/
```

Then run:

```bash
python train_gan.py \
  --data_dir ./data/SAT2SVI \
  --pix2pix_root ./pix2pixHD \
  --name sat2svi_pix2pix \
  --niter 100 \
  --niter_decay 100 \
  --batch_size 4
```

The trained generator checkpoint is saved to:

```text
pix2pixHD/checkpoints/sat2svi_pix2pix/latest_net_G.pth
```

### Stage 2: Fine-tune the diffusion UNet

Generate captions for the training images:

```bash
python step2_caption_generation.py \
  --image_dir ./data/SAT2SVI/train_B \
  --output_jsonl ./captions_train.jsonl \
  --city "New York" \
  --country "United States"
```

Fine-tune the UNet:

```bash
python train_unet.py \
  --captions_jsonl ./captions_train.jsonl \
  --output_dir ./unet_checkpoints \
  --num_epochs 15 \
  --batch_size 4 \
  --lr 1e-5
```

Checkpoints are saved every 5 epochs:

```text
unet_checkpoints/unet_epoch_N.pt
```

## Datasets

| Dataset | Description | Usage |
| --- | --- | --- |
| [CVUSA](https://mvrl.cse.wustl.edu/datasets/cvusa/) | US satellite-street pairs | Training and evaluation |
| [CVACT](https://github.com/Liumouliu/OriCNN) | Australia satellite-street pairs | Training and evaluation |
| [MultiCities Dataset](https://huggingface.co/datasets/Zongrong/MultiCities) | 50,000 pairs across Mumbai, Amsterdam, New York, Sao Paulo, and Sydney | Geographic identity evaluation |

## Citation

If you find this repository useful, please cite:

```bibtex
@article{li2026geoidentity,
  title   = {Bridging street view coverage disparities through geographic identity preserving generation from satellite view},
  author  = {Li, Zongrong and Zhang, Fan and Dai, Shaoqing and Zhao, Wufan},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {236},
  pages   = {622--639},
  year    = {2026},
  doi     = {10.1016/j.isprsjprs.2026.03.049}
}
```

## Contact

For questions about the code or paper, please contact **Zongrong Li** at [zongrong0122@gmail.com](mailto:zongrong0122@gmail.com).

## License

This project is released under the [MIT License](LICENSE). The Pix2PixHD dependency is subject to its own license; see [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
