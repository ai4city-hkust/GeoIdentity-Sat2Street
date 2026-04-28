"""
GeoIdentity-Sat2Street: End-to-End Inference Pipeline

Runs all four steps in sequence:
    1a. Polar transformation (satellite -> panoramic intermediate)
    1b. Pix2PixHD GAN      (panoramic intermediate -> coarse street view)
    2a. Caption generation  (coarse street view -> semantic prompt)
    2b. Diffusion refiner   (coarse street view + prompt -> final street view)

Usage:
    python run_pipeline.py \
        --satellite_dir  /path/to/satellite_images \
        --output_dir     /path/to/output \
        --city           Kathmandu --country Nepal \
        --unet_ckpt      /path/to/unet_epoch.pt \
        --gan_ckpt       /path/to/latest_net_G.pth \
        --pix2pix_root   ./pix2pixHD
"""

import argparse
import os
import subprocess
import sys


def run(cmd: list, description: str):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"FAILED: {description}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GeoIdentity-Sat2Street end-to-end pipeline")
    parser.add_argument("--satellite_dir",  required=True, help="Input satellite image directory")
    parser.add_argument("--output_dir",     required=True, help="Root output directory")
    parser.add_argument("--city",           default="",    help="City name for location tag, e.g. Kathmandu")
    parser.add_argument("--country",        default="",    help="Country name for location tag, e.g. Nepal")
    parser.add_argument("--unet_ckpt",      default=None,  help="Fine-tuned UNet checkpoint (.pt)")
    parser.add_argument("--gan_ckpt",       default=None,  help="Pix2PixHD checkpoint (latest_net_G.pth)")
    parser.add_argument("--pix2pix_root",   default="./pix2pixHD", help="pix2pixHD repo root")
    parser.add_argument("--checkpoint_name", default="sat2svi_pix2pix")
    parser.add_argument("--num_inference_steps", type=int, default=70)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    polar_dir   = os.path.join(args.output_dir, "1_polar")
    coarse_dir  = os.path.join(args.output_dir, "2_coarse_gan")
    captions    = os.path.join(args.output_dir, "3_captions.jsonl")
    refined_dir = os.path.join(args.output_dir, "4_refined")

    for d in [polar_dir, coarse_dir, refined_dir]:
        os.makedirs(d, exist_ok=True)

    # Step 1a: Polar transform
    run([
        "step1_polar_transform.py",
        "--input_dir",  args.satellite_dir,
        "--output_dir", polar_dir,
        "--device",     args.device,
    ], "Step 1a: Polar Transformation")

    # Step 1b: GAN inference
    gan_args = [
        "step1_gan_inference.py",
        "--polar_dir",       polar_dir,
        "--output_dir",      coarse_dir,
        "--pix2pix_root",    args.pix2pix_root,
        "--checkpoint_name", args.checkpoint_name,
    ]
    if args.gan_ckpt:
        gan_args += ["--checkpoint_path", args.gan_ckpt]
    run(gan_args, "Step 1b: Pix2PixHD GAN Inference")

    # Step 2a: Caption generation
    cap_args = [
        "step2_caption_generation.py",
        "--image_dir",    coarse_dir,
        "--output_jsonl", captions,
        "--device",       args.device,
    ]
    if args.city and args.country:
        cap_args += ["--city", args.city, "--country", args.country]
    run(cap_args, "Step 2a: Caption Generation")

    # Step 2b: Diffusion refinement
    diff_args = [
        "step2_diffusion_refiner.py",
        "--captions_jsonl",      captions,
        "--output_dir",          refined_dir,
        "--num_inference_steps", str(args.num_inference_steps),
        "--device",              args.device,
    ]
    if args.unet_ckpt:
        diff_args += ["--unet_ckpt", args.unet_ckpt]
    run(diff_args, "Step 2b: Diffusion Refinement")

    print(f"\nPipeline complete. Final outputs: {refined_dir}")


if __name__ == "__main__":
    main()
