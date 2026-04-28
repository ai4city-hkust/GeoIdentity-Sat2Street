"""
Step 2b: Diffusion-based Geographic Identity Refinement
Refines coarse GAN street views using Stable Diffusion + ControlNet conditioned on:
  - Semantic caption with location tag (from step2_caption_generation.py)
  - Structural priors: depth map (DepthAnything) + Canny edges
  - Fine-tuned UNet weights (optional)

For each image, generates three candidates (depth-only, canny-only, depth+canny)
and saves the one with highest SSIM against the coarse input as the final output.

Usage:
    python step2_diffusion_refiner.py \
        --captions_jsonl ./captions.jsonl \
        --output_dir     ./refined \
        --unet_ckpt      ./unet_checkpoints/unet_epoch_14.pt
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from transformers import pipeline as hf_pipeline


def load_unet_weights(pipe: StableDiffusionControlNetPipeline, unet_ckpt: str):
    state_dict = torch.load(unet_ckpt, map_location="cpu")
    cleaned = {
        k.replace("unet.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }
    pipe.unet.load_state_dict(cleaned, strict=False)
    print(f"Loaded fine-tuned UNet from {unet_ckpt}")


def build_pipe(controlnets, unet_ckpt: str) -> StableDiffusionControlNetPipeline:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnets,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    if unet_ckpt and os.path.exists(unet_ckpt):
        load_unet_weights(pipe, unet_ckpt)

    return pipe


def preprocess_depth(image: Image.Image, depth_estimator) -> Image.Image:
    depth = depth_estimator(image)["depth"]
    depth_np = np.array(depth).astype(np.float32)
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    return Image.fromarray((depth_norm * 255).astype(np.uint8)).convert("RGB")


def preprocess_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    img_np = np.array(image)
    edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), low, high)
    return Image.fromarray(edges).convert("RGB")


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(img1.convert("L"))
    b = np.array(img2.convert("L"))
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return ssim(a[:h, :w], b[:h, :w], data_range=255)


def main():
    parser = argparse.ArgumentParser(description="Diffusion-based geographic identity refinement")
    parser.add_argument("--captions_jsonl", required=True, help="JSONL from step2_caption_generation.py")
    parser.add_argument("--output_dir",     required=True, help="Directory to save final refined images")
    parser.add_argument("--unet_ckpt",      default=None,  help="Fine-tuned UNet .pt checkpoint (optional)")
    parser.add_argument("--num_inference_steps", type=int, default=70)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ControlNet models...")
    controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

    print("Building inference pipelines...")
    pipe_depth = build_pipe([controlnet_depth], args.unet_ckpt)
    pipe_canny = build_pipe([controlnet_canny], args.unet_ckpt)
    pipe_both  = build_pipe([controlnet_depth, controlnet_canny], args.unet_ckpt)

    print("Loading DepthAnything estimator...")
    depth_estimator = hf_pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)

    df = pd.read_json(args.captions_jsonl, lines=True)
    print(f"Processing {len(df)} images...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["image_path"]
        prompt = row.get("caption", "")
        basename = os.path.splitext(os.path.basename(img_path))[0]

        try:
            orig = Image.open(img_path).convert("RGB")
            depth_img = preprocess_depth(orig, depth_estimator)
            canny_img = preprocess_canny(orig)

            gen_depth = pipe_depth(
                prompt=prompt, image=[depth_img], num_inference_steps=args.num_inference_steps
            ).images[0]
            gen_canny = pipe_canny(
                prompt=prompt, image=[canny_img], num_inference_steps=args.num_inference_steps
            ).images[0]
            gen_both = pipe_both(
                prompt=prompt, image=[depth_img, canny_img], num_inference_steps=args.num_inference_steps
            ).images[0]

            scores = {
                "depth": compute_ssim(orig, gen_depth),
                "canny": compute_ssim(orig, gen_canny),
                "both":  compute_ssim(orig, gen_both),
            }
            best_key = max(scores, key=scores.get)
            best_img = {"depth": gen_depth, "canny": gen_canny, "both": gen_both}[best_key]

            save_path = os.path.join(args.output_dir, f"{basename}_refined.png")
            best_img.save(save_path)
            print(f"[{basename}] best={best_key.upper()} SSIM={scores[best_key]:.4f}")

        except Exception as e:
            print(f"Error on {img_path}: {e}")

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
