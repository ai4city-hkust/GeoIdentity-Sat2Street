"""
Step 1a: Polar Transformation
Converts satellite images (256x256) to panoramic-like intermediate images (256x512)
as a preprocessing step before GAN-based street view synthesis.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from imageio.v2 import imread
from imageio.v2 import imwrite as imsave
from tqdm import tqdm


def build_polar_grid(S: int, height: int, width: int, device: str) -> torch.Tensor:
    i = torch.arange(0, height).unsqueeze(1).repeat(1, width)
    j = torch.arange(0, width).unsqueeze(0).repeat(height, 1)

    y = S / 2.0 - S / 2.0 / height * (height - 1 - i) * torch.sin(2 * torch.pi * j / width)
    x = S / 2.0 + S / 2.0 / height * (height - 1 - i) * torch.cos(2 * torch.pi * j / width)

    grid_x = (y / (S / 2.0) - 1).unsqueeze(2)
    grid_y = (x / (S / 2.0) - 1).unsqueeze(2)
    grid = torch.cat((grid_x, grid_y), dim=2).unsqueeze(0).to(device)
    return grid


def transform_image(img_name: str, input_dir: str, output_dir: str, grid: torch.Tensor, device: str):
    input_path = os.path.join(input_dir, img_name)
    try:
        signal = imread(input_path)
        if signal.dtype != "uint8":
            signal = (signal * 255 / signal.max()).astype("uint8")
        if signal.ndim == 2:
            signal = np.stack([signal] * 3, axis=-1)
        elif signal.shape[-1] != 3:
            raise ValueError(f"Unsupported image shape: {signal.shape}")

        tensor = torch.from_numpy(signal).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        out = F.grid_sample(tensor, grid, mode="bilinear", align_corners=True)
        out_image = (out.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".jpg")
        imsave(output_path, out_image, format="jpg")
        return True
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Polar transformation for satellite images")
    parser.add_argument("--input_dir", required=True, help="Directory of input satellite images")
    parser.add_argument("--output_dir", required=True, help="Directory to save transformed images")
    parser.add_argument("--S", type=int, default=256, help="Satellite image size (default: 256)")
    parser.add_argument("--height", type=int, default=256, help="Output height (default: 256)")
    parser.add_argument("--width", type=int, default=512, help="Output width (default: 512)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    grid = build_polar_grid(args.S, args.height, args.width, args.device)

    images = [f for f in os.listdir(args.input_dir) if f.lower().endswith((".jpg", ".png"))]
    print(f"Found {len(images)} images in {args.input_dir}")

    success = 0
    for img_name in tqdm(images, desc="Polar transform"):
        if transform_image(img_name, args.input_dir, args.output_dir, grid, args.device):
            success += 1

    print(f"Done: {success}/{len(images)} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
