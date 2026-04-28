"""
Step 1b: Pix2PixHD GAN Inference
Runs pix2pixHD to synthesize coarse street view panoramas from polar-transformed
satellite images. Requires the pix2pixHD repo cloned alongside this project.

Setup:
    git clone https://github.com/NVIDIA/pix2pixHD
    pip install dominate tensorboardX

    Place your trained checkpoint at:
    pix2pixHD/checkpoints/<checkpoint_name>/latest_net_G.pth
"""

import argparse
import os
import shutil
import subprocess


def prepare_dataset(polar_dir: str, pix2pix_root: str, dataset_name: str):
    test_a_dir = os.path.join(pix2pix_root, "datasets", dataset_name, "test_A")
    os.makedirs(test_a_dir, exist_ok=True)

    images = [f for f in os.listdir(polar_dir) if f.lower().endswith((".jpg", ".png"))]
    for img in images:
        shutil.copy(os.path.join(polar_dir, img), os.path.join(test_a_dir, img))

    print(f"Copied {len(images)} images to {test_a_dir}")
    return test_a_dir


def run_inference(
    pix2pix_root: str,
    checkpoint_name: str,
    dataset_name: str,
    how_many: int,
    checkpoint_path: str = None,
):
    ckpt_dir = os.path.join(pix2pix_root, "checkpoints", checkpoint_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    if checkpoint_path:
        dest = os.path.join(ckpt_dir, "latest_net_G.pth")
        if not os.path.exists(dest):
            shutil.copy(checkpoint_path, dest)
            print(f"Copied checkpoint to {dest}")

    cmd = [
        "python", "test.py",
        "--name", checkpoint_name,
        "--no_instance",
        "--label_nc", "0",
        "--dataroot", os.path.join(pix2pix_root, "datasets"),
        "--resize_or_crop", "none",
        "--which_epoch", "latest",
        "--n_blocks_global", "6",
        "--n_downsample_global", "3",
        "--netG", "local",
        "--ngf", "64",
        "--n_local_enhancers", "1",
        "--how_many", str(how_many),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=pix2pix_root)
    if result.returncode != 0:
        raise RuntimeError("pix2pixHD inference failed")

    output_dir = os.path.join(pix2pix_root, "results", checkpoint_name, "test_latest", "images")
    print(f"Results saved to: {output_dir}")
    return output_dir


def collect_outputs(result_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for f in os.listdir(result_dir):
        if f.endswith("_synthesized_image.jpg"):
            base = f.replace("_synthesized_image.jpg", "")
            shutil.copy(
                os.path.join(result_dir, f),
                os.path.join(output_dir, f"{base}_fake.jpg"),
            )
            count += 1
    print(f"Collected {count} synthesized images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pix2PixHD GAN inference for street view synthesis")
    parser.add_argument("--polar_dir", required=True, help="Directory of polar-transformed satellite images")
    parser.add_argument("--output_dir", required=True, help="Directory to save coarse street view outputs")
    parser.add_argument("--pix2pix_root", default="./pix2pixHD", help="Path to cloned pix2pixHD repo")
    parser.add_argument("--checkpoint_name", default="sat2svi_pix2pix", help="Checkpoint folder name")
    parser.add_argument("--checkpoint_path", default=None, help="Path to latest_net_G.pth weight file")
    parser.add_argument("--how_many", type=int, default=10000, help="Max number of images to process")
    args = parser.parse_args()

    if not os.path.isdir(args.pix2pix_root):
        raise FileNotFoundError(
            f"pix2pixHD repo not found at {args.pix2pix_root}.\n"
            "Run: git clone https://github.com/NVIDIA/pix2pixHD"
        )

    prepare_dataset(args.polar_dir, args.pix2pix_root, args.checkpoint_name)

    result_dir = run_inference(
        args.pix2pix_root,
        args.checkpoint_name,
        args.checkpoint_name,
        args.how_many,
        args.checkpoint_path,
    )

    collect_outputs(result_dir, args.output_dir)


if __name__ == "__main__":
    main()
