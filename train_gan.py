"""
Train Step 1b: Pix2PixHD GAN Training
Trains the GAN to synthesize coarse street view panoramas from polar-transformed
satellite images.

Dataset structure expected by pix2pixHD:
    <dataroot>/
        train_A/   <- polar-transformed satellite images (input)
        train_B/   <- ground-truth street view images (target)
        test_A/
        test_B/

Setup:
    git clone https://github.com/NVIDIA/pix2pixHD
    pip install dominate tensorboardX

Usage:
    python train_gan.py \
        --data_dir  /path/to/SAT2SVI \
        --pix2pix_root ./pix2pixHD \
        --name sat2svi_pix2pix \
        --niter 100 --niter_decay 100
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Pix2PixHD GAN training for satellite-to-street-view")
    parser.add_argument("--data_dir",       required=True, help="Dataset root with train_A / train_B subdirs")
    parser.add_argument("--pix2pix_root",   default="./pix2pixHD", help="Path to cloned pix2pixHD repo")
    parser.add_argument("--name",           default="sat2svi_pix2pix", help="Experiment name (checkpoint folder)")
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--load_size",      type=int, default=512)
    parser.add_argument("--fine_size",      type=int, default=256)
    parser.add_argument("--niter",          type=int, default=100, help="Epochs at initial learning rate")
    parser.add_argument("--niter_decay",    type=int, default=100, help="Epochs to linearly decay learning rate")
    parser.add_argument("--save_epoch_freq",type=int, default=5)
    parser.add_argument("--display_freq",   type=int, default=200)
    parser.add_argument("--print_freq",     type=int, default=100)
    args = parser.parse_args()

    if not os.path.isdir(args.pix2pix_root):
        print(f"pix2pixHD repo not found at {args.pix2pix_root}")
        print("Run: git clone https://github.com/NVIDIA/pix2pixHD")
        sys.exit(1)

    datasets_dir = os.path.join(args.pix2pix_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    link_path = os.path.join(datasets_dir, os.path.basename(args.name))
    if not os.path.exists(link_path):
        os.symlink(os.path.abspath(args.data_dir), link_path)

    cmd = [
        sys.executable, "train.py",
        "--name",               args.name,
        "--label_nc",           "0",
        "--no_instance",
        "--dataroot",           os.path.join(args.pix2pix_root, "datasets"),
        "--resize_or_crop",     "none",
        "--batchSize",          str(args.batch_size),
        "--loadSize",           str(args.load_size),
        "--fineSize",           str(args.fine_size),
        "--netG",               "local",
        "--n_local_enhancers",  "1",
        "--ngf",                "64",
        "--n_downsample_global","3",
        "--n_blocks_global",    "6",
        "--save_epoch_freq",    str(args.save_epoch_freq),
        "--display_freq",       str(args.display_freq),
        "--print_freq",         str(args.print_freq),
        "--niter",              str(args.niter),
        "--niter_decay",        str(args.niter_decay),
    ]

    print(f"Starting pix2pixHD training: {args.name}")
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=args.pix2pix_root)

    if result.returncode == 0:
        ckpt = os.path.join(args.pix2pix_root, "checkpoints", args.name, "latest_net_G.pth")
        print(f"\nTraining complete. Generator checkpoint: {ckpt}")
    else:
        print("\nTraining failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
