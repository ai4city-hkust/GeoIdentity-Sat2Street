"""
Train Step 2b: UNet Fine-tuning for Geographic Identity Preservation
Fine-tunes the Stable Diffusion UNet on street view images conditioned on
semantic captions (with location metadata embedded in text).

Freezes: VAE, CLIP text encoder (uses SD original CLIP)
Trains:  UNet only

Input JSONL format (one record per line):
    {"image_path": "path/to/image.jpg", "caption": "scene description..."}
    Caption should include location tag, e.g.:
    "A two-lane road lined with palm trees. Scene is located in Kathmandu, Nepal. real photo, street-level view."

Usage:
    python train_unet.py \
        --captions_jsonl /path/to/captions.jsonl \
        --image_root     /path/to/train_B \
        --output_dir     ./unet_checkpoints \
        --num_epochs     15 \
        --batch_size     4
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from diffusers import DDPMScheduler, StableDiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer


class ImageCaptionDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str, tokenizer: CLIPTokenizer, image_size: int = 512):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                img_path = data["image_path"]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(image_root, img_path)
                self.samples.append((img_path, data["caption"]))

        self.tokenizer = tokenizer
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": self.transform(image),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }


def train(args):
    device = torch.device(args.device)

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )
    pipe.to(device)

    vae = pipe.vae
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    dataset = ImageCaptionDataset(args.captions_jsonl, args.image_root, tokenizer, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        unet.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * 0.18215
                text_embeddings = text_encoder(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )[0]

            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            loss = nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f"[Epoch {epoch+1}/{args.num_epochs}] step {step}/{len(dataloader)} | loss={loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.num_epochs:
            ckpt_path = os.path.join(args.output_dir, f"unet_epoch_{epoch}.pt")
            torch.save(unet.state_dict(), ckpt_path)
            print(f"Saved: {ckpt_path}")

    print(f"\nTraining complete. Checkpoints in: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SD UNet for geographic identity preservation")
    parser.add_argument("--captions_jsonl", required=True, help="JSONL with image_path and caption fields")
    parser.add_argument("--image_root",     default="", help="Prepended to relative image_path in JSONL")
    parser.add_argument("--output_dir",     required=True, help="Directory to save UNet checkpoints")
    parser.add_argument("--num_epochs",     type=int,   default=15)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-5)
    parser.add_argument("--image_size",     type=int,   default=512)
    parser.add_argument("--save_every",     type=int,   default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
