"""
Step 2a: Semantic Caption Generation
Generates geographic-identity-aware text prompts for each coarse street view image.

Pipeline per image:
    1. InstructBLIP -> detailed scene description
    2. LaMini-T5   -> compressed one-sentence caption
    3. Append location tag + "real photo, street-level view." suffix

Output: JSONL file with {"image_path": ..., "caption": ...} per line.

Usage:
    python step2_caption_generation.py \
        --image_dir  ./coarse_gan \
        --output_jsonl ./captions.jsonl \
        --city Kathmandu --country Nepal
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)

BLIP_MODEL = "Salesforce/instructblip-flan-t5-xl"
T5_MODEL = "MBZUAI/LaMini-T5-738M"

DETAIL_PROMPT = (
    "Describe the street scene in the image in detail. Include road type, number of lanes, buildings, "
    "vehicles, people, vegetation, weather, and lighting."
)

COMPRESSION_PROMPT_TEMPLATE = (
    "Summarize the detailed scene below into one fluent, descriptive sentence. "
    "Avoid any repetition, redundant words.\n\n{caption}"
)


def load_models(device: str):
    print("Loading InstructBLIP...")
    blip_processor = InstructBlipProcessor.from_pretrained(BLIP_MODEL)
    blip_model = InstructBlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL, torch_dtype=torch.float16
    ).to(device)

    print("Loading LaMini-T5...")
    t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(device)

    print("Models loaded.\n")
    return blip_processor, blip_model, t5_tokenizer, t5_model


def generate_caption(
    image_path: str,
    blip_processor,
    blip_model,
    t5_tokenizer,
    t5_model,
    device: str,
    city: str = "",
    country: str = "",
) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")

        blip_inputs = blip_processor(
            images=image, text=DETAIL_PROMPT, return_tensors="pt"
        ).to(device, torch.float16)
        with torch.inference_mode():
            blip_output = blip_model.generate(**blip_inputs, max_new_tokens=120)
        raw_caption = blip_processor.tokenizer.decode(blip_output[0], skip_special_tokens=True)

        compression_prompt = COMPRESSION_PROMPT_TEMPLATE.format(caption=raw_caption)
        t5_inputs = t5_tokenizer(compression_prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            t5_output = t5_model.generate(**t5_inputs, max_new_tokens=60)
        compressed = t5_tokenizer.decode(t5_output[0], skip_special_tokens=True)

        location_tag = f" Scene is located in {city}, {country}." if city and country else ""
        final_caption = f"{compressed}{location_tag} real photo, street-level view."
        return {"image_path": image_path, "caption": final_caption}

    except Exception as e:
        print(f"Error on {image_path}: {e}")
        return {"image_path": image_path, "caption": f"[ERROR] {e}"}


def main():
    parser = argparse.ArgumentParser(description="Batch caption generation for street view images")
    parser.add_argument("--image_dir",    required=True, help="Directory of coarse street view images")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL file path")
    parser.add_argument("--city",    default="", help="City name for location tag, e.g. Kathmandu")
    parser.add_argument("--country", default="", help="Country name for location tag, e.g. Nepal")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if bool(args.city) != bool(args.country):
        parser.error("--city and --country must be provided together or not at all")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        os.path.join(args.image_dir, f)
        for f in sorted(os.listdir(args.image_dir))
        if os.path.splitext(f.lower())[1] in exts
    ]
    print(f"Found {len(image_paths)} images in {args.image_dir}")
    if args.city:
        print(f"Location tag: {args.city}, {args.country}")

    blip_processor, blip_model, t5_tokenizer, t5_model = load_models(args.device)

    results = []
    for path in tqdm(image_paths, desc="Generating captions"):
        result = generate_caption(
            path, blip_processor, blip_model, t5_tokenizer, t5_model,
            args.device, args.city, args.country,
        )
        results.append(result)
        print(f"  {os.path.basename(path)} -> {result['caption'][:80]}...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    errors = sum(1 for r in results if r["caption"].startswith("[ERROR]"))
    print(f"\nDone: {len(results) - errors}/{len(results)} captions saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
