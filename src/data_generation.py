"""
Usage:
    python data_generation.py \\
        --caption_path /path/to/captions_train2014.json \\
        --image_dir /path/to/train2014 \\
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

# --- Constants ---
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


# --- Model Loading and Utilities ---
def load_models(device: torch.device) -> Dict[str, Any]:
    """Loads and initializes all required models and tokenizers."""
    print(f"Loading Llama 3 model: {LLAMA_MODEL_ID}")
    model_llama = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer_llama = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)

    print(f"Loading LLaVA model: {LLAVA_MODEL_ID}")
    processor_llava = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)
    model_llava = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID, torch_dtype=torch.bfloat16
    ).to(device)

    return {
        "llama": model_llama,
        "llama_tokenizer": tokenizer_llama,
        "llava": model_llava,
        "llava_processor": processor_llava,
    }


def generate_with_llama(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    device: torch.device,
) -> str:
    """Generic function to generate text using the Llama model."""
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response_ids = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text.strip().rstrip('.')


# --- Core Logic Functions ---
def get_candidate_object(
    caption: str, use_random: bool, models: Dict[str, Any], device: torch.device
) -> str:
    """
    Generates a candidate object name.

    Args:
        caption: The original image caption.
        use_random: If True, generates a random object. Otherwise, generates
                    an object contextually related to the caption.
        models: Dictionary containing the loaded models and tokenizers.
        device: The device to run inference on.

    Returns:
        The name of the generated object as a string.
    """
    if use_random:
        prompt = "Name a random object."
    else:
        prompt = (
            f"Name an object that is not mentioned in the caption, but is "
            f"likely to be in the image corresponding to the caption '{caption}'."
        )

    messages = [
        {"role": "system", "content": "You are a helpful chatbot that answers with only one word."},
        {"role": "user", "content": prompt},
    ]

    return generate_with_llama(models["llama"], models["llama_tokenizer"], messages, device)


def is_object_in_image(
    image_path: Path, object_name: str, models: Dict[str, Any], device: torch.device
) -> bool:
    """
    Uses LLaVA to check if a given object is present in an image.

    Args:
        image_path: Path to the image file.
        object_name: The name of the object to check for.
        models: Dictionary containing the loaded models and tokenizers.
        device: The device to run inference on.

    Returns:
        True if the object is likely in the image, False otherwise.
    """
    image = Image.open(image_path)
    prompt = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        f"USER: <image>\nIs there {object_name} in this image? Answer only with 'yes' or 'no'. ASSISTANT:"
    )

    processor = models["llava_processor"]
    model = models["llava"]

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs, max_new_tokens=10, pad_token_id=processor.tokenizer.eos_token_id
    )

    response = processor.decode(output[0], skip_special_tokens=True)
    # Extract the final part of the response after "ASSISTANT:"
    answer = response.split("ASSISTANT:")[-1].strip().lower()

    return "yes" in answer


def generate_negated_caption(
    original_caption: str, absent_object: str, models: Dict[str, Any], device: torch.device
) -> str:
    """
    Rewrites a caption to state the absence of a specific object.

    Args:
        original_caption: The original caption.
        absent_object: The object confirmed to be absent from the image.
        models: Dictionary containing the loaded models and tokenizers.
        device: The device to run inference on.

    Returns:
        The new, negated caption.
    """
    examples = [
        {
            "caption": "a woman riding a horse in a field",
            "object": "hat",
            "updated_caption": "a woman not wearing a hat riding a horse in a field"
        },
        {
            "caption": "a child playing with a ball in the park",
            "object": "tree",
            "updated_caption": "a child playing with a ball in the park with no tree"
        }
    ]

    messages = [{"role": "system", "content": "You are a helpful chatbot that generates a concise caption."}]
    for ex in examples:
        messages.append({"role": "user", "content": f"Add the absence of the {ex['object']} to the caption '{ex['caption']}'."})
        messages.append({"role": "assistant", "content": ex['updated_caption']})

    messages.append({"role": "user", "content": f"Add the absence of the {absent_object} to the caption '{original_caption}'."})

    return generate_with_llama(models["llama"], models["llama_tokenizer"], messages, device)


# --- Main Execution ---
def main(args: argparse.Namespace):
    """Main function to orchestrate the caption generation process."""
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Data and Models ---
    models = load_models(torch.device(device))
    coco_annotation_path = Path(args.caption_path)
    coco_images_path = Path(args.image_dir)
    coco = COCO(str(coco_annotation_path))

    with open(coco_annotation_path, 'r') as f:
        original_data = json.load(f)

    all_captions = []
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            all_captions.append((img_id, ann['id'], ann['caption']))

    captions_to_process = all_captions[args.caption_start:args.caption_end]
    print(f"Processing {len(captions_to_process)} captions from index {args.caption_start} to {args.caption_end}.")

    # --- Process Captions ---
    updated_annotations_map = {}
    for img_id, ann_id, caption in tqdm(captions_to_process, desc="Processing Captions"):
        candidate_object = get_candidate_object(caption, args.use_random_object, models, device)
        img_info = coco.loadImgs([img_id])[0]
        image_path = coco_images_path / img_info['file_name']

        if not is_object_in_image(image_path, candidate_object, models, device):
            updated_caption = generate_negated_caption(caption, candidate_object, models, device)
            updated_annotations_map[ann_id] = updated_caption
            # print(f"Original: '{caption}' -> Updated: '{updated_caption}' (Object: {candidate_object})")

    # --- Save Results ---
    final_annotations = []
    for ann in original_data['annotations']:
        if ann['id'] in updated_annotations_map:
            ann['updated_caption'] = updated_annotations_map[ann['id']]
            final_annotations.append(ann)

    updated_data = {
        "info": original_data["info"],
        "licenses": original_data["licenses"],
        "images": original_data["images"],
        "annotations": final_annotations,
    }

    output_filename = f"negationclip_captions_train2014_repro.json"
    output_path = Path(args.output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(updated_data, f, indent=4)

    print(f"\nProcessing complete. Found {len(final_annotations)} valid negated captions.")
    print(f"Updated annotations saved to {output_path}")


def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate negated COCO captions using LLMs.")
    
    parser.add_argument('--device', type=int, default=0, help='GPU device ID to use.')
    parser.add_argument('--caption_path', type=str, required=True, help='Path to the COCO captions annotation file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing COCO images.')
    parser.add_argument('--output_dir', type=str, default='./annotations', help='Directory to save the output JSON file.')
    
    parser.add_argument('--caption_start', type=int, default=0, help='Starting index of captions to process.')
    parser.add_argument('--caption_end', type=int, default=50000, help='Ending index of captions to process.')
    parser.add_argument('--use_random_object', action='store_true', help='Generate a random object instead of a contextual one.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)