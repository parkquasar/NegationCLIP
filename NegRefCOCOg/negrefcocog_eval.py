"""
NegRefCOCOg Evaluation Script

This script evaluates CLIP models on the NegRefCOCOg benchmark for referring expression comprehension.
It computes accuracy by comparing predicted bounding boxes against ground truth annotations.

Usage:
    python negrefcocog_eval.py --arch ViT-B/16 --load_dir /path/to/checkpoint.pt --device cuda:0
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class NegRefCOCOgEvaluator:
    """Evaluator for NegRefCOCOg benchmark using CLIP models."""

    def __init__(
        self,
        model_arch: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        image_dir: str = "./coco_images/train2014"
    ):
        """
        Initialize the evaluator.

        Args:
            model_arch: CLIP model architecture (e.g., 'ViT-B/16', 'ViT-B/32')
            checkpoint_path: Optional path to fine-tuned model checkpoint
            device: Device to run evaluation on ('cuda', 'cpu', or 'cuda:X')
            image_dir: Directory containing COCO images
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_dir = Path(image_dir)

        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_arch, device=self.device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.model.eval()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from: {checkpoint_path}")

    def _compute_similarity_scores(
        self,
        image_path: Path,
        text: str,
        bboxes: List[List[int]]
    ) -> List[float]:
        """
        Compute similarity scores between text and cropped image regions.

        Args:
            image_path: Path to the image file
            text: Text description / referring expression
            bboxes: List of bounding boxes in [x, y, w, h] format

        Returns:
            List of similarity scores for each bounding box
        """
        # Tokenize text
        text_tokens = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            # Encode text
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = []

            # Process each bounding box
            for bbox in bboxes:
                x, y, w, h = bbox

                # Crop and preprocess image region
                image = Image.open(image_path)
                cropped = image.crop((x, y, x + w, y + h))
                cropped_tensor = self.preprocess(cropped).unsqueeze(0).to(self.device)

                # Encode cropped image
                image_features = self.model.encode_image(cropped_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Compute similarity
                similarity = (image_features @ text_features.T).item()
                similarities.append(similarity)

        return similarities

    def evaluate(
        self,
        annotation_file: str,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on NegRefCOCOg annotations.

        Args:
            annotation_file: Path to JSON annotation file
            verbose: Whether to show progress bar

        Returns:
            Dictionary containing evaluation metrics
        """
        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        correct = 0
        total = 0

        # Iterate through annotations
        iterator = tqdm(annotations, desc="Evaluating") if verbose else annotations

        for item in iterator:
            image_file = item['image']
            phrase = item['phrase']
            ref_bbox = item['ref_bbox']
            bbox_list = item['bbox_list']

            # Compute similarities
            image_path = self.image_dir / image_file
            similarities = self._compute_similarity_scores(image_path, phrase, bbox_list)

            # Get predicted bbox (highest similarity)
            pred_idx = np.argmax(similarities)
            predicted_bbox = bbox_list[pred_idx]

            # Check if prediction matches ground truth
            if predicted_bbox == ref_bbox:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

        return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP models on NegRefCOCOg benchmark"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ViT-B/16",
        help="CLIP model architecture (default: ViT-B/16)"
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device to run evaluation on (default: cuda:1)"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="NegRefCOCOg.json",
        help="Path to annotation file (default: NegRefCOCOg.json)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./coco_images/train2014",
        help="Directory containing COCO images (default: ./coco_images/train2014)"
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Initialize evaluator
    evaluator = NegRefCOCOgEvaluator(
        model_arch=args.arch,
        checkpoint_path=args.load_dir,
        device=args.device,
        image_dir=args.image_dir
    )

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Evaluating on NegRefCOCOg Benchmark")
    print(f"{'='*60}")
    print(f"Model Architecture: {args.arch}")
    print(f"Checkpoint: {args.load_dir if args.load_dir else 'None (baseline)'}")
    print(f"Device: {args.device}")
    print(f"Annotation File: {args.annotation_file}")
    print(f"{'='*60}\n")

    results = evaluator.evaluate(args.annotation_file, verbose=True)

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
