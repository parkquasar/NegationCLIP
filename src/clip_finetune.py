"""
Usage:
    python clip_finetune.py --json_path <path_to_captions.json> \\
                               --image_dir <path_to_coco_images> \\
                               --clip_model "ViT-B/32" \\
"""

import argparse
import json
import os
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# --- Constants ---
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2


# --- Dataset Definition ---
class COCOCaptionDataset(Dataset):
    """
    Custom PyTorch Dataset for COCO captions.

    Args:
        captions_file (Path): Path to the JSON annotation file.
        image_dir (Path): Path to the directory containing COCO images.
        preprocess (callable): A function/transform to apply to the images.
    """
    def __init__(self, captions_file: Path, image_dir: Path, preprocess: callable):
        super().__init__()
        with open(captions_file, 'r') as f:
            coco_data = json.load(f)
        self.annotations = coco_data['annotations']
        self.images = {img['id']: img['file_name'] for img in coco_data['images']}
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        """
        Retrieves an item from the dataset.

        Returns:
            A tuple containing the preprocessed image, original caption,
            and the updated caption.
        """
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        image_file_name = self.images[image_id]
        image_path = self.image_dir / image_file_name

        image = Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image)

        original_caption = annotation['caption']
        updated_caption = annotation['updated_caption']

        return processed_image, original_caption, updated_caption


# --- Core Functions ---
def symmetric_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Calculates the symmetric contrastive loss."""
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2.0


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Trains the model for one epoch.

    Returns:
        A tuple containing the average training loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, _, updated_captions in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        text_inputs = clip.tokenize(updated_captions, truncate=True).to(device)

        # Forward pass
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_inputs)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Calculate loss
        loss = symmetric_contrastive_loss(logits_per_image, logits_per_text)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # For accuracy calculation
        labels = torch.arange(len(images), device=device)
        preds = logits_per_image.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Returns:
        A tuple containing the average validation loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, _, updated_captions in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            text_inputs = clip.tokenize(updated_captions, truncate=True).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            loss = symmetric_contrastive_loss(logits_per_image, logits_per_text)
            total_loss += loss.item()

            labels = torch.arange(len(images), device=device)
            preds = logits_per_image.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# --- Main Execution ---
def main(args: argparse.Namespace):
    """Main function to run the training and evaluation process."""
    # --- Setup ---
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # --- Model and Preprocessing ---
    model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    model = model.float()

    print("Freezing model parameters...")
    for param in model.visual.parameters():
        param.requires_grad = False
    for param in model.ln_final.parameters():
        param.requires_grad = False
    print("Vision encoder and final layer norm frozen.")

    # --- Data Loading ---
    captions_file = Path(args.json_path)
    image_dir = Path(args.image_dir)

    dataset = COCOCaptionDataset(captions_file, image_dir, preprocess)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split_idx = int(np.floor(VALIDATION_SPLIT * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
    print(f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # --- Optimizer and Training Loop ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    output_dir = Path(args.output_dir) / args.clip_model.replace('/', '-')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {output_dir}")

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.epoch):
        epoch_num = epoch + 1
        print(f"\n--- Epoch {epoch_num}/{args.epoch} ---")

        train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch_num} Training -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_dataloader, device)
        print(f"Epoch {epoch_num} Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_num
            checkpoint_path = output_dir / f'best_model_epoch_{best_epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🎉 New best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
    
    print(f"\nTraining finished. Best model was saved from epoch {best_epoch}.")


def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune CLIP text encoder on COCO Captions.")
    
    parser.add_argument('--device', type=int, default=0, help='GPU device ID to use.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the COCO-style JSON annotation file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing COCO images.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    
    parser.add_argument('--clip_model', type=str, default="ViT-B/32", help='Name of the CLIP model to use.')
    parser.add_argument('--epoch', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for the optimizer.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)