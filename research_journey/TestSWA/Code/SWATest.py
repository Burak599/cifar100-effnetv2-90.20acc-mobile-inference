import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from timm.data import Mixup
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch_lr_finder import LRFinder
import numpy as np
import random
import os
import ttach as tta


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility across runs.
    Covers Python's random, NumPy, and PyTorch (both CPU and CUDA).
    Also forces cuDNN into deterministic mode to avoid non-deterministic ops.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def analyze_per_class_accuracy(model, loader, device, classes):
    """
    Evaluates per-class accuracy on the given data loader and prints the
    10 classes with the lowest accuracy (i.e., the hardest to classify).

    Args:
        model:   The trained PyTorch model to evaluate.
        loader:  DataLoader for the evaluation split (typically validation set).
        device:  Device to run inference on.
        classes: List of class name strings, indexed by integer class label.
    """
    model.eval()

    # Accumulators for correct predictions and total samples per class
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            # Accumulate results for each sample in the batch
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Compute accuracy per class and sort ascending (worst classes first)
    accs = []
    for i in range(100):
        acc = 100 * class_correct[i] / class_total[i]
        accs.append((classes[i], acc))

    accs.sort(key=lambda x: x[1])
    print("\n--- Top 10 Most Difficult Classes ---")
    for name, acc in accs[:10]:
        print(f"{name}: {acc:.2f}%")


def main():
    print("Started!")

    # Fix all random seeds for reproducibility
    seed_everything(42)

    # Dictionary to track per-epoch metrics for later plotting
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Augmentation & Preprocessing ---
    # Training transform: random resized crop, RandAugment, horizontal flip, normalize
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
        #transforms.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.2, 3))
    ])

    # Validation transform: deterministic resize, center crop, and normalize (no augmentation)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    # --- Dataset & DataLoader Setup ---
    # Download CIFAR-100 and create train/validation splits
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_val)

    # num_workers=8 enables multi-process data loading; pin_memory speeds up CPU->GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # --- Model Setup ---
    # Load EfficientNetV2-S pretrained on ImageNet with dropout=0.3 for regularization
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, dropout=0.3)

    # Replace the final classification head to output 100 classes (CIFAR-100)
    # In EfficientNetV2-S, the last layer is index 1 of the classifier sequential
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 100)

    model = model.to(device)
    #batch_size = find_optimal_batch_size(model)

    # --- Load Pretrained Weights ---
    # Get relative script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to Resume2/Weights for best_model_10 and last_checkpoint_10
    checkpoint_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Test8Resume2", "Weights", "last_checkpoint_10.pth"))
    best_weight_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Test8Resume2", "Weights", "best_model_10.pth"))

    if os.path.exists(best_weight_path):
        print(f"\n--- Loading best checkpoint: {best_weight_path} ---")
        checkpoint = torch.load(best_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Reset training state since this is a fresh fine-tuning run
        start_epoch = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float('inf')
    else:
        print(f"ERROR: Checkpoint file not found: {best_weight_path}")
        return

    classes = train_dataset.classes

    # --- Reload Best Weights for Evaluation ---
    print("\n--- Computing Final Score ---")

    checkpoint_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Test8Resume2", "Weights", "best_model_10.pth"))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Weights loaded: {checkpoint_path}")

    # --- Soft Ensemble Setup ---
    # Load multiple saved checkpoints and combine their predictions via probability averaging
    paths = [
        os.path.abspath(os.path.join(current_dir, "..", "..", "Test8Resume2", "Weights", "best_model_10.pth")),
        os.path.abspath(os.path.join(current_dir, "..", "..", "Test8Resume4", "Weights", "best_model_12.pth"))
    ]
    ensemble_models = []

    print("\n--- Preparing Soft Ensemble ---")

    for p in paths:
        if os.path.exists(p):
            # Re-instantiate the architecture for each checkpoint
            m = models.efficientnet_v2_s(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, 100)

            checkpoint = torch.load(p)
            s_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            m.load_state_dict(s_dict)

            m = m.to(device)
            m.eval()
            ensemble_models.append(m)
            print(f"Model loaded: {p}")
        else:
            print(f"WARNING: Checkpoint not found: {p}")

    if len(ensemble_models) < 2:
        print("ERROR: Not enough models for ensemble. Exiting.")
        return

    # --- Ensemble Evaluation ---
    # Average softmax probabilities across all models and pick the highest-confidence class
    val_correct, val_total = 0, 0
    print(f"\nRunning Soft Ensemble with {len(ensemble_models)} models (no TTA)...")
    print("Evaluating, this may take a few seconds...")

    val_correct, val_total = 0, 0
    print(f"\nRunning Soft Ensemble with {len(ensemble_models)} models...")
    print("Evaluating...")

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Accumulate softmax probabilities from each model in the ensemble
            batch_probs = torch.zeros((images.size(0), 100)).to(device)

            for m in ensemble_models:
                outputs = m(images)
                # Convert logits to probabilities and accumulate
                batch_probs += torch.softmax(outputs, dim=1)

            # Average the accumulated probabilities across all ensemble members
            avg_probs = batch_probs / len(ensemble_models)

            # Select the class with the highest average probability
            predictions = avg_probs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    final_acc = 100. * val_correct / val_total

    print(f"\n{'='*40}")
    print(f"FINAL ENSEMBLE ACCURACY: {final_acc:.2f}%")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()