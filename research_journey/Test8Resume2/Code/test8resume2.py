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

# --- PATH CONFIGURATION ---

# Get the absolute path of the current directory where the script is located
BASE_PATH = "/home/burak/cifar100-effnetv2-90.20acc-mobile-inference/research_journey/Test8Resume2"

# Define output directories for organization
WEIGHTS_DIR = os.path.join(BASE_PATH, "user_weights")
PLOT_DIR = os.path.join(BASE_PATH, "user_plots")

# 1. PRETRAINED_WEIGHT_PATH: Move up 2 levels (cd .. / cd ..), enter Test8Resume1, then enter Weights
# This points to the previously trained model weights to start fine-tuning
PRETRAINED_WEIGHT_PATH = os.path.abspath(os.path.join(BASE_PATH, "..", "TestResume1", "user_weights", "user_best_model_9.pth"))

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 2. CHECKPOINT_SAVE_PATH: Saves the full training state whenever validation loss improves
CHECKPOINT_SAVE_PATH = os.path.join(WEIGHTS_DIR, "user_best_model_10.pth")

# 3. LAST_CHECKPOINT_PATH: Rolling checkpoint updated every epoch for crash recovery
LAST_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "user_last_checkpoint_10.pth")

# 4. FINAL_MODEL_SAVE_PATH: Saves only the model state_dict after the final epoch
FINAL_MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, "user_last_epochs_10.pth")

# 5. PLOT_SAVE_PATH: File path to save the training/validation curves as an image
PLOT_SAVE_PATH = os.path.join(PLOT_DIR, "user_training_history_plot_10.png")
# --------------------------


def seed_everything(seed=42):
    """
    Seeds all relevant random number generators to ensure reproducibility
    across runs. Covers Python, NumPy, and both single/multi-GPU PyTorch.
    Also disables cuDNN non-determinism.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Required for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_optimal_batch_size(model, input_shape=(3, 224, 224), device="cuda", threshold=0.8):
    """
    Finds the largest power-of-2 batch size that fits within a given fraction
    of total GPU memory (default: 80%).

    Runs forward + backward passes with synthetic data to simulate real
    training memory usage, then returns the largest safe batch size found.

    Args:
        model: The PyTorch model to test.
        input_shape (tuple): Shape of a single input sample (C, H, W).
        device (str): Target device, e.g. "cuda".
        threshold (float): Maximum fraction of GPU memory to use (0.0 - 1.0).

    Returns:
        int: The largest power-of-2 batch size that fits within the memory threshold.
    """
    model.to(device)
    model.train()  # Training mode allocates more memory due to stored activations

    # Query total GPU memory and compute target ceiling
    gpu_capacity = torch.cuda.get_device_properties(device).total_memory
    target_memory = gpu_capacity * threshold

    batch_size = 2
    best_batch = 2

    print(f"Target Memory Usage: {target_memory / 1e9:.2f} GB")

    try:
        while True:
            # Clear cached memory and reset peak stats before each trial
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Generate synthetic inputs and random class labels
            inputs = torch.randn(batch_size, *input_shape).to(device)
            targets = torch.randint(0, 100, (batch_size,)).to(device)

            # Full forward + backward pass to simulate real training memory load
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()

            # Measure peak memory allocated during this pass
            used_memory = torch.cuda.max_memory_allocated(device)

            if used_memory < target_memory:
                print(f"Batch {batch_size} tested: {used_memory / 1e9:.2f} GB used (OK)")
                best_batch = batch_size
                batch_size *= 2
            else:
                print(f"Batch {batch_size} exceeded limit! ({used_memory / 1e9:.2f} GB)")
                break

            # Safety cap to prevent runaway loop
            if batch_size > 2048:
                break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM: batch size {batch_size} did not fit.")
        else:
            print(f"Unexpected error: {e}")

    # Free memory and clear gradients before returning
    torch.cuda.empty_cache()
    model.zero_grad()

    print(f"--- Recommended Maximum Batch Size: {best_batch} ---")
    return best_batch


def plot_history(history):
    """
    Plots training and validation accuracy and loss curves side by side,
    then saves the figure to PLOT_SAVE_PATH.

    Args:
        history (dict): Dictionary with keys 'train_acc', 'val_acc',
                        'train_loss', 'val_loss', each containing a list
                        of per-epoch values.
    """
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_acc"], 'bo-', label='Train Accuracy')
    plt.plot(epochs, history["val_acc"], 'ro-', label='Validation Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], 'b--', label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'r--', label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)  # Save the figure to the configured output path
    plt.show()


def analyze_per_class_accuracy(model, loader, device, classes):
    """
    Evaluates per-class accuracy on a given data loader and prints
    the 10 classes with the lowest accuracy (hardest classes).

    Args:
        model: Trained PyTorch model.
        loader: DataLoader for the evaluation set.
        device: Device to run inference on.
        classes (list[str]): List of class name strings indexed by class ID.
    """
    model.eval()
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Compute per-class accuracy and collect into a sortable list
    accs = []
    for i in range(100):
        acc = 100 * class_correct[i] / class_total[i]
        accs.append((classes[i], acc))

    # Sort ascending and print the 10 worst-performing classes
    accs.sort(key=lambda x: x[1])
    print("\n--- 10 Hardest Classes ---")
    for name, acc in accs[:10]:
        print(f"{name}: {acc:.2f}%")


def main():
    print("Started!!")

    # Fix all random seeds for reproducibility
    seed_everything(42)

    # Dictionary to track per-epoch metrics throughout training
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Augmentation for Training ---
    # Same spatial pipeline as v8, but Mixup/CutMix strength is further reduced
    # below v9 levels — this is an even gentler final fine-tuning stage
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.2, 3))
    ])

    # --- Validation Transform ---
    # Resize to 256 → center crop to 224 → normalize (no augmentation)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    # --- Datasets and DataLoaders ---
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # --- Model Setup ---
    # Load EfficientNetV2-S pretrained on ImageNet with dropout=0.3,
    # then replace the final classifier head for 100 CIFAR-100 classes
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, dropout=0.3)

    # In EfficientNetV2-S, the classification head is at model.classifier[1]
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 100)

    model = model.to(device)

    # Batch size is fixed here; find_optimal_batch_size() can be used instead
    batch_size = 64
    print(f"Batch Size: {batch_size}")

    # --- Mixup / CutMix Augmentation ---
    # Further reduced vs v9 (mixup_alpha=0.2→0.2, cutmix_alpha=0.4→0.3, prob=0.4→0.3)
    # Minimal blending to preserve the sharpness of a nearly converged model
    mixup_fn = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=0.3,
        prob=0.3,
        switch_prob=0.5,
        mode='batch',
        num_classes=100
    )

    # --- Loss Function and Optimizer ---
    # Fine-tuning stage: SAM is dropped in favour of standard AdamW with a very
    # small learning rate (8e-6) to apply gentle, precise weight corrections
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    num_epochs = 50   # 50 epochs is sufficient for this final polishing stage
    base_lr = 8e-6    # Very low LR to avoid overshooting the converged optimum

    # Standard AdamW (no SAM) — fine-tuning does not need sharpness-aware updates
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    # CosineAnnealingLR: smoothly decays LR from base_lr to eta_min over num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=8e-7)

    # --- Optional: LR Range Test (disabled by default) ---
    # Uncomment the block below to run the LR finder before training.
    # It will plot the loss-vs-LR curve and suggest an optimal max_lr.
    """
    print("\n--- Searching for Optimal Learning Rate ---")
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")

    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']
    best_lr = lrs[np.argmin(np.gradient(losses))]
    print(f"--- Suggested Max LR: {best_lr:.2e} ---")

    result = lr_finder.plot()

    # lr_finder.plot() may return a tuple (axis, ...) or just an axis
    ax = result[0] if isinstance(result, (list, tuple)) else result

    fig = ax.get_figure()
    fig.savefig("lr_finder_result.png")
    print("--- LR Finder plot saved as 'lr_finder_result.png' ---")

    lr_finder.reset()
    """

    # --- Early Stopping Configuration ---
    patience = 1000   # Number of epochs to wait without val_loss improvement
    counter = 0
    best_val_loss = float('inf')

    # --- Load Pretrained Weights for Fine-Tuning ---
    # This script is a fine-tuning stage: it loads the best checkpoint from a
    # previous training run (PRETRAINED_WEIGHT_PATH) and starts a fresh training
    # history and optimizer state, applying only small corrective updates.
    # Unlike a resume, epoch counter and history are intentionally reset to zero.
    start_epoch = 0

    if os.path.exists(PRETRAINED_WEIGHT_PATH):
        print(f"\n--- Loading pretrained weights for fine-tuning... ---")
        checkpoint = torch.load(PRETRAINED_WEIGHT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Reset training state — this is a fresh fine-tuning run, not a resume
        start_epoch = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float('inf')
    else:
        print("WARNING: Pretrained weight file not found!")
        return

    # --- TensorBoard Writer ---
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(BASE_PATH, f"user_runs10/baseline_{current_time}")
    writer = SummaryWriter(log_dir)

    # ===================== TRAINING LOOP =====================
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Apply Mixup or CutMix to the current batch
            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            # Standard forward → backward → gradient clip → optimizer step
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent large gradient updates
            optimizer.step()

            # Accumulate loss and accuracy statistics once per batch
            running_loss += loss.item() * images.size(0)
            # When using soft labels (Mixup/CutMix), compare argmax of both
            if labels.ndim == 2:
                correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            else:
                correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        # CosineAnnealingLR steps once per epoch (not per iteration)
        scheduler.step()

        # Compute average training loss and accuracy for this epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total

        # ==================== VALIDATION LOOP ====================
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total

        # --- Early Stopping & Best Checkpoint Saving ---
        # Save the full training state whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),          # Consistent key name
                'optimizer_state_dict': optimizer.state_dict(),  # Consistent key name
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_val_loss,
                'history': history
            }
            torch.save(state, CHECKPOINT_SAVE_PATH)  # Save best checkpoint
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Record metrics for plotting
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Log scalars to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # --- Rolling Checkpoint (saved every epoch for crash recovery) ---
        # Overwrites the previous rolling checkpoint so training can always
        # be resumed from the last completed epoch via LAST_CHECKPOINT_PATH.
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_val_loss
        }
        torch.save(checkpoint_data, LAST_CHECKPOINT_PATH)  # Overwrite rolling checkpoint

    writer.close()

    # --- Post-Training Analysis ---
    classes = train_dataset.classes
    analyze_per_class_accuracy(model, val_loader, device, classes)

    # Plot and save training curves
    plot_history(history)

    # Save final model weights
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
    print("Model saved!")


if __name__ == "__main__":
    main()
