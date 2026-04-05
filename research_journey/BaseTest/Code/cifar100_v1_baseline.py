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
import os

# Get the absolute path of the current directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define output directories for organization
WEIGHTS_DIR = "user_weights"
PLOT_DIR = "user_plots"

# Automatically create directories if they do not exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 1. CHECKPOINT_SAVE_PATH: Saves the full training state (optimizer, scheduler, weights) 
# whenever validation loss reaches a new minimum.
CHECKPOINT_SAVE_PATH = os.path.join(WEIGHTS_DIR, "user_best_model.pth")

# 2. FINAL_MODEL_SAVE_PATH: Saves only the model state_dict after the final epoch.
FINAL_MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, "user_last_epochs.pth")

# 3. PLOT_SAVE_PATH: The file path where the training and validation Accuracy/Loss curves will be saved.
PLOT_SAVE_PATH = os.path.join(PLOT_DIR, "user_training_history_plot.png")
# --------------------------


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
                print(f"Batch {batch_size} tested: {used_memory / 1e9:.2f} GB used (Passed)")
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
            print(f"OOM: Batch size {batch_size} failed.")
        else:
            print(f"Error occurred: {e}")

    # Free memory and clear gradients before returning
    torch.cuda.empty_cache()
    model.zero_grad()

    print(f"--- Recommended Max Batch Size: {best_batch} ---")
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
    plt.plot(epochs, history["val_acc"], 'ro-', label='Val Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], 'b--', label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'r--', label='Val Loss')
    plt.title('Training & Validation Loss')
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
    print("\n--- Top 10 Most Difficult Classes ---")
    for name, acc in accs[:10]:
        print(f"{name}: %{acc:.2f}")


def main():
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
    # Resize → RandAugment (strong, magnitude=9) → Normalize → Random erasing
    # Note: RandomHorizontalFlip is absent in this baseline version
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])

    # --- Validation Transform ---
    # Resize to 256 only — CenterCrop is intentionally disabled (commented out)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        #transforms.CenterCrop(224),  # Disabled in this baseline
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
    # Load EfficientNetV2-S pretrained on ImageNet (no dropout override in this baseline),
    # then replace the final classifier head for 100 CIFAR-100 classes
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # In EfficientNetV2-S, the classification head is at model.classifier[1]
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 100)

    model = model.to(device)

    # Batch size is fixed here; find_optimal_batch_size() can be used instead
    batch_size = 64
    print(f"Batch Size: {batch_size}")

    # --- Loss Function and Optimizer ---
    # Baseline uses standard CrossEntropyLoss with label smoothing and AdamW.
    # No SAM, no Mixup/CutMix — this is the simplest training configuration.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)
    num_epochs = 10

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
    print(f"--- Max LR: {best_lr:.2e} ---")

    result = lr_finder.plot()

    # lr_finder.plot() may return a tuple (axis, ...) or just an axis
    ax = result[0] if isinstance(result, (list, tuple)) else result

    fig = ax.get_figure()
    fig.savefig("lr_finder_result.png")
    print("--- LR Finder plot saved as 'lr_finder_result.png'! ---")

    lr_finder.reset()
    """

    # --- Learning Rate Scheduler ---
    # OneCycleLR: warm up for 30% of training, then cosine anneal to near zero.
    # max_lr=3e-3 is aggressive for a baseline — suitable for a short 10-epoch run.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,          # Fraction of cycle spent increasing LR
        div_factor=25,          # Initial LR = max_lr / div_factor
        final_div_factor=1e4,   # Min LR = initial_lr / final_div_factor
        anneal_strategy='cos'   # Cosine annealing during the decay phase
    )

    # --- Early Stopping Configuration ---
    # patience=10 is much tighter than later versions (1000) — suitable for a
    # short exploratory baseline run
    patience = 10
    counter = 0
    best_val_loss = float('inf')

    # --- TensorBoard Writer ---
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = f"runs/baseline_{current_time}"
    writer = SummaryWriter(log_dir)

    # ===================== TRAINING LOOP =====================
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # OneCycleLR steps per iteration, not per epoch

            running_loss += loss.item() * images.size(0)

            # ndim == 2 check handles potential soft labels (e.g. from CutMix)
            # even though no Mixup/CutMix is used in this baseline
            if labels.ndim == 2:
                correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            else:
                correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

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

        # --- Early Stopping & Checkpoint Saving ---
        # Save the full training state whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_val_loss,
                'history': history
            }
            torch.save(state, CHECKPOINT_SAVE_PATH)  # Save best checkpoint
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
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

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    writer.close()

    # --- Post-Training Analysis ---
    classes = train_dataset.classes
    analyze_per_class_accuracy(model, val_loader, device, classes)

    # Plot and save training curves
    plot_history(history)

    # Save final model weights
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
    print("Model successfully saved!")


if __name__ == "__main__":
    main()