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

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DIR = os.path.join(BASE_PATH, "..", "user_weights")
PLOT_DIR = os.path.join(BASE_PATH, "..", "user_plots")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

CHECKPOINT_SAVE_PATH    = os.path.join(WEIGHTS_DIR, "user_best_model_3.pth")
FINAL_MODEL_SAVE_PATH   = os.path.join(WEIGHTS_DIR, "user_last_epochs_3.pth")
PLOT_SAVE_PATH          = os.path.join(PLOT_DIR,    "user_training_history_plot_3.png")


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
    torch.cuda.manual_seed_all(seed)  # Required when using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_optimal_batch_size(model, input_shape=(3, 224, 224), device="cuda", threshold=0.8):
    """
    Finds the largest power-of-2 batch size that fits within 80% of GPU memory.
    Runs forward and backward passes with increasing batch sizes until memory
    usage exceeds the threshold or an OOM error is raised.

    Args:
        model:        The PyTorch model to test.
        input_shape:  Shape of a single input tensor (C, H, W).
        device:       Target device, typically 'cuda'.
        threshold:    Fraction of total GPU memory to use as the upper limit.

    Returns:
        best_batch (int): The largest batch size that fits within the memory budget.
    """
    model.to(device)
    model.train()  # Train mode stores activations, giving a more realistic memory estimate

    # Retrieve total GPU memory and compute the target memory ceiling
    gpu_capacity = torch.cuda.get_device_properties(device).total_memory
    target_memory = gpu_capacity * threshold

    batch_size = 2
    best_batch = 2

    print(f"Hedef Bellek Kullanımı: {target_memory / 1e9:.2f} GB")

    try:
        while True:
            # Clear cached memory and reset peak stats before each test
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create random dummy inputs and integer class labels
            inputs = torch.randn(batch_size, *input_shape).to(device)
            targets = torch.randint(0, 100, (batch_size,)).to(device)

            # Perform a full forward and backward pass to simulate real training memory usage
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()

            # Measure peak memory allocated during this pass
            used_memory = torch.cuda.max_memory_allocated(device)

            if used_memory < target_memory:
                print(f"Batch {batch_size} denendi: {used_memory / 1e9:.2f} GB kullanıldı (Uygun)")
                best_batch = batch_size
                batch_size *= 2  # Double batch size for the next iteration
            else:
                print(f"Batch {batch_size} sınırı aştı! ({used_memory / 1e9:.2f} GB)")
                break

            # Safety cap to prevent testing unreasonably large batch sizes
            if batch_size > 2048:
                break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM: {batch_size} sığmadı.")
        else:
            print(f"Hata oluştu: {e}")

    # Release cached memory and clear gradients before returning
    torch.cuda.empty_cache()
    model.zero_grad()

    print(f"--- Önerilen En Yüksek Batch Size: {best_batch} ---")
    return best_batch


def plot_history(history):
    """
    Plots training and validation accuracy and loss curves side by side,
    then saves the figure to the path defined in PLOT_SAVE_PATH.

    Args:
        history (dict): Dictionary containing lists for 'train_acc', 'val_acc',
                        'train_loss', and 'val_loss', one value per epoch.
    """
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(12, 5))

    # --- Accuracy subplot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_acc"], 'bo-', label='Eğitim Başarımı')
    plt.plot(epochs, history["val_acc"], 'ro-', label='Doğrulama Başarımı')
    plt.title('Eğitim ve Doğrulama Doğruluğu (Acc)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # --- Loss subplot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], 'b--', label='Eğitim Kaybı')
    plt.plot(epochs, history["val_loss"], 'r--', label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)  # Save the figure to the configured output path
    plt.show()


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
    print("\n--- En Zorlanılan 10 Sınıf ---")
    for name, acc in accs[:10]:
        print(f"{name}: %{acc:.2f}")


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
    # Training transform: resize, RandAugment, horizontal flip, normalize, and random erasing
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandAugment(num_ops=1, magnitude=6),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
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

    # num_workers=8 enables multi-process data loading; pin_memory speeds up CPU→GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # --- Model Setup ---
    # Load EfficientNetV2-S pretrained on ImageNet as the backbone
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # Replace the final classification head to output 100 classes (CIFAR-100)
    # In EfficientNetV2-S, the last layer is index 1 of the classifier sequential
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 100)

    model = model.to(device)

    # Batch size can be found dynamically via find_optimal_batch_size(), but is fixed here
    #batch_size = find_optimal_batch_size(model)
    batch_size = 64
    print(f"Batch Size: {batch_size}")

    # Mixup / CutMix augmentation (disabled; uncomment to enable)
    #mixup_fn = Mixup(
    #    mixup_alpha=0.0,   # MixUp
    #    cutmix_alpha=0.0,  # CutMix
    #    num_classes=100
    #)

    # --- Loss Function & Optimizer ---
    # Label smoothing (0.1) reduces overconfidence and improves generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW optimizer with L2 weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)

    num_epochs = 40

    # --- LR Finder (disabled; uncomment to search for an optimal learning rate) ---
    """
    print("\n--- 🔍 En İyi Learning Rate Aranıyor ---")
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")
    
    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']
    best_lr = lrs[np.argmin(np.gradient(losses))]
    print(f"--- ✅ Önerilen Max LR: {best_lr:.2e} ---")
    
    result = lr_finder.plot() 
    
    # If result is a tuple, take the first element (the axis); otherwise use it directly
    ax = result[0] if isinstance(result, (list, tuple)) else result
    
    # Retrieve the figure and save it
    fig = ax.get_figure()
    fig.savefig("lr_finder_result.png")
    print("--- 📸 LR Finder grafiği 'lr_finder_result.png' olarak kaydedildi! ---")

    lr_finder.reset()
    """

    # --- Learning Rate Scheduler ---
    # OneCycleLR: warms up to max_lr then cosine-anneals to near zero
    # pct_start=0.3 means 30% of training is the warmup phase
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1.5e-3,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    # --- Early Stopping Configuration ---
    # Patience is set very high (1000) to effectively disable early stopping
    patience = 1000
    counter = 0
    best_val_loss = float('inf')

    # --- TensorBoard Logging Setup ---
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(BASE_PATH, "..", f"user_runs3/baseline_{current_time}")
    writer = SummaryWriter(log_dir)

    # ==================== Training Loop ====================
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup / CutMix application (disabled; uncomment if mixup_fn is enabled above)
            #if mixup_fn is not None:
                #images, labels = mixup_fn(images, labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # OneCycleLR is stepped per batch, not per epoch

            running_loss += loss.item() * images.size(0)

            # Handle both hard labels (standard) and soft labels (e.g., from CutMix)
            if labels.ndim == 2:  # CutMix soft labels: compare argmax of both
                correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            else:
                correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        # Compute average training loss and accuracy for this epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total

        # ==================== Validation Loop ====================
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
        # Save a full checkpoint whenever a new best validation loss is achieved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            # Bundle all relevant state for resumable checkpointing
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_val_loss,
                'history': history  # Full metric history for resuming plots
            }
            torch.save(state, CHECKPOINT_SAVE_PATH)  # Save checkpoint to configured path
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

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    writer.close()

    # Analyze which classes the model struggles with most
    classes = train_dataset.classes
    analyze_per_class_accuracy(model, val_loader, device, classes)

    # Plot and save the training history curves
    plot_history(history)

    # Save only the final model weights (lighter than a full checkpoint)
    torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
    print("Model kaydedildi!")


if __name__ == "__main__":
    main()
