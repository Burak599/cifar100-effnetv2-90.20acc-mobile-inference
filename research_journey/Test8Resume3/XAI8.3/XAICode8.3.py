"""
analysis.py
-----------
Post-training evaluation script for a CIFAR-100 EfficientNetV2-S model.

Produces the following outputs (all saved inside the XAI/ folder):
  1. Confusion Matrix           - heatmap showing which classes are confused
  2. Classification Report      - per-class Precision, Recall, F1-Score (saved as .txt and printed)
  3. Most Confused Pairs        - top-N class pairs with the highest misclassification count
  4. Grad-CAM visualizations    - gradient-weighted class activation maps for sample images
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import cv2

# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------

# Directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model checkpoint: go one level up, then into Weights/
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "..", "Weights", "best_model_11.pth")

# Output directory for all analysis artefacts
XAI_DIR = os.path.join(SCRIPT_DIR, "XAI")
os.makedirs(XAI_DIR, exist_ok=True)

# Number of most-confused pairs to display / save
TOP_N_CONFUSED_PAIRS = 20

# Number of Grad-CAM sample images to generate per class (set to 1 for a quick overview)
GRADCAM_SAMPLES = 5

# Batch size for inference (no gradient storage needed, so can be larger)
BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# CIFAR-100 STATISTICS  (same values used during training)
# ---------------------------------------------------------------------------
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ===========================================================================
# GRAD-CAM IMPLEMENTATION
# ===========================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Registers forward and backward hooks on a target convolutional layer to
    capture feature maps and their gradients, then combines them into a
    coarse spatial heatmap highlighting regions that most influenced the
    model's prediction.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model:        The PyTorch model to explain.
            target_layer: The convolutional layer whose activations are used
                          (typically the last Conv layer before the classifier).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None   # dL/dA stored by the backward hook
        self.activations = None  # A stored by the forward hook

        # Register hooks
        self._forward_hook  = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: stores the layer's output feature maps."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: stores the gradient of the loss w.r.t. feature maps."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Runs a forward + backward pass and returns a Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, C, H, W).
            class_idx:    Class index to explain. If None, uses the predicted class.

        Returns:
            cam (np.ndarray): Normalised heatmap in [0, 1], shape (H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for the chosen class score
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dimensions -> (C,)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze().cpu().numpy()                 # ReLU + squeeze

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        """Must be called when done to avoid memory leaks from lingering hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ===========================================================================
# MODEL LOADING
# ===========================================================================

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Rebuilds the EfficientNetV2-S architecture and loads weights from a checkpoint.

    The checkpoint can be either:
      - a full training state dict: {'model_state_dict': ..., 'epoch': ..., ...}
      - a bare model state dict:   {layer_name: tensor, ...}

    Args:
        checkpoint_path: Absolute path to the .pth checkpoint file.
        device:          Device to load the model onto.

    Returns:
        model (nn.Module): Loaded model in eval() mode.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Rebuild the same architecture that was used during training
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 100)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Assume the file IS the state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded and set to eval mode.")
    return model


# ===========================================================================
# INFERENCE  (collect all predictions and ground-truth labels)
# ===========================================================================

def get_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Runs full inference over the data loader and returns predictions and labels.

    Args:
        model:  Model in eval mode.
        loader: DataLoader (typically the validation / test split).
        device: Inference device.

    Returns:
        all_preds  (np.ndarray): Predicted class indices, shape (N,).
        all_labels (np.ndarray): Ground-truth class indices, shape (N,).
        all_images (list):       Raw PIL images in dataset order (for Grad-CAM).
    """
    all_preds  = []
    all_labels = []

    print("Running inference on the validation set...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(f"Inference complete. Total samples: {len(all_preds)}")
    return all_preds, all_labels


# ===========================================================================
# 1. CONFUSION MATRIX
# ===========================================================================

def plot_confusion_matrix(all_labels, all_preds, classes, save_dir):
    """
    Computes and plots the 100x100 confusion matrix as a heatmap.

    Because 100 class labels overlap badly on screen, tick labels are hidden
    by default and only the colour gradient conveys the confusion pattern.
    A zoomed version focusing on the top-20 most confused classes is also saved.

    Args:
        all_labels: Ground-truth class indices (N,).
        all_preds:  Predicted class indices (N,).
        classes:    List of 100 class name strings.
        save_dir:   Directory where the PNG files are written.
    """
    print("\n[1/4] Generating Confusion Matrix...")

    cm = confusion_matrix(all_labels, all_preds)

    # --- Full 100x100 matrix ---
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        ax=ax,
        cmap="Blues",
        xticklabels=False,   # Too dense to read at 100 classes
        yticklabels=False,
        linewidths=0,
        cbar_kws={"shrink": 0.7}
    )
    ax.set_title("Confusion Matrix (100 Classes)", fontsize=16, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    full_path = os.path.join(save_dir, "confusion_matrix_full.png")
    plt.savefig(full_path, dpi=150)
    plt.close()
    print(f"  Saved: {full_path}")

    # --- Zoomed matrix: top-20 most confused classes ---
    # Find which classes have the most off-diagonal errors
    np.fill_diagonal(cm.copy(), 0)   # zero out diagonal for ranking
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    confused_class_ids = np.argsort(cm_no_diag.sum(axis=1))[-20:][::-1]
    sub_cm    = cm[np.ix_(confused_class_ids, confused_class_ids)]
    sub_names = [classes[i] for i in confused_class_ids]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        sub_cm,
        ax=ax,
        cmap="Reds",
        annot=True,
        fmt="d",
        xticklabels=sub_names,
        yticklabels=sub_names,
        linewidths=0.4,
        cbar_kws={"shrink": 0.7}
    )
    ax.set_title("Confusion Matrix – Top 20 Most Confused Classes", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    zoom_path = os.path.join(save_dir, "confusion_matrix_top20.png")
    plt.savefig(zoom_path, dpi=150)
    plt.close()
    print(f"  Saved: {zoom_path}")

    return cm


# ===========================================================================
# 2. CLASSIFICATION REPORT
# ===========================================================================

def save_classification_report(all_labels, all_preds, classes, save_dir):
    """
    Computes per-class Precision, Recall, and F1-Score using sklearn and saves
    the report both as a plain-text file and prints it to stdout.

    Args:
        all_labels: Ground-truth class indices (N,).
        all_preds:  Predicted class indices (N,).
        classes:    List of 100 class name strings.
        save_dir:   Directory where the .txt file is written.
    """
    print("\n[2/4] Generating Classification Report...")

    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)

    # Print to console
    print(report)

    # Save to file
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")


# ===========================================================================
# 3. MOST CONFUSED PAIRS
# ===========================================================================

def plot_most_confused_pairs(cm, classes, save_dir, top_n=TOP_N_CONFUSED_PAIRS):
    """
    Extracts the top-N off-diagonal entries of the confusion matrix and
    displays them as a horizontal bar chart (e.g. "cat → dog: 47 errors").

    Args:
        cm:       Full 100x100 confusion matrix (numpy array).
        classes:  List of 100 class name strings.
        save_dir: Directory where the PNG and TXT files are written.
        top_n:    How many confused pairs to show.
    """
    print(f"\n[3/4] Finding Top-{top_n} Most Confused Pairs...")

    # Collect all off-diagonal entries
    pairs = []
    n = cm.shape[0]
    for true_idx in range(n):
        for pred_idx in range(n):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                pairs.append({
                    "true":  classes[true_idx],
                    "pred":  classes[pred_idx],
                    "count": int(cm[true_idx, pred_idx])
                })

    # Sort by error count descending
    pairs.sort(key=lambda x: x["count"], reverse=True)
    top_pairs = pairs[:top_n]

    # Print to console
    print(f"  {'True Class':<20} {'Predicted As':<20} {'Errors':>6}")
    print(f"  {'-'*20} {'-'*20} {'-'*6}")
    for p in top_pairs:
        print(f"  {p['true']:<20} {p['pred']:<20} {p['count']:>6}")

    # Save to text file
    txt_path = os.path.join(save_dir, "most_confused_pairs.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'True Class':<20} {'Predicted As':<20} {'Errors':>6}\n")
        f.write(f"{'-'*20} {'-'*20} {'-'*6}\n")
        for p in top_pairs:
            f.write(f"{p['true']:<20} {p['pred']:<20} {p['count']:>6}\n")
    print(f"  Saved: {txt_path}")

    # Bar chart
    labels = [f"{p['true']} → {p['pred']}" for p in top_pairs]
    counts = [p["count"] for p in top_pairs]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
    bars = ax.barh(labels[::-1], counts[::-1], color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Number of Misclassifications", fontsize=11)
    ax.set_title(f"Top-{top_n} Most Confused Class Pairs", fontsize=13)
    ax.set_xlim(0, max(counts) * 1.15)
    plt.tight_layout()
    bar_path = os.path.join(save_dir, "most_confused_pairs.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"  Saved: {bar_path}")


# ===========================================================================
# 4. GRAD-CAM
# ===========================================================================

def run_gradcam(model, val_dataset, classes, all_preds, all_labels, device, save_dir,
                n_samples=GRADCAM_SAMPLES):
    """
    Generates Grad-CAM heatmaps for a selection of correctly and incorrectly
    classified validation images and saves side-by-side visualisations.

    Target layer: the last convolutional block of EfficientNetV2-S's feature
    extractor (model.features[-1]), which captures the highest-level spatial
    features before global pooling.

    Args:
        model:       Model in eval mode.
        val_dataset: The raw validation dataset (with val transforms applied).
        classes:     List of 100 class name strings.
        all_preds:   Predicted class indices for every val sample.
        all_labels:  Ground-truth class indices for every val sample.
        device:      Inference device.
        save_dir:    Directory where the PNG files are written.
        n_samples:   Number of sample images to explain.
    """
    print(f"\n[4/4] Generating Grad-CAM for {n_samples} sample images...")

    # --- Select target layer ---
    # model.features[-1] is the last MBConv block before adaptive average pooling
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    # Inverse-normalisation transform to recover the original RGB image for display
    inv_mean = [-m / s for m, s in zip(CIFAR100_MEAN, CIFAR100_STD)]
    inv_std  = [1.0 / s for s in CIFAR100_STD]
    inv_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)

    # Separate correct / incorrect predictions for a balanced visualisation
    correct_idxs   = np.where(all_preds == all_labels)[0]
    incorrect_idxs = np.where(all_preds != all_labels)[0]

    # Sample indices: first half correct, second half incorrect
    half = n_samples // 2
    sample_idxs = (
        list(np.random.choice(correct_idxs,   min(half, len(correct_idxs)),   replace=False)) +
        list(np.random.choice(incorrect_idxs, min(n_samples - half, len(incorrect_idxs)), replace=False))
    )

    for i, idx in enumerate(sample_idxs):
        # Fetch the preprocessed tensor and its label
        img_tensor, label = val_dataset[idx]          # img_tensor: (C, H, W) normalised
        input_tensor = img_tensor.unsqueeze(0).to(device)   # (1, C, H, W)
        input_tensor.requires_grad_(True)

        pred_idx = int(all_preds[idx])
        true_idx = int(all_labels[idx])

        # Generate the heatmap for the predicted class
        cam = gradcam.generate(input_tensor, class_idx=pred_idx)

        # Reconstruct the displayable RGB image
        rgb = inv_normalize(img_tensor).permute(1, 2, 0).numpy()   # (H, W, 3)
        rgb = np.clip(rgb, 0, 1)

        # Resize the Grad-CAM heatmap to match the input image spatial size
        cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay: weighted sum of original image and heatmap
        overlay = 0.5 * rgb + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)

        # --- Plot: original | heatmap | overlay ---
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(rgb)
        axes[0].set_title(f"Original\nTrue: {classes[true_idx]}", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(cam_resized, cmap="jet")
        axes[1].set_title(f"Grad-CAM\nPred: {classes[pred_idx]}", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        status = "CORRECT" if pred_idx == true_idx else "WRONG"
        axes[2].set_title(f"Overlay [{status}]", fontsize=10,
                          color="green" if status == "CORRECT" else "red")
        axes[2].axis("off")

        plt.suptitle(
            f"Sample {i+1}/{len(sample_idxs)} | "
            f"True: {classes[true_idx]} | Pred: {classes[pred_idx]}",
            fontsize=11
        )
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"gradcam_sample_{i+1:02d}_{status.lower()}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")

    gradcam.remove_hooks()
    print("  Grad-CAM hooks removed.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("  CIFAR-100 Post-Training Analysis")
    print("=" * 60)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Validation transform (must match what was used during training) ---
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # --- Dataset & DataLoader ---
    # Using the standard CIFAR-100 test split as the validation set
    val_dataset = datasets.CIFAR100(
        root=os.path.join(SCRIPT_DIR, "..", "..", "..", "data"), # İki kez üste çık ve data klasörüne gir
        train=False,
        download=True,
        transform=transform_val
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    classes = val_dataset.classes   # 100 class name strings

    # --- Load model ---
    model = load_model(CHECKPOINT_PATH, device)

    # --- Run inference once; reuse results for all analysis steps ---
    all_preds, all_labels = get_predictions(model, val_loader, device)

    overall_acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nOverall Validation Accuracy: {overall_acc:.2f}%")

    # --- 1. Confusion Matrix ---
    cm = plot_confusion_matrix(all_labels, all_preds, classes, XAI_DIR)

    # --- 2. Classification Report ---
    save_classification_report(all_labels, all_preds, classes, XAI_DIR)

    # --- 3. Most Confused Pairs ---
    plot_most_confused_pairs(cm, classes, XAI_DIR, top_n=TOP_N_CONFUSED_PAIRS)

    # --- 4. Grad-CAM ---
    # Seed for reproducible sample selection
    np.random.seed(42)
    run_gradcam(model, val_dataset, classes, all_preds, all_labels,
                device, XAI_DIR, n_samples=GRADCAM_SAMPLES)

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {XAI_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()