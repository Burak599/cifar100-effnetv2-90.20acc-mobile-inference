"""
analysis.py
-----------
Combined FP16 ONNX inference test + full XAI analysis for CIFAR-100 EfficientNetV2-S.

Produces the following outputs (all saved inside the XAI/ folder one level up):
  1. Accuracy Report          - overall Top-1 accuracy printed to console
  2. Confusion Matrix         - full 100x100 heatmap + Top-20 zoomed heatmap
  3. Classification Report    - per-class Precision, Recall, F1-Score (.txt)
  4. Most Confused Pairs      - top-20 class pairs bar chart + .txt
  5. Grad-CAM visualizations  - 5 sample images with heatmap overlay (ONNX-based)

Directory layout expected:
  LastModelTest(FP16)/
  ├── Code/
  │   └── analysis.py          <- this file
  ├── weight/
  │   └── model_fp16.onnx
  └── XAI/                     <- all outputs go here (auto-created)

Grad-CAM approach (ONNX):
  Standard Grad-CAM requires backpropagation which ONNX Runtime does not support.
  Instead we use GradCAM++ approximation via ONNX intermediate layer outputs:
    1. Export the model with an additional output node at the last conv layer
       (model.features[-1]) using onnxruntime + onnx manipulation.
    2. Run two forward passes: one for the full prediction, one for the feature maps.
    3. Compute the importance weights from the feature maps using the predicted
       class score differences (finite-difference approximation of gradients).
  This is fully ONNX Runtime compatible — no PyTorch or .pth file needed.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import onnxruntime as ort
import onnx
from onnx import numpy_helper
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
from tqdm import tqdm


# ===========================================================================
# PATH CONFIGURATION
# ===========================================================================

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "pretrained_weights", "best_weight", "model_fp32.onnx"))
XAI_DIR         = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "User_XAI"))
os.makedirs(XAI_DIR, exist_ok=True)

# ===========================================================================
# SETTINGS
# ===========================================================================

BATCH_SIZE           = 1
TOP_N_CONFUSED_PAIRS = 20
GRADCAM_SAMPLES      = 5

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ===========================================================================
# REPRODUCIBILITY
# ===========================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)


# ===========================================================================
# ONNX GRAD-CAM
# ===========================================================================

def get_gradcam_session(onnx_path: str):
    """
    Loads the ONNX model and adds the last convolutional feature map
    as an additional output node so we can extract it during inference.

    EfficientNetV2-S feature extractor output before global pooling is
    typically named '/features/features.8/...' in the ONNX graph.
    We find the correct node automatically by inspecting the graph.

    Returns:
        session     : onnxruntime.InferenceSession with the extra output
        feat_name   : name of the intermediate feature map output node
        input_name  : name of the model input node
    """
    model_onnx = onnx.load(onnx_path)

    # Find the last Conv output before the global average pool
    # In EfficientNetV2-S the node just before /avgpool is the target
    # We walk the graph and find the last node that produces a 4-D feature map
    feat_name = None
    for node in model_onnx.graph.node:
        if node.op_type in ("Conv", "BatchNormalization", "SiLU", "Relu", "HardSwish", "Mul", "Add"):
            feat_name = node.output[0]

    if feat_name is None:
        raise RuntimeError("Could not find a suitable feature map node in the ONNX graph.")

    # Add the intermediate node as an extra graph output so ORT can return it
    intermediate_output = onnx.helper.make_tensor_value_info(
        feat_name, onnx.TensorProto.FLOAT, None
    )
    model_onnx.graph.output.append(intermediate_output)

    # Save modified model to a temp file
    modified_path = onnx_path.replace(".onnx", "_gradcam_tmp.onnx")
    onnx.save(model_onnx, modified_path)

    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(modified_path, providers=providers)
    except Exception:
        session = ort.InferenceSession(modified_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    return session, feat_name, input_name


def compute_gradcam_onnx(session, input_name, feat_name, input_np, class_idx):
    """
    Computes a Grad-CAM-like heatmap using ONNX intermediate feature maps.

    Since ONNX Runtime does not support backpropagation we use a
    Score-CAM-inspired approach:
      1. Get the feature maps A (shape: C x H x W) from the last conv layer.
      2. Upsample each channel map to input size.
      3. Use each upsampled map as a mask over the input and run a masked
         forward pass to get a perturbed score for the target class.
      4. Weight = max(0, score_masked - score_baseline).
      5. Final CAM = ReLU( sum_c( weight_c * A_c ) ), normalised to [0,1].

    For efficiency we use the raw channel activations weighted by their
    global average (classic CAM weights) as a fast approximation when the
    number of channels is large (>64), and the full Score-CAM otherwise.

    Args:
        session    : ONNX session with intermediate output appended.
        input_name : Name of the model's input node.
        feat_name  : Name of the intermediate feature map output node.
        input_np   : Input image as float32 numpy array, shape (1, 3, H, W).
        class_idx  : Target class index for the heatmap.

    Returns:
        cam (np.ndarray): Normalised heatmap in [0, 1], shape (H, W).
    """
    outputs     = session.run(None, {input_name: input_np})
    logits      = outputs[0]          # (1, 100)
    feature_map = outputs[-1]         # (1, C, h, w)  — last output is our appended node

    # feature_map may be fp16 — cast to float32 for numpy ops
    feature_map = feature_map.astype(np.float32)
    fm          = feature_map[0]      # (C, h, w)
    C, h, w     = fm.shape
    H, W        = input_np.shape[2], input_np.shape[3]

    if C <= 64:
        # Full Score-CAM: mask input with each upsampled channel map
        baseline_score = float(logits[0, class_idx])
        weights        = np.zeros(C, dtype=np.float32)

        for c in range(C):
            channel = fm[c]                                           # (h, w)
            # Normalise channel map to [0, 1]
            c_min, c_max = channel.min(), channel.max()
            if c_max - c_min < 1e-6:
                continue
            channel_norm = (channel - c_min) / (c_max - c_min)
            # Upsample to input resolution
            channel_up   = cv2.resize(channel_norm, (W, H))          # (H, W)
            # Apply as mask (broadcast over RGB channels)
            masked_input = input_np * channel_up[np.newaxis, np.newaxis, :, :]
            masked_out   = session.run(None, {input_name: masked_input.astype(np.float32)})
            masked_score = float(masked_out[0][0, class_idx])
            weights[c]   = max(0.0, masked_score - baseline_score)
    else:
        # Fast approximation: global-average-pool the activations (classic CAM weights)
        weights = np.maximum(0, fm.mean(axis=(1, 2)))                 # (C,)

    # Weighted sum of feature maps
    cam = np.einsum('c,chw->hw', weights, fm)                         # (h, w)
    cam = np.maximum(cam, 0)                                          # ReLU

    # Upsample to input size
    cam = cv2.resize(cam, (W, H))

    # Normalise to [0, 1]
    if cam.max() > 1e-6:
        cam = cam / cam.max()

    return cam


# ===========================================================================
# STEP 1 — ONNX INFERENCE TEST (accuracy)
# ===========================================================================

def run_onnx_fp16_test(onnx_path: str):
    """
    Runs full inference with the FP16 ONNX model on the CIFAR-100 test set.

    Returns:
        all_preds   (np.ndarray) : Predicted class indices, shape (N,).
        all_labels  (np.ndarray) : Ground-truth class indices, shape (N,).
        val_dataset              : The CIFAR-100 test dataset (reused for Grad-CAM).
    """
    print("\n" + "=" * 60)
    print("  STEP 1 — ONNX INFERENCE TEST")
    print("=" * 60)

    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"--> Model loaded: {onnx_path}")
    except Exception as e:
        print(f"--> WARNING: CUDA provider failed ({e}), falling back to CPU.")
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    data_root   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "data"))
    val_dataset = datasets.CIFAR100(root=data_root, train=False, download=True,
                                    transform=val_transform)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    all_preds  = []
    all_labels = []
    correct    = 0
    total      = 0

    print(f"\nEvaluating {len(val_dataset)} images...\n")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Inference"):
            inputs_np   = inputs.numpy()
            onnx_inputs = {input_name: inputs_np}
            onnx_out    = session.run(None, onnx_inputs)[0]

            preds = np.argmax(onnx_out, axis=1)
            all_preds.append(preds)
            all_labels.append(targets.numpy())

            total   += targets.size(0)
            correct += (preds == targets.numpy()).sum()

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy   = 100.0 * correct / total

    print(f"\n{'-' * 40}")
    print(f"  ONNX PATH  : {onnx_path}")
    print(f"  TOTAL SAMPLES   : {total}")
    print(f"  CORRECT MATCHES : {correct}")
    print(f"  MODEL ACCURACY  : {accuracy:.2f}%")
    print(f"{'-' * 40}\n")

    return all_preds, all_labels, val_dataset


# ===========================================================================
# STEP 2 — CONFUSION MATRIX
# ===========================================================================

def plot_confusion_matrix(all_labels, all_preds, classes, save_dir):
    """
    Saves two confusion matrix plots:
      - confusion_matrix_full.png  : full 100x100 heatmap
      - confusion_matrix_top20.png : zoomed Top-20 most confused classes
    """
    print("\n" + "=" * 60)
    print("  STEP 2 — CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(all_labels, all_preds)

    # Full 100x100
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, ax=ax, cmap="Blues", xticklabels=False, yticklabels=False,
                linewidths=0, cbar_kws={"shrink": 0.7})
    ax.set_title("Confusion Matrix (100 Classes)", fontsize=16, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    full_path = os.path.join(save_dir, "confusion_matrix_full.png")
    plt.savefig(full_path, dpi=150)
    plt.close()
    print(f"  Saved: {full_path}")

    # Top-20 zoomed
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    confused_ids = np.argsort(cm_no_diag.sum(axis=1))[-20:][::-1]
    sub_cm       = cm[np.ix_(confused_ids, confused_ids)]
    sub_names    = [classes[i] for i in confused_ids]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(sub_cm, ax=ax, cmap="Reds", annot=True, fmt="d",
                xticklabels=sub_names, yticklabels=sub_names,
                linewidths=0.4, cbar_kws={"shrink": 0.7})
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
# STEP 3 — CLASSIFICATION REPORT
# ===========================================================================

def save_classification_report(all_labels, all_preds, classes, save_dir):
    """Saves per-class Precision, Recall, F1-Score as classification_report.txt."""
    print("\n" + "=" * 60)
    print("  STEP 3 — CLASSIFICATION REPORT")
    print("=" * 60)

    report      = classification_report(all_labels, all_preds,
                                        target_names=classes, digits=4)
    print(report)

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")


# ===========================================================================
# STEP 4 — MOST CONFUSED PAIRS
# ===========================================================================

def plot_most_confused_pairs(cm, classes, save_dir, top_n=TOP_N_CONFUSED_PAIRS):
    """
    Saves a horizontal bar chart and .txt of the top-N most confused class pairs.
    """
    print("\n" + "=" * 60)
    print(f"  STEP 4 — TOP-{top_n} MOST CONFUSED PAIRS")
    print("=" * 60)

    pairs = []
    n     = cm.shape[0]
    for true_idx in range(n):
        for pred_idx in range(n):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                pairs.append({
                    "true":  classes[true_idx],
                    "pred":  classes[pred_idx],
                    "count": int(cm[true_idx, pred_idx])
                })

    pairs.sort(key=lambda x: x["count"], reverse=True)
    top_pairs = pairs[:top_n]

    print(f"\n  {'True Class':<20} {'Predicted As':<20} {'Errors':>6}")
    print(f"  {'-'*20} {'-'*20} {'-'*6}")
    for p in top_pairs:
        print(f"  {p['true']:<20} {p['pred']:<20} {p['count']:>6}")

    txt_path = os.path.join(save_dir, "most_confused_pairs.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'True Class':<20} {'Predicted As':<20} {'Errors':>6}\n")
        f.write(f"{'-'*20} {'-'*20} {'-'*6}\n")
        for p in top_pairs:
            f.write(f"{p['true']:<20} {p['pred']:<20} {p['count']:>6}\n")
    print(f"\n  Saved: {txt_path}")

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
# STEP 5 — GRAD-CAM (ONNX-based, no .pth required)
# ===========================================================================

def run_gradcam(all_preds, all_labels, val_dataset, classes, save_dir,
                onnx_path, n_samples=GRADCAM_SAMPLES):
    """
    Generates Grad-CAM-like heatmaps using only the ONNX model.

    No PyTorch .pth checkpoint required. Uses Score-CAM / fast-CAM
    approximation via ONNX intermediate layer feature map outputs.
    See compute_gradcam_onnx() for the full explanation.
    """
    print("\n" + "=" * 60)
    print(f"  STEP 5 — GRAD-CAM ({n_samples} samples, ONNX-based)")
    print("=" * 60)

    print("  Building ONNX session with intermediate feature map output...")
    try:
        session, feat_name, input_name = get_gradcam_session(onnx_path)
        print(f"  Feature map node: {feat_name}")
    except Exception as e:
        print(f"  ERROR building Grad-CAM session: {e}")
        print("  Skipping Grad-CAM.")
        return

    inv_mean      = [-m / s for m, s in zip(CIFAR100_MEAN, CIFAR100_STD)]
    inv_std       = [1.0 / s for s in CIFAR100_STD]
    inv_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)

    correct_idxs   = np.where(all_preds == all_labels)[0]
    incorrect_idxs = np.where(all_preds != all_labels)[0]
    half           = n_samples // 2

    sample_idxs = (
        list(np.random.choice(correct_idxs,   min(half, len(correct_idxs)),               replace=False)) +
        list(np.random.choice(incorrect_idxs, min(n_samples - half, len(incorrect_idxs)), replace=False))
    )

    for i, idx in enumerate(sample_idxs):
        img_tensor, _ = val_dataset[idx]
        input_np      = img_tensor.unsqueeze(0).numpy().astype(np.float32)   # (1,3,H,W)

        pred_idx = int(all_preds[idx])
        true_idx = int(all_labels[idx])

        try:
            cam = compute_gradcam_onnx(session, input_name, feat_name,
                                       input_np, pred_idx)
        except Exception as e:
            print(f"  WARNING: Grad-CAM failed for sample {idx}: {e}. Skipping.")
            continue

        # Recover displayable RGB
        rgb = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)
        H, W = rgb.shape[:2]

        cam_resized = cv2.resize(cam, (W, H))
        heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay     = np.clip(0.5 * rgb + 0.4 * heatmap, 0, 1)

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

    # Clean up temp modified ONNX file
    tmp_path = onnx_path.replace(".onnx", "_gradcam_tmp.onnx")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        print("  Temp ONNX file cleaned up.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("  CIFAR-100 ONNX TEST + XAI ANALYSIS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device : {device}")
    print(f"ONNX model   : {CHECKPOINT_PATH}")
    print(f"Output dir   : {XAI_DIR}\n")

    # Step 1 — inference + accuracy
    all_preds, all_labels, val_dataset = run_onnx_fp16_test(onnx_path=CHECKPOINT_PATH)

    classes = val_dataset.classes

    # Step 2 — confusion matrix
    cm = plot_confusion_matrix(all_labels, all_preds, classes, XAI_DIR)

    # Step 3 — classification report
    save_classification_report(all_labels, all_preds, classes, XAI_DIR)

    # Step 4 — most confused pairs
    plot_most_confused_pairs(cm, classes, XAI_DIR, top_n=TOP_N_CONFUSED_PAIRS)

    # Step 5 — Grad-CAM (ONNX-based)
    np.random.seed(42)
    run_gradcam(all_preds, all_labels, val_dataset, classes,
                XAI_DIR, CHECKPOINT_PATH, n_samples=GRADCAM_SAMPLES)

    print("\n" + "=" * 60)
    print(f"  ALL OUTPUTS SAVED TO: {XAI_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
