# EfficientNetV2-S CIFAR-100 — Advanced Optimization & Quantization

A high-accuracy, production-ready deep learning pipeline for CIFAR-100 image classification.
Achieving **90.20% validation accuracy** with EfficientNetV2-S, SAM optimization, SWA, and FP16 ONNX mobile deployment.

---

## 📌 Overview

This project is a comprehensive deep learning pipeline designed to achieve **high accuracy** and **low-latency inference** on the CIFAR-100 dataset (100 classes, 60,000 images) using the **EfficientNetV2-S** architecture.

The model was iteratively refined across **14 training stages** — from an 81.13% baseline to a final **90.20% validation accuracy** — using modern augmentation strategies, SAM optimization, and Stochastic Weight Averaging. The final model is exported to **FP16 ONNX (38 MB)** and runs **live in the browser** via a standalone HTML interface with WebGPU acceleration.

---

## 🚀 Key Features

| Feature | Details |
|:---|:---|
| **Architecture** | EfficientNetV2-S — superior parameter efficiency and training speed |
| **Data Augmentation** | MixUp + CutMix hybrid, RandomResizedCrop, Soft RandAugment, RandomErasing |
| **Optimizer** | Sharpness-Aware Minimization (SAM), ρ=0.05 |
| **Loss Function** | CrossEntropy + Label Smoothing (Stages 1–8.3) / Focal Loss (Stage 8.4) |
| **LR Scheduling** | OneCycleLR with dynamic warm-up |
| **Weight Averaging** | Stochastic Weight Averaging (SWA) across best checkpoints |
| **Explainability** | GradCAM + Confusion Matrix + Most Confused Pairs |
| **Export** | PyTorch (.pth) → ONNX FP32 → ONNX FP16 (38 MB) |
| **Deployment** | Live browser inference via WebGPU / WebGL / WASM fallback |

---

## 📁 Project Structure

```
CIFAR100Project/
│
├── data/
├── LRFinder/
├── ONNX/
│   └── Code/
│       ├── To_ONNX.py
│       └── ONNXTest.py
├── Quantization_ONNX/
│   └── Code/
│       ├── Quantization.py
│       └── QuantizationTest.py
├── LastModelTest(FP16)/
│   └── Code/
│       └── best_model_fp16Test.py
├── research_journey/
│   ├── test1/ ... test8.4/
│   │   └── Code/
│   └── TestSWA/
│       └── Code/
└── UseMobile/
    └── index.html
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA 12.x (optional — CPU inference also supported)
- 8 GB+ VRAM recommended for training

```
torch>=2.0.0
torchvision>=0.15.0
onnx>=1.14.0
onnxruntime-gpu>=1.16.0
timm>=0.9.0
numpy>=1.24.0
Pillow>=9.5.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

---

## 🛠️ Quick Start

**1. Clone the repository**

```bash
git clone https://github.com/Burak599/cifar100-effnetv2-90.20acc-mobile-inference.git
cd cifar100-effnetv2-90.20acc-mobile-inference
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download pre-trained weights**

All weights are hosted on HuggingFace: https://huggingface.co/brk9999/efficientnetv2-s-cifar100-fp16

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='brk9999/efficientnetv2-s-cifar100-fp16',
    local_dir='pretrained_weights/'
)
"
```

Pretrained weights stay in pretrained_weights/; training outputs always go to the user_weights/ folder in the active script's directory. These two paths remain strictly separate.

---

## 📊 Results

### Validation Accuracy Evolution

| ID | Val Acc | Key Changes |
|:---:|:---:|:---|
| 1 | 81.13% | Initial baseline |
| 2 | 81.87% | 40 epochs + seed fix + CenterCrop |
| 3 | 83.13% | Lower LR (1.5e-3) + Soft RandAug |
| 4 | 85.34% | MixUp (α=0.8) + LR (7e-4) + RandomErasing |
| 5 | 86.45% | MixUp + CutMix hybrid |
| 6 | 87.35% | SAM optimizer (ρ=0.05) + 120 epochs |
| 7 | 86.53% | Resume + Dropout (0.3) |
| 8 | 88.17% | 200 epochs + RandomResizedCrop + SAM |
| 8.1 | 89.18% | Base progressive refinement |
| **8.2** | **89.86%** | **Relaxed MixUp/CutMix + lower LR ⭐ Best single model** |
| 8.3 | 89.69% | No MixUp + reduced augmentation |
| 8.4 | 89.78% | Focal Loss + weight decay |
| SWA | 89.81% | Weight averaging (8.2 + 8.4) |
| **FULL** | **90.20%** | **🏆 Final best weights** |

### GradCAM — Sample Prediction

<img src="YOUR_GRADCAM_URL" width="900"/>

### Top-20 Most Confused Class Pairs

<img src="YOUR_CONFUSED_PAIRS_URL" width="700"/>

### Confusion Matrix

<img src="YOUR_CONFUSION_MATRIX_URL" width="700"/>

---

## 🏋️ Training

>Pretrained weights stay in pretrained_weights/; training outputs always go to the user_weights/ folder located inside each script's own directory. These two never interfere.

Each stage has its own self-contained script, no arguments needed:

```bash
# Stage 8.2 — Best single model (89.86%) ⭐
python "research_journey/Test8/Code/cifar100_test8.py"
python "research_journey/Test8.1/Code/test8resume.py"
python "research_journey/Test8.2/Code/test8resume2.py"

---

## 📤 Export Pipeline

```bash
# Step 1 — PyTorch → ONNX FP32
python "ONNX/Code/To_ONNX.py"

# Step 2 — Test ONNX FP32
python "ONNX/Code/ONNXTest.py"

# Step 3 — ONNX FP32 → FP16 (38 MB)
python "Quantization_ONNX/Code/Quantization.py"

# Step 4 — Test ONNX FP16
python "Quantization_ONNX/Code/QuantizationTest.py"
```

---

## 🔍 Inference

```bash
python "LastModelTest(FP16)/Code/best_model_fp16Test.py" (90.20%) ⭐
```

Loads the FP16 ONNX model, runs full evaluation on CIFAR-100 test set, and generates all XAI visualizations. 33333

---

## 📱 Mobile Inference

The model runs **live in the browser** — no server, no Python, no installation required.

```bash
cd UseMobile
python -m http.server 8080
# Open: http://localhost:8080
```

Or open `UseMobile/index.html` directly in Chrome / Edge / Firefox.

- Captures live camera feed and classifies every frame in real time
- Runs via **WebGPU** → falls back to WebGL → falls back to WASM (CPU)
- Displays predicted class name and live FPS counter
- Works completely offline after first load

---

## 📦 Model Weights

All weights on HuggingFace: https://huggingface.co/brk9999/efficientnetv2-s-cifar100-fp16

| Checkpoint | Val Acc | Format | Size |
|:---|:---:|:---:|:---:|
| Stage 8.2 ⭐ Best | 89.86% | `.pth` | ~85 MB |
| Stage 8.4 | 89.78% | `.pth` | ~85 MB |
| SWA Merged | 89.81% | `.pth` | ~85 MB |
| Final ONNX FP32 | 90.20% | `.onnx` | ~85 MB |
| **Final ONNX FP16** | **90.20%** | **`.onnx`** | **38 MB** |

---

## 🔬 Explainability (XAI)

Running `best_model_fp16Test.py` automatically generates:

- **GradCAM** (5 samples) — shows which image regions the model focuses on, with correct/incorrect label overlay
- **Confusion Matrix** — Top-20 most confused classes with raw count heatmap
- **Most Confused Pairs** — Top-20 class pairs by misclassification count (`oak_tree → maple_tree`: 25, `boy → man`: 14)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 🙏 Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) — EfficientNetV2-S backbone
- [SAM Optimizer](https://github.com/davda54/sam) — Sharpness-Aware Minimization
- [ONNX Runtime Web](https://onnxruntime.ai/) — browser inference engine
- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298) — Mingxing Tan & Quoc V. Le
- CIFAR-100 — Alex Krizhevsky

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [Burak599](https://github.com/Burak599) — ⭐ If this project helped you, please give it a star!
