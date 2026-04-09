import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
import os
import random

# --- PATH CONFIGURATION (Fixed Absolute Paths) ---
# Your fixed base path for FP16 weights
BASE_FP16_PATH = os.path.dirname(os.path.abspath(__file__))

# Fixed path to the FP16 model file
FP16_ONNX_PATH = os.path.join(BASE_FP16_PATH, "..", "Weight_fp16", "user_model_fp16.onnx")

# Data directory path
DATA_ROOT = "./data"
# --------------------------

# ==========================================
# 0. REPRODUCIBILITY (SEED)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def run_onnx_fp16_test(onnx_path=FP16_ONNX_PATH):
    # 1. ONNX Runtime Session Setup (Senin çalışan mantığın)
    providers = [
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]
    
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"--> FP16 Model Loaded Successfully: {onnx_path}")
    except Exception as e:
        print(f"--> HATA: {e}")
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name

    # 2. Data Preparation
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    correct = 0
    total = 0

    print(f"\n{'='*50}")
    print(f"FP16 INFERENCE TEST: Evaluating {len(test_set)} images...")
    print(f"{'='*50}\n")

    # 3. Inference Loop
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing FP16 Model"):
            inputs_np = inputs.numpy()
            
            # Forward pass
            onnx_inputs = {input_name: inputs_np}
            onnx_outputs = session.run(None, onnx_inputs)[0]
            
            # Extract predictions
            preds = np.argmax(onnx_outputs, axis=1)
            
            total += targets.size(0)
            correct += (preds == targets.numpy()).sum()

    # 4. Final Metrics
    accuracy = 100. * correct / total
    print(f"\n{'-'*30}")
    print(f"ONNX FP16 PATH  : {onnx_path}")
    print(f"TOTAL SAMPLES   : {total}")
    print(f"CORRECT MATCHES : {correct}")
    print(f"MODEL ACCURACY  : {accuracy:.2f}%")
    print(f"{'-'*30}\n")

if __name__ == '__main__':
    run_onnx_fp16_test()
