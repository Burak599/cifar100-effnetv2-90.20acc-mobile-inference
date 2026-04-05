import torch  
import torchvision  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader  
import onnxruntime as ort  
import numpy as np  
from tqdm import tqdm  
import os  
import random  

# --- PATH CONFIGURATION ---  
# Get the current directory of the script, move one level up, and enter the Weights folder  
current_dir = os.path.dirname(os.path.abspath(__file__))  
DEFAULT_ONNX_PATH = os.path.abspath(os.path.join(current_dir, "..", "Weights", "model_v1_mobile.onnx"))  
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

def run_onnx_pure_test(onnx_path=DEFAULT_ONNX_PATH):  
    # 1. ONNX Runtime Session Setup  
    providers = [  
        ('CUDAExecutionProvider', {'device_id': 0}),  
        'CPUExecutionProvider'  
    ]  
      
    try:  
        session = ort.InferenceSession(onnx_path, providers=providers)  
        print(f"--> ONNX Model Loaded Successfully: {onnx_path} (GPU ACTIVE)")  
    except Exception as e:  
        print(f"--> CUDA/GPU Error, falling back to CPUExecutionProvider: {e}")  
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])  

    input_name = session.get_inputs()[0].name  

    # 2. Data Preparation  
    val_transform = transforms.Compose([  
        transforms.Resize(224),  
        transforms.ToTensor(),  
        # Standard normalization for CIFAR-100 dataset  
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  
    ])  
      
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)  
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)  

    correct = 0  
    total = 0  

    print(f"\n{'='*50}")  
    print(f"ONNX INFERENCE TEST: Evaluating {len(test_set)} images...")  
    print(f"{'='*50}\n")  

    # 3. Inference Loop  
    with torch.no_grad():  
        for inputs, targets in tqdm(test_loader, desc="Testing ONNX Model"):  
            # Convert PyTorch Tensor to NumPy Array for ONNX Runtime  
            inputs_np = inputs.numpy()  
              
            # Forward pass through the ONNX model  
            onnx_inputs = {input_name: inputs_np}  
            onnx_outputs = session.run(None, onnx_inputs)[0]  
              
            # Extract predictions  
            preds = np.argmax(onnx_outputs, axis=1)  
              
            total += targets.size(0)  
            correct += (preds == targets.numpy()).sum()  

    # 4. Final Metrics  
    accuracy = 100. * correct / total  
    print(f"\n{'-'*30}")  
    print(f"ONNX MODEL PATH : {onnx_path}")  
    print(f"TOTAL SAMPLES   : {total}")  
    print(f"CORRECT MATCHES : {correct}")  
    print(f"MODEL ACCURACY  : {accuracy:.2f}%")  
    print(f"{'-'*30}\n")  

if __name__ == '__main__':  
    run_onnx_pure_test()