import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import os

def fast_gpu_onnx_test(onnx_path, batch_size=128): 
    """
    Performs a high-speed inference test using ONNX Runtime with CUDA acceleration.
    """
    print(f"⚡ GPU-Accelerated Test Starting: {onnx_path}")
    
    # Standard transformation pipeline for CIFAR-100 evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=False),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # num_workers set to 8 to leverage multi-core CPU performance during data loading
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # --- CUDA (GPU) Provider Configuration ---
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB limit
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]
    
    try:
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        print("✅ GPU Execution Provider Active!")
    except Exception as e:
        print(f"⚠️ GPU connection failed, falling back to CPU... Error: {e}")
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    input_name = ort_session.get_inputs()[0].name
    correct = 0
    total = 0

    # Inference Loop
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Processing"):
            # Convert PyTorch tensors to NumPy for ONNX Runtime compatibility
            inputs_np = images.numpy()
            
            # Run inference on batches for maximum throughput
            onnx_inputs = {input_name: inputs_np}
            onnx_outputs = ort_session.run(None, onnx_inputs)[0]
            
            # Calculate accuracy
            predictions = np.argmax(onnx_outputs, axis=1)
            total += labels.size(0)
            correct += (predictions == labels.numpy()).sum()

    accuracy = 100 * correct / total
    print(f"\n📊 --- TEST RESULTS ---")
    print(f"🔹 Final Accuracy: {accuracy:.4f}%")
    print(f"🔹 Status: Inference Completed Successfully.")

if __name__ == "__main__":
    # --- PATH CONFIGURATION ---
    # Move up 1 level (cd ..), then navigate to the ONNXWeight folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    TARGET_ONNX_PATH = os.path.abspath(
        os.path.join(current_dir, "..", "ONNXWeight", "model_v1_mobile.onnx")
    )
    
    # Run the test
    fast_gpu_onnx_test(TARGET_ONNX_PATH)