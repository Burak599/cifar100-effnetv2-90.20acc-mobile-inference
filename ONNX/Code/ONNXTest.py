import torch
import torchvision.models as models
import torch.nn as nn
import os

def convert_to_onnx(checkpoint_path, onnx_model_path="user_model_v1_mobile.onnx"):
    """
    Converts a trained PyTorch EfficientNetV2-S model to ONNX format for mobile deployment.
    """
    # 1. Initialize Architecture (EfficientNetV2-S with 100 classes)
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 100)
    
    # 2. Load Weights
    print(f"🔄 Loading: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check if weights are inside 'model_state_dict' or stored directly
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval() # Disable Dropout and BatchNorm for inference

    # 3. Dummy Input (Define input shape: 1 image, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. Export Process
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=15,          # Recommended version for mobile compatibility
        do_constant_folding=True,  # Merge constant nodes to optimize performance
        input_names=['input'],     # Input layer name (required for mobile APIs)
        output_names=['output'],   # Output layer name
        # Enable dynamic batch size for flexibility
        dynamic_axes=None 
    )
    
    print(f"✅ Success! Universal model is ready: {onnx_model_path}")

if __name__ == "__main__":
    # --- PATH CONFIGURATION ---
    # Move up 2 levels (cd .. / cd ..), then navigate to research_journey/Test8Resume2/Weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.abspath(
        os.path.join(
            current_dir, "..", "..", 
            "research_journey", "Test8Resume2", "Weights", 
            "best_model_10.test.pth"
        )
    )
    
    ONNX_OUTPUT_PATH = "user_model_v1_mobile.onnx"

    convert_to_onnx(CHECKPOINT_PATH, ONNX_OUTPUT_PATH)
