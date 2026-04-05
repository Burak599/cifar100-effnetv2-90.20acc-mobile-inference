import onnx
from onnxconverter_common import float16
import os

def optimize_onnx_to_fp16(input_path, output_path):
    """
    Directly converts an ONNX model to Float16 using the EXACT 
    function name discovered in the diagnostics: convert_float_to_float16.
    """
    print(f"🚀 Starting Pure ONNX Optimization: {os.path.basename(input_path)}")
    
    try:
        # 1. Load the Model
        model = onnx.load(input_path)
        
        # 2. Float16 Conversion
        # Using the actual function name from your library's directory
        print("Using float16.convert_float_to_float16...")
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        
        # 3. Save the Model
        onnx.save(model_fp16, output_path)
        
        # Metrics
        old_size = os.path.getsize(input_path) / (1024 * 1024)
        new_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\n✅ SUCCESS! FP16 Optimization Complete.")
        print(f"📉 Original: {old_size:.2f} MB")
        print(f"💎 New (FP16): {new_size:.2f} MB")
        print(f"📂 Saved at: {output_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # --- PATHS (cd .. logic as requested) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input Path: cd .. -> ONNX32Weights -> model_v1_mobile.onnx
    INPUT_ONNX = os.path.abspath(
        os.path.join(current_dir, "..", "ONNX32Weights", "model_v1_mobile.onnx")
    )
    
    # Output Path: cd .. -> Weight_fp16 -> model_fp16.onnx
    OUTPUT_DIR = os.path.abspath(
        os.path.join(current_dir, "..", "Weight_fp16")
    )
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model_fp16.onnx")
    
    if os.path.exists(INPUT_ONNX):
        optimize_onnx_to_fp16(INPUT_ONNX, FINAL_OUTPUT_PATH)
    else:
        print(f"❌ File not found: {INPUT_ONNX}")