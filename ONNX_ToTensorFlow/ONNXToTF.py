import os
import subprocess

# =========================
# Input ONNX model path
# =========================
onnx_path = "model_v1_mobile.onnx"

# =========================
# Output folder
# =========================
output_dir = "tf_model"

# =========================
# Run conversion using onnx2tf
# =========================
cmd = [
    "onnx2tf",
    "-i", onnx_path,
    "-o", output_dir
]

print("Starting ONNX → TensorFlow conversion...")

result = subprocess.run(cmd, capture_output=True, text=True)

# =========================
# Check result
# =========================
if result.returncode == 0:
    print(f"Conversion successful. TensorFlow model saved to: {output_dir}")
else:
    print("Conversion failed!")
    print(result.stderr)
