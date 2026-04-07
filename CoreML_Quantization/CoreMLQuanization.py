import coremltools as ct

# =========================
# TensorFlow SavedModel path
# =========================
tf_model_path = "tf_model"

print("Starting CoreML conversion...")

# =========================
# FP32 CoreML model
# =========================
model_fp32 = ct.convert(
    tf_model_path,
    source="tensorflow",
    convert_to="mlprogram"
)

model_fp32.save("model_fp32.mlpackage")

print("FP32 model saved")

# =========================
# FP16 CoreML model (MOBILE OPTIMIZED)
# =========================
model_fp16 = ct.convert(
    tf_model_path,
    source="tensorflow",
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16
)

model_fp16.save("model_fp16.mlpackage")

print("FP16 model saved")

print("DONE")
