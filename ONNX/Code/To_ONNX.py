import torch
import torchvision.models as models
import torch.nn as nn
import os
import onnx

def convert_to_onnx(checkpoint_path, output_path="user_model_fp32.onnx"):

    # Model
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(
        ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    )
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    print("🚀 Exporting clean ONNX...")

    torch.onnx.export(
        model,
        dummy,
        output_path,

        export_params=True,
        opset_version=18,

        do_constant_folding=True,

        input_names=["input"],
        output_names=["output"],

        dynamic_axes=None,

        training=torch.onnx.TrainingMode.EVAL,

        dynamo=False
    )

    print("🔍 Checking model...")
    model_proto = onnx.load(output_path)
    onnx.checker.check_model(model_proto)

    print("📦 Forcing single file...")
    onnx.save_model(
        model_proto,
        output_path,
        save_as_external_data=False
    )

    print("✅ DONE:", output_path)


if __name__ == "__main__":
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

    BASE_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "research_journey", "Test8Resume2", "user_weights")
    BASE_OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "ONNX", "ONNX_Weight")

    CHECKPOINT_PATH  = os.path.join(BASE_WEIGHTS_PATH, "user_best_model_10.pth")
    ONNX_OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH,  "user_model_fp32.onnx")

    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    convert_to_onnx(CHECKPOINT_PATH, ONNX_OUTPUT_PATH)
