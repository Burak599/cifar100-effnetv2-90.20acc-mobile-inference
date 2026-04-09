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

        # 🔥 KRİTİK: yeni exporter KULLANMA
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

    # Define the absolute base path for weights
    # This ensures the script finds the model regardless of where it's executed from
    BASE_WEIGHTS_PATH = "/home/burak/cifar100-effnetv2-90.20acc-mobile-inference/research_journey/Test8Resume1/user_weights"
    BASE_OUTPUT_PATH = "/home/burak/cifar100-effnetv2-90.20acc-mobile-inference/ONNX/ONNX_Weight"

    # Construct the full path to the specific checkpoint
    CHECKPOINT_PATH = os.path.join(BASE_WEIGHTS_PATH, "user_best_model_9.pth")
    ONNX_OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, "user_model_fp32.onnx")

    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)

    convert_to_onnx(CHECKPOINT_PATH, ONNX_OUTPUT_PATH)
