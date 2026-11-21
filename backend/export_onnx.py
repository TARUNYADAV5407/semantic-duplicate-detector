import os
import torch
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_dynamic, QuantType

model_dir = "./my_trained_model"
output_dir = "./onnx_model"
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)
model.eval()

dummy = tokenizer("What is overfitting?", return_tensors="pt")
fp32_path = "model_fp32.onnx"
quantized_path = os.path.join(output_dir, "model_quantized.onnx")

torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
    fp32_path,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
    opset_version=14,
)

# Quantize to ~23 MB and save directly into onnx_model/
quantize_dynamic(fp32_path, quantized_path, weight_type=QuantType.QUInt8)
tokenizer.save_pretrained(output_dir)

# Remove the large temporary FP32 file
if os.path.exists(fp32_path):
    os.remove(fp32_path)

print("Done! Saved quantized ONNX model and tokenizer to ./onnx_model/")