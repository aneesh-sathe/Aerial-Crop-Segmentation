import torch
import torch.nn as nn
import torchvision.transforms as T
import onnx
from models import UNet
from onnxruntime.quantization import quantize_dynamic, QuantType



model = UNet()  
model.load_state_dict(torch.load('saved_model/model.pth'))
model.eval()  


# batch size of 1, 3 input channels (RGB), and image size of 256x256
dummy_input = torch.randn(1, 3, 256, 256)

# Export the model to ONNX format
onnx_model_path = "saved_model/unet_model.onnx"
torch.onnx.export(
    model,                        
    dummy_input,                  
    onnx_model_path,              
    export_params=True,           
    opset_version=11,             
    input_names=['input'],        
    output_names=['output'],      
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Allow dynamic batch size
)

print("Model has been successfully converted to ONNX and saved as 'unet_model.onnx'")


quantized_model_path = "unet_model_quantized.onnx"
quantize_dynamic(
    onnx_model_path,              
    quantized_model_path,         
    weight_type=QuantType.QInt8   # Quantize weights to 8-bit integers
)

print(f"Model has been quantized to 8-bit and saved as {quantized_model_path}")
