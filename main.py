import onnxruntime as ort
from PIL import Image
import numpy as np
import torchvision.transforms as T

MODEL_PATH = 'saved_model/unet_model_quantized.onnx'
IMAGE_DIR = 'input_images/'  
OUTPUT_DIR = 'predicted_masks/' 

# Function to load and preprocess the input image
def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    return input_image.numpy()

# Function to post-process the output mask
def postprocess_mask(mask):
    mask = mask.squeeze()  # Remove batch dimension
    mask = (mask > 0.5).astype(np.uint8)  # Binarize mask (thresholding at 0.5)
    return mask


def load_onnx_model(model_path):
    return ort.InferenceSession(model_path)


def run_inference(onnx_session, input_image):
    inputs = {onnx_session.get_inputs()[0].name: input_image}
    output = onnx_session.run(None, inputs)
    return output[0]

# Save the predicted mask as an image
def save_mask_as_image(mask, output_path):
    mask_image = Image.fromarray(mask * 255)  # Convert to 0-255 scale
    mask_image.save(output_path)

def main():
    # Load ONNX model
    onnx_session = load_onnx_model(MODEL_PATH)
    
    while True:
        # Check for new images in the directory
        for image_name in os.listdir(IMAGE_DIR):
            image_path = os.path.join(IMAGE_DIR, image_name)
            output_path = os.path.join(OUTPUT_DIR, f'predicted_{image_name}')
            
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                print(f"Processing {image_path}...")
                

                input_image = preprocess_image(image_path)


                pred_mask = run_inference(onnx_session, input_image)


                pred_mask = postprocess_mask(pred_mask)


                save_mask_as_image(pred_mask, output_path)
                print(f"Predicted mask saved at {output_path}")


        # Sleep for a short period to avoid busy-waiting
        time.sleep(5)

if __name__ == '__main__':
    main()
