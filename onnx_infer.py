from PIL import Image
import onnxruntime as ort
import numpy as np
from onnxruntime import InferenceSession



def preprocess_image(image_path, input_size=None):
    img = Image.open(image_path).convert("RGB")
    if input_size:
        img = img.resize(input_size, Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    # Rearrange axes to (C, H, W)
    img_np = np.transpose(img_np, (2, 0, 1))
    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)
    return img_np


def prepre_input_dict_and_run_inference(session: InferenceSession, input_image):
    # Get model input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: input_image})
    output_image = outputs[0]  # Usually the first output
    return output_image


def postprocess_output(output_tensor):
    # Remove batch dimension
    output_tensor = output_tensor.squeeze(0)
    # Change to HWC
    output_tensor = np.transpose(output_tensor, (1, 2, 0))
    # Clip and scale back to 0-255
    output_tensor = np.clip(output_tensor, 0, 1)
    output_tensor = (output_tensor * 255.0).astype(np.uint8)
    return output_tensor






def postprocess_image(image_path, output_size=None):
    img = Image.open(image_path).convert("RGB")
    if output_size:
        img = img.resize(output_size, Image.LANCZOS)
    return img


images = []
for i in range(1, 101):
    idx = f'{i:03d}'
    images.append(preprocess_image(f"test_images/urban100x4/img_{idx}.png"))



ort_session = ort.InferenceSession("swinir_real_x4.onnx", providers=["CUDAExecutionProvider"])
# ort_session = ort.InferenceSession("swin2sr_real_x4.onnx", providers=["CUDAExecutionProvider"])


from time import time
inference_times = []
for i, image in enumerate(images):
    start_time = time()
    output_image = prepre_input_dict_and_run_inference(ort_session, image)
    end_time = time()
    inference_times.append(end_time - start_time)
    print(f"Inference time: {end_time - start_time} seconds")
    output_img_np = postprocess_output(output_image)
    output_img = Image.fromarray(output_img_np)
    output_img.save(f"results/onnx/urban100/img_{i:03d}.png")

print(f"Average inference time: {sum(inference_times) / len(inference_times)} seconds")