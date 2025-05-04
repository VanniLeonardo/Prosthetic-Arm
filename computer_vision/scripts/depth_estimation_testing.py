import requests
from PIL import Image
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

url = 'https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = DepthProImageProcessorFast.from_pretrained('apple/DepthPro-hf')
model = DepthProForDepthEstimation.from_pretrained('apple/DepthPro-hf').to(device)

inputs = image_processor(images=image, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = model(**inputs)

post_processed_output = image_processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)],
)

field_of_view = post_processed_output[0]['field_of_view']
focal_length = post_processed_output[0]['focal_length']
depth = post_processed_output[0]['predicted_depth']
depth = (depth - depth.min()) / (depth.max() - depth.min())
depth = depth * 255.
depth = depth.detach().cpu().numpy()
depth = Image.fromarray(depth.astype('uint8'))
