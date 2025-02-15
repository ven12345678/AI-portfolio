from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
  """Generates caption for a given image"""
  image = Image.open(image_path).convert("RGB")
  inputs = processor(images = image, return_tensors = "pt")

  with torch.no_grad():
    output = model.generate(**inputs)

  caption = processor.decode(output[0], skip_special_tokens = True)
  return caption
