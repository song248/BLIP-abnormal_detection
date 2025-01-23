from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

image_path = "./image/cat.png"
image = Image.open(image_path).convert("RGB")

question = "What is the name of the animal in the image?"
inputs = processor(image, question, return_tensors="pt")

with torch.no_grad():
    out = model.generate(**inputs)

answer = processor.decode(out[0], skip_special_tokens=True)
print("Answer:", answer)