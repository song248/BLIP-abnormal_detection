from transformers import BlipProcessor, BlipForImageTextRetrieval
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Image
image = Image.open("./image/cat.png").convert("RGB")

# Text
text = "A cat sitting on a grass"

inputs = processor(image, text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
itm_score = outputs.itm_score

probs = torch.softmax(itm_score, dim=1)
similarity_score = probs[:, 1].item()

print(f"Image-Text Similarity Score: {similarity_score:.4f}")