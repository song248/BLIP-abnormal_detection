import json
import pandas as pd
from tqdm import tqdm
import utils

import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval


# BLIP model load
processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with open("prompt.json", "r") as f:
    prompt_data = json.load(f)

# prompot from JSON
events = prompt_data["PROMPT_CFG"]
prompts = {}
for event in events:
    event_name = event["event"]
    normal_prompts = [p["sentence"] for p in event["prompts"]["normal"]]
    abnormal_prompts = [p["sentence"] for p in event["prompts"]["abnormal"]]
    prompts[event_name] = {
        "normal": normal_prompts,
        "abnormal": abnormal_prompts
    }

# 이미지와 텍스트 간의 유사도 계산
def calculate_similarity(image, text):
    inputs = processor(image, text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.itm_score, dim=-1)
    return probs[0][1].item() 

video_path = "./video/Explosion002_x264.mp4"
print("Extracting frames from video...")
frames = utils.extract_frames(video_path)

results = []
print("Calculating similarity between frames and prompts...")
for idx, frame in enumerate(tqdm(frames, desc="Processing Frames")):
    frame_results = {"frame": (idx + 1) * 15}
    for event_name, event_prompts in prompts.items():
        for i, prompt in enumerate(event_prompts["normal"]):
            similarity = calculate_similarity(frame, prompt)
            frame_results[f"{event_name}_normal_{i+1}"] = similarity
        for i, prompt in enumerate(event_prompts["abnormal"]):
            similarity = calculate_similarity(frame, prompt)
            frame_results[f"{event_name}_abnormal_{i+1}"] = similarity
    results.append(frame_results)

df = pd.DataFrame(results)
df.to_csv("output_similarity.csv", index=False)
print("***Complete***")