import json
import os
import pandas as pd
from tqdm import tqdm
import utils
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
device_ids = list(range(num_gpus))
torch.cuda.empty_cache()

torch.cuda.set_device(0)

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

if num_gpus > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load prompts from JSON
with open("prompt.json", "r") as f:
    prompt_data = json.load(f)

# Extract prompts from JSON
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

# Function to calculate similarity between image and text
def calculate_similarity(image, text):
    inputs = processor(image, text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.itm_score, dim=-1)
    return probs[0][1].item()

# Load video and extract frames
# video_path = "./video/falldown/falldown.mp4"
video_path = "./video/Explosion002_x264.mp4"
# output_dir = "./extracted_frames"
print("Extracting frames from video...")

frames = utils.extract_frames(video_path)
# Calculate similarity between frames and prompts
results = []
print("Calculating similarity between frames and prompts...")
for frame_idx, frame in enumerate(frames):
    frame_results = {"frame": (frame_idx + 1) * 15}
    for event_name, event_prompts in prompts.items():
        for i, prompt in enumerate(event_prompts["normal"]):
            similarity = calculate_similarity(frame, prompt)
            frame_results[f"{event_name}_normal_{i+1}"] = similarity
        for i, prompt in enumerate(event_prompts["abnormal"]):
            similarity = calculate_similarity(frame, prompt)
            frame_results[f"{event_name}_abnormal_{i+1}"] = similarity
    results.append(frame_results)

# Convert results to DataFrame
df = pd.DataFrame(results)

# Get the total number of frames in the video
cap = cv2.VideoCapture(video_path)
len_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Process the results to create the final model DataFrame
fire_columns = [col for col in df.columns if 'fire' in col]
falldown_columns = [col for col in df.columns if 'falldown' in col]
violence_columns = [col for col in df.columns if 'violence' in col]

df['fire_result'] = df[fire_columns].idxmax(axis=1).apply(lambda x: 1 if 'abnormal' in x else 0)
df['falldown_result'] = df[falldown_columns].idxmax(axis=1).apply(lambda x: 1 if 'abnormal' in x else 0)
df['violence_result'] = df[violence_columns].idxmax(axis=1).apply(lambda x: 1 if 'abnormal' in x else 0)

model = pd.DataFrame(index=range(len_frame), columns=['fire', 'falldown', 'violence'])

# Fill the first frame with 0
model.loc[0, ['fire', 'falldown', 'violence']] = 0

# Fill the rest of the frames
for i in range(len(df)):
    start_frame = df.loc[i, 'frame']
    end_frame = df.loc[i+1, 'frame'] if i+1 < len(df) else len_frame
    
    if start_frame == 15:
        model.loc[1:15, 'fire'] = df.loc[i, 'fire_result']
        model.loc[1:15, 'falldown'] = df.loc[i, 'falldown_result']
        model.loc[1:15, 'violence'] = df.loc[i, 'violence_result']
    elif start_frame == 30:
        model.loc[16:30, 'fire'] = df.loc[i, 'fire_result']
        model.loc[16:30, 'falldown'] = df.loc[i, 'falldown_result']
        model.loc[16:30, 'violence'] = df.loc[i, 'violence_result']
    else:
        model.loc[start_frame:end_frame-1, 'fire'] = df.loc[i, 'fire_result']
        model.loc[start_frame:end_frame-1, 'falldown'] = df.loc[i, 'falldown_result']
        model.loc[start_frame:end_frame-1, 'violence'] = df.loc[i, 'violence_result']

# Forward fill any remaining NaN values
model = model.fillna(method='ffill')

# Insert frame column at the beginning
model.insert(0, 'frame', range(len_frame))

# Save the final results to CSV
output_csv_path = 'integrated_results.csv'
model.to_csv(output_csv_path, index=False)

print(f"결과가 {output_csv_path} 파일로 저장되었습니다.")