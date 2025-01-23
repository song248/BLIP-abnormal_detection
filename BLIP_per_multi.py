import os, json
import pandas as pd
import utils
import cv2
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval


processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    
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

# Create result directory if it doesn't exist
if not os.path.exists("result"):
    os.makedirs("result")

# Process each video in the video directory
# video_dir = "./video"
video_dir = "./video/falldown"
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
total_videos = len(video_files)

for idx, video_file in enumerate(video_files):
    video_path = os.path.join(video_dir, video_file)
    print(f"Processing video {idx + 1}/{total_videos}: {video_file}...")

    # Load video and extract frames
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

    model_df = pd.DataFrame(index=range(len_frame), columns=['fire', 'falldown', 'violence'])
    model_df.loc[0, ['fire', 'falldown', 'violence']] = 0

    # Fill the rest of the frames
    for i in range(len(df)):
        start_frame = df.loc[i, 'frame']
        end_frame = df.loc[i+1, 'frame'] if i+1 < len(df) else len_frame
        
        model.loc[start_frame:end_frame-1, 'fire'] = df.loc[i, 'fire_result']
        model.loc[start_frame:end_frame-1, 'falldown'] = df.loc[i, 'falldown_result']
        model.loc[start_frame:end_frame-1, 'violence'] = df.loc[i, 'violence_result']

    model_df = model_df.fillna(method='ffill')
    model_df.insert(0, 'frame', range(len_frame))

    # Save the final results to CSV
    # output_csv_path = os.path.join("result", f"{os.path.splitext(video_file)[0]}_results.csv")
    output_csv_path = os.path.join("falldown_result", f"{os.path.splitext(video_file)[0]}.csv")
    model_df.to_csv(output_csv_path, index=False)

    print(f"Results for {video_file} saved to {output_csv_path}.")