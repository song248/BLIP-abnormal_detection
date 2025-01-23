import cv2, os
from PIL import Image
from tqdm import tqdm

# evary 15fps
def extract_frames(video_path, frame_interval=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                frames.append(Image.fromarray(frame_rgb))
            frame_count += 1
            pbar.update(1)

    cap.release()
    return frames

'''
if memory rackage use below code
save result in extracted_frames folder not in list(on memory)
'''

# def extract_frames(video_path, frame_interval=15, output_dir="extracted_frames"):
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Can't open video file: {video_path}")

#     frame_count = 0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if frame_count % frame_interval == 0:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame_image = Image.fromarray(frame_rgb)
#                 frame_image.save(os.path.join(output_dir, f"frame_{frame_count:05d}.jpg"))
#             frame_count += 1
#             pbar.update(1)

#     cap.release()
#     print(f"Save to {output_dir}")
