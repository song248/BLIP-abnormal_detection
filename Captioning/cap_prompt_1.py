import cv2
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm  # 진행 상황 표시를 위한 라이브러리

# BLIP 모델과 프로세서 로드
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# 제외할 단어 목록
bad_words = ["green", "standing"]  # 제외할 단어 리스트
bad_words_ids = [processor.tokenizer.encode(word, add_special_tokens=False) for word in bad_words]
bad_words_ids = [word_id for sublist in bad_words_ids for word_id in sublist]  # 단일 리스트로 변환

# 동영상 파일 경로
video_path = "./video/explosion.mp4"  # 동영상 파일 경로
output_json_path = "./caption_json/explosion_output.json"  # 출력 JSON 파일 경로

# 동영상 열기
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"동영상을 열 수 없습니다: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))  # 동영상의 FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 동영상의 총 프레임 수
frame_interval = 15  # 15프레임마다 이미지 추출
frame_count = 0
captions = {}

# tqdm을 사용하여 진행 상황 표시
with tqdm(total=total_frames, desc="동영상 처리 중") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 15프레임마다 이미지 추출 및 캡션 생성
        if frame_count % frame_interval == 0:
            # OpenCV 프레임을 PIL 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 이미지를 모델 입력으로 변환
            inputs = processor(pil_image, return_tensors="pt")

            # 캡션 생성 (특정 단어 제외, 하나의 캡션만 생성)
            outputs = model.generate(
                **inputs,
                num_return_sequences=1,  # 하나의 캡션만 생성
                num_beams=5,  # 빔 서치 사용 (더 나은 캡션 생성)
                bad_words_ids=[bad_words_ids],  # 제외할 단어 설정
            )

            # 생성된 캡션 디코딩
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # 캡션 저장
            captions[f"{frame_count // frame_interval * frame_interval}fps"] = caption

        frame_count += 1
        pbar.update(1)  # 진행 상황 업데이트

# 동영상 해제
cap.release()

# JSON 형식으로 저장
output_data = {
    "filename": video_path,
    "captions": captions
}

# JSON 파일로 저장
with open(output_json_path, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"\n캡셔닝 결과가 {output_json_path}에 저장되었습니다.")