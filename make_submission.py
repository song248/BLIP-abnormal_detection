import pandas as pd
import cv2

# 0. input_video.mp4를 이용해 전체 프레임 개수 확인 후 len_frame에 저장해두기
video_path = './video/Explosion002_x264.mp4'
cap = cv2.VideoCapture(video_path)
len_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# 1. csv 파일 읽어오기
csv_path = 'output_similarity.csv'
df = pd.read_csv(csv_path)

# 2. fire는 fire끼리 falldown은 falldown끼리 violence는 violence끼리 처리
# 각 이벤트별로 normal과 abnormal 컬럼을 분리
fire_columns = [col for col in df.columns if 'fire' in col]
falldown_columns = [col for col in df.columns if 'falldown' in col]
violence_columns = [col for col in df.columns if 'violence' in col]

# 3. csv 파일에 기록된 숫자는 유사도를 나타냄. 이벤트끼리 유사도를 비교해서 가장 유사도가 높은 것이 normal인지 abnormal인지 확인
# 각 이벤트별로 normal과 abnormal 중 더 높은 값을 선택
df['fire_result'] = df[fire_columns].idxmax(axis=1).apply(lambda x: 1 if 'abnormal' in x else 0)
df['falldown_result'] = df[falldown_columns].idxmax(axis=1).apply(lambda x: 1 if 'abnormal' in x else 0)
df['violence_result'] = df[violence_columns].idxmax(axis=1).apply(lambda x: 1 if 'abnormal' in x else 0)

# 4. csv의 frame 컬럼을 이용하여 fire, falldown, violence가 각각 normal인지 abnormal인지 표시하는 새로운 데이터 프레임 'model'을 만듦.
model = pd.DataFrame(index=range(len_frame), columns=['fire', 'falldown', 'violence'])

model.loc[0, ['fire', 'falldown', 'violence']] = 0

for i in range(len(df)):
    start_frame = df.loc[i, 'frame']
    end_frame = df.loc[i+1, 'frame'] if i+1 < len(df) else len_frame
    
    model.loc[start_frame:end_frame-1, 'fire'] = df.loc[i, 'fire_result']
    model.loc[start_frame:end_frame-1, 'falldown'] = df.loc[i, 'falldown_result']
    model.loc[start_frame:end_frame-1, 'violence'] = df.loc[i, 'violence_result']

# 5. len_frame만큼 데이터 프레임을 채움. 남는 칸은 마지막 값으로 채움.
model = model.fillna(method='ffill')
model.insert(0, 'frame', range(len_frame))
output_csv_path = 'model_results.csv'
model.to_csv(output_csv_path, index=False)

print(f"Save result to {output_csv_path}. Done!")