# BLIP-abnormal_detection
## Abnormal Behavior Detection Using BLIP model  
This repository is aimed at evaluating the performance of abnormal behavior detection using the BLIP model for CCTV
See how VLM performs in abnormal behavior detection  
https://huggingface.co/spaces/PIA-SPACE-LAB/PIA-SPACE_LeaderBoard

## Envrironment setup

### 1. Clone repository
```
git clone git@github.com:song248/BLIP-abnormal_detection.git
cd BLIP-abnormal_detection
```

### 2. Setup Conda
```
conda create -n blip python=3.11 -y
conda activate blip
```

### 3. Install package
```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
```
### 4. Prepare Dataset
Prepare video and prompt  
Prompt must be inclue normal, abnormal situation  
Please refer to the included JSON <b>'prompt.json'</b>

### 5. Execute main.py
```
python main.py
```
main.py = blip_cal_sim.py + make_submission.py

### ETC

citation  
> https://huggingface.co/Salesforce/blip-vqa-base
https://github.com/salesforce/BLIP?tab=readme-ov-file  
https://github.com/salesforce/BLIP/blob/main/models/blip_retrieval.py  
https://github.com/salesforce/BLIP/blob/main/train_retrieval.py  
https://github.com/dino-chiio/blip-vqa-finetune/blob/main/finetuning.py  
