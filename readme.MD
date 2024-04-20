===== Requierments =====

conda create -n ccp python
conda activate ccp
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install wandb
pip install einops
pip install torchinfo

===== Training =====

_train.py 실행

===== dataset =====

dataset/ 폴더에 데이터셋을 넣어주세요.
구조 
dataset/
    Training/
        image/
            AP10_City_IMAGE/
            AP10_City_SH/
            ...
        label/
    Validation/
        image/
            AP10_City_IMAGE/
            AP10_City_SH/
            ...
        label/