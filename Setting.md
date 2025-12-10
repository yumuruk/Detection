OS: Ubuntu 20.04 / 22.04

### 가상환경 생성

```
conda create -n mmdet python=3.10 -y

conda activate mmdet
```

### 필수 라이브러리 설치

Intel MKL FATAL ERROR 및 Segmentation fault방지 위해서 pip 대신 conda 사용

```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install "numpy<2.0.0" "opencv-python<4.10.0" # 버전 충돌 방지 (NumPy, OpenCV)

# OpenMMLab 관리 도구 설치
pip install -U openmim
pip install mmengine

 
mim install "mmcv==2.1.0" # 컴파일 없이 설치하기 위해 mim 사용

git clone https://github.com/open-mmlab/mmdetection.git # MMDetection 레포 클론 
 
cd mmdetection # 폴더 이동

python setup.py develop # 소스코드 연결

```
