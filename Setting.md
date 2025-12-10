OS: Ubuntu 20.04 / 22.04

### 가상환경 생성

```
conda create -n mmdet python=3.10 -y

conda activate mmdet
```

### 필수 라이브러리 설치

Intel MKL FATAL ERROR 및 Segmentation fault방지 위해서 pip 대신 conda 사용

> Intel MKL FATAL ERROR (Library Conflict)

원인: pip로 설치한 PyTorch에 내장된 연산 라이브러리(libiomp5.so)가 시스템 또는 Conda 환경의 MKL 라이브러리와 중복 로드되어 충돌 발생.



> Segmentation fault (Binary Incompatibility)

원인: pip로 배포되는 사전 빌드된(pre-built) PyTorch 바이너리가 현재 서버의 시스템 라이브러리(GCC/GLIBC 등)와 호환되지 않아 발생하는 메모리 참조 오류.



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

WS01이랑 WS09에 해당하는 환경세팅 yaml파일 형태로 저장

세팅한 이후 cehck_env.py 실행해서 제대로 설치됐는지 확인하면 됨
