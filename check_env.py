import torch, mmcv
from mmdet.apis import init_detector

print("="*40)
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Avail: {torch.cuda.is_available()}")
print(f"✅ GPU Count: {torch.cuda.device_count()}")
print(f"✅ MMCV: {mmcv.__version__}")
print("="*40)

if torch.cuda.is_available():
    print("GPU 인식 성공.")
else:
    print("GPU 인식 실패")
