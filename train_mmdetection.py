import argparse
import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector with Auto-Scaling LR')
    parser.add_argument('--config', default='my_configs/DETR.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--batch-size', type=int, default='8', help='Batch size per GPU (Overwrites config)')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    return args

def get_model_settings(config_name):
    """
    Config 파일 이름에 따라 모델별 Default Setting을 반환하는 함수
    반환값: (Optimizer유형, Base_LR, Base_Batch_Size, Weight_Decay)
    """
    config_name = config_name.lower()

    # 1. YOLOX 계열 (가장 먼저 체크, 설정이 까다로움)
    if 'yolox' in config_name:
        print("⚔️ Model detected: YOLOX")
        return {
            'type': 'YOLOX_SGD', # Main 함수에서 구분을 위해 별도 타입 지정
            'base_lr': 0.01,     # YOLOX Standard (Batch 64 기준)
            'base_batch': 64,    # YOLOX는 보통 8 GPU x 8 samples = 64 기준
            'weight_decay': 5e-4,
            'nesterov': True
        }

    # 2. DETR 계열 (AdamW 사용, 매우 낮은 LR)
    elif 'detr' in config_name or 'dino' in config_name:
        print("🤖 Model detected: DETR/Transformer-based")
        return {
            'type': 'AdamW',
            'base_lr': 0.0001,  # 1e-4
            'base_batch': 16,   # DETR 표준 배치
            'weight_decay': 0.0001
        }
    
    # 3. SSD 계열 (SGD 사용, 보통 LR이 조금 낮음)
    elif 'ssd' in config_name:
        print("🚀 Model detected: SSD")
        return {
            'type': 'SGD',
            'base_lr': 0.001,   # SSD는 설정에 따라 다르지만 보통 1e-3 ~ 1e-2
            'base_batch': 32,   # SSD 표준 배치
            'weight_decay': 5e-4
        }
    
    # 4. Faster R-CNN / RetinaNet 등 일반 CNN (SGD 사용, 높은 LR)
    else:
        print("📦 Model detected: CNN-based (Faster R-CNN/YOLOv3 etc.)")
        return {
            'type': 'SGD',
            'base_lr': 0.02,    # Standard ImageNet Pretrained LR
            'base_batch': 16,   # MMDetection Standard
            'weight_decay': 0.0001
        }

def main():
    args = parse_args()

    # 1. Config 로드
    cfg = Config.fromfile(args.config)
    
    # 2. Work Directory 설정
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = f'./work_dirs/{os.path.splitext(os.path.basename(args.config))[0]}'

    # 3. GPU 개수 확인 및 배치 사이즈 계산
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU found. This script requires GPU.")
    
    # 사용자 입력 Batch Size가 있으면 덮어쓰기
    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
    
    per_gpu_batch = cfg.train_dataloader.batch_size
    total_batch_size = per_gpu_batch * num_gpus
    
    # 4. 모델별 최적 설정 가져오기 (핵심 로직)
    settings = get_model_settings(os.path.basename(args.config))
    
    # 5. Linear Scaling Rule 적용
    # New LR = Base LR * (Total Batch Size / Base Batch Size)
    scaling_factor = total_batch_size / settings['base_batch']
    scaled_lr = settings['base_lr'] * scaling_factor

    print("="*50)
    print(f"📊 Auto-Scaling Configuration Report")
    print(f"   - GPU Count       : {num_gpus}")
    print(f"   - Per GPU Batch   : {per_gpu_batch}")
    print(f"   - Total Batch Size: {total_batch_size}")
    print(f"   - Model Type      : {settings['type']}")
    print(f"   - Base LR         : {settings['base_lr']} (at batch {settings['base_batch']})")
    print(f"   - Scaling Factor  : x{scaling_factor:.2f}")
    print(f"   ✅ Final LR       : {scaled_lr:.6f}")
    print("="*50)

    # 6. Config에 Optimizer 및 LR 적용
    
    # [Case A] YOLOX (Nesterov + Paramwise Config 필수)
    if settings['type'] == 'YOLOX_SGD':
        cfg.optim_wrapper.optimizer = dict(
            type='SGD',
            lr=scaled_lr,
            momentum=0.9,
            weight_decay=settings['weight_decay'],
            nesterov=True
        )
        # YOLOX 핵심: Norm과 Bias에는 Weight Decay 적용 안 함
        cfg.optim_wrapper.paramwise_cfg = dict(
            norm_decay_mult=0., 
            bias_decay_mult=0.
        )

    # [Case B] DETR / Transformer (AdamW + Gradient Clip)
    elif settings['type'] == 'AdamW':
        cfg.optim_wrapper.optimizer = dict(
            type='AdamW',
            lr=scaled_lr,
            weight_decay=settings['weight_decay']
        )
        # DETR은 clip_grad가 필수
        if cfg.get('optim_wrapper', {}).get('clip_grad', None) is None:
             cfg.optim_wrapper.clip_grad = dict(max_norm=0.1, norm_type=2)

    # [Case C] 일반 CNN (Standard SGD)
    else: 
        cfg.optim_wrapper.optimizer = dict(
            type='SGD',
            lr=scaled_lr,
            momentum=0.9,
            weight_decay=settings['weight_decay']
        )

    # 7. Resume 설정
    if args.resume:
        cfg.resume = True

    # 8. Runner 실행
    setup_cache_size_limit_of_dynamo()
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()