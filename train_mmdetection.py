import argparse
import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo


DATASET_REGISTRY = {
    'UJED': {
        'root': 'data/UJED/', # 기본 경로 (Config 파일의 data_root를 덮어씀)
        'classes': ('echinus', 'holothurian', 'scallop', 'starfish'),
        'palette': [(4, 42, 255), (11, 219, 235), (243, 243, 243), (0, 223, 183)]
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector with Auto-Scaling LR')
    parser.add_argument('--dataset', default='UJED', choices=DATASET_REGISTRY.keys(), help='Select dataset metadata (e.g., classes)')
    parser.add_argument('--data-root', type=str, default=None, help='Override data root path (e.g., data/UJED_CycleGAN/)')
    parser.add_argument('--config', default='my_configs/DETR.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--batch-size', type=int, default='8', help='Batch size per GPU (Overwrites config)')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    return args

def apply_dataset_to_cfg(cfg, dataset_name, override_root=None):
    """Config 객체에 선택된 데이터셋의 경로와 클래스 정보를 강제 주입"""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry!")
    
    info = DATASET_REGISTRY[dataset_name]
    
    # [핵심] 실제 사용할 루트 경로 결정
    real_root = override_root if override_root is not None else info['root']
    if not real_root.endswith('/'):
        real_root += '/'

    classes = info['classes']
    num_classes = len(classes)
    
    print(f"\n🔄 [Dataset Setup]")
    print(f"   - Metadata Source : {dataset_name}")
    print(f"   - Actual Data Root: {real_root}") 
    print(f"   - Classes ({num_classes}) : {classes}")

    # 1. Config 메타데이터 주입
    cfg.metainfo = {'classes': classes, 'palette': info.get('palette', None)}
    cfg.data_root = real_root

    # 2. 모델 Head 클래스 수 조정 (이전에 논의된 로직)
    if hasattr(cfg.model, 'bbox_head'):
        if hasattr(cfg.model.bbox_head, 'num_classes'):
            cfg.model.bbox_head.num_classes = num_classes
        elif isinstance(cfg.model.bbox_head, list):
             for head in cfg.model.bbox_head:
                 head.num_classes = num_classes
    if hasattr(cfg.model, 'roi_head') and hasattr(cfg.model.roi_head, 'bbox_head'):
        if hasattr(cfg.model.roi_head.bbox_head, 'num_classes'):
            cfg.model.roi_head.bbox_head.num_classes = num_classes

    # 3. Dataloader 경로 재설정 (스크린샷 기반 경로: annotations/json, images/valid)
    
    # Train
    if hasattr(cfg, 'train_dataloader'):
        cfg.train_dataloader.dataset.data_root = real_root
        cfg.train_dataloader.dataset.metainfo = cfg.metainfo
        cfg.train_dataloader.dataset.ann_file = 'annotations/instances_train.json'
        cfg.train_dataloader.dataset.data_prefix = dict(img='images/train/')

    # Val
    if hasattr(cfg, 'val_dataloader'):
        cfg.val_dataloader.dataset.data_root = real_root
        cfg.val_dataloader.dataset.metainfo = cfg.metainfo
        cfg.val_dataloader.dataset.ann_file = 'annotations/instances_val.json'
        cfg.val_dataloader.dataset.data_prefix = dict(img='images/valid/') # valid 폴더명 반영

    # Test
    if hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.dataset.data_root = real_root
        cfg.test_dataloader.dataset.metainfo = cfg.metainfo
        cfg.test_dataloader.dataset.ann_file = 'annotations/instances_test.json'
        cfg.test_dataloader.dataset.data_prefix = dict(img='images/test/')

    # Evaluator
    if hasattr(cfg, 'val_evaluator'):
        cfg.val_evaluator.ann_file = os.path.join(real_root, 'annotations/instances_val.json')
    if hasattr(cfg, 'test_evaluator'):
        cfg.test_evaluator.ann_file = os.path.join(real_root, 'annotations/instances_test.json')

    return cfg

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
    
    # [추가] 3단계: 데이터셋 정보로 Config 덮어쓰기
    cfg = apply_dataset_to_cfg(cfg, args.dataset, override_root=args.data_root)
    
    # 2. Work Directory 설정 (데이터셋 폴더 이름을 로그 폴더에 포함시킴)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        
        # 실제 데이터 폴더 이름을 따와서 로그 폴더명에 사용
        if args.data_root:
            dataset_folder = os.path.basename(os.path.normpath(args.data_root))
        else:
            dataset_folder = args.dataset
            
        cfg.work_dir = f'./work_dirs/{config_name}_{dataset_folder}'

    # 3. GPU 개수 확인 및 배치 사이즈 계산 (기존 로직)
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
    scaling_factor = total_batch_size / settings['base_batch']
    scaled_lr = settings['base_lr'] * scaling_factor

    print("="*50)
    print(f"📊 Auto-Scaling Configuration Report")
    print(f"   - GPU Count       : {num_gpus}")
    print(f"   - Per GPU Batch   : {per_gpu_batch}")
    print(f"   - Total Batch Size: {total_batch_size}")
    print(f"   - Model Type      : {settings['type']}")
    print(f"   - Base LR         : {settings['base_lr']} (at batch {settings['base_batch']})")
    print(f"   - Scaling Factor  : x{scaling_factor:.2f}")
    print(f"   ✅ Final LR       : {scaled_lr:.6f}")
    print("="*50)

    # 6. Config에 Optimizer 및 LR 적용 (기존 로직)
    if settings['type'] == 'YOLOX_SGD':
        cfg.optim_wrapper.optimizer = dict(type='SGD', lr=scaled_lr, momentum=0.9, weight_decay=settings['weight_decay'], nesterov=True)
        cfg.optim_wrapper.paramwise_cfg = dict(norm_decay_mult=0., bias_decay_mult=0.)
    elif settings['type'] == 'AdamW':
        cfg.optim_wrapper.optimizer = dict(type='AdamW', lr=scaled_lr, weight_decay=settings['weight_decay'])
        if cfg.get('optim_wrapper', {}).get('clip_grad', None) is None:
             cfg.optim_wrapper.clip_grad = dict(max_norm=0.1, norm_type=2)
    else: 
        cfg.optim_wrapper.optimizer = dict(type='SGD', lr=scaled_lr, momentum=0.9, weight_decay=settings['weight_decay'])

    # 7. Resume 설정
    if args.resume:
        cfg.resume = True

    # 8. Runner 실행
    setup_cache_size_limit_of_dynamo()
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()