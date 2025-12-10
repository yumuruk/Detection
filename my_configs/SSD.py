# =========================================================
# 1. 기본 설정 상속 (SSD512 사용)
# =========================================================
_base_ = [
    '../mmdetection/configs/ssd/ssd512_coco.py'
]

# =========================================================
# 2. 클래스 및 메타정보 (UJED 데이터셋)
# =========================================================
metainfo = {
    'classes': ('echinus', 'holothurian', 'scallop', 'starfish'),
    'palette': [(4, 42, 255), (11, 219, 235), (243, 243, 243), (0, 223, 183)]
}

# =========================================================
# 3. 모델 헤드 수정
# =========================================================
model = dict(
    bbox_head=dict(
        num_classes=4,  # 내 클래스 개수
        # SSD512에 최적화된 Anchor 비율 사용
        anchor_generator=dict(basesize_ratio_range=(0.1, 0.9))
    )
)

# =========================================================
# 4. 데이터 파이프라인 (Augmentation 전부 OFF)
# =========================================================
data_root = 'data/UJED/'
input_size = 512  # 512x512 고정

# [핵심 수정] Train Pipeline에서 RandomFlip, Expand, Crop 등 모든 증강 기법 제거
# 순수하게 이미지 로드 -> 512 리사이즈 -> 포맷 변환만 수행
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    # keep_ratio=False: 강제로 512x512 정사각형으로 변환 (SSD 필수)
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='PackDetInputs')
]

# Test Pipeline도 동일하게 구성 (meta_keys 자동 설정)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# =========================================================
# 5. 데이터로더 설정
# =========================================================
train_dataloader = dict(
    batch_size=16,     # GPU당 16장 (3090 7장 기준 Total 112)
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline  # 위에서 정의한 Clean Pipeline 적용
    )
)

# 검증 및 테스트는 반드시 Batch Size 1
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/valid/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')

# =========================================================
# 6. 학습 스케줄 (120 Epoch)
# =========================================================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=1)

# 체크포인트 저장 (Best mAP 기준)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5, 
        save_best='coco/bbox_mAP', 
        rule='greater', 
        max_keep_ckpts=3
    )
)

# Pretrained Weight 로드 (SSD512 COCO)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth'

# 작업 경로
work_dir = './work_dirs/ssd512_ujed_120e'