# =========================================================
# 1. 기본 설정 상속 (DETR ResNet-50, 150 Epoch)
# =========================================================
_base_ = [
    '../mmdetection/configs/detr/detr_r50_8xb2-150e_coco.py'
]

# =========================================================
# 2. 클래스 및 색상 정의 (UJED 데이터셋)
# =========================================================
metainfo = {
    'classes': ('echinus', 'holothurian', 'scallop', 'starfish'),
    'palette': [
        (4, 42, 255),    # echinus
        (11, 219, 235),  # holothurian
        (243, 243, 243), # scallop
        (0, 223, 183)    # starfish
    ]
}

# =========================================================
# 3. 모델 헤드 수정
# =========================================================
model = dict(
    bbox_head=dict(
        num_classes=4  # 내 클래스 개수 (배경 제외)
    )
)

# =========================================================
# 4. 데이터 경로 및 파이프라인 설정
# =========================================================
data_root = 'data/UJED/'

# DETR은 입력 이미지 크기에 민감합니다. 
# 640x480 고정보다는 Base Config의 Multi-scale Resize를 사용하는 것이 
# 수중 작은 객체 탐지에 훨씬 유리합니다. (따라서 pipeline을 굳이 재정의하지 않고 상속받아 씁니다)

# [Test Pipeline 수정] 
# Faster R-CNN때 겪었던 에러 방지를 위해 meta_keys를 제거한 버전을 명시합니다.
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True), # DETR 기본 권장 크기
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs'
        # meta_keys 삭제됨 -> 자동 설정
    )
]

# =========================================================
# 5. 데이터로더 설정 (RTX 3090 VRAM 활용)
# =========================================================
train_dataloader = dict(
    batch_size=4,      # DETR은 무거우므로 GPU당 4장 권장
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
    )
)

# ★ 중요: 검증 시에는 반드시 Batch Size = 1
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
        # [수정됨] valid -> val (중요! 폴더명 확인 필수)
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
# 6. 최적화 설정 (DETR 전용)
# =========================================================
# DETR은 SGD 대신 AdamW를 사용하며, 150 Epoch 이상 학습해야 성능이 나옵니다.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)

# 체크포인트 저장 설정
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,             # 5 epoch마다 저장
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=3
    )
)

# [수정됨] MMDetection v3.0 호환 체크포인트 (링크 업데이트됨)
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

# 작업 경로
work_dir = './work_dirs/detr_r50_ujed_150e'