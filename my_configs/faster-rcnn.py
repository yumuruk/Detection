# 파일 경로: Detection/my_configs/faster_rcnn_ujed.py

# 1. 원본 Config 상속
_base_ = [
    '../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
]

# =========================================================
# 2. 클래스 및 색상 정의
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
    roi_head=dict(
        bbox_head=dict(
            num_classes=4  # 내 클래스 개수
        )
    )
)

# =========================================================
# 4. 데이터 파이프라인 (Resize: 640x480 고정)
# =========================================================
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    # 원본 해상도(640, 480)로 리사이즈 (Augmentation 없음)
    # dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs' 
        # meta_keys=(...) 부분을 삭제했습니다. 
        # 이렇게 비워두면 MMDetection이 알아서 scale_factor를 포함한 필수 정보를 다 챙깁니다.
    )
]

# =========================================================
# 5. 데이터로더 설정 (Single GPU Optimization)
# =========================================================
data_root = 'data/UJED/'

train_dataloader = dict(
    batch_size=36,        # GPU 1장당 12장
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/valid/'),
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        pipeline=test_pipeline
    )
)

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')

# =========================================================
# 6. 학습 스케줄 및 옵티마이저 (100 Epoch 설정)
# =========================================================
# [변경] 100 Epoch 학습
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

# 학습률 최적화 (Batch 12 기준: 0.015)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)
)

# [변경] 학습률 감소 스케줄 조정
# 100 epoch니까 전체의 70%(70 epoch), 90%(90 epoch) 지점에서 학습률을 1/10로 줄임
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=True, milestones=[70, 90], gamma=0.1)
]

# [추가] 최고 성능 모델 자동 저장 (Early Stopping 효과)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,                 # 5 epoch마다 일반 저장 (용량 절약)
        save_best='coco/bbox_mAP',  # Validation mAP가 가장 높은 모델을
        rule='greater',             # 클수록 좋음
        max_keep_ckpts=3            # 최고 모델 3개만 남기고 나머지 삭제 (용량 관리)
    )
)

# Pretrained Weight 로드
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 로그 저장 경로
work_dir = './work_dirs/faster_rcnn_ujed_100e'