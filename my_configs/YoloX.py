# =========================================================
# 1. 기본 설정 상속 (YOLOX-s)
# =========================================================
_base_ = ['../mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py']

# =========================================================
# 2. 메타 정보
# =========================================================
metainfo = {
    'classes': ('echinus', 'holothurian', 'scallop', 'starfish'),
    'palette': [(4, 42, 255), (11, 219, 235), (243, 243, 243), (0, 223, 183)]
}

# =========================================================
# 3. 모델 헤드 수정
# =========================================================
model = dict(
    bbox_head=dict(num_classes=4)  # 클래스 4개
)

# =========================================================
# 4. 데이터 파이프라인 (Augmentation 제거)
# =========================================================
# 데이터 루트 (상대 경로 기준점)
data_root = 'data/UJED/' 

input_size = (640, 640)

# Test/Val Pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Train Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackDetInputs')
]

# -------------------------------------------------------
# 2. Train Dataloader
# -------------------------------------------------------
train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root, 
        metainfo=metainfo,    
        # Annotation 경로 (data/UJED/annotations/instances_train.json)
        ann_file='annotations/instances_train.json', 
        # [수정 완료] 이미지 경로 (data/UJED/images/train/)
        data_prefix=dict(img='images/train/'), 
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=None
    )
)

# -------------------------------------------------------
# 3. Val Dataloader
# -------------------------------------------------------
val_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        # Annotation 경로 (data/UJED/annotations/instances_val.json)
        ann_file='annotations/instances_val.json', 
        # [수정 완료] 이미지 경로 (data/UJED/images/val/)
        data_prefix=dict(img='images/valid/'), 
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None
    )
)

# -------------------------------------------------------
# 4. Test Dataloader
# -------------------------------------------------------
test_dataloader = val_dataloader

# -------------------------------------------------------
# 5. Evaluator
# -------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json', 
    metric='bbox',
    backend_args=None
)
test_evaluator = val_evaluator

# =========================================================
# 6. 학습 설정
# =========================================================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

custom_hooks = [] 

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='coco/bbox_mAP', rule='greater', max_keep_ckpts=3)
)

# [수정 완료] 404 에러 해결된 올바른 링크 (8xb8 -> 8x8)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

work_dir = './work_dirs/yolox_s_ujed_100e'