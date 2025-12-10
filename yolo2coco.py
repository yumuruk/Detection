import os
import json
import cv2
import yaml
import datetime  # 날짜 생성을 위해 추가
from tqdm import tqdm

# yolo 형식의 data.yaml에서 클래스 정보 로드
def load_config(yaml_path): 
    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # names가 딕셔너리인 경우와 리스트인 경우 모두 처리
    names = data_cfg.get('names')
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        return names
    else:
        raise ValueError("data.yaml의 'names' 형식을 인식할 수 없습니다.")

def yolo_to_coco(root_path, set_mode, classes):
    # 경로 설정 (images/train, labels/train 등)
    # data.yaml의 폴더명(valid)과 실제 폴더명이 일치해야 합니다.
    img_dir_name = 'valid' if set_mode == 'val' else set_mode 
    
    # 실제 폴더가 'valid'인지 'val'인지 확인
    if os.path.exists(os.path.join(root_path, 'images', 'valid')):
        folder_split = 'valid' if set_mode == 'val' or set_mode == 'valid' else set_mode
    else:
        folder_split = set_mode

    img_path = os.path.join(root_path, 'images', folder_split)
    label_path = os.path.join(root_path, 'labels', folder_split)
    
    # 이미지 폴더가 없으면 스킵
    if not os.path.exists(img_path):
        print(f"⚠️ Warning: {img_path} 경로를 찾을 수 없어 건너뜁니다.")
        return

    # 이미지 파일 리스트 불러오기
    images = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    # ★ 수정됨: info와 licenses 키 추가 (MMDetection 에러 방지용)
    current_date = datetime.datetime.now().strftime("%Y/%m/%d")
    coco_format = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Converted from YOLO format",
            "contributor": "",
            "url": "",
            "date_created": current_date
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 카테고리 정보 생성
    for i, cls in enumerate(classes):
        coco_format["categories"].append({"id": i, "name": cls, "supercategory": "object"})

    ann_id = 0
    
    print(f"🔄 Converting {folder_split} data... ({len(images)} files)")
    
    for img_id, img_file in enumerate(tqdm(images)):
        # 1. 이미지 정보 읽기
        image_full_path = os.path.join(img_path, img_file)
        image = cv2.imread(image_full_path)
        
        # 이미지가 깨져있거나 못 읽는 경우 예외처리
        if image is None:
            print(f"❌ Error reading image: {img_file}")
            continue
            
        height, width, _ = image.shape
        
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        # 2. 라벨 파일 읽기
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_full_path = os.path.join(label_path, label_file)

        if os.path.exists(label_full_path):
            with open(label_full_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue # 데이터 오염 방지
                
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])

                # YOLO (Normalized 0~1) -> COCO (Absolute xywh)
                abs_w = w * width
                abs_h = h * height
                abs_x = (x_c * width) - (abs_w / 2)
                abs_y = (y_c * height) - (abs_h / 2)

                coco_format["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
                ann_id += 1

    # 저장 경로 (annotations 폴더 자동 생성)
    save_dir = os.path.join(root_path, 'annotations')
    os.makedirs(save_dir, exist_ok=True)
    
    # MMDetection은 보통 instances_train.json 형태를 선호
    json_name = f'instances_{set_mode}.json'
    save_path = os.path.join(save_dir, json_name)
    
    with open(save_path, 'w') as f:
        json.dump(coco_format, f)
    
    print(f"✅ Saved to: {save_path}")

# 실행부
if __name__ == '__main__':
    # 1. 기본 설정
    ROOT_PATH = 'data/UJED'
    YAML_PATH = os.path.join(ROOT_PATH, 'data.yaml')
    
    # 2. 클래스 정보 로드
    try:
        CLASSES = load_config(YAML_PATH)
        print(f"📂 Loaded Classes from yaml: {CLASSES}")
    except Exception as e:
        print(f"❌ YAML 로드 실패: {e}")
        print("경로가 정확한지 확인해주세요.")
        exit()

    # 3. 변환 실행 (train, val, test)
    # 기존 데이터 덮어쓰기 되므로 주의하세요.
    for split in ['train', 'val', 'test']:
        yolo_to_coco(ROOT_PATH, split, CLASSES)