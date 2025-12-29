from ultralytics import YOLO
from pathlib import Path
import cv2
import yaml
from pathlib import Path
import json
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors

def yaml_load(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

dataset = "CBLA"
CKPT = f"/media/ssd1/hansung/Detection/runs/detect/{dataset}_yolov9e_640_200epochs_seed1/weights/best.pt"
DATA_YAML = f"/media/ssd1/hansung/Detection/data/{dataset}/data.yaml"

IMGSZ = 640
SPLIT = "test"
DEVICE = 6
CONF = 0.001
IOU_NMS = 0.7
MAX_DET = 100

CUSTOM_COLORS = {
    0: (0, 0, 255),      # Class 0: Red (빨강)
    1: (128, 0, 255),    # Class 1: Magenta (핫핑크)
    2: (0, 80, 255),    # Class 2: Orange (주황)
    3: (255, 0, 0),    # Class 3: Yellow (노랑)
}

"""
def draw_and_save_all_predictions(metrics_save_dir: Path, data_yaml: str, split: str):
    # 1) labels 폴더와 pred_images 폴더 세팅
    labels_dir = metrics_save_dir / "labels"
    out_dir = metrics_save_dir / "pred_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) split 이미지 폴더 찾기 (data.yaml 기준)
    data = yaml_load(data_yaml)
    base = Path(data.get("path", "")).expanduser()
    split_rel = data.get(split, split)  # ex) 'images/test'
    img_dir = (base / split_rel) if base else Path(split_rel)

    # 3) 각 txt → 해당 이미지 로드 → 박스 그려 저장
    # TXT 포맷: cls cx cy w h [conf]
    img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for txt in sorted(labels_dir.glob("*.txt")):
        stem = txt.stem  # 이미지 파일명과 동일
        img_path = None
        for ext in img_exts:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            # 이미지가 없으면 건너뜀
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        H, W = im.shape[:2]

        with open(txt, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else None

            # YOLO (cx,cy,w,h) normalized → xyxy pixel
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)

            # 그리기
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls}" + (f" {conf:.2f}" if conf is not None else "")
            cv2.putText(im, label, (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"{stem}.jpg"), im)
        
"""
        
        
def draw_from_predictions_json(metrics_save_dir: Path, data_yaml: str, split: str, vis_thresh: float = 0.4):
    """
    predictions.json(원본 픽셀 좌표)로 시각화.
    - predictions.json에 entry가 없는 이미지도 input만 저장
    - vis_thresh: 시각화 전용 confidence threshold
    """
    pred_json = metrics_save_dir / "predictions.json"
    assert pred_json.exists(), f"predictions.json not found: {pred_json}"

    # data.yaml에서 split 이미지 폴더 찾기
    data = yaml_load(data_yaml)
    base = Path(data.get("path", "")).expanduser()
    split_rel = data.get(split, split)  # e.g. 'images/test'
    img_dir = (base / split_rel) if base else Path(split_rel)

    # 이미지 목록 전체 확보
    img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    all_images = {p.name: p for p in img_dir.rglob("*") if p.suffix.lower() in img_exts}

    # predictions 읽기
    with open(pred_json, "r") as f:
        preds = json.load(f)

    groups = defaultdict(list)
    for d in preds:
        score = float(d.get("score", 0.0))
        if score < vis_thresh:
            continue
        img_id = str(d.get("image_id"))
        groups[img_id].append(d)

    out_dir = metrics_save_dir / "pred_images_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = data.get("names", None)
    count_drawn = 0

    for fname, img_path in all_images.items():
        im = cv2.imread(str(img_path))
        if im is None:
            continue

        annotator = Annotator(im, line_width=4, font_size=30, pil=True,example=str(names) if names else None)

        # detections 있는 경우 그리기
        dets = groups.get(fname, []) or groups.get(Path(fname).stem, [])
        for d in dets:
            x, y, w, h = d["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cls_id = int(d.get("category_id", 0)) -1
            conf = float(d.get("score", 0.0))
            label = f"{names[cls_id] if names and 0 <= cls_id < len(names) else cls_id} {conf:.2f}"
            
            box_color = CUSTOM_COLORS.get(cls_id, colors(cls_id, bgr=True))
            
            annotator.box_label([x1, y1, x2, y2], label, color=box_color)

        out_path = out_dir / fname
        cv2.imwrite(str(out_path), annotator.result())
        count_drawn += 1

    print(f"[viz-json] {count_drawn} images saved to: {out_dir}")

def main():
    assert Path(CKPT).exists()
    assert Path(DATA_YAML).exists()

    model = YOLO(CKPT)
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        split=SPLIT,
        batch=16,
        device=DEVICE,
        conf=CONF,
        iou=IOU_NMS,
        max_det=MAX_DET,
        save_json=True,   # predictions.json
        save_txt=True,    # ← txt 예측 저장
        save_conf=True,   # ← conf까지 저장
        plots=True,       # PR/Confusion/샘플 배치 이미지
        verbose=True
    )

    # 여기서 전체 이미지 렌더
    # draw_and_save_all_predictions(Path(metrics.save_dir), DATA_YAML, SPLIT)
    # print("Saved images to:", Path(metrics.save_dir) / "pred_images")
    draw_from_predictions_json(
        metrics_save_dir=Path(metrics.save_dir),
        data_yaml=DATA_YAML,
        split=SPLIT,
        vis_thresh=0.5      # ← 여기만 손봐서 보기 좋은 임계값으로
    )

    # mAP 출력
    print("\n=== Ultralytics DetMetrics ===")
    print(f"mAP50:     {float(metrics.box.map50):.3f}")
    print(f"mAP50-95:  {float(metrics.box.map):.3f}")

    # (선택) COCOeval은 그대로 유지…

# if __name__ == "__main__":
#     main()

def draw_ground_truth_boxes(metrics_save_dir: Path, data_yaml: str, split: str):
    """
    Draws and saves images with ground truth bounding boxes.
    Assumes labels are in YOLO format (cls cx cy w h).
    """
    # 1) Set up output directory
    out_dir = metrics_save_dir / "gt_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Find split image and label directories
    data = yaml_load(data_yaml)
    base = Path(data.get("path", "")).expanduser()
    split_rel = data.get(split, split)  # e.g., 'images/test'
    img_dir = (base / split_rel) if base else Path(split_rel)
    # Get the labels path from data.yaml, assuming it's structured like images
    labels_rel = data.get("labels", "labels")
    labels_dir = (base / labels_rel / split_rel.split('/')[-1]) if base else Path(labels_rel) / split_rel.split('/')[-1]
    
    # Check for common YOLO dataset structure if labels_dir isn't specified
    if not labels_dir.exists():
        labels_dir = img_dir.parent.parent / 'labels' / img_dir.name
    
    if not labels_dir.exists():
        print(f"Warning: Could not find ground truth labels directory at {labels_dir}. Skipping GT visualization.")
        return

    # 3) Iterate through label files, find images, and draw boxes
    img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    names = data.get("names", None)

    for txt in sorted(labels_dir.glob("*.txt")):
        stem = txt.stem
        img_path = None
        for ext in img_exts:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        H, W = im.shape[:2]

        with open(txt, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])

            # YOLO (cx,cy,w,h) normalized → xyxy pixel
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)

            # Get class name if available
            label = f"{names[cls_id] if names and 0 <= cls_id < len(names) else cls_id}"
            color = colors(cls_id, bgr=True)

            # Use Annotator for consistent drawing
            annotator = Annotator(im, line_width=1, example=str(names) if names else None)
            annotator.box_label([x1, y1, x2, y2], label, color=color)
            im = annotator.result()

        cv2.imwrite(str(out_dir / f"{stem}.jpg"), im)
    
    print(f"Saved {len(list(out_dir.glob('*.jpg')))} ground truth images to: {out_dir}")



def main():
    assert Path(CKPT).exists()
    assert Path(DATA_YAML).exists()

    model = YOLO(CKPT)
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        split=SPLIT,
        batch=1,
        device=DEVICE,
        conf=CONF,
        iou=IOU_NMS,
        max_det=MAX_DET,
        save_json=True,
        save_txt=True,
        save_conf=True,
        plots=True,
        verbose=True
    )

    # Visualize predictions
    draw_from_predictions_json(
        metrics_save_dir=Path(metrics.save_dir),
        data_yaml=DATA_YAML,
        split=SPLIT,
        vis_thresh=0.5
    )

    # Visualize ground truths
    draw_ground_truth_boxes(
        metrics_save_dir=Path(metrics.save_dir),
        data_yaml=DATA_YAML,
        split=SPLIT
    )

    # Print mAP
    print("\n=== Ultralytics DetMetrics ===")
    print(f"mAP50:     {float(metrics.box.map50):.3f}")
    print(f"mAP50-95:  {float(metrics.box.map):.3f}")

if __name__ == "__main__":
    main()