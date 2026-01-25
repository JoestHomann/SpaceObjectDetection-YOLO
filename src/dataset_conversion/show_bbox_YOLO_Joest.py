from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

# -------- Paths (as you defined) --------
dataset_path = Path("D:/SpaceObjectDetection-YOLO/data/spark-2022-stream-1")
labels_path = dataset_path / "labels"

train_img_path = dataset_path / "train" / "train"
val_img_path   = dataset_path / "val" / "val"

output_path       = dataset_path / "labels"
output_train_path = output_path / "train"
output_val_path   = output_path / "val"

# -------- Config --------
SPLIT = "val"          # "train" or "val"
NUM_SAMPLES = 10       # how many images to show
SEED = None              # set None for random
SHOW_EMPTY_LABELS = False  # show images even if label file missing/empty

# -------- Helpers --------
def clamp_int(v, lo, hi):
    return max(lo, min(hi, v))

def yolo_norm_to_xyxy(xc, yc, bw, bh, w, h):
    # YOLO normalized (0..1) -> pixel xyxy
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h

    # round to int pixels
    x1 = int(round(x1)); y1 = int(round(y1))
    x2 = int(round(x2)); y2 = int(round(y2))

    # clip to image bounds
    x1 = clamp_int(x1, 0, w - 1)
    y1 = clamp_int(y1, 0, h - 1)
    x2 = clamp_int(x2, 0, w - 1)
    y2 = clamp_int(y2, 0, h - 1)

    # ensure proper ordering
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    return x1, y1, x2, y2

def read_yolo_label_file(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return boxes
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc  = float(parts[1])
        yc  = float(parts[2])
        bw  = float(parts[3])
        bh  = float(parts[4])
        boxes.append((cls, xc, yc, bw, bh))
    return boxes

def list_jpgs(img_dir: Path):
    return sorted([p for p in img_dir.glob("*.jpg") if p.is_file()])

# -------- Main --------
if SEED is not None:
    random.seed(SEED)

if SPLIT == "train":
    img_dir = train_img_path
    lbl_dir = output_train_path
elif SPLIT == "val":
    img_dir = val_img_path
    lbl_dir = output_val_path
else:
    raise ValueError("SPLIT must be 'train' or 'val'")

images = list_jpgs(img_dir)
if not images:
    raise SystemExit(f"No .jpg images found in: {img_dir}")

samples = random.sample(images, min(NUM_SAMPLES, len(images)))

for img_path in samples:
    label_path = lbl_dir / (img_path.stem + ".txt")
    boxes = read_yolo_label_file(label_path)

    if (not boxes) and (not SHOW_EMPTY_LABELS):
        continue

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("Failed to read:", img_path)
        continue

    h, w = img_bgr.shape[:2]

    # Draw boxes
    for (cls, xc, yc, bw, bh) in boxes:
        x1, y1, x2, y2 = yolo_norm_to_xyxy(xc, yc, bw, bh, w, h)

        # Skip degenerate boxes
        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            continue

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img_bgr,
            f"class {cls}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Show
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    title = f"{SPLIT}: {img_path.name} | labels: {label_path.name if label_path.exists() else 'MISSING'} | boxes: {len(boxes)}"
    plt.title(title)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

print("Done.")
