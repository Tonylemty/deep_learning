# kaggle_competition_ready.py
# -----------------------------
# 這份程式碼可以直接貼到 Kaggle Notebook 執行（不用 argparse）
# 功能：
#   - 自動偵測競賽資料夾與 train/test CSV、圖像資料
#   - K-Fold 訓練 (預設 5-Fold)
#   - 可選擇 Mixup 與 Test Time Augmentation (TTA)
#   - 儲存每個 Fold 的最佳模型
#   - 進行推論並輸出 submission CSV
# -----------------------------

import os
from pathlib import Path
import random
import copy
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -------------------- Config (可修改) --------------------
EPOCHS = 8           # 訓練輪數，小規模測試可降低，正式可提高
BATCH = 64           # 批次大小
IMG_SIZE = 128       # 圖像尺寸
KFOLD = 5            # K-Fold 數量
LR = 2e-4            # 學習率
NUM_CLASSES = 88     # 分類數量（88個琴鍵）
USE_MIXUP = False    # 是否使用 Mixup 資料增強
USE_TTA = True       # 是否使用 Test Time Augmentation
SEED = 42            # 隨機種子
SAVE_MODELS_DIR = "/kaggle/working/models"  # 模型儲存目錄
OUTPUT_CSV = "/kaggle/working/output.csv"   # 最終 submission CSV
# --------------------------------------------------------

# -------------------- 固定隨機種子 --------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 選擇運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------- 自動偵測 Competition 與 Datasets --------------------
KAGGLE_INPUT = Path("/kaggle/input")
if not KAGGLE_INPUT.exists():
    raise RuntimeError("/kaggle/input not found — 請在 Kaggle Notebook 上執行此程式。")

# 找出 competition 資料夾 (包含 train_truth.csv & sample_truth.csv)
comp_dirs = []
for child in KAGGLE_INPUT.iterdir():
    if child.is_dir():
        files = [p.name.lower() for p in child.iterdir() if p.is_file()]
        if any(n in files for n in ("train_truth.csv", "train.csv")) and any(n in files for n in ("sample_truth.csv", "sample_submission.csv")):
            comp_dirs.append(child)

# 如果沒找到，fallback 至第一個資料夾
if len(comp_dirs) == 0:
    for child in KAGGLE_INPUT.iterdir():
        if child.is_dir():
            comp_dirs.append(child)
            break

if len(comp_dirs) == 0:
    raise RuntimeError("找不到 /kaggle/input 下的任何資料夾，請確認競賽已掛載。")

COMP_DIR = comp_dirs[0]
print("Detected competition dir:", COMP_DIR)

# CSV 檔案候選名稱
train_csv_candidates = ["train_truth.csv", "train.csv"]
sample_csv_candidates = ["sample_truth.csv", "sample_submission.csv"]

# 找到第一個符合的 CSV
def find_first_file(base_dir, candidates):
    for c in candidates:
        p = base_dir / c
        if p.exists():
            return str(p)
    # 遞迴搜尋
    for p in base_dir.rglob("*"):
        if p.is_file() and p.name.lower() in [x.lower() for x in candidates]:
            return str(p)
    return None

train_csv = find_first_file(COMP_DIR, train_csv_candidates)
sample_csv = find_first_file(COMP_DIR, sample_csv_candidates)

# 找圖像資料夾
def find_dataset_dir(name_keywords=("music-train","music_train")):
    for p in KAGGLE_INPUT.iterdir():
        if p.is_dir() and any(k in p.name.lower() for k in name_keywords):
            return str(p)
    for p in COMP_DIR.iterdir():
        if p.is_dir() and any(k in p.name.lower() for k in name_keywords):
            return str(p)
    return None

train_dir = find_dataset_dir(("music-train","music_train"))
test_dir  = find_dataset_dir(("music-test","music_test"))

# fallback：找資料夾內有多張圖像的資料夾
def find_image_folder(base_dir):
    for p in base_dir.iterdir():
        if p.is_dir():
            imgs = list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
            if len(imgs) >= 5:
                return str(p)
    return None

if train_dir is None:
    train_dir = find_image_folder(COMP_DIR)
if test_dir is None:
    test_dir = find_image_folder(COMP_DIR)

print("train_csv:", train_csv)
print("sample_csv:", sample_csv)
print("train_dir:", train_dir)
print("test_dir:", test_dir)

if train_csv is None:
    raise RuntimeError("找不到 train CSV，請確認競賽介面有上傳。")
if sample_csv is None or test_dir is None:
    print("[WARNING] 找不到 sample CSV 或 test_dir，預測階段會被跳過。")

# -------------------- Dataset class --------------------
class MelCepDataset(Dataset):
    """
    自定義 Dataset，用於 Mel-Cepstrum 圖像分類
    """
    def __init__(self, filenames, labels, root_dir, transform=None, is_test=False):
        self.filenames = list(filenames)
        self.labels = None if labels is None else list(labels)
        self.root = Path(root_dir) if root_dir is not None else None
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        # 判斷路徑
        if self.root is not None:
            candidate = self.root / fname
            if candidate.exists():
                p = candidate
            else:
                candidate2 = self.root / Path(fname).name
                if candidate2.exists():
                    p = candidate2
                else:
                    p = Path(fname)
                    if not p.exists():
                        raise FileNotFoundError(f"Cannot find image file for {fname} under {self.root}")
        else:
            p = Path(fname)
            if not p.exists():
                raise FileNotFoundError(f"File {fname} not found and no root_dir provided.")

        # 讀圖並轉 RGB
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.is_test or self.labels is None:
            return img, Path(fname).name
        else:
            label = int(self.labels[idx])
            return img, label

# -------------------- Transforms --------------------
def get_train_transforms(img_size):
    """
    訓練資料增強
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def get_val_transforms(img_size):
    """
    驗證 / 測試資料處理
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def get_tta_transforms(img_size):
    """
    測試時增強 (TTA)
    """
    tta = []
    tta.append(get_val_transforms(img_size))  # 原圖
    tta.append(transforms.Compose([            # 水平翻轉
        transforms.Resize((img_size,img_size)), 
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]))
    return tta

# -------------------- Model helper --------------------
def build_model(num_classes=NUM_CLASSES, pretrained=False):
    """
    建立 ResNet18 模型並修改輸出層
    """
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # 輸出對應琴鍵數量
    return model

# -------------------- Mixup helpers --------------------
def mixup_data(x, y, alpha=0.4):
    """
    將 batch 資料做 Mixup
    """
    if alpha <= 0:
        return x, (y, y, 1.0)
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b, lam)

def mixup_criterion(criterion, preds, y_a_y_b_lam):
    """
    Mixup 損失計算
    """
    y_a, y_b, lam = y_a_y_b_lam
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

# -------------------- 讀取 train csv --------------------
df = pd.read_csv(train_csv)
image_col = df.columns[0]
label_col = df.columns[1] if df.shape[1] > 1 else None
if label_col is None:
    raise RuntimeError("train csv 沒有 label column。")
images = df[image_col].astype(str).tolist()
labels = df[label_col].astype(int).tolist()
print(f"Loaded {len(images)} train samples.")

# -------------------- K-Fold split --------------------
indices = list(range(len(images)))
try:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    folds = list(skf.split(indices, labels))
except Exception:
    # fallback：簡單分割
    print("[WARN] sklearn StratifiedKFold not available; using simple splits.")
    perm = np.random.RandomState(SEED).permutation(len(indices))
    folds = []
    fold_size = len(indices) // KFOLD
    for k in range(KFOLD):
        v = perm[k*fold_size:(k+1)*fold_size].tolist()
        t = [i for i in indices if i not in v]
        folds.append((t, v))

# -------------------- 訓練 Loop --------------------
train_tf = get_train_transforms(IMG_SIZE)
val_tf = get_val_transforms(IMG_SIZE)
tta_tfs = get_tta_transforms(IMG_SIZE)
os.makedirs(SAVE_MODELS_DIR, exist_ok=True)
fold_model_paths = []
num_workers = 2 if torch.cuda.is_available() else 0

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"=== Fold {fold_idx} ===")
    train_imgs = [images[i] for i in train_idx]
    train_labs = [labels[i] for i in train_idx]
    val_imgs = [images[i] for i in val_idx]
    val_labs = [labels[i] for i in val_idx]

    # 建立 Dataset 與 DataLoader
    ds_train = MelCepDataset(train_imgs, train_labs, train_dir, transform=train_tf, is_test=False)
    ds_val   = MelCepDataset(val_imgs, val_labs, train_dir, transform=val_tf, is_test=False)
    loader_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    loader_val   = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # 建立模型
    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience = 0
    early_stop_patience = 6

    # -------------------- 每 Epoch --------------------
    for epoch in range(1, EPOCHS+1):
        # 訓練模式
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader_train:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            if USE_MIXUP:
                xb_m, yabl = mixup_data(xb, yb, alpha=0.4)
                out = model(xb_m)
                loss = mixup_criterion(criterion, out, yabl)
            else:
                out = model(xb)
                loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            preds = out.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        train_acc = correct / max(1, total)
        train_loss = running_loss / max(1, total)

        # 驗證模式
        model.eval()
        v_correct = 0
        v_total = 0
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                preds = out.argmax(dim=1)
                v_correct += (preds == yb).sum().item()
                v_total += xb.size(0)
                v_loss += loss.item() * xb.size(0)
        val_acc = v_correct / max(1, v_total)
        val_loss = v_loss / max(1, v_total)

        print(f"Fold{fold_idx} Epoch{epoch}/{EPOCHS} TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f} TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f}")

        # Early stopping 檢查
        if val_acc > best_val_acc + 1e-12:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= early_stop_patience:
            print(f"Fold {fold_idx} early stopping (patience={patience})")
            break

    # 保存最佳模型
    model.load_state_dict(best_weights)
    fold_path = Path(SAVE_MODELS_DIR) / f"fold_{fold_idx}.pth"
    torch.save(model.state_dict(), str(fold_path))
    fold_model_paths.append(str(fold_path))
    print("Saved model:", fold_path)

# -------------------- 推論 / Inference --------------------
if sample_csv is None or test_dir is None:
    print("[WARNING] sample csv or test dir missing -> skip inference. Trained models saved to", SAVE_MODELS_DIR)
else:
    print("Starting inference / ensemble with", len(fold_model_paths), "models")
    sample_df = pd.read_csv(sample_csv)
    test_images = sample_df.iloc[:,0].astype(str).tolist()
    ntest = len(test_images)
    all_probs = np.zeros((ntest, NUM_CLASSES), dtype=np.float32)

    for mp in fold_model_paths:
        print("Load", mp)
        state = torch.load(mp, map_location=device)
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        fold_probs = np.zeros((ntest, NUM_CLASSES), dtype=np.float32)
        tta_list = get_tta_transforms(IMG_SIZE) if USE_TTA else [get_val_transforms(IMG_SIZE)]

        for tta_idx, tf in enumerate(tta_list):
            ds_test = MelCepDataset(test_images, None, test_dir, transform=tf, is_test=True)
            loader_test = DataLoader(ds_test, batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
            ptr = 0
            with torch.no_grad():
                for xb, fnames in tqdm(loader_test, desc=f"TTA{tta_idx}"):
                    xb = xb.to(device)
                    out = model(xb)
                    p = torch.softmax(out, dim=1).cpu().numpy()
                    bs = p.shape[0]
                    fold_probs[ptr:ptr+bs] += p
                    ptr += bs
            fold_probs /= len(tta_list)  # 平均 TTA

        all_probs += fold_probs

    all_probs /= max(1, len(fold_model_paths))
    preds = all_probs.argmax(axis=1)

    # 輸出 CSV
    sample_cols = list(pd.read_csv(sample_csv, nrows=1).columns)
    if len(sample_cols) >= 2:
        out_df = pd.DataFrame({sample_cols[0]: test_images, sample_cols[1]: preds.astype(int)})
    else:
        out_df = pd.DataFrame({"image": test_images, "label": preds.astype(int)})

    out_df.to_csv(OUTPUT_CSV, index=False)
    print("Saved submission to", OUTPUT_CSV)

print("All done.")
