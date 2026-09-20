import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression


# =========================
# Config (CFG)
# -------------------------
# 把所有重要超參數集中管理，方便調整與追蹤版本差異
# =========================
CFG = dict(
    # ---- Cross Validation (CV) 設定 ----
    # 使用多個 seed 產生不同切分 -> ensemble 會更穩
    seeds=[42, 202, 777],
    folds=5,

    # 用 r = sqrt(x1^2 + x2^2) 做分箱後 stratified split
    # 目的是讓每個 fold 的半徑分布接近，CV 更可靠
    r_bins=25,

    # ---- Neural Network 訓練參數 ----
    epochs=320,
    batch_size=128,
    lr=5e-4,
    weight_decay=1e-4,     # AdamW 的 L2 正則
    patience=45,           # early stopping：多少 epoch 沒變好就停
    report_every=25,       # 每幾 epoch 印一次 log

    # ---- TTA (Test-Time Augmentation) 設定 ----
    # 做法：對 input 加一點點 noise，多跑幾次平均
    # 目的是減少 NN 預測的抖動，提高穩定性
    tta=5,
    tta_noise=0.0010,
    mix_tta=True,          # mix_tta=True 代表會融合 noise=0 + noise=tta_noise 的平均

    # ---- NN ensemble 權重設定 ----
    # 用 validation 的最佳 raw_mse 來決定每個模型的權重（表現好 -> 權重大）
    weight_by_val=True,
    weight_power=2.0,

    # ---- Ridge (v1 + v2) 設定 ----
    ridge_alphas=[0.2, 0.35, 0.6, 0.9, 1.3, 2.0, 3.0],

    # Ridge ensemble 權重：依 OOF MSE 反比加權（表現好 -> 權重大）
    ridge_weight_by_oof=True,
    ridge_weight_power=2.0,

    # ---- Stage1 stacking 設定 ----
    # positive=True -> 限制 stacking 的係數必須 >= 0
    # 這樣比較像「加權平均」，通常更穩不容易反向抵銷
    stacking_positive=True,

    # ---- Stage2 poly stacking 設定 ----
    stage2_poly_degree=2,
    stage2_ridge_alphas=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],

    # ---- Output 檔名 ----
    out_path="submission.csv",
)

# NN 架構種類：會各自訓練，最後一起加權 ensemble
ARCHS = ("base", "wide", "deep")


# ---------------------------
# Utils - set_seed
# ---------------------------
# 固定隨機種子，讓結果比較可重現
# 注意：torch 的 deterministic 會讓訓練變慢一點，但比較穩定
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Utils - find_file_in_kaggle_input
# ---------------------------
# Kaggle 有時候 input 資料會在 /kaggle/input/某個資料夾底下
# 這個函式會：
# 1) 先看當前目錄有沒有檔案
# 2) 再去 /kaggle/input 下面找同名檔案
# 3) 找不到就回傳 None
# ---------------------------
def find_file_in_kaggle_input(filename: str):
    if os.path.exists(filename):
        return filename
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    # 第一層資料夾找
    for child in root.iterdir():
        p = child / filename
        if p.exists():
            return str(p)
    # 遞迴找（較慢但保險）
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == filename.lower():
            return str(p)
    return None


# ---------------------------
# Utils - safe_submission_id
# ---------------------------
# Kaggle submission 最常見的坑之一：id 不唯一或格式錯
# 這個函式邏輯：
# 1) 優先使用 sample_submission.csv 的 id（通常正確）
# 2) sample 不行才用 test.csv 的 id（但 test 有時候 id 會重複）
# 3) 都不行就 fallback: 1..N
# ---------------------------
def safe_submission_id(sample_df, test_df):
    # prefer sample_submission ids if valid & unique
    if sample_df is not None and "id" in sample_df.columns and len(sample_df) == len(test_df):
        ids = sample_df["id"].values
        if pd.Series(ids).duplicated().sum() == 0:
            return ids
    # else use test ids only if unique
    if "id" in test_df.columns:
        ids = test_df["id"].values
        if pd.Series(ids).duplicated().sum() == 0:
            return ids
    # fallback: 1..N
    return np.arange(1, len(test_df) + 1, dtype=np.int64)


# =========================================================
# Feature Engineering
# ---------------------------------------------------------
# 你設計了兩套特徵：
# - feat_v1：偏 v8 style，較多高次項 + sin/cos + r 的 Fourier
# - feat_v2：更強化週期/半徑基底（exp(-c*r^2) 等），提供不同的視角
# =========================================================

def feat_v1(df: pd.DataFrame) -> np.ndarray:
    """
    v1 特徵：偏多項式 + trig + radial + 一些混合項
    用途：
    - 給 Ridge：讓線性模型能擬合非線性
    - 給 NN：因為 v1 特徵通常比較穩，NN 用起來容易收斂
    """
    x1 = df["x1"].astype(np.float32).values
    x2 = df["x2"].astype(np.float32).values

    r = np.sqrt(x1**2 + x2**2).astype(np.float32)
    theta = np.arctan2(x2, x1).astype(np.float32)
    eps = 1e-6  # 避免除以 0

    feats = [
        # 原始輸入
        x1, x2,

        # 交互項
        x1*x2,

        # 多項式項（到 5 次）
        x1**2, x2**2,
        x1**3, x2**3,
        x1**4, x2**4,
        x1**5, x2**5,

        # trig：描述週期性
        np.sin(x1), np.cos(x1),
        np.sin(x2), np.cos(x2),

        # tanh：提供平滑非線性變換
        np.tanh(x1), np.tanh(x2),

        # 高斯型：exp(-x^2) 常用來表達局部形狀
        np.exp(-x1**2), np.exp(-x2**2),

        # radial (r) 與其多項式
        r, r**2, r**3, r**4, r**5,

        # theta 的 sin/cos：提供角度資訊（極座標）
        np.sin(theta), np.cos(theta),

        # 一些混合多項式
        x1**2 * x2,
        x1 * x2**2,

        # 有理式（用 eps 防爆）
        x1/(x2+eps),
        x2/(x1+eps),

        # (x1+x2) 的 trig：提供斜方向週期資訊
        np.sin(x1+x2),
        np.cos(x1+x2),
    ]

    # radial Fourier：sin(k*pi*r), cos(k*pi*r)
    # k 越大表示越高頻，可能更貼合複雜形狀，但也可能更容易過擬合
    for k in range(1, 7):
        feats.append(np.sin(k * np.pi * r))
        feats.append(np.cos(k * np.pi * r))

    return np.vstack(feats).T.astype(np.float32)


def feat_v2(df: pd.DataFrame) -> np.ndarray:
    """
    v2 特徵：加入更多週期/角度 harmonics + radial basis
    用途：
    - 特別適合搭配 Ridge，因為它提供「另一種」特徵空間
    - 跟 v1 的 Ridge 形成多樣性，讓 stacking 有更大機會提升泛化
    """
    x1 = df["x1"].astype(np.float32).values
    x2 = df["x2"].astype(np.float32).values

    r = np.sqrt(x1**2 + x2**2).astype(np.float32)
    theta = np.arctan2(x2, x1).astype(np.float32)
    eps = 1e-6

    feats = [
        # 原始輸入 + 絕對值（有時能幫助模型抓到對稱性）
        x1, x2,
        np.abs(x1), np.abs(x2),

        # 交互項
        x1*x2,

        # 多項式（比 v1 輕一點，避免太爆）
        x1**2, x2**2,
        x1**3, x2**3,
        x1**4, x2**4,

        # radial + theta
        r, r**2, r**3, r**4,
        theta,
        np.sin(theta), np.cos(theta),
    ]

    # theta harmonics：sin(k*theta), cos(k*theta)
    # 用來描述角度方向的週期性結構
    for k in range(2, 9):
        feats.append(np.sin(k * theta))
        feats.append(np.cos(k * theta))

    # axis periodic：sin(k*pi*x1/x2), cos(k*pi*x1/x2)
    # 用來捕捉沿 x1 或 x2 軸方向的週期變化
    for k in range(1, 9):
        feats.append(np.sin(k * np.pi * x1))
        feats.append(np.cos(k * np.pi * x1))
        feats.append(np.sin(k * np.pi * x2))
        feats.append(np.cos(k * np.pi * x2))

    # radial Fourier：比 v1 更密（1..9）
    for k in range(1, 10):
        feats.append(np.sin(k * np.pi * r))
        feats.append(np.cos(k * np.pi * r))

    # radial basis：exp(-c*r^2)
    # 這種特徵常用在 RBF/核方法，能描述「距離中心越遠影響越小」的形狀
    for c in [0.5, 1.0, 2.0, 4.0, 8.0]:
        feats.append(np.exp(-c * (r**2)))

    # 一些保守的混合 trig/ratio
    feats += [
        x1/(x2+eps),
        x2/(x1+eps),
        np.sin(x1+x2), np.cos(x1+x2),
        np.sin(x1-x2), np.cos(x1-x2),
        np.sin(x1)*np.sin(x2),
        np.cos(x1)*np.cos(x2),
    ]

    return np.vstack(feats).T.astype(np.float32)


# =========================================================
# Neural Network: Residual MLP
# ---------------------------------------------------------
# 你使用 Residual block 的原因：
# - 訓練比較穩定
# - 讓模型學「修正量」而不是整個 mapping
# 同時搭配 BatchNorm + Dropout 增強穩定性/減少過擬合
# =========================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.12):
        super().__init__()
        # 先升維到 hidden，再降回 dim，形成 residual
        self.fc1 = nn.Linear(dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Residual block: x + F(x)
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.bn2(h)
        return x + h


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], n_blocks: int, dropout: float):
        super().__init__()

        # head：把 input_dim -> hidden_dims[0]
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # blocks：多個 residual block 堆疊
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dims[0], max(32, hidden_dims[0]//2), dropout=dropout)
            for _ in range(n_blocks)
        ])

        # tail：逐步降維到 1（回歸輸出）
        tail = []
        d = hidden_dims[0]
        for h in hidden_dims[1:]:
            tail += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        tail += [nn.Linear(d, 1)]
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        h = self.head(x)
        h = self.blocks(h)
        return self.tail(h)


def build_model(input_dim: int, arch: str) -> nn.Module:
    """
    根據 arch 產生不同容量的 MLP
    - base：較小、比較不容易 overfit
    - wide：更寬、表達力更強
    - deep：更深、表達力更強
    """
    if arch == "base":
        return ResidualMLP(input_dim, [256, 128], n_blocks=3, dropout=0.12)
    if arch == "wide":
        return ResidualMLP(input_dim, [512, 256], n_blocks=4, dropout=0.15)
    if arch == "deep":
        return ResidualMLP(input_dim, [512, 256, 128], n_blocks=4, dropout=0.15)
    raise ValueError(arch)


# ---------------------------------------------------------
# Loss: Hybrid(MSE + SmoothL1)
# ---------------------------------------------------------
# Kaggle 評分用 MSE
# 但訓練資料有 noise 時，SmoothL1 對 outlier 比較不敏感
# 混合可以讓訓練比較穩
# ---------------------------------------------------------
mse_none = nn.MSELoss(reduction="none")
s1_none  = nn.SmoothL1Loss(reduction="none")

def hybrid_loss_per_sample(pred, target):
    return 0.6 * mse_none(pred, target) + 0.4 * s1_none(pred, target)


@torch.no_grad()
def raw_mse_eval(model, X_tensor, y_scaled_tensor, device, y_mean, y_std, batch_size=1024):
    """
    計算「raw space」的 MSE（把標準化後的 y 還原）
    因為你訓練時 y 是標準化的，但評估想要對齊 Kaggle 的 MSE（原始尺度）
    """
    model.eval()
    loader = DataLoader(TensorDataset(X_tensor, y_scaled_tensor), batch_size=batch_size, shuffle=False)
    se, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze()
        ps = model(xb).squeeze()

        # 還原到原始 y 尺度
        pred = ps * y_std + y_mean
        yraw = yb * y_std + y_mean

        se += ((pred - yraw)**2).sum().item()
        n += xb.size(0)
    return se / max(1, n)


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    """
    單個 epoch 訓練
    - 支援 AMP (mixed precision)
    - 加上 gradient clipping 防止梯度爆炸
    """
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze()
        optimizer.zero_grad(set_to_none=True)

        # GPU 時可用 AMP 加速
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(xb).squeeze()
                loss = hybrid_loss_per_sample(pred, yb).mean()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(xb).squeeze()
            loss = hybrid_loss_per_sample(pred, yb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)


@torch.no_grad()
def predict_model(model, X_tensor, device, batch_size=1024, tta=1, noise_scale=0.0):
    """
    NN 推論（支援 TTA）
    - tta 次數：多跑幾次平均
    - noise_scale：對 input 加上小 noise，讓預測更平滑穩定
    """
    model.eval()
    out = None
    for _ in range(tta):
        preds = []
        loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
        for (xb,) in loader:
            xb = xb.to(device)

            # TTA：加微小 noise
            if noise_scale > 0:
                xb = xb + noise_scale * torch.randn_like(xb)

            p = model(xb).squeeze().detach().cpu().numpy()
            preds.append(np.atleast_1d(p))

        arr = np.concatenate(preds, axis=0)
        out = arr if out is None else (out + arr)

    return out / float(tta)


def train_fold(
    X_tr, y_tr,
    X_val, y_val,
    arch, device, seed, fold_idx,
    y_mean, y_std,
    epochs, patience, lr, weight_decay, batch_size, report_every
):
    """
    訓練單一 fold 的單一 arch
    - 以 raw_mse_eval 作為 early stopping 指標（對齊 Kaggle）
    - ReduceLROnPlateau：val 沒變好就降 lr
    """
    set_seed(seed)
    model = build_model(X_tr.shape[1], arch).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    # GPU 用 AMP
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_raw = float("inf")
    best_state = None
    no_imp = 0  # 連續幾次沒進步

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_raw = raw_mse_eval(model, X_val_t, y_val_t, device, y_mean, y_std, batch_size=batch_size)
        scheduler.step(val_raw)

        # early stopping：更新最佳權重
        if val_raw < best_raw - 1e-10:
            best_raw = val_raw
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        # 印 log
        if epoch == 1 or epoch % report_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[Seed {seed} Fold {fold_idx} {arch}] ep {epoch}/{epochs} | train {tr_loss:.6f} | val_raw_mse {val_raw:.6f} | lr {lr_now:.2e} | best {best_raw:.6f}")

        # 沒進步太久就停
        if no_imp >= patience:
            break

    # 回復最佳權重
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_raw)


# ---------------------------
# CV-safe helper (這裡你雖然有寫，但 v9 主流程實際沒有用到)
# ---------------------------
def cv_oof_and_test_predict(build_and_fit_fn, X, y, X_test, splits):
    """
    CV-safe：每個 fold 只用 train fold fit，然後預測 val fold 得到 OOF
    最後再用全部資料 fit 一次預測 test
    """
    oof = np.zeros(len(y), dtype=np.float64)
    for tr_idx, va_idx in splits:
        m = build_and_fit_fn()
        m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X[va_idx])
    m_full = build_and_fit_fn()
    m_full.fit(X, y)
    test_pred = m_full.predict(X_test)
    return oof, test_pred


# =========================
# Main
# =========================
start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- 讀取 Kaggle 檔案 ----
train_path = find_file_in_kaggle_input("train.csv")
test_path  = find_file_in_kaggle_input("test.csv")
sample_path = find_file_in_kaggle_input("sample_submission.csv")
if train_path is None or test_path is None:
    raise FileNotFoundError("找不到 train.csv 或 test.csv")

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path) if sample_path is not None else None

# ---- y 標準化：NN 訓練更穩 ----
y = train_df["y"].values.astype(np.float32)
y_mean = float(y.mean())
y_std  = float(y.std() + 1e-8)  # +1e-8 避免 std=0
y_s = ((y - y_mean) / y_std).astype(np.float32)

# ---- stratified bins by radius ----
# 用 r 做分箱，目的是 stratified kfold 讓每 fold 半徑分布相近
x1 = train_df["x1"].astype(np.float32).values
x2 = train_df["x2"].astype(np.float32).values
r = np.sqrt(x1*x1 + x2*x2).astype(np.float32)

# qcut：把 r 分成 CFG["r_bins"] 份（分位數分箱）
bins = pd.qcut(pd.Series(r), q=CFG["r_bins"], labels=False, duplicates="drop").to_numpy()

# 每個 seed 都做一組 splits：讓 NN 有更多切分多樣性
splits_by_seed = {
    seed: list(StratifiedKFold(n_splits=CFG["folds"], shuffle=True, random_state=seed).split(np.zeros(len(y)), bins))
    for seed in CFG["seeds"]
}

# Ridge 與 stage2 用固定 seed=42 的 splits（比較簡潔一致）
base_splits = list(StratifiedKFold(n_splits=CFG["folds"], shuffle=True, random_state=42).split(np.zeros(len(y)), bins))


# =========================================================
# Feature sets：v1 / v2 + StandardScaler
# ---------------------------------------------------------
# 注意：scaler 一定要 fit 在 train，然後 transform train/test
# 不要 fit 到 test（避免資料洩漏）
# =========================================================
X1_train = feat_v1(train_df)
X1_test  = feat_v1(test_df)
sc1 = StandardScaler().fit(X1_train)
X1_train_s = sc1.transform(X1_train).astype(np.float32)
X1_test_s  = sc1.transform(X1_test).astype(np.float32)

X2_train = feat_v2(train_df)
X2_test  = feat_v2(test_df)
sc2 = StandardScaler().fit(X2_train)
X2_train_s = sc2.transform(X2_train).astype(np.float32)
X2_test_s  = sc2.transform(X2_test).astype(np.float32)

# dummy init（只是避免某些 lint/型別檢查）
X_test_t = torch.FloatTensor(X1_train_s[:1])


# =========================================================
# Base model #1：NN ensemble（使用 v1 features）
# ---------------------------------------------------------
# 最後會產生：
# - oof_nn：訓練集的 OOF 預測（拿來 stacking）
# - test_nn：測試集預測（拿來最終輸出）
# =========================================================
X_nn_train = X1_train_s
X_nn_test  = X1_test_s
X_nn_test_t = torch.FloatTensor(X_nn_test)

# oof_nn：累積加權預測
oof_nn = np.zeros(len(train_df), dtype=np.float64)
oof_nn_wsum = np.zeros(len(train_df), dtype=np.float64)

# test 端的加權累積
test_nn_acc = np.zeros(len(test_df), dtype=np.float64)
total_nn_w = 0.0

for seed in CFG["seeds"]:
    print(f"\n===== NN Seed {seed} =====")

    # 每個 seed 有自己的 splits
    for fold_idx, (tr_idx, va_idx) in enumerate(splits_by_seed[seed], start=1):
        X_tr, X_va = X_nn_train[tr_idx], X_nn_train[va_idx]
        y_tr, y_va = y_s[tr_idx], y_s[va_idx]
        X_va_t = torch.FloatTensor(X_va)

        # 每個 fold 內，訓練多種 arch
        for arch in ARCHS:
            model, best_val_raw = train_fold(
                X_tr=X_tr, y_tr=y_tr,
                X_val=X_va, y_val=y_va,
                arch=arch, device=device, seed=seed, fold_idx=fold_idx,
                y_mean=y_mean, y_std=y_std,
                epochs=CFG["epochs"], patience=CFG["patience"],
                lr=CFG["lr"], weight_decay=CFG["weight_decay"],
                batch_size=CFG["batch_size"], report_every=CFG["report_every"],
            )

            # --- 權重：val 表現越好，權重越大 ---
            w = 1.0
            if CFG["weight_by_val"]:
                w = (1.0 / max(best_val_raw, 1e-12)) ** float(CFG["weight_power"])

            # --- val pred（TTA） ---
            # mix_tta=True：平均 (noise=0) 與 (noise=tta_noise)
            if CFG["mix_tta"]:
                p0 = predict_model(model, X_va_t, device, batch_size=CFG["batch_size"], tta=CFG["tta"], noise_scale=0.0)
                p1 = predict_model(model, X_va_t, device, batch_size=CFG["batch_size"], tta=CFG["tta"], noise_scale=CFG["tta_noise"])
                p_va_s = 0.5 * (p0 + p1)
            else:
                p_va_s = predict_model(model, X_va_t, device, batch_size=CFG["batch_size"], tta=CFG["tta"], noise_scale=CFG["tta_noise"])

            # 還原 y 尺度
            p_va = p_va_s * y_std + y_mean

            # 累積 OOF
            oof_nn[va_idx] += w * p_va
            oof_nn_wsum[va_idx] += w

            # --- test pred（TTA） ---
            if CFG["mix_tta"]:
                t0 = predict_model(model, X_nn_test_t, device, batch_size=CFG["batch_size"], tta=CFG["tta"], noise_scale=0.0)
                t1 = predict_model(model, X_nn_test_t, device, batch_size=CFG["batch_size"], tta=CFG["tta"], noise_scale=CFG["tta_noise"])
                p_te_s = 0.5 * (t0 + t1)
            else:
                p_te_s = predict_model(model, X_nn_test_t, device, batch_size=CFG["batch_size"], tta=CFG["tta"], noise_scale=CFG["tta_noise"])

            p_te = p_te_s * y_std + y_mean

            # test 端也依權重累積
            test_nn_acc += w * p_te
            total_nn_w += w

            print(f"[NN Seed {seed} Fold {fold_idx} {arch}] best_val_raw_mse = {best_val_raw:.6f}")

# 對 OOF 做權重平均（每筆資料自己的 wsum）
oof_nn = oof_nn / np.maximum(1e-12, oof_nn_wsum)
test_nn = test_nn_acc / max(1e-12, total_nn_w)

print("\n[OOF] NN raw MSE:", mean_squared_error(y, oof_nn))


# =========================================================
# Base model #2：Ridge ensemble（feature set v1 / v2）
# ---------------------------------------------------------
# 做法：
# - 不同 alpha 各自跑 CV，得到 oof_a 與 te_a
# - 根據每個 alpha 的 OOF MSE 做加權平均（表現越好權重越大）
# =========================================================
def ridge_ensemble_oof_test(Xtr, Xte, splits, alphas, weight_by_oof=True, power=2.0):
    oof_list, te_list, mses = [], [], []

    for a in alphas:
        oof_a = np.zeros(len(y), dtype=np.float64)
        te_a  = np.zeros(len(Xte), dtype=np.float64)

        # fold-by-fold 訓練與預測（CV-safe）
        for tr_idx, va_idx in splits:
            m = Ridge(alpha=float(a), random_state=42)
            m.fit(Xtr[tr_idx], y[tr_idx])
            oof_a[va_idx] = m.predict(Xtr[va_idx])

            # test 端：這裡是 fold 平均（簡單但常見）
            te_a += m.predict(Xte)

        te_a /= float(CFG["folds"])
        mse_a = mean_squared_error(y, oof_a)

        oof_list.append(oof_a)
        te_list.append(te_a)
        mses.append(mse_a)

        print(f"[OOF] Ridge alpha={a} raw MSE: {mse_a:.12f}")

    oof_mat = np.vstack(oof_list)   # shape: (n_alpha, n_train)
    te_mat  = np.vstack(te_list)    # shape: (n_alpha, n_test)
    mses = np.array(mses, dtype=np.float64)

    # 根據 OOF MSE 做權重
    if weight_by_oof:
        w = (1.0 / np.maximum(mses, 1e-12)) ** float(power)
        w = w / w.sum()
    else:
        w = np.ones(len(alphas), dtype=np.float64) / len(alphas)

    # 加權平均得到 ensemble 輸出
    oof = (w.reshape(-1,1) * oof_mat).sum(axis=0)
    te  = (w.reshape(-1,1) * te_mat).sum(axis=0)
    return oof, te, w, mses


print("\n===== Ridge on feature v1 =====")
oof_ridge1, test_ridge1, w1, mses1 = ridge_ensemble_oof_test(
    X1_train_s, X1_test_s, base_splits, CFG["ridge_alphas"],
    weight_by_oof=CFG["ridge_weight_by_oof"], power=CFG["ridge_weight_power"]
)
print("[OOF] Ridge-v1 ensemble raw MSE:", mean_squared_error(y, oof_ridge1))
print("[Ridge-v1] alpha weights:", {str(a): float(w) for a, w in zip(CFG["ridge_alphas"], w1)})

print("\n===== Ridge on feature v2 =====")
oof_ridge2, test_ridge2, w2, mses2 = ridge_ensemble_oof_test(
    X2_train_s, X2_test_s, base_splits, CFG["ridge_alphas"],
    weight_by_oof=CFG["ridge_weight_by_oof"], power=CFG["ridge_weight_power"]
)
print("[OOF] Ridge-v2 ensemble raw MSE:", mean_squared_error(y, oof_ridge2))
print("[Ridge-v2] alpha weights:", {str(a): float(w) for a, w in zip(CFG["ridge_alphas"], w2)})


# =========================================================
# Stage1：Linear stacking（使用 OOF 預測作為訓練資料）
# ---------------------------------------------------------
# X_meta = [oof_nn, oof_ridge1, oof_ridge2]
# 用線性回歸學最適合的組合係數
# positive=True：係數限制非負（更穩）
# =========================================================
X_meta = np.vstack([oof_nn, oof_ridge1, oof_ridge2]).T.astype(np.float64)
T_meta = np.vstack([test_nn, test_ridge1, test_ridge2]).T.astype(np.float64)

meta_lr = LinearRegression(positive=bool(CFG["stacking_positive"]), fit_intercept=True)
meta_lr.fit(X_meta, y)

stage1_oof = meta_lr.predict(X_meta)
stage1_test = meta_lr.predict(T_meta)
stage1_mse = mean_squared_error(y, stage1_oof)

print("\n[Stage1] Linear stacking OOF MSE =", stage1_mse)
print("[Stage1 Coefs] ",
      {"nn": float(meta_lr.coef_[0]), "ridge_v1": float(meta_lr.coef_[1]), "ridge_v2": float(meta_lr.coef_[2])},
      "| intercept:", float(meta_lr.intercept_))


# =========================================================
# Stage2：Polynomial(deg=2) + Ridge stacking（CV-safe）
# ---------------------------------------------------------
# 做法：
# 1) 對 base preds 做二次多項式展開（包含交互項）
# 2) 對每個 alpha：
#    - 用 base_splits 做 OOF（CV-safe）
# 3) 選 OOF MSE 最低的 alpha
# 4) 用全資料 fit 再預測 test
# =========================================================
print("\n===== Stage2: Poly(2) Ridge stacking (CV-safe) =====")
poly = PolynomialFeatures(degree=int(CFG["stage2_poly_degree"]), include_bias=False)
X_meta_poly = poly.fit_transform(X_meta)
T_meta_poly = poly.transform(T_meta)

best_stage2_mse = float("inf")
best_stage2_alpha = None
best_stage2_oof = None
best_stage2_test = None

for a in CFG["stage2_ridge_alphas"]:
    oof2 = np.zeros(len(y), dtype=np.float64)

    # CV-safe OOF
    for tr_idx, va_idx in base_splits:
        reg = Ridge(alpha=float(a), random_state=42)
        reg.fit(X_meta_poly[tr_idx], y[tr_idx])
        oof2[va_idx] = reg.predict(X_meta_poly[va_idx])

    mse2 = mean_squared_error(y, oof2)

    # 更新最佳
    if mse2 < best_stage2_mse:
        best_stage2_mse = mse2
        best_stage2_alpha = a
        best_stage2_oof = oof2

    print(f"[Stage2] alpha={a:g} | OOF MSE={mse2:.12f}")

# 用最佳 alpha 對全資料 fit，預測 test
reg_full = Ridge(alpha=float(best_stage2_alpha), random_state=42)
reg_full.fit(X_meta_poly, y)
best_stage2_test = reg_full.predict(T_meta_poly)

print(f"\n[Stage2 Best] alpha={best_stage2_alpha} | OOF MSE={best_stage2_mse:.12f}")


# =========================================================
# Blend：Stage1 與 Stage2 做線性混合
# ---------------------------------------------------------
# final = (1-t)*stage1 + t*stage2
# 用 OOF 搜尋最佳 t（0~1）
# =========================================================
print("\n===== Blend Stage1 & Stage2 =====")
best_t = 0.0
best_blend_mse = stage1_mse

for t in np.linspace(0, 1, 2001):  # step=0.0005
    oof_blend = (1.0 - t) * stage1_oof + t * best_stage2_oof
    mse_b = mean_squared_error(y, oof_blend)

    if mse_b < best_blend_mse:
        best_blend_mse = mse_b
        best_t = float(t)

final_oof = (1.0 - best_t) * stage1_oof + best_t * best_stage2_oof
final_test = (1.0 - best_t) * stage1_test + best_t * best_stage2_test

print(f"[Blend] best t={best_t:.4f} | OOF MSE={best_blend_mse:.12f}")


# =========================================================
# Mild clipping：避免 test 預測出現極端值
# ---------------------------------------------------------
# 這是保守的防呆做法：
# 以 train y 的 min/max 為基準，允許多 5% range
# 避免某些模型在 test 端出現不合理爆掉的值
# =========================================================
ymin, ymax = float(y.min()), float(y.max())
final_test = np.clip(final_test, ymin - 0.05*(ymax-ymin), ymax + 0.05*(ymax-ymin))


# =========================================================
# Submission：id 必須唯一，y 做 round
# ---------------------------------------------------------
# - id 用 safe_submission_id 避免 duplicate
# - y round 到 6 位小數（你原本就這樣做）
# =========================================================
sub_id = safe_submission_id(sample_df, test_df)
sub = pd.DataFrame({"id": sub_id})
sub["y"] = np.round(final_test.astype(float), 6)

dup = int(pd.Series(sub["id"]).duplicated().sum())
print("\ndup id count:", dup)

sub.to_csv(CFG["out_path"], index=False)

print("Saved:", CFG["out_path"], "| rows:", len(sub))
print("Total time:", time.time() - start)
print(sub.head())
