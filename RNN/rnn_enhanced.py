import os
import json
import random
from pathlib import Path
from collections import Counter
import math

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence

# ---------------------------
# 設定區域（可自行調整）
# ---------------------------

# Kaggle input 資料夾路徑
INPUT_DIR = "/kaggle/input"  # 若資料在子資料夾，改成 "/kaggle/input/your-dataset-folder"

# 資料檔名
TRAIN_JSON = "train.json"
TEST_JSON = "test.json"
VOCAB_JSON = "vocabulary.json"
SAMPLE_SUB = "sample_submission.csv"

# 隨機種子
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU（若有）或 CPU

# 訓練參數
NUM_FOLDS = 5          # K-Fold
BATCH_SIZE = 64
EMBED_DIM = 128        # Embedding 維度
HIDDEN_SIZE = 256      # LSTM 隱藏層維度
NUM_LAYERS = 1         # LSTM 層數
DROPOUT = 0.3
MAX_LEN = None         # 最大序列長度，None 則自動計算
LR = 1e-3              # 學習率
WEIGHT_DECAY = 1e-6    # 權重衰減
N_EPOCHS = 6           # 訓練輪數
PATIENCE = 2           # Early stopping 容忍次數
NUM_CLASSES = 5        # 分類類別數量

# 類別名稱與索引映射
LABELS = ["Category_A", "Category_B", "Category_C", "Category_D", "Category_E"]
label2idx = {l: i for i, l in enumerate(LABELS)}
idx2label = {i: l for l, i in label2idx.items()}

# ---------------------------
# 固定隨機數種子
# ---------------------------
def seed_everything(seed=SEED):
    """固定所有可能的隨機性以確保結果可重現"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ---------------------------
# 讀取資料
# ---------------------------
def find_file_in_input(filename):
    """
    在 Kaggle INPUT_DIR 中尋找檔案
    - 先檢查根目錄
    - 再搜尋第一層子資料夾
    """
    base = Path(INPUT_DIR)
    candidate = base / filename
    if candidate.exists():
        return str(candidate)
    for p in base.iterdir():
        fp = p / filename
        if fp.exists():
            return str(fp)
    raise FileNotFoundError(f"Can't find {filename} under {INPUT_DIR}.")

# 尋找各資料檔案路徑
train_path = find_file_in_input(TRAIN_JSON)
test_path = find_file_in_input(TEST_JSON)
vocab_path = find_file_in_input(VOCAB_JSON)
sample_sub_path = find_file_in_input(SAMPLE_SUB)

print("train_path:", train_path)
print("test_path:", test_path)
print("vocab_path:", vocab_path)
print("sample_submission:", sample_sub_path)

# 載入 JSON
with open(train_path, "r", encoding="utf-8") as f:
    train_json = json.load(f)
with open(test_path, "r", encoding="utf-8") as f:
    test_json = json.load(f)
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_json = json.load(f)

# 提取 sequences 與 labels
train_seqs = train_json["sequences"]
train_labels = train_json["labels"]
test_seqs = test_json["sequences"]

print(f"Train samples: {len(train_seqs)}, Test samples: {len(test_seqs)}")

# ---------------------------
# 序列長度與 padding 設定
# ---------------------------

# 取得 vocab_size
vocab_size = vocab_json.get("vocab_size", None)
if vocab_size is None:
    vocab_size = max(int(k) for k in vocab_json["idx_to_word"].keys()) + 1
print("vocab_size:", vocab_size)

# 計算 MAX_LEN（95 百分位數避免極端長序列）
if MAX_LEN is None:
    lengths = [len(s) for s in train_seqs]
    MAX_LEN = int(np.percentile(lengths, 95))
    MAX_LEN = max(16, MAX_LEN)  # 至少 16
print("MAX_LEN:", MAX_LEN)

# 特殊 token
PAD_IDX = 0
UNK_IDX = 1

# ---------------------------
# Dataset & DataLoader
# ---------------------------
class SeqDataset(Dataset):
    """文字序列資料集"""
    def __init__(self, sequences, labels=None, max_len=MAX_LEN):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_len]
        item = {"input": torch.tensor(seq, dtype=torch.long)}
        if self.labels is not None:
            item["label"] = torch.tensor(label2idx[self.labels[idx]], dtype=torch.long)
        return item

def collate_fn(batch):
    """
    用於 DataLoader 的 collate_fn
    - 將不同長度的序列 pad 到 MAX_LEN
    """
    inputs = [b["input"] for b in batch]
    padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
    if padded.size(1) > MAX_LEN:
        padded = padded[:, :MAX_LEN]
    elif padded.size(1) < MAX_LEN:
        pad_amt = MAX_LEN - padded.size(1)
        padded = F.pad(padded, (0, pad_amt), value=PAD_IDX)
    out = {"input_ids": padded}
    if "label" in batch[0]:
        out["labels"] = torch.stack([b["label"] for b in batch])
    return out

# ---------------------------
# Model 定義：Embedding + BiLSTM + Attention + FC
# ---------------------------
class Attention(nn.Module):
    """簡單的注意力機制"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)  # 將每個時間步投影成注意力分數

    def forward(self, x, mask=None):
        # x: (B, T, H)
        scores = self.proj(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, T)
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, H)
        return out, weights

class SeqClassifier(nn.Module):
    """Embedding + BiLSTM + Attention + FC 分類器"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, num_classes=NUM_CLASSES, pad_idx=PAD_IDX):
        super().__init__()
        # Embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size+2, embedding_dim=embed_dim, padding_idx=pad_idx)
        # BiLSTM
        self.encoder = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers>1 else 0.0)
        # 注意力
        self.attn = Attention(hidden_dim=hidden_size*2)
        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids):
        mask = (input_ids != PAD_IDX)  # 過濾 PAD token
        emb = self.embedding(input_ids)  # (B, T, E)
        outputs, _ = self.encoder(emb)   # (B, T, 2H)
        attn_out, weights = self.attn(outputs, mask=mask)  # (B, 2H)
        logits = self.fc(attn_out)       # (B, num_classes)
        return logits

# ---------------------------
# 訓練與驗證函式
# ---------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    preds, trues = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
        preds += logits.detach().cpu().argmax(1).tolist()
        trues += labels.detach().cpu().tolist()
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(trues, preds)
    return avg_loss, acc

@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, device):
    """驗證一輪"""
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item() * input_ids.size(0)
        preds += logits.detach().cpu().argmax(1).tolist()
        trues += labels.detach().cpu().tolist()
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(trues, preds)
    return avg_loss, acc

@torch.no_grad()
def predict_one_epoch(model, dataloader, device):
    """對一個 DataLoader 做預測，回傳 softmax 機率"""
    model.eval()
    all_logits = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)
        all_logits.append(logits.detach().cpu())
    all_logits = torch.cat(all_logits, dim=0)
    return F.softmax(all_logits, dim=1).numpy()

# ---------------------------
# Cross-validation 訓練與 Ensemble
# ---------------------------
labels_array = np.array([label2idx[l] for l in train_labels])
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

# 測試資料 Dataset / DataLoader
test_dataset = SeqDataset(test_seqs, labels=None, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 準備 OOF 與測試集累計預測
oof_preds = np.zeros((len(train_seqs), NUM_CLASSES))
test_preds = np.zeros((len(test_seqs), NUM_CLASSES))

fold_idx = 0
for train_idx, val_idx in skf.split(np.zeros(len(labels_array)), labels_array):
    fold_idx += 1
    print(f"\n========== Fold {fold_idx}/{NUM_FOLDS} ==========")
    # 取出該 fold 的訓練/驗證資料
    tr_seqs = [train_seqs[i] for i in train_idx]
    tr_labels = [train_labels[i] for i in train_idx]
    val_seqs = [train_seqs[i] for i in val_idx]
    val_labels = [train_labels[i] for i in val_idx]

    # Dataset / DataLoader
    train_dataset = SeqDataset(tr_seqs, labels=tr_labels, max_len=MAX_LEN)
    val_dataset = SeqDataset(val_seqs, labels=val_labels, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    model = SeqClassifier(vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS, dropout=DROPOUT, num_classes=NUM_CLASSES, pad_idx=PAD_IDX)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_cnt = 0

    # 訓練 N_EPOCHS 或早停
    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = valid_one_epoch(model, val_loader, criterion, DEVICE)
        print(f"Fold {fold_idx} Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping.")
                break

    # 載入最佳模型權重
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

    # 計算 OOF 預測
    val_loader_full = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    val_probs = predict_one_epoch(model, val_loader_full, DEVICE)
    oof_preds[val_idx] = val_probs

    # 累計測試集預測
    fold_test_probs = predict_one_epoch(model, test_loader, DEVICE)
    test_preds += fold_test_probs / NUM_FOLDS  # Fold 平均

# ---------------------------
# OOF Accuracy
# ---------------------------
oof_pred_labels = oof_preds.argmax(axis=1)
oof_true = labels_array
oof_acc = accuracy_score(oof_true, oof_pred_labels)
print(f"\nOOF Accuracy: {oof_acc:.6f}")

# ---------------------------
# 產生 submission.csv
# ---------------------------
submission_df = pd.read_csv(sample_sub_path)
pred_ids = np.arange(len(test_seqs))
pred_labels = test_preds.argmax(axis=1)
pred_labels_names = [idx2label[int(i)] for i in pred_labels]

submission = pd.DataFrame({"id": pred_ids, "category": pred_labels_names})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv (first 10 rows):")
print(submission.head(10))

# 選擇性：存 OOF 與測試預測權重
torch.save({'test_preds': test_preds, 'oof_preds': oof_preds}, "preds_fold_ensemble.pth")
print("Saved preds_fold_ensemble.pth")
