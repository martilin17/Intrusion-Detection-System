import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "| device:", device)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# =========================
# 0) Config
# =========================
# ── Stage-1 (CNN-B) ──────────────────────────────────────────────────────────
LR_B      = 1e-2    # 論文 Table 2
BATCH_B   = 512     # 論文 Table 2
EPOCHS_B  = 10      # 論文：Stage-1 固定 10 epochs

# ── Stage-2 (CNN-TL) ──────────────────────────────────────────────────────────
LR_T      = 1e-3    # 論文 Table 2：避免大幅改動 CNN-B layer-3 weights
BATCH_T   = 1024    # 論文 Table 2
EPOCHS_T  = 2000    # 論文：最多 2000 epochs，early stopping 提前結束
PATIENCE  = 100     # ✅ Fix-4: 50→100，給 Stage-2 更多時間跳出 false minimum

# ── 共用 ─────────────────────────────────────────────────────────────────────
DROPOUT   = 0.5
GRAD_CLIP = 1.0
# ✅ Fix-1: SEQ_LEN 8→32
#   SEQ_LEN=8 時 CNN-B 三次 MaxPool 後 L=1，CNN-T Temporal Attention 全被 bypass
#   SEQ_LEN=32:
#     CNN-B: 32→16→8→4  (block3 output L=4)
#     CNN-T: block1 MaxPool → L=2  (Attention 有效 ✅)
#            block2 MaxPool → L=1  (bypass，正常)
SEQ_LEN   = 32
STRIDE    = 1

# =========================
# 1) Load NSL-KDD
# =========================
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

train_file  = BASE_DIR / "Data" / "KDDTrain+.txt"
test_file   = BASE_DIR / "Data" / "KDDTest+.txt"
test21_file = BASE_DIR / "Data" / "KDDTest-21.txt"

train_df  = pd.read_csv(train_file,  header=None)
test_df   = pd.read_csv(test_file,   header=None)
test21_df = pd.read_csv(test21_file, header=None)

columns = (
    ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
     'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
     'root_shell','su_attempted','num_root','num_file_creations','num_shells',
     'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
     'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
     'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
     'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
     'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
     'dst_host_serror_rate','dst_host_srv_serror_rate',
     'dst_host_rerror_rate','dst_host_srv_rerror_rate','target','level']
)
for df in (train_df, test_df, test21_df):
    df.columns = columns

# =========================
# 2) Missing Value Patching
# =========================
for df in (train_df, test_df, test21_df):
    df.fillna(0, inplace=True)

# =========================
# 3) Macro categories (5-class)
# =========================
def macro_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df["target"].astype(str).str.strip()
    mapping = {
        **{k: 'Dos'   for k in ['apache2','back','land','neptune','mailbomb','pod',
                                  'processtable','smurf','teardrop','udpstorm','worm']},
        **{k: 'R2L'   for k in ['ftp_write','guess_passwd','httptunnel','imap','multihop',
                                  'named','phf','sendmail','snmpgetattack','snmpguess',
                                  'spy','warezclient','warezmaster','xlock','xsnoop']},
        **{k: 'Probe' for k in ['ipsweep','mscan','nmap','portsweep','saint','satan']},
        **{k: 'U2R'   for k in ['buffer_overflow','loadmodule','perl','ps','rootkit',
                                  'sqlattack','xterm']},
        'normal': 'Normal',
    }
    df["target"] = df["target"].replace(mapping)
    return df

train_df  = macro_target(train_df)
test_df   = macro_target(test_df)
test21_df = macro_target(test21_df)

print("\n[Train] macro label counts:\n", train_df["target"].value_counts())
print("\n[Test]  macro label counts:\n", test_df["target"].value_counts())
print("\n[Test-21] macro label counts:\n", test21_df["target"].value_counts())

CLASS_NAMES = ["Normal", "Dos", "Probe", "U2R", "R2L"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
NORMAL_ID   = CLASS_TO_ID["Normal"]

def to_y(df: pd.DataFrame) -> np.ndarray:
    y = df["target"].astype(str).str.strip().map(CLASS_TO_ID)
    if y.isna().any():
        raise ValueError(f"Unknown labels: {df.loc[y.isna(),'target'].unique()}")
    return y.astype(np.int64).to_numpy()

y_train  = to_y(train_df)
y_test   = to_y(test_df)
y_test21 = to_y(test21_df)

# =========================
# 4) Digitizing
# =========================
svc_le = LabelEncoder()
svc_le.fit(pd.concat([train_df["service"], test_df["service"], test21_df["service"]], axis=0).astype(str))
for df in (train_df, test_df, test21_df):
    df["service"] = svc_le.transform(df["service"].astype(str))

all_df = pd.concat([train_df, test_df, test21_df], axis=0, ignore_index=True)
all_df = pd.get_dummies(all_df, columns=["protocol_type", "flag"])

n_tr = len(train_df)
n_te = len(test_df)

train_enc  = all_df.iloc[:n_tr].copy()
test_enc   = all_df.iloc[n_tr:n_tr+n_te].copy()
test21_enc = all_df.iloc[n_tr+n_te:].copy()

def to_X(df_enc: pd.DataFrame) -> np.ndarray:
    return df_enc.drop(columns=["target", "level"]).to_numpy(dtype=np.float32)

X_train  = to_X(train_enc)
X_test   = to_X(test_enc)
X_test21 = to_X(test21_enc)

print("\nFeature dim:", X_train.shape[1])

# =========================
# 5) Split Base dataset (KDDTrain+) → 80/20 for Stage-1 train / val
# =========================
X_tr_b, X_val_b, y_tr_b, y_val_b = train_test_split(
    X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train
)

# =========================
# 6) Normalizing (MinMax [0,1]) — fit ONLY on Stage-1 train split
# =========================
scaler = MinMaxScaler()
scaler.fit(X_tr_b)

X_tr_b_s   = scaler.transform(X_tr_b).astype(np.float32)
X_val_b_s  = scaler.transform(X_val_b).astype(np.float32)
X_train_s  = scaler.transform(X_train).astype(np.float32)
X_test_s   = scaler.transform(X_test).astype(np.float32)
X_test21_s = scaler.transform(X_test21).astype(np.float32)

# =========================
# 7) Time Series Formatting (sliding window)
# =========================
def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
    N = X.shape[0]
    if N < seq_len:
        raise ValueError(f"N={N} < seq_len={seq_len}.")
    Xs, ys = [], []
    for s in range(0, N - seq_len + 1, stride):
        e = s + seq_len
        Xs.append(X[s:e])
        ys.append(y[e - 1])    # 視窗最後一筆當 label
    return np.stack(Xs).astype(np.float32), np.array(ys, dtype=np.int64)

# Stage-1 sequences (Base: KDDTrain+)
X_tr_b_seq,  y_tr_b_seq  = make_sequences(X_tr_b_s,  y_tr_b,  SEQ_LEN, STRIDE)
X_val_b_seq, y_val_b_seq = make_sequences(X_val_b_s, y_val_b, SEQ_LEN, STRIDE)

# Full dataset sequences (evaluation)
X_train_seq,  y_train_seq  = make_sequences(X_train_s,  y_train,  SEQ_LEN, STRIDE)
X_test_seq,   y_test_seq   = make_sequences(X_test_s,   y_test,   SEQ_LEN, STRIDE)
X_test21_seq, y_test21_seq = make_sequences(X_test21_s, y_test21, SEQ_LEN, STRIDE)

print("\nStage-1 seq shapes | tr_b:", X_tr_b_seq.shape, "| val_b:", X_val_b_seq.shape)

# Stage-2: KDDTest+ → 80/20 Target Train / Val
# 論文：KDDTest+ 全量作為 Target Training data
X_tr_t, X_val_t, y_tr_t, y_val_t = train_test_split(
    X_test_seq, y_test_seq, test_size=0.20, random_state=SEED, stratify=y_test_seq
)
print("Stage-2 seq shapes | tr_t:", X_tr_t.shape, "| val_t:", X_val_t.shape)

# =========================
# 8) Class Weights (inv_freq + sqrt 壓縮，平衡 DR 提升 vs FPR 控制)
# =========================
def compute_class_weights(y: np.ndarray, n_classes: int,
                          label: str = "",
                          mode: str = "sqrt",
                          max_ratio: float = None) -> torch.Tensor:
    # mode="sqrt"  : sqrt(1/freq)  — Stage-1 溫和壓縮，平衡 DR vs FPR
    # mode="linear": 1/freq        — 較強懲罰
    # max_ratio: cap，任何 class weight 不超過最小 weight 的 max_ratio 倍
    #   診斷：Stage-2 linear 讓 U2R weight ≈ 144x Normal → class collapse
    #   cap=15 保有足夠懲罰但不崩潰
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv_freq = np.sqrt(1.0 / counts) if mode == "sqrt" else (1.0 / counts)
    if max_ratio is not None:
        min_w    = inv_freq.min()
        inv_freq = np.minimum(inv_freq, min_w * max_ratio)
    inv_freq = inv_freq / inv_freq.sum() * n_classes
    if label:
        print(f"\n[Class Weights ({mode}, cap={max_ratio}) - {label}]")
        for i, (name, w) in enumerate(zip(CLASS_NAMES, inv_freq)):
            print(f"  {name:8s}: count={int(counts[i]):6d}, weight={w:.4f}")
    return torch.tensor(inv_freq, dtype=torch.float32)

# Stage-1: sqrt, no cap
cw_stage1 = compute_class_weights(
    y_tr_b_seq, len(CLASS_NAMES), "Stage-1 KDDTrain+",
    mode="sqrt", max_ratio=None
)
# ✅ Fix-A+C: Stage-2 改回 sqrt + cap=15
#   原 linear 導致 U2R weight ≈ 144x Normal → FPR 爆炸 class collapse
#   sqrt + cap=15: U2R weight 上限=15x Normal，足夠懲罰少數類又不崩潰
cw_stage2 = compute_class_weights(
    y_tr_t, len(CLASS_NAMES), "Stage-2 KDDTest+",
    mode="sqrt", max_ratio=8.0   # ✅ Fix: 15→8, 減少 attack bias，降低 FPR
)

# =========================
# 9) Dataset / DataLoader
# =========================
class SeqDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.X[i].transpose(0, 1), self.y[i]   # (F, L), label

# Stage-1 loaders
s1_train_loader = DataLoader(SeqDataset(X_tr_b_seq,  y_tr_b_seq),
                             batch_size=BATCH_B, shuffle=True,  num_workers=0)
s1_val_loader   = DataLoader(SeqDataset(X_val_b_seq, y_val_b_seq),
                             batch_size=BATCH_B, shuffle=False, num_workers=0)

# Stage-2 loaders
s2_train_loader = DataLoader(SeqDataset(X_tr_t,  y_tr_t),
                             batch_size=BATCH_T, shuffle=True,  num_workers=0)
s2_val_loader   = DataLoader(SeqDataset(X_val_t, y_val_t),
                             batch_size=BATCH_T, shuffle=False, num_workers=0)

# Evaluation loaders (full datasets, no shuffle)
eval_train_loader  = DataLoader(SeqDataset(X_train_seq,  y_train_seq),
                                batch_size=BATCH_B, shuffle=False, num_workers=0)
eval_test_loader   = DataLoader(SeqDataset(X_test_seq,   y_test_seq),
                                batch_size=BATCH_B, shuffle=False, num_workers=0)
eval_test21_loader = DataLoader(SeqDataset(X_test21_seq, y_test21_seq),
                                batch_size=BATCH_B, shuffle=False, num_workers=0)

# =========================
# 10) Model Definitions
# =========================

# ── Temporal Attention ────────────────────────────────────────────────────────
# 論文 Section 3.3：每個 ConvNet layer 的 Pooling 之後加入 Attention
# 讓模型在每個 time-step 關注不同的輸入部分
class TemporalAttention(nn.Module):
    """
    Temporal self-attention over L dimension.
    Input / Output: (B, C, L)
    1×1 Conv → Softmax → Scale（當 L=1 時 bypass，避免 trivial attention）
    """
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] <= 1:
            return x                                  # L=1: attention 無意義，直接傳遞
        w = torch.softmax(self.score(x), dim=2)       # (B, 1, L)
        return x * w                                  # (B, C, L)


# ── CNN-B：Stage-1 Baseline（3 ConvNet layers: 64→128→256）──────────────────
# 論文 Table 2 / Section 4: Layer-1=64, Layer-2=128, Layer-3=256
# Conv → BN → ReLU → MaxPool → Dropout（Stage-1 無 Attention）
class CNNB(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )
        # Global Average Pooling + FC head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)    # (B, 256, L')
        x = x.mean(dim=2)     # GAP → (B, 256)
        return self.fc(x)     # (B, n_classes)


# ── CNN-T：Stage-2 Transfer Module（2 ConvNet layers: 128→256 + Attention）──
# 論文 Table 2: Layer-1=128, Layer-2=256, Layer-3=N/A
# 每個 ConvNet layer: Conv → BN → ReLU → MaxPool → Attention → Dropout
# ceil_mode=True：防止 L=1 經 MaxPool 後變成 L=0
class CNNT(nn.Module):
    def __init__(self, in_ch: int, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            TemporalAttention(128),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),
            TemporalAttention(256),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x    # (B, 256, L'')


# ── CNN-TL：完整 Transfer Learning 模型 ─────────────────────────────────────
# 論文 Section 3.3 Freezing & Fine-tuning strategy:
#   CNN-B.block1, block2 → frozen（固定 weights）
#   CNN-B.block3          → fine-tune（保留 Stage-1 weights，LR=1e-3）
#   CNN-T (2 blocks)      → random init，從頭訓練
#   new FC head           → random init，從頭訓練
class CNNTL(nn.Module):
    def __init__(self, cnnb: CNNB, n_classes: int, dropout: float = 0.5):
        super().__init__()
        # 直接共享 CNN-B 的 blocks（不複製）
        self.cnnb_block1 = cnnb.block1   # ← will be frozen
        self.cnnb_block2 = cnnb.block2   # ← will be frozen
        self.cnnb_block3 = cnnb.block3   # ← fine-tune

        # CNN-T 接在 CNN-B.block3 output（256 channels）之後
        self.cnnt = CNNT(in_ch=256, dropout=dropout)

        # 全新 FC head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnnb_block1(x)    # frozen
        x = self.cnnb_block2(x)    # frozen
        x = self.cnnb_block3(x)    # fine-tune  → (B, 256, L')
        x = self.cnnt(x)           # CNN-T + Attention → (B, 256, L'')
        x = x.mean(dim=2)          # GAP → (B, 256)
        return self.fc(x)          # (B, n_classes)


in_ch = X_tr_b_seq.shape[2]    # feature dimension F

# =========================
# 11) Checkpoint utilities
# =========================
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def save_ckpt(path: Path, mdl: nn.Module, opt, sch, epoch: int, val_loss: float) -> None:
    torch.save({
        "model_state_dict":     mdl.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sch.state_dict() if sch else {},
        "epoch":    epoch,
        "val_loss": val_loss,
    }, path)
    print(f"  💾 Saved: {path.name}")

@torch.no_grad()
def eval_loss_fn(mdl: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    mdl.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(mdl(xb), yb)
        total += loss.item() * yb.size(0)
        n += yb.size(0)
    return total / max(1, n)

# =========================
# 12) Stage-1: Train CNN-B (10 epochs)
# =========================
cnnb = CNNB(in_ch=in_ch, n_classes=len(CLASS_NAMES), dropout=DROPOUT).to(device)

criterion_b = nn.CrossEntropyLoss(weight=cw_stage1.to(device))
optimizer_b = torch.optim.Adam(cnnb.parameters(), lr=LR_B)
scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_b, T_max=EPOCHS_B, eta_min=1e-4
)

print(f"\n{'='*65}")
print(f"  STAGE-1: Training CNN-B")
print(f"  params={sum(p.numel() for p in cnnb.parameters()):,}")
print(f"  LR={LR_B}  BS={BATCH_B}  Epochs={EPOCHS_B}  GradClip={GRAD_CLIP}")
print(f"{'='*65}")

CKPT_B_BEST = CKPT_DIR / "cnnb_best.ckpt"
CKPT_B_LAST = CKPT_DIR / "cnnb_last.ckpt"

best_b_loss = float("inf")

for ep in range(1, EPOCHS_B + 1):
    cnnb.train()
    total, n = 0.0, 0
    for xb, yb in s1_train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_b.zero_grad(set_to_none=True)
        loss = criterion_b(cnnb(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(cnnb.parameters(), max_norm=GRAD_CLIP)
        optimizer_b.step()
        total += loss.item() * yb.size(0)
        n     += yb.size(0)

    scheduler_b.step()
    tr_loss = total / max(1, n)
    va_loss = eval_loss_fn(cnnb, s1_val_loader, criterion_b)
    lr_now  = scheduler_b.get_last_lr()[0]

    is_best = va_loss < best_b_loss
    if is_best:
        best_b_loss = va_loss
        save_ckpt(CKPT_B_BEST, cnnb, optimizer_b, scheduler_b, ep, va_loss)

    tag = " ⭐ best" if is_best else ""
    print(f"[S1] Ep {ep:02d} | train {tr_loss:.6f} | val {va_loss:.6f} | lr {lr_now:.6f}{tag}")

save_ckpt(CKPT_B_LAST, cnnb, optimizer_b, scheduler_b, EPOCHS_B, va_loss)
print(f"\n✅ Stage-1 done. Best val loss: {best_b_loss:.6f}")

# Load best CNN-B weights 以備 Stage-2
ckpt_b = torch.load(CKPT_B_BEST, map_location=device, weights_only=True)
cnnb.load_state_dict(ckpt_b["model_state_dict"])
print("✅ Best CNN-B weights loaded for Stage-2")

# =========================
# 13) Stage-2: Build CNN-TL → Freeze → Train
# =========================
cnntl = CNNTL(cnnb=cnnb, n_classes=len(CLASS_NAMES), dropout=DROPOUT).to(device)

# ── Freeze CNN-B block1 & block2 ──────────────────────────────────────────────
# 論文：weights of layer-1 and layer-2 in CNN-B are fixed
for param in cnntl.cnnb_block1.parameters():
    param.requires_grad = False
for param in cnntl.cnnb_block2.parameters():
    param.requires_grad = False

# ── Param Groups：block3 極保守 fine-tune (LR=1e-4)，CNN-T/FC 從頭學 (LR=1e-3) ──
# 診斷：Stage-2 後 benign DR 從 94% 掉到 81%，是 block3 更新太激進破壞 Normal 特徵
# 論文："layer-3 is retrained using the previous weights" → 應保守微調
LR_BLOCK3 = LR_T * 0.1   # 1e-4

frozen_params    = (list(cnntl.cnnb_block1.parameters()) +
                    list(cnntl.cnnb_block2.parameters()))
block3_params    = list(cnntl.cnnb_block3.parameters())
new_params       = list(cnntl.cnnt.parameters()) + list(cnntl.fc.parameters())
trainable_params = block3_params + new_params   # 用於 grad clip

param_groups = [
    {"params": block3_params, "lr": LR_BLOCK3, "name": "cnnb_block3"},
    {"params": new_params,    "lr": LR_T,       "name": "cnnt+fc"},
]

print(f"\n{'='*65}")
print(f"  STAGE-2: Training CNN-TL (Transfer Learning)")
print(f"  CNN-B block1/2 → FROZEN   ({sum(p.numel() for p in frozen_params):,} params)")
print(f"  CNN-B block3   → fine-tune LR={LR_BLOCK3:.0e}  ({sum(p.numel() for p in block3_params):,} params)")
print(f"  CNN-T + FC     → train     LR={LR_T:.0e}       ({sum(p.numel() for p in new_params):,} params)")
print(f"  BS={BATCH_T}  MaxEpochs={EPOCHS_T}  Patience={PATIENCE}")
print(f"{'='*65}")

# label_smoothing 降至 0.05（cap 已降低，過度 smoothing 反而損失少數類 DR）
criterion_t = nn.CrossEntropyLoss(weight=cw_stage2.to(device), label_smoothing=0.05)
optimizer_t = torch.optim.Adam(param_groups)
# ✅ Cosine LR scheduler（param groups 各自按比例 decay）
scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_t, T_max=EPOCHS_T, eta_min=1e-6
)

CKPT_TL_BEST = CKPT_DIR / "cnntl_best.ckpt"
CKPT_TL_LAST = CKPT_DIR / "cnntl_last.ckpt"

best_t_loss    = float("inf")
patience_count = 0
stopped_epoch  = EPOCHS_T

for ep in range(1, EPOCHS_T + 1):
    cnntl.train()
    # ✅ Fix-2: Frozen blocks 必須保持 eval() mode
    #   .train() 會把整個 model 切成 train mode，包含 frozen 的 BatchNorm
    #   BatchNorm 在 train mode 會用 batch statistics 更新 running_mean/var
    #   → frozen layer 的 BN stats 被 Stage-2 data 污染，破壞 Stage-1 學到的特徵
    cnntl.cnnb_block1.eval()
    cnntl.cnnb_block2.eval()
    total, n = 0.0, 0
    for xb, yb in s2_train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_t.zero_grad(set_to_none=True)
        loss = criterion_t(cnntl(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP)
        optimizer_t.step()
        total += loss.item() * yb.size(0)
        n     += yb.size(0)

    scheduler_t.step()
    tr_loss = total / max(1, n)
    va_loss = eval_loss_fn(cnntl, s2_val_loader, criterion_t)
    lr_now  = scheduler_t.get_last_lr()[0]

    is_best = va_loss < best_t_loss
    if is_best:
        best_t_loss    = va_loss
        patience_count = 0
        save_ckpt(CKPT_TL_BEST, cnntl, optimizer_t, scheduler_t, ep, va_loss)
    else:
        patience_count += 1

    # 印出 log：前 10 epoch 每 epoch、之後每 50 epoch、以及 best
    if ep <= 10 or ep % 50 == 0 or is_best:
        tag = " ⭐ best" if is_best else ""
        print(f"[S2] Ep {ep:04d} | train {tr_loss:.6f} | val {va_loss:.6f} "
              f"| lr {lr_now:.7f} | patience {patience_count:3d}{tag}")

    # Early stopping
    if patience_count >= PATIENCE:
        stopped_epoch = ep
        print(f"\n⏹️  Early stopping at epoch {ep}  (best val: {best_t_loss:.6f})")
        break

save_ckpt(CKPT_TL_LAST, cnntl, optimizer_t, scheduler_t, stopped_epoch, va_loss)
print(f"\n✅ Stage-2 done. Best val loss: {best_t_loss:.6f}")

# Load best CNN-TL weights for evaluation
ckpt_tl = torch.load(CKPT_TL_BEST, map_location=device, weights_only=True)
cnntl.load_state_dict(ckpt_tl["model_state_dict"])
print("✅ Best CNN-TL weights loaded for evaluation")

# =========================
# 14) Metrics
# =========================

def metrics_overall(y_true: np.ndarray, y_pred: np.ndarray, normal_id: int = 0) -> dict:
    """Overall binary (Normal vs Attack) metrics."""
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64)
    acc = float((y_true == y_pred).mean())
    ta  = y_true != normal_id
    tn_ = y_true == normal_id
    dr  = float(((y_pred != normal_id) &  ta).sum() / ( ta.sum() + 1e-12))
    fpr = float(((y_pred != normal_id) & tn_).sum() / (tn_.sum() + 1e-12))
    tn  = int(((y_true == normal_id) & (y_pred == normal_id)).sum())
    fp  = int(((y_true == normal_id) & (y_pred != normal_id)).sum())
    fn  = int(((y_true != normal_id) & (y_pred == normal_id)).sum())
    tp  = int(((y_true != normal_id) & (y_pred != normal_id)).sum())
    return {"DR": dr, "ACC": acc, "FPR": fpr, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

def metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list:
    """One-vs-Rest per-class DR / ACC / FPR (matches paper Table 6 format)."""
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64)
    N = len(y_true); results = []
    for c in range(n_classes):
        tc = y_true == c; pc = y_pred == c
        tp = int(( tc &  pc).sum()); fn = int(( tc & ~pc).sum())
        fp = int((~tc &  pc).sum()); tn = int((~tc & ~pc).sum())
        results.append({
            "class_id": c,
            "DR":  tp / (tp + fn + 1e-12),
            "ACC": (tp + tn) / (N + 1e-12),
            "FPR": fp / (fp + tn + 1e-12),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "support": tp + fn,
        })
    return results

@torch.no_grad()
def predict_labels(mdl: nn.Module, loader: DataLoader):
    mdl.eval()
    preds, trues = [], []
    for xb, yb in loader:
        yhat = torch.argmax(mdl(xb.to(device)), dim=1).cpu().numpy()
        preds.append(yhat); trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)

# =========================
# 15) Paper Reference Numbers
# =========================
DISPLAY_NAMES = ["benign", "dos", "probe", "u2r", "r2l"]

# Table 6 CNN (no TL) — baseline reference for Stage-1
PAPER_CNN = {
    "KDDTrain+": {
        "benign": {"DR": 95.38, "ACC": 96.78, "FPR":  1.61},
        "dos":    {"DR": 99.98, "ACC": 98.12, "FPR":  2.95},
        "probe":  {"DR": 98.47, "ACC": 99.03, "FPR":  0.90},
        "u2r":    {"DR":  0.00, "ACC": 99.96, "FPR":  0.00},
        "r2l":    {"DR":  0.00, "ACC": 99.21, "FPR":  0.00},
        "overall":{"DR": 94.58, "ACC": 97.58, "FPR":  0.47},
    },
    "KDDTest+": {
        "benign": {"DR": 94.01, "ACC": 80.76, "FPR": 29.33},
        "dos":    {"DR": 85.44, "ACC": 91.57, "FPR":  5.38},
        "probe":  {"DR": 75.92, "ACC": 94.68, "FPR":  3.07},
        "u2r":    {"DR":  0.00, "ACC": 99.70, "FPR":  0.00},
        "r2l":    {"DR":  0.00, "ACC": 87.20, "FPR":  0.00},
        "overall":{"DR": 72.77, "ACC": 77.86, "FPR":  5.83},
    },
    "KDDTest-21": {
        "benign": {"DR": 79.51, "ACC": 64.51, "FPR": 38.81},
        "dos":    {"DR": 75.01, "ACC": 84.94, "FPR":  9.31},
        "probe":  {"DR": 75.73, "ACC": 90.02, "FPR":  6.34},
        "u2r":    {"DR":  0.00, "ACC": 99.43, "FPR":  0.00},
        "r2l":    {"DR":  0.00, "ACC": 75.65, "FPR":  0.00},
        "overall":{"DR": 47.03, "ACC": 57.30, "FPR": 10.94},
    },
}

# Table 6 CNN-TL (with TL) — target reference for Stage-2
PAPER_TL = {
    "KDDTrain+": {
        "benign": {"DR": 84.49, "ACC": 89.96, "FPR":  3.76},
        "dos":    {"DR": 95.18, "ACC": 96.55, "FPR":  2.66},
        "probe":  {"DR": 78.19, "ACC": 95.58, "FPR":  2.65},
        "u2r":    {"DR":  0.00, "ACC": 99.96, "FPR":  0.00},
        "r2l":    {"DR": 46.23, "ACC": 92.89, "FPR":  6.73},
        "overall":{"DR": 83.05, "ACC": 94.98, "FPR":  3.22},
    },
    "KDDTest+": {
        "benign": {"DR": 92.44, "ACC": 95.31, "FPR":  2.52},
        "dos":    {"DR": 95.73, "ACC": 98.34, "FPR":  0.38},
        "probe":  {"DR": 97.48, "ACC": 98.98, "FPR":  0.78},
        "u2r":    {"DR":  0.00, "ACC": 99.70, "FPR":  0.00},
        "r2l":    {"DR": 94.45, "ACC": 95.68, "FPR":  4.13},
        "overall":{"DR": 93.15, "ACC": 94.18, "FPR":  1.72},
    },
    "KDDTest-21": {
        "benign": {"DR": 73.56, "ACC": 92.46, "FPR":  3.34},
        "dos":    {"DR": 92.68, "ACC": 96.85, "FPR":  0.73},
        "probe":  {"DR": 97.09, "ACC": 98.10, "FPR":  1.64},
        "u2r":    {"DR":  0.00, "ACC": 99.43, "FPR":  0.00},
        "r2l":    {"DR": 94.45, "ACC": 93.16, "FPR":  7.25},
        "overall":{"DR": 88.46, "ACC": 90.01, "FPR":  2.83},
    },
}

# =========================
# 16) Pretty-print table
# =========================
def fmt_delta(v: float, higher_better: bool = True) -> str:
    sym = "✅" if (higher_better and v > 0) or (not higher_better and v < 0) \
          else ("⚠️ " if abs(v) < 1.5 else "❌")
    return f"{v:+6.2f}%{sym}"

def print_per_class_table(dataset_name: str,
                          per_class: list,
                          overall: dict,
                          paper_ref: dict = None,
                          model_tag: str = "Ours") -> None:
    W = 100 if paper_ref else 62
    print("\n" + "=" * W)
    print(f"  Dataset : {dataset_name}   Model: {model_tag}")
    print("=" * W)

    if paper_ref:
        hdr = (f"{'Traffic Type':<12}  {'DR (Ours)':>10}  {'DR (Paper)':>10}  {'ΔDR':>9}"
               f"  {'ACC (Ours)':>10}  {'ACC (Paper)':>11}  {'ΔACC':>9}"
               f"  {'FPR (Ours)':>10}  {'FPR (Paper)':>11}  {'ΔFPR':>9}")
    else:
        hdr = (f"{'Traffic Type':<12}  {'DR':>10}  {'ACC':>10}  {'FPR':>10}"
               f"  {'TP':>7}  {'FN':>7}  {'FP':>7}  {'TN':>7}  {'Support':>8}")
    print(hdr)
    print("-" * W)

    for m in per_class:
        cname = DISPLAY_NAMES[m["class_id"]]
        dr_s  = f"{m['DR']*100:6.2f}%"
        acc_s = f"{m['ACC']*100:6.2f}%"
        fpr_s = f"{m['FPR']*100:6.2f}%"

        if paper_ref and cname in paper_ref:
            p = paper_ref[cname]
            row = (f"  {cname:<12}  {dr_s:>10}  {p['DR']:>9.2f}%  "
                   f"{fmt_delta(m['DR']*100 - p['DR'])}"
                   f"  {acc_s:>10}  {p['ACC']:>10.2f}%  "
                   f"{fmt_delta(m['ACC']*100 - p['ACC'])}"
                   f"  {fpr_s:>10}  {p['FPR']:>10.2f}%  "
                   f"{fmt_delta(m['FPR']*100 - p['FPR'], higher_better=False)}")
        else:
            row = (f"  {cname:<12}  {dr_s:>10}  {acc_s:>10}  {fpr_s:>10}"
                   f"  {m['tp']:>7}  {m['fn']:>7}  {m['fp']:>7}  {m['tn']:>7}"
                   f"  {m['support']:>8}")
        print(row)

    print("-" * W)
    ov_dr  = f"{overall['DR'] *100:6.2f}%"
    ov_acc = f"{overall['ACC']*100:6.2f}%"
    ov_fpr = f"{overall['FPR']*100:6.2f}%"

    if paper_ref and "overall" in paper_ref:
        p = paper_ref["overall"]
        ov_row = (f"  {'OVERALL':<12}  {ov_dr:>10}  {p['DR']:>9.2f}%  "
                  f"{fmt_delta(overall['DR']*100 - p['DR'])}"
                  f"  {ov_acc:>10}  {p['ACC']:>10.2f}%  "
                  f"{fmt_delta(overall['ACC']*100 - p['ACC'])}"
                  f"  {ov_fpr:>10}  {p['FPR']:>10.2f}%  "
                  f"{fmt_delta(overall['FPR']*100 - p['FPR'], higher_better=False)}")
    else:
        ov_row = (f"  {'OVERALL':<12}  {ov_dr:>10}  {ov_acc:>10}  {ov_fpr:>10}"
                  f"  {overall['tp']:>7}  {overall['fn']:>7}"
                  f"  {overall['fp']:>7}  {overall['tn']:>7}")
    print(ov_row)
    print("=" * W)


def full_eval(ds_name: str, loader: DataLoader, mdl: nn.Module,
              paper_ref: dict = None, model_tag: str = "Ours"):
    y_true, y_pred = predict_labels(mdl, loader)
    overall   = metrics_overall(y_true, y_pred, normal_id=NORMAL_ID)
    per_class = metrics_per_class(y_true, y_pred, n_classes=len(CLASS_NAMES))
    print_per_class_table(ds_name, per_class, overall,
                          paper_ref=paper_ref, model_tag=model_tag)
    return overall, per_class

# =========================
# 17) Evaluation
# =========================

# ── Stage-1: CNN-B vs Paper CNN (no TL) ──────────────────────────────────────
print("\n\n" + "=" * 100)
print("  STAGE-1 RESULTS  ▌ CNN-B (no Transfer Learning) vs Paper CNN (no TL)")
print("=" * 100)

full_eval("KDDTrain+",  eval_train_loader,  cnnb,
          paper_ref=PAPER_CNN["KDDTrain+"],  model_tag="CNN-B  Stage-1")
full_eval("KDDTest+",   eval_test_loader,   cnnb,
          paper_ref=PAPER_CNN["KDDTest+"],   model_tag="CNN-B  Stage-1")
full_eval("KDDTest-21", eval_test21_loader, cnnb,
          paper_ref=PAPER_CNN["KDDTest-21"], model_tag="CNN-B  Stage-1")

# ── Stage-2: CNN-TL vs Paper CNN-TL ──────────────────────────────────────────
print("\n\n" + "=" * 100)
print("  STAGE-2 RESULTS  ▌ CNN-TL (with Transfer Learning) vs Paper CNN-TL")
print("=" * 100)

full_eval("KDDTrain+",  eval_train_loader,  cnntl,
          paper_ref=PAPER_TL["KDDTrain+"],  model_tag="CNN-TL Stage-2")
full_eval("KDDTest+",   eval_test_loader,   cnntl,
          paper_ref=PAPER_TL["KDDTest+"],   model_tag="CNN-TL Stage-2")
full_eval("KDDTest-21", eval_test21_loader, cnntl,
          paper_ref=PAPER_TL["KDDTest-21"], model_tag="CNN-TL Stage-2")