import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
LR_B = 1e-2
BATCH_B = 512
EPOCHS_B = 10

LR_T = 1e-3
BATCH_T = 1024
EPOCHS_T = 2000
PATIENCE = 100
S2_LABEL_SMOOTH = 0.0

DROPOUT = 0.5
GRAD_CLIP = 1.0
SEQ_LEN = 32
STRIDE = 1

# Hierarchical setup
RUN_TAG = "part2_hier_cnn_tl_v4"
SAVE_RUN_REPORT = True
BASELINE_REPORT = Path("checkpoints") / "report_part1_v4_2.json"

# Binary gate threshold tuning
GATE_TARGET_FPR = 0.045
GATE_TARGET_DR = 0.89
GATE_TUNE_TOPK = 8
GATE_TUNE_RELAX_FPR = 0.012
GATE_TUNE_RELAX_DR = 0.020
GATE_THR_MIN = 0.50
GATE_THR_MAX = 0.95
GATE_THR_STEPS = 160
GATE_SCORE_W_FPR = 1.2
GATE_SCORE_W_DR = 1.8
GATE_SCORE_W_DR_UNDER = 2.2

# Attack confidence threshold tuning
ATK_CONF_MIN = 0.20
ATK_CONF_MAX = 0.80
ATK_CONF_STEPS = 61

# Class-weight modes
BIN_S1_W_MODE, BIN_S1_W_CAP = "sqrt", None
BIN_S2_W_MODE, BIN_S2_W_CAP = "none", None
ATK_S1_W_MODE, ATK_S1_W_CAP = "sqrt", 8.0
ATK_S2_W_MODE, ATK_S2_W_CAP = "sqrt", 8.0
BIN_S2_NORMAL_WEIGHT_BOOST = 1.0
BIN_S2_FREEZE_BLOCK3 = False
BIN_S2_LR_NEW = 1e-3
BIN_S2_BLOCK3_LR_FACTOR = 0.10


# =========================
# 1) Labels
# =========================
CLASS_NAMES_5 = ["Normal", "Dos", "Probe", "U2R", "R2L"]
CLASS_TO_ID_5 = {c: i for i, c in enumerate(CLASS_NAMES_5)}
NORMAL_ID = CLASS_TO_ID_5["Normal"]

ATTACK_NAMES_4 = ["Dos", "Probe", "U2R", "R2L"]
FIVE_TO_ATTACK4 = {
    CLASS_TO_ID_5["Dos"]: 0,
    CLASS_TO_ID_5["Probe"]: 1,
    CLASS_TO_ID_5["U2R"]: 2,
    CLASS_TO_ID_5["R2L"]: 3,
}
ATTACK4_TO_FIVE = np.array(
    [
        CLASS_TO_ID_5["Dos"],
        CLASS_TO_ID_5["Probe"],
        CLASS_TO_ID_5["U2R"],
        CLASS_TO_ID_5["R2L"],
    ],
    dtype=np.int64,
)


# =========================
# 2) Data prep (same as Part-1)
# =========================
BASE_DIR = Path(__file__).resolve().parent
train_file = BASE_DIR / "Data" / "KDDTrain+.txt"
test_file = BASE_DIR / "Data" / "KDDTest+.txt"
test21_file = BASE_DIR / "Data" / "KDDTest-21.txt"

train_df = pd.read_csv(train_file, header=None)
test_df = pd.read_csv(test_file, header=None)
test21_df = pd.read_csv(test21_file, header=None)

columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "target",
    "level",
]
for df in (train_df, test_df, test21_df):
    df.columns = columns
    df.fillna(0, inplace=True)


def macro_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df["target"].astype(str).str.strip()
    mapping = {
        **{
            k: "Dos"
            for k in [
                "apache2",
                "back",
                "land",
                "neptune",
                "mailbomb",
                "pod",
                "processtable",
                "smurf",
                "teardrop",
                "udpstorm",
                "worm",
            ]
        },
        **{
            k: "R2L"
            for k in [
                "ftp_write",
                "guess_passwd",
                "httptunnel",
                "imap",
                "multihop",
                "named",
                "phf",
                "sendmail",
                "snmpgetattack",
                "snmpguess",
                "spy",
                "warezclient",
                "warezmaster",
                "xlock",
                "xsnoop",
            ]
        },
        **{k: "Probe" for k in ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"]},
        **{
            k: "U2R"
            for k in ["buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm"]
        },
        "normal": "Normal",
    }
    df["target"] = df["target"].replace(mapping)
    return df


train_df = macro_target(train_df)
test_df = macro_target(test_df)
test21_df = macro_target(test21_df)

print("\n[Train] macro label counts:\n", train_df["target"].value_counts())
print("\n[Test]  macro label counts:\n", test_df["target"].value_counts())
print("\n[Test-21] macro label counts:\n", test21_df["target"].value_counts())


def to_y_5(df: pd.DataFrame) -> np.ndarray:
    y = df["target"].astype(str).str.strip().map(CLASS_TO_ID_5)
    if y.isna().any():
        raise ValueError(f"Unknown labels: {df.loc[y.isna(), 'target'].unique()}")
    return y.astype(np.int64).to_numpy()


y_train_5 = to_y_5(train_df)
y_test_5 = to_y_5(test_df)
y_test21_5 = to_y_5(test21_df)

svc_le = LabelEncoder()
svc_le.fit(pd.concat([train_df["service"], test_df["service"], test21_df["service"]], axis=0).astype(str))
for df in (train_df, test_df, test21_df):
    df["service"] = svc_le.transform(df["service"].astype(str))

all_df = pd.concat([train_df, test_df, test21_df], axis=0, ignore_index=True)
all_df = pd.get_dummies(all_df, columns=["protocol_type", "flag"])

n_tr = len(train_df)
n_te = len(test_df)
train_enc = all_df.iloc[:n_tr].copy()
test_enc = all_df.iloc[n_tr : n_tr + n_te].copy()
test21_enc = all_df.iloc[n_tr + n_te :].copy()


def to_X(df_enc: pd.DataFrame) -> np.ndarray:
    return df_enc.drop(columns=["target", "level"]).to_numpy(dtype=np.float32)


X_train = to_X(train_enc)
X_test = to_X(test_enc)
X_test21 = to_X(test21_enc)
print("\nFeature dim:", X_train.shape[1])

X_tr_b, X_val_b, y_tr_b_5, y_val_b_5 = train_test_split(
    X_train, y_train_5, test_size=0.20, random_state=SEED, stratify=y_train_5
)

scaler = MinMaxScaler()
scaler.fit(X_tr_b)

X_tr_b_s = scaler.transform(X_tr_b).astype(np.float32)
X_val_b_s = scaler.transform(X_val_b).astype(np.float32)
X_train_s = scaler.transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)
X_test21_s = scaler.transform(X_test21).astype(np.float32)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
    n = X.shape[0]
    if n < seq_len:
        raise ValueError(f"N={n} < seq_len={seq_len}")
    xs, ys = [], []
    for s in range(0, n - seq_len + 1, stride):
        e = s + seq_len
        xs.append(X[s:e])
        ys.append(y[e - 1])
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


X_tr_b_seq, y_tr_b_seq_5 = make_sequences(X_tr_b_s, y_tr_b_5, SEQ_LEN, STRIDE)
X_val_b_seq, y_val_b_seq_5 = make_sequences(X_val_b_s, y_val_b_5, SEQ_LEN, STRIDE)

X_train_seq, y_train_seq_5 = make_sequences(X_train_s, y_train_5, SEQ_LEN, STRIDE)
X_test_seq, y_test_seq_5 = make_sequences(X_test_s, y_test_5, SEQ_LEN, STRIDE)
X_test21_seq, y_test21_seq_5 = make_sequences(X_test21_s, y_test21_5, SEQ_LEN, STRIDE)

print("\nStage-1 seq shapes | tr_b:", X_tr_b_seq.shape, "| val_b:", X_val_b_seq.shape)

X_tr_t, X_val_t, y_tr_t_5, y_val_t_5 = train_test_split(
    X_test_seq, y_test_seq_5, test_size=0.20, random_state=SEED, stratify=y_test_seq_5
)
print("Stage-2 seq shapes | tr_t:", X_tr_t.shape, "| val_t:", X_val_t.shape)


def to_binary(y5: np.ndarray) -> np.ndarray:
    return (y5 != NORMAL_ID).astype(np.int64)


def filter_attack(X_seq: np.ndarray, y5: np.ndarray):
    mask = y5 != NORMAL_ID
    X_atk = X_seq[mask]
    y_atk_4 = np.array([FIVE_TO_ATTACK4[int(v)] for v in y5[mask]], dtype=np.int64)
    return X_atk, y_atk_4


y_tr_b_seq_bin = to_binary(y_tr_b_seq_5)
y_val_b_seq_bin = to_binary(y_val_b_seq_5)
y_train_seq_bin = to_binary(y_train_seq_5)
y_test_seq_bin = to_binary(y_test_seq_5)
y_test21_seq_bin = to_binary(y_test21_seq_5)
y_tr_t_bin = to_binary(y_tr_t_5)
y_val_t_bin = to_binary(y_val_t_5)

X_tr_b_atk, y_tr_b_atk_4 = filter_attack(X_tr_b_seq, y_tr_b_seq_5)
X_val_b_atk, y_val_b_atk_4 = filter_attack(X_val_b_seq, y_val_b_seq_5)
X_tr_t_atk, y_tr_t_atk_4 = filter_attack(X_tr_t, y_tr_t_5)
X_val_t_atk, y_val_t_atk_4 = filter_attack(X_val_t, y_val_t_5)


# =========================
# 3) Dataset / loaders
# =========================
class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].transpose(0, 1), self.y[idx]  # (F, L), label


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    return DataLoader(SeqDataset(X, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


# Binary loaders
bin_s1_train = make_loader(X_tr_b_seq, y_tr_b_seq_bin, BATCH_B, True)
bin_s1_val = make_loader(X_val_b_seq, y_val_b_seq_bin, BATCH_B, False)
bin_s2_train = make_loader(X_tr_t, y_tr_t_bin, BATCH_T, True)
bin_s2_val = make_loader(X_val_t, y_val_t_bin, BATCH_T, False)

# Attack-4 loaders
atk_s1_train = make_loader(X_tr_b_atk, y_tr_b_atk_4, BATCH_B, True)
atk_s1_val = make_loader(X_val_b_atk, y_val_b_atk_4, BATCH_B, False)
atk_s2_train = make_loader(X_tr_t_atk, y_tr_t_atk_4, BATCH_T, True)
atk_s2_val = make_loader(X_val_t_atk, y_val_t_atk_4, BATCH_T, False)

# Evaluation loaders on full 5-class labels
eval_train_5 = make_loader(X_train_seq, y_train_seq_5, BATCH_B, False)
eval_test_5 = make_loader(X_test_seq, y_test_seq_5, BATCH_B, False)
eval_test21_5 = make_loader(X_test21_seq, y_test21_seq_5, BATCH_B, False)
eval_val_t_5 = make_loader(X_val_t, y_val_t_5, BATCH_T, False)


# =========================
# 4) Weight utils
# =========================
def compute_class_weights(
    y: np.ndarray, n_classes: int, label: str = "", mode: str = "sqrt", max_ratio: float = None
) -> torch.Tensor:
    if mode == "none":
        w = np.ones(n_classes, dtype=np.float32)
    else:
        counts = np.bincount(y, minlength=n_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        w = np.sqrt(1.0 / counts) if mode == "sqrt" else (1.0 / counts)
        if max_ratio is not None:
            w = np.minimum(w, w.min() * max_ratio)
        w = w / w.sum() * n_classes

    if label:
        counts = np.bincount(y, minlength=n_classes).astype(np.int64)
        print(f"\n[Class Weights ({mode}, cap={max_ratio}) - {label}]")
        for i in range(n_classes):
            print(f"  class-{i:<2d}: count={int(counts[i]):6d}, weight={float(w[i]):.4f}")
    return torch.tensor(w, dtype=torch.float32)


cw_bin_s1 = compute_class_weights(y_tr_b_seq_bin, 2, "BIN Stage-1", BIN_S1_W_MODE, BIN_S1_W_CAP)
cw_bin_s2 = compute_class_weights(y_tr_t_bin, 2, "BIN Stage-2", BIN_S2_W_MODE, BIN_S2_W_CAP)
cw_atk_s1 = compute_class_weights(y_tr_b_atk_4, 4, "ATK4 Stage-1", ATK_S1_W_MODE, ATK_S1_W_CAP)
cw_atk_s2 = compute_class_weights(y_tr_t_atk_4, 4, "ATK4 Stage-2", ATK_S2_W_MODE, ATK_S2_W_CAP)

if BIN_S2_NORMAL_WEIGHT_BOOST != 1.0:
    w = cw_bin_s2.numpy().copy()
    w[0] *= BIN_S2_NORMAL_WEIGHT_BOOST
    w = w / w.sum() * 2.0
    cw_bin_s2 = torch.tensor(w, dtype=torch.float32)
    print(f"\n[Class Weights (BIN Stage-2 boosted normal x{BIN_S2_NORMAL_WEIGHT_BOOST:.2f})]")
    print(f"  class-0(normal): weight={w[0]:.4f}")
    print(f"  class-1(attack): weight={w[1]:.4f}")


# =========================
# 5) Models
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        if x.shape[2] <= 1:
            return x
        w = torch.softmax(self.score(x), dim=2)
        return x * w


class CNNB(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=3, padding=1),
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
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=2)
        return self.fc(x)


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

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class CNNTL(nn.Module):
    def __init__(self, cnnb: CNNB, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.cnnb_block1 = cnnb.block1
        self.cnnb_block2 = cnnb.block2
        self.cnnb_block3 = cnnb.block3
        self.cnnt = CNNT(in_ch=256, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.cnnb_block1(x)
        x = self.cnnb_block2(x)
        x = self.cnnb_block3(x)
        x = self.cnnt(x)
        x = x.mean(dim=2)
        return self.fc(x)


IN_CH = X_tr_b_seq.shape[2]
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def save_ckpt(path: Path, mdl: nn.Module, opt, sch, epoch: int, val_loss: float):
    torch.save(
        {
            "model_state_dict": mdl.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sch.state_dict() if sch else {},
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
    print(f"  Saved: {path.name}")


@torch.no_grad()
def eval_loss(mdl: nn.Module, loader: DataLoader, criterion: nn.Module):
    mdl.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(mdl(xb), yb)
        total += loss.item() * yb.size(0)
        n += yb.size(0)
    return total / max(1, n)


def train_stage1(
    task_name: str,
    n_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    ckpt_prefix: str,
):
    model = CNNB(IN_CH, n_classes, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_B)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_B, eta_min=1e-4)

    print(f"\n{'='*72}\n  STAGE-1 {task_name}: CNN-B | classes={n_classes}\n{'='*72}")
    best_loss = float("inf")
    ckpt_best = CKPT_DIR / f"{ckpt_prefix}_s1_best.ckpt"
    ckpt_last = CKPT_DIR / f"{ckpt_prefix}_s1_last.ckpt"

    for ep in range(1, EPOCHS_B + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item() * yb.size(0)
            n += yb.size(0)
        scheduler.step()

        tr_loss = total / max(1, n)
        va_loss = eval_loss(model, val_loader, criterion)
        if va_loss < best_loss:
            best_loss = va_loss
            save_ckpt(ckpt_best, model, optimizer, scheduler, ep, va_loss)
            tag = " *best"
        else:
            tag = ""
        print(f"[{task_name} S1] Ep {ep:02d} | train {tr_loss:.6f} | val {va_loss:.6f}{tag}")

    save_ckpt(ckpt_last, model, optimizer, scheduler, EPOCHS_B, va_loss)
    ckpt = torch.load(ckpt_best, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def train_stage2(
    task_name: str,
    cnnb: CNNB,
    n_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    ckpt_prefix: str,
    lr_new: float = LR_T,
    block3_lr_factor: float = 0.1,
    freeze_block3: bool = False,
):
    model = CNNTL(cnnb=cnnb, n_classes=n_classes, dropout=DROPOUT).to(device)
    for p in model.cnnb_block1.parameters():
        p.requires_grad = False
    for p in model.cnnb_block2.parameters():
        p.requires_grad = False
    if freeze_block3:
        for p in model.cnnb_block3.parameters():
            p.requires_grad = False

    block3_params = [p for p in model.cnnb_block3.parameters() if p.requires_grad]
    new_params = list(model.cnnt.parameters()) + list(model.fc.parameters())
    trainable_params = block3_params + new_params
    param_groups = [{"params": new_params, "lr": lr_new}]
    if block3_params:
        param_groups.insert(0, {"params": block3_params, "lr": lr_new * block3_lr_factor})

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=S2_LABEL_SMOOTH)
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_T, eta_min=1e-6)

    print(f"\n{'='*72}\n  STAGE-2 {task_name}: CNN-TL | classes={n_classes}\n{'='*72}")
    best_loss = float("inf")
    patience_count = 0
    stop_epoch = EPOCHS_T
    ckpt_best = CKPT_DIR / f"{ckpt_prefix}_s2_best.ckpt"
    ckpt_last = CKPT_DIR / f"{ckpt_prefix}_s2_last.ckpt"

    for ep in range(1, EPOCHS_T + 1):
        model.train()
        model.cnnb_block1.eval()
        model.cnnb_block2.eval()
        if freeze_block3:
            model.cnnb_block3.eval()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if trainable_params:
                nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optimizer.step()
            total += loss.item() * yb.size(0)
            n += yb.size(0)
        scheduler.step()

        tr_loss = total / max(1, n)
        va_loss = eval_loss(model, val_loader, criterion)
        is_best = va_loss < best_loss
        if is_best:
            best_loss = va_loss
            patience_count = 0
            save_ckpt(ckpt_best, model, optimizer, scheduler, ep, va_loss)
        else:
            patience_count += 1

        if ep <= 10 or ep % 50 == 0 or is_best:
            tag = " *best" if is_best else ""
            print(
                f"[{task_name} S2] Ep {ep:04d} | train {tr_loss:.6f} | val {va_loss:.6f} "
                f"| patience {patience_count:3d}{tag}"
            )

        if patience_count >= PATIENCE:
            stop_epoch = ep
            print(f"Early stopping at ep={ep}, best_val={best_loss:.6f}")
            break

    save_ckpt(ckpt_last, model, optimizer, scheduler, stop_epoch, va_loss)
    ckpt = torch.load(ckpt_best, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, best_loss


# =========================
# 6) Train binary gate + attack-4 classifier
# =========================
bin_cnnb = train_stage1("BIN", 2, bin_s1_train, bin_s1_val, cw_bin_s1, "hier_bin")
bin_cnntl, bin_best = train_stage2(
    "BIN",
    bin_cnnb,
    2,
    bin_s2_train,
    bin_s2_val,
    cw_bin_s2,
    "hier_bin",
    lr_new=BIN_S2_LR_NEW,
    block3_lr_factor=BIN_S2_BLOCK3_LR_FACTOR,
    freeze_block3=BIN_S2_FREEZE_BLOCK3,
)
print(f"\n[BIN] Stage-2 best val loss: {bin_best:.6f}")

atk_cnnb = train_stage1("ATK4", 4, atk_s1_train, atk_s1_val, cw_atk_s1, "hier_atk4")
atk_cnntl, atk_best = train_stage2("ATK4", atk_cnnb, 4, atk_s2_train, atk_s2_val, cw_atk_s2, "hier_atk4")
print(f"\n[ATK4] Stage-2 best val loss: {atk_best:.6f}")


# =========================
# 7) Metrics
# =========================
def metrics_overall(y_true: np.ndarray, y_pred: np.ndarray, normal_id: int = 0):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    acc = float((y_true == y_pred).mean())
    ta = y_true != normal_id
    tn_mask = y_true == normal_id
    dr = float(((y_pred != normal_id) & ta).sum() / (ta.sum() + 1e-12))
    fpr = float(((y_pred != normal_id) & tn_mask).sum() / (tn_mask.sum() + 1e-12))
    tn = int(((y_true == normal_id) & (y_pred == normal_id)).sum())
    fp = int(((y_true == normal_id) & (y_pred != normal_id)).sum())
    fn = int(((y_true != normal_id) & (y_pred == normal_id)).sum())
    tp = int(((y_true != normal_id) & (y_pred != normal_id)).sum())
    return {"DR": dr, "ACC": acc, "FPR": fpr, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    n = len(y_true)
    out = []
    for c in range(n_classes):
        tc = y_true == c
        pc = y_pred == c
        tp = int((tc & pc).sum())
        fn = int((tc & ~pc).sum())
        fp = int((~tc & pc).sum())
        tn = int((~tc & ~pc).sum())
        out.append(
            {
                "class_id": c,
                "DR": tp / (tp + fn + 1e-12),
                "ACC": (tp + tn) / (n + 1e-12),
                "FPR": fp / (fp + tn + 1e-12),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "support": tp + fn,
            }
        )
    return out


@torch.no_grad()
def collect_hier_scores(gate_model: nn.Module, atk_model: nn.Module, loader: DataLoader):
    gate_model.eval()
    atk_model.eval()
    all_true5, all_gate_attack_prob, all_atk_pred4, all_atk_conf = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        gate_probs = torch.softmax(gate_model(xb), dim=1)
        atk_probs = torch.softmax(atk_model(xb), dim=1)
        atk_conf, atk_pred4 = torch.max(atk_probs, dim=1)

        all_true5.append(yb.numpy().astype(np.int64))
        all_gate_attack_prob.append(gate_probs[:, 1].cpu().numpy().astype(np.float32))
        all_atk_pred4.append(atk_pred4.cpu().numpy().astype(np.int64))
        all_atk_conf.append(atk_conf.cpu().numpy().astype(np.float32))

    return (
        np.concatenate(all_true5),
        np.concatenate(all_gate_attack_prob),
        np.concatenate(all_atk_pred4),
        np.concatenate(all_atk_conf),
    )


def hierarchical_pred_from_scores(
    gate_attack_prob: np.ndarray,
    atk_pred4: np.ndarray,
    atk_conf: np.ndarray,
    gate_thr: float,
    atk_conf_thr: float,
):
    y_pred5 = np.full_like(atk_pred4, NORMAL_ID, dtype=np.int64)
    promote = (gate_attack_prob >= gate_thr) & (atk_conf >= atk_conf_thr)
    if promote.any():
        y_pred5[promote] = ATTACK4_TO_FIVE[atk_pred4[promote]]
    return y_pred5


def tune_hier_thresholds(
    gate_model: nn.Module,
    atk_model: nn.Module,
    loader: DataLoader,
    target_fpr: float = 0.045,
    target_dr: float = 0.865,
    topk: int = 8,
    relax_fpr: float = 0.012,
    relax_dr: float = 0.012,
):
    y_true5, gate_attack_prob, atk_pred4, atk_conf = collect_hier_scores(gate_model, atk_model, loader)

    gate_grid = np.linspace(GATE_THR_MIN, GATE_THR_MAX, GATE_THR_STEPS)
    atk_grid = np.linspace(ATK_CONF_MIN, ATK_CONF_MAX, ATK_CONF_STEPS)
    candidates = []

    best_g, best_a, best_m, best_score = 0.5, 0.5, None, float("inf")
    for g in gate_grid:
        for a in atk_grid:
            y_pred5 = hierarchical_pred_from_scores(gate_attack_prob, atk_pred4, atk_conf, float(g), float(a))
            m = metrics_overall(y_true5, y_pred5, normal_id=NORMAL_ID)
            dr_under = max(0.0, target_dr - m["DR"])
            score = (
                (GATE_SCORE_W_FPR * abs(m["FPR"] - target_fpr))
                + (GATE_SCORE_W_DR * abs(m["DR"] - target_dr))
                + (GATE_SCORE_W_DR_UNDER * dr_under)
            )
            candidates.append((float(g), float(a), m, float(score)))
            if score < best_score:
                best_g, best_a, best_m, best_score = float(g), float(a), m, float(score)

    feasible = [
        (g, a, m, s)
        for (g, a, m, s) in candidates
        if (m["FPR"] <= target_fpr + relax_fpr) and (m["DR"] >= target_dr - relax_dr)
    ]
    if feasible:
        feasible.sort(
            key=lambda x: (x[3], -x[2]["ACC"], abs(x[2]["FPR"] - target_fpr), abs(x[2]["DR"] - target_dr))
        )
        best_g, best_a, best_m, best_score = feasible[0]
    else:
        # If no feasible point exists, minimize DR shortfall first, then FPR overflow.
        candidates.sort(
            key=lambda x: (
                max(0.0, target_dr - x[2]["DR"]),
                max(0.0, x[2]["FPR"] - target_fpr),
                x[3],
                -x[2]["ACC"],
            )
        )
        best_g, best_a, best_m, best_score = candidates[0]

    top = sorted(candidates, key=lambda x: x[3])[: max(1, topk)]
    print("\n[Hier Threshold Tuning on Val (gate + atk_conf)]")
    print(
        f"  chosen_gate_threshold={best_g:.4f} | chosen_atk_conf_threshold={best_a:.4f} | "
        f"chosen_score={best_score:.5f}"
    )
    print(f"  val_DR={best_m['DR']*100:.2f}% | val_FPR={best_m['FPR']*100:.2f}% | val_ACC={best_m['ACC']*100:.2f}%")
    print(f"  target_DR={target_dr*100:.2f}% | target_FPR={target_fpr*100:.2f}%")
    print("  top-threshold candidates:")
    for g, a, m, s in top:
        print(
            f"    g={g:.4f} | a={a:.4f} | score={s:.5f} | "
            f"DR={m['DR']*100:.2f}% | FPR={m['FPR']*100:.2f}% | ACC={m['ACC']*100:.2f}%"
        )
    return best_g, best_a, best_m


@torch.no_grad()
def predict_hierarchical_5(
    gate_model: nn.Module, atk_model: nn.Module, loader: DataLoader, gate_thr: float, atk_conf_thr: float
):
    gate_model.eval()
    atk_model.eval()
    trues, preds = [], []
    map_tensor = torch.tensor(ATTACK4_TO_FIVE, dtype=torch.long, device=device)

    for xb, yb in loader:
        xb = xb.to(device)
        gate_probs = torch.softmax(gate_model(xb), dim=1)
        gate_attack = gate_probs[:, 1] >= gate_thr

        yhat5 = torch.full((xb.size(0),), NORMAL_ID, dtype=torch.long, device=device)
        if gate_attack.any():
            atk_probs = torch.softmax(atk_model(xb[gate_attack]), dim=1)
            atk_conf, atk_pred4 = torch.max(atk_probs, dim=1)
            promote_idx = torch.nonzero(gate_attack, as_tuple=False).squeeze(1)
            promote_mask = atk_conf >= atk_conf_thr
            if promote_mask.any():
                yhat5[promote_idx[promote_mask]] = map_tensor[atk_pred4[promote_mask]]

        trues.append(yb.numpy())
        preds.append(yhat5.cpu().numpy())

    return np.concatenate(trues), np.concatenate(preds)


def print_table(ds_name: str, per_class: list, overall: dict):
    print("\n" + "=" * 82)
    print(f"  Dataset: {ds_name}   Model: Hierarchical CNN-TL")
    print("=" * 82)
    print(f"{'Class':<10} {'DR':>10} {'ACC':>10} {'FPR':>10} {'TP':>7} {'FN':>7} {'FP':>7} {'TN':>7}")
    print("-" * 82)
    for m in per_class:
        cname = CLASS_NAMES_5[m["class_id"]]
        print(
            f"{cname:<10} {m['DR']*100:9.2f}% {m['ACC']*100:9.2f}% {m['FPR']*100:9.2f}% "
            f"{m['tp']:7d} {m['fn']:7d} {m['fp']:7d} {m['tn']:7d}"
        )
    print("-" * 82)
    print(
        f"{'OVERALL':<10} {overall['DR']*100:9.2f}% {overall['ACC']*100:9.2f}% {overall['FPR']*100:9.2f}% "
        f"{overall['tp']:7d} {overall['fn']:7d} {overall['fp']:7d} {overall['tn']:7d}"
    )
    print("=" * 82)


def eval_hier_dataset(
    ds_name: str,
    loader: DataLoader,
    gate_model: nn.Module,
    atk_model: nn.Module,
    gate_thr: float,
    atk_conf_thr: float,
):
    y_true, y_pred = predict_hierarchical_5(gate_model, atk_model, loader, gate_thr, atk_conf_thr)
    overall = metrics_overall(y_true, y_pred, normal_id=NORMAL_ID)
    per_class = metrics_per_class(y_true, y_pred, len(CLASS_NAMES_5))
    print_table(ds_name, per_class, overall)
    return overall, per_class


# =========================
# 8) Evaluation + report
# =========================
gate_thr, atk_conf_thr, gate_val = tune_hier_thresholds(
    bin_cnntl,
    atk_cnntl,
    eval_val_t_5,
    target_fpr=GATE_TARGET_FPR,
    target_dr=GATE_TARGET_DR,
    topk=GATE_TUNE_TOPK,
    relax_fpr=GATE_TUNE_RELAX_FPR,
    relax_dr=GATE_TUNE_RELAX_DR,
)
print(f"[Hier] Apply thresholds: gate={gate_thr:.4f}, atk_conf={atk_conf_thr:.4f}")

train_overall, _ = eval_hier_dataset("KDDTrain+", eval_train_5, bin_cnntl, atk_cnntl, gate_thr, atk_conf_thr)
test_overall, _ = eval_hier_dataset("KDDTest+", eval_test_5, bin_cnntl, atk_cnntl, gate_thr, atk_conf_thr)
test21_overall, _ = eval_hier_dataset("KDDTest-21", eval_test21_5, bin_cnntl, atk_cnntl, gate_thr, atk_conf_thr)

baseline_stage2 = None
if BASELINE_REPORT.exists():
    with open(BASELINE_REPORT, "r", encoding="utf-8") as f:
        baseline_json = json.load(f)
    baseline_stage2 = baseline_json.get("stage2", {}).get("overall")

if baseline_stage2:
    print("\n[Delta vs Part-1 baseline (stage2 overall)]")
    for ds, ours in [("KDDTrain+", train_overall), ("KDDTest+", test_overall), ("KDDTest-21", test21_overall)]:
        base = baseline_stage2.get(ds, {})
        if base:
            d_dr = (ours["DR"] - base["DR"]) * 100.0
            d_acc = (ours["ACC"] - base["ACC"]) * 100.0
            d_fpr = (ours["FPR"] - base["FPR"]) * 100.0
            print(f"  {ds:<10} dDR={d_dr:+6.2f}% | dACC={d_acc:+6.2f}% | dFPR={d_fpr:+6.2f}%")

if SAVE_RUN_REPORT:
    report = {
        "run_tag": RUN_TAG,
        "seed": SEED,
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "part": "hierarchical_cnn_tl",
        "binary_gate": {
            "stage2_best_val_loss": float(bin_best),
            "stage2_setup": {
                "weight_mode": BIN_S2_W_MODE,
                "normal_weight_boost": BIN_S2_NORMAL_WEIGHT_BOOST,
                "freeze_block3": BIN_S2_FREEZE_BLOCK3,
                "lr_new": BIN_S2_LR_NEW,
                "block3_lr_factor": BIN_S2_BLOCK3_LR_FACTOR,
            },
            "gate_threshold": float(gate_thr),
            "attack_conf_threshold": float(atk_conf_thr),
            "threshold_targets": {"fpr": GATE_TARGET_FPR, "dr": GATE_TARGET_DR},
            "threshold_search_space": {
                "gate_thr_min": GATE_THR_MIN,
                "gate_thr_max": GATE_THR_MAX,
                "gate_thr_steps": GATE_THR_STEPS,
                "atk_conf_min": ATK_CONF_MIN,
                "atk_conf_max": ATK_CONF_MAX,
                "atk_conf_steps": ATK_CONF_STEPS,
                "score_weights": {
                    "fpr": GATE_SCORE_W_FPR,
                    "dr": GATE_SCORE_W_DR,
                    "dr_under": GATE_SCORE_W_DR_UNDER,
                },
            },
            "threshold_val_metrics_overall_5class_binary_view": gate_val,
        },
        "attack4_classifier": {
            "stage2_best_val_loss": float(atk_best),
            "classes": ATTACK_NAMES_4,
        },
        "overall_5class": {
            "KDDTrain+": train_overall,
            "KDDTest+": test_overall,
            "KDDTest-21": test21_overall,
        },
        "baseline_part1_stage2_overall": baseline_stage2,
    }
    out_path = CKPT_DIR / f"report_{RUN_TAG}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Report] Saved: {out_path}")
