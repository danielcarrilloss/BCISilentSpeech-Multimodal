import torch
import time
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#                        MODULES
# ============================================================

class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * w, dim=1)


class EEGNet(nn.Module):
    def __init__(self, eeg_ch=32, embed_dim=128, dropout=0.25):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (eeg_ch, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B,T,C) → (B,1,C,T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.mean(dim=-1).squeeze(-1)
        return self.fc(x)

class EEGEncoder(nn.Module):
    def __init__(self, eeg_ch=32, embed_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(eeg_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, T, C]
        x = x.permute(0, 2, 1)   # → [B, C, T]
        x = self.net(x)          # → [B, D, T]

        x = x.mean(dim=-1)       # temporal pooling final
        return x

# MODELS
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        out, weights = self.attn(query, context, context)
        return self.norm(out + query), weights


class TriModalFusion(nn.Module):
    def __init__(self, eeg_ch=32, emg_ch=5, imu_ch=36, embed_dim=128, num_classes=10):
        super().__init__()

        self.eeg_proj = nn.Linear(eeg_ch, embed_dim)
        self.emg_proj = nn.Linear(emg_ch, embed_dim)
        self.imu_proj = nn.Linear(imu_ch, embed_dim)

        self.pool = AttnPool(embed_dim)
        self.fusion_layer = CrossAttentionFusion(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_eeg, x_emg, x_imu):
        z_eeg = self.pool(self.eeg_proj(x_eeg))
        z_emg = self.pool(self.emg_proj(x_emg))
        z_imu = self.pool(self.imu_proj(x_imu))

        z_peripheral = z_emg + z_imu
        fused, _ = self.fusion_layer(z_eeg.unsqueeze(1), z_peripheral.unsqueeze(1))

        return self.classifier(fused.squeeze(1))

class EarlyFusion(nn.Module):
    def __init__(self, eeg_ch=32, emg_ch=5, imu_ch=36, embed_dim=128, num_classes=10):
        super().__init__()

        self.eeg_proj = nn.Linear(eeg_ch, embed_dim)
        self.emg_proj = nn.Linear(emg_ch, embed_dim)
        self.imu_proj = nn.Linear(imu_ch, embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_eeg, x_emg, x_imu):
        z_eeg = self.eeg_proj(x_eeg).mean(dim=1)
        z_emg = self.emg_proj(x_emg).mean(dim=1)
        z_imu = self.imu_proj(x_imu).mean(dim=1)

        fused = torch.cat([z_eeg, z_emg, z_imu], dim=1)
        return self.classifier(fused)
    

class SpectogramEEGEncoder(nn.Module):
    """
    Processes time-frequency representations with 2D convolutions.
    Much better at capturing speech-relevant spectral patterns.
    """
    def __init__(self, n_channels=32, n_freqs=22, embed_dim=128):
        super().__init__()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, (1,1)),
            nn.Sigmoid()
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(128, embed_dim)
    
    def forward(self, x):
        # x: (B, Freq, Time, Channels)

        # Spatial attention
        attn = self.spatial_attention(x)
        x = x * attn

        x = self.conv_block(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.fc(x)








# EVALUATION
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    all_weights = []

    with torch.no_grad():
        for b_eeg, b_emg, b_imu, b_y in test_loader:
            b_eeg = b_eeg.to(device)
            b_emg = b_emg.to(device)
            b_imu = b_imu.to(device)
            b_y   = b_y.to(device)

            outputs, weights = model(b_eeg, b_emg, b_imu, return_weights=True)

            all_weights.append(weights.cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()

    acc = correct / total

    print("\nAverage modality weights:")
    print(f"EEG: {all_weights[:,0].mean():.3f}")
    print(f"EMG: {all_weights[:,1].mean():.3f}")
    print(f"IMU: {all_weights[:,2].mean():.3f}")

    return acc if total > 0 else 0

# ============================================================
#              UNIFIED TRAIN + EVALUATE
# ============================================================

def train_and_evaluate(split, has_band, model_class, model_args,
                       mode="all", epochs=70, batch_size=32,
                       lr=1e-3, augment=True, patience=10):
    """
    Single function for training and evaluating one fold.
    Works for LOSO, within-subject CV, and ablation — all the same.
    """

    # --- NORMALIZE ---
    tr_eeg, te_eeg = normalize(split['train_eeg'], split['test_eeg'])
    tr_emg, te_emg = normalize(split['train_emg'], split['test_emg'])
    tr_imu, te_imu = normalize(split['train_imu'], split['test_imu'])
    tr_y, te_y = split['train_labels'], split['test_labels']

    tr_band, te_band = None, None
    if has_band:
        tr_band, te_band = normalize_1d(split['train_band'], split['test_band'])

    # --- AUGMENT ---
    if augment:
        aug = augment_data(tr_eeg, tr_emg, tr_imu, tr_y,
                           band=tr_band if has_band else None)
        tr_eeg = np.concatenate([tr_eeg, aug['eeg']])
        tr_emg = np.concatenate([tr_emg, aug['emg']])
        tr_imu = np.concatenate([tr_imu, aug['imu']])
        tr_y   = np.concatenate([tr_y,   aug['y']])
        if has_band:
            tr_band = np.concatenate([tr_band, aug['band']])

    # --- DATALOADERS ---
    train_tensors = [
        torch.FloatTensor(tr_eeg),
        torch.FloatTensor(tr_emg),
        torch.FloatTensor(tr_imu),
    ]
    test_tensors = [
        torch.FloatTensor(te_eeg),
        torch.FloatTensor(te_emg),
        torch.FloatTensor(te_imu),
    ]

    if has_band:
        train_tensors.append(torch.FloatTensor(tr_band))
        test_tensors.append(torch.FloatTensor(te_band))

    train_tensors.append(torch.LongTensor(tr_y))
    test_tensors.append(torch.LongTensor(te_y))

    train_loader = DataLoader(TensorDataset(*train_tensors),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True)
    test_loader  = DataLoader(TensorDataset(*test_tensors),
                              batch_size=64)

    # --- MODEL ---
    model = model_class(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- TRAIN ---
    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            if has_band:
                b_eeg, b_emg, b_imu, b_band, b_y = [b.to(device) for b in batch]
            else:
                b_eeg, b_emg, b_imu, b_y = [b.to(device) for b in batch]
                b_band = None

            # Determine what to feed based on mode
            feed = _build_feed(b_eeg, b_emg, b_imu, b_band, mode)

            optimizer.zero_grad()
            outputs = model(**feed, mode=mode)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # --- EVAL every 5 epochs ---
        if (epoch + 1) % 5 == 0:
            acc, weights = _evaluate_loader(model, test_loader, has_band, mode)
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

    # --- FINAL EVAL ---
    final_acc, final_weights = _evaluate_loader(model, test_loader, has_band, mode)
    best_acc = max(best_acc, final_acc)

    return best_acc, final_weights, model


def _build_feed(b_eeg, b_emg, b_imu, b_band, mode):
    """Build the keyword arguments for model.forward() based on mode."""
    feed = {}
    if mode in ("all", "eeg"):
        feed['x_eeg'] = b_eeg
        feed['x_band'] = b_band
    if mode in ("all", "emg"):
        feed['x_emg'] = b_emg
    if mode in ("all", "imu"):
        feed['x_imu'] = b_imu
    # Dual-modal
    if mode == "eeg+emg":
        feed = {'x_eeg': b_eeg, 'x_emg': b_emg, 'x_band': b_band}
    elif mode == "eeg+imu":
        feed = {'x_eeg': b_eeg, 'x_imu': b_imu, 'x_band': b_band}
    elif mode == "emg+imu":
        feed = {'x_emg': b_emg, 'x_imu': b_imu}
    return feed


def _evaluate_loader(model, loader, has_band, mode):
    """Evaluate model on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    all_weights = []

    with torch.no_grad():
        for batch in loader:
            if has_band:
                b_eeg, b_emg, b_imu, b_band, b_y = [b.to(device) for b in batch]
            else:
                b_eeg, b_emg, b_imu, b_y = [b.to(device) for b in batch]
                b_band = None

            feed = _build_feed(b_eeg, b_emg, b_imu, b_band, mode)
            outputs = model(**feed, mode=mode, return_weights=True)

            if isinstance(outputs, tuple):
                out, w = outputs
                if w is not None:
                    all_weights.append(w.cpu().numpy())
            else:
                out = outputs

            _, pred = torch.max(out, 1)
            total += b_y.size(0)
            correct += (pred == b_y).sum().item()

    acc = correct / total if total > 0 else 0
    weights = np.concatenate(all_weights, axis=0) if all_weights else None
    return acc, weights

# ============================================================
#                     FUSION MODEL
# ============================================================

class GatedFusion(nn.Module):
    def __init__(self, eeg_ch=32, emg_ch=5, imu_ch=36,
                 embed_dim=128, num_classes=10,
                 eeg_band_dim=None):
        super().__init__()

        self.eeg_encoder = EEGNet(eeg_ch, embed_dim)
        self.emg_proj = nn.Linear(emg_ch, embed_dim)
        self.imu_proj = nn.Linear(imu_ch, embed_dim)
        self.pool = AttnPool(embed_dim)

        # Optional band-power stream
        self.use_band = eeg_band_dim is not None
        if self.use_band:
            self.band_proj = nn.Sequential(
                nn.Linear(eeg_band_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

        self.gated_network = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def encode(self, x_eeg=None, x_emg=None, x_imu=None, x_band=None):
        """Encode each modality independently. Returns dict of embeddings."""
        z = {}
        if x_eeg is not None:
            z['eeg'] = self.eeg_encoder(x_eeg)
            # Optionally fuse band-power into EEG embedding
            if self.use_band and x_band is not None:
                z['eeg'] = z['eeg'] + self.band_proj(x_band)
        if x_emg is not None:
            z['emg'] = self.pool(self.emg_proj(x_emg))
        if x_imu is not None:
            z['imu'] = self.pool(self.imu_proj(x_imu))
        return z

    def forward(self, x_eeg=None, x_emg=None, x_imu=None, x_band=None,
                mode="all", return_weights=False):
        """
        Unified forward pass. 'mode' controls which modalities are used.
          - "all"       : gated tri-modal fusion
          - "eeg"       : EEG-only
          - "emg"       : EMG-only
          - "imu"       : IMU-only
          - "eeg+emg"   : dual-modal (any pair)
        """
        z = self.encode(x_eeg, x_emg, x_imu, x_band)

        # ---- UNIMODAL ----
        if mode in z and mode != "all":
            out = self.classifier(z[mode])
            return (out, None) if return_weights else out

        # ---- FULL TRIMODAL ----
        if not all(k in z for k in ['eeg', 'emg', 'imu']):
            # Fallback: use whatever is available, average them
            available = list(z.values())
            fused = torch.stack(available, dim=0).mean(dim=0)
            out = self.classifier(fused)
            return (out, None) if return_weights else out

        combined = torch.cat([z['eeg'], z['emg'], z['imu']], dim=1)
        weights = torch.softmax(self.gated_network(combined), dim=1)
        alpha, beta, gamma = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        fused = alpha * z['eeg'] + beta * z['emg'] + gamma * z['imu']

        out = self.classifier(fused)
        return (out, weights) if return_weights else out


# ============================================================
#                      UTILITIES
# ============================================================

def normalize(train_data, test_data):
    N, T, C = train_data.shape
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_data.reshape(-1, C))
    test_flat  = scaler.transform(test_data.reshape(-1, C))
    return (train_flat.reshape(N, T, C),
            test_flat.reshape(test_data.shape[0], T, C))


def normalize_1d(train_data, test_data):
    """For flat feature vectors like band-power (Trials, Features)."""
    scaler = StandardScaler()
    return scaler.fit_transform(train_data), scaler.transform(test_data)


def augment_data(eeg, emg, imu, y, band=None, noise_std=0.05):
    """Gaussian noise augmentation."""
    n = len(eeg)
    aug = {
        'eeg': eeg + np.random.randn(*eeg.shape).astype(np.float32) * noise_std,
        'emg': emg + np.random.randn(*emg.shape).astype(np.float32) * noise_std,
        'imu': imu + np.random.randn(*imu.shape).astype(np.float32) * noise_std,
        'y':   y.copy()
    }
    if band is not None:
        aug['band'] = band + np.random.randn(*band.shape).astype(np.float32) * noise_std * 0.1
    return aug


# ============================================================
#                  SPLIT FUNCTIONS
# ============================================================

def build_splits_loso(all_data, test_sub):
    """Build train/test arrays for one LOSO fold."""
    subject_ids = sorted(list(all_data.keys()))
    train_subs = [s for s in subject_ids if s != test_sub]

    def concat(subs, key):
        return np.concatenate([ses[key] for sid in subs for ses in all_data[sid]])

    split = {}
    for key in ['eeg', 'emg', 'imu', 'labels']:
        split[f'train_{key}'] = concat(train_subs, key)
        split[f'test_{key}']  = concat([test_sub], key)

    # Optional band-power
    has_band = 'eeg_band' in all_data[test_sub][0]
    if has_band:
        split['train_band'] = concat(train_subs, 'eeg_band')
        split['test_band']  = concat([test_sub], 'eeg_band')

    return split, has_band


def build_splits_kfold(all_data, sub_id, train_idx, test_idx):
    """Build train/test arrays for one within-subject fold."""
    sessions = all_data[sub_id]

    eeg = np.concatenate([ses['eeg'] for ses in sessions])
    emg = np.concatenate([ses['emg'] for ses in sessions])
    imu = np.concatenate([ses['imu'] for ses in sessions])
    y   = np.concatenate([ses['labels'] for ses in sessions])

    has_band = 'eeg_band' in sessions[0]
    band = np.concatenate([ses['eeg_band'] for ses in sessions]) if has_band else None

    split = {
        'train_eeg': eeg[train_idx], 'test_eeg': eeg[test_idx],
        'train_emg': emg[train_idx], 'test_emg': emg[test_idx],
        'train_imu': imu[train_idx], 'test_imu': imu[test_idx],
        'train_labels': y[train_idx], 'test_labels': y[test_idx],
    }
    if has_band:
        split['train_band'] = band[train_idx]
        split['test_band']  = band[test_idx]

    return split, has_band

# ============================================================
#                  HIGH-LEVEL RUNNERS
# ============================================================

def run_loso(all_data, model_class, model_args, mode="all", **kwargs):
    """Leave-One-Subject-Out with any mode."""
    subject_ids = sorted(list(all_data.keys()))
    results = []

    for test_sub in subject_ids:
        print(f"\n=== LOSO | Test: Sub {test_sub} | Mode: {mode} ===")

        split, has_band = build_splits_loso(all_data, test_sub)
        acc, weights, _ = train_and_evaluate(
            split, has_band, model_class, model_args,
            mode=mode, **kwargs
        )

        print(f"  Sub {test_sub}: {acc:.3f}")
        if weights is not None:
            print(f"  Weights → EEG:{weights[:,0].mean():.3f} "
                  f"EMG:{weights[:,1].mean():.3f} "
                  f"IMU:{weights[:,2].mean():.3f}")

        results.append(acc)

    mean, std = np.mean(results), np.std(results)
    print(f"\n>>> LOSO {mode.upper()}: {mean:.3f} ± {std:.3f}")
    return mean, std, results


def run_within_subject(all_data, model_class, model_args,
                       mode="all", n_folds=5, **kwargs):
    """Within-subject stratified K-fold CV."""
    subject_ids = sorted(list(all_data.keys()))
    all_results = {}

    for sub_id in subject_ids:
        sessions = all_data[sub_id]
        y = np.concatenate([ses['labels'] for ses in sessions])

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_accs = []

        # We need the full array for indexing
        eeg_full = np.concatenate([ses['eeg'] for ses in sessions])

        for fold, (train_idx, test_idx) in enumerate(skf.split(eeg_full, y)):
            split, has_band = build_splits_kfold(all_data, sub_id,
                                                  train_idx, test_idx)
            acc, _, _ = train_and_evaluate(
                split, has_band, model_class, model_args,
                mode=mode, **kwargs
            )
            fold_accs.append(acc)
            print(f"  Sub {sub_id} | Fold {fold+1}: {acc:.3f}")

        all_results[sub_id] = np.mean(fold_accs)
        print(f"  Sub {sub_id} MEAN: {all_results[sub_id]:.3f}")

    global_mean = np.mean(list(all_results.values()))
    print(f"\n>>> WITHIN-SUBJECT {mode.upper()}: {global_mean:.3f}")
    return all_results


def run_ablation(all_data, model_class, model_args, **kwargs):
    """Run LOSO for each modality mode. One unified call."""
    modes = ["eeg", "emg", "imu"]
    results = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"  ABLATION — MODE: {mode.upper()}")
        print(f"{'='*50}")

        mean, std, _ = run_loso(all_data, model_class, model_args,
                                mode=mode, **kwargs)
        results[mode] = (mean, std)

    print(f"\n{'='*50}")
    print("  ABLATION SUMMARY")
    print(f"{'='*50}")
    for mode, (m, s) in results.items():
        print(f"  {mode.upper():>8s}: {m:.3f} ± {s:.3f}")

    return results
