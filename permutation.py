import os
import json
import numpy as np
if not hasattr(np, "int"):
    np.int = int
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt



CSV_PATH = '../mlp/data/combined5.outlier2.csv'   # adjust path if needed
MODEL_PATH = os.path.join('saved_models_different', 'best_model.pt')
OUT_DIR = 'explain_outputs_fullcsv'
os.makedirs(OUT_DIR, exist_ok=True)

# runtime options
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
PERM_REPEATS = 5
PERM_TOP_K = None          # None => permute all features; or set e.g. 100 to permute only top-100 features
IG_SAMPLES = 200           # #val samples to run IG on (use <= dataset size)
SHAP_BG = 50               # SHAP Kernel background size
SHAP_NEVAL = 300           # number of rows to explain with SHAP (<= dataset rows)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# model architecture params (must match training)
HIDDEN_SIZES = [256, 128, 64]
OUTPUT_SIZE = 1
USE_LAYERNORM = False

# ---------------- Model ----------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.2, use_layernorm=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.LayerNorm(hidden_sizes[0]) if use_layernorm else nn.BatchNorm1d(hidden_sizes[0])
        self.drop1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.LayerNorm(hidden_sizes[1]) if use_layernorm else nn.BatchNorm1d(hidden_sizes[1])
        self.drop2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.LayerNorm(hidden_sizes[2]) if use_layernorm else nn.BatchNorm1d(hidden_sizes[2])
        self.drop3 = nn.Dropout(dropout_prob)

        self.out = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = self.fc1(x); x = torch.nn.functional.gelu(x); x = self.bn1(x); x = self.drop1(x)
        x = self.fc2(x); x = torch.nn.functional.gelu(x); x = self.bn2(x); x = self.drop2(x)
        x = self.fc3(x); x = torch.nn.functional.gelu(x); x = self.bn3(x); x = self.drop3(x)
        x = self.out(x)
        return x


def load_model(input_size, model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = MLP(input_size=input_size, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model
    
def compute_mae_preds(model, X_tensor, y_tensor):
    model.eval()
    preds = []; targs = []
    with torch.no_grad():
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=False)
        for xb, yb in loader:
            out = model(xb.to(DEVICE)).view(-1).cpu().numpy()
            preds.append(out); targs.append(yb.view(-1).cpu().numpy())
    preds = np.concatenate(preds); targs = np.concatenate(targs)
    mae = float(np.mean(np.abs(preds - targs)))
    return mae, preds, targs
    

def permutation_importance(model, X_tensor, y_tensor, feat_names, repeats=5, out_dir=OUT_DIR):
    base_mae, _, _ = compute_mae_preds(model, X_tensor, y_tensor)
    print("Base MAE (units matching model outputs):", base_mae)
    x_np = X_tensor.cpu().numpy()
    importances = []
    D = x_np.shape[1]
    for i, name in enumerate(feat_names):
        deltas = []
        for r in range(repeats):
            arr = x_np.copy()
            rng = np.random.RandomState(1000 + r)
            rng.shuffle(arr[:, i])
            mae_shuf, _, _ = compute_mae_preds(model, torch.tensor(arr, dtype=torch.float32), y_tensor)
            deltas.append(mae_shuf - base_mae)
        importances.append((name, float(np.mean(deltas)), float(np.std(deltas))))
    imp_df = pd.DataFrame(importances, columns=['feature','mae_increase','std']).sort_values('mae_increase', ascending=False)
    imp_df.to_csv(os.path.join(out_dir, 'permutation_importance_fullcsv.csv'), index=False)
    topk = imp_df.head(30)
    plt.figure(figsize=(8,6)); plt.barh(topk['feature'][::-1], topk['mae_increase'][::-1]); plt.xlabel('Increase in MAE'); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'permutation_importance_fullcsv.png')); plt.close()
    print("Saved permutation_importance_fullcsv.csv and PNG")
    return imp_df
    
def main():
    print("Device:", DEVICE)
    df = pd.read_csv(CSV_PATH)
    ncols = df.shape[1]
    print(f"Loaded CSV {CSV_PATH} shape {df.shape}")

    # select features: columns 1 .. (ncols - 12)
    cols_main = list(range(1, max(1, ncols - 12)))
    cols = sorted(set(cols_main))
    print(f"Selecting {len(cols)} feature columns (indices): {cols[:10]}{'...' if len(cols)>10 else ''}")

    df_feats = df.iloc[:, cols].copy()

    # attach metadata (keeps them but we won't drop rows)
    required = ['Age','FSID','TVT']
    for r in required:
        if r not in df.columns:
            raise KeyError(f"Missing required column: {r}")
    df_feats['age'] = df['Age']
    df_feats['FSID'] = df['FSID']
    df_feats['group'] = df['TVT']

    # full-dataset feature matrix & target
    drop_cols = ['age','FSID','group']
    X_all = df_feats.drop(columns=drop_cols).values.astype(np.float32)
    y_all_raw = df_feats['age'].values.astype(np.float32)

    # compute z-score stats from ENTIRE CSV (per your request)
    feat_means = np.nanmean(X_all, axis=0)
    feat_stds = np.nanstd(X_all, axis=0)
    feat_stds[feat_stds == 0] = 1e-8

    # normalize full dataset
    X_all_norm = (X_all - feat_means.reshape(1, -1)) / feat_stds.reshape(1, -1)
    y_mean = float(np.mean(y_all_raw))
    y_std = float(np.std(y_all_raw)) if np.std(y_all_raw) > 0 else 1.0
    y_all_norm = (y_all_raw - y_mean) / y_std

    # convert to tensors
    X_tensor = torch.tensor(X_all_norm, dtype=torch.float32)
    # We'll keep both raw and norm y as tensors for flexibility
    y_raw_tensor = torch.tensor(y_all_raw, dtype=torch.float32).view(-1,1)
    y_norm_tensor = torch.tensor(y_all_norm, dtype=torch.float32).view(-1,1)

    feat_names = df_feats.drop(columns=drop_cols).columns.tolist()
    print("Num samples:", X_tensor.shape[0], "Num features:", X_tensor.shape[1])

    # load model
    model = load_model(input_size=X_tensor.shape[1], model_path=MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    # detect whether model outputs raw ages or normalized targets
    model.eval()
    with torch.no_grad():
        # use a subset to detect
        sample_in = X_tensor[:min(256, X_tensor.shape[0])].to(DEVICE)
        out = model(sample_in).view(-1).cpu().numpy()
    out_mean = float(np.mean(out)); out_std = float(np.std(out))
    print("Model output stats (sample): mean={:.4f}, std={:.4f}".format(out_mean, out_std))
    is_raw_output = abs(out_mean - y_mean) < (3.0 * y_std)
    if is_raw_output:
        print("Detected: model outputs RAW ages.")
        y_for_metric = y_raw_tensor
    else:
        print("Detected: model outputs NORMALIZED targets.")
        y_for_metric = y_norm_tensor

    # optionally limit permutation to top-K features by simple correlation to speed up
    perm_feat_names = feat_names
    perm_X_tensor = X_tensor
    if PERM_TOP_K is not None and PERM_TOP_K < len(feat_names):
        # compute absolute Pearson correlation with raw age and pick top-K
        corrs = []
        for i in range(X_all.shape[1]):
            a = X_all[:, i]
            if np.all(np.isfinite(a)):
                c = np.corrcoef(a, y_all_raw)[0,1]
            else:
                c = 0.0
            corrs.append(abs(c) if not np.isnan(c) else 0.0)
        idx_sorted = np.argsort(corrs)[::-1][:PERM_TOP_K]
        perm_feat_names = [feat_names[i] for i in idx_sorted.tolist()]
        perm_X_tensor = torch.tensor(X_all_norm[:, idx_sorted], dtype=torch.float32)
        print(f"Permutation limited to top {len(perm_feat_names)} features (by abs corr with age).")

    # Run permutation importance (units matching model outputs)
    perm_df = permutation_importance(model, perm_X_tensor, y_for_metric, perm_feat_names, repeats=PERM_REPEATS)
    perm_df.to_csv(os.path.join(OUT_DIR, 'permutation_interpret.csv'), index=False)

    print("\nTop 20 by permutation importance:")
    print(perm_df.head(20).to_string(index=False))
    
if __name__ == '__main__':
    main()

