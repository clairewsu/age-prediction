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



CSV_PATH = 'nacc.synthseg.notnormcog.zscore.csv'   # adjust path if needed
MODEL_PATH = os.path.join('../saved_models_different', 'best_model.pt')
# ---- DEFINE GROUPS (from R) ----
groups = [
    ["csf","total.intracranial"],
    ["ctx.rh.insula","ctx.lh.insula","ctx.rh.superiortemporal",            
    "ctx.lh.parahippocampal","ctx.lh.rostralanteriorcingulate","ctx.rh.rostralanteriorcingulate",
    "ctx.rh.lateralorbitofrontal","ctx.rh.pericalcarine","ctx.rh.inferiortemporal","ctx.rh.paracentral"],
    ["left.inferior.lateral.ventricle","right.inferior.lateral.ventricle"],
    ["right.hippocampus","right.amygdala"],
]

# runtime options
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
PERM_REPEATS = 100
PERM_TOP_K = None          # None => permute all features; or set e.g. 100 to permute only top-100 features
IG_SAMPLES = 200           # #val samples to run IG on (use <= dataset size)
SHAP_BG = 50               # SHAP Kernel background size
SHAP_NEVAL = 300           # number of rows to explain with SHAP (<= dataset rows)


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

def load_feature_stats(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature-stats file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    if not {'mean', 'std'}.issubset(df.columns):
        raise ValueError("Feature-stats CSV must contain columns named 'mean' and 'std'.")
    return df[['mean', 'std']].copy()


def compute_mean_std_from_stats(features_df, stats_df):
    feat_cols = list(features_df.columns)

    stats_index = list(stats_df.index.astype(str))
    if set(stats_index) == set(map(str, feat_cols)):
        stats_ordered = stats_df.reindex(feat_cols)
        means = stats_ordered['mean'].to_numpy(dtype=np.float32)
        stds = stats_ordered['std'].to_numpy(dtype=np.float32)
        stds = np.where(stds == 0, 1e-8, stds).astype(np.float32)
        return means, stds

    if stats_df.shape[0] == features_df.shape[1]:
        print("Warning: feature-stats index does not match feature names; using positional order.")
        means = stats_df['mean'].to_numpy(dtype=np.float32)
        stds = stats_df['std'].to_numpy(dtype=np.float32)
        stds = np.where(stds == 0, 1e-8, stds).astype(np.float32)
        return means, stds

    raise ValueError(
        f"Feature-stats shape/names do not match selected features.\n"
        f"Stats rows: {stats_df.shape[0]}, selected features: {features_df.shape[1]}."
    )   
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
    

def permutation_importance(model, X_tensor, y_tensor, feat_names, repeats=5, out_dir=None, stem="run", seed=42, groups=None):
    base_mae, _, _ = compute_mae_preds(model, X_tensor, y_tensor)
    print("Base MAE:", base_mae)

    x_np = X_tensor.cpu().numpy()
    feat_to_idx = {f: i for i, f in enumerate(feat_names)}
    importances = []

    # ---- GROUPED PERMUTATION ----
    if groups is not None:
        for group in groups:
            idxs = [feat_to_idx[f] for f in group if f in feat_to_idx]
            if not idxs:
                continue

            deltas = []
            for r in range(repeats):
                arr = x_np.copy()
                rng = np.random.RandomState(seed + r)

                perm = rng.permutation(arr.shape[0])
                for idx in idxs:
                    arr[:, idx] = arr[perm, idx]

                mae_shuf, _, _ = compute_mae_preds(
                    model, torch.tensor(arr, dtype=torch.float32), y_tensor
                )
                deltas.append(mae_shuf - base_mae)

            importances.append((" + ".join(group), float(np.mean(deltas)), float(np.std(deltas))))

        imp_df = pd.DataFrame(importances, columns=["feature_group", "mae_increase", "std"]).sort_values("mae_increase", ascending=False)

    # ---- ORIGINAL SINGLE FEATURE PERMUTATION ----
    else:
        for i, name in enumerate(feat_names):
            deltas = []
            for r in range(repeats):
                arr = x_np.copy()
                rng = np.random.RandomState(seed + r)
                rng.shuffle(arr[:, i])

                mae_shuf, _, _ = compute_mae_preds(
                    model, torch.tensor(arr, dtype=torch.float32), y_tensor
                )
                deltas.append(mae_shuf - base_mae)

            importances.append((name, float(np.mean(deltas)), float(np.std(deltas))))

        imp_df = pd.DataFrame(importances, columns=["feature", "mae_increase", "std"]).sort_values("mae_increase", ascending=False)

    # ---- SAVE ----
    if out_dir is not None:
        imp_df.to_csv(os.path.join(out_dir, f"{stem}_permutation_importance.csv"), index=False)

    return imp_df
    
def main(csv_path,seed,feature_stats_path):
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = f'explaincdr_seed{seed}'
    os.makedirs(out_dir,exist_ok=True)
    print("Device:", DEVICE)
    df = pd.read_csv(csv_path)
    ncols = df.shape[1]
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"Loaded CSV {CSV_PATH} shape {df.shape}")

    # select features: columns 1 .. (ncols - 12)
    cols_main = list(range(1, max(1, ncols -8)))
    cols = sorted(set(cols_main))
    print(f"Selecting {len(cols)} feature columns (indices): {cols[:10]}{'...' if len(cols)>10 else ''}")

    df_feats = df.iloc[:, cols].copy()

    # attach metadata (keeps them but we won't drop rows)
    required = ['age','subject']
    for r in required:
        if r not in df.columns:
            raise KeyError(f"Missing required column: {r}")
    df_feats['age'] = df['age']
    df_feats['subject'] = df['subject']

    # full-dataset feature matrix & target
    drop_cols = ['age','subject']
    stats_df = load_feature_stats(feature_stats_path)
    
    
    feat_names = df_feats.drop(columns=drop_cols).columns.tolist()
    feat_means, feat_stds = compute_mean_std_from_stats(df_feats.drop(columns=drop_cols), stats_df)

    X_all = df_feats.drop(columns=drop_cols).values.astype(np.float32)
    y_all_raw = df_feats['age'].values.astype(np.float32)

    # --- CLEANING: drop rows with non-finite age OR non-finite features ---
    # (This must happen BEFORE you compute feat_means / feat_stds / normalize / create tensors)
    mask_good = np.isfinite(y_all_raw) & np.isfinite(X_all).all(axis=1)
    n_total = X_all.shape[0]
    n_keep = int(mask_good.sum())
    n_drop = n_total - n_keep
    if n_drop > 0:
        print(f"Dropping {n_drop} rows with non-finite target/features (keeping {n_keep} of {n_total})")
    X_all = X_all[mask_good]
    y_all_raw = y_all_raw[mask_good]
    
    X_all_norm = (X_all - feat_means.reshape(1, -1)) / feat_stds.reshape(1, -1)
    y_mean = float(np.mean(y_all_raw))
    y_std = float(np.std(y_all_raw)) if np.std(y_all_raw) > 0 else 1.0
    y_all_norm = (y_all_raw - y_mean) / y_std


    # --- convert to tensors (ensure targets are 1-D) ---
    X_tensor = torch.tensor(X_all_norm, dtype=torch.float32)
    y_raw_tensor = torch.tensor(y_all_raw, dtype=torch.float32).view(-1)
    y_norm_tensor = torch.tensor(y_all_norm, dtype=torch.float32).view(-1)

    # rebuild feature names to match X_all columns (drop metadata)
    feat_names = df_feats.drop(columns=drop_cols).columns.tolist()
    print("Num samples:", X_tensor.shape[0], "Num features:", X_tensor.shape[1])

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
    # get all grouped features
    grouped_feats = set(f for group in groups for f in group)

    # find leftover features
    leftover = [f for f in perm_feat_names if f not in grouped_feats]

    # turn leftovers into single-feature groups
    for f in leftover:
        groups.append([f])
    perm_df = permutation_importance(
        model, perm_X_tensor, y_for_metric, perm_feat_names,
        repeats=PERM_REPEATS, out_dir=out_dir, stem=stem, seed=seed,groups=groups
    )
    perm_df.to_csv(os.path.join(out_dir, f'{stem}_permutation_interpret.csv'), index=False)

    print("\nTop 20 by permutation importance:")
    print(perm_df.head(20).to_string(index=False))
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True)
parser.add_argument('--out', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--feature-stats', type=str, required=True)
args = parser.parse_args()

main(csv_path=args.csv,seed=args.seed,feature_stats_path=args.feature_stats)
    

