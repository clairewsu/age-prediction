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

# ---------------- CONFIG ----------------
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
SHAP_NEVAL = 1000           # number of rows to explain with SHAP (<= dataset rows)
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

# ---------------- Helpers ----------------
def load_model(input_size, model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = MLP(input_size=input_size, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def shap_kernel_with_noise_robust(model, X_tensor, y_tensor_for_noise, feat_names,
                                  requested_bg=50, n_eval=300, nsamples=100,
                                  out_dir=OUT_DIR, topk_fallback=120, batch_size=100, seed=42):
    """
    Robust Kernel SHAP:
      - Estimates noise variance from model residuals
      - Ensures background size is >= num_features (to satisfy LassoLarsIC), otherwise enlarges
      - If enlarging is impossible or unacceptable, falls back to running SHAP on top-K features
        selected by absolute Pearson correlation (fast)
      - Uses batching when computing shap_values for many samples
    Returns a DataFrame or None.
    """
    try:
        import shap
    except Exception as e:
        print("SHAP import failed:", e)
        return None

    rng = np.random.default_rng(seed)
    N, D = X_tensor.shape
    print(f"SHAP robust: N={N}, D={D}, requested_bg={requested_bg}")

    # estimate noise variance from residuals (up to 2000 samples)
    with torch.no_grad():
        take_n = min(2000, N)
        X_sub = X_tensor[:take_n].to(DEVICE)
        preds = model(X_sub).view(-1).cpu().numpy()
        y_for_noise_np = y_tensor_for_noise[:take_n].view(-1).cpu().numpy()
        resid = preds - y_for_noise_np
        noise_var = float(np.var(resid))
        if not np.isfinite(noise_var) or noise_var <= 0:
            noise_var = 1e-6
    print(f"Estimated noise_variance={noise_var:.6e} from {take_n} samples")

    # ensure background size is at least num_features (or a sensible minimum)
    bg_n = max(requested_bg, D, 300)  # require at least D and at least 300 for stability
    bg_n = min(bg_n, N)               # cannot exceed dataset size
    print(f"Using background size bg_n={bg_n}")

    bg_idx = rng.choice(N, size=bg_n, replace=False)
    background = X_tensor[bg_idx].cpu().numpy()

    # ensure n_eval is >= D if possible
    n_eval = int(min(n_eval, N))
    if n_eval < D:
        print(f"Warning: n_eval {n_eval} < D {D}; setting n_eval = {min(N, max(D, n_eval))}")
        n_eval = min(N, max(D, n_eval))

    # choose evaluation indices
    eval_idx = rng.choice(N, size=n_eval, replace=False)
    x_eval = X_tensor[eval_idx].cpu().numpy()

    def model_forward_numpy(x_numpy):
        xt = torch.tensor(x_numpy.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            out = model(xt).view(-1).cpu().numpy()
        return out

    # Try initializing KernelExplainer with sufficiently large background
    try:
        print("Initializing KernelExplainer with noise_variance and large background...")
        ke = shap.KernelExplainer(model_forward_numpy, background, noise_variance=noise_var)
    except Exception as e:
        print("KernelExplainer init failed even with large background:", e)
        # FALLBACK: reduce features (top-k) and run on reduced input
        if topk_fallback is None or topk_fallback >= D:
            print("No top-k fallback available or not requested; aborting SHAP.")
            return None
        print(f"Falling back to top-{topk_fallback} feature SHAP (by abs Pearson corr) to avoid ill-conditioning.")
        # compute abs Pearson correlation between raw features and raw target
        X_np = X_tensor.cpu().numpy()
        # try to get original raw target array for correlation
        try:
            # if y_tensor_for_noise is normalized, it still is valid for corr ranking
            y_np = y_tensor_for_noise.view(-1).cpu().numpy()
        except Exception:
            y_np = np.zeros(N)
        corrs = []
        for i in range(D):
            a = X_np[:, i]
            # compute corr safely
            if np.all(np.isfinite(a)):
                c = np.corrcoef(a, y_np)[0,1]
                if np.isnan(c):
                    c = 0.0
            else:
                c = 0.0
            corrs.append(abs(c))
        idx_sorted = np.argsort(corrs)[::-1][:topk_fallback]
        reduced_feat_names = [feat_names[i] for i in idx_sorted.tolist()]
        X_reduced = X_np[:, idx_sorted]
        # convert reduced inputs to tensor
        X_reduced_tensor = torch.tensor(X_reduced, dtype=torch.float32)
        # run the same function but with reduced D
        return shap_kernel_with_noise_robust(model, X_reduced_tensor, y_tensor_for_noise, reduced_feat_names,
                                            requested_bg=max(50, topk_fallback // 3), n_eval=min(n_eval, X_reduced_tensor.shape[0]),
                                            nsamples=nsamples, out_dir=out_dir, topk_fallback=None, batch_size=batch_size, seed=seed)

    # compute shap_values in batches to avoid memory bloat
    all_shap_batches = []
    try:
        total = x_eval.shape[0]
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            xb = x_eval[start:end]
            print(f"Computing SHAP batch rows {start}:{end} ...")
            shap_vals = ke.shap_values(xb, nsamples=nsamples)
            arr = np.array(shap_vals)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 1:
                arr = arr.reshape((xb.shape[0], -1))
            all_shap_batches.append(arr)
        all_shap = np.vstack(all_shap_batches)
    except Exception as e:
        print("KernelExplainer.shap_values failed during batching:", e)
        return None

    mean_abs_shap = np.mean(np.abs(all_shap), axis=0)
    shp_df = pd.DataFrame({'feature': feat_names, 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)
    out_csv = os.path.join(out_dir, 'shap_kernel_with_noise_robust.csv')
    shp_df.to_csv(out_csv, index=False)
    print("Saved", out_csv)
    return shp_df

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
    
def shap_kernel_with_noise(model, X_tensor, y_tensor_for_noise, feat_names,
                           background_size=50, n_eval=300, nsamples=100, out_dir=OUT_DIR, seed=42):
    """
    Kernel SHAP using an explicit noise_variance estimated from model residuals.
    - y_tensor_for_noise must match model output scale (raw or normalized).
    - Returns DataFrame of mean-abs SHAP per feature or None on failure.
    """
    try:
        import shap
    except Exception as e:
        print("SHAP import failed:", e)
        return None

    rng = np.random.default_rng(seed)
    N, D = X_tensor.shape

    # 1) choose background
    bg_n = min(background_size, max(1, N // 10))
    bg_idx = rng.choice(N, size=bg_n, replace=False)
    background = X_tensor[bg_idx].cpu().numpy()

    # 2) estimate noise variance from residuals (use up to 2000 points for speed)
    with torch.no_grad():
        take_n = min(2000, N)
        idx_for_var = np.arange(take_n)
        X_sub = X_tensor[idx_for_var].to(DEVICE)
        preds = model(X_sub).view(-1).cpu().numpy()
        y_for_noise_np = y_tensor_for_noise[idx_for_var].view(-1).cpu().numpy()
        resid = preds - y_for_noise_np
        noise_var = float(np.var(resid))
        # clamp into a stable positive range
        if not np.isfinite(noise_var) or noise_var <= 0:
            noise_var = 1e-6
    print(f"Estimated noise_variance={noise_var:.6e} from {take_n} samples (D={D}, bg_n={bg_n})")

    # 3) choose evaluation set (ensure n_eval >= D if possible)
    n_eval = int(min(n_eval, N))
    if n_eval < D:
        n_eval = max(n_eval, D)
        print(f"Increasing n_eval to {n_eval} to be >= num_features {D}")

    eval_idx = rng.choice(N, size=n_eval, replace=False)
    x_eval = X_tensor[eval_idx].cpu().numpy()

    # 4) wrapper for model -> numpy
    def model_forward_numpy(x_numpy):
        xt = torch.tensor(x_numpy.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            out = model(xt).view(-1).cpu().numpy()
        return out

    # 5) init KernelExplainer with noise_variance
    try:
        ke = shap.KernelExplainer(model_forward_numpy, background, noise_variance=noise_var)
    except Exception as e:
        print("KernelExplainer init failed with noise_variance:", e)
        # fallback: enlarge background to >= D and retry
        try:
            bg_n2 = max(bg_n, D)
            print(f"Retrying with larger background_size={bg_n2}...")
            bg_idx2 = rng.choice(N, size=bg_n2, replace=False)
            background2 = X_tensor[bg_idx2].cpu().numpy()
            ke = shap.KernelExplainer(model_forward_numpy, background2, noise_variance=noise_var)
        except Exception as e2:
            print("KernelExplainer fallback also failed:", e2)
            return None

    # 6) compute SHAP values
    try:
        print(f"Computing SHAP for {n_eval} samples with nsamples={nsamples} ...")
        shap_vals = ke.shap_values(x_eval, nsamples=nsamples)
        arr = np.array(shap_vals)
        # normalize shapes
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr.reshape((n_eval, -1))
    except Exception as e:
        print("KernelExplainer.shap_values failed:", e)
        return None

    # 7) aggregate & save
    mean_abs_shap = np.mean(np.abs(arr), axis=0)
    shp_df = pd.DataFrame({'feature': feat_names, 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)
    out_csv = os.path.join(out_dir, 'shap_kernel_with_noise.csv')
    shp_df.to_csv(out_csv, index=False)
    print("Saved", out_csv)
    return shp_df

# ---------------- Main ----------------
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
        
    shp_df = shap_kernel_with_noise(model, X_tensor, y_tensor_for_noise=y_norm_tensor,
                                    feat_names=feat_names,
                                    background_size=300,  # <- increase to >= D
                                    n_eval=300,
                                    nsamples=100)
    shp_df.to_csv(os.path.join(OUT_DIR, 'shap_pmo_interpret.csv'), index=False)
    print("\nTop 20 SHAP:")
    print(shp_df.head(20).to_string(index=False))
    
if __name__ == '__main__':
    main()
    
