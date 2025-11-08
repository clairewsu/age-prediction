#!/usr/bin/env python3
"""
evaluate.py

Evaluate a trained model on a CSV dataset and optionally plot Actual vs Predicted.
Uses feature means/stds from training (feature-stats CSV) to z-score evaluation data.

Usage examples:
  python evaluate.py --csv ../data/combined5.outlier2.csv --feature-stats saved_models_different/feature_stats_different.csv
  python evaluate.py --csv ../data/new_dataset.csv --no-tvt --cutoff 8 --feature-stats ./feature_stats_different.csv
"""
import os
import argparse
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from config import *
from model import MLP

OUTLIER_COL_REGEX = re.compile(r'outlier', flags=re.IGNORECASE)

def find_outlier_column(df):
    """Return the first column name that looks like an outlier flag, or None."""
    for col in df.columns:
        if OUTLIER_COL_REGEX.search(col):
            return col
    return None

def select_feature_columns_from_csv(df, cutoff=12):
    """
    Select feature columns according to the convention:
    columns 1 .. (ncols - cutoff). If that results in no features,
    fall back to all numeric columns except known metadata names.
    """
    ncols = df.shape[1]
    if cutoff < 0:
        raise ValueError("--cutoff must be >= 0")
    cols_main = list(range(1, max(1, ncols - cutoff)))
    cols = sorted(set(cols_main))
    if len(cols) > 0 and max(cols) < df.shape[1]:
        try:
            features_df = df.iloc[:, cols].copy()
            return features_df
        except Exception:
            pass
    exclude = {'Age', 'FSID', 'TVT'}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    return df.loc[:, numeric_cols].copy()

def coerce_and_fill(features_df):
    """Coerce to numeric and fill per-column NaNs with column means. Ensure not fully-NaN."""
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    fully_na = features_df.columns[features_df.isna().all()].tolist()
    if fully_na:
        raise ValueError(f"The following feature columns are entirely non-numeric: {fully_na}")
    return features_df.fillna(features_df.mean())

def load_feature_stats(path):
    """
    Load a feature-stats CSV with columns 'mean' and 'std'.
    Returns a DataFrame indexed by feature name (if present) or a DataFrame with positional index.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature-stats file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    # Accept if it has columns 'mean' and 'std'
    if not {'mean', 'std'}.issubset(df.columns):
        raise ValueError("Feature-stats CSV must contain columns named 'mean' and 'std'.")
    return df[['mean', 'std']].copy()

def compute_mean_std_from_stats(features_df, stats_df):
    """
    Resolve stats_df to match features_df columns.
    - If stats_df.index matches feature names, reorder accordingly.
    - Else if lengths match, use positional order (issue a warning).
    - Else raise.
    Returns (means_array, stds_array, means_series, stds_series)
    """
    feat_cols = list(features_df.columns)
    # Case 1: names match (set equality)
    stats_index = list(stats_df.index.astype(str))
    if set(stats_index) == set(map(str, feat_cols)):
        # reorder stats to feature order
        stats_ordered = stats_df.reindex(feat_cols)
        means = stats_ordered['mean'].to_numpy(dtype=np.float32)
        stds = stats_ordered['std'].to_numpy(dtype=np.float32)
        means_series = stats_ordered['mean']
        stds_series = stats_ordered['std']
        # replace zeros in stds
        stds = np.where(stds == 0, 1e-8, stds).astype(np.float32)
        stds_series = stds_series.replace(0, 1e-8)
        return means, stds, means_series, stds_series

    # Case 2: length matches, use positional order (fall back)
    if stats_df.shape[0] == features_df.shape[1]:
        print("Warning: feature-stats index does not match feature names; using positional order from stats file.")
        means = stats_df['mean'].to_numpy(dtype=np.float32)
        stds = stats_df['std'].to_numpy(dtype=np.float32)
        stds = np.where(stds == 0, 1e-8, stds).astype(np.float32)
        means_series = pd.Series(means, index=features_df.columns)
        stds_series = pd.Series(stds, index=features_df.columns)
        return means, stds, means_series, stds_series

    # Otherwise, cannot resolve
    raise ValueError(
        f"Feature-stats shape/names do not match selected features.\n"
        f"Stats file rows: {stats_df.shape[0]}, selected features: {features_df.shape[1]}.\n"
        f"Either provide a feature-stats CSV whose index matches feature names, or use --cutoff so features align."
    )

def evaluate(csv_path,
             model_path=None,
             feature_stats_path=None,
             out_csv='evaluation_predictions.csv',
             batch_size=BATCH_SIZE,
             device=None,
             has_tvt=True,
             has_age=True,
             cutoff=12,
             plot=True,
             plot_path='actual_vs_predicted_eval.png',
             keep_outliers=False):
    """
    Evaluate model on the given CSV using feature means/stds from training.
    """
    if device is None:
        device = DEVICE
    if model_path is None:
        model_path = os.path.join(SAVE_DIR, 'best_model.pt')

    # default feature-stats path if not provided
    if feature_stats_path is None:
        candidate = os.path.join(SAVE_DIR, 'feature_stats_different.csv')
        if os.path.isfile(candidate):
            feature_stats_path = candidate
        elif os.path.isfile('feature_stats_different.csv'):
            feature_stats_path = 'feature_stats_different.csv'
        else:
            raise FileNotFoundError(
                "Feature-stats CSV not provided and default paths not found.\n"
                "Provide --feature-stats PATH pointing to the CSV produced by training (columns: mean,std)."
            )

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.isfile(feature_stats_path):
        raise FileNotFoundError(f"Feature-stats file not found: {feature_stats_path}")

    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        raise ValueError("CSV is empty.")

    # Detect and optionally remove outliers if an outlier-like column exists
    outlier_col = find_outlier_column(df)
    if outlier_col is not None and not keep_outliers:
        outlier_vals = pd.to_numeric(df[outlier_col], errors='coerce').fillna(0).astype(np.float32)
        mask_outlier = outlier_vals != 0
        n_outliers = int(mask_outlier.sum())
        if n_outliers > 0:
            print(f"Outlier column detected ('{outlier_col}'). Removing {n_outliers} outlier rows.")
            df = df.loc[~mask_outlier].reset_index(drop=True)
        else:
            print(f"Outlier column detected ('{outlier_col}') but no truthy values found; no rows removed.")
    elif outlier_col is not None and keep_outliers:
        print(f"Outlier column detected ('{outlier_col}'), but --keep-outliers set: not removing any rows.")

    # select and prepare feature dataframe
    features_df = select_feature_columns_from_csv(df, cutoff=cutoff)
    features_df = coerce_and_fill(features_df)

    # load training feature stats and resolve ordering
    stats_df = load_feature_stats(feature_stats_path)
    feat_mean_arr, feat_std_arr, feat_means_series, feat_stds_series = compute_mean_std_from_stats(features_df, stats_df)

    # metadata
    if has_age:
        if 'Age' not in df.columns:
            raise KeyError("Expected 'Age' column for has_age=True.")
        age_arr = pd.to_numeric(df['Age'], errors='coerce').to_numpy(dtype=np.float32)
        if np.isnan(age_arr).any():
            raise ValueError("'Age' column contains NaNs or non-numeric values.")
    else:
        age_arr = np.zeros(len(df), dtype=np.float32)

    fsid_arr = df['FSID'].values if 'FSID' in df.columns else np.arange(len(df))
    if has_tvt and 'TVT' in df.columns:
        group_arr = df['TVT'].values
    else:
        group_arr = np.full(len(df), 3, dtype=np.int32)

    X = features_df.to_numpy(dtype=np.float32)
    y = age_arr

    # Use training stats (feat_mean_arr, feat_std_arr) to z-score evaluation data
    mean_vec = torch.tensor(feat_mean_arr, dtype=torch.float32, device=device)
    std_vec = torch.tensor(feat_std_arr, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

    if mean_vec.shape[0] != x_tensor.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch after applying training stats: "
            f"selected features={x_tensor.shape[1]}, stats length={mean_vec.shape[0]}.\n"
            "Adjust --cutoff or provide a matching --feature-stats file."
        )

    x_norm = (x_tensor - mean_vec) / std_vec
    dataset = TensorDataset(x_norm, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_size = x_norm.shape[1]
    model = MLP(input_size=input_size, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE,
                dropout_prob=DROPOUT, use_layernorm=USE_LAYERNORM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.L1Loss()
    preds = []
    losses = []

    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb)
            if has_age:
                loss = criterion(outputs, yb)
                losses.append(loss.item())
            preds.extend(outputs.view(-1).cpu().numpy().tolist())

    if has_age and losses:
        mae = float(np.mean(losses))
        print(f"Evaluation MAE: {mae:.4f} on {len(dataset)} samples")
    else:
        print(f"Ran inference on {len(dataset)} samples (no Age provided).")

    results = pd.DataFrame({
        'FSID': fsid_arr,
        'prediction': np.array(preds, dtype=np.float32)
    })
    if has_age:
        results['actual'] = age_arr
        results['diff'] = np.abs(results['actual'] - results['prediction'])
        results['realdiff'] = results['prediction'] - results['actual']

    results.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

    # Plot actual vs predicted (same style as training) if we have ground truth
    if has_age and plot:
        try:
            actual = results['actual'].to_numpy()
            pred = results['prediction'].to_numpy()
            min_val = float(min(actual.min(), pred.min()))
            max_val = float(max(actual.max(), pred.max()))

            plt.figure(figsize=(8, 6))
            plt.scatter(actual, pred, alpha=0.6, edgecolor='k')
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
            plt.xlabel('Actual Age')
            plt.ylabel('Predicted Age')
            plt.title('Actual vs Predicted Age (Evaluation)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
            try:
                plt.show()
            except Exception:
                pass
            plt.close()
        except Exception as e:
            print("Warning: failed to create plot:", e)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained model on a dataset CSV using training feature stats.")
    parser.add_argument('--csv', type=str, required=True, help="Path to dataset CSV.")
    parser.add_argument('--model', type=str, default=None,
                        help="Path to model file (.pt). Defaults to best_model.pt in SAVE_DIR.")
    parser.add_argument('--feature-stats', type=str, default=None,
                        help="Path to feature-stats CSV (index=feature names or positional; columns: mean,std). "
                             "Defaults to SAVE_DIR/feature_stats_different.csv or ./feature_stats_different.csv.")
    parser.add_argument('--out', type=str, default='evaluation_predictions.csv',
                        help="Path to save predictions CSV.")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size for DataLoader.")
    parser.add_argument('--device', type=str, default=None, help="'cpu' or 'cuda'. Defaults to config.DEVICE.")
    parser.add_argument('--no-tvt', action='store_true', help="If set, ignore TVT and apply to all rows.")
    parser.add_argument('--no-age', action='store_true', help="If set, skip MAE and plotting (for unlabeled data).")
    parser.add_argument('--cutoff', '-c', type=int, default=12,
                        help="How many columns from the end to exclude (default=12).")
    parser.add_argument('--no-plot', action='store_true', help="If set, skip drawing/saving the plot.")
    parser.add_argument('--plot-path', type=str, default='actual_vs_predicted_eval.png',
                        help="Path to save the Actual vs Predicted plot.")
    parser.add_argument('--keep-outliers', action='store_true',
                        help="If set, do NOT remove rows flagged by an outlier column (if present).")

    args = parser.parse_args()
    dev = torch.device(args.device) if args.device else DEVICE

    evaluate(csv_path=args.csv,
             model_path=args.model,
             feature_stats_path=args.feature_stats,
             out_csv=args.out,
             batch_size=args.batch_size,
             device=dev,
             has_tvt=not args.no_tvt,
             has_age=not args.no_age,
             cutoff=args.cutoff,
             plot=not args.no_plot,
             plot_path=args.plot_path,
             keep_outliers=args.keep_outliers)
