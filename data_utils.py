import numpy as np
import pandas as pd
import torch

REQUIRED_COLS = ['Age', 'FSID', 'TVT']

def load_and_select_features(csv_path):
    """
    Loads CSV and selects the feature columns (original behavior: cols 1 .. ncols-12).
    Coerces selected feature columns to numeric, fills NaNs with column means,
    and appends metadata columns 'age','FSID','group' (from 'Age','FSID','TVT').
    Returns (original_df, df1) where df1 has numeric feature columns followed by
    the three metadata columns.
    """
    df = pd.read_csv(csv_path)
    ncols = df.shape[1]
    cols_main = list(range(1, max(1, ncols - 12)))
    cols = sorted(set(cols_main))

    # Select candidate feature columns
    df_features = df.iloc[:, cols].copy()

    # Try to coerce to numeric; non-convertible values become NaN
    df_features = df_features.apply(pd.to_numeric, errors='coerce')

    # Report columns that became fully NaN (unlikely but important)
    fully_na = df_features.columns[df_features.isna().all()].tolist()
    if fully_na:
        raise ValueError(f"The following selected feature columns are entirely non-numeric or all-NaN: {fully_na}")

    # Fill remaining NaNs with the column mean (so a few bad cells won't crash)
    col_means = df_features.mean()
    df_features = df_features.fillna(col_means)

    # Attach metadata columns. Ensure they exist.
    for rc in REQUIRED_COLS:
        if rc not in df.columns:
            raise KeyError(f"Required column '{rc}' not in CSV")

    df_features['age'] = df['Age']
    df_features['FSID'] = df['FSID']
    df_features['group'] = df['TVT']

    # Now df_features (alias df1) has numeric feature columns followed by metadata
    return df, df_features

def compute_feature_mean_std(df1):
    """
    Compute per-feature mean and std (ddof=0) for the numeric feature columns.
    We assume df1 has the last three columns as metadata: ['age','FSID','group'].
    Returns (feature_means (pd.Series), feature_stds (pd.Series),
             mean_tensor (torch.tensor), std_tensor (torch.tensor)).
    The tensors are CPU tensors (caller can move them to DEVICE).
    """
    if not {'age', 'FSID', 'group'}.issubset(df1.columns):
        raise KeyError("df1 must contain metadata columns 'age', 'FSID', 'group'")

    # Numeric feature columns are everything except the last 3 metadata columns
    feature_cols = df1.columns[:-3]
    feature_df = df1.loc[:, feature_cols]

    # Ensure numeric dtype (should be, but double-check)
    feature_df = feature_df.apply(pd.to_numeric, errors='coerce')
    if feature_df.isna().any().any():
        # If any NaNs remain (unexpected), fill with column mean
        feature_df = feature_df.fillna(feature_df.mean())

    feature_means = feature_df.mean()
    feature_stds = feature_df.std(ddof=0).replace(0, 1e-8)

    mean_tensor = torch.tensor(feature_means.values, dtype=torch.float32)
    std_tensor = torch.tensor(feature_stds.values, dtype=torch.float32)

    return feature_means, feature_stds, mean_tensor, std_tensor

def filter_age_and_split(df, min_age=51):
    """
    Filters df (which should already have 'age' column) to keep age > min_age-1 (default >50),
    and splits into train/val/test by 'group' values 1/2/3 respectively.
    """
    if 'age' not in df.columns:
        raise KeyError("DataFrame must contain 'age' column for filtering")

    mask = (df['age'] > (min_age - 1))
    df = df[mask].reset_index(drop=True)

    train_df = df[df['group'] == 1].reset_index(drop=True)
    val_df = df[df['group'] == 2].reset_index(drop=True)
    test_df = df[df['group'] == 3].reset_index(drop=True)

    return train_df, val_df, test_df

def prep_xy(df_part, drop_cols=['age','FSID','group']):
    """
    Extract X (features) and y (age) from a dataframe part.
    Drop the metadata columns listed in drop_cols to form X.
    """
    X = df_part.drop(columns=drop_cols).values.astype(np.float32)
    y = df_part['age'].values.astype(np.float32)
    return X, y

def zscore_tensor(x_raw, mean_tensor, std_tensor, device=None):
    """
    Apply z-score normalization to a tensor (x_raw) using provided mean and std tensors.
    If device is provided, mean/std are moved to that device.
    """
    if device is not None:
        mean_tensor = mean_tensor.to(device)
        std_tensor = std_tensor.to(device)
    return (x_raw - mean_tensor) / std_tensor

def save_feature_stats_df(df1, out_path):
    """
    Saves the mean/std of feature columns to CSV. Assumes last 3 cols are metadata.
    """
    feature_cols = df1.columns[:-3]
    stats_df = pd.DataFrame({
        'mean': df1.loc[:, feature_cols].mean(),
        'std': df1.loc[:, feature_cols].std()
    })
    stats_df.to_csv(out_path, index=True)
