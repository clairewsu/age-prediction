#!/usr/bin/env python3
"""
train.py

Train an MLP model on the specified dataset with optional on-the-fly augmentation.
Usage examples:
  python train.py --csv ../data/combined5.outlier2.csv
  python train.py --csv ../data/new_dataset.csv --no-age-filter
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from config import *
from model import MLP
from dataset import AugmentedTensorDataset, load_noise_vector_from_json
from data_utils import (
    load_and_select_features,
    compute_feature_mean_std,
    filter_age_and_split,
    prep_xy,
    zscore_tensor,
    save_feature_stats_df,
)

def main(csv_path=CSV_PATH,
         noise_json_path=NOISE_JSON_PATH,
         save_dir=SAVE_DIR,
         batch_size=BATCH_SIZE,
         num_epochs=NUM_EPOCHS,
         lr=LR,
         nudge_factor=NUDGE_FACTOR,
         augment_prob=AUGMENT_PROB,
         clip_range=CLIP_RANGE,
         no_age_filter=False):

    os.makedirs(save_dir, exist_ok=True)
    print("Device:", DEVICE)

    # Load and select features
    df_orig, df1 = load_and_select_features(csv_path)
    print(f"Loaded CSV {csv_path} with shape {df_orig.shape}")

    feature_means, feature_stds, mean_tensor, std_tensor = compute_feature_mean_std(df1)
    df1_for_stats = df1.copy()

    # Optionally filter age > 50
    if no_age_filter:
        train_df = df1[df1["group"] == 1].reset_index(drop=True)
        val_df = df1[df1["group"] == 2].reset_index(drop=True)
        test_df = df1[df1["group"] == 3].reset_index(drop=True)
        print("No age filter applied; using all rows.")
    else:
        train_df, val_df, test_df = filter_age_and_split(df1)
        print("Applied age>50 filter.")

    print(f"Rows used: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Shuffle training
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare data
    X_train, y_train = prep_xy(train_df)
    X_val, y_val = prep_xy(val_df)
    X_test, y_test = prep_xy(test_df)

    ytrain = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
    yval = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)
    ytest = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(DEVICE)

    xtrain = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    xval_raw = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    xtest_raw = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    mean_t = mean_tensor.to(DEVICE)
    std_t = std_tensor.to(DEVICE)
    xval = zscore_tensor(xval_raw, mean_t, std_t, device=DEVICE)
    xtest = zscore_tensor(xtest_raw, mean_t, std_t, device=DEVICE)

    input_size = xtrain.shape[1]
    print("Num features:", input_size)

    # Load noise vector
    try:
        noise_vector = load_noise_vector_from_json(noise_json_path)
        print(f"Loaded noise vector from {noise_json_path} (len={len(noise_vector)})")
    except Exception as e:
        print("Warning: noise JSON not loaded:", e)
        noise_vector = None

    if noise_vector is not None:
        fs = feature_stds.values.astype(np.float32)
        noise_vector = np.asarray(noise_vector, dtype=np.float32)
        if noise_vector.shape[0] != fs.shape[0]:
            raise ValueError("Noise vector length does not match num features")
        noise_vector_normalized = noise_vector / fs
        noise_vector_to_use = noise_vector_normalized
    else:
        noise_vector_to_use = None

    val_dataset = TensorDataset(xval, yval)
    test_dataset = TensorDataset(xtest, ytest)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_size=input_size, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE,
                dropout_prob=DROPOUT, use_layernorm=USE_LAYERNORM).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses, val_losses = [], []
    best_val, best_epoch = float('inf'), -1

    for epoch in range(1, num_epochs + 1):
        xtrain_norm = zscore_tensor(xtrain, mean_t, std_t, device=DEVICE)
        train_dataset = AugmentedTensorDataset((xtrain_norm, ytrain),
                                               noise_vector=noise_vector_to_use,
                                               nudge=nudge_factor,
                                               augment_prob=augment_prob,
                                               clip_range=clip_range,
                                               seed=AUG_SEED,
                                               device=DEVICE)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        train_batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_batch_losses.append(loss.item())

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_batch_losses.append(loss.item())

        avg_train, avg_val = np.mean(train_batch_losses), np.mean(val_batch_losses)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val, best_epoch = avg_val, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:03d}  Train MAE: {avg_train:.4f}  Val MAE: {avg_val:.4f}  (best {best_val:.4f} @ {best_epoch})")

    # Save training curves
    pd.DataFrame({'train': train_losses, 'val': val_losses}).to_csv('losses_different.csv', index=False)

    # Evaluate best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt'), map_location=DEVICE))
    model.eval()
    preds, test_losses = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_losses.append(loss.item())
            preds.extend(outputs.view(-1).tolist())

    test_mae = np.mean(test_losses)
    print(f"Test MAE: {test_mae:.4f} (best epoch {best_epoch})")

    results = pd.DataFrame({
        'actual': y_test,
        'prediction': np.array(preds)
    })
    results['diff'] = np.abs(results['actual'] - results['prediction'])
    results['realdiff'] = results['prediction'] - results['actual']
    results['FSID'] = test_df['FSID'].values
    results.to_csv('test_predictions_different.csv', index=False)

    # Plot
    actual, pred = results['actual'].to_numpy(), results['prediction'].to_numpy()
    min_val, max_val = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, pred, alpha=0.6, edgecolor='k')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('Actual vs Predicted Age (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_different.png')
    plt.show()

    save_feature_stats_df(df1_for_stats, 'feature_stats_different.csv')
    print('Done. Artifacts saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP model with optional augmentation and age filtering.")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to training CSV.")
    parser.add_argument("--noise", type=str, default=NOISE_JSON_PATH, help="Path to noise JSON.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="Where to save models and outputs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--nudge-factor", type=float, default=NUDGE_FACTOR)
    parser.add_argument("--augment-prob", type=float, default=AUGMENT_PROB)
    parser.add_argument("--no-age-filter", action="store_true",
                        help="If set, do NOT remove subjects age ≤ 50.")
    args = parser.parse_args()

    main(csv_path=args.csv,
         noise_json_path=args.noise,
         save_dir=args.save_dir,
         batch_size=args.batch_size,
         num_epochs=args.epochs,
         lr=args.lr,
         nudge_factor=args.nudge_factor,
         augment_prob=args.augment_prob,
         no_age_filter=args.no_age_filter)

