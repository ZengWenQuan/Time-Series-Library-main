
import argparse
import os
import pandas as pd
import numpy as np
import yaml

def calculate_statistics(series):
    """Calculates a dictionary of statistics for a pandas Series."""
    return {
        'mean': float(series.mean()),
        'variance': float(series.var()),
        'std_dev': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        '25th_percentile': float(series.quantile(0.25)),
        'median_50th_percentile': float(series.quantile(0.5)),
        '75th_percentile': float(series.quantile(0.75)),
    }

def main():
    """Main function to update stats.yaml from training data for fixed targets."""
    parser = argparse.ArgumentParser(description="Update stats.yaml with statistics from the training set.")
    parser.add_argument('--train_dir', type=str, default='dataset/split_data/train', help="Path to the training data directory.")
    parser.add_argument('--stats_path', type=str, default='conf/stats.yaml', help="Path to the stats.yaml file to update.")
    args = parser.parse_args()

    # --- 1. Define fixed target list ---
    targets = ['Teff', 'logg', 'FeH', 'CFe']
    print(f"Updating statistics for fixed targets: {targets}")

    print(f"Updating '{args.stats_path}' using data from '{args.train_dir}'...")

    # --- 2. Define and check data file paths ---
    labels_path = os.path.join(args.train_dir, 'labels.csv')
    continuum_path = os.path.join(args.train_dir, 'continuum.csv')
    normalized_path = os.path.join(args.train_dir, 'normalized.csv')

    for path in [labels_path, continuum_path, normalized_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at {path}")
            return

    # --- 3. Load data ---
    print("Loading data...")
    labels_df = pd.read_csv(labels_path)
    continuum_df = pd.read_csv(continuum_path)
    normalized_df = pd.read_csv(normalized_path)

    # --- 4. Calculate new statistics ---
    print("Calculating new statistics...")
    new_stats = {}

    # Calculate stats only for specified fixed targets
    for target_col in targets:
        if target_col in labels_df.columns:
            print(f"  - Calculating stats for target: {target_col}")
            new_stats[target_col] = calculate_statistics(labels_df[target_col])
        else:
            print(f"  - Warning: Target '{target_col}' not found in labels.csv. Skipping.")

    # Calculate stats for combined flux
    print("  - Calculating stats for flux...")
    all_flux_values = pd.concat([continuum_df.iloc[:, 1:].stack(), normalized_df.iloc[:, 1:].stack()])
    new_stats['flux'] = calculate_statistics(all_flux_values)

    # --- 5. Load existing stats.yaml to preserve other keys ---
    try:
        with open(args.stats_path, 'r') as f:
            existing_stats = yaml.safe_load(f)
        print(f"Loaded existing stats from '{args.stats_path}'.")
    except FileNotFoundError:
        print(f"Warning: '{args.stats_path}' not found. A new file will be created.")
        existing_stats = {}

    # --- 6. Update the stats dictionary ---
    print("Updating statistics...")
    existing_stats.update(new_stats)

    # --- 7. Write the updated stats back to the file ---
    try:
        with open(args.stats_path, 'w') as f:
            yaml.dump(existing_stats, f, sort_keys=False, default_flow_style=False, indent=2)
        print(f"Successfully updated '{args.stats_path}'.")
    except Exception as e:
        print(f"Error writing to '{args.stats_path}': {e}")

if __name__ == '__main__':
    main()
