import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import glob

def load_data_splits(directory):
    """Loads predictions.csv from train, val, and test subdirectories and returns a dictionary of DataFrames."""
    splits = ['train', 'val', 'test']
    data_splits = {}
    for split in splits:
        file_path = os.path.join(directory, split, 'predictions.csv')
        if os.path.exists(file_path):
            print(f"Loading data for '{split}' from {directory}")
            data_splits[split] = pd.read_csv(file_path)
        else:
            print(f"Warning: No predictions.csv found for '{split}' split in {directory}")
    if not data_splits:
        raise FileNotFoundError(f"No data splits found in {directory}")
    return data_splits

def generate_plots_for_model(df, model_name, split_name, targets, output_dir):
    """Generates and saves one comprehensive scatter plot with 4 subplots for a single model and split."""
    print(f"  - Generating plot for {model_name} - {split_name} split...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Model: {model_name} - Split: {split_name} - Predicted vs. True (Filtered)', fontsize=16)

    for ax, target in zip(axes.flatten(), targets):
        true_col = f'{target}_true'
        pred_col = f'{target}_pred'
        
        if true_col not in df.columns or pred_col not in df.columns:
            print(f"    - Warning: Skipping subplot for '{target}' due to missing columns.")
            continue

        ax.scatter(df[true_col], df[pred_col], alpha=0.5, s=10)
        
        lims = [
            min(df[true_col].min(), df[pred_col].min()),
            max(df[true_col].max(), df[pred_col].max()),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(target)
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, f'{model_name}_{split_name}_scatter.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    - Saved plot to {plot_path}")

def main():
    """Main function to filter and visualize prediction results from directories."""
    parser = argparse.ArgumentParser(description="Filter and visualize prediction directories based on combined FeH error.")
    parser.add_argument('--dir1', type=str, default='dataset/test_all_results_flx1', help="Path to the first model's result directory.")
    parser.add_argument('--dir2', type=str, default='dataset/test_all_results_flx2', help="Path to the second model's result directory.")
    parser.add_argument('--percentage', type=float, default=10.0, help="The percentage of samples with the highest error to filter out (e.g., 10 for 10%%).")
    parser.add_argument('--output_dir', type=str, default='final_filtered_results', help="Directory to save all outputs.")
    
    args = parser.parse_args()

    try:
        # --- 1. Load Data Splits ---
        model1_dfs = load_data_splits(args.dir1)
        model2_dfs = load_data_splits(args.dir2)

        # --- 2. Combine data for error analysis ---
        df1_combined = pd.concat(model1_dfs.values(), ignore_index=True).set_index('obsid')
        df2_combined = pd.concat(model2_dfs.values(), ignore_index=True).set_index('obsid')

        # --- 3. Calculate Combined Error ---
        print("\nCalculating combined FeH error on full dataset...")
        error1 = (df1_combined['FeH_true'] - df1_combined['FeH_pred']).abs()
        error2 = (df2_combined['FeH_true'] - df2_combined['FeH_pred']).abs()

        analysis_df = pd.DataFrame({
            'FeH_true': df1_combined['FeH_true'],
            'total_error': error1 + error2
        }).dropna()

        # --- 4. Identify Samples to Drop ---
        print(f"Identifying samples to filter, protecting where FeH_true < -3...")
        filterable_samples = analysis_df[analysis_df['FeH_true'] >= -3]

        k = int(len(filterable_samples) * (args.percentage / 100.0))
        print(f"Filtering top {args.percentage}%% of {len(filterable_samples)} filterable samples, which is {k} samples.")

        if k > 0:
            samples_to_drop = filterable_samples.nlargest(k, 'total_error')
            obsids_to_drop = samples_to_drop.index
            print(f"\n--- Dropping {len(obsids_to_drop)} samples ---")
            print(samples_to_drop.head())
        else:
            obsids_to_drop = pd.Index([])
            print("\n--- No samples to drop ---")
        print("-------------------------------------")

        # --- 5. Filter, Save, and Plot Individual Splits ---
        print("\nFiltering, saving, and plotting individual splits...")
        targets = ['Teff', 'logg', 'FeH', 'CFe']

        for model_num, model_dfs in enumerate([model1_dfs, model2_dfs], 1):
            model_name = f"Model{model_num}"
            output_dir_model = os.path.join(args.output_dir, model_name.lower())
            print(f"Processing {model_name}...")

            for split_name, df in model_dfs.items():
                filtered_df = df[~df['obsid'].isin(obsids_to_drop)].copy()
                output_path_split = os.path.join(output_dir_model, split_name)
                os.makedirs(output_path_split, exist_ok=True)
                
                # Save filtered CSV
                filtered_df.to_csv(os.path.join(output_path_split, 'predictions.csv'), index=False)
                print(f"  - Saved filtered '{split_name}' to {output_path_split}")
                
                # Generate and save plot for this split
                generate_plots_for_model(filtered_df, model_name, split_name, targets, output_path_split)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()