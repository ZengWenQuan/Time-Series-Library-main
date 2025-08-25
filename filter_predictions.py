
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f'Model: {model_name} - Split: {split_name} - Predicted vs. True (Filtered)', fontsize=16)

    for ax, target in zip(axes.flatten(), targets):
        true_col = f'{target}_true'
        pred_col = f'{target}_pred'
        
        if true_col not in df.columns or pred_col not in df.columns:
            print(f"    - Warning: Skipping subplot for '{target}' due to missing columns.")
            continue

        true_vals = df[true_col]
        pred_vals = df[pred_col]

        ax.scatter(true_vals, pred_vals, alpha=0.5, s=10)
        
        lims = [
            min(true_vals.min(), pred_vals.min()),
            max(true_vals.max(), pred_vals.max()),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(target)
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.grid(True)

        # Calculate and add metrics to the plot
        mae = mean_absolute_error(true_vals, pred_vals)
        mse = mean_squared_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)
        metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, f'{model_name}_{split_name}_scatter.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    - Saved plot to {plot_path}")

def main():
    """Main function to filter and visualize prediction results from directories."""
    parser = argparse.ArgumentParser(description="Filter and visualize prediction directories based on combined error for multiple labels.")
    parser.add_argument('--dir1', type=str, default='dataset/test_all_results_flx1', help="Path to the first model's result directory.")
    parser.add_argument('--dir2', type=str, default='dataset/test_all_results_flx2', help="Path to the second model's result directory.")
    parser.add_argument('--output_dir', type=str, default='final_filtered_results', help="Directory to save all outputs.")
    parser.add_argument('--p_teff', type=float, default=5.0, help="Percentage of Teff samples to filter out.")
    parser.add_argument('--p_logg', type=float, default=5.0, help="Percentage of logg samples to filter out.")
    parser.add_argument('--p_feh', type=float, default=10.0, help="Percentage of FeH samples to filter out.")
    parser.add_argument('--p_cfe', type=float, default=5.0, help="Percentage of CFe samples to filter out.")

    args = parser.parse_args()

    try:
        # --- 0. Create Output Directory ---
        os.makedirs(args.output_dir, exist_ok=True)

        # --- 1. Load Data Splits ---
        model1_dfs = load_data_splits(args.dir1)
        model2_dfs = load_data_splits(args.dir2)

        # --- 2. Combine data for error analysis ---
        df1_combined = pd.concat(model1_dfs.values(), ignore_index=True).set_index('obsid')
        df2_combined = pd.concat(model2_dfs.values(), ignore_index=True).set_index('obsid')

        # --- 3. Calculate Combined Error for All Targets ---
        print("\nCalculating combined errors for all targets...")
        targets = ['Teff', 'logg', 'FeH', 'CFe']
        analysis_df = pd.DataFrame({'FeH_true': df1_combined['FeH_true']})

        for target in targets:
            true_col = f'{target}_true'
            pred1_col = f'{target}_pred'
            pred2_col = f'{target}_pred'
            error1 = (df1_combined[true_col] - df1_combined[pred1_col]).abs()
            error2 = (df2_combined[true_col] - df2_combined[pred2_col]).abs()
            analysis_df[f'{target}_total_error'] = error1 + error2
        
        analysis_df.dropna(inplace=True)

        # --- 4. Identify Samples to Drop based on multiple criteria ---
        print(f"Identifying samples to filter, protecting where FeH_true < -3...")
        filterable_samples = analysis_df[analysis_df['FeH_true'] >= -3] 
        
        obsids_to_drop = set()
        percentages = {'Teff': args.p_teff, 'logg': args.p_logg, 'FeH': args.p_feh, 'CFe': args.p_cfe}

        for target, percentage in percentages.items():
            if percentage > 0:
                k = int(len(filterable_samples) * (percentage / 100.0))
                print(f"- For '{target}', filtering top {percentage}%%, which is {k} samples.")
                if k > 0:
                    top_k_errors = filterable_samples.nlargest(k, f'{target}_total_error')
                    obsids_to_drop.update(top_k_errors.index)

        print(f"\n--- Total unique samples to be dropped: {len(obsids_to_drop)} ---")

        # --- 5. Filter, Save, and Plot Individual Splits ---
        print("\nFiltering, saving, and plotting individual splits...")

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
