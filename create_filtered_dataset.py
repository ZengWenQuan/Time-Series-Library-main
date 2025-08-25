
import pandas as pd
import argparse
import os
import glob
from sklearn.model_selection import train_test_split

def load_and_combine(directory, filename):
    """Loads and concatenates a specific CSV file from train, val, and test subdirectories."""
    csv_files = glob.glob(os.path.join(directory, '**', filename), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No '{filename}' files found in subdirectories of {directory}")
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)

def main():
    """Main function to create a new, filtered, and re-split dataset."""
    parser = argparse.ArgumentParser(description="Create a new dataset from filtered predictions.")
    parser.add_argument('--filtered_dir', type=str, default='final_filtered_results/model2', help="Path to the directory with filtered results (e.g., 'final_filtered_results/model2').")
    parser.add_argument('--original_data_dir', type=str, default='dataset/split_data', help="Path to the original data directory.")
    parser.add_argument('--output_dir', type=str, default='dataset/filtered_resplit_data', help="Directory to save the new dataset splits.")
    parser.add_argument('--val_size', type=float, default=0.2, help="The proportion of the dataset to include in the validation split.")
    parser.add_argument('--random_state', type=int, default=42, help="Random state for reproducibility.")

    args = parser.parse_args()

    try:
        # --- 0. Create Output Directory ---
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output will be saved to: {args.output_dir}")

        # --- 1. Get the list of obsids to keep ---
        print(f"Loading filtered obsids from {args.filtered_dir}...")
        filtered_predictions_df = load_and_combine(args.filtered_dir, 'predictions.csv')
        obsids_to_keep = filtered_predictions_df['obsid'].unique()
        print(f"Found {len(obsids_to_keep)} unique obsids to keep.")

        # --- 2. Load the original full dataset ---
        print(f"Loading original full dataset from {args.original_data_dir}...")
        original_labels_df = load_and_combine(args.original_data_dir, 'labels.csv')
        original_continuum_df = load_and_combine(args.original_data_dir, 'continuum.csv')
        original_normalized_df = load_and_combine(args.original_data_dir, 'normalized.csv')

        # --- 3. Filter the original data to keep only the high-quality samples ---
        print("Filtering original data based on obsid list...")
        # Ensure obsid is the index for easy filtering
        original_labels_df.set_index('obsid', inplace=True)
        original_continuum_df.set_index('obsid', inplace=True)
        original_normalized_df.set_index('obsid', inplace=True)

        # Select only the rows with the obsids we want to keep
        labels_filtered = original_labels_df.loc[obsids_to_keep]
        continuum_filtered = original_continuum_df.loc[obsids_to_keep]
        normalized_filtered = original_normalized_df.loc[obsids_to_keep]

        print(f"Filtered dataset size: {len(labels_filtered)}")

        # --- 4. Perform Stratified Split based on FeH distribution ---
        print(f"Performing stratified split ({1-args.val_size:.0%} train / {args.val_size:.0%} val) based on FeH...")
        
        # Create bins for FeH to stratify on. This groups similar FeH values together.
        feh_bins = pd.cut(labels_filtered['FeH'], bins=20, labels=False)

        # Get the indices for the train and validation sets
        train_indices, val_indices = train_test_split(
            labels_filtered.index, 
            test_size=args.val_size, 
            random_state=args.random_state,
            stratify=feh_bins
        )

        print(f"New train set size: {len(train_indices)}")
        print(f"New validation set size: {len(val_indices)}")

        # --- 5. Create and Save the new splits ---
        print("Saving new data splits...")
        # Create directories
        train_output_dir = os.path.join(args.output_dir, 'train')
        val_output_dir = os.path.join(args.output_dir, 'val')
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(val_output_dir, exist_ok=True)

        # Select data based on split indices and save
        # Train set
        labels_filtered.loc[train_indices].to_csv(os.path.join(train_output_dir, 'labels.csv'))
        continuum_filtered.loc[train_indices].to_csv(os.path.join(train_output_dir, 'continuum.csv'))
        normalized_filtered.loc[train_indices].to_csv(os.path.join(train_output_dir, 'normalized.csv'))
        print(f"  - Saved new train set to {train_output_dir}")

        # Validation set
        labels_filtered.loc[val_indices].to_csv(os.path.join(val_output_dir, 'labels.csv'))
        continuum_filtered.loc[val_indices].to_csv(os.path.join(val_output_dir, 'continuum.csv'))
        normalized_filtered.loc[val_indices].to_csv(os.path.join(val_output_dir, 'normalized.csv'))
        print(f"  - Saved new validation set to {val_output_dir}")

        print("\nScript finished successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
