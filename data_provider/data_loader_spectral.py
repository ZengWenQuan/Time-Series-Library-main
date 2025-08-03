import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import yaml
from utils.stellar_metrics import Scaler


class Dataset_Spectral(Dataset):
    def __init__(self, args, flag='train', label_scaler=None, feature_scaler=None):
        self.args = args
        self.feature_size = args.feature_size
        self.label_size = args.label_size
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.targets = args.targets
        self.ratio = args.split_ratio
        self.root_path = args.root_path
        
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        
        self.__read_data__()

    def __read_data__(self):
        # Load the datasets
        features_continuum_df = pd.read_csv(os.path.join(self.root_path, self.args.spectra_continuum_path))
        features_normalized_df = pd.read_csv(os.path.join(self.root_path, self.args.spectra_normalized_path))
        labels_df = pd.read_csv(os.path.join(self.root_path, self.args.label_path))
        
        # Find common obsids
        common_obsids = list(set(features_continuum_df['obsid']) & set(features_normalized_df['obsid']) & set(labels_df['obsid']))
        
        # Filter and sort dataframes by common obsids
        features_continuum_df = features_continuum_df[features_continuum_df['obsid'].isin(common_obsids)].sort_values(by='obsid').reset_index(drop=True)
        features_normalized_df = features_normalized_df[features_normalized_df['obsid'].isin(common_obsids)].sort_values(by='obsid').reset_index(drop=True)
        labels_df = labels_df[labels_df['obsid'].isin(common_obsids)].sort_values(by='obsid').reset_index(drop=True)

        # Get feature and target columns based on feature_size
        feature_cols_continuum = features_continuum_df.columns[1:self.feature_size+1]
        feature_cols_normalized = features_normalized_df.columns[1:self.feature_size+1]

        # Separate feature sets
        data_x_continuum = features_continuum_df[feature_cols_continuum].values
        data_x_normalized = features_normalized_df[feature_cols_normalized].values
        data_y = labels_df[self.targets].values

        # Split data based on ratio
        total_samples = len(common_obsids)
        train_ratio, val_ratio, _ = self.ratio
        train_boundary = int(total_samples * train_ratio)
        val_boundary = int(total_samples * (train_ratio + val_ratio))

        # Transform only the continuum features
        if self.feature_scaler:
            data_x_continuum_scaled = self.feature_scaler.transform(data_x_continuum)
        else:
            data_x_continuum_scaled = data_x_continuum
        
        # Concatenate scaled continuum features with the original normalized features
        data_x = np.concatenate((data_x_continuum_scaled, data_x_normalized), axis=1)

        # Transform labels if a scaler is provided
        if self.label_scaler:
            data_y_scaled = self.label_scaler.transform(data_y)
        else:
            data_y_scaled = data_y
        
        # Define borders for train, val, test sets
        if self.set_type == 0:  # train
            border1, border2 = 0, train_boundary
        elif self.set_type == 1:  # val
            border1, border2 = train_boundary, val_boundary
        else:  # test
            border1, border2 = val_boundary, total_samples

        self.data_x = data_x[border1:border2]
        self.data_y = data_y_scaled[border1:border2]
        self.obsids = labels_df.iloc[border1:border2, 0].values
        self.raw_data_y = data_y[border1:border2]

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]

        # Dummy values for mark arrays as they are not used in this task
        seq_x_mark = np.zeros((seq_x.shape[0], 1))
        seq_y_mark = np.zeros((seq_y.shape[0], 1))
        obsid = self.obsids[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, obsid

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        if self.label_scaler:
            return self.label_scaler.inverse_transform(data)
        else:
            return data
            
    def get_raw_targets(self):
        return self.raw_data_y
