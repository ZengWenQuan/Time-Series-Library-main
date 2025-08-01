
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class Dataset_Spectral(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # Load the datasets
        features_continuum_df = pd.read_csv(os.path.join(self.root_path, 'final_spectra_continuum.csv'))
        features_normalized_df = pd.read_csv(os.path.join(self.root_path, 'final_spectra_normalized.csv'))
        labels_df = pd.read_csv(os.path.join(self.root_path, 'removed_with_rv.csv'))
        
        # Merge the datasets on 'obsid'
        df_merged = pd.merge(features_continuum_df, labels_df, on='obsid')
        df_merged = pd.merge(df_merged, features_normalized_df, on='obsid', suffixes=['_continuum', '_normalized'])

        # Get feature and target columns
        feature_cols_continuum = [col for col in features_continuum_df.columns if col != 'obsid']
        feature_cols_normalized = [col for col in features_normalized_df.columns if col != 'obsid']
        target_cols = ['Teff', 'logg', 'FeH', 'CFe']

        # Separate features and targets
        self.data_x_continuum = df_merged[feature_cols_continuum].values
        self.data_x_normalized = df_merged[feature_cols_normalized].values
        self.data_y = df_merged[target_cols].values

        # Combine the two feature sets
        self.data_x = np.concatenate((self.data_x_continuum, self.data_x_normalized), axis=1)

        # Split data
        num_train = int(len(df_merged) * 0.7)
        num_test = int(len(df_merged) * 0.2)
        num_vali = len(df_merged) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_merged) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_merged)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = self.data_x[border1:border2]
        self.data_y = self.data_y[border1:border2]

    def __getitem__(self, index):
        # The model will expect seq_x, seq_y, seq_x_mark, seq_y_mark
        # For this simple case, we can return dummy values for the mark arrays
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        return seq_x, seq_y, np.zeros_like(seq_x), np.zeros_like(seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1
