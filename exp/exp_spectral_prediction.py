from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.stellar_metrics import calculate_metrics, save_regression_metrics, calculate_feh_classification_metrics, save_feh_classification_metrics, save_history_plot
import os
import yaml
import warnings
from utils.scaler import Scaler
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob

warnings.filterwarnings('ignore')


class Exp_Spectral_Prediction(Exp_Basic):
    """
    恒星参数估计（Stellar Parameter Estimation）实验类
    """

    def __init__(self, args):
        super(Exp_Spectral_Prediction, self).__init__(args)
        self.label_scaler = self.get_label_scaler()
        self.feature_scaler = self.get_feature_scaler()

        sample_batch = None
        # Only load data if not in prediction-only mode
        if not getattr(self.args, 'predict', False):
            self._get_data()
            if self.train_loader:
                sample_batch, _, _ = next(iter(self.train_loader))
                sample_batch = sample_batch.float().to(self.device)

        self.model = self._build_model(sample_batch=sample_batch)

    def _get_data(self):
        self.train_data, self.train_loader = data_provider(args=self.args, flag='train',
                                                          feature_scaler=self.feature_scaler,
                                                          label_scaler=self.label_scaler)
        self.vali_data, self.vali_loader = data_provider(args=self.args, flag='val',
                                                         feature_scaler=self.feature_scaler,
                                                         label_scaler=self.label_scaler)

    def _get_finetune_data(self):
        """
        Loads the finetuning dataset by setting a temporary flag in args.
        """
        # Set is_finetune flag in args
        self.args.is_finetune = True
        self.logger.info("Loading finetuning dataset...")

        self.finetune_train_data, self.finetune_train_loader = data_provider(args=self.args, flag='train',
                                                                             feature_scaler=self.feature_scaler,
                                                                             label_scaler=self.label_scaler)
        self.finetune_vali_data, self.finetune_vali_loader = data_provider(args=self.args, flag='val',
                                                                          feature_scaler=self.feature_scaler,
                                                                          label_scaler=self.label_scaler)

        test_path = os.path.join(self.args.root_path, 'test')
        self.finetune_test_data, self.finetune_test_loader = data_provider(args=self.args, flag='test',
                                                                          feature_scaler=self.feature_scaler,
                                                                          label_scaler=self.label_scaler) if os.path.exists(
            test_path) else (None, None)

        # Unset the flag to avoid side effects in other parts of the code
        self.args.is_finetune = False
        self.logger.info("Finetuning dataset loaded.")

    def get_feature_scaler(self):
        if self.args.stats_path:
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)
            return Scaler(scaler_type=self.args.features_scaler_type, stats_dict={'flux': stats['flux']},
                          target_names=['flux'])
        raise ValueError("没有提供特征统计数据文件路径")

    def get_label_scaler(self):
        if self.args.stats_path:
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)
            return Scaler(scaler_type=self.args.label_scaler_type, stats_dict=stats, target_names=self.targets)
        raise ValueError("没有提供标签统计数据文件路径")

    def calculate_and_save_all_metrics(self, preds, trues, phase, save_as):
        if preds is None or trues is None: return None
        # self.logger.info(f"Calculating and saving {save_as} metrics for {phase} set...")
        save_path = os.path.join(self.args.run_dir, 'metrics', save_as)

        reg_metrics = calculate_metrics(preds, trues, self.args.targets)
        save_regression_metrics(reg_metrics, save_path, self.args.targets, phase=phase)

        cls_metrics = calculate_feh_classification_metrics(preds, trues, self.args.feh_index)
        save_feh_classification_metrics(cls_metrics, save_path, phase=phase)
        return reg_metrics

    def predict_folder(self):
        # 1. Get paths
        predict_data_path = self.args.predict_data_path
        if not predict_data_path or not os.path.isdir(predict_data_path):
            self.logger.error(f"Invalid or missing --predict_data_path: {predict_data_path}")
            return

        # 2. Create output directory
        parent_dir = os.path.dirname(predict_data_path.rstrip('/'))
        dir_name = os.path.basename(predict_data_path.rstrip('/'))
        output_dir = os.path.join(parent_dir, f"{dir_name}_csv")
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Prediction results will be saved to: {output_dir}")

        # 3. Load model
        checkpoint_path = self.args.checkpoints
        if not (checkpoint_path and os.path.exists(checkpoint_path)):
            self.logger.error(f"No valid checkpoint path provided via --checkpoints. Aborting.")
            return
        self.logger.info(f"Loading model from provided checkpoint: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
        self.model.eval()

        # 4. Find feather files
        feather_files = glob.glob(os.path.join(predict_data_path, '*.feather'))
        if not feather_files:
            self.logger.warning(f"No .feather files found in {predict_data_path}")
            return
        self.logger.info(f"Found {len(feather_files)} feather files to predict.")

        # 5. Define custom Dataset for prediction
        class PredictionDataset(Dataset):
            def __init__(self, file_path, feature_scaler=None):
                df = pd.read_feather(file_path)
                self.obsids = None
                if 'obsid' in df.columns:
                    self.obsids = df['obsid'].values
                    df_features = df.drop(columns=['obsid'])
                else:
                    df_features = df

                self.data_x = df_features.values
                if feature_scaler:
                    self.data_x = feature_scaler.transform(self.data_x)

            def __getitem__(self, index):
                # Return features and obsid if available
                if self.obsids is not None:
                    return self.data_x[index], self.obsids[index]
                else:
                    return self.data_x[index], index  # Use row index as a fallback ID

            def __len__(self):
                return len(self.data_x)

        # 6. Loop through files and predict
        for file_path in feather_files:
            filename = os.path.basename(file_path)
            self.logger.info(f"Predicting on file: {filename}")

            dataset = PredictionDataset(file_path, self.feature_scaler)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                                    num_workers=self.args.num_workers, drop_last=False)

            all_preds = []
            all_obsids = []
            with torch.no_grad():
                for batch_x, batch_obsids in dataloader:
                    outputs = self.model(batch_x.float().to(self.device))
                    preds = outputs.detach().cpu().numpy()
                    all_preds.append(preds)
                    all_obsids.extend(list(batch_obsids))

            all_preds = np.concatenate(all_preds, axis=0)

            if self.label_scaler:
                all_preds = self.label_scaler.inverse_transform(all_preds)

            # Create result DataFrame
            pred_df = pd.DataFrame()
            pred_df['obsid'] = all_obsids

            for i, target_name in enumerate(self.targets):
                pred_df[f'{target_name}_pred'] = all_preds[:, i]

            # Save to CSV
            output_filename = os.path.splitext(filename)[0] + '.csv'
            output_path = os.path.join(output_dir, output_filename)
            pred_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved predictions to {output_path}")

        self.logger.info("--- Folder prediction completed ---")