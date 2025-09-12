import os
import warnings
import numpy as np
import torch
import argparse
import datetime
import sys
import shutil
import ast
import random

# Only import the experiment classes we need
from exp.exp_regression import Exp_Regression
from exp.exp_spectral_prediction import Exp_Spectral_Prediction
from utils.print_args import print_args

warnings.filterwarnings('ignore')

def fix_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectral Analysis and Regression Library')

    # --- Basic Config ---
    parser.add_argument('--task_name', type=str, required=True, choices=['regression', 'spectral_prediction'],
                        help='task name: regression or spectral_prediction')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='A custom name for the model run')
    parser.add_argument('--model', type=str, required=True, default='CustomFusionNet', help='Name of the model to use')
    parser.add_argument('--model_conf', type=str, required=True, help='Path to the model-specific .yaml configuration file')

    # --- Data Config ---
    parser.add_argument('--data', type=str, required=True, default='stellar', help='Dataset type (e.g., stellar)')
    parser.add_argument('--root_path', type=str, default='./dataset/split_data', help='Root path of the data file directory')
    parser.add_argument('--stats_path', type=str, default='conf/stats.yaml', help='Path to the pre-computed statistics file')
    parser.add_argument('--split_ratio', type=str, default='[0.8, 0.1, 0.1]', help='Train/validation/test split ratio as a list string')
    parser.add_argument('--show_stats', action='store_true', help='Whether to show statistics of data during loading')
    parser.add_argument('--continuum_filename', type=str, default='continuum.csv', help='Filename for the continuum spectra data')
    parser.add_argument('--normalized_filename', type=str, default='normalized.csv', help='Filename for the normalized spectra data')
    parser.add_argument('--labels_filename', type=str, default='labels.csv', help='Filename for the labels data')

    # --- Task-Specific Config ---
    parser.add_argument('--feature_size', type=int, default=4704, help='Feature size for EACH input branch (continuum and normalized)')
    parser.add_argument('--label_size', type=int, default=4, help='Number of target labels to predict')
    parser.add_argument('--label_scaler_type', type=str, default='standard', help='Scaler type for labels (e.g., standard, minmax)')
    parser.add_argument('--features_scaler_type', type=str, default='standard', help='Scaler type for features (e.g., standard, minmax)')
    
    # --- FeH Sampling Config ---
    parser.add_argument('--feh_index', type=int, default=2, help='Index of FeH in the label array')

    # --- Optimization ---
    parser.add_argument('--num_workers', type=int, default=10, help='Data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--train_epochs', type=int, default=50, help='Train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size of train input data')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty)')
    parser.add_argument('--max_grad_norm', type=float, default=20.0, help='Max norm of the gradients for clipping')
    parser.add_argument('--vali_interval', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--checkpoints', type=str, default=None, help='Path to a checkpoint to resume training from')

    # --- Finetuning ---
    parser.add_argument('--freeze_body', action='store_true', help='Freeze the body of the model and only finetune the head')

    # --- Finetuning ---
    parser.add_argument('--disable_finetune', dest='do_finetune', action='store_false', help='Disable the finetuning phase (which is on by default)')
    parser.add_argument('--finetune_lr', type=float, default=1e-1, help='Optional: Custom learning rate for finetuning')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='Optional: Custom number of epochs for finetuning')

    # --- GPU ---
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU type to use (e.g., cuda, mps)')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1', help='Device IDs for multiple GPUs')

    # --- Augmentation ---
    parser.add_argument('--seed', type=int, default=42, help="Randomization seed")

    args = parser.parse_args()
    
    fix_seed(args.seed)
    
    if hasattr(args, 'split_ratio') and isinstance(args.split_ratio, str):
        try:
            args.split_ratio = ast.literal_eval(args.split_ratio)
        except (ValueError, SyntaxError):
            print(f"Warning: Invalid format for split_ratio. Using default [0.8, 0.1, 0.1]")
            args.split_ratio = [0.8, 0.1, 0.1]
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU: cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
        print('Using CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    #print('Args in experiment:')
    #print_args(args)

    # --- Simplified Experiment Selection ---
    if args.task_name == 'regression':
        Exp = Exp_Regression
    elif args.task_name == 'spectral_prediction':
        Exp = Exp_Spectral_Prediction
    else:
        # This case should not be reached due to 'choices' in parser
        raise ValueError(f"Unknown task name: {args.task_name}")

    if args.is_training:
        for ii in range(args.itr):
            # --- Set up folder for this run ---
            setting = f'{args.model_id}_{args.task_name}_{args.model}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{ii}'
            run_dir = os.path.join('runs', args.task_name, args.model, datetime.datetime.now().strftime("%Y%m%d_%H%M%S-finetune"))
            os.makedirs(run_dir, exist_ok=True)
            args.run_dir = run_dir

            # Save model config used for this run
            if hasattr(args, 'model_conf') and args.model_conf and os.path.exists(args.model_conf):
                shutil.copy2(args.model_conf, run_dir)

            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp = Exp(args)
            exp.train()
        
            print(f'>>>>>>>testing after training : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test()

            if args.do_finetune:
                print(f'>>>>>>>start finetuning : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                exp.finetune(finetune_lr=args.finetune_lr, finetune_epochs=args.finetune_epochs)

                print(f'>>>>>>>testing after finetuning : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.test()

            torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model_id))
        exp.test()