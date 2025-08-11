import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import argparse
import matplotlib.pyplot as plt
import datetime
import sys
import shutil
import ast
import random

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_regression import Exp_Regression
from exp.exp_spectral_prediction import Exp_Spectral_Prediction
from exp.exp_dualpyramidnet import Exp_DualPyramidNet
from exp.exp_dualspectralnet import Exp_DualSpectralNet
from utils.print_args import print_args

warnings.filterwarnings('ignore')

def fix_seed(seed):
    """
    设置随机种子以确保实验可重复性
    """
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
    parser = argparse.ArgumentParser(description='Time Series Library')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, regression]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--model_conf', type=str, default=None, help='path to the model-specific configuration file')

    # -- data parser
    parser.add_argument('--data', type=str, required=True, default='ETTm1',
                        help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/spectral/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='spectral_data.csv', help='data file')
    parser.add_argument('--split_data_path', type=str, default='./dataset/split_data', help='path to the split data directory')
    parser.add_argument('--continuum_filename', type=str, default='continuum.csv', help='filename for the continuum spectra data')
    parser.add_argument('--normalized_filename', type=str, default='normalized.csv', help='filename for the normalized spectra data')
    parser.add_argument('--labels_filename', type=str, default='labels.csv', help='data file for labels')
    parser.add_argument('--feature_filename', type=str, default='features.csv', help='data file for features')
    parser.add_argument('--show_stats', action='store_true', help='whether to show statistics of data during loading')
    parser.add_argument('--plot_loss', type=bool, default=False, help='whether to plot the loss')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    parser.add_argument('--checkpoints', type=str, default=None, help='path to a checkpoint to resume training from')

    # forecasting task
    parser.add_argument('--feature_size', type=int, default=4798, help='feature size for EACH input branch (continuum and normalized)')
    parser.add_argument('--label_size', type=int, default=4, help='label size')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # regression task
    parser.add_argument('--apply_inverse_transform', type=int, default=0, help='whether to apply inverse transform on the output')
    parser.add_argument('--label_scaler_type', type=str, default='standard', help='scaler type for label')
    parser.add_argument('--split_ratio', type=str, default='[0.8, 0.1, 0.1]', help='train/val/test split ratio')
    parser.add_argument('--targets', type=str, default="['Teff', 'logg', 'FeH', 'CFe']", help='target columns for regression')
    parser.add_argument('--features_scaler_type', type=str, default='standard', help='scaler type for features')
    
    # FeH采样相关参数
    parser.add_argument('--use_feh_sampling', action='store_true', help='是否使用基于FeH的过采样/欠采样', default=False)
    parser.add_argument('--feh_sampling_strategy', type=str, default='balanced', 
                        help='FeH采样策略: auto(所有类别采样到最多类别数量), balanced(所有类别采样到平均数量), 也可以是字典形式的字符串')
    parser.add_argument('--feh_sampling_k_neighbors', type=int, default=5, help='FeH采样中SMOTE算法的邻居数量')
    parser.add_argument('--feh_index', type=int, default=2, help='FeH在标签数组中的索引位置')

    # model define
    parser.add_argument('--expand', type=int, default=0, help='expand channels')
    parser.add_argument('--d_conv', type=int, default=1, help='d_conv')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1, help='output channel independence')
    parser.add_argument('--decomp_method', type=str, default='None', help='decomposition method')
    parser.add_argument('--use_norm', type=int, default=1, help='use norm')
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=4, help='down sampling window')
    parser.add_argument('--down_sampling_method', type=str, default='avg', help='down sampling method')
    parser.add_argument('--seg_len', type=int, default=24, help='segment length')
    parser.add_argument('--list_inplanes', nargs='+', type=int, default=[1, 2, 2, 2], help='list inplanes')
    parser.add_argument('--num_rnn_sequence', type=int, default=1, help='num rnn sequence')
    parser.add_argument('--embedding_c', type=int, default=1, help='embedding c')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--batch_norm', type=int, default=1, help='batch norm')
    parser.add_argument('--n_fft', type=int, default=256, help='n fft')
    parser.add_argument('--hop_length', type=int, default=64, help='hop length')
    parser.add_argument('--conv_channel_1', type=int, default=16, help='conv channel 1')
    parser.add_argument('--conv_channel_2', type=int, default=32, help='conv channel 2')
    parser.add_argument('--conv_channel_3', type=int, default=64, help='conv channel 3')
    parser.add_argument('--inception_channel_1', type=int, default=16, help='inception channel 1')
    parser.add_argument('--inception_channel_2', type=int, default=32, help='inception channel 2')
    parser.add_argument('--inception_channel_3', type=int, default=64, help='inception channel 3')
    parser.add_argument('--pool_size', type=int, default=2, help='pool size')
    parser.add_argument('--ffn_hidden_size', type=int, default=128, help='ffn hidden size')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30000, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=20.0, help='max norm of the gradients')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--loss_threshold', type=float, default=100000.0, help='threshold for skipping batches with abnormally high loss')
    parser.add_argument('--lradj', type=str, default='warmup_cosine', help='adjust learning rate, options: [warmup_cosine, cos, step, exponential]')
    parser.add_argument('--cosine_t0', type=int, default=100, help='The number of epochs for the first restart of the cosine annealing scheduler.')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--vali_interval', type=int, default=5, help='vali interval')

    # DWT
    parser.add_argument('--dwt_level', type=int, default=3, help='dwt level')
    parser.add_argument('--wavelet', type=str, default='db4', help='wavelet')
    parser.add_argument('--ffn_dim', type=int, default=128, help='ffn dim')

    # Focal Loss
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma')
    parser.add_argument('--focal_threshold', type=float, default=0.5, help='focal loss threshold')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stats_path', type=str, default=None, help='path to the stats file')
    parser.add_argument('--spectra_continuum_path', type=str, default='final_spectra_continuum.csv', help='path to the spectra continuum file')
    parser.add_argument('--spectra_normalized_path', type=str, default='final_spectra_normalized.csv', help='path to the spectra normalized file')
    parser.add_argument('--label_path', type=str, default='removed_with_rv.csv', help='path to the label file')

    args = parser.parse_args()
    
    # 设置随机种子
    fix_seed(args.seed)
    
    # 解析字符串形式的列表参数
    if hasattr(args, 'split_ratio') and isinstance(args.split_ratio, str):
        try:
            args.split_ratio = ast.literal_eval(args.split_ratio)
        except (ValueError, SyntaxError):
            print("警告: split_ratio 参数格式错误，使用默认值 [0.8, 0.1, 0.1]")
            args.split_ratio = [0.8, 0.1, 0.1]
    
    if hasattr(args, 'targets') and isinstance(args.targets, str):
        try:
            args.targets = ast.literal_eval(args.targets)
        except (ValueError, SyntaxError):
            print("警告: targets 参数格式错误，使用默认值 ['Teff', 'logg', 'FeH', 'CFe']")
            args.targets = ['Teff', 'logg', 'FeH', 'CFe']
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'regression':
        Exp = Exp_Regression
    elif args.task_name == 'spectral_prediction':
            Exp = Exp_Spectral_Prediction
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        
        # 设置实验记录
        # 获取当前时间并加上8小时
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        
        # 创建runs目录结构
        runs_dir = os.path.join("runs", args.task_name, args.model)
        os.makedirs(runs_dir, exist_ok=True)
        
        # 创建时间戳目录
        run_dir = os.path.join(runs_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # 保存当前实验的模型配置文件
        if hasattr(args, 'model_conf') and args.model_conf and os.path.exists(args.model_conf):
            try:
                shutil.copy2(args.model_conf, run_dir)
                print(f"Saved model config to {run_dir}")
            except Exception as e:
                print(f"Warning: Could not save model config file. Error: {e}")
        
        # 创建metrics目录
        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # 保存当前使用的脚本
        script_path = sys.argv[0]
        if script_path != "run.py" and os.path.exists(script_path):
            script_dir = os.path.join(run_dir, "scripts")
            os.makedirs(script_dir, exist_ok=True)
            shutil.copy2(script_path, script_dir)
        else:
            # 创建一个包含所有参数的脚本
            script_dir = os.path.join(run_dir, "scripts")
            os.makedirs(script_dir, exist_ok=True)
            script_file = os.path.join(script_dir, f"train_{args.model}.sh")
            with open(script_file, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("# 激活conda环境\n")
                f.write("conda activate mp\n\n")
                f.write("# 运行训练\n")
                f.write("python run.py \\\n")
                for arg, value in vars(args).items():
                    if arg not in ['device', 'device_ids']:
                        if isinstance(value, bool):
                            if value:
                                f.write(f"  --{arg} \\\n")
                        elif isinstance(value, list):
                            f.write(f"  --{arg} {' '.join(map(str, value))} \\\n")
                        elif value is not None:
                            f.write(f"  --{arg} {value} \\\n")
            os.chmod(script_file, 0o755)  # 添加执行权限
        for ii in range(args.itr):
            
            
            # 设置实验参数
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            # 将metrics_dir传递给实验
            args.run_dir = run_dir

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp = Exp(args)  # 设置实验
            exp.train()

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test()
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        # 测试模式下也创建runs目录
        if args.is_training == 0:
            # 获取当前时间并加上8小时
            current_time = datetime.datetime.now() + datetime.timedelta(hours=8)
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            
            runs_dir = os.path.join("runs", args.task_name, args.model)
            os.makedirs(runs_dir, exist_ok=True)
            run_dir = os.path.join(runs_dir, timestamp)
            os.makedirs(run_dir, exist_ok=True)
            metrics_dir = os.path.join(run_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # 保存当前使用的脚本
            script_path = sys.argv[0]
            if script_path != "run.py" and os.path.exists(script_path):
                script_dir = os.path.join(run_dir, "scripts")
                os.makedirs(script_dir, exist_ok=True)
                shutil.copy2(script_path, script_dir)
            else:
                # 创建一个包含所有参数的脚本
                script_dir = os.path.join(run_dir, "scripts")
                os.makedirs(script_dir, exist_ok=True)
                script_file = os.path.join(script_dir, f"test_{args.model}.sh")
                with open(script_file, "w") as f:
                    f.write("#!/bin/bash\n\n")
                    f.write("# 激活conda环境\n")
                    f.write("conda activate mp\n\n")
                    f.write("# 运行测试\n")
                    f.write("python run.py \\\n")
                    for arg, value in vars(args).items():
                        if arg not in ['device', 'device_ids']:
                            if isinstance(value, bool):
                                if value:
                                    f.write(f"  --{arg} \\\n")
                            elif isinstance(value, list):
                                f.write(f"  --{arg} {' '.join(map(str, value))} \\\n")
                            elif value is not None:
                                f.write(f"  --{arg} {value} \\\n")
                os.chmod(script_file, 0o755)  # 添加执行权限
            
            args.metrics_dir = metrics_dir

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test()
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
