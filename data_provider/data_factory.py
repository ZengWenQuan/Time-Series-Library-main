from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader,     MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.data_loader_spectral import Dataset_Spectral
from data_provider.steller import  Dataset_Steller
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import numpy as np
import random
import torch

def fix_seed_worker(worker_id):
    """
    为DataLoader的每个worker设置随机种子
    """
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'steller': Dataset_Steller,
    'spectral': Dataset_Spectral
}


def data_provider(args, flag, label_scaler=None, feature_scaler=None, has_labels=True):
    Data = data_dict[args.data] # 根据args.dataset选择数据集类
    # timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST' or not has_labels) else True
    drop_last = False
    batch_size = args.batch_size
    # freq = args.freq

    # if args.task_name == 'anomaly_detection':
    #     drop_last = False
    #     data_set = Data(
    #         args = args,
    #         root_path=args.root_path,
    #         win_size=args.seq_len,
    #         flag=flag,
    #     )
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last)
    #     return data_set, data_loader
    # elif args.task_name == 'classification':
    #     drop_last = False
    #     data_set = Data(
    #         args = args,
    #         root_path=args.root_path,
    #         flag=flag,
    #     )

    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last,
    #         collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    #     )
    #     return data_set, data_loader
    if args.task_name == 'regression' or args.task_name == 'stellar_parameter_estimation':
        # 恒星光谱数据回归任务的特殊处理
        data_set = Data( # 创建数据集对象
            args = args,
            flag=flag,
            feature_scaler=feature_scaler,
            label_scaler=label_scaler
        )
        print(flag, len(data_set))
        data_loader = DataLoader( # 创建数据加载器对象
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif  args.task_name == 'spectral_prediction':
        data_set = Data(
            args=args,
            flag=flag,
            label_scaler=label_scaler,
            feature_scaler=feature_scaler,
            has_labels=has_labels
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            worker_init_fn=fix_seed_worker
        )
        return data_set, data_loader
    # else:
    #     if args.data == 'm4':
    #         drop_last = False
    #     data_set = Data(
    #         args = args,
    #         root_path=args.root_path,
    #         data_path=args.data_path,
    #         flag=flag,
    #         size=[args.seq_len, args.label_len, args.pred_len],
    #         features=args.features,
    #         target=args.target,
    #         timeenc=timeenc,
    #         freq=freq,
    #         seasonal_patterns=args.seasonal_patterns
    #     )
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last)
    #     return data_set, data_loader
