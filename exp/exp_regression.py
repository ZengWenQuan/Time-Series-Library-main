from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.stellar_metrics import calculate_metrics, format_metrics, calculate_feh_classification_metrics, format_feh_classification_metrics
from utils.stellar_metrics import save_regression_metrics, save_feh_classification_metrics
from utils.losses import RegressionFocalLoss

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils.stellar_metrics import Scaler

warnings.filterwarnings('ignore')


class Exp_Regression(Exp_Basic):
    """
    恒星参数估计（Stellar Parameter Estimation）实验类
    """
    def __init__(self, args):
        super(Exp_Regression, self).__init__(args)
        # 恒星参数名称
        self.targets = args.targets
        self.args=args
        self.label_scaler=self.get_label_scaler()
        info_model=self.args.run_dir+'/model.txt'
        with open(info_model,'w') as f:
            # 写入模型结构
            f.write("模型结构:\n")
            f.write(f"{self.model}\n\n")
            
            # 写入每层参数数量
            f.write("每层参数数量:\n")
            for name, param in self.model.named_parameters():
                f.write(f"  {name}: {param.numel():,} 参数\n")
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        选择损失函数
        支持的损失函数:
        - MSE: 均方误差损失，适用于一般回归问题
        - MAE: 平均绝对误差损失，对异常值更鲁棒
        - SmoothL1: 平滑L1损失，结合了MSE和MAE的优点
        - Huber: Huber损失，对异常值更鲁棒
        - LogCosh: Log-Cosh损失，类似于Huber损失但更平滑
        - FocalMSE: 基于MSE的Focal Loss，更关注难样本
        - FocalMAE: 基于MAE的Focal Loss，更关注难样本
        - FocalSmoothL1: 基于SmoothL1的Focal Loss，更关注难样本
        """
        if not hasattr(self.args, 'loss') or self.args.loss == 'MSE':
            return nn.MSELoss()
        elif self.args.loss == 'MAE':
            return nn.L1Loss()
        elif self.args.loss == 'SmoothL1':
            return nn.SmoothL1Loss()
        elif self.args.loss == 'Huber':
            return nn.HuberLoss(delta=1.0)
        elif self.args.loss == 'LogCosh':
            # 自定义Log-Cosh损失函数
            def logcosh_loss(pred, target):
                return torch.mean(torch.log(torch.cosh(pred - target)))
            return logcosh_loss
        elif self.args.loss == 'FocalMSE':
            # 获取Focal Loss参数
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='mse')
        elif self.args.loss == 'FocalMAE':
            # 获取Focal Loss参数
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='mae')
        elif self.args.loss == 'FocalSmoothL1':
            # 获取Focal Loss参数
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='smooth_l1')
        else:
            print(f"警告: 未知的损失函数 '{self.args.loss}'，使用默认的MSE损失")
            return nn.MSELoss()
    def get_label_scaler(self):
        
        df_raw = pd.read_csv(os.path.join(self.args.root_path,
                                          self.args.data_path))
        
        
        # 确定训练、验证、测试集的划分比例
        total_samples = len(df_raw)
        train_ratio, val_ratio, test_ratio = self.args.split_ratio
        train_boundary = int(total_samples * train_ratio) # 训练集边界
        val_boundary = int(total_samples * (train_ratio + val_ratio)) # 验证集边界
        
        # 根据实际需要预测的标签选择标签列
        targets = df_raw[self.targets].values
        if self.args.label_scaler_type:
            label_scaler = Scaler(scaler_type=self.args.label_scaler_type)
            # 仅在训练数据上拟合
            label_scaler.fit(targets[:train_boundary]) # 拟合训练集数据
            return label_scaler
        else:
            return None
        
    def vali(self, vali_data, vali_loader, criterion):
        
        if vali_data is None:
            return None, None, None
        total_loss = []
        self.model.eval()
        
        # 用于收集所有预测和真实值
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 恒星参数估计前向传播
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                # 计算损失
                pred = outputs.detach()
                true = batch_y.detach()
                loss = criterion(pred, true)
                total_loss.append(loss.item())
                
                # 收集预测和真实值
                all_preds.append(pred.cpu().numpy())
                all_trues.append(true.cpu().numpy())
                
        # 将所有批次的预测和真实值合并
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        
        # 如果需要应用反归一化
        if hasattr(self.args, 'apply_inverse_transform') and self.args.apply_inverse_transform:
            # 使用数据集对象的反归一化方法
            # 直接对整个预测和真实值数组进行反归一化
            all_preds = self.label_scaler.inverse_transform(all_preds)
            all_trues = self.label_scaler.inverse_transform(all_trues)  # 直接使用保存的原始标签
        
        # 计算指标
        # metrics_dict = calculate_metrics(all_preds, all_trues, self.targets)
        # feh_metrics = calculate_feh_classification_metrics(all_preds, all_trues)
        
        # 验证后将模型设回训练模式
        #self.model.train()
        
        # 返回平均损失和指标字典
        #return np.average(total_loss), metrics_dict, feh_metrics
        return np.average(total_loss) , all_preds , all_trues
    def train(self,setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        #path = os.path.join(self.args.checkpoints, setting)
        chechpoint_path=self.args.run_dir+'/'+'checkpoints'
        if not os.path.exists(chechpoint_path):
            os.makedirs(chechpoint_path)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        # 用于保存最佳模型的指标
        #best_metrics_dict = None
        #best_feh_metrics = None
        
        # 记录上一次验证的epoch
        #last_vali_epoch = 0
        train_loss_list=[]
        vali_loss_list=[]
        test_loss_list=[]
        lr_list=[]
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 使用恒星参数估计前向传播
                outputs = self.model(batch_x) #

                # 计算损失
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                loss.backward() #
                model_optim.step() # 更新模型参数

            #print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            # 只在指定间隔执行验证
            do_validation = (epoch + 1) % self.args.vali_interval == 0
            # 最后一个epoch也执行验证
            is_last_epoch = epoch == self.args.train_epochs - 1
                # 执行验证
            vali_loss, vali_preds, vali_trues = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_preds, test_trues = self.vali(test_data, test_loader, criterion)
            
            left_time = (self.args.train_epochs - epoch) *(time.time() - epoch_time)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss),end=" ")
            if test_data is not None:
                print("Test Loss: {0:.7f}".format(test_loss),end=" ")
                
            print("Vali Interval: {0},left_time: {1:.2f}h{2:.2f}m{3:.2f}s".format(
                self.args.vali_interval, left_time//3600, (left_time%3600)//60, left_time%60))
            
            # 计算指标
            metrics_dict_vali = calculate_metrics(vali_preds, vali_trues, self.args.targets)
            feh_metrics_vali = calculate_feh_classification_metrics(vali_preds, vali_trues)
            if test_data is not None:
                metrics_dict_test = calculate_metrics(test_preds, test_trues, self.args.targets)
                feh_metrics_test = calculate_feh_classification_metrics(test_preds, test_trues)
                
                
            # 早停并保存最佳模型
            prev_best_loss = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, chechpoint_path) #如果早停，保存最后一轮模型
            if vali_loss < prev_best_loss:
                # 保存最佳模型的指标
                save_dir = self.args.run_dir+"/metrics/best/"
                save_regression_metrics(metrics_dict_vali, save_dir, self.targets, phase="vali")
                save_feh_classification_metrics(feh_metrics_vali, save_dir, phase="vali")
                if test_data is not None:
                    save_regression_metrics(metrics_dict_test, save_dir, self.targets, phase="test")
                    save_feh_classification_metrics(feh_metrics_test, save_dir, phase="test")
            # 保存最新模型的指标
            save_dir = self.args.run_dir+"/metrics/last/"
            save_regression_metrics(metrics_dict_vali, save_dir, self.targets, phase="vali")
            save_feh_classification_metrics(feh_metrics_vali, save_dir, phase="vali")
            if test_data is not None:
                save_regression_metrics(metrics_dict_test, save_dir, self.targets, phase="test")
                save_feh_classification_metrics(feh_metrics_test, save_dir, phase="test")
            if do_validation or is_last_epoch :
                #or early_stopping.val_loss_min < prev_best_loss or early_stopping.early_stop:
                # 输出详细验证指标
                print("验证集指标:")
                print(format_metrics(metrics_dict_vali))
                print("\n验证集FeH分类指标:")
                print(format_feh_classification_metrics(feh_metrics_vali))
                if test_data is not None:
                    print("\n测试集指标:")
                    print(format_metrics(metrics_dict_test))
                    print("\n测试集FeH分类指标:")
                    print(format_feh_classification_metrics(feh_metrics_test))
                
                    
                
                # 更新上一次验证的epoch
                #last_vali_epoch = epoch
            # else:
            #     # 不执行验证，只输出训练损失
            #     print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
            #         epoch + 1, train_steps, train_loss))
            #     print(f"跳过验证 (下一次验证将在 epoch {((epoch + 1) // self.args.vali_interval + 1) * self.args.vali_interval})")
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            train_loss_list.append(train_loss)
            vali_loss_list.append(vali_loss)
            test_loss_list.append(test_loss)
            lr_list.append(self.args.learning_rate)
            
            if self.args.plot_loss and False:
                save_dir=self.args.run_dir+'/plot'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.rcParams['font.family'] = ['Times New Roman']
                skip_epoch=0
                plt.figure(figsize=(10,5))
                plt.plot(range(epoch+1)[skip_epoch:],train_loss_list[skip_epoch:],label='train_loss',color='red')
                plt.plot(range(epoch+1)[skip_epoch:],vali_loss_list[skip_epoch:],label='vali_loss',color='blue')
                plt.plot(range(epoch+1)[skip_epoch:],test_loss_list[skip_epoch:],label='test_loss',color='green')
                plt.xlabel('Epoch',fontsize=14)
                plt.ylabel('Loss',fontsize=14)
                plt.title('Loss Curve',fontsize=14)
                plt.legend()
                plt.savefig(save_dir+'/loss.png')
                plt.close()
                
                
                plt.figure(figsize=(10,5))
                plt.plot(range(epoch+1)[skip_epoch:],lr_list[skip_epoch:],label='lr',color='black')
                plt.xlabel('Epoch',fontsize=14)
                plt.ylabel('Learning Rate',fontsize=14)
                plt.title('Learning Rate Curve',fontsize=14)
                plt.legend()
                plt.savefig(save_dir+'/lr.png')
                plt.close()
            
        # 保存最后一轮的指标
        if hasattr(self.args, 'metrics_dir') and self.args.metrics_dir:
            
            save_dir = self.args.run_dir+'/metrics/last/'
            save_regression_metrics(metrics_dict_vali, save_dir, self.targets, phase="vali")
            save_feh_classification_metrics(feh_metrics_vali, save_dir, phase="vali")
            if test_data is not None:
                save_regression_metrics(metrics_dict_test, save_dir, self.targets, phase="test")
                save_feh_classification_metrics(feh_metrics_test, save_dir, phase="test")

        last_model_path = chechpoint_path + '/' + 'last.pth'
        torch.save(self.model.state_dict(), last_model_path)
        #self.model.load_state_dict(torch.load(last_model_path))
        print("the train is over,save the best model and the last model")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 使用恒星参数估计前向传播
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                # 收集预测结果
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('测试集形状:', preds.shape, trues.shape)
        
        # 计算指标
        raw_targets = test_data.get_raw_targets()
        
        # 对预测值进行反归一化，使用统一的标签缩放器
        preds_inverse = self.label_scaler.inverse_transform(preds)
        
        # 计算指标时使用反归一化后的数据
        metrics_dict = calculate_metrics(preds_inverse, raw_targets, self.targets)
        
        # 计算FeH分类指标
        feh_index = self.targets.index('FeH') if 'FeH' in self.targets else 2
        feh_pred = preds_inverse[:, feh_index:feh_index+1]
        feh_true = raw_targets[:, feh_index:feh_index+1]
        feh_metrics = calculate_feh_classification_metrics(feh_pred, feh_true, feh_index=0)
        
        # 保存反归一化后的数据以便后续分析
        preds_for_save = preds_inverse
        trues_for_save = raw_targets
        
        print("测试集回归指标:")
        print(format_metrics(metrics_dict))
        
        print("测试集FeH分类指标:")
        print(format_feh_classification_metrics(feh_metrics))
        
        # 保存指标到文件（如果指定了保存目录）
        if hasattr(self.args, 'run_dir') and self.args.run_dir:
            save_dir = self.args.run_dir+'/test_results/'
            
            # 保存测试指标
            save_regression_metrics(metrics_dict, save_dir, self.targets, phase="test")
            save_feh_classification_metrics(feh_metrics, save_dir, phase="test")
        
        # 保存每个标签的预测结果
        #folder_path = './test_results/' + setting + '/'
        folder_path=self.args.run_dir+'/test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存预测结果到本地
        np.save(folder_path + 'pred.npy', preds_for_save)
        np.save(folder_path + 'true.npy', trues_for_save)
        print(f"save test result to {folder_path}")
        return metrics_dict['mae'], metrics_dict['mse'], metrics_dict['rmse'] 