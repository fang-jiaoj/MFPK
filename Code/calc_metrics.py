from numba.np.arrayobj import default_lt
from sklearn.metrics import r2_score,mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os
from collections import defaultdict

class Metrics(object):
    def __init__(self,args):
        self.task_num = args.task_num
        self.task_names = args.task_names
        self.y_pred = []
        self.y_true = []

    def update(self,y_pred,y_true):
        self.y_pred.extend(y_pred.detach().cpu().numpy())
        self.y_true.extend(y_true.detach().cpu().numpy())
        self._multi_datasets()

    def update_test(self,y_pred):
        self.y_pred.extend(y_pred.detach().cpu().numpy())
        self._multi_datasets_test()

    def _multi_datasets(self):
        """内部方法：将数据按任务拆分"""
        self.target_total = []
        self.pred_total = []

        for i in range(self.task_num):
            pred_task_i = []
            target_task_i = []
            for j in range(len(self.y_true)):
                if not np.isnan(self.y_true[j][i]):
                    target_task_i.append(self.y_true[j][i])
                    pred_task_i.append(self.y_pred[j][i])
            self.target_total.append(target_task_i)
            self.pred_total.append(pred_task_i)

    def _multi_datasets_test(self):
        """将数据按照任务拆分"""
        self.pred_total = []

        for i in range(self.task_num):
            pred_task_i = []
            for j in range(len(self.y_pred)):
                if not np.isnan(self.y_pred[j][i]):
                    pred_task_i.append(self.y_pred[j][i])
            self.pred_total.append(pred_task_i)

    def multi_datasets(self):
        """兼容旧代码的接口"""
        return self.pred_total

    def fold_error(self):
        dict = defaultdict(list)

        for i in range(self.task_num):
            if len(self.target_total[i]) == 0 or len(self.pred_total[i]) == 0:
                print(f"[Skip] {self.task_names[i]} has no data.")
                dict[self.task_names[i]].extend([np.nan, np.nan, np.nan])
                continue

            if 'fup' in self.task_names[i]:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                print(self.task_names[i],len(self.target_total[i]))
                lst = [abs(10**a/10**b) for a,b in zip(self.pred_total[i],self.target_total[i])]
                
                if len(lst) == 0:
                    print(f"Warning: lst is empty for task {self.task_names[i]}")
                    dict[self.task_names[i]].extend([0, 0, 0])
                else:
                    two_newlist = [x for x in lst if 1/2 <= x <= 2]
                    three_newlist = [x for x in lst if 1/3 <= x <= 3]
                    five_newlist = [x for x in lst if 1/5 <= x <= 5]

                    two_fold_error = len(two_newlist) / len(lst) * 100
                    three_fold_error = len(three_newlist) /len(lst) * 100
                    five_fold_error = len(five_newlist) / len(lst) * 100
                    dict[self.task_names[i]].extend([two_fold_error,three_fold_error,five_fold_error])

            else:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                print(self.task_names[i], len(self.pred_total[i]))

                lst = [abs(10**a/10**b)for a,b in zip(self.pred_total[i],self.target_total[i])]
                if len(lst) == 0:
                    print(f"Warning: lst is empty for task {self.task_names[i]}")
                    dict[self.task_names[i]].extend([0, 0, 0])
                else:
                    two_newlist = [x for x in lst if 1 / 2 <= x <= 2]
                    three_newlist = [x for x in lst if 1 / 3 <= x <= 3]
                    five_newlist = [x for x in lst if 1 / 5 <= x <= 5]

                    two_fold_error = len(two_newlist) / len(lst) * 100
                    three_fold_error = len(three_newlist) / len(lst) * 100
                    five_fold_error = len(five_newlist) / len(lst) * 100
                    dict[self.task_names[i]].extend([two_fold_error,three_fold_error,five_fold_error])
        return dict

    def calculate_gmfe(self):
        dict = defaultdict(list)

        for i in range(self.task_num):
            if len(self.target_total[i]) == 0 or len(self.pred_total[i]) == 0:
                print(f"[Skip] {self.task_names[i]} has no data.")
                dict[self.task_names[i]].append(np.nan)
                continue

            if 'fup' in self.task_names[i]:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                lst = [abs(np.log10(10**a/10**b)) for a,b in zip(self.pred_total[i],self.target_total[i])]
                mean_abs = np.mean(lst)
                gmfe = 10 ** mean_abs
                dict[self.task_names[i]].append(gmfe)
            else:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                lst = [abs(np.log10(10**a / 10**b)) for a, b in zip(self.pred_total[i], self.target_total[i])]
                mean_abs = np.mean(lst)
                gmfe = 10 ** mean_abs
                dict[self.task_names[i]].append(gmfe)
        return  dict

    def calculate_afe(self):
        dict = defaultdict(list)

        for i in range(self.task_num):
            if len(self.target_total[i]) == 0 or len(self.pred_total[i]) == 0:
                print(f"[Skip] {self.task_names[i]} has no data.")
                dict[self.task_names[i]].append(np.nan)
                continue

            if 'fup' in self.task_names[i]:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                lst = [np.log10(10**a/10**b) for a,b in zip(self.pred_total[i],self.target_total[i])]
                mean = np.mean(lst)
                afe = 10 ** mean
                dict[self.task_names[i]].append(afe)
            else:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                lst = [np.log10(10**a / 10**b) for a, b in zip(self.pred_total[i], self.target_total[i])]
                mean = np.mean(lst)
                afe = 10 ** mean
                dict[self.task_names[i]].append(afe)
        return  dict

    def median_fold_error(self):
        dict = defaultdict(list)

        for i in range(self.task_num):
            if len(self.target_total[i]) == 0 or len(self.pred_total[i]) == 0:
                print(f"[Skip] {self.task_names[i]} has no data.")
                dict[self.task_names[i]].append(np.nan)
                continue

            if 'fup' in self.task_names[i]:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                lst = [abs(np.log10(10**a/10**b)) for a,b in zip(self.pred_total[i],self.target_total[i])]
                median_abs = np.median(lst)
                dict[self.task_names[i]].append(np.e ** median_abs)
            else:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                lst = [abs(np.log10(10**a / 10**b)) for a, b in zip(self.pred_total[i], self.target_total[i])]
                median_abs = np.median(lst)
                dict[self.task_names[i]].append(np.e ** median_abs)
        return dict

    def calculate_bias(self):
        dict = defaultdict(list)

        for i in range(self.task_num):
            if len(self.target_total[i]) == 0 or len(self.pred_total[i]) == 0:
                print(f"[Skip] {self.task_names[i]} has no data.")
                dict[self.task_names[i]].append(np.nan)
                continue

            if 'fup' in self.task_names[i]:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                lst = [(10**a - 10**b) for a,b in zip(self.pred_total[i], self.target_total[i])]
                bias = np.median(lst)
                dict[self.task_names[i]].append(bias)
            else:
                assert len(self.pred_total[i]) == len(
                    self.target_total[i]), "Prediction and target lengths do not match!"

                lst = [(10**a - 10**b) for a, b in zip(self.pred_total[i], self.target_total[i])]
                bias = np.median(lst)
                dict[self.task_names[i]].append(bias)
        return dict

    def calculate_rmse_r2(self):
        dict = defaultdict(list)
             
        # for i in range(self.task_num):
        #     if 'fup' in self.task_names[i]:
        #         assert len(self.pred_total[i]) == len(self.target_total[i]), "Prediction and target lengths do not match!"
        #
        #         rmse = np.sqrt(mean_squared_error(self.target_total[i],self.pred_total[i]))
        #         r2 = r2_score(self.target_total[i],self.pred_total[i])
        #         dict[self.task_names[i]].extend([rmse,r2])
        #
        #     else:
        #         assert len(self.pred_total[i]) == len(self.target_total[i]), "Prediction and target lengths do not match!"
        #
        #         pred_total = [10**j for j in self.pred_total[i]]
        #         target_total = [10**j for j in self.target_total[i]]
        #         rmse = np.sqrt(mean_squared_error(target_total,pred_total))
        #         r2 = r2_score(target_total,pred_total)
        #         dict[self.task_names[i]].extend([rmse, r2])
        for i in range(self.task_num):
            if len(self.target_total[i]) == 0 or len(self.pred_total[i]) == 0:
                print(f"[Skip] {self.task_names[i]} has no data.")
                dict[self.task_names[i]].extend([np.nan,np.nan])
                continue

            assert len(self.pred_total[i]) == len(self.target_total[i]), "Prediction and target lengths do not match!"
            assert len(self.pred_total[i]) == len(self.target_total[i]), "Prediction and target lengths do not match!"
            rmse = np.sqrt(mean_squared_error(self.target_total[i],self.pred_total[i]))
            r2 = r2_score(self.target_total[i],self.pred_total[i])
            dict[self.task_names[i]].extend([rmse,r2])
                
        return dict

    def calculate_pearson_r(self):
        dict = defaultdict(list)

        for i in range(self.task_num):
            if len(self.target_total[i]) == 0 or len(self.pred_total[i]) == 0:
                print(f"[Skip] {self.task_names[i]} has no data.")
                dict[self.task_names[i]].append(np.nan)
                continue

            assert len(self.pred_total[i]) == len(self.target_total[i]), "Prediction and target lengths do not match!"
            try:
                r = pearsonr(self.target_total[i],self.pred_total[i])[0]
                dict[self.task_names[i]].append(r)
            except Exception as e:
                # 捕获其他可能的错误（如全零值）
                print(f"[Error] {self.task_names[i]}: Pearson calculation failed: {str(e)}")
                dict[self.task_names[i]].append(np.nan)

        return dict










