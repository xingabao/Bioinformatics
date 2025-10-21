---
title: 融合机器学习算法 - 用 VotingClassifier 实现分类多模型的投票集成
date: 2025-10-19 09:06:53
tags: []
categories: [[]] 
---



```Python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 09:08:39 2025
 @author Abao Xing
 @email  albertxn7@gmail.com
 This scripts writen by Abao Xing

   ┏┓　　┏┓
  ┏┛┻━━━━┛┻┓
  ┃　　　　  ┃
  ┃　━　　━　 ┃
  ┃　┳┛　┗┳　 ┃
  ┃　　　　　 ┃
  ┃　　　┻　　┃
  ┃　　　　　 ┃
  ┗━━┓　　　┏━┛
  　　┃　　 ┃ 神兽保佑
  　　┃　　 ┃ 代码无BUG！！！
  　　┃　　 ┗━━━━━┓
  　　┃　　　　　　  ┣┓
 　　┃　　　　　　  ┏┛┃
 　　┗┓┓┏━━━━━┳┓┏━━┛
  　　┃┫┫　   ┃┫┫
  　　┗┻┛　   ┗┻┛

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message = ".*does not have valid feature names.*")
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
             
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

import winreg
def get_desktop():
    ''' 获得桌面路径 '''
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                         r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
    return winreg.QueryValueEx(key,"Desktop")[0]
windir = get_desktop()
wkdir = 'Z:/TData/big-data/SADAGY'

if __name__ == '__main__':
    
    # 读取数据
    df = pd.read_excel(f'{wkdir}/2501019_voting_classifier_ensemble_learning.xlsx')
    
    # 划分特征和目标变量
    X = df.drop(['class'], axis = 1)
    y = df['class']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = df['class'])
    df.head()
    
    # 硬投票
    if True:
        
        # 定义各个模型
        rf_clf = RandomForestClassifier(random_state = 42)
        xgb_clf = XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', random_state = 42)
        lgbm_clf = LGBMClassifier(random_state = 42, verbose = -1)
        gbm_clf = GradientBoostingClassifier(random_state = 42)
        adaboost_clf = AdaBoostClassifier(random_state = 42, algorithm = 'SAMME')
        catboost_clf = CatBoostClassifier(verbose = 0, random_state = 42)
        
        # 创建硬投票分类器
        voting_hard = VotingClassifier(
            estimators = [
                ('RandomForest', rf_clf),
                ('XGBoost', xgb_clf),
                ('LightGBM', lgbm_clf),
                ('GradientBoosting', gbm_clf),
                ('AdaBoost', adaboost_clf),
                ('CatBoost', catboost_clf)
            ],
            voting = 'hard'
        )
        
        # 训练硬投票分类器
        voting_hard.fit(X_train, y_train)
    
    # 软投票
    if True:
        
        # 创建软投票分类器
        voting_soft = VotingClassifier(
            estimators = [
                ('RandomForest', rf_clf),
                ('XGBoost', xgb_clf),
                ('LightGBM', lgbm_clf),
                ('GradientBoosting', gbm_clf),
                ('AdaBoost', adaboost_clf),
                ('CatBoost', catboost_clf)
            ],
            voting = 'soft',
            weights = [1, 1, 1, 1, 1, 1]
        )
        # 训练软投票分类器
        voting_soft.fit(X_train, y_train)
        
    if True:

        # 硬投票预测测试集
        y_pred_hard = voting_hard.predict(X_test)
        
        # 输出硬投票模型的评价指标
        print("Classification Report for Hard Voting:")
        print(classification_report(y_test, y_pred_hard))
        
        # 软投票预测测试集
        y_pred_soft = voting_soft.predict(X_test)
        
        # 输出软投票模型的评价指标
        print("Classification Report for Soft Voting:")
        print(classification_report(y_test, y_pred_soft))
        
    if True:
        
        # 硬投票的混淆矩阵
        conf_matrix_hard = confusion_matrix(y_test, y_pred_hard)
        
        # 软投票的混淆矩阵
        conf_matrix_soft = confusion_matrix(y_test, y_pred_soft)
        fig, axes = plt.subplots(1, 2, figsize = (16, 6), dpi=1200)
        
        # 绘制硬投票混淆矩阵热力图
        sns.heatmap(conf_matrix_hard, annot = True, annot_kws = {'size': 15}, fmt = 'd', cmap = 'YlGnBu', cbar_kws = {'shrink': 0.75}, ax = axes[0])
        axes[0].set_title('Confusion Matrix (Hard Voting)', fontsize = 15)
        axes[0].set_xlabel('Predicted Label', fontsize = 15)
        axes[0].set_ylabel('True Label', fontsize = 15)
       
        # 绘制软投票混淆矩阵热力图
        sns.heatmap(conf_matrix_soft, annot = True, annot_kws = {'size': 15}, fmt = 'd', cmap = 'YlGnBu', cbar_kws = {'shrink': 0.75}, ax = axes[1])
        axes[1].set_title('Confusion Matrix (Soft Voting)', fontsize = 15)
        axes[1].set_xlabel('Predicted Label', fontsize = 15)
        axes[1].set_ylabel('True Label', fontsize = 15)
        plt.tight_layout()
        plt.savefig(f"{windir}/混淆矩阵_硬投票_软投票.png", bbox_inches = 'tight')
        plt.close()
        
    if True:
        
        # 初始化字典存储每个模型的预测结果和ROC信息
        models = {
            'RandomForest': rf_clf,
            'XGBoost': xgb_clf,
            'LightGBM': lgbm_clf,
            'GradientBoosting': gbm_clf,
            'AdaBoost': adaboost_clf,
            'CatBoost': catboost_clf
        }
        
        # 绘制ROC曲线
        plt.figure(figsize = (10, 8))
        for name, model in models.items():
            # 获取预测概率
            y_proba = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            # 计算ROC曲线和AUC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            # 绘制ROC曲线
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
            
        # 添加对硬投票分类器的ROC曲线
        voting_hard.fit(X_train, y_train)
        y_pred_hard = voting_hard.predict(X_test)
        
        # 使用投票分类器计算硬投票下的AUC和假阳率、真阳率
        y_proba_hard = voting_hard.transform(X_test)[:, 1]
        fpr_hard, tpr_hard, _ = roc_curve(y_test, y_proba_hard)
        auc_score_hard = roc_auc_score(y_test, y_proba_hard)
        
        plt.plot(fpr_hard, tpr_hard, label = f"Voting (AUC = {auc_score_hard:.2f})", linestyle = '--')
        plt.plot([0, 1], [0, 1], 'k--', label = "Random Guessing")
        plt.xlabel('False Positive Rate (FPR)', fontsize = 18)
        plt.ylabel('True Positive Rate (TPR)', fontsize = 18)
        plt.title('ROC Curve of Base Models and Voting Classifier', fontsize = 18)
        plt.legend(loc = 'lower right')
        plt.grid()
        plt.savefig(f"{windir}/ROC Curve of Base Models and Voting Classifier.png", bbox_inches = 'tight', dpi = 1200)
        plt.close()
                
    if True:
        
        # 获取软投票分类器的预测概率
        # 选择正类的概率
        y_proba_soft = voting_soft.predict_proba(X_test)[:, 1] 
        
        # 计算软投票分类器的ROC曲线和AUC值
        fpr_soft, tpr_soft, _ = roc_curve(y_test, y_proba_soft)
        auc_score_soft = roc_auc_score(y_test, y_proba_soft)
        
        # 绘制ROC曲线
        plt.figure(figsize = (8, 6))
        plt.plot(fpr_soft, tpr_soft, label = f"Soft Voting (AUC = {auc_score_soft:.2f})")
        
        # 添加随机猜测的基线
        plt.plot([0, 1], [0, 1], 'k--', label = "Random Guessing")
        
        # 图形修饰
        plt.xlabel('False Positive Rate (FPR)', fontsize = 18)
        plt.ylabel('True Positive Rate (TPR)', fontsize = 18)
        plt.title('ROC Curve of Soft Voting Classifier', fontsize = 18)
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(f"{windir}/ROC Curve of Soft Voting Classifier.png", bbox_inches = 'tight', dpi = 1200)
        plt.show()
```



