---
title: Transformer+KAN+SHAP
date: 2025-05-07 08:27:30
tags: [Python, 机器学习, SHAP]
categories: [[案例分享, 机器学习, Transformer]]
---


# 环境设置

``` r
# 指定 Python 环境
reticulate::use_python("C:/ProgramData/Anaconda3/python.exe")

# 切换工作目录
wkdir <- dirname(rstudioapi::getActiveDocumentContext()$path)
```

# 导入所需库

``` python
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings("ignore")

# 加载自定义模块
import sys
sys.path.append(f'{r.wkdir}/modules')
from fftKAN import *
from effKAN import *
```

# 数据预处理函数

``` python
# lookback 表示观察的跨度
def split_data(feature, target, lookback):
    data_raw = feature
    target_raw = target
    data = []
    target = []
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
        target.append(target_raw[index: index + lookback])
    data = np.array(data)
    target = np.array(target)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    x_train = data[:train_set_size, :-1, :]
    y_train = target[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1]
    y_test = target[train_set_size:, -1, :]
    
    return [x_train, y_train, x_test, y_test]
```

<strong style="color:#00A087;font-size:16px;">`split_data()`函数用于将时间序列数据划分为训练集和测试集</strong>：

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>feature</strong>，输入特征数据
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>target</strong>，目标变量
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>lookback</strong>，时间窗口的大小，表示用前多少个时间点的数据来预测当前的点
</li>
</ul>

# 模型定义: Transformer + KAN

``` python
class TimeSeriesTransformer_ekan(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, hidden_space, dropout_rate = 0.1):
        super(TimeSeriesTransformer_ekan, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space = hidden_space
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,
            nhead=num_heads,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers = num_layers)
        self.e_kan = KAN([hidden_space, 10, num_outputs])
        self.transform_layer = nn.Linear(input_dim, hidden_space)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]
        x = self.e_kan(x)
        return x
```

<p>
这是 Transformer 结合 KNN (Kolmogorov-Arnold Network)
的模型，在输出层使用了 KAN (非线性网路结构)
替代了普通的线性层`nn.Linear`。
</p>
<p>
`self.e_kan = KAN([hidden_space, 10, num_outputs])`，定义了一个 KAN
网络，输入维度为 hidden_space，中间层有 10 个节点，输出维度为
num_outputs 。
</p>

# SHAP 解释函数

``` python
def explain_model_with_shap(model, data, background_samples = 50, seq_len = None, input_dim = None):
    model.eval()
    # 展平数据：从 (n_samples, seq_len, input_dim) 到 (n_samples, seq_len * input_dim)
    data_flattened = data.reshape(data.shape[0], -1)
    background_data = data_flattened[:background_samples]

    def model_wrapper(x):
        with torch.no_grad():
            x_reshaped = torch.FloatTensor(x).reshape(-1, seq_len, input_dim)
            return model(x_reshaped).numpy().flatten()

    explainer = shap.KernelExplainer(model_wrapper, background_data)
    shap_values = explainer.shap_values(data_flattened)
    return shap_values, explainer, data_flattened
```

<strong style="color:#00A087;font-size:16px;">定义一个函数，使用SHAP（SHapley
Additive exPlanations）方法解释模型的预测结果</strong>：

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>model.eval()</strong>，将模型设置为评估模式，不进行梯度更新
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>data_flattened</strong>，将输入数据展平为二维数组（从 \[样本数,
时间步, 特征数\] 到 \[样本数, 时间步\*特征数\]）
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>background_data</strong>，选取部分数据作为背景数据，用于 SHAP
计算
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>model_wrapper</strong>，定义一个包装函数，将展平的数据重新塑形为模型需要的形状，并返回预测结果
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>shap.KernelExplainer</strong>，使用SHAP的内核方法计算每个特征对预测的贡献，即
SHAP 值
</li>
<li style="margin-top:2px;margin-bottom:2px;">
输出
shap_values（每个特征的贡献值）、explainer（解释器对象）、data_flattened（展平后的数据）
</li>
</ul>

# 参数设置

``` python
parser = argparse.ArgumentParser()
args = parser.parse_args(args = [])
args.input_features = ['Open', 'High', 'Low', 'Volume', 'Close']
args.num_heads = 4
args.n_layers = 2
args.output_features = ['Close']
args.hidden_space = 32
args.dropout = 0.2
args.num_epochs = 300
args.vision = True
args.window_size = 20
args.model_name = 'Transformer-ekan'
args.path = f'{r.wkdir}/data/rlData.csv'
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

<strong style="color:#00A087;font-size:16px;">设置模型和训练参数</strong>：

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>input_features</strong>，输入特征列表，这里是股票的开盘价、最高价、最低价、成交量、收盘价
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>num_heads</strong>，目标变量
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>n_layers</strong>，Transformer 编码器层数
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>output_features</strong>，输出目标
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>hidden_space</strong>，隐藏层维度
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>dropout</strong>，Dropout 比例
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>vision</strong>，是否可视化数据
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>window_size</strong>，时间窗口大小
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>model_name</strong>，模型名称
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>path</strong>，数据文件路径
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>device</strong>，设备类型（优先使用GPU，若无则使用CPU）
</li>
</ul>

# 数据加载

``` python
data = pd.read_csv(args.path)
data = data.sort_values('Date')
    
data
##            Date        Open        High  ...       Close   Adj Close    Volume
## 0    2018-05-23  182.500000  186.910004  ...  186.899994  186.899994  16628100
## 1    2018-05-24  185.880005  186.800003  ...  185.929993  185.929993  12354700
## 2    2018-05-25  186.020004  186.330002  ...  184.919998  184.919998  10965100
## 3    2018-05-29  184.339996  186.809998  ...  185.740005  185.740005  16398900
## 4    2018-05-30  186.539993  188.000000  ...  187.669998  187.669998  13736900
## ..          ...         ...         ...  ...         ...         ...       ...
## 247  2019-05-17  184.839996  187.580002  ...  185.300003  185.300003  10485400
## 248  2019-05-20  181.880005  184.229996  ...  182.720001  182.720001  10352000
## 249  2019-05-21  184.570007  185.699997  ...  184.820007  184.820007   7502800
## 250  2019-05-22  184.729996  186.740005  ...  185.320007  185.320007   9203300
## 251  2019-05-23  182.419998  183.899994  ...  180.460007  180.460007  10396877
## 
## [252 rows x 7 columns]
```

# 可视化数据

``` python
if args.vision:
    sns.set_style("darkgrid")
    plt.figure(figsize = (11, 7))
    plt.plot(data[['Close']])
    plt.xticks(range(0, data.shape[0], 20), data['Date'].loc[::20], rotation = 45)
    plt.title("Stock Price", fontsize = 18, fontweight = 'bold')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price (USD)', fontsize = 18)
    plt.show()
```

<img src="Transformer+KAN+SHAP_files/figure-markdown_strict/unnamed-chunk-9-1.png" width="1056" />

# 数据标准化

``` python
features = data[args.input_features]
scaler = MinMaxScaler(feature_range = (-1, 1))
features_scaled = scaler.fit_transform(features)

target_scaler = MinMaxScaler(feature_range = (-1, 1))
target = data[args.output_features]
target_scaled = target_scaler.fit_transform(target)
```

<p>
对输入特征和目标变量进行标准化处理（范围为-1到1），以提高模型训练的稳定性。
</p>

# 划分数据集

``` python
x_train, y_train, x_test, y_test = split_data(features_scaled, target_scaled, args.window_size)
```

调用之前定义的 split_data 函数，将数据划分为训练集和测试集。

# 转换为张量

``` python
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)
```

<p>
将`NumPy`数组转换为`PyTorch`张量，便于模型训练。
</p>

# 模型初始化和训练

``` python
model = TimeSeriesTransformer_ekan(
    input_dim = len(args.input_features),
    num_heads = args.num_heads,
    num_layers = args.n_layers,
    num_outputs = len(args.output_features),
    hidden_space = args.hidden_space,
    dropout_rate = args.dropout
)
```

# 定义损失函数和优化器

``` python
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)
```

<p>
定义损失函数为均方误差`MSE`，优化器为`Adam`，学习率为`0.01`。
</p>

# 训练模型

``` python
MSE_hist = np.zeros(args.num_epochs)
R2_hist = np.zeros(args.num_epochs)

start_time = time.time()
result = []

for t in range(args.num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train)
    R2 = r2_score(y_train_pred.detach().numpy(), y_train.detach().numpy())
    print("Epoch ", t + 1, "MSE: ", loss.item(), 'R2', R2)
    MSE_hist[t] = loss.item()
    if R2 < 0:
        R2 = 0
    R2_hist[t] = R2
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
## Epoch  1 MSE:  0.2978668510913849 R2 -215.90505981445312
## Epoch  2 MSE:  0.20988529920578003 R2 -366.9470520019531
## Epoch  3 MSE:  0.1441890001296997 R2 -17.266427993774414
## Epoch  4 MSE:  0.08360082656145096 R2 -0.5605981349945068
## Epoch  5 MSE:  0.09039576351642609 R2 0.26518017053604126
## Epoch  6 MSE:  0.09294246137142181 R2 0.5167976021766663
## Epoch  7 MSE:  0.08413499593734741 R2 0.5497957468032837
## Epoch  8 MSE:  0.07082340121269226 R2 0.47064143419265747
## Epoch  9 MSE:  0.05963682755827904 R2 0.460324764251709
## Epoch  10 MSE:  0.05902103707194328 R2 0.46168339252471924
## Epoch  11 MSE:  0.060106586664915085 R2 0.4551217555999756
## Epoch  12 MSE:  0.0527922585606575 R2 0.515474796295166
## Epoch  13 MSE:  0.04780493676662445 R2 0.5844342708587646
## Epoch  14 MSE:  0.04749075695872307 R2 0.5788122415542603
## Epoch  15 MSE:  0.033786021173000336 R2 0.7561811208724976
## Epoch  16 MSE:  0.028839165344834328 R2 0.8007737994194031
## Epoch  17 MSE:  0.021978355944156647 R2 0.8664212226867676
## Epoch  18 MSE:  0.02348753623664379 R2 0.8911263346672058
## Epoch  19 MSE:  0.020583385601639748 R2 0.9123624563217163
## Epoch  20 MSE:  0.023393956944346428 R2 0.8851319551467896
## Epoch  21 MSE:  0.024205941706895828 R2 0.8577677011489868
## Epoch  22 MSE:  0.024759389460086823 R2 0.876966118812561
## Epoch  23 MSE:  0.027842970564961433 R2 0.8899805545806885
## Epoch  24 MSE:  0.022542832419276237 R2 0.9068772196769714
## Epoch  25 MSE:  0.017846733331680298 R2 0.9106296896934509
## Epoch  26 MSE:  0.01796991564333439 R2 0.8928827047348022
## Epoch  27 MSE:  0.01829621009528637 R2 0.8850983381271362
## Epoch  28 MSE:  0.0171201154589653 R2 0.8948999047279358
## Epoch  29 MSE:  0.017353717237710953 R2 0.9018740057945251
## Epoch  30 MSE:  0.016058819368481636 R2 0.9092409610748291
## Epoch  31 MSE:  0.016529982909560204 R2 0.9044861197471619
## Epoch  32 MSE:  0.015494206920266151 R2 0.9149088859558105
## Epoch  33 MSE:  0.015528894029557705 R2 0.918588399887085
## Epoch  34 MSE:  0.014900807291269302 R2 0.922192394733429
## Epoch  35 MSE:  0.013198306784033775 R2 0.9281350374221802
## Epoch  36 MSE:  0.014325499534606934 R2 0.9153225421905518
## Epoch  37 MSE:  0.014767245389521122 R2 0.9129051566123962
## Epoch  38 MSE:  0.014736580662429333 R2 0.9190813899040222
## Epoch  39 MSE:  0.013684464618563652 R2 0.9281110763549805
## Epoch  40 MSE:  0.013046672567725182 R2 0.9340126514434814
## Epoch  41 MSE:  0.013979458250105381 R2 0.9307637810707092
## Epoch  42 MSE:  0.01414431817829609 R2 0.9254735112190247
## Epoch  43 MSE:  0.014489187858998775 R2 0.9234504699707031
## Epoch  44 MSE:  0.013583404012024403 R2 0.9288392066955566
## Epoch  45 MSE:  0.014081467874348164 R2 0.9286097884178162
## Epoch  46 MSE:  0.01465035043656826 R2 0.9147261381149292
## Epoch  47 MSE:  0.01326502114534378 R2 0.9270995855331421
## Epoch  48 MSE:  0.010784022510051727 R2 0.9444246292114258
## Epoch  49 MSE:  0.01245618611574173 R2 0.9408636689186096
## Epoch  50 MSE:  0.011607872322201729 R2 0.9412036538124084
## Epoch  51 MSE:  0.011579558253288269 R2 0.9396814703941345
## Epoch  52 MSE:  0.011423959396779537 R2 0.9409884810447693
## Epoch  53 MSE:  0.012799317017197609 R2 0.9354100227355957
## Epoch  54 MSE:  0.011869007721543312 R2 0.9348561763763428
## Epoch  55 MSE:  0.01154865138232708 R2 0.9419567584991455
## Epoch  56 MSE:  0.01277134008705616 R2 0.9346367716789246
## Epoch  57 MSE:  0.011765565723180771 R2 0.9431921243667603
## Epoch  58 MSE:  0.01247556135058403 R2 0.9349058866500854
## Epoch  59 MSE:  0.01117069460451603 R2 0.9416759014129639
## Epoch  60 MSE:  0.011723140254616737 R2 0.9387801289558411
## Epoch  61 MSE:  0.011272897012531757 R2 0.9405821561813354
## Epoch  62 MSE:  0.011689144186675549 R2 0.9390334486961365
## Epoch  63 MSE:  0.011821219697594643 R2 0.9365214705467224
## Epoch  64 MSE:  0.011872473172843456 R2 0.9398102760314941
## Epoch  65 MSE:  0.011135043576359749 R2 0.9374039769172668
## Epoch  66 MSE:  0.0110648637637496 R2 0.9408621788024902
## Epoch  67 MSE:  0.011290518566966057 R2 0.9446777701377869
## Epoch  68 MSE:  0.010160441510379314 R2 0.9483175873756409
## Epoch  69 MSE:  0.012767720967531204 R2 0.9320811033248901
## Epoch  70 MSE:  0.010992834344506264 R2 0.9420775175094604
## Epoch  71 MSE:  0.011823303997516632 R2 0.9402464628219604
## Epoch  72 MSE:  0.010118992999196053 R2 0.9462583661079407
## Epoch  73 MSE:  0.011299378238618374 R2 0.9386039972305298
## Epoch  74 MSE:  0.011831192299723625 R2 0.9403679370880127
## Epoch  75 MSE:  0.010172625072300434 R2 0.9465368986129761
## Epoch  76 MSE:  0.009603764861822128 R2 0.9517666101455688
## Epoch  77 MSE:  0.009152399376034737 R2 0.9502028822898865
## Epoch  78 MSE:  0.011370530351996422 R2 0.9405347108840942
## Epoch  79 MSE:  0.012217461131513119 R2 0.9405039548873901
## Epoch  80 MSE:  0.011066476814448833 R2 0.9408552050590515
## Epoch  81 MSE:  0.011823568493127823 R2 0.9352561235427856
## Epoch  82 MSE:  0.011405867524445057 R2 0.9451730847358704
## Epoch  83 MSE:  0.010826836340129375 R2 0.944719135761261
## Epoch  84 MSE:  0.011737717315554619 R2 0.9344923496246338
## Epoch  85 MSE:  0.011200694367289543 R2 0.9416669607162476
## Epoch  86 MSE:  0.010950625874102116 R2 0.9458390474319458
## Epoch  87 MSE:  0.010463203303515911 R2 0.9464824199676514
## Epoch  88 MSE:  0.012583848088979721 R2 0.9311196804046631
## Epoch  89 MSE:  0.01049449760466814 R2 0.9472618699073792
## Epoch  90 MSE:  0.009781384840607643 R2 0.9530571103096008
## Epoch  91 MSE:  0.01101007778197527 R2 0.9394491314888
## Epoch  92 MSE:  0.010043242014944553 R2 0.947794497013092
## Epoch  93 MSE:  0.010211259126663208 R2 0.9491825103759766
## Epoch  94 MSE:  0.011086449958384037 R2 0.9403480291366577
## Epoch  95 MSE:  0.010303943417966366 R2 0.9458306431770325
## Epoch  96 MSE:  0.010996444150805473 R2 0.9427081346511841
## Epoch  97 MSE:  0.008778414689004421 R2 0.9561299085617065
## Epoch  98 MSE:  0.010812357999384403 R2 0.9480277299880981
## Epoch  99 MSE:  0.009326360188424587 R2 0.9512144923210144
## Epoch  100 MSE:  0.009986805729568005 R2 0.9442269206047058
## Epoch  101 MSE:  0.009843907319009304 R2 0.948183000087738
## Epoch  102 MSE:  0.010302797891199589 R2 0.9499773979187012
## Epoch  103 MSE:  0.010073578916490078 R2 0.9477660655975342
## Epoch  104 MSE:  0.009139630012214184 R2 0.95223468542099
## Epoch  105 MSE:  0.008659495040774345 R2 0.9550801515579224
## Epoch  106 MSE:  0.010690429247915745 R2 0.9472653865814209
## Epoch  107 MSE:  0.00929638184607029 R2 0.9503622055053711
## Epoch  108 MSE:  0.010904992930591106 R2 0.9416102766990662
## Epoch  109 MSE:  0.010579852387309074 R2 0.9475810527801514
## Epoch  110 MSE:  0.01027741003781557 R2 0.9492634534835815
## Epoch  111 MSE:  0.009993099607527256 R2 0.9443711042404175
## Epoch  112 MSE:  0.009958111681044102 R2 0.9469329714775085
## Epoch  113 MSE:  0.009505224414169788 R2 0.9536134600639343
## Epoch  114 MSE:  0.008814816363155842 R2 0.9548131227493286
## Epoch  115 MSE:  0.008779405616223812 R2 0.9532050490379333
## Epoch  116 MSE:  0.009141667746007442 R2 0.9517090320587158
## Epoch  117 MSE:  0.008758174255490303 R2 0.957345187664032
## Epoch  118 MSE:  0.009343898855149746 R2 0.9502443671226501
## Epoch  119 MSE:  0.00833556242287159 R2 0.9573606848716736
## Epoch  120 MSE:  0.009603414684534073 R2 0.9506988525390625
## Epoch  121 MSE:  0.009733635932207108 R2 0.9518366456031799
## Epoch  122 MSE:  0.009610765613615513 R2 0.9506756663322449
## Epoch  123 MSE:  0.009675065986812115 R2 0.9510132670402527
## Epoch  124 MSE:  0.009068235754966736 R2 0.9542862176895142
## Epoch  125 MSE:  0.010685114189982414 R2 0.9414517879486084
## Epoch  126 MSE:  0.009162258356809616 R2 0.9514691829681396
## Epoch  127 MSE:  0.009503367356956005 R2 0.953647792339325
## Epoch  128 MSE:  0.009567338041961193 R2 0.9513320922851562
## Epoch  129 MSE:  0.009279369376599789 R2 0.9494407773017883
## Epoch  130 MSE:  0.009109370410442352 R2 0.9535083770751953
## Epoch  131 MSE:  0.009335143491625786 R2 0.9505072236061096
## Epoch  132 MSE:  0.008862456306815147 R2 0.9543634653091431
## Epoch  133 MSE:  0.010241976007819176 R2 0.9468673467636108
## Epoch  134 MSE:  0.009531587362289429 R2 0.9526965618133545
## Epoch  135 MSE:  0.009000749327242374 R2 0.9531651735305786
## Epoch  136 MSE:  0.009271440096199512 R2 0.9472977519035339
## Epoch  137 MSE:  0.00833127461373806 R2 0.9600118398666382
## Epoch  138 MSE:  0.009623165242373943 R2 0.9543843269348145
## Epoch  139 MSE:  0.009550872258841991 R2 0.9467430114746094
## Epoch  140 MSE:  0.008669355884194374 R2 0.9536998867988586
## Epoch  141 MSE:  0.008904428221285343 R2 0.9581335783004761
## Epoch  142 MSE:  0.00951224472373724 R2 0.9497624039649963
## Epoch  143 MSE:  0.008225470781326294 R2 0.955711305141449
## Epoch  144 MSE:  0.007655432913452387 R2 0.9617536067962646
## Epoch  145 MSE:  0.009617741219699383 R2 0.9525779485702515
## Epoch  146 MSE:  0.009199470281600952 R2 0.9506943821907043
## Epoch  147 MSE:  0.009035908617079258 R2 0.9512618184089661
## Epoch  148 MSE:  0.007789858151227236 R2 0.9631843566894531
## Epoch  149 MSE:  0.008900074288249016 R2 0.9521812200546265
## Epoch  150 MSE:  0.008524342440068722 R2 0.955144464969635
## Epoch  151 MSE:  0.008699076250195503 R2 0.9586480259895325
## Epoch  152 MSE:  0.008004042319953442 R2 0.958063006401062
## Epoch  153 MSE:  0.01045997068285942 R2 0.9459646344184875
## Epoch  154 MSE:  0.008813305757939816 R2 0.955852210521698
## Epoch  155 MSE:  0.009082063101232052 R2 0.9527032375335693
## Epoch  156 MSE:  0.01134663075208664 R2 0.9460525512695312
## Epoch  157 MSE:  0.007205232046544552 R2 0.9611515402793884
## Epoch  158 MSE:  0.011992964893579483 R2 0.9266119599342346
## Epoch  159 MSE:  0.009959407150745392 R2 0.953726053237915
## Epoch  160 MSE:  0.008605774492025375 R2 0.9584540128707886
## Epoch  161 MSE:  0.008193757385015488 R2 0.95553058385849
## Epoch  162 MSE:  0.00929043535143137 R2 0.9472683668136597
## Epoch  163 MSE:  0.009513989090919495 R2 0.9476475715637207
## Epoch  164 MSE:  0.009052075445652008 R2 0.9561070203781128
## Epoch  165 MSE:  0.009934055618941784 R2 0.9555201530456543
## Epoch  166 MSE:  0.008421989157795906 R2 0.9516104459762573
## Epoch  167 MSE:  0.009009155444800854 R2 0.9472737908363342
## Epoch  168 MSE:  0.01047644205391407 R2 0.9478313326835632
## Epoch  169 MSE:  0.008151191286742687 R2 0.9605578184127808
## Epoch  170 MSE:  0.008254706859588623 R2 0.9585602879524231
## Epoch  171 MSE:  0.007791442330926657 R2 0.9594950079917908
## Epoch  172 MSE:  0.007816464640200138 R2 0.9601609706878662
## Epoch  173 MSE:  0.008463647216558456 R2 0.9570326209068298
## Epoch  174 MSE:  0.008728602901101112 R2 0.9575611352920532
## Epoch  175 MSE:  0.00865170732140541 R2 0.953685998916626
## Epoch  176 MSE:  0.007570682559162378 R2 0.9611909985542297
## Epoch  177 MSE:  0.007333747576922178 R2 0.9653509855270386
## Epoch  178 MSE:  0.007630809675902128 R2 0.9632920026779175
## Epoch  179 MSE:  0.008648314513266087 R2 0.9516938328742981
## Epoch  180 MSE:  0.006993351969867945 R2 0.9630697965621948
## Epoch  181 MSE:  0.008667580783367157 R2 0.9583492279052734
## Epoch  182 MSE:  0.007382080890238285 R2 0.9607289433479309
## Epoch  183 MSE:  0.009423547424376011 R2 0.9455956220626831
## Epoch  184 MSE:  0.007583572529256344 R2 0.9588890671730042
## Epoch  185 MSE:  0.008538267575204372 R2 0.960455596446991
## Epoch  186 MSE:  0.009383083321154118 R2 0.9572867155075073
## Epoch  187 MSE:  0.006276839412748814 R2 0.9662052392959595
## Epoch  188 MSE:  0.00809152889996767 R2 0.9554105401039124
## Epoch  189 MSE:  0.007534824311733246 R2 0.9594323635101318
## Epoch  190 MSE:  0.007782209198921919 R2 0.9619906544685364
## Epoch  191 MSE:  0.008638923987746239 R2 0.9598569273948669
## Epoch  192 MSE:  0.0071949344128370285 R2 0.963735044002533
## Epoch  193 MSE:  0.00776807451620698 R2 0.9546290636062622
## Epoch  194 MSE:  0.007876898162066936 R2 0.9601911306381226
## Epoch  195 MSE:  0.007446295581758022 R2 0.9633427262306213
## Epoch  196 MSE:  0.007592636626213789 R2 0.9609328508377075
## Epoch  197 MSE:  0.006312173325568438 R2 0.9682732224464417
## Epoch  198 MSE:  0.006196524482220411 R2 0.969232976436615
## Epoch  199 MSE:  0.008596575818955898 R2 0.9543557167053223
## Epoch  200 MSE:  0.007973997853696346 R2 0.9623196125030518
## Epoch  201 MSE:  0.006786488927900791 R2 0.9666585922241211
## Epoch  202 MSE:  0.007128921337425709 R2 0.9623326063156128
## Epoch  203 MSE:  0.00574052007868886 R2 0.9699428677558899
## Epoch  204 MSE:  0.00713359797373414 R2 0.9659086465835571
## Epoch  205 MSE:  0.007161654997617006 R2 0.9634640216827393
## Epoch  206 MSE:  0.006550908088684082 R2 0.9646643400192261
## Epoch  207 MSE:  0.0068421801552176476 R2 0.9656534194946289
## Epoch  208 MSE:  0.005673703271895647 R2 0.9708356261253357
## Epoch  209 MSE:  0.006449716165661812 R2 0.965156078338623
## Epoch  210 MSE:  0.006427771411836147 R2 0.9675875902175903
## Epoch  211 MSE:  0.008084830828011036 R2 0.9629685282707214
## Epoch  212 MSE:  0.0057359798811376095 R2 0.9713265895843506
## Epoch  213 MSE:  0.009129781275987625 R2 0.9447212815284729
## Epoch  214 MSE:  0.007758782710880041 R2 0.963043212890625
## Epoch  215 MSE:  0.00925472192466259 R2 0.9594692587852478
## Epoch  216 MSE:  0.0067895748652517796 R2 0.9648026823997498
## Epoch  217 MSE:  0.008429493755102158 R2 0.9468057155609131
## Epoch  218 MSE:  0.006647494621574879 R2 0.965067446231842
## Epoch  219 MSE:  0.008042261004447937 R2 0.9641276001930237
## Epoch  220 MSE:  0.006943634711205959 R2 0.9674563407897949
## Epoch  221 MSE:  0.007015637122094631 R2 0.9621544480323792
## Epoch  222 MSE:  0.006048776675015688 R2 0.9679921865463257
## Epoch  223 MSE:  0.007128388155251741 R2 0.9639651775360107
## Epoch  224 MSE:  0.0070009673945605755 R2 0.963036060333252
## Epoch  225 MSE:  0.0063621364533901215 R2 0.9652423858642578
## Epoch  226 MSE:  0.006185003090649843 R2 0.9663795828819275
## Epoch  227 MSE:  0.006438425276428461 R2 0.96901935338974
## Epoch  228 MSE:  0.006010518409311771 R2 0.9718626737594604
## Epoch  229 MSE:  0.005647418089210987 R2 0.9723436832427979
## Epoch  230 MSE:  0.005394381005316973 R2 0.9735841155052185
## Epoch  231 MSE:  0.005575934890657663 R2 0.9712538123130798
## Epoch  232 MSE:  0.006198697257786989 R2 0.9670788645744324
## Epoch  233 MSE:  0.005613729357719421 R2 0.9714838862419128
## Epoch  234 MSE:  0.0057612028904259205 R2 0.9734523296356201
## Epoch  235 MSE:  0.005692220292985439 R2 0.9728236198425293
## Epoch  236 MSE:  0.004565902519971132 R2 0.9766972064971924
## Epoch  237 MSE:  0.004955127369612455 R2 0.9735045433044434
## Epoch  238 MSE:  0.004718957934528589 R2 0.9755628108978271
## Epoch  239 MSE:  0.005052037071436644 R2 0.9748619794845581
## Epoch  240 MSE:  0.006029763724654913 R2 0.9698092937469482
## Epoch  241 MSE:  0.004822004120796919 R2 0.9760769605636597
## Epoch  242 MSE:  0.005735547747462988 R2 0.970954179763794
## Epoch  243 MSE:  0.005111425183713436 R2 0.9728104472160339
## Epoch  244 MSE:  0.005012674257159233 R2 0.9738149642944336
## Epoch  245 MSE:  0.004637125413864851 R2 0.9773094058036804
## Epoch  246 MSE:  0.00525005254894495 R2 0.9751859903335571
## Epoch  247 MSE:  0.005226116627454758 R2 0.9733985662460327
## Epoch  248 MSE:  0.00577901303768158 R2 0.9684407711029053
## Epoch  249 MSE:  0.005689190700650215 R2 0.9725633859634399
## Epoch  250 MSE:  0.005145838484168053 R2 0.9757484197616577
## Epoch  251 MSE:  0.006155603099614382 R2 0.9693459272384644
## Epoch  252 MSE:  0.006472359877079725 R2 0.9665019512176514
## Epoch  253 MSE:  0.006510278210043907 R2 0.9659515023231506
## Epoch  254 MSE:  0.0077912574633955956 R2 0.9584161043167114
## Epoch  255 MSE:  0.00648656627163291 R2 0.963390588760376
## Epoch  256 MSE:  0.005997247528284788 R2 0.9708623886108398
## Epoch  257 MSE:  0.0069855195470154285 R2 0.9664041996002197
## Epoch  258 MSE:  0.006738909520208836 R2 0.968481719493866
## Epoch  259 MSE:  0.0075574410147964954 R2 0.9587312340736389
## Epoch  260 MSE:  0.005418908316642046 R2 0.9739459753036499
## Epoch  261 MSE:  0.007847707718610764 R2 0.9621928334236145
## Epoch  262 MSE:  0.006736191920936108 R2 0.9658989310264587
## Epoch  263 MSE:  0.008485203608870506 R2 0.9530889391899109
## Epoch  264 MSE:  0.00800260528922081 R2 0.9544270634651184
## Epoch  265 MSE:  0.0073430538177490234 R2 0.9607998728752136
## Epoch  266 MSE:  0.008029432035982609 R2 0.96198970079422
## Epoch  267 MSE:  0.0071290056221187115 R2 0.9648715853691101
## Epoch  268 MSE:  0.007493348326534033 R2 0.9609792232513428
## Epoch  269 MSE:  0.0075207967311143875 R2 0.9591938853263855
## Epoch  270 MSE:  0.007436951622366905 R2 0.964181661605835
## Epoch  271 MSE:  0.007356744725257158 R2 0.9635609984397888
## Epoch  272 MSE:  0.0067794728092849255 R2 0.9644497036933899
## Epoch  273 MSE:  0.007375949062407017 R2 0.9634525179862976
## Epoch  274 MSE:  0.007576477713882923 R2 0.9636195302009583
## Epoch  275 MSE:  0.007976565510034561 R2 0.9568899869918823
## Epoch  276 MSE:  0.006386935710906982 R2 0.9703540205955505
## Epoch  277 MSE:  0.0064359973184764385 R2 0.9664402008056641
## Epoch  278 MSE:  0.006430755835026503 R2 0.9666799902915955
## Epoch  279 MSE:  0.005276428535580635 R2 0.9702255129814148
## Epoch  280 MSE:  0.006923682056367397 R2 0.9668565392494202
## Epoch  281 MSE:  0.007382635027170181 R2 0.966993510723114
## Epoch  282 MSE:  0.00800259131938219 R2 0.952319860458374
## Epoch  283 MSE:  0.006486250553280115 R2 0.9627496600151062
## Epoch  284 MSE:  0.006587837357074022 R2 0.9686839580535889
## Epoch  285 MSE:  0.007034036330878735 R2 0.9682321548461914
## Epoch  286 MSE:  0.005024658516049385 R2 0.9737381935119629
## Epoch  287 MSE:  0.006065655965358019 R2 0.965411365032196
## Epoch  288 MSE:  0.005467365495860577 R2 0.9718729853630066
## Epoch  289 MSE:  0.005746257025748491 R2 0.9742549061775208
## Epoch  290 MSE:  0.007465220056474209 R2 0.9659151434898376
## Epoch  291 MSE:  0.005705045070499182 R2 0.9685552716255188
## Epoch  292 MSE:  0.00576650770381093 R2 0.9675907492637634
## Epoch  293 MSE:  0.006097708363085985 R2 0.9670138359069824
## Epoch  294 MSE:  0.004715206101536751 R2 0.9778618812561035
## Epoch  295 MSE:  0.007890542037785053 R2 0.9644694924354553
## Epoch  296 MSE:  0.004611301701515913 R2 0.9773174524307251
## Epoch  297 MSE:  0.005941724870353937 R2 0.9666162133216858
## Epoch  298 MSE:  0.005209742579609156 R2 0.9708541035652161
## Epoch  299 MSE:  0.005461267661303282 R2 0.9733696579933167
## Epoch  300 MSE:  0.007012419868260622 R2 0.9698745012283325
```

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>MSE_hist</strong> 和 <strong>R2_hist</strong>：记录每轮的 MSE
和 R² 分数
</li>
<li style="margin-top:2px;margin-bottom:2px;">
每轮训练：前向传播计算预测值，计算损失和 R² 分数，反向传播更新参数
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>optimiser.zero_grad()</strong>，清空梯度
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>loss.backward()</strong>，计算梯度
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>optimiser.step()</strong>，更新参数
</li>
</ul>

# 结果评估

``` python
y_test_pred = model(x_test)
trainScore = mean_squared_error(y_train.detach().numpy(), y_train_pred.detach().numpy())
r2_train = r2_score(y_train.detach().numpy(), y_train_pred.detach().numpy())
print('Train Score: %.2f RMSE' % (trainScore))
## Train Score: 0.01 RMSE
print('Train R^2: %.2f' % (r2_train))
## Train R^2: 0.97

testScore = math.sqrt(mean_squared_error(y_test.detach().numpy(), y_test_pred.detach().numpy()))
r2_test = r2_score(y_test.detach().numpy(), y_test_pred.detach().numpy())
print('Test Score: %.2f RMSE' % (testScore))
## Test Score: 0.12 RMSE
print('Test R^2: %.2f' % (r2_test))
## Test R^2: 0.67
```

# SHAP

``` python
# SHAP 解释部分
print("Starting SHAP explanation...")
## Starting SHAP explanation...
shap_values, explainer, test_data_flattened = explain_model_with_shap(
    model, 
    x_test.numpy(), 
    # 为了快速测试设置为 1
    background_samples = 1, 
    seq_len = x_test.shape[1], 
    input_dim = x_test.shape[2]
)
##   0%|          | 0/46 [00:00<?, ?it/s]  4%|4         | 2/46 [00:00<00:04, 10.86it/s]  9%|8         | 4/46 [00:00<00:05,  8.00it/s] 11%|#         | 5/46 [00:00<00:05,  7.67it/s] 13%|#3        | 6/46 [00:00<00:05,  7.38it/s] 15%|#5        | 7/46 [00:00<00:05,  7.25it/s] 17%|#7        | 8/46 [00:01<00:05,  7.14it/s] 20%|#9        | 9/46 [00:01<00:05,  6.98it/s] 22%|##1       | 10/46 [00:01<00:05,  6.99it/s] 24%|##3       | 11/46 [00:01<00:05,  6.96it/s] 26%|##6       | 12/46 [00:01<00:04,  6.94it/s] 28%|##8       | 13/46 [00:01<00:04,  6.98it/s] 30%|###       | 14/46 [00:01<00:04,  7.00it/s] 33%|###2      | 15/46 [00:02<00:04,  7.03it/s] 35%|###4      | 16/46 [00:02<00:04,  7.00it/s] 37%|###6      | 17/46 [00:02<00:04,  7.01it/s] 39%|###9      | 18/46 [00:02<00:04,  6.98it/s] 41%|####1     | 19/46 [00:02<00:03,  6.99it/s] 43%|####3     | 20/46 [00:02<00:03,  6.94it/s] 46%|####5     | 21/46 [00:02<00:03,  6.96it/s] 48%|####7     | 22/46 [00:03<00:03,  6.96it/s] 50%|#####     | 23/46 [00:03<00:03,  6.96it/s] 52%|#####2    | 24/46 [00:03<00:03,  6.98it/s] 54%|#####4    | 25/46 [00:03<00:03,  6.92it/s] 57%|#####6    | 26/46 [00:03<00:02,  6.94it/s] 59%|#####8    | 27/46 [00:03<00:02,  6.90it/s] 61%|######    | 28/46 [00:03<00:02,  6.88it/s] 63%|######3   | 29/46 [00:04<00:02,  6.90it/s] 65%|######5   | 30/46 [00:04<00:02,  6.89it/s] 67%|######7   | 31/46 [00:04<00:02,  6.86it/s] 70%|######9   | 32/46 [00:04<00:02,  6.84it/s] 72%|#######1  | 33/46 [00:04<00:01,  6.86it/s] 74%|#######3  | 34/46 [00:04<00:01,  6.83it/s] 76%|#######6  | 35/46 [00:04<00:01,  6.93it/s] 78%|#######8  | 36/46 [00:05<00:01,  6.77it/s] 80%|########  | 37/46 [00:05<00:01,  6.66it/s] 83%|########2 | 38/46 [00:05<00:01,  6.57it/s] 85%|########4 | 39/46 [00:05<00:01,  6.48it/s] 87%|########6 | 40/46 [00:05<00:00,  6.59it/s] 89%|########9 | 41/46 [00:05<00:00,  6.71it/s] 91%|#########1| 42/46 [00:06<00:00,  6.79it/s] 93%|#########3| 43/46 [00:06<00:00,  6.59it/s] 96%|#########5| 44/46 [00:06<00:00,  6.65it/s] 98%|#########7| 45/46 [00:06<00:00,  6.76it/s]100%|##########| 46/46 [00:06<00:00,  6.79it/s]100%|##########| 46/46 [00:06<00:00,  6.95it/s]

# 聚合 SHAP 值：对每个特征在时间步上的 SHAP 值取平均
feature_names = args.input_features
n_features = len(args.input_features)
n_timesteps = x_test.shape[1]
shap_values_aggregated = np.zeros((test_data_flattened.shape[0], n_features))
test_data_aggregated = np.zeros((test_data_flattened.shape[0], n_features))

for i in range(n_features):
    feature_indices = [i + j * n_features for j in range(n_timesteps)]
    shap_values_aggregated[:, i] = np.mean(shap_values[:, feature_indices], axis = 1)
    test_data_aggregated[:, i] = np.mean(test_data_flattened[:, feature_indices], axis = 1)
    
# 绘制 SHAP 总结图
plt.figure(figsize = (11, 7))
shap.summary_plot(shap_values_aggregated, test_data_aggregated, feature_names = feature_names, plot_type = "bar")
```

<img src="Transformer+KAN+SHAP_files/figure-markdown_strict/unnamed-chunk-17-3.png" width="768" />

``` python

# 绘制 SHAP 详细图
plt.figure(figsize = (11, 7))
shap.summary_plot(shap_values_aggregated, test_data_aggregated, feature_names = feature_names)
```

<img src="Transformer+KAN+SHAP_files/figure-markdown_strict/unnamed-chunk-17-4.png" width="768" />

<p>
调用 SHAP 解释函数，计算测试数据的 SHAP 值，并绘制图。
</p>
