---
title: 基于 Transformer 和 KAN 的时间序列预测模型 + SHAP 模型可解释性
date: 2025-05-07 08:27:30
tags: [Python, 机器学习, SHAP]
categories: [[案例分享, 机器学习, Transformer]]
---


# 环境设置

``` r
# 指定 Python 环境
reticulate::use_python("C:/ProgramData/Anaconda3/python.exe")

# 切换工作目录
wkdir = dirname(rstudioapi::getActiveDocumentContext()$path)
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

![](/imgs/0445099e2c1e9f3a2fae1a42b5d64685.png)
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
## Epoch  1 MSE:  0.20225873589515686 R2 -31.886425018310547
## Epoch  2 MSE:  0.298414409160614 R2 -42.046592712402344
## Epoch  3 MSE:  0.23563598096370697 R2 -43.875423431396484
## Epoch  4 MSE:  0.15872986614704132 R2 -17.286842346191406
## Epoch  5 MSE:  0.12643669545650482 R2 -6.356443405151367
## Epoch  6 MSE:  0.09145685285329819 R2 -1.0092172622680664
## Epoch  7 MSE:  0.0745336040854454 R2 0.20615911483764648
## Epoch  8 MSE:  0.07983776926994324 R2 0.4293314218521118
## Epoch  9 MSE:  0.07939670234918594 R2 0.5352250337600708
## Epoch  10 MSE:  0.07990598678588867 R2 0.5464810132980347
## Epoch  11 MSE:  0.08307226002216339 R2 0.4392090439796448
## Epoch  12 MSE:  0.08004911243915558 R2 0.4856940507888794
## Epoch  13 MSE:  0.0795847475528717 R2 0.4719334840774536
## Epoch  14 MSE:  0.07717426121234894 R2 0.46455514430999756
## Epoch  15 MSE:  0.07702719420194626 R2 0.39811068773269653
## Epoch  16 MSE:  0.07859313488006592 R2 0.3260354995727539
## Epoch  17 MSE:  0.08135922998189926 R2 0.22676098346710205
## Epoch  18 MSE:  0.08019085228443146 R2 0.2370361089706421
## Epoch  19 MSE:  0.07962346822023392 R2 0.19053369760513306
## Epoch  20 MSE:  0.0786891058087349 R2 0.17718809843063354
## Epoch  21 MSE:  0.07530727237462997 R2 0.20182985067367554
## Epoch  22 MSE:  0.07261732220649719 R2 0.2662377953529358
## Epoch  23 MSE:  0.0729486271739006 R2 0.39610427618026733
## Epoch  24 MSE:  0.07484294474124908 R2 0.4820999503135681
## Epoch  25 MSE:  0.07016625255346298 R2 0.5492802858352661
## Epoch  26 MSE:  0.06698057800531387 R2 0.46985524892807007
## Epoch  27 MSE:  0.06927278637886047 R2 0.28384435176849365
## Epoch  28 MSE:  0.061243686825037 R2 0.5423012971878052
## Epoch  29 MSE:  0.05536247417330742 R2 0.6476640105247498
## Epoch  30 MSE:  0.05800174921751022 R2 0.6330298185348511
## Epoch  31 MSE:  0.05266817659139633 R2 0.6027801036834717
## Epoch  32 MSE:  0.044197194278240204 R2 0.6896723508834839
## Epoch  33 MSE:  0.040882233530282974 R2 0.7287555932998657
## Epoch  34 MSE:  0.04244617372751236 R2 0.6274962425231934
## Epoch  35 MSE:  0.037999190390110016 R2 0.6371495127677917
## Epoch  36 MSE:  0.03200247138738632 R2 0.7697381973266602
## Epoch  37 MSE:  0.02890932373702526 R2 0.8646442294120789
## Epoch  38 MSE:  0.028858479112386703 R2 0.8749728202819824
## Epoch  39 MSE:  0.024547487497329712 R2 0.8800598382949829
## Epoch  40 MSE:  0.027305975556373596 R2 0.8329728245735168
## Epoch  41 MSE:  0.027785031124949455 R2 0.8197386264801025
## Epoch  42 MSE:  0.02365117147564888 R2 0.8766265511512756
## Epoch  43 MSE:  0.024261469021439552 R2 0.8965966105461121
## Epoch  44 MSE:  0.024466168135404587 R2 0.9011127948760986
## Epoch  45 MSE:  0.01808491162955761 R2 0.91972815990448
## Epoch  46 MSE:  0.023742683231830597 R2 0.8634177446365356
## Epoch  47 MSE:  0.023093033581972122 R2 0.8548323512077332
## Epoch  48 MSE:  0.021937701851129532 R2 0.8700320720672607
## Epoch  49 MSE:  0.022022832185029984 R2 0.8752726316452026
## Epoch  50 MSE:  0.018366539850831032 R2 0.896073043346405
## Epoch  51 MSE:  0.0227207001298666 R2 0.8796857595443726
## Epoch  52 MSE:  0.01927117444574833 R2 0.8969408869743347
## Epoch  53 MSE:  0.01714526116847992 R2 0.8931755423545837
## Epoch  54 MSE:  0.02033025398850441 R2 0.8700670003890991
## Epoch  55 MSE:  0.01735217683017254 R2 0.8970661759376526
## Epoch  56 MSE:  0.01667058654129505 R2 0.9101118445396423
## Epoch  57 MSE:  0.01843738742172718 R2 0.9047363996505737
## Epoch  58 MSE:  0.020372120663523674 R2 0.9006676077842712
## Epoch  59 MSE:  0.01660245656967163 R2 0.9221010804176331
## Epoch  60 MSE:  0.016648301854729652 R2 0.9221247434616089
## Epoch  61 MSE:  0.016397720202803612 R2 0.9170522093772888
## Epoch  62 MSE:  0.016446013003587723 R2 0.9092898964881897
## Epoch  63 MSE:  0.016286415979266167 R2 0.9047813415527344
## Epoch  64 MSE:  0.017099885269999504 R2 0.8930974006652832
## Epoch  65 MSE:  0.01610066369175911 R2 0.9076640605926514
## Epoch  66 MSE:  0.014690566807985306 R2 0.9230966567993164
## Epoch  67 MSE:  0.015427093021571636 R2 0.923015296459198
## Epoch  68 MSE:  0.014979406259953976 R2 0.9232712984085083
## Epoch  69 MSE:  0.015858955681324005 R2 0.9135801792144775
## Epoch  70 MSE:  0.016440976411104202 R2 0.9061396718025208
## Epoch  71 MSE:  0.014070401899516582 R2 0.9245411157608032
## Epoch  72 MSE:  0.013009407557547092 R2 0.9333095550537109
## Epoch  73 MSE:  0.01386302150785923 R2 0.9295218586921692
## Epoch  74 MSE:  0.013586930930614471 R2 0.9268723130226135
## Epoch  75 MSE:  0.01260241400450468 R2 0.9303719401359558
## Epoch  76 MSE:  0.014276157133281231 R2 0.9230559468269348
## Epoch  77 MSE:  0.01513133104890585 R2 0.9209627509117126
## Epoch  78 MSE:  0.01235912274569273 R2 0.9328458309173584
## Epoch  79 MSE:  0.013148767873644829 R2 0.9292569756507874
## Epoch  80 MSE:  0.012890445068478584 R2 0.9352448582649231
## Epoch  81 MSE:  0.011547064408659935 R2 0.9441823959350586
## Epoch  82 MSE:  0.011410089209675789 R2 0.9427899122238159
## Epoch  83 MSE:  0.014889625832438469 R2 0.9143943190574646
## Epoch  84 MSE:  0.012940896674990654 R2 0.9291906356811523
## Epoch  85 MSE:  0.01248413510620594 R2 0.9352758526802063
## Epoch  86 MSE:  0.012234735302627087 R2 0.9404329061508179
## Epoch  87 MSE:  0.013958021998405457 R2 0.925429105758667
## Epoch  88 MSE:  0.011953799985349178 R2 0.9360503554344177
## Epoch  89 MSE:  0.01171583030372858 R2 0.9376858472824097
## Epoch  90 MSE:  0.011882662773132324 R2 0.939037561416626
## Epoch  91 MSE:  0.011556466110050678 R2 0.9400622248649597
## Epoch  92 MSE:  0.01083365548402071 R2 0.9444785118103027
## Epoch  93 MSE:  0.012246312573552132 R2 0.9337013363838196
## Epoch  94 MSE:  0.012195369228720665 R2 0.937350869178772
## Epoch  95 MSE:  0.011055802926421165 R2 0.9418314695358276
## Epoch  96 MSE:  0.011127471923828125 R2 0.9432144165039062
## Epoch  97 MSE:  0.011168841272592545 R2 0.9431397318840027
## Epoch  98 MSE:  0.010964706540107727 R2 0.9446061849594116
## Epoch  99 MSE:  0.012929677963256836 R2 0.9281521439552307
## Epoch  100 MSE:  0.011543658562004566 R2 0.9414356350898743
## Epoch  101 MSE:  0.010463657788932323 R2 0.9480983018875122
## Epoch  102 MSE:  0.010957881808280945 R2 0.9446573853492737
## Epoch  103 MSE:  0.010334246791899204 R2 0.9478861093521118
## Epoch  104 MSE:  0.009886440820991993 R2 0.9463023543357849
## Epoch  105 MSE:  0.01164489146322012 R2 0.9369932413101196
## Epoch  106 MSE:  0.011157775297760963 R2 0.9400005340576172
## Epoch  107 MSE:  0.011287794448435307 R2 0.944885790348053
## Epoch  108 MSE:  0.010159262455999851 R2 0.9520426392555237
## Epoch  109 MSE:  0.010824288241565228 R2 0.9423235654830933
## Epoch  110 MSE:  0.010752422735095024 R2 0.9390590190887451
## Epoch  111 MSE:  0.010514352470636368 R2 0.9454538822174072
## Epoch  112 MSE:  0.010996238328516483 R2 0.9444611668586731
## Epoch  113 MSE:  0.011360413394868374 R2 0.9443960189819336
## Epoch  114 MSE:  0.011175724677741528 R2 0.9395014643669128
## Epoch  115 MSE:  0.010253587737679482 R2 0.9456093311309814
## Epoch  116 MSE:  0.010191611014306545 R2 0.9457188248634338
## Epoch  117 MSE:  0.010211308486759663 R2 0.9504458904266357
## Epoch  118 MSE:  0.010046865791082382 R2 0.9473015069961548
## Epoch  119 MSE:  0.011700510047376156 R2 0.9378324747085571
## Epoch  120 MSE:  0.009451260790228844 R2 0.9509357810020447
## Epoch  121 MSE:  0.010614283382892609 R2 0.9462201595306396
## Epoch  122 MSE:  0.011142362840473652 R2 0.9467788934707642
## Epoch  123 MSE:  0.010354393161833286 R2 0.9457390904426575
## Epoch  124 MSE:  0.009864013642072678 R2 0.9472853541374207
## Epoch  125 MSE:  0.009295382536947727 R2 0.9501583576202393
## Epoch  126 MSE:  0.010780416429042816 R2 0.9442393183708191
## Epoch  127 MSE:  0.010747263208031654 R2 0.9460399746894836
## Epoch  128 MSE:  0.011286415159702301 R2 0.9413518905639648
## Epoch  129 MSE:  0.009668487124145031 R2 0.9498662352561951
## Epoch  130 MSE:  0.009119526483118534 R2 0.9521781206130981
## Epoch  131 MSE:  0.010137779638171196 R2 0.9477961659431458
## Epoch  132 MSE:  0.010773522779345512 R2 0.9436598420143127
## Epoch  133 MSE:  0.010250646620988846 R2 0.9464605450630188
## Epoch  134 MSE:  0.010532618500292301 R2 0.9454031586647034
## Epoch  135 MSE:  0.009705400094389915 R2 0.949856698513031
## Epoch  136 MSE:  0.0091252401471138 R2 0.95412278175354
## Epoch  137 MSE:  0.010239989496767521 R2 0.9456197619438171
## Epoch  138 MSE:  0.011189078912138939 R2 0.9390959143638611
## Epoch  139 MSE:  0.010134459473192692 R2 0.9516520500183105
## Epoch  140 MSE:  0.009851214475929737 R2 0.9527378678321838
## Epoch  141 MSE:  0.010534733533859253 R2 0.9392232894897461
## Epoch  142 MSE:  0.009433116763830185 R2 0.9495104551315308
## Epoch  143 MSE:  0.009926737286150455 R2 0.9517117738723755
## Epoch  144 MSE:  0.009388117119669914 R2 0.955207347869873
## Epoch  145 MSE:  0.009214591234922409 R2 0.9513780474662781
## Epoch  146 MSE:  0.010308733209967613 R2 0.9427282810211182
## Epoch  147 MSE:  0.009721271693706512 R2 0.9492425322532654
## Epoch  148 MSE:  0.009637282229959965 R2 0.9544447064399719
## Epoch  149 MSE:  0.009965136647224426 R2 0.951054036617279
## Epoch  150 MSE:  0.009241165593266487 R2 0.9465551376342773
## Epoch  151 MSE:  0.01002117246389389 R2 0.9448351860046387
## Epoch  152 MSE:  0.010310526937246323 R2 0.9501459002494812
## Epoch  153 MSE:  0.009744048118591309 R2 0.9531636834144592
## Epoch  154 MSE:  0.009579528123140335 R2 0.949242115020752
## Epoch  155 MSE:  0.010707440786063671 R2 0.9384672045707703
## Epoch  156 MSE:  0.009882104583084583 R2 0.9479899406433105
## Epoch  157 MSE:  0.00983772985637188 R2 0.9538400769233704
## Epoch  158 MSE:  0.010067986324429512 R2 0.9507231712341309
## Epoch  159 MSE:  0.00988831277936697 R2 0.9467902183532715
## Epoch  160 MSE:  0.009872454218566418 R2 0.9476045370101929
## Epoch  161 MSE:  0.009737255051732063 R2 0.9510957598686218
## Epoch  162 MSE:  0.00992168951779604 R2 0.9492301344871521
## Epoch  163 MSE:  0.009958731010556221 R2 0.9478836059570312
## Epoch  164 MSE:  0.009648142382502556 R2 0.948082685470581
## Epoch  165 MSE:  0.009274373762309551 R2 0.9537317752838135
## Epoch  166 MSE:  0.009255626238882542 R2 0.9549742937088013
## Epoch  167 MSE:  0.009442847222089767 R2 0.9535589218139648
## Epoch  168 MSE:  0.009413758292794228 R2 0.9516565203666687
## Epoch  169 MSE:  0.009726248681545258 R2 0.9456560015678406
## Epoch  170 MSE:  0.009528640657663345 R2 0.9477475881576538
## Epoch  171 MSE:  0.009518753737211227 R2 0.9514949321746826
## Epoch  172 MSE:  0.008933200500905514 R2 0.9567114114761353
## Epoch  173 MSE:  0.009507346898317337 R2 0.950848400592804
## Epoch  174 MSE:  0.0107524199411273 R2 0.9412626624107361
## Epoch  175 MSE:  0.009081174619495869 R2 0.9497719407081604
## Epoch  176 MSE:  0.009262737818062305 R2 0.954872727394104
## Epoch  177 MSE:  0.01159551553428173 R2 0.9475067257881165
## Epoch  178 MSE:  0.008793601766228676 R2 0.9552981853485107
## Epoch  179 MSE:  0.009172830730676651 R2 0.9482210278511047
## Epoch  180 MSE:  0.008899391628801823 R2 0.9508412480354309
## Epoch  181 MSE:  0.008961113169789314 R2 0.9548705220222473
## Epoch  182 MSE:  0.010228605009615421 R2 0.9528177976608276
## Epoch  183 MSE:  0.008195984177291393 R2 0.9557403326034546
## Epoch  184 MSE:  0.00946030206978321 R2 0.9475462436676025
## Epoch  185 MSE:  0.00893314927816391 R2 0.9521485567092896
## Epoch  186 MSE:  0.009931230917572975 R2 0.9541125893592834
## Epoch  187 MSE:  0.0089170653373003 R2 0.9558547139167786
## Epoch  188 MSE:  0.007836210541427135 R2 0.95893394947052
## Epoch  189 MSE:  0.009082172065973282 R2 0.9515615701675415
## Epoch  190 MSE:  0.010095741599798203 R2 0.9482653737068176
## Epoch  191 MSE:  0.009731031022965908 R2 0.9460965991020203
## Epoch  192 MSE:  0.009519870392978191 R2 0.9549015760421753
## Epoch  193 MSE:  0.00980305951088667 R2 0.9511070847511292
## Epoch  194 MSE:  0.009998088702559471 R2 0.9456890821456909
## Epoch  195 MSE:  0.010070920921862125 R2 0.9431175589561462
## Epoch  196 MSE:  0.008962555788457394 R2 0.9549666047096252
## Epoch  197 MSE:  0.008322876878082752 R2 0.9594370722770691
## Epoch  198 MSE:  0.007479468826204538 R2 0.9600977897644043
## Epoch  199 MSE:  0.00936348270624876 R2 0.9512057900428772
## Epoch  200 MSE:  0.008630257099866867 R2 0.9546411633491516
## Epoch  201 MSE:  0.008585885167121887 R2 0.9574077725410461
## Epoch  202 MSE:  0.009243194945156574 R2 0.9545102119445801
## Epoch  203 MSE:  0.009006887674331665 R2 0.9528292417526245
## Epoch  204 MSE:  0.009267548099160194 R2 0.9492133259773254
## Epoch  205 MSE:  0.00885502528399229 R2 0.9556372761726379
## Epoch  206 MSE:  0.007839334197342396 R2 0.9626772403717041
## Epoch  207 MSE:  0.007889660075306892 R2 0.9607027769088745
## Epoch  208 MSE:  0.009158053435385227 R2 0.9467917084693909
## Epoch  209 MSE:  0.008201127871870995 R2 0.9592642188072205
## Epoch  210 MSE:  0.009695758111774921 R2 0.9568880200386047
## Epoch  211 MSE:  0.008682046085596085 R2 0.9580089449882507
## Epoch  212 MSE:  0.008618245832622051 R2 0.9504194259643555
## Epoch  213 MSE:  0.0091047752648592 R2 0.945823609828949
## Epoch  214 MSE:  0.008653942495584488 R2 0.9575968980789185
## Epoch  215 MSE:  0.010378541424870491 R2 0.9530927538871765
## Epoch  216 MSE:  0.008754557929933071 R2 0.9560497403144836
## Epoch  217 MSE:  0.010468058288097382 R2 0.9399751424789429
## Epoch  218 MSE:  0.009039929136633873 R2 0.9459966421127319
## Epoch  219 MSE:  0.008150231093168259 R2 0.9600619077682495
## Epoch  220 MSE:  0.008759637363255024 R2 0.9594472050666809
## Epoch  221 MSE:  0.008761920034885406 R2 0.9561316967010498
## Epoch  222 MSE:  0.009602386504411697 R2 0.9435466527938843
## Epoch  223 MSE:  0.009304748848080635 R2 0.9470775127410889
## Epoch  224 MSE:  0.008283263072371483 R2 0.9592418074607849
## Epoch  225 MSE:  0.010027470998466015 R2 0.9540421366691589
## Epoch  226 MSE:  0.007696002256125212 R2 0.9626914858818054
## Epoch  227 MSE:  0.009052915498614311 R2 0.9517748355865479
## Epoch  228 MSE:  0.009078595787286758 R2 0.9480093121528625
## Epoch  229 MSE:  0.008896038867533207 R2 0.9561989307403564
## Epoch  230 MSE:  0.007574658375233412 R2 0.9640858173370361
## Epoch  231 MSE:  0.007233073003590107 R2 0.9638074636459351
## Epoch  232 MSE:  0.008781888522207737 R2 0.9508364796638489
## Epoch  233 MSE:  0.009323369711637497 R2 0.9511691331863403
## Epoch  234 MSE:  0.008625761605799198 R2 0.9579605460166931
## Epoch  235 MSE:  0.008406052365899086 R2 0.9610554575920105
## Epoch  236 MSE:  0.008183211088180542 R2 0.9547343850135803
## Epoch  237 MSE:  0.007108317222446203 R2 0.9614951610565186
## Epoch  238 MSE:  0.008179637603461742 R2 0.9595950841903687
## Epoch  239 MSE:  0.008626124821603298 R2 0.9592104554176331
## Epoch  240 MSE:  0.007012654561549425 R2 0.9626399278640747
## Epoch  241 MSE:  0.007322638761252165 R2 0.9607754945755005
## Epoch  242 MSE:  0.006714336108416319 R2 0.967159628868103
## Epoch  243 MSE:  0.006793661043047905 R2 0.9651563167572021
## Epoch  244 MSE:  0.007762091234326363 R2 0.9595820307731628
## Epoch  245 MSE:  0.008042494766414165 R2 0.9582329988479614
## Epoch  246 MSE:  0.007931752130389214 R2 0.9579026699066162
## Epoch  247 MSE:  0.007647352293133736 R2 0.964361846446991
## Epoch  248 MSE:  0.007763803470879793 R2 0.9565420746803284
## Epoch  249 MSE:  0.006216908805072308 R2 0.9688230156898499
## Epoch  250 MSE:  0.008562345057725906 R2 0.9614236950874329
## Epoch  251 MSE:  0.006510521750897169 R2 0.9671981334686279
## Epoch  252 MSE:  0.0068036518059670925 R2 0.9607090353965759
## Epoch  253 MSE:  0.006529017351567745 R2 0.9679325819015503
## Epoch  254 MSE:  0.006595964077860117 R2 0.9666531682014465
## Epoch  255 MSE:  0.006388965528458357 R2 0.9675566554069519
## Epoch  256 MSE:  0.006304628681391478 R2 0.9676119089126587
## Epoch  257 MSE:  0.006383887957781553 R2 0.967444658279419
## Epoch  258 MSE:  0.0074219414964318275 R2 0.9620017409324646
## Epoch  259 MSE:  0.0068730865605175495 R2 0.9645500779151917
## Epoch  260 MSE:  0.006524584721773863 R2 0.9663106799125671
## Epoch  261 MSE:  0.006852108985185623 R2 0.9632098078727722
## Epoch  262 MSE:  0.005741457920521498 R2 0.9721826910972595
## Epoch  263 MSE:  0.0063127693720161915 R2 0.9704563021659851
## Epoch  264 MSE:  0.008828637190163136 R2 0.9477763772010803
## Epoch  265 MSE:  0.0064428383484482765 R2 0.9700095653533936
## Epoch  266 MSE:  0.0080038383603096 R2 0.9644981026649475
## Epoch  267 MSE:  0.007069345563650131 R2 0.9614790081977844
## Epoch  268 MSE:  0.0082890335470438 R2 0.951194703578949
## Epoch  269 MSE:  0.006782536394894123 R2 0.9680345058441162
## Epoch  270 MSE:  0.009233972989022732 R2 0.9592376947402954
## Epoch  271 MSE:  0.006276641972362995 R2 0.9679719805717468
## Epoch  272 MSE:  0.008183509111404419 R2 0.9491691589355469
## Epoch  273 MSE:  0.0075199194252491 R2 0.9554727077484131
## Epoch  274 MSE:  0.006682570092380047 R2 0.9673746228218079
## Epoch  275 MSE:  0.007191981188952923 R2 0.9671205878257751
## Epoch  276 MSE:  0.0064865946769714355 R2 0.969824492931366
## Epoch  277 MSE:  0.006659356411546469 R2 0.9628877639770508
## Epoch  278 MSE:  0.006857100408524275 R2 0.961350679397583
## Epoch  279 MSE:  0.005178229883313179 R2 0.9731891751289368
## Epoch  280 MSE:  0.005541696213185787 R2 0.9723655581474304
## Epoch  281 MSE:  0.006241624243557453 R2 0.9700920581817627
## Epoch  282 MSE:  0.008030639961361885 R2 0.9623153209686279
## Epoch  283 MSE:  0.005591248162090778 R2 0.9699800610542297
## Epoch  284 MSE:  0.0058428505435585976 R2 0.9691315293312073
## Epoch  285 MSE:  0.006127728149294853 R2 0.9708541035652161
## Epoch  286 MSE:  0.0061877416446805 R2 0.970396101474762
## Epoch  287 MSE:  0.005941085983067751 R2 0.9708747267723083
## Epoch  288 MSE:  0.005854279734194279 R2 0.9699398279190063
## Epoch  289 MSE:  0.004996651783585548 R2 0.974057674407959
## Epoch  290 MSE:  0.005488466937094927 R2 0.9717930555343628
## Epoch  291 MSE:  0.006038226652890444 R2 0.970748782157898
## Epoch  292 MSE:  0.006797155365347862 R2 0.9672282934188843
## Epoch  293 MSE:  0.006360945291817188 R2 0.9652406573295593
## Epoch  294 MSE:  0.005848906934261322 R2 0.9678417444229126
## Epoch  295 MSE:  0.006622715853154659 R2 0.9676633477210999
## Epoch  296 MSE:  0.005139654502272606 R2 0.9750269651412964
## Epoch  297 MSE:  0.006728962529450655 R2 0.9666933417320251
## Epoch  298 MSE:  0.005661812610924244 R2 0.9708558917045593
## Epoch  299 MSE:  0.007148797158151865 R2 0.9645295143127441
## Epoch  300 MSE:  0.006579401902854443 R2 0.9649872183799744
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
## Test Score: 0.11 RMSE
print('Test R^2: %.2f' % (r2_test))
## Test R^2: 0.69
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
##   0%|          | 0/46 [00:00<?, ?it/s]  4%|4         | 2/46 [00:00<00:03, 12.58it/s]  9%|8         | 4/46 [00:00<00:05,  8.26it/s] 11%|#         | 5/46 [00:00<00:05,  7.84it/s] 13%|#3        | 6/46 [00:00<00:05,  7.52it/s] 15%|#5        | 7/46 [00:00<00:05,  7.34it/s] 17%|#7        | 8/46 [00:01<00:05,  7.28it/s] 20%|#9        | 9/46 [00:01<00:05,  7.03it/s] 22%|##1       | 10/46 [00:01<00:05,  6.99it/s] 24%|##3       | 11/46 [00:01<00:05,  6.92it/s] 26%|##6       | 12/46 [00:01<00:04,  6.86it/s] 28%|##8       | 13/46 [00:01<00:04,  6.87it/s] 30%|###       | 14/46 [00:01<00:04,  6.86it/s] 33%|###2      | 15/46 [00:02<00:04,  6.89it/s] 35%|###4      | 16/46 [00:02<00:04,  6.79it/s] 37%|###6      | 17/46 [00:02<00:04,  6.89it/s] 39%|###9      | 18/46 [00:02<00:04,  6.66it/s] 41%|####1     | 19/46 [00:02<00:04,  6.73it/s] 43%|####3     | 20/46 [00:02<00:03,  6.63it/s] 46%|####5     | 21/46 [00:02<00:03,  6.66it/s] 48%|####7     | 22/46 [00:03<00:03,  6.46it/s] 50%|#####     | 23/46 [00:03<00:03,  6.55it/s] 52%|#####2    | 24/46 [00:03<00:03,  6.63it/s] 54%|#####4    | 25/46 [00:03<00:03,  6.65it/s] 57%|#####6    | 26/46 [00:03<00:02,  6.68it/s] 59%|#####8    | 27/46 [00:03<00:02,  6.71it/s] 61%|######    | 28/46 [00:04<00:02,  6.77it/s] 63%|######3   | 29/46 [00:04<00:02,  6.78it/s] 65%|######5   | 30/46 [00:04<00:02,  6.66it/s] 67%|######7   | 31/46 [00:04<00:02,  6.68it/s] 70%|######9   | 32/46 [00:04<00:02,  6.73it/s] 72%|#######1  | 33/46 [00:04<00:01,  6.65it/s] 74%|#######3  | 34/46 [00:04<00:01,  6.42it/s] 76%|#######6  | 35/46 [00:05<00:01,  6.44it/s] 78%|#######8  | 36/46 [00:05<00:01,  6.54it/s] 80%|########  | 37/46 [00:05<00:01,  6.37it/s] 83%|########2 | 38/46 [00:05<00:01,  6.46it/s] 85%|########4 | 39/46 [00:05<00:01,  6.60it/s] 87%|########6 | 40/46 [00:05<00:00,  6.67it/s] 89%|########9 | 41/46 [00:05<00:00,  6.61it/s] 91%|#########1| 42/46 [00:06<00:00,  6.63it/s] 93%|#########3| 43/46 [00:06<00:00,  6.60it/s] 96%|#########5| 44/46 [00:06<00:00,  6.63it/s] 98%|#########7| 45/46 [00:06<00:00,  6.68it/s]100%|##########| 46/46 [00:06<00:00,  6.70it/s]100%|##########| 46/46 [00:06<00:00,  6.82it/s]

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

![](/imgs/5e438dff8e01f3e179447bd197cdf1f9.png)
``` python

# 绘制 SHAP 详细图
plt.figure(figsize = (11, 7))
shap.summary_plot(shap_values_aggregated, test_data_aggregated, feature_names = feature_names)
```

![](/imgs/da7003a9b0a59a71764f8428850d20dc.png)
<p>
调用 SHAP 解释函数，计算测试数据的 SHAP 值，并绘制图。
</p>
