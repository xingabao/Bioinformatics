---
title: 基于 Transformer 和 KAN 的时间序列预测模型 + SHAP 模型可解释性
date: 2025-05-07 08:27:30
tags: [Python, 机器学习, SHAP]
categories: [[案例分享, 机器学习, Transformer]]
---


<p>
时间序列预测是指基于历史数据预测未来值的过程，常见于金融（如股票价格预测）、天气预测等领域。传统方法包括
ARIMA、LSTM 等，而近年来 Transformer 因其强大的序列建模能力被广泛应用。
</p>
<p>
以下是基于 Transformer 和 KAN（Kolmogorov-Arnold
Network）的时间序列预测模型的详细解释，以及如何结合 SHAP（SHapley
Additive exPlanations）进行模型可解释性分析的内容。
</p>
<p>
<b>Transformer</b>
是一种基于注意力机制（Attention）的深度学习模型，最初用于自然语言处理任务（如机器翻译），包括编码器（Encoder）和解码器（Decoder），但在时间序列预测中通常只用编码器部分。关键组件包括多头注意力机制（Multi-Head
Attention）、前馈神经网络（Feed-Forward Neural
Network）、层归一化（Layer
Normalization）。其核心思想是通过注意力机制捕捉序列中不同位置之间的依赖关系，适用于时间序列预测，因为它能处理长距离依赖问题。
</p>
<p>
<b>KAN</b> 是一种新兴的神经网络结构，基于 Kolmogorov-Arnold
表示定理，理论上可以用较小的网络结构逼近任意连续函数。在时间序列预测中，KAN
可以作为非线性输出层，增强模型的表达能力。
</p>
<p>
<b>SHAP</b> (SHapley Additive
exPlanations)，是一种模型可解释性工具，基于博弈论中的 Shapley
值，用于量化每个特征对模型预测的贡献。
</p>

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
## Epoch  1 MSE:  0.2534732520580292 R2 -471.8250732421875
## Epoch  2 MSE:  0.1668313592672348 R2 -33.82643127441406
## Epoch  3 MSE:  0.0878085270524025 R2 -0.30102694034576416
## Epoch  4 MSE:  0.07575715333223343 R2 0.3833956718444824
## Epoch  5 MSE:  0.07859250158071518 R2 0.33137404918670654
## Epoch  6 MSE:  0.07542045414447784 R2 0.509617269039154
## Epoch  7 MSE:  0.07011733204126358 R2 0.6344890594482422
## Epoch  8 MSE:  0.05435286462306976 R2 0.6012566089630127
## Epoch  9 MSE:  0.05212904140353203 R2 0.4315526485443115
## Epoch  10 MSE:  0.09258008748292923 R2 0.0626031756401062
## Epoch  11 MSE:  0.07763289660215378 R2 -0.26717889308929443
## Epoch  12 MSE:  0.06637494266033173 R2 0.41642308235168457
## Epoch  13 MSE:  0.06139782816171646 R2 0.7198256254196167
## Epoch  14 MSE:  0.06764353066682816 R2 0.7355738878250122
## Epoch  15 MSE:  0.05358313396573067 R2 0.7311378717422485
## Epoch  16 MSE:  0.04331132769584656 R2 0.6838115453720093
## Epoch  17 MSE:  0.04264071583747864 R2 0.5376354455947876
## Epoch  18 MSE:  0.05008227378129959 R2 0.2926443815231323
## Epoch  19 MSE:  0.04249647259712219 R2 0.5228865742683411
## Epoch  20 MSE:  0.02876206859946251 R2 0.7658500671386719
## Epoch  21 MSE:  0.023902520537376404 R2 0.8628374934196472
## Epoch  22 MSE:  0.029222732409834862 R2 0.8650037050247192
## Epoch  23 MSE:  0.029559804126620293 R2 0.8808320760726929
## Epoch  24 MSE:  0.02456689067184925 R2 0.8950456380844116
## Epoch  25 MSE:  0.02231714501976967 R2 0.887019693851471
## Epoch  26 MSE:  0.021600019186735153 R2 0.8642966151237488
## Epoch  27 MSE:  0.020628048107028008 R2 0.8676158785820007
## Epoch  28 MSE:  0.02102423831820488 R2 0.8639687299728394
## Epoch  29 MSE:  0.02318122796714306 R2 0.8768531084060669
## Epoch  30 MSE:  0.020583178848028183 R2 0.8983039855957031
## Epoch  31 MSE:  0.01820877008140087 R2 0.9105284214019775
## Epoch  32 MSE:  0.020521488040685654 R2 0.9025799632072449
## Epoch  33 MSE:  0.018996352329850197 R2 0.9090265035629272
## Epoch  34 MSE:  0.01863311044871807 R2 0.8949334025382996
## Epoch  35 MSE:  0.016330722719430923 R2 0.9035411477088928
## Epoch  36 MSE:  0.01988879404962063 R2 0.8826991319656372
## Epoch  37 MSE:  0.019811738282442093 R2 0.8963184952735901
## Epoch  38 MSE:  0.016064802184700966 R2 0.9137381911277771
## Epoch  39 MSE:  0.017291486263275146 R2 0.9076024889945984
## Epoch  40 MSE:  0.016297100111842155 R2 0.9127324223518372
## Epoch  41 MSE:  0.013636012561619282 R2 0.9243469834327698
## Epoch  42 MSE:  0.017193172127008438 R2 0.9016556739807129
## Epoch  43 MSE:  0.013930974528193474 R2 0.9256582260131836
## Epoch  44 MSE:  0.015710987150669098 R2 0.9223694801330566
## Epoch  45 MSE:  0.015717405825853348 R2 0.9181807041168213
## Epoch  46 MSE:  0.01646864227950573 R2 0.9133821725845337
## Epoch  47 MSE:  0.016677148640155792 R2 0.9104026556015015
## Epoch  48 MSE:  0.016654308885335922 R2 0.9125961661338806
## Epoch  49 MSE:  0.01504489779472351 R2 0.9174494743347168
## Epoch  50 MSE:  0.015386948361992836 R2 0.9191188812255859
## Epoch  51 MSE:  0.013124234043061733 R2 0.9320604801177979
## Epoch  52 MSE:  0.012827855534851551 R2 0.9336647391319275
## Epoch  53 MSE:  0.013185215182602406 R2 0.9308905005455017
## Epoch  54 MSE:  0.013359582982957363 R2 0.9305475950241089
## Epoch  55 MSE:  0.014215911738574505 R2 0.9267830848693848
## Epoch  56 MSE:  0.012336105108261108 R2 0.9365473985671997
## Epoch  57 MSE:  0.013479623943567276 R2 0.9285085201263428
## Epoch  58 MSE:  0.014196997508406639 R2 0.9248655438423157
## Epoch  59 MSE:  0.012596558779478073 R2 0.9303416013717651
## Epoch  60 MSE:  0.011786160990595818 R2 0.9335236549377441
## Epoch  61 MSE:  0.012033735401928425 R2 0.934407651424408
## Epoch  62 MSE:  0.012221417389810085 R2 0.9395216703414917
## Epoch  63 MSE:  0.013094006106257439 R2 0.9290205836296082
## Epoch  64 MSE:  0.012437201105058193 R2 0.9362053871154785
## Epoch  65 MSE:  0.011427691206336021 R2 0.9391639232635498
## Epoch  66 MSE:  0.013520302250981331 R2 0.930102527141571
## Epoch  67 MSE:  0.01146305724978447 R2 0.9373044371604919
## Epoch  68 MSE:  0.011003624647855759 R2 0.9457361698150635
## Epoch  69 MSE:  0.012217302806675434 R2 0.9411989450454712
## Epoch  70 MSE:  0.01208106055855751 R2 0.9346969127655029
## Epoch  71 MSE:  0.011841426603496075 R2 0.9369077682495117
## Epoch  72 MSE:  0.01109832338988781 R2 0.9441723227500916
## Epoch  73 MSE:  0.011319847777485847 R2 0.9414624571800232
## Epoch  74 MSE:  0.009816925972700119 R2 0.9503077864646912
## Epoch  75 MSE:  0.0126412995159626 R2 0.9343464374542236
## Epoch  76 MSE:  0.010959724895656109 R2 0.9460115432739258
## Epoch  77 MSE:  0.013773021288216114 R2 0.9267539381980896
## Epoch  78 MSE:  0.011400336399674416 R2 0.9403164386749268
## Epoch  79 MSE:  0.011118900030851364 R2 0.9443305730819702
## Epoch  80 MSE:  0.010150941088795662 R2 0.9493393898010254
## Epoch  81 MSE:  0.010755038820207119 R2 0.9423977732658386
## Epoch  82 MSE:  0.010961504653096199 R2 0.9401872754096985
## Epoch  83 MSE:  0.011918892152607441 R2 0.9396145343780518
## Epoch  84 MSE:  0.010817240923643112 R2 0.9474574327468872
## Epoch  85 MSE:  0.01114434190094471 R2 0.941309928894043
## Epoch  86 MSE:  0.010893384926021099 R2 0.9393807053565979
## Epoch  87 MSE:  0.011898498050868511 R2 0.9381856918334961
## Epoch  88 MSE:  0.010246182791888714 R2 0.9482366442680359
## Epoch  89 MSE:  0.010482043027877808 R2 0.944943904876709
## Epoch  90 MSE:  0.011072216555476189 R2 0.9410621523857117
## Epoch  91 MSE:  0.011327333748340607 R2 0.9422070980072021
## Epoch  92 MSE:  0.009995638392865658 R2 0.9504632949829102
## Epoch  93 MSE:  0.00920066423714161 R2 0.9511006474494934
## Epoch  94 MSE:  0.009608440101146698 R2 0.9488811492919922
## Epoch  95 MSE:  0.010564160533249378 R2 0.9430632591247559
## Epoch  96 MSE:  0.00910515058785677 R2 0.9536101222038269
## Epoch  97 MSE:  0.009348835796117783 R2 0.9532167315483093
## Epoch  98 MSE:  0.010603196918964386 R2 0.9476780891418457
## Epoch  99 MSE:  0.009904292412102222 R2 0.9454771876335144
## Epoch  100 MSE:  0.009137008339166641 R2 0.9482314586639404
## Epoch  101 MSE:  0.010494035668671131 R2 0.9473217725753784
## Epoch  102 MSE:  0.011749385856091976 R2 0.9449583888053894
## Epoch  103 MSE:  0.0100778229534626 R2 0.94657301902771
## Epoch  104 MSE:  0.010181033983826637 R2 0.9475560188293457
## Epoch  105 MSE:  0.0095805823802948 R2 0.9494341611862183
## Epoch  106 MSE:  0.010941697284579277 R2 0.9464211463928223
## Epoch  107 MSE:  0.009568294510245323 R2 0.9513254761695862
## Epoch  108 MSE:  0.009573405608534813 R2 0.9504727721214294
## Epoch  109 MSE:  0.009274011477828026 R2 0.9500051140785217
## Epoch  110 MSE:  0.010410312563180923 R2 0.9459930062294006
## Epoch  111 MSE:  0.009834043681621552 R2 0.9497649669647217
## Epoch  112 MSE:  0.011148406192660332 R2 0.9427489638328552
## Epoch  113 MSE:  0.009367657825350761 R2 0.9536588788032532
## Epoch  114 MSE:  0.010902645997703075 R2 0.9476873278617859
## Epoch  115 MSE:  0.009864072315394878 R2 0.9446117877960205
## Epoch  116 MSE:  0.011270079761743546 R2 0.9303070306777954
## Epoch  117 MSE:  0.010049925185739994 R2 0.9492925405502319
## Epoch  118 MSE:  0.010705901309847832 R2 0.9512972831726074
## Epoch  119 MSE:  0.00943265575915575 R2 0.9538164734840393
## Epoch  120 MSE:  0.01056565623730421 R2 0.9444701075553894
## Epoch  121 MSE:  0.009429169818758965 R2 0.9474436044692993
## Epoch  122 MSE:  0.010300708934664726 R2 0.9434698224067688
## Epoch  123 MSE:  0.009549415670335293 R2 0.9508563876152039
## Epoch  124 MSE:  0.009696361608803272 R2 0.953175961971283
## Epoch  125 MSE:  0.00976331066340208 R2 0.9500521421432495
## Epoch  126 MSE:  0.009703857824206352 R2 0.9466226696968079
## Epoch  127 MSE:  0.008790210820734501 R2 0.9541271924972534
## Epoch  128 MSE:  0.008019505999982357 R2 0.9597827196121216
## Epoch  129 MSE:  0.00875046942383051 R2 0.9560425281524658
## Epoch  130 MSE:  0.00955173373222351 R2 0.9511801600456238
## Epoch  131 MSE:  0.00887457188218832 R2 0.950252890586853
## Epoch  132 MSE:  0.009711218066513538 R2 0.9495028257369995
## Epoch  133 MSE:  0.009793044999241829 R2 0.953754186630249
## Epoch  134 MSE:  0.009644728153944016 R2 0.9522054195404053
## Epoch  135 MSE:  0.009192757308483124 R2 0.9489296674728394
## Epoch  136 MSE:  0.010543744079768658 R2 0.9421007037162781
## Epoch  137 MSE:  0.010072208009660244 R2 0.9511125087738037
## Epoch  138 MSE:  0.009237739257514477 R2 0.9556540250778198
## Epoch  139 MSE:  0.007838279008865356 R2 0.9587422013282776
## Epoch  140 MSE:  0.00997531320899725 R2 0.9486216902732849
## Epoch  141 MSE:  0.007739574182778597 R2 0.9608839154243469
## Epoch  142 MSE:  0.008677362464368343 R2 0.9551751017570496
## Epoch  143 MSE:  0.00938377995043993 R2 0.9512699246406555
## Epoch  144 MSE:  0.010342868976294994 R2 0.9465201497077942
## Epoch  145 MSE:  0.008446292020380497 R2 0.9552205204963684
## Epoch  146 MSE:  0.008468082174658775 R2 0.9550501108169556
## Epoch  147 MSE:  0.009271743707358837 R2 0.9540414214134216
## Epoch  148 MSE:  0.007812335155904293 R2 0.9607552289962769
## Epoch  149 MSE:  0.00986523274332285 R2 0.9480332732200623
## Epoch  150 MSE:  0.008462023921310902 R2 0.9558300971984863
## Epoch  151 MSE:  0.007392578292638063 R2 0.9622674584388733
## Epoch  152 MSE:  0.007327120751142502 R2 0.9633309245109558
## Epoch  153 MSE:  0.010081881657242775 R2 0.9514141082763672
## Epoch  154 MSE:  0.007448465563356876 R2 0.960807204246521
## Epoch  155 MSE:  0.008892059326171875 R2 0.9520787000656128
## Epoch  156 MSE:  0.007791969925165176 R2 0.9616129398345947
## Epoch  157 MSE:  0.008075644262135029 R2 0.9622374176979065
## Epoch  158 MSE:  0.008860220201313496 R2 0.9516741037368774
## Epoch  159 MSE:  0.008123785257339478 R2 0.9550145268440247
## Epoch  160 MSE:  0.0066077737137675285 R2 0.9680257439613342
## Epoch  161 MSE:  0.0076627726666629314 R2 0.9621256589889526
## Epoch  162 MSE:  0.007752333767712116 R2 0.9600989818572998
## Epoch  163 MSE:  0.008637238293886185 R2 0.9551494717597961
## Epoch  164 MSE:  0.008350824005901814 R2 0.9582706689834595
## Epoch  165 MSE:  0.009142833761870861 R2 0.954505443572998
## Epoch  166 MSE:  0.0095977783203125 R2 0.9503512382507324
## Epoch  167 MSE:  0.00851521734148264 R2 0.9545641541481018
## Epoch  168 MSE:  0.007227341178804636 R2 0.9638073444366455
## Epoch  169 MSE:  0.008193510584533215 R2 0.9614974856376648
## Epoch  170 MSE:  0.007496950216591358 R2 0.9587215185165405
## Epoch  171 MSE:  0.0077207996509969234 R2 0.9582918882369995
## Epoch  172 MSE:  0.007721614558249712 R2 0.9619179368019104
## Epoch  173 MSE:  0.008746491745114326 R2 0.957348644733429
## Epoch  174 MSE:  0.007308714557439089 R2 0.9614310264587402
## Epoch  175 MSE:  0.007730916142463684 R2 0.9577051997184753
## Epoch  176 MSE:  0.008248593658208847 R2 0.9607486128807068
## Epoch  177 MSE:  0.0075026629492640495 R2 0.963122546672821
## Epoch  178 MSE:  0.006710813380777836 R2 0.9635106921195984
## Epoch  179 MSE:  0.008089091628789902 R2 0.954879879951477
## Epoch  180 MSE:  0.007934331893920898 R2 0.9637376070022583
## Epoch  181 MSE:  0.007672728970646858 R2 0.9649863243103027
## Epoch  182 MSE:  0.010104522109031677 R2 0.9359973669052124
## Epoch  183 MSE:  0.008162758313119411 R2 0.9584996104240417
## Epoch  184 MSE:  0.007695219945162535 R2 0.9648655652999878
## Epoch  185 MSE:  0.006981140002608299 R2 0.9646787047386169
## Epoch  186 MSE:  0.0072520277462899685 R2 0.957394003868103
## Epoch  187 MSE:  0.007335434667766094 R2 0.9626728296279907
## Epoch  188 MSE:  0.008078549057245255 R2 0.9626415967941284
## Epoch  189 MSE:  0.0067289723083376884 R2 0.9651387333869934
## Epoch  190 MSE:  0.0071083311922848225 R2 0.9598923921585083
## Epoch  191 MSE:  0.007669317536056042 R2 0.9617222547531128
## Epoch  192 MSE:  0.008615112863481045 R2 0.958169162273407
## Epoch  193 MSE:  0.006361066829413176 R2 0.967210590839386
## Epoch  194 MSE:  0.006892209406942129 R2 0.9642789959907532
## Epoch  195 MSE:  0.007057217415422201 R2 0.9629371762275696
## Epoch  196 MSE:  0.008126940578222275 R2 0.9619240760803223
## Epoch  197 MSE:  0.007130241487175226 R2 0.9634702205657959
## Epoch  198 MSE:  0.006516146939247847 R2 0.9646458625793457
## Epoch  199 MSE:  0.006300664506852627 R2 0.967575192451477
## Epoch  200 MSE:  0.005974531173706055 R2 0.970913827419281
## Epoch  201 MSE:  0.005898010917007923 R2 0.9719399213790894
## Epoch  202 MSE:  0.006594320293515921 R2 0.9640745520591736
## Epoch  203 MSE:  0.0071221389807760715 R2 0.9627928137779236
## Epoch  204 MSE:  0.006670053116977215 R2 0.9679027795791626
## Epoch  205 MSE:  0.0065369210205972195 R2 0.968093991279602
## Epoch  206 MSE:  0.008040967397391796 R2 0.9531829953193665
## Epoch  207 MSE:  0.007492115255445242 R2 0.9624799489974976
## Epoch  208 MSE:  0.0070355054922401905 R2 0.9675500392913818
## Epoch  209 MSE:  0.007728135213255882 R2 0.9617397785186768
## Epoch  210 MSE:  0.006674226373434067 R2 0.9611905813217163
## Epoch  211 MSE:  0.0075251199305057526 R2 0.96036696434021
## Epoch  212 MSE:  0.008181837387382984 R2 0.9607510566711426
## Epoch  213 MSE:  0.00678792130202055 R2 0.9682092070579529
## Epoch  214 MSE:  0.0062492117285728455 R2 0.9672576785087585
## Epoch  215 MSE:  0.0070623308420181274 R2 0.9591363072395325
## Epoch  216 MSE:  0.006537482142448425 R2 0.9686934947967529
## Epoch  217 MSE:  0.006276443600654602 R2 0.9705959558486938
## Epoch  218 MSE:  0.005660415161401033 R2 0.9711925983428955
## Epoch  219 MSE:  0.007159233093261719 R2 0.9612800478935242
## Epoch  220 MSE:  0.0056326608173549175 R2 0.972314178943634
## Epoch  221 MSE:  0.008215436711907387 R2 0.9605125784873962
## Epoch  222 MSE:  0.005672534462064505 R2 0.9698215126991272
## Epoch  223 MSE:  0.0072356644086539745 R2 0.9607755541801453
## Epoch  224 MSE:  0.007422951515763998 R2 0.9626718163490295
## Epoch  225 MSE:  0.006514041684567928 R2 0.9700199365615845
## Epoch  226 MSE:  0.006257223896682262 R2 0.9675881862640381
## Epoch  227 MSE:  0.007253670133650303 R2 0.9590235352516174
## Epoch  228 MSE:  0.006671955343335867 R2 0.9655918478965759
## Epoch  229 MSE:  0.006116991862654686 R2 0.9702662825584412
## Epoch  230 MSE:  0.005888959858566523 R2 0.9710013270378113
## Epoch  231 MSE:  0.0053434851579368114 R2 0.9720838665962219
## Epoch  232 MSE:  0.0058745830319821835 R2 0.9695208072662354
## Epoch  233 MSE:  0.006677330005913973 R2 0.9680531024932861
## Epoch  234 MSE:  0.005740263033658266 R2 0.9702969193458557
## Epoch  235 MSE:  0.005975949577987194 R2 0.9704180955886841
## Epoch  236 MSE:  0.005419286899268627 R2 0.9736351370811462
## Epoch  237 MSE:  0.0051394919864833355 R2 0.9740492701530457
## Epoch  238 MSE:  0.00662276754155755 R2 0.9643526673316956
## Epoch  239 MSE:  0.005940120667219162 R2 0.9720928072929382
## Epoch  240 MSE:  0.006082853768020868 R2 0.9710600972175598
## Epoch  241 MSE:  0.006576190702617168 R2 0.9645625352859497
## Epoch  242 MSE:  0.005714558996260166 R2 0.9685084223747253
## Epoch  243 MSE:  0.005458399187773466 R2 0.9728614091873169
## Epoch  244 MSE:  0.0055341534316539764 R2 0.9731367826461792
## Epoch  245 MSE:  0.006101683247834444 R2 0.9692975282669067
## Epoch  246 MSE:  0.005448825191706419 R2 0.9713589549064636
## Epoch  247 MSE:  0.0060656871646642685 R2 0.9680804014205933
## Epoch  248 MSE:  0.005952788982540369 R2 0.9719140529632568
## Epoch  249 MSE:  0.0059254225343465805 R2 0.9703701138496399
## Epoch  250 MSE:  0.005546108353883028 R2 0.9720331430435181
## Epoch  251 MSE:  0.005714273080229759 R2 0.9705783724784851
## Epoch  252 MSE:  0.005989579949527979 R2 0.9685412645339966
## Epoch  253 MSE:  0.0051178280264139175 R2 0.9750781059265137
## Epoch  254 MSE:  0.005292404443025589 R2 0.9734353423118591
## Epoch  255 MSE:  0.005918257869780064 R2 0.9707419872283936
## Epoch  256 MSE:  0.006343513261526823 R2 0.9659266471862793
## Epoch  257 MSE:  0.00540657015517354 R2 0.9722336530685425
## Epoch  258 MSE:  0.005896111950278282 R2 0.9727492928504944
## Epoch  259 MSE:  0.005143587943166494 R2 0.9741325378417969
## Epoch  260 MSE:  0.006720585282891989 R2 0.9625327587127686
## Epoch  261 MSE:  0.00551197212189436 R2 0.9719552993774414
## Epoch  262 MSE:  0.004950292874127626 R2 0.9758775234222412
## Epoch  263 MSE:  0.004967639222741127 R2 0.9764528870582581
## Epoch  264 MSE:  0.005225048866122961 R2 0.9742004871368408
## Epoch  265 MSE:  0.0058432770892977715 R2 0.9682109355926514
## Epoch  266 MSE:  0.005777351092547178 R2 0.9712411761283875
## Epoch  267 MSE:  0.00480499304831028 R2 0.9764816164970398
## Epoch  268 MSE:  0.0058189881965518 R2 0.9701387882232666
## Epoch  269 MSE:  0.006018139887601137 R2 0.9702509641647339
## Epoch  270 MSE:  0.005513140466064215 R2 0.9716259837150574
## Epoch  271 MSE:  0.005333675071597099 R2 0.9714084267616272
## Epoch  272 MSE:  0.004741562530398369 R2 0.9766141176223755
## Epoch  273 MSE:  0.005834192503243685 R2 0.9723095297813416
## Epoch  274 MSE:  0.005673604551702738 R2 0.9705700278282166
## Epoch  275 MSE:  0.006197819951921701 R2 0.9668320417404175
## Epoch  276 MSE:  0.006081063766032457 R2 0.9689193964004517
## Epoch  277 MSE:  0.00556461326777935 R2 0.9731989502906799
## Epoch  278 MSE:  0.0045838975347578526 R2 0.9778389930725098
## Epoch  279 MSE:  0.005211235489696264 R2 0.9736252427101135
## Epoch  280 MSE:  0.00524947652593255 R2 0.9724346995353699
## Epoch  281 MSE:  0.00530390115454793 R2 0.9724984169006348
## Epoch  282 MSE:  0.0053209057077765465 R2 0.9742041230201721
## Epoch  283 MSE:  0.004938586615025997 R2 0.9760962724685669
## Epoch  284 MSE:  0.005471555981785059 R2 0.9723605513572693
## Epoch  285 MSE:  0.005935856141149998 R2 0.9676690697669983
## Epoch  286 MSE:  0.005724429618567228 R2 0.9717093706130981
## Epoch  287 MSE:  0.005768186412751675 R2 0.9718997478485107
## Epoch  288 MSE:  0.005588470492511988 R2 0.9713769555091858
## Epoch  289 MSE:  0.005520069506019354 R2 0.9721653461456299
## Epoch  290 MSE:  0.005756934639066458 R2 0.9706947207450867
## Epoch  291 MSE:  0.005612320266664028 R2 0.9703035354614258
## Epoch  292 MSE:  0.005594941787421703 R2 0.9730437994003296
## Epoch  293 MSE:  0.0057985736057162285 R2 0.9713171124458313
## Epoch  294 MSE:  0.005252736154943705 R2 0.9722656011581421
## Epoch  295 MSE:  0.005076351575553417 R2 0.9746055603027344
## Epoch  296 MSE:  0.006129700690507889 R2 0.9704755544662476
## Epoch  297 MSE:  0.0045348163694143295 R2 0.9764673709869385
## Epoch  298 MSE:  0.005580593831837177 R2 0.9709889888763428
## Epoch  299 MSE:  0.004618881735950708 R2 0.9770481586456299
## Epoch  300 MSE:  0.00563901849091053 R2 0.9732192158699036
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
## Test Score: 0.10 RMSE
print('Test R^2: %.2f' % (r2_test))
## Test R^2: 0.74
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
##   0%|          | 0/46 [00:00<?, ?it/s]  4%|4         | 2/46 [00:00<00:04, 10.47it/s]  9%|8         | 4/46 [00:00<00:05,  7.34it/s] 11%|#         | 5/46 [00:00<00:05,  6.91it/s] 13%|#3        | 6/46 [00:00<00:05,  6.75it/s] 15%|#5        | 7/46 [00:00<00:05,  6.82it/s] 17%|#7        | 8/46 [00:01<00:05,  6.91it/s] 20%|#9        | 9/46 [00:01<00:05,  6.95it/s] 22%|##1       | 10/46 [00:01<00:05,  6.96it/s] 24%|##3       | 11/46 [00:01<00:05,  6.96it/s] 26%|##6       | 12/46 [00:01<00:04,  6.99it/s] 28%|##8       | 13/46 [00:01<00:04,  6.94it/s] 30%|###       | 14/46 [00:01<00:04,  6.91it/s] 33%|###2      | 15/46 [00:02<00:04,  6.98it/s] 35%|###4      | 16/46 [00:02<00:04,  7.02it/s] 37%|###6      | 17/46 [00:02<00:04,  7.01it/s] 39%|###9      | 18/46 [00:02<00:04,  6.97it/s] 41%|####1     | 19/46 [00:02<00:03,  6.98it/s] 43%|####3     | 20/46 [00:02<00:03,  6.95it/s] 46%|####5     | 21/46 [00:02<00:03,  6.95it/s] 48%|####7     | 22/46 [00:03<00:03,  6.98it/s] 50%|#####     | 23/46 [00:03<00:03,  6.90it/s] 52%|#####2    | 24/46 [00:03<00:03,  6.93it/s] 54%|#####4    | 25/46 [00:03<00:03,  6.95it/s] 57%|#####6    | 26/46 [00:03<00:02,  6.93it/s] 59%|#####8    | 27/46 [00:03<00:02,  6.97it/s] 61%|######    | 28/46 [00:03<00:02,  7.04it/s] 63%|######3   | 29/46 [00:04<00:02,  7.01it/s] 65%|######5   | 30/46 [00:04<00:02,  6.90it/s] 67%|######7   | 31/46 [00:04<00:02,  6.92it/s] 70%|######9   | 32/46 [00:04<00:02,  6.97it/s] 72%|#######1  | 33/46 [00:04<00:01,  6.96it/s] 74%|#######3  | 34/46 [00:04<00:01,  6.92it/s] 76%|#######6  | 35/46 [00:05<00:01,  6.94it/s] 78%|#######8  | 36/46 [00:05<00:01,  6.92it/s] 80%|########  | 37/46 [00:05<00:01,  6.97it/s] 83%|########2 | 38/46 [00:05<00:01,  7.03it/s] 85%|########4 | 39/46 [00:05<00:00,  7.02it/s] 87%|########6 | 40/46 [00:05<00:00,  6.94it/s] 89%|########9 | 41/46 [00:05<00:00,  6.93it/s] 91%|#########1| 42/46 [00:06<00:00,  6.86it/s] 93%|#########3| 43/46 [00:06<00:00,  6.80it/s] 96%|#########5| 44/46 [00:06<00:00,  6.77it/s] 98%|#########7| 45/46 [00:06<00:00,  6.79it/s]100%|##########| 46/46 [00:06<00:00,  6.83it/s]100%|##########| 46/46 [00:06<00:00,  6.97it/s]

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

# 代码简洁版

``` python
# 导入所需库
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
wkdir = 'E:/src'
sys.path.append(f'{wkdir}/modules')
from fftKAN import *
from effKAN import *

# 数据预处理函数
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

# 模型定义
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

# SHAP 解释函数
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

if __name__ == '__main__':
    
    # 参数设置
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
    args.path = f'{wkdir}/data/rlData.csv'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    data = pd.read_csv(args.path)
    data = data.sort_values('Date')
    
    # 可视化数据，绘制收盘价随时间变化的折线图
    if args.vision:
        sns.set_style("darkgrid")
        plt.figure(figsize = (11, 7))
        plt.plot(data[['Close']])
        plt.xticks(range(0, data.shape[0], 20), data['Date'].loc[::20], rotation = 45)
        plt.title("Stock Price", fontsize = 18, fontweight = 'bold')
        plt.xlabel('Date', fontsize = 18)
        plt.ylabel('Close Price (USD)', fontsize = 18)
        plt.show()
        
    # 数据标准化
    features = data[args.input_features]
    scaler = MinMaxScaler(feature_range = (-1, 1))
    features_scaled = scaler.fit_transform(features)
    
    target_scaler = MinMaxScaler(feature_range = (-1, 1))
    target = data[args.output_features]
    target_scaled = target_scaler.fit_transform(target)
    
    # 划分数据集
    x_train, y_train, x_test, y_test = split_data(features_scaled, target_scaled, args.window_size)
    
    # 转换为张量
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    
    # 模型初始化和训练 
    model = TimeSeriesTransformer_ekan(
        input_dim = len(args.input_features),
        num_heads = args.num_heads,
        num_layers = args.n_layers,
        num_outputs = len(args.output_features),
        hidden_space = args.hidden_space,
        dropout_rate = args.dropout
    )
    
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    # 训练模
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
        
    # 结果评估
    y_test_pred = model(x_test)
    trainScore = mean_squared_error(y_train.detach().numpy(), y_train_pred.detach().numpy())
    r2_train = r2_score(y_train.detach().numpy(), y_train_pred.detach().numpy())
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Train R^2: %.2f' % (r2_train))

    testScore = math.sqrt(mean_squared_error(y_test.detach().numpy(), y_test_pred.detach().numpy()))
    r2_test = r2_score(y_test.detach().numpy(), y_test_pred.detach().numpy())
    print('Test Score: %.2f RMSE' % (testScore))
    print('Test R^2: %.2f' % (r2_test))
    
    # SHAP 解释部分
    print("Starting SHAP explanation...")
    shap_values, explainer, test_data_flattened = explain_model_with_shap(
        model, 
        x_test.numpy(), 
        background_samples = 50, 
        seq_len = x_test.shape[1], 
        input_dim = x_test.shape[2]
    )
    
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

    plt.figure(figsize = (11, 7))
    shap.summary_plot(shap_values_aggregated, test_data_aggregated, feature_names = feature_names)
```
