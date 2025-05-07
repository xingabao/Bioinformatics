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
## Epoch  1 MSE:  0.19178439676761627 R2 -97.6021499633789
## Epoch  2 MSE:  0.11515504866838455 R2 -4.259119033813477
## Epoch  3 MSE:  0.10730896145105362 R2 0.5661811828613281
## Epoch  4 MSE:  0.12128999829292297 R2 0.45202988386154175
## Epoch  5 MSE:  0.08234210312366486 R2 0.1890367865562439
## Epoch  6 MSE:  0.11613549292087555 R2 -1.5446171760559082
## Epoch  7 MSE:  0.09201673418283463 R2 -0.8488613367080688
## Epoch  8 MSE:  0.09303460270166397 R2 -0.7730255126953125
## Epoch  9 MSE:  0.08826945722103119 R2 -0.43811941146850586
## Epoch  10 MSE:  0.0826805904507637 R2 -0.01560509204864502
## Epoch  11 MSE:  0.07445799559354782 R2 0.30534589290618896
## Epoch  12 MSE:  0.07272742688655853 R2 0.5112962126731873
## Epoch  13 MSE:  0.07502595335245132 R2 0.5616952776908875
## Epoch  14 MSE:  0.07915326207876205 R2 0.5954674482345581
## Epoch  15 MSE:  0.07735110074281693 R2 0.5956048965454102
## Epoch  16 MSE:  0.07923386991024017 R2 0.579035758972168
## Epoch  17 MSE:  0.0737772136926651 R2 0.5429388880729675
## Epoch  18 MSE:  0.06966660916805267 R2 0.4993346333503723
## Epoch  19 MSE:  0.06978463381528854 R2 0.4385448694229126
## Epoch  20 MSE:  0.07087114453315735 R2 0.3469070792198181
## Epoch  21 MSE:  0.0712207704782486 R2 0.309411883354187
## Epoch  22 MSE:  0.0689103826880455 R2 0.27520591020584106
## Epoch  23 MSE:  0.06583337485790253 R2 0.2900269031524658
## Epoch  24 MSE:  0.058043573051691055 R2 0.4121251702308655
## Epoch  25 MSE:  0.05837249383330345 R2 0.41844213008880615
## Epoch  26 MSE:  0.051497653126716614 R2 0.5678671598434448
## Epoch  27 MSE:  0.04302489385008812 R2 0.678180456161499
## Epoch  28 MSE:  0.03679583594202995 R2 0.7578483819961548
## Epoch  29 MSE:  0.0390644297003746 R2 0.7635668516159058
## Epoch  30 MSE:  0.03235171362757683 R2 0.8363774418830872
## Epoch  31 MSE:  0.028603410348296165 R2 0.8638669848442078
## Epoch  32 MSE:  0.0269683375954628 R2 0.8727587461471558
## Epoch  33 MSE:  0.02322244830429554 R2 0.8821458220481873
## Epoch  34 MSE:  0.026886489242315292 R2 0.8225810527801514
## Epoch  35 MSE:  0.02455974742770195 R2 0.8529604077339172
## Epoch  36 MSE:  0.028817811980843544 R2 0.8583551645278931
## Epoch  37 MSE:  0.02652980014681816 R2 0.8826791644096375
## Epoch  38 MSE:  0.025522224605083466 R2 0.8849178552627563
## Epoch  39 MSE:  0.021615559235215187 R2 0.8953350186347961
## Epoch  40 MSE:  0.019921625033020973 R2 0.8928285241127014
## Epoch  41 MSE:  0.020783282816410065 R2 0.8784900903701782
## Epoch  42 MSE:  0.020824924111366272 R2 0.8749976754188538
## Epoch  43 MSE:  0.019794750958681107 R2 0.8840230107307434
## Epoch  44 MSE:  0.017945921048521996 R2 0.898942232131958
## Epoch  45 MSE:  0.01860768534243107 R2 0.8972787857055664
## Epoch  46 MSE:  0.01851726695895195 R2 0.8966484665870667
## Epoch  47 MSE:  0.016049236059188843 R2 0.9111850261688232
## Epoch  48 MSE:  0.017436383292078972 R2 0.9044294357299805
## Epoch  49 MSE:  0.016457797959446907 R2 0.9110966920852661
## Epoch  50 MSE:  0.015587953850626945 R2 0.9159865379333496
## Epoch  51 MSE:  0.016235381364822388 R2 0.9100028276443481
## Epoch  52 MSE:  0.015225127339363098 R2 0.916687548160553
## Epoch  53 MSE:  0.01422305777668953 R2 0.9263167381286621
## Epoch  54 MSE:  0.015904393047094345 R2 0.9189725518226624
## Epoch  55 MSE:  0.014586790464818478 R2 0.9221493005752563
## Epoch  56 MSE:  0.015185055322945118 R2 0.9179830551147461
## Epoch  57 MSE:  0.01355196163058281 R2 0.9292134046554565
## Epoch  58 MSE:  0.015717433765530586 R2 0.9206165075302124
## Epoch  59 MSE:  0.012122849933803082 R2 0.933146059513092
## Epoch  60 MSE:  0.013825392350554466 R2 0.9204477071762085
## Epoch  61 MSE:  0.012928365729749203 R2 0.9257997274398804
## Epoch  62 MSE:  0.014591323211789131 R2 0.9274843335151672
## Epoch  63 MSE:  0.013292551040649414 R2 0.9348023533821106
## Epoch  64 MSE:  0.013512367382645607 R2 0.9319292306900024
## Epoch  65 MSE:  0.011580933816730976 R2 0.9387495517730713
## Epoch  66 MSE:  0.012459618039429188 R2 0.9328042268753052
## Epoch  67 MSE:  0.013030794449150562 R2 0.9302496910095215
## Epoch  68 MSE:  0.014300240203738213 R2 0.9224230051040649
## Epoch  69 MSE:  0.011774887330830097 R2 0.9443750977516174
## Epoch  70 MSE:  0.014734257943928242 R2 0.9332401156425476
## Epoch  71 MSE:  0.013253651559352875 R2 0.9271889925003052
## Epoch  72 MSE:  0.012673098593950272 R2 0.9272987842559814
## Epoch  73 MSE:  0.011663287878036499 R2 0.9356896281242371
## Epoch  74 MSE:  0.012995307333767414 R2 0.9317533373832703
## Epoch  75 MSE:  0.011102271266281605 R2 0.9463801980018616
## Epoch  76 MSE:  0.013528362847864628 R2 0.9303154945373535
## Epoch  77 MSE:  0.011956882663071156 R2 0.9384117722511292
## Epoch  78 MSE:  0.01295546069741249 R2 0.9338594079017639
## Epoch  79 MSE:  0.013340983539819717 R2 0.9305988550186157
## Epoch  80 MSE:  0.012283382937312126 R2 0.9330556392669678
## Epoch  81 MSE:  0.012401635758578777 R2 0.9354619383811951
## Epoch  82 MSE:  0.010316195897758007 R2 0.9486016035079956
## Epoch  83 MSE:  0.010807721875607967 R2 0.9455956220626831
## Epoch  84 MSE:  0.010959566570818424 R2 0.9419273734092712
## Epoch  85 MSE:  0.010682275518774986 R2 0.9418985843658447
## Epoch  86 MSE:  0.01051393523812294 R2 0.9454401731491089
## Epoch  87 MSE:  0.011032268404960632 R2 0.9446873664855957
## Epoch  88 MSE:  0.010321903973817825 R2 0.9471704363822937
## Epoch  89 MSE:  0.009893239475786686 R2 0.9512066841125488
## Epoch  90 MSE:  0.010933170095086098 R2 0.9423196911811829
## Epoch  91 MSE:  0.011105647310614586 R2 0.9404904842376709
## Epoch  92 MSE:  0.010918023996055126 R2 0.9412364363670349
## Epoch  93 MSE:  0.010198162868618965 R2 0.9473950862884521
## Epoch  94 MSE:  0.011594627052545547 R2 0.940468966960907
## Epoch  95 MSE:  0.00991944968700409 R2 0.9484924674034119
## Epoch  96 MSE:  0.010319136083126068 R2 0.9475435018539429
## Epoch  97 MSE:  0.009975352324545383 R2 0.9476333260536194
## Epoch  98 MSE:  0.009895812720060349 R2 0.9457210898399353
## Epoch  99 MSE:  0.009843782521784306 R2 0.949234127998352
## Epoch  100 MSE:  0.011196908541023731 R2 0.9466854929924011
## Epoch  101 MSE:  0.010244348086416721 R2 0.9496625065803528
## Epoch  102 MSE:  0.01078728586435318 R2 0.9409816861152649
## Epoch  103 MSE:  0.010301274247467518 R2 0.9423493146896362
## Epoch  104 MSE:  0.010046308860182762 R2 0.9469430446624756
## Epoch  105 MSE:  0.009839901700615883 R2 0.9544753432273865
## Epoch  106 MSE:  0.00929940678179264 R2 0.9551283121109009
## Epoch  107 MSE:  0.009538104757666588 R2 0.9492160677909851
## Epoch  108 MSE:  0.011033694259822369 R2 0.937552273273468
## Epoch  109 MSE:  0.010660802945494652 R2 0.9437330365180969
## Epoch  110 MSE:  0.00970819965004921 R2 0.9542999267578125
## Epoch  111 MSE:  0.008565757423639297 R2 0.9580900073051453
## Epoch  112 MSE:  0.010481173172593117 R2 0.9402580261230469
## Epoch  113 MSE:  0.008635572157800198 R2 0.9508949518203735
## Epoch  114 MSE:  0.010567103512585163 R2 0.9491052031517029
## Epoch  115 MSE:  0.01003353577107191 R2 0.9526596069335938
## Epoch  116 MSE:  0.009212781675159931 R2 0.9516239166259766
## Epoch  117 MSE:  0.009458602406084538 R2 0.9476789832115173
## Epoch  118 MSE:  0.007542564067989588 R2 0.9616516828536987
## Epoch  119 MSE:  0.009635122492909431 R2 0.9543884992599487
## Epoch  120 MSE:  0.008313117548823357 R2 0.9569514393806458
## Epoch  121 MSE:  0.009641324169933796 R2 0.9446430206298828
## Epoch  122 MSE:  0.008440828882157803 R2 0.9563095569610596
## Epoch  123 MSE:  0.009154248051345348 R2 0.9575388431549072
## Epoch  124 MSE:  0.008368334732949734 R2 0.958701491355896
## Epoch  125 MSE:  0.009418881498277187 R2 0.9439796209335327
## Epoch  126 MSE:  0.007991752587258816 R2 0.9593879580497742
## Epoch  127 MSE:  0.00799400731921196 R2 0.9630898833274841
## Epoch  128 MSE:  0.008032913319766521 R2 0.95933997631073
## Epoch  129 MSE:  0.008782845921814442 R2 0.9527826905250549
## Epoch  130 MSE:  0.010227333754301071 R2 0.9383296966552734
## Epoch  131 MSE:  0.009973082691431046 R2 0.9520425796508789
## Epoch  132 MSE:  0.00941250566393137 R2 0.9563052654266357
## Epoch  133 MSE:  0.008959271013736725 R2 0.9562976956367493
## Epoch  134 MSE:  0.010199476033449173 R2 0.9395914673805237
## Epoch  135 MSE:  0.009774817153811455 R2 0.9459201097488403
## Epoch  136 MSE:  0.009351782500743866 R2 0.9567587375640869
## Epoch  137 MSE:  0.00793848279863596 R2 0.9633986949920654
## Epoch  138 MSE:  0.007497046608477831 R2 0.9604674577713013
## Epoch  139 MSE:  0.010137762874364853 R2 0.939140796661377
## Epoch  140 MSE:  0.007527503184974194 R2 0.9582058787345886
## Epoch  141 MSE:  0.007786872796714306 R2 0.9621811509132385
## Epoch  142 MSE:  0.00905624683946371 R2 0.9584528803825378
## Epoch  143 MSE:  0.007295875810086727 R2 0.9642433524131775
## Epoch  144 MSE:  0.008860515430569649 R2 0.952159583568573
## Epoch  145 MSE:  0.007905327714979649 R2 0.9539268016815186
## Epoch  146 MSE:  0.007926681078970432 R2 0.9605296850204468
## Epoch  147 MSE:  0.008663184940814972 R2 0.9609581232070923
## Epoch  148 MSE:  0.007662136573344469 R2 0.9617140293121338
## Epoch  149 MSE:  0.007597471587359905 R2 0.9575866460800171
## Epoch  150 MSE:  0.00863315723836422 R2 0.9512831568717957
## Epoch  151 MSE:  0.007370935287326574 R2 0.9639314413070679
## Epoch  152 MSE:  0.007238175719976425 R2 0.9646619558334351
## Epoch  153 MSE:  0.006634398363530636 R2 0.9671022295951843
## Epoch  154 MSE:  0.0082968445494771 R2 0.960102379322052
## Epoch  155 MSE:  0.0077222418040037155 R2 0.960525393486023
## Epoch  156 MSE:  0.0098393764346838 R2 0.9459074139595032
## Epoch  157 MSE:  0.006695483345538378 R2 0.9645319581031799
## Epoch  158 MSE:  0.007374966517090797 R2 0.9628350734710693
## Epoch  159 MSE:  0.011077425442636013 R2 0.9475194215774536
## Epoch  160 MSE:  0.00668466929346323 R2 0.9660290479660034
## Epoch  161 MSE:  0.006807106081396341 R2 0.9622051119804382
## Epoch  162 MSE:  0.007412811741232872 R2 0.9592909216880798
## Epoch  163 MSE:  0.007954650558531284 R2 0.9596652388572693
## Epoch  164 MSE:  0.006525097880512476 R2 0.967961847782135
## Epoch  165 MSE:  0.006629915442317724 R2 0.9687573909759521
## Epoch  166 MSE:  0.00656061340123415 R2 0.9659274220466614
## Epoch  167 MSE:  0.007534875068813562 R2 0.9579246640205383
## Epoch  168 MSE:  0.007938968949019909 R2 0.9595691561698914
## Epoch  169 MSE:  0.007895450107753277 R2 0.9618131518363953
## Epoch  170 MSE:  0.008361906744539738 R2 0.9610474705696106
## Epoch  171 MSE:  0.007483711000531912 R2 0.9622556567192078
## Epoch  172 MSE:  0.007432035636156797 R2 0.9599983096122742
## Epoch  173 MSE:  0.006882436107844114 R2 0.962751030921936
## Epoch  174 MSE:  0.007949475198984146 R2 0.9591315388679504
## Epoch  175 MSE:  0.0073630004189908504 R2 0.9635286927223206
## Epoch  176 MSE:  0.007726449053734541 R2 0.9617764353752136
## Epoch  177 MSE:  0.0069922492839396 R2 0.9651075005531311
## Epoch  178 MSE:  0.006671065464615822 R2 0.964409589767456
## Epoch  179 MSE:  0.007251058705151081 R2 0.9613932371139526
## Epoch  180 MSE:  0.006724624428898096 R2 0.9664217829704285
## Epoch  181 MSE:  0.0072695668786764145 R2 0.9630431532859802
## Epoch  182 MSE:  0.006354009732604027 R2 0.9676737189292908
## Epoch  183 MSE:  0.006880416534841061 R2 0.9660972952842712
## Epoch  184 MSE:  0.006916868966072798 R2 0.9627417325973511
## Epoch  185 MSE:  0.007226546760648489 R2 0.9637784957885742
## Epoch  186 MSE:  0.006891767960041761 R2 0.9653438329696655
## Epoch  187 MSE:  0.00598677434027195 R2 0.9694749712944031
## Epoch  188 MSE:  0.00597013346850872 R2 0.9701234102249146
## Epoch  189 MSE:  0.00666645634919405 R2 0.9662463068962097
## Epoch  190 MSE:  0.006168515421450138 R2 0.9692153334617615
## Epoch  191 MSE:  0.006771284621208906 R2 0.9653322696685791
## Epoch  192 MSE:  0.005957279819995165 R2 0.9687253832817078
## Epoch  193 MSE:  0.006522681564092636 R2 0.9668076634407043
## Epoch  194 MSE:  0.006280134432017803 R2 0.9674874544143677
## Epoch  195 MSE:  0.00638793408870697 R2 0.9695163369178772
## Epoch  196 MSE:  0.0065450482070446014 R2 0.9668936729431152
## Epoch  197 MSE:  0.006108627654612064 R2 0.9675158858299255
## Epoch  198 MSE:  0.006115186493843794 R2 0.9691495299339294
## Epoch  199 MSE:  0.006294677499681711 R2 0.9688394069671631
## Epoch  200 MSE:  0.0059805032797157764 R2 0.9699112772941589
## Epoch  201 MSE:  0.006412376184016466 R2 0.9692266583442688
## Epoch  202 MSE:  0.0065093147568404675 R2 0.9670207500457764
## Epoch  203 MSE:  0.005105788353830576 R2 0.9735230207443237
## Epoch  204 MSE:  0.0068105971440672874 R2 0.9634036421775818
## Epoch  205 MSE:  0.005655080545693636 R2 0.9713810682296753
## Epoch  206 MSE:  0.00560990022495389 R2 0.9740450382232666
## Epoch  207 MSE:  0.0059644305147230625 R2 0.9706023335456848
## Epoch  208 MSE:  0.005902386270463467 R2 0.9681909084320068
## Epoch  209 MSE:  0.006224923301488161 R2 0.9660131335258484
## Epoch  210 MSE:  0.0064316256903111935 R2 0.9678651690483093
## Epoch  211 MSE:  0.006411944516003132 R2 0.9703030586242676
## Epoch  212 MSE:  0.006336397957056761 R2 0.9685736894607544
## Epoch  213 MSE:  0.005718912463635206 R2 0.970257043838501
## Epoch  214 MSE:  0.006079385522753 R2 0.9674644470214844
## Epoch  215 MSE:  0.005961387883871794 R2 0.9705788493156433
## Epoch  216 MSE:  0.006385700777173042 R2 0.9695829153060913
## Epoch  217 MSE:  0.00551938870921731 R2 0.9721552133560181
## Epoch  218 MSE:  0.005599146243184805 R2 0.9703459739685059
## Epoch  219 MSE:  0.005898566450923681 R2 0.9693739414215088
## Epoch  220 MSE:  0.005617812741547823 R2 0.9723753929138184
## Epoch  221 MSE:  0.005489769857376814 R2 0.972437858581543
## Epoch  222 MSE:  0.006302519701421261 R2 0.9682247638702393
## Epoch  223 MSE:  0.0050707547925412655 R2 0.9737696647644043
## Epoch  224 MSE:  0.005958175752311945 R2 0.96907639503479
## Epoch  225 MSE:  0.005135675892233849 R2 0.974398672580719
## Epoch  226 MSE:  0.006818938069045544 R2 0.9657344222068787
## Epoch  227 MSE:  0.005383857060223818 R2 0.9730949997901917
## Epoch  228 MSE:  0.005976186599582434 R2 0.9693691730499268
## Epoch  229 MSE:  0.005321640055626631 R2 0.9726920127868652
## Epoch  230 MSE:  0.0058106607757508755 R2 0.9714874029159546
## Epoch  231 MSE:  0.0059808362275362015 R2 0.9703096747398376
## Epoch  232 MSE:  0.0060668508522212505 R2 0.9685224890708923
## Epoch  233 MSE:  0.00637073116376996 R2 0.9668856263160706
## Epoch  234 MSE:  0.004986935760825872 R2 0.9759079813957214
## Epoch  235 MSE:  0.006159973330795765 R2 0.9701669812202454
## Epoch  236 MSE:  0.0047853002324700356 R2 0.9754735827445984
## Epoch  237 MSE:  0.0061004855670034885 R2 0.9677600264549255
## Epoch  238 MSE:  0.005289617460221052 R2 0.9738962650299072
## Epoch  239 MSE:  0.006310881581157446 R2 0.9679137468338013
## Epoch  240 MSE:  0.005487554706633091 R2 0.9729105830192566
## Epoch  241 MSE:  0.005944752600044012 R2 0.9702807068824768
## Epoch  242 MSE:  0.005584438797086477 R2 0.9706501960754395
## Epoch  243 MSE:  0.0059288316406309605 R2 0.9703902006149292
## Epoch  244 MSE:  0.005165284499526024 R2 0.9748358726501465
## Epoch  245 MSE:  0.0051946151070296764 R2 0.974245011806488
## Epoch  246 MSE:  0.005276110954582691 R2 0.9729243516921997
## Epoch  247 MSE:  0.005967170465737581 R2 0.9694318175315857
## Epoch  248 MSE:  0.004801316186785698 R2 0.9755889773368835
## Epoch  249 MSE:  0.005702749826014042 R2 0.9721812605857849
## Epoch  250 MSE:  0.005205148831009865 R2 0.9739764928817749
## Epoch  251 MSE:  0.0046859863214194775 R2 0.9753831028938293
## Epoch  252 MSE:  0.0050020101480185986 R2 0.974845826625824
## Epoch  253 MSE:  0.00564985116943717 R2 0.9714954495429993
## Epoch  254 MSE:  0.005000963341444731 R2 0.9763684272766113
## Epoch  255 MSE:  0.004990378860384226 R2 0.9738895893096924
## Epoch  256 MSE:  0.005659600254148245 R2 0.970112681388855
## Epoch  257 MSE:  0.005368035286664963 R2 0.9729671478271484
## Epoch  258 MSE:  0.005981837399303913 R2 0.9711027145385742
## Epoch  259 MSE:  0.004915089812129736 R2 0.9762650728225708
## Epoch  260 MSE:  0.004720646422356367 R2 0.9751933217048645
## Epoch  261 MSE:  0.005384501069784164 R2 0.9708820581436157
## Epoch  262 MSE:  0.004913149401545525 R2 0.9757048487663269
## Epoch  263 MSE:  0.004713958129286766 R2 0.9765925407409668
## Epoch  264 MSE:  0.005471335258334875 R2 0.9731633067131042
## Epoch  265 MSE:  0.005166992545127869 R2 0.9751152992248535
## Epoch  266 MSE:  0.004886224865913391 R2 0.974766194820404
## Epoch  267 MSE:  0.006215560715645552 R2 0.9654456973075867
## Epoch  268 MSE:  0.005104606505483389 R2 0.9746893644332886
## Epoch  269 MSE:  0.005704686976969242 R2 0.9741338491439819
## Epoch  270 MSE:  0.005496397614479065 R2 0.9730241298675537
## Epoch  271 MSE:  0.00629106629639864 R2 0.9647162556648254
## Epoch  272 MSE:  0.004872417543083429 R2 0.9745984673500061
## Epoch  273 MSE:  0.004875859245657921 R2 0.9757074117660522
## Epoch  274 MSE:  0.005860995966941118 R2 0.971680223941803
## Epoch  275 MSE:  0.00551152927801013 R2 0.9730200171470642
## Epoch  276 MSE:  0.005610702559351921 R2 0.9708433747291565
## Epoch  277 MSE:  0.005102898459881544 R2 0.9734987616539001
## Epoch  278 MSE:  0.004946876782923937 R2 0.9745447039604187
## Epoch  279 MSE:  0.004989406559616327 R2 0.9756315350532532
## Epoch  280 MSE:  0.005284270737320185 R2 0.9744126200675964
## Epoch  281 MSE:  0.005618950817734003 R2 0.9710121154785156
## Epoch  282 MSE:  0.004425379913300276 R2 0.9764961004257202
## Epoch  283 MSE:  0.004844602197408676 R2 0.9756308794021606
## Epoch  284 MSE:  0.005038614384829998 R2 0.9758918285369873
## Epoch  285 MSE:  0.004765322431921959 R2 0.9767976403236389
## Epoch  286 MSE:  0.00486151035875082 R2 0.9738835692405701
## Epoch  287 MSE:  0.004474706016480923 R2 0.9767695069313049
## Epoch  288 MSE:  0.004245744552463293 R2 0.9794406294822693
## Epoch  289 MSE:  0.004816101398319006 R2 0.9767546653747559
## Epoch  290 MSE:  0.004993516486138105 R2 0.9745946526527405
## Epoch  291 MSE:  0.0052416021935641766 R2 0.9722975492477417
## Epoch  292 MSE:  0.004457000643014908 R2 0.978620171546936
## Epoch  293 MSE:  0.005160861182957888 R2 0.9749547839164734
## Epoch  294 MSE:  0.004845053423196077 R2 0.9751296639442444
## Epoch  295 MSE:  0.005577846430242062 R2 0.9695637822151184
## Epoch  296 MSE:  0.006244901567697525 R2 0.9703384041786194
## Epoch  297 MSE:  0.00488935736939311 R2 0.9773527383804321
## Epoch  298 MSE:  0.005875481758266687 R2 0.9691973328590393
## Epoch  299 MSE:  0.005324223078787327 R2 0.9696080684661865
## Epoch  300 MSE:  0.0049764919094741344 R2 0.9745597839355469
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
## Train Score: 0.00 RMSE
print('Train R^2: %.2f' % (r2_train))
## Train R^2: 0.98

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
##   0%|          | 0/46 [00:00<?, ?it/s]  4%|4         | 2/46 [00:00<00:03, 11.92it/s]  9%|8         | 4/46 [00:00<00:05,  8.24it/s] 11%|#         | 5/46 [00:00<00:05,  7.67it/s] 13%|#3        | 6/46 [00:00<00:05,  7.44it/s] 15%|#5        | 7/46 [00:00<00:05,  7.28it/s] 17%|#7        | 8/46 [00:01<00:05,  7.24it/s] 20%|#9        | 9/46 [00:01<00:05,  7.22it/s] 22%|##1       | 10/46 [00:01<00:05,  7.09it/s] 24%|##3       | 11/46 [00:01<00:04,  7.04it/s] 26%|##6       | 12/46 [00:01<00:04,  7.01it/s] 28%|##8       | 13/46 [00:01<00:04,  7.01it/s] 30%|###       | 14/46 [00:01<00:04,  6.96it/s] 33%|###2      | 15/46 [00:02<00:04,  7.03it/s] 35%|###4      | 16/46 [00:02<00:04,  7.02it/s] 37%|###6      | 17/46 [00:02<00:04,  7.01it/s] 39%|###9      | 18/46 [00:02<00:04,  6.93it/s] 41%|####1     | 19/46 [00:02<00:03,  6.86it/s] 43%|####3     | 20/46 [00:02<00:03,  6.85it/s] 46%|####5     | 21/46 [00:02<00:03,  6.90it/s] 48%|####7     | 22/46 [00:03<00:03,  6.91it/s] 50%|#####     | 23/46 [00:03<00:03,  6.99it/s] 52%|#####2    | 24/46 [00:03<00:03,  6.84it/s] 54%|#####4    | 25/46 [00:03<00:03,  6.87it/s] 57%|#####6    | 26/46 [00:03<00:02,  6.74it/s] 59%|#####8    | 27/46 [00:03<00:02,  6.80it/s] 61%|######    | 28/46 [00:03<00:02,  6.87it/s] 63%|######3   | 29/46 [00:04<00:02,  6.92it/s] 65%|######5   | 30/46 [00:04<00:02,  7.02it/s] 67%|######7   | 31/46 [00:04<00:02,  6.97it/s] 70%|######9   | 32/46 [00:04<00:02,  6.92it/s] 72%|#######1  | 33/46 [00:04<00:01,  6.83it/s] 74%|#######3  | 34/46 [00:04<00:01,  6.90it/s] 76%|#######6  | 35/46 [00:04<00:01,  6.94it/s] 78%|#######8  | 36/46 [00:05<00:01,  6.96it/s] 80%|########  | 37/46 [00:05<00:01,  7.01it/s] 83%|########2 | 38/46 [00:05<00:01,  7.02it/s] 85%|########4 | 39/46 [00:05<00:01,  6.96it/s] 87%|########6 | 40/46 [00:05<00:00,  6.92it/s] 89%|########9 | 41/46 [00:05<00:00,  6.96it/s] 91%|#########1| 42/46 [00:05<00:00,  7.01it/s] 93%|#########3| 43/46 [00:06<00:00,  7.03it/s] 96%|#########5| 44/46 [00:06<00:00,  6.99it/s] 98%|#########7| 45/46 [00:06<00:00,  7.00it/s]100%|##########| 46/46 [00:06<00:00,  6.98it/s]100%|##########| 46/46 [00:06<00:00,  7.06it/s]

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
