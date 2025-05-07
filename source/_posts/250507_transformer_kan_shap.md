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
## Epoch  1 MSE:  0.3244590163230896 R2 -81.86808013916016
## Epoch  2 MSE:  0.17048083245754242 R2 -51.55203628540039
## Epoch  3 MSE:  0.10463625192642212 R2 -1.697335958480835
## Epoch  4 MSE:  0.0592690147459507 R2 0.46434980630874634
## Epoch  5 MSE:  0.0949481874704361 R2 0.4184958338737488
## Epoch  6 MSE:  0.08789702504873276 R2 0.5781103372573853
## Epoch  7 MSE:  0.06131557747721672 R2 0.5778844356536865
## Epoch  8 MSE:  0.07036063820123672 R2 0.1699284315109253
## Epoch  9 MSE:  0.06590444594621658 R2 0.18549025058746338
## Epoch  10 MSE:  0.053950607776641846 R2 0.4713819622993469
## Epoch  11 MSE:  0.06267283856868744 R2 0.5055198669433594
## Epoch  12 MSE:  0.05565633624792099 R2 0.6001260280609131
## Epoch  13 MSE:  0.04797773435711861 R2 0.6685233116149902
## Epoch  14 MSE:  0.04191407188773155 R2 0.686025857925415
## Epoch  15 MSE:  0.03984205797314644 R2 0.7030754089355469
## Epoch  16 MSE:  0.02472391352057457 R2 0.830919086933136
## Epoch  17 MSE:  0.028077220544219017 R2 0.8325182795524597
## Epoch  18 MSE:  0.02609783038496971 R2 0.8712986707687378
## Epoch  19 MSE:  0.022654861211776733 R2 0.8980116844177246
## Epoch  20 MSE:  0.02597144804894924 R2 0.876849353313446
## Epoch  21 MSE:  0.025426605716347694 R2 0.8630183339118958
## Epoch  22 MSE:  0.02632393315434456 R2 0.8550807237625122
## Epoch  23 MSE:  0.022798174992203712 R2 0.8839166760444641
## Epoch  24 MSE:  0.020341666415333748 R2 0.8981319665908813
## Epoch  25 MSE:  0.01860513910651207 R2 0.9097527265548706
## Epoch  26 MSE:  0.018755914643406868 R2 0.9079084396362305
## Epoch  27 MSE:  0.016408761963248253 R2 0.918725311756134
## Epoch  28 MSE:  0.015335631556808949 R2 0.9211610555648804
## Epoch  29 MSE:  0.01755468361079693 R2 0.9047802090644836
## Epoch  30 MSE:  0.01684446446597576 R2 0.8946171402931213
## Epoch  31 MSE:  0.016944915056228638 R2 0.8912290930747986
## Epoch  32 MSE:  0.016034487634897232 R2 0.8965347409248352
## Epoch  33 MSE:  0.015906553715467453 R2 0.9019235372543335
## Epoch  34 MSE:  0.015745919197797775 R2 0.9165442585945129
## Epoch  35 MSE:  0.015720762312412262 R2 0.9214410185813904
## Epoch  36 MSE:  0.017328165471553802 R2 0.9119930863380432
## Epoch  37 MSE:  0.014093161560595036 R2 0.9217148423194885
## Epoch  38 MSE:  0.017954397946596146 R2 0.8904478549957275
## Epoch  39 MSE:  0.015532765537500381 R2 0.9150664806365967
## Epoch  40 MSE:  0.013720463961362839 R2 0.9326291084289551
## Epoch  41 MSE:  0.01698736473917961 R2 0.915220320224762
## Epoch  42 MSE:  0.012984556145966053 R2 0.9324168562889099
## Epoch  43 MSE:  0.015280836261808872 R2 0.9169896841049194
## Epoch  44 MSE:  0.014594401232898235 R2 0.9231131672859192
## Epoch  45 MSE:  0.011561625637114048 R2 0.9412159323692322
## Epoch  46 MSE:  0.013118362985551357 R2 0.9328393936157227
## Epoch  47 MSE:  0.011626704595983028 R2 0.9389486908912659
## Epoch  48 MSE:  0.01443588174879551 R2 0.9235434532165527
## Epoch  49 MSE:  0.012889456935226917 R2 0.938549280166626
## Epoch  50 MSE:  0.013233843259513378 R2 0.9328243136405945
## Epoch  51 MSE:  0.014230630360543728 R2 0.9236214756965637
## Epoch  52 MSE:  0.01093862671405077 R2 0.939804196357727
## Epoch  53 MSE:  0.012389265932142735 R2 0.9350196123123169
## Epoch  54 MSE:  0.010775900445878506 R2 0.9450511336326599
## Epoch  55 MSE:  0.012510859407484531 R2 0.9337466359138489
## Epoch  56 MSE:  0.010805000551044941 R2 0.9458743333816528
## Epoch  57 MSE:  0.01252948958426714 R2 0.9365915060043335
## Epoch  58 MSE:  0.01289566233754158 R2 0.9293097853660583
## Epoch  59 MSE:  0.012318012304604053 R2 0.9378422498703003
## Epoch  60 MSE:  0.011715873144567013 R2 0.9429422616958618
## Epoch  61 MSE:  0.011672834865748882 R2 0.9367096424102783
## Epoch  62 MSE:  0.011626672931015491 R2 0.9409716725349426
## Epoch  63 MSE:  0.010594070889055729 R2 0.9477125406265259
## Epoch  64 MSE:  0.010051948949694633 R2 0.9455795288085938
## Epoch  65 MSE:  0.011740678921341896 R2 0.9377477169036865
## Epoch  66 MSE:  0.011194184422492981 R2 0.9413071870803833
## Epoch  67 MSE:  0.009772321209311485 R2 0.9475803375244141
## Epoch  68 MSE:  0.011213688179850578 R2 0.9434808492660522
## Epoch  69 MSE:  0.011703606694936752 R2 0.9394267201423645
## Epoch  70 MSE:  0.012748615816235542 R2 0.9336779713630676
## Epoch  71 MSE:  0.010481767356395721 R2 0.945489227771759
## Epoch  72 MSE:  0.010638110339641571 R2 0.9440062046051025
## Epoch  73 MSE:  0.010934988968074322 R2 0.9361864924430847
## Epoch  74 MSE:  0.010798148810863495 R2 0.9434691667556763
## Epoch  75 MSE:  0.010740821249783039 R2 0.9473460912704468
## Epoch  76 MSE:  0.009689059108495712 R2 0.9485390186309814
## Epoch  77 MSE:  0.011560985818505287 R2 0.9371325969696045
## Epoch  78 MSE:  0.010330232791602612 R2 0.950807511806488
## Epoch  79 MSE:  0.010188899002969265 R2 0.9456671476364136
## Epoch  80 MSE:  0.01043661218136549 R2 0.9411596059799194
## Epoch  81 MSE:  0.009942508302628994 R2 0.9517512321472168
## Epoch  82 MSE:  0.010461654514074326 R2 0.9492571353912354
## Epoch  83 MSE:  0.009435007348656654 R2 0.9502252340316772
## Epoch  84 MSE:  0.010358642786741257 R2 0.9400075078010559
## Epoch  85 MSE:  0.010189270600676537 R2 0.9526662230491638
## Epoch  86 MSE:  0.010159716941416264 R2 0.9530544281005859
## Epoch  87 MSE:  0.008362014777958393 R2 0.956559419631958
## Epoch  88 MSE:  0.012197624891996384 R2 0.9222647547721863
## Epoch  89 MSE:  0.009446141310036182 R2 0.9539985656738281
## Epoch  90 MSE:  0.009423636831343174 R2 0.9569090604782104
## Epoch  91 MSE:  0.01038883626461029 R2 0.9445056915283203
## Epoch  92 MSE:  0.010128379799425602 R2 0.9392440915107727
## Epoch  93 MSE:  0.01106493454426527 R2 0.9479466080665588
## Epoch  94 MSE:  0.009343083947896957 R2 0.9558821320533752
## Epoch  95 MSE:  0.009741284884512424 R2 0.9465798735618591
## Epoch  96 MSE:  0.008375469595193863 R2 0.9521741271018982
## Epoch  97 MSE:  0.008074765093624592 R2 0.958798885345459
## Epoch  98 MSE:  0.00861338060349226 R2 0.9594277739524841
## Epoch  99 MSE:  0.008906305767595768 R2 0.9551287889480591
## Epoch  100 MSE:  0.008286421187222004 R2 0.9560644030570984
## Epoch  101 MSE:  0.009665402583777905 R2 0.9463827013969421
## Epoch  102 MSE:  0.007746145129203796 R2 0.9601619243621826
## Epoch  103 MSE:  0.008528935723006725 R2 0.957419753074646
## Epoch  104 MSE:  0.007865269668400288 R2 0.9597781300544739
## Epoch  105 MSE:  0.008457592688500881 R2 0.9559739232063293
## Epoch  106 MSE:  0.006837176159024239 R2 0.9650614857673645
## Epoch  107 MSE:  0.008872193284332752 R2 0.953698456287384
## Epoch  108 MSE:  0.007684888318181038 R2 0.960578203201294
## Epoch  109 MSE:  0.007681715302169323 R2 0.9611596465110779
## Epoch  110 MSE:  0.007999641820788383 R2 0.9593503475189209
## Epoch  111 MSE:  0.007668097037822008 R2 0.9590728878974915
## Epoch  112 MSE:  0.007488598581403494 R2 0.9624501466751099
## Epoch  113 MSE:  0.00844129454344511 R2 0.9590429663658142
## Epoch  114 MSE:  0.008229208178818226 R2 0.9588971734046936
## Epoch  115 MSE:  0.008194651454687119 R2 0.955600917339325
## Epoch  116 MSE:  0.006940991152077913 R2 0.9627959728240967
## Epoch  117 MSE:  0.007827707566320896 R2 0.9618640542030334
## Epoch  118 MSE:  0.008509906008839607 R2 0.9603421092033386
## Epoch  119 MSE:  0.007452942430973053 R2 0.961605429649353
## Epoch  120 MSE:  0.009044270031154156 R2 0.9497575163841248
## Epoch  121 MSE:  0.006833908148109913 R2 0.9671173691749573
## Epoch  122 MSE:  0.009936590678989887 R2 0.9525114297866821
## Epoch  123 MSE:  0.0076332190074026585 R2 0.9621607661247253
## Epoch  124 MSE:  0.009095695801079273 R2 0.9514036178588867
## Epoch  125 MSE:  0.007868892513215542 R2 0.9534273743629456
## Epoch  126 MSE:  0.007162081077694893 R2 0.9607499837875366
## Epoch  127 MSE:  0.007639337796717882 R2 0.9629003405570984
## Epoch  128 MSE:  0.008493781089782715 R2 0.9613790512084961
## Epoch  129 MSE:  0.007792034652084112 R2 0.9584614634513855
## Epoch  130 MSE:  0.00777973048388958 R2 0.9565096497535706
## Epoch  131 MSE:  0.0077254255302250385 R2 0.9589155316352844
## Epoch  132 MSE:  0.0066995141096413136 R2 0.9670074582099915
## Epoch  133 MSE:  0.008813153952360153 R2 0.9573349952697754
## Epoch  134 MSE:  0.007352829445153475 R2 0.9641203880310059
## Epoch  135 MSE:  0.007498316932469606 R2 0.9637142419815063
## Epoch  136 MSE:  0.008097006939351559 R2 0.9525023698806763
## Epoch  137 MSE:  0.006934758275747299 R2 0.9623078107833862
## Epoch  138 MSE:  0.006993445567786694 R2 0.9674987196922302
## Epoch  139 MSE:  0.008020367473363876 R2 0.9635803699493408
## Epoch  140 MSE:  0.007395551539957523 R2 0.9611970782279968
## Epoch  141 MSE:  0.007483285386115313 R2 0.9589468836784363
## Epoch  142 MSE:  0.0065333242528140545 R2 0.9662991166114807
## Epoch  143 MSE:  0.006413144990801811 R2 0.9693188071250916
## Epoch  144 MSE:  0.006639889441430569 R2 0.9660854339599609
## Epoch  145 MSE:  0.006816761568188667 R2 0.9651182293891907
## Epoch  146 MSE:  0.006609441712498665 R2 0.9663437008857727
## Epoch  147 MSE:  0.006651867646723986 R2 0.964220404624939
## Epoch  148 MSE:  0.006357830483466387 R2 0.9679238796234131
## Epoch  149 MSE:  0.006612980272620916 R2 0.9675978422164917
## Epoch  150 MSE:  0.0070242914371192455 R2 0.9625675678253174
## Epoch  151 MSE:  0.0063431235030293465 R2 0.9677268862724304
## Epoch  152 MSE:  0.006844973191618919 R2 0.9664768576622009
## Epoch  153 MSE:  0.006120477803051472 R2 0.969608724117279
## Epoch  154 MSE:  0.006451460532844067 R2 0.9651440382003784
## Epoch  155 MSE:  0.0063308426178991795 R2 0.9686074256896973
## Epoch  156 MSE:  0.00505833001807332 R2 0.9758238792419434
## Epoch  157 MSE:  0.0054681613110005856 R2 0.9719182848930359
## Epoch  158 MSE:  0.005567214917391539 R2 0.9713070392608643
## Epoch  159 MSE:  0.005757444072514772 R2 0.9710779786109924
## Epoch  160 MSE:  0.005853920243680477 R2 0.9699321389198303
## Epoch  161 MSE:  0.005961836781352758 R2 0.9710411429405212
## Epoch  162 MSE:  0.0067397370003163815 R2 0.9673551321029663
## Epoch  163 MSE:  0.006686049979180098 R2 0.9652424454689026
## Epoch  164 MSE:  0.006711904890835285 R2 0.9641988277435303
## Epoch  165 MSE:  0.00563047407194972 R2 0.9732588529586792
## Epoch  166 MSE:  0.006188321392983198 R2 0.97017502784729
## Epoch  167 MSE:  0.005795064382255077 R2 0.968379557132721
## Epoch  168 MSE:  0.005685267969965935 R2 0.9705266952514648
## Epoch  169 MSE:  0.006108333356678486 R2 0.9695872664451599
## Epoch  170 MSE:  0.006538196466863155 R2 0.9667947888374329
## Epoch  171 MSE:  0.005939288064837456 R2 0.9694963097572327
## Epoch  172 MSE:  0.0070615001022815704 R2 0.9658304452896118
## Epoch  173 MSE:  0.005732809193432331 R2 0.9688127040863037
## Epoch  174 MSE:  0.006034162826836109 R2 0.9683396816253662
## Epoch  175 MSE:  0.0070047141052782536 R2 0.9678507447242737
## Epoch  176 MSE:  0.005719627253711224 R2 0.9706769585609436
## Epoch  177 MSE:  0.0073847356252372265 R2 0.9576202630996704
## Epoch  178 MSE:  0.006651111878454685 R2 0.9693489670753479
## Epoch  179 MSE:  0.006894728168845177 R2 0.9680904746055603
## Epoch  180 MSE:  0.007411858066916466 R2 0.9590674042701721
## Epoch  181 MSE:  0.006789466831833124 R2 0.960418164730072
## Epoch  182 MSE:  0.007036895956844091 R2 0.9681001901626587
## Epoch  183 MSE:  0.005786989349871874 R2 0.9721340537071228
## Epoch  184 MSE:  0.006410521920770407 R2 0.9662808179855347
## Epoch  185 MSE:  0.007466496434062719 R2 0.9600175023078918
## Epoch  186 MSE:  0.005784387234598398 R2 0.9710282683372498
## Epoch  187 MSE:  0.007745926268398762 R2 0.9628615379333496
## Epoch  188 MSE:  0.006697583477944136 R2 0.9656645059585571
## Epoch  189 MSE:  0.006180599331855774 R2 0.9664428234100342
## Epoch  190 MSE:  0.007264450192451477 R2 0.9615222215652466
## Epoch  191 MSE:  0.0062132952734827995 R2 0.9709522128105164
## Epoch  192 MSE:  0.005515663418918848 R2 0.9740461111068726
## Epoch  193 MSE:  0.005795839708298445 R2 0.9679241180419922
## Epoch  194 MSE:  0.006208892911672592 R2 0.9651920795440674
## Epoch  195 MSE:  0.005484140012413263 R2 0.9743022918701172
## Epoch  196 MSE:  0.005077993497252464 R2 0.975899875164032
## Epoch  197 MSE:  0.005721480119973421 R2 0.9704201817512512
## Epoch  198 MSE:  0.0050033098086714745 R2 0.9736150503158569
## Epoch  199 MSE:  0.005121196154505014 R2 0.9753851294517517
## Epoch  200 MSE:  0.005812486167997122 R2 0.9715610146522522
## Epoch  201 MSE:  0.0069578057155013084 R2 0.9674170017242432
## Epoch  202 MSE:  0.005587434861809015 R2 0.9700823426246643
## Epoch  203 MSE:  0.005549272987991571 R2 0.969587504863739
## Epoch  204 MSE:  0.005138449836522341 R2 0.9749905467033386
## Epoch  205 MSE:  0.006292460951954126 R2 0.970911979675293
## Epoch  206 MSE:  0.004755394998937845 R2 0.9752234220504761
## Epoch  207 MSE:  0.006699582561850548 R2 0.9628714919090271
## Epoch  208 MSE:  0.0065208557061851025 R2 0.9668937921524048
## Epoch  209 MSE:  0.005455526523292065 R2 0.9731776118278503
## Epoch  210 MSE:  0.006147307809442282 R2 0.9703337550163269
## Epoch  211 MSE:  0.005149289034307003 R2 0.9733278751373291
## Epoch  212 MSE:  0.0067632379941642284 R2 0.9638308882713318
## Epoch  213 MSE:  0.006321932189166546 R2 0.9673579335212708
## Epoch  214 MSE:  0.0067110126838088036 R2 0.9685299396514893
## Epoch  215 MSE:  0.004800408147275448 R2 0.9763007760047913
## Epoch  216 MSE:  0.008877823129296303 R2 0.948781430721283
## Epoch  217 MSE:  0.0063559552654623985 R2 0.9667490124702454
## Epoch  218 MSE:  0.005255629774183035 R2 0.9753440022468567
## Epoch  219 MSE:  0.00560385175049305 R2 0.9730678796768188
## Epoch  220 MSE:  0.005410523619502783 R2 0.972830593585968
## Epoch  221 MSE:  0.005659180227667093 R2 0.969674289226532
## Epoch  222 MSE:  0.008001357316970825 R2 0.9602741599082947
## Epoch  223 MSE:  0.005836199037730694 R2 0.9716172218322754
## Epoch  224 MSE:  0.0060666403733193874 R2 0.968362033367157
## Epoch  225 MSE:  0.0056483931839466095 R2 0.9712015390396118
## Epoch  226 MSE:  0.006127805914729834 R2 0.9683095812797546
## Epoch  227 MSE:  0.006586187053471804 R2 0.9667210578918457
## Epoch  228 MSE:  0.00586677948012948 R2 0.9703112840652466
## Epoch  229 MSE:  0.0055565317161381245 R2 0.9713637232780457
## Epoch  230 MSE:  0.006407930050045252 R2 0.9690821170806885
## Epoch  231 MSE:  0.005277611315250397 R2 0.9731870889663696
## Epoch  232 MSE:  0.007048908621072769 R2 0.9624230861663818
## Epoch  233 MSE:  0.006325109861791134 R2 0.9693465232849121
## Epoch  234 MSE:  0.006223865319043398 R2 0.9702410697937012
## Epoch  235 MSE:  0.004752731882035732 R2 0.9748126268386841
## Epoch  236 MSE:  0.005285304505378008 R2 0.9732733964920044
## Epoch  237 MSE:  0.005420430563390255 R2 0.9729294180870056
## Epoch  238 MSE:  0.006155931390821934 R2 0.971455454826355
## Epoch  239 MSE:  0.006284170318394899 R2 0.9665245413780212
## Epoch  240 MSE:  0.005237511824816465 R2 0.9711976647377014
## Epoch  241 MSE:  0.005261408630758524 R2 0.9747508764266968
## Epoch  242 MSE:  0.005572688765823841 R2 0.9735466837882996
## Epoch  243 MSE:  0.005289089400321245 R2 0.9725960493087769
## Epoch  244 MSE:  0.005330247804522514 R2 0.9723680019378662
## Epoch  245 MSE:  0.0052888719365000725 R2 0.9732720255851746
## Epoch  246 MSE:  0.004754737950861454 R2 0.9759392738342285
## Epoch  247 MSE:  0.00506703183054924 R2 0.9752662181854248
## Epoch  248 MSE:  0.004562671296298504 R2 0.9771904349327087
## Epoch  249 MSE:  0.005110258236527443 R2 0.9733473062515259
## Epoch  250 MSE:  0.004920970648527145 R2 0.974741518497467
## Epoch  251 MSE:  0.004764130804687738 R2 0.9761291146278381
## Epoch  252 MSE:  0.004768327809870243 R2 0.9775505661964417
## Epoch  253 MSE:  0.004132575821131468 R2 0.979833722114563
## Epoch  254 MSE:  0.005411818623542786 R2 0.9690700173377991
## Epoch  255 MSE:  0.00469746720045805 R2 0.9753133654594421
## Epoch  256 MSE:  0.0059452666901052 R2 0.9725741147994995
## Epoch  257 MSE:  0.004900065716356039 R2 0.9763451814651489
## Epoch  258 MSE:  0.006010513287037611 R2 0.9680899977684021
## Epoch  259 MSE:  0.004747600294649601 R2 0.9741546511650085
## Epoch  260 MSE:  0.005909695290029049 R2 0.971888542175293
## Epoch  261 MSE:  0.005697569344192743 R2 0.9737469553947449
## Epoch  262 MSE:  0.005107860546559095 R2 0.9723159670829773
## Epoch  263 MSE:  0.006482184398919344 R2 0.9624252319335938
## Epoch  264 MSE:  0.004581693094223738 R2 0.977620005607605
## Epoch  265 MSE:  0.006278727203607559 R2 0.9715064764022827
## Epoch  266 MSE:  0.005419895984232426 R2 0.9752801656723022
## Epoch  267 MSE:  0.006021031644195318 R2 0.9642603993415833
## Epoch  268 MSE:  0.00615469878539443 R2 0.9639031291007996
## Epoch  269 MSE:  0.004832975100725889 R2 0.977458119392395
## Epoch  270 MSE:  0.005858351942151785 R2 0.9749304056167603
## Epoch  271 MSE:  0.00444658612832427 R2 0.9771818518638611
## Epoch  272 MSE:  0.005695972591638565 R2 0.9670841693878174
## Epoch  273 MSE:  0.0051566604524850845 R2 0.9720604419708252
## Epoch  274 MSE:  0.0049045151099562645 R2 0.9769071936607361
## Epoch  275 MSE:  0.005690482445061207 R2 0.9740431308746338
## Epoch  276 MSE:  0.004303578287363052 R2 0.979483425617218
## Epoch  277 MSE:  0.00586294662207365 R2 0.9667097330093384
## Epoch  278 MSE:  0.00543146301060915 R2 0.9692402482032776
## Epoch  279 MSE:  0.004692430142313242 R2 0.9766706824302673
## Epoch  280 MSE:  0.005409920588135719 R2 0.9757351875305176
## Epoch  281 MSE:  0.005142100155353546 R2 0.9761001467704773
## Epoch  282 MSE:  0.004941625986248255 R2 0.9732593894004822
## Epoch  283 MSE:  0.005364151205867529 R2 0.9700453281402588
## Epoch  284 MSE:  0.004848805256187916 R2 0.9754607081413269
## Epoch  285 MSE:  0.004652945790439844 R2 0.9778115749359131
## Epoch  286 MSE:  0.004593329504132271 R2 0.9789403080940247
## Epoch  287 MSE:  0.004854775499552488 R2 0.97463458776474
## Epoch  288 MSE:  0.005140540190041065 R2 0.9716817140579224
## Epoch  289 MSE:  0.004603697918355465 R2 0.9767618775367737
## Epoch  290 MSE:  0.0048814332112669945 R2 0.977929413318634
## Epoch  291 MSE:  0.004467095248401165 R2 0.9786310195922852
## Epoch  292 MSE:  0.004333497025072575 R2 0.9770505428314209
## Epoch  293 MSE:  0.004864835180342197 R2 0.9727289080619812
## Epoch  294 MSE:  0.004368470050394535 R2 0.9782506823539734
## Epoch  295 MSE:  0.0045540365390479565 R2 0.9781267642974854
## Epoch  296 MSE:  0.004784087184816599 R2 0.9770055413246155
## Epoch  297 MSE:  0.0045201475732028484 R2 0.9770331382751465
## Epoch  298 MSE:  0.004492541309446096 R2 0.976433515548706
## Epoch  299 MSE:  0.0045410399325191975 R2 0.976487934589386
## Epoch  300 MSE:  0.004730215296149254 R2 0.9758826494216919
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
## Test R^2: 0.65
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
##   0%|          | 0/46 [00:00<?, ?it/s]  4%|4         | 2/46 [00:00<00:03, 12.96it/s]  9%|8         | 4/46 [00:00<00:05,  8.39it/s] 11%|#         | 5/46 [00:00<00:05,  7.83it/s] 13%|#3        | 6/46 [00:00<00:05,  7.50it/s] 15%|#5        | 7/46 [00:00<00:05,  7.40it/s] 17%|#7        | 8/46 [00:01<00:05,  7.19it/s] 20%|#9        | 9/46 [00:01<00:05,  6.99it/s] 22%|##1       | 10/46 [00:01<00:05,  6.93it/s] 24%|##3       | 11/46 [00:01<00:05,  6.81it/s] 26%|##6       | 12/46 [00:01<00:04,  6.90it/s] 28%|##8       | 13/46 [00:01<00:04,  6.97it/s] 30%|###       | 14/46 [00:01<00:04,  6.97it/s] 33%|###2      | 15/46 [00:02<00:04,  6.85it/s] 35%|###4      | 16/46 [00:02<00:04,  6.97it/s] 37%|###6      | 17/46 [00:02<00:04,  6.90it/s] 39%|###9      | 18/46 [00:02<00:04,  6.94it/s] 41%|####1     | 19/46 [00:02<00:03,  6.91it/s] 43%|####3     | 20/46 [00:02<00:03,  6.91it/s] 46%|####5     | 21/46 [00:02<00:03,  6.95it/s] 48%|####7     | 22/46 [00:03<00:03,  6.94it/s] 50%|#####     | 23/46 [00:03<00:03,  6.95it/s] 52%|#####2    | 24/46 [00:03<00:03,  6.91it/s] 54%|#####4    | 25/46 [00:03<00:03,  6.86it/s] 57%|#####6    | 26/46 [00:03<00:02,  6.86it/s] 59%|#####8    | 27/46 [00:03<00:02,  6.84it/s] 61%|######    | 28/46 [00:03<00:02,  6.82it/s] 63%|######3   | 29/46 [00:04<00:02,  6.89it/s] 65%|######5   | 30/46 [00:04<00:02,  6.88it/s] 67%|######7   | 31/46 [00:04<00:02,  6.97it/s] 70%|######9   | 32/46 [00:04<00:02,  6.91it/s] 72%|#######1  | 33/46 [00:04<00:01,  6.75it/s] 74%|#######3  | 34/46 [00:04<00:01,  6.82it/s] 76%|#######6  | 35/46 [00:04<00:01,  6.90it/s] 78%|#######8  | 36/46 [00:05<00:01,  6.83it/s] 80%|########  | 37/46 [00:05<00:01,  6.91it/s] 83%|########2 | 38/46 [00:05<00:01,  6.99it/s] 85%|########4 | 39/46 [00:05<00:01,  6.92it/s] 87%|########6 | 40/46 [00:05<00:00,  6.94it/s] 89%|########9 | 41/46 [00:05<00:00,  6.97it/s] 91%|#########1| 42/46 [00:05<00:00,  6.96it/s] 93%|#########3| 43/46 [00:06<00:00,  6.91it/s] 96%|#########5| 44/46 [00:06<00:00,  6.95it/s] 98%|#########7| 45/46 [00:06<00:00,  6.94it/s]100%|##########| 46/46 [00:06<00:00,  6.95it/s]100%|##########| 46/46 [00:06<00:00,  7.03it/s]

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
