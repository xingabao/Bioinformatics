---
title: 基于双向长短期记忆神经网络和一维卷积神经网络预测未来气象结果
date: 2025-05-10 19:39:13
tags: [Python, 机器学习, 时间序列]
categories: [[案例分享, 机器学习]]
---


<p>
在深度学习领域，序列数据的处理一直是一个关键任务。时间序列模型是根据系统观测得到的时间序列数据，通过曲线拟合和参数估计来建立数学模型的理论和方法。
</p>
<p>
在很多的时间序列预测任务中，利用卷积神经网络（CNN）和长短期记忆网络（LSTM）的混合模型是目前常见的深度学习解决方案之一。CNN
和 LSTM 各自有不同的特长，CNN 擅长局部模式的捕捉，LSTM
擅长捕捉序列的长依赖关系。通过混合这两种网络，可以非常好地学习时间序列数据中的复杂模式。
</p>

# 设置运行环境

``` r
# 指定 Python 环境
reticulate::use_python("C:/ProgramData/Anaconda3/python.exe")

# 切换工作目录
wkdir = dirname(rstudioapi::getActiveDocumentContext()$path)
```

# 导入所需库

``` python
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape, Flatten
```

# 自定义函数

``` python
# 定义一个函数，对输入的训练集、验证集和测试集进行数据归一化处理
def normalize_dataframe(train_set, val_set, test_set):
    # 初始化`MinMaxScaler`对象，用于将数据归一化到 [0, 1] 范围
    scaler = MinMaxScaler()
    # 在训练集上拟合归一化模型，计算每个特征的最小值和最大值
    # 这一步不会对训练集，仅记录归一化参数
    scaler.fit(train_set)
    # 使用训练集拟合的归一化模型对训练集、验证集和测试集进行转换
    # 转换后的数据保持原有的列名和索
    train = pd.DataFrame(scaler.transform(train_set), columns = train_set.columns, index = train_set.index)
    val = pd.DataFrame(scaler.transform(val_set), columns = val_set.columns, index = val_set.index)
    test = pd.DataFrame(scaler.transform(test_set), columns = test_set.columns, index = test_set.index)
    
    # 返回归一化后的训练集、验证集和测试集
    return train, val, test

# 定义一个函数，用于准备时间序列数据，将其转换为适合模型输入的格式
def prepare_data(data, win_size):
    X = []  # 存储输入特征（时间窗口内的数据）
    y = []  # 存储目标值（时间窗口后的数据）

    # 遍历数据，创建时间窗口大小为 win_size 的输入和对应的目标值
    for i in range(len(data) - win_size):
        # 提取一个时间窗口的数据作为输入
        temp_x = data[i:i + win_size]
        # 提取时间窗口后的数据作为目标值
        temp_y = data[i + win_size]    
        X.append(temp_x)
        y.append(temp_y)
        
    # 将列表转换为 numpy 数组，便于后续模型输入
    X = np.asarray(X)
    y = np.asarray(y)
    X = np.expand_dims(X, axis = -1)
    
    # 返回输入特征和目标值
    return X, y
```

# 全局环境变量

``` python
# 全局环境变量
win_size = 30                 # 准备时间序列数据，设置时间窗口大小为 30
epoch_size = 100              # 设置 epoch 次数为 100（这里测试设置值较小，具体根据实际设置）
batch_size = 32               # 设置批量大小
verbose = 0                   # 是否打印中间过程，0 表示静默状态
train_ratio = 0.7             # 训练集比例
val_ratio = 0.1               # 验证集比例
test_ratio = 0.2              # 测试集比例
```

# 设置随机种子

``` python
# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)   

# 增强 TensorFlow 的确定性
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

# 加载数据

``` python
# 加载数据
df = pd.read_csv('data/weather.csv')
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Day'].astype(str), format = '%Y-%j')
df.set_index('Date', inplace = True)
df.drop(['Year', 'Day'], axis = 1, inplace = True)
col = 'Temperature'
df = df[[col]]

# 生成时间范围
start_date = pd.Timestamp('1990-01-01')
end_date = pd.Timestamp('2023-03-01')
date_range = pd.date_range(start = start_date, end = end_date, freq = 'D')

# 检查时间范围中是否包含 DataFrame 中的所有日期
missing_dates = date_range[~date_range.isin(df.index)]
print("Missing Dates:")
```

    ## Missing Dates:

``` python
print(missing_dates)
```

    ## DatetimeIndex([], dtype='datetime64[ns]', freq='D')

# 可视化数据集

``` python
plt.figure(figsize = (15, 5))
plt.plot(df[col], color = '#00A087',  alpha = 0.3)
plt.title('')
plt.xticks(rotation = 0)
## (array([ 6574.,  8035.,  9496., 10957., 12418., 13879., 15340., 16801.,
##        18262., 19723.]), [Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, '')])
plt.show()
```

![](/imgs/153c1049c2906567944c66d807a6a1d6.png)
# 数据处理

## 数据集划分

``` python
# 计算划分的索引
train_split = int(train_ratio * len(df))
val_split = int((train_ratio + val_ratio) * len(df))

# 划分数据集
train_set = df.iloc[:train_split]
val_set = df.iloc[train_split:val_split]
test_set = df.iloc[val_split:]
```

## 可视化训练集, 验证集和测试集数据

``` python
plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(train_set, color = 'g',  alpha = 0.3)
plt.title('Training Data')

plt.subplot(3, 1, 2)
plt.plot(val_set, color = 'b',  alpha = 0.3)
plt.title('Validation Data')

plt.subplot(3, 1, 3)
plt.plot(test_set, color = 'r',  alpha = 0.3)
plt.title('Testing Data')
plt.xticks(rotation = 0)
## (array([16801., 17167., 17532., 17897., 18262., 18628., 18993., 19358.]), [Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, '')])
plt.show()
```

![](/imgs/5c70c3ff8121b74da43a44c9df8a0532.png)
## 归一化处理

``` python
# 对训练集, 验证集和测试集进行归一化处理
train, val, test = normalize_dataframe(train_set, val_set, test_set)
```

## 可视化归一化后的训练集, 验证集和测试集数据

``` python
plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(train, color = 'g',  alpha = 0.3)
plt.title('Training Data')

plt.subplot(3, 1, 2)
plt.plot(val, color = 'b',  alpha = 0.3)
plt.title('Validation Data')

plt.subplot(3, 1, 3)
plt.plot(test, color = 'r',  alpha = 0.3)
plt.title('Testing Data')
plt.xticks(rotation = 0)
## (array([16801., 17167., 17532., 17897., 18262., 18628., 18993., 19358.]), [Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, ''), Text(0, 0, '')])
plt.show()
```

![](/imgs/c85bb50f0c3a2f1e1c10f64d62e36f3c.png)
## 准备时间序列数据

``` python
# 训练集
X_train, y_train = prepare_data(train['Temperature'].values, win_size)

# 验证集
X_val, y_val= prepare_data(val['Temperature'].values, win_size)

# 测试集
X_test, y_test = prepare_data(test['Temperature'].values, win_size)

df_max = list(np.max(train_set))[0]
df_min = list(np.min(train_set))[0]

print("训练集形状:", X_train.shape, y_train.shape)
## 训练集形状: (8449, 30, 1) (8449,)
print("验证集形状:", X_val.shape, y_val.shape)
## 验证集形状: (1181, 30, 1) (1181,)
print("测试集形状:", X_test.shape, y_test.shape)
## 测试集形状: (2393, 30, 1) (2393,)
```

# 构建预测模型

## 双向长短时记忆网络

<p>
双向长短时记忆网络 (Bi-LSTM)，是一种特殊的 LSTM
结构，它同时处理序列的正向和方向信息。这意味着对于给定的时间步，双向
LSTM 不仅考虑了之前的时间步信息，像标准的单向 LSTM
一样，还考虑了未来时间步的信息。这使得 Bi-LSTM
在处理自然语言任务、语音识别和其他需要理解上下文的任务时非常有用。
</p>
<p>
`Bi-LSTM`特别适合时间序列数据，因为它可以同时考虑过去和未来的依赖关系；对于温度预测任务，这种模型能够更好地捕捉数据中的长期依赖和趋势。相比单向，双向通过正向和方向处理序列，能够更全面地理解数据的上下文，尤其是在序列数据中存在前后依赖关系。
</p>
<p>
这个代码使用的模型是基于`Bi-LSTM`的深度学习模型，专门用于时间序列预测任务。模型通过`Bi-LSTM`层提取时间序列特征，并通过多个全连接层进行回归预测，最终输出单一的预测值（如温度）。
</p>

### 构建 Bi-LSTM 模型

<p>

构建 Bi-LSTM 模型，双向长短时记忆网络&lt;/&gt;

``` python
# 创建一个顺序模型
model = Sequential()
# 添加双向 LSTM 层，128 个单元，激活函数为 relu，输入形状为 (时间窗口大小, 特征数量)
model.add(Bidirectional(LSTM(128, activation = 'relu'), input_shape = (X_train.shape[1], X_train.shape[2])))
# 添加全连接层，64 个神经元，relu 激活函数
model.add(Dense(64, activation = 'relu'))
# 添加全连接层，32 个神经元，relu 激活函数
model.add(Dense(32, activation = 'relu'))
# 添加全连接层，16 个神经元，relu 激活函数
model.add(Dense(16, activation = 'relu'))
# 输出层，1 个神经元，用于预测单个数值；使用 sigmoid 激活函数，将输出限制在 0 到 1 之间
model.add(Dense(1, activation = 'sigmoid'))
```

### 编译模型

``` python
# 编译模型，优化器为 adam，损失函数为均方误差 (mse)
model.compile(optimizer = 'adam', loss = 'mse')
```

### 训练模型

``` python
# 训练模型，设置 epoch 次数为 10（这里测试设置值较小，具体根据实际设置），批量大小为 32，使用验证集评估模型
history = model.fit(X_train, y_train, epochs = epoch_size, batch_size = batch_size, validation_data = (X_val, y_val), verbose = verbose)
```

### 绘制训练过程中的损失曲线

``` python
plt.figure()
plt.plot(history.history['loss'], c = 'b', label = 'loss')
plt.plot(history.history['val_loss'], c = 'g', label = 'val_loss')
plt.legend()
plt.show()
```

![](/imgs/e1b875bbe5fbf4a3243ae05f8efa8ce1.png)
### 使用模型对测试集进行预测

``` python
y_pred = model.predict(X_test, verbose = verbose)
```

### 计算模型性能指标

``` python
# 计算均方误差（MSE）
mse = metrics.mean_squared_error(y_test, np.array([i for arr in y_pred for i in arr]))
# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
# 计算平均绝对误差（MAE）
mae = metrics.mean_absolute_error(y_test, np.array([i for arr in y_pred for i in arr]))
# 计算 R² 拟合优度
r2 = r2_score(y_test, np.array([i for arr in y_pred for i in arr]))

print("均方误差 (MSE):", mse)
## 均方误差 (MSE): 0.0009024836310498792
print("均方根误差 (RMSE):", rmse)
## 均方根误差 (RMSE): 0.03004136533265223
print("平均绝对误差 (MAE):", mae)
## 平均绝对误差 (MAE): 0.023367103850682486
print("拟合优度:", r2)
## 拟合优度: 0.9709102190822203
```

### 打印模型结构摘要

``` python
model.summary()
## Model: "sequential"
## ┌─────────────────────────────────┬────────────────────────┬───────────────┐
## │ Layer (type)                    │ Output Shape           │       Param # │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ bidirectional (Bidirectional)   │ (None, 256)            │       133,120 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ dense (Dense)                   │ (None, 64)             │        16,448 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ dense_1 (Dense)                 │ (None, 32)             │         2,080 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ dense_2 (Dense)                 │ (None, 16)             │           528 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ dense_3 (Dense)                 │ (None, 1)              │            17 │
## └─────────────────────────────────┴────────────────────────┴───────────────┘
##  Total params: 456,581 (1.74 MB)
##  Trainable params: 152,193 (594.50 KB)
##  Non-trainable params: 0 (0.00 B)
##  Optimizer params: 304,388 (1.16 MB)
```

### 未来输出预测

``` python
# 取出预测的最后一个时间步的输出作为下一步的输入
last_output = model.predict(X_test, verbose = verbose)[-1]

# 预测的时间步数
steps = 10  # 假设向后预测 10 个时间步
predicted = []
for i in range(steps):
    # 将最后一个输出加入 X_test，继续向后预测
    input_data = np.append(X_test[-1][1:], last_output).reshape(1, X_test.shape[1], X_test.shape[2])
    # 使用模型进行预测
    next_output = model.predict(input_data, verbose = verbose)
    # 将预测的值加入结果列表
    predicted.append(next_output[0][0])
    last_output = next_output[0]

print("向后预测的值:", predicted)
## 向后预测的值: [0.5347221, 0.54252774, 0.5505376, 0.55875146, 0.56716347, 0.5757619, 0.58452773, 0.5930122, 0.60100317, 0.6083987]
```

``` python
series_1 = y_pred*(df_max - df_min) + df_min
series_2 = np.array(predicted)*(df_max - df_min) + df_min

plt.figure(figsize = (15,4), dpi = 300)

plt.subplot(3 ,1, 1)
plt.plot(train_set, color = 'c', label = 'Training Data')
plt.plot(val_set, color = 'r', label = 'Validation Data')
plt.plot(test_set, color = 'b', label = 'Testing Data')
plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(test_set, color = 'b', label = 'Training Data')
plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(test_set, color = 'b', label = 'Training Data')
plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-03-11'))
## (18993.0, 19427.0)
plt.legend()

plt.show()
```

![](/imgs/24e7db63bcff89771fe5d4439f302279.png)
## 混合神经网络模型

<p>
混合神经网络模型，结合双向长短期记忆网络（Bidirectional LSTM,
Bi-LSTM）和一维卷积神经网络（1D
CNN），用于处理时间序列或序列数据的任务（如二分类问题）。Bi-LSTM
擅长捕捉序列中的长期依赖关系和上下文信息，而 1D CNN
擅长提取局部特征。通过结合两者的优势，模型可以在全局和局部特征提取上都表现优异。
</p>
<p>
一个`Bi-LSTM + 1D CNN`混合模型，结合了`Bi-LSTM`的长期依赖建模能力和`1D CNN`的局部特征提取能力，最终用于二分类任务（通过`sigmoid`等激活函数输出概率）。这种混合架构在处理复杂序列数据时通常表现优异，适用于时间序列、信号处理等领域。如果需要进一步优化，可以引入残差连接、注意力机制或调整网络层数和参数。
</p>

### 构建混合神经网络模型

<p>

构建混合神经网络模型，结合了 Bi-LSTM 和 1D CNN 两个模型&lt;/&gt;

``` python
# 创建一个顺序模型
model = Sequential()
# 添加双向长短期记忆层，分别从正向和反向处理输入序列，捕捉序列中前后依赖关系
# 每个方向有 128 个隐藏单元，因此总共 256 个隐藏单元
# 使用激活函数 ReLU，引入非线性，增强模型的学习能力
model.add(Bidirectional(LSTM(128, activation = 'relu'), input_shape = (X_train.shape[1], X_train.shape[2])))
# 添加重塑层，将`Bi-LSTM`的输出重塑为形状为`(256, 1)`的二维张量
# 这一步是为了将`Bi-LSTM`输出调整为适合后续`1D CNN`层处理的形状
model.add(Reshape((256, 1)))
# 添加一维卷积层`Conv1D`，64 个卷积核（过滤器），每个卷积核会提取不同的特征
# 卷积核的大小为 7，表示每次卷积操作覆盖 7 个时间步（适用于时间序列数据）
# 使用`ReLU`激活函数，引入非线性，增强模型的学习能力
model.add(Conv1D(filters = 64, kernel_size = 7, activation = 'relu'))
# 添加一维最大池化层，池化窗口大小为 2，表示将输入数据的大小减半（下采样），提取主要特征，减少计算量
model.add(MaxPooling1D(pool_size = 2))
# 添加展平层，将多维输入，例如卷积层输出的特征图展平成一维向量，以便后续全连接层处理
model.add(Flatten())
# 添加全连接层，32 个神经元，relu 激活函数
model.add(Dense(32, activation = 'relu'))
# 添加全连接层，16 个神经元，relu 激活函数
model.add(Dense(16, activation = 'relu'))
# 输出层，1 个神经元，用于预测单个数值；使用 sigmoid 激活函数，将输出限制在 0 到 1 之间
model.add(Dense(1, activation = 'sigmoid'))
```

### 编译模型

``` python
# 编译模型，优化器为 adam，损失函数为均方误差 (mse)
model.compile(optimizer = 'adam', loss = 'mse')
```

### 训练模型

``` python
# 训练模型，设置 epoch 次数为 10（这里测试设置值较小，具体根据实际设置），批量大小为 32，使用验证集评估模型
history = model.fit(X_train, y_train, epochs = epoch_size, batch_size = batch_size, validation_data = (X_val, y_val), verbose = verbose)
```

### 绘制训练过程中的损失曲线

``` python
plt.figure()
plt.plot(history.history['loss'], c = 'b', label = 'loss')
plt.plot(history.history['val_loss'], c = 'g', label = 'val_loss')
plt.legend()
plt.show()
```

![](/imgs/837a964822c9d4ca0fa71190ec17522f.png)
### 使用模型对测试集进行预测

``` python
y_pred = model.predict(X_test, verbose = verbose)
```

### 计算模型性能指标

``` python
# 计算均方误差（MSE）
mse = metrics.mean_squared_error(y_test, np.array([i for arr in y_pred for i in arr]))
# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
# 计算平均绝对误差（MAE）
mae = metrics.mean_absolute_error(y_test, np.array([i for arr in y_pred for i in arr]))
# 计算 R² 拟合优度
r2 = r2_score(y_test, np.array([i for arr in y_pred for i in arr]))

print("均方误差 (MSE):", mse)
## 均方误差 (MSE): 0.0009550017622248667
print("均方根误差 (RMSE):", rmse)
## 均方根误差 (RMSE): 0.030903102792840507
print("平均绝对误差 (MAE):", mae)
## 平均绝对误差 (MAE): 0.024207261518905218
print("拟合优度:", r2)
## 拟合优度: 0.9692174006448219
```

### 打印模型结构摘要

``` python
model.summary()
## Model: "sequential_1"
## ┌─────────────────────────────────┬────────────────────────┬───────────────┐
## │ Layer (type)                    │ Output Shape           │       Param # │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ bidirectional_1 (Bidirectional) │ (None, 256)            │       133,120 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ reshape (Reshape)               │ (None, 256, 1)         │             0 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ conv1d (Conv1D)                 │ (None, 250, 64)        │           512 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ max_pooling1d (MaxPooling1D)    │ (None, 125, 64)        │             0 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ flatten (Flatten)               │ (None, 8000)           │             0 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ dense_4 (Dense)                 │ (None, 32)             │       256,032 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ dense_5 (Dense)                 │ (None, 16)             │           528 │
## ├─────────────────────────────────┼────────────────────────┼───────────────┤
## │ dense_6 (Dense)                 │ (None, 1)              │            17 │
## └─────────────────────────────────┴────────────────────────┴───────────────┘
##  Total params: 1,170,629 (4.47 MB)
##  Trainable params: 390,209 (1.49 MB)
##  Non-trainable params: 0 (0.00 B)
##  Optimizer params: 780,420 (2.98 MB)
```

### 未来输出预测

``` python
# 取出预测的最后一个时间步的输出作为下一步的输入
last_output = model.predict(X_test, verbose = verbose)[-1]

# 预测的时间步数
steps = 10  # 假设向后预测 10 个时间步
predicted = []
for i in range(steps):
    # 将最后一个输出加入 X_test，继续向后预测
    input_data = np.append(X_test[-1][1:], last_output).reshape(1, X_test.shape[1], X_test.shape[2])
    # 使用模型进行预测
    next_output = model.predict(input_data, verbose = verbose)
    # 将预测的值加入结果列表
    predicted.append(next_output[0][0])
    last_output = next_output[0]

print("向后预测的值:", predicted)
## 向后预测的值: [0.53138125, 0.5355914, 0.5394832, 0.54305017, 0.54631245, 0.5493123, 0.55207515, 0.55461955, 0.5569496, 0.55904627]
```

``` python
series_1 = y_pred*(df_max - df_min) + df_min
series_2 = np.array(predicted)*(df_max - df_min) + df_min

plt.figure(figsize = (15,4), dpi = 300)

plt.subplot(3 ,1, 1)
plt.plot(train_set, color = 'c', label = 'Training Data')
plt.plot(val_set, color = 'r', label = 'Validation Data')
plt.plot(test_set, color = 'b', label = 'Testing Data')
plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(test_set, color = 'b', label = 'Training Data')
plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(test_set, color = 'b', label = 'Training Data')
plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-03-11'))
## (18993.0, 19427.0)
plt.legend()

plt.show()
```

![](/imgs/c9443de57904c22f0f997c57bd68e8a4.png)
# 版本信息

``` python
import sys
import platform
import pkg_resources

def session_info():
    print("Python Session Information")
    print("==========================")
    
    # Python 版本信息
    print(f"Python Version: {sys.version}")
    print(f"Python Implementation: {platform.python_implementation()}")
    print(f"Python Build: {platform.python_build()}")
    
    # 操作系统信息
    print("\nOperating System Information")
    print(f"OS: {platform.system()}")
    print(f"OS Release: {platform.release()}")
    print(f"OS Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # 已安装的包及其版本
    print("\nInstalled Packages")
    print("------------------")
    installed_packages = sorted(
        [(dist.key, dist.version) for dist in pkg_resources.working_set],
        key=lambda x: x[0].lower()
    )
    for package, version in installed_packages:
        print(f"{package}: {version}")

# 调用函数
session_info()
## Python Session Information
## ==========================
## Python Version: 3.9.12 (main, Apr  4 2022, 05:22:27) [MSC v.1916 64 bit (AMD64)]
## Python Implementation: CPython
## Python Build: ('main', 'Apr  4 2022 05:22:27')
## 
## Operating System Information
## OS: Windows
## OS Release: 10
## OS Version: 10.0.26100
## Machine: AMD64
## Processor: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel
## 
## Installed Packages
## ------------------
## absl-py: 2.2.2
## aiohttp: 3.8.1
## aiosignal: 1.2.0
## alabaster: 0.7.12
## anaconda-client: 1.9.0
## anaconda-navigator: 2.1.4
## anaconda-project: 0.10.2
## anyio: 3.5.0
## appdirs: 1.4.4
## argon2-cffi: 21.3.0
## argon2-cffi-bindings: 21.2.0
## arrow: 1.2.2
## astroid: 2.6.6
## astropy: 5.0.4
## asttokens: 2.0.5
## astunparse: 1.6.3
## async-timeout: 4.0.1
## atomicwrites: 1.4.0
## attrs: 21.4.0
## automat: 20.2.0
## autopep8: 1.6.0
## babel: 2.9.1
## backcall: 0.2.0
## backports.functools-lru-cache: 1.6.4
## backports.tempfile: 1.0
## backports.weakref: 1.0.post1
## bcrypt: 3.2.0
## beautifulsoup4: 4.11.1
## binaryornot: 0.4.4
## biopython: 1.79
## bitarray: 2.4.1
## bkcharts: 0.2
## black: 19.10b0
## bleach: 4.1.0
## bokeh: 2.4.2
## boto3: 1.21.32
## botocore: 1.24.32
## bottleneck: 1.3.4
## brotlipy: 0.7.0
## cachetools: 4.2.2
## causal-learn: 0.1.4.1
## certifi: 2021.10.8
## cffi: 1.15.0
## chardet: 4.0.0
## charset-normalizer: 2.0.4
## clarabel: 0.10.0
## click: 8.0.4
## cloudpickle: 2.0.0
## clyent: 1.2.2
## colorama: 0.4.4
## colorcet: 2.0.6
## comtypes: 1.1.10
## conda: 4.12.0
## conda-build: 3.21.8
## conda-content-trust: 0+unknown
## conda-pack: 0.6.0
## conda-package-handling: 1.8.1
## conda-repo-cli: 1.0.4
## conda-token: 0.3.0
## conda-verify: 3.4.2
## constantly: 15.1.0
## cookiecutter: 1.7.3
## cryptography: 3.4.8
## cssselect: 1.1.0
## cvxpy: 1.6.5
## cycler: 0.11.0
## cython: 0.29.28
## cytoolz: 0.11.0
## daal4py: 2021.5.0
## dask: 2022.2.1
## datashader: 0.13.0
## datashape: 0.5.4
## debugpy: 1.5.1
## decorator: 5.1.1
## defusedxml: 0.7.1
## diff-match-patch: 20200713
## distributed: 2022.2.1
## docutils: 0.17.1
## docxcompose: 1.4.0
## docxtpl: 0.19.1
## dowhy: 0.12
## entrypoints: 0.4
## et-xmlfile: 1.1.0
## executing: 0.8.3
## fastjsonschema: 2.15.1
## filelock: 3.6.0
## flake8: 3.9.2
## flask: 1.1.2
## flatbuffers: 25.2.10
## fonttools: 4.25.0
## frozenlist: 1.2.0
## fsspec: 2022.2.0
## future: 0.18.2
## gast: 0.6.0
## gensim: 4.1.2
## glob2: 0.7
## google-api-core: 1.25.1
## google-auth: 1.33.0
## google-cloud-core: 1.7.1
## google-cloud-storage: 1.31.0
## google-crc32c: 1.1.2
## google-pasta: 0.2.0
## google-resumable-media: 1.3.1
## googleapis-common-protos: 1.53.0
## graphviz: 0.20.3
## greenlet: 1.1.1
## grpcio: 1.71.0
## h5py: 3.13.0
## heapdict: 1.0.1
## holoviews: 1.14.8
## hvplot: 0.7.3
## hyperlink: 21.0.0
## idna: 3.3
## imagecodecs: 2021.8.26
## imageio: 2.9.0
## imagesize: 1.3.0
## importlib-metadata: 4.11.3
## incremental: 21.3.0
## inflection: 0.5.1
## iniconfig: 1.1.1
## intake: 0.6.5
## intervaltree: 3.1.0
## ipykernel: 6.9.1
## ipython: 8.2.0
## ipython-genutils: 0.2.0
## ipywidgets: 7.6.5
## isort: 5.9.3
## itemadapter: 0.3.0
## itemloaders: 1.0.4
## itsdangerous: 2.0.1
## jdcal: 1.4.1
## jedi: 0.18.1
## jinja2: 2.11.3
## jinja2-time: 0.2.0
## jmespath: 0.10.0
## joblib: 1.4.2
## json5: 0.9.6
## jsonschema: 4.4.0
## jupyter: 1.0.0
## jupyter-client: 6.1.12
## jupyter-console: 6.4.0
## jupyter-core: 4.9.2
## jupyter-server: 1.13.5
## jupyterlab: 3.3.2
## jupyterlab-pygments: 0.1.2
## jupyterlab-server: 2.10.3
## jupyterlab-widgets: 1.0.0
## keras: 3.9.2
## keyring: 23.4.0
## kiwisolver: 1.3.2
## lazy-object-proxy: 1.6.0
## libarchive-c: 2.9
## libclang: 18.1.1
## lightgbm: 4.6.0
## llvmlite: 0.43.0
## locket: 0.2.1
## looseversion: 1.3.0
## lxml: 4.8.0
## markdown: 3.3.4
## markdown-it-py: 3.0.0
## markupsafe: 2.0.1
## matplotlib: 3.5.1
## matplotlib-inline: 0.1.2
## mccabe: 0.6.1
## mdurl: 0.1.2
## menuinst: 1.4.18
## mistune: 0.8.4
## mkl-fft: 1.3.1
## mkl-random: 1.2.2
## mkl-service: 2.4.0
## ml-dtypes: 0.5.1
## mock: 4.0.3
## momentchi2: 0.1.8
## mpmath: 1.2.1
## msgpack: 1.0.2
## multidict: 5.1.0
## multipledispatch: 0.6.0
## munkres: 1.1.4
## mypy-extensions: 0.4.3
## namex: 0.0.9
## navigator-updater: 0.2.1
## nbclassic: 0.3.5
## nbclient: 0.5.13
## nbconvert: 6.4.4
## nbformat: 5.3.0
## nest-asyncio: 1.5.5
## networkx: 3.2.1
## nltk: 3.7
## nose: 1.3.7
## notebook: 6.4.8
## numba: 0.60.0
## numexpr: 2.8.1
## numpy: 1.26.4
## numpydoc: 1.2
## olefile: 0.46
## opencv-python: 4.11.0.86
## openpyxl: 3.0.9
## opt-einsum: 3.4.0
## optree: 0.15.0
## osqp: 1.0.3
## packaging: 21.3
## pandas: 1.5.3
## pandocfilters: 1.5.0
## panel: 0.13.0
## param: 1.12.0
## paramiko: 2.8.1
## parsel: 1.6.0
## parso: 0.8.3
## partd: 1.2.0
## pathspec: 0.7.0
## patsy: 1.0.1
## pep8: 1.7.1
## pexpect: 4.8.0
## pickleshare: 0.7.5
## pillow: 9.0.1
## pims: 0.7
## pip: 21.2.4
## pkginfo: 1.8.2
## plotly: 5.6.0
## pluggy: 1.0.0
## poyo: 0.5.0
## prometheus-client: 0.13.1
## prompt-toolkit: 3.0.20
## protego: 0.1.16
## protobuf: 5.29.4
## psutil: 5.8.0
## ptyprocess: 0.7.0
## pure-eval: 0.2.2
## py: 1.11.0
## pyasn1: 0.4.8
## pyasn1-modules: 0.2.8
## pycodestyle: 2.7.0
## pycosat: 0.6.3
## pycparser: 2.21
## pyct: 0.4.6
## pycurl: 7.44.1
## pydispatcher: 2.0.5
## pydocstyle: 6.1.1
## pydot: 3.0.4
## pyerfa: 2.0.0
## pyflakes: 2.3.1
## pygments: 2.19.1
## pyhamcrest: 2.0.2
## pyjwt: 2.1.0
## pylint: 2.9.6
## pyls-spyder: 0.4.0
## pymysql: 1.1.1
## pynacl: 1.4.0
## pyodbc: 4.0.32
## pyopenssl: 21.0.0
## pyparsing: 3.2.3
## pypdf2: 3.0.1
## pyreadline: 2.1
## pyrsistent: 0.18.0
## pysocks: 1.7.1
## pytest: 7.1.1
## python-dateutil: 2.8.2
## python-docx: 1.1.2
## python-lsp-black: 1.0.0
## python-lsp-jsonrpc: 1.0.0
## python-lsp-server: 1.2.4
## python-slugify: 5.0.2
## python-snappy: 0.6.0
## pytz: 2021.3
## pyviz-comms: 2.0.2
## pywavelets: 1.3.0
## pywin32: 302
## pywin32-ctypes: 0.2.0
## pywinpty: 2.0.2
## pyyaml: 6.0
## pyzmq: 22.3.0
## qdarkstyle: 3.0.2
## qstylizer: 0.1.10
## qtawesome: 1.0.3
## qtconsole: 5.3.0
## qtpy: 2.0.1
## queuelib: 1.5.0
## regex: 2022.3.15
## requests: 2.27.1
## requests-file: 1.5.1
## rich: 14.0.0
## rope: 0.22.0
## rsa: 4.7.2
## rtree: 0.9.7
## ruamel-yaml-conda: 0.15.100
## s3transfer: 0.5.0
## scikit-image: 0.19.2
## scikit-learn: 1.6.1
## scikit-learn-intelex: 2021.20220215.102710
## scipy: 1.13.1
## scrapy: 2.6.1
## scs: 3.2.7.post2
## seaborn: 0.11.2
## send2trash: 1.8.0
## service-identity: 18.1.0
## setuptools: 61.2.0
## shap: 0.47.2
## sip: 4.19.13
## six: 1.16.0
## slicer: 0.0.8
## slicerator: 1.1.0
## smart-open: 5.1.0
## sniffio: 1.2.0
## snowballstemmer: 2.2.0
## sortedcollections: 2.1.0
## sortedcontainers: 2.4.0
## soupsieve: 2.3.1
## sphinx: 4.4.0
## sphinxcontrib-applehelp: 1.0.2
## sphinxcontrib-devhelp: 1.0.2
## sphinxcontrib-htmlhelp: 2.0.0
## sphinxcontrib-jsmath: 1.0.1
## sphinxcontrib-qthelp: 1.0.3
## sphinxcontrib-serializinghtml: 1.1.5
## spyder: 5.1.5
## spyder-kernels: 2.1.3
## sqlalchemy: 1.4.32
## stack-data: 0.2.0
## statsmodels: 0.14.4
## sympy: 1.13.3
## tables: 3.6.1
## tabulate: 0.8.9
## tbb: 0.2
## tblib: 1.7.0
## tenacity: 8.0.1
## tensorboard: 2.19.0
## tensorboard-data-server: 0.7.2
## tensorflow: 2.19.0
## tensorflow-io-gcs-filesystem: 0.31.0
## termcolor: 3.1.0
## terminado: 0.13.1
## testpath: 0.5.0
## text-unidecode: 1.3
## textdistance: 4.2.1
## threadpoolctl: 3.6.0
## three-merge: 0.1.1
## tifffile: 2021.7.2
## tinycss: 0.4
## tldextract: 3.2.0
## toml: 0.10.2
## tomli: 1.2.2
## toolz: 0.11.2
## torch: 2.7.0+cu128
## torchaudio: 2.7.0+cu128
## torchvision: 0.22.0+cu128
## tornado: 6.1
## tqdm: 4.64.0
## trackpy: 0.5.0
## traitlets: 5.1.1
## twisted: 22.2.0
## twisted-iocpsupport: 1.0.2
## typed-ast: 1.4.3
## typing-extensions: 4.13.1
## ujson: 5.1.0
## unidecode: 1.2.0
## urllib3: 1.26.9
## w3lib: 1.21.0
## watchdog: 2.1.6
## wcwidth: 0.2.5
## webencodings: 0.5.1
## websocket-client: 0.58.0
## werkzeug: 2.0.3
## wheel: 0.37.1
## widgetsnbextension: 3.5.2
## win-inet-pton: 1.1.0
## win-unicode-console: 0.5
## wincertstore: 0.2
## wrapt: 1.12.1
## xarray: 0.20.1
## xgboost: 2.1.4
## xlrd: 2.0.1
## xlsxwriter: 3.0.3
## xlwings: 0.24.9
## yapf: 0.31.0
## yarl: 1.6.3
## zict: 2.0.0
## zipp: 3.7.0
## zope.interface: 5.4.0
```

# 代码简洁版

``` python
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape, Flatten

# 定义一个函数，对输入的训练集、验证集和测试集进行数据归一化处理
def normalize_dataframe(train_set, val_set, test_set):
    # 初始化`MinMaxScaler`对象，用于将数据归一化到 [0, 1] 范围
    scaler = MinMaxScaler()
    # 在训练集上拟合归一化模型，计算每个特征的最小值和最大值
    # 这一步不会对训练集，仅记录归一化参数
    scaler.fit(train_set)
    # 使用训练集拟合的归一化模型对训练集、验证集和测试集进行转换
    # 转换后的数据保持原有的列名和索
    train = pd.DataFrame(scaler.transform(train_set), columns = train_set.columns, index = train_set.index)
    val = pd.DataFrame(scaler.transform(val_set), columns = val_set.columns, index = val_set.index)
    test = pd.DataFrame(scaler.transform(test_set), columns = test_set.columns, index = test_set.index)
    
    # 返回归一化后的训练集、验证集和测试集
    return train, val, test

# 定义一个函数，用于准备时间序列数据，将其转换为适合模型输入的格式
def prepare_data(data, win_size):
    X = []  # 存储输入特征（时间窗口内的数据）
    y = []  # 存储目标值（时间窗口后的数据）

    # 遍历数据，创建时间窗口大小为 win_size 的输入和对应的目标值
    for i in range(len(data) - win_size):
        # 提取一个时间窗口的数据作为输入
        temp_x = data[i:i + win_size]
        # 提取时间窗口后的数据作为目标值
        temp_y = data[i + win_size]    
        X.append(temp_x)
        y.append(temp_y)
        
    # 将列表转换为 numpy 数组，便于后续模型输入
    X = np.asarray(X)
    y = np.asarray(y)
    X = np.expand_dims(X, axis = -1)
    
    # 返回输入特征和目标值
    return X, y



if __name__ == '__main__':
    
    # 全局环境变量
    win_size = 30                 # 准备时间序列数据，设置时间窗口大小为 30
    epoch_size = 10               # 设置 epoch 次数为 10（这里测试设置值较小，具体根据实际设置）
    batch_size = 32               # 设置批量大小
    verbose = 0                   # 是否打印中间过程，0 表示静默状态
    train_ratio = 0.7             # 训练集比例
    val_ratio = 0.1               # 验证集比例
    test_ratio = 0.2              # 测试集比例
    
    # 设置工作目录
    wkdir = 'E:/BaiduSyncdisk/005.Bioinformatics/Bioinformatics/src/250510_bilstm_weather'
    os.chdir(wkdir)
    
    # 设置随机种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)   
    
    # 增强 TensorFlow 的确定性
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # 加载数据
    df = pd.read_csv('data/weather.csv')
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Day'].astype(str), format = '%Y-%j')
    df.set_index('Date', inplace = True)
    df.drop(['Year', 'Day'], axis = 1, inplace = True)
    col = 'Temperature'
    df = df[[col]]
    
    # 生成时间范围
    start_date = pd.Timestamp('1990-01-01')
    end_date = pd.Timestamp('2023-03-01')
    date_range = pd.date_range(start = start_date, end = end_date, freq = 'D')
    
    # 检查时间范围中是否包含 DataFrame 中的所有日期
    missing_dates = date_range[~date_range.isin(df.index)]
    print("Missing Dates:")
    print(missing_dates)

    # 可视化数据集
    plt.figure(figsize = (15, 5))
    plt.plot(df[col], color = '#00A087',  alpha = 0.3)
    plt.title('')
    plt.xticks(rotation = 0)
    plt.show()

    # 数据处理
    # 计算划分的索引
    train_split = int(train_ratio * len(df))
    val_split = int((train_ratio + val_ratio) * len(df))
    
    # 划分数据集
    train_set = df.iloc[:train_split]
    val_set = df.iloc[train_split:val_split]
    test_set = df.iloc[val_split:]

    # 可视化训练集, 验证集和测试集数据
    plt.figure(figsize = (15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(train_set, color = 'g',  alpha = 0.3)
    plt.title('Training Data')
    
    plt.subplot(3, 1, 2)
    plt.plot(val_set, color = 'b',  alpha = 0.3)
    plt.title('Validation Data')
    
    plt.subplot(3, 1, 3)
    plt.plot(test_set, color = 'r',  alpha = 0.3)
    plt.title('Testing Data')
    plt.xticks(rotation = 0)
    plt.show()
    
    # 对训练集, 验证集和测试集进行归一化处理
    train, val, test = normalize_dataframe(train_set, val_set, test_set)
    
    # 可视化归一化后的训练集, 验证集和测试集数据
    plt.figure(figsize = (15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(train, color = 'g',  alpha = 0.3)
    plt.title('Training Data')
    
    plt.subplot(3, 1, 2)
    plt.plot(val, color = 'b',  alpha = 0.3)
    plt.title('Validation Data')
    
    plt.subplot(3, 1, 3)
    plt.plot(test, color = 'r',  alpha = 0.3)
    plt.title('Testing Data')
    plt.xticks(rotation = 0)
    plt.show()
    
    # 训练集
    X_train, y_train = prepare_data(train['Temperature'].values, win_size)
    
    # 验证集
    X_val, y_val= prepare_data(val['Temperature'].values, win_size)
    
    # 测试集
    X_test, y_test = prepare_data(test['Temperature'].values, win_size)
    
    df_max = list(np.max(train_set))[0]
    df_min = list(np.min(train_set))[0]

    print("训练集形状:", X_train.shape, y_train.shape)
    print("验证集形状:", X_val.shape, y_val.shape)
    print("测试集形状:", X_test.shape, y_test.shape)
      
    # 双向长短时记忆网络
    if False:
        
        # 创建一个顺序模型
        model = Sequential()
        # 添加双向 LSTM 层，128 个单元，激活函数为 relu，输入形状为 (时间窗口大小, 特征数量)
        model.add(Bidirectional(LSTM(128, activation = 'relu'), input_shape = (X_train.shape[1], X_train.shape[2])))
        # 添加全连接层，64 个神经元，relu 激活函数
        model.add(Dense(64, activation = 'relu'))
        # 添加全连接层，32 个神经元，relu 激活函数
        model.add(Dense(32, activation = 'relu'))
        # 添加全连接层，16 个神经元，relu 激活函数
        model.add(Dense(16, activation = 'relu'))
        # 输出层，1 个神经元，用于预测单个数值；使用 sigmoid 激活函数，将输出限制在 0 到 1 之间
        model.add(Dense(1, activation = 'sigmoid'))
    
    # 混合神经网络模型
    else:
        
        # 创建一个顺序模型
        model = Sequential()
        # 添加双向长短期记忆层，分别从正向和反向处理输入序列，捕捉序列中前后依赖关系
        # 每个方向有 128 个隐藏单元，因此总共 256 个隐藏单元
        # 使用激活函数 ReLU，引入非线性，增强模型的学习能力
        model.add(Bidirectional(LSTM(128, activation = 'relu'), input_shape = (X_train.shape[1], X_train.shape[2])))
        # 添加重塑层，将`Bi-LSTM`的输出重塑为形状为`(256, 1)`的二维张量
        # 这一步是为了将`Bi-LSTM`输出调整为适合后续`1D CNN`层处理的形状
        model.add(Reshape((256, 1)))
        # 添加一维卷积层`Conv1D`，64 个卷积核（过滤器），每个卷积核会提取不同的特征
        # 卷积核的大小为 7，表示每次卷积操作覆盖 7 个时间步（适用于时间序列数据）
        # 使用`ReLU`激活函数，引入非线性，增强模型的学习能力
        model.add(Conv1D(filters = 64, kernel_size = 7, activation = 'relu'))
        # 添加一维最大池化层，池化窗口大小为 2，表示将输入数据的大小减半（下采样），提取主要特征，减少计算量
        model.add(MaxPooling1D(pool_size = 2))
        # 添加展平层，将多维输入，例如卷积层输出的特征图展平成一维向量，以便后续全连接层处理
        model.add(Flatten())
        # 添加全连接层，32 个神经元，relu 激活函数
        model.add(Dense(32, activation = 'relu'))
        # 添加全连接层，16 个神经元，relu 激活函数
        model.add(Dense(16, activation = 'relu'))
        # 输出层，1 个神经元，用于预测单个数值；使用 sigmoid 激活函数，将输出限制在 0 到 1 之间
        model.add(Dense(1, activation = 'sigmoid'))
        
    # 编译模型，优化器为 adam，损失函数为均方误差 (mse)
    model.compile(optimizer = 'adam', loss = 'mse')
    
    # 训练模型，设置 epoch 次数为 10（这里测试设置值较小，具体根据实际设置），批量大小为 32，使用验证集评估模型
    history = model.fit(X_train, y_train, epochs = epoch_size, batch_size = batch_size, validation_data = (X_val, y_val), verbose = verbose)
    
    # 绘制训练过程中的损失曲线
    plt.figure()
    plt.plot(history.history['loss'], c = 'b', label = 'loss')
    plt.plot(history.history['val_loss'], c = 'g', label = 'val_loss')
    plt.legend()
    plt.show()
    
    # 使用模型对测试集进行预测
    y_pred = model.predict(X_test, verbose = verbose)
    
    # 计算模型性能指标
    # 计算均方误差（MSE）
    mse = metrics.mean_squared_error(y_test, np.array([i for arr in y_pred for i in arr]))
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)
    # 计算平均绝对误差（MAE）
    mae = metrics.mean_absolute_error(y_test, np.array([i for arr in y_pred for i in arr]))
    # 计算 R² 拟合优度
    r2 = r2_score(y_test, np.array([i for arr in y_pred for i in arr]))
    
    print("均方误差 (MSE):", mse)
    print("均方根误差 (RMSE):", rmse)
    print("平均绝对误差 (MAE):", mae)
    print("拟合优度:", r2)

    # 打印模型结构摘要
    model.summary()

    # 取出预测的最后一个时间步的输出作为下一步的输入
    last_output = model.predict(X_test, verbose = verbose)[-1]
    
    # 预测的时间步数
    steps = 10  # 假设向后预测 10 个时间步
    predicted = []
    for i in range(steps):
        # 将最后一个输出加入 X_test，继续向后预测
        input_data = np.append(X_test[-1][1:], last_output).reshape(1, X_test.shape[1], X_test.shape[2])
        # 使用模型进行预测
        next_output = model.predict(input_data, verbose = verbose)
        # 将预测的值加入结果列表
        predicted.append(next_output[0][0])
        last_output = next_output[0]
    
    print("向后预测的值:", predicted)

    series_1 = y_pred*(df_max - df_min) + df_min
    series_2 = np.array(predicted)*(df_max - df_min) + df_min

    plt.figure(figsize = (15,4), dpi = 300)
    
    plt.subplot(3 ,1, 1)
    plt.plot(train_set, color = 'c', label = 'Training Data')
    plt.plot(val_set, color = 'r', label = 'Validation Data')
    plt.plot(test_set, color = 'b', label = 'Testing Data')
    plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
    plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(test_set, color = 'b', label = 'Training Data')
    plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
    plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(test_set, color = 'b', label = 'Training Data')
    plt.plot(pd.date_range(start = '2016-08-12', end = '2023-03-01', freq = 'D'), series_1, color = 'y', label = 'Testing Data Predition')
    plt.plot(pd.date_range(start = '2023-03-02', end = '2023-03-11', freq = 'D'), series_2, color = 'magenta', linestyle = '-.', label = 'Futrue Prediction')
    plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-03-11'))
    plt.legend()
    
    plt.show()
```
