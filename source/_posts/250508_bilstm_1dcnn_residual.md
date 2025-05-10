---
title: 基于长短期记忆神经网络和一维卷积神经网络以及残差网络进行时间序列预测
date: 2025-05-10 11:16:07
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
reticulate::use_python("C:/ProgramData/Anaconda3.2019.07/python.exe")

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
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Add
```

# 自定义函数

``` python
# 定义一个函数，用于检测是否支持 GPU 加速运算
def check_tensorflow_gpu():
    print("TensorFlow 版本:", tf.__version__)
    if tf.test.is_gpu_available():
        print("GPU is available")
    else:
        print("GPU is not available, using CPU")

# 定义一个函数，用于对数据进行归一化处理
def normalize_dataframe(DFTrain, DFTest):
    # 创建 MinMaxScaler 对象，用于将数据归一化到 [0, 1] 范围
    scaler = MinMaxScaler()
    # 在训练集上拟合归一化模型，计算每个特征的最小值和最大值
    scaler.fit(DFTrain)
    # 对训练集和测试集应用归一化变换，并保留原始数据的列名和索引
    train_data = pd.DataFrame(scaler.transform(DFTrain), columns = DFTrain.columns, index = DFTrain.index)
    test_data = pd.DataFrame(scaler.transform(DFTest), columns = DFTest.columns, index = DFTest.index)
    
    # 返回归一化后的训练集和测试集
    return train_data, test_data

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
    
    # 返回输入特征和目标值
    return X, y
```

# 全局环境变量

``` python
# 全局环境变量
win_size = 30                 # 准备时间序列数据，设置时间窗口大小为 30
tra_val_ratio = 0.7           # 测试和训练集比例
epoch_size = 10               # 设置 epoch 次数为 10（这里测试设置值较小，具体根据实际设置）
batch_size = 32               # 设置批量大小
verbose = 0                   # 是否打印中间过程，0 表示静默状态
```

# 设置随机种子

``` python
# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

# 增强 TensorFlow 的确定性
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

# 加载数据

``` python
# 读取 Excel 文件，设置第一列为索引，并解析日期列
DF = pd.read_excel('data/data.xlsx', index_col = 0, parse_dates = ['日期'])
col = '平均氣溫'
# 提取`平均氣溫`列作为研究对象
DF = DF[[col]]
# 划分训练集和测试集
DFTrain = DF[DF.index < '2020-01-01']
DFTest = DF[DF.index >= '2020-01-01']
```

# 可视化训练集和测试集

``` python
plt.figure(figsize = (15, 5))
plt.subplot(1, 2, 1)
plt.plot(DFTrain[col], color = 'b',  alpha = 0.5)
plt.title('Train Data')
plt.xticks(rotation = 0)
## (array([728659., 730120., 731581., 733042., 734503., 735964., 737425.]), <a list of 7 Text xticklabel objects>)
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(DFTest[col], color = 'r',  alpha = 0.5)
plt.title('Test Data')
plt.grid(True)
plt.xticks(rotation = 0)
## (array([737060., 737425., 737791., 738156., 738521., 738886., 739252.]), <a list of 7 Text xticklabel objects>)
plt.show()
```

![](/imgs/cd351202b06b002b19db4737712972e2.png)
# 数据处理和划分

``` python
# 对训练集和测试集进行归一化处理
data_train, data_test = normalize_dataframe(DFTrain, DFTest)

# 查看训练数据
data_train.head()

# 查看测试数据
##                 平均氣溫
## 日期                  
## 2000-01-01  0.496575
## 2000-01-02  0.554795
## 2000-01-03  0.506849
## 2000-01-04  0.503425
## 2000-01-05  0.530822
data_test.head()

# 准备时间序列数据，设置时间窗口大小为 30
##                 平均氣溫
## 日期                  
## 2020-01-01  0.469178
## 2020-01-02  0.476027
## 2020-01-03  0.510274
## 2020-01-04  0.506849
## 2020-01-05  0.547945
win_size = 30

# 准备时间序列数据，设置时间窗口大小为 30
X, y = prepare_data(data_train.values, win_size)

# 划分训练集和验证集，70% 为训练集，30% 为验证集
train_size = int(len(X) * 0.7)  

# 划分训练集和验证集的输入特征
X_train, X_val = X[:train_size], X[train_size:]

# 划分训练集和验证集的目标值
y_train, y_val = y[:train_size], y[train_size:]

# 准备测试集数据，将测试数据转换为模型输入格式
X_test, y_test = prepare_data(data_test.values, win_size)

# 打印各数据集的形状，便于检查
print("训练集形状:", X_train.shape, y_train.shape)
## 训练集形状: (5092, 30, 1) (5092, 1)
print("验证集形状:", X_val.shape, y_val.shape)
## 验证集形状: (2183, 30, 1) (2183, 1)
print("测试集形状:", X_test.shape, y_test.shape)
## 测试集形状: (1917, 30, 1) (1917, 1)
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
history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data = (X_val, y_val), verbose = 0) 
```

### 绘制训练过程中的损失曲线

``` python
plt.figure()
plt.plot(history.history['loss'], c = 'b', label = 'loss')
plt.plot(history.history['val_loss'], c = 'g', label = 'val_loss')
plt.legend()
plt.show()
```

![](/imgs/9e005e323ba40ef0837440e9531316b4.png)
### 使用模型对测试集进行预测

``` python
y_pred = model.predict(X_test)
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
## 均方误差 (MSE): 0.002590255189090242
print("均方根误差 (RMSE):", rmse)
## 均方根误差 (RMSE): 0.05089454969925799
print("平均绝对误差 (MAE):", mae)
## 平均绝对误差 (MAE): 0.0402230674545089
print("拟合优度:", r2)
## 拟合优度: 0.9180396800901244
```

### 打印模型结构摘要

``` python
model.summary()
## Model: "sequential"
## _________________________________________________________________
## Layer (type)                 Output Shape              Param #   
## =================================================================
## bidirectional (Bidirectional (None, 256)               133120    
## _________________________________________________________________
## dense (Dense)                (None, 64)                16448     
## _________________________________________________________________
## dense_1 (Dense)              (None, 32)                2080      
## _________________________________________________________________
## dense_2 (Dense)              (None, 16)                528       
## _________________________________________________________________
## dense_3 (Dense)              (None, 1)                 17        
## =================================================================
## Total params: 152,193
## Trainable params: 152,193
## Non-trainable params: 0
## _________________________________________________________________
```

## 一维卷积神经网络

<p>
一维卷积神经网络（1D Convolutional Neural Network, 1D
CNN）是一种专门用于处理一维序列数据的深度学习模型。与二维卷积神经网络（2D
CNN）主要用于图像处理不同，1D CNN
适用于时间序列、信号数据或文本序列等一维数据。
</p>
<p>
一维卷积神经网络是一种高效的深度学习模型，特别适合处理时间序列、信号和序列数据。其核心在于通过卷积操作提取局部特征，并结合池化和全连接层实现分类或回归任务。通过合理设计网络结构，它可以在许多序列相关任务中取得优异表现。
</p>
<p>
如果需要处理长期依赖关系，可以将其与`LSTM`或`Transformer`等模型结合使用。
</p>

### 构建 1D CNN 模型

<p>

构建 1D CNN 模型，一维卷积神经网络&lt;/&gt;

``` python
# 创建一个顺序模型
model = Sequential()
# 添加一维卷积层`Conv1D`，64 个卷积核（过滤器），每个卷积核会提取不同的特征
# 卷积核的大小为 7，表示每次卷积操作覆盖 7 个时间步（适用于时间序列数据）
# 使用`ReLU`激活函数，引入非线性，增强模型的学习能力
model.add(Conv1D(filters = 64, kernel_size = 7, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
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
history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data = (X_val, y_val), verbose = 0) 
```

### 绘制训练过程中的损失曲线

``` python
plt.figure()
plt.plot(history.history['loss'], c = 'b', label = 'loss')
plt.plot(history.history['val_loss'], c = 'g', label = 'val_loss')
plt.legend()
plt.show()
```

![](/imgs/3665428409ef7e9edc26c9cb8b5c5ebb.png)
### 使用模型对测试集进行预测

``` python
y_pred = model.predict(X_test)
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
## 均方误差 (MSE): 0.0035650473366629193
print("均方根误差 (RMSE):", rmse)
## 均方根误差 (RMSE): 0.05970801735665755
print("平均绝对误差 (MAE):", mae)
## 平均绝对误差 (MAE): 0.046815115724165815
print("拟合优度:", r2)
## 拟合优度: 0.8871955082119273
```

### 打印模型结构摘要

``` python
model.summary()
## Model: "sequential_1"
## _________________________________________________________________
## Layer (type)                 Output Shape              Param #   
## =================================================================
## conv1d (Conv1D)              (None, 24, 64)            512       
## _________________________________________________________________
## max_pooling1d (MaxPooling1D) (None, 12, 64)            0         
## _________________________________________________________________
## flatten (Flatten)            (None, 768)               0         
## _________________________________________________________________
## dense_4 (Dense)              (None, 32)                24608     
## _________________________________________________________________
## dense_5 (Dense)              (None, 16)                528       
## _________________________________________________________________
## dense_6 (Dense)              (None, 1)                 17        
## =================================================================
## Total params: 25,665
## Trainable params: 25,665
## Non-trainable params: 0
## _________________________________________________________________
```

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
history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data = (X_val, y_val), verbose = 0) 
```

### 绘制训练过程中的损失曲线

``` python
plt.figure()
plt.plot(history.history['loss'], c = 'b', label = 'loss')
plt.plot(history.history['val_loss'], c = 'g', label = 'val_loss')
plt.legend()
plt.show()
```

![](/imgs/f8286a22680fb8040845007811ca7808.png)
### 使用模型对测试集进行预测

``` python
y_pred = model.predict(X_test)
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
## 均方误差 (MSE): 0.0027621990066841262
print("均方根误差 (RMSE):", rmse)
## 均方根误差 (RMSE): 0.0525566266676632
print("平均绝对误差 (MAE):", mae)
## 平均绝对误差 (MAE): 0.03589670057350113
print("拟合优度:", r2)
## 拟合优度: 0.9125990693132883
```

### 打印模型结构摘要

``` python
model.summary()
## Model: "sequential_2"
## _________________________________________________________________
## Layer (type)                 Output Shape              Param #   
## =================================================================
## bidirectional_1 (Bidirection (None, 256)               133120    
## _________________________________________________________________
## reshape (Reshape)            (None, 256, 1)            0         
## _________________________________________________________________
## conv1d_1 (Conv1D)            (None, 250, 64)           512       
## _________________________________________________________________
## max_pooling1d_1 (MaxPooling1 (None, 125, 64)           0         
## _________________________________________________________________
## flatten_1 (Flatten)          (None, 8000)              0         
## _________________________________________________________________
## dense_7 (Dense)              (None, 32)                256032    
## _________________________________________________________________
## dense_8 (Dense)              (None, 16)                528       
## _________________________________________________________________
## dense_9 (Dense)              (None, 1)                 17        
## =================================================================
## Total params: 390,209
## Trainable params: 390,209
## Non-trainable params: 0
## _________________________________________________________________
```

## 混合神经网络模型+残差网络

<p>
结合双向长短期记忆网络（Bidirectional LSTM,
Bi-LSTM）、一维卷积神经网络（1D CNN）以及残差网络（Residual Network,
ResNet）的思想，用于处理时间序列或序列数据的任务（如二分类问题）。Bi-LSTM
擅长捕捉序列中的长期依赖关系和上下文信息，1D CNN
擅长提取局部特征，而残差连接可以缓解深层网络中的梯度消失问题，提升模型训练的稳定性和性能。
</p>

### 构建混合神经网络模型+残差网络

<p>

构建混合神经网络模型，结合了 Bi-LSTM 和 1D CNN 以及残差网络&lt;/&gt;

``` python
# 定义残差块函数
def residual_block(input_layer, filters, kernel_size):
    # 第一个卷积层
    # `filters`，指定卷积核数量，定义特征提取的维度
    # `kernel_size`，卷积核大小，定义每次卷积操作覆盖的时间步长
    # `activation = 'relu'`，激活函数，使用`RuLU`激活函数，引入非线性
    # `padding = 'same'`，使用 same 填充，确保输出形状与输入形状相同，便于残差连接
    residual = Conv1D(filters = filters, kernel_size = kernel_size, activation = 'relu', padding = 'same')(input_layer)
    # 第二个卷积层
    # 继续进行特征处理，参数与第一个卷积层相同
    residual = Conv1D(filters = filters, kernel_size = kernel_size, activation = 'relu', padding = 'same')(residual)
    # 残差连接，将输入层与经过两个卷积层处理的输出相加，形成残差连接
    # 残差拼接有助于缓解梯度消失问题，增强深层网络的训练效果
    residual = Add()([input_layer, residual])
    
    return residual

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
# 获取当前模型的中间输出，用于后续残差块的输入
intermediate_output = model.layers[-1].output
# 调用残差块函数，构建残差块
# 将`MaxPooling1D()`的输入传入残差块
residual_output = residual_block(model.layers[-1].output, filters = 64, kernel_size = 7)
# 对残差块输出进行最大池化操作，继续下采样，进一步减少维度
residual_output = MaxPooling1D(pool_size = 2)(residual_output)
# 添加展平层，将多维输入展平成一维向量，以便后续全连接层处理
residual_output = Flatten()(residual_output)
# 添加全连接层，32 个神经元，relu 激活函数
residual_output = Dense(32, activation = 'relu')(residual_output)
# 添加全连接层，16 个神经元，relu 激活函数
residual_output = Dense(16, activation = 'relu')(residual_output)
# 输出层，1 个神经元，用于预测单个数值；使用 sigmoid 激活函数，将输出限制在 0 到 1 之间
output_layer = Dense(1, activation = 'sigmoid')(residual_output)
# 构建最终模型
# 使用`Model`将整个网络连接起来，允许非顺序结构，如残差连接
model = Model(inputs = model.input, outputs = output_layer)
```

### 编译模型

``` python
# 编译模型，优化器为 adam，损失函数为均方误差 (mse)
model.compile(optimizer = 'adam', loss = 'mse')
```

### 训练模型

``` python
# 训练模型，设置 epoch 次数为 10（这里测试设置值较小，具体根据实际设置），批量大小为 32，使用验证集评估模型
history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data = (X_val, y_val), verbose = 0) 
```

### 绘制训练过程中的损失曲线

``` python
plt.figure()
plt.plot(history.history['loss'], c = 'b', label = 'loss')
plt.plot(history.history['val_loss'], c = 'g', label = 'val_loss')
plt.legend()
plt.show()
```

![](/imgs/14a99056872b20e13ac5c670d8fbc007.png)
### 使用模型对测试集进行预测

``` python
y_pred = model.predict(X_test)
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
## 均方误差 (MSE): 0.002461460275704593
print("均方根误差 (RMSE):", rmse)
## 均方根误差 (RMSE): 0.049613105886495285
print("平均绝对误差 (MAE):", mae)
## 平均绝对误差 (MAE): 0.0344665479800505
print("拟合优度:", r2)
## 拟合优度: 0.9221149821485135
```

### 打印模型结构摘要

``` python
model.summary()
## Model: "model"
## __________________________________________________________________________________________________
## Layer (type)                    Output Shape         Param #     Connected to                     
## ==================================================================================================
## bidirectional_2_input (InputLay [(None, 30, 1)]      0                                            
## __________________________________________________________________________________________________
## bidirectional_2 (Bidirectional) (None, 256)          133120      bidirectional_2_input[0][0]      
## __________________________________________________________________________________________________
## reshape_1 (Reshape)             (None, 256, 1)       0           bidirectional_2[0][0]            
## __________________________________________________________________________________________________
## conv1d_2 (Conv1D)               (None, 250, 64)      512         reshape_1[0][0]                  
## __________________________________________________________________________________________________
## max_pooling1d_2 (MaxPooling1D)  (None, 125, 64)      0           conv1d_2[0][0]                   
## __________________________________________________________________________________________________
## conv1d_3 (Conv1D)               (None, 125, 64)      28736       max_pooling1d_2[0][0]            
## __________________________________________________________________________________________________
## conv1d_4 (Conv1D)               (None, 125, 64)      28736       conv1d_3[0][0]                   
## __________________________________________________________________________________________________
## add (Add)                       (None, 125, 64)      0           max_pooling1d_2[0][0]            
##                                                                  conv1d_4[0][0]                   
## __________________________________________________________________________________________________
## max_pooling1d_3 (MaxPooling1D)  (None, 62, 64)       0           add[0][0]                        
## __________________________________________________________________________________________________
## flatten_2 (Flatten)             (None, 3968)         0           max_pooling1d_3[0][0]            
## __________________________________________________________________________________________________
## dense_10 (Dense)                (None, 32)           127008      flatten_2[0][0]                  
## __________________________________________________________________________________________________
## dense_11 (Dense)                (None, 16)           528         dense_10[0][0]                   
## __________________________________________________________________________________________________
## dense_12 (Dense)                (None, 1)            17          dense_11[0][0]                   
## ==================================================================================================
## Total params: 318,657
## Trainable params: 318,657
## Non-trainable params: 0
## __________________________________________________________________________________________________
```

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
## Python Version: 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)]
## Python Implementation: CPython
## Python Build: ('default', 'Apr 24 2019 15:29:51')
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
## -: portlib-metadata
## -arkupsafe: 1.1.1
## -mportlib-metadata: 0.17
## absl-py: 2.1.0
## alabaster: 0.7.12
## anaconda-client: 1.7.2
## anaconda-navigator: 1.9.7
## anaconda-project: 0.8.3
## asn1crypto: 0.24.0
## astor: 0.8.1
## astroid: 2.2.5
## astropy: 3.2.1
## atomicwrites: 1.3.0
## attrs: 19.1.0
## babel: 2.7.0
## backcall: 0.1.0
## backports.functools-lru-cache: 1.5
## backports.os: 0.1.1
## backports.shutil-get-terminal-size: 1.0.0
## backports.tempfile: 1.0
## backports.weakref: 1.0.post1
## beautifulsoup4: 4.7.1
## bitarray: 0.9.3
## bkcharts: 0.2
## bleach: 3.1.0
## bokeh: 1.2.0
## boto: 2.49.0
## bottleneck: 1.2.1
## cachetools: 5.5.2
## certifi: 2019.6.16
## cffi: 1.12.3
## chardet: 3.0.4
## click: 7.0
## cloudpickle: 1.2.1
## clyent: 1.2.2
## colorama: 0.4.1
## comtypes: 1.1.7
## conda: 4.7.10
## conda-build: 3.18.8
## conda-package-handling: 1.3.11
## conda-verify: 3.4.2
## contextlib2: 0.5.5
## cryptography: 2.7
## cycler: 0.10.0
## cython: 0.29.12
## cytoolz: 0.10.0
## dask: 2.1.0
## decorator: 4.4.0
## defusedxml: 0.6.0
## distributed: 2.1.0
## docutils: 0.14
## entrypoints: 0.3
## et-xmlfile: 1.0.1
## fastcache: 1.1.0
## filelock: 3.0.12
## flask: 1.1.1
## flatbuffers: 25.2.10
## future: 0.17.1
## gast: 0.2.2
## gevent: 1.4.0
## glob2: 0.7
## google-auth: 2.40.1
## google-auth-oauthlib: 0.4.6
## google-pasta: 0.2.0
## greenlet: 0.4.15
## grpcio: 1.62.3
## h5py: 2.9.0
## heapdict: 1.0.0
## html5lib: 1.0.1
## idna: 2.8
## imageio: 2.5.0
## imagesize: 1.1.0
## importlib-metadata: 6.7.0
## ipykernel: 5.1.1
## ipython: 7.6.1
## ipython-genutils: 0.2.0
## ipywidgets: 7.5.0
## isort: 4.3.21
## itsdangerous: 1.1.0
## jdcal: 1.4.1
## jedi: 0.13.3
## jinja2: 2.10.1
## joblib: 0.13.2
## json5: 0.8.4
## jsonschema: 3.0.1
## jupyter: 1.0.0
## jupyter-client: 5.3.1
## jupyter-console: 6.0.0
## jupyter-core: 4.5.0
## jupyterlab: 1.0.2
## jupyterlab-server: 1.0.0
## keras: 2.11.0
## keras-applications: 1.0.8
## keras-preprocessing: 1.1.2
## keyring: 18.0.0
## kiwisolver: 1.1.0
## lazy-object-proxy: 1.4.1
## libarchive-c: 2.8
## llvmlite: 0.29.0
## locket: 0.2.0
## lxml: 4.3.4
## markdown: 3.4.4
## markupsafe: 1.1.1
## matplotlib: 3.1.0
## mccabe: 0.6.1
## menuinst: 1.4.16
## mistune: 0.8.4
## mkl-fft: 1.0.12
## mkl-random: 1.0.2
## mkl-service: 2.0.2
## mock: 3.0.5
## more-itertools: 7.0.0
## mpmath: 1.1.0
## msgpack: 0.6.1
## multipledispatch: 0.6.0
## navigator-updater: 0.2.1
## nbconvert: 5.5.0
## nbformat: 4.4.0
## networkx: 2.3
## nltk: 3.4.4
## nose: 1.3.7
## notebook: 6.0.0
## numba: 0.44.1
## numexpr: 2.6.9
## numpy: 1.16.4
## numpydoc: 0.9.1
## oauthlib: 3.2.2
## olefile: 0.46
## openpyxl: 2.6.2
## opt-einsum: 3.3.0
## packaging: 19.0
## pandas: 0.24.2
## pandocfilters: 1.4.2
## parso: 0.5.0
## partd: 1.0.0
## path.py: 12.0.1
## pathlib2: 2.3.4
## patsy: 0.5.1
## pep8: 1.7.1
## pickleshare: 0.7.5
## pillow: 6.1.0
## pip: 19.1.1
## pkginfo: 1.5.0.1
## pluggy: 0.12.0
## ply: 3.11
## prometheus-client: 0.7.1
## prompt-toolkit: 2.0.9
## protobuf: 3.19.6
## psutil: 5.6.3
## py: 1.8.0
## pyasn1: 0.5.1
## pyasn1-modules: 0.3.0
## pycodestyle: 2.5.0
## pycosat: 0.6.3
## pycparser: 2.19
## pycrypto: 2.6.1
## pycurl: 7.43.0.3
## pyflakes: 2.1.1
## pygments: 2.4.2
## pylint: 2.3.1
## pyodbc: 4.0.26
## pyopenssl: 19.0.0
## pyparsing: 2.4.0
## pyreadline: 2.1
## pyrsistent: 0.14.11
## pysocks: 1.7.0
## pytest: 5.0.1
## pytest-arraydiff: 0.3
## pytest-astropy: 0.5.0
## pytest-doctestplus: 0.3.0
## pytest-openfiles: 0.3.2
## pytest-remotedata: 0.3.1
## python-dateutil: 2.8.0
## pytz: 2019.1
## pywavelets: 1.0.3
## pywin32: 223
## pywinpty: 0.5.5
## pyyaml: 5.1.1
## pyzmq: 18.0.0
## qtawesome: 0.5.7
## qtconsole: 4.5.1
## qtpy: 1.8.0
## requests: 2.22.0
## requests-oauthlib: 2.0.0
## rope: 0.14.0
## rsa: 4.9.1
## ruamel-yaml: 0.15.46
## scikit-image: 0.15.0
## scikit-learn: 0.21.2
## scipy: 1.2.1
## seaborn: 0.9.0
## send2trash: 1.5.0
## setuptools: 41.0.1
## simplegeneric: 0.8.1
## singledispatch: 3.4.0.3
## six: 1.12.0
## snowballstemmer: 1.9.0
## sortedcollections: 1.1.2
## sortedcontainers: 2.1.0
## soupsieve: 1.8
## sphinx: 2.1.2
## sphinxcontrib-applehelp: 1.0.1
## sphinxcontrib-devhelp: 1.0.1
## sphinxcontrib-htmlhelp: 1.0.2
## sphinxcontrib-jsmath: 1.0.1
## sphinxcontrib-qthelp: 1.0.2
## sphinxcontrib-serializinghtml: 1.1.3
## sphinxcontrib-websupport: 1.1.2
## spyder: 3.3.6
## spyder-kernels: 0.5.1
## sqlalchemy: 1.3.5
## statsmodels: 0.10.0
## sympy: 1.4
## tables: 3.5.2
## tblib: 1.4.0
## tensorboard: 1.15.0
## tensorboard-data-server: 0.6.1
## tensorboard-plugin-wit: 1.8.1
## tensorflow-estimator: 1.15.1
## tensorflow-gpu: 1.15.5
## termcolor: 2.3.0
## terminado: 0.8.2
## testpath: 0.4.2
## toolz: 0.10.0
## tornado: 6.0.3
## tqdm: 4.32.1
## traitlets: 4.3.2
## typing-extensions: 4.7.1
## unicodecsv: 0.14.1
## urllib3: 1.24.2
## wcwidth: 0.1.7
## webencodings: 0.5.1
## werkzeug: 0.15.4
## wheel: 0.33.4
## widgetsnbextension: 3.5.0
## win-inet-pton: 1.1.0
## win-unicode-console: 0.5
## wincertstore: 0.2
## wrapt: 1.11.2
## xlrd: 1.2.0
## xlsxwriter: 1.1.8
## xlwings: 0.15.8
## xlwt: 1.3.0
## zict: 1.0.0
## zipp: 0.5.1
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
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Add

# 定义一个函数，用于检测是否支持 GPU 加速运算
def check_tensorflow_gpu():
    print("TensorFlow 版本:", tf.__version__)
    if tf.test.is_gpu_available():
        print("GPU is available")
    else:
        print("GPU is not available, using CPU")

# 定义一个函数，用于对数据进行归一化处理
def normalize_dataframe(DFTrain, DFTest):
    # 创建 MinMaxScaler 对象，用于将数据归一化到 [0, 1] 范围
    scaler = MinMaxScaler()
    # 在训练集上拟合归一化模型，计算每个特征的最小值和最大值
    scaler.fit(DFTrain)
    # 对训练集和测试集应用归一化变换，并保留原始数据的列名和索引
    train_data = pd.DataFrame(scaler.transform(DFTrain), columns = DFTrain.columns, index = DFTrain.index)
    test_data = pd.DataFrame(scaler.transform(DFTest), columns = DFTest.columns, index = DFTest.index)
    
    # 返回归一化后的训练集和测试集
    return train_data, test_data

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
    
    # 返回输入特征和目标值
    return X, y



if __name__ == '__main__':
    
    # 全局环境变量
    win_size = 30                 # 准备时间序列数据，设置时间窗口大小为 30
    tra_val_ratio = 0.7           # 测试和训练集比例
    epoch_size = 10               # 设置 epoch 次数为 10（这里测试设置值较小，具体根据实际设置）
    batch_size = 32               # 设置批量大小
    verbose = 0                   # 是否打印中间过程，0 表示静默状态
    
    # 设置工作目录
    wkdir = 'E:/BaiduSyncdisk/005.Bioinformatics/Bioinformatics/src/250508_multiple_timeseries_model'
    os.chdir(wkdir)
    
    # 设置随机种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    
    # 增强 TensorFlow 的确定性
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 加载数据
    # 读取 Excel 文件，设置第一列为索引，并解析日期列
    DF = pd.read_excel('data/data.xlsx', index_col = 0, parse_dates = ['日期'])
    # 提取`平均氣溫`列作为研究对象
    DF = DF[['平均氣溫']]
    # 划分训练集和测试集
    DFTrain = DF[DF.index < '2020-01-01']
    DFTest = DF[DF.index >= '2020-01-01']

    # 可视化训练集和测试集数据
    plt.figure(figsize = (15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(DFTrain['平均氣溫'], color = 'b',  alpha = 0.5)
    plt.title('Train Data')
    plt.xticks(rotation = 0)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(DFTest['平均氣溫'], color = 'r',  alpha = 0.5)
    plt.title('Test Data')
    plt.grid(True)
    plt.xticks(rotation = 0)
    plt.show()

    # 对训练集和测试集进行归一化处理
    data_train, data_test = normalize_dataframe(DFTrain, DFTest)
    
    # 准备时间序列数据，设置时间窗口大小为 30
    X, y = prepare_data(data_train.values, win_size)
    
    # 划分训练集和验证集，当值为 0.7 则表示 70% 为训练集，30% 为验证集
    train_size = int(len(X) * tra_val_ratio)  
    
    # 划分训练集和验证集的输入特征
    X_train, X_val = X[:train_size], X[train_size:]
    
    # 划分训练集和验证集的目标值
    y_train, y_val = y[:train_size], y[train_size:]
    
    # 准备测试集数据，将测试数据转换为模型输入格式
    X_test, y_test = prepare_data(data_test.values, win_size)
    
    # 打印各数据集的形状，便于检查
    print("训练集形状:", X_train.shape, y_train.shape)
    print("验证集形状:", X_val.shape, y_val.shape)
    print("测试集形状:", X_test.shape, y_test.shape)

    # 构建 Bi-LSTM 模型，双向长短时记忆网络
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
    
    # 构建 1D CNN 模型，一维卷积神经网络
    elif False:
        
        # 创建一个顺序模型
        model = Sequential()
        # 添加一维卷积层`Conv1D`，64 个卷积核（过滤器），每个卷积核会提取不同的特征
        # 卷积核的大小为 7，表示每次卷积操作覆盖 7 个时间步（适用于时间序列数据）
        # 使用`ReLU`激活函数，引入非线性，增强模型的学习能力
        model.add(Conv1D(filters = 64, kernel_size = 7, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
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
       
    # 构建混合神经网络模型，结合了 Bi-LSTM 和 1D CNN 两个模型
    elif False:
        
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
        
    # 构建混合神经网络模型，结合了 Bi-LSTM 和 1D CNN 以及残差网络
    else:
        
        # 定义残差块函数
        def residual_block(input_layer, filters, kernel_size):
            # 第一个卷积层
            # `filters`，指定卷积核数量，定义特征提取的维度
            # `kernel_size`，卷积核大小，定义每次卷积操作覆盖的时间步长
            # `activation = 'relu'`，激活函数，使用`RuLU`激活函数，引入非线性
            # `padding = 'same'`，使用 same 填充，确保输出形状与输入形状相同，便于残差连接
            residual = Conv1D(filters = filters, kernel_size = kernel_size, activation = 'relu', padding = 'same')(input_layer)
            # 第二个卷积层
            # 继续进行特征处理，参数与第一个卷积层相同
            residual = Conv1D(filters = filters, kernel_size = kernel_size, activation = 'relu', padding = 'same')(residual)
            # 残差连接，将输入层与经过两个卷积层处理的输出相加，形成残差连接
            # 残差拼接有助于缓解梯度消失问题，增强深层网络的训练效果
            residual = Add()([input_layer, residual])
            
            return residual

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
        # 获取当前模型的中间输出，用于后续残差块的输入
        intermediate_output = model.layers[-1].output
        # 调用残差块函数，构建残差块
        # 将`MaxPooling1D()`的输入传入残差块
        residual_output = residual_block(model.layers[-1].output, filters = 64, kernel_size = 7)
        # 对残差块输出进行最大池化操作，继续下采样，进一步减少维度
        residual_output = MaxPooling1D(pool_size = 2)(residual_output)
        # 添加展平层，将多维输入展平成一维向量，以便后续全连接层处理
        residual_output = Flatten()(residual_output)
        # 添加全连接层，32 个神经元，relu 激活函数
        residual_output = Dense(32, activation = 'relu')(residual_output)
        # 添加全连接层，16 个神经元，relu 激活函数
        residual_output = Dense(16, activation = 'relu')(residual_output)
        # 输出层，1 个神经元，用于预测单个数值；使用 sigmoid 激活函数，将输出限制在 0 到 1 之间
        output_layer = Dense(1, activation = 'sigmoid')(residual_output)
        # 构建最终模型
        # 使用`Model`将整个网络连接起来，允许非顺序结构，如残差连接
        model = Model(inputs = model.input, outputs = output_layer)
    
        
    # 编译模型，优化器为 adam，损失函数为均方误差 (mse)
    model.compile(optimizer = 'adam', loss = 'mse')
    
    # 训练模型，设置 epoch 次数为 10，批量大小为 32，使用验证集评估模型
    history = model.fit(X_train, y_train, epochs = epoch_size, batch_size = batch_size, validation_data = (X_val, y_val), verbose = verbose)
    
    # 绘制训练过程中的损失曲线
    plt.figure()
    plt.plot(history.history['loss'], c = 'b', label = 'loss')
    plt.plot(history.history['val_loss'], c = 'g', label = 'val_loss')
    plt.legend()
    plt.show()
    
    # 使用模型对测试集进行预测
    y_pred = model.predict(X_test)
    
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

    # 获取当前环境信息
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
```
