---
title: 跟着顶刊学分析之微生物溯源
date: 2025-05-23 16:45:59
tags: [R]
categories: [[跟着顶刊学分析, 微生物溯源]]
---


<p>
FEAST (fast expectation-maximization for microbial source
tracking)，一个快速准确的微生物来源追溯工具，可以用于医疗保健、公共卫生、环境研究和农业，于
2019 年发表在 <b>Nature Methods</b> 中。
</p>
<p>
<a href='https://doi.org/10.1038/s41592-019-0431-x'>FEAST: fast
expectation-maximization for microbial source tracking</a>
</p>
<p>
分析微生物组数据的组成结构的一个主要挑战是确定其潜在来源，与传统溯源工具`SourceTracker`相比，`FEAST`算法展现出三大显著优势：首先，其计算效率实现<b>30-300</b>倍的突破性提升；其次，突破性支持复杂微生物群落体系解析，可同时处理数千个潜在源环境；更重要的是，在目标群落包含未知来源微生物类群时仍能保持高精度解析能力。
</p>
<p>
本示例参考 <iMeta> <a href='https://doi.org/10.1002/imt2.103'>Linking
biodiversity and ecological function through extensive microeukaryotic
movement across different habitats in six urban parks</a> 的 Figure 3A
和 B。
</p>

<img src="/imgs/10.1002-imt2.103.fig3.ab.png" width="75%" style="display: block; margin: auto;" />

<p style="text-align:justify;font-size:15px;line-height:20px">
Fig.3(A,B) Connectivity of microeukaryotes across different habitats and
potential migrated fungal taxa. Fast expectation-maximization microbial
source tracking analysis for V4 (A) and V9 (B) regions at the
zero-radius operational taxonomic unit (zOTU) level.
</p>

# 加载包和数据

``` r
# install.packages("vegan")
# install.packages("dplyr")
# install.packages("doParallel")
# install.packages("foreach")
# install.packages("reshape2")
# install.packages("ggplot2")
# install.packages("cowplot")
# install.packages("Rcpp")
# install.packages("RcppArmadillo")
# devtools::install_github("cozygene/FEAST")

# 加载所需要的包
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(vegan)))
suppressMessages(suppressWarnings(library(doParallel)))
suppressMessages(suppressWarnings(library(foreach)))
suppressMessages(suppressWarnings(library(mgcv)))
suppressMessages(suppressWarnings(library(reshape2)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(cowplot)))
suppressMessages(suppressWarnings(library(Rcpp)))
suppressMessages(suppressWarnings(library(RcppArmadillo)))

# 数据加载 V4
metadata_file = "data/group_soiltow.csv"
count_matrix = "data/water_soil.csv"

metadata <- read.csv(file = metadata_file, header = TRUE, sep = ",", row.names = 1)
otus <- read.csv(file = count_matrix, header = TRUE, sep = ",", row.names = 1)
otus <- t(as.matrix(otus))

# 查看数据
metadata
##              Env SourceSink id
## DLSs01      soil     Source NA
## DLSs02      soil     Source NA
## DLSs03      soil     Source NA
## DPSs202001  soil     Source NA
## DPSs202002  soil     Source NA
## DPSs202003  soil     Source NA
## HLs202001   soil     Source NA
## HLs202002   soil     Source NA
## HLs202003   soil     Source NA
## SLTs202001  soil     Source NA
## SLTs202002  soil     Source NA
## SLTs202003  soil     Source NA
## SLs01       soil     Source NA
## SLs02       soil     Source NA
## SLs03       soil     Source NA
## XSs202001   soil     Source NA
## XSs202002   soil     Source NA
## XsSs202003  soil     Source NA
## DLSw202001 water       Sink  1
## DLSw202002 water       Sink  2
## DLSw202003 water       Sink  3
## DPSw202001 water       Sink  4
## DPSw202002 water       Sink  5
## DPSw202003 water       Sink  6
## HLw202001  water       Sink  7
## HLw202002  water       Sink  8
## HLw202003  water       Sink  9
## SLTw202001 water       Sink 10
## SLTw202002 water       Sink 11
## SLTw202003 water       Sink 12
## SLw202001  water       Sink 13
## SLw202002  water       Sink 14
## SLw202003  water       Sink 15
## XSw202001  water       Sink 16
## XSw202002  water       Sink 17
## XSw202003  water       Sink 18

dim(otus)
## [1]    36 12266
```

# 加载额外函数

``` r
# default 1000
EM_iterations = 1000

# if you use different sources for each sink, different_sources_flag = 1, otherwise = 0
different_sources_flag = 0

# https://github.com/liuchen92/FEAST
# 有些代码需求调整一下啊，否则可能会报错
# 522 行代码: 
# sinks_rarefy = rarefy(matrix(sinks, nrow = 1), maxdepth = apply(totalsource_old, 1, sum)[1]) (原)
# sinks_rarefy = rarefy(as.data.frame(matrix(sinks, nrow = 1)), maxdepth = apply(totalsource_old, 1, sum)[1]) (改)
# 535 行代码: 
# unknown_source_rarefy = rarefy(matrix(unknown_source, nrow = 1), maxdepth = COVERAGE) (原)
# unknown_source_rarefy = rarefy(as.data.frame(matrix(unknown_source, nrow = 1)), maxdepth = COVERAGE) (改)
source("FEAST_src/src.R") 
```

# 数据处理

确保 OTU 表和元数据配对。

``` r
# Extract only those samples in common between the two tables
common.sample.ids <- intersect(rownames(metadata), rownames(otus))
otus <- otus[common.sample.ids, ]
metadata <- metadata[common.sample.ids, ]

# Double-check that the mapping file and otu table
# had overlapping samples
if (length(common.sample.ids) <= 1) {
  message <- paste(sprintf('Error: there are %d sample ids in common '),
                   'between the metadata file and data table')
  stop(message)
}

if (different_sources_flag == 0) {
  metadata$id[metadata$SourceSink == 'Source'] = NA
  metadata$id[metadata$SourceSink == 'Sink'] = c(1:length(which(metadata$SourceSink == 'Sink')))
}

envs <- metadata$Env
Ids <- na.omit(unique(metadata$id))
Proportions_est <- list()
```

# 进行 FEAST 分析

主要作用是对每个样本进行一次溯源分析，估算各个样本对它的贡献比例。

``` r
# 因为具有随机性，每次运行结果不一样的，可以自行设定随机种子
set.seed(1234175)

for(it in 1:length(Ids)){
  if(different_sources_flag == 1){
    train.ix <- which(metadata$SourceSink == 'Source' & metadata$id == Ids[it])
    test.ix <- which(metadata$SourceSink == 'Sink' & metadata$id == Ids[it])
  } else {
    train.ix <- which(metadata$SourceSink == 'Source')
    test.ix <- which(metadata$SourceSink == 'Sink' & metadata$id == Ids[it])
  }
  
  num_sources <- length(train.ix)
  COVERAGE =  min(rowSums(otus[c(train.ix, test.ix), ])) 
  str(COVERAGE)

  sources <- as.data.frame(as.matrix(rarefy(as.data.frame(otus[train.ix,]), COVERAGE)))
  sinks <- as.data.frame(as.matrix(rarefy(as.data.frame(t(as.matrix(otus[test.ix,]))), COVERAGE)))
  
  
  print(paste("Number of OTUs in the sink sample = ",length(which(sinks > 0))))
  print(paste("Seq depth in the sources and sink samples = ",COVERAGE))
  print(paste("The sink is:", envs[test.ix]))
  
  # Estimate source proportions for each sink
  FEAST_output <- FEAST(source = sources, sinks = t(sinks), env = envs[train.ix], em_itr = EM_iterations, COVERAGE = COVERAGE)
  Proportions_est[[it]] <- FEAST_output$data_prop[, 1]
  
  
  names(Proportions_est[[it]]) <- c(as.character(envs[train.ix]), "unknown")
  
  if(length(Proportions_est[[it]]) < num_sources + 1){
    
    tmp = Proportions_est[[it]]
    Proportions_est[[it]][num_sources] = NA
    Proportions_est[[it]][num_sources + 1] = tmp[num_sources]
  }
  
  print("Source mixing proportions")
  print(Proportions_est[[it]])
}
##  num 86235
## [1] "Number of OTUs in the sink sample =  2395"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 2.373234e-06 1.283929e-05 4.739721e-05 3.146515e-04 1.081324e-01 8.094386e-03 1.579931e-02 2.894069e-02 8.155425e-02 5.445537e-01 2.577812e-02 1.171612e-04 1.067092e-04 5.044292e-05 5.457985e-05 
##         soil         soil         soil      unknown 
## 2.532620e-04 1.622015e-02 1.829376e-03 1.681383e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2523"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 3.889828e-05 1.525195e-06 1.117246e-04 8.038632e-03 3.785298e-02 1.637728e-02 2.741566e-04 5.241198e-03 1.778192e-02 5.361979e-01 1.270950e-01 1.933521e-04 1.409010e-05 2.483056e-05 6.800795e-05 
##         soil         soil         soil      unknown 
## 6.318950e-04 2.775094e-05 1.122462e-05 2.500176e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  3021"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 7.988138e-05 3.808868e-05 5.778474e-05 1.483824e-02 1.081487e-02 1.752882e-02 1.775024e-03 6.237512e-03 3.281048e-03 7.227866e-01 7.215726e-02 6.936103e-02 6.509192e-05 3.245632e-05 1.286299e-04 
##         soil         soil         soil      unknown 
## 4.601348e-03 3.259272e-04 8.695947e-04 7.502082e-02 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2002"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 2.467743e-06 2.016527e-10 4.500065e-05 4.168413e-04 1.203435e-03 2.734859e-05 1.818273e-03 1.289035e-01 3.260235e-02 7.976899e-02 5.357446e-02 1.388128e-04 1.001814e-04 5.249166e-05 5.938489e-05 
##         soil         soil         soil      unknown 
## 3.957446e-03 7.982142e-03 4.681435e-04 6.888788e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2392"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 1.858930e-04 5.521816e-14 2.371067e-07 2.944044e-05 8.386825e-04 8.174583e-05 3.547330e-02 4.727666e-02 3.462112e-02 9.586780e-02 3.623000e-02 5.211156e-02 7.633941e-05 1.634543e-05 1.435384e-04 
##         soil         soil         soil      unknown 
## 1.568299e-03 1.297201e-02 3.820682e-03 6.786864e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2107"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 4.513034e-07 2.757246e-06 3.131089e-05 2.591090e-02 9.040759e-03 1.130738e-03 1.500494e-02 5.851312e-02 3.778950e-02 2.621470e-03 6.125437e-03 1.260032e-04 1.473455e-04 6.207257e-05 1.555516e-04 
##         soil         soil         soil      unknown 
## 2.628223e-02 2.345885e-02 2.983823e-03 7.906127e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  3369"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 4.436293e-05 3.825966e-05 1.115491e-04 2.621842e-03 2.651220e-02 1.566928e-03 2.469641e-04 3.629713e-01 9.435515e-02 7.006173e-04 2.357900e-03 3.633220e-05 1.370690e-04 4.229128e-05 1.553882e-04 
##         soil         soil         soil      unknown 
## 2.284871e-04 9.111208e-02 1.046303e-03 4.157149e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2153"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 5.060619e-07 4.238346e-15 5.133635e-05 1.658001e-04 2.266642e-02 8.339385e-05 3.910068e-05 2.573275e-01 3.205428e-02 1.135226e-01 5.447708e-02 2.355611e-04 1.170099e-04 5.791111e-05 1.262617e-04 
##         soil         soil         soil      unknown 
## 9.523774e-04 3.786437e-02 3.116843e-04 4.799469e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2753"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 4.592451e-05 1.220534e-13 9.039803e-05 3.166722e-04 2.311432e-02 2.377031e-04 3.051420e-04 1.298119e-01 4.787729e-03 1.662530e-01 1.319061e-01 6.320147e-02 2.900905e-05 1.109139e-04 1.697483e-04 
##         soil         soil         soil      unknown 
## 1.243189e-03 4.158595e-02 1.827319e-03 4.349636e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2467"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 2.439986e-05 2.789953e-07 3.715871e-05 1.287213e-03 7.165193e-02 1.335127e-03 5.402352e-04 3.238563e-02 3.098788e-02 9.887313e-04 5.687712e-02 2.922136e-04 5.906572e-05 5.781291e-05 1.327281e-04 
##         soil         soil         soil      unknown 
## 2.255157e-02 7.041053e-02 1.151336e-02 6.988670e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2902"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 5.407014e-05 1.438758e-06 2.606040e-05 6.069120e-04 9.981355e-03 4.121905e-02 2.974744e-03 1.120120e-01 2.810943e-03 1.000632e-03 1.162093e-01 9.147944e-05 1.880628e-04 5.221135e-05 1.820410e-04 
##         soil         soil         soil      unknown 
## 1.074081e-02 1.689527e-02 4.834408e-02 6.366095e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2285"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 1.301808e-04 9.903458e-12 4.538408e-05 1.465752e-04 1.243683e-03 4.457985e-05 1.608254e-02 2.246077e-02 7.481728e-02 8.871777e-02 1.188632e-01 2.082201e-04 2.086206e-04 7.437774e-11 1.010761e-04 
##         soil         soil         soil      unknown 
## 2.044466e-03 3.621512e-03 6.038443e-04 6.706603e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2556"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 1.194924e-04 4.529600e-05 4.300092e-05 9.066231e-02 5.483757e-03 1.117409e-05 1.142865e-09 1.652363e-01 8.239933e-04 4.576609e-02 1.193923e-02 4.438130e-05 2.817395e-04 2.137628e-04 2.116386e-06 
##         soil         soil         soil      unknown 
## 4.644909e-02 3.639445e-02 1.567089e-03 5.949167e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2297"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 2.710071e-05 8.426057e-07 2.934029e-05 1.191880e-03 9.199456e-02 3.455579e-03 3.606958e-04 5.188395e-02 2.735425e-02 2.819756e-02 8.750873e-03 6.631289e-05 6.865814e-05 3.795983e-05 9.400427e-05 
##         soil         soil         soil      unknown 
## 8.123896e-04 8.128507e-02 2.019802e-03 7.023692e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2717"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 8.989832e-06 2.123293e-05 3.581263e-05 2.308722e-03 1.537354e-02 3.373665e-02 6.636455e-04 1.452912e-01 9.528756e-03 4.310366e-02 3.402126e-02 1.122007e-04 1.416246e-04 1.932541e-05 1.483264e-04 
##         soil         soil         soil      unknown 
## 7.846299e-04 6.163245e-02 3.336177e-02 6.197062e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2458"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 6.417801e-07 3.597128e-05 5.722292e-05 1.919292e-04 1.393938e-04 3.378286e-05 7.402761e-07 1.108268e-01 1.302255e-01 5.234308e-03 1.187145e-01 1.063309e-04 9.705875e-05 1.156402e-05 1.933701e-04 
##         soil         soil         soil      unknown 
## 2.688407e-04 2.020866e-04 4.871850e-05 6.336113e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  3080"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 3.757648e-05 2.090283e-05 1.189458e-04 1.664155e-04 7.746612e-05 1.275432e-07 1.274463e-03 2.104089e-01 1.089801e-01 2.587192e-02 9.553513e-02 6.227267e-02 1.108829e-05 8.079599e-05 2.196598e-04 
##         soil         soil         soil      unknown 
## 1.950334e-03 8.330172e-04 1.863436e-04 4.919541e-01 
##  num 86235
## [1] "Number of OTUs in the sink sample =  2926"
## [1] "Seq depth in the sources and sink samples =  86235"
## [1] "The sink is: water"
## [1] "Source mixing proportions"
##         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil         soil 
## 2.475600e-06 1.392951e-05 1.136505e-04 3.813466e-02 1.235573e-04 3.864183e-05 3.043981e-05 1.733159e-01 5.248783e-02 5.558045e-05 1.376765e-02 1.804816e-04 1.151054e-04 2.018138e-04 4.607440e-05 
##         soil         soil         soil      unknown 
## 4.865664e-02 3.864785e-04 2.030563e-04 6.721260e-01
```

# 估算贡献度

``` r
# 输出结果，获得贡献度
# 1 - 最后一行结果数据的平均值
res = data.frame(Proportions_est)
print(1 - mean(unlist(res[nrow(res), ])))
## [1] 0.4609555
```

# 写出到文件

``` r
write.csv(Proportions_est, file = "res.csv", quote = FALSE)
```

# 版本信息

``` r
sessionInfo()
## R version 4.4.3 (2025-02-28 ucrt)
## Platform: x86_64-w64-mingw32/x64
## Running under: Windows 11 x64 (build 26100)
## 
## Matrix products: default
## 
## 
## locale:
## [1] LC_COLLATE=Chinese (Simplified)_China.utf8  LC_CTYPE=Chinese (Simplified)_China.utf8    LC_MONETARY=Chinese (Simplified)_China.utf8 LC_NUMERIC=C                               
## [5] LC_TIME=Chinese (Simplified)_China.utf8    
## 
## time zone: Asia/Shanghai
## tzcode source: internal
## 
## attached base packages:
## [1] parallel  stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
##  [1] ggrepel_0.9.6          RcppArmadillo_14.4.1-1 Rcpp_1.0.14            cowplot_1.1.3          ggplot2_3.5.1          reshape2_1.4.4         mgcv_1.9-1             nlme_3.1-167          
##  [9] doParallel_1.0.17      iterators_1.0.14       foreach_1.5.2          vegan_2.6-10           lattice_0.22-6         permute_0.9-7          dplyr_1.1.4           
## 
## loaded via a namespace (and not attached):
##  [1] Matrix_1.7-2      gtable_0.3.6      compiler_4.4.3    tidyselect_1.2.1  stringr_1.5.1     cluster_2.1.8     scales_1.3.0      splines_4.4.3     yaml_2.3.10       fastmap_1.2.0    
## [11] R6_2.6.1          plyr_1.8.9        generics_0.1.4    knitr_1.50        MASS_7.3-64       tibble_3.2.1      munsell_0.5.1     pillar_1.10.2     rlang_1.1.5       stringi_1.8.7    
## [21] xfun_0.51         cli_3.6.4         withr_3.0.2       magrittr_2.0.3    digest_0.6.37     grid_4.4.3        rstudioapi_0.17.1 lifecycle_1.0.4   vctrs_0.6.5       evaluate_1.0.3   
## [31] glue_1.8.0        codetools_0.2-20  colorspace_2.1-1  rmarkdown_2.29    tools_4.4.3       pkgconfig_2.0.3   htmltools_0.5.8.1
```

# 代码简洁版

``` r
# 加载所需要的包
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(vegan)))
suppressMessages(suppressWarnings(library(doParallel)))
suppressMessages(suppressWarnings(library(foreach)))
suppressMessages(suppressWarnings(library(mgcv)))
suppressMessages(suppressWarnings(library(reshape2)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(cowplot)))
suppressMessages(suppressWarnings(library(Rcpp)))
suppressMessages(suppressWarnings(library(RcppArmadillo)))

# 切换工作目录
wkdir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wkdir)

# 数据加载 V4
metadata_file = "data/group_soiltow.csv"
count_matrix = "data/water_soil.csv"

metadata <- read.csv(file = metadata_file, header = TRUE, sep = ",", row.names = 1)
otus <- read.csv(file = count_matrix, header = TRUE, sep = ",", row.names = 1)
otus <- t(as.matrix(otus))

# default 1000
EM_iterations = 1000

# if you use different sources for each sink, different_sources_flag = 1, otherwise = 0
different_sources_flag = 0

# https://github.com/liuchen92/FEAST
source("FEAST_src/src.R") 

# Extract only those samples in common between the two tables
common.sample.ids <- intersect(rownames(metadata), rownames(otus))
otus <- otus[common.sample.ids, ]
metadata <- metadata[common.sample.ids, ]

# Double-check that the mapping file and otu table
# had overlapping samples
if (length(common.sample.ids) <= 1) {
  message <- paste(sprintf('Error: there are %d sample ids in common '),
                   'between the metadata file and data table')
  stop(message)
}

if (different_sources_flag == 0) {
  metadata$id[metadata$SourceSink == 'Source'] = NA
  metadata$id[metadata$SourceSink == 'Sink'] = c(1:length(which(metadata$SourceSink == 'Sink')))
}

envs <- metadata$Env
Ids <- na.omit(unique(metadata$id))
Proportions_est <- list()

# 因为具有随机性，每次运行结果不一样的，可以自行设定随机种子
set.seed(1234175)

for(it in 1:length(Ids)){
  if(different_sources_flag == 1){
    train.ix <- which(metadata$SourceSink == 'Source' & metadata$id == Ids[it])
    test.ix <- which(metadata$SourceSink == 'Sink' & metadata$id == Ids[it])
  } else {
    train.ix <- which(metadata$SourceSink == 'Source')
    test.ix <- which(metadata$SourceSink == 'Sink' & metadata$id == Ids[it])
  }
  
  num_sources <- length(train.ix)
  COVERAGE =  min(rowSums(otus[c(train.ix, test.ix), ])) 
  str(COVERAGE)
  
  sources <- as.data.frame(as.matrix(rarefy(as.data.frame(otus[train.ix,]), COVERAGE)))
  sinks <- as.data.frame(as.matrix(rarefy(as.data.frame(t(as.matrix(otus[test.ix,]))), COVERAGE)))
  
  
  print(paste("Number of OTUs in the sink sample = ",length(which(sinks > 0))))
  print(paste("Seq depth in the sources and sink samples = ",COVERAGE))
  print(paste("The sink is:", envs[test.ix]))
  
  # Estimate source proportions for each sink
  FEAST_output <- FEAST(source = sources, sinks = t(sinks), env = envs[train.ix], em_itr = EM_iterations, COVERAGE = COVERAGE)
  Proportions_est[[it]] <- FEAST_output$data_prop[, 1]
  
  
  names(Proportions_est[[it]]) <- c(as.character(envs[train.ix]), "unknown")
  
  if(length(Proportions_est[[it]]) < num_sources + 1){
    
    tmp = Proportions_est[[it]]
    Proportions_est[[it]][num_sources] = NA
    Proportions_est[[it]][num_sources + 1] = tmp[num_sources]
  }
  
  print("Source mixing proportions")
  print(Proportions_est[[it]])
}

# 输出结果，获得贡献度
# 1 - 最后一行结果数据的平均值
res = data.frame(Proportions_est)
print(1 - mean(unlist(res[nrow(res), ])))

# 写出到文件
write.csv(Proportions_est, file = "res.csv", quote = FALSE)
```
