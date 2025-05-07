---
title: 250424_sci_partial_least_squares_path_model
date: 2025-04-24 10:00:00
tags: [R, 偏最小二乘路径建模, 因果推断]
categories: [[跟着顶刊学分析, 偏最小二乘路径建模]]
---


<p>
偏最小二乘路径建模 (PLS-PM)
是一种结构方程建模方法，主要用于分析具有复杂因果关系的多变量数据，特别是在处理潜在变量（无法直接观测但可以通过其他可观测变量来间接反映的变量）之间的关系时非常有用。它允许研究观测变量（observed
variables）和潜变量（latent variable）之间复杂多元关系的统计方法。
</p>

# 10.1038/s41467-024-53753-w (Fig.3)

这个示例根据[`The biogeography of soil microbiome potential growth rates`](https://doi.org/10.1038/s41467-024-53753-w)，这篇文献整理而来。通过`PLS-PM`方法分析了土壤微生物组潜在生长率的数据，探索了环境因素、土壤特性、微生物特性和群落结构对生长率的影响。代码的主要步骤包括数据加载、变量选择、数据预处理（对数变换和标准化）、模型定义（块和路径）、模型拟合以及结果分析。每个步骤都基于研究的科学假设和统计分析的需求，确保了模型的准确性和可解释性。

<img src="/imgs/10.1038-s41467-024-53753-w.Fig.3.webp" width="75%" style="display: block; margin: auto;" />

<p style="text-align:justify;font-size:15px;line-height:20px">
The partial least squares path model depicts factors influencing
microbial potential growth rates through direct and indirect pathways.
Blue and red arrows indicate positive and negative effects,
respectively, while the indicated values on the arrows are the path
coefficients for the inner model. The path coefficients for outer models
of the partial least squares path modeling are shown in Supplementary
Table 1. C is carbon, N is nitrogen, and P is phosphorus. The soil
microbiome in acid-neutral soils with high organic matter and nutrients
(resource-rich) in humid regions, dominated by Basidiomycota,
Acidobacteriota, and Proteobacteria, exhibits a large genome size and
low biomass C:P and N:P ratios, indicating a high potential growth rate.
Conversely, in resource-poor, dry, and hypersaline soils, the
microbiome, dominated by Ascomycota, Actinobacteriota, and
Gemmatimonadota, displays a lower potential growth rate, suggesting that
resource acquisition and stress tolerance tradeoff with growth rate.
Source data are provided as a Source Data file.
</p>

# 关键要点

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>研究背景</strong>：研究通过偏最小二乘路径建模（PLS-PM）分析土壤微生物组潜在生长率（Gmass）的生物地理分布，揭示环境因素、土壤资源、微生物群落结构及其特性如何通过直接和间接路径影响生长率
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>核心发现</strong>：微生物群落结构通过改变微生物特性（基因组大小、核糖体
RNA
拷贝数、最适温度、生物量化学计量）间接影响潜在生长率，而非直接作用。基因组大小是关键的生命历史特性，影响微生物丰度、生长率和代谢能力
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>环境与生长率关系</strong>：资源丰富、酸中性、湿润地区的土壤微生物组（以
Basidiomycota、Acidobacteriota 和 Proteobacteria
主导）具有较大基因组大小和较低的生物量 C:P 和 N:P
比，表现出高潜在生长率；而资源匮乏、干旱、高盐土壤（以
Ascomycota、Actinobacteriota 和 Gemmatimonadota
主导）则生长率较低，反映资源获取和应激耐受性与生长率的权衡
</li>
</ul>

# 偏最小二乘路径建模

## 加载包和数据

``` r
# https://www.nature.com/articles/s41467-024-53753-w
# 加载所需要的包
suppressMessages(suppressWarnings(library(plspm)))

# 数据加载
growth = data.frame(readxl::read_excel("data/41467_2024_53753_MOESM4_ESM.xlsx"))

# 从`growth`数据框中选择特定的列，创建一个新的数据框`semdata`
semdata = growth[, c("MBCtoP", "MBNtoP", "F_genome_size", "B_genome_size",
                      "optimum_tmp", "rrn_copy_number", "AI", "MAT", "pH",
                      "CS", "SOC", "TN", "TP", "DOC", "AvaiN",
                      "Ascomycota", "Actinobacteriota", "Gemmatimonadota",
                      "Acidobacteriota", "Proteobacteria", "Basidiomycota",
                      "RelGrowth")]
```

<strong style="color:#00A087;font-size:16px;">这些列代表了土壤微生物组生长率分析中所需的关键变量</strong>，包括：

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>微生物特性</strong>：MBCtoP（微生物生物量碳与磷的比值）、MBNtoP（微生物生物量氮与磷的比值）、F_genome_size（真菌基因组大小）、B_genome_size（细菌基因组大小）、optimum_tmp（最适生长温度）、rrn_copy_number（核糖体
RNA 运转数）
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>环境因素</strong>：AI（干燥指数）、MAT（年平均温度）、pH（土壤
pH 值）、CS（可能为黏土含量或土壤结构）
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>土壤资源</strong>：SOC（土壤有机碳）、TN（总氮）、TP（总磷）、DOC（溶解有机碳）、AvaiN（可用氮）
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>微生物群落结构</strong>：Ascomycota、Actinobacteriota、Gemmatimonadota、Acidobacteriota、Proteobacteria、Basidiomycota（这些是微生物门水平的相对丰度）
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>变量</strong>：RelGrowth（相对生长率）
</li>
</ul>
<p>
这些变量的选择基于研究的目标，即探索环境因素、土壤特性、微生物特性和群落结构如何影响土壤微生物组的生长率。
</p>

``` r
# 查看数据
head(semdata)
##     MBCtoP   MBNtoP F_genome_size B_genome_size optimum_tmp rrn_copy_number       AI       MAT   pH       CS       SOC       TN        TP       DOC     AvaiN Ascomycota Actinobacteriota
## 1 19.81672 2.612245      37.82561      2.342668    32.69841        2.292500 27.76859 12.183333 8.10 82.24883 18.010254 1.727460 0.7846446 113.84430 34.246023  0.5945183        0.2435402
## 2 37.14067 3.723636      34.76863      2.863967    30.32043        3.362793 64.87466 17.591667 4.62 79.51402 23.722305 1.324222 0.2232410  44.52458 33.874911  0.1436178        0.3093456
## 3 41.38054 2.620952      36.30179      2.262327    32.01724        2.308163 12.85593  4.779167 8.55 44.78182  7.623631 0.814682 0.2412368  63.31475  3.491716  0.3631388        0.6022700
## 4 43.94107 3.034074      32.90958      2.639655    33.12637        4.198864 54.70828 17.637501 4.43 89.41315 25.342554 1.353268 0.1461584  74.36433 23.948556  0.1671228        0.2585124
## 5 23.25162 1.996916      34.22209      2.876363    31.93342        3.051222 54.21411 15.104167 4.51 82.55045 24.722395 1.465851 0.2445615  74.96242 24.131474  0.2200513        0.2262336
## 6 21.67857 3.095814      37.98236      2.562593    26.35777        1.811798 27.28562  9.387500 5.80 89.41634 18.414808 1.318665 0.2573506  72.04095 23.630484  0.7940347        0.2621750
##   Gemmatimonadota Acidobacteriota Proteobacteria Basidiomycota RelGrowth
## 1     0.043789745      0.26104806     0.20739757    0.02566877 1.5401573
## 2     0.010504709      0.15342510     0.39318200    0.03175714 1.6388373
## 3     0.069910650      0.09281172     0.07135957    0.03112073 0.6328229
## 4     0.003984545      0.19363278     0.29936408    0.01109485 1.2623599
## 5     0.012557353      0.18695162     0.44598728    0.10708755 1.3188750
## 6     0.050028174      0.20051517     0.22349674    0.06519018 1.1602684
```

## 数据处理

``` r
# 对数变换
# 1.减少数据的偏斜度，使数据分布更接近正态分布
# 2.稳定变量的方差，减少异方差性
# 3.线性化非线性关系，便于后续的统计分析
semdata$AI = log(semdata$AI)
semdata$SOC = log(semdata$SOC)
semdata$TN = log(semdata$TN)
semdata$TP = log(semdata$TP)
semdata$DOC = log(semdata$DOC)
semdata$AvaiN = log(semdata$AvaiN)
semdata$RelGrowth = log(semdata$RelGrowth)

# 数据标准化，使每个变量的均值为 0，标准差为 1
# 1.消除变量之间的量纲差异（例如，温度和碳含量单位不同）
# 2.确保所有变量在模型中具有相等的权重，避免某些变量因数值范围大而在分析中占据主导地位
semdata = data.frame(scale(semdata))
```

## 定义潜在变量块

``` r
# 定义潜在变量块
# 定义偏最小二乘路径模型 (PLS-PM) 中的潜在变量块，每个块包含一组相关的观测变量
# 在 PLS-PM 中，潜在变量是不可直接观测的概念，通过一组观测变量来表示
dat_blocks = list(
  MAT = c("MAT"),
  AI = c("AI"),
  Soil = c("CS", "pH"),
  Resources = c("SOC", "TN", "TP", "DOC", "AvaiN"),
  Structure = c("Ascomycota", "Gemmatimonadota", "Actinobacteriota", "Acidobacteriota", "Proteobacteria", "Basidiomycota"),
  Traits = c("MBCtoP", "MBNtoP", "F_genome_size", "B_genome_size", "optimum_tmp", "rrn_copy_number"),
  Growth = "RelGrowth")

# 每个块代表一个特定的潜在概念：
# MAT：年平均温度
# AI：干燥指数
# Soil：土壤特性（包括黏土含量和 pH）
# Resources：土壤资源（包括有机碳、总氮、总磷、溶解有机碳、可用氮）
# Structure：微生物群落结构（包括多个微生物门的相对丰度）
# Traits：微生物特性（包括生物量比值、基因组大小、生长温度等）
# Growth：相对生长率（目标变量）
```

## 定义路径矩阵

``` r
# 定义路径矩阵
MAT = c(0, 0, 0, 0, 0, 0, 0)
AI = c(0, 0, 0, 0, 0, 0, 0)
Soil = c(1, 1, 0, 0, 0, 0, 0)
Resources = c(1, 1, 1, 0, 0, 0, 0)
Structure = c(1, 1, 1, 1, 0, 0, 0)
Traits = c(1, 0, 0, 1, 1, 0, 0)
Growth = c(0, 1, 0, 1, 0, 1, 0)

# 定义潜在变量块之间的路径关系，即内模型
# 路径矩阵是一个方阵，行和列对应于潜在变量块，矩阵中的值表示块之间的关系

dat_path = rbind(MAT, AI, Soil, Resources, Structure, Traits, Growth)
colnames(dat_path) = rownames(dat_path)

dat_path
##           MAT AI Soil Resources Structure Traits Growth
## MAT         0  0    0         0         0      0      0
## AI          0  0    0         0         0      0      0
## Soil        1  1    0         0         0      0      0
## Resources   1  1    1         0         0      0      0
## Structure   1  1    1         1         0      0      0
## Traits      1  0    0         1         1      0      0
## Growth      0  1    0         1         0      1      0
```

<p>
1 表示有从列块到行块的路径（即列块影响行块）； 0 表示没有路径。
</p>
<p>
路径矩阵反映了研究的假设，即环境因素影响土壤特性和资源，进而影响微生物群落结构和特性，最终影响生长率。
</p>

## 拟合 PLS-PM 模型

``` r
# 拟合 PLS-PM 模型
# 使用 plspm 函数拟合偏最小二乘路径模型
# 输出 dat_pls 是一个包含模型结果的对象，包括路径系数、内模型、外模型等信息
dat_pls = plspm(semdata, dat_path, dat_blocks)

# 查看模型输出，提供模型的整体摘要
# 这些输出帮助研究者理解模型的拟合质量、变量之间的关系以及每个观测变量对其所属块的贡献
summary(dat_pls)
## PARTIAL LEAST SQUARES PATH MODELING (PLS-PM) 
## 
## ---------------------------------------------------------- 
## MODEL SPECIFICATION 
## 1   Number of Cases      336 
## 2   Latent Variables     7 
## 3   Manifest Variables   22 
## 4   Scale of Data        Standardized Data 
## 5   Non-Metric PLS       FALSE 
## 6   Weighting Scheme     centroid 
## 7   Tolerance Crit       1e-06 
## 8   Max Num Iters        100 
## 9   Convergence Iters    6 
## 10  Bootstrapping        FALSE 
## 11  Bootstrap samples    NULL 
## 
## ---------------------------------------------------------- 
## BLOCKS DEFINITION 
##         Block         Type   Size   Mode
## 1         MAT    Exogenous      1      A
## 2          AI    Exogenous      1      A
## 3        Soil   Endogenous      2      A
## 4   Resources   Endogenous      5      A
## 5   Structure   Endogenous      6      A
## 6      Traits   Endogenous      6      A
## 7      Growth   Endogenous      1      A
## 
## ---------------------------------------------------------- 
## BLOCKS UNIDIMENSIONALITY 
##            Mode  MVs  C.alpha    DG.rho  eig.1st  eig.2nd
## MAT           A    1    1.000  1.00e+00     1.00    0.000
## AI            A    1    1.000  1.00e+00     1.00    0.000
## Soil          A    2    0.000  8.89e-32     1.45    0.555
## Resources     A    5    0.901  9.29e-01     3.64    0.751
## Structure     A    6    0.000  3.58e-04     3.60    1.078
## Traits        A    6    0.171  2.62e-01     1.84    1.243
## Growth        A    1    1.000  1.00e+00     1.00    0.000
## 
## ---------------------------------------------------------- 
## OUTER MODEL 
##                       weight  loading  communality  redundancy
## MAT                                                           
##   1 MAT                1.000    1.000        1.000      0.0000
## AI                                                            
##   2 AI                 1.000    1.000        1.000      0.0000
## Soil                                                          
##   3 CS                -0.513    0.807        0.651      0.3740
##   3 pH                 0.660   -0.888        0.789      0.4529
## Resources                                                     
##   4 SOC                0.297    0.967        0.934      0.5488
##   4 TN                 0.238    0.949        0.900      0.5287
##   4 TP                 0.131    0.621        0.385      0.2263
##   4 DOC                0.227    0.870        0.757      0.4449
##   4 AvaiN              0.257    0.808        0.652      0.3831
## Structure                                                     
##   5 Ascomycota         0.161   -0.632        0.399      0.2842
##   5 Gemmatimonadota    0.219   -0.769        0.592      0.4214
##   5 Actinobacteriota   0.277   -0.921        0.848      0.6038
##   5 Acidobacteriota   -0.207    0.788        0.621      0.4425
##   5 Proteobacteria    -0.233    0.821        0.674      0.4800
##   5 Basidiomycota     -0.176    0.682        0.465      0.3311
## Traits                                                        
##   6 MBCtoP             0.326   -0.665        0.442      0.1245
##   6 MBNtoP             0.340   -0.660        0.436      0.1227
##   6 F_genome_size     -0.289    0.498        0.248      0.0698
##   6 B_genome_size     -0.260    0.463        0.214      0.0604
##   6 optimum_tmp        0.329   -0.520        0.270      0.0761
##   6 rrn_copy_number   -0.321    0.384        0.148      0.0416
## Growth                                                        
##   7 RelGrowth          1.000    1.000        1.000      0.6036
## 
## ---------------------------------------------------------- 
## CROSSLOADINGS 
##                           MAT       AI    Soil  Resources  Structure   Traits   Growth
## MAT                                                                                   
##   1 MAT                1.0000   0.0377   0.293    -0.1627     0.4280  -0.0816  -0.0746
## AI                                                                                    
##   2 AI                 0.0377   1.0000   0.709     0.6369     0.7051   0.3951   0.7288
## Soil                                                                                  
##   3 CS                 0.1768   0.4629   0.807     0.6744     0.4544   0.2947   0.4457
##   3 pH                -0.3072  -0.7151  -0.888    -0.4635    -0.7861  -0.4017  -0.5733
## Resources                                                                             
##   4 SOC               -0.2003   0.6792   0.683     0.9666     0.4302   0.4311   0.6829
##   4 TN                -0.2327   0.5412   0.552     0.9488     0.2496   0.3443   0.5738
##   4 TP                -0.4131   0.2134   0.152     0.6207     0.0233   0.2755   0.2991
##   4 DOC               -0.2653   0.4537   0.491     0.8703     0.3113   0.3528   0.5077
##   4 AvaiN              0.2591   0.6811   0.724     0.8076     0.5981   0.3626   0.5892
## Structure                                                                             
##   5 Ascomycota        -0.0485  -0.4188  -0.436    -0.3491    -0.6317  -0.3170  -0.3233
##   5 Gemmatimonadota   -0.5070  -0.5525  -0.572    -0.2018    -0.7692  -0.3000  -0.3582
##   5 Actinobacteriota  -0.4432  -0.6770  -0.742    -0.4768    -0.9207  -0.3518  -0.5051
##   5 Acidobacteriota    0.4736   0.5050   0.664     0.2455     0.7882   0.1272   0.3357
##   5 Proteobacteria     0.3895   0.6074   0.558     0.2949     0.8209   0.4182   0.4456
##   5 Basidiomycota     -0.0145   0.4707   0.472     0.3565     0.6817   0.4287   0.4416
## Traits                                                                                
##   6 MBCtoP             0.0887  -0.2236  -0.177    -0.2955    -0.1428  -0.6650  -0.2375
##   6 MBNtoP             0.0816  -0.1654  -0.198    -0.3084    -0.2070  -0.6600  -0.2009
##   6 F_genome_size     -0.0563   0.1492   0.138     0.2307     0.1668   0.4978   0.2246
##   6 B_genome_size      0.4101   0.3704   0.494     0.0989     0.6015   0.4631   0.3204
##   6 optimum_tmp        0.3095  -0.0874  -0.060    -0.1320    -0.0265  -0.5197  -0.3023
##   6 rrn_copy_number   -0.0429   0.3041   0.321     0.2571     0.2586   0.3844   0.1940
## Growth                                                                                
##   7 RelGrowth         -0.0746   0.7288   0.607     0.6456     0.5216   0.4557   1.0000
## 
## ---------------------------------------------------------- 
## INNER MODEL 
## $Soil
##             Estimate   Std. Error    t value   Pr(>|t|)
## Intercept   2.24e-17       0.0358   6.27e-16   1.00e+00
## MAT         2.67e-01       0.0358   7.46e+00   7.49e-13
## AI          6.99e-01       0.0358   1.95e+01   3.40e-57
## 
## $Resources
##              Estimate   Std. Error     t value   Pr(>|t|)
## Intercept    1.83e-18       0.0353    5.19e-17   1.00e+00
## MAT         -3.44e-01       0.0381   -9.02e+00   1.53e-17
## AI           2.33e-01       0.0517    4.51e+00   8.94e-06
## Soil         5.87e-01       0.0540    1.09e+01   9.46e-24
## 
## $Structure
##              Estimate   Std. Error     t value   Pr(>|t|)
## Intercept    1.20e-16       0.0295    4.07e-15   1.00e+00
## MAT          2.75e-01       0.0356    7.72e+00   1.35e-13
## AI           4.78e-01       0.0445    1.07e+01   3.16e-23
## Soil         4.02e-01       0.0526    7.64e+00   2.36e-13
## Resources   -1.07e-01       0.0459   -2.32e+00   2.08e-02
## 
## $Traits
##              Estimate   Std. Error     t value   Pr(>|t|)
## Intercept   -6.38e-17       0.0465   -1.37e-15   1.00e+00
## MAT         -2.31e-01       0.0565   -4.09e+00   5.32e-05
## Resources    2.04e-01       0.0562    3.64e+00   3.15e-04
## Structure    4.28e-01       0.0613    6.98e+00   1.63e-11
## 
## $Growth
##              Estimate   Std. Error     t value   Pr(>|t|)
## Intercept   -3.99e-17       0.0346   -1.15e-15   1.00e+00
## AI           5.03e-01       0.0456    1.10e+01   2.54e-24
## Resources    2.64e-01       0.0461    5.72e+00   2.37e-08
## Traits       1.46e-01       0.0387    3.78e+00   1.87e-04
## 
## ---------------------------------------------------------- 
## CORRELATIONS BETWEEN LVs 
##                MAT      AI   Soil  Resources  Structure   Traits   Growth
## MAT         1.0000  0.0377  0.293     -0.163      0.428  -0.0816  -0.0746
## AI          0.0377  1.0000  0.709      0.637      0.705   0.3951   0.7288
## Soil        0.2934  0.7093  1.000      0.652      0.752   0.4163   0.6069
## Resources  -0.1627  0.6369  0.652      1.000      0.415   0.4196   0.6456
## Structure   0.4280  0.7051  0.752      0.415      1.000   0.4136   0.5216
## Traits     -0.0816  0.3951  0.416      0.420      0.414   1.0000   0.4557
## Growth     -0.0746  0.7288  0.607      0.646      0.522   0.4557   1.0000
## 
## ---------------------------------------------------------- 
## SUMMARY INNER MODEL 
##                  Type     R2  Block_Communality  Mean_Redundancy    AVE
## MAT         Exogenous  0.000              1.000           0.0000  1.000
## AI          Exogenous  0.000              1.000           0.0000  1.000
## Soil       Endogenous  0.574              0.720           0.4134  0.720
## Resources  Endogenous  0.587              0.726           0.4264  0.726
## Structure  Endogenous  0.712              0.600           0.4272  0.600
## Traits     Endogenous  0.282              0.293           0.0825  0.293
## Growth     Endogenous  0.604              1.000           0.6036  1.000
## 
## ---------------------------------------------------------- 
## GOODNESS-OF-FIT 
## [1]  0.5503
## 
## ---------------------------------------------------------- 
## TOTAL EFFECTS 
##              relationships  direct  indirect    total
## 1                MAT -> AI   0.000    0.0000   0.0000
## 2              MAT -> Soil   0.267    0.0000   0.2670
## 3         MAT -> Resources  -0.344    0.1568  -0.1870
## 4         MAT -> Structure   0.275    0.1273   0.4020
## 5            MAT -> Traits  -0.231    0.1337  -0.0977
## 6            MAT -> Growth   0.000   -0.0637  -0.0637
## 7               AI -> Soil   0.699    0.0000   0.6992
## 8          AI -> Resources   0.233    0.4107   0.6439
## 9          AI -> Structure   0.478    0.2124   0.6900
## 10            AI -> Traits   0.000    0.4269   0.4269
## 11            AI -> Growth   0.503    0.2324   0.7353
## 12       Soil -> Resources   0.587    0.0000   0.5874
## 13       Soil -> Structure   0.402   -0.0626   0.3394
## 14          Soil -> Traits   0.000    0.2653   0.2653
## 15          Soil -> Growth   0.000    0.1938   0.1938
## 16  Resources -> Structure  -0.107    0.0000  -0.1066
## 17     Resources -> Traits   0.204   -0.0456   0.1588
## 18     Resources -> Growth   0.264    0.0232   0.2872
## 19     Structure -> Traits   0.428    0.0000   0.4278
## 20     Structure -> Growth   0.000    0.0626   0.0626
## 21        Traits -> Growth   0.146    0.0000   0.1463

# dat_pls$path_coefs           显示路径系数，表示块之间关系的强度和方向
# dat_pls$inner_model          显示内模型（块之间关系）的细节
# dat_pls$inner_summary        总结内模型的统计信息，如 R² 值
# dat_pls$effects              显示总效应（直接效应 + 间接效应）
# dat_pls$outer_model          显示外模型（块与其观测变量的关系）
# dat_pls$gof                  计算模型的良好度指标（Goodness of Fit），通常大于 0.7 表示模型拟合良好
```

## 生成路径图，查看因果关系的路径图

``` r
# 生成路径图，查看因果关系的路径图
innerplot(dat_pls, lcol = 'grey5')
```

![](/imgs/d5dd99ad40c8cf0b315245bf24f74974.png)
## 生成外模型图

``` r
# 生成外模型图
outerplot(dat_pls, lcol = 'grey9', arr.width = 0.05, box.size = 0.1)
```

![](/imgs/db6abd12cbad8750b79bb7ae9ccec4d8.png)
# 综合结论总结

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>微生物群落结构对潜在生长率的影响路径</strong>：PLS-PM
模型表明，微生物群落结构（由主要真菌和细菌门类组成，如
Ascomycota、Basidiomycota、Actinobacteriota
等）通过调节微生物特性（Traits 块，包括基因组大小、rrn
拷贝数、最适温度、生物量化学计量）间接影响潜在生长率（Gmass）。模型中，Structure
到 Traits 的路径系数为 0.428（p &lt; 0.001），Traits 到 Growth
的路径系数为 0.146（p &lt; 0.001），而 Structure 到 Growth
的直接路径不显著（被移除）。这表明微生物群落组成通过塑造功能特性（如代谢能力和生长潜力）间接调控生长率，而非直接决定生长率。这种间接作用强调了微生物生态系统中群落结构与功能特性之间的复杂相互作用
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>基因组大小的关键作用</strong>：真菌和细菌基因组大小（F_genome_size
和 B_genome_size）对潜在生长率有正向影响（外模型载荷分别为 0.498 和
0.463），表明较大基因组具有更广泛的代谢能力，支持更高的生长潜力
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>统计显著性</strong>：大多数路径系数高度显著（p &lt; 0.001），如
AI -&gt; Growth、Resources -&gt; Growth 和 Structure -&gt;
Traits，验证了模型路径的稳健性
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>模型拟合度</strong>：PLS-PM 模型的总体良好度（Goodness-of-Fit,
GoF）为 0.5503，低于理想值 0.7，但仍表明模型具有一定的解释力
</li>
</ul>

# 代码简洁版

``` r
# 加载所需要的包
suppressMessages(suppressWarnings(library(plspm)))

# 数据加载
growth = data.frame(readxl::read_excel("data/41467_2024_53753_MOESM4_ESM.xlsx"))

# 从`growth`数据框中选择特定的列，创建一个新的数据框`semdata`
semdata = growth[, c("MBCtoP", "MBNtoP", "F_genome_size", "B_genome_size",
                      "optimum_tmp", "rrn_copy_number", "AI", "MAT", "pH",
                      "CS", "SOC", "TN", "TP", "DOC", "AvaiN",
                      "Ascomycota", "Actinobacteriota", "Gemmatimonadota",
                      "Acidobacteriota", "Proteobacteria", "Basidiomycota",
                      "RelGrowth")]

# 对数变换
semdata$AI = log(semdata$AI)
semdata$SOC = log(semdata$SOC)
semdata$TN = log(semdata$TN)
semdata$TP = log(semdata$TP)
semdata$DOC = log(semdata$DOC)
semdata$AvaiN = log(semdata$AvaiN)
semdata$RelGrowth = log(semdata$RelGrowth)

# 数据标准化
semdata = data.frame(scale(semdata))

# 定义潜在变量块
dat_blocks = list(
  MAT = c("MAT"),
  AI = c("AI"),
  Soil = c("CS", "pH"),
  Resources = c("SOC", "TN", "TP", "DOC", "AvaiN"),
  Structure = c("Ascomycota", "Gemmatimonadota", "Actinobacteriota", "Acidobacteriota", "Proteobacteria", "Basidiomycota"),
  Traits = c("MBCtoP", "MBNtoP", "F_genome_size", "B_genome_size", "optimum_tmp", "rrn_copy_number"),
  Growth = "RelGrowth")

# 定义路径矩阵
MAT = c(0, 0, 0, 0, 0, 0, 0)
AI = c(0, 0, 0, 0, 0, 0, 0)
Soil = c(1, 1, 0, 0, 0, 0, 0)
Resources = c(1, 1, 1, 0, 0, 0, 0)
Structure = c(1, 1, 1, 1, 0, 0, 0)
Traits = c(1, 0, 0, 1, 1, 0, 0)
Growth = c(0, 1, 0, 1, 0, 1, 0)
dat_path = rbind(MAT, AI, Soil, Resources, Structure, Traits, Growth)
colnames(dat_path) = rownames(dat_path)

# 拟合 PLS-PM 模型
dat_pls = plspm(semdata, dat_path, dat_blocks)

# 查看模型输出，提供模型的整体摘要
summary(dat_pls)

# 生成路径图，查看因果关系的路径图
innerplot(dat_pls, lcol = 'grey5')

# 生成外模型图
outerplot(dat_pls, lcol = 'grey9', arr.width = 0.05, box.size = 0.1)
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
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] plspm_0.5.1
## 
## loaded via a namespace (and not attached):
##  [1] digest_0.6.37     fastmap_1.2.0     cellranger_1.1.0  shape_1.4.6.1     xfun_0.51         readxl_1.4.5      magrittr_2.0.3    turner_0.1.9      glue_1.8.0        tibble_3.2.1     
## [11] knitr_1.50        pkgconfig_2.0.3   htmltools_0.5.8.1 rmarkdown_2.29    lifecycle_1.0.4   cli_3.6.4         amap_0.8-20       vctrs_0.6.5       compiler_4.4.3    rstudioapi_0.17.1
## [21] tools_4.4.3       tester_0.2.0      pillar_1.10.1     evaluate_1.0.3    yaml_2.3.10       rlang_1.1.5       diagram_1.6.5
```
