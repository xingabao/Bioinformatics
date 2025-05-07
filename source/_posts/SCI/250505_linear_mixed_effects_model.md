---
title: 跟着顶刊学分析之线性混合效应模型
date: 2025-05-05 13:50:32
tags: [R, 线性混合效应模型]
categories: [[跟着顶刊学分析, 线性混合效应模型]]
---


各个数据模型都包含一定的假设，当数据满足这些假设时，使用这些模型得到的结果才会更准确。比如广义线性模型需要满足线性和独立性等条件。为了适应更多数据类型，开发出了各种模型，可根据数据情况，选择合适的模型。比如，如果数据不满住线性，则可考虑广义可加模型；如果不满足独立性，则可以考虑混合效应模型。

多水平模型同时包含多个分类因子数据，最主要特点就是非独立性，在多个水平上都存在残差，因此适合混合效应模型，总的来说，其思想就是把最高水平上的差异估计出来，这就使得残差变小，估计的结果更为可靠。比如，一份包含不同施肥处理的多土层微生物数据，其中不同施肥处理与不同土壤深度即是嵌套数据（多水平数据）。

# 10.1038/s41467-025-56439-z (Fig.2)

这个示例根据[`Temperature-dependent variations in under-canopy herbaceous foliar diseases following shrub encroachment in grasslands`](https://www.nature.com/articles/s41467-025-56439-z)，这篇文献整理而来。

线性混合效应模型 (Linear mixed-effects model):

</br></br>

<img src="/imgs/41467_2025_56439_Fig2_HTML.webp" width="65%" style="display: block; margin: auto;" />

<p style="text-align:justify;font-size:15px;line-height:20px">
The linear mixed-effects models illustrate the effects of a mean annual
temperature (MAT) and b mean annual precipitation (MAP) on
log-transformed response ratio of herbaceous community foliar fungal
pathogen load between shrub patches and grassland patches’ plots (i.e.,
LRR pathogen load), c MAT and d MAP on log-transformed response ratio of
herbaceous plant community aboveground biomass between shrub patches and
grassland patches’ plots (i.e., LRR herbaceous biomass), e LRR
herbaceous biomass on LRR pathogen load. Parameter significance was
assessed by F-tests (two-tailed). Solid lines and shadows indicate
significant effects (P &lt; 0.05) and 95% confidence intervals; n = 320
for these linear mixed-effects models. Data points represent the
log-transformed response ratios of the original data.
</p>

# 为什么需要混合效应模型

<p>
在现实数据中，常常遇到<b>嵌套</b>或<b>重复测量</b>的数据结构，例如：1)
学生成绩，学生嵌套在班级，班级嵌套在学校中；2)
医疗实验，同一个病人多次测量；3) 农业试验，同一块地不同时间反复观测。
</p>

传统的线性模型，如线性回归，假设所有观测值相互独立，但这些嵌套/重复测量数据违反了独立性假设。

# 什么是线性混合效应模型

<p>
线性混合效应模型是扩展线性回归模型以考虑固定效应和随机效应的统计模型，它是一种结合了<b>固定效应</b>
(Fixed Effects) 和<b>随机效应</b> (Random Effects)
的统计模型，能处理上述复杂结构的数据。
</p>

## 固定效应

类似于传统线性回归中的回归系数，表示整体趋势或总体规律；例如，药物对血压的平均影响。

## 随机效应

反应的是群体间的变异，用于建模数据的层级结构或重复测量；例如，由于不同班级、不同病人本身的差异带来的波动。

# 线性混合效应模型

## 加载包和数据

``` r
# 加载所需要的包
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(lme4)))
suppressMessages(suppressWarnings(library(lmerTest)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(patchwork)))

# 用于存放模型以及画图必须信息
models = list()
xs = list()
ys = list()

# 数据加载
Result_data = openxlsx::read.xlsx("data/10.1038nc2025.xlsx", sheet = "Fig1 to Fig6", startRow = 23)  

# 查看数据格式
str(Result_data)
## 'data.frame':    320 obs. of  17 variables:
##  $ Site          : chr  "HK21" "HK21" "HK21" "HK21" ...
##  $ Plot          : num  1 2 3 4 1 2 3 4 1 2 ...
##  $ MAT           : num  -5.6 -5.6 -5.6 -5.6 -3.76 ...
##  $ MAP           : num  463 463 463 463 518 518 518 518 509 509 ...
##  $ LRRPL         : num  -2.51 -2 -1.69 -2.1 -4.39 ...
##  $ LRRbiomass    : num  -0.302 -0.445 -0.842 -0.607 -2.264 ...
##  $ LRRTemperature: num  -0.3372 -0.3372 -0.3372 -0.3372 0.0443 ...
##  $ LRRHumidity   : num  0.317 0.317 0.317 0.317 0.124 ...
##  $ LRRSoilPC1    : num  -2 -2.2 -3.12 -1.56 NA ...
##  $ LRRSR         : num  1.288 0.934 -0.14 -0.357 -1.012 ...
##  $ LRRShannon    : num  1.005 1.216 0.475 0.347 -0.225 ...
##  $ LRRSimpson    : num  0.5945 0.9423 0.3941 0.313 0.0956 ...
##  $ LRRCWMHN      : num  0.864 0.434 1.216 1.188 1.54 ...
##  $ LRRCWMSLA     : num  -0.24 -0.207 0.392 0.332 0.294 ...
##  $ LRRMPD        : num  2.373 3.354 1.262 0.972 0.379 ...
##  $ Beta          : num  0.784 0.846 0.767 0.824 0.867 ...
##  $ Timeline      : chr  "1985-2015" "1985-2015" "1985-2015" "1985-2015" ...

# 简要生成数据摘要
summary(Result_data)
##      Site                Plot           MAT                 MAP            LRRPL           LRRbiomass       LRRTemperature      LRRHumidity          LRRSoilPC1           LRRSR         
##  Length:320         Min.   :1.00   Min.   :-5.600942   Min.   : 75.0   Min.   :-5.0186   Min.   :-3.26093   Min.   :-0.77164   Min.   :-0.887893   Min.   :-4.77375   Min.   :-1.79176  
##  Class :character   1st Qu.:1.75   1st Qu.: 0.006637   1st Qu.:275.0   1st Qu.:-0.9900   1st Qu.:-0.80824   1st Qu.:-0.30298   1st Qu.: 0.004256   1st Qu.:-0.41195   1st Qu.:-0.47000  
##  Mode  :character   Median :2.50   Median : 1.736203   Median :372.5   Median : 0.0000   Median :-0.12834   Median :-0.16330   Median : 0.170402   Median :-0.02943   Median :-0.04831  
##                     Mean   :2.50   Mean   : 2.206784   Mean   :393.0   Mean   :-0.1809   Mean   :-0.05418   Mean   :-0.17633   Mean   : 0.189813   Mean   :-0.05917   Mean   :-0.11399  
##                     3rd Qu.:3.25   3rd Qu.: 4.057243   3rd Qu.:507.5   3rd Qu.: 0.3139   3rd Qu.: 0.58472   3rd Qu.:-0.04711   3rd Qu.: 0.369955   3rd Qu.: 0.35961   3rd Qu.: 0.22314  
##                     Max.   :4.00   Max.   : 9.983211   Max.   :861.0   Max.   : 4.6833   Max.   : 6.65136   Max.   : 0.17491   Max.   : 0.711632   Max.   : 5.21610   Max.   : 2.56495  
##                                                                                                                                                    NA's   :39                           
##    LRRShannon         LRRSimpson          LRRCWMHN         LRRCWMSLA            LRRMPD               Beta          Timeline        
##  Min.   :-7.40996   Min.   :-6.68496   Min.   :-0.8082   Min.   :-3.34100   Min.   :-12.16651   Min.   :0.0000   Length:320        
##  1st Qu.:-0.41212   1st Qu.:-0.29139   1st Qu.: 0.3463   1st Qu.:-0.40574   1st Qu.: -0.40400   1st Qu.:0.4187   Class :character  
##  Median :-0.05061   Median :-0.02937   Median : 0.7313   Median : 0.09851   Median : -0.03839   Median :0.5556   Mode  :character  
##  Mean   :-0.17899   Mean   :-0.17392   Mean   : 0.7619   Mean   : 0.09199   Mean   : -0.25013   Mean   :0.5614                     
##  3rd Qu.: 0.19266   3rd Qu.: 0.12383   3rd Qu.: 1.1890   3rd Qu.: 0.53506   3rd Qu.:  0.15443   3rd Qu.:0.7143                     
##  Max.   : 7.59293   Max.   : 6.71028   Max.   : 2.8694   Max.   : 4.13218   Max.   : 12.14125   Max.   :1.0000                     
## 
```

## Fig.2a

``` r
# 建立线性混合效应模型，MAT 为固定效应，Site 为随机效应
model1 = lmer(LRRPL ~ MAT + (1|Site), data = Result_data)

# 保存模型到列表中，用于后续画图
models[[1]] = model1
xs[[1]] = 'MAT'
ys[[1]] = 'LRRPL'

# 查看模型摘要，包括固定效应、随机效应估计等
# MAT (年均温度) 对 LRRPL (病原负荷) 影响显著
# MAT 每增加 1°C，LRRPL 预计增加 0.09887 (p = 0.00369）。说明温度升高可能会增加草本植物的病原负荷
# 随机效应 (Site) 影响较小但仍然重要
# 站点 (Site) 的方差 (0.5067) 较小，但仍然影响 LRRPL 变化。说明不同站点的 LRRPL 基础水平不同，但趋势一致
summary(model1)
## Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
## Formula: LRRPL ~ MAT + (1 | Site)
##    Data: Result_data
## 
## REML criterion at convergence: 1117.8
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -3.8382 -0.4402 -0.1010  0.3857  3.6084 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  Site     (Intercept) 0.5067   0.7118  
##  Residual             1.5327   1.2380  
## Number of obs: 320, groups:  Site, 80
## 
## Fixed effects:
##             Estimate Std. Error       df t value Pr(>|t|)   
## (Intercept) -0.39911    0.12820 78.00000  -3.113  0.00259 **
## MAT          0.09887    0.03303 78.00000   2.993  0.00369 **
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##     (Intr)
## MAT -0.569

# 对模型进行方差分析，检验固定效应（如 MAT）是否显著
# MAT (年均温度) 显著影响 LRRPL (病原负荷) (p = 0.003694 < 0.01)
# F 值较大 (8.9604)，表明 MAT 的影响不是随机的，而是系统性的
anova(model1)
## Type III Analysis of Variance Table with Satterthwaite's method
##     Sum Sq Mean Sq NumDF DenDF F value   Pr(>F)   
## MAT 13.733  13.733     1    78  8.9604 0.003694 **
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# 使用 performance 包计算模型的R²（拟合优度）, 用于衡量模型对数据的解释能力
# Marginal R²（边际 R²） = 0.047，表示固定效应 (MAT) 解释的 LRRPL 变异比例
# Conditional R²（条件 R²） = 0.284，说明考虑站点 (Site) 影响后，模型解释能力大幅提升
performance::r2(model1)
## # R2 for Mixed Models
## 
##   Conditional R2: 0.284
##      Marginal R2: 0.047
```

## Fig.2b

``` r
# 用线性混合效应模型拟合数据
model2 = lmer(LRRPL ~ MAP + (1|Site), data = Result_data)

# 保存模型到列表中，用于后续画图
models[[2]] = model2
xs[[2]] = 'MAP'
ys[[2]] = 'LRRPL'

# 查看模型摘要，包括固定效应、随机效应估计等
summary(model2)
## Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
## Formula: LRRPL ~ MAP + (1 | Site)
##    Data: Result_data
## 
## REML criterion at convergence: 1132
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -3.7009 -0.4458 -0.0113  0.4029  3.7457 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  Site     (Intercept) 0.5803   0.7618  
##  Residual             1.5327   1.2380  
## Number of obs: 320, groups:  Site, 80
## 
## Fixed effects:
##               Estimate Std. Error         df t value Pr(>|t|)
## (Intercept)  0.2040442  0.2757213 77.9999986   0.740    0.461
## MAP         -0.0009795  0.0006436 77.9999986  -1.522    0.132
## 
## Correlation of Fixed Effects:
##     (Intr)
## MAP -0.917

# 对模型进行方差分析，检验固定效应是否显著
anova(model2)
## Type III Analysis of Variance Table with Satterthwaite's method
##     Sum Sq Mean Sq NumDF DenDF F value Pr(>F)
## MAP 3.5505  3.5505     1    78  2.3165  0.132

# 使用 performance 包计算模型的R²（拟合优度）, 用于衡量模型对数据的解释能力
performance::r2(model2)
## # R2 for Mixed Models
## 
##   Conditional R2: 0.284
##      Marginal R2: 0.013
```

## Fig.2c

``` r
# 用线性混合效应模型拟合数据
model3 = lmer(LRRbiomass ~ MAT + (1|Site), Result_data)

# 保存模型到列表中，用于后续画图
models[[3]] = model3
xs[[3]] = 'MAT'
ys[[3]] = 'LRRbiomass'

# 查看模型摘要，包括固定效应、随机效应估计等
summary(model3)
## Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
## Formula: LRRbiomass ~ MAT + (1 | Site)
##    Data: Result_data
## 
## REML criterion at convergence: 946.8
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -3.1299 -0.4853 -0.0518  0.4513  5.8800 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  Site     (Intercept) 0.3339   0.5778  
##  Residual             0.8773   0.9366  
## Number of obs: 320, groups:  Site, 80
## 
## Fixed effects:
##             Estimate Std. Error       df t value Pr(>|t|)    
## (Intercept) -0.27410    0.10108 78.00000  -2.712  0.00823 ** 
## MAT          0.09966    0.02604 78.00000   3.827  0.00026 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##     (Intr)
## MAT -0.569

# 对模型进行方差分析，检验固定效应是否显著
anova(model3)
## Type III Analysis of Variance Table with Satterthwaite's method
##     Sum Sq Mean Sq NumDF DenDF F value    Pr(>F)    
## MAT 12.847  12.847     1    78  14.644 0.0002604 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# 使用 performance 包计算模型的R²（拟合优度）, 用于衡量模型对数据的解释能力
performance::r2(model3)
## # R2 for Mixed Models
## 
##   Conditional R2: 0.332
##      Marginal R2: 0.077
```

## Fig.2d

``` r
# 用线性混合效应模型拟合数据
model4 = lmer(LRRbiomass ~ MAP + (1|Site), Result_data)

# 保存模型到列表中，用于后续画图
models[[4]] = model4
xs[[4]] = 'MAP'
ys[[4]] = 'LRRbiomass'

# 查看模型摘要，包括固定效应、随机效应估计等
summary(model4)
## Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
## Formula: LRRbiomass ~ MAP + (1 | Site)
##    Data: Result_data
## 
## REML criterion at convergence: 958.2
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -2.8727 -0.4608 -0.0580  0.4095  5.7476 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  Site     (Intercept) 0.3583   0.5986  
##  Residual             0.8773   0.9366  
## Number of obs: 320, groups:  Site, 80
## 
## Fixed effects:
##               Estimate Std. Error         df t value Pr(>|t|)   
## (Intercept)  0.5871594  0.2134992 77.9999997   2.750  0.00740 **
## MAP         -0.0016318  0.0004983 77.9999997  -3.274  0.00158 **
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##     (Intr)
## MAP -0.917

# 对模型进行方差分析，检验固定效应是否显著
anova(model4)
## Type III Analysis of Variance Table with Satterthwaite's method
##     Sum Sq Mean Sq NumDF DenDF F value  Pr(>F)   
## MAP 9.4065  9.4065     1    78  10.722 0.00158 **
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# 使用 performance 包计算模型的R²（拟合优度）, 用于衡量模型对数据的解释能力
performance::r2(model4)
## # R2 for Mixed Models
## 
##   Conditional R2: 0.332
##      Marginal R2: 0.059
```

## Fig.2e

``` r
# 用线性混合效应模型拟合数据
model5 = lmer(LRRPL ~ LRRbiomass + (1|Site), Result_data)

# 保存模型到列表中，用于后续画图
models[[5]] = model5
xs[[5]] = 'LRRPL'
ys[[5]] = 'LRRbiomass'

# 查看模型摘要，包括固定效应、随机效应估计等
summary(model5)
## Linear mixed model fit by REML. t-tests use Satterthwaite's method ['lmerModLmerTest']
## Formula: LRRPL ~ LRRbiomass + (1 | Site)
##    Data: Result_data
## 
## REML criterion at convergence: 1104.7
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -3.06170 -0.55313 -0.05235  0.45779  3.13242 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  Site     (Intercept) 0.4309   0.6564  
##  Residual             1.5073   1.2277  
## Number of obs: 320, groups:  Site, 80
## 
## Fixed effects:
##              Estimate Std. Error        df t value Pr(>|t|)    
## (Intercept)  -0.16318    0.10056  76.60839  -1.623    0.109    
## LRRbiomass    0.32781    0.07016 310.69787   4.672 4.45e-06 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##            (Intr)
## LRRbiomass 0.038

# 对模型进行方差分析，检验固定效应是否显著
anova(model5)
## Type III Analysis of Variance Table with Satterthwaite's method
##            Sum Sq Mean Sq NumDF DenDF F value    Pr(>F)    
## LRRbiomass 32.903  32.903     1 310.7  21.828 4.447e-06 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# 使用 performance 包计算模型的R²（拟合优度）, 用于衡量模型对数据的解释能力
performance::r2(model5)
## # R2 for Mixed Models
## 
##   Conditional R2: 0.275
##      Marginal R2: 0.067
```

## 批量画图

``` r
xlabs = list()
ylabs = list()
xlabs[[1]] = expression(MAT ~ (degree*C)); ylabs[[1]] = "LRR pathogen load"
xlabs[[2]] = "MAP (mm)"; ylabs[[2]] = "LRR pathogen load"
xlabs[[3]] = expression(MAT ~ (degree*C)); ylabs[[3]] = "LRR herbaceous biomass"
xlabs[[4]] = "MAP (mm)"; ylabs[[4]] = "LRR herbaceous biomass"
xlabs[[5]] = "LRR herbaceous biomass"; ylabs[[5]] = "LRR pathogen load"

fig.alb = list()
for (index in 1:length(models)) {
  
  model = models[[index]]
  xlab = xlabs[[index]]
  ylab = ylabs[[index]]
  
  F_value = round(anova(model)$`F value`[1], 3)
  p_ = anova(model)$`Pr(>F)`[1]
  p_value = ifelse(p_ < 0.001, "*P* < 0.001", sprintf("*P* = %.3f", p_))
  
  R2_marginal = round(performance::r2(model)$R2_marginal, 3)
  
  stat_text = paste0(
    "*F*<sub>1,239</sub> = ", F_value,"<br>",
    p_value, "<br>",
    "*R*<sup>2</sup> = ", R2_marginal
  )
  
  ert = ggplot(Result_data, aes(x = .data[[xs[[index]]]], y = .data[[ys[[index]]]])) +
    geom_point(
      alpha = 0.45, 
      size = 4, 
      color = "#0069B7FF",
      pch = 21,
      fill = "#0069B7FF", 
      stroke = 0
      ) +
    geom_smooth(method = "lm", se = TRUE, color = "#98103E") +
    labs(y = ylab, x = xlab) +
    ggtext::geom_richtext(
      x = I(0.99),
      y = I(0.88), 
      label = stat_text,
      size = 3.5, 
      fill = NA, 
      hjust = 1,
      label.color = NA
    ) +
    theme_classic()
  
  fig.alb[[index]] = ert
}

(fig.alb[[1]] | fig.alb[[2]] | patchwork::plot_spacer()) /
  (fig.alb[[3]] | fig.alb[[4]] | fig.alb[[5]]) + 
  patchwork::plot_annotation(tag_levels = 'a')
```

![](/imgs/3eb9fd400fa8c177d70371e28fc8eba2.png)
# 代码简洁版

``` r
# 加载所需要的包
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(lme4)))
suppressMessages(suppressWarnings(library(lmerTest)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(patchwork)))

no = basename(dirname(rstudioapi::getActiveDocumentContext()$path))
wkdir = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wkdir)

# 用于存放模型以及画图必须信息
models = list()
xs = list()
ys = list()

# 数据加载
# Result_data = openxlsx::read.xlsx("data/10.1038nc2025.xlsx", sheet = "Fig1 to Fig6", startRow = 23)  
Result_data = openxlsx::read.xlsx("E:/BaiduSyncdisk/005.Bioinformatics/Bioinformatics/src/data/10.1038nc2025.xlsx", sheet = "Fig1 to Fig6", startRow = 23)  

# 查看数据格式
str(Result_data)

# 简要生成数据摘要
summary(Result_data)
 
# Fig.2a
# 建立线性混合效应模型，MAT 为固定效应，Site 为随机效应
model1 = lmer(LRRPL ~ MAT + (1|Site), data = Result_data)

# 保存模型到列表中，用于后续画图
models[[1]] = model1
xs[[1]] = 'MAT'
ys[[1]] = 'LRRPL'

# 查看模型摘要，包括固定效应、随机效应估计等
# MAT (年均温度) 对 LRRPL (病原负荷) 影响显著
# MAT 每增加 1°C，LRRPL 预计增加 0.09887 (p = 0.00369）。说明温度升高可能会增加草本植物的病原负荷
# 随机效应 (Site) 影响较小但仍然重要
# 站点 (Site) 的方差 (0.5067) 较小，但仍然影响 LRRPL 变化。说明不同站点的 LRRPL 基础水平不同，但趋势一致
summary(model1)

# 对模型进行方差分析，检验固定效应（如 MAT）是否显著
# MAT (年均温度) 显著影响 LRRPL (病原负荷) (p = 0.003694 < 0.01)
# F 值较大 (8.9604)，表明 MAT 的影响不是随机的，而是系统性的
anova(model1)

# 使用 performance 包计算模型的R²（拟合优度）, 用于衡量模型对数据的解释能力
# Marginal R²（边际 R²） = 0.047，表示固定效应 (MAT) 解释的 LRRPL 变异比例
# Conditional R²（条件 R²） = 0.284，说明考虑站点 (Site) 影响后，模型解释能力大幅提升
performance::r2(model1)

# Fig.2b
model2 = lmer(LRRPL ~ MAP + (1|Site), data = Result_data)
models[[2]] = model2
xs[[2]] = 'MAP'
ys[[2]] = 'LRRPL'
summary(model2)
anova(model2)
performance::r2(model2)

# Fig.2c
model3 = lmer(LRRbiomass ~ MAT + (1|Site), Result_data)
models[[3]] = model3
xs[[3]] = 'MAT'
ys[[3]] = 'LRRbiomass'
summary(model3)
anova(model3)
performance::r2(model3)

# Fig.2d
model4 = lmer(LRRbiomass ~ MAP + (1|Site), Result_data)
models[[4]] = model4
xs[[4]] = 'MAP'
ys[[4]] = 'LRRbiomass'
summary(model4)
anova(model4)
performance::r2(model4)

# Fig.2e
model5 = lmer(LRRPL ~ LRRbiomass + (1|Site), Result_data)
models[[5]] = model5
xs[[5]] = 'LRRPL'
ys[[5]] = 'LRRbiomass'
summary(model5)
anova(model5)
performance::r2(model5)

# 批量画图
xlabs = list()
ylabs = list()
xlabs[[1]] = expression(MAT ~ (degree*C)); ylabs[[1]] = "LRR pathogen load"
xlabs[[2]] = "MAP (mm)"; ylabs[[2]] = "LRR pathogen load"
xlabs[[3]] = expression(MAT ~ (degree*C)); ylabs[[3]] = "LRR herbaceous biomass"
xlabs[[4]] = "MAP (mm)"; ylabs[[4]] = "LRR herbaceous biomass"
xlabs[[5]] = "LRR herbaceous biomass"; ylabs[[5]] = "LRR pathogen load"

fig.alb = list()
for (index in 1:length(models)) {
  
  model = models[[index]]
  xlab = xlabs[[index]]
  ylab = ylabs[[index]]
  
  F_value = round(anova(model)$`F value`[1], 3)
  p_ = anova(model)$`Pr(>F)`[1]
  p_value = ifelse(p_ < 0.001, "*P* < 0.001", sprintf("*P* = %.3f", p_))
  
  R2_marginal = round(performance::r2(model)$R2_marginal, 3)
  
  stat_text = paste0(
    "*F*<sub>1,239</sub> = ", F_value,"<br>",
    p_value, "<br>",
    "*R*<sup>2</sup> = ", R2_marginal
  )
  
  ert = ggplot(Result_data, aes(x = .data[[xs[[index]]]], y = .data[[ys[[index]]]])) +
    geom_point(
      alpha = 0.45, 
      size = 4, 
      color = "#0069B7FF",
      pch = 21,
      fill = "#0069B7FF", 
      stroke = 0
    ) +
    geom_smooth(method = "lm", se = TRUE, color = "#98103E") +
    labs(y = ylab, x = xlab) +
    ggtext::geom_richtext(
      x = I(0.99),
      y = I(0.88), 
      label = stat_text,
      size = 3.5, 
      fill = NA, 
      hjust = 1,
      label.color = NA
    ) +
    theme_classic()
  
  fig.alb[[index]] = ert
}

gg = (fig.alb[[1]] | fig.alb[[2]] | patchwork::plot_spacer()) /
  (fig.alb[[3]] | fig.alb[[4]] | fig.alb[[5]]) + 
  patchwork::plot_annotation(tag_levels = 'a')

ggsave(gg, filename = paste0(wkdir, '/LMM.png'), width = 11, height = 7, dpi = 300, device = 'png', bg = '#FFFFFF')
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
## [1] patchwork_1.3.0 ggplot2_3.5.1   lmerTest_3.1-3  lme4_1.1-37     Matrix_1.7-2    dplyr_1.1.4    
## 
## loaded via a namespace (and not attached):
##  [1] generics_0.1.3      xml2_1.3.8          stringi_1.8.7       lattice_0.22-6      digest_0.6.37       magrittr_2.0.3      evaluate_1.0.3      grid_4.4.3          fastmap_1.2.0      
## [10] zip_2.3.2           ggtext_0.1.2        mgcv_1.9-1          scales_1.3.0        numDeriv_2016.8-1.1 reformulas_0.4.0    Rdpack_2.6.3        cli_3.6.4           rlang_1.1.5        
## [19] rbibutils_2.3       performance_0.13.0  litedown_0.6        commonmark_1.9.5    munsell_0.5.1       splines_4.4.3       withr_3.0.2         yaml_2.3.10         tools_4.4.3        
## [28] nloptr_2.2.1        minqa_1.2.8         colorspace_2.1-1    boot_1.3-31         vctrs_0.6.5         R6_2.6.1            lifecycle_1.0.4     stringr_1.5.1       MASS_7.3-64        
## [37] insight_1.1.0       pkgconfig_2.0.3     pillar_1.10.1       openxlsx_4.2.8      gtable_0.3.6        glue_1.8.0          Rcpp_1.0.14         xfun_0.51           tibble_3.2.1       
## [46] tidyselect_1.2.1    rstudioapi_0.17.1   knitr_1.50          farver_2.1.2        htmltools_0.5.8.1   nlme_3.1-167        labeling_0.4.3      rmarkdown_2.29      compiler_4.4.3     
## [55] markdown_2.0        gridtext_0.1.5
```
