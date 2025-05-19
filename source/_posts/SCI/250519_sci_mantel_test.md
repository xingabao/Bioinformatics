---
title: 跟着顶刊学分析之曼特尔检验
date: 2025-04-27 13:22:55
tags: [R]
categories: [[跟着顶刊学分析, 曼特尔检验]]
---


<p>
曼特尔检验 (Mantel Test)
是一种在统计学中常用的非参数统计方法，旨在检测两个或多个独立距离矩阵之间是否存在相关性。该方法由
Nathan Mantel 和 William Haenszel 于 1967
年首次提出，并广泛应用于生态学和微生物多样性研究等领域。相较于只能处理两列数据的相关系数，曼特尔检验能够灵活地分析单个或一组环境因子与整个微生物群落地相关性。
</p>
<p>
在应用曼特尔检验时，首先假设两个矩阵之间不存在相关关系，并计算它们之间的原始相关系数；接着，通过置换检验随机改变其中一个矩阵的行和列顺序，并重新计算置换后的相关系数。这个过程重复多次，以生成一个相关系数的分布。最后，将原始观测到的相关系数与这个分布进行比较，以确定其是否具有统计学意义。如果观测到的相关性的
P 值小于预设的显著性水平，则可以拒绝原假设，认为两个矩阵之间存在相关性。
</p>
<p>

此外，曼特尔检验的应用不仅局限于生物领域，它同样适用于其他领域如地理学、社会科学等，帮助研究人员揭示不同矩阵之间的潜在联系&lt;/&gt;

<p>
本示例参考 iMeta <a href='https://doi.org/10.1002/imt2.224'>Probio-M9, a
breast milk-originated probiotic, alleviates mastitis and enhances
antibiotic efficacy: Insights into the gut–mammary axis</a> 的 Figure 2E
和 Nature Communications
<a href='https://doi.org/10.1038/s41467-024-52996-x'>Biodiversity of
mudflat intertidal viromes along the Chinese coasts</a> 的 Figure 5a。
</p>

# 10.1002/imt2.224 Fig.2E

<img src="/imgs/10.1002-imt2.224-fig.2e.png" width="75%" style="display: block; margin: auto;" />

<p style="text-align:justify;font-size:15px;line-height:20px">
Fig.2(E) The heatmap illustrates Pearson’s correlation between
significant differential abundant KOs and SGBs, while the network shows
the correlations between cytokine levels, including interferon (IFN)-γ,
interleukin (IL)-10, IL-1β, IL-2, and IL-6, and these KOs and SGBs,
evaluated by the partial Mantel test. Mantel’s p and r values are
indicated by different line colors and thickness, respectively, while
Pearson’s correlation is represented by the color scale.
</p>

## 加载包和数据

``` r
# 加载所需要的包
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(linkET)))
suppressMessages(suppressWarnings(library(ggplot2)))

# 数据加载
varespec = read.delim("data/varespec.txt", header = TRUE, sep = "\t")
varechem = read.delim("data/varechem.txt", header = TRUE, sep = "\t")

# 查看数据
head(varespec)
##      IFN.γ    IL.1β     IL.6     IL.2    IL.10
## 1 74.09582 70.61972 97.07855 27.58282 14.76930
## 2 74.08789 74.64455 99.08221 35.25620 17.79727
## 3 74.59502 74.57874 90.09028 33.50049 15.87683
## 4 73.11008 69.67854 98.09761 36.37342 19.34643
## 5 72.88947 68.45753 90.10275 43.23114 20.34632
## 6 71.11008 65.68943 95.14529 34.34545 18.45622

head(varechem)
##       SGB136     SGB130    SGB036   SGB063     SGB134     SGB142     SGB154     SGB019 SGB087     SGB034 SGB078    SGB047    SGB183     SGB133     SGB243     SGB105    SGB095     SGB221     SGB044
## 1 0.06869513 0.11673915 0.0000000 0.031427 0.17533973 0.00000000 0.07403608 0.00000000      0 0.02627083      0 0.0000000 0.0000000 0.00000000 0.00000000 0.14336938 0.2682917 1.15992330 0.11419496
## 2 0.07299563 0.21187916 0.0000000 0.000000 0.00000000 0.07167242 0.15134813 0.05820489      0 0.02699741      0 0.1423929 1.9020954 0.03076021 0.00000000 0.04439715 0.3109148 0.26340985 0.05762080
## 3 0.06755042 0.21368471 0.0000000 0.000000 0.00000000 0.10364100 0.08930372 0.00000000      0 0.51242703      0 0.0000000 0.1941449 0.00000000 0.00000000 0.05867881 0.2157102 0.44498524 0.18070021
## 4 0.05618834 0.17248255 0.0000000 0.000000 0.00000000 0.04161875 0.20480755 0.03834866      0 0.04224905      0 0.1381581 0.0000000 0.00000000 0.03843803 0.02764098 0.2734038 0.03968174 0.09481099
## 5 0.06140434 0.16796456 0.1532939 0.000000 0.03444199 0.18139623 0.16276625 0.04139398      0 0.07384385      0 0.1191059 1.7622347 0.00000000 0.00000000 0.03175899 0.1475364 0.56748170 0.61345834
## 6 0.06423240 0.09366044 0.0000000 0.000000 0.00000000 0.22429633 0.10163527 0.00000000      0 0.19491150      0 0.2114573 1.3923233 0.15903293 0.58136064 0.03268280 0.5469152 0.45748657 0.00000000
##        K23352      K19449      K18979      K18122      K13587      K08679      K07106      K03234      K03073
## 1 0.000217122 0.000269816 0.000492548 0.000124433 0.000066900 0.000139967 0.000263257 0.000136755 0.000482436
## 2 0.000231683 0.000099800 0.000384572 0.000077600 0.000149707 0.000247812 0.000217646 0.000197460 0.000483561
## 3 0.000247582 0.000217360 0.000384702 0.000157832 0.000086900 0.000157948 0.000119221 0.000135437 0.000535649
## 4 0.000268665 0.000225916 0.000441278 0.000222608 0.000095700 0.000153844 0.000116003 0.000150727 0.000582952
## 5 0.000217140 0.000277284 0.000550173 0.000249651 0.000091200 0.000118685 0.000109456 0.000137470 0.000565600
## 6 0.000261931 0.000205327 0.000471094 0.000178867 0.000112382 0.000168895 0.000159143 0.000162577 0.000538786
```

## 执行曼特尔检验

``` r
# Mantel test
mantel <- mantel_test(
  varespec, 
  varechem,
  spec_select = list(
    Spec01 = 1,
    Spec02 = 2,
    Spec03 = 3,
    Spec04 = 4,
    Spec05 = 5)) %>% 
  mutate(
    rd = cut(r, breaks = c(-Inf, 0.2, 0.4, Inf), labels = c("< 0.2", "0.2 - 0.4", ">= 0.4")),
    pd = cut(p, breaks = c(-Inf, 0.01, 0.05, Inf), labels = c("< 0.01", "0.01 - 0.05", ">= 0.05"))
  )
## `mantel_test()` using 'bray' dist method for 'spec'.
## `mantel_test()` using 'euclidean' dist method for 'env'.
```

## 画图

``` r
set_corrplot_style(scale = ggplot2::scale_fill_viridis_c())

qcorrplot(correlate(varechem), type = "lower", diag = FALSE) +
  geom_square() +
  geom_couple(
    aes(colour = pd, size = rd), 
    data = mantel, 
    curvature = nice_curvature()
  ) +
  scale_size_manual(values = c(0.5, 1, 2)) +
  scale_colour_manual(values = color_pal(3)) +
  guides(
    size = guide_legend(title = "Mantel's r", override.aes = list(colour = "grey35"),  order = 2),
    colour = guide_legend(title = "Mantel's p", override.aes = list(size = 3), order = 1),
    fill = guide_colorbar(title = "Pearson's r", order = 3)
  )
```

![](/imgs/b5e5a70dde09dfc3dccbbc4a8727d89f.png)
# 10.1002/imt2.224 Fig.2E

<img src="/imgs/10.1038-s41467-024-52996-x-fig.5a.png" width="75%" style="display: block; margin: auto;" />

<p style="text-align:justify;font-size:15px;line-height:20px">
Fig.5 Analyses were performed for vOTUs, mOTUs, vOTUs assigned with
hosts, and microbial hosts. a Partial mantel tests showing the
relationship between environmental factors and viral/microbial
communities. The edge color and width represent the Mantel’s r and p
value, respectively. The color gradient in heatmap represents the
Pearson’s correlation coefficients between different environmental
factors. The stars in the heatmap indicate significance levels: \*
(P &lt; 0.05), \*\* (P &lt; 0.01), and \*\*\* (P &lt; 0.001). All
statistical tests were two-tailed.
</p>

## 加载包和数据

``` r
# 加载所需要的包
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(Hmisc)))
suppressMessages(suppressWarnings(library(linkET)))
suppressMessages(suppressWarnings(library(ggplot2)))

# 数据加载
varespec = read.delim("data/varespec.txt", header = TRUE, sep = "\t")
varechem = read.delim("data/varechem.txt", header = TRUE, sep = "\t")
env <- read.csv('data/Environment.CSV', row.names = 1, check.names = FALSE)
geo <- read.csv('data/distance.CSV', row.names = 1)
host <- t(read.delim('data/host.txt', row.names = 1))
mOTUs <- t(read.delim('data/mOTUs.txt', row.names = 1))
virus <- t(read.delim('data/virus.txt', row.names = 1))
vOTUs <- t(read.delim('data/vOTUs.txt', row.names = 1))

spec <- as.data.frame(cbind(host, mOTUs, virus, vOTUs))
```

## 执行曼特尔检验

``` r
# Mantel test
mantel <- mantel_test(
  spec, env[1:11],
  spec_select = list(
    "host" = colnames(host),
    "mOTUs" = colnames(mOTUs),
    "virus" = colnames(virus),
    "vOTUs" = colnames(vOTUs)),
  env_ctrl = geo,
  mantel_fun = "mantel.partial",
  permutations = 999) %>%
  mutate(
    r_value = cut(r, breaks = c(-Inf, 0.1, 0.2, Inf), labels = c("r < 0.1", "0.1 ≤ r ≤ 0.2", "r ≥ 0.2")),
    p_value = cut(p, breaks = c(-Inf, 0.001, 0.05, Inf), labels = c("p < 0.001", "0.001 ≤ p ≤ 0.05", "p ≥ 0.05"))
  )
## `mantel_test()` using 'bray' dist method for 'spec'.
## `mantel_test()` using 'gower' dist method for 'env'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
## `mantel_test()` using 'euclidean' dist method for 'env_ctrl'.
```

## 画图

``` r
env_numeric <- apply(env[, 1:11], 2, as.numeric)

qcorrplot(rcorr(env_numeric), type = "upper", diag = FALSE, grid_size = 0.4, grid_col = "lightgray", parse = ) +
  geom_square(linetype = 0) +
  geom_couple(aes(colour = p_value, size = r_value), data = mantel, curvature = 0.1) +
  set_corrplot_style(colours = c("#FF8040", "white", "#5BC2CD")) +
  scale_size_manual(breaks = c("r < 0.1", "0.1 ≤ r ≤ 0.2", "r ≥ 0.2"), values = c(0.2, 0.5, 0.8)) +
  scale_colour_manual(breaks = c("p ≥ 0.05", "0.001 ≤ p ≤ 0.05", "p < 0.001"), values = c("gray", "#9ACD32", "#87CEEB")) +
  guides(
    size = guide_legend(title = "Mantel's r", override.aes = list(colour = "grey35"), order = 2),
    colour = guide_legend(title = "Mantel's p", override.aes = list(size = 2), order = 1),
    fill = guide_colorbar(title = "Pearson's r", order = 3)
  ) +
  geom_mark(
    only_mark = TRUE,
    size = 6, 
    sig_level = c(0.05, 0.01, 0.001), 
    sig_thres = 0.05
  ) + theme(
    axis.text.x.top = element_text(vjust = -0.1, hjust = 0, angle = 60)
  )
```

![](/imgs/4702c3db791b79e2973573aa8e156509.png)
# 版本信息

``` r
sessionInfo()
## R version 4.4.1 (2024-06-14 ucrt)
## Platform: x86_64-w64-mingw32/x64
## Running under: Windows 10 x64 (build 19045)
## 
## Matrix products: default
## 
## 
## locale:
## [1] LC_COLLATE=English_United States.utf8  LC_CTYPE=English_United States.utf8    LC_MONETARY=English_United States.utf8 LC_NUMERIC=C                           LC_TIME=English_United States.utf8    
## 
## time zone: Asia/Shanghai
## tzcode source: internal
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] Hmisc_5.2-2    ggplot2_3.5.1  linkET_0.0.7.4 dplyr_1.1.4   
## 
## loaded via a namespace (and not attached):
##  [1] magic_1.6-1       generics_0.1.3    stringi_1.8.4     lattice_0.22-6    digest_0.6.37     magrittr_2.0.3    evaluate_1.0.3    grid_4.4.1        fastmap_1.2.0     Matrix_1.7-1     
## [11] ape_5.8           nnet_7.3-19       backports_1.5.0   Formula_1.2-5     gridExtra_2.3     mgcv_1.9-1        purrr_1.0.2       viridisLite_0.4.2 scales_1.3.0      permute_0.9-7    
## [21] ade4_1.7-23       abind_1.4-8       cli_3.6.3         rlang_1.1.4       munsell_0.5.1     splines_4.4.1     base64enc_0.1-3   withr_3.0.2       yaml_2.3.10       vegan_2.6-8      
## [31] geometry_0.5.2    tools_4.4.1       parallel_4.4.1    checkmate_2.3.2   htmlTable_2.4.3   colorspace_2.1-1  vctrs_0.6.5       R6_2.5.1          rpart_4.1.23      lifecycle_1.0.4  
## [41] stringr_1.5.1     htmlwidgets_1.6.4 MASS_7.3-61       foreign_0.8-87    cluster_2.1.6     pkgconfig_2.0.3   pillar_1.10.1     gtable_0.3.6      Rcpp_1.0.14       data.table_1.16.4
## [51] glue_1.8.0        xfun_0.49         tibble_3.2.1      tidyselect_1.2.1  FD_1.0-12.3       rstudioapi_0.17.1 knitr_1.49        farver_2.1.2      htmltools_0.5.8.1 nlme_3.1-166     
## [61] rmarkdown_2.29    labeling_0.4.3    compiler_4.4.1
```
