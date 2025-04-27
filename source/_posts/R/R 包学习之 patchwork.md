---
title: R 包学习之 patchwork
date: 2025-04-27 15:36:12
tags: [R package]
categories: [R]
---


<!-- version: 1.3.0 -->

# 介绍

<p>
<b>patchwork</b> 是一个专为 R
语言设计的绘图组合工具包，它的目标是使将多个单独的 <b>ggplot</b> 与
<font style="background-color:#F5864F">gridExtra::grid.arrange()</font>
和 <font style="background-color:#F5864F">cowplot::plot_grid()</font>
类似，但 <b>patchwork</b>
设计鼓励探索和迭代，并且能够扩展到任意复杂的布局。
</p>

# 安装

You can install patchwork from CRAN using install.packages(‘patchwork’).

``` r
install.packages('patchwork')
```

Alternatively you can grab the development version from
[Github](https://github.com/thomasp85/patchwork) using devtools:

``` r
# install.packages("devtools")
devtools::install_github('thomasp85/patchwork')
```

# 简单示例

使用 **patchwork** 是非常简单的：只要将不同的图 `+` 一起就行了！

``` r
library(ggplot2)
library(patchwork)

p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))

p1 + p2
```

![](/imgs/ed6ee88f80f4fcc9ab7724cc20ef6a67.svg)
在组合多个图形时，最后加入的图形会成为当前活动的图形，后续添加的任何
ggplot2 元素，如图层、标签等，都会自动加到该图形上。

``` r
library(ggplot2)
library(patchwork)

p1 <- ggplot(mtcars) + 
  geom_point(aes(mpg, disp)) + 
  ggtitle('Plot 1')

p2 <- ggplot(mtcars) + 
  geom_boxplot(aes(gear, disp, group = gear)) + 
  ggtitle('Plot 2')

p1 + p2 + labs(subtitle = 'This will appear in the last plot')
```

![](/imgs/6dcdfb50d8e6d22e74025ab7388c7ca4.svg)
**patchwork**
同时也提供了丰富的功能，可以支持创建任意复杂的布局，并且这些布局中的元素可以完全对齐。

``` r
library(ggplot2)
library(patchwork)

p3 <- ggplot(mtcars) + geom_smooth(aes(disp, qsec))
p4 <- ggplot(mtcars) + geom_bar(aes(carb))

(p1 | p2 | p3) /
      p4
```

![](/imgs/962f42195c3a158ad7123e833a7fb388.svg)
# 函数说明

## plot_arithmetic, 排列 ggplot2 对象

排列方法无非就是水平排列或垂直排列，这里可以用
(<strong style="color:red">|</strong>,
<strong style="color:red">-</strong>) 和
<strong style="color:red">/</strong> 分别实现。

### +, 将图形组合在一起

将多个 ggplot2 图形组合在一起，形成一个嵌套的组合图形。

``` r
library(ggplot2)
library(patchwork)

# 绘制多个图
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
p3 <- ggplot(mtcars) + geom_bar(aes(gear)) + facet_wrap(~cyl)
```

``` r
#  p1、p2 和 p3 被组合到同一个嵌套层级中，plot_layout(ncol = 1) 对所有图形生效，最终将 p1、p2 和 p3 垂直排列
p1 + p2 + p3 + plot_layout(ncol = 1)
```

![](/imgs/147c2d419c9c6aaecc0fd898c5fc421a.svg)
``` r
# p3 不会嵌套到 p1 + p2 的组合中，而是与 p1 + p2 保持同一级别。因此，这里 plot_layout(ncol = 1) 会分别作用于 p1 + p2 和 p3，而不会将它们作为整体排列
p1 + p2 - p3 + plot_layout(ncol = 1)
```

![](/imgs/1afe9e264fe2dcc8c6d4ab23858f8f61.svg)
### -, 将图形放在同一级别 (非嵌套)

与 <strong style="color:red">|</strong>
一样，<strong style="color:red">-</strong> 作用将左右两图并列放置
(同层级组合)，类似于”水平连接图形”，但不会改变嵌套关系，可以看作是”平行组合”。

``` r
library(ggplot2)
library(patchwork)

p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
```

``` r
# 等价于 p1 | p2
p1 - p2
```

![](/imgs/6065be93e6e31c4bc04c4456d82726a5.svg)
``` r
# 显式合并
merge(p1) - p2
```

![](/imgs/bea0cc5e2825425513ae0fb1f7d859f1.svg)
当 p1 本身可能是一个复杂的 patchwork 组合（例如 (p1 + p2) - p3），显式
merge() 可以确保逻辑清晰。如果 p1 是通过函数返回的（可能是 ggplot 或
patchwork 对象），merge()
能强制统一类型。一般情况下，运行以下代码会发现两者输出完全相同。因此，在简单场景下直接使用 -
即可，显式 merge() 主要用于特殊需求或代码严谨性。

### |, 将图形水平排列

``` r
library(ggplot2)
library(patchwork)

p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))

p1 | p2
```

![](/imgs/14044a89949fe7907a15646fe9c2c729.svg)
### /, 将图形垂直排列

``` r
library(ggplot2)
library(patchwork)

p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))

p1 / p2
```

![](/imgs/c8a0621be70705c77dceb7d30e4c2d5b.svg)
## plot_arithmetic, 拼接 gg 对象

上面部分主要是一些布局运算，为了减少代码重复，patchwork
提供了两个运算符，用于将 ggplot
元素（几何图形、主题、面等）添加到拼版中的多个/所有图中。<strong style="color:red">\*</strong>
将元素添加到当前嵌套级别的所有图中，而
<strong style="color:red">&</strong> 将递归到嵌套图中。

### &, 递归应用到所有子图

``` r
library(ggplot2)
library(patchwork)

# 绘制四个图
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
p3 <- ggplot(mtcars) + geom_bar(aes(gear)) + facet_wrap(~cyl)
p4 <- ggplot(mtcars) + geom_bar(aes(carb))

# 递归修改所有图层
(p1 + (p2 + p3) + p4 + plot_layout(ncol = 1)) & theme_bw()
```

![](/imgs/56c0314de21a1279d711477f4addc427.svg)
### \*, 应用于当前层级，不递归

``` r
library(ggplot2)
library(patchwork)

# 绘制四个图
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
p3 <- ggplot(mtcars) + geom_bar(aes(gear)) + facet_wrap(~cyl)
p4 <- ggplot(mtcars) + geom_bar(aes(carb))

# 对当前图层进行修改，不递归
(p1 + (p2 + p3) + p4 + plot_layout(ncol = 1)) * theme_bw()
```

![](/imgs/fd96b0f43ce505b2b3c06fd3faa81203.svg)
<strong style="color:red">\*</strong> 和
<strong style="color:red">&</strong> 运算符在 patchwork
包中用于组合图形时，虽然表面效果相似，但存在以下关键区别：<strong style="color:red">\*</strong>
运算符，递归应用到所有子图，会修改每个子图的属性；<strong style="color:red">&</strong>
运算符，仅作用于最外层的组合图，不会影响子图的独立属性。

## multipage_align(), 对齐图形

有时，有必要确保单独的图彼此对齐，但仍然作为单独的图存在。
例如，如果它们需要成为幻灯片的一部分，并且您不希望在幻灯片之间切换时标题和面板四处跳跃。
\*\* patchwork \*\* 提供了一系列实用程序来实现这一目标。但目前只能对齐
ggplot2 对象。

### get_dim(), 获取尺寸

提取单个图形的尺寸信息 (包括标题、坐标轴、图例等区域的大小)，输入一个
ggplot2 对象，并返回存储图形各部分的尺寸。

### get_max_dim(), 获取最大尺寸

计算多个图形的最大尺寸 (取所有图形的标题/坐标轴等区域的最大值)，可以是 A
图的一个值，B 的一个值，最后范围所有纬度的最大值组合。

### set_dim(), 设置尺寸

#### 示例一

将指定尺寸强制应用到另一个图形。

``` r
library(ggplot2)
library(patchwork)

p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp)) +
  ggtitle('Plot 1')

p2 <- ggplot(mtcars) +
  geom_bar(aes(gear)) +
  facet_wrap(~cyl) +
  ggtitle('Plot 4') +
  theme(plot.margin = margin(50, 50, 10, 10))
```

``` r
p1
```

![](/imgs/228ccaa77341fdca4c4f02b08a44601d.svg)
``` r
p2
```

![](/imgs/66be420398a86fef7d0a86141ccf6912.svg)
``` r
# Align a plot to p2
p2_dim <- get_dim(p2)
set_dim(p1, p2_dim)
```

![](/imgs/03a7ecbd955cc88b5622b82606f426fe.svg)
#### 示例二

``` r
library(ggplot2)
library(patchwork)

p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp)) +
  ggtitle('Plot 1')

p2 <- ggplot(mtcars) +
  geom_boxplot(aes(gear, disp, group = gear)) +
  ggtitle('Plot 2')

p3 <- ggplot(mtcars) +
  geom_point(aes(hp, wt, colour = mpg)) +
  ggtitle('Plot 3') +
  theme(plot.margin = margin(75, 75, 10, 10))

p4 <- ggplot(mtcars) +
  geom_bar(aes(gear)) +
  facet_wrap(~cyl) +
  ggtitle('Plot 4') +
  theme(plot.margin = margin(50, 50, 10, 50))
```

``` r
p1
```

![](/imgs/ad8c50bec17d0be49cbde027f856e4ac.svg)
``` r
p2
```

![](/imgs/bd0733def0c534cfb044f05be82429e8.svg)
``` r
p3
```

![](/imgs/8c22d95e9cac2562405858a7a8aa5310.svg)
``` r
p4
```

![](/imgs/f0ce00e73a6ca45213b7a30511c5cfd6.svg)
``` r
# Align a plot to the maximum dimensions of a list of plots
max_dims <- get_max_dim(p1, p2, p3, p4)
set_dim(p2, max_dims)
```

![](/imgs/e55e6c99e9c596af104b3d094739f198.svg)
### align_patches(), 批量对齐图形

``` r
library(ggplot2)
library(patchwork)

p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp)) +
  ggtitle('Plot 1')

p2 <- ggplot(mtcars) +
  geom_boxplot(aes(gear, disp, group = gear)) +
  ggtitle('Plot 2')

p3 <- ggplot(mtcars) +
  geom_point(aes(hp, wt, colour = mpg)) +
  ggtitle('Plot 3') +
  theme(plot.margin = margin(75, 75, 10, 10))

p4 <- ggplot(mtcars) +
  geom_bar(aes(gear)) +
  facet_wrap(~cyl) +
  ggtitle('Plot 4') +
  theme(plot.margin = margin(50, 50, 10, 50))

# Align a list of plots with each other
aligned_plots <- align_patches(p1, p2, p3, p4)
```

``` r
aligned_plots[[1]]
```

![](/imgs/dc1ff923f64c3aa6b25281b985158e8f.svg)
``` r
aligned_plots[[2]]
```

![](/imgs/7fb7279400a173200d2aa652808011cf.svg)
``` r
aligned_plots[[3]]
```

![](/imgs/48528a61a215acfe4e4c6d2ed85a4864.svg)
``` r
aligned_plots[[4]]
```

![](/imgs/6aa4c99bcea5b3c3f2d919f16b86fef3.svg)
``` r
# Aligned plots still behave like regular ggplots
aligned_plots[[4]] + theme_bw()
```

![](/imgs/afb36c6ea035b6723d324a9367956fb3.svg)
## area(), 在布局中指定打印区域

`area()` 是 **patchwork**
包中用于自定义拼图布局的核心函数，它通过定义矩阵网格区域来控制每个子图的放置位置。

核心功能是将画布划分为网格，并指定每个子图占据的单元格范围。

``` r
# Usage
area(t, l, b = t, r = l)

# Arguments
- t, b 分别为顶部和底部的行号
- l, r 分别为左边和右边的行号

# 例如：
areas <- c(area(1, 1, 2, 1), area(2, 3, 3, 3))

# 通过数字或字符布局，明确指定子图占据的矩形区域，其中相同字母代表一个子图。
# 等价于 (这种更好理解一些)：
areas < -"A##
          A#B
          ##B"
```

具体示例：

``` r
library(ggplot2)
library(patchwork)

# 绘制三个图
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
p3 <- ggplot(mtcars) + geom_bar(aes(gear)) + facet_wrap(~cyl)

# 设置这三个图的布局
layout <- c(
  area(1, 1),
  area(1, 3, 3),
  area(3, 1, 3, 2)
)

# 通过 plot() 函数预览布局
plot(layout)
```

![](/imgs/7f66570674673f6c97afeef40e7e3539.svg)
``` r
# 应用布局
p1 + p2 + p3 + plot_layout(design = layout)
```

![](/imgs/7bd728d83fb1e51467fa7c236f16ca0f.svg)
⚠️ 注意：子图数量需与 area() 定义的区域数量一致，否则会报错。

## free(), 将绘图从对齐中解放出来

当使用 patchwork 组合多个图形时，默认会尝试对齐各个图形的不同部分
(如面板、标签等)。`free()`
函数允许在指定某些图形的某些部分不参与这种对齐。

核心功能是将画布划分为网格，并指定每个子图占据的单元格范围。

``` r
# Usage
free(x, type = c("panel", "label", "space"), side = "trbl")

# Arguments
- x 要处理的 ggplot2 或 patchwork 对象
- type 释放对齐的类型，有三种选择：
    "panel" (默认)：允许面板区域根据需要以填充空白空间；
    "label"：保持轴标签与轴的接近性，即使其他图形的较长轴文本会将其对齐；
    "space"：不保留任何空间，允许轴占用其他空白区域。
-   side 指定应用释放的边，可以是"t"(上)、"r"(右)、"b"(下)、"l"(左) 的组合
```

具体示例：

``` r
library(ggplot2)
library(patchwork)

# 创建条形图，y 轴为 gear（档位）
p1 <- ggplot(mtcars) +
  geom_bar(aes(y = factor(gear), fill = factor(gear))) +
  scale_y_discrete(
    "",                                                  # 不显示 y 轴标题
    labels = c("3 gears are often enough",               # 为 y 轴的每个档位提供较长的标签
               "But, you know, 4 is a nice number",
               "I would def go with 5 gears in a modern car")
  )

# 创建散点图，x 轴为 mpg，y 轴为 disp
p2 <- ggplot(mtcars) +
  geom_point(aes(mpg, disp))

# 将条形图和散点图按照垂直方式组合
# 有时，由于轴标签过长，某些图形可能难以实现良好的排版对齐。
# 当与其他图形组合时，由于标签过长，结果可能显示效果不好
p1 / p2
```

![](/imgs/baa626b36e7cb899c3d788d2951fd441.svg)
可以通过使用 free() 函数调整布局（这里使用默认的 “panel”
类型进行自由调整）

``` r
# 调用 free() 函数对 p1 进行自由调整，然后与 p2 垂直组合
free(p1) / p2
```

![](/imgs/fec5eefae3e3984826a7481c9699097d.svg)
如果想让面板右侧对齐，可以选择仅自由调整左侧（使用 side = “l” 参数）

``` r
free(p1, side = "l") / p2
```

![](/imgs/38436b4e47a5b8f8a96ac03f7f0447d4.svg)
我们仍然可以像以前一样收集图例

``` r
# 使用 plot_layout(guides = "collect") 将图例统一放置
free(p1) / p2 + plot_layout(guides = "collect")
```

![](/imgs/00cac25bd2cb11717de4cdc5dafb4b22.svg)
可以通过 “label” 类型来以另一种方式修复布局

``` r
# 对 p2 使用 "label" 类型的自由调整，然后与 p1 垂直组合
p1 / free(p2, "label")
```

![](/imgs/008c8420c024fa54825918804b6ac764.svg)
另一个问题是，长标签没有充分利用可用的自由空间

``` r
# 使用 plot_spacer() 添加一个空白占位符，与其他图形组合
plot_spacer() + p1 + p2 + p2
```

![](/imgs/e00a6c9e3ab903d0c329b46500e69a37.svg)
这个问题可以通过 “space” 类型修复

``` r
# 对 p1 使用 "space" 类型的自由调整，同时仅调整左侧 (side = "l")，然后与其他图形组合
plot_spacer() + free(p1, "space", "l") + p2 + p2 
```

![](/imgs/8c2b1712057786c21e50a28013e9c826.svg)
## guide_area(), 添加一个区域来存放收集到的图例

`guide_area()`
函数主要用于为组合图图形布局预留一个专门的区域来放置图例。当使用`plot_layout(guides = "collect")`
收集图例时，默认情况下，patchwork
会将所有图例放置在图形的边缘（比如上方、右侧等）。但是，通过使用`guide_area()`函数，就可以指定一个明确的位置，用于显示这些收集的图例，而不是让它们依赖于默认的布局规则。

如果没有收集到图例，`guide_area()`会类似于`plot_spacer()`，即作为一个空白区域。

具体示例：

``` r
library(ggplot2)
library(patchwork)

# 绘制三个图
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp, colour = factor(gear)))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
p3 <- ggplot(mtcars) + geom_bar(aes(gear)) + facet_wrap(~cyl)
```

``` r
# 默认行为：图例保留在各自的图中
p1 + p2 + p3
```

![](/imgs/c73c93e18cb31420589d69ec3a66f902.svg)
``` r
# 收集图例，并放置在布局的边缘
p1 + p2 + p3 + plot_layout(guides = 'collect', ncol = 2)
```

![](/imgs/5d6489ef38cc25c4ea983e223938740a.svg)
``` r
# 使用 guide_area() 指定图例区域
p1 + p2 + p3 + guide_area() + plot_layout(guides = 'collect')
```

![](/imgs/d01c1039401a7135f6017e775eb60f45.svg)
## insert_element(), 嵌入图形

`insert_element()`用于在主图图形中嵌入另一个图形或图形元素，并允许自由控制嵌入图形的位置、大小和层级。它为我们在组合多个图形时提供了更多灵活性，比如将一个小图嵌入到另一个图的特定位置，类似于”内嵌小图”的功能。支持嵌入到主图的上层或下层，并可选择是否裁剪嵌入的图。

``` r
# Usage
inset_element(
  p,
  left,
  bottom,
  right,
  top,
  align_to = "panel",
  on_top = TRUE,
  clip = TRUE,
  ignore_tag = FALSE
)

# Arguments
- p 需要嵌入的图形或对象，可以是一个 ggplot2 对象、一个 grob 对象 (grid::circleGrob())、一个图片 (例如通过 png::readPNG() 读取的图片文件)、其他支持的对象类型
- left, bottom, right, top  
     定义嵌入对象的边界位置，取值范围为 0 至 1，例如 left = 0.6，bottom = 0.6，right = 1，top = 1 表示嵌入图位于主图的右上角，占据主图 40% 的宽度和高度；如果使用 unit 对象，可以直接指定具体的尺寸
- align_to 定义嵌入图的边界相对于哪个区域
     "panel" (默认)，嵌入图相对于主图的绘图区 (即坐标区域) 对齐
     "plot" 嵌入图相对于主图的整个绘图区域 (包括轴标签等) 对齐
     "full" 嵌入图相对于整个布局区域对齐 (包括背景)
- on_top 逻辑值，指定嵌入图是否位于主图的上方，默认嵌入图放置在主图的上层 (TRUE，默认)；否则嵌入图相对于整个布局在主图的下层 (FALSE，但仍位于背景上)
- clip 逻辑值，指定嵌入图是否裁剪到主图的边界内，默认嵌入图超过边界的部分会被裁剪掉 (TRUE)；否则嵌入图不会被裁剪，可以超出主图范围 (FALSE)
- ignore_tag 逻辑值，指定嵌入图是否被自动标注忽略，默默嵌入图会参与自动标注 (FALSE)；否则嵌入图不会被标注 (TRUE)
```

具体示例：

``` r
library(grid)
library(ggplot2)
library(patchwork)

# 创建两个 ggplot 图形
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))                     # 主图
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))    # 嵌入图
```

``` r
# 将 p2 嵌入到 p1 的右上角
p1 + inset_element(p2, 0.6, 0.6, 1, 1)
```

![](/imgs/e429c43db1c93f7e5b95e857fc265b6e.svg)
``` r
# 将 p2 嵌入到布局区域的左上角
p1 + inset_element(p2, 0, 0.6, 0.4, 1, align_to = 'full')
```

![](/imgs/e9b6913ba7383f4ca13aac4b52c15345.svg)
``` r
# 嵌入一个圆形 grob 到主图中
p1 + inset_element(grid::circleGrob(), 0.4, 0.4, 0.6, 0.6)
```

![](/imgs/1f5e776618a1ee40b268e984feafe960.svg)
``` r
# 嵌入图片
logo <- system.file('help', 'figures', 'logo.png', package = 'patchwork')       # 获取 R 内置图片路径
logo <- png::readPNG(logo, native = TRUE)                                       # 读取图片
p1 + inset_element(logo, 0.8, 0.8, 1, 1, align_to = 'full')                     # 嵌入图片到主图的右上角
```

![](/imgs/738d17d5708b0655ab44f02d0c07035b.svg)
``` r
# 嵌入图形后，修改主题
p1 + inset_element(p2, 0.6, 0.6, 1, 1) + theme_classic()
```

![](/imgs/81e8066d74de2cb1f659564735e3dd8e.svg)
自动标注和忽略标注

``` r
# 自动标注嵌入图
p1 +
  inset_element(p2, 0.6, 0.6, 1, 1) +
  plot_annotation(tag_levels = '1')
```

![](/imgs/aa873daaf0fc9bb0defd44cd3e3fdafe.svg)
``` r
# 忽略嵌入图的标注
p1 +
  inset_element(p2, 0.6, 0.6, 1, 1, ignore_tag = TRUE) +
  plot_annotation(tag_levels = '1')
```

![](/imgs/8b551434080e8577b89992fae8d7e103.svg)
## plot_annotation(), 添加注释

`plot_annotation()`用于为组合图形添加全局注释，比如标题、副标题、说明文字，以及对子图的自动标记。它适用于整个组合图的顶层布局，能够增强图形的整体表达性和美观性。这个函数可以通过`+`操作符直接添加到组合图中，与`plot_layout()`类似，**但其作用仅限于组合图的最顶层**，不会影响嵌套布局中的子图。

``` r
# Usage
plot_annotation(
  title = waiver(),
  subtitle = waiver(),
  caption = waiver(),
  tag_levels = waiver(),
  tag_prefix = waiver(),
  tag_suffix = waiver(),
  tag_sep = waiver(),
  theme = waiver()
)

# Arguments
- title, subtitle, caption
    为组合图添加全局标题、副标题或说明文字
- tag_levels 用于设置对子图的自动标记格式，支持以下类型:
      'a'：小写字母（a, b, c...）
      'A'：大写字母（A, B, C...）
      '1'：数字（1, 2, 3...）
      'i'：小写罗马数字（i, ii, iii...）
      'I'：大写罗马数字（I, II, III...）
      也可以是一个列表，定义多个层级的标记顺序（详见下面的多级标记示例）
            plot_annotation(tag_levels = 'A')           # 使用大写字母标记子图
            plot_annotation(tag_levels = c('A', '1'))   # 多级标记
- tag_prefix, tag_suffix
      用于设置标记的前缀或后缀
            plot_annotation(tag_levels = 'A', tag_prefix = "Figure ", tag_suffix = ":")
- tag_sep 设置多级标记之间的分隔符，默认分隔符为 . (如 A.1)
- theme 用于定义注释部分的样式，如标题的字体大小、颜色等；仅影响与注释相关的主题元素
```

具体示例：

``` r
library(ggplot2)
library(patchwork)

# 创建两个 ggplot 图形
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
```

``` r
# 添加全局标题和说明文字
p1 + p2 + plot_annotation(
  title = "This is a Title",
  subtitle = "This is a Subtitle",
  caption = "This is a Caption"
)
```

![](/imgs/e07865251d1c4ee39c9cc98a51037fc0.svg)
``` r
# 自定义标题样式
p1 + p2 +
  plot_annotation(
    title = "Customized Title",
    theme = theme(plot.title = element_text(size = 16, face = "bold", color = "blue"))
  )
```

![](/imgs/00624a240caa3b85531401b485d6c2ac.svg)
``` r
# 垂直排列两个子图，并自动标记
p1 / p2 + plot_annotation(tag_levels = "A")
```

![](/imgs/2c348b819848b50e203c595d82905702.svg)
``` r
# 嵌套布局的多级标记
# 如果组合图中有嵌套布局，可以通过 plot_layout(tag_level = 'new' 来创建新标记层级)
p1 / ((p2 | p2) + plot_layout(tag_level = 'new')) +
  plot_annotation(tag_levels = c('A', '1'))
```

![](/imgs/ee67bb09f66afb735232efd113ea5633.svg)
``` r
# 自定义标记顺序
p1 / ((p2 | p2) + plot_layout(tag_level = 'new')) +
  plot_annotation(tag_levels = list(c("&", "%"), "1"))
```

![](/imgs/4ff2b07ffde9c0d9d8230c22ea2691ce.svg)
## plot_layout(), 定义组合图布局

`play_layout()`，是一个功能强大的布局管理函数，用于定义组合图形的排列方式。它允许用户控制图形的行列布局、宽高比例、图例的展示方式、多层嵌套的布局规则，以及复杂的自定义布局设计。通过`play_layout()`，用户可以轻松实现多图组合的精确控制，从简单的行列排列到复杂的区域划分。

<strong style="color:#00A087;font-size:16px;">核心功能</strong>：

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>控制行列布局</strong>：使用 ncol 和 nrow
参数设置图形网格的列数和行数；默认情况下，patchwork
会自动根据图形数量和屏幕大小布局图形
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>图例管理</strong>：使用 guides
参数合并或保留图例，避免重复的图例展示
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>自定义网格比例</strong>：使用 widths 和 heights
参数设置每列和每行的相对宽度或高度，支持灵活调整布局
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>嵌套布局</strong>：当图形中有嵌套组合时，可以使用 plot_layout()
在各个嵌套层级定义布局规则
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>自定义复杂布局</strong>：使用 design
参数，您可以定义更复杂的布局方式，例如以文本或固定区域划分的方式精确放置图形
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>轴和标题的处理</strong>：使用 axes 和 axis_titles
参数，可以控制是否合并或保留单独的坐标轴和标题
</li>
</ul>

``` r
# Usage
plot_layout(
  ncol = waiver(),
  nrow = waiver(),
  byrow = waiver(),
  widths = waiver(),
  heights = waiver(),
  guides = waiver(),
  tag_level = waiver(),
  design = waiver(),
  axes = waiver(),
  axis_titles = axes
)

# Arguments
- ncol, nrow 控制网格布局的列数和行数，默认为自动布局
- byrow 控制图形的填充顺序，类似于矩阵中的 byrow 参数，默认 (TRUE) 按行填充
- widths, heights 设置每列相关宽度或每行相对高度
- guides 控制图例的展示方式，避免重复图例
      'collect'：合并所有子图的图例，并放置在布局外部
      'keep'：保留每个子图自己的图例
      'auto'：根据上层布局自动决定是否合并图例
- tag_level 控制自动标准的层级
      'keep'：保持当前标注层级
      'new'：进入新的标注层级
- design 自定义复杂布局，通过区域划分精确放置图形，可以通过数字或字符定义布局；通过 area() 函数，使用区域的行列范围自定义布局
- axes 控制是否合并重复的坐标轴
- axis_titles 控制是否合并重复的坐标轴标签
      'keep'：保留所有子图的坐标轴和标题
      'collect'：合并重复的坐标轴和标题
      'collect_x' / 'collect_y'：仅合并 x 或 y 轴
```

具体示例：

``` r
library(ggplot2)
library(patchwork)

# 定义一些 ggplot2 图形
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
p3 <- ggplot(mtcars) + geom_bar(aes(gear)) + facet_wrap(~cyl)
p4 <- ggplot(mtcars) + geom_bar(aes(carb))
p5 <- ggplot(mtcars) + geom_violin(aes(cyl, mpg, group = cyl))
p6 <- ggplot(mtcars) + geom_point(aes(mpg, disp, color = cyl))
p7 <- ggplot(mtcars) + geom_point(aes(mpg, hp, color = cyl))
```

``` r
# 自动布局
# 默认情况下，patchwork 会自动排列图形
# 将 5 个图自动排列，patchwork 会根据屏幕大小和图形数量自动生成行列布局
# 图形被紧凑地排列在网格中，行列数由 patchwork 自动决定
p1 + p2 + p3 + p4 + p5
```

![](/imgs/a7197eb9f77b4c45b7331730f6fa850d.svg)
``` r
# 按列填充网格
# 使用 byrow 参数，按列主要顺序填充图形
# 通过 plot_layout(byrow = FALSE)，图形将按照列优先的顺序依次排列
p1 + p2 + p3 + p4 + p5 + plot_layout(byrow = FALSE)
```

![](/imgs/a67f861e8973d641f9399d0bc245c07a.svg)
``` r
# 自定义网格维度
# 设置网格的列数为 2，并调整列的宽度比例
# ncol = 2：将图形排列为 2 列；widths = c(1, 2)：设置两列的相对宽度，第二列的宽度是第一列的两倍。
p1 + p2 + p3 + p4 + p5 + plot_layout(ncol = 2, widths = c(1, 2))
```

![](/imgs/44b6799a1a34783a72a5ea60273e5da4.svg)
``` r
# 嵌套布局
# 定义不同嵌套层级的布局
# 图形 p3 和 p4 被嵌套在一个单独的垂直布局中（ncol = 1）
# 嵌套布局作为整体，和其他图（p1, p2, p5）一起排列，最后变成 2*2
p1 +
  p2 +
  (p3 +
     p4 +
     # 嵌套布局：p3 和 p4 垂直排列
     plot_layout(ncol = 1)
  ) +
  p5 +
  # 外层布局：第一列宽度为第二列的两倍
  plot_layout(widths = c(2, 1))
```

![](/imgs/f00fc9cd46390092aeb2759291b27a88.svg)
``` r
# 自定义复杂布局
# 使用 area() 函数定义布局
design <- c(
  area(1, 1, 2),     # 图1占据第1列的所有行
  area(1, 2, 1, 3),  # 图2占据第1行的第2-3列
  area(2, 3, 3),     # 图3占据第2-3行的第3列
  area(3, 1, 3, 2),  # 图4占据第3行的第1-2列
  area(2, 2)         # 图5占据第2行的第2列
)
p1 + p2 + p3 + p4 + p5 + plot_layout(design = design)
```

![](/imgs/b2188fd94590c281face3cee08c0b889.svg)
``` r
# 使用字符串定义布局
# 将布局规则写成字符串，其中数字或字母表示每个图形的位置，# 表示空白区域
design <- "
  122
  153
  443
"
p1 + p2 + p3 + p4 + p5 + plot_layout(design = design)
```

![](/imgs/81aca0c9840a0297c3f2836ccbb11f42.svg)
``` r
# 带空白区域的布局
# 表示空白区域，图形只占据非空白部分
design <- "
  1##
  123
  ##3
"
# 图形被排列成非规则的布局，其中有空白区域
p1 + p2 + p3 + plot_layout(design = design)
```

![](/imgs/9758eb674728ca294b1d5def31c337e1.svg)
``` r
# 合并图例到布局外
# guides = 'collect'：将重复的图例合并为一个，并放置在布局外部
# 图例被合并为一个，避免重复
p6 + p7 + plot_layout(guides = 'collect')
```

![](/imgs/cb382dfc8701f75e30b06bd2d6c3d293.svg)
``` r
# 合并图例并调整位置
# guides = 'collect'：合并图例
# theme(legend.position = 'bottom')：将合并后的图例放置在布局底部
# 图例统一显示在整个图形的底部
p6 + p7 + plot_layout(guides = 'collect') &
  theme(legend.position = 'bottom')
```

![](/imgs/5d461f45ef32a4b1c9d684c8b55c221d.svg)
## plot_spacer(), 定义一个完全空白的区域

`plot_spacer()`用于在组合图中插入一个完全空白的区域，从而为图形之间增加间隔或者实现更灵活的布局。这个空白区域是一个透明的占位符，它不会显示任何内容，但会占据一定的空间。在某些场景下（如调整图形间距或创建复杂布局时），`plot_spacer()`是一个非常有用的工具。

``` r
# Usage
plot_spacer()

# Value
plot_spacer() 没有参数，因为它的功能非常简单，只生成一个空白的占位符
```

具体示例：

``` r
library(ggplot2)
library(patchwork)

# 定义一些 ggplot2 图形
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
```

``` r
# 组合图形，并在两者之间插入一个空白区域
p1 + plot_spacer() + p2
```

![](/imgs/1ed71f06b7dbcf4ac3549a1180b17f63.svg)
``` r
# 使用 `theme()` 的 `plot.margin` 参数对每个图形进行边距调整
(p1 + theme(plot.margin = unit(c(0, 30, 0, 0), "pt"))) +    # 给 p1 添加右侧 30 pt 的外边距
  (p2 + theme(plot.margin = unit(c(0, 0, 0, 30), "pt")))    # 给 p2 添加左侧 30 pt 的外边距
```

![](/imgs/a0cd718907c61d09ccbfe2e41756187b.svg)
## wrap_elements(), 将任意图形融入组合图中

`wrap_element()`就像一个转换器，它可以把非 ggplot2
的内容（比如文本、矩形、基础图形等）打包起来，让它们能像 ggplot2
图一样加入组合图中。这为用户提供了一个极大的灵活性，允许将多种类型的图形、文本或其他内容整合到单一布局中。

<strong style="color:#00A087;font-size:16px;">核心功能</strong>：

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>包装图形或对象</strong>：将非 ggplot2 的图形（如 grid
图形、基础图形、文本、表格等）转换为 patchwork
兼容的对象。支持的对象包括：grob（grid 图形对象，如
textGrob()、rectGrob() 等）；基础 R 图形（通过公式 ~ 传入，如 ~
plot()）；表格对象（如 gt 表格）
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>灵活放置区域</strong>：支持将内容放置在以下三个区域，‘panel’
数据绘制的核心区域（不包含坐标轴、标签等）；‘plot’
包括坐标轴、标签等的完整绘图区；‘full’
整个图形的全部区域，包括标题、标签、边距等
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>支持标题和主题</strong>：包装后的元素仍然可以添加标题
ggtitle()、标签 labs()，并设置样式
theme()，但主题设置仅对背景、边距、标题样式等有效
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>自动标签支持</strong>：如果图形中启用了自动标签（tag），包装的元素可以选择是否包含标签（通过参数
ignore_tag 控制）
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>裁剪内容</strong>：使用 clip 参数控制内容是否超出指定区域。如果
clip = TRUE（默认），超出部分会被裁剪
</li>
</ul>

``` r
# Usage
wrap_elements(
  panel = NULL,
  plot = NULL,
  full = NULL,
  clip = TRUE,
  ignore_tag = FALSE
)

# Arguments
- panel 将对象放置在绘图的核心区域（只包含数据绘制部分，不包含坐标轴和标题），支持 grob、ggplot 对象等
- plot 将对象放置在完整绘图区域（包含坐标轴、标题等），与 panel 相同
- full 将对象放置在整个图形区域（包括标题、标签、边距等），与 panel 相同
- clip 是否裁剪超出指定区域的内容，默认值为 TRUE，如果设置为 FALSE，内容可以超出指定的区域范围
- ignore_tag 是否忽略自动标签，默认值为 FALSE，如果设置为 TRUE，包装的对象不会包含自动生成的标签
```

具体示例：

``` r
library(grid)
library(ggplot2)
library(patchwork)
library(gridGraphics)
```

``` r
# 组合多个 grob 对象
# - `wrap_elements()` 将 `textGrob` 和 `rectGrob` 包装成为 patchwork 兼容对象
# - `panel` 区域表示绘图的核心区域（只包含绘图内容）
# - `full` 区域表示整个图形区域，包括标题、标签和边距
# - 通过 `+` 将两个包装的 grob 对象组合在一起，形成一个布局
wrap_elements(panel = textGrob('Here are some text')) +       # 在 panel 区域放置文本 'Here are some text'
  wrap_elements(
    panel = rectGrob(gp = gpar(fill = 'steelblue')),          # 在 panel 区域放置一个蓝色矩形
    full = rectGrob(gp = gpar(fill = 'goldenrod'))            # 在 full 区域放置一个金色矩形
  )
```

![](/imgs/7a3e25bb1892882db78e3005e68bbc1b.svg)
``` r
# 给包装的元素添加标题
wrap_elements(panel = textGrob('Here are some text')) +
  wrap_elements(
    panel = rectGrob(gp = gpar(fill = 'steelblue')),
    full = rectGrob(gp = gpar(fill = 'goldenrod'))
  ) +
  ggtitle('Title for the amazing rectangles')
```

![](/imgs/79b213bcf904bb66169151afe425372e.svg)
``` r
# 将 ggplot 图形包装为 panel 区域
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p1 + wrap_elements(panel = p1 + ggtitle('Look at me shrink'))
```

![](/imgs/1a7b6749935c26a915514836ff9f5b62.svg)
``` r
# 包装基础 R 图形
# - 使用公式 `~` 将基础 R 图形包装为 `full` 区域
# - `wrap_elements(full = ...)` 将基础图形转换为 patchwork 兼容对象，并放置在整个图形区域（full）
p1 + wrap_elements(full = ~ plot(mtcars$mpg, mtcars$disp))
```

![](/imgs/1684e514b1901bc046d83c4f02c13b17.svg)
``` r
# 这是包装基础 R 图形的简写形式
# 直接传入公式 `~ plot(...)`，效果等价于 `wrap_elements(full = ~ plot(...))`
p1 + ~ plot(mtcars$mpg, mtcars$disp)
```

![](/imgs/a59c10bbb04702b5b187733895fbc193.svg)
## wrap_plots(), 把 ggplot2 图拼接起来

`wrap_plots()`用于将多个 ggplot2
图形（或其他可组合内容）以网格布局的方式组合在一起。这种方式特别适合在编程中动态组合图形，例如当图形数量未知或需要自定义复杂布局时。

`wrap_plots()`和`wrap_elements()`之间的差别，`wrap_plots()`用于动态组合多个
ggplot2 图形，并提供网格布局控制；`wrap_elements()` 用于包装任意非
ggplot2 内容，与 ggplot2
图形一起组合。简单记忆：<span style="color:#00A087;font-weight:bold;">多个
ggplot2 图形？用 wrap_plots() ！非 ggplot2 内容？那就用
wrap_elements() ！</span>

``` r
# Usage
wrap_plots(
  ...,
  ncol = NULL,
  nrow = NULL,
  byrow = NULL,
  widths = NULL,
  heights = NULL,
  guides = NULL,
  tag_level = NULL,
  design = NULL,
  axes = NULL,
  axis_titles = axes
)

# Arguments
- ... 直接传入多个 ggplot2 图形，或传入一个包含图形的列表
- ncol, nrow
      设置网格布局的列数和行数，如果不指定，函数会自动根据图形数量推算
- byrow 布局方向控制，默认为 TRUE (按行填充图形)；若为 FALSE，则按列填充
- widths, heights   
      控制每列或每行的相对宽度或高度。值为向量，元素的长度会自动重复以匹配网格的行列数
- guides 控制图例的展示方式
      "collect"：将图例组合在一起，而不是在每个图形中单独显示
      "keep"：保持每个图形的独立图例
      "auto"：根据上层布局决定
- tag_level 控制自动标记（如 A、B、C…）的层次和行为
- design 自定义图形布局的参数，可以使用字符网格，如 "#BB\nAA#" 或区域定义 area() 函数来指定图形的位置
- axes, axis_titles 控制坐标轴和坐标轴标题的展示方式
      "keep"：保留每个图形的独立坐标轴或标题
      "collect"：合并重复的坐标轴或标题
      "collect_x"/"collect_y"：仅合并 x 轴或 y 轴
      
# Value
一个 patchwork 对象
```

具体示例：

``` r
library(ggplot2)
library(patchwork)

# 创建多个 ggplot 图形
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp))
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))
p3 <- ggplot(mtcars) + geom_bar(aes(gear)) + facet_wrap(~cyl)
p4 <- ggplot(mtcars) + geom_bar(aes(carb))
p5 <- ggplot(mtcars) + geom_violin(aes(cyl, mpg, group = cyl))
```

``` r
# 将多个图形作为单独参数传递，组合成网格布局
wrap_plots(p1, p2, p3, p4, p5)
```

![](/imgs/f9e865e3fa2f597e256296489ae800b2.svg)
``` r
# 或者将图形存储在列表中，动态传递给 wrap_plots 组合
plots <- list(p1, p2, p3, p4, p5)
wrap_plots(plots)
```

![](/imgs/08c9586394126b13163dc2f9329275e9.svg)
``` r
# 自定义布局：用字符网格设计图形的排列方式
# 表示空白区域，A 和 B 表示图形的对应位置
design <- "#BB
           AA#"
# 指定图形 p1 和 p2 匹配到布局的 B 和 A
wrap_plots(B = p1, A = p2, design = design)
```

![](/imgs/ff4ba5b0f9c8d411dfba03ec63c6ae41.svg)
``` r
# 不使用命名参数时，图形按顺序填充到布局中
wrap_plots(p1, p2, design = design)
```

![](/imgs/fddf27abcdd5615fc67c4a5bc4d1cee4.svg)
## wrap_ggplot_grob(), gtable 对象融入组合图

`wrap_ggplot_grob()`用于将一个由`ggplotGrob()`转换的 gtable 对象包装成
patchwork 兼容的对象。这种包装方式确保了 gtable 对象在 patchwork
中能够正确的对齐，并保持布局的一致性。

<strong style="color:#00A087;font-size:16px;">核心功能</strong>：

<ul style="margin-top:0;">
<li style="margin-top:2px;margin-bottom:2px;">
<strong>包装 gtable 对象</strong>：将由 ggplot2::ggplotGrob() 转换得到的
gtable 对象包装为 patchwork 兼容对象（table_patch）；这种包装的 gtable
对象可以与其他 ggplot 图形组合，作为布局的一部分
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>确保对齐</strong>：与直接使用 wrap_elements() 或将 gtable
对象直接添加到 patchwork 不同，wrap_ggplot_grob()
会确保包装的内容在组合图中正确对齐；如果直接添加 gtable
对象，可能会导致对齐问题
</li>
<li style="margin-top:2px;margin-bottom:2px;">
<strong>支持自定义 gtable 修改</strong>：在将 ggplot 对象转换成 gtable
后，可以对 gtable 对象进行修改（如添加文本、水印等），然后通过
wrap_ggplot_grob() 包装后加入到组合图中
</li>
</ul>

``` r
# Usage
wrap_ggplot_grob(x)

# Arguments
- x 一个由 ggplot2::ggplotGrob() 转换得到的 gtable 对象

# Value
返回一个 table_patch 对象，这是一个可以被添加到 patchwork 布局中的对象
```

具体示例：

``` r
library(grid)
library(gtable)
library(ggplot2)
library(patchwork)

# 创建两个 ggplot 图形
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp)) + ggtitle('disp and mpg seems connected')
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear))

# 将 p2 转换为 gtable 对象
p2_table <- ggplotGrob(p2)

# 创建一个文本 grob（用作水印）
stamp <- textGrob('TOP SECRET', rot = 35, gp = gpar(fontsize = 36, fontface = 'bold'))

# 将水印添加到 gtable 的顶部
p2_table <- gtable_add_grob(
  p2_table,      # 原始 gtable 对象
  stamp,         # 要添加的 grob 对象
  t = 1,         # 添加到 gtable 的第 1 行
  l = 1,         # 添加到 gtable 的第 1 列
  b = nrow(p2_table), # 跨越 gtable 的所有行
  r = ncol(p2_table)  # 跨越 gtable 的所有列
)
```

``` r
# 直接添加 gtable 对象会导致对齐问题
p1 + p2_table
```

![](/imgs/2ab0e7256b558bc513a377bd50f80389.svg)
``` r
# 使用 wrap_ggplot_grob() 确保对齐
p1 + wrap_ggplot_grob(p2_table)
```

![](/imgs/77757a157b9d71028d615d2ac0a07335.svg)
## wrap_table(), gt 对象融入组合图

`wrap_table()`用于将 表格（gt 表或数据框表格） 包装成 patchwork
兼容对象，从而可以与 ggplot
图形组合使用。它与`wrap_elements()`功能类似，但**专为表格设计**，提供了一些额外的布局选项和功能，使表格在组合图形时更容易调整和控制。

``` r
# Usage
wrap_table(
  table,
  panel = c("body", "full", "rows", "cols"),
  space = c("free", "free_x", "free_y", "fixed"),
  ignore_tag = FALSE
)

# Arguments
- table 输入表格
      支持 gt 表格（需要 gt 包支持）
      支持可以被转换为数据框的对象（如 data.frame）
- panel 指明表格的哪一部分与绘图区域（panel region）对齐
      "body"（默认值）：仅对齐表格的主体部分（数据单元格），行列标题会在绘图区域外显示
      "full"：将整个表格（包括行列标题和主体）放置在绘图区域内
      "rows"：将所有行（包括列标题）放置在绘图区域内，但行标题会在左侧显示
      "cols"：将所有列（包括行标题）放置在绘图区域内，但列标题会在顶部显示
- space 控制表格尺寸是否影响整体布局
      "free"（默认值）：表格的尺寸不会影响布局的宽度或高度（尺寸自由）
      "fixed"：表格的宽度和高度会固定，并影响布局的整体尺寸
      "free_x"：表格的宽度不会影响布局，但高度会固定
      "free_y"：表格的高度不会影响布局，但宽度会固定
- ignore_tag 控制自动标签是否忽略表格
      FALSE（默认值）：表格会自动生成标签（如 A、B...）
      TRUE：表格不会生成标签，常用于表格不需要标签的场景
      
# Value
返回一个 wrapped_table 对象，可以与其他 patchwork 对象（如 ggplot 图形）组合
```

具体示例：

``` r
library(gt)
library(ggplot2)
library(patchwork)

# 创建 ggplot 图形 p1
p1 <- ggplot(airquality) +
  geom_line(aes(x = Day, y = Temp, colour = month.name[Month])) +               # 绘制每日气温折线图，按月份分类
  labs(colour = "Month")                                                        # 设置图例标题为 "Month"

# 创建第二个 ggplot 图形 p2
p2 <- ggplot(airquality) +
  geom_boxplot(aes(y = month.name[Month], x = Temp)) +                          # 绘制每月气温的箱线图
  scale_y_discrete(name = NULL, limits = month.name[9:5], guide = "none")       # 设置 y 轴顺序为 9 月到 5 月，并隐藏图例

# 创建数据表格
table <- data.frame(
  Month = month.name[5:9],                                                      # 表格的月份列（5 月到 9 月）
  "Mean temp." = tapply(airquality$Temp, airquality$Month, mean),               # 平均温度
  "Min temp." = tapply(airquality$Temp, airquality$Month, min),                 # 最低温度
  "Max temp." = tapply(airquality$Temp, airquality$Month, max)                  # 最高温度
)

# 查看数据
table
##       Month Mean.temp. Min.temp. Max.temp.
## 5       May   65.54839        56        81
## 6      June   79.10000        65        93
## 7      July   83.90323        73        92
## 8    August   83.96774        72        97
## 9 September   76.90000        63        93

# 将数据框转换为 gt 表格
gt_tab <- gt(table, rowname_col = "Month")                                      # 将 "Month" 列作为行名
```

``` r
# 默认情况下，直接将表格与 ggplot 图形组合
# - 由于 `patchwork` 自动识别表格类型，默认使用 `wrap_table()` 包装表格
# - 表格的行列标题会位于绘图区域外，仅数据（主体部分）对齐绘图区域
p1 + gt_tab
```

![](/imgs/5cec604078167a44015f21ba71f3bb54.svg)
``` r
# 使用 wrap_table() 显式控制表格布局
# - 使用 `panel = "full"` 参数，将表格的主体和标题（包括行标题和列标题）完全放入绘图区域
# - 表格整体与图形组合，而不是仅对齐数据部分
p1 + wrap_table(gt_tab, panel = "full")
```

![](/imgs/543052e72e0fb82c2f267914424afb5b.svg)
``` r
# 使用 wrap_table() 控制表格的尺寸影响布局
# - 设置 `space = "fixed"` 参数，表格的宽度和高度会固定，并影响布局的尺寸
# - 表格和图形的大小根据表格的固定尺寸调整，确保表格和图形比例一致
wrap_table(gt_tab, space = "fixed") + p2
```

![](/imgs/6d55679214d22fd102e4e906f6b3464a.svg)
# Quick Tips
