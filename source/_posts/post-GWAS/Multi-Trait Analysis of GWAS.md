---
title: Multi-Trait Analysis of GWAS
date: 2025-05-02 23:33:04
tags: [R, psot-GWAS, MTAG]
categories: [[教学示例, psot-GWAS, MTAG]]
---
多性状 GWAS 分析 (Multi-Trait Analysis of GWAS)

`mtag`是一款基于 Python 的命令行工具，用于联合分析多组 GWAS summary data，该方法由 Turley 等人 2018 提出。It can also be used as a tool to meta-analyze GWAS results.

# 安装程序

To run `mtag`, you will need to have `Python 2.7` installed with the following packages:

- `numpy (>=1.13.1)`
- `scipy`
- `pandas (>=0.18.1)`
- `argparse`
- `bitarray` (for `ldsc`)
- `joblib`

(Note: if you already have the Python 3 version of the Anaconda distribution installed, then you will need to create and activate a Python 2.7 environment to run `mtag`. See [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) for details.)

## 基于 Linux 环境 (未完整测试)

`mtag` may be downloaded by cloning this github repository:

```shell
git clone https://github.com/omeed-maghzian/mtag.git
cd mtag

/tools/Python-2.7.16/bin/pip install joblib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

To test that the tool has been successfully installed, type:

```shell
$ /tools/Python-2.7.16/python mtag.py -h
usage: mtag.py [-h] [--sumstats [{File1},{File2}...]]
               [--gencov_path FILE_PATH] [--residcov_path FILE_PATH]
               [--out DIR/PREFIX] [--make_full_path] [--meta_format]
               [--snp_name SNP_NAME] [--z_name Z_NAME] [--beta_name BETA_NAME]
               [--se_name SE_NAME] [--n_name N_NAME] [--n_value N1, N2,...]
               [--eaf_name EAF_NAME] [--no_chr_data] [--chr_name CHR_NAME]
               [--bpos_name BPOS_NAME] [--a1_name A1_NAME] [--a2_name A2_NAME]
               [--p_name P_NAME] [--include SNPLIST1,SNPLIST2,..]
               [--exclude SNPLIST1,SNPLIST2,..] [--only_chr CHR_A,CHR_B,..]
               [--homogNs_frac FRAC] [--homogNs_dist D] [--maf_min MAF_MIN]
               [--n_min N_MIN] [--n_max N_MAX] [--info_min INFO_MIN]
               [--incld_ambig_snps] [--no_allele_flipping] [--use_beta_se]
               [--no_overlap] [--perfect_gencov] [--equal_h2] [--force]
               [--fdr] [--skip_mtag] [--grid_file GRID_FILE] [--fit_ss]
               [--intervals INTERVALS] [--cores CORES] [--p_sig P_SIG]
               [--n_approx] [--ld_ref_panel FOLDER_PATH]
               [--time_limit TIME_LIMIT] [--std_betas] [--tol TOL]
               [--numerical_omega] [--verbose] [--chunksize CHUNKSIZE]
               [--stream_stdout] [--median_z_cutoff MEDIAN_Z_CUTOFF]

**mtag: Multitrait Analysis of GWAS** This program is the implementation of
MTAG method described by Turley et. al. Requires the input of a comma-
separated list of GWAS summary statistics with identical columns. It is
recommended to pass the column names manually to the program using the options
below. The implementation of MTAG makes use of the LD Score Regression (ldsc)
for cleaning the data and estimating residual variance-covariance matrix, so
the input must also be compatible ./munge_sumstats.py command in the ldsc
distribution included with mtag. The default estimation method for the genetic
covariance matrix Omega is GMM (as described in the paper). Note below: any
list of passed to the options below must be comma-separated without
whitespace.

optional arguments:
  -h, --help            show this help message and exit
```

## 基于 Windows 环境 (测试通过)

`mtag` may be downloaded by cloning this github repository:

```shell
git clone https://github.com/omeed-maghzian/mtag.git
cd mtag

/tools/Python-2.7.16/bin/pip install joblib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装 Python 环境:

```R
MTAG_install()
```

To test that the tool has been successfully installed, type:

```shell
C:/Users/Administrator/.conda/envs/MTAG/python.exe E:/tools/mtag/mtag.py -h
```

# 官方手册

## Tutorial 1: The Basics

这部分入门教程旨在帮助练习如何在命令行中使用`mtag`，但它并不全面。在开始之前，先按照[安装程序](#安装程序)部署好环境，并查看帮助文档信息。

### 示例数据

当对多个 GWAS summary data 进行数据分析时，使用 MATG 是非常有用的。GWAS 的汇总统计数据必须按照`mtag`能够读取的格式进行整理。这里所用的示例数据已经完成了这些预处理，只需要将压缩文件解压到`mtag`目录的文件夹中，即可直接运行下方的`mtag`命令。

这里以  [Okbay et. al. (2016)](http://www.nature.com/ng/journal/v48/n6/full/ng.3552.html#access) 关于神经质 (Neuroticism) 和主观幸福感 (Subjective well-being) 的 GWAS 结果为例，演示`mtag`的使用方法。这里使用了原始 GWAS 结果中，来自 Hapmap3 中随机抽取的一部分 SNP 的子集。神经质和主观幸福感的 GWAS summary data 可以从这里下载：[https://thessgac.com/](https://thessgac.com/) (Note: requires registering for an account).

这些示例文件已经按命令行工具默认输出格式整理好了，也就是说，它们是以空白字符分隔的 .txt 文件，包含以下这些列，顺序不限：

```
以下列是 MTAG 正常运行所必需的：
snpid    chr    bpos    a1    a2    freq    z    pval    n
```

>+ **snpid**：唯一的 SNP 标识符，通常为 rsid 编号，多个 GWAS 文件会通过该列进行匹配。当对 GWAS 数据进行标准化和估算残差协方差矩阵时，该 SNP 标识符也会传递给 ldsc，也可以通过 `--snp_name` 选项指定其他列名。
>
>- **a1/a2**：该位点观测到的等位基因，a1 被视为效应等位基因，其符号也应反映在 Z 分数上，这两列也会传递给 ldsc 程序；mtag 会检查并调整 a1 与 a2，确保在所有输入文件中保持一致，也可以用 `--a1_name` 和 `--a2_name` 指定其他列名。
>
>- **freq**：效应等位基因（a1）的频率，用于过滤稀有变异并用于将 MTAG 结果转换为非标准化效应值，可通过 `--eaf_name` 指定其他列名。
>
>- **z**：与 SNP 效应值相关的 Z 分数，与样本量列一起，几乎在 mtag 程序的每一个关键步骤中都会用到；可通过 `--z_name` 指定其他列名。
>
>  注意：如果输入的 GWAS 数据中同时包含效应值（beta）和标准误，则 MTAG 不再强制要求 z 列；要使用该选项，请确保仓库为最新版本，并指定 `--use_beta_se` 标志，默认的列名分别是 beta 和 se，可通过 `--beta_name` 和 `--se_name` 指定。
>
>- **n**：该 SNP 的样本量，和 z 一样，是分析的核心组成部分，可通过 `--n_name` 指定其他列名。
>
>
>
>其他列虽然不是 mtag.py 直接使用，但在 ldsc 实现的数据标准化过程中会用到。因此，至少需要保证这些列的格式能够被 ldsc 识别，详见[这里](https://github.com/bulik/ldsc/wiki/Heritability-and-Genetic-Correlation#sumstats-format)。染色体编号 (chr) 和碱基对位置 (bpos) 也会作为参考列写入结果文件，也可以分别通过 `--chr_name` 和 `--bpos_name` 选项指定其他列名。



## Tutorial 2: Special Options



## Tutorial 3: maxFDR Calculation







# 封装函数



https://github.com/JonJala/mtag/wiki/Tutorial-1:-The-Basics
