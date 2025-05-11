---
title: 基于 LaTeX 生成一份完整的携带者筛查检测报告
date: 2025-05-12 01:49:18
tags: [LaTeX]
categories: [[案例分享, LaTeX]]
---


`LaTeX`是一种高质量的排版系统，广泛用于创建科学文档、学术论文和书籍等。本完整示例采用`LaTeX`完整生成了一份 PDF 检测报告，包括 PDF 合并、表格、页眉页脚、固定文本、固定图片、文本框、多行文本、插入图片、调整图片样式、图片和文本之间对齐、不规则表格、表格文本换行、自定义颜色、自定义水平缩进等等众多细节调整。开发环境使用了`VS Code`，使用`LaTeXmk`进行编译。

# 完整代码

```tex
\documentclass{article}

\usepackage[fontset=windows]{ctex}
\usepackage{pdfpages}
\usepackage[paperwidth=21cm, paperheight=29.7cm, hmargin=0cm, vmargin={1.32cm,0cm}]{geometry}
\usepackage{graphicx}
\usepackage{xeCJK}
\usepackage[absolute,overlay]{textpos}
\usepackage{tabularx} 
\usepackage{array}
\usepackage[export]{adjustbox}
\usepackage{fontspec}
\usepackage{changepage}
\usepackage{colortbl}
\usepackage{fancyhdr}
\usepackage{makecell}
\usepackage{datetime2}
\usepackage{tcolorbox}

\tcbuselibrary{skins}
\DTMsetdatestyle{iso}                                     % 显示当前日期为 YYYY-MM-DD 格式

\renewcommand{\arraystretch}{1.6}                         % 行间距
\setlength{\arrayrulewidth}{0.25pt}                       % 边框粗细
\setCJKfamilyfont{YaHei}{微软雅黑}
\newcolumntype{Y}[1]{>{\centering\arraybackslash}p{#1}}   % 自定义标签列：粗体、居中对齐
\newcolumntype{L}[1]{>{\raggedleft\bfseries}p{#1}}        % 自定义标签列：粗体、右对齐
\newcolumntype{C}[1]{p{#1}}                               % 普通内容列，左对齐
\arrayrulecolor[HTML]{0059BA}                             % 定义蓝色线条
\definecolor{separationline}{RGB}{0,89,186}               % 定义分隔线颜色

% 自定义水平缩进
\newcommand{\myindent}{\hspace{0.820cm}}
\newcommand{\myindenr}{\hspace{0.220cm}}
\newcommand{\mypassage}{\hspace{1.565cm}}

% 设定页眉页脚
\pagestyle{fancy}
\fancyhf{}                                                % 清除默认页眉页脚
\fancyhead[C]{\includegraphics[width=\paperwidth]{data/imgs/top-banner.png}}
\fancyfoot[C]{\includegraphics[height=0.84cm]{data/imgs/bottom-banner.png}}
\renewcommand{\headrulewidth}{0pt}                        % 移除页眉下方的线
\renewcommand{\footrulewidth}{0pt}                        % 移除页脚上方的线
\setlength{\headheight}{3cm}                              % 根据图片实际高度调整，确保页眉区域足够容纳图片
\setlength{\headsep}{30pt}                                % 页眉与正文之间的间距
\setlength{\textheight}{22.5cm}  
\setlength{\footskip}{2cm}                                % 根据页脚图片高度调整

% 定义自定义字体命令
\newfontfamily\EASTPAKBOLD[
    Path = ./fonts/ ,
    UprightFont = EASTPAK-BOLD.TTF
]{EASTPAK-BOLD}

% 单元格的一些设置
\setlength{\tabcolsep}{10pt}                              % 单元格左右间隙
\setlength{\extrarowheight}{0pt}                          % 彻底消除额外行间距
\renewcommand{\arraystretch}{1.5}                         % 行高最小

\begin{document}

% 合并 data 目录下的 first-page.pdf
\includepdf[pages=-]{data/first-page.pdf}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 添加实验操作人, 报告攥写人, 审核人和检验专用章
\clearpage                                                % 确保插入内容的页面是新页
\noindent
\begin{textblock*}{10cm}(1.565cm, 27.0cm)
    {\CJKfamily{YaHei}\fontsize{8pt}{14pt}\selectfont
    \textcolor[RGB]{68,68,68}{实验操作人: 张三}}
\end{textblock*}

\noindent
\begin{textblock*}{10cm}(6.065cm, 27.0cm)
    {\CJKfamily{YaHei}\fontsize{8pt}{14pt}\selectfont
    \textcolor[RGB]{68,68,68}{报告攥写人: 李四}}
\end{textblock*}

\begin{textblock*}{10cm}(10.565cm, 27.0cm)
  {\CJKfamily{YaHei}\fontsize{8pt}{14pt}\selectfont
  \textcolor[RGB]{68,68,68}{审核人: 王五}}
\end{textblock*}

\begin{textblock*}{10cm}(15.065cm, 27.0cm)
  {\CJKfamily{YaHei}\fontsize{8pt}{14pt}\selectfont
  \textcolor[RGB]{68,68,68}{日期: \today}}
\end{textblock*}

\begin{textblock*}{10cm}(14.20cm, 23.72cm)
  \includegraphics[width=4.0cm]{data/imgs/company_seal.png}
\end{textblock*}
% 添加实验操作人, 报告攥写人, 审核人和检验专用章
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 第二页开始使用页眉页脚
\thispagestyle{fancy}

% 主标题具体上顶端的间距，起始间距
\vspace{1.20cm}

% 添加中文标题
\begin{minipage}[t]{21cm}
  {\CJKfamily{YaHei}\fontsize{24.57pt}{32pt}\selectfont
  \myindent\textbf{\textcolor[RGB]{68,68,68}{单基因病（扩展性）携带者筛查检测报告}}}
\end{minipage}%

% 添加间距
\vspace{0.30cm}

% 添加英文标题
\begin{minipage}[t]{21cm}
  {\EASTPAKBOLD\fontsize{14.29pt}{32pt}\selectfont
  \myindent\textbf{\textcolor[RGB]{68,68,68}{Expanded Carrier Screening, ECS}}}
\end{minipage}

% 添加间距
\vspace{1.00cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 基本信息
% 添加图标和文本，且图片和文字水平对齐
\myindent
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/basic-info.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{基本信息}}}

% 添加间距
\vspace{-0.20cm}

% 添加渐变色条
\myindent\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 添加间距
\vspace{-0.05cm}

% 添加表格
{\CJKfamily{YaHei}
\fontsize{9.00pt}{11pt}\selectfont
\begin{adjustwidth}{1.565cm}{1.565cm}
\begin{tabularx}{\linewidth}{p{3.2cm}|p{3.5cm}|p{4.8cm}|l}
\hline
\textbf{姓名：}某某某 & \textbf{年龄：}29岁 & \textbf{性别：}女 & \textbf{电话：}12345678901 \\
\hline
\textbf{样本类型：}外周血 & \textbf{样本性状：}肉眼未见异常 & \textbf{身份证号：}12345678912345678 & \textbf{条码号：}123456789012 \\
\hline
\textbf{送检医生：} & \multicolumn{2}{l}{\textbf{送检单位：}} &  \\
\hline
\textbf{采样时间：}2025-04-16 & \textbf{收样时间：}2025-04-17 & \multicolumn{2}{l}{\textbf{报告时间：}2025-04-22} \\
\hline
\end{tabularx}
\end{adjustwidth}
}
% 基本信息
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 添加间距
\vspace{1.00cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 检测项目信息
\myindent
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/porj-info.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{检测项目信息}}}

% 添加间距
\vspace{-0.20cm}

% 添加渐变色条
\myindent\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 添加间距
\vspace{-0.05cm}

% 添加表格
{\CJKfamily{YaHei}
\fontsize{9.00pt}{11pt}\selectfont
\begin{adjustwidth}{1.565cm}{1.565cm}
\begin{tabularx}{\linewidth}{p{3.2cm}|l}
\hline
\textbf{检测项目} & 单基因病（扩展性）携带者筛查（434项） \\
\hline
\textbf{检测方法} & 多重扩增子测序 \\
\hline
\textbf{项目编号} & GRS-MG-01  \\
\hline
\end{tabularx}
\end{adjustwidth}
}
% 检测项目信息
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 添加间距
\vspace{1.00cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 检测结论
\myindent
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/porj-info.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{检测结论}}}

% 添加间距
\vspace{-0.20cm}

% 添加渐变色条
\myindent\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 添加间距
\vspace{-0.05cm}

% 添加表格
{\CJKfamily{YaHei}
\fontsize{9.00pt}{11pt}\selectfont
\begin{adjustwidth}{1.565cm}{1.565cm}
\begin{tabularx}{\linewidth}{l}
\hline
\hspace{0.50cm}本次检测，在受检者中检出1个致病突变。  \\
\hline
\end{tabularx}
\end{adjustwidth}
}
% 检测结论
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 添加间距
\vspace{1.00cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 结果信息
\myindent
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/gene-icon.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{结果信息}}}

% 添加间距
\vspace{-0.20cm}

% 添加渐变色条
\myindent\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 添加间距
\vspace{-0.05cm}

% 添加表格
{
\CJKfamily{YaHei}
\fontsize{9.00pt}{11pt}\selectfont
\begin{adjustwidth}{1.565cm}{1.565cm}
  \begin{tabularx}{\linewidth}{
    Y{1.2cm}|C{1.6cm}|Y{1.8cm}|Y{2.0cm}|Y{1.1cm}|Y{1.6cm}|Y{0.7cm}|l
  }
  \hline
  \textbf{基因} & \textbf{染色体位置} & \textbf{转录本编号} & \textbf{核苷酸变化（氨基酸变化）} & \textbf{基因型} & \textbf{致病性分类} & \textbf{遗传方式} & \textbf{疾病} \\
  \hline
  ERCC2 & 19:45352235-45352235 & \makecell[tl]{NM_0004656\\700.4} & \makecell[tl]{c.2164C$>$T\\(p.Arg722Trp)} & 杂合 & 致病变异 & AR & \makecell[tl]{着色性干皮病\\D组} \\
  \hline
  GAA & 17:78012345-78012345 & \makecell[tl]{NM_000152.5} & \makecell[tl]{c.2560C$>$T\\(p.Arg854Trp)} & 杂合 & 致病变异 & AR & \makecell[tl]{糖原累积病II型} \\
  \hline
  \end{tabularx}
\end{adjustwidth}
}
\noindent\mypassage{\CJKfamily{YaHei}\fontsize{7pt}{10pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{备注:}}}

\noindent\mypassage{\CJKfamily{YaHei}\fontsize{7pt}{10pt}\selectfont{\textcolor[RGB]{68,68,68}{1) AD表示常染色体显性遗传，AR表示常染色体隐性遗传，XLD表示X染色体连锁显性遗传，XLR表示X染色体连锁隐性遗传；}}

\noindent\mypassage{\CJKfamily{YaHei}\fontsize{7pt}{10pt}\selectfont{\textcolor[RGB]{68,68,68}{2) 本报告参考hg38⼈类基因组版本。}}

% 结果信息
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 开始新的一页
\newpage 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 检测结果说明及建议
%
\myindent\myindenr\includegraphics[width=17.88cm]{data/imgs/result_introduction_recommendation.png}

% 添加间距
\vspace{1.00cm}

% logo + 检测结果
\myindent\myindenr
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/basic-info.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{检测结果}}}

% 添加渐变色条
\myindent\myindenr\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 主要文本
\vspace{0.20cm}
\noindent\begin{minipage}{19.385cm}
  \noindent
  \parshape=1        % 只定义一个统一的缩进规则
  1.565cm 17.820cm   % 首行缩进 1.565cm，文本宽度为 17.820cm
  {\CJKfamily{YaHei}\fontsize{9pt}{20pt}\selectfont{\textcolor[RGB]{68,68,68}{
  本次检测，在受检者中检出着色性干皮病D组相关的ERCC2基因的1个杂合型致病变异c.2164C>T。根据常染色体隐性遗传方式，受检者有1/2的可能将该变异遗传给后代。若配偶也携带有该基因的致病或可能致病变异，则后代有1/4的风险患病，建议配偶进行遗传咨询及相关检测。
  }}}
\end{minipage}

% 检测结果说明及建议
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 添加间距
\vspace{1.00cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 基因与疾病
%
\myindent\myindenr\includegraphics[width=17.88cm]{data/imgs/gene_and_disease.png}

% 添加间距
\vspace{1.00cm}

% logo + 疾病名称
\myindent\myindenr
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/gene-icon.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{着色性干皮病D组}}}

% 添加渐变色条
\myindent\myindenr\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 主要文本
\vspace{0.20cm}
\noindent\begin{minipage}{19.385cm}
  \noindent
  \parshape=1        % 只定义一个统一的缩进规则
  1.565cm 17.820cm   % 首行缩进 1.565cm，文本宽度为 17.820cm
  {\CJKfamily{YaHei}\fontsize{9pt}{20pt}\selectfont{\textcolor[RGB]{68,68,68}{
    着色性干皮病是一种常染色体隐性遗传病，表现为皮肤对日光过敏、暴露在阳光下的区域具有发展成癌症的高度趋势，有时还会表现出神经性异常。皮肤显现显著的斑点和其他着色异常。着色性干皮病互补D组（XP-D）是一种临床异质型疾病，患者表现出典型的XP症状（皮肤光敏感性致灼伤和斑点，皮肤干燥和皮肤癌），并有不同程度的神经性异常（从无异常到严重的神经性异常）有关。一些XP-D患者出现科凯恩综合征的症状，包括恶病质性侏儒、色素性视网膜、共济失调和神经传导速度下降。这种兼有着色性干皮病和科凯恩综合征特征的表型被称为XP-CS复合征。
  }}}
\end{minipage}

% 基因与疾病
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 开始新的一页
\newpage 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 底部温馨提示
\clearpage
\noindent
\begin{textblock*}{10cm}(7.565cm, 24.0cm)

  \noindent\begin{minipage}{19.385cm}
    \noindent
    \parshape=1
    1.565cm 9.820cm
    {\CJKfamily{YaHei}\fontsize{7pt}{20pt}\selectfont{\textcolor[RGB]{68,68,68}{
      温馨提示：\newline
      1：请扫描左方的奥测医学二维码，点击“检测服务”，在选择点击“报告解读”，填写“报告解读预约”：联系人、联系电话、预约时间段，我们将竭诚为您专业提供报告咨询及解读服务。\newline\newline
      2：若想获得检测范围内疾病、基因、携带率等相关信息，请扫描左方的二维码，查看“单基因病（扩展性）携带者筛查”，了解更多相关疾病信息。
    }}}
  \end{minipage}
\end{textblock*}

\begin{textblock*}{10cm}(1.565cm, 23.7cm)
  \includegraphics[width=3.2cm]{data/imgs/aoce_qc.png}
\end{textblock*}

\begin{textblock*}{10cm}(5.065cm, 23.7cm)
  \includegraphics[width=3.2cm]{data/imgs/reporter_qc.png}
\end{textblock*}

% 底部温馨提示
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 建议

% logo + 建议
\myindent\myindenr
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/gene-icon.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{建议}}}

% 添加渐变色条
\myindent\myindenr\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 添加间距
\vspace{1.00cm}

% 图一
\myindent\myindent\myindenr
\raisebox{-0.5\height}{\includegraphics[width=5.88cm]{data/imgs/recommendation_one.png}}
\myindent\myindenr\raisebox{0.3 \height}{\includegraphics[width=0.5cm]{data/imgs/gene-icon.png}}
\noindent\begin{minipage}{19.385cm}
  \noindent
  \parshape=1
  0.220cm 8.820cm
  {\CJKfamily{YaHei}\fontsize{9pt}{20pt}\selectfont{\textcolor[RGB]{68,68,68}{
    受检者如有生育要求，建议妊娠前再次接受遗传咨询，获得关于再发风险的指导、辅助生殖技术及产前诊断等详细信息，检测结果与疾病相关性需咨询医院临床医疗人员。
  }}}
\end{minipage}

% 添加间距
\vspace{0.50cm}
\hspace{1.5cm}\color{separationline}\rule{17.00cm}{0.25pt}
\vspace{0.30cm}

% 图二
\myindent\myindent\myindenr
\raisebox{-0.5\height}{\includegraphics[width=5.88cm]{data/imgs/recommendation_two.png}}
\myindent\myindenr\raisebox{2.2 \height}{\includegraphics[width=0.5cm]{data/imgs/gene-icon.png}}
\noindent\begin{minipage}{19.385cm}
  \noindent
  \parshape=1
  0.220cm 8.820cm
  {\CJKfamily{YaHei}\fontsize{9pt}{20pt}\selectfont{\textcolor[RGB]{68,68,68}{
    对于常染色体隐性疾病携带者，虽不立即产生临床症状，但该位点仍有遗传风险。建议对配偶进行携带者筛查，根据常染色体隐性遗传方式，若配偶也携带有该基因的致病变异，则后代有1/4的患病风险。建议该类夫妇进行遗传咨询和产前诊断，在知情同意的原则下尊重夫妇双方的选择，以降低子代生育风险。对于选择辅助生殖的夫妇，可选择胚胎植入前遗传学诊断（PGD）技术助孕，挑选健康的胚胎移植，降低孕育患儿的风险。
  }}}
\end{minipage}

% 建议
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 添加间距
\vspace{1.00cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 参考文献

% logo + 参考文献
\myindent\myindenr
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/basic-info.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{参考文献}}}

% 添加渐变色条
\myindent\myindenr\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 添加间距
\vspace{1.00cm}

% 参考文献内容
\vspace{0.20cm}
\noindent\begin{minipage}{19.385cm}
  \noindent
  \parshape=1
  1.565cm 17.820cm
  {\CJKfamily{YaHei}\fontsize{7pt}{30pt}\selectfont{\textcolor[RGB]{68,68,68}{
    [1].Righetti S, Dive L, Archibald A D, et al. Correspondence on "Screening for autosomal recessive and X-linked conditions during pregnancy and preconception: a practice resource of the American College of Medical Genetics and Genomics (ACMG)" by Gregg etal. 2022.\newline
    [2].Romero, Stephanie, Rink, et al. Carrier Screening in the Age of Genomic Medicine[J]. Obstetrics \& Gynecology Journal of the American College of Obstetricians \& Gynecologists, 2017.\newline
    [3]. Mcgurk K A, Zheng S L, Henry A, et al. Correspondence on ACMG STATEMENT: ACMG SF v3.0 list for reporting of secondary findings in clinical exome and genome sequencing: a policy statement of the American College of Medical Genetics and Genomics (ACMG) by Miller et al[J]. 2022.\newline
    [4]. Murray, M.F., Giovanni, M.A., Doyle, D.L. et al. DNA-based screening and population health: a point to consider statement for programs and sponsoring organizations from the American College of Medical Genetics and Genomics (ACMG). Genet Med 23, 989–995 (2021).\newline
    [5]. Hendrickson B C, Donohoe C, Akmaev V R. Differences in SMN1 allele frequencies among ethnic groups within North America.\newline
    [6]. Peyser A, Singer T, Mullin C, et al. Comparing ethnicity-based and expanded carrier screening methods at a single fertility center reveals significant differences in carrier rates and carrier couple rates[J]. Genet Med, 2019(6).\newline
    [7]. Gregg, A.R. Message from ACMG President: overcoming disparities. Genet Med 22, 1758 (2020). https://doi.org/10.1038/s41436-020-0882-6.\newline
    [8]. Richards S, Aziz N, Bale S, et al. Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology. Genetics in medicine, 2015, 17(5): 405.\newline
    [9]. 遗传变异分类标准与指南.中国科学:生命科学,2017(06):76-96.\newline
  }}}
\end{minipage}

% 参考文献
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 开始新的一页
\newpage 

%
\myindent\myindenr\includegraphics[width=17.88cm]{data/imgs/appendix.png}

% 添加间距
\vspace{1.00cm}

% logo + 检测方法说明
\myindent\myindenr
\raisebox{-0.15\height}{\includegraphics[width=0.5cm]{data/imgs/porj-info.png}}
\myindenr
{\CJKfamily{YaHei}\fontsize{12pt}{32pt}\selectfont\textbf{\textcolor[RGB]{68,68,68}{检测方法说明}}}

% 添加渐变色条
\myindent\myindenr\includegraphics[width=17.88cm,height=0.15cm]{data/imgs/bar-span.png}

% 检测方法说明
\vspace{0.50cm}
\noindent\begin{minipage}{19.385cm}
  \noindent
  \parshape=1
  1.565cm 17.820cm
  {\CJKfamily{YaHei}\fontsize{9pt}{32pt}\selectfont{\textcolor[RGB]{68,68,68}{
    本项目利用多重扩增子靶向测序技术，设计了14,044个扩增子用以分析420个基因的编码区，同时包括基因外显子上下游25bp区域，以及某些特定的基因间区域、内含子和高度同源区域。该项目靶向区域包括自ClinVar和本地数据库超过28,530个假定的携带者单核苷酸变异(SNVs)和插入缺失变异(indels)。通过进行高通量测序和生物信息学分析，获取目标区域的基因变异信息，快速准确地检出致病/可能致病变异。\newline\newline
    同时，利用本公司的专利技术，可以对相关基因的拷贝数变异（CNV）进行检测及分析，除了以下基因\textit{USH2A} (\textit{CDS5}); \textit{SLC3A1} (\textit{CDS9}); \textit{PREPL} (\textit{CDS2}); \textit{NEB} (\textit{CDS74}, 82-85, 91-93, 98-101, 160); \textit{VPS13A} (\textit{CDS74}); \textit{FANCC} (\textit{CDS11}); \textit{ATM} (\textit{CDS11}, 42); \textit{PAH} (\textit{CDS1}, 10); \textit{GALC} (\textit{CDS1}); \textit{HEXA} (\textit{CDS1}); \textit{CLN3} (\textit{CDS7}); \textit{ITGB3} (\textit{CDS15}); \textit{SAMHD1} (\textit{CDS1}); \textit{DMD} (\textit{CDS1}, 8, 18, 26, 66, 83, 85); \textit{GLA} (\textit{CDS4}) 以外，其他相关具有CNV致病变异的基因发生单个区段100Kb以上的缺失和重复可以被检测。但是为了保证准确性，拷贝数检出需要三个或更多的扩增子测序数据，因此算法对单个外显子级别的CNV的敏感性可能取决于邻近区域的覆盖范围、扩增子的特异性。\textbf{\textcolor[RGB]{0,89,186}{本项目在CNV检测方面有明显的提升，包括外显子水平del/dup的检测，但NGS非常规CNV检测方法，存在一定程度假阳性和假阴性的风险。}}\newline\newline
    针对检测范围包含假基因、高度相关的同源基因或其他同源性相关问题的基因通过设计特殊的分析方法来检测，可以检测范围包括：\textit{SMN1}基因7号外显子缺失，\textit{HBA1}/\textit{HBA2}基因--SEA, -alpha3.7, -alpha4.2大片段缺失。\textbf{\textcolor[RGB]{0,89,186}{在其他高度同源基因或假基因的检测中存在假阳性及假阴性风险。}}
  }}}
\end{minipage}

% 添加间距
\vspace{1.00cm}

% 绘制蓝色虚线文本框
\tcbset{
  colframe=white,          % 设置边框颜色为白色。由于边框颜色为白色，可能在白色背景上不可见
  colback=white,           % 设置文本框背景颜色为白色
  boxrule=0.15mm,          % 设置边框线宽为 0.15 毫米
  rounded corners,         % 启用圆角边框效果
  arc=5mm,                 % 设置圆角的半径为 5 毫米
  left=0.5cm,              % 设置文本框内容的左内间距为 0.5 厘米
  right=0.5cm,             % 设置文本框内容的右内间距为 0.5 厘米
  top=0.5cm,               % 设置文本框内容的上内间距为 0.5 厘米
  bottom=0.5cm,            % 设置文本框内容的下内间距为 0.5 厘米
  width=17.87cm,           % 设置文本框的总宽度为 17.87 厘米
  center,                  % 设置文本框在页面上居中对齐
  enhanced,                % 启用 tcolorbox 的增强功能，支持更高级的样式（如 TikZ 绘图）
  borderline={0.15mm}{0mm}{dash pattern=on 5pt off 2pt, draw=separationline} % 定义边框线样式
                           % 第一个参数 {0.15mm}：边框线宽为 0.15 毫米
                           % 第二个参数 {0mm}：边框线与文本框边界的偏移量为 0 毫米（即无偏移）
                           % 第三个参数 {dash pattern=on 5pt off 2pt, draw=separationline}：
                           %   - dash pattern=on 5pt off 2pt：设置虚线样式，实线段长 5 点，空白段长 2 点
                           %   - draw=separationline：设置边框线颜色为 separationline（需提前定义此颜色）
}

% 参考文献内容
\vspace{0.20cm}
\begin{tcolorbox}
  {\CJKfamily{YaHei}\fontsize{7pt}{30pt}\selectfont{\textcolor[RGB]{68,68,68}{
  声明:\newline
  1.	该项目基于高通量测序方法学，在相同核苷酸数目大于8个时，对插入缺失突变的检出灵敏度受到限制。 同时在序列复杂度低、区域拷贝数变化大、存在大的插入缺失以及与其他基因组位点高度同源的区域，变异的检出也会收到影响。\newline
  2.	如果文献以及数据库之前没有报道过的相关变异，本次检出不会被报出。\newline
  3.	本检测不能排除受检者下一代会因为新发（de novo）变异而导致相关疾病发生的可能。\newline
  4.	本检测在检出2个拷贝的SMN17号外显子时，无法确定SMN1基因的2个拷贝是“2+0”型（携带者，2个拷贝在同一个染色体上）还是“1+1”基因型（正常，2个拷贝在一对同源染色体上）。\newline
  5.	本项目包含基因无法检测AR基因动态突变，F8:(intron 1 inversion、intron 22 inversion)，以及相关基因生殖细胞嵌合突变。\newline
  6.	本项目在同一个基因上检出2个变异时，无法分辨检出的2个变异是一个等位基因上形成顺式还是一对等位基因上形成反式。\newline
  7.	本检测仅对承诺范围内的疾病、基因和位点进⾏携带者变异情况筛查。但鉴于当前医学检测技术⽔平的限制和受检者个体差异等不同原因，即使在检测⼈员已经履⾏了⼯作职责和操作程序的前提下，仍可能有⼀些位点未检出。\newline
  8.	本检测只分析已知致病变异和疑似致病变异，不包括临床意义未明变异、可能良性变异和良性变异。\newline
  9.	数据解读规则参考美国医学遗传学和基因组学学院（American College of Medical Genetics and Genomics, ACMG）相关指南。变异致病性的判定依据现有的临床表型、⽂献报道和数据库及⽣物信息学软件判定，受科学发展的阶段性限制。随着时间推移，我们会获得更多关于这些基因的信息，我们的解读结果有可能会发⽣变化。\newline
  10.	以上结论均为实验室检测数据，仅用于变异检测目的，不代表最终诊断结果，临床表型相关性解释请咨询医生。\newline
  本报告结果只对送检样品负责。本中⼼对以上检测结果保留最终解释权，如有疑义，请在收到结果后的7个工作日内与我们联系。
  }}}
\end{tcolorbox}

% 合并 data 目录下的 last-page.pdf
\includepdf[pages=-]{data/last-page.pdf}

\end{document}
```

