---
title: 本地配置 Latex + VScode 环境
date: 2025-04-23 22:11:45
tags: [Latex, VScode]
categories: [Latex]
---
https://blog.csdn.net/zheliku/article/details/146968842

要在本地配置 LaTeX 和 VS Code 环境，首先需要安装一个 LaTeX 发行版（如 TeX Live、MikTeX 或 MacTeX），它提供了编译 LaTeX 文件所需的工具；然后安装 VS Code 编辑器，并添加 LaTeX Workshop 插件，这个插件可以帮助你进行代码高亮、自动补全、实时预览等操作。安装完成后，需要在 VS Code 的设置中配置 LaTeX 的编译工具链，并选择适合的编译流程（如 `pdflatex` 或 `xelatex`）。最后，通过创建一个 `.tex` 文件编写测试文档，保存后插件会自动编译，生成 PDF 文件，你可以直接在 VS Code 中查看输出结果。如果需要支持中文，可以使用 `ctex` 宏包，确保正确安装中文字体相关的包。这样，一个简单高效的 LaTeX 工作环境就搭建完成了！

# 安装 Latex 环境

本文基于 Windows 11 系统进行演示。

## 下载 texlive.iso 文件

打开清华大学开源软件镜像站：[https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/Images/](https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/Images/)，下载`texlive.iso`文件：

<img src='/imgs/微信截图_20250423220747.png'>

## 使用资源管理器打开

右键使用`Windows资源管理器`打开这个 .iso 文件。

<img src='/imgs/wechat_2025-04-24_140141_566.png'>

## 运行安装程序

进入上一步生成的`TeXLive2025`驱动器中，找到`install-tl-windows.bat`文件，右键，选择**以管理员身份运行**进行安装，期间会弹出一个交互框，可以参考下面设置进行配置。

<img src='/imgs/wechat_2025-04-24_140710_598.png'>

## 设置配置

<img src='/imgs/wechat_2025-04-24_141539_951.png'>

```shell
# Installation root: 指定安装路径, 默认即可
# 是否安装 TeXworks 前端, 取消勾选 (默认是勾选的), 因为我们这里使用 VScode 就不需要这个了。
```

设置完成后点击安装，需要等待一段时间，预计半个多小时吧，期间不需要做什么了，不要关掉程序即可，可以自行安排做其他事情。

当出现运行到下面这一步时候，表示整个安装程序完成，点击关闭即可。

<img src='/imgs/wechat_2025-04-24_150407_533.png'>

## 检查安装

从下面搜索栏检索 pow，会显示一些关联的程序，点击打开`Windows PowerShell`，然后输入`latex`，检查是否安装成功。

<img src='/imgs/wechat_2025-04-24_150709_416.png'>

<img src='/imgs/wechat_2025-04-24_151028_951.png'>



安装成功之后，就可以将下载的安装包以及解压出来的中间文件清理掉啦。

# 安装 VScode 环境

## 下载安装包

打开 VScode 官网：[https://code.visualstudio.com/](https://code.visualstudio.com/)，下载安装包。

<img src='/imgs/wechat_2025-04-24_151248_280.png'>

## 运行安装包

运行下载的安装包，这里我下载的是这个版本 VSCodeUserSetup-x64-1.99.3.exe，勾选`创建桌面快捷方式`和`添加到系统 PATH`。

> 安装路径自行修改，但严禁路径含中文或空格。



提示出现这个，点击`确定`直接进入安装即可。

<img src='/imgs/wechat_2025-04-24_151611_984.png'>



中间有几步确认的地方，比如安装路径等等，根据自己情况设置就好。对于`选择附加任务`部分，可以参考这个勾选，影响不大。

<img src='/imgs/wechat_2025-04-24_151821_421.png'>

然后点击下一步继续点击安装即可，这个安装很快，不到一分钟即可。

## 安装插件及环境配置

### 汉化 VScode (可选)

一些小伙伴不习惯英文界面，可以通过以下步骤进行汉化操作。

打开 VScode，点击左侧插件图标，搜索`Chinese`，安装汉化插件。安装完成后，点击重启，启用汉化即可。

<img src='/imgs/wechat_2025-04-24_152356_123.png'>

### 安装 Latex 插件

点击左侧插件图标，搜索`latex`，安装`LaTeX Workshop`插件。

<img src='/imgs/wechat_2025-04-24_152701_560.png'>



点击左下角设置按钮，选择设置，打开设置面板。

<img src='/imgs/wechat_2025-04-24_152909_278.png'>



点击右上角文件按钮，打开设置的 json 配置文件，将下面的代码拷贝到 json 文件中，然后保存即可。

```json
{
    "workbench.colorTheme": "Default Light Modern",

    "editor.wordWrap": "on",

    //------------------------------LaTeX 配置----------------------------------

    // 右键菜单
    "latex-workshop.showContextMenu": true,

    // 从使用的包中自动补全命令和环境
    "latex-workshop.intellisense.package.enabled": true,

    // 编译出错时设置是否弹出气泡设置
    "latex-workshop.message.error.show": false,
    "latex-workshop.message.warning.show": false,

    // 输出路径
    "latex-workshop.latex.outDir": "./build",

    // 编译工具和命令
    "latex-workshop.latex.tools": [
        {
            "name": "xelatex",
            "command": "xelatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "--output-directory=%OUTDIR%",
                "%DOCFILE%"
            ]
        },
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "--output-directory=%OUTDIR%",
                "%DOCFILE%"
            ]
        },
        {
            "name": "latexmk",
            "command": "latexmk",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-pdf",
                "--output-directory=%OUTDIR%",
                "%DOCFILE%"
            ]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
                "%OUTDIR%/%DOCFILE%",
                // "--output-directory=%OUTDIR%",
            ]
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "XeLaTeX",
            "tools": [
                "xelatex"
            ]
        },
        {
            "name": "PDFLaTeX",
            "tools": [
                "pdflatex"
            ]
        },
        {
            "name": "BibTeX",
            "tools": [
                "bibtex"
            ]
        },
        {
            "name": "LaTeXmk",
            "tools": [
                "latexmk"
            ]
        },
        {
            "name": "xelatex -> bibtex -> xelatex*2",
            "tools": [
                "xelatex",
                "bibtex",
                "xelatex",
                "xelatex"
            ]
        },
        {
            "name": "pdflatex -> bibtex -> pdflatex*2",
            "tools": [
                "pdflatex",
                "bibtex",
                "pdflatex",
                "pdflatex"
            ]
        },
    ],
    
    // 文件清理。此属性必须是字符串数组
    "latex-workshop.latex.clean.fileTypes": [
        "*.aux",
        "*.bbl",
        "*.blg",
        "*.idx",
        "*.ind",
        "*.lof",
        "*.lot",
        "*.out",
        "*.toc",
        "*.acn",
        "*.acr",
        "*.alg",
        "*.glg",
        "*.glo",
        "*.gls",
        "*.ist",
        "*.fls",
        "*.log",
        "*.fdb_latexmk",
        // "*.synctex.gz"
    ],

    // 设置为 onBuilt 在构建成功后清除辅助文件
    "latex-workshop.latex.autoClean.run": "onBuilt",

    // 使用第一个编译组合
    "latex-workshop.latex.recipe.default": "lastUsed",
    
    // 设置是否自动编译
    "latex-workshop.latex.autoBuild.run": "never",

    // 用于反向同步的内部查看器的键绑定。ctrl/cmd +点击(默认)或双击
    "latex-workshop.view.pdf.internal.synctex.keybinding": "double-click",
}
```

<img src='/imgs/wechat_2025-04-24_153455_679.png'>



到此，整个配置环境完成了，电脑该重启的可以重启了。

## 配置说明 (可选)

### 编译输出路径

这里设置为当前目录下的`./build`文件夹，设置为空则是同目录。

```json
// 输出路径
"latex-workshop.latex.outDir": "./build",
```

### 编译工具命令

定义了单个工具，具体内容看注释。参考文档：[Compile · James-Yu/LaTeX-Workshop Wiki](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile#placeholders)。

```json
// 编译工具和命令
"latex-workshop.latex.tools": [
    {
        "name": "xelatex",
        "command": "xelatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "--output-directory=%OUTDIR%",
            "%DOCFILE%"
        ]
    },
    {
        "name": "pdflatex",
        "command": "pdflatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "--output-directory=%OUTDIR%",
            "%DOCFILE%"
        ]
    },
    {
        "name": "latexmk",
        "command": "latexmk",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-pdf",
            "--output-directory=%OUTDIR%",
            "%DOCFILE%"
        ]
    },
    {
        "name": "bibtex",
        "command": "bibtex",
        "args": [
            "%OUTDIR%/%DOCFILE%",
            // "--output-directory=%OUTDIR%",
        ]
    }
],
```

### 编译工具链

定义编译工具链，由一个或多个工具组成：

```json
"latex-workshop.latex.recipes": [
    // 只运行一次 XeLaTeX，用于支持中文、字体等特性
    {
        "name": "XeLaTeX",         // 菜单显示的名字
        "tools": [
            "xelatex"              // 调用的工具：xelatex
        ]
    },
    // 只运行一次 PDFLaTeX，适用于简单的英文文档
    {
        "name": "PDFLaTeX",
        "tools": [
            "pdflatex"             // 调用的工具：pdflatex
        ]
    },
    // 只运行一次 BibTeX，生成参考文献的辅助工具（需单独运行）
    {
        "name": "BibTeX",
        "tools": [
            "bibtex"               // 调用的工具：bibtex
        ]
    },
    // 使用 Latexmk 自动化工具，智能判断需要运行哪些步骤（推荐，自动化程度高）
    {
        "name": "LaTeXmk",
        "tools": [
            "latexmk"              // 调用的工具：latexmk
        ]
    },
    // 先运行 XeLaTeX，再运行 BibTeX，然后再连续运行两次 XeLaTeX
    // 用于需要参考文献的文档，确保引用和目录都正确
    {
        "name": "xelatex -> bibtex -> xelatex*2",
        "tools": [
            "xelatex",             // 第一步：xelatex 编译一次，生成 aux 文件
            "bibtex",              // 第二步：用 bibtex 处理参考文献
            "xelatex",             // 第三步：xelatex 再编译一次，整合引用
            "xelatex"              // 第四步：再次 xelatex，确保所有引用都到位
        ]
    },
    // 先运行 PDFLaTeX，再运行 BibTeX，然后再连续运行两次 PDFLaTeX
    // 和上面类似，只是用 pdflatex 代替 xelatex
    {
        "name": "pdflatex -> bibtex -> pdflatex*2",
        "tools": [
            "pdflatex",            // 第一步：pdflatex 编译一次
            "bibtex",              // 第二步：处理参考文献
            "pdflatex",            // 第三步：再编译一次
            "pdflatex"             // 第四步：再编译一次，确保引用和目录正确
        ]
    }
],
```

可以理解为一个**编译流程**，比如**先用 xelatex 编译，再用 bibtex 处理文献，再连续编译两次 xelatex**，确保目录、引用、参考文献全部正确。

### 文件清理

编译过程中，latex 会生成许多中间文件。

- `clean.fileTypes`：定义编译完成后清理的文件后缀名
- `autoClean.run`：决定编译完成后是否清理文件

```json
// 文件清理。此属性必须是字符串数组
"latex-workshop.latex.clean.fileTypes": [
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.idx",
    "*.ind",
    "*.lof",
    "*.lot",
    "*.out",
    "*.toc",
    "*.acn",
    "*.acr",
    "*.alg",
    "*.glg",
    "*.glo",
    "*.gls",
    "*.ist",
    "*.fls",
    "*.log",
    "*.fdb_latexmk",
    // "*.synctex.gz"
],

// 设置为 onBuilt 在构建成功后清除辅助文件
"latex-workshop.latex.autoClean.run": "onBuilt",
```

### 自动编译

可以选择在文件内容更改时或保存文件时自动编译。**文件较大时，编译时间会很长**，因此谨慎选择，个人不习惯自动编译，因此选择`never`。

```json
// 使用上一个编译组合
"latex-workshop.latex.recipe.default": "lastUsed",

// 设置是否自动编译
"latex-workshop.latex.autoBuild.run": "never",
```

### 反向定位

设置从 pdf 到 tex 文件的反向定位方式，个人习惯鼠标双击定位。

```json
// 用于反向同步的内部查看器的键绑定。ctrl/cmd +点击(默认)或双击
"latex-workshop.view.pdf.internal.synctex.keybinding": "double-click",
```

### 搜索对应内容配置

 上述所有内容可直接在用户设置中搜索对应内容配置，并查看相关介绍，以反向定位为例：

<img src='/imgs/wechat_2025-04-24_154952_091.png'>

# 简单示例

## 创建目录

创建 test 文件夹，使用 VScode 打开该文件夹。

<img src='/imgs/wechat_2025-04-24_161252_556.png'>

## 新建 new.tex 文件

<img src='/imgs/wechat_2025-04-24_161407_796.png'>



然后在新建的 .tex 文件内，输入以下测试内容：

```tex
\documentclass{article} 

\begin{document}

hello world!

\end{document}
```

<img src='/imgs/wechat_2025-04-24_162153_610.png'>

## 编译文件

进入 TEX 插件页面，选择编译方式，点击编译。这里选择 LaTeXmk，或者选择 XeLaTeX 或 PDFLaTeX 也可。

若左侧没有 TEX 插件栏，新建 tex 文件即可显示。

<img src='/imgs/wechat_2025-04-24_162410_907.png'>



回到文件页面，可看到当前目录下新建了 build 文件夹或者直接进入工作目录，编译的 pdf 放在该目录中。右上角点击双栏按钮可在右侧查看编译后的 pdf。

说明：`new.synctex.gz”`存放反向编译内容，删除后不可以双击 pdf 定位到对应 tex 位置。

<img src='/imgs/wechat_2025-04-24_162607_322.png'>

## 快速编译

编译一次后，可点击左上角编译按钮进行快速编译，选择的编译方式由如下设置决定：

```json
// 使用上一个编译组合
"latex-workshop.latex.recipe.default": "lastUsed",
```

<img src='/imgs/wechat_2025-04-24_163023_686.png'>

# 如何加快编译

当 .tex 文件很大的时候，编译有时候会很慢，可以通过一些设置尽可能加快编译过程。

## 取消压缩

 最立竿见影的方法是取消 LaTeX 编译器对 PDF 的压缩，方法是在 tex 文档开头加入如下代码：

```tex
\documentclass{article} 

\usepackage{...}

\special{dvipdfmx:config z 0} % 取消 PDF 压缩，加快速度，最终版本生成时注释该行

\begin{document}

hello world!

\end{document}
```

> 该方法可以将速度加快 3~5 倍（60s -> 15s），但代价是 PDF 大小会成倍增加（60MB -> 700MB）。

## 使用 PDF 代替图片

 对于图像文件，建议保存为 .pdf 。

因为 .pdf 图像的编译速度比 .png 文件快，其编译过程不需要调用 libpng 库。

> 参考链接：[https://cn.overleaf.com/learn/how-to/Optimising_very_large_image_files](https://cn.overleaf.com/learn/how-to/Optimising_very_large_image_files)。



