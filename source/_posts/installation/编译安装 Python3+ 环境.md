---
title: 编译安装 Python3+ 环境
date: 2025-05-01 19:37:45
tags: [Python]
categories: [[安装说明, Python]]
---

# Python3.0+，一种面向对象、解释型计算机程序语言

[Python官方下载网址](https://www.python.org/downloads/source/)，本例以 [Python-3.10.8.tgz](https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz) 为例。

# 下载或本地上传 Python 安装包

```shell
# https://www.python.org/downloads/source/

cd /tools
wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz
```

# 解压缩文件

```shell
tar -xzvf Python-3.10.8.tgz
```

# 编译安装

```shell
cd /tools/Python-3.10.8

./configure --prefix=/tools/Python-3.10.8  --enable-optimizations
./configure --prefix=/tools/Python-3.10.8                             # 跳过`test`步骤.
make
make install
```

```python
# 测试安装结果
$ ./python 
Python 3.10.8 (main, May  1 2025, 19:44:42) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import os
>>> 
```

# 安装 pip 命令

```shell
# 在`Python`的`HOME`目录下：
cd /tools/Python-3.10.8 

# apt-get install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
./bin/python3.10 get-pip.py 
```

# 安装环境依赖或解决报错

## configure: error: no acceptable C compiler found in $PATH

```shell
# ERROR: configure: error: no acceptable C compiler found in $PATH
checking for gcc... no
checking for cc... no
checking for cl.exe... no
configure: error: in `/root/tools/Python-3.8.3':
configure: error: no acceptable C compiler found in $PATH
See `config.log' for more details

$ apt-get install cmake
```

## ModuleNotFoundError: No module named 'zlib'

```shell
  File "<frozen zipimport>", line 520, in _get_decompress_func
ModuleNotFoundError: No module named 'zlib'
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "<frozen zipimport>", line 568, in _get_data
  File "<frozen zipimport>", line 523, in _get_decompress_func
zipimport.ZipImportError: can't decompress data; zlib not available

Makefile:1186: recipe for target 'install' failed
make: *** [install] Error 1

$ apt-get install zlib1g zlib1g-dev
```

## ModuleNotFoundError: No module named '_ctypes'

```shell
# Error: ModuleNotFoundError: No module named '_ctypes'
Traceback (most recent call last):
  File "setup.py", line 8, in <module>
    from setuptools import find_packages, setup
  File "/root/tools/Python-3.8.3/lib/python3.8/site-packages/setuptools/__init__.py", line 20, in <module>
    from setuptools.dist import Distribution, Feature
  File "/root/tools/Python-3.8.3/lib/python3.8/site-packages/setuptools/dist.py", line 35, in <module>
    from setuptools import windows_support
  File "/root/tools/Python-3.8.3/lib/python3.8/site-packages/setuptools/windows_support.py", line 2, in <module>
    import ctypes
  File "/root/tools/Python-3.8.3/Lib/ctypes/__init__.py", line 7, in <module>
    from _ctypes import Union, Structure, Array
ModuleNotFoundError: No module named '_ctypes'

$ apt-get install libffi-dev
```

## the ssl module in Python is not available

```shell
# WARNING: the ssl module in Python is not available
WARNING: pip is configured with locations that require TLS/SSL, however the ssl module in Python is not available.
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError("Can't connect to HTTPS URL because the SSL module is not available.",)': /simple/pip/

$ apt-get install openssl libssl-dev
```

## ModuleNotFoundError: No module named '_bz2'

```shell
# Error ...
import bz2
  File "/home/albert/mysoft/Python-3.6.8/Lib/bz2.py", line 23, in <module>
    from _bz2 import BZ2Compressor, BZ2Decompressor
ModuleNotFoundError: No module named '_bz2'

$ apt-get install libbz2-dev
```

