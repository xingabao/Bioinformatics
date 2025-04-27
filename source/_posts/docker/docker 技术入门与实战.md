---
title: docker 技术入门与实战
date: 2021-05-09 14:49:23
tags: [docker]
categories: [[术语介绍, docker]]
---

categories: [[术语介绍, docker]]
---

# Docker 与容器

## 什么是 Docker ?

### Docker 开源项目背景

Docker 是基于 Go 语言实现的开源容器项目，它诞生于 2013 年年初，最初发起者是 dotCloud 公司。Docker 自开源后受到业界广泛的关注和参与，目前已有 80 多个相关开源组建项目，逐渐形成了围绕 Docker 容器的完整的生态体系。目前，Docker 项目已加入 Linux 基金会，并遵循 Apache 2.0 协议，全部开源代码均在`http://github.com/docker`项目仓库中进行维护。

现在主流的操作系统，包括 Linux 各大发行版、macOS、Windows等都已经支持 Docker 。Docker 的构想是要实现 **Build, Ship and Run Any App, Anywhere**，即通过对应用的封装`Packaging`、分发`Distribution`、部署`Depolyment`和运行`Runtime`生命周期进行管理，达到应用组件级别的`一次封装，到处运行`。这里的应用组件，即可以是一个 Web 应用、一个编译环境，也可以是一套数据库平台服务，甚至是一个操作系统或集群。

基于 Linux 平台上的多项开源技术，Docker 提供了高效、敏捷和轻量级的容器方案，并支持部署到本地环境和多种主流云平台。可以说，Docker 首次为应用的开发、运行和部署提供了`一站式`的实用解决方案。

### Linux 容器技术

与大部分新兴技术的诞生一样，Docker 也并非是从石头缝里面蹦出来的，而是站在前任的肩膀上。其中，最重要的就是 Linux Containers，LXC，也就是 Linux 容器技术。容器有效地将由单个操作系统管理地资源划分到孤立的组中，以更好地在孤立的组之间平衡有冲突的资源使用需求。与虚拟化相比，这样既不需要指令级模拟，也不需要即时翻译。容器可以在核心 CPU 本地运行指令，而不需要任何专门的解释机制。此外，也避免了准虚拟化和系统调用替换中的复杂性。

### Linux 容器到 Docker

在 LXC 的基础上，Docker 进一步优化了容器的使用体验，Docker 提供了各种容器管理工具，让用户无须关注底层的操作，更加简单明了地管理和使用容器。其次，Docker 通过引入分层文件系统构建和高效地镜像机制，降低了迁移难度，极大地改善了用户的体验。

简单来讲，读者可以将 Docker 容器理解为一种轻量级的沙盒，每个容器内运行着一个应用，不同的容器相互隔离，容器之间也可以通过网络相互通信。容器的创建和停止十分快速，几乎跟创建和终止原生应用一致；另外，容器自身对系统资源的额外需求也十分有限，远远低于传统虚拟机。很多时候，甚至直接把容器当作应用本身也没有任何问题。

## 为什么要使用 Docker

### Docker 容器虚拟化的好处

在云时代，开发者创建的应用必须要能很方便地在网络上传播，也就是应用必须脱离底层物理硬件的限制；同时必须是**任何时间任何地点**可获取的。因此，开发者们需要一种新型的创建分布式应用程序的方式，快速分发和部署，而这正式 Docker 所能够提供的最大优势。

举个简单的例子，假设用户试图基于最常见的`LAMP`，即 Linux + Apache + MySQL + PHP 组合来构建网络。按照传统的做法，首先需要安装 Apache、MySQL 和 PHP 以及它们各自运行所依赖的环境，之后分别对它们进行配置，包括创建合适的用户、配置参数等。经过大量的操作后，还需要进行功能测试，看是否工作正常，如果不正常，则进行调试追踪，意味着更多的时间代价和不可控的风险。可以想象，如果应用数目变多，事情会变得更加难处理。**更为可怕的是**，一旦需要服务器迁移，例如从亚马逊云迁移到其他云，往往需要对每个应用都进行重新部署和调试，这些琐碎而无趣的体力活，极大地降低了用户的工作效率。**究其根源**，是这些应用直接运行在底层操作系统上，无法保证同一份应用在不同的环境中行为一致。

**为了解决这个问题**，Docker 提供了一种更为聪明的方式，通过容器来打包应用、解耦应用和运行平台。这意味着迁移的时候，只需要在新服务器上启动需要的容器就可以了，无论新旧服务器是否是同一类型的平台，这无疑将帮助我们节约大量的宝贵时间，并降低部署过程出现问题的风险。

### Docker 在开发和运维中的优势

对于开发和运维人员来说，最梦寐以求的效果可能就是一次创建或配置，之后可以在任意地方、任意时间让应用正常运行，而 Docker 恰恰是可以实现这一终极目标的利器。具体来说，在开发和运维过程中，Docker 具有如下几个方面的优势：

- **更快速的交付和部署**，使用 Docker，开发人员可以使用镜像来构建一套标准的开发环境。开发完成之后，测试和运维人员可以直接使用完全相同的环境来部署代码。只要是开发测试过的代码，就可以确保在生产环境中无缝运行。DocKer 可以快速创建和删除容器，实现快速迭代，节约开发、测试、部署的大量时间。并且，整个过程全程可见，使团队更容易理解应用的创建和工作过程。
- **更高效的资源利用**，运行 Docker 容器不需要额外的虚拟化管理程序，VMM，Virtual Machine Manager，以及 Hypervisor 的支持，Docker 是内核级的虚拟化，可以实现更高的性能，同时对资源的额外需求很低。与传统的虚拟机方式相比，Docker 的性能要提高 1~2 个数量级。
- **更轻松的迁移和扩展**，Docker 容器几乎可以在任意的平台上运行，包括物理机、虚拟机、公有云、私有云、个人电脑、服务器等，同时支持主流的操作系统发行版本。这种兼容性让用户在不同平台之间轻松地迁移应用。
- **更简单的更新管理**，使用 Dockerfile，只需要小小的配置修改，就可以替代以往大量的更新工作，所有修改都以增量的方式被分发和更新，从而实现自动化并且高效的容器管理。

### Docker 与虚拟机比较

**Docker 容器技术与传统虚拟机技术的比较**：

<img src="/imgs/3ff552af22381e4fa65df8920651d10c.png">

## 核心概念

Docker 大部分的操作都围绕着它的三大核心概念：`镜像`、`容器`和`仓库`。因此，准确把握这三大核心概念对于掌握 Docker 技术尤为重要。

### Image 镜像

Docker 镜像类似于虚拟机镜像，可以将它理解为一个只读的模板。例如，一个镜像可以包含一个基本的操作系统环境，里面仅安装了 Apache 应用程序，可以把它称为一个 Apache 镜像。`镜像是创建 Docker 容器的基础`，通过版本管理和增量的文件系统，Docker 提供了一套十分简单的机制来创建和更新现有的镜像，用户甚至可以从网上下载一个已经做好的应用镜像，并直接使用。

### Container 容器

Docker 容器类似于一个轻量级的沙箱，Docker 利用容器来运行和隔离应用。容器是从镜像创建的应用运行实例。它可以启动、开始、停止、删除，而这些容器都是彼此相互隔离，互不可见的。**可以把容器看作一个简易的 Linux 系统环境**，包括 root 用户权限、进程空间、用户空间和网络空间等，以及运行在其中的应用程序打包而成的盒子。`镜像自身是只读的，容器从镜像启动的时候，会在镜像的最上层创建一个可写层`。 

### Repository 仓库

Docker 仓库类似于代码仓库，是 Docker 集中存放镜像文件的场所。根据所存储的镜像公开分享是否，Docker 仓库可以分为公开仓库和私有仓库两种形式。目前，最大的公开仓库是官方提供的 Docker Hub，其中存放着数量庞大的镜像供用户下载。国内不少云服务器提供商，如腾讯云、阿里云等也提供了仓库的本地源，可以提供稳定的国内访问。

如果，用户不希望公开分享自己的镜像文件，Docker 也支持用户在本地网络内创建一个只能自己访问的私有仓库。当用户创建了自己的镜像之后就可以使用 push 命令将它上传到指定的共有或私有仓库，这样用户下次在另外一台机器上使用该镜像时，只需要将其从仓库上 pull 下来就可以了。



# 安装和配置 Docker 服务

Docker 引擎是使用 Docker 容器的核心组件，可以在主流的操作系统和云平台上使用，包括 Linux 操作系统、MacOS 和 Windows 操作系统，以及 IBM、亚马逊、微软等知名云平台。

## Ubuntu 环境下安装 Docker

参照`Docker 安装和配置.md`文档。

## CentOs 环境下安装 Docker

...

## MacOS 环境下安装 Docker

...

## Windows 环境下安装 Docker

...

## 通过脚本安装 Docker



## 小结，配置 Docker 服务

为了避免每次使用 Docker 命令时都需要切换到特权身份，可以将当前用户加入安装中自动创建的 docker 用户组，代码如下：

```shell
# 1.创建一个`docker`组.
$ sudo groupadd docker
# 2.添加当前用户到`docker`组. 用户更新组信息, 退出并重新登陆后即可生效.
$ sudo usermod -aG docker USER_NAME
# 3.登出, 重新登录`shell`.
# 4.验证`docker`命令是否可以运行.
$ docker run hello-world
```

----

Docker 服务启动时实际上是调用了`dockerd`命令，支持多种启动参数。因此，用户可以直接通过执行 dockerd 命令来启动 Docker 服务，如下面的命令启动 Docker 服务，开启 Debug 模式，并监听在本地的 2376 端口：

```shell
$ dockerd -D -H tcp://127.0.0.1:2376
```

这些选项可以写入`/etc/docker/`路径下的`daemon.json`文件中，由 dockerd 服务启动时读取：

```json
{
    "debug": true,
    "host": ["tcp://127.0.0.1:2376"]
}
```

----

当然，操作系统也对 Docker 服务进行了封装，以使用 Upstart 来管理启动服务的 Ubuntu 系统为例，默认配置文件为`/etc/default/docker`，可以通过修改其中的 DOCKER_OPTS 来修改服务启动的参数，例如，让 Docker 服务开启网络 2375 端口的监听：

```shell
# 1.设置`/etc/default/docker`文件.
DOCKER_OPTS="$DOCKER_OPTS -H tcp://0.0.0.0:2375 -H unix:///var/run/docker.sock”
# 2.对于`Ubuntu`系统, 修改之后, 通过`service`命令来重启服务.
$ sudo service docker restart
# 2.对于`Centos、RedHat`系统, 服务是通过`systemd`来管理, 配置文件路径为:`/etc/systemd/system/docker.serviced/docker.conf`. 更新配置后需要通过`systemctl`命令来管理服务.
$ sudo systemctl daemon-reload
$ sudo systemctl start docker.service
```

----

如果服务工作不正常，可以通过查看 Docker 服务的日志信息来确定问题，例如，在 RedHat 系统上的日志文件可能为`/var/log/messages`，在 Ubuntu 或 CentOS 系统上可以执行`journalctl -u docker.service `进行查看。每次重启 Docker 服务后，可以通过查看 Docker 信息，`docker info`，以确保服务已经正常运行。 

```shell
ubuntu@VM-0-9-ubuntu:/etc/docker$ journalctl -u docker.service 
-- Logs begin at Sat 2020-05-23 10:58:12 CST, end at Sat 2020-05-23 21:24:08 CST. --
May 23 21:23:56 VM-0-9-ubuntu systemd[1]: Stopping Docker Application Container Engine...
May 23 21:23:56 VM-0-9-ubuntu dockerd[1305]: time="2020-05-23T21:23:56.007229131+08:00" level=info msg=" ...'terminated'"
May 23 21:23:56 VM-0-9-ubuntu dockerd[1305]: time="2020-05-23T21:23:56.038412984+08:00" level=info msg="stopping stream ...
May 23 21:23:56 VM-0-9-ubuntu dockerd[1305]: time="2020-05-23T21:23:56.075912280+08:00" level=info msg="Daemon complete"
May 23 21:23:56 VM-0-9-ubuntu dockerd[1305]: time="2020-05-23T21:23:56.075983680+08:00" level=info msg="stop ...
May 23 21:23:56 VM-0-9-ubuntu systemd[1]: Stopped Docker Application Container Engine.
May 23 21:23:56 VM-0-9-ubuntu systemd[1]: Starting Docker Application Container Engine...
May 23 21:23:58 VM-0-9-ubuntu dockerd[26413]: time="2020-05-23T21:23:58.420390094+08:00" level=info msg="Starting up"
May 23 21:23:58 VM-0-9-ubuntu dockerd[26413]: time="2020-05-23T21:23:58.423500047+08:00" level=info msg="parsed: \"unix\""
......
```

```shell
ubuntu@VM-0-9-ubuntu:/etc/docker$ sudo docker info
Client:
 Debug Mode: false

Server:
 Containers: 0
  Running: 0
  Paused: 0
  Stopped: 0
 Images: 9
 Server Version: 19.03.8
 Storage Driver: overlay2
  Backing Filesystem: <unknown>
  Supports d_type: true
  Native Overlay Diff: true
 Logging Driver: json-file
 Cgroup Driver: cgroupfs
 Plugins:
  Volume: local
  Network: bridge host ipvlan macvlan null overlay
  Log: awslogs fluentd gcplogs gelf journald json-file local logentries splunk syslog
 Swarm: inactive
 Runtimes: runc
 Default Runtime: runc
 Init Binary: docker-init
 containerd version: 7ad184331fa3e55e52b890ea95e65ba581ae3429
 runc version: dc9208a3303feef5b3839f4323d9beb36df0a9dd
 init version: fec3683
 Security Options:
  apparmor
  seccomp
   Profile: default
 Kernel Version: 4.4.0-130-generic
 Operating System: Ubuntu 16.04.1 LTS
 OSType: linux
 Architecture: x86_64
 CPUs: 2
 Total Memory: 3.734GiB
 Name: VM-0-9-ubuntu
 ID: 3EL7:IA5R:UIAZ:W4Z2:422D:SV5R:XQ6K:RMMQ:O3Z7:LSPR:VRKU:7ESR
 Docker Root Dir: /var/lib/docker
 Debug Mode: false
 Registry: https://index.docker.io/v1/
 Labels:
 Experimental: false
 Insecure Registries:
  127.0.0.0/8
 Live Restore Enabled: false

WARNING: No swap limit support
```



# 使用 Docker 镜像

`镜像`是 Docker 三大核心概念中最重要的，Docker 运行容器的前需要本地存在对应的镜像，如果镜像不存在，Docker 会尝试先从默认镜像仓库下载，用户也可以通过配置，使用自定义的镜像仓库。

## 获取镜像

镜像是运行容器的前提，官方的`Docker Hub`网站已经提供了数十万个镜像供大家开放下载。可以使用`docker [image] pull`命令直接从 Docker Hub 镜像源来下载镜像，该命令的格式如下：

```shell
# 其中, `NAME`是镜像仓库名称, 用来区分镜像, `TAG`是镜像的标签, 往往用来表示版本信息.
# 通常情况下, 描述一个镜像需要包括`名称+标签`信息.
$ docker [image] pull NAME[:TAG]
```

```shell
# 例如, 获取一个`Ubuntu 18.04`系统的基础镜像可以使用如下的命令:
# 对于Docker镜像来说, 如果不显式指定`TAG`, 则默认会选择`latest`标签, 这会下载仓库中最新版本的镜像.
$ docker pull ubuntu:18.04
$ docker pull ubuntu
```

---

严格来讲，镜像的仓库名称中还应该添加仓库的地址作为前缀，只是默认使用的是官方的 Docker Hub 服务，该前缀可以忽略。例如，`docker pull ubuntu:18.04`的命令相当于`docker pull registry.hub.docker.com/ubuntu:18.04`命令，即从默认的注册器中的 Ubuntu 仓库来下载标记为 18.04 的镜像。如果从非官方的仓库下载，则需要在仓库名称前指定完整的仓库地址，例如从网易的镜像源来下载 ubuntu:18.04 镜像，可以使用如下命令，此时下载的镜像名称为 **hub.c.163.com/public/ubuntu:18.04** ：`docker pull hub.c.163.com/public/ubuntu:18.04 `。

---

`docker pull`子命令支持的选项主要包括：`-a`，--all-tags=true|false，表示是否获取仓库的所有镜像，默认为否；`--disable-content-trust`，取消镜像的内容校验，默认值为 true 。

---

下载镜像到本地后，即可随时使用该镜像了，例如，利用该镜像创建一个容器，在其中运行 bash 应用，执行打印 Hello World 命令：

```shell
test@VM-0-9-ubuntu:~$ docker run -it ubuntu:18.04 bash
root@a7ba948af80d:/# 
root@a7ba948af80d:/# echo "Hello World"
Hello World
root@a7ba948af80d:/# exit
exit
```

## 查看镜像信息

### 列出镜像

使用`docker images`或`docker image ls`命令可以列出本地主机已有镜像的基本信息，例如：

```shell
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
test/ubuntu         dev                 6bb86ec47445        4 weeks ago         125MB
test/ununt          16.04               6bb86ec47445        4 weeks ago         125MB
ubuntu              16.04               005d2078bdfa        4 weeks ago         125MB
```

```properties
# 在列出的信息中, 可以看到几个字段信息.
REPOSITORY : 来自于哪个仓库, 比如`ubuntu`表示`ubuntu`系列的基础镜像.
TAG : 镜像的标签信息, 比如`18.04`、`lastest`表示不同的版本信息. 标签只是标记, 并不能标识镜像的内容.
IMAGE ID : 镜像的ID信息, 唯一标识镜像, 如果两个镜像的`ID`相同, 说明它们实际上指向了同一个镜像, 只是具有不同标签标记而已.
CREATED : 创建时间, 说明镜像最后的更新时间.
SIZE : 镜像大小.
```

其中镜像 ID 信息十分重要，它唯一标识了镜像。在使用镜像 ID 的时候，一般可以使用该 ID 的前若干个字符组成的可区分串来替代完整的 ID 。TAG 信息用于标记来自同一个仓库的不同镜像，例如 ubuntu 仓库中有多个镜像，通过 TAG 信息来区分发行版本，如 18.04、18.10 等。镜像大小信息只表示了该镜像的逻辑体积大小，实际上由相同的镜像层本地只会存储一份，物理上占用的内存空间会小于各镜像逻辑体积之和。

---

`docker images`子命令主要支持如下选项，用户可以自由组合：`-a`，--all=true|false，列出所有，包括临时文件镜像文件，默认为 false；`--digests`，可选值为 true 或 false，用于列出镜像的数字摘要值，默认值为 false；`-f`，--filter=[]，用于过滤列出的镜像，如 dangling=true，则只显示没有被使用的镜像，也可指定带有特定标注的镜像等；`--format`，--format="TEMPLATE"，控制输出格式，如 .ID 代表 ID 信息，.Repository 代表仓库信息等；`--no-trunc`，可选值为 true 或 false，对输出结果中太长的部分是否进行截断，如镜像 ID 信息，默认是 true；`-q`，--quite=true|false，仅输出 ID 信息，默认值 false。

```shell
test@VM-0-9-ubuntu:~$ docker images --help
Usage:  docker images [OPTIONS] [REPOSITORY[:TAG]]
List images
Options:
  -a, --all             Show all images (default hides intermediate images)
      --digests         Show digests
  -f, --filter filter   Filter output based on conditions provided
      --format string   Pretty-print images using a Go template
      --no-trunc        Don't truncate output
  -q, --quiet           Only show numeric IDs
```

### 添加镜像标签

为了方便在后续工作中使用特定的镜像，还可以使用`docker tag`命令来为本地镜像任意添加新的标签。例如：

```shell
test@VM-0-9-ubuntu:~$ docker tag ubuntu:18.04 myubuntu:18.04
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
test/ubuntu         dev                 6bb86ec47445        4 weeks ago         125MB
test/ununt          16.04               6bb86ec47445        4 weeks ago         125MB
ubuntu              16.04               005d2078bdfa        4 weeks ago         125MB
myubuntu            18.04               c3c304cb4f22        4 weeks ago         64.2MB
ubuntu              18.04               c3c304cb4f22        4 weeks ago         64.2MB
```

比较后可以发现，`ubuntu:18.04`和`myubuntu:18.04`的镜像 ID 是一致的，说明它们实际上是指向了同一个镜像文件，只是别名不同而已。docker tag 命令添加的标签实际上起到了类似链接的作用。

### 查看镜像的详细信息

使用`docker [image] inspect`命令可以获取该镜像的详细信息，包括制作者、适应架构、各层的数字摘要等：

```shell
# 使用方法:
$ docker image inspect ubuntu:18.04
$ docker inspect ubuntu:18.04
# 实例:
test@VM-0-9-ubuntu:~$ docker inspect ubuntu:18.04      
[
    {
        "Id": "sha256:c3c304cb4f22ceb8a6fcc29a0cd6d3e4383ba9eb9b5fb552f87de7c0ba99edac",
        "RepoTags": [
            "myubuntu:18.04",
            "ubuntu:18.04"
        ],
        "RepoDigests": [
            "ubuntu@sha256:3235326357dfb65f1781dbc4df3b834546d8bf914e82cce58e6e6b676e23ce8f"
        ],
        "Parent": "",
        "Comment": "",
        ...
```

上面代码返回的是一个 JSON 格式的消息，如果我们只要其中一项内容时，可以使用`-f`来指定，例如，获取镜像的 Architecture ：

```shell
# 使用方法:
$ docker [image] inspect -f {{".Architecture"}} ubuntu:18.04
# 实例:
test@VM-0-9-ubuntu:~$ docker inspect -f {{".Architecture"}} ubuntu:18.04
amd64
```

### 查看镜像历史

既然镜像文件由多个层组成，那么怎么知道各个层的内容具体是什么，这时候可以使用 history 子命令，该命令将列出各层的创建信息。例如，查看 unbuntu:18.04 镜像的创建过程，可以使用如下命令：

```shell
# 使用方法:
$ docker history ubuntu:18.04
# 实例:
# 注意, 过长的命令被自动截断了, 可以使用前面提到的`--no-trunc`选项来输出完整的命令.
test@VM-0-9-ubuntu:~$ docker history ubuntu:18.04
IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
c3c304cb4f22        4 weeks ago         /bin/sh -c #(nop)  CMD ["/bin/bash"]            0B                  
<missing>           4 weeks ago         /bin/sh -c mkdir -p /run/systemd && echo 'do…   7B                  
<missing>           4 weeks ago         /bin/sh -c set -xe   && echo '#!/bin/sh' > /…   745B                
<missing>           4 weeks ago         /bin/sh -c [ -z "$(apt-get indextargets)" ]     987kB               
<missing>           4 weeks ago         /bin/sh -c #(nop) ADD file:c3e6bb316dfa6b81d…   63.2MB              
```

## 搜索镜像

可以使用`docker search`命令可以搜索 Docker Hub 官方仓库中的镜像，语法为`docker search [option] keyword`。支持的命令选项：

- **-f | --filter filter**：过滤输出内容；
- **--format string**：格式化输出内容；
- **--limit int**：限制输出结果个数，默认为 25 个；
- **--no-trunc**：不截断输出结果。

例如，搜索官方提供的带 nginx 关键字的镜像，如下所示：

```shell
test@VM-0-9-ubuntu:~$ docker search --filter=is-official=true nginx
NAME                DESCRIPTION                STARS               OFFICIAL            AUTOMATED
nginx               Official build of Nginx.   13224               [OK]     
```

再比如，搜索所有收藏数超过 4，且关键词包括 tensorflow 的镜像：

```shell
test@VM-0-9-ubuntu:~$ docker search --filter=stars=4 tensorflow
NAME                             DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
tensorflow/tensorflow            Official Docker images for the machine learn…   1689                                    
jupyter/tensorflow-notebook      Jupyter Notebook Scientific Python Stack w/ …   217                                     
tensorflow/serving               Official images for TensorFlow Serving (http…   88                                      
xblaster/tensorflow-jupyter      Dockerized Jupyter with tensorflow              54                 [OK]
rocm/tensorflow                  Tensorflow with ROCm backend support            44                                      
floydhub/tensorflow              tensorflow                                      24                 [OK]
bitnami/tensorflow-serving       Bitnami Docker Image for TensorFlow Serving     14                 [OK]
opensciencegrid/tensorflow-gpu   TensorFlow GPU set up for OSG                   12                                      
ibmcom/tensorflow-ppc64le        Community supported ppc64le docker images fo…   5                                       
tensorflow/tf_grpc_test_server   Testing server for GRPC-based distributed ru…   4                                       
```

## 删除和清理镜像

### 使用标签删除镜像

使用`docker rmi`或`docker image rm`命令可以删除镜像，命令格式为：

```shell
# 其中, `IMAGE`可以为标签或ID.
$ docker rmi IMAGE [IMAGE ...]

# 支持的选项, 包括:
-f, --force      强制删除镜像, 即使有容器需要依赖它.
	--no-prune   不要清理未带标签的父镜像.
	
# 例如, 如果要删除`myubuntu:18.04`镜像, 可以使用如下命令:
# `docker rmi`命令只是删除了该镜像多个标签中的指定标签而已, 并不影响镜像文件. 因此, 下面操作相当于只是删除了指定镜像的一个标签副本而已.
# 但是当镜像只剩一个标签的时候就要小心了, 此时再使用`docker rmi`命令会彻底删除镜像.
test@VM-0-9-ubuntu:~$ docker rmi myubuntu:18.04
Untagged: myubuntu:18.04
```

### 使用 ID 来删除镜像

当使用`docker rmi`命令，并且后面跟上镜像的 ID 时，会先尝试删除所有指向该镜像的标签，然后删除该镜像文件本身。**注意**，当有该镜像创建的容器时，镜像文件默认是无法被删除的，此时如果要强行删除镜像，可以使用`-f`参数，形如 **docker rmi -f ubuntu:18.04** 。

不过，通常并不推荐使用`-f`参数来强制删除一个存在的容器依赖的镜像，正确的做法是，先删除依赖该镜像的所有容器，再来删除镜像：

```shell
# 1.首先删除容器 a21c08...
$ docker rm a21c08...
# 2.然后使用`ID`来删除镜像, 此时会正常打印出删除的各层信息.
$ docker rmi 8f1bd21...
```

### 清理镜像

使用 Docker 一段时间后，系统中可能会遗留一些临时的镜像文件，以及一些没有被使用的镜像，可以通过`docker image prune`命令来进行清理，支持的选项包括：

```shell
Options:
  -a, --all             删除所有无用的镜像, 不光是临时镜像.
      --filter filter   只清理符合给定过滤器的镜像(e.g. 'until=<timestamp>')
  -f, --force           强制删除镜像, 而不进行提示确认.
```

例如，如下命令会自动清理临时的遗留镜像文件层，最后会提示释放的存储空间：

```shell
test@VM-0-9-ubuntu:~$ docker image prune -f
Total reclaimed space: 0B
```

## 创建镜像

创建镜像的方法主要有三种：**基于已有镜像的容器创建**、**基于本地模板导入**和**基于 Dockerfile 创建**。

### 基于已有容器创建

该方法主要是使用`docker [container] commit`命令，其命令格式：

```shell
# Create a new image from a container's changes.
$ docker [container] commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]

Options:
  -a, --author string    "", 作者信息(e.g., "Albert丶XN <albertxn@126.com>")
  -c, --change list      [], 提交的时候执行`Dockerfile`指令, 包括`CMD|ENTRYPOINT|ENV|EXPOSE|LABEL|ONBUILD|USER|VOLUME`等
  -m, --message string   "", 提交信息
  -p, --pause            true, 提交时暂停容器运行.
```

----

下面将演示如何使用该命令创建一个新镜像：

```shell
# 1.首先, 启动一个镜像, 并在其中进行修改操作. 例如, 创建一个`test`文件, 之后退出, 代码如下:
test@VM-0-9-ubuntu:~$ docker run -it ubuntu:18.04 /bin/bash
root@d902843014e7:/# touch test
root@d902843014e7:/# exit
exit
# 2.此时, 该容器与原镜像相比, 已经发生了改变, 可以使用`docker [container] commit`命令来提交为一个新的镜像.
#   提交时可以使用`ID`或名称来指定容器:
test@VM-0-9-ubuntu:~$ docker container commit -m "Added a new file" -a "Albert丶XN" d902843014e7 test:0.1
sha256:5414532bfd461753f507a1e56e8e4d397feb88cec26403cdb3cbb54e94287ea3
# 3.顺利的话, 会返回新创建镜像的ID信息, 如上所示.
#   此时, 查看本地镜像列表, 会发现新创建的镜像已经存在了:
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED              SIZE
test                0.1                 5414532bfd46        About a minute ago   64.2MB
test/ubuntu         dev                 6bb86ec47445        4 weeks ago          125MB
test/ununt          16.04               6bb86ec47445        4 weeks ago          125MB
ubuntu              16.04               005d2078bdfa        4 weeks ago          125MB
ubuntu              18.04               c3c304cb4f22        4 weeks ago          64.2MB
```

### 基于本地模板导入

用户也可以直接从一个操作系统模板文件导入一个镜像，主要使用`docker import`命令，其命令格式：

```shell
# Import the contents from a tarball to create a filesystem image.
$ docker import [OPTIONS] file|URL|- [REPOSITORY[:TAG]]

Options:
  -c, --change list      Apply Dockerfile instruction to the created image
  -m, --message string   Set commit message for imported image
```

----

要直接导入一个镜像，可以使用`OpenVZ`提供的模板来创建，或者用其他已导出的镜像模板来创建。OpenVZ 模板的下载地址为：[URL](https://wiki.openvz.org/Download/template/precreated) ，例如，下载了 [ubuntu-16.04](http://download.openvz.org/template/precreated/ubuntu-16.04-x86_64.tar.gz) 的模板压缩包，之后使用以下命令导入即可：

```shell
# 1.下载`ubuntu-16.04`模板压缩包.
wget http://download.openvz.org/template/precreated/ubuntu-16.04-x86_64.tar.gz
# 2.下载完模板压缩包之后, 使用以下命令导入即可.
test@VM-0-9-ubuntu:~$ cat ./ubuntu-16.04-x86_64.tar.gz | docker import - ubuntu:16.04
sha256:d7a59f1b3c8ea221dcd1fbeb49a3f449878e55a30e599a8ca4a8d396be7ca03c
# 3.然后查看新导入的镜像.
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED              SIZE
ubuntu              16.04               d7a59f1b3c8e        About a minute ago   505MB
test                0.1                 5414532bfd46        18 minutes ago       64.2MB
ubuntu              18.04               c3c304cb4f22        4 weeks ago          64.2MB
```

### 基于 Docerfile 创建

基于`Dockerfile`创建是最常见的一种方式，Dockerfile 是一个文本文件，利用给定的质量描述基于某个父镜像创建新镜像的过程，使用`docker [image] build`命令。下面给出 Dockerfile 的一个简单示例，基于 debian:stretch-slim 镜像安装 Python3 环境，构成一个新的 python:3 镜像：

```shell
# 1.编写一个`Dockerfile`文件.
FROM debian:stretch-slim
LABEL version="1.0" maintainer="docker user <docker_user@github>"
RUN apt-get update && \
	apt-get install -y python3 && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*
# 2.使用`docker [image] build`命令创建镜像(在`Dockerfile`所在路径下执行):
#   建议新建一个文件夹.
test@VM-0-9-ubuntu:~/Dockrfile$ docker image build -t python:3 .
Sending build context to Docker daemon  2.048kB
Step 1/3 : FROM debian:stretch-slim
stretch-slim: Pulling from library/debian
e62d08fa1eb1: Pull complete 
Digest: sha256:b385ea429b383b690c955043d79050d1cb76346fbca67e3ed3649d5019dd6749
Status: Downloaded newer image for debian:stretch-slim
 ---> fa41698012c7
Step 2/3 : LABEL version="1.0" maintainer="docker user <docker_user@github>"
 ---> Running in c95cd139ce9e
Removing intermediate container c95cd139ce9e
 ---> 74611010b895
Step 3/3 : RUN apt-get update && apt-get install -y python3 && apt-get clean && rm -rf /var/lib/apt/lists/*
 ---> Running in 3b2dc94cebb8
Ign:1 http://deb.debian.org/debian stretch InRelease
Get:2 http://security.debian.org/debian-security stretch/updates InRelease [94.3 kB]
...
 ---> 97b0518f8505
Successfully built 97b0518f8505
Successfully tagged python:3
```

## 存出和载入镜像

用户可以使用`docker [image] save`和`docker [image] load`命令来存出和载入镜像。

### 存出镜像

如果要导出镜像到本地文件，可以使用`docker [image] save` 命令：

```shell
# Save one or more images to a tar archive (streamed to STDOUT by default).
Usage:  docker image save [OPTIONS] IMAGE [IMAGE...]

Options:
  -o, --output string   导出镜像到指定的位置.
```

例如，导出本地的 ubuntu:18.04 镜像文文件 ubuntu_18.04.tar，如下所示：

```shell
# 通过`docker save`存出的镜像可以分享给他人.
$ docker save -o /home/test/ubuntu_18.04.tar ubuntu:18.04
```

### 载入镜像

可以使用`docker [image] load`将导出的 tar 文件再导入到本地镜像库，命令格式：

```shell
# Load an image from a tar archive or STDIN.
Usage:  docker image load [OPTIONS]

Options:
  -i, --input string   从指定文件中读入镜像内容.
  -q, --quiet          Suppress the load output
```

例如，从文件 ubuntu_18.04.tar 导入镜像到本地镜像列表，如下所示：

```shell
# 使用方法:
$ docker load -i /home/test/ubuntu_18.04.tar
$ docker load < /home/test/ubuntu_18.04.tar
# 示例:
test@VM-0-9-ubuntu:~$ docker load -i /home/test/ubuntu_18.04.tar
Loaded image: ubuntu:18.04
```

## 上传镜像

可以使用`docker [image] push`命令来上传镜像到仓库，默认上到的 Docker Hub 官方仓库，需要登录，命令格式为：

```shell
# Push an image or a repository to a registry。
Usage:  docker image push [OPTIONS] NAME[:TAG]

Options:
      --disable-content-trust   Skip image signing (default true)
```

例如，用户上传本地的 **ubuntu:18.04** 镜像，可以先添加新的标签，然后使用 docker push 命令上传镜像：

```shelL
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
python              3                   97b0518f8505        4 minutes ago       95.2MB
ubuntu              16.04               d7a59f1b3c8e        33 minutes ago      505MB
debian              stretch-slim        fa41698012c7        9 days ago          55.3MB
ubuntu              18.04               c3c304cb4f22        4 weeks ago         64.2MB

# 1.登录`Docker`仓库, 登录成功后, 登录信息会被记录到本地`~/.docker`目录下.
test@VM-0-9-ubuntu:~$ docker login 
Login with your Docker ID to push and pull images from Docker Hub. If you don't have a Docker ID, head over to https://hub.docker.com to create one.
Username: -------
Password: -------
WARNING! Your password will be stored unencrypted in /home/test/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store
Login Succeeded
# 2.先添加新的标签, 修改为:`USER/REPOSITORY:TAG`.
test@VM-0-9-ubuntu:~$ docker tag ubuntu:18.04 albertxn97/ubuntu:18.04
# 3.使用`docker [image] push`命令上传镜像.
#   第一次上传时, 会提示输入登录信息或进行注册, 之后登录信息会被记录到本地``目录下》
test@VM-0-9-ubuntu:~$ docker push albertxn97/ubuntu:18.04
The push refers to repository [docker.io/albertxn97/ubuntu]
28ba7458d04b: Mounted from library/mongo 
838a37a24627: Mounted from library/mongo 
a6ebef4a95c3: Pushed 
b7f7d2967507: Pushed 
18.04: digest: sha256:b58746c8a89938b8c9f5b77de3b8cf1fe78210c696ab03a1442e235eea65d84f size: 1152
# 4.登出`Docker`仓库.
test@VM-0-9-ubuntu:~$ docker logout
Removing login credentials for https://index.docker.io/v1/
```



# 操作 Docker 容器

容器是 Docker 的另一个核心概念，简单来说，**容器是镜像的一个运行示例**。所不同的是，镜像是静态的只读文件，而容器带有运行时需要的可写文件层，同时，容器中的应用进程处于运行状态。如果认为虚拟机是模拟运行的一套操作系统，包括内核、应用运行态环境和其他系统环境，以及跑在上面的应用。**那么 Docker 容器就是独立运行的一个或一组应用，以及它们必需的运行环境**。 

因此，总的来说，容器是直接提供应用服务的组件，也是 Docker 整个技术栈中最为核心的概念。围绕容器，Docker 提供了十分丰富的操作命令，允许用户高效地管理容器的整个生命周期。

## 创建容器

### 新建容器后启动容器

#### 新建容器

可以使用`docker [container] create`命令新建一个容器，例如：

```shell
# 使用`docker [container] create`命令新建的容器处于停止状态, 可以使用`docker [container] start`命令来启动它.
test@VM-0-9-ubuntu:~$ docker create -it ubuntu:18.04
a8c9e1c541f8cf26bb04b0176402997a9fc4bb3ba2f3b42493c16e03d820b303
test@VM-0-9-ubuntu:~$ docker ps -a
CONTAINER ID    IMAGE          COMMAND       CREATED         STATUS       PORTS     NAMES
a8c9e1c541f8    ubuntu:18.04   "/bin/bash"   36 seconds ago  Created                blissful_neumann
test@VM-0-9-ubuntu:~$ docker container start a8c9e1c541f8
a8c9e1c541f8
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE          COMMAND       CREATED         STATUS       PORTS      NAMES
a8c9e1c541f8    ubuntu:18.04    "/bin/bash"  6 minutes ago   Up 3 seconds            blissful_neumann
```

```properties
# `docker [container] create`命令与容器运行模式相关的选项.
# `docker [container] create`命令与容器环境和配置爱相关的选项.
# `docker [container] create`命令与容器资源限制和安全保护相关的选项.
# --log-driver string : "json-file", 指定容器的日志驱动类型, 可以为`json-file`, `syslog`, `journald`, `gelf`, `fluentd`, `awslogs`, `splunk`, `etwlogs`, `gcplogs`或`none`.
Usage:  docker container create [OPTIONS] IMAGE [COMMAND] [ARG...]

Create a new container
Options:
      --add-host list                  : [], 在容器内添加一个主机名到`IP`地址的映射关系(通过`/etc/hosts`文件)(host:ip).
  -a, --attach list                    : [], 是否绑定到标准输入、输出和错误.
      --blkio-weight uint16            : 0, 容器读写块设备的`I/O`性能权重, 10 ~ 1000
      --blkio-weight-device list       : [], [DEVICE_NAME:WEIGHT], 指定各个块设备的`I/O`性能权重.
      --cap-add list                   : [], 增加容器的`Linux`指定安全能力.
      --cap-drop list                  : [], 移除容器的`Linux`指定安全能力.
      --cgroup-parent string           : "", 容器`cgroups`限制的创建路径.
      --cidfile string                 : "", 指定容器的进行`ID`号写到文件.
      --cpu-period int                 : int, 限制容器在`CFS`调度器下的`CPU`占用时间片段.
      --cpu-quota int                  : int, 限制容器在`CFS`调度下的`CPU`配额.
      --cpu-rt-period int              : Limit CPU real-time period in microseconds
      --cpu-rt-runtime int             : Limit CPU real-time runtime in microseconds
  -c, --cpu-shares int                 : CPU shares (relative weight)
      --cpus decimal                   : Number of CPUs
      --cpuset-cpus string             : "", 限制容器能使用哪些`CPU`核心(0-3, 0,1).
      --cpuset-mems string             : "", `NUMA`架构下使用哪些核心的内存(0-3, 0,1).
      --device list                    : [], 映射物理机上的设备到容器内.
      --device-cgroup-rule list        : Add a rule to the cgroup allowed devices list
      --device-read-bps list           : [], 挂载设备的读吞吐率(以`bps`为单位)限制.
      --device-read-iops list          : [], 挂载设备的读速率(以每秒`i/o`次数为单位)限制.
      --device-write-bps list          : [], 挂载设备的写吞吐率(以`bps`为单位)限制.
      --device-write-iops list         : [], 挂载设备的写速率(以每秒`i/o`次数为单位)限制.
      --disable-content-trust          : Skip image verification (default true)
      --dns list                       : [], 自定义的`DNS`服务器.
      --dns-option list                : [], 自定义的`DNS`选项.
      --dns-search list                : [], `DNS`搜索域.
      --domainname string              : Container NIS domain name
      --entrypoint string              : "", 镜像存在入口命令时, 覆盖为新的命令.
  -e, --env list                       : [], 指定容器内环境变量.
      --env-file list                  : [], 从文件中读取环境变量到容器内.
      --expose list                    : Expose a port or a range of ports
      --gpus gpu-request               : GPU devices to add to the container ('all' to pass all GPUs)
      --group-add list                 : [], 运行容器的用户组.
      --health-cmd string              : "", 指定要执行的健康检查的命令.
      --health-interval duration       : 0s, 指定健康检查命令执行的时间间隔, 例如`5s`(ms|s|m|h).
      --health-retries int             : int, 指定失败多少次, 容器会被标记为不健康的.
      --health-start-period duration   : 0s, 指定在多少秒后才正式开始计算失败次数, 例如`5s`(ms|s|m|h).
      --health-timeout duration        : 0s, 健康检查的超时时间, 超时会被认为是检查失败, 例如`5s`(ms|s|m|h).
      --help                           : 打印帮助信息.
  -h, --hostname string                : "", 指定容器内的主机名.
      --init                           : Run an init inside the container that forwards signals and reaps processes
  -i, --interactive                    : 保持标准输入打开.
      --ip string                      : "", 指定容器的`IPv4`地址(e.g., 172.30.100.104).
      --ip6 string                     : "", 指定容器的`IPv6`地址(e.g., 2001:db8::33).
      --ipc string                     : "", 容器`IPC`命名空间, 可以为其他容器或主机.
      --isolation string               : Container isolation technology
      --kernel-memory bytes            : Kernel memory limit
  -l, --label list                     : [], 以键值对方式指定容器的标签信息.
      --label-file list                : [], 从文件最后读取标签信息.
      --link list                      : [<name or id>:alias], 链接到其他容器.
      --link-local-ip list             : [], 容器的本地链接地址列表.
      --log-driver string              : "json-file", 指定容器的日志驱动类型, 可以为`json-file`, `syslog`, `journald`, `gelf` ...
      --log-opt list                   : [], 传递使用的隔离机制.
      --mac-address string             : "", 指定容器的`Mac`地址(e.g., 92:d0:c6:0a:29:33).
  -m, --memory bytes                   : Memory limit
      --memory-reservation bytes       : Memory soft limit
      --memory-swap bytes              : Swap limit equal to memory plus swap: '-1' to enable unlimited swap
      --memory-swappiness int          : Tune container memory swappiness (0 to 100) (default -1)
      --mount mount                    : Attach a filesystem mount to the container
      --name string                    : "", 指定容器的别名.
      --network network                : "bridge", 指定容器网络模式, 包括`bridge`, `none`, 其他容器内网络, host的网络或某个现有网络.
      --network-alias list             : [], 容器在网络中的别名.
      --no-healthcheck                 : Disable any container-specified HEALTHCHECK
      --oom-kill-disable               : Disable OOM Killer
      --oom-score-adj int              : Tune host's OOM preferences (-1000 to 1000)
      --pid string                     : "", 容器的`PID`命名空间.
      --pids-limit int                 : Tune container pids limit (set -1 for unlimited)
      --privileged                     : Give extended privileges to this container
  -p, --publish list                   : [], 指定如何映射到本地主机端口, 例如`-p 11234-12234:1234-2234`.
  -P, --publish-all                    : 通过`NAT`机制将容器标记暴露的端口自动映射到本地主机的临时端口.
      --read-only                      : Mount the container's root filesystem as read only
      --restart string                 : "no", 容器重启策略, 包括`no`, `no-failure[:max-retry]`, `always`, `unless-stopped`等.
      --rm                             : 容器退出后是否自动删除, 不能跟`-d`同时使用.
      --runtime string                 : Runtime to use for this container
      --security-opt list              : Security Options
      --shm-size bytes                 : Size of /dev/shm
      --stop-signal string             : Signal to stop a container (default "SIGTERM")
      --stop-timeout int               : Timeout (in seconds) to stop a container
      --storage-opt list               : Storage driver options for the container
      --sysctl map                     : Sysctl options (default map[])
      --tmpfs list                     : Mount a tmpfs directory
  -t, --tty                            : 是否分配一个伪终端.
      --ulimit ulimit                  : Ulimit options (default [])
  -u, --user string                    : Username or UID (format: <name|uid>[:<group|gid>])
      --userns string                  : "", 启用`userns-remap`时配置用户命名空间的模式.
      --uts string                     : "", 容器的`UTS`命名空间.
  -v, --volume list                    : [], 挂载主机上的文件卷到容器内.
      --volume-driver string           : "", 挂载文件卷的驱动类型.
      --volumes-from list              : [], 从其他容器挂载卷.
  -w, --workdir string                 : "", 容器内的默认工作目录.
```

#### 启动容器

使用`docker [container] start`命令来启动一个已经创建的容器，例如，启动上一步创建的容器，示例参考`新建容器`部分。

### 新建并启动容器

除了创建容器后通过`start`命令来启动，也可以直接新建并启动容器，所需要的命令主要为`docker [container] run`，等价于先执行`create`命令，在执行`start`命令，例如，下面的命令输出一个 **Hello World**，之后容器自动终止：

```shell
test@VM-0-9-ubuntu:~$ docker run ubuntu:18.04 /bin/echo "Hello World"
Hello World
```

当利用`docker [container] run`来创建并启动容器时，Docker 在后台的标准操作包括：

- 检查本地是否存在指定的镜像，不存在就从共有仓库下载；
- 利用镜像创建一个容器，并启动该容器；
- 分配一个文件系统给容器，并在只读的镜像层外面挂载一层可读写层；
- 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去；
- 从网桥的地址池配置一个 IP 地址给容器；
- 执行用户指定的应用程序；
- 执行完毕后容器被自动终止。

---

下面的命令启动一个 bash 终端，允许用户进行交互：

```shell
test@VM-0-9-ubuntu:~$ docker run -it ubuntu:18.04 /bin/bash
root@180874fc147f:/# 
```

其中，`-t`选项让 Docker 分配一个伪终端并绑定到容器的标准输入上，`-i`，则让容器的标准输入保持打开。更多的命令选项可以通过`man docker-run`命令来查看，用户可以按`Ctrl + D`或输入`exit`命令来退出容器。

---

某些时候，执行`run`命令时候，会因为命令无法正常执行容器会出错直接退出，此时可以查看退出的错误代码：

```properties
# 默认情况下, 常见错误代码包括:
125 : Docker daemon 执行错误, 例如指定了不支持的 Docker 命令参数;
126 : 所指定命令无法执行, 例如权限出错;
127 : 容器内命令无法找到.
```

### 守护态运行

更多的时候，需要让 Docker 容器在后台以守护态形式运行。此时，可以通过添加`-d`参数来实现，例如：

```shell
test@VM-0-9-ubuntu:~$ docker run -d ubuntu:18.04 /bin/bash -c "while true; do echo hello world; sleep 1; done"
ce5c79a86a935b74d9e6155e10729459701b1df324724cc9ab830eae6cd8cd4a
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID     IMAGE          COMMAND                 CREATED         STATUS          PORTS       NAMES
ce5c79a86a93     ubuntu:18.04   "/bin/bash -c 'while…"  4 seconds ago   Up 3 seconds                serene_proskuriakova
a8c9e1c541f8     ubuntu:18.04   "/bin/bash"             23 hours ago    Up 23 hours                 blissful_neumann
```

### 查看容器输出

要获取容器的输出信息，可以通过`docker [container] logs`命令，该命令支持的选项包括：

```shell
# Fetch the logs of a container.
Usage:  docker logs [OPTIONS] CONTAINER

Options:
      --details        打印详细信息.
  -f, --follow         持续保持输出.
      --since string   输出从某个时间开始的日志 (e.g. 2013-01-02T13:23:37) 或 (e.g. 42m for 42 minutes).
      --tail string    输出最近的若干日志 (default "all")
  -t, --timestamps     显示时间戳信息.
      --until string   输出某个时间之前的日志 (e.g. 2013-01-02T13:23:37) 或 (e.g. 42m for 42 minutes).
```

```shell
# 查看某容器的输出, 可以使用如下命令:
test@VM-0-9-ubuntu:~$ docker logs ce5c79a86a93
hello world
hello world
...
```

## 停止容器

### 暂停容器

可以使用`docker [container] pause CONTAINER [CONTAINER ...]`命令来暂停一个容器，例如，启动一个容器，并将其暂停：

```shell
test@VM-0-9-ubuntu:~$ docker run --name mytest -itd ubuntu:18.04 /bin/bash
5b70a3ec5da46ab783935ebf27cf4911325e8a49858b5a183bd56c8e9b95b44a
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID     IMAGE           COMMAND          CREATED           STATUS                    PORTS             NAMES
5b70a3ec5da4     ubuntu:18.04    "/bin/bash"      11 seconds ago    Up 10 seconds                               mytest
test@VM-0-9-ubuntu:~$ docker pause mytest
mytest
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID     IMAGE           COMMAND          CREATED           STATUS                     PORTS            NAMES
5b70a3ec5da4     ubuntu:18.04    "/bin/bash"      11 seconds ago    Up About a minute (Paused)                  mytest
```

处于 **paused** 状态的容器，可以使用`docker [container] unpause CONTAINER [CONTAINER ...]`命令来恢复到运行状态。

### 终止容器

可以使用`docker [container] stop`来终止一个运行中的容器，该命令的格式为`docker [container] stop [-t|--time [=10]] CONTAINER [CONTAINER ...]`。该命令会首先向容器发送 **SIGTERM** 信号，等待一段超时时间后，默认 **10s**，再发送一次 **SIGKILL** 信号来终止容器：

```shell
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE          COMMAND                 CREATED           STATUS            PORTS        NAMES
5b70a3ec5da4    ubuntu:18.04   "/bin/bash"             11 seconds ago    Up 10 seconds                  mytest
ce5c79a86a93    ubuntu:18.04   "/bin/bash -c 'while…"  15 minutes ago    Up 15 minutes                  serene_proskuriakova
a8c9e1c541f8    ubuntu:18.04   "/bin/bash"             24 hours ago      Up 24 hours                    blissful_neumann
test@VM-0-9-ubuntu:~$ docker stop --time 5 5b70a3ec5da4 ce5c79a86a93
5b70a3ec5da4
ce5c79a86a93
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE          COMMAND                 CREATED           STATUS            PORTS        NAMES
a8c9e1c541f8    ubuntu:18.04   "/bin/bash"             24 hours ago      Up 24 hours                    blissful_neumann
```

---

此时，执行`docker [container] prune`命令，会自动清楚掉所有处于停止状态的容器。此外，还以通过`docker [container] kill`命令直接发送 **SIGKILL**信号来强行终止容器，也可以通过`docker ps -qa`命令查看所有容器的 ID 信息，例如：

```shell
# 查看所有容器信息.
test@VM-0-9-ubuntu:~$ docker ps -a
CONTAINER ID   IMAGE         COMMAND                 CREATED         STATUS               PORTS    NAMES
5b70a3ec5da4  ubuntu:18.04   "/bin/bash"             10 minutes ago  Exited (0) Abo ...            mytest
ce5c79a86a93  ubuntu:18.04   "/bin/bash -c 'while…"  25 minutes ago  Exited (137)   ...            serene_proskuriakova
31e7a315e582  ubuntu:18.04   "/bin/bashddd"          32 minutes ago  Created                       festive_antonell
741ba8c91a30  ubuntu:18.04   "/bin/bashd"            33 minutes ago  Created                       hopeful_keller
180874fc147f  ubuntu:18.04   "/bin/bash"             46 minutes ago  Exited (127) 4 ...            objective_matsumoto
ba9471123659  ubuntu:18.04   "/bin/echo 'Hello Wo…"  53 minutes ago  Exited (0) 53  ...            happy_swartz
a8c9e1c541f8  ubuntu:18.04   "/bin/bash"             24 hours ago    Up 24 hours                   blissful_neumann
d902843014e7  ubuntu:18.04   "/bin/bash"             2 days ago      Exited (0) 2 d ...            objective_edison
a7ba948af80d  ubuntu:18.04   "bash"                  4 days ago      Exited (0) 4 d ...            stupefied_liskov
# 自动清楚掉处于停止状态的容器.
test@VM-0-9-ubuntu:~$ docker container prune
WARNING! This will remove all stopped containers.
Are you sure you want to continue? [y/N] y
Deleted Containers:
5b70a3ec5da46ab783935ebf27cf4911325e8a49858b5a183bd56c8e9b95b44a
ce5c79a86a935b74d9e6155e10729459701b1df324724cc9ab830eae6cd8cd4a
31e7a315e5826dee533e9d27f1e61b156ed89289db3cc1fa689e3273da584983
741ba8c91a301df831cab0100e865959b5484d178400952e7f2e98fee52b4be6
180874fc147f66925ac69d8430bc0baabff60c90e5a427cc67714a803e49584f
ba94711236597aedf290f0e2e378759595312f267a6fa70a5cbf784573eab492
d902843014e74c3eb24616f5b3faf6c73cac0e6b433bc137a97fe1ae6df56ecb
a7ba948af80dffc7e335e8e6f0f0d1e3351482f4e6f68b29663549063fb4d3aa

Total reclaimed space: 60B
# 查看正在运行的容器.
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE          COMMAND                 CREATED           STATUS            PORTS        NAMES
a8c9e1c541f8    ubuntu:18.04   "/bin/bash"             24 hours ago      Up 24 hours                    blissful_neumann
# 查看所有容器的`ID`信息.
test@VM-0-9-ubuntu:~$ docker ps -aq
a8c9e1c541f8
# 直接杀死一个容器.
test@VM-0-9-ubuntu:~$ docker kill a8c9e1c541f8
a8c9e1c541f8
# 查看正在运行的容器.
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE          COMMAND                 CREATED           STATUS            PORTS        NAMES
```

---

处于终止状态的容器，可以通过`docker [container] start`命令来重新启动，而`docker [container] restart`命令会将一个运行态的容器先终止，然后再启动：

```shell
# 查看正在运行的容器.
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE          COMMAND                 CREATED           STATUS            PORTS        NAMES
# 查看所有容器的`ID`信息.
test@VM-0-9-ubuntu:~$ docker ps -aq           
a8c9e1c541f8
# 重新启动一个容器.
test@VM-0-9-ubuntu:~$ docker start a8c9e1c541f8
a8c9e1c541f8
# 查看正在运行的容器.
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE          COMMAND                 CREATED           STATUS            PORTS        NAMES
a8c9e1c541f8    ubuntu:18.04   "/bin/bash"             24 hours ago      Up 24 hours                    blissful_neumann
# 先终止容器运行, 再重新启动容器.
test@VM-0-9-ubuntu:~$ docker restart a8c9e1c541f8
a8c9e1c541f8
```

## 进入容器

在使用`-d`参数时，容器启动后会进入后台，用户无法看到容器中的信息，也无法进行操作。这个时候如果需要进入容器进行，推荐使用官方的`attach`或`exec`命令。

### attach 命令

`attach`命令是 Docker 自带的命令，命令格式为：

```shell
# Attach local standard input, output, and error streams to a running container.
Usage:  docker attach [OPTIONS] CONTAINER

Options:
      --detach-keys string   指定退出`attach`模式的快捷键序列.
      --no-stdin             是否关闭标准输入, 默认是保持打开.
      --sig-proxy            是否代理收到的系统信号给应用进程, 默认是`true`.
```

下面示例如何使用该命令：

```shell
test@VM-0-9-ubuntu:~$ docker run -itd ubuntu:18.04
b0a1eaef4d7df358e8f625bc11368ef3218b9b33d72b67a28c24ec940d370586
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE           COMMAND        CREATED           STATUS           PORTS            NAMES
b0a1eaef4d7d    ubuntu:18.04    "/bin/bash"    9 seconds ago     Up 8 seconds                      vigilant_zhukovsky
test@VM-0-9-ubuntu:~$ docker attach b0a1eaef4d7d
root@b0a1eaef4d7d:/# exit
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE           COMMAND        CREATED           STATUS           PORTS            NAMES
```

然而，使用 **attach** 命令有时候并不方便，当多个窗口同时 **attach** 到同一个容器的时候，所有窗口都会同步显示，当某个窗口因命令阻塞时，其他窗口也无法执行操作了。

### exec 命令

从 Docker 的 1.3.0 版本起，Docker 提供了一个更加方便的工具 exec 命令，可以在运行中容器内直接执行任意命令，命令格式为：

```shell
# Run a command in a running container.
Usage:  docker exec [OPTIONS] CONTAINER COMMAND [ARG...]

Options:
  -d, --detach               在容器中后台执行命令.
      --detach-keys string   指定将容器切回后台的按键.
  -e, --env list             指定环境变量列表.
  -i, --interactive          打开标准输入接受用户输入命令, 默认值为`false`.
      --privileged           是否给执行命令以高权限, 默认值为`false`.
  -t, --tty                  分配伪终端, 默认值为`false`.
  -u, --user string          执行命令的用户名或ID.
  -w, --workdir string       容器内的工作路径.
```

例如，进入一个创建好的容器中，并启动一个 bash ：

```shell
test@VM-0-9-ubuntu:~$ docker run -itd ubuntu:18.04
b189a2a3f61244b48a93fdcbb6649d3b6b7096324327d700304a84257e38f50c
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE           COMMAND        CREATED           STATUS           PORTS            NAMES
b189a2a3f612    ubuntu:18.04    "/bin/bash"    9 seconds ago     Up 8 seconds                      vigilant_zhukovsky
test@VM-0-9-ubuntu:~$ docker exec -it b189a2a3f612 /bin/bash
root@b189a2a3f612:/# exit
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE           COMMAND        CREATED           STATUS           PORTS            NAMES
b189a2a3f612    ubuntu:18.04    "/bin/bash"    52 seconds ago    Up 51 seconds                     vigilant_zhukovsky
```

```shell
# 进一步地, 可以在容器中查看容器中的用户和进程信息.
root@b189a2a3f612:/# w
 15:24:52 up 19 days, 23:37,  0 users,  load average: 0.01, 0.02, 0.00
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
root@b189a2a3f612:/# ps -ef
UID        PID  PPID  C STIME TTY          TIME CMD
root         1     0  0 15:22 pts/0    00:00:00 /bin/bash
root        23     0  2 15:24 pts/1    00:00:00 /bin/bash
root        34    23  0 15:24 pts/1    00:00:00 ps -ef
```

## 删除容器

可以使用`docker [container] rm`命令来删除处于终止或退出状态的容器，命令格式为：

```shell
# Remove one or more containers.
Usage:  docker rm [OPTIONS] CONTAINER [CONTAINER...]

Options:
  -f, --force     是否强行终止并删除一个运行中的容器.
  -l, --link      删除容器的连接, 但保留容器.
  -v, --volumes   删除容器挂载的数据卷.
```

例如，查看处于终止状态的容器，并删除：

```shell
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID     IMAGE           COMMAND         CREATED           STATUS            PORTS           NAMES
b189a2a3f612     ubuntu:18.04    "/bin/bash"     7 minutes ago     Up 7 minutes                      mystifying_torvalds
test@VM-0-9-ubuntu:~$ docker rm b189a2a3f612
Error response from daemon: You cannot remove a running container b189a2a3f61244b48a93fdcbb6..... Stop the container before attempting removal or force remove
test@VM-0-9-ubuntu:~$ docker rm -f  b189a2a3f612
b189a2a3f612
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID     IMAGE           COMMAND         CREATED           STATUS            PORTS           NAMES
```

默认情况下，`docker rm`命令只能删除已经处于终止或退出状态的容器，并不能删除还处于运行状态的容器。如果要直接删除一个正在运行中的容器，可以添加`-f`参数。Docker 会先发送 SIGKILL 信号给容器，终止其中的应用，之后强行删除。

## 导入和到导出容器

某些时候，需要将容器从一个系统迁移到另一个系统，此时可以使用 Docker 的导入和导出功能，这也是 Docker 自身提供的一个重要特性。

### 导出容器

导出容器是指，导出一个已经创建的容器到一个文件，不管此时这个容器是否处于运行状态，可以使用`docker [container] export`命令，该命令格式为：

```shell
# Export a container's filesystem as a tar archive.
Usage:  docker export [OPTIONS] CONTAINER

Options:
  -o, --output string   通过`-o`选项来指定导出的`tar`文件名, 也可以直接通过重定向来实现.
```

```shell
# 导出`f0bebe1a8d7b`容器.
test@VM-0-9-ubuntu:~$ docker export f0bebe1a8d7b -o ubuntu_18.04.tar
test@VM-0-9-ubuntu:~$ ll
total 312644
drwxr-xr-x 12 test test      4096 Jun  3 10:27 ./
drwxr-xr-x  4 root root      4096 Mar 18 10:09 ../
-rw-------  1 test test  66612736 May 25 09:37 ubuntu_18.04.tar
```

之后，可将导出的 tar 文件传输到其他机器上，然后再通过导入命令导入到系统中，实现容器的迁移。

### 导入容器

导出的文件又可以使用`docker import`命令导入变成镜像，该命令格式为：

```shell
# Import the contents from a tarball to create a filesystem image.
Usage:  docker import [OPTIONS] file|URL|- [REPOSITORY[:TAG]]

Options:
  -c, --change list      在导入的同时执行对容器进行修改的`Dockerfile`指令.
  -m, --message string   Set commit message for imported image
```

```shell
# 将上一节导出的`ubuntu_18.04.tar`文件导入到系统中:
test@VM-0-9-ubuntu:~$ docker import ubuntu_18.04.tar albert/ubuntu:18.04
sha256:841b36a191f2c3e0e5a57ea041b7625f64d048f0fbca3eb197b9c201253d400a
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
albert/ubuntu       18.04               841b36a191f2        11 seconds ago      64.2MB
ubuntu              18.04               c3c304cb4f22        5 weeks ago         64.2MB
```

`docker import`与`docker load`命令十分相似。实际上，即可以使用 docker load 命令来导入镜像存出文件到本地镜像，也可以使用 docker import 命令来导入一个容器快照到本地镜像库。这两者的区别在于：**容器快照文件将丢弃所有的历史纪录和元数据信息，既可以仅保存容器当时的快照状态，而镜像存储文件将保存完整的记录，体积更大。此外，从容器快照文件导入时可以重新指定标签等元数据信息**。

## 查看容器

### 查看容器详情

查看容器详情可以使用`docker [container] inspect`命令。当查看某个容器的具体信息时，会议 json 格式返回包括容器 ID、创建时间、路径、状态、镜像、配置等在内的各项信息：

```shell
# Display detailed information on one or more containers.
Usage:  docker container inspect [OPTIONS] CONTAINER [CONTAINER...]
```

```SHELL
# 查看看一个容器的具体信息.
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
albert/ubuntu       18.04               841b36a191f2        10 minutes ago      64.2MB
ubuntu              18.04               c3c304cb4f22        5 weeks ago         64.2MB
genopro             1.0                 b74c517933a6        21 months ago       13.1GB
qh/genopro          1.0                 b74c517933a6        21 months ago       13.1GB
test@VM-0-9-ubuntu:~$ docker inspect 841b36a191f2
[
    {
        "Id": "sha256:841b36a191f2c3e0e5a57ea041b7625f64d048f0fbca3eb197b9c201253d400a",
        "RepoTags": [
            "albert/ubuntu:18.04"
        ],
	...
```

### 查看容器内进程

查看容器内进程可以使用`docker [container] top`命令，这个命令类似于 Linux 系统中的 top 命令，会打印出容器内的进程信息，包括 PID、用户、时间、命令等，命令格式如下：

```shell
# Display the running processes of a container.
Usage:  docker container top CONTAINER [ps OPTIONS]
```

```shell
# 例如, 查看某容器内的进程信息, 命令如下:
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID     IMAGE            COMMAND          CREATED            STATUS           PORTS       NAMES
05ca037fecc8     841b36a191f2     "/bin/bash"      11 seconds ago     Up 10 seconds                affectionate_einstein
test@VM-0-9-ubuntu:~$ docker top 05ca037fecc8
UID          PID          PPID          C           STIME          TTY          TIME          CMD
root         17435        17416         1           11:01          pts/0        00:00:00      /bin/bash
```

### 查看统计信息

查看统计信息，可以使用`docker [container] stats`命令，会显示 CPU、内存、存储、网络等使用的统计情况：

```shell
# Display a live stream of container(s) resource usage statistics.
Usage:  docker container stats [OPTIONS] [CONTAINER...]

Options:
  -a, --all             输出所有容器的统计信息, 默认仅在运行中.
      --format string   格式化输出信息.
      --no-stream       不持续输出, 默认会自动更新持续实时结果.
      --no-trunc        不截断输出信息.
```

```shell
# 例如, 查看当前运行中容器的系统资源使用统计.
test@VM-0-9-ubuntu:~$ docker container stats
CONTAINER ID     NAME                    CPU %       MEM USAGE / LIMIT   MEM %      NET I/O          BLOCK I/O        PIDS
05ca037fecc8     affectionate_einstein   0.00%       408KiB / 125.8GiB   0.00%      4.74kB / 0B      0B / 4.1kB       1
CONTAINER ID     NAME                    CPU %       MEM USAGE / LIMIT   MEM %      NET I/O          BLOCK I/O        PIDS
05ca037fecc8     affectionate_einstein   0.00%       08KiB / 125.8GiB    0.00%      4.74kB / 0B      0B / 4.1kB       1
```

## 其他容器命令

### 复制文件

`docker [container] cp`命令支持在容器和主机之间复制文件，命令格式为：

```shell
# Copy files/folders between a container and the local filesystem.
Usage:  docker cp [OPTIONS] CONTAINER:SRC_PATH DEST_PATH|-
        docker cp [OPTIONS] SRC_PATH|- CONTAINER:DEST_PATH
        
Options:
  -a, --archive       打包模式, 复制文件会带有原始的`uid/gid`信息.
  -L, --follow-link   跟随软连接. 当原始路径为软连接时, 默认只复制链接信息, 使用该选项会复制链接的目标内容.
```

```SHELL
# 例如, 将本地的路径`testcp`复制到`05ca037fecc8`容器的`/tmp`路径下:
test@VM-0-9-ubuntu:~$ touch testcp
test@VM-0-9-ubuntu:~$ docker cp ./testcp 05ca037fecc8:/tmp/
test@VM-0-9-ubuntu:~$ docker exec -it 05ca037fecc8 /bin/bash
root@05ca037fecc8:/# ll /tmp
total 8
drwxrwxrwt 1 root root 4096 Jun  3 04:06 ./
drwxr-xr-x 1 root root 4096 Jun  3 03:01 ../
-rw-r--r-- 1 1005  888    0 Jun  3 04:05 testcp
```

```shell
# 例如, 将`05ca037fecc8`容器的`/tmp`路径下将`testcp`拷贝到本地路径下:
test@VM-0-9-ubuntu:~$ docker exec -it 05ca037fecc8 /bin/bash
root@05ca037fecc8:/# ll /tmp
total 8
drwxrwxrwt 1 root root 4096 Jun  3 04:06 ./
drwxr-xr-x 1 root root 4096 Jun  3 03:01 ../
-rw-r--r-- 1 1005  888    0 Jun  3 04:05 testcp
root@05ca037fecc8:/# exit
exit
test@VM-0-9-ubuntu:~$ docker cp  05ca037fecc8:/tmp/testcp ./
test@VM-0-9-ubuntu:~$ ll
total 4.2G
drwxr-xr-x 2 test test 4.0K Jun  3 12:09 ./
drwxr-xr-x 3 test test 4.0K Jun  3 10:36 ../
-rw-r--r-- 1 test test    0 Jun  3 12:05 testcp
```

### 查看变更

`docker [container] diff`命令用于查看容器内文件系统的变更，其命令格式如下：

```shell
# Inspect changes to files or directories on a container's filesystem.
Usage:  docker diff CONTAINER
```

```shell
# 例如, 查看`05ca037fecc8`容器内的数据修改:
test@VM-0-9-ubuntu:~$ docker diff 05ca037fecc8
C /tmp
A /tmp/testcp
C /root
C /root/.bash_history
```

### 查看端口映射

`docker [container] port`命令可以用于查看容器的端口映射情况，其命令格式如下：

```shell
# List port mappings or a specific mapping for the container.
Usage:  docker port CONTAINER [PRIVATE_PORT[/PROTO]]
```

```shell
# 例如, 查看`05ca037fecc8`容器的端口映射情况:
test@VM-0-9-ubuntu:~$ docker port 05ca037fecc8
test@VM-0-9-ubuntu:~$ 
```

### 更新配置

`docker [container] update`命令可以用于更新容器的一些运行时的配置，主要是一些资源限制份额，其命令格式如下：

```shell
# Update configuration of one or more containers.
Usage:  docker update [OPTIONS] CONTAINER [CONTAINER...]

Options:
      --blkio-weight uint16        更新块 IO 限制, 10~1000 (default 0, 表示无限)
      --cpu-period int             限制 CPU 调度器 CFS 使用时间, 单位为微秒, 最小 1000 .
      --cpu-quota int              限制 CPU 调度器 CFS 配额, 单位为微秒, 最小 1000 .
      --cpu-rt-period int          限制 CPU 调度器的实时周期, 单位为微秒.
      --cpu-rt-runtime int         限制 CPU 调度器的实时运行时, 单位为微秒.
  -c, --cpu-shares int             限制 CPU 使用份额.
      --cpus decimal               限制 CPU 个数.
      --cpuset-cpus string         允许使用的 CPU 核数 (0-3, 0,1)
      --cpuset-mems string         允许使用的内存块 (0-3, 0,1)
      --kernel-memory bytes        限制使用的内核内存.
  -m, --memory bytes               限制使用的内存.
      --memory-reservation bytes   内存软限制.
      --memory-swap bytes          内存加上缓存区的限制, -1 表示为对缓冲区无限制.
      --restart string             容器退出后的重启策略.
```

```shell
# 限制容器`test`总配额为`1秒`.
$ docker update --cpu-quota 1000000 test
# 限制容器`test`所占用时间为`10%`.
$ docker update --cpu-period 100000 test
```



# 访问 Docker 仓库

仓库 Repository 是集中存放镜像的地方，又分为公共仓库和私有仓库。有时候容器把仓库与注册服务器 Registry 混淆，实际上注册服务器是存放仓库的具体服务器，一个注册服务器上可以有多个仓库，而每个仓库下面可以有多个镜像。从这方面来说，仓库可以被认为是一个具体的项目或目录。例如，对于 **private-docker.com/ubuntu** 来说，**private-docker.com** 是注册服务器地址，**ubuntu** 是仓库名。

## Docker Hub 公共镜像市场

Docker Hub 是 Docker 官方提供的最大的公共镜像仓库，目前超过 100,000 的镜像，地址为：https://hub.docker.com/。大部分对镜像的需求，都可以通过在 Docker Hub 中直接下载镜像来实现。

**登录**：

可以通过执行`docker login`命令来输入用户名和密码来完成注册和登录。登录成功后，本地用户目录下会自动创建 **.docker/config.json**文件，用来保存用户的认证信息。登录成功的用户可以上传个人制作的镜像到 Docker Hub 。

```shell
test@VM-0-9-ubuntu:~$ docker login
Login with your Docker ID to push and pull images from Docker Hub. 
If you don't have a Docker ID, head over to https://hub.docker.com to create one.
Username: albertxn
Password: 
WARNING! Your password will be stored unencrypted in /home/test/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store
Login Succeeded
```

**查询和下载镜像**：

用户无须登录即可通过`docker search`命令来查找官方仓库中的镜像，并利用`docker [image] pull`命令来将它下载到本地：

```shell
# https://hub.docker.com/_/centos/?tab=tags
test@VM-0-9-ubuntu:~/.docker$ docker search centos
NAME                      DESCRIPTION                         STARS               OFFICIAL            AUTOMATED
centos                    he official build of CentOS.        6036                [OK]                
ansible/centos7-ansible   Ansible on Centos7                  130                                     [OK]
consol/centos-xfce-vnc    Centos container with "headless" …  115                                     [OK]
jdeathe/centos-ssh        OpenSSH / Supervisor / EPEL/IUS …   114                                     [OK]
...
```

根据是否为官方提供，可将这些镜像资源分类两类：一种是类似与 centos 这样的基础镜像，也称为根镜像。这些镜像是由 Docker 公司创建、验证、支持、提供，这样的镜像往往使用单个单词作为名字；另一种类型的镜像，比如 ansible/centos7-ansible 镜像，是由 Docker 用户 ansible 创建并维护的，带有用户名称为前缀，表示是某用户下的某个仓库，可以通过 **user_name/镜像名** 来指定使用某个用户提供的镜像。

**上传镜像**：

用户也可以在登录后通过`docker push`命令来将本地镜像推送到 Docker Hub 。

**自动创建**：

自动创建是 Docker Hub 提供的自动化服务，这一功能可以自动跟随项目代码的变更而重新构建镜像。例如，用户构建了某应用镜像，如果应用发布新版本，用户需要手动更新镜像。而自动创建则允许用户通过 Docker Hub 指定跟踪一个目标网站，目前支持 GitHub 或 BitBucket 上的项目，一旦项目发生新的提交，则自动执行创建：

1. 创建并登录 Docker Hub，以及目标网站，如 GitHub；
2. 在目标网站中允许 Docker Hub 访问服务；
3. 在 Docker Hub 中配置一个**自动创建**类型的项目；
4. 选取一个目标网站中的项目，需要含有 Dockerfile 和分支；
5. 指定 Dockerfile 的位置，并提交创建。

之后，可以在 Docker Hub 的**自动创建**页面中跟踪每次创建的状态。

## 第三方镜像市场

国内有不少云服务商都提供了 Docker 镜像市场，包括腾讯云、网易云、阿里云等。

**查看镜像**：

访问 https://hub.daocloud.io/，通过相应的检索即可看到已存在的仓库和存储的镜像，包括 Ubuntu、Java、MongoDB、MySQL、Nginx 等热门仓库和镜像，且其中的镜像会保持与 Docker Hub 中官方镜像的同步。

**下载镜像**：

下载镜像也是使用`docker pull`命令，但是要在镜像名称前面添加注册服务器的具体地址，格式为：

```shell
daocloud.io/<namespace>/<repository>:<tag>
```

例如，要下载 Docker 官方仓库中的 node:latest 镜像，可以使用如下命令：

```shell
# 正常情况下, 镜像下载会比直接从 Docker Hub 中下载快很多.
# 下载后, 可以更新镜像的标签, 与官方标签保持一致, 方便使用: docker tag daocloud.io/library/nginx:latest node:latest
test@VM-0-9-ubuntu:~$ docker pull daocloud.io/library/nginx:latest
Using default tag: latest
latest: Pulling from library/nginx
123275d6e508: Pull complete 
6cd6a943ce27: Pull complete 
a50b5ac4a7fb: Pull complete 
Digest: sha256:6b3b6c113f98e901a8b1473dee4c268cf37e93d72bc0a01e57c65b4ab99e58ee
Status: Downloaded newer image for daocloud.io/library/nginx:latest
daocloud.io/library/nginx:latest
test@VM-0-9-ubuntu:~$ docker tag daocloud.io/library/nginx:latest node:latest
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
daocloud.io/library/nginx   latest              e791337790a6        7 weeks ago         127MB
node                        latest              e791337790a6        7 weeks ago         127MB
```

## 搭建本地私有仓库

**使用 registry 镜像创建私有仓库**：

安装 Docker 后，可以通过官方提供的 registory 镜像来简单搭建一套本地私有仓库环境：

```shell
# https://hub.docker.com/_/registry/?tab=tags
test@VM-0-9-ubuntu:~$ docker run -d -p 5001:5000 registry:2.7.1
Unable to find image 'registry:2.7.1' locally
2.7.1: Pulling from library/registry
...
Digest: sha256:7d081088e4bfd632a88e3f3bcd9e007ef44a796fddfe3261407a3f9f04abe1e7
Status: Downloaded newer image for registry:2.7.1
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
registry                    2.7.1               708bc6af7e5e        4 months ago        25.8MB
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE           COMMAND         CREATED         STATUS           PORTS                    NAMES
8f2ecad20dd9    registry:2.7.1  "/entrypoin…"   5 seconds ago   Up 4 seconds     0.0.0.0:5001->5000/tcp   distracted_feynman
```

这将自动下载并启动一个 registry 容器，创建本地的私有化仓库服务。默认情况下，仓库回被创建在容器的 **/var/lib/registory** 目录下，可以通过 **-v** 参数来将镜像文件存放在本地的指定路径，例如，下面的例子将上传的镜像放到 **/opt/data/registory** 目录：

```shell
# 此时将改变镜像文件的存放路径, 且启动一个私有仓库服务, 监听端口为`5001`:
# `-v $HOME/registry:/var/lib/registry`, 默认情况下, 会将仓库存放于容器内的`/var/lib/registry`目录下, 指定本地目录挂载到容器.
# `-p 5001:5000`, 端口映射. 即本地`5001`端口映射到容器中的`5000`端口.
# `--restart=always`, 在容器退出时总是重启容器, 主要应用在生产环境.
# ``--privileged=true`, 在 CentOS7 中的安全模块 selinux 把权限禁掉了, 参数给容器加特权, 不加上传镜像会报类似权限错误.
$ docker run -d -p 5001:5000 -v /opt/data/registory:/var/lib/registory registry:2.7.1
$ docker run -d -p 5001:5000 -v $HOME/registry:/var/lib/registry --restart=always --privileged=true --name registry registry:2.7.1
```

**管理私有仓库**：

```shell
# 查看私有仓库所有镜像:
test@VM-0-9-ubuntu:~$ curl http://127.0.0.1:5001/v2/_catalog
{"repositories":[]}
# 获取某个镜像的标签列表:
test@VM-0-9-ubuntu:~$ curl http://127.0.0.1:5001/v2/ububtu/tags/list    
{"name":"ububtu","tags":["18.04"]}
```

```shell
# 上传一个镜像到私有仓库:
# 0.启动一个容器.
test@VM-0-9-ubuntu:~$ docker run -d -p 5001:5000 -v $HOME/registry:/var/lib/registry --restart=always --privileged=true --name registry registry:2.7.1
0994ab5b1d41b1847ed2d0afd70d5589def98f32d8401f091e72c81c8070f7f1
# 1.查看私有仓库所有的镜像.
test@VM-0-9-ubuntu:~$ curl http://127.0.0.1:5001/v2/_catalog
{"repositories":[]}
# 2.使用`docker images`命令查看镜像.
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
ubuntu                      18.04               c3c304cb4f22        6 weeks ago         64.2MB
# 3.使用`docker tag`命令将这个镜像标记为`127.0.0.1:5001/ubuntu:18.04`.
test@VM-0-9-ubuntu:~$ docker tag ubuntu:18.04 127.0.0.1:5001/ububtu:18.04
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
127.0.0.1:5001/ububtu       18.04               c3c304cb4f22        6 weeks ago         64.2MB
ubuntu                      18.04               c3c304cb4f22        6 weeks ago         64.2MB
# 4.使用`docker push`上传标记的镜像.
test@VM-0-9-ubuntu:~$ docker push 127.0.0.1:5001/ububtu:18.04
The push refers to repository [127.0.0.1:5001/ububtu]
28ba7458d04b: Pushed 
838a37a24627: Pushed 
a6ebef4a95c3: Pushed 
b7f7d2967507: Pushed 
18.04: digest: sha256:b58746c8a89938b8c9f5b77de3b8cf1fe78210c696ab03a1442e235eea65d84f size: 1152
# 5.挂载路径下存在了相应文件夹.
test@VM-0-9-ubuntu:~$ ll $HOME/registry
total 12
drwxrwxr-x  3 test test 4096 Jun  9 09:58 ./
drwxr-xr-x 18 test test 4096 Jun  9 09:29 ../
drwxr-xr-x  3 root root 4096 Jun  9 09:58 docker/
# 6.查看私有仓库所有的镜像.
test@VM-0-9-ubuntu:~$ curl http://127.0.0.1:5001/v2/_catalog
{"repositories":["ububtu"]}
# 7.从私有仓库中下载和安装镜像(预先删掉之前的镜像).
test@VM-0-9-ubuntu:~$ docker pull 127.0.0.1:5001/ububtu:18.04
18.04: Pulling from ububtu
Digest: sha256:b58746c8a89938b8c9f5b77de3b8cf1fe78210c696ab03a1442e235eea65d84f
Status: Downloaded newer image for 127.0.0.1:5001/ububtu:18.04
127.0.0.1:5001/ububtu:18.04
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
127.0.0.1:5001/ububtu       18.04               c3c304cb4f22        6 weeks ago         64.2MB
ubuntu                      18.04               c3c304cb4f22        6 weeks ago         64.2MB
```



# Docker 数据管理

在生产环境中使用 Docker，往往需要对数据进行持久化，或者需要在多个容器之间进行数据共享，这必然涉及容器的数据管理操作。容器中的管理数据主要有两种方式：**数据卷**，容器内数据直接映射到本地主机环境；**数据卷容器**，使用特定容器维护数据卷。

## 数据卷

数据卷是一个可供容器使用的特殊目录，将主机操作系统目录直接映射进容器，类似于 Linux 中的 mount 行为，数据卷可以提供很多有用的特性：

1. 数据卷可以在容器之间共享和重用，容器间传递数据将变得高效与方便；
2. 对数据卷内数据的修改会立马生效，无论是容器内操作还是本地操作；
3. 对数据卷的更新不会影响镜像，解耦开应用和数据；
4. 卷会一直存在，知道没有容器使用，可以安全地卸载它。

### 创建数据卷

Docker 提供了 volume 子命令来管理数据卷，如下命令可以快速在本地创建一个数据卷：

```shell
ubuntu@VM-0-9-ubuntu:~$ docker volume
Usage:  docker volume COMMAND
Manage volumes
Commands:
  create      创建一个数据卷
  inspect     查看详细信息
  ls          列出已有数据卷
  prune       清理无用数据卷
  rm          删除数据卷
Run 'docker volume COMMAND --help' for more information on a command.
```

```shell
# 快速在本地创建一个数据卷.
test@VM-0-9-ubuntu:~$ docker volume create -d local test
test
# 此时, 查看`/var/lib/docker/volumes`路径下, 会发现所创建地数据卷位置（需要一定地权限）.
ubuntu@VM-0-9-ubuntu:~$ sudo ls -l /var/lib/docker/volumes
total 156
drwxr-xr-x 3 root root  4096 Jun  8 12:19 1688a7a3c89a5f3783dbca32e7fd5c8e03cd4680c683f61b3dc8406684526f23
drwxr-xr-x 3 root root  4096 Apr 26 14:46 ffb8829214e3c6a151bed2574a4c67ec8433219180cfa6046596fa634ca1fb6c
-rw------- 1 root root 65536 Jun  9 10:45 metadata.db
drwxr-xr-x 3 root root  4096 Jun  9 10:45 test
```

### 绑定数据卷

除了使用 volume 子命令来管理数据卷之外，还可以在创建容器时将主机本地的任意路径挂载到容器内作为数据卷，这种形式创建的数据卷称为绑定数据卷。在用 **docker [container] run** 命令的时候，可以用 **--mount** 选项来使用数据卷，它支持三种类型的数据卷，包括：

- **vloume**，普通数据卷，映射到主机 /var/lib/docker/volumes 路径下；
- **bind**，绑定数据卷，映射到主机指定路径下；
- **tmpfs**，临时数据卷，只存在于内存中。

下面使用 training/webapp 镜像创建一个 Web 容器，并创建一个数据卷挂载到容器的 /opt/webapp 目录：

```shell
# 安装`nginx:1.19.0`镜像.
test@VM-0-9-ubuntu:~$ docker pull nginx:1.19.0
1.19.0: Pulling from library/nginx
afb6ec6fdc1c: Already exists 
dd3ac8106a0b: Pull complete 
8de28bdda69b: Pull complete 
a2c431ac2669: Pull complete 
e070d03fd1b5: Pull complete 
Digest: sha256:15c65919f2b5889636c671b99ca4e70eff1e78c3114d60600b29286e476e6876
Status: Downloaded newer image for nginx:1.19.0
docker.io/library/nginx:1.19.0
# 使用`nginx:1.19.0`镜像创建一个`Web`容器, 并创建一个数据卷挂载到容器的`/opt/webapp`目录.
test@VM-0-9-ubuntu:~$ docker run -d -p 8080:80 --name web --mount type=bind,source=$HOME/webapp,destination=/opt/webapp nginx:1.19.0
80fe111dadf2abf41fb75d28a323bbbc6fc2a0ca601ac6f706143436bdf7c1d2
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID   IMAGE           COMMAND                  CREATED           STATUS            PORTS                    NAMES
80fe111dadf2   nginx:1.19.0    "/docker-entrypoint.…"   3 seconds ago     Up 2 seconds      0.0.0.0:8080->80/tcp     web
# 或者使用`-v`标记可以在容器内创建一个数据卷.
Total reclaimed space: 1.114kB
test@VM-0-9-ubuntu:~$ docker run -d -p 8080:80 --name web -v $HOME/webapp:/opt/webapp nginx:1.19.0
0cb474603a64baa28a2c632d02f99463bfefc006a7ccfb9c6b85a5dab8072c29
```

这个功能在进行应用测试的时候十分方便，比如用户可以放置一些程序或数据到本地目录中实时进行更新，然后在容器内运行和使用。如果使用`--mount`标志来绑定挂载 Docker 主机中不存在的文件或目录，Docker 不会自动为你创建，而是产生报错。如果使用`-v`或`--volume`标志来绑定挂载 Docker 主机中不存在的文件或目录，`-v`会为你创建一个端点，它会始终创建目录。

---

Docker 挂载数据卷的默认权限是读写`rw`，用户也可以通过`ro`指定为只读：

```shell
# 加了`:ro`之后, 容器内对所挂载数据卷内的数据就无法修改了.
test@VM-0-9-ubuntu:~$ docker run -d -p 8080:80 --name web -v $HOME/webapp:/opt/webapp:ro nginx:1.19.0
0cb474603a64baa28a2c632d02f99463bfefc006a7ccfb9c6b85a5dab8072c29
```

## 数据卷容器

如果用户需要在多个容器之间共享一些持续更新的数据，最简单的方式是使用数据卷容器。数据卷容器也是一个容器，但是它的目的是专门提供数据卷给其他容器挂载。

首先，创建一个数据卷容器 dbdata，并在其中创建一个数据卷挂载到 /dbdata：

```shell
test@VM-0-9-ubuntu:~$ docker run -it -v /dbdata --name dbdata ubuntu:18.04
root@76fde0cbb7fc:/# 
root@76fde0cbb7fc:/# ls
bin  boot  dbdata  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
```

然后，可以在其他容器中使用`--volumes-from`来挂载 dbdata 容器中的数据卷，例如创建 db1 和 db2 两个容器，并从 dbdata 容器挂载数据卷：

```shell
docker run -it --volumes-from dbdata --name db1 ubuntu:18.04
test@VM-0-9-ubuntu:~$ docker run -it --volumes-from dbdata --name db1 ubuntu:18.04
root@44e25e4fe451:/# 
test@VM-0-9-ubuntu:~$ docker run -it --volumes-from dbdata --name db2 ubuntu:18.04
root@b3c94fe9a6e4:/# 
```

此时，容器 db1 和 db2 都挂载同一个数据卷到相同的 /dbdata 目录，三个容器任何一方在该目录下的写入，其他容器都可以看到。可以多次使用 **--volumes-from** 参数来从多个容器挂载多个数据卷，还可以从其他已经挂载了容器卷的容器来挂载数据卷：

```shell
test@VM-0-9-ubuntu:~$ docker run -d --name db3 --volumes-from db1 ubuntu:18.04
2be8d94c3d52f16d1bee4a4184cb4c319b25f33222e2cdb17c543109b96fe314
```

使用 **--volumes-from** 参数所挂载数据卷的容器自身并不需要保持在运行状态。如果删除了挂载的容器，包括 dbdata、db1 和 db2，数据卷并不会自动删除，如果要删除一个数据卷，必须在删除最后一个还挂载它的容器时显式使用`docker rm -v`命令来指定同时删除关联的容器。

## 利用数据卷容器来迁移数据

可以利用数据卷容器对其中的数据卷进行备份、恢复，以实现数据的迁移。

### 备份

使用下面的命令来备份 **dbdata** 数据卷容器内的数据卷：

```shell
test@VM-0-9-ubuntu:~/webapp$ docker run --volumes-from dbdata -v $(pwd):/backup --name worker ubuntu:18.04 tar -zvcPf /backup/backup.tar /dbdata 
/dbdata/
/dbdata/ceshi.txt
test@VM-0-9-ubuntu:~/webapp$ ll
total 12
drwxrwxr-x  2 test test 4096 Jun  9 13:36 ./
drwxr-xr-x 19 test test 4096 Jun  9 11:05 ../
-rw-r--r--  1 root root  149 Jun  9 13:36 backup.tar
```

首先利用 ubuntu:18.04 镜像创建了一个容器 worker，使用`--volumes-from dbdata`参数来让 worker 容器挂载 dbdata 容器的数据卷；使用`-v $(pwd):/backup`参数来挂载本地的当前目录到 worker 容器的 /backup 目录。worker 容器启动后，使用`tar -zvcPf /backup/backup.tar /dbdata`命令将 /dbdata 下内容备份为容器内的 /backup/backup.tar，即宿主机当前目录下的 **backup.tar** 。

`tar -xvf backup.tar`可以解压该文件。

### 恢复

如果要恢复数据到一个容器，可以安装下面的操作：

```shell
# 1.创建一个带有数据卷的容器`dbdata2`.
test@VM-0-9-ubuntu:~$ docker run -itd -v /dbdata --name dbdata2 ubuntu:18.04 /bin/bash
# 2.然后创建另一个新的容器, 挂载 dbdata2 的容器, 并使用 untar 解压备份文件到所挂载的容器卷中.
test@VM-0-9-ubuntu:~$ docker run --volumes-from dbdata2 -v /home/test/webapp:/backup ubuntu:18.04 tar -xvf /backup/backup.tar
tar: Removing leading `/' from member names
/dbdata/
/dbdata/ceshi.txt
# 3.进入第一步创建的容器`dbdata2`, 并查看文件是否存在.
test@VM-0-9-ubuntu:~$ docker attach dbdata2
root@bce497fcb518:/# ll /dbdata/          
total 8
drwxr-xr-x 2 root root 4096 Jun  9 04:31 ./
drwxr-xr-x 1 root root 4096 Jun  9 06:01 ../
-rw-r--r-- 1 root root    0 Jun  9 04:31 ceshi.txt
```



# 端口映射和容器互联

在实践中，经常会碰到需要多个服务组件容器共同协作的情况，这往往需要多个容器之间能够互相访问到对方的服务。Docker 除了通过网络访问外，还提供了两个很方便的功能来满足服务访问的基本需求：一个是允许映射容器内应用的服务端到本机宿主主机；另一个是互联机制实现多个容器间通过容器名来快速访问。

## 端口映射实现容器访问

### 从外部访问容器应用

在启动容器的时候，如果不指定对应参数，在容器外部是无法通过网络来访问容器内的网络应用和服务的。当容器中运行一些网络应用，要让外部访问这些应用时，可以通过`-P`或`-p`参数来指定端口映射。当使用 -P 标记时，Docker 会随机映射一个端口到内部容器开放的网络端口：

```shell
# 使用`-P`随机端口.
test@VM-0-9-ubuntu:~$ docker run -d -P --name web -v $HOME/webapp:/opt/webapp:ro nginx:1.19.0
a91567287b5acccc0e3a2f9e9caca50d7adcd575a0569a641a9e5b8c6c629ae4
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID  IMAGE          COMMAND                  CREATED             STATUS           PORTS                   NAMES
a91567287b5a  nginx:1.19.0   "/docker-entrypoint.…"   4 seconds ago       Up 2 seconds     0.0.0.0:32777->80/tcp   web
test@VM-0-9-ubuntu:~$ docker logs a91567287b5a
2020/06/09 06:26:49 [error] 28#28: *1 open() "/usr/share/nginx/html/favicon.ico" failed (2: No such file or directory), client: 59.41.66.146, server: localhost, request: "GET /favicon.ico HTTP/1.1", host: "49.235.24.68:32777", referrer: "http://49.235.24.68:32777/"
```

```shell
test@VM-0-9-ubuntu:~$ docker run -d -p 12345:80 --name web -v $HOME/webapp:/opt/webapp:ro nginx:1.19.0
6973577c4033d6c528408969cd32181c61fc70334a66e28de907318d6e510d89
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID     IMAGE            COMMAND                  CREATED             STATUS          PORTS                   NAMES
6973577c4033     nginx:1.19.0     "/docker-entrypoint.…"   3 seconds ago       Up 2 seconds    0.0.0.0:12345->80/tcp   web
```

### 映射多个接口地址

```shell
# 多次使用`-p`标记可以绑定多个端口.
test@VM-0-9-ubuntu:~$ docker run -d -p 12345:80 -p 54321:80 --name web -v $HOME/webapp:/opt/webapp:ro nginx:1.19.0
fa43a3fb7a2a79da9c3ab73ddaa87b4678571c370ab7949f3d05e384db4a2888
```

### 映射到指定地址的指定端口

```shell
$ docker run -d -p 127.0.0.1:12345:80 --name web -v $HOME/webapp:/opt/webapp:ro nginx:1.19.0
```

### 映射到指定地址的任意端口

```shell
# 绑定任意端口到容器的`80`端口, 本地主机会自动分配一个端口.
$ docker run -d -p 127.0.0.1::80 --name web -v $HOME/webapp:/opt/webapp:ro nginx:1.19.0
# 使用`udp`标记来指定`udp`端口.
$ docker run -d -p 127.0.0.1:12345:80/udp --name web -v $HOME/webapp:/opt/webapp:ro nginx:1.19.0
```

### 查看映射端口配置

使用`docker port`来查看当前映射的端口配置，也可以查看到绑定的地址：

```shell
test@VM-0-9-ubuntu:~$ docker run -d -p 12345:80 -p 54321:80 --name web -v $HOME/webapp:/opt/webapp:ro nginx:1.19.0
8f51778bfd0dfba408e6042659ee7f5390727f086d0c3e8b85a3515c0223a57c
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID   IMAGE          COMMAND   CREATED         STATUS          PORTS                                          NAMES
8f51778bfd0d   nginx:1.19.0   "/docn…"  19 seconds ago  Up 17 seconds   0.0.0.0:12345->80/tcp, 0.0.0.0:54321->80/tcp   web
test@VM-0-9-ubuntu:~$ docker port 8f51778bfd0d
80/tcp -> 0.0.0.0:54321
80/tcp -> 0.0.0.0:12345
```

容器有自己的内部网络和 IP 地址，使用`docker [container] inspect ID`可以获取容器的具体信息。

## 互联机制实现便捷互访

容器的互联是一种让多个容器中的应用进行快速交互的方式，它会在源和接收容器之间创建连接关系，接收容器可以通过容器名快速访问到源容器，而不用指定具体的 IP 地址。

---

**自定义容器命名**：

连接系统依据容器的名称来执行，因此，首先需要自定义一个好记的容器名，虽然当创建容器的时候，系统默认会分配一个名字，但自定义命名容器有两个好处，一是自定义的命名，比较好记，比如一个 Web 应用容器我们可以给它起名叫 web，一目了然；当要连接到其他容器的时候，即便重启，也可以使用容器名而不用改变，比如连接 web 容器到 db 容器。

使用`--name`标记可以为容器自定义命名：

```shell
test@VM-0-9-ubuntu:~$ docker run -d -p 12345:80 --name web nginx:1.19.0
0444145654cd09a973b381298de63413ccfe2d1ec1d0d095e120ad1348d5398a
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE           COMMAND                  CREATED           STATUS           PORTS                   NAMES
0444145654cd    nginx:1.19.0    "/docker-entrypoint.…"   7 seconds ago     Up 6 seconds     0.0.0.0:12345->80/tcp   web
```

注意：容器的名称是唯一的，如果已经命名了一个叫 web 的容器，当你要再次使用 web 这个名称的时候，需要先用`docker rm`命令删除之前创建的同名容器。另外，在执行`docker [container] run`的时候，如果添加了`--rm`标记，则容器在终止后立刻删除，且`--rm`和`-d`参数不能同时使用。

----

**容器互联**：

使用`--link`参数可以让容器之间安全地进行交互，下面先创建一个新的数据库容器：

```shell
# 1.创建一个`db`容器.
test@VM-0-9-ubuntu:~$ docker run -d --name db training/postgres
4ac3e0aa8e65fbbd61d9774ba5d72b39b0785c37b8f2066cbabbc53daf3a6472
# 2.创建一个`web`容器, 并将它连接到`db`容器.
test@VM-0-9-ubuntu:~$ docker run -d -p 12345:80 --name web --link db:db nginx:1.19.0
49a4a530d6b80b14b636bbb548d00eb3d1eaa7750596fd4eebc1a3683d9c0ae3
# 3.查看容器连接.
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE              COMMAND            CREATED             STATUS              PORTS                   NAMES
93c842299bdb    nginx:1.19.0       "/docker-entry…"   About a minute ago  Up About a minute   0.0.0.0:12345->80/tcp   web
4ac3e0aa8e65    training/postgres  "su postgres -…"   3 minutes ago       Up 3 minutes        5432/tcp                db
```

`--link`参数地格式为`--link name:alias`，其中 name 是要链接的容器的名称，alias 是别名。上述代码结果可以看到自定义命名的容器：db 和 web 。Docker 相当于在两个互联的容器之间创建了一个虚拟通道，而且不用映射它们的端口到宿主主机上。在启动 db 容器的时候并没有使用 **-p** 和 **-P** 标记，从而避免了暴露数据库服务端口到外部网络上。

使用 env 命令来查看 web 容器的环境变量：

```shell
# 使用 env 命令来查看 web 容器的环境变量:
test@VM-0-9-ubuntu:~$ docker run --rm --name web2 --link db:db nginx:1.19.0 env 
HOSTNAME=5e8ff5eb589e
DB_PORT=tcp://172.18.0.3:5432
HOME=/root
DB_PORT_5432_TCP=tcp://172.18.0.3:5432
DB_NAME=/web2/db
PKG_RELEASE=1~buster
NGINX_VERSION=1.19.0
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
NJS_VERSION=0.4.1
DB_ENV_PG_VERSION=9.3
PWD=/
DB_PORT_5432_TCP_ADDR=172.18.0.3
DB_PORT_5432_TCP_PORT=5432
DB_PORT_5432_TCP_PROTO=tcp
```

```shell
# 查看父容器的`/etc/hosts`文件:
# 这里有两个`hosts`信息, 第一个是`web`容器, 使用自己的`ID`作为默认主机名, 第二个是`db`容器的`IP`和主机名.
test@VM-0-9-ubuntu:~$ docker run -it --rm --link db:db nginx:1.19.0 /bin/bash   
root@11729d2bb3b4:/# cat /etc/hosts 
127.0.0.1       localhost
::1     localhost ip6-localhost ip6-loopback
fe00::0 ip6-localnet
ff00::0 ip6-mcastprefix
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
172.18.0.3      db f539d85d8b25
172.18.0.5      11729d2bb3b4
# 可以在`web`容器中安装`ping`命令来测试跟`db`容器的连通.
root@11729d2bb3b4:/# ping db
PING db (172.18.0.3) 56(84) bytes of data.
64 bytes from db (172.18.0.3): icmp_seq=1 ttl=64 time=0.099 ms
64 bytes from db (172.18.0.3): icmp_seq=2 ttl=64 time=0.063 ms
64 bytes from db (172.18.0.3): icmp_seq=3 ttl=64 time=0.087 ms
```



# 使用 Dockerfile 创建镜像

Dockerfile 是一个文本格式的配置文件，用户可以使用 Dockerfile 来快速创建自定义镜像。

## 基本结构

Dockerfile 由一行行命令语句组成，并且支持以`#`开头的注释行。一般而言，Dockerfile 主体内容分为四个部分：**基础镜像信息**、**维护者信息**、**镜像操作者指令**和**容器启动时指令**。

```shell
# 下面给出一个简单的示例:

# escape=\ (backlash)
# This dockerfile uses the ubuntu:xeniel image
# VERSION 2 - EDITION 1
# Author: Albert丶XN
# Command format: Instruction [arguments / command] ...

# Base image to use, this must be set as the first line.
FROM ubuntu:xeniel

# Maintainer: docker_user <docker_user at email.com> (@docker_user)
LABEL maintainer Albert丶XN<albertxn@126.com>

# Commands to update the image
RUN echo "deb http://archive.ubuntu.com/ubuntu xeniel main universe" >> /etc/apt/sources.list
RUN apt-get update && apt-get install -y nginx
RUN echo "\ndaemon off;" >> /etc/nginx/nginx.conf

# Command when creating a new container
CMD /usr/sbin/nginx
```

首先可以通过注释来指定解析器命令，后续通过注释说明镜像的相关信息。主体部分首先使用 FROM 指令指明所基于的镜像名称，接下来一般是使用 LABEL 指令说明维护者信息。后面则是镜像操作指令，例如，RUN 指令将对镜像执行跟随的命令，每运行一条 RUN 指令，镜像添加新的一层，并提交。最后 CMD 指令，来指定运行容器时的操作命令。

---

## 指令说明

Dockerfile 中指令的一般格式为 INSTRUCATION arguments，包括**配置指令**和**操作指令**，如下所示：

```properties
# 配置指令:
ARG			: 定义创建镜像过程中使用的变量.
FROM		:指定所创建镜像的基础镜像.
LABEL		:为生成的镜像添加元数据标签信息.
EXPOSE		: 声明镜像内服务监听的端口.
ENV			: 指定环境变量.
ENTRYPOINT	: 指定镜像的默认入口命令.
VOLUME		: 创建一个数据卷挂载点.
USER		: 指定运行容器时的用户名和UID.
WORKDIR		: 配置工作目录.
ONBUILD		: 创建子镜像时指定自动执行的操作命令.
STOPSIGNAL	: 指定退出的信号值.
HEALTHCHECK	: 配置所启动容器如何进行健康检查.
SHELL		: 指定默认shell类型.
```

```properties
# 操作指令:
RUN			 : 运行指定命令.
CMD		 	 : 启动容器时指定默认执行的命令.
ADD			 : 添加内容到镜像.
COPY		 : 复制内容到镜像.
```

### 配置指令

#### ARG

`ARG`用于定义创建镜像过程中使用的变量。格式：`ARG <name>[=<default value>]`。在执行 docker build 时，可以通过 **--build-arg**来为变量赋值。当镜像编译成功后，ARG指定的变量将不再存在，而ENV指定的变量将在镜像中保留，例如：

```shell
FROM ubuntu
ARG CONT_IMG_VER v2.0.1
ENV CONT_IMG_VER v1.0.0
RUN echo $CONT_IMG_VER
```

```shell
# 可以使用`ARG`或`ENV`指令指定可用于`RUN`指令的变量. 使用`ENV`定义的环境变量始终会覆盖同一名称的`ARG`指令定义的变量.
# 在这种情况中, `RUN`指令解析`CONT_IMG_VER`变量的值为`v1.0.0`而不是`ARG`设置并由用户传递过来的`v2.0.1`.
$ docker build --build-arg CONT_IMG_VER=v2.0.1 Dockerfile
```

如果要在构建期间使用多个环境变量，例如，让我们说设置用户名和密码：

```shell
FROM ubuntu:16.04
ARG SMB_PASS
ARG SMB_USER
RUN useradd -ms /bin/bash $SMB_USER
RUN echo "$SMB_PASS\n$SMB_PASS" | smbpasswd -a $SMB_USER
```

```shell
$ docker build --build-arg SMB_PASS=swed24sw --build-arg SMB_USER=Ubuntu . -t IMAGE_TAG
```

---

Docker 内置了一些镜像创建变量，用户可以直接使用而无须声明，不区分大小写，包括：**HTTP_PROXY**、**HTTPS_PROXY**、**FTP_PROXY** 和 **NO_PROXY** 。

#### FROM

`FROM`，指定所创建镜像的基础镜像。格式：`FROM <image>:<tag> [AS <name>]`或`FROM <image>@<digest> [AS <name>]`或`FROM <image> [AS <name>]`。任何 Dockerfile 中第一条指令必须为 FROM 指令，并且，如果在同一个 Dockfile 中创建多个镜像时，可以使用多个 FROM 指令。为了保证镜像精简，可以选用一些体积较小的镜像作为基础镜像，例如：

```shell
ARG VERSION=9.3
FROM debian:${VERSION}
```

#### LABEL

`LABEL`指令可以为生成的镜像添加元数据标签信息，这些信息可以用来辅助过滤出特定镜像，格式：`KEY <key>=<value> <key>=<value> <key>=<value> … `，例如：

```shell
LABEL version="1.0.0-rc3"
LABEL author="yeasy@github" date="2020-06-03"
LABEL description="This text illustrates \
	that label-values can span multiple lines."
```

#### EXPOSE

`EXPOSE`，用于声明镜像内服务监听的端口，格式：`EXPOSE <port> [<port>/<protocol>...]`，例如：

```shell
EXPOSE 22 80 8443
```

注意，该指令只是起到声明作用，并不会自动完成端口映射。如果要映射端口出来，在启动容器时可以使用 -p 参数（Docker 主机会自动分配一个宿主机的临时端口）或 -p HOST_PORT:CONTAINER_PORT 参数（具体指定所映射的本地端口）。

#### ENV

`ENV`，指定环境变量，在镜像生成过程中会被后续 RUN 指令使用，在镜像启动的容器中也会存在，格式：`ENV <key> <value>`或`ENV <key>=<value>`，例如：

```shell
ENV APP_VERSION=1.0.0
ENV APP_HOME=/usr/local/app
ENV PATH $PATH:/usr/local/bin
```

指令指定的环境变量在运行时间可以被覆盖掉，如`docker run –env <key>=<value> built_image`。注意，当一条 ENV 指令中同时为多个环境变量赋值并且值也是从环境变量读取时，会为变量都赋值后再更新。如下面的指令，最终结果为：key1=value1 key2=value2：

```shell
ENV key1=value2
ENV key1=value1 key2=${key1}
```

#### ENTRYPOINT

`ENTRYPOINT`，指定镜像的默认入口命令，该入口命令会在启动容器时作为根命令执行，所有传入值作为该命的参数，支持两种格式：

```shell
# 1.`exec`调用执行.
ENTRYPOINT ["executable", "param1", "param2"]
# 2.`shell`中执行.
ENTRYPOINT command param1 param2
```

此时，CMD 指令指定值将作为根命令的参数。每个 Dockerfile 中只能有一个 ENTRYPOINT，当指定多个时，只有最后一个起效。在运行时，可以被 **--entrypoint** 参数覆盖掉。

#### VOLUME

`VOLUME`用于创建一个数据卷挂载点，格式：`VOLUME ["/data"]`。运行容器时，可以从本地主机或其他容器挂载数据卷，一般用来存放数据库和需要保持的数据等。

#### USER

`USER`，指定运行容器时的用户名或 UID，后续的 RUN 等指令也会使用指定的用户身份，格式：`USER daemon`。当服务不需要管理员权限时，可以通过该命令指定运行用户，并且可以在 Dockerfile 中创建所需要的用户。

#### WORKDIR

`WORKDIR`，为后续的 RUN、CMD、ENTRYPOINT 指令配置工作目录，格式：`WORKDIR /path/to/workdir`，可以使用多个 WORKDIR 指令，后续命令如果参数是相对路径，则会基于之前命令指定的路径，因此，为了避免出错，推荐 WORKDIR 指令中只使用绝对路径。

#### ONBUILD

`ONBUILD`，指定当基于所生成镜像创建子镜像时，自动执行的操作命令，格式：`ONBUILD [INSTRUCTION]`。

#### STOPSIGNAL

`STOPSIGNAL`，指定所创建镜像启动的容器接收退出的信号值：`STOPSIGNAL signal`。

#### HEALTHCHECK

`HEALTHCHECK`，配置所启动容器如何进行健康检查，基本格式有两种：

```shell
# 1.根据所执行命令返回值是否为`0`来判断.
HEALTHCHECK [OPTIONS] CMD command
# 2.禁止基础镜像中的健康检查.
HEALTHCHECK NONE
```

#### SHELL

`SHELL`，指定其他命令使用 shell 时的默认 shell 类型：`SHELL ["executable", "parameters"]`，默认值：`["/bin/sh", "-c"]`

### 操作指令

#### RUN

`RUN`，运行指定命令，格式：`RUN <command>`或`RUN ["executable", "param1", "param2"]`。注意后者指令会被解析为 JSON 数组，因此必须用双引号。前者默认将在 shell 终端中运行命令，即 **/bin/sh -c**，后者则使用 exec 执行，不会启动 shell 环境。

指定使用其他终端类型可以通过第二种方式实现，例如：**RUN [“/bin/bash”, “-c”, “echo hello”]** 。

每条 RUN 指令将在当前镜像基础上执行指定命令，并提交为新的镜像层。当命令较长时，可以使用 **\\** 来换行，例如：

```SHELL
RUN apt-get update \
	&& apt-get install -y libsnappy-dev zliblg-dev libbz2-dev \
	&& rm -rf /var/cache/apt \
	&& rm -rf /var/lib/apt/lists/*
```

#### CMD

`CMD`，该指令用来指定启动容器时默认执行的命令，支持三种格式：

```shell
# 1.推荐方式, 相当于执行`executable param1 param2`.
CMD ["executable", "param1", "param2"]
# 2.在默认的`Shell`中执行, 提供给需要交互的应用.
CMD command param1 param2
# 3.提供给`ENTRYPOINT`的默认参数.
CMD ["param1", "param2"]
```

每个 Dockerfile 只能有一条 CMD 命令。如果指定了多条命令，只有最后一条会被执行。如果用户启动容器时手动指定了运行的命令，则会覆盖掉 CMD 指定的命令。

#### ADD

`ADD`，添加内容到镜像，格式：`ADD <src> <dest>`，该命令将复制指定的 \<src\> 路径下的内容到容器中的 \<dest\> 路径下。其中，\<src\> 可以是 Dockerfile 所在目录的一个相对路径（文件或目录），也可以是一个 URL，还可以是一个 tar 文件（自动解压为目录）\<dest\> 可以是镜像内绝对路径，或者对于工作目录（WORKDIR）的相对路径。路径支持正则格式，例如：`ADD *.c /code/`。

#### COPY

`COPY`，复制内容到镜像，格式：`COPY <src> <dest>`。复制本地主机的 \<src\>（为 Dockerfile 所在目录的相对路径，文件或目录）下内容到镜像中的 \<dest\>。目标路径不存在时，会自动创建。

路径同样支持正则格式，COPY 与 ADD 指令功能类似，当使用本地目录为源目录时，推荐使用 COPY 。

## 创建镜像

编写完成 Dockerfile 之后，可以通过`docker [image] build`命令来创建镜像，基本格式为：

```shell
$ docker build [OPTIONS] PATH | URL | -
```

该命令将读取指定路径下，包括子目录的 Dockerfile，并将该路径下所有数据作为上下文发送给 Docker 服务端。Docker 服务端在校验 Dockerfile 格式通过后，逐条执行其中定义的指令，碰到 ADD、COPY 和 RUN 指令会生成一层新的镜像。最终如果创建镜像成功，会返回最终镜像的 ID 。如果上下文过大，会导致发送大量数据给服务端，延缓创建过程。因此，除非是生成镜像所必需文件，不然不要放在上下文路径下。如果使用非上下文路径下的 Dockerfile，可以通过`-f`选项来指定其路径。要指定生成镜像的标签信息，可以通过`-t`选项，该选项可以重复使用多次，为镜像一次添加多个名称，例如：

```shell
# 上下文路径为`/tmp/docker_builder/`, 并希望生成镜像标签为`builder/first_image:1.0.0`, 可以使用下面的命令:
$ docker build -t builder/first_image:1.0.0 /tmp/docker_builder/
```

### 命令选项

`docker [image] build`命令支持一系列的选项，可以调整创建镜像过程的行为，格式为：

```shell
# Build an image from a Dockerfile.
Usage:  docker build [OPTIONS] PATH | URL | -

Options:
      --add-host list              添加自定义的主机名到IP的映射
      --build-arg list             添加创建时的变量
      --cache-from stringSlice     使用指定镜像作为缓存源
      --cgroup-parent string       继承的上层 cgroup
      --compress                   使用 gzip 来压缩创建上下文数据
      --cpu-period int             分配的 CFS 调度器时长
      --cpu-quota int              CFS 调度器总份额
  -c, --cpu-shares int             CPU 权重
      --cpuset-cpus string         多 CPU 允许使用的 CPU
      --cpuset-mems string         多 CPU 允许使用的内存
      --disable-content-trust      不进行镜像校验, (default true)
  -f, --file string                Dockerfile 名称 (Default is 'PATH/Dockerfile')
      --force-rm                   总是删除中间过程的容器
      --help                       帮助文档
      --iidfile string             将镜像 ID 写入到文件
      --isolation string           容器的隔离机制
      --label list                 配置镜像的元数据
  -m, --memory bytes               限制使用内存量
      --memory-swap bytes          限制内存和缓存的总量
      --network string             指定 RUN 命令时的网络模式 (default "default")
      --no-cache                   创建镜像时不适用缓存
      --pull                       总是尝试获取镜像的最新版本
  -q, --quiet                      不打印创建过程中的日志信息
      --rm                         创建成功后自动删除中间过程容器 (default true)
      --security-opt stringSlice   指定安全相关的选项
      --shm-size bytes             `/dev/shm`的大小
  -t, --tag list                   指定镜像的标签列表 'name:tag'
      --target string              指定创建的目标阶段
      --ulimit ulimit              指定`ulimit`的配置
```

### 选择父镜像

大部分情况下，生成新的镜像都需要通过 FROM 指令来指定父镜像，父镜像是生成镜像的基础，会直接影响到所生成镜像的大小和功能。用户可以选择两种镜像作为父镜像，一种是所谓的基础镜像，即 baseimage，另外一种是普通的镜像（往往由第三方创建，基于基础镜像）。

基础镜像比较特殊，其 Dockerfile 中往往不存在 FROM 指令，或者基于 scratch 镜像（FROM scratch），这意味着其在整个镜像树中处于根的位置。普通镜像也可以作为父镜像来使用，包括常见的 busybox、debian、ubuntu 等。

### 使用 .dockerignore 文件

可以通过`.dockerignore`文件（每一行添加一条匹配模式）来让 Docker 忽略匹配路径或文件。在创建镜像时候不将无关数据发送到服务端，例如，下面的例子中包括了 6 行忽略的模式：

```shell
# .dockerignore 文件中可以定义忽略模式.
# .dockerignore 文件中模式语法支持 Golang 风格的路径正则格式.
# "*" 表示任意多个字符.
# "?" 代表单个字符.
# "!" 表示不匹配, 即不忽略指定的路径或文件.
*/temp*
*/*/temp*
tmp?
~*
Dockerfile
!README.md
```

## 最佳实践

所谓的最佳实践，就是从需求出发，来定制适合自己、高效方便的镜像。首先，要尽量吃透每个指令的含义和执行效果，多编写一些简单的例子进行测试，弄清楚了再攥写正式的 Dockerfile 。此外，Docker Hub 官方仓库中提供了大量的优秀镜像和对应的 Dockerfile，可以通过阅读它们来学习如何攥写高效的 Dockerfile 。

- **精简镜像用途**：尽量让每个镜像的用途都比较集中单一，避免构造大而复杂、多功能 的镜像。
- **选用合适的基础镜像**：容器的核心是应用。选择过大的父镜像，会造成最终生成应用镜像的臃肿，推荐选用瘦身过的应用镜像，或者较为小巧的系统镜像。
- **提供注释和维护者信息**：Dockerfile 也是一种代码，需要考虑方便后续的扩展和他人的使用。
- **正确使用版本号**：使用明确的版本号信息，如 1.0，2.0 等，而非依赖于默认的 latest 。通过版本号可以避免环境不一致导致的问题。
- **减少镜像层数**：如果希望所生成镜像的层数尽量少，则要尽量合并 RUN、ADD 和 COPY 指令。通常情况下，多个 RUN 指令可以合并为一条 RUN 指令。
- **恰当使用多步创建**：通过多步创建，可以将编译和运行等过程分开，保证最终生成的镜像只包括应用所需要的最小化环境。
- **使用 .dockeignore 文件**：使用它可以标记在执行 docker build 时忽略的路径和文件，避免发送不必要的数据内容，从而加快整个镜像的创建过程。
- **及时删除临时文件和缓存文件**：特别是在执行 apt-get 指令后，**/var/cache/apt** 下面会缓存了一些安装包。
- **提高生成速度**：如合理使用 cache，减少内容目录下的文件，或使用 .dockerignore 文件指定等。
- **调整合理的指令顺序**：在开启 cache 的情况下，内容不变的指令尽量放在前面，这样可以尽量复用。
- **减少外部源的干扰**：如果确实要从外部引入数据，需要指定持久的地址，并带版本信息等，让他人可以复用而不出错。



# 操作系统

目前常用的 Linux 发行版主要包括 Debian/Ubuntu 系列和 CentOS/Fedora 系列。使用 Docker，只需要一个命令就能快速获取一个 Linux 发行版镜像，这是以往各种虚拟化技术都难以实现的。这些镜像一般都很精简，但是可以支持完整 Linux 系统的大部分功能。

## BusyBox

BusyBox 是一个集成了一百多个常用 Linux 命令（如 cat、echo、grep、mount 等）的精简工具箱，它只有不到 2Mb 大小。

在 Docker Hub 中搜索 busybox 相关的镜像，如下所示：

```shell
test@VM-0-9-ubuntu:~$ docker search busybox
NAME                      DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
busybox                   Busybox base image.                             1909                [OK]                
progrium/busybox                                                          71                                      [OK]
radial/busyboxplus        Full-chain, Internet enabled, busybox made f…   30                                      [OK]                                  
......                                    
trollin/busybox                                                           0                                       
ggtools/busybox-ubuntu    Busybox ubuntu version with extra goodies       0                                       [OK]
```

从上结果可以看到，最受欢迎的镜像同时带有 **OFFICIAL** 标记，说明它是官方镜像。可以使用`docker pull`命令下载镜像 **busybox:latest**：

```shell
test@VM-0-9-ubuntu:~$ docker pull busybox:latest
latest: Pulling from library/busybox
76df9210b28c: Pull complete 
Digest: sha256:95cf004f559831017cdf4628aaf1bb30133677be8702a8c5f2994629f637a209
Status: Downloaded newer image for busybox:latest
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY                     TAG                 IMAGE ID            CREATED             SIZE
busybox                        latest              1c35c4412082        17 hours ago        1.22MB
```

----

运行一个 busybox，启动一个 busybox 镜像，并在容器中执行 grep 命令：

```shell
test@VM-0-9-ubuntu:~$ docker run -it busybox
/ # grep
BusyBox v1.31.1 (2020-06-02 02:36:36 UTC) multi-call binary.

Usage: grep [-HhnlLoqvsriwFE] [-m N] [-A/B/C N] PATTERN/-e PATTERN.../-f FILE [FILE]...

Search for PATTERN in FILEs (or stdin)

        -H      Add 'filename:' prefix
        -h      Do not add 'filename:' prefix
        -n      Add 'line_no:' prefix
        -l      Show only names of files that match
        -L      Show only names of files that don't match
        -c      Show only count of matching lines
		...
        -B N    Print N lines of leading context
        -C N    Same as '-A N -B N'
        -e PTRN Pattern to match
        -f FILE Read pattern from file
```

busybox 镜像虽然小巧，但包含了大量常见的 Linux 命令，读者可以用它快速熟悉 Linux 命令。

## Alpine

Alpine 操作系统是一个面向安全的轻型 Linux 发行版，关注安全，性能和资源效能。Alpine 采用了 musl libc 和 BusyBox 以减少系统的体积和运行时间资源消耗，比 BusyBox 功能上更完善。Apline Docker 镜像继承了 Apline Linux 发行版的这些优势，这可以带来多个优势，如镜像下载速度加快、镜像安全性提高、主机之间的切换更方便、占用更少磁盘空间等。

## Debian/Ubuntu

Debian 和 Ubuntu 都是目前较为流行的 Debian 系的服务器操作 系统，十分适合研发场景。Docker Hub 上提供了它们的官方镜像，国内各大容器云服务器都提供了完整的支持。

### Debian 系统简介及官方镜像使用

Debian 是基于 GPL 授权的开源操作系统，是目前个人电脑与服务器中最受欢迎的开源操作系统之一，由 Debian 计划组织维护。

```shell
test@VM-0-9-ubuntu:~$ docker search debian
NAME                 DESCRIPTION                                     STARS         OFFICIAL        AUTOMATED
ubuntu               Ubuntu is a Debian-based Linux operating sys…   10960          [OK]      
debian               Debian is a Linux distribution that's compos…   3501           [OK]                
arm32v7/debian       Debian is a Linux distribution that's compos…   66                                      
itscaro/debian-ssh   debian:jessie                                   28                             OK]
arm64v8/debian       Debian is a Linux distribution that's compos…   23                                      
...                     
```

### Ubuntu 系统简介及官方镜像使用

Ubuntu 是以桌面应用为主的 GUN/Linux 开源操作系统，每两年推出一个长期支持，Long Term Support，LTS 版本：

```shell
test@VM-0-9-ubuntu:~$ docker search --filter=stars=100 ubuntu
NAME                             DESCRIPTION                                     STARS           OFFICIAL       AUTOMATED
ubuntu                           Ubuntu is a Debian-based Linux operating sys…   10960           [OK]                
dorowu/ubuntu-desktop-lxde-vnc   Docker image to provide HTML5 VNC interface …   433                            [OK]
rastasheep/ubuntu-sshd           Dockerized SSH service, built on top of offi…   244                            [OK]
consol/ubuntu-xfce-vnc           Ubuntu container with "headless" VNC session…   219                            [OK]
ubuntu-upstart                   Upstart is an event-based replacement for th…   109             [OK]                
```

```shell
# 1.下载和安装`ubuntu:16.04`.
test@VM-0-9-ubuntu:~$ docker pull ubuntu:16.04
16.04: Pulling from library/ubuntu
e92ed755c008: Pull complete 
b9fd7cb1ff8f: Pull complete 
ee690f2d57a1: Pull complete 
53e3366ec435: Pull complete 
Digest: sha256:db6697a61d5679b7ca69dbde3dad6be0d17064d5b6b0e9f7be8d456ebb337209
Status: Downloaded newer image for ubuntu:16.04
# 2.运行`ubuntu:16.04`, 新建一个容器.
test@VM-0-9-ubuntu:~$ docker run -it ubuntu:16.04
# 3.查看容器的版本.
root@95cda544868c:/# cat /etc/lsb-release 
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=16.04
DISTRIB_CODENAME=xenial
DISTRIB_DESCRIPTION="Ubuntu 16.04.6 LTS"
# 4.`apt-get update`
root@95cda544868c:/# apt-get update
Get:1 http://archive.ubuntu.com/ubuntu xenial InRelease [247 kB]
Get:2 http://security.ubuntu.com/ubuntu xenial-security InRelease [109 kB]
Get:3 http://archive.ubuntu.com/ubuntu xenial-updates InRelease [109 kB]           
Get:4 http://archive.ubuntu.com/ubuntu xenial-backports InRelease [107 kB]        
Get:5 http://archive.ubuntu.com/ubuntu xenial/main amd64 Packages [1558 kB]                                                 ...                                                                        
Fetched 16.5 MB in 5min 7s (53.6 kB/s)   
Reading package lists... Done
# 5.查看`curl`, 命令不存在.
root@95cda544868c:/# curl --help               
bash: curl: command not found
# 6.安装`curl`命令.
root@95cda544868c:/# apt-get install curl
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following additional packages
.......
After this operation, 19.0 MB of additional disk space will be used.
Do you want to continue? [Y/n] y
Get:1 http://archive.ubuntu.com/ubuntu xenial/main amd64 libffi6 amd64 3.2.1-4 [17.8 kB]
.......
127 added, 0 removed; done.
Running hooks in /etc/ca-certificates/update.d...
done.
# 7.`curl`命令安装成功, 打印帮助文档.
root@95cda544868c:/# curl --help
Usage: curl [options...] <url>
Options: (H) means HTTP/HTTPS only, (F) means FTP only
     --anyauth       Pick "any" authentication method (H)
 -a, --append        Append to target file when uploading (F/SFTP)
     --basic         Use HTTP Basic Aut
     ......
```

## CentOS/Fedora

### CentOS 系统简介及官方镜像使用

CentOS 是基于 Redhat 的 Linux 发行版。CentOS 是目前企业级服务器的常用操作系统。Fedora 则主要面向个人桌面用户。CentOS，全称为 Community Enterprise Operatng System，社区企业操作系统，是基于 Red Hat Enterprise Linux 源代码编译而成。由于 CentOS 与 RedHat Linux 源于相同的代码基础，所以很多成本敏感且需要高稳定性的公司就使用 CentOS 来替代商业版的 RedHat 。

```shell
test@VM-0-9-ubuntu:~$ docker search --filter=stars=100 centos
NAME                      DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
centos                    The official build of CentOS.                   6029                [OK]                
ansible/centos7-ansible   Ansible on Centos7                              129                                     [OK]
consol/centos-xfce-vnc    Centos container with "headless" VNC session…   115                                     [OK]
jdeathe/centos-ssh        OpenSSH / Supervisor / EPEL/IUS/SCL Repos - …   114                                     [OK]
```

### Fedora 系统简介与官方镜像使用

Fedora 是由 Fedora Project 社区开发，Red Hat 公司赞助的 Linux 发行版。它的目标是创建一套新颖、多功能并且自由和开源的操作系统。

```shell
test@VM-0-9-ubuntu:~$ docker search --filter=stars=100 fedora
NAME                DESCRIPTION                        STARS               OFFICIAL            AUTOMATED
fedora              Official Docker builds of Fedora   881                 [OK]                
```



# 为镜像添加 SSH 服务

很多时候，系统管理员都习惯通过 SSH 服务来远程登录和关联服务器，但是 Docker 的很多镜像是不带 SSH 服务的，可以通过`docker commit`命令创建或基于`Dockerfile`创建。

## 基于 commit 命令创建

Docker 提供了`docker commit`命令，支持用户提交自己对定制容器的修改，并生成新的镜像。命令格式：`docker commit CONTAINER [REPOSITORY[:ATG]]`，下面展示，如何使用 docker commit 命令为 ubuntu:18.04 镜像添加 SSH 服务。

### 准备工作

首先，获取 ubuntu:18.04 镜像，并创建一个容器：

```shell
test@VM-0-9-ubuntu:~$ docker pull ubuntu:18.04
18.04: Pulling from library/ubuntu
Digest: sha256:3235326357dfb65f1781dbc4df3b834546d8bf914e82cce58e6e6b676e23ce8f
Status: Image is up to date for ubuntu:18.04
docker.io/library/ubuntu:18.04
test@VM-0-9-ubuntu:~$ docker run -it ubuntu:18.04 /bin/bash
root@7d4c8140e968:/# 
```

### 配置软件源

检查软件源，并使用 **apt-get update** 命令来更新软件源信息：

```shell
# 1.使用官方源:
root@7d4c8140e968:/# apt-get update
Get:1 http://archive.ubuntu.com/ubuntu bionic InRelease [242 kB]
Get:2 http://archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]             
Get:3 http://archive.ubuntu.com/ubuntu bionic-backports InRelease [74.6 kB]                     
......                
Fetched 18.0 MB in 9s (1922 kB/s)                                                                                             Reading package lists... Done
# 2.如果默认的官方源速度慢的话, 也可以替换成国内`163`, 阿里等镜像的源, 以`163源`为例, 在容器内创建`/etc/apt/sources.list.d/163.list`文件:
cd /etc/apt/sources.list.d && touch 163.list
echo "deb http://mirrors.163.com/ubuntu/ bionic main restricted universe multiverse" >> 163.list
echo "deb http://mirrors.163.com/ubuntu/ bionic-security main restricted universe multiverse" >> 163.list
echo "deb http://mirrors.163.com/ubuntu/ bionic-updates main restricted universe multiverse" >> 163.list
echo "deb http://mirrors.163.com/ubuntu/ bionic-proposed main restricted universe multiverse" >> 163.list
echo "deb http://mirrors.163.com/ubuntu/ bionic-backports main restricted universe multiverse" >> 163.list
echo "deb-src http://mirrors.163.com/ubuntu/ bionic main restricted universe multiverse" >> 163.list
echo "deb-src http://mirrors.163.com/ubuntu/ bionic-security main restricted universe multiverse" >> 163.list
echo "deb-src http://mirrors.163.com/ubuntu/ bionic-updates main restricted universe multiverse" >> 163.list
echo "deb-src http://mirrors.163.com/ubuntu/ bionic-proposed main restricted universe multiverse" >> 163.list
echo "deb-src http://mirrors.163.com/ubuntu/ bionic-backports main restricted universe multiverse" >> 163.list
apt-get update
```

### 安装和配置 SSH 服务

1. 更新软件包缓存后可以安装 SSH 服务了，选择主流的 openssh-server 作为服务端：

```shell
root@7d4c8140e968:~# apt-get install openssh-server
...
Running hooks in /etc/ca-certificates/update.d...
done.
```

2. 如果需要正常启动 SSH 服务，则目录 **/var/run/sshd** 必须存在：

```shell
root@7d4c8140e968:/# mkdir -p /var/run/sshd
root@7d4c8140e968:/# /usr/sbin/sshd -D &
[1] 3882
```

3. 此时查看容器的 22 端口（ SSH 服务默认监听的端口），可见此端口已经处于监听状态：

```shell
# apt-get install net-tools
root@7d4c8140e968:/# netstat -tunlp
Active Internet connections (only servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State       PID/Program name    
tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN      3882/sshd           
tcp6       0      0 :::22                   :::*                    LISTEN      3882/sshd 
```

4. 修改 SSH 服务的安全登录配置，取消 pam 登录限制：

```shell
root@7d4c8140e968:~# sed -ri "s/session required pam_loginuid.so/#session required pam_loginuid.so/g" /etc/pam.d/sshd
```

5. 在 root 用户目录下创建 .ssh 目录，并复制需要登录的公钥信息（一般为本地主机用户目录下的 **.ssh/id_rsa.pub** 文件，可由 **ssh-keygen -t rsa** 命令生成）到 **authorized_keys** 文件中。

```shell
# 由`ssh-keygen -t rsa`命令生成`$HOME/.ssh/id_rsa.pub`:
test@VM-0-9-ubuntu:~$ ssh-keygen -t rsa 
Generating public/private rsa key pair.
Enter file in which to save the key (/home/test/.ssh/id_rsa): 
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /home/test/.ssh/id_rsa.
Your public key has been saved in /home/test/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:jKAUpmP1gIJFquX6cy+nTpVQUNBVbmq0kL9Ag9OGc/4 test@VM-0-9-ubuntu
The key's randomart image is:
+---[RSA 2048]----+
|.o*oo=o....      |
|o=..o=.. .       |
|=.o B.O . o      |
|o= . X O +       |
|. o   * S        |
| .   . + .       |
|.   .   E        |
| ...o .          |
|  .+o=.          |
+----[SHA256]-----+
```

```shell
# 复制主机中`$HOME/.ssh/id_rsa.pub`文件内容到`/root/.ssh/authorized_keys`文件中.
# 容器中运行:
mkdir -p /root/.ssh
# 宿主机中运行:
docker cp $HOME/.ssh/id_rsa.pub 7d4c8140e968:/root/.ssh/authorized_keys
```

6. 修改配置

```shell
# apt-get install vim
root@7d4c8140e968:~# vim /etc/ssh/sshd_config

PubkeyAuthentication yes                   # 启用公钥私钥配对认证方式.
AuthorizedKeysFile .ssh/authorized_keys    # 公钥文件路径.
PermitRootLogin yes                        # `root`能使用`ssh`登录.
```

7. 修改密码

```shell
root@7d4c8140e968:/# passwd
Enter new UNIX password: 
Retype new UNIX password: 
```

8. 创建自动启动 SSH 服务的可执行文件 **run.sh**，并添加可执行权限：

```shell
root@7d4c8140e968:~# cd /
root@7d4c8140e968:/# vim run.sh
#!/bin/bash
/usr/sbin/sshd -D
root@7d4c8140e968:/# chmod +x run.sh
```

9. 最后，退出容器：

```SHELL
root@7d4c8140e968:/# exit
exit
```

### 保存镜像

将所退出的容器用`docker commit`命令保存为一个新的 **sshd_ubuntu:18.04** 镜像：

```shell
test@VM-0-9-ubuntu:~$ docker commit 7d4c8140e968 sshd_ubuntu:18.04
sha256:9a69cb3bf7fbb0a8c77cbae0314bedf7216785ee6eb2b2e14821614f4f088b41
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
sshd_ubuntu         18.04               4c15cf5ae87c        7 seconds ago       245MB
ubuntu              18.04               c3c304cb4f22        6 weeks ago         64.2MB
```

### 使用镜像

启动容器，并添加端口映射 *26 --> 22*，其中 26 是宿主主机的端口，22 是容器的 SSH 服务监听端口：

```shell
# sudo systemctl enable docker.service
# sudo /lib/systemd/systemd-sysv-install enable docker
# docker run -p 26:22 -d sshd_ubuntu:18.04 /run.sh --restart=always
# docker update --restart=always <CONTAINER ID>
test@VM-0-9-ubuntu:~/.ssh$ docker run -p 26:22 -d sshd_ubuntu:18.04 /run.sh --restart=always
a6ee5cf86324912b993a17146a025f0ef571efdb213f25e34e879d2b53cf5527
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID      IMAGE               COMMAND     CREATED           STATUS           PORTS                NAMES
3564c7265f1c      sshd_ubuntu:18.04   "/run.sh"   22 seconds ago    Up 21 seconds    0.0.0.0:26->22/tcp   intelligent_gates
```

```shell
# 未测试
docker run -p 1000:22 -d sshd-ubuntu:20.04 --memory 24G --memory-swap 32G /run.sh --restart=always

# 2023-05-09 已测试
docker run -p 1000:22 -d --name probe-guest sshd-ubuntu:20.04 /run.sh --restart=always
```



在宿主主机或其他主机上，可以通过 SSH 访问 26 端口来登录容器：

```shell
ssh root@49.235.24.68 -p 26
```

## 基于 Dockerfile 创建

### 创建工作目录

首先，创建一个`sshd_ubuntu`工作路径：

```shell
test@VM-0-9-ubuntu:~$ mkdir sshd_ubuntu
```

在其中，创建`Dockerfile`和`run.sh`文件：

```shell
test@VM-0-9-ubuntu:~/sshd_ubuntu$ cd sshd_ubuntu
test@VM-0-9-ubuntu:~/sshd_ubuntu$ touch Dockerfile run.sh
test@VM-0-9-ubuntu:~/sshd_ubuntu$ ll
total 8
drwxrwxr-x 2 test test 4096 Jun  8 09:07 ./
drwxrwxr-x 3 test test 4096 Jun  8 09:05 ../
-rw-rw-r-- 1 test test    0 Jun  8 09:07 Dockerfile
-rw-rw-r-- 1 test test    0 Jun  8 09:07 run.sh
```

### 编写 run.sh 脚本和 authorized_keys 文件

```shell
# run.sh
test@VM-0-9-ubuntu:~/sshd_ubuntu$ vim run.sh
#!/bin/bash
/usr/sbin/sshd -D
```

```shell
# authorized_keys
# 由`ssh-keygen -t rsa`命令生成`$HOME/.ssh/id_rsa.pub`:
test@VM-0-9-ubuntu:~$ ssh-keygen -t rsa 
Generating public/private rsa key pair.
Enter file in which to save the key (/home/test/.ssh/id_rsa): 
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /home/test/.ssh/id_rsa.
Your public key has been saved in /home/test/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:jKAUpmP1gIJFquX6cy+nTpVQUNBVbmq0kL9Ag9OGc/4 test@VM-0-9-ubuntu
The key's randomart image is:
+---[RSA 2048]----+
|.o*oo=o....      |
|o=..o=.. .       |
|=.o B.O . o      |
|o= . X O +       |
|. o   * S        |
| .   . + .       |
|.   .   E        |
| ...o .          |
|  .+o=.          |
+----[SHA256]-----+
test@VM-0-9-ubuntu:~/sshd_ubuntu$ cat ~/.ssh/id_rsa.pub > authorized_keys
```

### 编写 Dockerfile 文件

下面是 Dockerfile 的内容及各部分的注释，与上一节的 docker commit 命令创建镜像过程基本一致：

```shell
# 设置继承镜像.
FROM ubuntu:18.04

# 提供一些作者的信息.
MAINTAINER Albert丶XN (albertxn@126.com)

# 下面开始运行命令, 此处更改 ubuntu 的源为国内 163 的源.
RUN cd /etc/apt/
RUN echo "deb http://mirrors.163.com/ubuntu/ bionic main restricted universe multiverse" > sources.list
RUN echo "deb http://mirrors.163.com/ubuntu/ bionic-security main restricted universe multiverse" >> sources.list
RUN echo "deb http://mirrors.163.com/ubuntu/ bionic-updates main restricted universe multiverse" >> sources.list
RUN echo "deb http://mirrors.163.com/ubuntu/ bionic-proposed main restricted universe multiverse" >> sources.list
RUN echo "deb http://mirrors.163.com/ubuntu/ bionic-backports main restricted universe multiverse" >> sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ bionic main restricted universe multiverse" >> sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ bionic-security main restricted universe multiverse" >> sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ bionic-updates main restricted universe multiverse" >> sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ bionic-proposed main restricted universe multiverse" >> sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ bionic-backports main restricted universe multiverse" >> sources.list
RUN apt-get update

# 安装`SSH`服务.
RUN apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd
RUN mkdir -p /root/.ssh

# 取消`pam`限制.
RUN sed -ri "s/session required pam_loginuid.so/#session required pam_loginuid.so/g" /etc/pam.d/sshd

# 配置`SSH`远程登录
RUN sed -ie 's/#PermitRootLogin.*/PermitRootLogin yes/g' /etc/ssh/sshd_config
RUN sed -ie 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/g' /etc/ssh/sshd_config
RUN sed 's/#AuthorizedKeysFile/AuthorizedKeysFile/g' /etc/ssh/sshd_config

# 复制配置文件到相应位置, 并赋予脚本可执行权限.
ADD authorized_keys /root/.ssh/authorized_keys
ADD run.sh /run.sh
RUN chmod 755 /run.sh

# 开放端口
EXPOSE 22

# 设置自启动命令
CMD ["/run.sh"]
```

### 创建镜像

在 **sshd_ubuntu** 目录下，使用`docker build`命令来创建镜像：

```shell
# `.`表示使用当前目录中的`Dockerfile`.
test@VM-0-9-ubuntu:~/sshd_ubuntu$ docker build -t sshd_ubuntu:18.04 . 
Sending build context to Docker daemon  5.632kB
Step 1/23 : FROM ubuntu:18.04
 ---> c3c304cb4f22
Step 2/23 : MAINTAINER Albert丶XN (albertxn@126.com)
 ---> Running in a4fdb1cf6aef
Removing intermediate container a4fdb1cf6aef
 ---> b9976500b78e
Step 3/23 : RUN cd /etc/apt/
 ---> Running in 004ef2e573c2
 ...
Step 23/23 : CMD ["/run.sh"]
 ---> Running in 19ba7d4241b9
Removing intermediate container 19ba7d4241b9
 ---> a70ecf724414
Successfully built a70ecf724414
Successfully tagged sshd_ubuntu:18.04
```

```shell
test@VM-0-9-ubuntu:~/Dockrfile/sshd_ubuntu$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
sshd_ubuntu         18.04               a70ecf724414        51 seconds ago      208MB
ubuntu              18.04               c3c304cb4f22        6 weeks ago         64.2MB
```

如果使用 Dockerfile 创建自定义镜像，那么需要注意的是 Docker 会自动删除中间临时创建的层，还需注意每一步的操作和编写的 Dockerfile 中命令的对应关系。命令执行完毕后，如果显示 **Successfully built …..** 字样，则说明镜像创建成功，如上代码所示。

### 测试镜像，运行容器

直接启动镜像，映射容器的 22 端口到本地的 26 端口：

```shell
test@VM-0-9-ubuntu:~$ docker run -p 26:22 -d sshd_ubuntu:18.04 /run.sh --restart=always
af4d310f87a6747ccbe53524e16c56a8e260912adb4fe97b222c1c05728b3b4f
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE              COMMAND                 CREATED        STATUS        PORTS                NAMES
af4d310f87a6    sshd_ubuntu:18.04  "/run.sh --restart=a…"  3 seconds ago  Up 2 seconds  0.0.0.0:26->22/tcp   gracious_lumiere
```

```shell
# 在宿主主机或其他主机上, 可以通过 SSH 访问 26 端口来登录容器:
test@VM-0-9-ubuntu:~$ ssh root@49.235.24.68 -p 26
Welcome to Ubuntu 18.04.4 LTS (GNU/Linux 4.4.0-130-generic x86_64)
 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage
This system has been minimized by removing packages and content that are
not required on a system that users do not log into.
To restore this content, you can run the 'unminimize' command.
The programs included with the Ubuntu system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.
Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law.
root@af4d310f87a6:~#
```



# Web 服务与应用

待学习…



# 数据库应用

目前主流数据库主要包括关系型`SQL`和非关系型`NoSQL`两种。关系型数据库是建立在关系模型基础上的数据库，代表实现有 **MySQL**、**Oracle**、**PostGreSQL**、**MariaDB** 和 **SQLServer** 等。非关系型数据库是新兴的数据库技术，它放弃了传统关系型数据库的部分强一致性限制，带来性能上的提升，使其更适合于需要大规模并行处理的场景。非关系型数据库是关系型数据库的良好补充，代表产品有 **MongoDB**、**Redis** 等。

## MySQL

MySQL 是全球最流行的开源关系型数据库之一，由于其具有高性能、成熟可靠、高适应性、易用性而得到广泛的应用。

### 安装和启动一个 MySQL 数据库

```shell
# 用户可以使用官方镜像快速启动一个 MySQL 实例:
# https://hub.docker.com/_/mysql?tab=tags
test@VM-0-9-ubuntu:~$ docker run --name myMySQL -e MYSQL_ROOT_PASSWORD=fd12306 -d mysql:5.7
Unable to find image 'mysql:5.7' locally
5.7: Pulling from library/mysql
afb6ec6fdc1c: Pull complete 
....
Digest: sha256:d16d9ef7a4ecb29efcd1ba46d5a82bda3c28bd18c0f1e3b86ba54816211e1ac4
Status: Downloaded newer image for mysql:5.7
3ce89d14ed4b9f1f02d6e14339f21e0fbf604581eecd35a3415fb4bafb0ad3c7
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mysql               5.7                 a4fdfd462add        2 weeks ago         448MB
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID   IMAGE        COMMAND                  CREATED             STATUS              PORTS                 NAMES
3ce89d14ed4b   mysql:5.7    "docker-entrypoint.s…"   48 seconds ago      Up 47 seconds       3306/tcp, 33060/tcp   myMySQL
```

```shell
# 先通过`docker pull`获得镜像, 再新建一个容器:
test@VM-0-9-ubuntu:~$ docker pull mysql:5.7
5.7: Pulling from library/mysql
afb6ec6fdc1c: Pull complete 
....
Digest: sha256:d16d9ef7a4ecb29efcd1ba46d5a82bda3c28bd18c0f1e3b86ba54816211e1ac4
Status: Downloaded newer image for mysql:5.7
docker.io/library/mysql:5.7
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mysql               5.7                 a4fdfd462add        2 weeks ago         448MB
test@VM-0-9-ubuntu:~$ docker run -itd --name myMySQL -p 3306:3306 -e MYSQL_ROOT_PASSWORD=fd12306 mysql:5.7
2ea42370f864d83ac03779b7b22be483b947086ea11a143c1f7f80a6880cb825
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID  IMAGE      COMMAND                 CREATED         STATUS          PORTS                               NAMES
2ea42370f864  mysql:5.7  "docker-entrypoint.s…"  48 seconds ago  Up 47 seconds   0.0.0.0:3306->3306/tcp, 33060/tcp   myMySQL
```

```shell
# 修改默认端口, 并自定义挂载一些路径至容器:
# `-p 3300:3306`, 将容器的`3306`端口映射到主机的`3300`端口.
# `-v $PWD/conf:/etc/mysql/conf.d`, 将主机当前目录下的`conf/my.cnf`挂载到容器的`/etc/mysql/my.cnf`.
# `-v $PWD/logs:/logs`, 将主机当前目录下的`logs`目录挂载到容器的`/logs`.
# `-v $PWD/data:/var/lib/mysql`, 将主机当前目录下的`data`目录挂载到容器的`/var/lib/mysql`.
# `-e MYSQL_ROOT_PASSWORD=123456`, 初始化`root`用户的密码.
test@VM-0-9-ubuntu:~$ docker run -p 3300:3306 --name mymysql -v $PWD/conf:/etc/mysql/conf.d -v $PWD/logs:/logs -v $PWD/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=123456 -d mysql
```

### 进入一个 MySQL 容器

```shell
# 进入一个创建好的 MySQL 容器并登录数据库, 如果未创建, 可以通过上一步进行从操作.
# docker exec -it 容器ID bash
test@VM-0-9-ubuntu:~$ docker exec -it 2ea42370f864 bash
root@2ea42370f864:/# 
root@2ea42370f864:/# mysql -u root -p
Enter password: 
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 3
Server version: 5.7.30 MySQL Community Server (GPL)
Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.
mysql> 
```

```shell
# 修改用户密码.
mysql> ALTER USER 'root'@'localhost' IDENTIFIED BY '123456';
# 添加远程登录用户.
mysql> CREATE USER 'test'@'%' IDENTIFIED WITH mysql_native_password BY '123456';
mysql> GRANT ALL PRIVILEGES ON *.* TO 'test'@'%';
```

```shell
# ERROR, Client does not support authentication protocol requested  by server.
# 1.进入容器:
$ docker exec -it 容器ID /bin/bash
# 2.进入`MySQL`:
$ mysql -uroot -p
# 3.授权:
mysql> GRANT ALL ON *.* TO 'root'@'%';
# 4.刷新权限:
mysql> flush privileges;
# 5.更新加密规则:
mysql> ALTER USER 'root'@'localhost' IDENTIFIED BY 'password' PASSWORD EXPIRE NEVER;
# 6.更新`root`用户密码:
mysql> ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY '123456';
# 7.刷新权限:
mysql> flush privileges;
```

### MySQL 中导入和导出数据

```shell
# Backup
docker exec CONTAINER /usr/bin/mysqldump -u root --password=root DATABASE > backup.sql
# Restore
docker exec -i CONTAINER /usr/bin/mysql -u root --password=root DATABASE < backup.sql
```

```shell
# 已有一个`MySQL`容器`2ea42370f864`, 导入一个数据库:
# 如数据库未创建则需要先进行创建:
# 	1.`docker exec -it 2ea42370f864 /bin/bash`
#	2.`mysql -u root -p`
#	3.`CREATE DATABASE KBASE;`.
$ docker exec -i 2ea42370f864 /usr/bin/mysql -u root --password=fd12306 KBASE < /home/test/kbase.sql
```

## MongoDB

MongoDB 是一款可扩展、高性能的开源文档数据库，是当今最流行的 NoSQL 数据库之一。它采用 C++ 开发，支持复杂的数据类型和强大的查询语言，提供了关系数据库的绝大部分功能。由于其高性能、易部署、易使用等特点，MongoDB 已经在很多领域都得到了广泛的应用。

### 安装和启动一个 MongoDB 数据库

```shell
# 用户可以使用官方镜像快速启动一个 MongoDB 实例:
# https://hub.docker.com/_/mongo/?tab=tags
test@VM-0-9-ubuntu:~$ docker run --name mymongo -d mongo:4.4.0-rc8
Unable to find image 'mongo:4.4.0-rc8' locally
4.4.0-rc8: Pulling from library/mongo
...
Digest: sha256:a982894be1d5cc9479be27ec17a0f6d75b00c2bcb1c0b8ae7bf287ed695e9e66
Status: Downloaded newer image for mongo:4.4.0-rc8
382c201df5670e15fd3ce913c0adc80cebb2078e149d1ee6f427ac356044ab26
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID    IMAGE            COMMAND                 CREATED             STATUS              PORTS            NAMES
382c201df567    mongo:4.4.0-rc8  "docker-entrypoint.s…"  About a minute ago  Up About a minute   27017/tcp        mymongo
```

```shell
# 先通过`docker pull`获得镜像, 再新建一个容器:
test@VM-0-9-ubuntu:~$ docker pull mongo:4.4.0-rc8
5.7: Pulling from library/mysql
afb6ec6fdc1c: Pull complete 
....
Digest: sha256:a982894be1d5cc9479be27ec17a0f6d75b00c2bcb1c0b8ae7bf287ed695e9e66
Status: Downloaded newer image for mongo:4.4.0-rc8
docker.io/library/mysql:4.4.0-rc8
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mongo               4.4.0-rc8           29c751d0b555        3 days ago          492MB
test@VM-0-9-ubuntu:~$ docker run -p 27000:27017 -v $PWD/db:/data/db -d mongo:4.4.0-rc8  
3f5f58cd1c9fb35eb471f0982903b9924342deafc473e34f6f6e19156039751f
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID  IMAGE             COMMAND        CREATED         STATUS         PORTS                      NAMES
3f5f58cd1c9f  mongo:4.4.0-rc8   "docker-en…"   4 seconds ago   Up 3 seconds   0.0.0.0:27000->27017/tcp   intelligent_panini
```

### 进入一个 MongoDB 数据库

```shell
# docker exec -it 容器ID /bin/bash
test@VM-0-9-ubuntu:~$ docker exec -it 3f5f58cd1c9f /bin/bash
root@3f5f58cd1c9f:/# 
root@3f5f58cd1c9f:/# mongo
MongoDB shell version v4.4.0-rc8
connecting to: mongodb://127.0.0.1:27017/?compressors=disabled&gssapiServiceName=mongodb
Implicit session: session { "id" : UUID("39c00bc6-76f6-4c55-a3e2-d6a88fa33e32") }
MongoDB server version: 4.4.0-rc8
...
> 
> show dbs
admin   0.000GB
config  0.000GB
local   0.000GB
> 
> db.stats()
{
        "db" : "test",
        "collections" : 0,
        "views" : 0,
        "objects" : 0,
        "avgObjSize" : 0,
        "dataSize" : 0,
        "storageSize" : 0,
        "totalSize" : 0,
        "indexes" : 0,
        "indexSize" : 0,
        "scaleFactor" : 1,
        "fileSize" : 0,
        "fsUsedSize" : 0,
        "fsTotalSize" : 0,
        "ok" : 1
}
> 
> exit
bye
```

### 通过端口访问 MongoDB 数据库

```shell
test@VM-0-9-ubuntu:~$ docker run -it --link 3f5f58cd1c9f:db alpine:3.12.0 /bin/sh
/ # 
/ # ls
bin    dev    etc    home   lib    media  mnt    opt    proc   root   run    sbin   srv    sys    tmp    usr    var
/ # ping db
PING db (172.18.0.3): 56 data bytes
64 bytes from 172.18.0.3: seq=0 ttl=64 time=0.124 ms
64 bytes from 172.18.0.3: seq=1 ttl=64 time=0.081 ms
64 bytes from 172.18.0.3: seq=2 ttl=64 time=0.085 ms
64 bytes from 172.18.0.3: seq=3 ttl=64 time=0.075 ms
```

### 宿主机上直接使用 MongDB 数据库

如果想直接在宿主机上使用 MongoDB，可以在`docker run`指令后面加入`entrypoint`指令，这样就可以非常方便的直接进入 **mongo cli**：

```shell
test@VM-0-9-ubuntu:~$ docker run -it --link 3f5f58cd1c9f:db --entrypoint mongo mongo:4.4.0-rc8 --host db
MongoDB shell version v4.4.0-rc8
connecting to: mongodb://db:27017/?compressors=disabled&gssapiServiceName=mongodb
Implicit session: session { "id" : UUID("6dbf4f55-873d-4855-84f0-aaaa78315803") }
MongoDB server version: 4.4.0-rc8
Welcome to the MongoDB shell.
For interactive help, type "help".
For more comprehensive documentation, see
        http://docs.mongodb.org/
Questions? Try the support group
        http://groups.google.com/group/mongodb-user
---
....
> 
> db.version()
4.4.0-rc8
> 
> db.stats()
{
        "db" : "test",
        "collections" : 0,
        "views" : 0,
        "objects" : 0,
        "avgObjSize" : 0,
        "dataSize" : 0,
        "storageSize" : 0,
        "totalSize" : 0,
        "indexes" : 0,
        "indexSize" : 0,
        "scaleFactor" : 1,
        "fileSize" : 0,
        "fsUsedSize" : 0,
        "fsTotalSize" : 0,
        "ok" : 1
}
> 
> show dbs
admin   0.000GB
config  0.000GB
local   0.000GB
> 
> exit
bye
```

## Redis

Redis，全称 REmote DIctionary Server 是一个开源的基于内存的数据结构存储系统，可以作为数据库、缓存和消息的中间件。

### 安装和启动一个 Redis 数据库

```shell
# 用户可以使用官方镜像快速启动一个 Redis 实例:
# https://hub.docker.com/_/redis?tab=tags
test@VM-0-9-ubuntu:~$ docker run --name myredis -d redis:alpine3.12
Unable to find image 'redis:alpine3.12' locally
alpine3.12: Pulling from library/redis
...
Digest: sha256:50ce670996835d83e070a6b26ef168de774333cf6317cd1bad45f84da1421e24
Status: Downloaded newer image for redis:alpine3.12
94ccd00f0fd4ce1b942e2a95eeebc017bf8ffbf638a07342fb892b1bd2f10545
[xingabao@mu01 ~]$ docker ps
CONTAINER ID    IMAGE               COMMAND                  CREATED             STATUS              PORTS         NAMES
94ccd00f0fd4    redis:alpine3.12    "docker-entrypoint..."   20 seconds ago      Up 17 seconds       6379/tcp      myredis
```

```shell
# 先通过`docker pull`获得镜像, 再新建一个容器:
test@VM-0-9-ubuntu:~$ docker pull redis:alpine3.12
alpine3.12: Pulling from library/redis
...
Digest: sha256:50ce670996835d83e070a6b26ef168de774333cf6317cd1bad45f84da1421e24
Status: Downloaded newer image for redis:alpine3.12
docker.io/library/redis:alpine3.12
test@VM-0-9-ubuntu:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
redis               alpine3.12          e008f2ff99d0        3 days ago          31.5MB

test@VM-0-9-ubuntu:~$ docker run -itd --name redis-test -p 6379:6379 redis:alpine3.12
34a7fb8189f03ff1f18a7d1d2515fff091760d32975d6ebedf4bbd23104ba2d7
test@VM-0-9-ubuntu:~$ docker ps
CONTAINER ID   IMAGE              COMMAND            CREATED         STATUS           PORTS                      NAMES
34a7fb8189f0   redis:alpine3.12   "docker-entryp…"   5 seconds ago   Up 4 seconds     0.0.0.0:6379->6379/tcp     redis-test
```

### 进入一个 Redis 数据库

```shell
# docker exec -it 容器ID sh
test@VM-0-9-ubuntu:~$ docker exec -it 34a7fb8189f0 sh  
/data # 
/data # uptime
 07:37:13 up 1 day,  2:14,  load average: 0.00, 0.01, 0.00
/data # ps -ef
PID   USER     TIME  COMMAND
    1 redis     0:00 redis-server
   24 root      0:00 sh
   32 root      0:00 ps -ef
/data # free
              total        used        free      shared  buff/cache   available
Mem:        3915340      756708      192768       40100     2965864     3165416
Swap:             0           0           0
```

### 通过端口访问 Redis 数据库

```shell
test@VM-0-9-ubuntu:~$ docker run -it --link redis-test:db alpine:3.12.0 /bin/sh
/ # 
/ # ls
bin    dev    etc    home   lib    media  mnt    opt    proc   root   run    sbin   srv    sys    tmp    usr    var
/ # ping db
PING db (172.18.0.4): 56 data bytes
64 bytes from 172.18.0.4: seq=0 ttl=64 time=0.121 ms
64 bytes from 172.18.0.4: seq=1 ttl=64 time=0.108 ms
64 bytes from 172.18.0.4: seq=2 ttl=64 time=0.104 ms
64 bytes from 172.18.0.4: seq=3 ttl=64 time=0.111 ms
```

### 宿主机上直接使用 Redis 数据库

```shell
test@VM-0-9-ubuntu:~$ docker run -it --link redis-test:db --entrypoint redis-cli redis:alpine3.12 -h db
db:6379> 
db:6379> ping
PONG
db:6379> set 1 2
OK
db:6379> get 1
"2"
db:6379> exit
```



# 分布式处理与大数据平台

分布式系统和大数据处理平台是目前业界关注的热门技术，目前分布式处理的三大重量级武器：`Hadoop`、`Spark`和`Storm`，以及新一代的数据采集和分析 引擎`Elasticsearch`。

## Hadoop

Hadoop 是 Apache 软件基金会旗下的一个开源分布式计算平台。作为当今大数据处理领域的经典分布式平台，Hadoop 主要基于 Java 语言实现，由三个核心子系统组成：`HDFS`、`YARN`、`MapReduce`，其中，HDFS 是一套分布式文件系统，YARN 是资源管理系统，MapReduce 是运行在 YARN 上的应用，负责分布式处理管理。如果从操作系统的角度来看，HDFS 相当于 Linux 的 ext3/ext4 文件系统，而 Yarn 相当于 Linux 的进程调度和内存分配模块。Hadoop 还包括列数据库`HBase`、分布式数据库`Cassandra`、支持 SQL 语句`Hive`、流处理引擎`Pig`、分布式应用协调服务`Zookeeper`等相关项目

Hadoop 的核心子系统说明如下：

- **HDFS**：一个高度容错性的分布式文件系统，适合部署在大量廉价的机器上，提供高吞吐量的数据访问；
- **YARN**：Yet Another Resource Negotiator，资源管理器，可为上层应用提供统一的资源管理和调度，兼容多计算框架；
- **MapReduce**：是一种分布式编程模型，把对大规模数据集的处理分发给网络上的多个节点，之后收集处理结果进行规约 。

---

**使用官方镜像**：

用户可以通过`docker pull`指令直接从官方镜像中下载并安装 Hadoop 镜像：

```shell
# https://hub.docker.com/r/sequenceiq/hadoop-docker/tags
# 下载安装`sequenceiq/hadoop-docker:2.7.1`.
test@VM-0-9-ubuntu:~$ docker pull sequenceiq/hadoop-docker:2.7.1
2.7.1: Pulling from sequenceiq/hadoop-docker
Image docker.io/sequenceiq/hadoop-docker:2.7.1 uses outdated schema1 manifest format. Please upgrade to a schema2 image for better future compatibility. More information at https://docs.docker.com/registry/spec/deprecated-schema-v1/
...
Digest: sha256:2da37e4eeea57bc99dd64987391ce9e1384c63b4fa56b7525a60849a758fb950
Status: Downloaded newer image for sequenceiq/hadoop-docker:2.7.1
docker.io/sequenceiq/hadoop-docker:2.7.1
# 完成镜像拉取后, 使用`docker run`指令运行镜像, 同时打开`bash`命令行.
test@VM-0-9-ubuntu:~$ docker run -it sequenceiq/hadoop-docker:2.7.1 /etc/bootstrap.sh -bash
/
Starting sshd:                                             [  OK  ]
20/06/09 04:29:33 WARN util.NativeCodeLoader: ...
...
starting resourcemanager, logging to /usr/local/hadoop/logs/yarn--resourcemanager-d09b65037a3f.out
localhost: starting nodemanager, logging to /usr/local/hadoop/logs/yarn-root-nodemanager-d09b65037a3f.out
bash-4.1# 
# 用户此时可以查看各种配置信息和执行操作, 例如查看`namenode`日志等信息.
bash-4.1# cat /usr/local/hadoop/logs/hadoop-root-namenode-d09b65037a3f.out 
ulimit -a for user root
core file size          (blocks, -c) unlimited
data seg size           (kbytes, -d) unlimited
scheduling priority             (-e) 0
file size               (blocks, -f) unlimited
pending signals                 (-i) 15137
max locked memory       (kbytes, -l) 64
max memory size         (kbytes, -m) unlimited
open files                      (-n) 1048576
pipe size            (512 bytes, -p) 8
POSIX message queues     (bytes, -q) 819200
real-time priority              (-r) 0
stack size              (kbytes, -s) 8192
cpu time               (seconds, -t) unlimited
max user processes              (-u) unlimited
virtual memory          (kbytes, -v) unlimited
file locks                      (-x) unlimited
# 用户需要验证 Hadoop 环境是否安装成功, 首先进入 Hadoop 容器的 bash 命令行环境, 进入 Hadoop 目录
bash-4.1# echo $HADOOP_PREFIX
/usr/local/hadoop
bash-4.1# cd $HADOOP_PREFIX
# 然后通过运行 Hadoop 内置的实例程序来进行测试:
bash-4.1# bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.1.jar grep input output "dfs[a-z.]+"
20/06/09 04:39:30 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
20/06/09 04:39:31 INFO client.RMProxy: Connecting to ResourceManager at /0.0.0.0:8032
20/06/09 04:39:32 INFO input.FileInputFormat: Total input paths to process : 31
20/06/09 04:39:33 INFO mapreduce.JobSubmitter: number of splits:31
20/06/09 04:39:33 INFO mapreduce.JobSubmitter: Submitting tokens for job: job_1591691391784_0001
20/06/09 04:39:33 INFO impl.YarnClientImpl: Submitted application application_1591691391784_0001
20/06/09 04:39:33 INFO mapreduce.Job: The url to track the job: http://d09b65037a3f:8088/proxy/application_1591691391784_0001/
20/06/09 04:39:33 INFO mapreduce.Job: Running job: job_1591691391784_0001
20/06/09 04:39:41 INFO mapreduce.Job: Job job_1591691391784_0001 running in uber mode : false
20/06/09 04:39:41 INFO mapreduce.Job:  map 0% reduce 0%
# 最后用户可以使用`hdfs`指令检查输出结果:
bash-4.1# bin/hdfs dfs -cat output/*
20/06/09 04:41:32 WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
6       dfs.audit.logger
4       dfs.class
3       dfs.server.namenode.
2       dfs.period
2       dfs.audit.log.maxfilesize
2       dfs.audit.log.maxbackupindex
1       dfsmetrics.log
1       dfsadmin
1       dfs.servers
1       dfs.replication
1       dfs.file
```

## Spark

Apache Spark 是一个围绕速度、易用性和复杂分析构建的大数据处理框架，基于 Scala 开发。与 Hadoop 和 Storm 等其他大数据及 MapReduce 技术相比，Spark 支持更灵活的函数定义，可以将应用处理速度提升 1~2 个数量级，并且提供了众多方便的使用工具，包括 SQL 查询、流处理、机器学习和图像处理等。Spark 目前支持 Scala、Java、Python、Clojure、R 程序设计语言编写应用，除了 Spark 核心 API 之外，Spark 生态系统中还包括其他附加库，可以在大数据分析和机器学习领域提供更多的能力。

-----

**使用官方镜像**：

用户可以通过`docker pull`指令直接从官方镜像中下载并安装 Spark 镜像：

```shell
# https://github.com/sequenceiq/docker-spark
# 下载安装`sequenceiq/spark:1.6.0`.
test@VM-0-9-ubuntu:~$ docker pull sequenceiq/spark:1.6.0
1.6.0: Pulling from sequenceiq/spark
Image docker.io/sequenceiq/spark:1.6.0 uses outdated schema1 manifest format. Please upgrade to a schema2 image for better future compatibility. More information at https://docs.docker.com/registry/spec/deprecated-schema-v1/
...
Digest: sha256:64fbdd1a9ffb6076362359c3895d089afc65a533c0ef021ad4ae6da3f8b2a413
Status: Downloaded newer image for sequenceiq/spark:1.6.0
docker.io/sequenceiq/spark:1.6.0
# 用户在运行容器时, 需要映射`YARN UI`需要的端口.
test@VM-0-9-ubuntu:~$ docker run -it -p 8088:8088 -p 8042:8042 -h sandbox sequenceiq/spark:1.6.0 /bin/bash
/
Starting sshd:                                             [  OK  ]
Starting namenodes on [sandbox]
sandbox: starting namenode, logging to /usr/local/hadoop/logs/hadoop-root-namenode-sandbox.out
localhost: starting datanode, logging to /usr/local/hadoop/logs/hadoop-root-datanode-sandbox.out
Starting secondary namenodes [0.0.0.0]
0.0.0.0: starting secondarynamenode, logging to /usr/local/hadoop/logs/hadoop-root-secondarynamenode-sandbox.out
starting yarn daemons
starting resourcemanager, logging to /usr/local/hadoop/logs/yarn--resourcemanager-sandbox.out
localhost: starting nodemanager, logging to /usr/local/hadoop/logs/yarn-root-nodemanager-sandbox.out
bash-4.1#
# 启动后, 可以使用 bash 命令行来查看 namenode 日志信息.
bash-4.1# cat  /usr/local/hadoop/logs/hadoop-root-namenode-sandbox.out     
ulimit -a for user root
core file size          (blocks, -c) unlimited
data seg size           (kbytes, -d) unlimited
scheduling priority             (-e) 0
file size               (blocks, -f) unlimited
pending signals                 (-i) 15137
max locked memory       (kbytes, -l) 64
max memory size         (kbytes, -m) unlimited
open files                      (-n) 1048576
pipe size            (512 bytes, -p) 8
POSIX message queues     (bytes, -q) 819200
real-time priority              (-r) 0
stack size              (kbytes, -s) 8192
cpu time               (seconds, -t) unlimited
max user processes              (-u) unlimited
virtual memory          (kbytes, -v) unlimited
file locks                      (-x) unlimited
```

```shell
# 用户还可以使用 daemon 模式运行此 Spark 环境.
test@VM-0-9-ubuntu:~$ docker run -d -h sandbox sequenceiq/spark:1.6.0 -d
cadce68f6819314de9ce19e848af1e3e38a752e29d6fa2143a66d9388716af11
# 使用`docker ps`指令查看运行详情.
CONTAINER ID  IMAGE                   COMMAND          CREATED        STATUS         PORTS                  NAMES
cadce68f6819  sequenceiq/spark:1.6.0  "/etc/boots..."  51 seconds ago Up 49 seconds  22/tcp, 8030-8033/tcp, suspicious_liskov
```

## Storm

Apache Storm 是一个实时流计算框架，基于 Clojure 等语言实现。Storm 集群与 Hadoop 集群在工作方式上十分相似，唯一区别在于 Hadoop 上运行的是 MapReduce 任务，在 Storm 上运行的则是 topology 。MapReduce 任务完成处理即会结束，而 topology 则永远在等待消息并处理，直到停止。

## Elasticsearch

Elasticsearch 是基于 Lucene 的开源搜索服务，Java 实现。它是分布式、多租户的全文搜索引擎，支持 RESTful Web 接口。Elasticsearch 支持实时分布式数据库存储和分析查询功能，可以轻松扩展到上百台服务器，同时支持处理 PB 级结构化或非结构化数据。如果配合 Logstash、Kibana 等组件，可以快速构建一套日志消息分析平台。





