**GNN4DP 环境跨平台迁移与部署技术指南**

本指南记录了将 WSL2 本地配置好的深度学习环境（基于 PyTorch 2.6.0 \+ CUDA 12.1）通过 Docker 封装并部署到超算中心（Singularity/Apptainer）的标准化流程。

## ---

** 环境核心参数**

* **基础镜像**: nvidia/cuda:12.1.1-devel-ubuntu22.04 (与本地 Torch CUDA 版本严格对齐)  
* **迁移工具**: conda-pack (解决物理路径迁移后的硬编码修复问题)  
* **目标格式**: Singularity 镜像文件 (.sif)

## ---

** 第一阶段：Docker 镜像封装 (WSL2)**

编写 Dockerfile 实现环境的标准化封装。

### **1\. Dockerfile 配置**

Dockerfile

\# 1\. 基础镜像对齐 CUDA 版本  
FROM nvidia/cuda:12.1.1\-devel-ubuntu22.04

\# 2\. 环境变量设置  
ENV DEBIAN\_FRONTEND=noninteractive

\# 3\. 安装系统级依赖（解决 OpenCV 等库的底层依赖问题）  
RUN apt-get update && apt-get install \-y \--no-install-recommends \\  
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \\  
    && rm \-rf /var/lib/apt/lists/\*

\# 4\. 创建目录并解压环境包  
RUN mkdir \-p /app/GNN4DP  
ADD GNN4DP.tar.gz /app/GNN4DP/

\# 5\. 【核心步骤】路径修复  
RUN ln -s /app/GNN4DP/bin/python /usr/bin/python
RUN /app/GNN4DP/bin/conda-unpack

\# 6\. 配置最终搜索路径  
ENV PATH=/app/GNN4DP/bin:$PATH  
ENV CONDA\_DEFAULT\_ENV=GNN4DP

WORKDIR /app  
CMD \["bash"\]

### **2\. 构建与导出镜像**

Bash

\# 构建镜像  
docker build \-t gnn4dp\_hpc:v1 .

\# 导出镜像为单一 tar 文件以便传输  
docker save gnn4dp\_hpc:v1 \-o gnn4dp\_image.tar

## ---

**🚀 第二阶段：超算端转换与测试 (HPC)**

将 gnn4dp\_image.tar 上传至超算中心后执行。

1. **转换为 Singularity 格式**：  
   Bash  
   module load singularity   
   singularity build gnn4dp.sif docker-archive://gnn4dp\_image.tar

2. **环境验证测试**：  
   使用 \--nv 标志确保容器能调用超算的 NVIDIA 显卡。  
   Bash  
   singularity shell --nv ~/Docker_image/gnn4dp.sif 进入环境

## ---

**⚠️ 避坑总结**

\[\!IMPORTANT\]

1. **CUDA 版本对齐**：必须通过 torch.version.cuda 确认版本。如果 Torch 是 12.1，基础镜像千万不能用 12.4，否则会导致显卡驱动初始化失败。  
2. **路径硬编码**：Conda 环境迁移后，pip 和某些 Python 脚本内的路径会指向原 WSL2 路径。必须通过 conda-unpack 命令自动重写这些二进制文件和脚本。  
3. **空间管理**：Singularity 在集群上尝试把 .sif 镜像解压到临时目录 /tmp 时空间不够。 echo 'export SINGULARITY_TMPDIR=/fs1/home/USER/tmp_singularity' >> ~/.bashrc

---
