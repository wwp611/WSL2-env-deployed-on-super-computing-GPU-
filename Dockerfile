# 完美对齐你的 PyTorch CUDA 12.1 版本
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 避免安装时卡在时区选择等交互界面
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统基础依赖 (以防你的某些包底层需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建存放环境的目录
RUN mkdir -p /app/GNN4DP

# 自动解压环境包到目标目录
ADD GNN4DP.tar.gz /app/GNN4DP/

RUN ln -s /app/GNN4DP/bin/python /usr/bin/python
# 【核心步骤】自动修复 Conda 环境内所有写死的硬编码路径
RUN /app/GNN4DP/bin/conda-unpack

# 设置环境变量，让系统默认使用这个打包进来的环境
ENV PATH=/app/GNN4DP/bin:$PATH
ENV CONDA_DEFAULT_ENV=GNN4DP

# 设置工作目录
WORKDIR /app
CMD ["bash"]