#!/bin/bash

# RECALL环境迁移脚本
echo "开始RECALL环境迁移设置..."

# 1. 创建conda环境
echo "正在创建conda环境..."
conda env create -f environment.yml

# 2. 激活环境
echo "激活环境..."
conda activate recall

# 3. 安装MuJoCo依赖
echo "安装MuJoCo依赖..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Linux安装指南
  echo "检测到Linux系统，安装相应依赖..."
  sudo apt-get update -q
  sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS安装指南
  echo "检测到macOS系统，安装相应依赖..."
  brew install gcc

elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  # Windows安装指南
  echo "检测到Windows系统，请确保已安装Visual C++ Build Tools"
  echo "如果遇到问题，请访问：https://github.com/openai/mujoco-py/blob/master/README.md"
fi

# 4. 安装mujoco-py
echo "安装mujoco-py..."
pip install mujoco-py<2.1,>=2.0

# 5. 安装metaworld
echo "安装metaworld..."
cd metaworld
pip install -e .
cd ..

# 6. 安装当前项目
echo "安装RECALL项目..."
pip install -e .

echo "环境设置完成！"
echo ""
echo "使用方法:"
echo "1. 激活环境: conda activate recall"
echo "2. 运行实验: python run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=100 --log_every=10 --cl_method=recall"
echo ""
echo "备注：如果遇到MuJoCo安装问题，请参考官方文档：https://github.com/openai/mujoco-py/blob/master/README.md" 