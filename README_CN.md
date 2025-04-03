# RECALL 环境迁移指南

这是 RECALL（Replay-enhanced Continual Reinforcement Learning）项目的环境迁移指南。本指南将帮助你在新的计算机上重新建立运行环境。

## 环境要求

- Python 3.8.20
- CUDA 支持的 GPU（用于 TensorFlow GPU 加速）
- Conda 包管理器

## 迁移步骤

### 1. 克隆本仓库

```bash
git clone https://your-repository-url.git
cd RECALL
```

### 2. 使用自动化脚本设置环境

我们提供了一个自动化脚本来处理环境的设置过程：

```bash
chmod +x setup_env.sh
./setup_env.sh
```

这个脚本会:
- 创建名为 `recall` 的 Conda 环境
- 安装所有必要的依赖项
- 安装 MuJoCo 和相关库
- 安装 MetaWorld 框架
- 安装项目本身

### 3. 手动设置环境（如果自动脚本失败）

如果自动脚本无法正常工作，你可以手动执行以下步骤：

```bash
# 创建并激活 Conda 环境
conda env create -f environment.yml
conda activate recall

# 安装 MuJoCo 依赖（根据操作系统而异）
# 在 Linux 上:
sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev patchelf

# 安装 mujoco-py
pip install mujoco-py<2.1,>=2.0

# 安装 MetaWorld
cd metaworld
pip install -e .
cd ..

# 安装项目
pip install -e .
```

### 4. 验证安装

为了验证环境是否正确设置，你可以运行一个简单的测试：

```bash
python run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=100 --log_every=10 --cl_method=recall
```

## 常见问题解决

### MuJoCo 安装问题

如果在安装 MuJoCo 时遇到问题，请参考官方文档：https://github.com/openai/mujoco-py/blob/master/README.md

### TensorFlow 和 CUDA 兼容性

本项目使用 TensorFlow 2.6.0，请确保你的 CUDA 和 cuDNN 版本与之兼容。推荐:
- CUDA 11.2
- cuDNN 8.1.0

### 显存不足错误

如果遇到显存不足错误，可以尝试减小批量大小或模型大小：

```bash
python run_cl.py --tasks=CW6_0 --seed=0 --batch_size=64 --hidden_sizes 128 128 128 --steps_per_task=100 --log_every=10 --cl_method=recall
```

## 环境详情

本项目的核心依赖项包括：
- TensorFlow 2.6.0 GPU 版本
- MuJoCo 物理引擎和 mujoco-py 接口
- MetaWorld 基准测试环境
- Gymnasium/Gym 强化学习接口
- 各种数据处理和可视化库（NumPy, Pandas, Matplotlib等）

## 参数说明

运行实验时，你可以使用多种命令行参数来控制实验设置：

- `--tasks`: 指定任务序列（如 CW6_0, CW10 等）
- `--seed`: 随机种子，用于实验重现
- `--steps_per_task`: 每个任务的训练步数
- `--log_every`: 评估和记录之间的步数
- `--cl_method`: 连续学习方法（如 ft, ewc, recall 等）
- `--policy_reg_coef`: 策略正则化系数
- `--hidden_sizes`: 神经网络隐藏层大小

更多参数详情，请运行：

```bash
python run_cl.py --help
``` 