# 微表情识别系统 (Micro-Expression Recognition System)

> **项目重构说明**：本项目已完成全面重构升级。原项目代码位于 `master` 分支，重构后的代码在 `main` 分支。重构内容包括：模块化目录结构、模型参数量优化（68M→3M，减少 95%）、统一推理管道、阶段式训练支持等。详见 [重构前后对比](docs/comparison.md)。

基于 CNN+LSTM 深度学习架构的微表情识别系统，使用 CASME2 数据集进行训练，支持 5 类微表情的自动识别。

## 支持的微表情类别

| 编号 | 英文 | 中文 |
|------|------|------|
| 0 | surprise | 惊讶 |
| 1 | repression | 压抑 |
| 2 | happiness | 高兴 |
| 3 | disgust | 厌恶 |
| 4 | others | 其他 |

---

## 项目结构

```
MicroExpressionRecognizer/
├── main.py                        # 启动器（Tkinter GUI，推荐入口）
├── config.ini                     # 统一配置文件（所有参数集中管理）
├── requirements.txt               # Python 依赖列表
│
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── config.py                  # 配置管理器（单例模式）
│   ├── detector.py                # 人脸检测 + 68 点关键点提取
│   └── inference.py               # 统一推理管道（所有界面共享）
│
├── data/                          # 数据处理
│   └── prepare_dataset.py         # CASME2 数据预处理脚本
│
├── models/                        # 模型定义
│   ├── microexpression_model.py   # CNN+LSTM 模型（GAP + BN + BiLSTM）
│   └── weights/                   # 训练好的权重（不上传）
│
├── training/                      # 训练相关
│   ├── trainer.py                 # 完整训练器（支持阶段式训练）
│   ├── quick_test.py              # 快速验证脚本（少量数据，5 个 epoch）
│   └── metrics.py                 # 评估指标 + 可视化
│
├── interfaces/                    # 用户界面
│   ├── desktop/
│   │   ├── gui_main.py            # PyQt5 桌面应用
│   │   └── ui_layout.py           # UI 布局定义
│   └── webapp/
│       └── gradio_app.py          # Gradio Web 应用
│
├── utils/                         # 工具
│   └── dataset_loader.py          # PyTorch Dataset 类
│
└── docs/                          # 文档
    ├── training-guide.md          # 训练详细指南
    ├── comparison.md              # 重构前后对比
    └── model-architecture.md      # 模型架构详解
```

---

## 环境配置（从零开始）

### 第 1 步：安装 Python 环境

推荐使用 Anaconda 或 Miniconda。

```bash
# 创建虚拟环境
conda create -n microexp python=3.10

# 激活环境
conda activate microexp
```

### 第 2 步：安装 PyTorch（根据是否有 GPU 选择）

**有 NVIDIA 显卡（推荐）：**
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

**没有 GPU（CPU 模式）：**
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

### 第 3 步：安装其他依赖

```bash
pip install opencv-python numpy pandas scikit-learn matplotlib seaborn tqdm Pillow gradio PyQt5 dlib
```

或者直接：
```bash
pip install -r requirements.txt
```

### 第 4 步：安装 dlib 依赖（可能需要）

dlib 需要编译，如果安装失败：

**Windows 用户：**
```bash
# 方法1：使用 conda
conda install -c conda-forge dlib

# 方法2：使用预编译的 whl 文件
# 从 https://github.com/z-mahmud22/Dlib_Windows_Python3.x 下载对应版本的 whl 文件
pip install dlib-19.24.1-cp310-cp310-win_amd64.whl
```

### 第 5 步：验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import cv2; import dlib; print('OpenCV + dlib OK')"
```

---

## 数据准备

### 需要的数据文件

本项目使用 CASME2 数据集，需要以下文件：

1. **原始视频帧图像** → 放在 `data/CASME2-RAW/` 目录
2. **标注 Excel 文件** → 放在 `data/CASME2-coding-20140508.xlsx`
3. **辅助文件**（项目已包含）：
   - `utils/haarcascade_frontalface_default.xml` — OpenCV 人脸检测模型
   - `utils/shape_predictor_68_face_landmarks.dat` — dlib 68 点关键点模型

### 数据目录结构

```
data/
├── CASME2-coding-20140508.xlsx    # 标注文件（你需要提供）
├── CASME2-RAW/                    # 原始帧图像（你需要提供）
│   ├── sub01/
│   │   ├── EP02_01f/
│   │   │   ├── img001.jpg
│   │   │   ├── img002.jpg
│   │   │   └── ...
│   │   └── EP02_02f/
│   │       └── ...
│   ├── sub02/
│   │   └── ...
│   └── ...
```

> **注意**：CASME2 数据集需要申请才能获取，请访问其官方网站。

### 运行数据预处理

```bash
python data/prepare_dataset.py
```

该脚本会自动完成：
1. 分析类别分布，排除样本最少的 2 个类别
2. 提取人脸 68 点关键点（dlib）
3. 提取帧间光流特征（Farneback 算法）
4. 创建 32 帧序列样本（以 apex 帧为中心）
5. 数据增强（水平翻转）
6. 按 80/20 分割训练/测试集
7. 生成 `cls_train.txt`、`cls_test.txt`、`class_mapping.txt`

---

## 使用方法

### 方式一：启动器界面（推荐）

```bash
# 确保在 microexp 环境中
conda activate microexp

# 启动
python main.py
```

启动后界面包含以下功能按钮：

| 按钮 | 功能 |
|------|------|
| 1. 数据准备 | 运行数据预处理脚本 |
| 2. 模型训练 | 运行完整训练流程 |
| 3. 桌面界面 | 打开 PyQt5 识别界面 |
| 4. 网页界面 | 打开 Gradio Web 界面 |
| 查看系统信息 | 显示 Python/PyTorch/CUDA 版本 |
| 查看可用模型 | 列出已训练的模型及准确率 |
| 切换模型 | 选择使用哪个模型进行预测 |

### 方式二：命令行直接运行

```bash
# 数据准备
python data/prepare_dataset.py

# 完整训练
python training/trainer.py

# 快速验证（少量数据，5 个 epoch，确认 pipeline 正常）
python training/quick_test.py

# 桌面界面
python interfaces/desktop/gui_main.py

# Web 界面
python interfaces/webapp/gradio_app.py
```

---

## 模型训练

### 标准训练

编辑 `config.ini` 中的 `[training]` 部分：

```ini
[training]
stage = 0              # 0=标准训练，1=只训练CNN，2=联合微调
batch_size = 8         # 批次大小（显存不够减小此值）
num_epochs = 30        # 训练轮数
learning_rate = 0.0001 # 学习率
early_stopping_patience = 10  # 验证集无改善时提前停止的容忍轮数
```

然后运行：
```bash
python training/trainer.py
```

### 阶段式训练（推荐）

**第 1 阶段 — 只训练 CNN 特征提取器：**

修改 `config.ini`：
```ini
[training]
stage = 1
stage1_learning_rate = 0.001
stage1_epochs = 25
```

运行 `python training/trainer.py`

**第 2 阶段 — 联合微调全部网络：**

修改 `config.ini`：
```ini
[training]
stage = 2
stage1_checkpoint = models/weights/你的阶段1模型目录/stage1_best.pth
learning_rate = 0.0001
num_epochs = 30
```

运行 `python training/trainer.py`

### 快速验证（推荐首次使用）

在正式训练前，用少量数据验证 pipeline 是否正常：

```bash
python training/quick_test.py
```

该脚本自动抽取每类 50 个训练样本 + 20 个测试样本，训练 5 个 epoch，确认数据加载和模型前向/反向传播正常。

### 训练输出

训练结果保存在 `models/weights/MicroExpModel_<时间戳>_<准确率>/` 目录：

| 文件 | 内容 |
|------|------|
| `best_model.pth` | 最佳模型权重 |
| `training.log` | 训练日志 |
| `results.json` | 评估结果（JSON） |
| `loss_curve.png` | 损失曲线 |
| `accuracy_curve.png` | 准确率曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `training_overview.png` | 综合概览图 |

---

## 预测使用

### 桌面界面（PyQt5）

启动后支持三种输入方式：
- **图像预测** — 选择一张图片进行识别
- **视频预测** — 选择视频文件逐帧识别
- **摄像头预测** — 实时摄像头识别

界面下方显示 5 类表情的概率条，上方标注识别结果。

### Web 界面（Gradio）

启动后浏览器自动打开，支持：
- 上传图片识别
- 摄像头实时识别

### 切换模型

在启动器界面点击"切换模型"，选择已训练的模型。切换后下次启动界面会使用新模型。

---

## 模型架构

三流 CNN + BiLSTM 架构，参数量约 3M：

```
输入: 32帧序列 (128×128 灰度) + 68点关键点 + 光流

图像流:  Conv+BN → MaxPool → Conv+BN → MaxPool → Conv+BN → GAP → 128维
光流流:  Conv+BN → MaxPool → Conv+BN → MaxPool → Conv+BN → GAP → 128维
关键点:  FC+BN → ReLU → FC+BN → ReLU → 128维

融合:    拼接 → 384维
时序:    BiLSTM (384→256, 2层)
分类:    FC → ReLU → Dropout → FC → 5类
```

---

## 常见问题

### Q: 运行报错 `ModuleNotFoundError: No module named 'torch'`
A: 确保已激活 conda 环境：`conda activate microexp`

### Q: dlib 安装失败
A: 使用 conda 安装：`conda install -c conda-forge dlib`，或下载预编译 whl 文件。

### Q: CUDA 不可用 / 训练很慢
A: 检查显卡驱动是否正确安装。没有 GPU 也可以用 CPU 训练，只是较慢。

### Q: 找不到人脸检测模型文件
A: 确保 `utils/haarcascade_frontalface_default.xml` 和 `utils/shape_predictor_68_face_landmarks.dat` 存在。这些文件已包含在项目中。

### Q: 数据准备报错找不到 CASME2 数据
A: 请确认 CASME2 原始数据已按目录结构放在 `data/CASME2-RAW/` 下。

### Q: 训练时显存不足 (CUDA out of memory)
A: 在 `config.ini` 中减小 `batch_size`（如改为 4 或 2）。

---

## 依赖列表

| 包名 | 用途 |
|------|------|
| PyTorch + torchvision | 深度学习框架 |
| OpenCV | 图像处理、人脸检测 |
| dlib | 68 点人脸关键点提取 |
| numpy | 数值计算 |
| pandas | 读取 Excel 标注文件 |
| scikit-learn | 评估指标、数据分割 |
| matplotlib + seaborn | 可视化绘图 |
| tqdm | 训练进度条 |
| Pillow | 图像处理（中文标注） |
| PyQt5 | 桌面 GUI |
| Gradio | Web 界面 |
