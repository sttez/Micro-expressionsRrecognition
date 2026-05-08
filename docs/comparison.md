# 重构前后对比记录

## 1. 目录结构

### 重构前
```
MicroExpressionRecognizer/
├── main.py                     # Tkinter 启动器
├── config.ini                  # 简单配置（3个section）
├── trainner.py                 # 训练脚本（拼写错误）
├── prepare_casme2_dataset.py   # 数据预处理
├── prepare_casme2_dataset.py-v2.py  # 损坏的v2版本
├── fix_paths.py                # 路径修复工具
├── 模型参数.py                  # 模型分析工具
├── interface/
│   ├── GUI_main.py             # PyQt5 GUI
│   └── Ui_Emo_gui2.py          # UI布局
├── webapp/
│   └── gradio_app.py           # Gradio Web
├── models/
│   └── microexpression_model.py
├── utils/
│   ├── dataset_loader.py
│   ├── model_preprocess.py     # 含假随机landmarks
│   └── visualizer.py           # 孤立未使用
├── data/
│   ├── CNN.py                  # 独立demo
│   ├── lstm.py                 # 空文件
│   ├── static.py               # 独立demo
│   ├── 人脸对其.py              # 独立demo
│   └── 光流.py                 # 独立demo
├── requirements.txt            # 内容被覆盖，不是依赖
├── 1.txt                       # 内容被覆盖
└── 多个过时文档 .md
```

### 重构后
```
MicroExpressionRecognizer/
├── main.py                     # 修复后的启动器
├── config.ini                  # 扩展配置（5个section）
├── core/
│   ├── config.py               # 配置管理器（单例）
│   ├── detector.py             # 统一人脸检测器
│   └── inference.py            # 统一推理管道
├── data/
│   └── prepare_dataset.py      # 数据预处理（重构）
├── models/
│   └── microexpression_model.py # 优化后的模型
├── training/
│   ├── trainer.py              # 训练器（支持阶段式）
│   └── metrics.py              # 评估指标
├── interfaces/
│   ├── desktop/
│   │   ├── gui_main.py         # PyQt5 GUI（重构）
│   │   └── ui_layout.py        # UI布局
│   └── webapp/
│       └── gradio_app.py       # Gradio Web（重构）
├── utils/
│   └── dataset_loader.py       # Dataset（重构）
├── docs/                       # 文档目录
├── requirements.txt            # 真正的依赖文件
└── README.md                   # 完整项目文档
```

## 2. 模型架构

| 对比项 | 重构前 | 重构后 |
|--------|--------|--------|
| CNN 特征压缩 | 直接 flatten (128×16×16=32768) | 全局平均池化 GAP (128) |
| BatchNorm | 无 | 每层 Conv/FC 后加 BN |
| LSTM 方向 | 单向 | 双向 (Bidirectional) |
| LSTM 输入维度 | 65664 | 384 |
| LSTM 参数量 | ~67M | ~2M |
| 总参数量 | ~68M | ~3M |
| 参数减少 | — | **95%** |
| 冻结/解冻 | 不支持 | 支持 freeze_backbone / unfreeze_all |

## 3. 推理管道

| 对比项 | 重构前 | 重构后 |
|--------|--------|--------|
| 人脸检测 | 各接口各自初始化 | 统一 `core/detector.py` |
| Landmarks | webapp 使用**随机假数据** | 统一 dlib 精确检测 |
| 接口预处理 | 三个接口三套不同实现 | 统一 `core/inference.py` |
| 路径配置 | 硬编码绝对路径 | `core/config.py` 统一管理 |
| 推理一致性 | 各接口结果可能不同 | 所有接口共享同一引擎 |

## 4. 训练系统

| 对比项 | 重构前 | 重构后 |
|--------|--------|--------|
| 阶段式训练 | 不支持 | 支持 stage 0/1/2 |
| 超参数配置 | 代码硬编码 | config.ini 配置 |
| Early Stopping | 无 | 支持（patience 可配置） |
| 日志 | 简单 print | 完整 logging 系统 |
| 可视化 | 内联+孤立模块两套 | 统一 `training/metrics.py` |
| 重复代码 | 最后有重复代码块 | 已清理 |

## 5. 接口系统

| 对比项 | 重构前 | 重构后 |
|--------|--------|--------|
| 启动器 (main.py) | `PYTHONUNBUFFERED='1.txt'` bug | 修复为 `'1'` |
| 启动器 | bare except 吞异常 | 捕获具体异常 |
| 桌面 GUI | 5处硬编码路径 | 从 config 读取 |
| Web 应用 | 每帧重新加载 CascadeClassifier | 复用检测器 |
| Web 应用 | 无 webcam tab | 新增 webcam tab |
| Web 应用 | 写临时文件到磁盘 | 内存处理 |

## 6. 代码质量

| 对比项 | 重构前 | 重构后 |
|--------|--------|--------|
| 配置管理 | config.ini + 硬编码混合 | 单例 Config 类 |
| 模块职责 | 交叉依赖、职责不清 | core/data/models/training/interfaces 分层 |
| Keras 依赖 | 引入 keras 仅用于 to_categorical | 移除，使用 numpy |
| 废弃文件 | 17个废弃/孤立文件 | 全部清理 |
| 文档 | 多个过时.md + 损坏的 requirements.txt | 完整 README + 设计文档 + 训练指南 |
| 路径处理 | Windows `\\` vs `/` 混用 | Config.path() 统一解析 |

## 7. 新增功能

- **阶段式训练**: 先训练 CNN（冻结 LSTM），再联合微调
- **统一推理引擎**: `InferenceEngine` 类被所有接口共享
- **人脸检测器**: `FaceDetector` 支持 dlib 精确检测 + OpenCV fallback
- **Webcam 支持**: Gradio 界面新增摄像头识别 tab
- **配置管理器**: `Config` 单例，路径自动解析，模型自动查找
- **模型分析**: 参数量统计工具函数
