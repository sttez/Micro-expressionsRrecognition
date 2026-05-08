# 微表情识别项目重构设计文档

> 日期: 2026-05-08
> 状态: 自检通过，待用户审批

## 1. 目标

对 CASME2 微表情识别项目进行全面重构：模块化设计、模型优化、统一推理管道、阶段式训练支持、清理废弃代码，使项目结构清晰、功能完整、可维护。

## 2. 整体架构

### 2.1 目标目录结构

```
MicroExpressionRecognizer/
├── main.py                     # Tkinter 启动器（修复 bug）
├── config.ini                  # 统一配置（扩展版）
├── core/
│   ├── __init__.py
│   ├── config.py               # 配置管理器（单例，路径自动解析）
│   ├── detector.py             # 人脸检测 + landmarks 提取（统一，替代假随机 landmarks）
│   └── inference.py            # 统一推理管道（被所有接口共享）
├── data/
│   ├── __init__.py
│   └── prepare_dataset.py      # 数据预处理（从原 v1 重构）
├── models/
│   ├── __init__.py
│   └── microexpression_model.py # CNN+LSTM 模型（GAP + BN + 双向 LSTM）
├── training/
│   ├── __init__.py
│   ├── trainer.py              # 训练器（支持阶段式训练）
│   └── metrics.py              # 评估指标 + 可视化
├── interfaces/
│   ├── __init__.py
│   ├── desktop/
│   │   ├── __init__.py
│   │   ├── gui_main.py         # PyQt5 主窗口（接入 InferenceEngine）
│   │   └── ui_layout.py        # UI 布局（从 Ui_Emo_gui2.py 迁移）
│   └── webapp/
│       ├── __init__.py
│       └── gradio_app.py       # Gradio Web 应用（接入 InferenceEngine，新增 webcam）
├── utils/
│   ├── __init__.py
│   └── dataset_loader.py       # PyTorch Dataset（接入 core.config）
├── models/weights/             # 训练权重
├── data/CASME2-RAW/            # 原始数据（不动）
├── docs/
│   ├── superpowers/specs/      # 本文档
│   ├── comparison.md           # 重构前后对比记录
│   └── training-guide.md       # 训练指南
├── requirements.txt            # 真正的依赖清单
└── README.md                   # 完整项目说明
```

### 2.2 模块依赖关系

```
main.py ──→ 启动子进程调用 interfaces/ 或 training/

interfaces/desktop/gui_main.py ──→ core/inference.py ──→ core/detector.py
                               └──→ core/config.py

interfaces/webapp/gradio_app.py ──→ core/inference.py ──→ core/detector.py
                                └──→ core/config.py

training/trainer.py ──→ utils/dataset_loader.py
                   └──→ models/microexpression_model.py
                   └──→ training/metrics.py
                   └──→ core/config.py

data/prepare_dataset.py ──→ core/detector.py
                        └──→ core/config.py
```

核心原则：`core/` 是基础设施层，被所有模块依赖，不依赖任何项目内部模块。

## 3. 模型优化

### 3.1 当前问题

三流 CNN 输出直接拼接：`32768 + 128 + 32768 = 65664` 维送入 LSTM，参数量巨大。

### 3.2 优化方案

在每个 CNN 流后加全局平均池化 (AdaptiveAvgPool2d)，加 BatchNorm，改双向 LSTM：

```
图像流:  Conv+BN×3 → GAP     → (T, 128)
光流流:  Conv+BN×3 → GAP     → (T, 128)
关键点流: FC+BN     → (T, 128)

拼接:    (T, 384)  ← 从 65664 降到 384
LSTM:    BiLSTM(384, 256, 2层) → (T, 512)
输出:    Linear(512, 128) → ReLU → Dropout → Linear(128, 5)
```

### 3.3 参数量对比

| 指标 | 当前模型 | 优化后 |
|------|----------|--------|
| LSTM 输入维度 | 65664 | 384 |
| LSTM 参数量 | ~67M | ~2M |
| 总参数量（估算） | ~68M | ~3M |
| LSTM 方向 | 单向 | 双向 |
| BatchNorm | 无 | 有 |

参数量减少约 95%。

### 3.4 兼容性

优化后的模型结构不同，**无法加载旧的 best_model.pth**，需要重新训练。这是已预期的行为。

## 4. 阶段式训练

### 4.1 策略

通过 config.ini 的 `[training] stage` 参数控制：

| stage | 行为 |
|-------|------|
| 0 | 标准全量训练（兼容原行为） |
| 1 | 仅训练 CNN 特征提取器（冻结 LSTM + Classifier） |
| 2 | 联合微调（从 stage1 checkpoint 加载，解冻全部） |

### 4.2 冻结机制

```python
def freeze_backbone(model):
    """冻结 image_cnn, flow_cnn, landmark_fc"""
    for name, param in model.named_parameters():
        if 'lstm' not in name and 'classifier' not in name:
            param.requires_grad = False

def unfreeze_all(model):
    """解冻全部参数"""
    for param in model.parameters():
        param.requires_grad = True
```

### 4.3 训练流程

```
main():
  if stage == 1:
    创建模型 → 冻结 LSTM+Classifier → 训练 stage1_epochs → 保存 stage1_best.pth
  if stage == 2:
    加载 stage1_checkpoint → 解冻全部 → 训练 num_epochs（early stopping）→ 保存 best_model.pth
  if stage == 0:
    标准全量训练 → 保存 best_model.pth
  统一评估 + 可视化
```

### 4.4 config.ini 扩展

```ini
[training]
stage = 0
stage1_checkpoint =
pretrained_model =
batch_size = 8
num_epochs = 30
learning_rate = 0.0001
stage1_learning_rate = 0.001
stage1_epochs = 25
early_stopping_patience = 10
train_split = 0.8
```

## 5. 统一推理管道

### 5.1 core/detector.py

统一人脸检测器，被所有接口共享。解决 webapp 使用假随机 landmarks 的关键问题。

```python
class FaceDetector:
    def __init__(self, config):
        # 优先 dlib（精确 68 点），降级 OpenCV（Haar + 中心估算）
        self.use_dlib = self._init_dlib()

    def detect(self, frame) -> (face_roi, landmarks):
        """返回裁剪人脸和 68 点 landmarks"""
```

### 5.2 core/inference.py

```python
class InferenceEngine:
    def __init__(self, config):
        self.model = self._load_model()
        self.detector = FaceDetector(config)

    def predict_image(self, image) -> (label, probabilities):
        """单张图片推理"""

    def predict_frame(self, frame, frame_buffer) -> (label, probabilities):
        """实时视频帧推理"""

    def predict_video(self, video_path) -> list[(label, probabilities, timestamp)]:
        """逐帧视频推理"""
```

### 5.3 接口改造

- **desktop/gui_main.py**: 删除硬编码路径，删除重复预处理函数，改用 `InferenceEngine`
- **webapp/gradio_app.py**: 删除假 landmarks，删除临时文件 I/O，改用 `InferenceEngine`，增加 webcam tab
- **main.py**: 修复 `PYTHONUNBUFFERED` bug，删除 bare except

### 5.4 统一数据流

```
输入 → core/detector 检测人脸+landmarks → core/inference 预处理+光流
    → model 前向推理 → (label, probabilities) → 接口显示
```

## 6. 统一配置系统

### 6.1 core/config.py

```python
class Config:
    """单例配置管理器"""
    def path(self, section, key) -> str:
        """路径配置，自动解析为绝对路径"""
    def get(self, section, key, fallback=None) -> str:
    def getint(self, section, key, fallback=None) -> int:
    def getfloat(self, section, key, fallback=None) -> float:
```

### 6.2 config.ini 完整版

```ini
[paths]
project_root =
raw_data_dir = data/CASME2-RAW
sequences_dir = data/sequences
excel_path = data/CASME2-coding-20140508.xlsx
cascade_path = utils/haarcascade_frontalface_default.xml
predictor_path = utils/shape_predictor_68_face_landmarks.dat
weights_dir = models/weights
label_train = cls_train.txt
label_test = cls_test.txt
class_mapping = class_mapping.txt

[preprocessing]
image_size = 128
sequence_length = 32
min_sequences_per_class = 475
excluded_classes = fear,sadness
valid_classes = surprise,repression,happiness,disgust,others

[training]
stage = 0
stage1_checkpoint =
pretrained_model =
batch_size = 8
num_epochs = 30
learning_rate = 0.0001
stage1_learning_rate = 0.001
stage1_epochs = 25
early_stopping_patience = 10
train_split = 0.8

[webapp]
port = 7860
share = false

[desktop]
font_path =
```

## 7. 清理计划

### 7.1 删除文件

| 文件 | 原因 |
|------|------|
| `prepare_casme2_dataset.py-v2.py` | 缩进损坏，功能已合并到 `data/prepare_dataset.py` |
| `1.txt` | 内容被覆盖，无意义 |
| `fix_paths.py` | 路径问题由 `core/config.py` 统一解决 |
| `data/lstm.py` | 空文件 |
| `data/CNN.py` | 独立 demo |
| `data/static.py` | 独立 demo |
| `data/人脸对其.py` | 独立 demo |
| `data/光流.py` | 独立 demo |
| `utils/visualizer.py` | 功能由 `training/metrics.py` 覆盖 |
| `utils/model_preprocess.py` | 功能由 `core/inference.py` + `core/detector.py` 替代 |
| `cls_train.txt.backup` / `cls_test.txt.backup` | `fix_paths.py` 生成，不再需要 |
| `comprehensive-readme-enhanced.md` | 过时文档 |
| `data-processing-readme-enhanced.md` | 过时文档 |
| `model-training-readme-enhanced.md` | 过时文档 |
| `CASME2数据预处理脚本README.md` | 过时文档 |
| `CASME2模型训练脚本README.md` | 过时文档 |
| `数据处理.md` | 过时文档 |
| `模型训练.md` | 过时文档 |

### 7.2 移动/重命名

| 操作 | 说明 |
|------|------|
| `trainner.py` → `training/trainer.py` | 重构 + 修正拼写 |
| `interface/Ui_Emo_gui2.py` → `interfaces/desktop/ui_layout.py` | 迁移 + 重命名 |
| `interface/GUI_main.py` → `interfaces/desktop/gui_main.py` | 迁移 + 重命名 |
| `webapp/gradio_app.py` → `interfaces/webapp/gradio_app.py` | 迁移 |
| `models/microexpression_model.py` | 保留位置，优化结构 |
| `utils/dataset_loader.py` | 保留位置，接入 `core.config` |

## 8. 新增文档

| 文档 | 内容 |
|------|------|
| `README.md` | 项目概述、安装说明、使用指南、架构简介、训练流程 |
| `docs/training-guide.md` | 详细训练指南：数据准备、阶段式训练、超参数调优、评估方法 |
| `docs/comparison.md` | 重构前后对比：目录结构、模型参数、推理管道、接口、代码质量 |
| `docs/model-architecture.md` | 模型架构详解：网络结构图、各层参数、输入输出格式 |

## 9. 重构执行顺序

1. 创建目录结构 (`core/`, `interfaces/`, `training/`, `docs/`)
2. 写 `core/config.py`
3. 写 `core/detector.py`
4. 写 `core/inference.py`
5. 优化 `models/microexpression_model.py`
6. 重构 `training/trainer.py`
7. 写 `training/metrics.py`
8. 改造 `interfaces/desktop/`
9. 改造 `interfaces/webapp/`
10. 改造 `main.py`
11. 改造 `data/prepare_dataset.py`
12. 更新 `utils/dataset_loader.py`
13. 扩展 `config.ini`
14. 写 `requirements.txt`
15. 清理废弃文件
16. 写文档 (`README.md`, `docs/training-guide.md`, `docs/comparison.md`, `docs/model-architecture.md`)

## 10. 成功标准

- [ ] 三个接口（Tkinter/PyQt5/Gradio）能正常启动并使用统一推理管道
- [ ] 无硬编码路径，所有路径通过 `core/config.py` 获取
- [ ] 模型参数量降至 ~3M，支持阶段式训练（stage 0/1/2）
- [ ] 无假随机 landmarks，webapp 预测结果可信
- [ ] `requirements.txt` 包含正确的依赖列表
- [ ] 废弃文件全部清理，新文档齐全
- [ ] 代码无重复，模块职责单一
