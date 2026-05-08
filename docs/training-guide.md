# 训练指南

## 准备工作

### 1. 数据准备

确保 CASME2 原始数据放置在 `data/CASME2-RAW/` 目录下，结构为：
```
data/CASME2-RAW/
├── sub01/
│   ├── EP02_01f/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── EP02_02f/
│       └── ...
├── sub02/
│   └── ...
└── ...
```

同时需要 CASME2 标注文件：`data/CASME2-coding-20140508.xlsx`

### 2. 运行数据预处理

```bash
python data/prepare_dataset.py
```

该步骤会：
1. 分析类别分布，排除样本最少的 2 个类别（fear、sadness）
2. 提取所有帧的人脸关键点（68 点）和光流特征
3. 创建 32 帧序列样本（智能选择起始帧，以 apex 帧为中心）
4. 数据增强（水平翻转）
5. 分割训练/测试集（80/20）
6. 生成 `cls_train.txt`、`cls_test.txt`、`class_mapping.txt`

## 快速验证（推荐首次使用）

在正式训练前，用少量数据确认 pipeline 正常：

```bash
python training/quick_test.py
```

该脚本自动抽取每类 50 个训练样本 + 20 个测试样本，训练 5 个 epoch。如果运行成功并输出"测试通过"，说明数据加载和模型推理均正常，可以进行完整训练。

## 标准训练

### 配置

编辑 `config.ini` 的 `[training]` 部分：

```ini
[training]
stage = 0              ; 标准训练
batch_size = 8
num_epochs = 30
learning_rate = 0.0001
early_stopping_patience = 10
train_split = 0.8
```

### 运行

```bash
python training/trainer.py
```

### 输出

训练结果保存在 `models/weights/MicroExpModel_<时间戳>_<准确率>/` 目录下：
- `best_model.pth` — 最佳模型权重
- `training.log` — 训练日志
- `results.json` — 评估结果（JSON 格式）
- `loss_curve.png` — 损失曲线
- `accuracy_curve.png` — 准确率曲线
- `f1_curve.png` — F1 分数曲线
- `confusion_matrix.png` — 混淆矩阵
- `training_overview.png` — 综合概览
- `metrics_comparison.png` — 指标对比

## 阶段式训练（推荐）

### 第一步：训练 CNN 特征提取器

修改 `config.ini`：
```ini
[training]
stage = 1
stage1_learning_rate = 0.001
stage1_epochs = 25
```

运行：
```bash
python training/trainer.py
```

此阶段冻结 LSTM 和分类器，只训练图像/光流/关键点特征提取器。完成后会在模型目录生成 `stage1_best.pth`。

### 第二步：联合微调

修改 `config.ini`：
```ini
[training]
stage = 2
stage1_checkpoint = models/weights/MicroExpModel_<时间戳>_<准确率>/stage1_best.pth
learning_rate = 0.0001
num_epochs = 30
```

运行：
```bash
python training/trainer.py
```

此阶段加载阶段 1 的权重，解冻全部参数，联合训练。

## 超参数调优建议

| 参数 | 建议范围 | 说明 |
|------|----------|------|
| batch_size | 4-16 | 显存不够时减小 |
| learning_rate | 1e-4 ~ 1e-3 | 阶段 1 可用较大值 |
| num_epochs | 10-50 | 配合 early stopping |
| early_stopping_patience | 5-15 | 验证集不改善的容忍轮数 |
| stage1_epochs | 15-30 | CNN 预训练轮数 |

## 评估

训练完成后，查看以下文件了解模型表现：

1. **分类报告** — `results.json` 中的 `classification_report` 字段
2. **混淆矩阵** — `confusion_matrix.png`
3. **训练曲线** — `loss_curve.png`、`accuracy_curve.png`
4. **过拟合检测** — `metrics_comparison.png` 中的 Overfitting Detection 子图

## 使用训练好的模型

训练好的模型会被自动识别。启动任一界面即可使用：

```bash
# 桌面界面
python interfaces/desktop/gui_main.py

# Web 界面
python interfaces/webapp/gradio_app.py
```

系统会自动查找 `models/weights/` 下最新的 `best_model.pth`。

### 切换模型

如果训练了多个模型，可以通过启动器界面切换：

1. 启动 `python main.py`
2. 点击"切换模型"按钮
3. 选择要使用的模型
4. 重新启动界面即可生效

切换操作会更新 `config.ini` 中的 `[model] selected_path` 配置，所有界面下次启动时自动读取。
