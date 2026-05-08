# 模型架构详解

## 网络结构

```
输入:
  图像序列:   (B, T, 1, 128, 128)    灰度人脸图像
  关键点序列: (B, T, 136)            68 点 × 2 坐标
  光流序列:   (B, T-1, 2, 128, 128)  光流 dx, dy

图像流:
  Conv2d(1, 32, 3) + BN + ReLU + MaxPool2d(2)    → (B*T, 32, 64, 64)
  Conv2d(32, 64, 3) + BN + ReLU + MaxPool2d(2)   → (B*T, 64, 32, 32)
  Conv2d(64, 128, 3) + BN + ReLU + MaxPool2d(2)  → (B*T, 128, 16, 16)
  AdaptiveAvgPool2d(1)                             → (B*T, 128, 1, 1)
  Reshape                                          → (B, T, 128)

光流流:
  Conv2d(2, 32, 3) + BN + ReLU + MaxPool2d(2)    → (B*(T-1), 32, 64, 64)
  Conv2d(32, 64, 3) + BN + ReLU + MaxPool2d(2)   → (B*(T-1), 64, 32, 32)
  Conv2d(64, 128, 3) + BN + ReLU + MaxPool2d(2)  → (B*(T-1), 128, 16, 16)
  AdaptiveAvgPool2d(1)                             → (B*(T-1), 128, 1, 1)
  Reshape                                          → (B, T-1, 128)

关键点流:
  Linear(136, 128) + BN + ReLU                     → (B*T, 128)
  Reshape                                          → (B, T, 128)

特征融合:
  截取图像流和关键点流到 T-1 时间步
  Concatenate [图像流, 关键点流, 光流流]          → (B, T-1, 384)

时序建模:
  BiLSTM(input=384, hidden=256, layers=2, dropout=0.3)
  输出: (B, T-1, 512)  [双向 256×2]
  取最后时间步: (B, 512)

分类:
  Linear(512, 128) + ReLU
  Dropout(0.5)
  Linear(128, num_classes)
  输出: (B, num_classes)
```

## 参数量统计

| 层 | 参数量 |
|----|--------|
| image_cnn | ~185K |
| image_pool | 0 |
| flow_cnn | ~185K |
| flow_pool | 0 |
| landmark_fc | ~17.5K |
| lstm | ~2M |
| classifier | ~83K |
| **总计** | **~3M** |

## 关键设计决策

### 1. 全局平均池化 (GAP)

原模型直接将 CNN 输出 flatten（128×16×16=32768 维）送入 LSTM，导致 LSTM 输入维度极高。使用 GAP 将空间维度压缩到 1 维，特征维度从 32768 降到 128，大幅减少参数量。

### 2. BatchNorm

在每个 Conv 和 FC 层后加 BatchNorm，好处：
- 加速收敛
- 允许使用较大学习率
- 一定程度的正则化效果

### 3. 双向 LSTM

原模型使用单向 LSTM，只能利用前文信息。双向 LSTM 可以同时利用前后文，在微表情识别中，微表情的结束阶段往往和开始阶段有重要关联。

### 4. 光流对齐

光流序列比图像序列少 1 帧（因为光流是帧间差）。模型在特征融合时将图像流和关键点流截取到与光流相同的时间长度（T-1），确保时间维度对齐。

## 输入输出格式

### 训练输入
- `images`: FloatTensor (B, 32, 1, 128, 128) — 灰度图像序列
- `landmarks`: FloatTensor (B, 32, 136) — 关键点序列
- `flows`: FloatTensor (B, 31, 2, 128, 128) — 光流序列
- `labels`: LongTensor (B,) — 类别标签

### 推理输出
- `outputs`: FloatTensor (B, 5) — 5 类 logits
- 经 softmax 后得到各类别概率
