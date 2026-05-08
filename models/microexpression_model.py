import torch
import torch.nn as nn


class MicroExpressionModel(nn.Module):
    """CNN+LSTM 微表情识别模型（优化版：GAP + BN + 双向 LSTM）"""

    def __init__(self, num_classes, sequence_length=32):
        super(MicroExpressionModel, self).__init__()
        self.sequence_length = sequence_length

        # 图像特征提取器: Conv+BN+ReLU+Pool × 3 → GAP
        self.image_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # (B*T, 32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # (B*T, 64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # (B*T, 128, 16, 16)
        )
        self.image_pool = nn.AdaptiveAvgPool2d(1)  # (B*T, 128, 1, 1)

        # 光流特征提取器: Conv+BN+ReLU+Pool × 3 → GAP
        self.flow_cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flow_pool = nn.AdaptiveAvgPool2d(1)

        # 关键点特征处理: FC + BN + ReLU
        self.landmark_fc = nn.Sequential(
            nn.Linear(68 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 拼接后特征维度: 128 + 128 + 128 = 384
        combined_size = 128 + 128 + 128

        # 双向 LSTM
        self.lstm = nn.LSTM(
            combined_size, 256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # 分类器（双向输出 256*2=512）
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, landmarks, flows):
        """
        Args:
            images:   (B, T, 1, H, W)  灰度图像序列
            landmarks: (B, T, 136)      关键点序列
            flows:    (B, T-1, 2, H, W) 光流序列
        """
        batch_size, T, C_img, H, W = images.size()
        B_flow, T_flow, C_flow, H_flow, W_flow = flows.size()

        # 图像流: (B*T, 1, H, W) → CNN → GAP → (B, T, 128)
        images = images.view(batch_size * T, C_img, H, W)
        img_feat = self.image_cnn(images)
        img_feat = self.image_pool(img_feat).view(batch_size, T, -1)  # (B, T, 128)

        # 光流流: (B*T_flow, 2, H, W) → CNN → GAP → (B, T_flow, 128)
        flows = flows.view(B_flow * T_flow, C_flow, H_flow, W_flow)
        flow_feat = self.flow_cnn(flows)
        flow_feat = self.flow_pool(flow_feat).view(batch_size, T_flow, -1)  # (B, T_flow, 128)

        # 关键点流: (B, T, 136) → FC → (B, T, 128)
        # BatchNorm1d 需要 (B*N, 128) 输入
        landmarks_flat = landmarks.view(batch_size * T, -1)
        lm_feat = self.landmark_fc(landmarks_flat).view(batch_size, T, -1)  # (B, T, 128)

        # 对齐时间维度：截取到 T_flow
        combined = torch.cat((
            img_feat[:, :T_flow, :],
            lm_feat[:, :T_flow, :],
            flow_feat
        ), dim=2)  # (B, T_flow, 384)

        # LSTM
        lstm_out, _ = self.lstm(combined)  # (B, T_flow, 512)

        # 取最后时间步
        last_hidden = lstm_out[:, -1, :]  # (B, 512)

        return self.classifier(last_hidden)


def freeze_backbone(model):
    """冻结图像/光流/关键点特征提取器"""
    for name, param in model.named_parameters():
        if 'lstm' not in name and 'classifier' not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """解冻全部参数"""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """统计全部参数量"""
    return sum(p.numel() for p in model.parameters())
