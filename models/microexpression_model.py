import torch
import torch.nn as nn

# 定义一个简单的模型示例 (CNN + LSTM)
class MicroExpressionModel(nn.Module):
    def __init__(self, num_classes, sequence_length=32):
        super(MicroExpressionModel, self).__init__()
        self.sequence_length = sequence_length

        # 图像特征提取器 (简单的CNN)
        # 输入: (T, C, H, W) -> (T, 1.txt, 128, 128)
        # 假设输入图像是灰度图，通道数为 1.txt
        self.image_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: (T, 32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: (T, 64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 输出: (T, 128, 16, 16)
        )
        # 计算CNN输出展平后的特征维度
        self._cnn_output_size = 128 * 16 * 16

        # 关键点特征处理 (简单的全连接层)
        # 输入: (T, 关键点特征维度) -> (T, 68*2)
        self.landmark_fc = nn.Sequential(
            nn.Linear(68 * 2, 128),
            nn.ReLU()
        )
        self._landmark_output_size = 128


        # 光流特征提取器 (简单的CNN)
        # 输入: (T', C, H, W) -> (T', 2, 128, 128)
        # 假设输入光流有两个通道 (dx, dy)
        self.flow_cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: (T', 32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: (T', 64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 输出: (T', 128, 16, 16)
        )
        self._flow_cnn_output_size = 128 * 16 * 16


        # 将图像CNN输出、关键点FC输出、光流CNN输出合并
        # 注意：光流序列比图像和关键点序列短一帧。
        # 这里采用一个简单的方法：将图像CNN输出和关键点FC输出在时间维度上与光流CNN输出对齐 (忽略图像/关键点序列的最后一帧)。
        self._combined_feature_size = self._cnn_output_size + self._landmark_output_size + self._flow_cnn_output_size


        # 双层 LSTM 处理序列特征
        # 输入: (T', 特征维度)
        # batch_first=True 表示输入张量形状为 (batch, sequence, feature)
        self.lstm = nn.LSTM(self._combined_feature_size, 256, num_layers=2, batch_first=True, bidirectional=False) # 可以尝试双向LSTM

        # 分类器
        # 输入: LSTM的最后一个时间步的隐藏状态
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # 添加Dropout防止过拟合
            nn.Linear(128, num_classes)
        )

    def forward(self, images, landmarks, flows):
        # images: (B, T, C_img, H, W)
        # landmarks: (B, T, landmark_feature_size)
        # flows: (B, T', C_flow, H, W)
        batch_size, T, C_img, H, W = images.size() # 图像序列尺寸
        B_flow, T_flow, C_flow, H_flow, W_flow = flows.size() # 光流序列尺寸

        # 处理图像序列 (将时间维度和批量维度合并，然后分开)
        images = images.view(batch_size * T, C_img, H, W) # (B*T, C, H, W)
        image_features = self.image_cnn(images) # (B*T, feature_size)
        image_features = image_features.view(batch_size, T, -1) # (B, T, feature_size)

        # 处理关键点序列
        # 输入: (B, T, landmark_feature_size)
        landmark_features = self.landmark_fc(landmarks) # (B, T, landmark_output_size)


        # 处理光流序列 (将时间维度和批量维度合并，然后分开)
        flows = flows.view(B_flow * T_flow, C_flow, H_flow, W_flow) # (B*T', C, H, W)
        flow_features = self.flow_cnn(flows) # (B*T', flow_feature_size)
        flow_features = flow_features.reshape(batch_size, T_flow, -1)# (B, T', flow_feature_size)


        # 合并特征 (对齐时间维度)
        # 忽略图像和关键点序列的最后一帧，使其与光流序列长度一致 (T' = T - 1.txt)
        combined_features = torch.cat((image_features[:, :T_flow, :],
                                       landmark_features[:, :T_flow, :],
                                       flow_features), dim=2) # (B, T', combined_feature_size)


        # 将合并后的特征输入到 LSTM
        # 输入形状: (B, T', combined_feature_size)
        # 输出形状: (B, T', hidden_size)
        lstm_out, _ = self.lstm(combined_features)

        # 取 LSTM 最后一个时间步的输出用于分类
        # 形状: (B, hidden_size)
        lstm_last_hidden_state = lstm_out[:, -1, :]

        # 输入到分类器
        # 形状: (B, num_classes)
        output = self.classifier(lstm_last_hidden_state)

        return output

