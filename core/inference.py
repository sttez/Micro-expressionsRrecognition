import torch
import numpy as np
import cv2
import os
import logging
from collections import deque

from core.config import Config
from core.detector import FaceDetector
from models.microexpression_model import MicroExpressionModel

logger = logging.getLogger(__name__)

LABELS = ['surprise', 'repression', 'happiness', 'disgust', 'others']
CHINESE_LABELS = ['惊讶', '压抑', '高兴', '厌恶', '其他']


class InferenceEngine:
    """统一推理管道，被所有接口共享"""

    def __init__(self, config=None, model_path=None):
        if config is None:
            config = Config()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = config.getint('preprocessing', 'sequence_length', 32)
        self.image_size = config.getint('preprocessing', 'image_size', 128)
        self.num_classes = config.get_num_classes()
        self.labels = config.get_class_names() if config.get_class_names() else LABELS
        self.detector = FaceDetector(config)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path=None):
        """加载模型"""
        model = MicroExpressionModel(
            num_classes=self.num_classes,
            sequence_length=self.sequence_length
        ).to(self.device)

        if model_path is None:
            model_path = self.config.get_model_path()
        if model_path is None:
            logger.warning("未找到已训练模型，使用随机初始化参数")
            model.eval()
            return model

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")

        model.eval()
        return model

    def switch_model(self, model_path):
        """切换到指定模型"""
        self.model = self._load_model(model_path)

    def predict_image(self, image):
        """
        单张图片推理
        Args:
            image: BGR 图像 (numpy array) 或文件路径 (str)
        Returns:
            label: 预测类别名称
            probs: 各类别概率数组
            face_rect: 人脸矩形 (x, y, w, h)
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return None, np.zeros(self.num_classes), None

        face, landmarks, face_rect = self.detector.detect(image)
        if face is None:
            return None, np.zeros(self.num_classes), None

        # 构建序列（单帧重复）
        images_seq = np.stack([face] * self.sequence_length)[:, np.newaxis, :, :]
        landmarks_seq = np.stack([landmarks.flatten()] * self.sequence_length)
        flow_seq = np.zeros((self.sequence_length - 1, 2, self.image_size, self.image_size), dtype=np.float32)

        return self._infer(images_seq, landmarks_seq, flow_seq)

    def predict_frame(self, frame, frame_buffer, landmark_buffer, gray_buffer):
        """
        实时视频帧推理
        Args:
            frame: 当前 BGR 帧
            frame_buffer: deque of preprocessed face images
            landmark_buffer: deque of landmark arrays
            gray_buffer: deque of grayscale images
        Returns:
            label, probs, face_rect
        """
        face, landmarks, face_rect = self.detector.detect(frame)
        if face is None:
            return None, np.zeros(self.num_classes), None

        gray = cv2.cvtColor(
            cv2.resize(frame, (self.image_size, self.image_size)),
            cv2.COLOR_BGR2GRAY
        )

        frame_buffer.append(face)
        landmark_buffer.append(landmarks.flatten())
        gray_buffer.append(gray)

        if len(frame_buffer) < self.sequence_length:
            return None, np.zeros(self.num_classes), face_rect

        # 构建输入
        images_seq = np.array(list(frame_buffer)[-self.sequence_length:])[:, np.newaxis, :, :]
        landmarks_seq = np.array(list(landmark_buffer)[-self.sequence_length:])

        # 计算光流
        gray_list = list(gray_buffer)[-self.sequence_length:]
        flows = []
        for i in range(1, len(gray_list)):
            flow = cv2.calcOpticalFlowFarneback(
                gray_list[i - 1], gray_list[i], None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(np.transpose(np.clip(flow / 20.0, -1, 1), (2, 0, 1)))
        flow_seq = np.array(flows)

        return self._infer(images_seq, landmarks_seq, flow_seq)

    def predict_video(self, video_path):
        """
        逐帧视频推理
        Args:
            video_path: 视频文件路径
        Returns:
            results: list of (frame_idx, label, prob, face_rect)
            fps: 视频帧率
            total_frames: 总帧数
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0, 0

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_buffer = deque(maxlen=self.sequence_length)
        landmark_buffer = deque(maxlen=self.sequence_length)
        gray_buffer = deque(maxlen=self.sequence_length)

        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face, landmarks, face_rect = self.detector.detect(frame)
            if face is not None:
                gray = cv2.cvtColor(
                    cv2.resize(frame, (self.image_size, self.image_size)),
                    cv2.COLOR_BGR2GRAY
                )
                frame_buffer.append(face)
                landmark_buffer.append(landmarks.flatten())
                gray_buffer.append(gray)

                if len(frame_buffer) >= self.sequence_length:
                    images_seq = np.array(list(frame_buffer))[:, np.newaxis, :, :]
                    landmarks_seq = np.array(list(landmark_buffer))
                    gray_list = list(gray_buffer)
                    flows = []
                    for i in range(1, len(gray_list)):
                        flow = cv2.calcOpticalFlowFarneback(
                            gray_list[i - 1], gray_list[i], None,
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        flows.append(np.transpose(np.clip(flow / 20.0, -1, 1), (2, 0, 1)))
                    flow_seq = np.array(flows)

                    label, probs, _ = self._infer(images_seq, landmarks_seq, flow_seq)
                    results.append((frame_idx, label, probs, face_rect))

            frame_idx += 1

        cap.release()
        return results, fps, total_frames

    def _infer(self, images_seq, landmarks_seq, flow_seq):
        """模型前向推理"""
        images_t = torch.tensor(images_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        landmarks_t = torch.tensor(landmarks_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        flows_t = torch.tensor(flow_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(images_t, landmarks_t, flows_t)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        max_idx = np.argmax(probs)
        label = self.labels[max_idx]
        return label, probs, None

    def create_frame_buffers(self):
        """创建帧缓冲区，供实时推理使用"""
        return (
            deque(maxlen=self.sequence_length),
            deque(maxlen=self.sequence_length),
            deque(maxlen=self.sequence_length),
        )
