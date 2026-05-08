import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """统一人脸检测器，被所有接口共享"""

    def __init__(self, config):
        self.image_size = config.getint('preprocessing', 'image_size', 128)
        self.cascade_path = config.path('paths', 'cascade_path')
        self.predictor_path = config.path('paths', 'predictor_path')
        self._face_cascade = None
        self._dlib_detector = None
        self._dlib_predictor = None
        self._init_detectors()

    def _init_detectors(self):
        """初始化检测器，优先 dlib"""
        # OpenCV cascade（始终加载作为基础检测）
        if os.path.exists(self.cascade_path):
            self._face_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self._face_cascade.empty():
                logger.error(f"CascadeClassifier 加载失败: {self.cascade_path}")
                self._face_cascade = None
        else:
            logger.warning(f"Cascade 文件不存在: {self.cascade_path}")

        # dlib（精确 68 点 landmarks）
        try:
            import dlib
            if os.path.exists(self.predictor_path):
                self._dlib_detector = dlib.get_frontal_face_detector()
                self._dlib_predictor = dlib.shape_predictor(self.predictor_path)
                logger.info("dlib 人脸检测器初始化成功")
            else:
                logger.warning(f"dlib 关键点模型不存在: {self.predictor_path}")
        except ImportError:
            logger.warning("dlib 未安装，将使用 OpenCV fallback 检测")

    @property
    def use_dlib(self):
        return self._dlib_predictor is not None

    def detect(self, frame):
        """
        检测人脸并提取 68 点 landmarks
        Args:
            frame: BGR 图像 (numpy array)
        Returns:
            face_roi: 裁剪并 resize 后的灰度人脸 (image_size, image_size)
            landmarks: 68 点坐标，归一化到 [0,1]，shape=(68,2)
            face_rect: 人脸在原图中的矩形 (x, y, w, h)，未检测到返回 None
        """
        if frame is None or frame.size == 0:
            return None, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 检测人脸
        faces = self._detect_faces(gray)
        if len(faces) == 0:
            return None, None, None

        x, y, w, h = faces[0]
        # 扩展裁剪区域（留边距）
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face_roi = gray[y1:y2, x1:x2]
        face_roi = cv2.resize(face_roi, (self.image_size, self.image_size))
        face_roi = face_roi.astype(np.float32) / 255.0

        # 提取 landmarks
        landmarks = self._extract_landmarks(gray, x, y, w, h)
        return face_roi, landmarks, (x, y, w, h)

    def _detect_faces(self, gray):
        """检测人脸，返回矩形列表"""
        if self._face_cascade is not None:
            faces = self._face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                return faces
        return np.array([])

    def _extract_landmarks(self, gray, x, y, w, h):
        """提取 68 点 landmarks，归一化到 [0,1]"""
        if self._dlib_predictor is not None:
            import dlib
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = self._dlib_predictor(gray, rect)
            landmarks = np.zeros((68, 2), dtype=np.float32)
            for i in range(68):
                pt = shape.part(i)
                landmarks[i] = [pt.x / self.image_size, pt.y / self.image_size]
            return landmarks
        else:
            # Fallback: 使用人脸中心作为粗略估计
            # 在人脸框内均匀分布 68 个点
            landmarks = np.zeros((68, 2), dtype=np.float32)
            cx = (x + w / 2) / self.image_size
            cy = (y + h / 2) / self.image_size
            for i in range(68):
                angle = 2 * np.pi * i / 68
                rx = (w / 2) / self.image_size * 0.8
                ry = (h / 2) / self.image_size * 0.8
                landmarks[i] = [cx + rx * np.cos(angle), cy + ry * np.sin(angle)]
            return landmarks

    def detect_faces(self, frame):
        """
        检测图像中所有人脸
        Returns:
            list of (x, y, w, h)
        """
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        faces = self._detect_faces(gray)
        return [tuple(f) for f in faces] if len(faces) > 0 else []
