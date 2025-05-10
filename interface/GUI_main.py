# -*- coding: utf-8 -*-

# Fix encoding issues first - 增强的编码修复部分
import sys
import locale
import os
import ctypes


# Regular imports
import torch
import cv2
import numpy as np
import dlib
import traceback
import urllib.request
from collections import deque

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from interface.Ui_Emo_gui2 import Ui_MainWindow
from models.microexpression_model import MicroExpressionModel
from PIL import Image, ImageDraw, ImageFont

"""
微表情识别 GUI - 完整修复版
================================
* 适配 MicroExpressionModel (CNN+LSTM)
* 处理32帧序列输入
* 直接显示5类情绪识别
* 解决模型加载兼容性问题
* 修复Windows系统编码问题
* 自动下载缺失的cascade文件
"""

# -------------------------- 基本配置 --------------------------
# 5个情绪类别（与训练时的顺序保持一致）
LABELS = [
    'surprise',  # 惊讶
    'repression',  # 压抑
    'happiness',  # 高兴
    'disgust',  # 厌恶
    'others'  # 其他/中性
]

# 中文显示标签
CHINESE_LABELS = [
    '惊讶',
    '压抑',
    '高兴',
    '厌恶',
    '其他'
]

# 模型路径
MODEL_PATH = r"D:\college\MicroExpressionRecognizer\models\weights\MicroExpModel_20250509_133707_temp\best_model.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 检测器路径
FACE_CASCADE_PATH = r'D:\college\MicroExpressionRecognizer\utils\haarcascade_frontalface_default.xml'
LANDMARK_MODEL_PATH = r'D:\college\MicroExpressionRecognizer\utils\shape_predictor_68_face_landmarks.dat'

# 序列参数
SEQUENCE_LENGTH = 32
IMAGE_SIZE = 128
FRAME_BUFFER_SIZE = 32  # 缓存帧数


# -------------------------- 辅助函数 --------------------------
def download_file(url, file_path):
    """Download a file from URL if it doesn't exist"""
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print(f"正在下载文件到 {file_path}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print("下载完成!")
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            print(f"请手动从以下地址下载:")
            print(url)
            print(f"并保存到: {file_path}")
            return False
    return True


def check_and_download_dependencies():
    """Check and download missing dependency files"""
    # Download face cascade if missing
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    if not download_file(cascade_url, FACE_CASCADE_PATH):
        raise FileNotFoundError(f"无法获取face cascade文件: {FACE_CASCADE_PATH}")

    # Check if landmark model exists
    if not os.path.exists(LANDMARK_MODEL_PATH):
        print(f"警告: 关键点检测器文件不存在: {LANDMARK_MODEL_PATH}")
        print("请下载 shape_predictor_68_face_landmarks.dat 文件")
        print("下载地址: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("解压后放置到: utils/shape_predictor_68_face_landmarks.dat")


def safe_print(message):
    """安全的打印函数，处理Windows系统的编码问题"""
    try:
        print(message)
    except UnicodeEncodeError:
        try:
            print(message.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
        except:
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            print(safe_message)


def load_model_flexible(model_path, device, num_classes, sequence_length):
    """
    灵活的模型加载函数，可以处理不同的保存格式
    """
    model = MicroExpressionModel(num_classes=num_classes, sequence_length=sequence_length).to(device)

    try:
        # 首先尝试使用weights_only=True安全加载
        safe_print("尝试安全加载模型...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        load_success = True
    except Exception as e:
        safe_print(f"安全加载失败: {e}")
        safe_print("尝试常规加载...")
        try:
            # 如果失败，使用常规加载方式
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            load_success = True
        except Exception as e2:
            safe_print(f"常规加载也失败: {e2}")
            load_success = False
            raise e2

    if load_success:
        # 检查加载的数据类型并相应处理
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # 如果是完整的checkpoint（包含元数据）
                model.load_state_dict(checkpoint['model_state_dict'])
                safe_print("成功加载完整checkpoint")
                safe_print("训练信息:")
                safe_print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
                safe_print(f"  - 最佳验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")
                safe_print(f"  - 最佳F1分数: {checkpoint.get('best_val_f1', 'N/A')}")
            else:
                # 如果是只包含state_dict的字典
                model.load_state_dict(checkpoint)
                safe_print("成功加载state_dict")
        else:
            # 如果直接是state_dict
            model.load_state_dict(checkpoint)
            safe_print("成功加载纯state_dict")

    model.eval()
    return model


# -------------------------- 主窗体类 --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 强制设置UI字体编码
        font = self.font()
        font.setFamily("Microsoft YaHei")  # 使用微软雅黑字体以支持中文
        self.setFont(font)

        # 为所有UI元素强制使用UTF-8编码和中文友好字体
        for widget in self.findChildren(QtWidgets.QWidget):
            widget_font = widget.font()
            widget_font.setFamily("Microsoft YaHei")
            widget.setFont(widget_font)

        # 初始化进度条和标签
        self.ui.progress_bars = getattr(self.ui, 'progress_bars', [])
        self.ui.percent_labels = getattr(self.ui, 'percent_labels', [])

        # ------------ 检查并下载依赖文件 ------------
        try:
            check_and_download_dependencies()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"依赖文件检查失败: {e}")
            sys.exit(1)

        # ------------ 加载模型 ------------
        try:
            self.model = load_model_flexible(
                MODEL_PATH,
                DEVICE,
                num_classes=len(LABELS),
                sequence_length=SEQUENCE_LENGTH
            )
            safe_print("[SUCCESS] 模型加载成功！")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")
            sys.exit(1)

        # ------------ 加载检测器 ------------
        try:
            self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
            if self.face_cascade.empty():
                raise ValueError("人脸级联分类器加载失败")

            # 仅在文件存在时加载关键点检测器
            if os.path.exists(LANDMARK_MODEL_PATH):
                self.predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)
                safe_print("[SUCCESS] 人脸与关键点检测加载成功！")
            else:
                self.predictor = None
                safe_print("[WARNING] 关键点检测器未加载，将使用零值代替")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"检测模型加载失败: {e}")
            self.face_cascade = None
            self.predictor = None

        # ------------ 状态变量 ------------
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.landmark_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.gray_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

        self.cap = None
        self.current_mode = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 情绪颜色映射（BGR格式）
        self.emotion_colors = {
            'surprise': (255, 255, 0),  # 青色
            'repression': (128, 0, 128),  # 紫色
            'happiness': (0, 255, 0),  # 绿色
            'disgust': (0, 0, 255),  # 红色
            'others': (128, 128, 128),  # 灰色
        }

        # 绑定按钮事件
        self.ui.pushButton_image.clicked.connect(self.run_image_predict)
        self.ui.pushButton_video.clicked.connect(self.run_video_predict)
        self.ui.pushButton_camera.clicked.connect(self.run_camera_predict)
        self.ui.pushButton_stop.clicked.connect(self.stop_detection)

        self.update_status_bar()

    def update_status_bar(self, msg="就绪"):
        """更新状态栏信息"""
        self.statusBar().showMessage(f"{msg} | 模型: MicroExpressionModel | 类别: {len(LABELS)}类")

    def preprocess_frame(self, frame):
        """预处理单帧图像"""
        resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        return normalized, gray

    def extract_landmarks(self, img):
        """提取面部关键点"""
        if self.predictor is None:
            return np.zeros(136, dtype=np.float32)

        try:
            # 调整图像到模型尺寸
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # 转为灰度图
            if len(img_resized.shape) == 3:
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_resized

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                # 使用第一个检测到的人脸
                x, y, w, h = faces[0]
                # 创建dlib的矩形对象
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                # 获取关键点
                shape = self.predictor(gray, rect)

                # 提取关键点并归一化
                landmarks = []
                for i in range(68):
                    pt = shape.part(i)
                    # 归一化坐标到[0, 1.txt]
                    normalized_x = pt.x / IMAGE_SIZE
                    normalized_y = pt.y / IMAGE_SIZE
                    landmarks.extend([normalized_x, normalized_y])

                return np.array(landmarks, dtype=np.float32)
            else:
                return np.zeros(136, dtype=np.float32)

        except Exception as e:
            safe_print(f"关键点提取失败: {e}")
            return np.zeros(136, dtype=np.float32)

    def compute_optical_flow(self, prev_gray, curr_gray):
        """计算光流特征"""
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            # 归一化光流
            flow = np.clip(flow / 20.0, -1, 1)
            return flow
        except Exception as e:
            safe_print(f"光流计算失败: {e}")
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2), dtype=np.float32)

    def predict_sequence(self):
        """对缓存的序列进行预测"""
        if len(self.frame_buffer) < SEQUENCE_LENGTH:
            # 序列不足，返回空结果
            return None, np.zeros(len(LABELS))

        try:
            # 准备图像序列
            images = np.array(list(self.frame_buffer)[-SEQUENCE_LENGTH:])
            images_tensor = torch.from_numpy(images).unsqueeze(0).unsqueeze(2).float()

            # 准备关键点序列
            landmarks = np.array(list(self.landmark_buffer)[-SEQUENCE_LENGTH:])
            landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).float()

            # 准备光流序列（31帧）
            flows = []
            gray_list = list(self.gray_buffer)[-SEQUENCE_LENGTH:]
            for i in range(1, len(gray_list)):
                flow = self.compute_optical_flow(gray_list[i - 1], gray_list[i])
                flows.append(flow)

            flows = np.array(flows).transpose(0, 3, 1, 2)
            flows_tensor = torch.from_numpy(flows).unsqueeze(0).float()

            # 推理
            with torch.no_grad():
                images_tensor = images_tensor.to(DEVICE)
                landmarks_tensor = landmarks_tensor.to(DEVICE)
                flows_tensor = flows_tensor.to(DEVICE)

                outputs = self.model(images_tensor, landmarks_tensor, flows_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # 找到最高概率的类别
            max_idx = np.argmax(probs)
            predicted_label = LABELS[max_idx]

            return predicted_label, probs

        except Exception as e:
            traceback.print_exc()
            safe_print(f"序列预测失败: {e}")
            return None, np.zeros(len(LABELS))

    def process_frame(self, frame):
        """处理单帧并更新缓存"""
        # 预处理
        processed, gray = self.preprocess_frame(frame)

        # 提取关键点
        landmarks = self.extract_landmarks(frame)

        # 更新缓存
        self.frame_buffer.append(processed)
        self.landmark_buffer.append(landmarks)
        self.gray_buffer.append(gray)

        # 进行预测
        label, probs = self.predict_sequence()

        return label, probs

    def update_prob_bars(self, probs):
        """更新概率进度条和百分比标签"""
        try:
            for i in range(len(LABELS)):
                if i < len(self.ui.progress_bars):
                    # 将概率转换为百分比
                    prob_value = int(probs[i] * 100)
                    self.ui.progress_bars[i].setValue(prob_value)

                    if i < len(self.ui.percent_labels):
                        self.ui.percent_labels[i].setText(f"{prob_value}%")
        except Exception as e:
            safe_print(f"更新概率条失败: {e}")

    def process_and_draw(self, frame):
        """处理帧并绘制结果（支持中文）"""
        try:
            disp = frame.copy()
            label, probs = self.process_frame(frame)

            text = ""
            faces = self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
            if label is not None:
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    text_x, text_y = x, max(y - 10, 30)
                else:
                    text_x, text_y = 10, 30

                label_idx = LABELS.index(label)
                chinese_label = CHINESE_LABELS[label_idx]
                confidence = probs.max() * 100
                color = self.emotion_colors.get(label, (255, 255, 255))
                text = f"{chinese_label} ({label}) [{confidence:.1f}%]"

            else:
                text = f"正在收集帧... ({len(self.frame_buffer)}/{SEQUENCE_LENGTH})"
                text_x, text_y = 10, 30
                color = (255, 255, 0)

            # 转为RGB后再用PIL绘制中文
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(disp_rgb)
            draw = ImageDraw.Draw(pil_img)

            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)  # 微软雅黑
            except IOError:
                font = ImageFont.load_default()
                print("警告: 中文字体加载失败，使用默认字体")

            draw.text((text_x, text_y), text, font=font, fill=(color[2], color[1], color[0]))

            # 转回OpenCV图像
            disp = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            self.update_prob_bars(probs)

            rgb_image = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qt_image)

        except Exception as e:
            traceback.print_exc()
            safe_print(f"处理帧失败: {e}")
            return QPixmap()

    def update_frame(self):
        """定时器回调：更新帧"""
        if self.current_mode in ('video', 'camera') and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # 只在摄像头模式下进行镜像翻转
                    if self.current_mode == 'camera':
                        frame = cv2.flip(frame, 1)  # 水平镜像

                    pixmap = self.process_and_draw(frame)
                    self.ui.display_frame.setPixmap(pixmap)
                else:
                    self.stop_detection()
                    if self.current_mode == 'video':
                        self.update_status_bar("视频结束")
                    else:
                        self.update_status_bar("摄像头错误")
            except Exception as e:
                safe_print(f"更新帧失败: {e}")
                self.stop_detection()

    def run_image_predict(self):
        """图像预测模式"""
        try:
            path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.png *.jpg *.bmp)")
            if not path:
                return

            self.stop_detection()
            self.current_mode = 'image'

            # 清空缓存
            self.frame_buffer.clear()
            self.landmark_buffer.clear()
            self.gray_buffer.clear()

            img = cv2.imread(path)
            if img is None:
                QMessageBox.critical(self, "错误", "无法读取图像")
                return

            # 对于单张图片，我们将其复制为32帧序列
            processed, gray = self.preprocess_frame(img)
            landmarks = self.extract_landmarks(img)

            # 填充缓存
            for _ in range(SEQUENCE_LENGTH):
                self.frame_buffer.append(processed)
                self.landmark_buffer.append(landmarks)
                self.gray_buffer.append(gray)

            # 处理并显示
            pixmap = self.process_and_draw(img)
            self.ui.display_frame.setPixmap(pixmap)

            self.update_status_bar(f"已加载图像: {os.path.basename(path)}")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"图像处理失败: {e}")

    def run_video_predict(self):
        """视频预测模式"""
        try:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
            if not path:
                return

            self.stop_detection()
            self.current_mode = 'video'

            # 清空缓存
            self.frame_buffer.clear()
            self.landmark_buffer.clear()
            self.gray_buffer.clear()

            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开视频文件")
                return

            self.timer.start(30)  # 约33 FPS
            self.update_status_bar(f"已加载视频: {os.path.basename(path)}")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"视频处理失败: {e}")

    def run_camera_predict(self):
        """摄像头预测模式"""
        try:
            self.stop_detection()
            self.current_mode = 'camera'

            # 清空缓存
            self.frame_buffer.clear()
            self.landmark_buffer.clear()
            self.gray_buffer.clear()

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头")
                return

            self.timer.start(30)
            self.update_status_bar("摄像头模式")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"摄像头启动失败: {e}")

    def stop_detection(self):
        """停止检测"""
        try:
            if self.timer.isActive():
                self.timer.stop()

            if self.cap and self.cap.isOpened():
                self.cap.release()

            self.cap = None
            self.current_mode = None

            # 清空缓存
            self.frame_buffer.clear()
            self.landmark_buffer.clear()
            self.gray_buffer.clear()

            self.update_status_bar("已停止")

        except Exception as e:
            safe_print(f"停止检测失败: {e}")


# -------------------------- 主函数 --------------------------
def main():
    """程序入口点"""
    app = QtWidgets.QApplication(sys.argv)

    # 为整个应用设置默认字体
    font = app.font()
    font.setFamily("Microsoft YaHei")
    app.setFont(font)

    # 设置全局样式表确保中文正确显示
    app.setStyleSheet("""
        * {
            font-family: "Microsoft YaHei", "SimHei", "sans-serif";
        }
    """)

    # 确保Qt使用UTF-8解析文本
    QtCore.QTextCodec.setCodecForLocale(QtCore.QTextCodec.codecForName("UTF-8"))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":

    main()