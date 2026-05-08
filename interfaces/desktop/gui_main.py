import sys
import os
import cv2
import numpy as np
import traceback
from collections import deque

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from interfaces.desktop.ui_layout import Ui_MainWindow
from core.config import Config
from core.inference import InferenceEngine
from PIL import Image, ImageDraw, ImageFont

LABELS = ['surprise', 'repression', 'happiness', 'disgust', 'others']
CHINESE_LABELS = ['惊讶', '压抑', '高兴', '厌恶', '其他']
SEQUENCE_LENGTH = 32


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 设置字体
        font = self.font()
        font.setFamily("Microsoft YaHei")
        self.setFont(font)
        for widget in self.findChildren(QtWidgets.QWidget):
            wfont = widget.font()
            wfont.setFamily("Microsoft YaHei")
            widget.setFont(wfont)

        # 初始化推理引擎
        try:
            self.engine = InferenceEngine(self.config)
            self.current_model_path = self.config.get_model_path()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理引擎初始化失败: {e}")
            sys.exit(1)

        # 状态变量
        self.frame_buffer, self.landmark_buffer, self.gray_buffer = self.engine.create_frame_buffers()
        self.cap = None
        self.current_mode = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.emotion_colors = {
            'surprise': (255, 255, 0),
            'repression': (128, 0, 128),
            'happiness': (0, 255, 0),
            'disgust': (0, 0, 255),
            'others': (128, 128, 128),
        }

        self.font_path = self.config.get_font_path()

        # 添加菜单栏
        menubar = self.menuBar()
        model_menu = menubar.addMenu('模型')
        switch_action = model_menu.addAction('切换模型')
        switch_action.triggered.connect(self.switch_model)
        list_action = model_menu.addAction('查看可用模型')
        list_action.triggered.connect(self.show_model_list)

        # 绑定按钮
        self.ui.pushButton_image.clicked.connect(self.run_image_predict)
        self.ui.pushButton_video.clicked.connect(self.run_video_predict)
        self.ui.pushButton_camera.clicked.connect(self.run_camera_predict)
        self.ui.pushButton_stop.clicked.connect(self.stop_detection)

        self.update_status_bar()

    def update_status_bar(self, msg="就绪"):
        model_name = os.path.basename(os.path.dirname(self.current_model_path)) if self.current_model_path else "无模型"
        self.statusBar().showMessage(f"{msg} | 模型: {model_name} | {len(LABELS)}类")

    def show_model_list(self):
        """显示可用模型列表"""
        models = self.config.list_models()
        if not models:
            QMessageBox.information(self, "模型列表", "暂无已训练的模型")
            return
        text = "可用模型（按时间排序）:\n\n"
        for i, (name, path, acc) in enumerate(models, 1):
            marker = " *" if path == self.current_model_path else ""
            text += f"{i}. {name}\n   准确率: {acc}{marker}\n\n"
        text += "* = 当前使用"
        QMessageBox.information(self, "模型列表", text)

    def switch_model(self):
        """切换模型"""
        models = self.config.list_models()
        if not models:
            QMessageBox.information(self, "提示", "暂无已训练的模型，请先训练")
            return

        # 弹出选择对话框
        model_names = [f"{name} ({acc})" for name, path, acc in models]
        item, ok = QtWidgets.QInputDialog.getItem(self, "切换模型", "选择模型:", model_names, 0, False)
        if ok and item:
            idx = model_names.index(item)
            name, path, acc = models[idx]
            try:
                self.engine.switch_model(path)
                self.current_model_path = path
                self.update_status_bar(f"已切换到: {name}")
                QMessageBox.information(self, "成功", f"已切换到模型:\n{name}\n准确率: {acc}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"切换失败: {e}")

    def update_prob_bars(self, probs):
        try:
            for i in range(len(LABELS)):
                if i < len(self.ui.progress_bars):
                    val = int(probs[i] * 100)
                    self.ui.progress_bars[i].setValue(val)
                    if i < len(self.ui.percent_labels):
                        self.ui.percent_labels[i].setText(f"{val}%")
        except Exception:
            pass

    def draw_results(self, frame, label, probs, face_rect):
        """在帧上绘制识别结果（支持中文）"""
        disp = frame.copy()
        text = ""

        if label is not None and face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text_x, text_y = x, max(y - 10, 30)
            idx = LABELS.index(label) if label in LABELS else 0
            cn_label = CHINESE_LABELS[idx]
            confidence = probs.max() * 100
            color = self.emotion_colors.get(label, (255, 255, 255))
            text = f"{cn_label} ({label}) [{confidence:.1f}%]"
        else:
            text = f"正在收集帧... ({len(self.frame_buffer)}/{SEQUENCE_LENGTH})"
            text_x, text_y = 10, 30
            color = (255, 255, 0)

        # PIL 绘制中文
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(disp_rgb)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(self.font_path, 20) if self.font_path else ImageFont.load_default()
        except IOError:
            font = ImageFont.load_default()
        draw.text((text_x, text_y), text, font=font, fill=(color[2], color[1], color[0]))

        disp = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        self.update_prob_bars(probs)

        rgb_image = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def process_and_draw(self, frame):
        label, probs, face_rect = self.engine.predict_frame(
            frame, self.frame_buffer, self.landmark_buffer, self.gray_buffer
        )
        return self.draw_results(frame, label, probs, face_rect)

    def update_frame(self):
        if self.current_mode in ('video', 'camera') and self.cap:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if self.current_mode == 'camera':
                    frame = cv2.flip(frame, 1)
                pixmap = self.process_and_draw(frame)
                self.ui.display_frame.setPixmap(pixmap)
            else:
                self.stop_detection()
                self.update_status_bar("视频结束" if self.current_mode == 'video' else "摄像头错误")

    def clear_buffers(self):
        self.frame_buffer.clear()
        self.landmark_buffer.clear()
        self.gray_buffer.clear()

    def run_image_predict(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.png *.jpg *.bmp)")
        if not path:
            return
        self.stop_detection()
        self.current_mode = 'image'
        self.clear_buffers()

        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "错误", "无法读取图像")
            return

        face, landmarks, face_rect = self.engine.detector.detect(img)
        if face is None:
            self.update_status_bar("未检测到人脸")
            return

        gray = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2GRAY)
        for _ in range(SEQUENCE_LENGTH):
            self.frame_buffer.append(face)
            self.landmark_buffer.append(landmarks.flatten())
            self.gray_buffer.append(gray)

        label, probs, _ = self.engine.predict_frame(
            img, self.frame_buffer, self.landmark_buffer, self.gray_buffer
        )
        pixmap = self.draw_results(img, label, probs, face_rect)
        self.ui.display_frame.setPixmap(pixmap)
        self.update_status_bar(f"已加载: {os.path.basename(path)}")

    def run_video_predict(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if not path:
            return
        self.stop_detection()
        self.current_mode = 'video'
        self.clear_buffers()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频")
            return
        self.timer.start(30)
        self.update_status_bar(f"视频: {os.path.basename(path)}")

    def run_camera_predict(self):
        self.stop_detection()
        self.current_mode = 'camera'
        self.clear_buffers()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return
        self.timer.start(30)
        self.update_status_bar("摄像头模式")

    def stop_detection(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.current_mode = None
        self.clear_buffers()
        self.update_status_bar("已停止")


def main():
    app = QtWidgets.QApplication(sys.argv)
    font = app.font()
    font.setFamily("Microsoft YaHei")
    app.setFont(font)
    app.setStyleSheet('* { font-family: "Microsoft YaHei", "SimHei", "sans-serif"; }')

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
