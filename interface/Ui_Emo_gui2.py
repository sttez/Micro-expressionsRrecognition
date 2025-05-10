import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QProgressBar, QLabel, QFrame, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 700)  # 调整窗口大小
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 创建水平布局作为主布局
        self.main_layout = QHBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # 左侧区域 - 视频显示
        self.left_frame = QFrame(self.centralwidget)
        self.left_frame.setFrameShape(QFrame.StyledPanel)
        self.left_frame.setMinimumWidth(640)  # 确保视频区域足够宽
        self.left_layout = QVBoxLayout(self.left_frame)

        # 视频显示区域
        self.display_frame = QtWidgets.QLabel(self.left_frame)
        self.display_frame.setMinimumSize(640, 480)
        self.display_frame.setAlignment(Qt.AlignCenter)
        self.display_frame.setStyleSheet("background-color: #e0e0e0;")
        self.display_frame.setText("视频显示区域")
        self.left_layout.addWidget(self.display_frame)

        # 按钮区域
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(10)

        self.pushButton_image = QtWidgets.QPushButton(self.left_frame)
        self.pushButton_image.setObjectName("pushButton_image")
        self.pushButton_image.setText("图像检测")
        self.button_layout.addWidget(self.pushButton_image)

        self.pushButton_video = QtWidgets.QPushButton(self.left_frame)
        self.pushButton_video.setObjectName("pushButton_video")
        self.pushButton_video.setText("视频检测")
        self.button_layout.addWidget(self.pushButton_video)

        self.pushButton_camera = QtWidgets.QPushButton(self.left_frame)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.pushButton_camera.setText("摄像头检测")
        self.button_layout.addWidget(self.pushButton_camera)

        self.pushButton_stop = QtWidgets.QPushButton(self.left_frame)
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.pushButton_stop.setText("停止")
        self.button_layout.addWidget(self.pushButton_stop)

        self.left_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.left_frame, 7)  # 左侧区域占70%宽度

        # 右侧区域 - 表情标签和进度条（现在只有5个）
        self.right_frame = QFrame(self.centralwidget)
        self.right_frame.setFrameShape(QFrame.StyledPanel)
        self.right_frame.setMaximumWidth(300)  # 限制右侧区域宽度
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(10, 10, 10, 10)  # 增加一些内边距
        self.right_layout.setSpacing(10)  # 增加间距，因为现在只有5个元素

        # 添加标题
        title_label = QLabel("情绪识别结果", self.right_frame)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        self.right_layout.addWidget(title_label)

        # 表情标签和进度条（现在只有5个）
        self.labels = [
            '惊讶 (surprise)',
            '压抑 (repression)',
            '高兴 (happiness)',
            '厌恶 (disgust)',
            '其他 (others)'
        ]

        self.progress_bars = []
        self.percent_labels = []

        for label in self.labels:
            # 创建水平布局
            hbox = QHBoxLayout()
            hbox.setSpacing(5)

            # 添加表情标签
            emotion_label = QLabel(label, self.right_frame)
            emotion_label.setMinimumWidth(110)
            emotion_label.setMaximumWidth(110)  # 稍微增加宽度以适应中文
            emotion_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            emotion_label.setStyleSheet("font-size: 14px;")  # 稍微增大字体
            hbox.addWidget(emotion_label)

            # 添加进度条
            progress_bar = QProgressBar(self.right_frame)
            progress_bar.setMinimumWidth(100)
            progress_bar.setMaximumHeight(25)  # 增加进度条高度
            progress_bar.setValue(0)
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #5cb85c;
                    border-radius: 4px;
                }
            """)
            self.progress_bars.append(progress_bar)
            hbox.addWidget(progress_bar)

            # 添加百分比标签
            percent_label = QLabel("0%", self.right_frame)
            percent_label.setMinimumWidth(50)
            percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            percent_label.setStyleSheet("font-size: 14px; font-weight: bold;")
            self.percent_labels.append(percent_label)
            hbox.addWidget(percent_label)

            self.right_layout.addLayout(hbox)

        # 添加说明文字
        info_label = QLabel("注：实时显示当前检测到的表情概率分布", self.right_frame)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 12px; color: #666; margin-top: 20px;")
        info_label.setWordWrap(True)
        self.right_layout.addWidget(info_label)

        # 添加一个弹性空间，使内容居中显示
        self.right_layout.addStretch(1)

        self.main_layout.addWidget(self.right_frame, 3)  # 右侧区域占30%宽度

        # 设置中央部件
        MainWindow.setCentralWidget(self.centralwidget)

        # 状态栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # 设置窗口标题
        MainWindow.setWindowTitle("微表情识别系统")


# 测试UI布局
if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())