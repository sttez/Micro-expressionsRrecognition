import sys
import os
import threading
import subprocess
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import platform
import queue
import time
from pathlib import Path
import torch

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class ProcessOutputReader(threading.Thread):
    """用于读取子进程输出的线程类"""

    def __init__(self, process, output_queue):
        threading.Thread.__init__(self)
        self.process = process
        self.output_queue = output_queue
        self.daemon = True

    def run(self):
        """读取输出"""
        try:
            # 实时读取标准输出
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.output_queue.put(line.strip())

            # 读取标准错误输出
            for line in iter(self.process.stderr.readline, ''):
                if line:
                    self.output_queue.put(f"[ERROR] {line.strip()}")
        except Exception as e:
            self.output_queue.put(f"读取输出时出错: {str(e)}")


class MicroExpressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("微表情识别系统 v1.0")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # 设置窗口最小尺寸
        self.root.minsize(800, 600)

        # 居中窗口
        self.center_window()

        # 创建消息队列用于线程间通信
        self.message_queue = queue.Queue()

        # 当前运行的进程
        self.current_process = None

        # 初始化UI
        self.setup_ui()

        # 启动消息检查
        self.check_queue()

        # 检查环境
        self.check_environment()

    def center_window(self):
        """居中窗口"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_ui(self):
        """设置UI界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 标题
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        ttk.Label(title_frame, text="微表情识别系统",
                  font=("Arial", 24, "bold")).grid(row=0, column=0)
        ttk.Label(title_frame, text="Micro-Expression Recognition System",
                  font=("Arial", 12)).grid(row=1, column=0)
        ttk.Label(title_frame, text="基于深度学习的CASME2数据集微表情识别",
                  font=("Arial", 10)).grid(row=2, column=0)

        # 左侧功能区
        left_frame = ttk.LabelFrame(main_frame, text="功能模块", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # 创建按钮框架
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 功能按钮
        self.prepare_btn = ttk.Button(button_frame, text="1.txt. 数据准备",
                                      command=self.run_data_preparation, width=25)
        self.prepare_btn.grid(row=0, column=0, pady=5, sticky=tk.W)

        self.train_btn = ttk.Button(button_frame, text="2. 模型训练",
                                    command=self.run_training, width=25)
        self.train_btn.grid(row=1, column=0, pady=5, sticky=tk.W)

        self.gui_btn = ttk.Button(button_frame, text="3. 桌面界面",
                                  command=self.run_desktop_ui, width=25)
        self.gui_btn.grid(row=2, column=0, pady=5, sticky=tk.W)

        self.web_btn = ttk.Button(button_frame, text="4. 网页界面",
                                  command=self.run_web_interface, width=25)
        self.web_btn.grid(row=3, column=0, pady=5, sticky=tk.W)

        # 系统信息和模型查看按钮
        ttk.Separator(button_frame, orient='horizontal').grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))

        self.info_btn = ttk.Button(button_frame, text="查看系统信息",
                                   command=self.show_system_info, width=25)
        self.info_btn.grid(row=5, column=0, pady=5, sticky=tk.W)

        self.models_btn = ttk.Button(button_frame, text="查看可用模型",
                                     command=self.list_available_models, width=25)
        self.models_btn.grid(row=6, column=0, pady=5, sticky=tk.W)

        # 右侧信息区
        right_frame = ttk.LabelFrame(main_frame, text="运行状态", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

        # 状态标签
        self.status_label = ttk.Label(right_frame, text="就绪", font=("Arial", 12))
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # 进度条 - 修复初始状态显示问题

        # 进度条 - 初始状态为空
        style = ttk.Style()
        style.configure("Ready.Horizontal.TProgressbar", troughcolor='white', background='green')

        self.progress = ttk.Progressbar(right_frame, mode='determinate', maximum=100)
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        # 初始状态进度条
        self.progress['value'] = 100
        # 输出文本框
        self.output_text = scrolledtext.ScrolledText(right_frame, height=20, width=50)
        self.output_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置右侧框架的行列权重
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)

        # 底部控制区
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # 停止按钮（初始禁用）
        self.stop_btn = ttk.Button(bottom_frame, text="停止当前任务",
                                   command=self.stop_current_task, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=0, padx=5)

        # 清空输出按钮
        self.clear_btn = ttk.Button(bottom_frame, text="清空输出",
                                    command=self.clear_output)
        self.clear_btn.grid(row=0, column=1, padx=5)

        # 退出按钮
        self.exit_btn = ttk.Button(bottom_frame, text="退出程序",
                                   command=self.on_closing)
        self.exit_btn.grid(row=0, column=2, padx=5)

        # 配置主框架的行权重
        main_frame.rowconfigure(1, weight=1)

    def log(self, message, level="INFO"):
        """记录日志到输出框"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}\n"
        self.output_text.insert(tk.END, formatted_message)
        self.output_text.see(tk.END)
        self.output_text.update_idletasks()

    def set_status(self, status):
        """设置状态标签"""
        self.status_label.config(text=status)
        # 根据状态设置进度条显示
        if status == "就绪":
            self.progress.configure(mode='determinate', style="Ready.Horizontal.TProgressbar")
            self.progress['value'] = 100
            self.progress.stop()  # 确保停止任何动画
        elif "正在运行" in status:
            self.progress.configure(mode='indeterminate', style="TProgressbar")
            self.progress.start(10)  # 开始动画

    def enable_buttons(self, state=True):
        """启用/禁用按钮"""
        state = tk.NORMAL if state else tk.DISABLED
        self.prepare_btn.config(state=state)
        self.train_btn.config(state=state)
        self.gui_btn.config(state=state)
        self.web_btn.config(state=state)
        self.info_btn.config(state=state)
        self.models_btn.config(state=state)
        self.stop_btn.config(state=tk.DISABLED if state == tk.NORMAL else tk.NORMAL)

    def check_queue(self):
        """检查消息队列"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.log(message)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)

    def run_in_thread(self, target, *args):
        """在新线程中运行函数"""
        thread = threading.Thread(target=target, args=args)
        thread.daemon = True
        thread.start()

    def check_environment(self):
        """检查系统环境"""
        self.log("正在检查系统环境...")

        # 检查必要的Python包
        required_packages = ['torch', 'gradio', 'cv2', 'dlib', 'numpy', 'pandas', 'sklearn']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.log(f"警告: 缺少以下Python包: {', '.join(missing_packages)}", "WARNING")
            self.log("请运行: pip install torch gradio opencv-python dlib numpy pandas scikit-learn", "WARNING")

        # 检查文件和目录
        config_path = os.path.join(current_dir, "config.ini")
        if not os.path.exists(config_path):
            self.log("警告: 配置文件不存在，将在首次运行数据准备时自动创建", "WARNING")

        model_path = os.path.join(current_dir, "models", "weights")
        if not os.path.exists(model_path):
            self.log("警告: 模型目录不存在", "WARNING")

        data_path = os.path.join(current_dir, "data")
        if not os.path.exists(data_path):
            self.log("警告: 数据目录不存在", "WARNING")

        self.log("环境检查完成")

    def run_subprocess(self, script_name, description):
        """运行子进程的通用方法"""
        try:
            self.set_status(f"正在运行: {description}")
            self.enable_buttons(False)

            script_path = os.path.join(current_dir, script_name)
            if not os.path.exists(script_path):
                self.message_queue.put(f"错误: 找不到脚本 {script_name}")
                return

            # 创建进程，确保输出被正确捕获
            self.current_process = subprocess.Popen(
                [sys.executable, "-u", script_path],  # -u 参数确保输出不被缓冲
                cwd=current_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
                env={**os.environ, 'PYTHONUNBUFFERED': '1.txt'}  # 设置环境变量确保不缓冲
            )

            # 创建输出读取器线程
            output_reader = ProcessOutputReader(self.current_process, self.message_queue)
            output_reader.start()

            # 等待进程完成
            self.current_process.wait()
            rc = self.current_process.returncode

            if rc == 0:
                self.message_queue.put(f"{description}完成")
            else:
                self.message_queue.put(f"{description}失败 (返回码: {rc})")

        except Exception as e:
            self.message_queue.put(f"运行过程中出错: {str(e)}")
        finally:
            self.current_process = None
            self.enable_buttons(True)
            self.set_status("就绪")

    def run_data_preparation(self):
        """运行数据准备"""
        self.log("开始准备数据集...")
        self.run_in_thread(self.run_subprocess, "prepare_casme2_dataset.py", "数据准备")

    def run_training(self):
        """运行模型训练"""
        config_path = os.path.join(current_dir, "config.ini")
        if not os.path.exists(config_path):
            messagebox.showerror("错误", "配置文件不存在，请先运行数据准备！")
            return

        self.log("开始训练模型...")
        self.run_in_thread(self.run_subprocess, "trainner.py", "模型训练")

    def run_desktop_ui(self):
        """运行桌面GUI"""
        self.log("启动桌面界面...")
        gui_script = os.path.join(current_dir, "interface", "GUI_main.py")

        if not os.path.exists(gui_script):
            messagebox.showerror("错误", "找不到GUI脚本！")
            return

        try:
            subprocess.Popen([sys.executable, gui_script], cwd=current_dir)
            self.log("桌面界面已启动")
        except Exception as e:
            self.log(f"启动桌面界面失败: {str(e)}", "ERROR")

    def run_web_interface(self):
        """运行Web界面 - 修复问题1：更新Gradio脚本路径"""
        self.log("启动Web界面...")

        def run_gradio():
            try:
                # 修复：Gradio脚本在webapp目录下
                gradio_script = os.path.join(current_dir, "webapp", "gradio_app.py")

                if not os.path.exists(gradio_script):
                    self.message_queue.put(f"错误: 找不到Gradio脚本！路径: {gradio_script}")
                    return

                self.message_queue.put("正在启动Gradio服务器...")
                self.current_process = subprocess.Popen(
                    [sys.executable, gradio_script],
                    cwd=current_dir
                )

                # 等待服务器启动
                time.sleep(3)

                # 打开浏览器
                url = "http://127.0.0.1:7860"
                webbrowser.open(url)
                self.message_queue.put(f"已在浏览器中打开: {url}")

                # 等待进程结束
                self.current_process.wait()

            except Exception as e:
                self.message_queue.put(f"运行Web界面时出错: {str(e)}")
            finally:
                self.current_process = None

        self.run_in_thread(run_gradio)

    def show_system_info(self):
        """显示系统信息"""
        self.log("系统信息:", "INFO")
        self.log(f"操作系统: {platform.system()} {platform.version()}")
        self.log(f"Python版本: {platform.python_version()}")

        try:
            self.log(f"PyTorch版本: {torch.__version__}")
            self.log(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
            if torch.cuda.is_available():
                self.log(f"CUDA版本: {torch.version.cuda}")
                self.log(f"GPU设备: {torch.cuda.get_device_name(0)}")
        except:
            self.log("PyTorch未安装", "WARNING")

        self.log(f"项目路径: {current_dir}")
        self.log(f"配置文件: {os.path.join(current_dir, 'config.ini')}")

    def list_available_models(self):
        """列出可用的模型"""
        models_path = os.path.join(current_dir, "models", "weights")

        if not os.path.exists(models_path):
            self.log("暂无已训练的模型", "INFO")
            return

        self.log("可用的模型:", "INFO")

        models = []
        for item in os.listdir(models_path):
            item_path = os.path.join(models_path, item)
            if os.path.isdir(item_path) and "best_model.pth" in os.listdir(item_path):
                models.append(item)

        if not models:
            self.log("暂无已训练的模型", "INFO")
        else:
            for i, model in enumerate(models, 1):
                self.log(f"{i}. {model}")

    def stop_current_task(self):
        """停止当前任务"""
        if self.current_process:
            self.current_process.terminate()
            self.log("正在停止当前任务...", "WARNING")

    def clear_output(self):
        """清空输出"""
        self.output_text.delete(1.0, tk.END)

    def on_closing(self):
        """关闭窗口时的处理"""
        if self.current_process:
            if messagebox.askokcancel("确认", "有任务正在运行，确定要退出吗？"):
                self.current_process.terminate()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = MicroExpressionApp(root)

    # 设置关闭窗口的处理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main()