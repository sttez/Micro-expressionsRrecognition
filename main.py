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
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class ProcessOutputReader(threading.Thread):
    def __init__(self, process, output_queue):
        threading.Thread.__init__(self)
        self.process = process
        self.output_queue = output_queue
        self.daemon = True

    def run(self):
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.output_queue.put(line.strip())
            for line in iter(self.process.stderr.readline, ''):
                if line:
                    self.output_queue.put(f"[STDERR] {line.strip()}")
        except Exception as e:
            self.output_queue.put(f"读取输出错误: {e}")


class MicroExpressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("微表情识别系统")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.center_window()
        self.message_queue = queue.Queue()
        self.current_process = None
        self.setup_ui()
        self.check_queue()
        self.check_environment()

    def center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 标题
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        ttk.Label(title_frame, text="微表情识别系统", font=("Arial", 24, "bold")).grid(row=0, column=0)
        ttk.Label(title_frame, text="Micro-Expression Recognition System", font=("Arial", 12)).grid(row=1, column=0)

        # 左侧按钮
        left_frame = ttk.LabelFrame(main_frame, text="功能模块", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.prepare_btn = ttk.Button(btn_frame, text="1. 数据准备", command=self.run_data_preparation, width=25)
        self.prepare_btn.grid(row=0, column=0, pady=5, sticky=tk.W)
        self.train_btn = ttk.Button(btn_frame, text="2. 模型训练", command=self.run_training, width=25)
        self.train_btn.grid(row=1, column=0, pady=5, sticky=tk.W)
        self.gui_btn = ttk.Button(btn_frame, text="3. 桌面界面", command=self.run_desktop_ui, width=25)
        self.gui_btn.grid(row=2, column=0, pady=5, sticky=tk.W)
        self.web_btn = ttk.Button(btn_frame, text="4. 网页界面", command=self.run_web_interface, width=25)
        self.web_btn.grid(row=3, column=0, pady=5, sticky=tk.W)

        ttk.Separator(btn_frame, orient='horizontal').grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))
        self.info_btn = ttk.Button(btn_frame, text="查看系统信息", command=self.show_system_info, width=25)
        self.info_btn.grid(row=5, column=0, pady=5, sticky=tk.W)
        self.models_btn = ttk.Button(btn_frame, text="查看可用模型", command=self.list_available_models, width=25)
        self.models_btn.grid(row=6, column=0, pady=5, sticky=tk.W)
        self.switch_btn = ttk.Button(btn_frame, text="切换模型", command=self.switch_model, width=25)
        self.switch_btn.grid(row=7, column=0, pady=5, sticky=tk.W)

        # 右侧状态
        right_frame = ttk.LabelFrame(main_frame, text="运行状态", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

        self.status_label = ttk.Label(right_frame, text="就绪", font=("Arial", 12))
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        self.progress = ttk.Progressbar(right_frame, mode='determinate', maximum=100)
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.progress['value'] = 100

        self.output_text = scrolledtext.ScrolledText(right_frame, height=20, width=50)
        self.output_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)

        # 底部按钮
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        self.stop_btn = ttk.Button(bottom_frame, text="停止当前任务", command=self.stop_current_task, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=0, padx=5)
        self.clear_btn = ttk.Button(bottom_frame, text="清空输出", command=self.clear_output)
        self.clear_btn.grid(row=0, column=1, padx=5)
        self.exit_btn = ttk.Button(bottom_frame, text="退出程序", command=self.on_closing)
        self.exit_btn.grid(row=0, column=2, padx=5)

        main_frame.rowconfigure(1, weight=1)

    def log(self, message, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{ts}] {level}: {message}\n")
        self.output_text.see(tk.END)

    def set_status(self, status):
        self.status_label.config(text=status)
        if status == "就绪":
            self.progress['value'] = 100
        elif "正在运行" in status:
            self.progress['value'] = 0

    def enable_buttons(self, state=True):
        s = tk.NORMAL if state else tk.DISABLED
        for btn in [self.prepare_btn, self.train_btn, self.gui_btn, self.web_btn, self.info_btn, self.models_btn]:
            btn.config(state=s)
        self.stop_btn.config(state=tk.DISABLED if state else tk.NORMAL)

    def check_queue(self):
        try:
            while True:
                self.log(self.message_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def run_in_thread(self, target, *args):
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()

    def check_environment(self):
        self.log("检查系统环境...")
        required = ['torch', 'cv2', 'numpy', 'sklearn']
        missing = []
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            self.log(f"缺少包: {', '.join(missing)}", "WARNING")
        else:
            self.log("环境检查通过")

    def run_subprocess(self, script_name, description):
        try:
            self.set_status(f"正在运行: {description}")
            self.enable_buttons(False)
            script_path = os.path.join(current_dir, script_name)
            if not os.path.exists(script_path):
                self.message_queue.put(f"错误: 找不到 {script_name}")
                return
            self.current_process = subprocess.Popen(
                [sys.executable, "-u", script_path],
                cwd=current_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            reader = ProcessOutputReader(self.current_process, self.message_queue)
            reader.start()
            self.current_process.wait()
            rc = self.current_process.returncode
            self.message_queue.put(f"{description} 完成" if rc == 0 else f"{description} 失败 (code: {rc})")
        except Exception as e:
            self.message_queue.put(f"运行出错: {e}")
        finally:
            self.current_process = None
            self.enable_buttons(True)
            self.set_status("就绪")

    def run_data_preparation(self):
        self.log("开始数据准备...")
        self.run_in_thread(self.run_subprocess, "data/prepare_dataset.py", "数据准备")

    def run_training(self):
        self.log("开始模型训练...")
        self.run_in_thread(self.run_subprocess, "training/trainer.py", "模型训练")

    def run_desktop_ui(self):
        self.log("启动桌面界面...")
        script = os.path.join(current_dir, "interfaces", "desktop", "gui_main.py")
        if not os.path.exists(script):
            messagebox.showerror("错误", "找不到桌面界面脚本")
            return
        try:
            subprocess.Popen([sys.executable, script], cwd=current_dir)
            self.log("桌面界面已启动")
        except Exception as e:
            self.log(f"启动失败: {e}", "ERROR")

    def run_web_interface(self):
        self.log("启动 Web 界面...")
        def run_gradio():
            try:
                script = os.path.join(current_dir, "interfaces", "webapp", "gradio_app.py")
                if not os.path.exists(script):
                    self.message_queue.put(f"错误: 找不到 Gradio 脚本")
                    return
                self.message_queue.put("正在启动 Gradio...")
                self.current_process = subprocess.Popen([sys.executable, script], cwd=current_dir)
                time.sleep(3)
                webbrowser.open("http://127.0.0.1:7860")
                self.message_queue.put("已在浏览器打开: http://127.0.0.1:7860")
                self.current_process.wait()
            except Exception as e:
                self.message_queue.put(f"Web 界面错误: {e}")
            finally:
                self.current_process = None
        self.run_in_thread(run_gradio)

    def show_system_info(self):
        self.log(f"系统: {platform.system()} {platform.version()}")
        self.log(f"Python: {platform.python_version()}")
        try:
            self.log(f"PyTorch: {torch.__version__}")
            self.log(f"CUDA: {'可用' if torch.cuda.is_available() else '不可用'}")
            if torch.cuda.is_available():
                self.log(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            self.log("PyTorch 未安装", "WARNING")
        self.log(f"项目路径: {current_dir}")

    def list_available_models(self):
        from core.config import Config
        cfg = Config()
        models = cfg.list_models()
        if not models:
            self.log("暂无已训练模型")
        else:
            self.log(f"共 {len(models)} 个可用模型（按时间排序）:")
            for i, (name, path, acc) in enumerate(models, 1):
                self.log(f"  {i}. {name}  准确率: {acc}")

    def switch_model(self):
        from core.config import Config
        cfg = Config()
        models = cfg.list_models()
        if not models:
            messagebox.showinfo("提示", "暂无已训练的模型，请先训练")
            return

        # 弹出选择对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("切换模型")
        dialog.geometry("450x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="选择要使用的模型:", font=("Arial", 11)).pack(pady=10)

        listbox = tk.Listbox(dialog, font=("Arial", 10), height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        current_path = cfg.get_model_path()
        for i, (name, path, acc) in enumerate(models):
            marker = " *" if path == current_path else ""
            listbox.insert(tk.END, f"{name}  准确率: {acc}{marker}")

        def confirm_selection():
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("提示", "请选择一个模型")
                return
            idx = sel[0]
            name, path, acc = models[idx]
            cfg.set_selected_model(path)
            self.log(f"已切换到模型: {name} (准确率: {acc})")
            self.log("下次启动界面时将使用该模型")
            dialog.destroy()

        ttk.Button(dialog, text="确认", command=confirm_selection).pack(pady=10)

    def stop_current_task(self):
        if self.current_process:
            self.current_process.terminate()
            self.log("正在停止...", "WARNING")

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)

    def on_closing(self):
        if self.current_process:
            if messagebox.askokcancel("确认", "有任务运行中，确定退出？"):
                self.current_process.terminate()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = MicroExpressionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
