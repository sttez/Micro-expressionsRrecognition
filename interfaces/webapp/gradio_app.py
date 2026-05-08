import gradio as gr
from PIL import Image
import torch
import os
import numpy as np
import cv2
import sys
import time
import tempfile

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from core.config import Config
from core.inference import InferenceEngine, LABELS, CHINESE_LABELS

config = Config()
engine = InferenceEngine(config)
device = engine.device


def predict_image(img: Image.Image):
    if img is None:
        return "请上传图像", None, None

    original_img = img.convert("RGB")
    img_array = np.array(original_img)
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    label, probs, face_rect = engine.predict_image(bgr)
    if label is None:
        return "未检测到人脸", original_img, None

    idx = LABELS.index(label) if label in LABELS else 0
    confidence = probs[idx] * 100
    result_text = f"{CHINESE_LABELS[idx]} ({label}) {confidence:.2f}%"

    # 绘制人脸框
    if face_rect:
        x, y, w, h = face_rect
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    processed_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return result_text, original_img, processed_img


def predict_video(video_file):
    if video_file is None:
        return "请上传视频", None

    try:
        video_path = video_file.name if hasattr(video_file, 'name') else video_file
        print(f"处理视频: {video_path}")

        results, fps, total_frames = engine.predict_video(video_path)
        if not results:
            return "未检测到人脸或视频为空", None

        # 重新写入带标注的视频
        output_path = os.path.join(tempfile.gettempdir(), f"processed_{int(time.time())}.mp4")
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        result_dict = {r[0]: (r[1], r[2], r[3]) for r in results}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in result_dict:
                label, probs, face_rect = result_dict[frame_idx]
                if face_rect:
                    x, y, w, h = face_rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    idx = LABELS.index(label) if label in LABELS else 0
                    text = f"{CHINESE_LABELS[idx]} {probs[idx]*100:.1f}%"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # 统计结果
        labels_list = [r[1] for r in results]
        most_common = max(set(labels_list), key=labels_list.count)
        avg_conf = np.mean([r[2][LABELS.index(r[1])] for r in results]) * 100
        result_text = (
            f"识别完成\n"
            f"主要表情: {most_common}\n"
            f"平均置信度: {avg_conf:.2f}%\n"
            f"检测帧数: {len(results)}/{frame_idx}"
        )

        return result_text, output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"处理失败: {e}", None


with gr.Blocks(title="微表情识别系统") as demo:
    gr.Markdown("## 微表情识别系统")
    gr.Markdown("基于 CNN+LSTM 深度学习的微表情识别")

    with gr.Tab("图像识别"):
        with gr.Row():
            img_input = gr.Image(type="pil", label="上传图像")
            with gr.Column():
                result_label = gr.Label(label="预测结果")
                with gr.Row():
                    original_out = gr.Image(label="原图", type="pil")
                    processed_out = gr.Image(label="检测结果", type="pil")
        img_button = gr.Button("识别微表情")
        img_button.click(fn=predict_image, inputs=img_input, outputs=[result_label, original_out, processed_out])

    with gr.Tab("视频识别"):
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="上传视频", file_types=[".mp4", ".avi"])
                process_button = gr.Button("开始处理", variant="primary")
            with gr.Column():
                video_result = gr.Textbox(label="识别结果", lines=4)
        processed_video = gr.Video(label="处理后视频", format="mp4", interactive=False)

        def process_video_wrapper(video_file):
            if video_file is None:
                return "请先上传视频", None
            result_text, output_path = predict_video(video_file)
            if output_path and os.path.exists(output_path):
                return result_text, output_path
            return result_text, None

        process_button.click(fn=process_video_wrapper, inputs=video_input, outputs=[video_result, processed_video])
        video_input.change(lambda x: ("等待处理...", None), inputs=video_input, outputs=[video_result, processed_video])

    with gr.Tab("摄像头识别"):
        webcam_input = gr.Image(sources=["webcam"], type="pil", label="摄像头")
        webcam_output = gr.Label(label="预测结果")
        webcam_button = gr.Button("识别")
        webcam_button.click(fn=predict_image, inputs=webcam_input, outputs=[webcam_output, gr.Image(visible=False), gr.Image(visible=False)])

    gr.Markdown("""
    ### 支持类别
    - Surprise（惊讶） | Repression（压抑） | Happiness（快乐） | Disgust（厌恶） | Others（其他）
    """)


if __name__ == "__main__":
    port = config.getint('webapp', 'port', 7860)
    share = config.getboolean('webapp', 'share', False)
    demo.launch(server_name="127.0.0.1", server_port=port, share=share, debug=True)
