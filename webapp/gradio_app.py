import gradio as gr
from PIL import Image
import torch
import os
import numpy as np
import cv2
import sys
import threading
import time
import tempfile
import shutil

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.model_preprocess import process_single_image, process_video_for_model, preprocess_webcam_frame, \
    detect_face_and_landmarks
from models.microexpression_model import MicroExpressionModel

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = MicroExpressionModel(num_classes=5, sequence_length=32)
model_path = os.path.join(parent_dir, "models", "weights", "MicroExpModel_20250509_133707_temp/best_model.pth")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型权重: {model_path}")
else:
    print(f"警告: 模型文件不存在: {model_path}")
    print("将使用随机初始化的模型参数")

model.to(device)
model.eval()

label_map = ['surprise', 'repression', 'happiness', 'disgust', 'others']


def predict_image(img: Image.Image):
    if img is None:
        return "请上传图像", None, None
    original_img = img.convert("RGB")
    temp_path = os.path.join(current_dir, "temp", "input.jpg")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    original_img.save(temp_path)
    image_sequence, landmarks_sequence, flow_sequence = process_single_image(temp_path)
    if image_sequence is None:
        return "未检测到人脸", original_img, None
    images_tensor = torch.tensor(image_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    landmarks_tensor = torch.tensor(landmarks_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    flows_tensor = torch.tensor(flow_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(images_tensor, landmarks_tensor, flows_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
    label = label_map[pred_idx]
    confidence = prob[pred_idx]
    processed_img = Image.fromarray(image_sequence[0, 0].astype("uint8"))
    return f"{label} ({confidence * 100:.2f}%)", original_img, processed_img


def convert_video_to_h264(input_path, output_path):
    """使用OpenCV重新编码视频为H.264格式"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用H.264编码器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 使用avc1编码，更兼容
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return output_path


def predict_video(video_file):
    if video_file is None:
        return "请上传视频", None

    try:
        # 使用输入文件的完整路径
        video_path = video_file.name if hasattr(video_file, 'name') else video_file
        print(f"处理视频: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "无法打开视频文件", None

        # 创建临时输出文件
        output_filename = f"processed_{int(time.time())}.mp4"
        output_path = os.path.join(current_dir, "temp", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 使用mp4v编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        prev_face = None
        prev_landmarks = None
        frame_count = 0
        results = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"开始处理视频，总帧数: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:  # 每30帧输出一次进度
                print(f"处理进度: {frame_count}/{total_frames}")

            face_gray, landmarks, flow, prev_face, prev_landmarks = preprocess_webcam_frame(
                frame, prev_face, prev_landmarks
            )

            if face_gray is not None:
                image_seq = np.array([face_gray for _ in range(32)])[:, np.newaxis, :, :]
                landmark_seq = np.array([landmarks for _ in range(32)])
                flow_seq = np.array([flow if flow is not None else np.zeros((2, 128, 128)) for _ in range(31)])

                x = torch.tensor(image_seq, dtype=torch.float32).unsqueeze(0).to(device)
                y = torch.tensor(landmark_seq, dtype=torch.float32).unsqueeze(0).to(device)
                z = torch.tensor(flow_seq, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(x, y, z)
                    pred = torch.argmax(output, dim=1).item()
                    prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                    label = f"{label_map[pred]} ({prob[pred] * 100:.2f}%)"
                    results.append((frame_count, label_map[pred], prob[pred]))

                # 使用 OpenCV 检测人脸位置并画框
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    os.path.join(parent_dir, "utils", "haarcascade_frontalface_default.xml"))
                faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

                for (x1, y1, w, h) in faces:
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        print(f"视频处理完成，输出文件: {output_path}")

        # 生成结果摘要
        if results:
            most_common = max(set([r[1] for r in results]), key=[r[1] for r in results].count)
            avg_confidence = sum([r[2] for r in results]) / len(results)
            result_text = f"识别完成\n主要表情: {most_common}\n平均置信度: {avg_confidence * 100:.2f}%\n检测到表情帧数: {len(results)}/{frame_count}"
        else:
            result_text = "未检测到人脸"

        # 确保文件存在
        if not os.path.exists(output_path):
            return "视频处理失败", None

        # 尝试转换为H.264编码
        h264_output = output_path.replace('.mp4', '_h264.mp4')
        converted_path = convert_video_to_h264(output_path, h264_output)

        if converted_path and os.path.exists(converted_path):
            print(f"视频转换为H.264编码: {converted_path}")
            return result_text, converted_path
        else:
            print(f"使用原始编码: {output_path}")
            return result_text, output_path

    except Exception as e:
        print(f"视频处理错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"处理失败: {str(e)}", None


def predict_webcam_frame(img):
    if img is None:
        return "未检测到图像"
    frame = np.array(img.convert("RGB"))
    temp_img = Image.fromarray(frame)
    result, _, _ = predict_image(temp_img)
    return result


with gr.Blocks(title="微表情识别系统") as demo:
    gr.Markdown("## 微表情识别系统")
    gr.Markdown("基于深度学习的微表情识别系统")

    with gr.Tab("图像识别"):
        with gr.Row():
            img_input = gr.Image(type="pil", label="上传图像")
            with gr.Column():
                result_label = gr.Label(label="预测结果")
                with gr.Row():
                    original_out = gr.Image(label="原图", type="pil")
                    processed_out = gr.Image(label="处理后", type="pil")
        img_button = gr.Button("识别微表情")
        img_button.click(fn=predict_image, inputs=img_input, outputs=[result_label, original_out, processed_out])

    with gr.Tab("视频识别"):
        gr.Markdown("""
        ### 视频识别说明
        - 支持的格式：.mp4, .avi
        - 处理时间取决于视频长度和计算机性能
        - 如果视频无法播放，可能是编码问题
        """)

        with gr.Row():
            with gr.Column():
                # 视频上传组件
                video_input = gr.File(label="上传视频文件", file_types=[".mp4", ".avi"])
                process_button = gr.Button("开始处理", variant="primary")

            with gr.Column():
                # 识别结果
                video_result = gr.Textbox(label="识别结果", lines=4)

        # 处理后的视频播放器
        processed_video = gr.Video(
            label="处理后视频",
            format="mp4",
            interactive=False,
            autoplay=False,
            show_download_button=True
        )

        # 下载链接
        download_file = gr.File(label="下载处理后的视频", visible=False)


        # 处理函数
        def process_video_wrapper(video_file):
            if video_file is None:
                return "请先上传视频", None, None

            try:
                # 处理视频
                result_text, processed_path = predict_video(video_file)

                if processed_path and os.path.exists(processed_path):
                    print(f"返回视频文件: {processed_path}")
                    # 返回处理后的视频路径
                    return result_text, processed_path, gr.File(value=processed_path, visible=True)
                else:
                    return result_text, None, None

            except Exception as e:
                print(f"处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return f"处理失败: {str(e)}", None, None


        # 绑定处理函数
        process_button.click(
            fn=process_video_wrapper,
            inputs=video_input,
            outputs=[video_result, processed_video, download_file]
        )

        # 上传新视频时清空结果
        video_input.change(
            fn=lambda x: ("等待处理...", None, gr.File(visible=False)),
            inputs=video_input,
            outputs=[video_result, processed_video, download_file]
        )

    gr.Markdown("""
    ### 使用说明：
    - 上传图像、视频进行微表情识别。
    - 视频处理可能需要一些时间，请耐心等待。
    - 处理后的视频可以直接下载。

    ### 支持的微表情类别：
    - Surprise（惊讶）
    - Repression（压抑）
    - Happiness（快乐）
    - Disgust（厌恶）
    - Others（其他）

    ### 故障排除：
    - 如果视频无法播放，尝试下载文件后用本地播放器查看
    - 可以尝试不同格式的视频文件
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,  # 启用调试模式
        show_error=True  # 显示详细错误信息
    )