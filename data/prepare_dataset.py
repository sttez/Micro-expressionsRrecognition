import cv2
import dlib
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import random
from pathlib import Path
from collections import defaultdict
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

config = Config()


def makedir(path):
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {path}")


def initialize_detectors():
    """初始化检测器"""
    predictor_path = config.path('paths', 'predictor_path')
    cascade_path = config.path('paths', 'cascade_path')

    if not os.path.exists(predictor_path):
        logger.error(f"关键点模型不存在: {predictor_path}")
        raise FileNotFoundError(f"找不到关键点模型: {predictor_path}")

    if not os.path.exists(cascade_path):
        logger.error(f"Cascade 文件不存在: {cascade_path}")
        raise FileNotFoundError(f"找不到 Cascade: {cascade_path}")

    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    logger.info("检测器初始化成功")
    return predictor, detector, face_cascade


def analyze_class_distribution():
    """分析类别分布"""
    logger.info("分析类别分布...")
    excel_path = config.path('paths', 'excel_path')
    df = pd.read_excel(excel_path)

    emotion_counts = {}
    sequence_info = defaultdict(list)
    valid_sequences = 0
    invalid_sequences = 0

    for _, row in df.iterrows():
        emotion = str(row['Estimated Emotion']).strip().lower()
        subject = str(row['Subject']).zfill(2)
        filename = str(row['Filename']).strip()

        try:
            onset = int(float(str(row['OnsetFrame']).strip()))
            apex = int(float(str(row['ApexFrame']).strip()))
            offset = int(float(str(row['OffsetFrame']).strip()))
            if onset < 0 or apex < 0 or offset < 0 or not (onset <= apex <= offset):
                invalid_sequences += 1
                continue
        except (ValueError, TypeError):
            invalid_sequences += 1
            continue

        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0
        emotion_counts[emotion] += 1
        sequence_info[emotion].append({
            'subject': subject, 'filename': filename,
            'onset': onset, 'apex': apex, 'offset': offset
        })
        valid_sequences += 1

    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1])
    logger.info(f"有效: {valid_sequences} | 无效: {invalid_sequences}")

    excluded_classes = [e for e, _ in sorted_emotions[:2]]
    valid_classes = [e for e, _ in sorted_emotions[2:]]
    logger.info(f"排除: {excluded_classes} | 保留: {valid_classes}")

    return valid_classes, sequence_info


def extract_face_landmarks(img_path, predictor, detector):
    """提取面部关键点"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            return None
        landmarks = predictor(gray, faces[0])
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    except Exception as e:
        logger.error(f"提取关键点失败 {img_path}: {e}")
        return None


def extract_optical_flow(prev_path, next_path):
    """提取光流"""
    try:
        prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_path, cv2.IMREAD_GRAYSCALE)
        if prev is None or next_img is None:
            return None
        return cv2.calcOpticalFlowFarneback(prev, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    except Exception as e:
        logger.error(f"提取光流失败: {e}")
        return None


def preprocess_image(img_path, image_size):
    """预处理图像"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, (image_size, image_size))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32) / 255.0
    except Exception as e:
        logger.error(f"处理图像出错 {img_path}: {e}")
        return None


def select_sequence_start(onset, apex, offset, total_frames, sequence_length, loop):
    """智能选择序列起始帧"""
    expression_duration = offset - onset + 1
    if expression_duration >= sequence_length:
        max_start = offset - sequence_length + 1
        min_start = max(0, onset)
        if loop == 0:
            ideal_start = apex - sequence_length // 2
        else:
            range_width = max_start - min_start
            ideal_start = min_start + random.randint(0, range_width) if range_width > 0 else min_start
    else:
        ideal_start = apex - sequence_length // 2

    ideal_start = max(0, min(ideal_start, total_frames - sequence_length))
    return ideal_start


def main():
    """主数据准备流程"""
    logger.info("=== 开始数据准备 ===")

    # 读取配置
    raw_data_dir = config.path('paths', 'raw_data_dir')
    sequences_dir = config.path('paths', 'sequences_dir')
    optical_flow_dir = config.path('paths', 'optical_flow_dir')
    landmarks_dir = config.path('paths', 'landmarks_dir')
    image_size = config.getint('preprocessing', 'image_size', 128)
    sequence_length = config.getint('preprocessing', 'sequence_length', 32)
    min_sequences = config.getint('preprocessing', 'min_sequences_per_class', 475)
    train_pct = config.getfloat('training', 'train_split', 0.8)

    # 初始化检测器
    predictor, detector, face_cascade = initialize_detectors()

    # 分析类别分布
    valid_classes, sequence_info = analyze_class_distribution()
    if not valid_classes:
        logger.error("无有效类别，终止")
        return

    num_classes = len(valid_classes)
    emotion_to_idx = {e: i for i, e in enumerate(valid_classes)}
    logger.info(f"类别映射: {emotion_to_idx}")

    # 预处理原始图像（提取关键点和光流）
    logger.info("预处理原始图像...")
    excel_path = config.path('paths', 'excel_path')
    df = pd.read_excel(excel_path)

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="预处理"):
        emotion = str(row['Estimated Emotion']).strip().lower()
        if emotion not in valid_classes:
            continue
        subject = str(row['Subject']).zfill(2)
        filename = str(row['Filename']).strip()

        seq_path = os.path.join(raw_data_dir, f"sub{subject}", filename)
        if not os.path.exists(seq_path):
            continue

        lm_dir = os.path.join(landmarks_dir, f"sub{subject}", filename)
        of_dir = os.path.join(optical_flow_dir, f"sub{subject}", filename)
        makedir(lm_dir)
        makedir(of_dir)

        frames = sorted([f for f in os.listdir(seq_path) if f.lower().endswith('.jpg')])
        for i, fname in enumerate(frames):
            frame_id = fname.replace('.jpg', '')
            lm_file = os.path.join(lm_dir, f"{frame_id}.json")
            if not os.path.exists(lm_file):
                lm = extract_face_landmarks(os.path.join(seq_path, fname), predictor, detector)
                if lm:
                    with open(lm_file, 'w') as f:
                        json.dump({"landmarks": lm}, f)

            if i > 0:
                of_file = os.path.join(of_dir, f"{frame_id}.npz")
                if not os.path.exists(of_file):
                    flow = extract_optical_flow(
                        os.path.join(seq_path, frames[i - 1]),
                        os.path.join(seq_path, fname)
                    )
                    if flow is not None:
                        np.savez_compressed(of_file, flow=flow.astype(np.float16))

    # 创建序列样本
    logger.info("创建序列样本...")
    all_sequences = []
    for emotion in valid_classes:
        infos = sequence_info[emotion]
        makedir(os.path.join(sequences_dir, 'train', emotion))
        makedir(os.path.join(sequences_dir, 'test', emotion))

        seq_idx = 0
        loop = 0
        while len([s for s in all_sequences if s['emotion'] == emotion]) < min_sequences:
            for info in infos:
                if len([s for s in all_sequences if s['emotion'] == emotion]) >= min_sequences:
                    break

                subject = info['subject']
                filename = info['filename']
                onset, apex, offset = info['onset'], info['apex'], info['offset']

                raw_seq = os.path.join(raw_data_dir, f"sub{subject}", filename)
                if not os.path.exists(raw_seq):
                    continue

                frames = sorted([f for f in os.listdir(raw_seq) if f.lower().endswith('.jpg')])
                total = len(frames)
                if total < sequence_length:
                    continue

                start = select_sequence_start(onset, apex, offset, total, sequence_length, loop)
                selected = frames[start:start + sequence_length]
                if len(selected) < sequence_length:
                    continue

                # 提取图像序列
                images = []
                for fname in selected:
                    img = preprocess_image(os.path.join(raw_seq, fname), image_size)
                    if img is None:
                        break
                    images.append(img)

                if len(images) != sequence_length:
                    continue

                # 提取关键点
                lm_paths = []
                for fname in selected:
                    frame_id = fname.replace('.jpg', '')
                    lm_file = os.path.join(landmarks_dir, f"sub{subject}", filename, f"{frame_id}.json")
                    lm_paths.append(lm_file if os.path.exists(lm_file) else 'none')

                # 提取光流
                of_paths = []
                for fname in selected[1:]:
                    frame_id = fname.replace('.jpg', '')
                    of_file = os.path.join(optical_flow_dir, f"sub{subject}", filename, f"{frame_id}.npz")
                    of_paths.append(of_file if os.path.exists(of_file) else 'none')

                # 保存序列
                out_name = f"{emotion}_{seq_idx:04d}"
                npy_path = os.path.join(sequences_dir, 'all', emotion, f"{out_name}.npy")
                makedir(os.path.dirname(npy_path))
                np.save(npy_path, np.array(images))

                all_sequences.append({
                    'emotion': emotion,
                    'npy_path': npy_path,
                    'landmark_paths': lm_paths,
                    'flow_paths': of_paths,
                    'label_idx': emotion_to_idx[emotion]
                })
                seq_idx += 1
            loop += 1

    # 数据增强（翻转）
    logger.info("数据增强...")
    augmented = []
    for seq in all_sequences:
        images = np.load(seq['npy_path'])
        flipped = np.flip(images, axis=2).copy()  # 水平翻转
        aug_name = os.path.basename(seq['npy_path']).replace('.npy', '_flip.npy')
        aug_path = os.path.join(os.path.dirname(seq['npy_path']), aug_name)
        np.save(aug_path, flipped)
        augmented.append({**seq, 'npy_path': aug_path})
    all_sequences.extend(augmented)

    # 分割训练/测试集
    logger.info("分割训练/测试集...")
    train_seqs, test_seqs = train_test_split(
        all_sequences, train_size=train_pct,
        stratify=[s['label_idx'] for s in all_sequences],
        random_state=42
    )

    # 复制文件到 train/test 目录
    for seq_list, split in [(train_seqs, 'train'), (test_seqs, 'test')]:
        for seq in seq_list:
            src = seq['npy_path']
            dst = os.path.join(sequences_dir, split, seq['emotion'], os.path.basename(src))
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
                seq['npy_path'] = dst

    # 生成标签文件
    logger.info("生成标签文件...")
    train_txt = config.get_label_train()
    test_txt = config.get_label_test()

    for seq_list, txt_path in [(train_seqs, train_txt), (test_seqs, test_txt)]:
        with open(txt_path, 'w') as f:
            for seq in seq_list:
                label = seq['label_idx']
                npy_rel = os.path.relpath(seq['npy_path'], project_root)
                lm_str = '|'.join(os.path.relpath(p, project_root) if os.path.exists(p) else 'none'
                                  for p in seq['landmark_paths'])
                of_str = '|'.join(os.path.relpath(p, project_root) if os.path.exists(p) else 'none'
                                  for p in seq['flow_paths'])
                onehot = ','.join(['1' if i == label else '0' for i in range(num_classes)])
                f.write(f"{label};{npy_rel};{lm_str};{of_str};{onehot}\n")

    # 写类别映射
    mapping_path = config.get_class_mapping()
    with open(mapping_path, 'w') as f:
        for emotion, idx in emotion_to_idx.items():
            f.write(f"{idx}: {emotion}\n")

    logger.info(f"训练集: {len(train_seqs)} | 测试集: {len(test_seqs)}")
    logger.info("=== 数据准备完成 ===")


if __name__ == '__main__':
    main()
