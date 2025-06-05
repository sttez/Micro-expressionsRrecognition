import cv2
import dlib
import numpy as np
import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import configparser
import random
from pathlib import Path
from keras.utils import to_categorical
from collections import defaultdict

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


# 读取配置文件
def load_config():
    """加载配置文件"""
    config = configparser.ConfigParser()
    config_path = 'config.ini'

    if not os.path.exists(config_path):
        config['PATHS'] = {
            'raw_data_dir': "data/CASME2-RAW",
            'sequences_dir': "data/sequences",
            'optical_flow_dir': "data/optical_flow",  # 原始图像的光流
            'landmarks_dir': "data/landmarks",  # 原始图像的关键点
            'coding_excel': "data/CASME2-coding-20140508.xlsx",
            'face_cascade': "utils/haarcascade_frontalface_default.xml",
            'shape_predictor': "utils/shape_predictor_68_face_landmarks.dat"
        }
        config['PARAMETERS'] = {
            'train_pct': '0.8',
            'image_size': '128',
            'sequence_length': '32',
            'min_sequences_per_class': '475'
        }
        config['CLASSES'] = {
            'excluded_classes': '',
            'valid_classes': ''
        }

        with open(config_path, 'w') as f:
            config.write(f)
        logger.info(f"已创建默认配置文件: {config_path}")

    config.read(config_path)
    return config


# 加载配置
config = load_config()

# 配置路径
raw_data_dir = config['PATHS']['raw_data_dir']
sequences_dir = config['PATHS']['sequences_dir']
optical_flow_dir = config['PATHS']['optical_flow_dir']
landmarks_dir = config['PATHS']['landmarks_dir']
coding_excel = config['PATHS']['coding_excel']
face_cascade_path = config['PATHS']['face_cascade']
shape_predictor_path = config['PATHS']['shape_predictor']

# 配置参数
train_pct = float(config['PARAMETERS']['train_pct'])
IMAGE_SIZE = int(config['PARAMETERS']['image_size'])
SEQUENCE_LENGTH = int(config['PARAMETERS']['sequence_length'])
MIN_SEQUENCES = int(config['PARAMETERS']['min_sequences_per_class'])


def makedir(path):
    """创建目录"""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {path}")


# 初始化预处理器
def initialize_detectors():
    """初始化人脸检测器和关键点预测器"""
    try:
        if not os.path.exists(shape_predictor_path):
            logger.error(f"面部关键点预测器文件不存在: {shape_predictor_path}")
            raise FileNotFoundError(f"无法找到面部关键点预测器文件: {shape_predictor_path}")

        if not os.path.exists(face_cascade_path):
            logger.error(f"人脸级联分类器文件不存在: {face_cascade_path}")
            raise FileNotFoundError(f"无法找到人脸级联分类器文件: {face_cascade_path}")

        predictor = dlib.shape_predictor(shape_predictor_path)
        detector = dlib.get_frontal_face_detector()
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        logger.info("成功初始化人脸检测器和关键点预测器")
        return predictor, detector, face_cascade
    except Exception as e:
        logger.error(f"初始化检测器失败: {str(e)}")
        raise


# 全局变量用于存储检测器
try:
    predictor, detector, face_cascade = initialize_detectors()
except Exception as e:
    logger.critical(f"无法初始化检测器，程序终止: {str(e)}")
    exit(1)


def analyze_class_distribution():
    """分析各个类别的样本数量，确定要删除的类别"""
    logger.info("分析各个类别的样本分布...")

    df = pd.read_excel(coding_excel)

    # 统计每个情绪类别的序列数量
    emotion_counts = {}
    sequence_info = defaultdict(list)
    valid_sequences = 0
    invalid_sequences = 0

    for _, row in df.iterrows():
        emotion = str(row['Estimated Emotion']).strip().lower()
        subject = str(row['Subject']).zfill(2)
        filename = str(row['Filename']).strip()

        # 处理可能的特殊值
        try:
            onset_value = str(row['OnsetFrame']).strip()
            apex_value = str(row['ApexFrame']).strip()
            offset_value = str(row['OffsetFrame']).strip()

            # 检查是否包含有效的数值
            if any(val in ['/', '', 'NaN', 'nan', None] for val in [onset_value, apex_value, offset_value]):
                logger.warning(f"跳过无效序列: Subject {subject}, Filename {filename} "
                               f"(onset: {onset_value}, apex: {apex_value}, offset: {offset_value})")
                invalid_sequences += 1
                continue

            # 转换为整数
            onset = int(float(onset_value))
            apex = int(float(apex_value))
            offset = int(float(offset_value))

            # 验证帧值的合理性
            if onset < 0 or apex < 0 or offset < 0:
                logger.warning(f"跳过负值帧序列: Subject {subject}, Filename {filename}")
                invalid_sequences += 1
                continue

            if not (onset <= apex <= offset):
                logger.warning(f"跳过帧顺序异常的序列: Subject {subject}, Filename {filename} "
                               f"(onset: {onset}, apex: {apex}, offset: {offset})")
                invalid_sequences += 1
                continue

        except (ValueError, TypeError) as e:
            logger.warning(f"解析帧值时出错: Subject {subject}, Filename {filename}, 错误: {str(e)}")
            invalid_sequences += 1
            continue

        # 如果一切正常，添加到统计中
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0
        emotion_counts[emotion] += 1

        # 保存序列信息
        sequence_info[emotion].append({
            'subject': subject,
            'filename': filename,
            'onset': onset,
            'apex': apex,
            'offset': offset
        })
        valid_sequences += 1

    # 按样本数量排序
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1])

    logger.info("数据集分析结果：")
    logger.info(f"有效序列数: {valid_sequences}")
    logger.info(f"无效序列数: {invalid_sequences}")
    logger.info(f"总序列数: {valid_sequences + invalid_sequences}")

    logger.info("\n各类别样本数量统计：")
    for emotion, count in sorted_emotions:
        logger.info(f"{emotion}: {count} sequences")

    # 确保有足够的类别
    if len(sorted_emotions) < 3:
        raise ValueError(f"数据集中只有 {len(sorted_emotions)} 个类别，"
                         f"至少需要3个类别才能继续处理")

    # 选择样本最少的两个类别进行删除
    excluded_classes = [emotion for emotion, _ in sorted_emotions[:2]]
    valid_classes = [emotion for emotion, _ in sorted_emotions[2:]]

    logger.info(f"\n将删除的类别: {excluded_classes}")
    logger.info(f"保留的类别: {valid_classes}")

    # 统计保留类别的序列数量
    logger.info("\n保留类别的序列统计：")
    retained_counts = {}
    for emotion in valid_classes:
        count = len(sequence_info[emotion])
        retained_counts[emotion] = count
        logger.info(f"{emotion}: {count} sequences")

    # 确保有足够的数据
    total_valid_sequences = sum(len(sequence_info[emotion]) for emotion in valid_classes)
    logger.info(f"\n保留的总序列数: {total_valid_sequences}")

    if total_valid_sequences < 50:  # 设置一个合理的最小阈值
        logger.warning(f"警告：有效序列总数较少 ({total_valid_sequences})，"
                       f"可能影响模型训练效果")

    # 更新配置
    config['CLASSES']['excluded_classes'] = ','.join(excluded_classes)
    config['CLASSES']['valid_classes'] = ','.join(valid_classes)
    with open('config.ini', 'w') as f:
        config.write(f)

    return valid_classes, sequence_info


def preprocess_all_images():
    """预处理所有原始图像，提取关键点和光流"""
    logger.info("开始预处理所有原始图像...")

    # 读取编码信息
    df = pd.read_excel(coding_excel)
    valid_classes = config['CLASSES']['valid_classes'].split(',')

    processed_count = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="预处理原始图像"):
        emotion = str(row['Estimated Emotion']).strip().lower()
        if emotion not in valid_classes:
            continue

        subject = str(row['Subject']).zfill(2)
        filename = str(row['Filename']).strip()

        # 原始序列路径
        sequence_path = os.path.join(raw_data_dir, f"sub{subject}", filename)
        if not os.path.exists(sequence_path):
            logger.warning(f"序列路径不存在: {sequence_path}")
            continue

        # 创建对应的输出目录
        subject_landmarks_dir = os.path.join(landmarks_dir, f"sub{subject}", filename)
        subject_optical_flow_dir = os.path.join(optical_flow_dir, f"sub{subject}", filename)
        makedir(subject_landmarks_dir)
        makedir(subject_optical_flow_dir)

        # 获取所有帧
        frames = sorted([f for f in os.listdir(sequence_path) if f.lower().endswith('.jpg')])

        # 处理每一帧
        for i, frame_name in enumerate(frames):
            frame_path = os.path.join(sequence_path, frame_name)
            frame_id = frame_name.replace('.jpg', '')

            # 检查是否已经处理过
            landmarks_file = os.path.join(subject_landmarks_dir, f"{frame_id}.json")
            if os.path.exists(landmarks_file):
                continue

            # 提取关键点
            landmarks = extract_face_landmarks(frame_path)
            if landmarks is not None:
                with open(landmarks_file, 'w') as f:
                    json.dump({"landmarks": landmarks}, f)

            # 计算光流（除了第一帧）
            if i > 0:
                prev_frame_path = os.path.join(sequence_path, frames[i - 1])
                optical_flow_file = os.path.join(subject_optical_flow_dir, f"{frame_id}.npz")

                if not os.path.exists(optical_flow_file):
                    flow = extract_optical_flow_original(prev_frame_path, frame_path)
                    if flow is not None:
                        # 保存原始尺寸的光流
                        flow = flow.astype(np.float16)
                        np.savez_compressed(optical_flow_file, flow=flow)

            processed_count += 1

    logger.info(f"预处理完成，共处理 {processed_count} 帧图像")


def extract_face_landmarks(img_path):
    """提取面部关键点"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用dlib检测人脸
        faces = detector(gray)

        if len(faces) == 0:
            return None

        # 对第一个检测到的人脸提取关键点
        face = faces[0]
        landmarks = predictor(gray, face)

        # 提取68个关键点坐标
        landmarks_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

        return landmarks_points

    except Exception as e:
        logger.error(f"提取关键点失败 {img_path}: {str(e)}")
        return None


def extract_optical_flow_original(prev_img_path, next_img_path):
    """提取原始尺寸的光流特征"""
    try:
        prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_img_path, cv2.IMREAD_GRAYSCALE)

        if prev_img is None or next_img is None:
            return None

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return flow

    except Exception as e:
        logger.error(f"提取光流失败: {str(e)}")
        return None


def preprocess_image(img_path):
    """预处理图像：调整尺寸、转灰度、归一化"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"无法读取图像: {img_path}")
            return None

        # 调整尺寸为128x128
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # 转换为灰度图
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 归一化到[0,1.txt]区间
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        logger.error(f"处理图像时出错 {img_path}: {str(e)}")
        return None


def select_sequence_start(onset, apex, offset, total_frames, sequence_length, loop):
    """智能选择序列起始帧"""
    # 计算微表情的持续时间
    expression_duration = offset - onset + 1

    if expression_duration >= sequence_length:
        max_start = offset - sequence_length + 1
        min_start = max(0, onset)

        if loop == 0:
            # 第一次循环，让apex在序列中间位置
            ideal_start = apex - sequence_length // 2
        else:
            # 后续循环，增加随机性
            range_width = max_start - min_start
            if range_width > 0:
                random_offset = random.randint(0, range_width)
                ideal_start = min_start + random_offset
            else:
                ideal_start = min_start
    else:
        ideal_start = apex - sequence_length // 2

        if loop > 0:
            max_offset = min(sequence_length // 4, onset - ideal_start,
                             ideal_start + sequence_length - offset - 1)
            if max_offset > 0:
                random_offset = random.randint(-max_offset, max_offset)
                ideal_start += random_offset

    # 确保起始帧在有效范围内
    start_idx = max(0, min(ideal_start, total_frames - sequence_length))

    return start_idx


def create_sequence_samples(sequence_info, valid_classes):
    """创建标准化长度的序列样本，只处理图像序列"""
    logger.info("创建标准化序列样本...")

    sequence_samples = defaultdict(list)

    for emotion in valid_classes:
        sequences = sequence_info[emotion]
        num_sequences = len(sequences)

        # 计算需要的循环次数
        loops_needed = max(1, int(np.ceil(MIN_SEQUENCES / num_sequences)))
        logger.info(f"{emotion}: {num_sequences} sequences, will loop {loops_needed} times")

        # 对每个序列进行循环采样
        for loop in range(loops_needed):
            for seq in sequences:
                subject = seq['subject']
                filename = seq['filename']
                onset = seq['onset']
                apex = seq['apex']
                offset = seq['offset']

                # 获取序列的所有帧
                sequence_path = os.path.join(raw_data_dir, f"sub{subject}", filename)
                if not os.path.exists(sequence_path):
                    logger.warning(f"序列路径不存在: {sequence_path}")
                    continue

                # 获取该序列的所有帧
                frames = sorted([f for f in os.listdir(sequence_path) if f.lower().endswith('.jpg')])

                # 如果序列长度小于32帧，跳过
                if len(frames) < SEQUENCE_LENGTH:
                    logger.warning(f"序列长度不足 {SEQUENCE_LENGTH} 帧: {sequence_path}")
                    continue

                # 智能选择起始帧
                start_idx = select_sequence_start(
                    onset, apex, offset,
                    len(frames), SEQUENCE_LENGTH,
                    loop
                )

                # 提取32帧序列
                selected_frames = frames[start_idx:start_idx + SEQUENCE_LENGTH]

                # 处理选中的帧（只处理图像）
                sequence_data = []
                valid_sequence = True

                for frame_name in selected_frames:
                    frame_path = os.path.join(sequence_path, frame_name)
                    img_processed = preprocess_image(frame_path)

                    if img_processed is None:
                        valid_sequence = False
                        break

                    sequence_data.append(img_processed)

                if valid_sequence and len(sequence_data) == SEQUENCE_LENGTH:
                    # 创建序列ID
                    sequence_id = f"{subject}_{filename}_loop{loop}_start{start_idx}"

                    # 记录帧信息，用于映射关键点和光流
                    frame_info = []
                    for frame_name in selected_frames:
                        frame_id = frame_name.replace('.jpg', '')
                        frame_info.append({
                            'subject': subject,
                            'filename': filename,
                            'frame_id': frame_id
                        })

                    sequence_samples[emotion].append({
                        'id': sequence_id,
                        'frames': np.array(sequence_data),
                        'frame_info': frame_info,  # 用于映射到原始特征
                        'emotion': emotion,
                        'subject': subject,
                        'filename': filename,
                        'loop': loop,
                        'start_idx': start_idx,
                        'onset': onset,
                        'apex': apex,
                        'offset': offset
                    })

    # 打印最终的序列数量
    logger.info("最终序列数量统计：")
    for emotion in valid_classes:
        count = len(sequence_samples[emotion])
        logger.info(f"{emotion}: {count} sequences")

    return sequence_samples


def split_sequences(sequence_samples, valid_classes):
    """划分训练集和测试集"""
    logger.info("划分训练集和测试集...")

    train_sequences = []
    test_sequences = []

    for emotion in valid_classes:
        sequences = sequence_samples[emotion]

        # 分离序列ID和数据
        seq_ids = [seq['id'] for seq in sequences]

        # 分层抽样
        train_ids, test_ids = train_test_split(
            seq_ids,
            test_size=1 - train_pct,
            random_state=42
        )

        # 根据ID分配序列
        for seq in sequences:
            if seq['id'] in train_ids:
                train_sequences.append(seq)
            else:
                test_sequences.append(seq)

    logger.info(f"训练集: {len(train_sequences)} sequences")
    logger.info(f"测试集: {len(test_sequences)} sequences")

    return train_sequences, test_sequences


def apply_augmentation(frame):
    """对单帧图像应用数据增强"""
    # 随机水平翻转
    if random.random() > 0.5:
        frame = cv2.flip(frame, 1)

    # 随机亮度调整
    brightness_factor = random.uniform(0.8, 1.2)
    frame = np.clip(frame * brightness_factor, 0, 1)

    # 随机对比度调整
    contrast_factor = random.uniform(0.8, 1.2)
    frame = np.clip((frame - 0.5) * contrast_factor + 0.5, 0, 1)

    return frame


def save_sequences(train_sequences, test_sequences, valid_classes):
    """保存序列到文件系统"""
    logger.info("保存序列数据...")

    # 创建目录
    for dataset in ['train', 'test']:
        for emotion in valid_classes:
            makedir(os.path.join(sequences_dir, dataset, emotion))

    # 保存训练集序列（包含数据增强）
    train_info = []
    for seq in tqdm(train_sequences, desc="保存训练集序列"):
        emotion = seq['emotion']
        seq_id = seq['id']
        frames = seq['frames']
        frame_info = seq['frame_info']

        # 应用数据增强
        augmented_frames = []
        for frame in frames:
            if random.random() > 0.5:  # 50%概率应用增强
                frame = apply_augmentation(frame)
            augmented_frames.append(frame)

        # 保存序列数据
        base_path = os.path.join(sequences_dir, 'train', emotion, seq_id)
        frames_path = f"{base_path}_frames.npy"
        np.save(frames_path, np.array(augmented_frames))

        # 记录信息
        emotion_idx = valid_classes.index(emotion)
        train_info.append({
            'seq_id': seq_id,
            'emotion': emotion,
            'emotion_idx': emotion_idx,
            'frames_path': frames_path,
            'frame_info': frame_info  # 保留帧信息用于映射
        })

    # 保存测试集序列（不应用数据增强）
    test_info = []
    for seq in tqdm(test_sequences, desc="保存测试集序列"):
        emotion = seq['emotion']
        seq_id = seq['id']
        frames = seq['frames']
        frame_info = seq['frame_info']

        # 保存序列数据
        base_path = os.path.join(sequences_dir, 'test', emotion, seq_id)
        frames_path = f"{base_path}_frames.npy"
        np.save(frames_path, frames)

        # 记录信息
        emotion_idx = valid_classes.index(emotion)
        test_info.append({
            'seq_id': seq_id,
            'emotion': emotion,
            'emotion_idx': emotion_idx,
            'frames_path': frames_path,
            'frame_info': frame_info  # 保留帧信息用于映射
        })

    return train_info, test_info


def generate_label_files(train_info, test_info, valid_classes):
    """生成标签文件，映射到原始的关键点和光流文件"""
    logger.info("生成标签文件...")

    # 写入训练集标签文件
    with open('cls_train.txt', 'w') as f:
        for info in train_info:
            # 获取One-Hot编码
            onehot = to_categorical(info['emotion_idx'], num_classes=len(valid_classes))
            onehot_str = ','.join(map(str, onehot))

            # 构建关键点和光流文件路径列表
            landmarks_paths = []
            flow_paths = []

            for frame in info['frame_info']:
                subject = frame['subject']
                filename = frame['filename']
                frame_id = frame['frame_id']

                # 关键点文件路径
                landmark_path = os.path.join(landmarks_dir, f"sub{subject}",
                                             filename, f"{frame_id}.json")
                landmarks_paths.append(landmark_path)

                # 光流文件路径（第一帧没有光流）
                if frame == info['frame_info'][0]:
                    flow_paths.append("None")
                else:
                    flow_path = os.path.join(optical_flow_dir, f"sub{subject}",
                                             filename, f"{frame_id}.npz")
                    flow_paths.append(flow_path)

            # 将路径列表转为字符串
            landmarks_paths_str = '|'.join(landmarks_paths)
            flow_paths_str = '|'.join(flow_paths)

            # 写入格式：情绪索引;图像序列路径;关键点路径列表;光流路径列表;One-Hot编码
            f.write(
                f"{info['emotion_idx']};{info['frames_path']};{landmarks_paths_str};{flow_paths_str};{onehot_str}\n")

    # 写入测试集标签文件
    with open('cls_test.txt', 'w') as f:
        for info in test_info:
            # 获取One-Hot编码
            onehot = to_categorical(info['emotion_idx'], num_classes=len(valid_classes))
            onehot_str = ','.join(map(str, onehot))

            # 构建关键点和光流文件路径列表
            landmarks_paths = []
            flow_paths = []

            for frame in info['frame_info']:
                subject = frame['subject']
                filename = frame['filename']
                frame_id = frame['frame_id']

                # 关键点文件路径
                landmark_path = os.path.join(landmarks_dir, f"sub{subject}",
                                             filename, f"{frame_id}.json")
                landmarks_paths.append(landmark_path)

                # 光流文件路径（第一帧没有光流）
                if frame == info['frame_info'][0]:
                    flow_paths.append("None")
                else:
                    flow_path = os.path.join(optical_flow_dir, f"sub{subject}",
                                             filename, f"{frame_id}.npz")
                    flow_paths.append(flow_path)

            # 将路径列表转为字符串
            landmarks_paths_str = '|'.join(landmarks_paths)
            flow_paths_str = '|'.join(flow_paths)

            # 写入格式：情绪索引;图像序列路径;关键点路径列表;光流路径列表;One-Hot编码
            f.write(
                f"{info['emotion_idx']};{info['frames_path']};{landmarks_paths_str};{flow_paths_str};{onehot_str}\n")

    # 写入类别映射文件
    with open('class_mapping.txt', 'w') as f:
        for idx, emotion in enumerate(valid_classes):
            f.write(f"{idx}: {emotion}\n")

    logger.info(f"生成训练集标签文件: cls_train.txt ({len(train_info)} samples)")
    logger.info(f"生成测试集标签文件: cls_test.txt ({len(test_info)} samples)")
    logger.info(f"生成类别映射文件: class_mapping.txt")


def visualize_distribution(train_info, test_info, valid_classes):
    """可视化数据分布"""
    logger.info("生成数据分布可视化...")

    # 统计各类别数量
    train_counts = {emotion: 0 for emotion in valid_classes}
    test_counts = {emotion: 0 for emotion in valid_classes}

    for info in train_info:
        train_counts[info['emotion']] += 1

    for info in test_info:
        test_counts[info['emotion']] += 1

    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 训练集分布
    x = range(len(valid_classes))
    width = 0.35
    train_values = [train_counts[emotion] for emotion in valid_classes]
    bars1 = ax1.bar(x, train_values, width)
    ax1.set_xlabel('Emotion Classes')
    ax1.set_ylabel('Number of Sequences')
    ax1.set_title('Training Set Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_classes, rotation=45)

    # 在条形图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # 测试集分布
    test_values = [test_counts[emotion] for emotion in valid_classes]
    bars2 = ax2.bar(x, test_values, width)
    ax2.set_xlabel('Emotion Classes')
    ax2.set_ylabel('Number of Sequences')
    ax2.set_title('Test Set Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_classes, rotation=45)

    # 在条形图上添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('sequence_distribution.png', dpi=300)
    plt.close()

    # 打印统计信息
    logger.info("\n最终数据集统计：")
    total_train = sum(train_values)
    total_test = sum(test_values)

    for i, emotion in enumerate(valid_classes):
        logger.info(f"{emotion}:")
        logger.info(f"  训练集: {train_values[i]} sequences ({train_values[i] / total_train * 100:.1f}%)")
        logger.info(f"  测试集: {test_values[i]} sequences ({test_values[i] / total_test * 100:.1f}%)")
        logger.info(f"  总计: {train_values[i] + test_values[i]} sequences")


def main():
    """主函数"""
    try:
        # 创建必要的目录
        for dir_path in [sequences_dir, optical_flow_dir, landmarks_dir]:
            makedir(dir_path)

        # 步骤1：分析类别分布，确定要删除的类别
        valid_classes, sequence_info = analyze_class_distribution()

        # 步骤2：预处理所有原始图像（提取关键点和光流）
        preprocess_all_images()

        # 步骤3：创建标准化序列样本（只处理图像）
        sequence_samples = create_sequence_samples(sequence_info, valid_classes)

        # 步骤4：划分训练集和测试集
        train_sequences, test_sequences = split_sequences(sequence_samples, valid_classes)

        # 步骤5：保存序列数据（训练集应用数据增强）
        train_info, test_info = save_sequences(train_sequences, test_sequences, valid_classes)

        # 步骤6：生成标签文件（映射到原始特征文件）
        generate_label_files(train_info, test_info, valid_classes)

        # 步骤7：可视化数据分布
        visualize_distribution(train_info, test_info, valid_classes)

        logger.info("数据处理完成！")

        # 输出处理总结
        logger.info("\n处理总结：")
        logger.info(f"有效类别数: {len(valid_classes)}")
        logger.info(f"训练序列数: {len(train_info)}")
        logger.info(f"测试序列数: {len(test_info)}")
        logger.info(f"序列长度: {SEQUENCE_LENGTH} 帧")
        logger.info(f"图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
        logger.info("数据包含：预处理图像序列、面部关键点、光流特征")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()