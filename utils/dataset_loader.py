import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
import logging

# 配置日志记录
logger = logging.getLogger(__name__)

class CASME2Dataset(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        适配新数据格式的数据集类
        新格式: 情绪索引;图像序列路径;关键点路径列表;光流路径列表;One-Hot编码
        """
        self.sequence_paths = []
        self.landmark_paths = []
        self.flow_paths = []
        self.labels = []

        self.sequence_list = []
        self.transform = transform

        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 读取txt文件，解析每一行的序列信息
        with open(txt_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                parts = line.strip().split(';')
                if len(parts) != 5:
                    logger.warning(f"第 {line_num} 行格式错误，期望5个部分，实际{len(parts)}个")
                    continue

                try:
                    label_id = int(parts[0])
                    frames_path = os.path.join(self.root_dir, parts[1].strip())
                    landmarks_paths = [os.path.join(self.root_dir, p.strip()) for p in parts[2].split('|') if p.strip().lower() != 'none']
                    flow_paths = [os.path.join(self.root_dir, p.strip()) for p in parts[3].split('|') if p.strip().lower() != 'none']
                    onehot = [float(x) for x in parts[4].split(',')]
                    # 设定根目录为脚本根路径的上两级
                    if not os.path.exists(frames_path):
                        print(f"序列文件不存在: {frames_path}")
                        continue

                    self.sequence_paths.append(frames_path)
                    self.landmark_paths.append(landmarks_paths)
                    self.flow_paths.append(flow_paths)
                    self.labels.append(onehot)

                    self.sequence_list.append({
                        'label_id': label_id,
                        'frames_path': frames_path,
                        'landmarks_paths': landmarks_paths,
                        'flow_paths': flow_paths,
                        'onehot': onehot
                    })

                except Exception as e:
                    logger.warning(f"解析第 {line_num} 行时出错: {str(e)}")
                    continue

        logger.info(f"从 {txt_file} 加载了 {len(self.sequence_list)} 个序列")

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        info = self.sequence_list[idx]

        # 图像序列
        try:
            images = np.load(info['frames_path'])
            if len(images.shape) == 3:
                images = np.expand_dims(images, axis=1)
            images = torch.from_numpy(images).float()
        except Exception as e:
            logger.error(f"加载图像序列失败 {info['frames_path']}: {str(e)}")
            images = torch.zeros(32, 1, 128, 128)

        # 关键点
        landmarks_list = []
        for path in info['landmarks_paths']:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    points = np.array(data['landmarks'] if isinstance(data, dict) and 'landmarks' in data else data, dtype=np.float32)
                    points[:, 0] /= 128.0
                    points[:, 1] /= 128.0
                    landmarks_list.append(torch.from_numpy(points).flatten())
            except Exception as e:
                logger.warning(f"加载关键点失败 {path}: {str(e)}")
                landmarks_list.append(torch.zeros(136))

        landmarks = torch.stack(landmarks_list, dim=0)

        # 光流
        flows_list = []
        for path in info['flow_paths']:
            try:
                flow_data = np.load(path)
                flow = flow_data['flow'].astype(np.float32)
                if flow.ndim == 3 and flow.shape[2] == 2:
                    if flow.shape[:2] != (128, 128):
                        flow = np.stack([
                            cv2.resize(flow[:, :, 0], (128, 128)),
                            cv2.resize(flow[:, :, 1], (128, 128))
                        ], axis=0)
                    else:
                        flow = np.transpose(flow, (2, 0, 1))
                else:
                    logger.warning(f"光流格式异常 {path}, shape: {flow.shape}")
                    flow = np.zeros((2, 128, 128), dtype=np.float32)
                flows_list.append(torch.from_numpy(flow) / 20.0)
            except Exception as e:
                logger.warning(f"加载光流失败 {path}: {str(e)}")
                flows_list.append(torch.zeros(2, 128, 128))

        flows = torch.stack(flows_list, dim=0)
        label = torch.tensor(info['label_id'], dtype=torch.long)

        if self.transform:
            images = self.transform(images)

        return images, landmarks, flows, label
