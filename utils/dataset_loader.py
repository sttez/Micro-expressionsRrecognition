import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
import logging

logger = logging.getLogger(__name__)


class CASME2Dataset(Dataset):
    """CASME2 数据集加载器"""

    def __init__(self, txt_file, image_size=128, transform=None):
        self.sequence_list = []
        self.transform = transform
        self.image_size = image_size
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        with open(txt_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                parts = line.strip().split(';')
                if len(parts) != 5:
                    logger.warning(f"第 {line_num} 行格式错误: 期望5部分, 实际{len(parts)}")
                    continue
                try:
                    label_id = int(parts[0])
                    frames_path = os.path.join(self.root_dir, parts[1].strip())
                    landmarks_paths = [os.path.join(self.root_dir, p.strip())
                                       for p in parts[2].split('|') if p.strip().lower() != 'none']
                    flow_paths = [os.path.join(self.root_dir, p.strip())
                                  for p in parts[3].split('|') if p.strip().lower() != 'none']

                    if not os.path.exists(frames_path):
                        continue

                    self.sequence_list.append({
                        'label_id': label_id,
                        'frames_path': frames_path,
                        'landmarks_paths': landmarks_paths,
                        'flow_paths': flow_paths,
                    })
                except Exception as e:
                    logger.warning(f"解析第 {line_num} 行出错: {e}")

        logger.info(f"从 {txt_file} 加载了 {len(self.sequence_list)} 个序列")

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        info = self.sequence_list[idx]

        # 图像序列
        try:
            images = np.load(info['frames_path'])
            if len(images.shape) == 3:
                images = np.expand_dims(images, axis=1)
            images = torch.from_numpy(images).float()
        except Exception as e:
            logger.error(f"加载图像失败 {info['frames_path']}: {e}")
            images = torch.zeros(32, 1, self.image_size, self.image_size)

        # 关键点
        landmarks_list = []
        for path in info['landmarks_paths']:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    points = np.array(data['landmarks'] if isinstance(data, dict) and 'landmarks' in data else data, dtype=np.float32)
                    points[:, 0] /= float(self.image_size)
                    points[:, 1] /= float(self.image_size)
                    landmarks_list.append(torch.from_numpy(points).flatten())
            except Exception:
                landmarks_list.append(torch.zeros(136))
        landmarks = torch.stack(landmarks_list, dim=0)

        # 光流
        flows_list = []
        for path in info['flow_paths']:
            try:
                flow_data = np.load(path)
                flow = flow_data['flow'].astype(np.float32)
                if flow.ndim == 3 and flow.shape[2] == 2:
                    if flow.shape[:2] != (self.image_size, self.image_size):
                        flow = np.stack([
                            cv2.resize(flow[:, :, 0], (self.image_size, self.image_size)),
                            cv2.resize(flow[:, :, 1], (self.image_size, self.image_size))
                        ], axis=0)
                    else:
                        flow = np.transpose(flow, (2, 0, 1))
                else:
                    flow = np.zeros((2, self.image_size, self.image_size), dtype=np.float32)
                flows_list.append(torch.from_numpy(flow) / 20.0)
            except Exception:
                flows_list.append(torch.zeros(2, self.image_size, self.image_size))
        flows = torch.stack(flows_list, dim=0)

        label = torch.tensor(info['label_id'], dtype=torch.long)

        if self.transform:
            images = self.transform(images)

        return images, landmarks, flows, label
