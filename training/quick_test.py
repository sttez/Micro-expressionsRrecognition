"""
快速测试训练脚本 - 用少量数据验证 pipeline
直接从标签文件抽取样本索引，避免遍历加载全部数据
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
import time
from collections import defaultdict
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.config import Config
from utils.dataset_loader import CASME2Dataset
from models.microexpression_model import MicroExpressionModel
from sklearn.metrics import f1_score


def sample_indices_from_label_file(txt_file, samples_per_class=50):
    """直接从标签文件按类别抽样索引，不加载数据"""
    label_indices = defaultdict(list)
    with open(txt_file, 'r') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            parts = line.strip().split(';')
            if len(parts) >= 1:
                try:
                    label = int(parts[0])
                    label_indices[label].append(idx)
                except ValueError:
                    pass

    selected = []
    for label, indices in label_indices.items():
        np.random.shuffle(indices)
        n = min(samples_per_class, len(indices))
        selected.extend(indices[:n])
        print(f"  class {label}: {n}/{len(indices)}")

    np.random.shuffle(selected)
    return selected


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    train_txt = config.get_label_train()
    test_txt = config.get_label_test()

    # 从标签文件直接抽样（不加载数据）
    print("训练集抽样:")
    train_indices = sample_indices_from_label_file(train_txt, samples_per_class=50)
    print(f"  总计: {len(train_indices)}\n")

    print("测试集抽样:")
    test_indices = sample_indices_from_label_file(test_txt, samples_per_class=20)
    print(f"  总计: {len(test_indices)}\n")

    # 加载数据集（此时才真正加载数据）
    print("加载数据...")
    t0 = time.time()
    full_train = CASME2Dataset(train_txt)
    test_dataset = CASME2Dataset(test_txt)
    print(f"加载完成 [{time.time()-t0:.1f}s]\n")

    train_subset = Subset(full_train, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    batch_size = 8
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 创建模型
    num_classes = config.get_num_classes()
    seq_len = config.getint('preprocessing', 'sequence_length', 32)
    model = MicroExpressionModel(num_classes=num_classes, sequence_length=seq_len).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练 5 个 epoch
    num_epochs = 5
    print(f"开始训练 ({num_epochs} epochs)...\n")

    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_preds, train_labels = [], []

        t0 = time.time()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', ncols=100)
        for images, landmarks, flows, labels in pbar:
            images, landmarks, flows, labels = images.to(device), landmarks.to(device), flows.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, landmarks, flows)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            cur_loss = train_loss / train_total
            cur_acc = 100. * train_correct / train_total
            pbar.set_postfix(Loss=f'{cur_loss:.4f}', Acc=f'{cur_acc:.1f}%')

        epoch_time = time.time() - t0
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, landmarks, flows, labels in tqdm(test_loader, desc='  Validating', ncols=100):
                images, landmarks, flows, labels = images.to(device), landmarks.to(device), flows.to(device), labels.to(device)
                outputs = model(images, landmarks, flows)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_f1 = f1_score(val_labels, val_preds, average='weighted') if val_total > 0 else 0

        print(f"Epoch {epoch}/{num_epochs} [{epoch_time:.1f}s] - "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

    print(f"\n=== 测试通过 ===")
    print(f"最终训练准确率: {train_acc:.4f}")
    print(f"最终验证准确率: {val_acc:.4f}")
    print(f"模型前向推理和反向传播正常")


if __name__ == '__main__':
    main()
