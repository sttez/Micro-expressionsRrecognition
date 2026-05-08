import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import sys
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm

from core.config import Config
from utils.dataset_loader import CASME2Dataset
from models.microexpression_model import MicroExpressionModel, freeze_backbone, unfreeze_all
from training.metrics import (
    plot_loss_curve, plot_accuracy_curve, plot_f1_curve,
    plot_confusion_matrix, plot_training_overview, plot_metrics_comparison,
    generate_report
)


def setup_logging(model_dir):
    """配置日志"""
    log_file = os.path.join(model_dir, 'training.log')
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def create_model_folder(config, model_name="MicroExpModel"):
    """创建模型保存文件夹（临时名称，训练后重命名）"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_name = f"{model_name}_{timestamp}_temp"
    weights_dir = config.get_weights_dir()
    os.makedirs(weights_dir, exist_ok=True)
    model_dir = os.path.join(weights_dir, temp_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir, model_name, timestamp


def rename_model_folder(temp_dir, model_name, timestamp, best_accuracy, logger=None):
    """训练结束后根据准确率重命名文件夹"""
    acc_str = f"{best_accuracy:.2f}"
    new_name = f"{model_name}_{timestamp}_{acc_str}"
    parent = os.path.dirname(temp_dir)
    new_dir = os.path.join(parent, new_name)
    counter = 1
    original = new_dir
    while os.path.exists(new_dir):
        new_dir = f"{original}_{counter}"
        counter += 1
    try:
        os.rename(temp_dir, new_dir)
        return new_dir
    except PermissionError:
        import shutil
        time.sleep(2)
        try:
            shutil.move(temp_dir, new_dir)
            return new_dir
        except Exception:
            if os.path.exists(new_dir) and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return new_dir if os.path.exists(new_dir) else temp_dir


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0
    all_preds = []
    all_labels = []

    epoch_start = time.time()
    step_times = []

    progress_bar = tqdm(train_loader, desc='Training', ncols=100)
    for images, landmarks, flows, labels in progress_bar:
        step_start = time.time()
        lr = optimizer.param_groups[0]['lr']

        images = images.to(device)
        landmarks = landmarks.to(device)
        flows = flows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, landmarks, flows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        step_times.append((time.time() - step_start) * 1000)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        corrects += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 实时显示当前 loss 和 accuracy
        cur_loss = running_loss / total
        cur_acc = 100. * corrects / total
        progress_bar.set_postfix(Loss=f'{cur_loss:.4f}', Acc=f'{cur_acc:.1f}%')

    epoch_time = time.time() - epoch_start
    avg_step = np.mean(step_times) if step_times else 0

    from sklearn.metrics import f1_score
    return (
        running_loss / total,
        corrects / total,
        f1_score(all_labels, all_preds, average='weighted'),
        epoch_time, avg_step, lr
    )


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc='Validation', ncols=100)
        for images, landmarks, flows, labels in val_bar:
            images = images.to(device)
            landmarks = landmarks.to(device)
            flows = flows.to(device)
            labels = labels.to(device)

            outputs = model(images, landmarks, flows)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            cur_loss = running_loss / total
            cur_acc = 100. * corrects / total
            val_bar.set_postfix(Loss=f'{cur_loss:.4f}', Acc=f'{cur_acc:.1f}%')

    from sklearn.metrics import f1_score
    return (
        running_loss / total,
        corrects / total,
        f1_score(all_labels, all_preds, average='weighted'),
        all_preds, all_labels
    )


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs, device, model_dir, logger, patience=10):
    """完整训练流程（含 early stopping）"""
    history = {
        'train_losses': [], 'train_accs': [], 'train_f1s': [],
        'val_losses': [], 'val_accs': [], 'val_f1s': []
    }
    best_val_acc = 0.0
    patience_counter = 0

    logger.info(f"开始训练 | 设备: {device} | Epochs: {num_epochs}")
    logger.info(f"训练集批次数: {len(train_loader)} | 验证集批次数: {len(val_loader)}")
    logger.info("")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1, epoch_time, avg_step, lr = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        if scheduler:
            scheduler.step(val_loss)

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['train_f1s'].append(train_f1)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        history['val_f1s'].append(val_f1)

        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"LR: {lr:.6f} - "
            f"耗时: {int(epoch_time)}s/{int(avg_step)}ms - "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} - "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_f1': val_f1,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(model_dir, 'best_model.pth'))
            logger.info(f">> 保存最佳模型 (acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f">> Early stopping: {patience} 轮无改善")
            break

        # 定期绘图
        if epoch % 5 == 0 or epoch == num_epochs:
            plot_loss_curve(history, model_dir)
            plot_accuracy_curve(history, model_dir)
            plot_f1_curve(history, model_dir)
            if epoch >= num_epochs * 0.8:
                plot_training_overview(history, model_dir)
                plot_metrics_comparison(history, model_dir)

    logger.info(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
    return history, best_val_acc * 100, val_preds, val_labels


def main():
    """主训练入口（支持阶段式训练）"""
    torch.manual_seed(42)
    np.random.seed(42)

    config = Config()
    model_dir, model_name, timestamp = create_model_folder(config)
    logger = setup_logging(model_dir)

    logger.info(f"=== 开始新的训练任务 ===")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"模型目录: {model_dir}\n")

    # 读取训练配置
    stage = config.getint('training', 'stage', 0)
    batch_size = config.getint('training', 'batch_size', 8)
    num_epochs = config.getint('training', 'num_epochs', 30)
    learning_rate = config.getfloat('training', 'learning_rate', 0.0001)
    stage1_lr = config.getfloat('training', 'stage1_learning_rate', 0.001)
    stage1_epochs = config.getint('training', 'stage1_epochs', 25)
    patience = config.getint('training', 'early_stopping_patience', 10)
    val_ratio = 1.0 - config.getfloat('training', 'train_split', 0.8)
    sequence_length = config.getint('preprocessing', 'sequence_length', 32)
    valid_classes = config.get_class_names()
    num_classes = len(valid_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"阶段: {stage} | 类别: {valid_classes} | 类别数: {num_classes}")
    logger.info(f"批次: {batch_size} | 学习率: {learning_rate} | 序列长度: {sequence_length}")
    logger.info("")

    # 加载数据
    logger.info("加载数据集...")
    train_txt = config.get_label_train()
    test_txt = config.get_label_test()

    full_dataset = CASME2Dataset(train_txt)
    if len(full_dataset) == 0:
        logger.error("训练集为空，终止训练")
        return

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset = CASME2Dataset(test_txt)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logger.info(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}\n")

    # 创建模型
    model = MicroExpressionModel(num_classes=num_classes, sequence_length=sequence_length).to(device)
    criterion = nn.CrossEntropyLoss()

    # 阶段式训练
    if stage == 1:
        # 阶段1: 冻结 LSTM+Classifier，只训练 CNN/FC
        logger.info("=== 阶段1: 训练 CNN 特征提取器 ===")
        freeze_backbone(model)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=stage1_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        history, best_acc, _, _ = train(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            stage1_epochs, device, model_dir, logger, patience
        )
        # 保存阶段1 checkpoint
        stage1_path = os.path.join(model_dir, 'stage1_best.pth')
        torch.save({'model_state_dict': model.state_dict()}, stage1_path)
        logger.info(f"阶段1完成，checkpoint 保存至: {stage1_path}")

    elif stage == 2:
        # 阶段2: 加载阶段1权重，解冻全部，联合训练
        logger.info("=== 阶段2: 联合微调 ===")
        stage1_ckpt = config.get('training', 'stage1_checkpoint', '')
        if stage1_ckpt and os.path.exists(stage1_ckpt):
            checkpoint = torch.load(stage1_ckpt, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"已加载阶段1权重: {stage1_ckpt}")
        else:
            logger.info("无阶段1权重，从头训练")

        unfreeze_all(model)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        history, best_acc, val_preds, val_labels = train(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs, device, model_dir, logger, patience
        )

    else:
        # stage == 0: 标准全量训练
        logger.info("=== 标准训练模式 ===")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        history, best_acc, val_preds, val_labels = train(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs, device, model_dir, logger, patience
        )

    # 测试集评估
    logger.info("\n=== 测试集评估 ===")
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(model, test_loader, criterion, device)
    logger.info(f"测试准确率: {test_acc:.4f} | 测试F1: {test_f1:.4f}")

    report, _ = generate_report(test_labels, test_preds, valid_classes)
    logger.info(f"\n分类报告:\n{report}")
    plot_confusion_matrix(test_labels, test_preds, valid_classes, model_dir)

    # 保存结果
    results = {
        'model_name': model_name, 'timestamp': timestamp,
        'stage': stage, 'best_val_accuracy': best_acc,
        'test_accuracy': test_acc * 100, 'test_f1_score': test_f1,
        'classification_report': report,
        'config': {
            'num_classes': num_classes, 'classes': valid_classes,
            'sequence_length': sequence_length, 'batch_size': batch_size,
            'learning_rate': learning_rate, 'num_epochs': num_epochs
        }
    }
    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=str)

    # 重命名文件夹
    try:
        final_dir = rename_model_folder(model_dir, model_name, timestamp, best_acc, logger)
    except Exception as e:
        logger.warning(f"重命名失败: {e}")
        final_dir = model_dir

    logger.info(f"模型保存在: {final_dir}")
    logger.info(f"最佳验证准确率: {best_acc:.2f}%")
    logger.info(f"最终测试准确率: {test_acc * 100:.2f}%")

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


if __name__ == '__main__':
    main()
