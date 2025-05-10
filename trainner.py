"""
CASME2 微表情识别训练脚本
特点：
- 分离的训练曲线图（损失、准确率、F1分数）
- 详细的训练日志格式
- 智能的文件夹命名系统
- 完整的结果保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import logging
import configparser
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入自定义模块
from utils.dataset_loader import CASME2Dataset
from models.microexpression_model import MicroExpressionModel

def create_model_folder(model_name="MicroExpModel"):
    """
    创建模型保存文件夹
    使用临时名称，训练结束后根据准确率重命名
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_folder_name = f"{model_name}_{timestamp}_temp"

    # 在models/weights目录下创建文件夹
    weights_dir = os.path.join(current_dir, 'models','weights')
    os.makedirs(weights_dir, exist_ok=True)

    model_dir = os.path.join(weights_dir, temp_folder_name)
    os.makedirs(model_dir, exist_ok=True)

    return model_dir, model_name, timestamp


def rename_model_folder(temp_dir, model_name, timestamp, best_accuracy, logger=None):
    """
    训练结束后，根据最佳准确率重命名文件夹
    """
    # 格式化准确率为两位小数
    accuracy_str = f"{best_accuracy:.2f}"

    # 新的文件夹名称
    new_folder_name = f"{model_name}_{timestamp}_{accuracy_str}"

    # 获取父目录和新路径
    parent_dir = os.path.dirname(temp_dir)
    new_dir = os.path.join(parent_dir, new_folder_name)

    # 如果目标文件夹已存在，添加编号
    counter = 1
    original_new_dir = new_dir
    while os.path.exists(new_dir):
        new_dir = f"{original_new_dir}_{counter}"
        counter += 1

    try:
        # 重命名文件夹
        os.rename(temp_dir, new_dir)
        return new_dir  # 成功返回新路径
    except PermissionError:
        # 如果重命名失败，使用复制方法
        import shutil
        import time

        # 等待片刻，让文件句柄释放
        time.sleep(2)

        try:
            # 尝试使用shutil.move
            shutil.move(temp_dir, new_dir)
            return new_dir  # 成功返回新路径
        except Exception as e:
            # 最后检查是否真的失败
            if os.path.exists(new_dir):
                # 新文件夹存在，尝试删除临时文件夹
                if os.path.exists(temp_dir):
                    try:
                        time.sleep(1)  # 等待资源释放
                        shutil.rmtree(temp_dir)
                        if logger:
                            logger.info(f"已删除临时文件夹: {temp_dir}")
                    except:
                        if logger:
                            logger.warning(f"无法删除临时文件夹: {temp_dir}")

                if logger:
                    logger.info(f"文件夹已成功重命名为: {new_dir}")
                return new_dir
            else:
                # 真的失败了
                if logger:
                    logger.warning(f"无法重命名文件夹，保留临时名称: {temp_dir}")
                return temp_dir

def setup_logging(model_dir):
    """
    配置日志系统
    """
    log_file = os.path.join(model_dir, 'training.log')

    # 创建logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 设置简单格式（因为我们会手动格式化日志信息）
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def plot_loss_curve(history, save_dir):
    """绘制损失曲线图"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_losses']) + 1)

    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss',
             linewidth=2, marker='o', markersize=5)
    plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss',
             linewidth=2, marker='s', markersize=5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_curve(history, save_dir):
    """绘制准确率曲线图"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_accs']) + 1)

    # 将准确率转换为百分比形式
    train_accs_percent = [acc * 100 for acc in history['train_accs']]
    val_accs_percent = [acc * 100 for acc in history['val_accs']]

    plt.plot(epochs, train_accs_percent, 'b-', label='Training Accuracy',
             linewidth=2, marker='o', markersize=5)
    plt.plot(epochs, val_accs_percent, 'r-', label='Validation Accuracy',
             linewidth=2, marker='s', markersize=5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_curve(history, save_dir):
    """绘制F1分数曲线图"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_f1s']) + 1)

    plt.plot(epochs, history['train_f1s'], 'b-', label='Training F1 Score',
             linewidth=2, marker='o', markersize=5)
    plt.plot(epochs, history['val_f1s'], 'r-', label='Validation F1 Score',
             linewidth=2, marker='s', markersize=5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Training and Validation F1 Score', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'f1_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_overview(history, save_dir):
    """绘制综合概览图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    epochs = range(1, len(history['train_losses']) + 1)

    # 使用双Y轴显示不同尺度的指标
    ax2 = ax.twinx()

    # 左Y轴：损失
    loss_line1 = ax.plot(epochs, history['train_losses'], 'b-',
                        label='Training Loss', linewidth=2)
    loss_line2 = ax.plot(epochs, history['val_losses'], 'b--',
                        label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')

    # 右Y轴：准确率和F1分数
    train_accs_percent = [acc * 100 for acc in history['train_accs']]
    val_accs_percent = [acc * 100 for acc in history['val_accs']]

    acc_line1 = ax2.plot(epochs, train_accs_percent, 'r-',
                        label='Training Accuracy', linewidth=2)
    acc_line2 = ax2.plot(epochs, val_accs_percent, 'r--',
                        label='Validation Accuracy', linewidth=2)
    f1_line1 = ax2.plot(epochs, history['train_f1s'], 'g-',
                       label='Training F1', linewidth=2)
    f1_line2 = ax2.plot(epochs, history['val_f1s'], 'g--',
                       label='Validation F1', linewidth=2)
    ax2.set_ylabel('Accuracy (%) / F1 Score', fontsize=12)

    # 合并图例
    lines = loss_line1 + loss_line2 + acc_line1 + acc_line2 + f1_line1 + f1_line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=10)

    plt.title('Training Progress Overview', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(history, save_dir):
    """创建详细的指标对比图"""
    epochs = range(1, len(history['train_losses']) + 1)

    # 创建一个2x2的子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 损失对比
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation', linewidth=2)
    ax1.set_title('Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率对比
    train_accs_percent = [acc * 100 for acc in history['train_accs']]
    val_accs_percent = [acc * 100 for acc in history['val_accs']]
    ax2.plot(epochs, train_accs_percent, 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, val_accs_percent, 'r-', label='Validation', linewidth=2)
    ax2.set_title('Accuracy Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # F1分数对比
    ax3.plot(epochs, history['train_f1s'], 'b-', label='Training', linewidth=2)
    ax3.plot(epochs, history['val_f1s'], 'r-', label='Validation', linewidth=2)
    ax3.set_title('F1 Score Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 学习曲线差异（过拟合检测）
    loss_diff = [train - val for train, val in zip(history['train_losses'], history['val_losses'])]
    acc_diff = [train - val for train, val in zip(train_accs_percent, val_accs_percent)]

    ax4.plot(epochs, loss_diff, 'purple', label='Loss Difference', linewidth=2)
    ax4.plot(epochs, acc_diff, 'orange', label='Accuracy Difference (%)', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Overfitting Detection')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training - Validation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()

    running_loss = 0.0
    corrects = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    # 记录开始时间
    epoch_start_time = time.time()
    step_times = []

    progress_bar = tqdm(train_loader, desc='Training', ncols=100)

    for batch_idx, (images, landmarks, flows, labels) in enumerate(progress_bar):
        step_start_time = time.time()
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        images = images.to(device)
        landmarks = landmarks.to(device)
        flows = flows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, landmarks, flows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 记录步骤时间
        step_time = (time.time() - step_start_time) * 1000  # 转换为毫秒
        step_times.append(step_time)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        corrects += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        current_loss = running_loss / total_samples
        current_acc = 100. * corrects / total_samples
        progress_bar.set_postfix(Loss=f'{current_loss:.4f}', Acc=f'{current_acc:.2f}%')

    epoch_time = time.time() - epoch_start_time
    avg_step_time = np.mean(step_times)

    epoch_loss = running_loss / total_samples
    epoch_acc = corrects / total_samples  # 不乘以100，为了日志格式
    epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1, epoch_time, avg_step_time,current_lr

def validate(model, val_loader, criterion, device):
    """验证模型性能"""
    model.eval()

    running_loss = 0.0
    corrects = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', ncols=100)

        for images, landmarks, flows, labels in progress_bar:
            images = images.to(device)
            landmarks = landmarks.to(device)
            flows = flows.to(device)
            labels = labels.to(device)

            outputs = model(images, landmarks, flows)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            corrects += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / total_samples
    val_acc = corrects / total_samples  # 不乘以100，为了日志格式
    val_f1 = f1_score(all_labels, all_predictions, average='weighted')

    return val_loss, val_acc, val_f1, all_predictions, all_labels, all_probs

def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs, device, model_dir, logger):
    """完整的训练流程"""

    history = {
        'train_losses': [],
        'train_accs': [],
        'train_f1s': [],
        'val_losses': [],
        'val_accs': [],
        'val_f1s': []
    }

    best_val_acc = 0.0

    logger.info("开始训练...")
    logger.info(f"设备: {device}")
    logger.info(f"训练集批次数: {len(train_loader)}")
    logger.info(f"验证集批次数: {len(val_loader)}")
    logger.info("")

    for epoch in range(1, num_epochs + 1):
        # 训练一个epoch
        train_loss, train_acc, train_f1, epoch_time, avg_step_time, current_lr = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 验证
        val_loss, val_acc, val_f1, val_predictions, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )

        # 调整学习率
        if scheduler:
            scheduler.step(val_loss)  # 添加 val_loss 参数

        # 记录历史
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['train_f1s'].append(train_f1)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        history['val_f1s'].append(val_f1)

        # 格式化日志信息
        log_message = (
            f"Epoch {epoch}/{num_epochs} - "
            f"学习率: {current_lr:.6f} - "  # 添加学习率输出
            f"耗时: {int(epoch_time)}s, 每步: {int(avg_step_time)}ms - "
            f"训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f} - "
            f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f} - "
            f"F1 分数: {val_f1:.4f}"
        )
        logger.info(log_message)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_f1': val_f1,
                'optimizer_state_dict': optimizer.state_dict()
            }

            model_path = os.path.join(model_dir, 'best_model.pth')
            torch.save(checkpoint, model_path)
            logger.info(f">> 保存新的最佳模型 (准确率: {best_val_acc:.4f})")

        # 定期更新训练曲线
        if epoch % 5 == 0 or epoch == num_epochs:
            plot_loss_curve(history, model_dir)
            plot_accuracy_curve(history, model_dir)
            plot_f1_curve(history, model_dir)

            # 在训练后期生成更多图表
            if epoch >= num_epochs * 0.8:
                plot_training_overview(history, model_dir)
                plot_metrics_comparison(history, model_dir)

    logger.info("")
    logger.info(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")

    return history, best_val_acc * 100, val_predictions, val_labels, val_probs

def main():
    """主函数"""

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建模型文件夹
    model_dir, model_name, timestamp = create_model_folder()

    # 设置日志
    logger = setup_logging(model_dir)
    logger.info(f"=== 开始新的训练任务 ===")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"模型目录: {model_dir}\n")

    # 正确设置 config.ini 的路径（位于项目根目录）
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, 'config.ini')

    config = configparser.ConfigParser()
    config.read(config_path)

    if 'PARAMETERS' not in config or 'CLASSES' not in config:
        logger.error(f"配置文件缺失 'PARAMETERS' 或 'CLASSES' 部分: {config_path}")
        return

    try:
        sequence_length = int(config['PARAMETERS']['sequence_length'])
        valid_classes = config['CLASSES']['valid_classes'].split(',')
    except KeyError as e:
        logger.error(f"配置参数缺失: {e}")
        return

    num_classes = len(valid_classes)

    batch_size = 8
    num_epochs = 1
    learning_rate = 0.0001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("加载数据集...")

    train_txt = os.path.join(project_root, 'cls_train.txt')
    test_txt = os.path.join(project_root, 'cls_test.txt')

    logger.info(f"训练文件路径: {train_txt}")
    logger.info(f"测试文件路径: {test_txt}")

    train_dataset = CASME2Dataset(train_txt)
    if len(train_dataset) == 0:
        logger.error("训练集为空，终止训练。")
        return

    test_dataset = CASME2Dataset(test_txt)
    val_size = int(len(test_dataset) * 0.5)
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logger.info(f"训练集: {len(train_dataset)} 样本")
    #logger.info(f"验证集: {len(val_dataset)} 样本")
    logger.info(f"测试集: {len(test_dataset)} 样本")
    logger.info(f"类别: {valid_classes}\n")

    model = MicroExpressionModel(num_classes=num_classes, sequence_length=sequence_length).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控验证损失
        factor=0.5,  # 每次降低50%
        patience=3,  # 3个epoch无改善才降低
        min_lr=1e-6,  # 最小学习率
    )

    history, best_val_acc, val_predictions, val_labels, val_probs = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs, device, model_dir, logger
    )

    logger.info("\n=== 在测试集上评估 ===")
    test_loss, test_acc, test_f1, test_predictions, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )
    logger.info(f"测试准确率: {test_acc:.4f}")
    logger.info(f"测试F1分数: {test_f1:.4f}")

    report = classification_report(test_labels, test_predictions, target_names=valid_classes, digits=4)
    logger.info(f"\n分类报告:\n{report}")

    plot_confusion_matrix(test_labels, test_predictions, valid_classes, model_dir)

    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc * 100,
        'test_f1_score': test_f1,
        'training_history': {
            'train_losses': history['train_losses'],
            'train_accuracies': [acc * 100 for acc in history['train_accs']],
            'train_f1_scores': history['train_f1s'],
            'val_losses': history['val_losses'],
            'val_accuracies': [acc * 100 for acc in history['val_accs']],
            'val_f1_scores': history['val_f1s']
        },
        'classification_report': report,
        'config': {
            'num_classes': num_classes,
            'valid_classes': valid_classes,
            'sequence_length': sequence_length,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs
        }
    }

    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=str)

    # 在main函数最后部分
    logger.info(f"\n=== 训练完成 ===")
    try:
        final_model_dir = rename_model_folder(model_dir, model_name, timestamp, best_val_acc, logger)
        model_save_path = final_model_dir
    except Exception as e:
        logger.warning(f"重命名文件夹失败: {e}")
        model_save_path = model_dir

    # 在main函数最后部分
    logger.info(f"\n=== 训练完成 ===")
    try:
        final_model_dir = rename_model_folder(model_dir, model_name, timestamp, best_val_acc, logger)
    except Exception as e:
        logger.warning(f"重命名文件夹失败: {e}")
        final_model_dir = model_dir

    logger.info(f"模型保存在: {final_model_dir}")
    logger.info(f"最佳验证准确率: {best_val_acc:.2f}%")
    logger.info(f"最终测试准确率: {test_acc * 100:.2f}%")

    # 关闭logger的所有文件处理器
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    # 尝试删除临时文件夹（如果还存在）
    if final_model_dir != model_dir and os.path.exists(model_dir):
        try:
            import shutil
            import time
            time.sleep(1)  # 等待文件系统释放
            shutil.rmtree(model_dir)
            print(f"成功删除临时文件夹: {model_dir}")  # 使用print而不是logger
        except Exception as e:
            print(f"最终仍无法删除临时文件夹: {model_dir}, 错误: {e}")


if __name__ == '__main__':
    main()