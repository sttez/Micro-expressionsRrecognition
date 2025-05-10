import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import configparser

# 设置字体，优先使用微软雅黑或 SimHei
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


def get_config_info():
    """从配置文件获取类别信息"""
    config = configparser.ConfigParser()
    config.read('config.ini')

    if 'CLASSES' in config and 'valid_classes' in config['CLASSES']:
        class_names = config['CLASSES']['valid_classes'].split(',')
        # 去除可能的空格
        class_names = [name.strip() for name in class_names]
        return class_names
    else:
        # 如果配置文件不存在或格式不对，返回默认值
        return ['surprise', 'repression', 'happiness', 'disgust', 'others']


def show_image(tensor_image, title=None):
    """显示单张图像（不需要修改）"""
    image = tensor_image.squeeze().cpu().numpy()
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def plot_training_history(train_loss, train_acc, val_loss=None, val_acc=None):
    """绘制训练历史（增强版）"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', marker='o', markersize=3)
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy', marker='o', markersize=3)
    if val_acc is not None:
        plt.plot(val_acc, label='Validation Accuracy', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def show_confusion_matrix(y_true, y_pred, class_names=None):
    """显示混淆矩阵（改进版）"""
    if class_names is None:
        class_names = get_config_info()

    # 确保预测值和真实值在有效范围内
    valid_indices = (y_true < len(class_names)) & (y_pred < len(class_names))
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    cm = confusion_matrix(y_true, y_pred)

    # 创建一个更美观的混淆矩阵
    plt.figure(figsize=(10, 8))

    # 使用百分比显示
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 创建混合显示（数量 + 百分比）
    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_training_summary(losses, accs, f1s, y_true, y_pred, y_probs, num_classes=None, save_dir='results', prefix=""):
    """绘制训练总结（完全改进版）"""
    os.makedirs(save_dir, exist_ok=True)

    # 获取类别名称
    class_names = get_config_info()
    if num_classes is None:
        num_classes = len(class_names)

    # 1.txt. Loss & Accuracy & F1 曲线
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(losses) + 1)

    plt.subplot(2, 1, 1)
    plt.plot(epochs, losses, 'b-', label='Train Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, accs, 'g-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, f1s, 'r-', label='F1 Score', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Metrics', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}training_curves.png"), dpi=300)
    plt.close()

    # 2. 混淆矩阵（带类别名称）
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    # 计算百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 创建标签（数量 + 百分比）
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'

    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=class_names[:cm.shape[1]],
                yticklabels=class_names[:cm.shape[0]])

    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}confusion_matrix.png"), dpi=300)
    plt.close()

    # 3. ROC 曲线（带类别名称）
    y_probs = np.array(y_probs)
    if (isinstance(y_probs, np.ndarray) and
            len(np.unique(y_true)) >= 2 and
            len(np.unique(y_pred)) >= 2 and
            len(y_probs.shape) == 2 and
            y_probs.shape[1] == num_classes):

        try:
            # One-Hot 编码
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

            # 绘制每个类别的 ROC 曲线
            plt.figure(figsize=(10, 8))

            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])

                # 使用实际的类别名称
                class_name = class_names[i] if i < len(class_names) else f'Class {i}'
                plt.plot(fpr, tpr, linewidth=2,
                         label=f'{class_name} (AUC = {auc:.4f})')

            # 绘制对角线
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

            # 图表设置
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title("ROC Curves for Each Class", fontsize=16)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}roc_curves.png"), dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error plotting ROC curves: {e}")


# 新增：类别分布可视化
def plot_class_distribution(labels, dataset_name='Dataset'):
    """绘制类别分布图"""
    class_names = get_config_info()
    unique_labels, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_labels, counts, color='skyblue', edgecolor='navy', alpha=0.7)

    # 在条形图上添加数值
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        plt.text(label, count + max(counts) * 0.01, str(count),
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xlabel('Emotion Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Class Distribution in {dataset_name}', fontsize=16)
    plt.xticks(unique_labels, [class_names[i] for i in unique_labels], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()