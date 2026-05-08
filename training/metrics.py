import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix


def plot_loss_curve(history, save_dir):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_losses']) + 1)
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=5)
    plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_curve(history, save_dir):
    """绘制准确率曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_accs']) + 1)
    train_accs = [acc * 100 for acc in history['train_accs']]
    val_accs = [acc * 100 for acc in history['val_accs']]
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=5)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=5)
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
    """绘制 F1 分数曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_f1s']) + 1)
    plt.plot(epochs, history['train_f1s'], 'b-', label='Training F1', linewidth=2, marker='o', markersize=5)
    plt.plot(epochs, history['val_f1s'], 'r-', label='Validation F1', linewidth=2, marker='s', markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Training and Validation F1 Score', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_curve.png'), dpi=300, bbox_inches='tight')
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


def plot_training_overview(history, save_dir):
    """综合概览图（双Y轴）"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    epochs = range(1, len(history['train_losses']) + 1)
    ax2 = ax.twinx()

    # 左Y轴：损失
    ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], 'b--', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')

    # 右Y轴：准确率和F1
    train_accs = [a * 100 for a in history['train_accs']]
    val_accs = [a * 100 for a in history['val_accs']]
    ax2.plot(epochs, train_accs, 'r-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r--', label='Val Acc', linewidth=2)
    ax2.plot(epochs, history['train_f1s'], 'g-', label='Train F1', linewidth=2)
    ax2.plot(epochs, history['val_f1s'], 'g--', label='Val F1', linewidth=2)
    ax2.set_ylabel('Accuracy (%) / F1', fontsize=12)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=10)
    plt.title('Training Progress Overview', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(history, save_dir):
    """2x2 指标对比图"""
    epochs = range(1, len(history['train_losses']) + 1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.plot(epochs, history['train_losses'], 'b-', label='Train')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    train_accs = [a * 100 for a in history['train_accs']]
    val_accs = [a * 100 for a in history['val_accs']]
    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Val')
    ax2.set_title('Accuracy (%)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3.plot(epochs, history['train_f1s'], 'b-', label='Train')
    ax3.plot(epochs, history['val_f1s'], 'r-', label='Val')
    ax3.set_title('F1 Score'); ax3.legend(); ax3.grid(True, alpha=0.3)

    loss_diff = [t - v for t, v in zip(history['train_losses'], history['val_losses'])]
    acc_diff = [t - v for t, v in zip(train_accs, val_accs)]
    ax4.plot(epochs, loss_diff, 'purple', label='Loss Diff')
    ax4.plot(epochs, acc_diff, 'orange', label='Acc Diff')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Overfitting Detection'); ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(y_true, y_pred, class_names):
    """生成分类报告和 F1 分数"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return report, f1
