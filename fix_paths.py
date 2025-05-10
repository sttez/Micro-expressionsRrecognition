"""
路径格式修复脚本
用于修复标签文件中混用的路径分隔符问题
"""

import os
import sys


def fix_label_files():
    """修复标签文件中的路径格式"""

    # 要修复的文件列表
    files_to_fix = ['cls_train.txt', 'cls_test.txt']

    for filename in files_to_fix:
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"警告：文件 {filename} 不存在，跳过...")
            continue

        print(f"正在修复 {filename}...")

        # 创建备份文件
        backup_filename = filename + '.backup'
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已创建备份文件：{backup_filename}")

        # 读取原文件
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 统计修复的行数
        fixed_count = 0

        # 修复路径并写回
        with open(filename, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines):
                # 将所有反斜杠替换为正斜杠
                original_line = line
                fixed_line = line.replace('\\', '/')

                if original_line != fixed_line:
                    fixed_count += 1
                    print(f"  修复第 {i + 1} 行的路径格式")

                f.write(fixed_line)

        print(f"✓ 完成！共修复了 {fixed_count} 行")
        print()

    print("所有文件修复完成！")
    print("\n下一步：")
    print("1.txt. 检查修复后的文件是否正确")
    print("2. 运行训练代码")
    print("\n如果出现问题，可以从备份文件恢复：")
    print("  将 .backup 文件重命名回原文件名")


if __name__ == "__main__":
    print("=== 标签文件路径格式修复工具 ===")
    print()

    # 显示当前工作目录
    print(f"当前工作目录：{os.getcwd()}")
    print()

    # 检查必要文件是否存在
    if not os.path.exists('cls_train.txt') and not os.path.exists('cls_test.txt'):
        print("错误：在当前目录下没有找到标签文件！")
        print("请确保在正确的目录下运行此脚本。")
        print("应该包含：cls_train.txt 和/或 cls_test.txt")
        sys.exit(1)

    # 执行修复
    fix_label_files()