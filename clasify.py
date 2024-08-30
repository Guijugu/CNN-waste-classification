import os
import random
from PIL import Image

# 数据集路径
dataset_path = r"D:\pythonProject\chengxu\waste"
# 程序文件夹路径
program_path = r"D:\pythonProject\chengxu"

# 检查图片是否有效
def is_valid(file):
    valid = True
    try:
        Image.open(file).load()
    except:
        valid = False
    return valid

# 删除无效的图片
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(root, file)
            if not is_valid(file_path):
                print(f"删除无效的图片: {file_path}")
                os.remove(file_path)

# 分割数据集
def split_dataset(dataset_path, program_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                all_files.append(os.path.join(root, file))

    random.shuffle(all_files)
    train_split = int(train_ratio * len(all_files))
    val_split = int((train_ratio + val_ratio) * len(all_files))

    train_files = all_files[:train_split]
    val_files = all_files[train_split:val_split]
    test_files = all_files[val_split:]

    with open(os.path.join(program_path, "train.txt"), "w", encoding="utf-8") as train_file:
        for file in train_files:
            train_file.write(f"{file}\n")

    with open(os.path.join(program_path, "val.txt"), "w", encoding="utf-8") as val_file:
        for file in val_files:
            val_file.write(f"{file}\n")

    with open(os.path.join(program_path, "test.txt"), "w", encoding="utf-8") as test_file:
        for file in test_files:
            test_file.write(f"{file}\n")

# 执行数据集分割
split_dataset(dataset_path, program_path)
