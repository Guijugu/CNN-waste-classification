import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import WasteDataset  # 假设你已经将之前的代码保存为 dataset.py
import matplotlib.pyplot as plt

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self, num_main_categories, num_subcategories):
        super(CNNModel, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.base_model.fc.in_features  # 获取输入特征的数量
        self.base_model.fc = nn.Identity()  # 移除最后一层
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.fc2_main = nn.Linear(256, num_main_categories)
        self.fc2_sub = nn.Linear(256, num_subcategories)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(self.relu(self.fc1(x)))
        main_category = self.fc2_main(x)
        subcategory = self.fc2_sub(x)
        return main_category, subcategory

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
val_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\val.txt", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # 增加批量大小

# 加载标签映射
main_category_to_idx = torch.load("main_category_to_idx.pth")
subcategory_to_idx = torch.load("subcategory_to_idx.pth")

# 初始化模型
num_main_categories = len(main_category_to_idx)
num_subcategories = len(subcategory_to_idx)
model = CNNModel(num_main_categories=num_main_categories, num_subcategories=num_subcategories)
model.load_state_dict(torch.load("best_cnn_model.pth"))
model.eval()

# 验证模型
criterion_main = nn.CrossEntropyLoss()
criterion_sub = nn.CrossEntropyLoss()
val_loss = 0.0
correct_main = 0
correct_sub = 0
total = 0

accuracies_main = []
accuracies_sub = []

with torch.no_grad():
    for images, main_categories, subcategories in val_loader:
        outputs_main, outputs_sub = model(images)
        main_categories = torch.tensor([main_category_to_idx[main_category] for main_category in main_categories])
        subcategories = torch.tensor([subcategory_to_idx[subcategory] for subcategory in subcategories])
        loss_main = criterion_main(outputs_main, main_categories)
        loss_sub = criterion_sub(outputs_sub, subcategories)
        val_loss += (loss_main + loss_sub).item()

        _, predicted_main = torch.max(outputs_main, 1)
        _, predicted_sub = torch.max(outputs_sub, 1)
        total += main_categories.size(0)
        correct_main += (predicted_main == main_categories).sum().item()
        correct_sub += (predicted_sub == subcategories).sum().item()

        accuracies_main.append(100 * correct_main / total)
        accuracies_sub.append(100 * correct_sub / total)

val_loss /= len(val_loader)
accuracy_main = 100 * correct_main / total
accuracy_sub = 100 * correct_sub / total

print(f"Validation Loss: {val_loss}")
print(f"Main Category Accuracy: {accuracy_main}%")
print(f"Subcategory Accuracy: {accuracy_sub}%")

# 绘制准确率变化图
plt.figure(figsize=(10, 5))
plt.plot(accuracies_main, label='Main Category Accuracy')
plt.plot(accuracies_sub, label='Subcategory Accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Change Over Batches')
plt.legend()
plt.show()

