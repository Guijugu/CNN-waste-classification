import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import WasteDataset  # 假设你已经将之前的代码保存为 dataset.py

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self, num_main_categories, num_subcategories):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2_main = nn.Linear(128, num_main_categories)
        self.fc2_sub = nn.Linear(128, num_subcategories)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.dropout(self.relu(self.fc1(x)))  # 应用 Dropout
        main_category = self.fc2_main(x)
        subcategory = self.fc2_sub(x)
        return main_category, subcategory

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
test_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\test.txt", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 获取子分类数量
subcategories = set()
for _, _, subcategory in test_dataset:
    subcategories.add(subcategory)
num_subcategories = len(subcategories)

# 初始化模型
num_main_categories = 4  # 有害垃圾、厨余垃圾、可回收物、其他垃圾
model = CNNModel(num_main_categories=num_main_categories, num_subcategories=num_subcategories)
model.load_state_dict(torch.load("best_cnn_model.pth"))
model.eval()

# 验证模型
criterion_main = nn.CrossEntropyLoss()
criterion_sub = nn.CrossEntropyLoss()
test_loss = 0.0
correct_main = 0
correct_sub = 0
total = 0

with torch.no_grad():
    for images, main_categories, subcategories in test_loader:
        outputs_main, outputs_sub = model(images)
        main_categories = torch.tensor([int(main_category) for main_category in main_categories])
        subcategories = torch.tensor([int(subcategory) for subcategory in subcategories])
        loss_main = criterion_main(outputs_main, main_categories)
        loss_sub = criterion_sub(outputs_sub, subcategories)
        test_loss += (loss_main + loss_sub).item()

        _, predicted_main = torch.max(outputs_main, 1)
        _, predicted_sub = torch.max(outputs_sub, 1)
        total += main_categories.size(0)
        correct_main += (predicted_main == main_categories).sum().item()
        correct_sub += (predicted_sub == subcategories).sum().item()

test_loss /= len(test_loader)
accuracy_main = 100 * correct_main / total
accuracy_sub = 100 * correct_sub / total

print(f"Test Loss: {test_loss}")
print(f"Main Category Accuracy: {accuracy_main}%")
print(f"Subcategory Accuracy: {accuracy_sub}%")
