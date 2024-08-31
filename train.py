import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import WasteDataset  # 假设你已经将之前的代码保存为 dataset.py
from torch.optim.lr_scheduler import StepLR

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

# 数据转换和增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
train_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\train.txt", transform=transform)
val_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\val.txt", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 获取子分类数量
subcategories = set()
main_categories = set()
for _, main_category, subcategory in train_dataset:
    main_categories.add(main_category)
    subcategories.add(subcategory)
num_main_categories = len(main_categories)
num_subcategories = len(subcategories)

# 创建标签映射
main_category_to_idx = {category: idx for idx, category in enumerate(main_categories)}
subcategory_to_idx = {category: idx for idx, category in enumerate(subcategories)}

# 保存标签映射
torch.save(main_category_to_idx, "main_category_to_idx.pth")
torch.save(subcategory_to_idx, "subcategory_to_idx.pth")

# 初始化模型、损失函数和优化器
model = CNNModel(num_main_categories=num_main_categories, num_subcategories=num_subcategories)
criterion_main = nn.CrossEntropyLoss()
criterion_sub = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 学习率调度

# 训练模型
num_epochs = 5 # 根据需要调整
best_val_loss = float('inf')  # 用于保存最佳验证损失
early_stop_counter = 0  # 早停计数器
early_stop_patience = 3  # 早停耐心值

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, main_categories, subcategories in train_loader:
        optimizer.zero_grad()
        outputs_main, outputs_sub = model(images)
        main_categories = torch.tensor([main_category_to_idx[main_category] for main_category in main_categories])
        subcategories = torch.tensor([subcategory_to_idx[subcategory] for subcategory in subcategories])
        loss_main = criterion_main(outputs_main, main_categories)  # 假设 main_categories 是整数标签
        loss_sub = criterion_sub(outputs_sub, subcategories)  # 假设 subcategories 是整数标签
        loss = loss_main + loss_sub
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()  # 更新学习率

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, main_categories, subcategories in val_loader:
            outputs_main, outputs_sub = model(images)
            main_categories = torch.tensor([main_category_to_idx[main_category] for main_category in main_categories])
            subcategories = torch.tensor([subcategory_to_idx[subcategory] for subcategory in subcategories])
            loss_main = criterion_main(outputs_main, main_categories)
            loss_sub = criterion_sub(outputs_sub, subcategories)
            val_loss += (loss_main + loss_sub).item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_cnn_model.pth")
        early_stop_counter = 0  # 重置早停计数器
    else:
        early_stop_counter += 1

    # 检查是否需要早停
    if early_stop_counter >= early_stop_patience:
        print("早停触发，停止训练")
        break

print("训练完成！")

