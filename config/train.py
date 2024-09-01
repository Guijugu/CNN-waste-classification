import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import WasteDataset  # 假设你已经将之前的代码保存为 dataset.py
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm  # 导入 tqdm 库

# 使用预训练模型
class CNNModel(nn.Module):
    def __init__(self, num_main_categories, num_subcategories):
        super(CNNModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
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

# 数据转换和增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
train_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\train.txt", transform=transform)
val_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\val.txt", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 增加批量大小
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

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
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 调整学习率
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # 学习率调度

# 训练模型
num_epochs = 10  # 根据需要调整
best_val_loss = float('inf')  # 用于保存最佳验证损失
early_stop_counter = 0  # 早停计数器
early_stop_patience = 3  # 早停耐心值

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")  # 添加进度条
    for images, main_categories, subcategories in progress_bar:
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
        progress_bar.set_postfix(loss=running_loss/len(train_loader))  # 更新进度条上的损失

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
