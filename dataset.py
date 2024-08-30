import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义数据集类
class WasteDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data = []
        self.transform = transform
        with open(txt_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 1:
                    path = parts[0]
                    # 从路径中提取主分类和子分类
                    main_category = os.path.basename(os.path.dirname(path)).split('_')[0]
                    subcategory = os.path.basename(os.path.dirname(path)).split('_')[1]
                    self.data.append((path, main_category, subcategory))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, main_category, subcategory = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, main_category, subcategory

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
train_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\train.txt", transform=transform)
val_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\val.txt", transform=transform)
test_dataset = WasteDataset(txt_file=r"D:\pythonProject\chengxu\test.txt", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
