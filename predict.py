import torch
import torch.nn as nn
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms

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

# 加载模型
num_main_categories = 4  # 有害垃圾、厨余垃圾、可回收物、其他垃圾
num_subcategories = 10  # 根据你的实际子分类数量修改
model = CNNModel(num_main_categories=num_main_categories, num_subcategories=num_subcategories)
model.load_state_dict(torch.load("best_cnn_model.pth"))
model.eval()

# 图形化界面
class WasteClassifierApp:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        self.root = tk.Tk()
        self.root.title("Waste Classifier")
        self.label = tk.Label(self.root, text="请选择一张图片进行分类")
        self.label.pack()
        self.button = tk.Button(self.root, text="选择图片", command=self.load_image)
        self.button.pack()
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path).convert("RGB")
            image = self.transform(image).unsqueeze(0)
            self.classify_image(image)

    def classify_image(self, image):
        self.model.eval()
        with torch.no_grad():
            outputs_main, outputs_sub = self.model(image)
            main_category = torch.argmax(outputs_main, dim=1).item()
            subcategory = torch.argmax(outputs_sub, dim=1).item()
            self.result_label.config(text=f"主分类: {main_category}, 子分类: {subcategory}")

    def run(self):
        self.root.mainloop()

# 启动图形化界面
app = WasteClassifierApp(model, transform)
app.run()
