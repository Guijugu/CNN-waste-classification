import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

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

# 加载标签映射
main_category_to_idx = torch.load("main_category_to_idx.pth")
subcategory_to_idx = torch.load("subcategory_to_idx.pth")
idx_to_main_category = {v: k for k, v in main_category_to_idx.items()}
idx_to_subcategory = {v: k for k, v in subcategory_to_idx.items()}

# 初始化模型
num_main_categories = len(main_category_to_idx)
num_subcategories = len(subcategory_to_idx)
model = CNNModel(num_main_categories=num_main_categories, num_subcategories=num_subcategories)
model.load_state_dict(torch.load("best_cnn_model.pth"))
model.eval()

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建图形化界面
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification")
        self.label = Label(root, text="Upload an Image for Classification")
        self.label.pack()
        self.upload_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        self.image_label = Label(root)
        self.image_label.pack()
        self.result_label = Label(root, text="")
        self.result_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.classify_image(file_path)

    def classify_image(self, file_path):
        image = Image.open(file_path)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs_main, outputs_sub = model(image)
            _, predicted_main = torch.max(outputs_main, 1)
            _, predicted_sub = torch.max(outputs_sub, 1)
            main_category = idx_to_main_category[predicted_main.item()]
            subcategory = idx_to_subcategory[predicted_sub.item()]

        self.result_label.config(text=f"Main Category: {main_category}\nSubcategory: {subcategory}")

# 运行应用程序
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
