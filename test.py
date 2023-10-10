import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import numpy as np
from torch.nn import DataParallel
from testdata import TestCustomDataset
from resnet import CustomNet

#数据在哪里
class_to_folder = {
    0: "ADI",
    1: "BAC",
    2: "DEB",
    3: "LYM",
    4: "MUC",
    5: "MUS",
    6: "NOR",
    7: "STR",
    8: "TUM",
    # 添加更多类别和文件夹映射
}
print(f'{class_to_folder} 标签映射好了')
df = 'test.csv'
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print('transform出发！')
test_dataset = TestCustomDataset(df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=35, shuffle=False, num_workers=0)
print('数据加载好了')
resnet = CustomNet()
resnet.load_state_dict(torch.load('saved_models/resnet_fold_1_best.pth'))
resnet.to('cuda:0')
resnet.eval()
print('模型加载好了')
result_dir = 'result'
with torch.no_grad():
    for inputs, paths in test_loader:
        inputs = inputs.to('cuda:0')
        outputs = resnet(inputs)
        _, predicted_classes = torch.max(outputs, 1)
        
        # 遍历每个批次中的每个样本
        for predicted_class, path in zip(predicted_classes, paths):
            predicted_class = predicted_class.item()
            # import ipdb;ipdb.set_trace()
            print(f'{path} 预测结果是: {predicted_class}')
            patient_id = path.split('/')[0]
            filename = path.split('/')[1]
            target_folder = os.path.join(result_dir, patient_id, class_to_folder[predicted_class])
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy(path, os.path.join(target_folder, filename))
            print(f'{path}复制到{target_folder}完成。')
print('预测完成！')
