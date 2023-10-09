import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import lmdb
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import KFold
from torch.nn import DataParallel
from PIL import Image
import numpy as np
import cv2
from data import CustomDataset
from resnet import CustomNet
from sklearn.model_selection import train_test_split

# 1. 数据预处理
df = pd.read_csv('shuffled_data.csv')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 加载模型和微调
resnet = CustomNet(num_classes=9, pretrained=True)

# 3. 定义训练器和评估器
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# resnet = DataParallel(resnet)  # 使用 DataParallel 在多个 GPU 上并行训练
resnet.to(device)  # 将模型移动到 GPU 上
optimizer = optim.Adam(resnet.parameters(), lr=0.001, weight_decay=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# 4. 设置交叉验证
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
results = []

# 创建测试集 DataLoader
test_dataset = CustomDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=135, shuffle=False, num_workers=0)
test_loader = tqdm(test_loader, desc=f'Fold {fold} Testing')

for train_idx, val_idx in kf.split(train_df):
    fold += 1
    train_fold_df = train_df.iloc[train_idx]
    val_fold_df = train_df.iloc[val_idx]

    train_dataset = CustomDataset(train_fold_df, transform=transform)
    val_dataset = CustomDataset(val_fold_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=135, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=135, shuffle=False, num_workers=0)

    train_loader = tqdm(train_loader, desc=f'Fold {fold} Training')
    val_loader = tqdm(val_loader, desc=f'Fold {fold} Validation')

    # 训练模型
    num_epochs = 50
    best_auc = 0.0
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        resnet.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

        train_accuracy = correct_preds / total_preds

        # 在验证集上计算多分类AUC和准确率
        resnet.eval()
        all_probs = []
        all_labels = []
        correct_preds_val = 0
        total_preds_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = resnet(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                _, predicted = torch.max(outputs.data, 1)
                total_preds_val += labels.size(0)
                correct_preds_val += (predicted == labels).sum().item()

        val_accuracy = correct_preds_val / total_preds_val

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

        print(f'Fold {fold}, Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Acc: {train_accuracy}, Val Acc: {val_accuracy}, AUC: {auc}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_auc = auc
            torch.save(resnet.state_dict(), f'saved_models/resnet_fold_{fold}_best.pth')

    # 记录验证集结果
    results.append({
        'fold': fold,
        'best_auc': best_auc,
        'best_val_accuracy': best_val_accuracy,
    })

    # 测试集评估
    resnet.eval()
    all_test_probs = []
    all_test_labels = []
    correct_preds_test = 0
    total_preds_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = resnet(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_test_probs.append(probs.cpu().numpy())
            all_test_labels.append(labels.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total_preds_test += labels.size(0)
            correct_preds_test += (predicted == labels).sum().item()

    test_accuracy = correct_preds_test / total_preds_test
    all_test_probs = np.concatenate(all_test_probs)
    all_test_labels = np.concatenate(all_test_labels)
    test_auc = roc_auc_score(all_test_labels, all_test_probs, multi_class='ovr', average='macro')

    print(f'Fold {fold}, Test Acc: {test_accuracy}, Test AUC: {test_auc}')

    # 记录测试集结果
    results[fold - 1]['test_accuracy'] = test_accuracy
    results[fold - 1]['test_auc'] = test_auc

# 将结果输出到 CSV 文件
results_df = pd.DataFrame(results)
results_df.to_csv('cross_validation_results.csv', index=False)

# for train_idx, val_idx in kf.split(df):
#     fold += 1
#     train_fold_df = df.iloc[train_idx]
#     val_fold_df = df.iloc[val_idx]

#     train_dataset = CustomDataset(train_fold_df, transform=transform)
#     val_dataset = CustomDataset(val_fold_df, transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2) 

#     # 训练模型
#     num_epochs = 50
#     best_auc = 0.0

#     for epoch in range(num_epochs):
#         resnet.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             outputs = resnet(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
        
#         # 在验证集上计算多分类AUC
#         resnet.eval()
#         all_probs = []
#         all_labels = []
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 outputs = resnet(inputs)
#                 probs = torch.softmax(outputs, dim=1)
#                 all_probs.append(probs.cpu().numpy())
#                 all_labels.append(labels.cpu().numpy())
        
#         all_probs = np.concatenate(all_probs)
#         all_labels = np.concatenate(all_labels)
#         auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        
#         print(f'Fold {fold}, Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, AUC: {auc}')
        
#         if auc > best_auc:
#             best_auc = auc
#             torch.save(resnet.state_dict(), f'resnet_fold_{fold}_best.pth')
    
#     # 记录结果
#     results.append({
#         'fold': fold,
#         'best_auc': best_auc
#     })

# # 将结果输出到 CSV 文件
# results_df = pd.DataFrame(results)
# results_df.to_csv('cross_validation_results.csv', index=False)