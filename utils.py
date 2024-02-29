import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.special import softmax
from tqdm import tqdm
import numpy as np

def get_transforms():
    """
    Returns preprocessed and data-augmented transforms.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def select_model(model_name, num_classes=9, pretrained=True):
    """
    Select a model based on its name.
    """
    if model_name == 'efficientnet':
        from trainmodel import CustomEfficientNet
        return CustomEfficientNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vit':
        from trainmodel import CustomViT
        return CustomViT(num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train a model for an epoch.
    """
    model.train()
    train_loss, train_total, train_preds, train_labels = 0, 0, [], []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_total += labels.size(0)
        train_preds.extend(outputs.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    return train_loss / train_total, train_preds, train_labels

def validate_epoch(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    val_loss, val_total, val_preds, val_labels = 0, 0, [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_total += labels.size(0)
            val_preds.extend(outputs.detach().cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    return val_loss / val_total, val_preds, val_labels

def evaluate_model(model, data_loader, device):
    """
    Evaluate the performance of the model on a given dataset.
    """
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 在这个块内，不计算梯度
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算性能指标
    accuracy, auc_score, f1 = calculate_metrics(all_preds, all_labels)
    return accuracy, auc_score, f1, all_preds, all_labels

def calculate_metrics(predictions, labels):
    """
    Accuracy, AUC, and F1 scores are calculated and returned.
    """
    preds_softmax = softmax(np.array(predictions), axis=1)
    accuracy = accuracy_score(labels, np.argmax(preds_softmax, axis=1))
    auc_score = roc_auc_score(np.array(labels), preds_softmax, multi_class='ovr')
    f1 = f1_score(np.array(labels), np.argmax(preds_softmax, axis=1), average='macro')
    return accuracy, auc_score, f1

def save_model(model, path):
    """
    Save the model to the specified path.
    """
    torch.save(model.state_dict(), path)