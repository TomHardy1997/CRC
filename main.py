import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import utils
from torch.utils.data import DataLoader
from data import CustomDataset

# Parameter settings
model_types = ['efficientnet', 'vit']  # A list of the types of models that will be trained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = 'splits'
n_folds = 10
num_classes = 9  
patience = 5
# Data Conversion
transform = utils.get_transforms()

# Cross-validate each model type
for model_type in model_types:
    print(f"Starting training for model type: {model_type}")
    results = []
    start_time = time.time()

    for fold in range(1, n_folds + 1):
        print(f"Starting Fold {fold} for {model_type}")
        train_df = pd.read_csv(f'{splits_dir}/train_fold_{fold}.csv')
        val_df = pd.read_csv(f'{splits_dir}/val_fold_{fold}.csv')

        train_dataset = CustomDataset(train_df, transform=transform)
        val_dataset = CustomDataset(val_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=210,shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=210, shuffle=False)

        model = utils.select_model(model_type, num_classes=num_classes).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        num_epochs_no_improvement = 0
        for epoch in range(1, 31):  # The epochs can be adjusted as needed
            train_loss, train_preds, train_labels = utils.train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_labels = utils.validate_epoch(model, val_loader, criterion, device)
            # val_df = pd.DataFrame({'val_label': val_labels, 'val_preds': val_preds})
            # val_df.to_csv(f'results/{model_type}_val_fold_{fold}.csv', index=None)
            train_accuracy, train_auc, train_f1 = utils.calculate_metrics(train_preds, train_labels)
            val_accuracy, val_auc, val_f1 = utils.calculate_metrics(val_preds, val_labels)

            print(f"Fold {fold}, Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f},Train ACC:{train_accuracy:.4f}, Val ACC:{val_accuracy:.4f}, Train F1 Score:{train_f1:.4f}, Val F1 Score:{val_f1:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_epochs_no_improvement = 0
                val_df = pd.DataFrame({'val_label': val_labels, 'val_preds': val_preds})
                val_df.to_csv(f'results/{model_type}_val_fold_{fold}.csv', index=None)
                utils.save_model(model, f'saved_models/{model_type}_fold_{fold}.pth')
            else:
                num_epochs_no_improvement += 1
                if num_epochs_no_improvement == patience:
                    print(f"Early stopping at fold {fold}")
                    break
        # Save the results of each fold
        results.append({
            'fold': fold,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_f1': train_f1,
            'val_f1': val_f1,
        })

        # Save the model
        utils.save_model(model, f'saved_models/{model_type}_fold_{fold}.pth')

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Training execution time for {model_type}: {execution_time:.2f} minutes")
    pd.DataFrame(results).to_csv(f'{model_type}_cross_validation_results.csv', index=False)