import torch
import pandas as pd
from torch.utils.data import DataLoader
import utils
from data import CustomDataset

def eval(model_types, test_csv, num_classes=9, batch_size=210):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    test_df = pd.read_csv(test_csv)
    transform = utils.get_transforms()
    test_dataset = CustomDataset(test_df, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for model_type in model_types:
        for fold in range(1, 11):
            model_path = f'saved_models/{model_type}_fold_{fold}.pth'
            model = utils.select_model(model_type, num_classes=num_classes,pretrained=False)

            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            model.to(device)
            model.load_state_dict(torch.load(model_path))

            accuracy, auc_score, f1, all_preds, all_labels = utils.evaluate_model(model, test_loader, device)
            print(f"Model: {model_type}, Fold {fold}: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f}, F1 Score = {f1:.4f}")
            test_df = pd.DataFrame({'test_label':all_labels,'test_preds':all_preds})
            test_df.to_csv(f'results/{model_type}_test_fold_{fold}.csv', index=None)
            results.append({
                'Model': model_type,
                'Fold': fold,
                'Accuracy': accuracy,
                'AUC': auc_score,
                'F1 Score': f1
            })

    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{model_type}_evaluation_results.csv', index=False)

if __name__ == "__main__":
    model_types = ['efficientnet', 'vit']  
    test_csv = 'splits/independent_test_set.csv'  
    eval(model_types, test_csv)
