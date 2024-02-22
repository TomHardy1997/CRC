import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import glob

#
sns.set(style="white", palette="cubehelix")

n_classes = 9  # Total number of categories
all_mean_tpr = []  # Store the average TPR of all folds
all_mean_fpr = np.linspace(0, 1, 100)  # Define a unified FPR
model_type = 'vit'
stage = 'val'

# Go through each file
for file_path in glob.glob(f'result_plot/{model_type}_{stage}_fold_*.csv'):
    df = pd.read_csv(file_path)
    y_true = pd.get_dummies(df[f'{stage}_label']).values
    y_score = np.vstack(df[f'{stage}_preds'].apply(eval))

    # Calculate the macro-averaged ROC curve
    tprs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        tpr_interp = np.interp(all_mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    all_mean_tpr.append(mean_tpr)

# Calculate the mean and standard deviation of the macromean TPR for all folds
mean_of_mean_tpr = np.mean(all_mean_tpr, axis=0)
std_of_mean_tpr = np.std(all_mean_tpr, axis=0)

# Calculate the AUC of the macro-averaged ROC curve
mean_auc = auc(all_mean_fpr, mean_of_mean_tpr)

# Plot the macro average ROC curve
plt.figure(figsize=(8, 6))
main_line_color = sns.cubehelix_palette(start=.5, rot=-.75, light=0.8, dark=0.3)[4]
plt.plot(all_mean_fpr, mean_of_mean_tpr, color=main_line_color, label=f'Macro-average ROC (AUC = {mean_auc:.2f})', lw=2, alpha=.8)
plt.fill_between(all_mean_fpr, mean_of_mean_tpr - std_of_mean_tpr, mean_of_mean_tpr + std_of_mean_tpr, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title(f'{model_type.capitalize()} Macro-Average {stage.capitalize()} ROC Across Folds', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
sns.despine()  # Remove borders to provide cleaner visuals
plt.savefig(f'{model_type}_{stage}_macro_avg_roc_across_folds_high_quality.svg', format='svg')
plt.show()
