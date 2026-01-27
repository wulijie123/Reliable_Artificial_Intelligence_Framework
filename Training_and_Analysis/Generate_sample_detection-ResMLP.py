import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, \
    recall_score, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian

# Set global style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.linear1(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x + residual
class SNGPRegressor(nn.Module):
    def __init__(self, input_dim=25):
        super(SNGPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.predict = nn.Linear(128, 1)

    def forward(self, x, **kwargs):
        h = self.net(x)
        return self.predict(h, **kwargs)



train_data = pd.read_excel('../data/train_set.xlsx')
test_data = pd.read_excel('../data/test_set.xlsx')
target_col = train_data.columns[-1]

x_train_s = train_data.drop(columns=[target_col]).to_numpy().astype(np.float32)
y_train = train_data[target_col].to_numpy().astype(np.float32)
x_test_s = test_data.drop(columns=[target_col]).to_numpy().astype(np.float32)
y_test = test_data[target_col].to_numpy().astype(np.float32)

x_train_tensor = torch.FloatTensor(x_train_s).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
x_test_tensor = torch.FloatTensor(x_test_s).to(device)

GP_KWARGS = {
    'num_inducing': 1024,
    'gp_scale': 0.5,
    'gp_kernel_type': 'gaussian',
    'gp_random_feature_type': 'rff',
    'gp_bias': 0.0,
    'gp_input_normalization': True,
    'gp_cov_discount_factor': -1,
    'gp_cov_ridge_penalty': 1.0,
    'gp_scale_random_features': False,
    'gp_use_custom_random_features': True,
    'gp_output_bias_trainable': True,
    'gp_output_imagenet_initializer': True,
    'num_classes': 1
}

eval_kwargs = {'return_random_features': False, 'return_covariance': True,
               'update_precision_matrix': False, 'update_covariance_matrix': False}

model = SNGPRegressor(input_dim=x_train_s.shape[1]).to(device)
model = convert_to_sn_my(model,
                         spec_norm_replace_list=["Linear", "Conv1d"],
                         spec_norm_bound=10)
replace_layer_with_gaussian(container=model,
                            signature="predict",
                            **GP_KWARGS)
model.load_state_dict(torch.load('./MLP-80-0.80-2.pth', map_location=device))
model.eval()
model.predict.update_covariance_matrix()

with torch.no_grad():
    train_result = model(x_train_tensor, **eval_kwargs)
    train_var = train_result[1].diag().cpu().numpy()

#set the threshold quantiles
threshold = 80
variance_threshold = np.percentile(train_var, threshold)
print(f'Threshold selection：{threshold}  Selected variance threshold: {variance_threshold:.6f}')

model.eval()
with torch.no_grad():
    test_result = model(x_test_tensor, **eval_kwargs)
    test_pred = test_result[0].cpu().numpy()
    test_var = test_result[1].diag().cpu().numpy()

full_test_r2 = r2_score(np.expm1(y_test), np.expm1(test_pred))
full_test_mae = mean_absolute_error(np.expm1(y_test), np.expm1(test_pred))
full_test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(test_pred)))
ood_mask = test_var > variance_threshold
filtered_test_pred = test_pred[~ood_mask]
filtered_y_test = y_test[~ood_mask]

print('\n' + '=' * 50)
print(f"{'Test Set Metrics':^50}")
print('=' * 50)
print(f"{'Before OOD Filtering':<25}: R2 = {full_test_r2:.4f}, MAE = {full_test_mae:.4f}, RMSE = {full_test_rmse:.4f}")
Filtered = ood_mask.sum() / len(y_test) * 100
print(f"{'OOD Samples Filtered':<25}: {ood_mask.sum()} ({Filtered:.2f}%)")

if len(filtered_y_test) > 0:
    filtered_r2 = r2_score(np.expm1(filtered_y_test), np.expm1(filtered_test_pred))
    filtered_mae = mean_absolute_error(np.expm1(filtered_y_test), np.expm1(filtered_test_pred))
    filtered_RMSE = np.sqrt(mean_squared_error(np.expm1(filtered_y_test), np.expm1(filtered_test_pred)))
    print(f"{'After OOD Filtering':<25}: R2 = {filtered_r2:.4f}, MAE = {filtered_mae:.4f}, RMSE = {filtered_RMSE:.4f}")
    print(f"{'Improvement in R2':<25}: {(filtered_r2 - full_test_r2):+.4f}")
else:
    print("Warning: All test samples filtered as OOD!")
print('=' * 50)


# Functions for generating OOD samples
def generate_pseudo_ood(x_train, train_var_diag, threshold_percentile=80, num_samples=100,
                        disturb_all=False, disturb_strength=0.5, disturb_features_idx=None):
    threshold = np.percentile(train_var_diag, threshold_percentile)
    high_var_indices = np.where(train_var_diag > threshold)[0]
    if len(high_var_indices) == 0:

        ood_indices = np.random.choice(np.arange(len(x_train)), size=num_samples, replace=True)
    else:
        ood_indices = np.random.choice(high_var_indices, size=num_samples, replace=len(high_var_indices) < num_samples)
    base_samples = x_train[ood_indices].copy()

    if disturb_all:
        pseudo_ood = base_samples.copy()

        pseudo_ood[:, :12] += np.random.normal(
            scale=disturb_strength,
            size=(base_samples.shape[0], min(12, base_samples.shape[1]))
        )
        if base_samples.shape[1] > 12:
            end = min(24, base_samples.shape[1])
            pseudo_ood[:, 12:end] = 1 - pseudo_ood[:, 12:end]
    else:
        pseudo_ood = base_samples.copy()
        for idx in (disturb_features_idx or []):
            if idx < pseudo_ood.shape[1]:
                if idx < 12:
                    pseudo_ood[:, idx] += np.random.normal(
                        scale=disturb_strength,
                        size=base_samples.shape[0]
                    )
                elif 12 <= idx < 24:
                    pseudo_ood[:, idx] = 1 - pseudo_ood[:, idx]
    return pseudo_ood

results_df = pd.DataFrame(columns=[
    'Noise_Type', 'OOD_Samples', 'AUC',
    'Accuracy', 'Precision', 'Recall', 'F1_Score',
    'TPR_at_FPR_0.1', 'TPR_at_FPR_0.2'
])


colors_global = ['#6B9BD2', '#4A7BB5', '#2E5F8A']
colors_local = ['#E8969A', '#D67276', '#C44E52']

saved_mixtures = []

plt.figure(figsize=(8, 6))

train_var_diag = train_result[1].diag().cpu().numpy()
disturb_idx = [0, 1, 2, 3, 4, 5, 6, 17, 18, 19]
sample_sizes = [1000, 2000, 3000]

# Global noise
for i, num_samples in enumerate(sample_sizes):
    pseudo_ood_samples = generate_pseudo_ood(
        x_train_s, train_var_diag, threshold_percentile=80,
        num_samples=num_samples, disturb_all=True,
        disturb_strength=1.0, disturb_features_idx=disturb_idx
    )

    id_indices = np.where(test_var <= variance_threshold)[0]
    id_test_data = x_test_s[id_indices]
    id_test_labels = y_test[id_indices]

    mixed_data = np.concatenate([id_test_data, pseudo_ood_samples], axis=0)
    true_ood_labels = np.concatenate([
        np.zeros(len(id_test_labels)),
        np.ones(len(pseudo_ood_samples))
    ])

    mixed_tensor = torch.FloatTensor(mixed_data).to(device)
    model.eval()
    with torch.no_grad():
        mixed_result = model(mixed_tensor, **eval_kwargs)
        mixed_var = mixed_result[1].diag().cpu().numpy()

    predicted_ood_mask = mixed_var > variance_threshold

    fpr, tpr, _ = roc_curve(true_ood_labels, mixed_var)
    roc_auc = auc(fpr, tpr)

    accuracy = accuracy_score(true_ood_labels, predicted_ood_mask)
    precision = precision_score(true_ood_labels, predicted_ood_mask, zero_division=0)
    recall = recall_score(true_ood_labels, predicted_ood_mask, zero_division=0)
    f1 = f1_score(true_ood_labels, predicted_ood_mask, zero_division=0)

    tpr_at_fpr_01 = tpr[np.where(fpr <= 0.1)[0][-1]] if len(np.where(fpr <= 0.1)[0]) > 0 else 0
    tpr_at_fpr_02 = tpr[np.where(fpr <= 0.2)[0][-1]] if len(np.where(fpr <= 0.2)[0]) > 0 else 0

    results_df = pd.concat([results_df, pd.DataFrame({
        'Noise_Type': ['Global'],
        'OOD_Samples': [num_samples],
        'AUC': [roc_auc],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1],
        'TPR_at_FPR_0.1': [tpr_at_fpr_01],
        'TPR_at_FPR_0.2': [tpr_at_fpr_02]
    })], ignore_index=True)

    plt.plot(fpr, tpr, lw=2, color=colors_global[i],
             label=f'Global-{num_samples} (AUC={roc_auc:.3f})')

    saved_mixtures.append({
        'label': f'Global-{num_samples}',
        'mixed_var': mixed_var,
        'true_ood_labels': true_ood_labels,
        'color': colors_global[i],
        'linestyle': '-'
    })

# Local noise
for i, num_samples in enumerate(sample_sizes):
    pseudo_ood_samples = generate_pseudo_ood(
        x_train_s, train_var_diag, threshold_percentile=80,
        num_samples=num_samples, disturb_all=False,
        disturb_strength=1.0, disturb_features_idx=disturb_idx
    )

    id_indices = np.where(test_var <= variance_threshold)[0]
    id_test_data = x_test_s[id_indices]
    id_test_labels = y_test[id_indices]

    mixed_data = np.concatenate([id_test_data, pseudo_ood_samples], axis=0)
    true_ood_labels = np.concatenate([
        np.zeros(len(id_test_labels)),
        np.ones(len(pseudo_ood_samples))
    ])

    mixed_tensor = torch.FloatTensor(mixed_data).to(device)
    model.eval()
    with torch.no_grad():
        mixed_result = model(mixed_tensor, **eval_kwargs)
        mixed_var = mixed_result[1].diag().cpu().numpy()

    predicted_ood_mask = mixed_var > variance_threshold

    fpr, tpr, _ = roc_curve(true_ood_labels, mixed_var)
    roc_auc = auc(fpr, tpr)

    accuracy = accuracy_score(true_ood_labels, predicted_ood_mask)
    precision = precision_score(true_ood_labels, predicted_ood_mask, zero_division=0)
    recall = recall_score(true_ood_labels, predicted_ood_mask, zero_division=0)
    f1 = f1_score(true_ood_labels, predicted_ood_mask, zero_division=0)

    tpr_at_fpr_01 = tpr[np.where(fpr <= 0.1)[0][-1]] if len(np.where(fpr <= 0.1)[0]) > 0 else 0
    tpr_at_fpr_02 = tpr[np.where(fpr <= 0.2)[0][-1]] if len(np.where(fpr <= 0.2)[0]) > 0 else 0

    results_df = pd.concat([results_df, pd.DataFrame({
        'Noise_Type': ['Local'],
        'OOD_Samples': [num_samples],
        'AUC': [roc_auc],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1],
        'TPR_at_FPR_0.1': [tpr_at_fpr_01],
        'TPR_at_FPR_0.2': [tpr_at_fpr_02]
    })], ignore_index=True)

    plt.plot(fpr, tpr, lw=2, color=colors_local[i],
             label=f'Local-{num_samples} (AUC={roc_auc:.3f})', linestyle='--')

    saved_mixtures.append({
        'label': f'Local-{num_samples}',
        'mixed_var': mixed_var,
        'true_ood_labels': true_ood_labels,
        'color': colors_local[i],
        'linestyle': '--'
    })

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.7)
plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.legend(loc='lower right', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig("../result/image_result/MLP-ood_roc_comparison_combined.png", dpi=600, bbox_inches='tight')
plt.show()

results_df.to_excel("../result/data_result/MLP_ood_detection_metrics.xlsx", index=False)
print("指标结果已保存到: ../result/data_result/MLP_ood_detection_metrics.xlsx")

print("\nOOD检测性能汇总:")
print(results_df.to_string(index=False, float_format='%.4f'))


from sklearn.metrics import precision_recall_curve, average_precision_score

plt.figure(figsize=(8, 6))

pr_tables = []

for i, num_samples in enumerate(sample_sizes):

    # =========================
    # Global noise
    # =========================
    pseudo_ood_samples = generate_pseudo_ood(
        x_train_s,
        train_var_diag,
        threshold_percentile=80,
        num_samples=num_samples,
        disturb_all=True,
        disturb_strength=1.0,
        disturb_features_idx=disturb_idx
    )

    id_indices = np.where(test_var <= variance_threshold)[0]
    id_test_data = x_test_s[id_indices]

    mixed_data = np.concatenate([id_test_data, pseudo_ood_samples], axis=0)
    true_ood_labels = np.concatenate([
        np.zeros(len(id_test_data)),
        np.ones(len(pseudo_ood_samples))
    ])

    mixed_tensor = torch.FloatTensor(mixed_data).to(device)
    model.eval()
    with torch.no_grad():
        mixed_result = model(mixed_tensor, **eval_kwargs)
        mixed_var = mixed_result[1].diag().cpu().numpy()

    precision, recall, _ = precision_recall_curve(true_ood_labels, mixed_var)
    ap = average_precision_score(true_ood_labels, mixed_var)

    plt.plot(
        recall,
        precision,
        color=colors_global[i],
        linewidth=2,
        label=f'Global-{num_samples} (AP = {ap:.3f})'
    )

    pr_tables.append(pd.DataFrame({
        'recall': recall,
        'precision': precision,
        'scenario': 'Global',
        'num_ood_samples': num_samples
    }))

    # =========================
    # Local noise
    # =========================
    pseudo_ood_samples = generate_pseudo_ood(
        x_train_s,
        train_var_diag,
        threshold_percentile=80,
        num_samples=num_samples,
        disturb_all=False,
        disturb_strength=1.0,
        disturb_features_idx=disturb_idx
    )

    mixed_data = np.concatenate([id_test_data, pseudo_ood_samples], axis=0)
    true_ood_labels = np.concatenate([
        np.zeros(len(id_test_data)),
        np.ones(len(pseudo_ood_samples))
    ])

    mixed_tensor = torch.FloatTensor(mixed_data).to(device)
    model.eval()
    with torch.no_grad():
        mixed_result = model(mixed_tensor, **eval_kwargs)
        mixed_var = mixed_result[1].diag().cpu().numpy()

    precision, recall, _ = precision_recall_curve(true_ood_labels, mixed_var)
    ap = average_precision_score(true_ood_labels, mixed_var)

    plt.plot(
        recall,
        precision,
        color=colors_local[i],
        linewidth=2,
        linestyle='--',
        label=f'Local-{num_samples} (AP = {ap:.3f})'
    )

    pr_tables.append(pd.DataFrame({
        'recall': recall,
        'precision': precision,
        'scenario': 'Local',
        'num_ood_samples': num_samples
    }))

# =========================
# Figure styling
# =========================
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.xlim(0, 1)
plt.ylim(0, 1.02)
handles, labels = plt.gca().get_legend_handles_labels()

# Sort by Global/Local in the labels and sample size
def legend_sort_key(label):
    if label.startswith('Global'):
        group = 0
    else:
        group = 1
    num = int(label.split('-')[1].split()[0])
    return (group, num)

sorted_items = sorted(zip(handles, labels), key=lambda x: legend_sort_key(x[1]))
sorted_handles, sorted_labels = zip(*sorted_items)
plt.legend(
    sorted_handles,
    sorted_labels,
    frameon=False,
    fontsize=14
)
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    "../result/image_result/MLP_ood_PR_comparison_combined.png",
    dpi=800,
    bbox_inches='tight'
)
plt.show()

# =========================
# Save PR curve data
# =========================
pr_df = pd.concat(pr_tables, ignore_index=True)
pr_df.to_excel(
    "../result/data_result/MLP_ood_PR_curve.xlsx",
    index=False
)

print("Standard PR curves and data have been saved.")
