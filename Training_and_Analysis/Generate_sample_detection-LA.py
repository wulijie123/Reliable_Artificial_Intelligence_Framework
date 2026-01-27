import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, \
    recall_score, f1_score
import matplotlib.pyplot as plt
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearAttention(nn.Module):

    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.k_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.o_proj = nn.Linear(num_heads * self.d_k, d_model)



    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()

        # Project to Q, K, V
        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.v_proj(v).view(batch_size, seq_len, self.num_heads, self.d_k)

        # Feature Transformation (ELU + 1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Compute normalization factor
        z = 1 / (torch.einsum('blhd,bhd->blh', q, k.sum(dim=1)) + 1e-6)

        # Compute cumulative sum
        k_cum = torch.cumsum(k, dim=1)  # [b, l, h, d_k]
        v_cum = torch.cumsum(v, dim=1)  # [b, l, h, d_k]

        # Calculate attention output
        context = torch.einsum('blhd,blhd->blh', q, k_cum)  # [b, l, h]
        context = context.unsqueeze(-1) * v_cum  # [b, l, h, d_k]
        context = context * z.unsqueeze(-1)  # 应用归一化

        # Merge heads
        context = context.contiguous().view(batch_size, seq_len, -1)

        # Project output
        output = self.o_proj(context)
        return output
class EfficientTransformerRegressor(nn.Module):

    def __init__(self, input_dim=25, d_model=64, num_heads=8,
                 num_layers=6, d_ff=256):
        super().__init__()

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # Use standard positional encoding instead of learnable
        max_len = 100
        self.position_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.position_emb, std=0.02)

        # Encoding Layer
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "attention": LinearAttention(d_model, num_heads),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model)
            })
            self.layers.append(layer)

        # Global avg pooling + regression
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU()
        )
        self.predict = nn.Linear(d_model // 2, 1)

    def forward(self, x, **kwargs):
        # Input shape: [batch, input_dim]
        # Add sequence dimension: [batch, seq_len=1, input_dim]
        x = x.unsqueeze(1)

        x = self.embedding(x)

        # Positional Encoding
        seq_len = x.size(1)
        x = x + self.position_emb[:, :seq_len]

        for layer in self.layers:
            # Attention sublayer
            attn_output = layer["attention"](x, x, x)
            x = layer["norm1"](x + attn_output)

            # FFN sublayer
            ffn_output = layer["ffn"](x)
            x = layer["norm2"](x + ffn_output)

        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, d_model]
        x = self.regressor(x)
        return self.predict(x, **kwargs)  #self.regressor(x)


# Data loading and model initialization
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

model = EfficientTransformerRegressor(input_dim=x_train_s.shape[1]).to(device)

model = convert_to_sn_my(model,
                         spec_norm_replace_list=["Linear", "Conv1d"],
                         spec_norm_bound=10)
replace_layer_with_gaussian(container=model,
                            signature="predict",
                            **GP_KWARGS)
model.load_state_dict(torch.load('./LA-80-0.77-2.pth', map_location='cuda:0'))
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

# Performance evaluation
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
    ood_indices = np.random.choice(high_var_indices, size=num_samples, replace=False)
    base_samples = x_train[ood_indices].copy()

    if disturb_all:
        pseudo_ood = base_samples.copy()
        pseudo_ood[:, :12] += np.random.normal(
            scale=disturb_strength,
            size=(base_samples.shape[0], 12)
        )
        pseudo_ood[:, 12:24] = 1 - pseudo_ood[:, 12:24]
    else:
        pseudo_ood = base_samples.copy()
        for idx in disturb_features_idx:
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


plt.figure(figsize=(8, 6))
ax = plt.gca()

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

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(true_ood_labels, predicted_ood_mask)
    precision = precision_score(true_ood_labels, predicted_ood_mask)
    recall = recall_score(true_ood_labels, predicted_ood_mask)
    f1 = f1_score(true_ood_labels, predicted_ood_mask)

    # Calculate TPR under a specific FPR
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

    # Plotting ROC curves
    plt.plot(fpr, tpr, lw=2, color=colors_global[i],
             label=f'Global-{num_samples} (AUC={roc_auc:.3f})')

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

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(true_ood_labels, predicted_ood_mask)
    precision = precision_score(true_ood_labels, predicted_ood_mask)
    recall = recall_score(true_ood_labels, predicted_ood_mask)
    f1 = f1_score(true_ood_labels, predicted_ood_mask)

    # Calculate TPR for a specific FPR
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

    # Plot the ROC curve
    plt.plot(fpr, tpr, lw=2, color=colors_local[i],
             label=f'Local-{num_samples} (AUC={roc_auc:.3f})', linestyle='--')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.7)

plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)

plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
#plt.title('LA Model - OOD Detection ROC Curves', fontsize=18)
plt.legend(loc='lower right', frameon=False, fontsize=18)

plt.tight_layout()
plt.savefig("../result/image_result/LA-ood_roc_comparison_combined.png", dpi=600, bbox_inches='tight')
plt.show()

results_df.to_excel("../result/data_result/LA_ood_detection_metrics.xlsx", index=False)
print("指标结果已保存到: ../result/data_result/LA_ood_detection_metrics.xlsx")

print("\nLA模型 OOD检测性能汇总:")
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
    "../result/image_result/LA_ood_PR_comparison_combined.png",
    dpi=800,
    bbox_inches='tight'
)
plt.show()

# =========================
# Save PR curve data
# =========================
pr_df = pd.concat(pr_tables, ignore_index=True)
pr_df.to_excel(
    "../result/data_result/LA_ood_PR_curve.xlsx",
    index=False
)

print("Standard PR curves and data have been saved.")
