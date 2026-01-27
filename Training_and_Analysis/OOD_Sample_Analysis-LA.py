import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian

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

threshold_data = pd.read_excel('../data/threshold_set.xlsx')
x_threshold_s = threshold_data.to_numpy().astype(np.float32)
x_threshold_tensor = torch.FloatTensor(x_threshold_s).to(device)


GP_KWARGS = {
    'num_inducing': 1024,   #512
    'gp_scale': 0.5,
    'gp_kernel_type': 'gaussian',     #gaussian/linear
    'gp_random_feature_type': 'rff',  #rff/orf
    'gp_bias': 0.0,
    'gp_input_normalization': True,   #
    'gp_cov_discount_factor': -1,
    'gp_cov_ridge_penalty': 1.0, #1e-2
    'gp_scale_random_features': False,
    'gp_use_custom_random_features': True,
    'gp_output_bias_trainable': True,  #False
    'gp_output_imagenet_initializer': True,
    'num_classes': 1  # 回归任务输出维度为1
}
eval_kwargs = {'return_random_features': False, 'return_covariance': True,
               'update_precision_matrix': False, 'update_covariance_matrix': False}

model = EfficientTransformerRegressor(input_dim=x_threshold_s.shape[1]).to(device)

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
    threshold_result = model(x_threshold_tensor, **eval_kwargs)
    threshold_var = threshold_result[1].diag().cpu().numpy()

    train_result = model(x_train_tensor, **eval_kwargs)
    train_var = train_result[1].diag().cpu().numpy()

#set the threshold quantiles
threshold = 80
variance_threshold = np.percentile(train_var, threshold)
print(f'Select the threshold：{threshold}  Selected variance threshold: {variance_threshold:.6f}')
model.eval()
with torch.no_grad():
    test_result= model(x_test_tensor, **eval_kwargs)
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

#t-SNE
#LA
model.eval()
with torch.no_grad():
    embedding_test = model.extract_features(x_test_tensor).cpu().numpy()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 24

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced_features = tsne.fit_transform(embedding_test)

ood_points = reduced_features[ood_mask]
id_points = reduced_features[~ood_mask]

plt.figure(figsize=(7.5, 6))
plt.scatter(id_points[:, 0], id_points[:, 1], c='blue', label='ID', alpha=0.6, s=30)
plt.scatter(ood_points[:, 0], ood_points[:, 1], c='red', label='OOD', alpha=0.6, s=30)

plt.xlabel("Component 1", fontsize=24)
plt.ylabel("Component 2", fontsize=24)

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(6))

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.legend(fontsize=24, loc='lower center', bbox_to_anchor=(0.5, 1.01),
           ncol=2, frameon=False, scatterpoints=1, markerscale=2.0)

plt.tight_layout()
plt.savefig("../result/image_result/LA-embedding_space_OOD_vs_ID.png", dpi=600, bbox_inches='tight')
plt.show()


#1D LDA visualization is performed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
id_features = test_features[~ood_mask].cpu().numpy()
ood_features = test_features[ood_mask].cpu().numpy()

lda_features = np.vstack([id_features, ood_features])
lda_labels = np.hstack([np.zeros(len(id_features)), np.ones(len(ood_features))])

lda = LinearDiscriminantAnalysis(n_components=1)
lda_transformed = lda.fit_transform(lda_features, lda_labels).flatten()
lda_id = lda_transformed[lda_labels == 0]
lda_ood = lda_transformed[lda_labels == 1]

# plot the KDE curve
kde_id = gaussian_kde(lda_id)
kde_ood = gaussian_kde(lda_ood)
x_vals = np.linspace(min(lda_transformed), max(lda_transformed), 300)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"

plt.figure(figsize=(8, 5))
plt.plot(x_vals, kde_id(x_vals), label='ID', color='#FDDDB8', linewidth=2)
plt.fill_between(x_vals, kde_id(x_vals), alpha=0.4, color='#FDDDB8')

plt.plot(x_vals, kde_ood(x_vals), label='OOD', color='#8AD2AD', linewidth=2)
plt.fill_between(x_vals, kde_ood(x_vals), alpha=0.4, color='#8AD2AD')

plt.xlabel("LDA Component 1", fontsize=16, fontweight='bold')
plt.ylabel("Density", fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("../result/image_result/LA-lda_1d_distribution.png", dpi=400)
plt.show()

#SHAP Analysis
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcdefaults()
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 17
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 17
plt.rcParams["xtick.labelsize"] = 17
plt.rcParams["ytick.labelsize"] = 17
plt.rcParams["legend.fontsize"] = 17
plt.rcParams["figure.titlesize"] = 17
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"

# Background Data & Explanatory Data
x_background = torch.tensor(x_train_s[:6000], dtype=torch.float32).to(device)
x_explain = torch.tensor(x_test_s[:2000], dtype=torch.float32).to(device)

# Wrapper Model
class WrappedSNGP(torch.nn.Module):
    def __init__(self, sngp_model):
        super().__init__()
        self.model = sngp_model.eval()

    def forward(self, x):
        pred_mean, _ = self.model(x, **eval_kwargs)
        return pred_mean

model2 = WrappedSNGP(model)

# create an explainer and calculate the SHAP values
explainer = shap.GradientExplainer(model2, x_background)
shap_values = explainer.shap_values(x_explain)

# construct the feature column names
numeric_cols = [
    'COD', 'pH', 'Cr_VI', 'Wastewater_Flow', 'Total_Nitrogen', 'Total_Phosphorus',
    'Total_Iron', 'Total_Copper', 'Total_Chromium', 'Total_Zinc', 'Total_Nickel', 'Ammonia_Nitrogen'
]

other_cols = [
    'Process_01', 'Process_02', 'Process_03', 'Process_04', 'Process_05', 'Process_06',
    'Process_07', 'Process_14', 'Process_15', 'Process_16', 'Process_17', 'Company_Scale'
]

industry_ohe_cols = [f"Industry_Type_{i}" for i in range(1, 21)]  # 20类

all_columns = numeric_cols + industry_ohe_cols + other_cols

x_explain_np = x_explain.cpu().numpy()
shap_df = pd.DataFrame(x_explain_np, columns=all_columns)

# merge the SHAP values of industry classifications
industry_idx = [shap_df.columns.get_loc(c) for c in industry_ohe_cols]

shap_vals = shap_values.squeeze()  # [Number of Samples, Number of Features]
industry_shap = shap_vals[:, industry_idx].mean(axis=1)
shap_vals_merged = np.delete(shap_vals, industry_idx, axis=1)
shap_vals_merged = np.concatenate([shap_vals_merged, industry_shap.reshape(-1, 1)], axis=1)

# construct feature values corresponding to the color bar
industry_vals = x_explain_np[:, industry_idx].sum(axis=1)
non_industry_cols = [c for c in shap_df.columns if c not in industry_ohe_cols]
shap_df_plot = shap_df[non_industry_cols].copy()
shap_df_plot["Industry_Type"] = industry_vals

feature_names = non_industry_cols + ["Industry_Type"]

# generate a swarm plot with the Spectral color palette
plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_vals_merged,
    shap_df_plot,
    feature_names=feature_names,
    plot_type="dot",
    show=False,
    max_display=25,
    cmap=plt.cm.Spectral
)

ax = plt.gca()
plt.xlabel("SHAP value", fontsize=17, fontweight='normal')
plt.ylabel("Features", fontsize=17, fontweight='normal')
plt.xticks(fontsize=17, fontweight='normal')
plt.yticks(fontsize=17, fontweight='normal')

plt.xlim(-5, 5)

cbar = plt.gcf().axes[-1]
cbar.set_ylabel("Feature value", fontsize=17, fontweight='normal')
cbar.tick_params(labelsize=17)

plt.tight_layout()
plt.savefig("../result/image_result/LA_SHAP_Beeswarm.png", dpi=400, bbox_inches='tight')
plt.show()

# Combined Plot

# calculate feature importance
feature_importance = np.abs(shap_vals_merged).mean(axis=0)
feature_names_array = np.array(feature_names)

# rank by importance in descending order
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_idx]
sorted_features = feature_names_array[sorted_idx]

fig, ax = plt.subplots(figsize=(18, 9))

colors = plt.cm.coolwarm(np.linspace(0, 1, len(sorted_importance)))

#Horizontal Bar Chart
bar_height = 0.8
extra_gap = 0.3
base_positions = np.arange(len(sorted_importance)-1, -1, -1)
y_pos = base_positions * (bar_height + extra_gap)
bars = ax.barh(
    y_pos,
    sorted_importance,
    color=colors,
    alpha=0.85,
    edgecolor='white',
    linewidth=0.8,
    height=bar_height
)

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_features, fontsize=17, fontweight='normal')

ax.set_xlabel("Mean |SHAP value|", fontsize=17, fontweight='normal')

for bar, value in zip(bars, sorted_importance):
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
            f'{value:.3f}', ha='left', va='center', fontsize=17, fontweight='normal')

ax.set_xlim(0, sorted_importance.max() * 1.2)
ax.grid(axis='x', alpha=0.25, linestyle='-')

ax.set_ylim(-1, y_pos[0] + bar_height + extra_gap)

# embed a donut chart
try:
    ax_inset = inset_axes(ax, width="70%", height="70%", loc='lower right', borderpad=3,
                         bbox_to_anchor=(-0.3, 0.05), bbox_transform=ax.transAxes)
except:
    ax_inset = ax.inset_axes([0.50, 0.15, 0.5, 0.5])

total_importance = np.sum(sorted_importance)
percentages = (sorted_importance / total_importance) * 100

n_top_features = 6
top_percentages = percentages[:n_top_features]
other_percentage = np.sum(percentages[n_top_features:])
top_labels = [f'{feat}' for feat in sorted_features[:n_top_features]]

if other_percentage > 0:
    top_percentages = np.append(top_percentages, other_percentage)
    top_labels.append('Other features')
    colors_pie = list(colors[:n_top_features]) + ['#e0e0e0']
else:
    colors_pie = colors[:n_top_features]

# Plot donut chart
wedges, texts = ax_inset.pie(
    top_percentages,
    labels=[None] * len(top_percentages),   # 不显示扇区外围标签
    startangle=90,
    colors=colors_pie,
    wedgeprops=dict(width=0.7, alpha=0.9, edgecolor='white', linewidth=1.2),
    textprops={'fontsize': 17, 'fontweight': 'normal'},
    labeldistance=1.18
)

ax_inset.set_aspect('equal')
ax_inset.axis('off')

radius_out = 0.8
for i, w in enumerate(wedges):
    ang = (w.theta2 + w.theta1) / 2.0
    ang_rad = np.deg2rad(ang)
    x = radius_out * np.cos(ang_rad)
    y = radius_out * np.sin(ang_rad)
    pct_text = f"{top_percentages[i]:.1f}%"

    ax_inset.text(x, y, pct_text,
                  ha='center', va='center',
                  fontsize=10, fontweight='normal')

plt.tight_layout()
plt.savefig("../result/image_result/LA_SHAP_Comprehensive_Analysis.png", dpi=400, bbox_inches='tight')
plt.show()

# export SHAP analysis results to Excel
excel_path = "../result/image_result/LA_SHAP_Analysis_Results.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # 1. generate a feature importance ranking table
    importance_df = pd.DataFrame({
        'Feature': sorted_features,
        'Mean_Absolute_SHAP': sorted_importance,
        'Percentage_Contribution': percentages[sorted_idx],
        'Rank': range(1, len(sorted_features) + 1)
    })
    importance_df.to_excel(writer, sheet_name='Feature_Importance_Ranking', index=False)

    # 2. Sample-Level SHAP Value Table
    sample_shap_df = pd.DataFrame(shap_vals_merged, columns=feature_names)
    sample_shap_df['Sample_ID'] = range(1, len(sample_shap_df) + 1)

    cols = ['Sample_ID'] + feature_names
    sample_shap_df = sample_shap_df[cols]
    sample_shap_df.to_excel(writer, sheet_name='Sample_SHAP_Values', index=False)

    # 3.Sample-Level Feature Value Table
    sample_feature_df = shap_df_plot.copy()
    sample_feature_df['Sample_ID'] = range(1, len(sample_feature_df) + 1)

    feature_cols = ['Sample_ID'] + feature_names
    sample_feature_df = sample_feature_df[feature_cols]
    sample_feature_df.to_excel(writer, sheet_name='Sample_Feature_Values', index=False)

    # 4. Feature Statistical Analysis Table
    stats_data = []
    for i, feature in enumerate(feature_names):
        stats_data.append({
            'Feature': feature,
            'Mean_SHAP': np.mean(shap_vals_merged[:, i]),
            'Std_SHAP': np.std(shap_vals_merged[:, i]),
            'Mean_Feature_Value': np.mean(shap_df_plot[feature]),
            'Std_Feature_Value': np.std(shap_df_plot[feature]),
            'Correlation_SHAP_Feature': np.corrcoef(shap_vals_merged[:, i], shap_df_plot[feature])[0, 1] if len(
                shap_vals_merged[:, i]) > 1 else 0
        })

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_excel(writer, sheet_name='Feature_Statistics', index=False)

print("=" * 60)
print("FEATURE IMPORTANCE RANKING (by mean |SHAP value|)")
print("=" * 60)
for i, (feature, importance) in enumerate(zip(sorted_features, sorted_importance), 1):
    print(f"{i:2d}. {feature:25s}: {importance:.6f} ({percentages[i - 1]:.2f}%)")
print("=" * 60)

print("SHAP analysis completed! The following files have been generated:")
print("1. LA_SHAP_Beeswarm.png - Swarm plot")
print("2. LA_SHAP_Comprehensive_Analysis.png - Comprehensive analysis plot")
print(f"3. {excel_path} - SHAP analysis results Excel file")
print("\nThe Excel file contains the following worksheets:")
print("   - Feature_Importance_Ranking: Feature importance ranking")
print("   - Sample_SHAP_Values: SHAP values for each sample")
print("   - Sample_Feature_Values: Feature values for each sample")
print("   - Feature_Statistics: Feature statistical information")



