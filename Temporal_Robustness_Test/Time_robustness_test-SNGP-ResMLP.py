import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian

# =====================================================
# Device
# =====================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =====================================================
# Model definition
# =====================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
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
    def __init__(self, input_dim=44):
        super().__init__()
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


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    train_path = "../data/train_set.xlsx"
    test_path = "../data/test_set.xlsx"
    threshold_path = "../data/threshold_set.xlsx"
    robust_path = "../data/robustness_test_data-7.xlsx"

    model_ckpt = "../all_code/MLP-best_test_model.pth"

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    threshold_df = pd.read_excel(threshold_path)
    robust_df = pd.read_excel(robust_path)

    # target column
    target_col = train_df.columns[-1]

    # inverse log transform (keep consistent)
    for df in [train_df, test_df, robust_df]:
        df['HWsum'] = np.expm1(df['HWsum'])

    # -------------------------------------------------
    # Prepare tensors
    # -------------------------------------------------
    X_test = torch.FloatTensor(
        test_df.drop(columns=[target_col]).values.astype(np.float32)
    ).to(device)
    y_test = test_df[target_col].values.astype(np.float32)

    X_threshold = torch.FloatTensor(
        threshold_df.values.astype(np.float32)
    ).to(device)

    X_robust = torch.FloatTensor(
        robust_df.drop(columns=[target_col]).values.astype(np.float32)
    ).to(device)
    y_robust = robust_df[target_col].values.astype(np.float32)

    # -------------------------------------------------
    # Build model + SNGP
    # -------------------------------------------------
    model = SNGPRegressor(input_dim=X_test.shape[1])

    model = convert_to_sn_my(
        model,
        spec_norm_replace_list=["Linear", "Conv1d"],
        spec_norm_bound=10
    )

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

    replace_layer_with_gaussian(
        container=model,
        signature="predict",
        **GP_KWARGS
    )

    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    eval_kwargs = {
        'return_random_features': False,
        'return_covariance': True,
        'update_precision_matrix': False,
        'update_covariance_matrix': False
    }

    # -------------------------------------------------
    # Step 1: Compute variance threshold (from threshold set)
    # -------------------------------------------------
    with torch.no_grad():
        model.predict.update_covariance_matrix()
        _, var_threshold = model(X_threshold, **eval_kwargs)
        all_var = var_threshold.diag().cpu().numpy()

    variance_threshold = np.percentile(all_var, 80)

    # -------------------------------------------------
    # Step 2: Robustness inference
    # -------------------------------------------------
    with torch.no_grad():
        y_pred_robust, y_var_robust = model(X_robust, **eval_kwargs)
        y_pred_robust = y_pred_robust.cpu().numpy().flatten()
        y_var_robust = y_var_robust.diag().cpu().numpy()

    ood_mask = y_var_robust > variance_threshold

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    y_pred_robust = np.expm1(y_pred_robust)
    full_r2 = r2_score(y_robust, y_pred_robust)
    full_mae = mean_absolute_error(y_robust, y_pred_robust)
    full_rmse = np.sqrt(
        mean_squared_error(y_robust, y_pred_robust)
    )

    filtered_y = y_robust[~ood_mask]
    filtered_pred = y_pred_robust[~ood_mask]

    # -------------------------------------------------
    # Print results (exact requested format)
    # -------------------------------------------------
    print('\n' + '=' * 50)
    print(f"{'Final Model Robustness Check Metrics':^50}")
    print('=' * 50)
    print(
        f"{'Before OOD Filtering':<25}: "
        f"R2 = {full_r2:.2f}, "
        f"MAE = {full_mae:.2f}, "
        f"RMSE = {full_rmse:.2f}"
    )
    print(
        f"{'OOD Samples Filtered':<25}: "
        f"{ood_mask.sum()} "
        f"({ood_mask.sum() / len(y_robust) * 100:.2f}%)"
    )

    if len(filtered_y) > 0:
        filtered_r2 = r2_score(
            filtered_y, filtered_pred
        )
        filtered_mae = mean_absolute_error(
            filtered_y, filtered_pred
        )
        filtered_rmse = np.sqrt(
            mean_squared_error(
                filtered_y, filtered_pred
            )
        )

        print(
            f"{'After OOD Filtering':<25}: "
            f"R2 = {filtered_r2:.2f}, "
            f"MAE = {filtered_mae:.2f}, "
            f"RMSE = {filtered_rmse:.2f}"
        )
        print(
            f"{'Improvement in R2':<25}: "
            f"{(filtered_r2 - full_r2):+.2f}"
        )
    else:
        print("Warning: All robustness samples filtered as OOD!")

    print('=' * 50)



    import matplotlib.pyplot as plt
    import os

    # =====================================================
    # Create output folder
    # =====================================================
    output_dir = "./noSNGP-result"
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # Plot predicted vs true values
    # =====================================================
    plt.figure(figsize=(12, 6))

    # Full prediction
    plt.plot(y_robust, label='True Values', color='black', linewidth=2)
    plt.plot(y_pred_robust, label='Predicted Values (All)', color='#2E86AB', linewidth=2, alpha=0.7)

    # Filtered prediction (after OOD removal)
    if len(filtered_y) > 0:
        plt.plot(
            np.arange(len(y_robust))[~ood_mask],
            filtered_pred,
            label='Predicted Values (Filtered)',
            color='#A23B72',
            linewidth=2,
            alpha=0.9
        )

    plt.xlabel("Sample Index", fontsize=16)
    plt.ylabel("HWsum", fontsize=16)
    plt.title("Predicted vs True HWsum", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "ResMLP-Predicted_vs_True_HWsum.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()
