from numpy.random import default_rng
from sklearn.neighbors import KDTree
from scipy.stats import gaussian_kde
from sklearn import metrics
import matplotlib.patheffects as PathEffects
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian


def calc_dist(train_X, calib_X, random_seed=None, nearest_neighbors=10, metric="minkowski"):
    np.random.seed(random_seed)

    std_scaler = StandardScaler().fit(train_X)
    train_X_std_scaled = std_scaler.transform(train_X)
    calib_X_std_scaled = std_scaler.transform(calib_X)

    kdtree = KDTree(train_X_std_scaled, metric=metric)
    dist, ind = kdtree.query(calib_X_std_scaled, k=nearest_neighbors)

    return dist.mean(axis=1)


class ConformalPrediction():
    def __init__(self, residuals_calib, heurestic_uncertainty_calib, alpha) -> None:
        # score function
        scores = abs(residuals_calib / heurestic_uncertainty_calib)
        scores = np.array(scores)

        n = len(residuals_calib)
        qhat = torch.quantile(torch.from_numpy(scores), np.ceil(n * (1 - alpha)) / n)
        qhat_value = np.float64(qhat.numpy())
        self.qhat = qhat_value
        pass

    def predict(self, heurestic_uncertainty_test):
        return heurestic_uncertainty_test * self.qhat, self.qhat


def error_list(true_list, pred_list):
    true_value = np.array(true_list)
    pred_value = np.array(pred_list)
    list_of_error = []

    for i in range(len(true_list)):
        error = pred_value[i] - true_value[i]
        list_of_error.append(error)

    return list_of_error


def plot_scatter_residuals_vs_heurestic_uncertainty(heurestic_uncertainty_test, residuals_test, qhats, ps,
                                                    xlabel="dist", title=None, dpi=300):
    heurestic_uncertainty_test = copy.deepcopy(heurestic_uncertainty_test)
    residuals_test = copy.deepcopy(residuals_test)
    qhat1, qhat2 = qhats
    p_1, p_2, p_3 = ps

    min_xxx = np.min(heurestic_uncertainty_test)
    max_xxx = np.max(heurestic_uncertainty_test)
    xxx = max_xxx - min_xxx
    yyy = np.max(residuals_test)

    ref_fx = np.linspace(np.amin(heurestic_uncertainty_test), np.amax(heurestic_uncertainty_test), 100)  # 画斜线用的，不管

    xy = np.vstack([heurestic_uncertainty_test, residuals_test])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    heurestic_uncertainty_test, residuals_test, z = heurestic_uncertainty_test[idx], residuals_test[idx], z[idx]

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(10, 6))  # 设置图形大小
    # 68%的线
    ax.plot(ref_fx, ref_fx * qhat1, c=colors[0])
    ax.plot(ref_fx, - ref_fx * qhat1, c=colors[0])
    # 95%的线
    ax.plot(ref_fx, ref_fx * qhat2, c=colors[1])
    ax.plot(ref_fx, - ref_fx * qhat2, c=colors[1])

    ax.fill_between(ref_fx, ref_fx * qhat1, - ref_fx * qhat1, alpha=0.2, color=colors[0], linewidth=0)
    ax.fill_between(ref_fx, ref_fx * qhat2, - ref_fx * qhat2, alpha=0.2, color=colors[1], linewidth=0)
    ax.plot(ref_fx, [0] * 100, "--", color="darkgray")

    # x_lower_lim = np.quantile(heurestic_uncertainty_test, 1e-4)
    # x_upper_lim = np.quantile(heurestic_uncertainty_test, 0.998)

    x_lower_lim = heurestic_uncertainty_test
    x_upper_lim = heurestic_uncertainty_test

    # force share-y
    ax.set_xlim([min_xxx, max_xxx])

    # add text of prob
    prob_posi = (x_upper_lim - x_lower_lim) * .8
    # txt = ax.text(min_xxx + (xxx) * 0.4, yyy / 8, "{:.0f}%".format(p_1 * 100), color=colors[2])
    txt = ax.text(5.2, 1200, "{:.0f}%".format(p_1 * 100), color=colors[2], fontsize=14)  # 68，增大字体
    txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='w')])  # 文本描边
    # txt = ax.text(min_xxx + (xxx) * 0.05, yyy / 1.3, "{:.0f}%".format(p_2 * 100), color=colors[2])
    txt = ax.text(1.2, 1200, "{:.0f}%".format(p_2 * 100), color=colors[2], fontsize=14)  # 95，增大字体
    txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='w')])
    txt = ax.text(0.03, 1200, "{:.0f}%".format(p_3 * 100), color="k", fontsize=14)  # 1-95，增大字体
    txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='w')])

    ax.scatter(heurestic_uncertainty_test, residuals_test, c=z, s=20, alpha=0.5, cmap='summer')
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel("residual", fontsize=22)

    # 增大刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=16)

    if title is not None:
        ax.set_title(title)

    return fig, ax



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


def evaluate_conformal_performance(conf_levels, list_error_calib, calib_dist, list_error_test, test_dist, Y_pred_test):
    """
    Evaluate the performance of control points (CP) at different confidence levels
    :param conf_levels: List of confidence levels [0.68, 0.80, 0.90, 0.95, 0.99]
    :return: DataFrame containing the evaluation metrics
    """
    results = []

    for conf_level in conf_levels:
        alpha = 1 - conf_level

        # calculate the CP model
        model_cp = ConformalPrediction(list_error_calib, calib_dist, alpha=alpha)
        test_uncertainty, qhat = model_cp.predict(test_dist)

        # Calculate the actual confidence level
        covered = np.sum(np.abs(list_error_test) <= test_uncertainty)
        observed_conf = covered / len(list_error_test)

        # Calculate the average interval width
        lower = Y_pred_test - test_uncertainty
        upper = Y_pred_test + test_uncertainty
        lower = np.maximum(lower, 0)
        avg_interval_width = np.mean(upper - lower)  # Two-sided interval width

        # Calculate the confidence level deviation
        conf_diff = observed_conf - conf_level

        # Coverage rate of raw predicted values
        if Y_pred_test is not None and Y_test is not None:
            coverage_pred = np.mean((Y_test >= lower) & (Y_test <= upper))
        else:
            coverage_pred = np.nan

        results.append({
            "Expected Confidence": conf_level,
            "Observed Confidence": observed_conf,
            "Confidence Difference": conf_diff,
            "Avg. Interval Width": avg_interval_width,
            "Qhat": qhat,
            "Coverage Rate": coverage_pred
        })

    return pd.DataFrame(results)


csv_name = "../result/MLP-cp_performance_metrics_detailed-7.csv"
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_list = "../data/robustness_test_data-7.xlsx"
    train_list = "../data/train_set.xlsx"
    calib_list = "../data/calib_set.xlsx"
    threshold_list = "../data/threshold_set.xlsx"

    test_df = pd.read_excel(test_list)
    train_df = pd.read_excel(train_list)
    calib_df = pd.read_excel(calib_list)
    threshold_df = pd.read_excel(threshold_list)

    train_df['HWsum'] = np.expm1(train_df['HWsum'])
    test_df['HWsum'] = np.expm1(test_df['HWsum'])
    calib_df['HWsum'] = np.expm1(calib_df['HWsum'])

    target_col = train_df.columns[-1]

    # Data preparation
    X_train = train_df.drop(columns=[target_col]).values.astype(np.float32)
    Y_train = train_df[target_col].values.astype(np.float32)
    X_test = test_df.drop(columns=[target_col]).values.astype(np.float32)
    Y_test = test_df[target_col].values.astype(np.float32)
    X_calib = calib_df.drop(columns=[target_col]).values.astype(np.float32)
    Y_calib = calib_df["HWsum"].values.astype(np.float32)
    X_threshold = threshold_df.values.astype(np.float32)

    # Convert to tensors
    X_test = torch.FloatTensor(X_test).to(device)
    X_calib = torch.FloatTensor(X_calib).to(device)
    X_threshold = torch.FloatTensor(X_threshold).to(device)

    model = SNGPRegressor(input_dim=44)  # EfficientTransformerRegressor SNGPRegressor
    model = convert_to_sn_my(model,
                             spec_norm_replace_list=["Linear", "Conv1d"],
                             spec_norm_bound=10)
    # 替换输出层为GP层
    GP_KWARGS = {
        'num_inducing': 1024,  # 512
        'gp_scale': 0.5,
        'gp_kernel_type': 'gaussian',  # gaussian/linear
        'gp_random_feature_type': 'rff',  # rff/orf
        'gp_bias': 0.0,
        'gp_input_normalization': True,  #
        'gp_cov_discount_factor': -1,
        'gp_cov_ridge_penalty': 1.0,  # 1e-2
        'gp_scale_random_features': False,
        'gp_use_custom_random_features': True,
        'gp_output_bias_trainable': True,  # False
        'gp_output_imagenet_initializer': True,
        'num_classes': 1  # 回归任务输出维度为1
    }

    replace_layer_with_gaussian(container=model,
                                signature="predict",
                                **GP_KWARGS)

    model.load_state_dict(torch.load('./MLP-best_test_model.pth', map_location='cuda:0'))

    model.to(device)

    eval_kwargs = {'return_random_features': False, 'return_covariance': True,
                   'update_precision_matrix': False, 'update_covariance_matrix': False}

    with torch.no_grad():
        model.eval()

        model.predict.update_covariance_matrix()
        Y_pred_test, Y_var_test = model(X_test, **eval_kwargs)
        Y_pred_test = Y_pred_test.cpu().numpy().flatten()
        Y_var_test = Y_var_test.diag().cpu().numpy()

        Y_pred_calib, Y_var_calib = model(X_calib, **eval_kwargs)
        Y_pred_calib = Y_pred_calib.cpu().numpy().flatten()
        Y_var_calib = Y_var_calib.diag().cpu().numpy()

        Y_threshold_test, Y_var_threshold = model(X_threshold, **eval_kwargs)
        all_var = Y_var_threshold.diag().cpu().numpy()

    variance_threshold = np.percentile(all_var, 80)
    ood_mask = Y_var_test > variance_threshold
    ood_mask_calib = Y_var_calib > variance_threshold

    X_test = X_test.cpu().numpy()[~ood_mask]
    X_calib = X_calib.cpu().numpy()[~ood_mask_calib]
    Y_pred_test = Y_pred_test[~ood_mask]
    Y_test = Y_test[~ood_mask]
    Y_pred_calib = Y_pred_calib[~ood_mask_calib]
    Y_calib = Y_calib[~ood_mask_calib]

    X_test = torch.FloatTensor(X_test).to(device)
    X_calib = torch.FloatTensor(X_calib).to(device)

    num_nearest_neighbors = 10
    metric = "minkowski"
    seed = 1

    list_error_test = error_list(Y_test, np.expm1(Y_pred_test))
    list_error_calib = error_list(Y_calib, np.expm1(Y_pred_calib))

    train_X = np.array(X_train)
    calib_X = np.array(X_calib.cpu().numpy())
    test_X = np.array(X_test.cpu().numpy())
    list_error_test = np.squeeze(list_error_test)
    list_error_calib = np.squeeze(list_error_calib)
    calib_dist = calc_dist(train_X, calib_X, nearest_neighbors=num_nearest_neighbors, metric=metric, random_seed=42)
    test_dist = calc_dist(train_X, test_X, nearest_neighbors=num_nearest_neighbors, metric=metric, random_seed=42)
    alpha1 = 1 - 0.68
    model_cp1 = ConformalPrediction(list_error_calib, calib_dist, alpha=alpha1)
    test_uncertainty1, qhat1 = model_cp1.predict(test_dist)

    alpha2 = 1 - 0.95
    model_cp2 = ConformalPrediction(list_error_calib, calib_dist, alpha=alpha2)
    test_uncertainty2, qhat2 = model_cp2.predict(test_dist)

    print(type(test_uncertainty1))
    shap_68 = np.mean(test_uncertainty1)
    shap_95 = np.mean(test_uncertainty2)
    pred_mean = np.mean(Y_pred_test)  # 计算预测均值

    print(f"预测均值: {pred_mean:.4f}")
    print(f"68%置信区间平均宽度: {shap_68:.4f}")
    print(f"95%置信区间平均宽度: {shap_95:.4f}")

    # 设置图形样式
    fontsize = 12  # 增大字体
    rc = {
        'figure.figsize': (10, 6),  # 统一设置图形大小
        'font.size': fontsize,
        'axes.labelsize': 16,  # 增大坐标轴标签字体
        'axes.titlesize': fontsize,
        'xtick.labelsize': 14,  # 增大x轴刻度字体
        'ytick.labelsize': 14,  # 增大y轴刻度字体
        'legend.fontsize': fontsize
    }
    matplotlib.rcParams.update(rc)
    plt.rcParams["font.family"] = "Arial"
    # mpl.rcParams.update(mpl.rcParamsDefault)
    # 0-68，1-95
    colors = ["#65B5FF", "#BDE0FF", "#00498E"]
    # quantify uncertainty
    overconfident_idx1 = np.argwhere(abs(list_error_test) > test_uncertainty1)
    overconfident_idx2 = np.argwhere(abs(list_error_test) > test_uncertainty2)

    p1 = 1 - len(overconfident_idx1) / len(test_uncertainty1)
    p2 = 1 - len(overconfident_idx2) / len(test_uncertainty2)
    p3 = 1 - p2

    # fig_filename = "./figures/panel_method_comparison/cp_feature_dpi300.png"
    fig, ax = plot_scatter_residuals_vs_heurestic_uncertainty(test_dist, list_error_test, (qhat1, qhat2), (p1, p2, p3),
                                                              xlabel="dist")
    plt.xlim([0, 6])
    plt.ylim([-1500, 1500])
    plt.tight_layout()
    save_path = f'../result/image_result/MLP-CP-conformal-uncertainty,png'
    plt.savefig(save_path, dpi=400)
    plt.show()

    from scipy.stats import norm

    expected_stds = np.linspace(norm.ppf(1e-3),
                                norm.ppf(1 - 1e-3), 15)

    expected_ps = []
    observed_ps = []
    test_uncertainties = []

    for std in expected_stds:
        expected_p = norm.cdf(std)
        expected_ps.append(expected_p)

        alpha = 1 - expected_p
        model_cp = ConformalPrediction(list_error_calib, calib_dist, alpha=alpha)
        test_uncertainty, qhat = model_cp.predict(test_dist)

        count = np.sum(np.abs(list_error_test) < test_uncertainty)
        observed_ps.append(count / len(list_error_test))
        test_uncertainties.append(np.mean(test_uncertainty))

    trapezoid = np.trapz

    area = 0

    for i in range(1, len(observed_ps) + 1):
        trap = np.abs(trapezoid(observed_ps[i - 1:i + 1], expected_ps[i - 1:i + 1])
                      - trapezoid(expected_ps[i - 1:i + 1], expected_ps[i - 1:i + 1]))
        area += trap

    import matplotlib.ticker as mtick

    fig, ax = plt.subplots(figsize=(10, 6))  # 设置图形大小
    expected_ps = np.array(expected_ps) * 100  # convert to %
    observed_ps = np.array(observed_ps) * 100  # convert to %

    ax.plot(expected_ps, observed_ps, linewidth=2)
    ax.plot(expected_ps, expected_ps, "--", alpha=0.4, linewidth=2)

    ax.fill_between(expected_ps, expected_ps, observed_ps, alpha=0.2, linewidth=0)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    # 移除等比例设置，让图形区域也保持10:6的比例
    # ax.set_aspect('equal')

    ax.set_xlabel("Expected conf. level", fontsize=20)
    ax.set_ylabel("Observed conf. level", fontsize=20)

    # 增大刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.locator_params(axis='y', nbins=6)
    ax.locator_params(axis='x', nbins=6)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.text(39, 7, "miscalc. area = {:.3f}".format(area), fontsize=16)  # 增大文本字体并添加背景

    # 调整图形区域比例，确保内部绘图区域也是10:6
    plt.tight_layout()

    save_path = f'../result/image_result/MLP-CP-Statistical-calibration-performance.png'
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()
###############################################################################################################
    # 修改置信水平范围为0.5-0.99，间隔0.01
    confidence_levels = [round(i * 0.001, 3) for i in range(1001)]

    # 评估性能
    performance_df = evaluate_conformal_performance(
        confidence_levels,
        list_error_calib,
        calib_dist,
        list_error_test,
        test_dist,
        np.expm1(Y_pred_test)
    )

    # 打印性能统计
    print("\n" + "=" * 80)
    print("Conformal Prediction Performance Metrics (0.50-0.99, step=0.01)")
    print("=" * 80)
    print(f"Total confidence levels evaluated: {len(performance_df)}")

    # 只保存一个详细的CSV文件
    performance_df.to_csv(csv_name, index=False)
    print("Detailed performance metrics saved to '../result/MLP-cp_performance_metrics_detailed.csv'")

    # 设置Arial字体，不加粗
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'normal'  # 不加粗
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'

    # 图1: 区间宽度与置信水平关系（10:6比例）
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    expected_conf = performance_df["Expected Confidence"] * 100
    interval_widths = performance_df["Avg. Interval Width"]

    # 绘制线图
    ax1.plot(expected_conf, interval_widths,
             linewidth=2, color='#2E86AB', alpha=0.8)

    # 标记关键点但不添加文本标注
    key_points = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]
    for point in key_points:
        idx = confidence_levels.index(point)
        ax1.plot(expected_conf[idx], interval_widths[idx], 'o',
                 markersize=6, color='#A23B72')

    ax1.set_xlabel('Confidence Level (%)', fontsize=18)
    ax1.set_ylabel('Average Prediction Interval Width', fontsize=18)
    ax1.set_xticks([50, 60, 70, 80, 90, 100])

    # 增大刻度字体
    ax1.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig('../result/image_result/MLP-cp_interval_width_vs_confidence.png',
                dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()

    # 图2: 置信水平偏差分布（10:6比例）
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    confidence_diff = performance_df["Confidence Difference"] * 100  # 转换为百分比

    # 使用插值方法平滑曲线
    from scipy.interpolate import CubicSpline

    # 创建插值函数
    cs = CubicSpline(expected_conf, confidence_diff)

    # 创建更密集的x值用于平滑曲线
    x_smooth = np.linspace(expected_conf.min(), expected_conf.max(), 500)
    y_smooth = cs(x_smooth)

    # 修复填充区域空白问题 - 使用单一填充函数
    # 首先绘制负值区域
    ax2.fill_between(x_smooth, y_smooth, 0,
                     where=(y_smooth <= 0),
                     color='#F25F5C', alpha=0.5, label='Under-confident')

    # 然后绘制正值区域
    ax2.fill_between(x_smooth, 0, y_smooth,
                     where=(y_smooth >= 0),
                     color='#18A558', alpha=0.5, label='Over-confident')

    # 绘制平滑的数据线
    ax2.plot(x_smooth, y_smooth, linewidth=1.5, color='black', alpha=0.7)

    ax2.set_xlabel('Confidence Level (%)', fontsize=22)
    ax2.set_ylabel('Confidence Difference (%)', fontsize=22)

    # 移除图例边框
    legend = ax2.legend(fontsize=14, frameon=False)
    ax2.set_xticks([50, 60, 70, 80, 90, 100])

    # 增大刻度字体
    ax2.tick_params(axis='both', which='major', labelsize=18)

    # 添加零线参考
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('../result/image_result/MLP-cp_confidence_difference_distribution.png',
                dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()

    print("\nSummary of Key Performance Metrics:")
    print("=" * 60)
    print(f"Average interval width range: {interval_widths.min():.2f} - {interval_widths.max():.2f}")
    print(f"Average confidence level deviation: {confidence_diff.mean():.3f}%")
    print(f"Maximum over-confidence deviation: {confidence_diff.max():.3f}%")
    print(f"Maximum under-confidence deviation: {confidence_diff.min():.3f}%")

    print("\nPerformance at Key Confidence Levels:")
    print("=" * 60)
    key_performance_df = performance_df[performance_df['Expected Confidence'].isin(key_points)].copy()
    for _, row in key_performance_df.iterrows():
        conf_level = row['Expected Confidence']
        observed_conf = row['Observed Confidence']
        coverage_rate = row['Coverage Rate']
        avg_width = row['Avg. Interval Width']
        conf_diff = row['Confidence Difference']

        print(f"{conf_level:.0%} Confidence Level:")
        print(f"  Observed Confidence: {observed_conf:.3f} | Prediction Coverage Rate: {coverage_rate:.3f}")
        print(f"  Interval Width: {avg_width:.2f} | Confidence Deviation: {conf_diff:.3f}")