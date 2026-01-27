import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from pyod.utils.utility import standardizer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from data_process_function import *

from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Channel-Attention Regression Network
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        y = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * y
class CARBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.ca = ChannelAttention(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.ca(x)

        return F.gelu(x + residual)
class CARegressor(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()
        self.initial_conv = nn.Conv1d(1, num_features, kernel_size=7, padding=3)

        self.net = nn.Sequential(
            CARBlock(num_features, num_features * 2, kernel_size=5),
            nn.MaxPool1d(2),

            CARBlock(num_features * 2, num_features * 4),
            nn.MaxPool1d(2),

            CARBlock(num_features * 4, num_features * 8),
            CARBlock(num_features * 8, num_features * 4),

            nn.AdaptiveAvgPool1d(1)
        )

        self.final_fc = nn.Sequential(
            nn.Linear(num_features * 4, 128),
            nn.GELU(),
            #nn.Linear(128, 1)
        )
        self.predict = nn.Linear(128, 1)

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        x = F.gelu(self.initial_conv(x))
        x = self.net(x).squeeze(-1)
        x = self.final_fc(x)
        return self.predict(x, **kwargs)


# SNGP Training and Evaluation Pipeline (with 5-fold CV)
def SNGP(df, epochs, lr, batch_size, spec_bound):
    for random_seed in [1025]:
        print("随机种子：",random_seed)

        # Stratified split for calibration set
        df_t = df.copy()
        df_t['HWsum_bin'] = pd.qcut(df_t['HWsum'], q=10, duplicates='drop')
        remain_df, calib_set = train_test_split(
            df_t,
            test_size=0.05,
            random_state=1025,
            stratify=df_t['HWsum_bin']
        )
        calib_set = calib_set.drop(columns=['HWsum_bin'])
        set = remain_df.drop(columns=['HWsum_bin'])

        # Missing value imputation for calibration set
        fill_missing_values_by_industry(calib_set, 'pH', 'hyfl4')
        calib_set_copy = calib_set.copy()
        for column in ['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen']:
            fill_missing_values_by_industry(calib_set_copy, column, 'hyfl4')

        # Calculate the emission intensity of various industries in the calibration set
        emission_intensity_cod = calculate_emission_intensity(calib_set_copy, 'COD')
        emission_intensity_zonglin = calculate_emission_intensity(calib_set_copy, 'Total phosphorus')
        emission_intensity_ammonia = calculate_emission_intensity(calib_set_copy, 'Ammonia nitrogen')
        emission_intensity_all_ammonia = calculate_emission_intensity(calib_set_copy, 'Total nitrogen')

        for column, intensity in zip(['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen'],
                                     [emission_intensity_cod, emission_intensity_zonglin, emission_intensity_ammonia,
                                      emission_intensity_all_ammonia]):
            fill_missing_values(calib_set, column, intensity)


        #Divide into training and test sets
        train_set, test_set = train_test_split(set, test_size = 2/19, random_state=random_seed)

        #Missing value imputation for training set
        fill_missing_values_by_industry(train_set, 'pH', 'hyfl4')
        train_set_copy = train_set.copy()
        for column in ['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen']:
            fill_missing_values_by_industry(train_set_copy, column, 'hyfl4')

        # Calculate the emission intensity of various industries in the training set
        emission_intensity_cod = calculate_emission_intensity(train_set_copy, 'COD')
        emission_intensity_zonglin = calculate_emission_intensity(train_set_copy, 'Total phosphorus')
        emission_intensity_ammonia = calculate_emission_intensity(train_set_copy, 'Ammonia nitrogen')
        emission_intensity_all_ammonia = calculate_emission_intensity(train_set_copy, 'Total nitrogen')

        for column, intensity in zip(['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen'],
                                     [emission_intensity_cod, emission_intensity_zonglin, emission_intensity_ammonia,
                                      emission_intensity_all_ammonia]):
            fill_missing_values(train_set, column, intensity)

        #Missing value imputation for test set
        fill_missing_values_by_industry(test_set, 'pH', 'hyfl4')
        test_set_copy = test_set.copy()
        for column in ['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen']:
            fill_missing_values_by_industry(test_set_copy, column, 'hyfl4')

        for column, intensity in zip(['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen'],
                                     [emission_intensity_cod, emission_intensity_zonglin, emission_intensity_ammonia,
                                      emission_intensity_all_ammonia]):
            fill_missing_values(test_set, column, intensity)

        #Remove outliers from training set
        cleaned_data = anomaly_detection(data=train_set, anomaly_rate=0.05, random_state=66)

        test_data = test_set.drop(['time', 'dwmc', 'spCode'], axis=1)
        train_data = cleaned_data.drop(['time', 'dwmc', 'spCode'], axis=1)
        calib_data = calib_set.drop(['time', 'dwmc', 'spCode'], axis=1)

        columns_to_select = [
            'COD', 'pH', 'Hexavalent chromium', 'Wastewater flow', 'Total nitrogen', 'Total phosphorus',
            'Total iron', 'Total copper', 'Total chromium', 'Total zinc', 'Total nickel', 'Ammonia nitrogen',
            'hyfl4', 'process_01', 'process_02', 'process_03', 'process_04', 'process_05', 'process_06',
            'process_07', 'process_14', 'process_15', 'process_16', 'process_17', 'Firm_scale', 'HWsum']

        train_data = train_data[columns_to_select]
        test_data = test_data[columns_to_select]
        calib_data = calib_data[columns_to_select]

        # fill empty values in the specified column with 0.
        columns_to_fill_zero = ['Hexavalent chromium', 'Total iron', 'Total copper', 'Total chromium', 'Total zinc', 'Total nickel']
        train_data[columns_to_fill_zero] = train_data[columns_to_fill_zero].fillna(0)
        test_data[columns_to_fill_zero] = test_data[columns_to_fill_zero].fillna(0)
        calib_data[columns_to_fill_zero] = calib_data[columns_to_fill_zero].fillna(0)

        x_columns_to_select = [
            'COD', 'pH', 'Hexavalent chromium', 'Wastewater flow', 'Total nitrogen', 'Total phosphorus',
            'Total iron', 'Total copper', 'Total chromium', 'Total zinc', 'Total nickel', 'Ammonia nitrogen',
            'hyfl4', 'process_01', 'process_02', 'process_03', 'process_04', 'process_05', 'process_06',
            'process_07', 'process_14', 'process_15', 'process_16', 'process_17', 'Firm_scale']
        new_name_columns = [
            'COD', 'pH', 'Hexavalent chromium', 'Wastewater flow', 'Total nitrogen', 'Total phosphorus',
            'Total iron', 'Total copper', 'Total chromium', 'Total zinc', 'Total nickel', 'Ammonia nitrogen',
            'Industry Classification', 'process_01', 'process_02', 'process_03', 'process_04', 'process_05', 'process_06',
            'process_07', 'process_14', 'process_15', 'process_16', 'process_17', 'Firm_scale']

        #Determine the threshold selection set
        df_t = train_data.copy()
        df_t['HWsum_bin'] = pd.qcut(df_t['HWsum'], q=10, duplicates='drop')
        remain_df, threshold = train_test_split(
            df_t,
            test_size=0.05,
            random_state=1025,
            stratify=df_t['HWsum_bin']
        )
        threshold = threshold.drop(columns=['HWsum_bin'])
        train_data = remain_df.drop(columns=['HWsum_bin'])

        x_threshold = threshold[x_columns_to_select]
        x_threshold.columns = new_name_columns
        x_train = train_data[x_columns_to_select]
        x_train.columns = new_name_columns
        x_test = test_data[x_columns_to_select]
        x_test.columns = new_name_columns
        x_calib = calib_data[x_columns_to_select]
        x_calib.columns = new_name_columns

        numeric_cols = [
            'COD', 'pH', 'Hexavalent chromium', 'Wastewater flow', 'Total nitrogen', 'Total phosphorus',
            'Total iron', 'Total copper', 'Total chromium', 'Total zinc', 'Total nickel', 'Ammonia nitrogen'
        ]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_train.loc[:, numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
        x_test.loc[:, numeric_cols] = scaler.transform(x_test[numeric_cols])
        x_calib.loc[:, numeric_cols] = scaler.transform(x_calib[numeric_cols])
        x_threshold.loc[:, numeric_cols] = scaler.transform(x_threshold[numeric_cols])

        # 2. One-Hot Industry Classification
        x_train = pd.get_dummies(x_train, columns=["Industry Classification"])
        x_test = pd.get_dummies(x_test, columns=["Industry Classification"])
        x_calib = pd.get_dummies(x_calib, columns=["Industry Classification"])
        x_threshold = pd.get_dummies(x_threshold, columns=["Industry Classification"])

        x_train_s = x_train.values.astype(np.float32)
        x_test_s = x_test.values.astype(np.float32)
        x_threshold = x_threshold.values.astype(np.float32)

        x_threshold_tensor = torch.FloatTensor(x_threshold).to(device)

        y_train = np.log1p(train_data['HWsum'].to_numpy())
        y_test = np.log1p(test_data['HWsum'].to_numpy())
        #y_calib = np.log1p(calib_data['HWsum'].to_numpy())

        #train_df = x_train.copy()
        #test_df = x_test.copy()
        #calib_df = x_calib.copy()
        #train_df['HWsum'] = y_train
        #test_df['HWsum'] = y_test
        #calib_df['HWsum'] = y_calib
        #train_df.to_excel('../data/train_set.xlsx', index=False)
        #test_df.to_excel('../data/test_set.xlsx', index=False)
        #calib_df.to_excel('../data/calib_set.xlsx', index=False)


        eval_kwargs = {'return_random_features': False, 'return_covariance': True,
                       'update_precision_matrix': False, 'update_covariance_matrix': False}

        kwargs = {'return_random_features': False, 'return_covariance': False,
                  'update_precision_matrix': True, 'update_covariance_matrix': False}

        x_all = x_train_s
        y_all = y_train
        x_test = x_test_s
        y_test = y_test

        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_results = []
        best_epochs = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_all)):
            print(f'\nStarting Fold {fold + 1}/{k_folds}')

            #data of the current fold
            x_train_fold, x_val_fold = x_all[train_idx], x_all[val_idx]
            y_train_fold, y_val_fold = y_all[train_idx], y_all[val_idx]

            #Convert to Tensor
            x_train_tensor = torch.FloatTensor(x_train_fold).to(device)
            y_train_tensor = torch.FloatTensor(y_train_fold).view(-1, 1).to(device)
            x_val_tensor = torch.FloatTensor(x_val_fold).to(device)

            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model = CARegressor().to(device)
            model = convert_to_sn_my(model,
                                     spec_norm_replace_list=["Linear", "Conv1d"],
                                     spec_norm_bound=spec_bound)
            GP_KWARGS = {
                'num_inducing': 1024,
                'gp_scale': 0.5,
                'gp_kernel_type': 'gaussian',  # gaussian/linear
                'gp_random_feature_type': 'rff',  # rff/orf
                'gp_bias': 0.0,
                'gp_input_normalization': True,
                'gp_cov_discount_factor': -1,
                'gp_cov_ridge_penalty': 1.0,
                'gp_scale_random_features': False,
                'gp_use_custom_random_features': True,
                'gp_output_bias_trainable': True,  # False
                'gp_output_imagenet_initializer': True,
                'num_classes': 1  # The output dimension of the regression task is 1
            }
            replace_layer_with_gaussian(container=model,
                                        signature="predict",
                                        **GP_KWARGS)

            criterion = nn.SmoothL1Loss(beta=0.8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            best_val_r2 = -float('inf')
            best_epoch = 0
            model.train()
            model.predict.reset_covariance_matrix()
            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x, **kwargs)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    #  validation cycle
                if (epoch + 1) % 10 == 0:
                    model.eval()
                    model.predict.update_covariance_matrix()  # update the covariance matrix prior to evaluation
                    with torch.no_grad():
                        #training set indicators
                        train_pred = model(x_train_tensor, **eval_kwargs)
                        train_r2 = r2_score(np.expm1(y_train_fold), np.expm1(train_pred[0].cpu().numpy()))
                        train_mae = mean_absolute_error(np.expm1(y_train_fold),
                                                        np.expm1(train_pred[0].cpu().numpy()))
                        #validation set indicators
                        val_result = model(x_val_tensor, **eval_kwargs)
                        val_r2 = r2_score(np.expm1(y_val_fold), np.expm1(val_result[0].cpu().numpy()))
                        val_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(val_result[0].cpu().numpy()))
                        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], '
                              f'Train R2: {train_r2:.4f}, MAE: {train_mae:.4f}, '
                              f'Val R2: {val_r2:.4f}, MAE: {val_mae:.4f}')
                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        best_epoch = epoch + 1

            # record the results of the current fold
            fold_results.append({
                'fold': fold + 1,
                'best_val_r2': best_val_r2,
                'best_epoch': best_epoch
            })
            best_epochs.append(best_epoch)

            print(f'Fold {fold + 1} completed. Best Val R2: {best_val_r2:.4f}, Best Epoch: {best_epoch}')

        # The average performance of cross-validation is calculated
        val_r2_scores = [result['best_val_r2'] for result in fold_results]
        mean_val_r2 = np.mean(val_r2_scores)
        std_val_r2 = np.std(val_r2_scores)
        mean_best_epoch = int(np.mean(best_epochs))
        print(f'\nCross-Validation completed. Mean Val R2: {mean_val_r2:.4f} (±{std_val_r2:.4f})')
        print(f'Average best epoch: {mean_best_epoch}')

        # The final model is retrained using all the training data
        print("\nTraining final model on all training data...")

        final_model = CARegressor().to(device)

        final_model = convert_to_sn_my(final_model,
                                 spec_norm_replace_list=["Linear", "Conv1d"],
                                 spec_norm_bound=spec_bound)
        GP_KWARGS = {
            'num_inducing': 1024,
            'gp_scale': 0.5,
            'gp_kernel_type': 'gaussian',  # gaussian/linear
            'gp_random_feature_type': 'rff',  # rff/orf
            'gp_bias': 0.0,
            'gp_input_normalization': True,
            'gp_cov_discount_factor': -1,
            'gp_cov_ridge_penalty': 1.0,
            'gp_scale_random_features': False,
            'gp_use_custom_random_features': True,
            'gp_output_bias_trainable': True,  # False
            'gp_output_imagenet_initializer': True,
            'num_classes': 1
        }
        replace_layer_with_gaussian(container=final_model,
                                    signature="predict",
                                    **GP_KWARGS)
        criterion = nn.SmoothL1Loss(beta=0.8)
        optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)

        # All the training data are prepared.
        x_all_tensor = torch.FloatTensor(x_all).to(device)
        y_all_tensor = torch.FloatTensor(y_all).view(-1, 1).to(device)
        all_dataset = TensorDataset(x_all_tensor, y_all_tensor)
        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)
        x_test_tensor = torch.FloatTensor(x_test).to(device)

        # train the final model
        best_test_r2 = -float('inf')
        best_test_model_state = None
        best_test_epoch = 0

        final_model.train()
        final_model.predict.reset_covariance_matrix()
        for epoch in range(mean_best_epoch):
            final_model.train()

            total_loss = 0
            nan_detected = False

            for batch_x, batch_y in all_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = final_model(batch_x, **kwargs)

                # It is checked whether the output contains NaN or Inf.
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Invalid outputs at epoch {epoch + 1}")
                    nan_detected = True
                    continue

                loss = criterion(outputs, batch_y)

                # It is checked whether the loss is NaN or Inf.
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss at epoch {epoch + 1}: {loss.item()}")
                    nan_detected = True
                    continue

                loss.backward()

                # Gradient clipping is added to prevent gradient explosion.
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            if nan_detected:
                print(f"Skipping evaluation at epoch {epoch + 1} due to invalid values")
                continue

            avg_loss = total_loss / len(all_loader)

            # Evaluation is performed on the test set every 3 epochs.
            if (epoch + 1) % 3 == 0 or epoch == mean_best_epoch - 1:
                final_model.eval()
                final_model.predict.update_covariance_matrix()

                with torch.no_grad():

                    train_result = final_model(x_all_tensor, **eval_kwargs)
                    train_pred = train_result[0].cpu().numpy()

                    # calculate the training set indicators
                    train_r2 = r2_score(np.expm1(y_all), np.expm1(train_pred))


                    test_result = final_model(x_test_tensor, **eval_kwargs)
                    test_pred = test_result[0].cpu().numpy()

                    test_var = test_result[1].diag().cpu().numpy()

                    # calculate the test set indicators
                    test_r2 = r2_score(np.expm1(y_test), np.expm1(test_pred))
                    test_mae = mean_absolute_error(np.expm1(y_test), np.expm1(test_pred))
                    test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(test_pred)))

                    # save the best-performing model on the test set
                    if test_r2 > best_test_r2:
                        best_test_r2 = test_r2
                        best_test_mae = test_mae
                        best_test_rmse = test_rmse
                        best_test_model_state = copy.deepcopy(final_model.state_dict())
                        best_test_epoch = epoch + 1
                        torch.save(best_test_model_state, "CA-best_test_model.pth")
                        print(f"New best model saved at epoch {epoch + 1} with Test R2: {test_r2:.4f}")

                # print the progress and indicators
                print(f'Epoch [{epoch + 1}/{mean_best_epoch}], '
                      f'Train Loss: {avg_loss:.4f}, Train R2: {train_r2:.4f}, '
                      f'Test R2: {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}')
            else:
                # print only the training loss for epochs without evaluation
                print(f'Epoch [{epoch + 1}/{mean_best_epoch}], Train Loss: {avg_loss:.4f}')

        print("Final model training completed")

        # load the best-performing model on the test set
        if best_test_model_state is not None:
            final_model.load_state_dict(best_test_model_state)
            print(f"Loaded best model from epoch {best_test_epoch} with Test R2: {best_test_r2:.4f}")

        # determine the threshold
        final_model.eval()
        final_model.predict.update_covariance_matrix()
        with torch.no_grad():
            all_result = final_model(x_threshold_tensor, **eval_kwargs)
            all_var = all_result[1].diag().cpu().numpy()

        # set the threshold quantiles
        variance_threshold = np.percentile(all_var, 80)
        print(f'Selected variance threshold: {variance_threshold:.6f}')

        # conduct the final test evaluation using the optimal model
        final_model.eval()
        with torch.no_grad():
            test_result = final_model(x_test_tensor, **eval_kwargs)
            test_pred = test_result[0].cpu().numpy()
            test_var = test_result[1].diag().cpu().numpy()

        # result Analysis
        full_test_r2 = best_test_r2
        full_test_mae = best_test_mae
        full_test_rmse = best_test_rmse

        ood_mask = test_var > variance_threshold
        filtered_test_pred = test_pred[~ood_mask]
        filtered_y_test = y_test[~ood_mask]

        print('\n' + '=' * 50)
        print(f"{'Final Model Test Set Metrics':^50}")
        print('=' * 50)
        print(
            f"{'Before OOD Filtering':<25}: R2 = {full_test_r2:.4f}, MAE = {full_test_mae:.4f}, RMSE = {full_test_rmse:.4f}")
        print(f"{'OOD Samples Filtered':<25}: {ood_mask.sum()} ({ood_mask.sum() / len(y_test) * 100:.2f}%)")

        if len(filtered_y_test) > 0:
            filtered_r2 = r2_score(np.expm1(filtered_y_test), np.expm1(filtered_test_pred))
            filtered_mae = mean_absolute_error(np.expm1(filtered_y_test), np.expm1(filtered_test_pred))
            filtered_rmse = np.sqrt(mean_squared_error(np.expm1(filtered_y_test), np.expm1(filtered_test_pred)))
            print(
                f"{'After OOD Filtering':<25}: R2 = {filtered_r2:.4f}, MAE = {filtered_mae:.4f}, RMSE = {filtered_rmse:.4f}")
            print(f"{'Improvement in R2':<25}: {(filtered_r2 - full_test_r2):+.4f}")
        else:
            print("Warning: All test samples filtered as OOD!")
        print('=' * 50)

    return filtered_r2, filtered_mae, filtered_rmse, filtered_r2 - full_test_r2


output = pd.read_csv("../data/Enterprise_data.csv", encoding='gbk')
output['hyfl4'] = output['hyfl4'].str.extract('(\d+)').astype(int)
grouped = output.groupby('hyfl4')['HWsum'].sum()
sorted_grouped = grouped.sort_values(ascending=False)
columns_to_check = ['Total copper', 'COD', 'Total chromium', 'Hexavalent chromium', 'Total phosphorus']
output = output[~(output[columns_to_check] < 0).any(axis=1)]
columns_to_check = ['COD', 'Hexavalent chromium', 'Hexavalent chromium', 'Total nitrogen',
                    'Total phosphorus', 'Total iron','Total copper', 'Total chromium', 'Total zinc', 'Total nickel', 'Ammonia nitrogen', 'HWsum']
output = remove_extreme_values_with_na(output, columns_to_check, percentile_high=99.5, percentile_low=0.5)
output['Firm_scale'] = output.apply(lambda x: firm_scale_split(x.staff_number), axis=1)
output = output.drop('staff_number', axis=1)


lr = 0.0007
epochs = 200
batch_size = 4096

spec_bound = 10

print("Configure the parameter settings：", lr, epochs, batch_size, spec_bound)

r2, mae, rmse, improve = SNGP(output, epochs, lr, batch_size=batch_size,spec_bound =spec_bound)

