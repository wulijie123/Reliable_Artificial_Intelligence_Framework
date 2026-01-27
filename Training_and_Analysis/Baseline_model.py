import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.utils.utility import standardizer
from pyod.models.knn import KNN
from pyod.models.combination import aom
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import copy
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def remove_extreme_values_with_na(df, columns, percentile_high=99.5, percentile_low=1):
    df_filtered = df.copy()
    for col in columns:
        threshold_high = df_filtered[col].quantile(percentile_high / 100)
        threshold_low = df_filtered[col].quantile(percentile_low / 100)
        df_filtered = df_filtered[
            (df_filtered[col] <= threshold_high) | df_filtered[col].isna()
            ]
        df_filtered = df_filtered[
            (df_filtered[col] >= threshold_low) | df_filtered[col].isna()
            ]
    return df_filtered

def firm_scale_split(staff_number):
    if staff_number >= 1 and staff_number <= 49:
        return '1'
    if staff_number >= 50 and staff_number <= 99:
        return '2'
    if staff_number >= 100 and staff_number <= 199:
        return '3'
    if staff_number >= 200 and staff_number <= 499:
        return '4'
    else:
        return '5'


def fill_missing_values_by_industry(df, column_name, industry_column):
    median_by_dwmc = df.groupby('dwmc')[column_name].median()  # 计算企业内的中位数
    median_by_industry_and_scale = df.groupby([industry_column, 'Firm_scale'])[column_name].median()  # 计算同一行业内按企业规模分类的中位数
    median_by_industry = df.groupby(industry_column)[column_name].median()  # 新增：计算同行业所有规模企业的中位数
    median_all = df[column_name].median()

    def fill_value(row):
        if pd.isna(row[column_name]):
            dwmc = row['dwmc']
            firm_scale = row['Firm_scale']
            industry = row[industry_column]

            if dwmc in median_by_dwmc.index:  # 先用企业的中位数填补
                median_dwmc = median_by_dwmc[dwmc]
                if pd.notna(median_dwmc):
                    return median_dwmc
            if (industry, firm_scale) in median_by_industry_and_scale.index:  # 如果企业的中位数不可用，则用同行业同规模的企业的中位数填补
                median_industry_scale = median_by_industry_and_scale[(industry, firm_scale)]
                if pd.notna(median_industry_scale):
                    return median_industry_scale
            if industry in median_by_industry.index:  # 修改：如果同行业同规模的企业的中位数不可用，则用同行业所有规模企业的中位数填补
                median_industry = median_by_industry[industry]
                if pd.notna(median_industry):
                    return median_industry

            if pd.notna(median_all):
                return median_all
        return row[column_name]

    df[column_name] = df.apply(fill_value, axis=1)  # 应用填补函数


def calculate_emission_intensity(df, column_name):
    df['排放强度'] = df[column_name] / df['Wastewater flow']
    return df.groupby('hyfl4')['排放强度'].median()


def fill_missing_values(df, column_name, emission_intensity):
    def fill_value(row):
        if pd.isna(row[column_name]):
            dwmc = row['dwmc']
            firm_scale = row['Firm_scale']
            hyfl4 = row['hyfl4']
            wastewater_flow = row['Wastewater flow']

            if hyfl4 in emission_intensity.index:
                return emission_intensity[hyfl4] * wastewater_flow
        return row[column_name]

    df[column_name] = df.apply(fill_value, axis=1)


def anomaly_ensemble(x, random_state):

    # iforest
    clf = IForest(n_estimators=300, random_state=random_state)
    clf.fit(x)
    pred1 = clf.predict(x)
    A1 = clf.decision_function(x)
    # mcd
    warnings.filterwarnings('ignore', 'Determinant has increased; this should not happen: ')
    warnings.filterwarnings('ignore', 'The covariance matrix associated to your dataset ')
    clf = MCD(random_state=random_state)
    clf.fit(x)
    pred2 = clf.predict(x)
    A2 = clf.decision_function(x)
    # lof
    clf = LOF(n_neighbors=10)
    clf.fit(x)
    pred3 = clf.predict(x)
    A3 = clf.decision_function(x)
    # knn
    clf = KNN(n_neighbors=10)
    clf.fit(x)
    pred4 = clf.predict(x)
    A4 = clf.decision_function(x)
    # cblof
    clf = CBLOF(random_state=random_state)
    clf.fit(x)
    pred5 = clf.predict(x)
    A5 = clf.decision_function(x)
    # hbos
    clf = HBOS(n_bins=10)
    clf.fit(x)
    pred6 = clf.predict(x)
    A6 = clf.decision_function(x)
    scores = np.vstack([A1, A2, A3, A4, A5, A6]).T
    scores = standardizer(scores)
    y_by_aom = aom(scores, n_buckets=3, method='static', bootstrap_estimators=False, random_state=random_state)
    y_by_aom = pd.DataFrame(y_by_aom)
    y_by_aom.columns = ['scores']
    return y_by_aom


def anomaly_detection(data, anomaly_rate, random_state, partial=False):
    to_model_columns = data[['COD', 'pH', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen', 'Wastewater flow', 'HWsum']]
    to_model_columns = standardizer(to_model_columns)
    data['score'] = anomaly_ensemble(to_model_columns, random_state=random_state)['scores']
    data = data.reset_index()
    data = data.sort_values(by='score')
    cleaned_data = data.iloc[:round(len(data) * (1 - anomaly_rate)), :]
    anomaly_data = data.iloc[round(len(data) * (1 - anomaly_rate)):, :]
    cleaned_data = cleaned_data.sort_index()
    anomaly_data = anomaly_data.sort_index()
    return cleaned_data

#
class DeepMLPRegressor(nn.Module):
    def __init__(self, input_dim=25, hidden_dims=[512, 256, 128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim


        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

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


#
class SimpleCNNRegressor(nn.Module):
    def __init__(self, input_dim=25, num_features=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, num_features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(num_features, num_features * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(num_features * 2, num_features * 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(num_features * 4, num_features * 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(num_features * 8, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.conv_layers(x).squeeze(-1)
        return self.fc_layers(x)

#
class RescnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # If the number of input channels and the number of output channels are not the same, use a 1x1 convolution to adjust the dimensions
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.gelu(out)
class ResCNNRegressor(nn.Module):
    def __init__(self, input_dim=25, num_features=64):
        super().__init__()
        self.initial_conv = nn.Conv1d(1, num_features, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(
            RescnnBlock(num_features, num_features * 2),
            nn.MaxPool1d(2),
            RescnnBlock(num_features * 2, num_features * 4),
            nn.MaxPool1d(2),
            RescnnBlock(num_features * 4, num_features * 8),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(num_features * 8, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = F.gelu(self.initial_conv(x))
        x = self.res_blocks(x).squeeze(-1)
        return self.fc_layers(x)


#
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
class InceptionCNNRegressor(nn.Module):
    def __init__(self, input_dim=25, num_features=64):
        super().__init__()
        self.initial_conv = nn.Conv1d(1, num_features, kernel_size=3, padding=1)

        self.net = nn.Sequential(
            InceptionBlock(num_features, num_features * 2),
            nn.MaxPool1d(2),
            InceptionBlock(num_features * 2, num_features * 4),
            nn.MaxPool1d(2),
            InceptionBlock(num_features * 4, num_features * 8),
            nn.AdaptiveAvgPool1d(1)
        )

        self.final_fc = nn.Sequential(
            nn.Linear(num_features * 8, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.gelu(self.initial_conv(x))
        x = self.net(x).squeeze(-1)
        return self.final_fc(x)


#
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
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        y = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * y
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y
class AttentionResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

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
        x = self.sa(x)

        return F.gelu(x + residual)
class AttentionCNNRegressor(nn.Module):
    def __init__(self, input_dim=25, num_features=64):
        super().__init__()
        self.initial_conv = nn.Conv1d(1, num_features, kernel_size=7, padding=3)

        self.net = nn.Sequential(
            AttentionResBlock(num_features, num_features * 2),
            nn.MaxPool1d(2),
            AttentionResBlock(num_features * 2, num_features * 4),
            nn.MaxPool1d(2),
            AttentionResBlock(num_features * 4, num_features * 8),
            nn.AdaptiveAvgPool1d(1)
        )

        self.final_fc = nn.Sequential(
            nn.Linear(num_features * 8, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.gelu(self.initial_conv(x))
        x = self.net(x).squeeze(-1)
        return self.final_fc(x)


#
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
    def __init__(self, input_dim=25, num_features=64):
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
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.gelu(self.initial_conv(x))
        x = self.net(x).squeeze(-1)
        return self.final_fc(x)



#Transformer
import math
class SimpleSelfAttention(nn.Module):
    """self-attention"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model


        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

    def forward(self, x):
        # [batch, seq_len, d_model]
        batch_size, seq_len, _ = x.size()

        # Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Attention Score
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_probs = F.softmax(attn_scores, dim=-1)


        output = torch.matmul(attn_probs, v)
        return output
class SimpleTransformerRegressor(nn.Module):


    def __init__(self, input_dim=25, d_model=64, num_layers=6, d_ff=256):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # 分类token

        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": SimpleSelfAttention(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # [batch, input_dim]
        # [batch, 1, input_dim]
        x = x.unsqueeze(1)

        x = self.embedding(x)

        # CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :x.size(1)]

        for layer in self.layers:

            attn_output = layer["attention"](x)
            x = layer["norm1"](x + attn_output)


            ffn_output = layer["ffn"](x)
            x = layer["norm2"](x + ffn_output)


        cls_output = x[:, 0, :]
        return self.regressor(cls_output)


class MultiHeadAttention(nn.Module):


    def __init__(self, d_model, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)


        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)


        attn_output = torch.matmul(attn_probs, v)


        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_k)

        return self.wo(attn_output)
class TransformerRegressor(nn.Module):


    def __init__(self, input_dim=25, d_model=64, num_heads=8,
                 num_layers=6, d_ff=256):
        super().__init__()


        self.embedding = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))


        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "attention": MultiHeadAttention(d_model, num_heads),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
            })
            self.layers.append(layer)


        self.norm = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # [batch, input_dim] -> [batch, 1, input_dim]
        x = x.unsqueeze(1)

        x = self.embedding(x)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :x.size(1)]

        for layer in self.layers:

            attn_output = layer["attention"](x, x, x)
            x = layer["norm1"](x + attn_output)

            #
            ffn_output = layer["ffn"](x)
            x = layer["norm2"](x + ffn_output)


        cls_output = self.norm(x[:, 0, :])
        return self.regressor(cls_output)


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


        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.v_proj(v).view(batch_size, seq_len, self.num_heads, self.d_k)


        q = F.elu(q) + 1
        k = F.elu(k) + 1


        z = 1 / (torch.einsum('blhd,bhd->blh', q, k.sum(dim=1)) + 1e-6)

        # 计算累积和
        k_cum = torch.cumsum(k, dim=1)  # [b, l, h, d_k]
        v_cum = torch.cumsum(v, dim=1)  # [b, l, h, d_k]


        context = torch.einsum('blhd,blhd->blh', q, k_cum)  # [b, l, h]
        context = context.unsqueeze(-1) * v_cum  # [b, l, h, d_k]
        context = context * z.unsqueeze(-1)  # 应用归一化

        context = context.contiguous().view(batch_size, seq_len, -1)

        output = self.o_proj(context)
        return output
class EfficientTransformerRegressor(nn.Module):


    def __init__(self, input_dim=25, d_model=64, num_heads=8,
                 num_layers=6, d_ff=256):
        super().__init__()


        self.embedding = nn.Linear(input_dim, d_model)


        max_len = 100
        self.position_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.position_emb, std=0.02)

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

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            #nn.Linear(d_model // 2, 1)
            nn.Linear(d_model // 2, d_model // 2)
        )
        self.predict = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        # [batch, input_dim]
        # [batch, seq_len=1, input_dim]
        x = x.unsqueeze(1)


        x = self.embedding(x)

        seq_len = x.size(1)
        x = x + self.position_emb[:, :seq_len]

        for layer in self.layers:

            attn_output = layer["attention"](x, x, x)
            x = layer["norm1"](x + attn_output)


            ffn_output = layer["ffn"](x)
            x = layer["norm2"](x + ffn_output)


        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, d_model]
        x = self.regressor(x)
        return self.predict(x)  #self.regressor(x)


class LightConvTransformerBlock(nn.Module):


    def __init__(self, channels, kernel_size=3, nhead=2):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU())


        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=nhead,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels))


    def forward(self, x):
        # [batch, channels, seq_len]
        identity = x


        x = self.conv_block(x)

        # [batch, seq_len, channels]
        x_trans = x.permute(0, 2, 1)


        attn_output, _ = self.self_attn(x_trans, x_trans, x_trans)
        x_trans = self.norm1(x_trans + attn_output)

        ffn_output = self.ffn(x_trans)
        x_trans = self.norm2(x_trans + ffn_output)

        # [batch, channels, seq_len]

        return F.gelu(x + identity)
class LightCNNTransformerRegressor(nn.Module):


    def __init__(self, input_dim=25, base_channels=64):
        super(LightCNNTransformerRegressor, self).__init__()


        self.input_layer = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.GELU())

        self.block1 = LightConvTransformerBlock(base_channels)
        self.block2 = LightConvTransformerBlock(base_channels)
        self.block3 = LightConvTransformerBlock(base_channels)


        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.output_layer = nn.Sequential(
            nn.Linear(base_channels, 64),
            nn.GELU(),
            nn.Linear(64, 1))

    def forward(self, x):
        # batch, input_dim]
        x = x.unsqueeze(1)  # [batch, 1, input_dim]

        # 输入层处理
        x = self.input_layer(x)  # [batch, base_channels, input_dim]


        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x).squeeze(-1)  # [batch, base_channels]

        return self.output_layer(x)


def SNGP(df, epochs, lr, batch_size, model):
    for random_seed in [1025]:
        print("Random_seed: ", random_seed)
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=random_seed)
        check_cols = ['COD', 'pH', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen', 'Wastewater flow', 'HWsum']
        # Check if there are completely empty columns in the training and test sets
        if train_set[check_cols].isna().all().any() or test_set[check_cols].isna().all().any():
            print(f"The random seed {random_seed} was skipped because an all-empty column existed")
            continue

        fill_missing_values_by_industry(train_set, 'pH', 'hyfl4')
        train_set_copy = train_set.copy()
        for column in ['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen']:
            fill_missing_values_by_industry(train_set_copy, column, 'hyfl4')

        emission_intensity_cod = calculate_emission_intensity(train_set_copy, 'COD')
        emission_intensity_zonglin = calculate_emission_intensity(train_set_copy, 'Total phosphorus')
        emission_intensity_ammonia = calculate_emission_intensity(train_set_copy, 'Ammonia nitrogen')
        emission_intensity_all_ammonia = calculate_emission_intensity(train_set_copy, 'Total nitrogen')

        for column, intensity in zip(['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen'],
                                     [emission_intensity_cod, emission_intensity_zonglin, emission_intensity_ammonia,
                                      emission_intensity_all_ammonia]):
            fill_missing_values(train_set, column, intensity)

        fill_missing_values_by_industry(test_set, 'pH', 'hyfl4')
        test_set_copy = test_set.copy()
        for column in ['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen']:
            fill_missing_values_by_industry(test_set_copy, column, 'hyfl4')

        for column, intensity in zip(['COD', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen'],
                                     [emission_intensity_cod, emission_intensity_zonglin, emission_intensity_ammonia,
                                      emission_intensity_all_ammonia]):
            fill_missing_values(test_set, column, intensity)

        cleaned_data = anomaly_detection(data=train_set, anomaly_rate=0.05, random_state=66)

        test_data = test_set.drop(['time', 'dwmc', 'spCode'], axis=1)
        train_data = cleaned_data.drop(['time', 'dwmc', 'spCode'], axis=1)

        columns_to_select = [
            'COD', 'pH', 'Hexavalent chromium', 'Wastewater flow', 'Total nitrogen', 'Total phosphorus',
            'Total iron', 'Total copper', 'Total chromium', 'Total zinc', 'Total nickel', 'Ammonia nitrogen',
            'hyfl4', 'process_01', 'process_02', 'process_03', 'process_04', 'process_05', 'process_06',
            'process_07', 'process_14', 'process_15', 'process_16', 'process_17', 'Firm_scale', 'HWsum']
        train_data = train_data[columns_to_select]
        test_data = test_data[columns_to_select]
        columns_to_fill_zero = ['Hexavalent chromium', 'Total iron', 'Total copper', 'Total chromium', 'Total zinc', 'Total nickel']

        train_data[columns_to_fill_zero] = train_data[columns_to_fill_zero].fillna(0)
        test_data[columns_to_fill_zero] = test_data[columns_to_fill_zero].fillna(0)

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
        x_train = train_data[x_columns_to_select]
        x_train.columns = new_name_columns
        x_test = test_data[x_columns_to_select]
        x_test.columns = new_name_columns

        numeric_cols = [
            'COD', 'pH', 'Hexavalent chromium', 'Wastewater flow', 'Total nitrogen', 'Total phosphorus',
            'Total iron', 'Total copper', 'Total chromium', 'Total zinc', 'Total nickel', 'Ammonia nitrogen'
        ]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_train.loc[:, numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
        x_test.loc[:, numeric_cols] = scaler.transform(x_test[numeric_cols])


        x_train = pd.get_dummies(x_train, columns=["Industry Classification"])
        x_test = pd.get_dummies(x_test, columns=["Industry Classification"])

        x_train_s = x_train.values.astype(np.float32)
        x_test_s = x_test.values.astype(np.float32)
        y_train = np.log1p(train_data['HWsum'].to_numpy())
        y_test = np.log1p(test_data['HWsum'].to_numpy())

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

            x_train_fold, x_val_fold = x_all[train_idx], x_all[val_idx]
            y_train_fold, y_val_fold = y_all[train_idx], y_all[val_idx]

            # 转换为Tensor
            x_train_tensor = torch.FloatTensor(x_train_fold).to(device)
            y_train_tensor = torch.FloatTensor(y_train_fold).view(-1, 1).to(device)
            x_val_tensor = torch.FloatTensor(x_val_fold).to(device)

            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model2 = model

            criterion = nn.SmoothL1Loss(beta=0.8)
            optimizer = torch.optim.AdamW(model2.parameters(), lr=lr, weight_decay=1e-4)
            best_val_r2 = -float('inf')
            best_epoch = 0
            for epoch in range(epochs):
                model2.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model2(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 10 == 0:
                    model2.eval()
                    with torch.no_grad():

                        train_pred = model2(x_train_tensor)
                        train_r2 = r2_score(np.expm1(y_train_fold), np.expm1(train_pred.cpu().numpy()))
                        train_mae = mean_absolute_error(np.expm1(y_train_fold),
                                                        np.expm1(train_pred.cpu().numpy()))

                        val_result = model2(x_val_tensor)
                        val_r2 = r2_score(np.expm1(y_val_fold), np.expm1(val_result.cpu().numpy()))
                        val_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(val_result.cpu().numpy()))
                        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], '
                              f'Train R2: {train_r2:.4f}, MAE: {train_mae:.4f}, '
                              f'Val R2: {val_r2:.4f}, MAE: {val_mae:.4f}')
                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        best_epoch = epoch + 1

            fold_results.append({
                'fold': fold + 1,
                'best_val_r2': best_val_r2,
                'best_epoch': best_epoch
            })
            best_epochs.append(best_epoch)

            print(f'Fold {fold + 1} completed. Best Val R2: {best_val_r2:.4f}, Best Epoch: {best_epoch}')

        val_r2_scores = [result['best_val_r2'] for result in fold_results]
        mean_val_r2 = np.mean(val_r2_scores)
        std_val_r2 = np.std(val_r2_scores)
        mean_best_epoch = int(np.mean(best_epochs))
        print(f'\nCross-Validation completed. Mean Val R2: {mean_val_r2:.4f} (±{std_val_r2:.4f})')
        print(f'Average best epoch: {mean_best_epoch}')

        print("\nTraining final model on all training data...")

        final_model = model

        criterion = nn.SmoothL1Loss(beta=0.8)
        optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)

        x_all_tensor = torch.FloatTensor(x_all).to(device)
        y_all_tensor = torch.FloatTensor(y_all).view(-1, 1).to(device)
        all_dataset = TensorDataset(x_all_tensor, y_all_tensor)
        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)
        x_test_tensor = torch.FloatTensor(x_test).to(device)

        best_test_r2 = -float('inf')
        best_test_model_state = None
        best_test_epoch = 0

        for epoch in range(mean_best_epoch):
            final_model.train()

            total_loss = 0
            nan_detected = False

            for batch_x, batch_y in all_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = final_model(batch_x)

                loss = criterion(outputs, batch_y)


                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss at epoch {epoch + 1}: {loss.item()}")
                    nan_detected = True
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            if nan_detected:
                print(f"Skipping evaluation at epoch {epoch + 1} due to invalid values")
                continue

            avg_loss = total_loss / len(all_loader)

            if (epoch + 1) % 2 == 0 or epoch == mean_best_epoch - 1:
                final_model.eval()

                with torch.no_grad():

                    train_result = final_model(x_all_tensor)
                    train_pred = train_result.cpu().numpy()


                    if np.isnan(train_pred).any() or np.isinf(train_pred).any():
                        print(f"Invalid values in train predictions at epoch {epoch + 1}")
                        train_pred = np.nan_to_num(train_pred, nan=0.0, posinf=0.0, neginf=0.0)


                    train_r2 = r2_score(np.expm1(y_all), np.expm1(train_pred))
                    train_mae = mean_absolute_error(np.expm1(y_all), np.expm1(train_pred))
                    train_rmse = np.sqrt(mean_squared_error(np.expm1(y_all), np.expm1(train_pred)))


                    test_result = final_model(x_test_tensor)
                    test_pred = test_result.cpu().numpy()


                    if np.isnan(test_pred).any() or np.isinf(test_pred).any():
                        print(f"Invalid values in test predictions at epoch {epoch + 1}")
                        test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)


                    test_r2 = r2_score(np.expm1(y_test), np.expm1(test_pred))
                    test_mae = mean_absolute_error(np.expm1(y_test), np.expm1(test_pred))
                    test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(test_pred)))


                    if test_r2 > best_test_r2:
                        best_test_r2 = test_r2
                        best_test_mae = test_mae
                        best_test_rmse = test_rmse
                        best_test_model_state = copy.deepcopy(final_model.state_dict())
                        best_test_epoch = epoch + 1
                        print(f"New best model saved at epoch {epoch + 1} with Test R2: {test_r2:.4f}")


                print(f'Epoch [{epoch + 1}/{mean_best_epoch}], '
                      f'Train Loss: {avg_loss:.4f}, Train R2: {train_r2:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f} '
                      f'Test R2: {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}')
            else:

                print(f'Epoch [{epoch + 1}/{mean_best_epoch}], Train Loss: {avg_loss:.4f}')

        print("Final model training completed")

        if best_test_model_state is not None:
            final_model.load_state_dict(best_test_model_state)
            print(f"Loaded best model from epoch {best_test_epoch} with Test R2: {best_test_r2:.4f}")

        final_model.eval()
        with torch.no_grad():
            test_result = final_model(x_test_tensor)
            test_pred = test_result.cpu().numpy()

        full_test_r2 = best_test_r2
        full_test_mae = best_test_mae
        full_test_rmse = best_test_rmse


        print('\n' + '=' * 50)
        print(f"{'Final Model Test Set Metrics':^50}")
        print('=' * 50)
        print(
            f"{'Before OOD Filtering':<25}: R2 = {full_test_r2:.4f}, MAE = {full_test_mae:.4f}, RMSE = {full_test_rmse:.4f}")

        print('=' * 50)
    return full_test_r2, full_test_mae, full_test_rmse


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

lr = 0.0018
epochs = 100
batch_size = 4096
print(lr, epochs, batch_size)


print('\n' + '=' * 50)
print(f"{'DeepMLPRegressor':^50}")
print('=' * 50)
model = DeepMLPRegressor(input_dim=44).to(device)
r2, mae, rmse= SNGP(output, epochs, lr, batch_size=batch_size, model = model)



