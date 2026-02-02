import warnings
import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.utils.utility import standardizer
from pyod.models.combination import aom


# ============================================================
# Function: Remove extreme values while preserving missing values
# Description:
#   Remove upper and lower extreme values based on given percentiles
#   for specified columns, while retaining NaN entries.
# ============================================================
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


# ============================================================
# Function: Firm size categorization
# Description:
#   Categorize firms into size classes based on staff number.
# ============================================================
def firm_scale_split(staff_number):
    if 1 <= staff_number <= 49:
        return '1'
    if 50 <= staff_number <= 99:
        return '2'
    if 100 <= staff_number <= 199:
        return '3'
    if 200 <= staff_number <= 499:
        return '4'
    else:
        return '5'


# ============================================================
# Missing value imputation strategy (Method 01)
# Description:
#   Hierarchical median-based imputation:
#   1) Firm-level median
#   2) Industry + firm size median
#   3) Industry-level median
#   4) Global median
# ============================================================
def fill_missing_values_by_industry(df, column_name, industry_column):
    median_by_dwmc = df.groupby('dwmc')[column_name].median()
    median_by_industry_and_scale = df.groupby(
        [industry_column, 'Firm_scale']
    )[column_name].median()
    median_by_industry = df.groupby(industry_column)[column_name].median()
    median_all = df[column_name].median()

    def fill_value(row):
        if pd.isna(row[column_name]):
            dwmc = row['dwmc']
            firm_scale = row['Firm_scale']
            industry = row[industry_column]

            # Firm-level median
            if dwmc in median_by_dwmc.index:
                median_dwmc = median_by_dwmc[dwmc]
                if pd.notna(median_dwmc):
                    return median_dwmc

            # Industry + firm size median
            if (industry, firm_scale) in median_by_industry_and_scale.index:
                median_industry_scale = median_by_industry_and_scale[(industry, firm_scale)]
                if pd.notna(median_industry_scale):
                    return median_industry_scale

            # Industry-level median
            if industry in median_by_industry.index:
                median_industry = median_by_industry[industry]
                if pd.notna(median_industry):
                    return median_industry

            # Global median fallback
            if pd.notna(median_all):
                return median_all

        return row[column_name]

    df[column_name] = df.apply(fill_value, axis=1)


# ============================================================
# Missing value imputation strategy (Method 02)
# Description:
#   Estimate emission intensity and impute missing emissions
#   using industry-level median intensity.
# ============================================================
def calculate_emission_intensity(df, column_name):
    df['Emission_intensity'] = df[column_name] / df['Wastewater flow']
    return df.groupby('hyfl4')['Emission_intensity'].median()


def fill_missing_values(df, column_name, emission_intensity):
    def fill_value(row):
        if pd.isna(row[column_name]):
            hyfl4 = row['hyfl4']
            wastewater_flow = row['Wastewater flow']

            if hyfl4 in emission_intensity.index:
                return emission_intensity[hyfl4] * wastewater_flow

        return row[column_name]

    df[column_name] = df.apply(fill_value, axis=1)


# ============================================================
# Utility function: Check for NaN values in anomaly scores
# ============================================================
def check_scores(scores):
    if np.any(np.isnan(scores)):
        print("NaN values detected in scores.")
    else:
        print("No NaN values detected in scores.")


# ============================================================
# Anomaly detection ensemble framework
# Description:
#   Combine multiple unsupervised detectors and aggregate
#   anomaly scores using AOM (Average of Maximum).
# ============================================================
def anomaly_ensemble(x, random_state):
    # Isolation Forest
    clf = IForest(n_estimators=300, random_state=random_state)
    clf.fit(x)
    A1 = clf.decision_function(x)

    # Minimum Covariance Determinant
    warnings.filterwarnings('ignore', 'Determinant has increased; this should not happen: ')
    warnings.filterwarnings('ignore', 'The covariance matrix associated to your dataset ')
    clf = MCD(random_state=random_state)
    clf.fit(x)
    A2 = clf.decision_function(x)

    # Local Outlier Factor
    clf = LOF(n_neighbors=10)
    clf.fit(x)
    A3 = clf.decision_function(x)

    # k-Nearest Neighbors
    clf = KNN(n_neighbors=10)
    clf.fit(x)
    A4 = clf.decision_function(x)

    # Clustering-Based LOF
    clf = CBLOF(random_state=random_state)
    clf.fit(x)
    A5 = clf.decision_function(x)

    # Histogram-Based Outlier Score
    clf = HBOS(n_bins=10)
    clf.fit(x)
    A6 = clf.decision_function(x)

    # Stack and standardize anomaly scores
    scores = np.vstack([A1, A2, A3, A4, A5, A6]).T
    scores = standardizer(scores)

    # Aggregate scores using AOM
    y_by_aom = aom(
        scores,
        n_buckets=3,
        method='static',
        bootstrap_estimators=False,
        random_state=random_state
    )

    y_by_aom = pd.DataFrame(y_by_aom, columns=['scores'])
    return y_by_aom


# ============================================================
# Full anomaly detection pipeline
# Description:
#   1) Standardize selected features
#   2) Compute ensemble anomaly scores
#   3) Remove top anomalies based on specified rate
# ============================================================
def anomaly_detection(data, anomaly_rate, random_state, partial=False):
    to_model_columns = data[
        ['COD', 'pH', 'Total phosphorus', 'Ammonia nitrogen', 'Total nitrogen', 'Wastewater flow', 'HWsum']
    ]
    to_model_columns = standardizer(to_model_columns)

    data['score'] = anomaly_ensemble(
        to_model_columns,
        random_state=random_state
    )['scores']

    data = data.reset_index()
    data = data.sort_values(by='score')

    cleaned_data = data.iloc[:round(len(data) * (1 - anomaly_rate)), :]
    anomaly_data = data.iloc[round(len(data) * (1 - anomaly_rate)):, :]

    cleaned_data = cleaned_data.sort_index()
    anomaly_data = anomaly_data.sort_index()

    return cleaned_data
