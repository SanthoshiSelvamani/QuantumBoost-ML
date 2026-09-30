"""
Data preprocessing utilities for QuantumBoost ML.
Provides functions for cleaning, normalizing, and preparing time-series data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def handle_missing_values(df, strategy='interpolate', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    strategy : str
        Strategy for handling missing values:
        - 'interpolate': Linear interpolation
        - 'mean': Fill with column mean
        - 'drop': Drop rows with missing values
        - 'ffill': Forward fill
    columns : list or None
        Columns to apply the strategy to (default: all numeric columns)
        
    Returns:
    --------
    DataFrame with missing values handled
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'interpolate':
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].bfill().ffill()
        elif strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'drop':
            pass
        elif strategy == 'ffill':
            df[col] = df[col].ffill().bfill()
    
    if strategy == 'drop':
        df = df.dropna(subset=columns)
    
    return df


def normalize_series(series, method='minmax'):
    """
    Normalize a pandas Series.
    
    Parameters:
    -----------
    series : Series
        Input Series to normalize
    method : str
        Normalization method: 'minmax' or 'standard'
        
    Returns:
    --------
    tuple of (normalized_values, scaler)
        The scaler can be used to inverse_transform later
    """
    values = series.values.reshape(-1, 1)
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    normalized = scaler.fit_transform(values)
    
    return normalized.flatten(), scaler


def generate_lag_features(df, value_col, lags=[1, 2, 3, 7, 14], drop_na=True):
    """
    Generate lag features for time-series forecasting.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    value_col : str
        Name of the value column to create lags for
    lags : list of int
        List of lag periods to create
    drop_na : bool
        Whether to drop rows with NaN values created by lagging
        
    Returns:
    --------
    DataFrame with lag features added
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
    
    if drop_na:
        df = df.dropna()
    
    return df


def train_test_split_time_series(df, test_size=0.2, shuffle=False):
    """
    Split time-series data into train and test sets preserving temporal order.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    test_size : float
        Proportion of data to use for testing (0-1)
    shuffle : bool
        Whether to shuffle (should be False for time-series)
        
    Returns:
    --------
    tuple of (train_df, test_df)
    """
    n = len(df)
    train_size = int(n * (1 - test_size))
    
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    return train_df, test_df


def prepare_lstm_data(df, value_col, sequence_length=10, target_col=None):
    """
    Prepare data for LSTM model (create sequences).
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with numeric values
    value_col : str
        Name of the main value column
    sequence_length : int
        Number of time steps in each sequence
    target_col : str or None
        Target column (default: same as value_col)
        
    Returns:
    --------
    tuple of (X, y) numpy arrays
        X shape: (samples, sequence_length, features)
        y shape: (samples,)
    """
    if target_col is None:
        target_col = value_col
    
    feature_cols = [col for col in df.columns 
                   if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
                   and col != target_col]
    
    if value_col not in feature_cols:
        feature_cols = [value_col] + feature_cols
    
    data = df[feature_cols].values
    targets = df[target_col].values
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_sklearn_data(df, value_col, lag_cols=None):
    """
    Prepare data for scikit-learn models (XGBoost, RandomForest).
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with lag features
    value_col : str
        Name of the target column
    lag_cols : list or None
        List of lag column names to use as features
        
    Returns:
    --------
    tuple of (X, y) numpy arrays
    """
    if lag_cols is None:
        lag_cols = [col for col in df.columns if 'lag_' in col]
    
    if not lag_cols:
        raise ValueError("No lag columns found. Run generate_lag_features first.")
    
    X = df[lag_cols].values.astype(np.float32)
    y = df[value_col].values.astype(np.float32)
    
    return X, y


def compute_metrics(y_true, y_pred):
    """
    Compute forecasting metrics.
    Returns a dictionary with MAE, RMSE, MAPE
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    metrics = {}
    metrics["MAE"] = round(mae, 4)
    metrics["RMSE"] = round(rmse, 4)
    metrics["MAPE"] = round(mape, 2) if not np.isnan(mape) else "N/A"

    return metrics