"""
Data cleaning utilities for QuantumBoost ML.
Provides comprehensive data cleaning and export functionality.
"""

import pandas as pd
import numpy as np
import io
from datetime import datetime


def clean_dataframe(df, options):
    """
    Clean DataFrame based on specified options.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame to clean
    options : dict
        Cleaning options:
        - remove_duplicates: bool
        - handle_missing: str ('drop', 'mean', 'median', 'mode', 'interpolate', 'ffill', 'bfill')
        - remove_outliers: bool
        - outlier_method: str ('iqr', 'zscore')
        - outlier_threshold: float
        - trim_whitespace: bool
        - standardize_case: str ('lower', 'upper', 'title', None)
        - remove_empty_columns: bool
        - remove_constant_columns: bool
        
    Returns:
    --------
    tuple of (cleaned_df, cleaning_report)
    """
    df = df.copy()
    report = {
        'original_rows': len(df),
        'original_columns': len(df.columns),
        'steps': [],
        'rows_removed': 0,
        'columns_removed': 0,
        'values_modified': 0
    }
    
    if options.get('remove_empty_columns', False):
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            report['steps'].append(f"Removed {len(empty_cols)} empty column(s): {', '.join(empty_cols)}")
            report['columns_removed'] += len(empty_cols)
    
    if options.get('remove_constant_columns', False):
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            report['steps'].append(f"Removed {len(constant_cols)} constant column(s): {', '.join(constant_cols)}")
            report['columns_removed'] += len(constant_cols)
    
    if options.get('remove_duplicates', False):
        before_count = len(df)
        df = df.drop_duplicates()
        removed = before_count - len(df)
        if removed > 0:
            report['steps'].append(f"Removed {removed} duplicate row(s)")
            report['rows_removed'] += removed
    
    if options.get('trim_whitespace', False):
        string_cols = df.select_dtypes(include=['object']).columns
        modified = 0
        for col in string_cols:
            original = df[col].copy()
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            modified += (original != df[col]).sum()
        if modified > 0:
            report['steps'].append(f"Trimmed whitespace in {len(string_cols)} column(s), {modified} value(s) modified")
            report['values_modified'] += modified
    
    case_option = options.get('standardize_case')
    if case_option:
        string_cols = df.select_dtypes(include=['object']).columns
        modified = 0
        for col in string_cols:
            original = df[col].copy()
            if case_option == 'lower':
                df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
            elif case_option == 'upper':
                df[col] = df[col].apply(lambda x: x.upper() if isinstance(x, str) else x)
            elif case_option == 'title':
                df[col] = df[col].apply(lambda x: x.title() if isinstance(x, str) else x)
            modified += (original != df[col]).sum()
        if modified > 0:
            report['steps'].append(f"Standardized case to {case_option} in {len(string_cols)} column(s)")
            report['values_modified'] += modified
    
    if options.get('remove_outliers', False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        method = options.get('outlier_method', 'iqr')
        threshold = options.get('outlier_threshold', 1.5)
        
        before_count = len(df)
        
        if method == 'iqr':
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                    mask = (df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))
                    df = df[mask]
        elif method == 'zscore':
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    if std_val > 0:
                        z_scores = np.abs((df[col] - mean_val) / std_val)
                        mask = (df[col].isna()) | (z_scores < threshold)
                        df = df[mask]
        
        removed = before_count - len(df)
        if removed > 0:
            report['steps'].append(f"Removed {removed} outlier row(s) using {method} method (threshold: {threshold})")
            report['rows_removed'] += removed
    
    missing_strategy = options.get('handle_missing')
    if missing_strategy and missing_strategy != 'none':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        missing_numeric_before = df[numeric_cols].isna().sum().sum() if len(numeric_cols) > 0 else 0
        missing_categorical_before = df[categorical_cols].isna().sum().sum() if len(categorical_cols) > 0 else 0
        total_missing_before = missing_numeric_before + missing_categorical_before
        
        if missing_strategy == 'drop':
            before_count = len(df)
            df = df.dropna()
            removed = before_count - len(df)
            report['steps'].append(f"Dropped {removed} row(s) with missing values")
            report['rows_removed'] += removed
        elif missing_strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
            for col in categorical_cols:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            report['steps'].append(f"Filled {missing_numeric_before} numeric missing value(s) with mean, {missing_categorical_before} categorical with mode")
            report['values_modified'] += total_missing_before
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            for col in categorical_cols:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            report['steps'].append(f"Filled {missing_numeric_before} numeric missing value(s) with median, {missing_categorical_before} categorical with mode")
            report['values_modified'] += total_missing_before
        elif missing_strategy == 'mode':
            for col in numeric_cols:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            for col in categorical_cols:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            report['steps'].append(f"Filled {total_missing_before} missing value(s) with column mode")
            report['values_modified'] += total_missing_before
        elif missing_strategy == 'interpolate':
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].bfill().ffill()
            for col in categorical_cols:
                df[col] = df[col].ffill().bfill()
            report['steps'].append(f"Interpolated {missing_numeric_before} numeric missing value(s), forward-filled {missing_categorical_before} categorical")
            report['values_modified'] += total_missing_before
        elif missing_strategy == 'ffill':
            for col in numeric_cols:
                df[col] = df[col].ffill().bfill()
            for col in categorical_cols:
                df[col] = df[col].ffill().bfill()
            report['steps'].append(f"Forward-filled {total_missing_before} missing value(s)")
            report['values_modified'] += total_missing_before
        elif missing_strategy == 'bfill':
            for col in numeric_cols:
                df[col] = df[col].bfill().ffill()
            for col in categorical_cols:
                df[col] = df[col].bfill().ffill()
            report['steps'].append(f"Backward-filled {total_missing_before} missing value(s)")
            report['values_modified'] += total_missing_before
    
    report['final_rows'] = len(df)
    report['final_columns'] = len(df.columns)
    
    return df, report


def export_to_csv(df):
    """Export DataFrame to CSV bytes."""
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()


def export_to_excel(df):
    """Export DataFrame to Excel bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    output.seek(0)
    return output.getvalue()


def get_cleaning_preview(df):
    """
    Get preview of data issues for cleaning.
    
    Returns:
    --------
    dict with data quality information
    """
    preview = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicate_rows': int(df.duplicated().sum()),
        'missing_values': {},
        'outliers': {},
        'data_types': {},
        'empty_columns': [],
        'constant_columns': []
    }
    
    for col in df.columns:
        preview['missing_values'][col] = int(df[col].isna().sum())
        preview['data_types'][col] = str(df[col].dtype)
    
    preview['empty_columns'] = df.columns[df.isna().all()].tolist()
    preview['constant_columns'] = [col for col in df.columns if df[col].nunique() <= 1]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = ((col_data < lower) | (col_data > upper)).sum()
            preview['outliers'][col] = int(outlier_count)
    
    preview['total_missing'] = sum(preview['missing_values'].values())
    preview['total_outliers'] = sum(preview['outliers'].values())
    
    return preview
