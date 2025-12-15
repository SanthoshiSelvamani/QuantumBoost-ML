"""
Data inspection utilities for QuantumBoost ML.
Provides functions for loading, inspecting, and visualizing datasets.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta


def load_data(file_path_or_bytes, sheet_name=None, file_type=None):
    """
    Load data from CSV or Excel file.
    
    Parameters:
    -----------
    file_path_or_bytes : str or bytes or file-like object
        Path to the file or file contents
    sheet_name : str or None
        Sheet name for Excel files (default: first sheet)
    file_type : str or None
        'csv' or 'excel' - auto-detected if None
        
    Returns:
    --------
    DataFrame
    """
    if file_type == 'csv' or (isinstance(file_path_or_bytes, str) and file_path_or_bytes.endswith('.csv')):
        return pd.read_csv(file_path_or_bytes)
    else:
        return pd.read_excel(file_path_or_bytes, sheet_name=sheet_name, engine='openpyxl')


def get_excel_sheet_names(file_path_or_bytes):
    """Get list of sheet names from an Excel file."""
    try:
        xl = pd.ExcelFile(file_path_or_bytes, engine='openpyxl')
        return xl.sheet_names
    except Exception:
        return []


def sample_dataset_generator(n=1095):
    """
    Generate a sample time-series dataset with trend, seasonality, and noise.
    
    Parameters:
    -----------
    n : int
        Number of data points (default: 1095 for ~3 years of daily data)
        
    Returns:
    --------
    DataFrame with 'date' and 'value' columns
    """
    np.random.seed(42)
    
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n)]
    
    t = np.arange(n)
    trend = 0.02 * t + 50
    
    weekly_seasonality = 5 * np.sin(2 * np.pi * t / 7)
    yearly_seasonality = 15 * np.sin(2 * np.pi * t / 365)
    
    noise = np.random.normal(0, 3, n)
    
    values = trend + weekly_seasonality + yearly_seasonality + noise
    
    outlier_indices = np.random.choice(n, size=10, replace=False)
    for idx in outlier_indices:
        values[idx] += np.random.choice([-1, 1]) * np.random.uniform(30, 50)
    
    missing_indices = np.random.choice([i for i in range(n) if i not in outlier_indices], 
                                       size=15, replace=False)
    values = values.astype(float)
    for idx in missing_indices:
        values[idx] = np.nan
    
    df = pd.DataFrame({'date': dates, 'value': values})
    
    dup_idx = np.random.choice(range(10, n-10), size=2, replace=False)
    dup_rows = df.iloc[dup_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def inspect_dataframe(df, time_column=None, value_column=None):
    """
    Perform comprehensive inspection of a DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        The dataset to inspect
    time_column : str or None
        Name of the time/date column (auto-detected if None)
    value_column : str or None
        Name of the value column for time-series (auto-detected if None)
        
    Returns:
    --------
    dict with inspection results suitable for Flask templates
    """
    results = {}
    
    results['head'] = df.head(10).to_html(classes='data-table', index=False)
    results['tail'] = df.tail(10).to_html(classes='data-table', index=False)
    
    results['shape'] = {'rows': df.shape[0], 'columns': df.shape[1]}
    
    missing = df.isnull().sum()
    results['missing_values'] = missing.to_dict()
    results['total_missing'] = int(missing.sum())
    
    results['duplicate_rows_count'] = int(df.duplicated().sum())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().round(2)
        results['descriptive_stats'] = stats.to_html(classes='data-table stats-table')
    else:
        results['descriptive_stats'] = '<p>No numeric columns found</p>'
    
    results['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    results['columns'] = df.columns.tolist()
    
    if time_column is None:
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                time_column = col
                break
    results['detected_time_column'] = time_column
    
    if value_column is None:
        for col in numeric_cols:
            if col != time_column and 'value' in col.lower():
                value_column = col
                break
        if value_column is None and numeric_cols:
            value_column = numeric_cols[0] if numeric_cols[0] != time_column else (
                numeric_cols[1] if len(numeric_cols) > 1 else None
            )
    results['detected_value_column'] = value_column
    
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr().round(2)
            results['correlation_matrix'] = corr_matrix.to_html(classes='data-table corr-table')
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr_matrix.values, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax.set_yticklabels(numeric_cols)
            plt.colorbar(im, ax=ax, label='Correlation')
            ax.set_title('Correlation Heatmap')
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}', 
                           ha='center', va='center', fontsize=8)
            results['correlation_plot'] = _fig_to_base64(fig)
        except Exception as e:
            results['correlation_matrix'] = f'<p>Error computing correlation: {e}</p>'
            results['correlation_plot'] = None
    else:
        results['correlation_matrix'] = '<p>Need at least 2 numeric columns for correlation</p>'
        results['correlation_plot'] = None
    
    outliers_summary = {}
    if numeric_cols:
        try:
            fig, axes = plt.subplots(1, min(len(numeric_cols), 4), figsize=(12, 4))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for idx, col in enumerate(numeric_cols[:4]):
                col_data = df[col].dropna()
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_count = ((col_data < lower) | (col_data > upper)).sum()
                outliers_summary[col] = {
                    'count': int(outlier_count),
                    'lower_bound': round(lower, 2),
                    'upper_bound': round(upper, 2)
                }
                
                axes[idx].boxplot(col_data)
                axes[idx].set_title(f'{col}\n({outlier_count} outliers)')
                axes[idx].set_ylabel('Value')
            
            plt.tight_layout()
            results['outlier_boxplot'] = _fig_to_base64(fig)
        except Exception as e:
            results['outlier_boxplot'] = None
    else:
        results['outlier_boxplot'] = None
    
    results['outliers_by_iqr'] = outliers_summary
    
    if value_column and value_column in df.columns:
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            plot_df = df.copy()
            
            if time_column and time_column in df.columns:
                plot_df = plot_df.sort_values(time_column)
                x_data = plot_df[time_column]
            else:
                x_data = range(len(plot_df))
            
            ax.plot(x_data, plot_df[value_column], linewidth=1)
            ax.set_xlabel(time_column if time_column else 'Index')
            ax.set_ylabel(value_column)
            ax.set_title(f'Time Series Preview: {value_column}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            results['line_chart_preview'] = _fig_to_base64(fig)
        except Exception as e:
            results['line_chart_preview'] = None
    else:
        results['line_chart_preview'] = None
    
    return results
