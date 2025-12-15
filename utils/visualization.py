"""
Visualization utilities for QuantumBoost ML.
Provides various chart types for data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def generate_pie_chart(df, column, title=None, top_n=10):
    """
    Generate a pie chart for categorical data.
    
    Parameters:
    -----------
    df : DataFrame
    column : str
        Column name to create pie chart for
    title : str
        Chart title
    top_n : int
        Show top N categories, group rest as 'Other'
    """
    value_counts = df[column].value_counts()
    
    if len(value_counts) > top_n:
        top_values = value_counts.head(top_n - 1)
        other_sum = value_counts.tail(len(value_counts) - top_n + 1).sum()
        value_counts = pd.concat([top_values, pd.Series({'Other': other_sum})])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
    
    if len(value_counts) > 6:
        wedges, texts, autotexts = ax.pie(
            value_counts.values,
            autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(value_counts),
            pctdistance=0.75
        )
        ax.legend(wedges, [f'{label} ({val:,})' for label, val in zip(value_counts.index, value_counts.values)],
                  title=column, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    else:
        wedges, texts, autotexts = ax.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(value_counts)
        )
        for text in texts:
            text.set_fontsize(10)
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('black')
    
    ax.set_title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
    ax.axis('equal')
    plt.tight_layout()
    
    return _fig_to_base64(fig)


def generate_bar_chart(df, column, value_column=None, title=None, orientation='vertical', top_n=20):
    """
    Generate a bar chart.
    
    Parameters:
    -----------
    df : DataFrame
    column : str
        Category column
    value_column : str or None
        Value column for aggregation (if None, count occurrences)
    title : str
        Chart title
    orientation : str
        'vertical' or 'horizontal'
    top_n : int
        Show top N categories
    """
    if value_column:
        data = df.groupby(column)[value_column].sum().sort_values(ascending=False).head(top_n)
        ylabel = f'Sum of {value_column}'
    else:
        data = df[column].value_counts().head(top_n)
        ylabel = 'Count'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(data)))
    
    if orientation == 'horizontal':
        bars = ax.barh(range(len(data)), data.values, color=colors)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
    else:
        bars = ax.bar(range(len(data)), data.values, color=colors)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
    
    for bar, val in zip(bars, data.values):
        if orientation == 'horizontal':
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:,.0f}', 
                   va='center', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:,.0f}', 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_title(title or f'Bar Chart: {column}', fontsize=14, fontweight='bold')
    ax.grid(axis='y' if orientation == 'vertical' else 'x', alpha=0.3)
    plt.tight_layout()
    
    return _fig_to_base64(fig)


def generate_scatter_plot(df, x_column, y_column, color_column=None, title=None):
    """
    Generate a scatter plot.
    
    Parameters:
    -----------
    df : DataFrame
    x_column : str
        X-axis column
    y_column : str
        Y-axis column
    color_column : str or None
        Column for color coding points
    title : str
        Chart title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_column and color_column in df.columns:
        categories = df[color_column].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
        
        for cat, color in zip(categories, colors):
            mask = df[color_column] == cat
            ax.scatter(df.loc[mask, x_column], df.loc[mask, y_column], 
                      c=[color], label=str(cat), alpha=0.7, s=50)
        ax.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(df[x_column], df[y_column], alpha=0.6, s=50, c='steelblue')
    
    try:
        valid_mask = df[x_column].notna() & df[y_column].notna()
        x_valid = df.loc[valid_mask, x_column]
        y_valid = df.loc[valid_mask, y_column]
        if len(x_valid) > 1:
            z = np.polyfit(x_valid, y_valid, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend line')
    except Exception:
        pass
    
    ax.set_xlabel(x_column, fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    ax.set_title(title or f'Scatter Plot: {x_column} vs {y_column}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return _fig_to_base64(fig)


def generate_area_plot(df, time_column, value_columns, title=None, stacked=True):
    """
    Generate an area plot.
    
    Parameters:
    -----------
    df : DataFrame
    time_column : str
        X-axis (time/index) column
    value_columns : list of str
        Columns to plot as areas
    title : str
        Chart title
    stacked : bool
        Whether to stack the areas
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_df = df.copy()
    if time_column in plot_df.columns:
        plot_df = plot_df.sort_values(time_column)
        x_data = plot_df[time_column]
    else:
        x_data = range(len(plot_df))
    
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(value_columns)))
    
    if stacked and len(value_columns) > 1:
        ax.stackplot(x_data, [plot_df[col].values for col in value_columns],
                    labels=value_columns, colors=colors, alpha=0.8)
    else:
        for col, color in zip(value_columns, colors):
            ax.fill_between(x_data, plot_df[col], alpha=0.6, label=col, color=color)
            ax.plot(x_data, plot_df[col], linewidth=1.5, color=color)
    
    ax.set_xlabel(time_column if time_column else 'Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title or 'Area Plot', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if hasattr(x_data, 'dtype') and np.issubdtype(x_data.dtype, np.datetime64):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return _fig_to_base64(fig)


def generate_histogram(df, column, bins=30, title=None):
    """
    Generate a histogram.
    
    Parameters:
    -----------
    df : DataFrame
    column : str
        Column to create histogram for
    bins : int
        Number of bins
    title : str
        Chart title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = df[column].dropna()
    
    n, bins_edges, patches = ax.hist(data, bins=bins, edgecolor='white', 
                                     color='steelblue', alpha=0.7)
    
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return _fig_to_base64(fig)


def generate_line_chart(df, time_column, value_columns, title=None):
    """
    Generate a line chart.
    
    Parameters:
    -----------
    df : DataFrame
    time_column : str
        X-axis column
    value_columns : list of str
        Columns to plot as lines
    title : str
        Chart title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_df = df.copy()
    if time_column in plot_df.columns:
        plot_df = plot_df.sort_values(time_column)
        x_data = plot_df[time_column]
    else:
        x_data = range(len(plot_df))
    
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(value_columns)))
    
    for col, color in zip(value_columns, colors):
        ax.plot(x_data, plot_df[col], linewidth=2, label=col, color=color, marker='o', 
               markersize=3, alpha=0.8)
    
    ax.set_xlabel(time_column if time_column else 'Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title or 'Line Chart', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if hasattr(x_data, 'dtype') and np.issubdtype(x_data.dtype, np.datetime64):
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return _fig_to_base64(fig)


def generate_box_plot(df, columns, title=None):
    """
    Generate a box plot for multiple columns.
    
    Parameters:
    -----------
    df : DataFrame
    columns : list of str
        Columns to create box plots for
    title : str
        Chart title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if isinstance(columns, str):
        columns = [columns]
    
    data = [df[col].dropna().values for col in columns]
    
    bp = ax.boxplot(data, patch_artist=True, labels=columns)
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(columns)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title or 'Box Plot Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return _fig_to_base64(fig)


def get_available_visualizations(df):
    """
    Get available visualization options based on DataFrame structure.
    
    Returns:
    --------
    dict with available chart types and suitable columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            if col not in datetime_cols:
                datetime_cols.append(col)
    
    return {
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'datetime_columns': datetime_cols,
        'all_columns': df.columns.tolist(),
        'available_charts': {
            'pie_chart': len(categorical_cols) > 0 or len(numeric_cols) > 0,
            'bar_chart': True,
            'scatter_plot': len(numeric_cols) >= 2,
            'area_plot': len(numeric_cols) > 0,
            'histogram': len(numeric_cols) > 0,
            'line_chart': len(numeric_cols) > 0,
            'box_plot': len(numeric_cols) > 0
        }
    }
