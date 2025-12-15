# QuantumBoost ML

A Flask web application for time-series forecasting using a custom Quantum-Inspired Gradient Optimization algorithm.

## Overview

QuantumBoost ML provides:
- **Data Upload**: Support for CSV and Excel files with automatic sheet detection
- **Data Inspection**: Comprehensive analysis including statistics, correlations, outliers, and visualizations
- **Model Training**: LSTM, XGBoost, and Random Forest models with optimizer comparison
- **QuantumBoost Optimizer**: Custom optimizer using quantum tunneling and probabilistic jumps
- **PDF Reports**: Downloadable reports with metrics, comparisons, and plots

## Project Structure

```
├── app.py                    # Main Flask application with routes
├── main.py                   # Entry point for Gunicorn
├── optimizer/
│   └── quantum_boost.py      # QuantumBoost optimizer implementation
├── models/
│   ├── lstm_model.py         # LSTM model with manual training loop
│   ├── xgboost_model.py      # XGBoost with hyperparameter optimization
│   └── random_forest.py      # Random Forest with hyperparameter optimization
├── utils/
│   ├── data_inspection.py    # Data loading and inspection utilities
│   ├── preprocessing.py      # Data preprocessing functions
│   ├── data_cleaning.py      # Data cleaning and export utilities
│   └── visualization.py      # Chart generation utilities
├── templates/
│   ├── base.html             # Base template with header/footer
│   ├── index.html            # Upload page
│   ├── inspect.html          # Data inspection dashboard
│   ├── clean.html            # Data cleaning page
│   ├── clean_results.html    # Cleaning results with export
│   ├── visualize.html        # Interactive visualization page
│   ├── results.html          # Training results page
│   └── select_sheet.html     # Excel sheet selection
└── static/
    └── style.css             # Application styling
```

## Running the Application

```bash
# Start with Flask
python app.py

# Or with Gunicorn (production)
gunicorn --bind 0.0.0.0:5000 main:app
```

## QuantumBoost Optimizer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| initial_lr | 0.01 | Starting learning rate |
| tunneling_rate | 0.05 | Probability of quantum tunneling escapes |
| jump_prob | 0.03 | Probability of stochastic jumps |
| temperature | 1.0 | Magnitude of probabilistic jumps |
| decay | 0.99 | Learning rate decay per epoch |

## Key Features

### Data Inspection
- Head/tail preview (10 rows each)
- Shape, missing values, duplicates summary
- Descriptive statistics for numeric columns
- Correlation heatmap
- Outlier detection with IQR method
- Time series line preview

### Preprocessing Options
- Missing value handling: interpolate, mean, forward fill, drop
- Normalization: min-max or standard scaling
- Lag feature generation
- Duplicate removal

### Data Cleaning
- Remove duplicate rows
- Trim whitespace from text columns
- Standardize text case (lower/upper/title)
- Remove empty or constant columns
- Handle missing values (multiple strategies)
- Remove outliers (IQR or Z-score methods)
- Export cleaned data to CSV or Excel

### Visualizations
- Pie charts for categorical distributions
- Bar charts (vertical/horizontal) with aggregations
- Scatter plots with trend lines
- Area plots (stacked or overlaid)
- Histograms with mean/median markers
- Line charts for time series
- Box plots for distribution comparison

### Model Training
- Compares QuantumBoost vs Adam vs SGD optimizers
- Computes MAE, RMSE, MAPE metrics
- Generates loss curves and forecast plots

### PDF Report
- Dataset information
- Preprocessing steps applied
- Metrics comparison table
- All visualization plots

## Recent Changes

- December 2024: Initial implementation with full feature set
- December 9, 2024: Configured for Replit environment
  - Made TensorFlow/LSTM optional (unavailable due to package size constraints)
  - XGBoost and Random Forest models are available and work fully
  - Set up deployment configuration with Gunicorn
- December 10, 2024: Added Data Cleaning and Visualization modules
  - New data cleaning module with multiple cleaning options
  - Export cleaned data to CSV or Excel format
  - New visualization page with interactive charts:
    - Pie charts, Bar charts, Scatter plots
    - Area plots, Histograms, Line charts, Box plots
  - Updated navigation with 5-step workflow

## Environment Notes

- **LSTM Model**: Requires TensorFlow which is a very large package. Currently disabled due to storage constraints. XGBoost is used as the default model.
- **Session**: Uses SESSION_SECRET environment variable for Flask sessions
- **Storage**: Uses local file storage in `instance/` directory for uploads and results

## User Preferences

- No JavaScript frameworks required
- Clean, minimal UI design
- Matplotlib for all visualizations
- ReportLab for PDF generation
