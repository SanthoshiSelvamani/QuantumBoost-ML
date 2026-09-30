"""
QuantumBoost ML - Flask Web Application
A time-series forecasting app using quantum-inspired optimization.

How to run:
    python app.py
    
Then open http://localhost:5000 in your browser.
"""

import os
import io
import logging
import base64
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.data_inspection import load_data, sample_dataset_generator, inspect_dataframe, get_excel_sheet_names
from utils.preprocessing import (
    handle_missing_values, normalize_series, generate_lag_features,
    train_test_split_time_series, prepare_lstm_data, prepare_sklearn_data, compute_metrics
)
from optimizer.quantum_boost import QuantumBoostOptimizer
from models import (
    TENSORFLOW_AVAILABLE, train_xgboost, train_random_forest, train_random_forest_classifier,
    train_decision_tree, train_decision_tree_classifier, train_svm, train_svm_classifier,
    train_knn, train_knn_classifier, train_logistic_regression, train_naive_bayes
)
if TENSORFLOW_AVAILABLE:
    from models import build_lstm, train_lstm, train_lstm_keras, evaluate_lstm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.secret_key = "quantumboost_secret_key"

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'results')
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'dataset_history.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def load_dataset_history():
    """Load dataset history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_dataset_history(history):
    """Save dataset history to JSON file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def add_to_history(session_id, data_source, rows, columns, column_names):
    """Add a dataset entry to history."""
    history = load_dataset_history()
    entry = {
        'session_id': session_id,
        'data_source': data_source,
        'rows': rows,
        'columns': columns,
        'column_names': column_names[:5],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    history.insert(0, entry)
    history = history[:20]
    save_dataset_history(history)


def save_dataframe(df, session_id):
    """Save DataFrame to server-side storage."""
    filepath = os.path.join(UPLOAD_FOLDER, f'{session_id}.parquet')
    df.to_parquet(filepath, index=False)
    return filepath


def load_dataframe(session_id):
    """Load DataFrame from server-side storage."""
    filepath = os.path.join(UPLOAD_FOLDER, f'{session_id}.parquet')
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    return None


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_results(results, session_id):
    """Save results to server-side storage."""
    filepath = os.path.join(RESULTS_FOLDER, f'{session_id}_results.json')
    serializable = {
        'model_type': results.get('model_type'),
        'preprocessing_steps': results.get('preprocessing_steps'),
        'comparison_data': convert_to_serializable(results.get('comparison_data')),
        'quantum_metrics': convert_to_serializable(results.get('quantum_metrics')),
        'plots': results.get('plots')
    }
    with open(filepath, 'w') as f:
        json.dump(serializable, f)
    return filepath


def load_results(session_id):
    """Load results from server-side storage."""
    filepath = os.path.join(RESULTS_FOLDER, f'{session_id}_results.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def get_or_create_session_id():
    """Get existing session ID or create a new one."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


@app.route('/')
def landing():
    """Landing page with overview."""
    return render_template('landing.html')


@app.route('/app')
def index():
    """Index page with upload form and model selection."""
    history = load_dataset_history()
    return render_template(
        'index.html',
        tensorflow_available=TENSORFLOW_AVAILABLE,
        dataset_history=history
    )


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload or sample dataset generation."""
    try:
        session_id = get_or_create_session_id()
        use_sample = request.form.get('use_sample') == 'true'
        
        if use_sample:
            df = sample_dataset_generator(n=1095)
            session['data_source'] = 'Sample Dataset (3 years daily)'
        else:
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            filename = file.filename.lower()
            file_bytes = io.BytesIO(file.read())
            
            if filename.endswith('.csv'):
                df = load_data(file_bytes, file_type='csv')
                session['data_source'] = f'Uploaded: {file.filename}'
            elif filename.endswith(('.xls', '.xlsx')):
                sheet_name = request.form.get('sheet_name')
                sheet_names = get_excel_sheet_names(file_bytes)
                file_bytes.seek(0)
                
                if len(sheet_names) > 1 and not sheet_name:
                    temp_path = os.path.join(UPLOAD_FOLDER, f'{session_id}_temp.xlsx')
                    file_bytes.seek(0)
                    with open(temp_path, 'wb') as f:
                        f.write(file_bytes.read())
                    session['temp_file'] = temp_path
                    session['temp_filename'] = file.filename
                    session['sheet_names'] = sheet_names
                    return render_template('select_sheet.html', 
                                         sheets=sheet_names, 
                                         filename=file.filename)
                
                df = load_data(file_bytes, sheet_name=sheet_name, file_type='excel')
                session['data_source'] = f'Uploaded: {file.filename}'
            else:
                flash('Unsupported file format. Please upload CSV or Excel file.', 'error')
                return redirect(url_for('index'))
        
        default_model = 'lstm' if TENSORFLOW_AVAILABLE else 'xgboost'
        session['model_type'] = request.form.get('model_type', default_model)
        session['epochs'] = int(request.form.get('epochs', 30))
        session['batch_size'] = int(request.form.get('batch_size', 32))
        session['n_estimators'] = int(request.form.get('n_estimators', 100))
        session['max_depth'] = int(request.form.get('max_depth', 6))
        
        save_dataframe(df, session_id)
        
        add_to_history(session_id, session['data_source'], len(df), len(df.columns), df.columns.tolist())
        
        return redirect(url_for('inspect'))
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/select_sheet', methods=['POST'])
def select_sheet():
    """Handle Excel sheet selection."""
    try:
        session_id = get_or_create_session_id()
        sheet_name = request.form.get('sheet_name')
        temp_path = session.get('temp_file')
        
        if not temp_path or not os.path.exists(temp_path):
            flash('Session expired. Please upload the file again.', 'error')
            return redirect(url_for('index'))
        
        df = load_data(temp_path, sheet_name=sheet_name, file_type='excel')
        session['data_source'] = f"Uploaded: {session.get('temp_filename', 'Excel file')} (Sheet: {sheet_name})"
        
        save_dataframe(df, session_id)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        session.pop('temp_file', None)
        session.pop('temp_filename', None)
        session.pop('sheet_names', None)
        
        return redirect(url_for('inspect'))
        
    except Exception as e:
        logger.error(f"Sheet selection error: {e}")
        flash(f'Error loading sheet: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/load_history/<history_session_id>')
def load_history(history_session_id):
    """Load a dataset from history."""
    try:
        df = load_dataframe(history_session_id)
        if df is None:
            flash('Dataset no longer available.', 'error')
            return redirect(url_for('index'))
        
        history = load_dataset_history()
        data_source = 'Historical Dataset'
        for entry in history:
            if entry['session_id'] == history_session_id:
                data_source = entry['data_source']
                break
        
        new_session_id = str(uuid.uuid4())
        session['session_id'] = new_session_id
        session['data_source'] = data_source
        session['model_type'] = 'xgboost'
        session['epochs'] = 30
        session['batch_size'] = 32
        session['n_estimators'] = 100
        session['max_depth'] = 6
        
        save_dataframe(df, new_session_id)
        
        flash(f'Loaded: {data_source}', 'success')
        return redirect(url_for('inspect'))
        
    except Exception as e:
        logger.error(f"Load history error: {e}")
        flash(f'Error loading dataset: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/delete_history/<history_session_id>', methods=['POST'])
def delete_history(history_session_id):
    """Delete a dataset from history."""
    try:
        history = load_dataset_history()
        history = [entry for entry in history if entry['session_id'] != history_session_id]
        save_dataset_history(history)
        
        filepath = os.path.join(UPLOAD_FOLDER, f'{history_session_id}.parquet')
        if os.path.exists(filepath):
            os.remove(filepath)
        
        flash('Dataset removed from history.', 'success')
    except Exception as e:
        logger.error(f"Delete history error: {e}")
        flash(f'Error removing dataset: {str(e)}', 'error')
    
    return redirect(url_for('index'))


@app.route('/inspect')
def inspect():
    """Data inspection page."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        df = load_dataframe(session_id)
        if df is None:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        inspection = inspect_dataframe(df)
        
        return render_template('inspect.html',
                             inspection=inspection,
                             data_source=session.get('data_source', 'Unknown'),
                             model_type=session.get('model_type', 'lstm'),
                             columns=df.columns.tolist())
        
    except Exception as e:
        logger.error(f"Inspection error: {e}")
        flash(f'Error inspecting data: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/train', methods=['GET', 'POST'])
def train():
    """Train the model with selected preprocessing and optimizer options."""
    if request.method == 'GET':
        session_id = session.get('session_id')
        if not session_id:
            flash('Please upload a dataset first before training.', 'info')
            return redirect(url_for('index'))
        flash('Please configure training options and click "Proceed to Training" to start.', 'info')
        return redirect(url_for('inspect'))
    
    try:
        session_id = session.get('session_id')
        if not session_id:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        df = load_dataframe(session_id)
        if df is None:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        time_column = request.form.get('time_column')
        value_column = request.form.get('value_column')
        missing_strategy = request.form.get('missing_strategy', 'interpolate')
        normalization = request.form.get('normalization', 'minmax')
        remove_duplicates = request.form.get('remove_duplicates') == 'on'
        lag_features = request.form.get('lag_features', '1,2,3,7,14')
        
        model_type = session.get('model_type', 'lstm')
        epochs = session.get('epochs', 30)
        batch_size = session.get('batch_size', 32)
        n_estimators = session.get('n_estimators', 100)
        max_depth = session.get('max_depth', 6)
        
        preprocessing_steps = []
        
        if remove_duplicates:
            before_count = len(df)
            df = df.drop_duplicates()
            after_count = len(df)
            preprocessing_steps.append(f"Removed {before_count - after_count} duplicate rows")
        
        df = handle_missing_values(df, strategy=missing_strategy)
        preprocessing_steps.append(f"Missing values handled using: {missing_strategy}")
        
        if value_column and value_column in df.columns:
            normalized_values, scaler = normalize_series(df[value_column], method=normalization)
            df[f'{value_column}_normalized'] = normalized_values
            preprocessing_steps.append(f"Normalized {value_column} using: {normalization}")
        
        try:
            lags = [int(x.strip()) for x in lag_features.split(',') if x.strip()]
        except:
            lags = [1, 2, 3, 7, 14]
        
        work_col = f'{value_column}_normalized' if f'{value_column}_normalized' in df.columns else value_column
        df = generate_lag_features(df, work_col, lags=lags, drop_na=True)
        preprocessing_steps.append(f"Generated lag features: {lags}")
        
        train_df, test_df = train_test_split_time_series(df, test_size=0.2)
        preprocessing_steps.append(f"Train/test split: {len(train_df)}/{len(test_df)} samples")
        
        results = {}
        comparison_data = []
        
        if model_type == 'lstm' and not TENSORFLOW_AVAILABLE:
            flash('LSTM model requires TensorFlow which is not available. Using XGBoost instead.', 'warning')
            model_type = 'xgboost'
            session['model_type'] = 'xgboost'
        
        if model_type == 'lstm':
            sequence_length = 10
            X_train, y_train = prepare_lstm_data(train_df, work_col, sequence_length=sequence_length)
            X_test, y_test = prepare_lstm_data(test_df, work_col, sequence_length=sequence_length)
            
            if len(X_train) < 10 or len(X_test) < 5:
                flash('Not enough data for LSTM training. Try a larger dataset.', 'error')
                return redirect(url_for('inspect'))
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            for opt_name in ['quantum', 'adam', 'sgd']:
                model = build_lstm(input_shape, units=32)
                
                if opt_name == 'quantum':
                    optimizer = QuantumBoostOptimizer(
                        initial_lr=0.01, tunneling_rate=0.05,
                        jump_prob=0.03, temperature=1.0, decay=0.99
                    )
                    history = train_lstm(model, optimizer, X_train, y_train, 
                                       X_test, y_test, epochs=epochs, batch_size=batch_size)
                else:
                    history = train_lstm_keras(model, opt_name, X_train, y_train,
                                             X_test, y_test, epochs=epochs, 
                                             batch_size=batch_size, learning_rate=0.01)
                
                predictions, metrics = evaluate_lstm(model, X_test, y_test)
                
                results[opt_name] = {
                    'history': history,
                    'predictions': predictions.tolist(),
                    'metrics': metrics,
                    'y_test': y_test.tolist()
                }
                
                comparison_data.append({
                    'Optimizer': opt_name.upper() if opt_name != 'quantum' else 'QuantumBoost',
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'Time (s)': history['training_time']
                })
        
        elif model_type == 'xgboost':
            X_train, y_train = prepare_sklearn_data(train_df, work_col)
            X_test, y_test = prepare_sklearn_data(test_df, work_col)
            
            for opt_name in ['quantum', 'adam', 'sgd']:
                result = train_xgboost(X_train, y_train, X_test, y_test,
                                      optimizer_name=opt_name,
                                      n_estimators=n_estimators,
                                      max_depth=max_depth)
                
                results[opt_name] = {
                    'history': {'optimizer_state': result['optimizer_state'],
                               'training_time': result['training_time']},
                    'predictions': result['val_predictions'].tolist(),
                    'metrics': result['val_metrics'],
                    'y_test': y_test.tolist()
                }
                
                comparison_data.append({
                    'Optimizer': opt_name.upper() if opt_name != 'quantum' else 'QuantumBoost',
                    'MAE': result['val_metrics']['MAE'],
                    'RMSE': result['val_metrics']['RMSE'],
                    'MAPE': result['val_metrics']['MAPE'],
                    'Time (s)': result['training_time']
                })
        
        elif model_type == 'random_forest' or model_type == 'random_forest_regressor':
            X_train, y_train = prepare_sklearn_data(train_df, work_col)
            X_test, y_test = prepare_sklearn_data(test_df, work_col)
            
            for opt_name in ['quantum', 'adam', 'sgd']:
                result = train_random_forest(X_train, y_train, X_test, y_test,
                                            optimizer_name=opt_name,
                                            n_estimators=n_estimators,
                                            max_depth=max_depth)
                
                results[opt_name] = {
                    'history': {'optimizer_state': result['optimizer_state'],
                               'training_time': result['training_time']},
                    'predictions': result['val_predictions'].tolist(),
                    'metrics': result['val_metrics'],
                    'y_test': y_test.tolist()
                }
                
                comparison_data.append({
                    'Optimizer': opt_name.upper() if opt_name != 'quantum' else 'QuantumBoost',
                    'MAE': result['val_metrics']['MAE'],
                    'RMSE': result['val_metrics']['RMSE'],
                    'MAPE': result['val_metrics']['MAPE'],
                    'Time (s)': result['training_time']
                })
        
        elif model_type == 'random_forest_classifier':
            import time as time_module
            X_train, y_train = prepare_sklearn_data(train_df, work_col)
            X_test, y_test = prepare_sklearn_data(test_df, work_col)
            
            result = train_random_forest_classifier(X_train, y_train, X_test, y_test,
                                                    n_estimators=n_estimators,
                                                    max_depth=max_depth)
            
            results['quantum'] = {
                'history': {'optimizer_state': {}, 'training_time': result['training_time']},
                'predictions': result['val_predictions'].tolist(),
                'metrics': {
                    'Accuracy': result['val_accuracy'],
                    'Train Accuracy': result['train_accuracy']
                },
                'y_test': y_test.astype(int).tolist(),
                'confusion_matrix': result['confusion_matrix'],
                'classification_report': result['classification_report']
            }
            
            comparison_data.append({
                'Model': 'Random Forest Classifier',
                'Train Accuracy': f"{result['train_accuracy']:.4f}",
                'Test Accuracy': f"{result['val_accuracy']:.4f}",
                'Time (s)': result['training_time']
            })
        
        elif model_type in ['decision_tree_regressor', 'svm_regressor', 'knn_regressor', 'linear_regression', 'bayesian_ridge']:
            import time
            X_train, y_train = prepare_sklearn_data(train_df, work_col)
            X_test, y_test = prepare_sklearn_data(test_df, work_col)
            
            start_time = time.time()
            
            if model_type == 'decision_tree_regressor':
                result = train_decision_tree(X_train, y_train, X_test, y_test, max_depth=max_depth)
            elif model_type == 'svm_regressor':
                result = train_svm(X_train, y_train, X_test, y_test)
            elif model_type == 'knn_regressor':
                result = train_knn(X_train, y_train, X_test, y_test, n_neighbors=5)
            elif model_type == 'linear_regression':
                result = train_logistic_regression(X_train, y_train, X_test, y_test)
            elif model_type == 'bayesian_ridge':
                result = train_naive_bayes(X_train, y_train, X_test, y_test)
            
            training_time = time.time() - start_time
            
            metrics = result['metrics']
            results['quantum'] = {
                'history': {'optimizer_state': {}, 'training_time': training_time},
                'predictions': result['predictions'].tolist(),
                'metrics': {
                    'MAE': metrics['test_mae'],
                    'RMSE': metrics['test_rmse'],
                    'MAPE': abs(metrics['test_mae'] / np.mean(y_test) * 100) if np.mean(y_test) != 0 else 0,
                    'R2': metrics['test_r2']
                },
                'y_test': y_test.tolist()
            }
            
            comparison_data.append({
                'Optimizer': model_type.upper().replace('_', ' '),
                'MAE': metrics['test_mae'],
                'RMSE': metrics['test_rmse'],
                'MAPE': abs(metrics['test_mae'] / np.mean(y_test) * 100) if np.mean(y_test) != 0 else 0,
                'Time (s)': training_time
            })
        
        elif model_type in ['decision_tree_classifier', 'svm_classifier', 'knn_classifier']:
            import time as time_module
            X_train, y_train = prepare_sklearn_data(train_df, work_col)
            X_test, y_test = prepare_sklearn_data(test_df, work_col)
            
            if model_type == 'decision_tree_classifier':
                result = train_decision_tree_classifier(X_train, y_train, X_test, y_test, max_depth=max_depth)
                model_display_name = 'Decision Tree Classifier'
            elif model_type == 'svm_classifier':
                result = train_svm_classifier(X_train, y_train, X_test, y_test)
                model_display_name = 'SVM Classifier'
            elif model_type == 'knn_classifier':
                result = train_knn_classifier(X_train, y_train, X_test, y_test, n_neighbors=5)
                model_display_name = 'KNN Classifier'
            
            metrics = result['metrics']
            results['quantum'] = {
                'history': {'optimizer_state': {}, 'training_time': 0},
                'predictions': result['predictions'].tolist(),
                'metrics': {
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1_score']
                },
                'y_test': result['predictions'].tolist(),
                'confusion_matrix': metrics['confusion_matrix'],
                'is_classifier': True
            }
            
            comparison_data.append({
                'Model': model_display_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}"
            })
        
        else:
            X_train, y_train = prepare_sklearn_data(train_df, work_col)
            X_test, y_test = prepare_sklearn_data(test_df, work_col)
            
            for opt_name in ['quantum', 'adam', 'sgd']:
                result = train_random_forest(X_train, y_train, X_test, y_test,
                                            optimizer_name=opt_name,
                                            n_estimators=n_estimators,
                                            max_depth=max_depth)
                
                results[opt_name] = {
                    'history': {'optimizer_state': result['optimizer_state'],
                               'training_time': result['training_time']},
                    'predictions': result['val_predictions'].tolist(),
                    'metrics': result['val_metrics'],
                    'y_test': y_test.tolist()
                }
                
                comparison_data.append({
                    'Optimizer': opt_name.upper() if opt_name != 'quantum' else 'QuantumBoost',
                    'MAE': result['val_metrics']['MAE'],
                    'RMSE': result['val_metrics']['RMSE'],
                    'MAPE': result['val_metrics']['MAPE'],
                    'Time (s)': result['training_time']
                })
        
        plots = generate_result_plots(results, model_type)
        
        comparison_df = pd.DataFrame(comparison_data)
        for col in ['MAE', 'RMSE', 'MAPE', 'Time (s)']:
            if col in comparison_df.columns:
                comparison_df[col] = comparison_df[col].apply(lambda x: f'{x:.8f}' if isinstance(x, (int, float)) else x)
        comparison_html = comparison_df.to_html(classes='data-table comparison-table', index=False)
        
        confusion_matrix_data = results['quantum'].get('confusion_matrix') if model_type == 'random_forest_classifier' else None
        
        results_to_save = {
            'model_type': model_type,
            'preprocessing_steps': preprocessing_steps,
            'comparison_data': comparison_data,
            'quantum_metrics': results['quantum']['metrics'],
            'plots': plots,
            'confusion_matrix': confusion_matrix_data
        }
        save_results(results_to_save, session_id)
        
        return render_template('results.html',
                             model_type=model_type,
                             preprocessing_steps=preprocessing_steps,
                             comparison_table=comparison_html,
                             quantum_metrics=results['quantum']['metrics'],
                             plots=plots,
                             confusion_matrix=confusion_matrix_data,
                             data_source=session.get('data_source', 'Unknown'))
        
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        flash(f'Error during training: {str(e)}', 'error')
        return redirect(url_for('inspect'))


def generate_result_plots(results, model_type):
    """Generate visualization plots for results."""
    plots = {}
    
    if model_type == 'lstm':
        fig, ax = plt.subplots(figsize=(10, 5))
        for opt_name, data in results.items():
            label = 'QuantumBoost' if opt_name == 'quantum' else opt_name.upper()
            ax.plot(data['history']['train_losses'], label=f'{label} (Train)', linestyle='-')
            ax.plot(data['history']['val_losses'], label=f'{label} (Val)', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plots['loss_curve'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    quantum_state = results['quantum']['history'].get('optimizer_state', {})
    
    lr_history = quantum_state.get('lr_history', [])
    if lr_history:
        ax.plot(lr_history, label='Learning Rate', linewidth=2)
        
        tunneling = quantum_state.get('tunneling_events', [])
        jump = quantum_state.get('jump_events', [])
        
        if tunneling and len(lr_history) > 0:
            t_epochs = [min(t // 100, len(lr_history)-1) for t in tunneling[:20]]
            t_vals = [lr_history[e] for e in t_epochs]
            ax.scatter(t_epochs, t_vals, color='green', marker='^', 
                      s=100, label='Tunneling Events', zorder=5)
        
        if jump and len(lr_history) > 0:
            j_epochs = [min(j // 100, len(lr_history)-1) for j in jump[:20]]
            j_vals = [lr_history[e] for e in j_epochs]
            ax.scatter(j_epochs, j_vals, color='red', marker='v',
                      s=100, label='Jump Events', zorder=5)
        
        ax.set_xlabel('Epoch/Iteration')
        ax.set_ylabel('Learning Rate / Value')
        ax.set_title('QuantumBoost Optimizer Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No optimizer trajectory data available',
               ha='center', va='center', transform=ax.transAxes)
    
    plots['optimizer_trajectory'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    quantum_data = results['quantum']
    y_test = np.array(quantum_data['y_test'])
    predictions = np.array(quantum_data['predictions'])
    
    ax.plot(y_test, label='Actual', linewidth=2)
    ax.plot(predictions, label='Predicted (QuantumBoost)', linewidth=2, alpha=0.8)
    
    std_err = np.std(y_test - predictions)
    ax.fill_between(range(len(predictions)),
                   predictions - 1.96 * std_err,
                   predictions + 1.96 * std_err,
                   alpha=0.2, label='95% Confidence')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Forecast: Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plots['forecast'] = fig_to_base64(fig)
    
    return plots

def safe_format(val, suffix=""):
    try:
        return f"{float(val):.8f}{suffix}"
    except (ValueError, TypeError):
        return f"{val}{suffix}"


@app.route('/download_pdf')
def download_pdf():

    print("DOWNLOAD PDF ROUTE HIT")

    """Generate and download PDF report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.enums import TA_CENTER
        
        session_id = session.get('session_id')
        if not session_id:
            flash('No results available. Please train a model first.', 'error')
            return redirect(url_for('index'))
        
        results = load_results(session_id)
        if not results:
            flash('No results available. Please train a model first.', 'error')
            return redirect(url_for('index'))
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                    fontSize=24, spaceAfter=30, alignment=TA_CENTER)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'],
                                      fontSize=14, spaceAfter=12, spaceBefore=20)
        
        story = []
        
        story.append(Paragraph("QuantumBoost ML Results Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                              styles['Normal']))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Dataset Information", heading_style))
        story.append(Paragraph(f"Source: {session.get('data_source', 'Unknown')}", 
                              styles['Normal']))
        story.append(Paragraph(f"Model: {results.get('model_type', 'Unknown').upper()}", 
                              styles['Normal']))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("Preprocessing Steps", heading_style))
        for step in results.get('preprocessing_steps', []):
            story.append(Paragraph(f"• {step}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("Model Comparison Results", heading_style))
        comparison = results.get('comparison_data', [])
        if comparison:
            def format_num(val):
                try:
                    return f'{float(val):.8f}'
                except (ValueError, TypeError):
                    return str(val)
            
            table_data = [['Optimizer', 'MAE', 'RMSE', 'MAPE', 'Time (s)']]
            for row in comparison:
                table_data.append([
                    str(row['Optimizer']),
                    format_num(row['MAE']),
                    format_num(row['RMSE']),
                    format_num(row['MAPE']),
                    format_num(row['Time (s)'])
                ])
            
            table = Table(table_data, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.0*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("QuantumBoost Final Metrics", heading_style))
        metrics = results.get('quantum_metrics', {})
        mae_val = metrics.get('MAE', 'N/A')
        rmse_val = metrics.get('RMSE', 'N/A')
        mape_val = metrics.get('MAPE', 'N/A')
        story.append(Paragraph(f"MAE: {mae_val:.8f}" if isinstance(mae_val, (int, float)) else f"MAE: {mae_val}", styles['Normal']))
        story.append(Paragraph(f"RMSE: {rmse_val:.8f}" if isinstance(rmse_val, (int, float)) else f"RMSE: {rmse_val}", styles['Normal']))
        story.append(Paragraph(f"MAPE: {mape_val:.8f}%" if isinstance(mape_val, (int, float)) else f"MAPE: {mape_val}%", styles['Normal']))
        story.append(Spacer(1, 20))
        
        plots = results.get('plots', {})
        for plot_name, plot_data in plots.items():
            if plot_data:   
                story.append(Paragraph(f"{plot_name.replace('_', ' ').title()}", heading_style))
                img_buffer = io.BytesIO(base64.b64decode(plot_data))
                img = Image(img_buffer, width=6*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 30))
        story.append(Paragraph("About QuantumBoost Optimizer", heading_style))
        story.append(Paragraph(
            "QuantumBoost is a quantum-inspired optimization algorithm that combines "
            "standard gradient descent with quantum tunneling and probabilistic jumps "
            "to escape local minima and explore the loss landscape more effectively.",
            styles['Normal']
        ))
        
        doc.build(story)
        buffer.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'quantumboost_results_{timestamp}.pdf'
        
        return send_file(buffer, mimetype='application/pdf',
                        as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}", exc_info=True)
        flash(f'Error generating PDF: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/clean')
def clean():
    """Data cleaning page."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        df = load_dataframe(session_id)
        if df is None:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        from utils.data_cleaning import get_cleaning_preview
        preview = get_cleaning_preview(df)
        
        return render_template('clean.html',
                             preview=preview,
                             data_source=session.get('data_source', 'Unknown'),
                             head=df.head(10).to_html(classes='data-table', index=False))
        
    except Exception as e:
        logger.error(f"Cleaning page error: {e}")
        flash(f'Error loading cleaning page: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/clean_data', methods=['POST'])
def clean_data():
    """Apply data cleaning and show results."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        df = load_dataframe(session_id)
        if df is None:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        from utils.data_cleaning import clean_dataframe
        
        options = {
            'remove_duplicates': request.form.get('remove_duplicates') == 'on',
            'handle_missing': request.form.get('handle_missing', 'none'),
            'remove_outliers': request.form.get('remove_outliers') == 'on',
            'outlier_method': request.form.get('outlier_method', 'iqr'),
            'outlier_threshold': float(request.form.get('outlier_threshold', 1.5)),
            'trim_whitespace': request.form.get('trim_whitespace') == 'on',
            'standardize_case': request.form.get('standardize_case') or None,
            'remove_empty_columns': request.form.get('remove_empty_columns') == 'on',
            'remove_constant_columns': request.form.get('remove_constant_columns') == 'on'
        }
        
        cleaned_df, report = clean_dataframe(df, options)
        
        save_dataframe(cleaned_df, f'{session_id}_cleaned')
        session['cleaned_session_id'] = f'{session_id}_cleaned'
        
        return render_template('clean_results.html',
                             report=report,
                             data_source=session.get('data_source', 'Unknown'),
                             head=cleaned_df.head(10).to_html(classes='data-table', index=False),
                             tail=cleaned_df.tail(10).to_html(classes='data-table', index=False))
        
    except Exception as e:
        logger.error(f"Cleaning error: {e}")
        flash(f'Error cleaning data: {str(e)}', 'error')
        return redirect(url_for('clean'))


@app.route('/export_cleaned/<format>')
def export_cleaned(format):
    """Export cleaned data to CSV or Excel."""
    try:
        cleaned_id = session.get('cleaned_session_id')
        if not cleaned_id:
            flash('No cleaned data available. Please clean data first.', 'error')
            return redirect(url_for('clean'))
        
        df = load_dataframe(cleaned_id)
        if df is None:
            flash('No cleaned data available. Please clean data first.', 'error')
            return redirect(url_for('clean'))
        
        from utils.data_cleaning import export_to_csv, export_to_excel
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            data = export_to_csv(df)
            return send_file(
                io.BytesIO(data),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'cleaned_data_{timestamp}.csv'
            )
        elif format == 'excel':
            data = export_to_excel(df)
            return send_file(
                io.BytesIO(data),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'cleaned_data_{timestamp}.xlsx'
            )
        else:
            flash('Invalid export format', 'error')
            return redirect(url_for('clean'))
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        flash(f'Error exporting data: {str(e)}', 'error')
        return redirect(url_for('clean'))


@app.route('/visualize')
def visualize():
    """Visualization page."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        df = load_dataframe(session_id)
        if df is None:
            flash('No data loaded. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        from utils.visualization import get_available_visualizations
        viz_options = get_available_visualizations(df)
        
        return render_template('visualize.html',
                             viz_options=viz_options,
                             data_source=session.get('data_source', 'Unknown'))
        
    except Exception as e:
        logger.error(f"Visualization page error: {e}")
        flash(f'Error loading visualization page: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """Generate a chart based on user selection."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return {'error': 'No data loaded'}, 400
        
        df = load_dataframe(session_id)
        if df is None:
            return {'error': 'No data loaded'}, 400
        
        from utils.visualization import (
            generate_pie_chart, generate_bar_chart, generate_scatter_plot,
            generate_area_plot, generate_histogram, generate_line_chart, generate_box_plot
        )
        
        chart_type = request.form.get('chart_type')
        
        if chart_type == 'pie':
            column = request.form.get('column')
            chart = generate_pie_chart(df, column)
        elif chart_type == 'bar':
            column = request.form.get('column')
            value_column = request.form.get('value_column') or None
            orientation = request.form.get('orientation', 'vertical')
            chart = generate_bar_chart(df, column, value_column, orientation=orientation)
        elif chart_type == 'scatter':
            x_column = request.form.get('x_column')
            y_column = request.form.get('y_column')
            color_column = request.form.get('color_column') or None
            chart = generate_scatter_plot(df, x_column, y_column, color_column)
        elif chart_type == 'area':
            time_column = request.form.get('time_column')
            value_columns = request.form.getlist('value_columns')
            stacked = request.form.get('stacked') == 'on'
            chart = generate_area_plot(df, time_column, value_columns, stacked=stacked)
        elif chart_type == 'histogram':
            column = request.form.get('column')
            bins = int(request.form.get('bins', 30))
            chart = generate_histogram(df, column, bins)
        elif chart_type == 'line':
            time_column = request.form.get('time_column')
            value_columns = request.form.getlist('value_columns')
            chart = generate_line_chart(df, time_column, value_columns)
        elif chart_type == 'box':
            columns = request.form.getlist('columns')
            chart = generate_box_plot(df, columns)
        else:
            return {'error': 'Invalid chart type'}, 400
        
        return {'chart': chart, 'chart_type': chart_type}
        
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return {'error': str(e)}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
