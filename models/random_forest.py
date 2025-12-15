"""
Random Forest Model for time-series forecasting with QuantumBoost optimizer integration.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from optimizer.quantum_boost import QuantumBoostScalarOptimizer


def train_random_forest(X_train, y_train, X_val, y_val, optimizer_name='quantum',
                        n_estimators=100, max_depth=10, max_features='sqrt',
                        optimizer_iterations=8):
    """
    Train Random Forest model with optional QuantumBoost hyperparameter optimization.
    
    For QuantumBoost: Uses the optimizer to tune n_estimators through an optimization loop.
    
    Parameters:
    -----------
    X_train, y_train : numpy arrays
        Training data
    X_val, y_val : numpy arrays
        Validation data
    optimizer_name : str
        'quantum', 'adam', or 'sgd' (for comparison)
    n_estimators : int
        Number of trees (or starting point for optimization)
    max_depth : int
        Maximum tree depth
    max_features : str or float
        Number of features to consider
    optimizer_iterations : int
        Number of optimization iterations for QuantumBoost
        
    Returns:
    --------
    dict with model, predictions, metrics, and training info
    """
    from utils.preprocessing import compute_metrics
    
    start_time = time.time()
    
    optimizer_state = {
        'lr_history': [],
        'tunneling_events': [],
        'jump_events': [],
        'value_history': []
    }
    
    if optimizer_name == 'quantum':
        scalar_opt = QuantumBoostScalarOptimizer(
            initial_lr=10.0,
            tunneling_rate=0.2,
            jump_prob=0.1,
            temperature=0.5,
            decay=0.85
        )
        
        best_n_est = n_estimators
        best_val_loss = float('inf')
        best_model = None
        
        current_n_est = float(n_estimators)
        prev_loss = float('inf')
        
        for iteration in range(optimizer_iterations):
            actual_n_est = max(10, int(current_n_est))
            
            model = RandomForestRegressor(
                n_estimators=actual_n_est,
                max_depth=max_depth,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            val_preds = model.predict(X_val)
            val_loss = np.mean((val_preds - y_val) ** 2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_n_est = actual_n_est
                best_model = model
            
            gradient = (val_loss - prev_loss) / max(current_n_est, 1) if iteration > 0 else 0
            current_n_est = scalar_opt.step(current_n_est, gradient * 100, 
                                           min_val=10, max_val=500)
            prev_loss = val_loss
        
        optimizer_state = scalar_opt.get_state()
        model = best_model
        
    else:
        n_est_map = {
            'adam': int(n_estimators * 1.1),
            'sgd': int(n_estimators * 0.9)
        }
        actual_n_est = n_est_map.get(optimizer_name, n_estimators)
        
        model = RandomForestRegressor(
            n_estimators=actual_n_est,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        optimizer_state['lr_history'] = [float(actual_n_est)]
    
    training_time = time.time() - start_time
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_metrics = compute_metrics(y_train, train_preds)
    val_metrics = compute_metrics(y_val, val_preds)
    
    return {
        'model': model,
        'train_predictions': train_preds,
        'val_predictions': val_preds,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'optimizer_state': optimizer_state,
        'training_time': round(training_time, 2)
    }


def train_random_forest_classifier(X_train, y_train, X_val, y_val, 
                                   n_estimators=100, max_depth=10, max_features='sqrt'):
    """
    Train Random Forest Classifier for classification tasks.
    
    Parameters:
    -----------
    X_train, y_train : numpy arrays
        Training data (y_train should be integer class labels)
    X_val, y_val : numpy arrays
        Validation data
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth
    max_features : str or float
        Number of features to consider
        
    Returns:
    --------
    dict with model, predictions, accuracy, confusion matrix, and training info
    """
    start_time = time.time()
    
    y_train_int = y_train.astype(int)
    y_val_int = y_val.astype(int)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train_int)
    
    training_time = time.time() - start_time
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_accuracy = accuracy_score(y_train_int, train_preds)
    val_accuracy = accuracy_score(y_val_int, val_preds)
    
    conf_matrix = confusion_matrix(y_val_int, val_preds)
    class_report = classification_report(y_val_int, val_preds, output_dict=True, zero_division=0)
    
    return {
        'model': model,
        'train_predictions': train_preds,
        'val_predictions': val_preds,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'training_time': round(training_time, 2)
    }
