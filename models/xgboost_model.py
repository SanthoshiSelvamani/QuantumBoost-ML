"""
XGBoost Model for time-series forecasting with QuantumBoost optimizer integration.
"""

import numpy as np
import xgboost as xgb
import time
from optimizer.quantum_boost import QuantumBoostScalarOptimizer


def train_xgboost(X_train, y_train, X_val, y_val, optimizer_name='quantum',
                  n_estimators=100, max_depth=6, learning_rate=0.1,
                  optimizer_iterations=10):
    """
    Train XGBoost model with optional QuantumBoost hyperparameter optimization.
    
    For QuantumBoost: Uses the optimizer to tune the learning_rate hyperparameter
    through an optimization loop.
    
    Parameters:
    -----------
    X_train, y_train : numpy arrays
        Training data
    X_val, y_val : numpy arrays
        Validation data
    optimizer_name : str
        'quantum', 'adam', or 'sgd' (for comparison, adam/sgd use fixed lr)
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Initial learning rate
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
            initial_lr=0.05,
            tunneling_rate=0.15,
            jump_prob=0.1,
            temperature=0.3,
            decay=0.9
        )
        
        best_lr = learning_rate
        best_val_loss = float('inf')
        best_model = None
        
        current_lr = learning_rate
        prev_loss = float('inf')
        
        for iteration in range(optimizer_iterations):
            params = {
                'objective': 'reg:squarederror',
                'max_depth': max_depth,
                'learning_rate': current_lr,
                'n_estimators': max(10, n_estimators // optimizer_iterations),
                'random_state': 42,
                'verbosity': 0
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            val_preds = model.predict(X_val)
            val_loss = np.mean((val_preds - y_val) ** 2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = current_lr
                best_model = model
            
            gradient = (val_loss - prev_loss) / (current_lr + 1e-8) if iteration > 0 else 0
            current_lr = scalar_opt.step(current_lr, gradient, min_val=0.001, max_val=0.5)
            prev_loss = val_loss
        
        optimizer_state = scalar_opt.get_state()
        model = best_model
        
    else:
        lr_map = {
            'adam': learning_rate * 1.2,
            'sgd': learning_rate * 0.8
        }
        actual_lr = lr_map.get(optimizer_name, learning_rate)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'learning_rate': actual_lr,
            'n_estimators': n_estimators,
            'random_state': 42,
            'verbosity': 0
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        optimizer_state['lr_history'] = [actual_lr]
    
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
