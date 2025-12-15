"""
LSTM Model for time-series forecasting with custom optimizer support.
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import time


def build_lstm(input_shape, units=64):
    """
    Build a simple LSTM model for forecasting.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    units : int
        Number of LSTM units
        
    Returns:
    --------
    Keras Model
    """
    model = keras.Sequential([
        keras.layers.LSTM(units, input_shape=input_shape, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units // 2),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    return model


def train_lstm(model, optimizer, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train LSTM model with custom QuantumBoost optimizer using manual training loop.
    
    Parameters:
    -----------
    model : Keras Model
        The LSTM model to train
    optimizer : QuantumBoostOptimizer
        Custom optimizer instance
    X_train, y_train : numpy arrays
        Training data
    X_val, y_val : numpy arrays
        Validation data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    dict with training history and optimizer state
    """
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    
    n_samples = len(X_train)
    n_batches = max(1, n_samples // batch_size)
    
    for epoch in range(epochs):
        epoch_losses = []
        
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            with tf.GradientTape() as tape:
                predictions = model(X_batch, training=True)
                loss = tf.reduce_mean(tf.square(predictions - y_batch.reshape(-1, 1)))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            
            grad_arrays = [g.numpy() if g is not None else np.zeros_like(v.numpy()) 
                          for g, v in zip(gradients, model.trainable_variables)]
            param_arrays = [v.numpy() for v in model.trainable_variables]
            
            updated_params = optimizer.step(param_arrays, grad_arrays)
            
            for var, new_val in zip(model.trainable_variables, updated_params):
                var.assign(new_val)
            
            epoch_losses.append(float(loss))
        
        optimizer.end_epoch()
        
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        val_preds = model(X_val, training=False)
        val_loss = float(tf.reduce_mean(tf.square(val_preds - y_val.reshape(-1, 1))))
        val_losses.append(val_loss)
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'optimizer_state': optimizer.get_state(),
        'training_time': round(training_time, 2)
    }


def train_lstm_keras(model, optimizer_name, X_train, y_train, X_val, y_val, 
                     epochs=50, batch_size=32, learning_rate=0.01):
    """
    Train LSTM model with Keras built-in optimizers for comparison.
    
    Parameters:
    -----------
    model : Keras Model
        The LSTM model to train
    optimizer_name : str
        'adam' or 'sgd'
    X_train, y_train : numpy arrays
        Training data
    X_val, y_val : numpy arrays
        Validation data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimizer
        
    Returns:
    --------
    dict with training history
    """
    start_time = time.time()
    
    if optimizer_name.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': history.history['loss'],
        'val_losses': history.history['val_loss'],
        'optimizer_state': {
            'lr_history': [learning_rate] * epochs,
            'tunneling_events': [],
            'jump_events': [],
            'param_norms': []
        },
        'training_time': round(training_time, 2)
    }


def evaluate_lstm(model, X_test, y_test):
    """
    Evaluate LSTM model on test data.
    
    Parameters:
    -----------
    model : Keras Model
        Trained model
    X_test, y_test : numpy arrays
        Test data
        
    Returns:
    --------
    tuple of (predictions, metrics_dict)
    """
    from utils.preprocessing import compute_metrics
    
    predictions = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_test, predictions)
    
    return predictions, metrics
