"""
QuantumBoost Optimizer: A quantum-inspired gradient optimization algorithm.

This optimizer combines standard gradient descent with quantum-inspired techniques:
1. Quantum tunneling: Small random perturbations to escape local minima
2. Probabilistic jumps: Larger random jumps to explore the loss landscape
3. Adaptive learning rate: RMSprop-like adjustment based on gradient history
"""

import numpy as np


class QuantumBoostOptimizer:
    """
    Quantum-Inspired Gradient Optimizer for machine learning models.
    
    Parameters:
    -----------
    initial_lr : float
        Starting learning rate (default: 0.01)
    tunneling_rate : float
        Probability and intensity of tunneling escapes from local minima (default: 0.05)
    jump_prob : float
        Probability of a stochastic jump each step (default: 0.03)
    temperature : float
        Controls magnitude of probabilistic jumps (default: 1.0)
    decay : float
        Learning rate decay per epoch (default: 0.99)
    seed : int or None
        Random seed for reproducibility (default: None)
    """
    
    def __init__(self, initial_lr=0.01, tunneling_rate=0.05, jump_prob=0.03, 
                 temperature=1.0, decay=0.99, seed=None):
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.tunneling_rate = tunneling_rate
        self.jump_prob = jump_prob
        self.temperature = temperature
        self.decay = decay
        
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        
        self.lr_history = [initial_lr]
        self.tunneling_events = []
        self.jump_events = []
        self.param_norms = []
        self.step_count = 0
        self.epoch_count = 0
        
        self.squared_grad_avg = None
        self.epsilon = 1e-8
        self.beta = 0.9
    
    def step(self, params, grads):
        """
        Perform one optimization step.
        
        Parameters:
        -----------
        params : list of numpy arrays
            Current model parameters
        grads : list of numpy arrays
            Gradients of the loss with respect to parameters
            
        Returns:
        --------
        updated_params : list of numpy arrays
            Updated parameters after optimization step
        """
        self.step_count += 1
        updated_params = []
        
        if self.squared_grad_avg is None:
            self.squared_grad_avg = [np.zeros_like(g) for g in grads]
        
        tunneling_occurred = False
        jump_occurred = False
        total_param_norm = 0.0
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            param = np.array(param, dtype=np.float32)
            grad = np.array(grad, dtype=np.float32)
            
            self.squared_grad_avg[i] = (self.beta * self.squared_grad_avg[i] + 
                                        (1 - self.beta) * np.square(grad))
            
            adaptive_lr = self.lr / (np.sqrt(self.squared_grad_avg[i]) + self.epsilon)
            
            new_param = param - adaptive_lr * grad
            
            if np.random.random() < self.tunneling_rate:
                tunneling_occurred = True
                grad_norm = np.linalg.norm(grad)
                if grad_norm > self.epsilon:
                    normalized_grad = grad / grad_norm
                else:
                    normalized_grad = np.zeros_like(grad)
                
                tunneling_noise = np.random.normal(0, 0.01, size=param.shape).astype(np.float32)
                tunneling_direction = -normalized_grad * 0.1 + tunneling_noise
                new_param = new_param + tunneling_direction * self.lr
            
            if np.random.random() < self.jump_prob:
                jump_occurred = True
                jump_noise = np.random.normal(0, self.temperature * 0.1, size=param.shape).astype(np.float32)
                new_param = new_param + jump_noise
            
            updated_params.append(new_param)
            total_param_norm += np.linalg.norm(new_param)
        
        if tunneling_occurred:
            self.tunneling_events.append(self.step_count)
        if jump_occurred:
            self.jump_events.append(self.step_count)
        
        self.param_norms.append(total_param_norm)
        
        return updated_params
    
    def end_epoch(self):
        """Call at the end of each epoch to decay learning rate."""
        self.epoch_count += 1
        self.lr = self.lr * self.decay
        self.lr_history.append(self.lr)
    
    def get_state(self):
        """
        Get optimizer state for visualization.
        
        Returns:
        --------
        dict with keys:
            - lr_history: list of learning rates over epochs
            - tunneling_events: list of step indices where tunneling occurred
            - jump_events: list of step indices where jumps occurred
            - param_norms: list of parameter norms over steps
        """
        return {
            'lr_history': self.lr_history.copy(),
            'tunneling_events': self.tunneling_events.copy(),
            'jump_events': self.jump_events.copy(),
            'param_norms': self.param_norms.copy(),
            'step_count': self.step_count,
            'epoch_count': self.epoch_count
        }
    
    def reset(self):
        """Reset optimizer state for a new training run."""
        self.lr = self.initial_lr
        self.lr_history = [self.initial_lr]
        self.tunneling_events = []
        self.jump_events = []
        self.param_norms = []
        self.step_count = 0
        self.epoch_count = 0
        self.squared_grad_avg = None


class QuantumBoostScalarOptimizer:
    """
    Scalar version of QuantumBoost for hyperparameter optimization.
    Used for XGBoost and RandomForest where direct gradient updates aren't possible.
    """
    
    def __init__(self, initial_lr=0.01, tunneling_rate=0.1, jump_prob=0.05,
                 temperature=0.5, decay=0.95, seed=None):
        self.lr = initial_lr
        self.tunneling_rate = tunneling_rate
        self.jump_prob = jump_prob
        self.temperature = temperature
        self.decay = decay
        
        if seed is not None:
            np.random.seed(seed)
        
        self.lr_history = [initial_lr]
        self.tunneling_events = []
        self.jump_events = []
        self.value_history = []
        self.step_count = 0
    
    def step(self, current_value, gradient, min_val=0.001, max_val=1.0):
        """
        Optimize a scalar hyperparameter.
        
        Parameters:
        -----------
        current_value : float
            Current hyperparameter value
        gradient : float
            Estimated gradient (can be numerical approximation)
        min_val : float
            Minimum allowed value
        max_val : float
            Maximum allowed value
            
        Returns:
        --------
        new_value : float
            Updated hyperparameter value
        """
        self.step_count += 1
        
        new_value = current_value - self.lr * gradient
        
        if np.random.random() < self.tunneling_rate:
            self.tunneling_events.append(self.step_count)
            tunneling_noise = np.random.normal(0, 0.01)
            new_value += tunneling_noise - 0.1 * self.lr * np.sign(gradient)
        
        if np.random.random() < self.jump_prob:
            self.jump_events.append(self.step_count)
            jump_noise = np.random.normal(0, self.temperature * 0.1)
            new_value += jump_noise
        
        new_value = np.clip(new_value, min_val, max_val)
        
        self.lr *= self.decay
        self.lr_history.append(self.lr)
        self.value_history.append(new_value)
        
        return new_value
    
    def get_state(self):
        """Get optimizer state for visualization."""
        return {
            'lr_history': self.lr_history.copy(),
            'tunneling_events': self.tunneling_events.copy(),
            'jump_events': self.jump_events.copy(),
            'value_history': self.value_history.copy(),
            'step_count': self.step_count
        }
