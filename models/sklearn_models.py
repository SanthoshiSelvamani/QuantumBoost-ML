"""
Additional sklearn-based ML models for classification and regression.
Includes: Logistic Regression, Decision Tree, Naive Bayes, SVM, KNN
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_logistic_regression(X_train, y_train, X_test, y_test, **kwargs):
    """
    Train Logistic Regression model.
    Note: For regression tasks, we use a binning approach or treat as classification.
    For time series regression, we'll use Ridge regression instead.
    """
    from sklearn.linear_model import Ridge
    
    model = Ridge(alpha=kwargs.get('alpha', 1.0))
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return {
        'model': model,
        'predictions': test_pred,
        'train_predictions': train_pred,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }


def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=10, min_samples_split=2, **kwargs):
    """
    Train Decision Tree Regressor model.
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return {
        'model': model,
        'predictions': test_pred,
        'train_predictions': train_pred,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'feature_importance': model.feature_importances_.tolist()
    }


def train_decision_tree_classifier(X_train, y_train, X_test, y_test, max_depth=10, min_samples_split=2, **kwargs):
    """
    Train Decision Tree Classifier model.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_train_class = (y_train > np.median(y_train)).astype(int)
    y_test_class = (y_test > np.median(y_train)).astype(int)
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train_class)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return {
        'model': model,
        'predictions': test_pred,
        'train_predictions': train_pred,
        'is_classifier': True,
        'metrics': {
            'accuracy': accuracy_score(y_test_class, test_pred),
            'precision': precision_score(y_test_class, test_pred, zero_division=0),
            'recall': recall_score(y_test_class, test_pred, zero_division=0),
            'f1_score': f1_score(y_test_class, test_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test_class, test_pred).tolist()
        },
        'feature_importance': model.feature_importances_.tolist()
    }


def train_naive_bayes(X_train, y_train, X_test, y_test, **kwargs):
    """
    Train Naive Bayes model.
    For regression, we discretize the target and use classification,
    then map back to regression values.
    """
    from sklearn.linear_model import BayesianRidge
    
    model = BayesianRidge()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return {
        'model': model,
        'predictions': test_pred,
        'train_predictions': train_pred,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }


def train_svm(X_train, y_train, X_test, y_test, kernel='rbf', C=1.0, **kwargs):
    """
    Train Support Vector Machine (SVR) model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVR(kernel=kernel, C=C)
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': test_pred,
        'train_predictions': train_pred,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }


def train_svm_classifier(X_train, y_train, X_test, y_test, kernel='rbf', C=1.0, **kwargs):
    """
    Train Support Vector Machine (SVC) Classifier model.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_train_class = (y_train > np.median(y_train)).astype(int)
    y_test_class = (y_test > np.median(y_train)).astype(int)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel=kernel, C=C)
    model.fit(X_train_scaled, y_train_class)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': test_pred,
        'train_predictions': train_pred,
        'is_classifier': True,
        'metrics': {
            'accuracy': accuracy_score(y_test_class, test_pred),
            'precision': precision_score(y_test_class, test_pred, zero_division=0),
            'recall': recall_score(y_test_class, test_pred, zero_division=0),
            'f1_score': f1_score(y_test_class, test_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test_class, test_pred).tolist()
        }
    }


def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5, **kwargs):
    """
    Train K-Nearest Neighbors Regressor model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': test_pred,
        'train_predictions': train_pred,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }


def train_knn_classifier(X_train, y_train, X_test, y_test, n_neighbors=5, **kwargs):
    """
    Train K-Nearest Neighbors Classifier model.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_train_class = (y_train > np.median(y_train)).astype(int)
    y_test_class = (y_test > np.median(y_train)).astype(int)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train_class)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': test_pred,
        'train_predictions': train_pred,
        'is_classifier': True,
        'metrics': {
            'accuracy': accuracy_score(y_test_class, test_pred),
            'precision': precision_score(y_test_class, test_pred, zero_division=0),
            'recall': recall_score(y_test_class, test_pred, zero_division=0),
            'f1_score': f1_score(y_test_class, test_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test_class, test_pred).tolist()
        }
    }
