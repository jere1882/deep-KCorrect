import numpy as np
from sklearn.metrics import r2_score
from models.models import MLP, SelfAttentionMLP, ResidualMLP
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb


def calculate_zero_shot(X_train, y_train, X_test, y_test, n_neighbors = 64):
    """Train and evaluate a zero-shot model using KNN with standardized data"""
    # Scale properties
    scaler = {"mean": y_train.mean(), "std": y_train.std()}
    y_train_scaled = (y_train - scaler["mean"]) / scaler["std"]
    
    # Create and train KNN model
    print("training with n_neighbors = ", n_neighbors)
    neigh = KNeighborsRegressor(weights="distance", n_neighbors=n_neighbors)
    neigh.fit(X_train, y_train_scaled)
    y_pred = neigh.predict(X_test)

    results = {}
    results["preds"] = y_pred * scaler["std"] + scaler["mean"]
    results["mae"] = np.mean(np.abs(results["preds"] - y_test))
    results["r2"] = r2_score(y_test, results["preds"])

    return results

def calculate_few_shot_xgb(
    X_support: np.ndarray, y_support: np.ndarray, X_query: np.ndarray, y_query: np.ndarray,
    n_estimators: int = 100, learning_rate: float = 0.1
) -> dict:
    """Train and evaluate a few-shot model using XGBoost with standardized data
    
    Args:
        X_support (ndarray): Support set features (few examples for training)
        y_support (ndarray): Support set labels
        X_query (ndarray): Query set features to predict
        y_query (ndarray): Query set labels for evaluation
        n_estimators (int): Number of boosting rounds
        learning_rate (float): Learning rate for the model
        
    Returns:
        dict: Dictionary containing predictions, MAE, and R2 score
    """
    # Scale properties
    scaler = {"mean": y_support.mean(), "std": y_support.std()}
    y_support_scaled = (y_support - scaler["mean"]) / scaler["std"]
    
    print(f"Training few-shot XGBoost with {len(X_support)} support examples")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective='reg:squarederror',
        # Few-shot specific parameters
        min_child_weight=1,  # Allow smaller leaf sizes
        max_depth=3,         # Prevent overfitting on small data
        subsample=1.0,       # Use all data points due to small sample size
    )
    model.fit(X_support, y_support_scaled)
    y_pred = model.predict(X_query)
    
    results = {}
    results["preds"] = y_pred * scaler["std"] + scaler["mean"]
    results["mae"] = np.mean(np.abs(results["preds"] - y_query))
    results["r2"] = r2_score(y_query, results["preds"])
    
    return results

def initialize_model(config):
    """Initialize a model based on the configuration."""
    model_type = config['model_type']
    # Convert DictConfig to a regular dictionary
    parameters = {k: v for k, v in config.items() if k != 'model_type'}

    # Define a mapping from string names to PyTorch activation classes
    activation_mapping = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        # Add more mappings as needed
    }

    # Convert the activation function string to an actual class
    if 'act' in parameters and isinstance(parameters['act'], str):
        act_class = activation_mapping.get(parameters['act'])
        if act_class is None:
            raise ValueError(f"Unsupported activation function: {parameters['act']}")
        parameters['act'] = [act_class() for _ in range(len(parameters['n_hidden']) + 1)]

    model_mapping = {
        "MLP": MLP,
        "SelfAttentionMLP": SelfAttentionMLP,
        "ResidualMLP": ResidualMLP
    }

    if model_type in model_mapping:
        model_class = model_mapping[model_type]
        model = model_class(**parameters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model