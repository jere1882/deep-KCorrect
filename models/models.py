import lightning as L
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
import torch.nn.functional as F
import torchvision
from lightning import Trainer
from numpy import ndarray
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def few_shot(
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    max_epochs: int = 10,
    hidden_dims: list[int] = [64, 64],
    lr: float = 1e-3,
) -> ndarray:
    """Train a few-shot model using a simple neural network"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_features = y_train.shape[1] if len(y_train.shape) > 1 else 1
    model = MLP(
        n_in=X_train.shape[1],
        n_out=num_features,
        n_hidden=hidden_dims,
        act=[nn.ReLU()] * (len(hidden_dims) + 1),
        dropout=0.1,
    )

    # Set up the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.cuda()
    model.train()
    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.cuda()).squeeze()
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Make predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy()
    return preds

def residual_few_shot(
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    max_epochs: int = 10,
    hidden_dims: list[int] = [64, 64],
    lr: float = 1e-3,
) -> ndarray:
    """Train a few-shot model using a simple neural network"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_features = y_train.shape[1] if len(y_train.shape) > 1 else 1
    model = ResidualMLP(
        n_in=X_train.shape[1],
        n_out=num_features,
        n_hidden=hidden_dims,
        act=[nn.ReLU()] * (len(hidden_dims) + 1),
        dropout=0.1,
    )

    # Set up the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.cuda()
    model.train()
    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.cuda()).squeeze()
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Make predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy()
    return preds

def attention_few_shot(
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    max_epochs: int = 10,
    hidden_dims: list[int] = [64, 64],
    lr: float = 1e-3,
) -> ndarray:
    """Train a few-shot model using a simple neural network"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_features = y_train.shape[1] if len(y_train.shape) > 1 else 1
    model = SelfAttentionMLP(
        n_in=X_train.shape[1],
        n_out=num_features,
        hidden_dims=hidden_dims,
        dropout=0.1,
    )

    # Set up the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.cuda()
    model.train()
    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.cuda()).squeeze()
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Make predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy()
    return preds


def zero_shot(
    X_train: ndarray, y_train: ndarray, X_test: ndarray, n_neighbors: int = 64
) -> ndarray:
    """Train a zero-shot model using KNN"""
    print("training with n_neighbors = ", n_neighbors)
    neigh = KNeighborsRegressor(weights="distance", n_neighbors=n_neighbors)
    neigh.fit(X_train, y_train)
    preds = neigh.predict(X_test)
    return preds

def zero_shot_with_uncertainty(
    X_train: ndarray, y_train: ndarray, X_test: ndarray, n_neighbors: int = 64
) -> tuple[ndarray, ndarray]:
    """
    Train a zero-shot model using KNN and return predictions with certainty scores.
    
    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        X_test (ndarray): Test features.
        n_neighbors (int): Number of neighbors to consider.

    Returns:
        tuple: A tuple containing predictions (ndarray) and certainty scores (ndarray).
    """
    print("Training with n_neighbors =", n_neighbors)
    neigh = KNeighborsRegressor(weights="distance", n_neighbors=n_neighbors)
    neigh.fit(X_train, y_train)
    
    # Get predictions
    preds = neigh.predict(X_test)
    
    # Compute certainty scores based on the inverse of mean distances
    distances, _ = neigh.kneighbors(X_test, return_distance=True)
    certainty_scores = 1 / (distances.mean(axis=1) + 1e-9)  # Add epsilon to avoid division by zero

    return preds, certainty_scores

class MLP(nn.Sequential):
    """MLP model"""

    def __init__(self, n_in, n_out, n_hidden=(16, 16, 16), act=None, dropout=0):
        if act is None:
            act = [
                nn.LeakyReLU(),
            ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_) - 2):
            layer.append(nn.Linear(n_[i], n_[i + 1]))
            layer.append(act[i])
            layer.append(nn.Dropout(p=dropout))
        layer.append(nn.Linear(n_[-2], n_[-1]))
        super(MLP, self).__init__(*layer)


class ResidualBlock(nn.Module):
    def __init__(self, n_units, activation=nn.ReLU, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_units, n_units),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(n_units, n_units),
        )

    def forward(self, x):
        return x + self.layer(x)

class ResidualMLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=(64, 64), activation=nn.ReLU, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(n_in, n_hidden[0]), activation()]
        for hidden_dim in n_hidden:
            layers.append(ResidualBlock(hidden_dim, activation, dropout))
        layers.append(nn.Linear(n_hidden[-1], n_out))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SelfAttentionMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden_dims=(64, 64), num_heads=4, dropout=0.1):
        super().__init__()
        self.linear_in = nn.Linear(n_in, hidden_dims[0])
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dims[0], num_heads=num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], n_out),
        )

    def forward(self, x):
        x = self.linear_in(x)
        x = x.unsqueeze(0)  # Add sequence dimension
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(0)  # Remove sequence dimension
        return self.mlp(x)


class ConditionalFlowStack(dist.conditional.ConditionalComposeTransformModule):
    """Normalizing flow stack for conditional distribution"""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dims: int,
        num_flows: int,
        device: str = "cuda",
    ):
        coupling_transforms = [
            T.conditional_spline(
                input_dim,
                context_dim,
                count_bins=8,
                hidden_dims=hidden_dims,
                order="quadratic",
            ).to(device)
            for _ in range(num_flows)
        ]

        super().__init__(coupling_transforms, cache_size=1)
