import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam, SGD
from omegaconf import DictConfig
import logging
import torch
import os
import numpy as np

class AstroCLIPLightningModel(pl.LightningModule):
    def __init__(self, model, optimizer_config, metrics):
        super(AstroCLIPLightningModel, self).__init__()
        self.model = model
        self.metrics = metrics
        self.validation_outputs = []  # Store validation outputs
        # Check for both dict and DictConfig
        if isinstance(optimizer_config, (dict, DictConfig)):
            self.optimizer_config = optimizer_config
        else:
            print(optimizer_config)
            raise ValueError("optimizer_config must be a dictionary or DictConfig")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        logging.debug(f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        results = {}
        
        if 'mse' in self.metrics:
            mse = F.mse_loss(y_hat, y)
            self.log('val_mse', mse, prog_bar=True, logger=True)
            results['mse'] = mse
        
        if 'mae' in self.metrics:
            mae = F.l1_loss(y_hat, y)
            self.log('val_mae', mae, prog_bar=True, logger=True)
            results['mae'] = mae
        
        # Add predictions to results
        results['predictions'] = y_hat.detach()  # Detach to avoid tracking gradients
        
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        
        logging.debug(f"Validation Batch {batch_idx}, Results: {results}")
        self.validation_outputs.append(results)  # Store results for aggregation
        return results

    def on_validation_epoch_end(self):
        # Aggregate validation results and log them
        avg_mse = torch.stack([x['mse'] for x in self.validation_outputs if 'mse' in x]).mean() if 'mse' in self.metrics else None
        avg_mae = torch.stack([x['mae'] for x in self.validation_outputs if 'mae' in x]).mean() if 'mae' in self.metrics else None
        
        if avg_mse is not None:
            self.log('avg_val_mse', avg_mse, prog_bar=True, logger=True)
            logging.info(f"Epoch {self.current_epoch} Validation MSE: {avg_mse.item()}")
        
        if avg_mae is not None:
            self.log('avg_val_mae', avg_mae, prog_bar=True, logger=True)
            logging.info(f"Epoch {self.current_epoch} Validation MAE: {avg_mae.item()}")
        
        # Clear the outputs for the next epoch
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer_type = self.optimizer_config['type']
        lr = self.optimizer_config['learning_rate']
        
        if optimizer_type == 'Adam':
            optimizer = Adam(self.parameters(), lr=lr)
        elif optimizer_type == 'SGD':
            optimizer = SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        print("Predicting step batch: ", batch_idx)
        # Unpack the batch to get the features
        x, _ = batch  # Ignore the target variable during prediction
        
        # Forward pass to get predictions
        predictions = self(x)
        
        # Ensure predictions are detached and converted to numpy if needed
        return predictions.detach().cpu().numpy()