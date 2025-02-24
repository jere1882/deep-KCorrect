import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.astroclip_dataset import AstroclipDataset
from models.lightning_model import AstroCLIPLightningModel
from utils.data_utils import load_data, load_dataset_between_redshifts
from utils.train_utils import initialize_model
import os
import numpy as np
# Set precision for Tensor Cores
torch.set_float32_matmul_precision('medium')  # or 'high' for more performance

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.debug("Loaded main configuration: %s", cfg)

    # Debug statement to check optimizer configuration
    logging.debug("Optimizer configuration: %s", cfg.training.optimizer)

    # Load and preprocess data using paths from config
    X_train, y_train, X_val, y_val = load_data(
        train_path=cfg.data.train_path,
        test_path=cfg.data.test_path,
        z_min=cfg.data.z_min,
        z_max=cfg.data.z_max,
        target_variable=cfg.data.target_variable,
        include_redshift=cfg.data.include_redshift
    )
    logging.debug("Data loaded successfully.")


    # Create dataset and dataloader for training
    train_dataset = AstroclipDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)

    # Create dataset and dataloader for validation
    val_dataset = AstroclipDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, drop_last=False)

    # Initialize model using the configuration
    model = initialize_model(cfg.model)
    logging.debug("Model initialized: %s", model)

    # Initialize LightningModule with model, optimizer settings, and metrics
    lightning_model = AstroCLIPLightningModel(
        model=model,
        optimizer_config=cfg.training.optimizer,
        metrics=cfg.data.metrics
    )
    logging.debug("Lightning model initialized with metrics: %s", cfg.data.metrics)

    # Set up TensorBoard logger
    logger = TensorBoardLogger("tensorboard_logs", name="AstroCLIP")

    # Ensure the directory exists
    os.makedirs(logger.log_dir, exist_ok=True)

    # Log the configuration to a file in the TensorBoard log directory
    config_log_dir = os.path.join(logger.log_dir, "config.txt")
    with open(config_log_dir, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logging.debug("Configuration logged to TensorBoard directory: %s", config_log_dir)

    # Set up ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,  # Save checkpoints in the TensorBoard log directory
        filename='best-checkpoint',  # Filename for the best checkpoint
        save_top_k=1,  # Save only the best model
        monitor='val_loss',  # Monitor validation loss
        mode='min'  # Save the model with the minimum validation loss
    )

    # Set up PyTorch Lightning Trainer with settings from training.yaml
    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        logger=logger,
        devices=cfg.training.devices,  # Specify the number of GPUs to use
        enable_progress_bar=True, 
        enable_model_summary=True,
        callbacks=[checkpoint_callback]  # Add the checkpoint callback
    )
    logging.debug("Trainer initialized with max epochs: %d and GPUs: %s", cfg.training.max_epochs, cfg.training.devices)

    # Train the model
    trainer.fit(lightning_model, train_loader, val_loader)  # Pass both train and validation loaders
    logging.debug("Training completed.")

    # Use the existing checkpoint callback to get the best checkpoint path
    best_checkpoint_path = checkpoint_callback.best_model_path

    # Load the model from the best checkpoint
    model = AstroCLIPLightningModel.load_from_checkpoint(
        best_checkpoint_path,
        model=model,  # Pass the model
        optimizer_config=cfg.training.optimizer,  # Pass the optimizer configuration
        metrics=cfg.data.metrics  # Pass the metrics
    )

    # Initialize a new Trainer without accelerator
    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        logger=logger,
        devices=1,  # Use a single device (CPU or GPU)
        accelerator='auto',  # Automatically choose CPU or GPU
        enable_progress_bar=True, 
        enable_model_summary=True
    )

    # Run predictions
    predictions = trainer.predict(
        model=model,
        dataloaders=val_loader
    )

    # Ensure all predictions are numpy arrays and concatenate them
    concatenated_predictions = np.concatenate([pred if isinstance(pred, np.ndarray) else pred.numpy() for pred in predictions], axis=0)

    # Save the concatenated predictions
    np.save(os.path.join(logger.log_dir, 'last_epoch_predictions.npy'), concatenated_predictions)

if __name__ == "__main__":
    main()