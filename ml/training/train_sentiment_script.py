# we need to do this to import from src
# when this script is run directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models import SentimentLightningModule
from training import SentimentDataModule
import torch

def train_sentiment_model():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Data module
    data_module = SentimentDataModule(
        data_dir="../data/processed",
        batch_size=16
    )
    
    # Model 
    model = SentimentLightningModule(
        learning_rate=2e-5,
        num_classes=3
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='sentiment-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    
    # Logger
    logger = TensorBoardLogger("logs", name="sentiment_model")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='auto',  # Uses GPU if available
        devices='auto',
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    
    # Test the model
    print("Running test...")
    trainer.test(model, datamodule=data_module)
    
    # Save final model
    print("Saving final model...")
    torch.save(model.model.state_dict(), 'checkpoints/final_sentiment_model.pth')
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Final model saved to: checkpoints/final_sentiment_model.pth")
    
    return trainer, model

if __name__ == "__main__":
    trainer, model = train_sentiment_model()