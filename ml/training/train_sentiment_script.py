import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models import SentimentLightningModule
from training import SentimentDataModule
import torch

torch.set_float32_matmul_precision('high')

def train_sentiment_model():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    print("Setting up data module...")
    try:
        data_module = SentimentDataModule(
            data_dir="data/processed",
            batch_size=16
        )
        data_module.setup(stage="fit")
        print(f"Data loaded: {len(data_module.train_texts)} train, {len(data_module.val_texts)} val samples")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None, None
    
    print("Creating model...")
    model = SentimentLightningModule(
        learning_rate=2e-5,
        num_classes=3
    )
    
    # Create directories with proper error handling
    try:
        Path('checkpoints').mkdir(exist_ok=True, mode=0o755)
        Path('logs').mkdir(exist_ok=True, mode=0o755)
        
        # Test write permissions
        test_file = Path('checkpoints/.write_test')
        test_file.write_text('test')
        test_file.unlink()  # Delete test file
        print("Checkpoint directory permissions OK")
        
    except PermissionError as e:
        print(f"Permission denied for checkpoints directory: {e}")
        print("Run: sudo chown -R $USER:$USER checkpoints/ logs/")
        return None, None
    except Exception as e:
        print(f"Failed to create directories: {e}")
        return None, None
    
    print("Setting up callbacks...")
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
        verbose=True,
        mode='min'
    )
    
    logger = TensorBoardLogger("logs", name="sentiment_model")
    
    print("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_progress_bar=True,
        deterministic=True
    )
    
    print("Starting training...")
    try:
        trainer.fit(model, datamodule=data_module)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        return None, None
    
    print("Running test...")
    try:
        trainer.test(model, datamodule=data_module)
        print("Testing completed!")
    except Exception as e:
        print(f"Testing failed: {e}")
    
    # Save final model
    print("Saving final model...")
    try:
        final_model_path = Path('checkpoints/final_sentiment_model.pth')
        torch.save(model.model.state_dict(), final_model_path)
        print(f"Model saved to: {final_model_path}")
    except Exception as e:
        print(f"Failed to save final model: {e}")
        print("But training checkpoints should still be available!")
    
    print("\nTraining Summary:")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
    
    # List all created checkpoints
    checkpoints = list(Path('checkpoints').glob('*.ckpt'))
    print(f"Created {len(checkpoints)} checkpoint files:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")
    
    return trainer, model

if __name__ == "__main__":
    print("Customer Sentiment Analyzer - Training Script")
    print("=" * 50)
    
    # Check data first
    data_dir = Path("data/processed")
    required_files = ['train.parquet', 'val.parquet', 'test.parquet']
    
    print("Checking data files...")
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"  Found: {file_path}")
        else:
            print(f"  Missing: {file_path}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing data files: {missing_files}")
        print("Run: uv run python data/data_pipeline.py --num-samples 10000")
        sys.exit(1)
    
    trainer, model = train_sentiment_model()
    
    if trainer and model:
        print("\nAll done! Ready to deploy.")
    else:
        print("\nTraining failed. Check the errors above.")
        sys.exit(1)
