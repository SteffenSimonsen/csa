import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from .sentiment_model import SentimentModel

class SentimentLightningModule(pl.LightningModule):
    def __init__(self, learning_rate: float = 2e-5, num_classes: int = 3):
        super().__init__()
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Save hyperparameters to checkpoint
        self.save_hyperparameters()
        
        # Model
        self.model = SentimentModel(num_classes=num_classes)
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        
    def forward(self, texts):
        return self.model(texts)
        
    def training_step(self, batch, batch_idx):
        texts, labels = batch
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        batch_size = len(labels)
        
        # Forward pass
        logits = self(texts)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute predictions for metrics
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        batch_size = len(labels)
        
        # Forward pass
        logits = self(texts)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute predictions for metrics
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        texts, labels = batch
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        batch_size = len(labels)
        
        # Forward pass
        logits = self(texts)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute predictions for metrics
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        
        # Log metrics
        self.log('test_loss', loss, batch_size=batch_size)
        self.log('test_acc', self.test_accuracy, batch_size=batch_size)
        self.log('test_f1', self.test_f1, batch_size=batch_size)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        # Optional: Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }