import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

class SentimentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/processed", batch_size: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Will be populated in setup()
        self.train_texts = None
        self.train_labels = None
        self.val_texts = None  
        self.val_labels = None
        self.test_texts = None
        self.test_labels = None
        
    def setup(self, stage: Optional[str] = None):
        # Load the parquet files for train, val, test
        if stage == "fit" or stage is None:
            # Load training data
            train_df = pd.read_parquet(f"{self.data_dir}/train.parquet")
            self.train_texts = train_df['text'].tolist()
            self.train_labels = train_df['sentiment'].tolist()
            
            # Load validation data  
            val_df = pd.read_parquet(f"{self.data_dir}/val.parquet")
            self.val_texts = val_df['text'].tolist()
            self.val_labels = val_df['sentiment'].tolist()
            
        if stage == "test" or stage is None:
            # Load test data
            test_df = pd.read_parquet(f"{self.data_dir}/test.parquet")
            self.test_texts = test_df['text'].tolist()
            self.test_labels = test_df['sentiment'].tolist()
            
    def collate_fn(self, batch):
        # batch is a list of (text, label) tuples
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch] 
        return texts, labels
        
    def train_dataloader(self):
        # Combine texts and labels into tuples
        dataset = list(zip(self.train_texts, self.train_labels))
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4,  
            persistent_workers=True
        )
        
    def val_dataloader(self):
        dataset = list(zip(self.val_texts, self.val_labels))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False, 
            collate_fn=self.collate_fn,
            num_workers=4
        )
        
    def test_dataloader(self):
        dataset = list(zip(self.test_texts, self.test_labels))  
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4
        )
