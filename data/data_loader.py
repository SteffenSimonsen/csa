# data/src/data_loader.py
import pandas as pd
from datasets import load_dataset
from typing import Iterator, List, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split
from .data_preprocessing import clean_and_label_sample

def load_streaming_dataset(split="train", streaming=True):
    """Load the Amazon reviews dataset in streaming or download mode https://huggingface.co/datasets/gmongaras/Amazon-Reviews-2023 """
    if split != "train":
        raise ValueError("Only 'train' split available. Use create_splits() for train/val/test splits.")
    
    return load_dataset(
        "gmongaras/Amazon-Reviews-2023", 
        split=split, 
        streaming=streaming
    )


def process_batch(dataset_iterator: Iterator, batch_size: int = 1000) -> List[Dict]:
    """Process a batch of samples from streaming dataset"""
    processed_samples = []
    
    for i, sample in enumerate(dataset_iterator):
        if i >= batch_size:
            break
        cleaned = clean_and_label_sample(sample)
        processed_samples.append(cleaned)
    
    return processed_samples

def create_splits(processed_samples: List[Dict], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split processed data into train/validation/test sets"""
    df = pd.DataFrame(processed_samples)
    
    # First split: separate test set
    train_val, test = train_test_split(df, test_size=test_ratio, stratify=df['sentiment'], random_state=42)
    
    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val['sentiment'], random_state=42)
    
    return train, val, test

def save_splits(train, val, test, base_path="processed"):
    """Save train/val/test splits to separate parquet files"""
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    train.to_parquet(f"{base_path}/train.parquet", index=False)
    val.to_parquet(f"{base_path}/val.parquet", index=False) 
    test.to_parquet(f"{base_path}/test.parquet", index=False)

def save_processed_data(processed_samples: List[Dict], filepath: str):
    """Save processed samples to parquet file"""
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(processed_samples)
    df.to_parquet(filepath, index=False)

def load_processed_data(filepath: str) -> pd.DataFrame:
    """Load processed data from parquet file"""
    return pd.read_parquet(filepath)
