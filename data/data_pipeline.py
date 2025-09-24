import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import save_processed_data, create_splits, save_splits
from data.data_preprocessing import clean_and_label_sample

def download_and_process_dataset(
    num_samples=None,
    output_dir="data/processed",
    raw_dir="data/raw"
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset in streaming mode...")
    
    dataset = load_dataset(
        "gmongaras/Amazon-Reviews-2023",
        split="train",
        streaming=True
    )
    
    processed_samples = []
    batch_size = 10000
    
    if num_samples:
        print(f"Processing up to {num_samples} samples...")
        pbar_total = num_samples
    else:
        print(f"Processing all available samples...")
        pbar_total = None
    
    iterator = iter(dataset)
    
    # Enhanced progress bar with dynamic description
    pbar = tqdm(total=pbar_total, 
                desc="Downloading & Processing",
                unit=" samples",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    while True:
        if num_samples and len(processed_samples) >= num_samples:
            break
            
        chunk = []
        for i in range(batch_size):
            if num_samples and len(processed_samples) + len(chunk) >= num_samples:
                break
            try:
                sample = next(iterator)
                cleaned = clean_and_label_sample(sample)
                chunk.append(cleaned)
            except StopIteration:
                break
        
        if not chunk:
            break
            
        processed_samples.extend(chunk)
        pbar.update(len(chunk))
        
        # Update description with current count
        if num_samples is None:
            pbar.set_description(f"Processed {len(processed_samples):,} samples")
        
        if len(processed_samples) % 50000 == 0:
            pbar.write(f"\n💾 Checkpoint saved at {len(processed_samples):,} samples")
            save_processed_data(processed_samples, f"{output_dir}/checkpoint.parquet")
    
    pbar.close()
    
    print(f"\nProcessed {len(processed_samples)} samples")
    
    print("Creating train/val/test splits...")
    train, val, test = create_splits(processed_samples)
    
    print(f"Train: {len(train)} samples")
    print(f"Val: {len(val)} samples")
    print(f"Test: {len(test)} samples")
    
    print(f"\nSaving splits to {output_dir}...")
    save_splits(train, val, test, base_path=output_dir)
    
    print("\n✅ Dataset pipeline complete!")
    print(f"Files saved in: {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process Amazon reviews dataset")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all available samples)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    download_and_process_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
