import pytest
import torch
import tempfile
from pathlib import Path
import pandas as pd

from data.data_loader import create_splits
from ml.training.lightning_datamodule import SentimentDataModule
from ml.models.sentiment_lightning_module import SentimentLightningModule
from ml.inference.predictor import SentimentPredictor

class TestPipelineIntegration:
    
    @pytest.fixture
    def sample_data(self):
        """Create minimal sample data for testing"""
        data = {
            'text': [
                'This product is great!',
                'Terrible quality, very disappointed',
                'It is okay, nothing special',
                'Amazing! Best purchase ever!',
                'Not worth the money'
            ],
            'title': ['Great', 'Bad', 'Okay', 'Amazing', 'Poor'],
            'rating': [5.0, 1.0, 3.0, 5.0, 2.0],
            'sentiment': [2, 0, 1, 2, 0],
            'asin': ['B001', 'B002', 'B003', 'B004', 'B005'],
            'parent_asin': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'verified_purchase': [True, False, True, True, False],
            'helpful_vote': [10, 2, 5, 15, 1]
        }
        return pd.DataFrame(data)
    
    def test_full_pipeline_single_step(self, sample_data, tmp_path):
        """Test: Data loading → Training step → Model save → Prediction"""
        
        # 1. Create train/val/test splits
        train_data = sample_data.iloc[:3]
        val_data = sample_data.iloc[3:4]
        test_data = sample_data.iloc[4:5]
        
        # 2. Save to temporary parquet files
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        
        train_data.to_parquet(data_dir / "train.parquet")
        val_data.to_parquet(data_dir / "val.parquet")
        test_data.to_parquet(data_dir / "test.parquet")
        
        # 3. Create DataModule
        dm = SentimentDataModule(data_dir=str(data_dir), batch_size=2)
        dm.setup(stage="fit")
        
        # Verify data loaded correctly
        assert dm.train_texts is not None
        assert dm.val_texts is not None
        assert len(dm.train_texts) == 3
        assert len(dm.val_texts) == 1
        
        # 4. Create model
        model = SentimentLightningModule(learning_rate=2e-5, num_classes=3)
        
        # 5. Simulate one training step
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        model.train()
        loss = model.training_step(batch, batch_idx=0)
        
        # Verify training step worked
        assert loss is not None
        assert loss.item() > 0  # Loss should be positive
        
        # 6. Save model state
        model_path = tmp_path / "test_model.pth"
        torch.save(model.model.state_dict(), model_path)
        
        # Verify model saved
        assert model_path.exists()
        
        # 7. Load model and make prediction
        predictor = SentimentPredictor(str(model_path))
        result = predictor.predict("This is a test review")
        
        # Verify prediction works
        assert 'prediction' in result
        assert result['prediction']['label'] in ['negative', 'neutral', 'positive']
        
        print(f"✅ Full pipeline test passed!")
        print(f"   - Loaded {len(dm.train_texts)} training samples")
        print(f"   - Training loss: {loss.item():.4f}")
        print(f"   - Prediction: {result['prediction']['label']} (confidence: {result['prediction']['confidence']:.2f})")
    
    def test_dataloader_batch_format(self, sample_data, tmp_path):
        """Test DataLoader returns correct batch format"""
        
        # Save minimal data
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        sample_data.to_parquet(data_dir / "train.parquet")
        sample_data.to_parquet(data_dir / "val.parquet")
        
        # Create DataModule
        dm = SentimentDataModule(data_dir=str(data_dir), batch_size=2)
        dm.setup(stage="fit")
        
        # Get a batch
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        texts, labels = batch
        
        # Verify batch structure
        assert isinstance(texts, list)
        assert isinstance(labels, list)
        assert len(texts) == len(labels)
        assert len(texts) <= 2  # batch_size
        assert all(isinstance(t, str) for t in texts)
        assert all(isinstance(l, int) for l in labels)
        assert all(0 <= l <= 2 for l in labels)
    
    def test_model_forward_backward_cycle(self):
        """Test model can do forward and backward pass"""
        
        model = SentimentLightningModule()
        
        # Create dummy batch
        texts = ["Great product", "Bad quality"]
        labels = torch.tensor([2, 0])
        
        # Forward pass
        model.train()
        logits = model(texts)
        
        # Verify logits shape
        assert logits.shape == (2, 3)  # (batch_size, num_classes)
        
        # Backward pass
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        
        # Verify gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients
        
        # Verify loss is reasonable
        assert 0 < loss.item() < 10  # Loss should be in reasonable range