# ml/__init__.py
from .models import SentimentModel
from .inference import SentimentPredictor

# Only import training modules if available (for local dev)
try:
    from .models import SentimentLightningModule
    __all__ = ['SentimentModel', 'SentimentLightningModule', 'SentimentPredictor']
except ImportError:
    # pytorch_lightning not installed (production API)
    __all__ = ['SentimentModel', 'SentimentPredictor']
