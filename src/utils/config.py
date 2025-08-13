"""
Configuration management for BacterioDesign platform.
Centralizes all settings, database connections, and model parameters.
"""

import os
from typing import Dict, Any
from pathlib import Path

class Config:
    """Central configuration class for the BacterioDesign platform."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = DATA_DIR / "models"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Database configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bacteriodesign.db")
    BAGEL4_API_URL = "https://bagel4.molgenrug.nl/api/v1/"
    
    # Machine Learning parameters
    ML_CONFIG = {
        "similarity_threshold": 0.60,
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "n_estimators": 100
    }
    
    # Neural Network parameters
    NN_CONFIG = {
        "embedding_dim": 128,
        "hidden_units": [256, 128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10
    }
    
    # Sequence analysis parameters
    SEQUENCE_CONFIG = {
        "min_length": 20,
        "max_length": 150,
        "amino_acids": "ACDEFGHIKLMNPQRSTVWY",
        "bacteriocin_classes": ["Class I", "Class II", "Class III", "Class IV"]
    }
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, 
                         cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Initialize configuration
config = Config()
config.ensure_directories()
