"""
Comprehensive tests for bacteriocin classification system.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.sequence_processor import SequenceProcessor
from core.feature_extractor import FeatureExtractor
from models.bacteriocin_classifier import BacteriocinClassifier

class TestBacteriocinClassifier:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        # Sample bacteriocin sequences
        positive_sequences = [
            "MKKIEKFEEQSKKLKNIWSSLCTPGCKTGALQTCFLQTLTCNCKISK",
            "MAGIKKGGKWIKELVKKCSSPGCITGVLQACFNGSNNCKISK",
            "MREINNQIFSQIDQAQANINNIAGILAERSDQKSAETLRRWRGQGAK"
        ]
        
        # Sample non-bacteriocin sequences  
        negative_sequences = [
            "MAQTFRKKMRQLLKDLDAKAAEELAVFQKSQEAAATLAQEIEQLERRKAQLEQEAQLAKERQAEQLLQAQAAA",
            "MKVALVLSCLVLLVALTGCEGCGSCSSTSPTCPPSNPYCNKGTYSGRCPDWDRPGSRPGSCG",
            "MLHLVLSALLLASAAARAADIPEAFLKDVKKLYPGVPVKPKDAKKAEALAAVFQQMQEAALSLKEV"
        ]
        
        # Create feature extractor
        feature_extractor = FeatureExtractor()
        
        # Extract features
        all_sequences = positive_sequences + negative_sequences
        features_list = []
        
        for seq in all_sequences:
            features = feature_extractor.extract_all_features(seq)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list).fillna(0)
        labels = [1] * len(positive_sequences) + [0] * len(negative_sequences)
        
        return features_df, pd.Series(labels)
    
    def test_model_initialization(self, tmp_path):
        """Test classifier initialization."""
        classifier = BacteriocinClassifier(tmp_path)
        assert classifier.models_dir == tmp_path
        assert len(classifier.models) == 4  # RF, GB, SVM, LR
        assert not classifier.is_trained
    
    def test_training(self, sample_data, tmp_path):
        """Test model training."""
        features_df, labels = sample_data
        classifier = BacteriocinClassifier(tmp_path)
        
        results = classifier.train(features_df, labels, test_size=0.3, cv_folds=3)
        
        assert 'ensemble_auc' in results
        assert 'test_accuracy' in results
        assert results['ensemble_auc'] > 0
        assert classifier.is_trained
    
    def test_prediction(self, sample_data, tmp_path):
        """Test model prediction."""
        features_df, labels = sample_data
        classifier = BacteriocinClassifier(tmp_path)
        
        # Train model
        classifier.train(features_df, labels)
        
        # Test predictions
        predictions = classifier.predict(features_df)
        probabilities = classifier.predict_proba(features_df)
        
        assert len(predictions) == len(features_df)
        assert len(probabilities) == len(features_df)
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_model_saving_loading(self, sample_data, tmp_path):
        """Test model persistence."""
        features_df, labels = sample_data
        classifier = BacteriocinClassifier(tmp_path)
        
        # Train and save
        classifier.train(features_df, labels)
        classifier._save_models()
        
        # Create new classifier and load
        new_classifier = BacteriocinClassifier(tmp_path)
        new_classifier.load_models()
        
        assert new_classifier.is_trained
        
        # Test predictions are consistent
        pred1 = classifier.predict_proba(features_df)
        pred2 = new_classifier.predict_proba(features_df)
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

if __name__ == "__main__":
    pytest.main([__file__])
