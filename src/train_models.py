"""
Complete training pipeline for BacterioDesign platform.
Trains all models: classifier, activity predictor, and sequence generator.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent))

from core.sequence_processor import SequenceProcessor
from core.feature_extractor import FeatureExtractor
from core.database_manager import DatabaseManager
from models.bacteriocin_classifier import BacteriocinClassifier
from models.design_generator import BacteriocinDesignGenerator
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacterioDesignTrainer:
    """Complete training pipeline for BacterioDesign platform."""
    
    def __init__(self):
        self.sequence_processor = SequenceProcessor()
        self.feature_extractor = FeatureExtractor()
        self.database_manager = DatabaseManager()
        self.classifier = BacteriocinClassifier(config.MODELS_DIR / "classifier")
        self.design_generator = BacteriocinDesignGenerator(config.MODELS_DIR / "generator")
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare comprehensive training dataset.
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Preparing training data...")
        
        # Get bacteriocin data (positive examples)
        bacteriocins_df = self.database_manager.fetch_bagel4_data()
        
        if bacteriocins_df.empty:
            raise ValueError("No bacteriocin data available. Check BAGEL4 connection.")
        
        # Filter valid sequences
        valid_bacteriocins = []
        for _, row in bacteriocins_df.iterrows():
            if pd.notna(row['sequence']) and self.sequence_processor.validate_sequence(row['sequence']):
                valid_bacteriocins.append(row['sequence'])
        
        logger.info(f"Found {len(valid_bacteriocins)} valid bacteriocin sequences")
        
        # Generate negative examples (non-bacteriocin proteins)
        # In a real implementation, you would use a curated negative dataset
        # Here we create synthetic negative examples for demonstration
        negative_sequences = self._generate_negative_examples(len(valid_bacteriocins))
        
        # Combine datasets
        all_sequences = valid_bacteriocins + negative_sequences
        all_labels = [1] * len(valid_bacteriocins) + [0] * len(negative_sequences)
        
        # Extract features
        logger.info("Extracting features...")
        feature_list = []
        valid_sequences = []
        valid_labels = []
        
        for seq, label in zip(all_sequences, all_labels):
            try:
                features = self.feature_extractor.extract_all_features(seq)
                feature_list.append(features)
                valid_sequences.append(seq)
                valid_labels.append(label)
            except Exception as e:
                logger.warning(f"Could not extract features for sequence: {e}")
                continue
        
        features_df = pd.DataFrame(feature_list).fillna(0)
        labels_series = pd.Series(valid_labels)
        
        logger.info(f"Prepared {len(features_df)} training examples")
        logger.info(f"Positive examples: {sum(labels_series)}")
        logger.info(f"Negative examples: {len(labels_series) - sum(labels_series)}")
        
        return features_df, labels_series, valid_sequences
    
    def _generate_negative_examples(self, num_examples: int) -> list:
        """Generate synthetic negative examples (non-bacteriocin proteins)."""
        # This is a simplified approach - in practice, use real protein sequences
        # from non-bacteriocin sources (e.g., house-keeping proteins)
        
        import random
        
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        negative_sequences = []
        
        for _ in range(num_examples):
            # Generate random protein-like sequence
            length = random.randint(50, 200)  # Longer than typical bacteriocins
            
            # Create sequence with realistic amino acid composition
            sequence = ""
            for _ in range(length):
                # Weight amino acids based on natural abundance
                aa_weights = {
                    'L': 0.096, 'A': 0.089, 'G': 0.074, 'S': 0.071, 'V': 0.066,
                    'E': 0.063, 'T': 0.059, 'K': 0.053, 'I': 0.051, 'D': 0.048,
                    'R': 0.047, 'P': 0.046, 'N': 0.044, 'Q': 0.041, 'F': 0.040,
                    'Y': 0.033, 'H': 0.024, 'C': 0.014, 'M': 0.022, 'W': 0.012
                }
                
                sequence += random.choices(
                    list(aa_weights.keys()),
                    weights=list(aa_weights.values())
                )[0]
            
            negative_sequences.append(sequence)
        
        return negative_sequences
    
    def train_classification_models(self, features_df: pd.DataFrame, 
                                  labels_series: pd.Series) -> dict:
        """Train bacteriocin classification models."""
        logger.info("Training classification models...")
        
        # Hyperparameter tuning
        best_params = self.classifier.hyperparameter_tuning(features_df, labels_series)
        logger.info(f"Best parameters found: {best_params}")
        
        # Train ensemble classifier
        training_results = self.classifier.train(
            features_df, labels_series, 
            test_size=0.2, cv_folds=5
        )
        
        # Display results
        logger.info(f"Ensemble AUC: {training_results['ensemble_auc']:.3f}")
        logger.info(f"Test Accuracy: {training_results['test_accuracy']:.3f}")
        
        return training_results
    
    def train_design_models(self, sequences: list) -> dict:
        """Train neural network models for bacteriocin design."""
        logger.info("Training design models...")
        
        # Filter sequences for generator training (bacteriocins only)
        bacteriocin_sequences = [seq for seq in sequences if len(seq) <= 100]
        
        if len(bacteriocin_sequences) < 50:
            logger.warning("Insufficient bacteriocin sequences for reliable generator training")
            return {}
        
        # Train sequence generator
        generator_history = self.design_generator.train_generator(
            bacteriocin_sequences, epochs=50
        )
        
        # Prepare activity prediction data
        # (This would use experimental activity data in a real implementation)
        activity_features = []
        activity_labels = []
        
        for seq in bacteriocin_sequences:
            try:
                features = self.feature_extractor.extract_all_features(seq)
                activity_features.append(list(features.values()))
                # Simulate activity label (in practice, use experimental data)
                activity_labels.append(1)  # Assume all bacteriocins are active
            except:
                continue
        
        if len(activity_features) > 20:
            activity_df = pd.DataFrame(activity_features)
            activity_series = pd.Series(activity_labels)
            
            predictor_history = self.design_generator.train_activity_predictor(
                activity_df, activity_series, epochs=50
            )
        else:
            logger.warning("Insufficient data for activity predictor training")
            predictor_history = {}
        
        return {
            'generator_history': generator_history,
            'predictor_history': predictor_history
        }
    
    def evaluate_models(self, features_df: pd.DataFrame, 
                       labels_series: pd.Series) -> dict:
        """Evaluate trained models."""
        logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        # Test classifier
        if self.classifier.is_trained:
            test_predictions = self.classifier.predict_proba(features_df)
            test_accuracy = ((test_predictions > 0.5) == labels_series).mean()
            evaluation_results['classifier_accuracy'] = test_accuracy
            
            # Feature importance
            feature_importance = self.classifier.get_feature_importance()
            evaluation_results['top_features'] = feature_importance.head(10).to_dict('records')
        
        # Test generator
        if self.design_generator.sequence_generator:
            # Generate test sequences
            test_sequences = self.design_generator.generate_novel_bacteriocins(5)
            evaluation_results['sample_generated_sequences'] = test_sequences
        
        return evaluation_results
    
    def run_complete_training(self) -> dict:
        """Run complete training pipeline."""
        logger.info("Starting complete BacterioDesign training pipeline...")
        
        try:
            # Prepare data
            features_df, labels_series, sequences = self.prepare_training_data()
            
            # Train classification models
            classification_results = self.train_classification_models(features_df, labels_series)
            
            # Train design models
            design_results = self.train_design_models(sequences)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(features_df, labels_series)
            
            # Save all models
            self.classifier.save_models()
            self.design_generator.save_models()
            
            # Compile final results
            final_results = {
                'classification_results': classification_results,
                'design_results': design_results,
                'evaluation_results': evaluation_results,
                'training_data_size': len(features_df),
                'status': 'success'
            }
            
            logger.info("Training pipeline completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {'status': 'failed', 'error': str(e)}

def main():
    """Main training script."""
    trainer = BacterioDesignTrainer()
    results = trainer.run_complete_training()
    
    if results['status'] == 'success':
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Training data size: {results['training_data_size']}")
        print(f"Ensemble AUC: {results['classification_results']['ensemble_auc']:.3f}")
        print(f"Test Accuracy: {results['classification_results']['test_accuracy']:.3f}")
        print("\nTop 5 Important Features:")
        for feature in results['evaluation_results']['top_features'][:5]:
            print(f"  {feature['feature']}: {feature['importance']:.3f}")
        print("\nSample Generated Sequences:")
        for i, seq in enumerate(results['evaluation_results']['sample_generated_sequences'][:3]):
            print(f"  {i+1}: {seq}")
    else:
        print(f"Training failed: {results['error']}")

if __name__ == "__main__":
    main()
