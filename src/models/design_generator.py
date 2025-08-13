"""
Neural network-based bacteriocin design and optimization.
Implements sequence generation, activity prediction, and optimization algorithms.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import pickle
import random
from collections import Counter

class BacteriocinDesignGenerator:
    """Neural network for designing novel bacteriocins."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Sequence parameters
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {i: aa for i, aa in enumerate(self.amino_acids)}
        self.vocab_size = len(self.amino_acids)
        
        # Model parameters
        self.max_sequence_length = 100
        self.embedding_dim = 128
        self.hidden_units = [256, 128, 64]
        self.dropout_rate = 0.3
        
        # Models
        self.sequence_generator = None
        self.activity_predictor = None
        self.property_optimizer = None
        
        self.is_trained = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def build_sequence_generator(self) -> keras.Model:
        """
        Build LSTM-based sequence generator for de novo bacteriocin design.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Embedding(
                input_dim=self.vocab_size + 1,  # +1 for padding
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                mask_zero=True
            ),
            layers.LSTM(
                self.hidden_units[0],
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.LSTM(
                self.hidden_units[1],
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.LSTM(
                self.hidden_units[2],
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ),
            layers.Dense(self.hidden_units[1], activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_activity_predictor(self, input_dim: int) -> keras.Model:
        """
        Build neural network for predicting bacteriocin activity.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Dense(self.hidden_units[0], activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(self.hidden_units[1], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(self.hidden_units[2], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(1, activation='sigmoid')  # Binary classification: active/inactive
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        return model
    
    def build_property_optimizer(self, input_dim: int) -> keras.Model:
        """
        Build neural network for optimizing bacteriocin properties.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model for multi-property prediction
        """
        # Shared base layers
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(self.hidden_units[0], activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(self.hidden_units[1], activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        shared_features = layers.Dense(self.hidden_units[2], activation='relu')(x)
        
        # Multiple output heads for different properties
        stability_output = layers.Dense(1, activation='sigmoid', name='stability')(shared_features)
        solubility_output = layers.Dense(1, activation='sigmoid', name='solubility')(shared_features)
        activity_output = layers.Dense(1, activation='sigmoid', name='activity')(shared_features)
        toxicity_output = layers.Dense(1, activation='sigmoid', name='toxicity')(shared_features)
        
        model = models.Model(
            inputs=inputs,
            outputs=[stability_output, solubility_output, activity_output, toxicity_output]
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'stability': 'binary_crossentropy',
                'solubility': 'binary_crossentropy',
                'activity': 'binary_crossentropy',
                'toxicity': 'binary_crossentropy'
            },
            loss_weights={
                'stability': 1.0,
                'solubility': 1.0,
                'activity': 2.0,  # Higher weight for antimicrobial activity
                'toxicity': 1.5   # Higher weight to avoid toxic compounds
            },
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequence_data(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for training the generator.
        
        Args:
            sequences: List of bacteriocin sequences
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        encoded_sequences = []
        
        for seq in sequences:
            if len(seq) > self.max_sequence_length:
                seq = seq[:self.max_sequence_length]
            
            # Encode sequence
            encoded = [self.aa_to_idx.get(aa, 0) for aa in seq]
            
            # Pad sequence
            while len(encoded) < self.max_sequence_length:
                encoded.append(0)  # Padding token
            
            encoded_sequences.append(encoded)
        
        encoded_sequences = np.array(encoded_sequences)
        
        # For sequence generation, input is sequence[:-1], target is sequence[1:]
        X = encoded_sequences[:, :-1]
        y = encoded_sequences[:, 1:]
        
        return X, y
    
    def train_generator(self, sequences: List[str], epochs: int = 100) -> Dict:
        """
        Train the sequence generator model.
        
        Args:
            sequences: Training sequences
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        self.logger.info("Training bacteriocin sequence generator...")
        
        # Prepare data
        X, y = self.prepare_sequence_data(sequences)
        
        # Build model
        self.sequence_generator = self.build_sequence_generator()
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            self.models_dir / "sequence_generator.h5",
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        history = self.sequence_generator.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        self.logger.info("Sequence generator training completed")
        return history.history
    
    def train_activity_predictor(self, features: pd.DataFrame, 
                               activities: pd.Series, epochs: int = 100) -> Dict:
        """
        Train the activity prediction model.
        
        Args:
            features: Sequence features
            activities: Activity labels (1 for active, 0 for inactive)
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        self.logger.info("Training activity predictor...")
        
        # Build model
        self.activity_predictor = self.build_activity_predictor(features.shape[1])
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max'
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            self.models_dir / "activity_predictor.h5",
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
        
        # Train model
        history = self.activity_predictor.fit(
            features.values, activities.values,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        self.logger.info("Activity predictor training completed")
        return history.history
    
    def generate_novel_bacteriocins(self, num_sequences: int = 10, 
                                   seed_sequence: Optional[str] = None,
                                   temperature: float = 1.0) -> List[str]:
        """
        Generate novel bacteriocin sequences.
        
        Args:
            num_sequences: Number of sequences to generate
            seed_sequence: Optional seed sequence to start generation
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            List of generated sequences
        """
        if self.sequence_generator is None:
            raise ValueError("Sequence generator must be trained first")
        
        generated_sequences = []
        
        for _ in range(num_sequences):
            if seed_sequence:
                # Start with seed sequence
                current_seq = list(seed_sequence[:20])  # Use first 20 residues as seed
            else:
                # Random start
                current_seq = [random.choice(self.amino_acids)]
            
            # Generate sequence
            for _ in range(self.max_sequence_length - len(current_seq)):
                # Encode current sequence
                encoded = [self.aa_to_idx.get(aa, 0) for aa in current_seq[-20:]]  # Last 20 residues
                while len(encoded) < 20:
                    encoded = [0] + encoded  # Pad at beginning
                
                # Predict next amino acid
                input_seq = np.array([encoded])
                predictions = self.sequence_generator.predict(input_seq, verbose=0)[0]
                
                # Apply temperature sampling
                predictions = np.log(predictions + 1e-8) / temperature
                predictions = np.exp(predictions) / np.sum(np.exp(predictions))
                
                # Sample next amino acid
                next_aa_idx = np.random.choice(len(predictions), p=predictions)
                
                if next_aa_idx < len(self.amino_acids):
                    next_aa = self.idx_to_aa[next_aa_idx]
                    current_seq.append(next_aa)
                else:
                    break  # Stop token or padding
            
            generated_sequences.append(''.join(current_seq))
        
        return generated_sequences
    
    def optimize_bacteriocin_properties(self, sequences: List[str], 
                                      features_extractor) -> List[Dict]:
        """
        Optimize bacteriocin sequences for desired properties.
        
        Args:
            sequences: Input sequences to optimize
            features_extractor: Feature extraction function
            
        Returns:
            List of optimization results
        """
        if self.property_optimizer is None:
            raise ValueError("Property optimizer must be trained first")
        
        optimization_results = []
        
        for seq in sequences:
            # Extract features
            features = features_extractor.extract_all_features(seq)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Predict properties
            predictions = self.property_optimizer.predict(feature_vector)
            stability, solubility, activity, toxicity = predictions
            
            # Calculate optimization score
            # Higher is better for stability, solubility, activity
            # Lower is better for toxicity
            optimization_score = (
                stability[0][0] * 0.25 +
                solubility[0][0] * 0.25 +
                activity[0][0] * 0.35 +
                (1 - toxicity[0][0]) * 0.15
            )
            
            result = {
                'sequence': seq,
                'stability': stability[0][0],
                'solubility': solubility[0][0],
                'activity': activity[0][0],
                'toxicity': toxicity[0][0],
                'optimization_score': optimization_score,
                'recommended_for_synthesis': optimization_score > 0.7 and toxicity[0][0] < 0.3
            }
            
            optimization_results.append(result)
        
        # Sort by optimization score
        optimization_results.sort(key=lambda x: x['optimization_score'], reverse=True)
        
        return optimization_results
    
    def save_models(self):
        """Save all trained models."""
        if self.sequence_generator:
            self.sequence_generator.save(self.models_dir / "sequence_generator.h5")
        
        if self.activity_predictor:
            self.activity_predictor.save(self.models_dir / "activity_predictor.h5")
        
        if self.property_optimizer:
            self.property_optimizer.save(self.models_dir / "property_optimizer.h5")
        
        # Save tokenizers and parameters
        model_params = {
            'aa_to_idx': self.aa_to_idx,
            'idx_to_aa': self.idx_to_aa,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length
        }
        
        with open(self.models_dir / "model_params.pkl", 'wb') as f:
            pickle.dump(model_params, f)
        
        self.logger.info(f"Models saved to {self.models_dir}")
    
    def load_models(self):
        """Load pre-trained models."""
        try:
            # Load models
            generator_path = self.models_dir / "sequence_generator.h5"
            if generator_path.exists():
                self.sequence_generator = keras.models.load_model(generator_path)
            
            predictor_path = self.models_dir / "activity_predictor.h5"
            if predictor_path.exists():
                self.activity_predictor = keras.models.load_model(predictor_path)
            
            optimizer_path = self.models_dir / "property_optimizer.h5"
            if optimizer_path.exists():
                self.property_optimizer = keras.models.load_model(optimizer_path)
            
            # Load parameters
            params_path = self.models_dir / "model_params.pkl"
            if params_path.exists():
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
                    self.aa_to_idx = params['aa_to_idx']
                    self.idx_to_aa = params['idx_to_aa']
                    self.vocab_size = params['vocab_size']
                    self.max_sequence_length = params['max_sequence_length']
            
            self.is_trained = True
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
