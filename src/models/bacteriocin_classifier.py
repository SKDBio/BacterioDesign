"""
Machine learning models for bacteriocin identification and classification.
Implements ensemble methods combining multiple algorithms for robust prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import logging
from pathlib import Path

class BacteriocinClassifier:
    """Ensemble classifier for bacteriocin identification."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_weights = {}
        self.is_trained = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual classification models."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            )
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, cv_folds: int = 5) -> Dict:
        """
        Train ensemble of classification models.
        
        Args:
            X: Feature matrix
            y: Target labels (1 for bacteriocin, 0 for non-bacteriocin)
            test_size: Fraction of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results and metrics
        """
        self.logger.info("Starting bacteriocin classifier training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train individual models and collect results
        model_results = {}
        cv_scores = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_score = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=cv_folds, scoring='roc_auc'
            ).mean()
            cv_scores[name] = cv_score
            
            # Test predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            model_results[name] = {
                'cv_score': cv_score,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(
                    y_test, y_pred, output_dict=True
                )
            }
            
            self.logger.info(f"{name} - CV AUC: {cv_score:.3f}, Test AUC: {auc_score:.3f}")
        
        # Calculate ensemble weights based on cross-validation performance
        total_cv_score = sum(cv_scores.values())
        self.ensemble_weights = {
            name: score / total_cv_score 
            for name, score in cv_scores.items()
        }
        
        # Generate ensemble predictions
        ensemble_proba = self._ensemble_predict_proba(X_test_scaled)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        # Save models and scaler
        self._save_models()
        
        self.is_trained = True
        
        results = {
            'individual_models': model_results,
            'ensemble_auc': ensemble_auc,
            'ensemble_weights': self.ensemble_weights,
            'test_accuracy': (ensemble_pred == y_test).mean(),
            'feature_names': list(X.columns)
        }
        
        self.logger.info(f"Training complete. Ensemble AUC: {ensemble_auc:.3f}")
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict bacteriocin probability for new sequences.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self._ensemble_predict_proba(X_scaled)
        return (probabilities > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict bacteriocin probability for new sequences.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self._ensemble_predict_proba(X_scaled)
    
    def _ensemble_predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions using weighted average."""
        ensemble_proba = np.zeros(X_scaled.shape[0])
        
        for name, model in self.models.items():
            proba = model.predict_proba(X_scaled)[:, 1]
            weight = self.ensemble_weights.get(name, 1.0)
            ensemble_proba += weight * proba
        
        return ensemble_proba
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        importance_data = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data.append({
                    'model': name,
                    'importance': model.feature_importances_
                })
        
        if not importance_data:
            return pd.DataFrame()
        
        # Average importance across models
        avg_importance = np.mean([item['importance'] for item in importance_data], axis=0)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
    
    def _save_models(self):
        """Save trained models and scaler."""
        # Save individual models
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name}_model.joblib"
            joblib.dump(model, model_path)
        
        # Save scaler and ensemble weights
        joblib.dump(self.scaler, self.models_dir / "scaler.joblib")
        joblib.dump(self.ensemble_weights, self.models_dir / "ensemble_weights.joblib")
        
        self.logger.info(f"Models saved to {self.models_dir}")
    
    def load_models(self):
        """Load pre-trained models."""
        try:
            # Load individual models
            for name in self.models.keys():
                model_path = self.models_dir / f"{name}_model.joblib"
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
            
            # Load scaler and weights
            scaler_path = self.models_dir / "scaler.joblib"
            weights_path = self.models_dir / "ensemble_weights.joblib"
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            if weights_path.exists():
                self.ensemble_weights = joblib.load(weights_path)
            
            self.is_trained = True
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform hyperparameter tuning for individual models.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with best parameters for each model
        """
        self.logger.info("Starting hyperparameter tuning...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        }
        
        best_params = {}
        
        for name, model in self.models.items():
            if name in param_grids:
                self.logger.info(f"Tuning {name}...")
                
                grid_search = GridSearchCV(
                    model, param_grids[name],
                    cv=5, scoring='roc_auc',
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_scaled, y)
                best_params[name] = grid_search.best_params_
                
                # Update model with best parameters
                self.models[name] = grid_search.best_estimator_
                
                self.logger.info(f"{name} best score: {grid_search.best_score_:.3f}")
        
        return best_params
