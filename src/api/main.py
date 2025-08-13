"""
FastAPI application for BacterioDesign platform.
Provides REST API endpoints for bacteriocin analysis and design.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import io
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from core.sequence_processor import SequenceProcessor
from core.feature_extractor import FeatureExtractor
from core.database_manager import DatabaseManager
from models.bacteriocin_classifier import BacteriocinClassifier
from models.design_generator import BacteriocinDesignGenerator
from utils.config import config

# Initialize FastAPI app
app = FastAPI(
    title="BacterioDesign API",
    description="AI-Powered Bacteriocin Discovery & Engineering Platform",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
sequence_processor = SequenceProcessor()
feature_extractor = FeatureExtractor()
database_manager = DatabaseManager()
classifier = BacteriocinClassifier(config.MODELS_DIR / "classifier")
design_generator = BacteriocinDesignGenerator(config.MODELS_DIR / "generator")

# Pydantic models
class SequenceInput(BaseModel):
    sequence: str
    sequence_id: Optional[str] = None

class SequenceAnalysisResponse(BaseModel):
    sequence_id: str
    sequence: str
    is_bacteriocin: bool
    confidence: float
    similar_bacteriocins: List[Dict]
    properties: Dict

class DesignRequest(BaseModel):
    num_sequences: int = 5
    seed_sequence: Optional[str] = None
    optimization_target: str = "activity"  # activity, stability, solubility

class DesignResponse(BaseModel):
    generated_sequences: List[Dict]
    optimization_results: List[Dict]

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    try:
        # Load pre-trained models if available
        if (config.MODELS_DIR / "classifier").exists():
            classifier.load_models()
        
        if (config.MODELS_DIR / "generator").exists():
            design_generator.load_models()
        
        # Initialize database with BAGEL4 data
        database_manager.fetch_bagel4_data(update_local=True)
        
    except Exception as e:
        print(f"Warning: Could not load pre-trained models: {e}")

@app.get("/")
async def root():
    """API health check."""
    return {
        "message": "BacterioDesign API is running",
        "version": "2.1.0",
        "status": "healthy"
    }

@app.post("/analyze/sequence", response_model=SequenceAnalysisResponse)
async def analyze_sequence(sequence_input: SequenceInput):
    """
    Analyze a single protein sequence for bacteriocin potential.
    
    Args:
        sequence_input: Input sequence data
        
    Returns:
        Analysis results including prediction and similar sequences
    """
    try:
        sequence = sequence_input.sequence.upper().strip()
        
        # Validate sequence
        if not sequence_processor.validate_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Extract features
        features = feature_extractor.extract_all_features(sequence)
        feature_df = pd.DataFrame([features])
        
        # Predict bacteriocin probability
        if classifier.is_trained:
            probability = classifier.predict_proba(feature_df)[0]
            is_bacteriocin = probability > 0.5
        else:
            # Fallback heuristic if model not trained
            is_bacteriocin = _heuristic_bacteriocin_check(sequence, features)
            probability = 0.7 if is_bacteriocin else 0.3
        
        # Find similar bacteriocins if predicted as bacteriocin
        similar_bacteriocins = []
        if is_bacteriocin or probability > 0.3:
            similar_bacteriocins = database_manager.find_similar_bacteriocins(
                sequence, threshold=0.6
            )[:5]  # Top 5 matches
        
        return SequenceAnalysisResponse(
            sequence_id=sequence_input.sequence_id or f"seq_{hash(sequence)%10000}",
            sequence=sequence,
            is_bacteriocin=is_bacteriocin,
            confidence=float(probability),
            similar_bacteriocins=similar_bacteriocins,
            properties=features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch_sequences(file: UploadFile = File(...)):
    """
    Analyze multiple sequences from FASTA file.
    
    Args:
        file: FASTA file upload
        
    Returns:
        Batch analysis results
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Create temporary file
        temp_path = config.DATA_DIR / "temp" / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        # Process FASTA file
        sequences_df = sequence_processor.batch_process_fasta(str(temp_path))
        
        if sequences_df.empty:
            raise HTTPException(status_code=400, detail="No valid sequences found in file")
        
        # Analyze each sequence
        results = []
        for _, row in sequences_df.iterrows():
            sequence_input = SequenceInput(
                sequence=row['sequence'],
                sequence_id=row['id']
            )
            
            try:
                analysis_result = await analyze_sequence(sequence_input)
                results.append(analysis_result.dict())
            except Exception as e:
                results.append({
                    'sequence_id': row['id'],
                    'error': str(e)
                })
        
        # Clean up temporary file
        temp_path.unlink(missing_ok=True)
        
        return {
            'total_sequences': len(results),
            'results': results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/design/generate", response_model=DesignResponse)
async def generate_bacteriocins(design_request: DesignRequest):
    """
    Generate novel bacteriocin sequences using neural networks.
    
    Args:
        design_request: Design parameters
        
    Returns:
        Generated sequences with optimization scores
    """
    try:
        if not design_generator.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Design generator model not available. Please train the model first."
            )
        
        # Generate sequences
        generated_sequences = design_generator.generate_novel_bacteriocins(
            num_sequences=design_request.num_sequences,
            seed_sequence=design_request.seed_sequence,
            temperature=0.8  # Moderate randomness
        )
        
        # Analyze generated sequences
        sequence_results = []
        optimization_results = []
        
        for i, seq in enumerate(generated_sequences):
            # Extract features
            features = feature_extractor.extract_all_features(seq)
            
            # Predict activity if classifier is available
            activity_score = 0.5  # Default
            if classifier.is_trained:
                feature_df = pd.DataFrame([features])
                activity_score = float(classifier.predict_proba(feature_df)[0])
            
            sequence_result = {
                'sequence_id': f"generated_{i+1}",
                'sequence': seq,
                'length': len(seq),
                'predicted_activity': activity_score,
                'molecular_weight': features.get('molecular_weight', 0),
                'isoelectric_point': features.get('isoelectric_point', 0),
                'hydrophobic_ratio': features.get('hydrophobic_ratio', 0)
            }
            
            sequence_results.append(sequence_result)
        
        # Optimize sequences for desired properties
        if design_generator.property_optimizer:
            optimization_results = design_generator.optimize_bacteriocin_properties(
                generated_sequences, feature_extractor
            )
        else:
            # Simple heuristic optimization
            for seq_result in sequence_results:
                score = (
                    seq_result['predicted_activity'] * 0.4 +
                    min(seq_result['hydrophobic_ratio'] * 2, 1.0) * 0.3 +
                    (1.0 if 20 < len(seq_result['sequence']) < 100 else 0.5) * 0.3
                )
                
                optimization_results.append({
                    'sequence': seq_result['sequence'],
                    'optimization_score': score,
                    'predicted_activity': seq_result['predicted_activity'],
                    'recommended_for_synthesis': score > 0.6
                })
        
        return DesignResponse(
            generated_sequences=sequence_results,
            optimization_results=optimization_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Design generation failed: {str(e)}")

@app.get("/database/bacteriocins")
async def get_bacteriocin_database():
    """Get bacteriocin database statistics and sample records."""
    try:
        bacteriocins_df = database_manager.get_local_bacteriocins()
        
        if bacteriocins_df.empty:
            # Try to fetch from BAGEL4
            bacteriocins_df = database_manager.fetch_bagel4_data()
        
        stats = {
            'total_records': len(bacteriocins_df),
            'unique_organisms': bacteriocins_df['organism'].nunique() if 'organism' in bacteriocins_df.columns else 0,
            'bacteriocin_classes': bacteriocins_df['bacteriocin_class'].value_counts().to_dict() if 'bacteriocin_class' in bacteriocins_df.columns else {},
            'sample_records': bacteriocins_df.head(5).to_dict('records') if not bacteriocins_df.empty else []
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.post("/similarity/search")
async def search_similar_bacteriocins(sequence_input: SequenceInput, threshold: float = 0.6):
    """
    Search for bacteriocins similar to query sequence.
    
    Args:
        sequence_input: Query sequence
        threshold: Similarity threshold (0-1)
        
    Returns:
        List of similar bacteriocins
    """
    try:
        sequence = sequence_input.sequence.upper().strip()
        
        if not sequence_processor.validate_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        similar_bacteriocins = database_manager.find_similar_bacteriocins(
            sequence, threshold=threshold
        )
        
        return {
            'query_sequence': sequence,
            'threshold': threshold,
            'total_matches': len(similar_bacteriocins),
            'similar_bacteriocins': similar_bacteriocins
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@app.get("/models/status")
async def get_model_status():
    """Get status of all trained models."""
    return {
        'classifier_trained': classifier.is_trained,
        'generator_trained': design_generator.is_trained,
        'database_records': len(database_manager.get_local_bacteriocins()),
        'api_version': "2.1.0"
    }

def _heuristic_bacteriocin_check(sequence: str, features: Dict) -> bool:
    """Heuristic bacteriocin check when ML model is not available."""
    # Simple heuristics based on known bacteriocin characteristics
    length_ok = 20 <= len(sequence) <= 150
    has_cysteines = sequence.count('C') >= 2
    has_glycines = 'GG' in sequence
    cationic = features.get('net_charge', 0) > 0
    
    return length_ok and (has_cysteines or has_glycines) and cationic

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
