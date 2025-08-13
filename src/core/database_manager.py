"""
Database management system for bacteriocin data.
Handles BAGEL4 API integration, local database operations, and similarity analysis.
"""

import requests
import pandas as pd
import sqlite3
from typing import List, Dict, Optional, Tuple
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from Bio import Align
from Bio.Align import PairwiseAligner
import logging
from datetime import datetime

Base = declarative_base()

class BacteriocinRecord(Base):
    """SQLAlchemy model for bacteriocin records."""
    __tablename__ = 'bacteriocins'
    
    id = Column(Integer, primary_key=True)
    accession = Column(String(50), unique=True)
    name = Column(String(200))
    sequence = Column(Text)
    organism = Column(String(200))
    bacteriocin_class = Column(String(50))
    activity = Column(Text)
    molecular_weight = Column(Float)
    isoelectric_point = Column(Float)
    length = Column(Integer)
    bagel4_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DatabaseManager:
    """Comprehensive database management for bacteriocin analysis."""
    
    def __init__(self, database_url: str = "sqlite:///./bacteriodesign.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Configure pairwise aligner for similarity analysis
        self.aligner = PairwiseAligner()
        self.aligner.match_score = 2
        self.aligner.mismatch_score = -1
        self.aligner.open_gap_score = -2
        self.aligner.extend_gap_score = -0.5
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fetch_bagel4_data(self, update_local: bool = True) -> pd.DataFrame:
        """
        Fetch bacteriocin data from BAGEL4 database.
        
        Args:
            update_local: Whether to update local database
            
        Returns:
            DataFrame with BAGEL4 bacteriocin data
        """
        try:
            # BAGEL4 API endpoint for bacteriocin data
            url = "https://bagel4.molgenrug.nl/api/bacteriocins"
            
            self.logger.info("Fetching data from BAGEL4 database...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse the JSON response
            bacteriocins = []
            for record in data.get('bacteriocins', []):
                bacteriocins.append({
                    'bagel4_id': record.get('id'),
                    'name': record.get('name'),
                    'sequence': record.get('sequence'),
                    'organism': record.get('organism'),
                    'bacteriocin_class': record.get('class'),
                    'activity': record.get('activity'),
                    'accession': record.get('accession')
                })
            
            df = pd.DataFrame(bacteriocins)
            
            if update_local and not df.empty:
                self._update_local_database(df)
            
            self.logger.info(f"Successfully fetched {len(df)} bacteriocin records from BAGEL4")
            return df
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching BAGEL4 data: {e}")
            # Fallback to local data
            return self.get_local_bacteriocins()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return pd.DataFrame()
    
    def _update_local_database(self, df: pd.DataFrame):
        """Update local database with BAGEL4 data."""
        try:
            for _, row in df.iterrows():
                # Check if record exists
                existing = self.session.query(BacteriocinRecord).filter_by(
                    bagel4_id=row.get('bagel4_id')
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in row.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new record
                    record = BacteriocinRecord(**row.to_dict())
                    self.session.add(record)
            
            self.session.commit()
            self.logger.info("Local database updated successfully")
            
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error updating local database: {e}")
    
    def get_local_bacteriocins(self) -> pd.DataFrame:
        """Retrieve bacteriocin data from local database."""
        try:
            query = self.session.query(BacteriocinRecord)
            df = pd.read_sql(query.statement, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving local data: {e}")
            return pd.DataFrame()
    
    def find_similar_bacteriocins(self, query_sequence: str, 
                                threshold: float = 0.60) -> List[Dict]:
        """
        Find bacteriocins similar to query sequence.
        
        Args:
            query_sequence: Query protein sequence
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of similar bacteriocin records with similarity scores
        """
        # Get reference bacteriocins
        ref_bacteriocins = self.get_local_bacteriocins()
        
        if ref_bacteriocins.empty:
            self.logger.warning("No reference bacteriocins found. Fetching from BAGEL4...")
            ref_bacteriocins = self.fetch_bagel4_data()
        
        similar_bacteriocins = []
        
        for _, record in ref_bacteriocins.iterrows():
            if pd.isna(record['sequence']) or not record['sequence']:
                continue
                
            # Calculate sequence similarity
            similarity = self._calculate_similarity(query_sequence, record['sequence'])
            
            if similarity >= threshold:
                similar_record = record.to_dict()
                similar_record['similarity_score'] = similarity
                similar_record['alignment_score'] = self._get_alignment_score(
                    query_sequence, record['sequence']
                )
                similar_bacteriocins.append(similar_record)
        
        # Sort by similarity score
        similar_bacteriocins.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        self.logger.info(f"Found {len(similar_bacteriocins)} similar bacteriocins "
                        f"above {threshold:.1%} similarity threshold")
        
        return similar_bacteriocins
    
    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate sequence similarity using global alignment.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Similarity score (0-1)
        """
        try:
            alignments = self.aligner.align(seq1, seq2)
            if not alignments:
                return 0.0
            
            best_alignment = alignments[0]
            
            # Calculate percentage identity
            matches = 0
            total_positions = 0
            
            for i in range(len(best_alignment.aligned[0])):
                start1, end1 = best_alignment.aligned[0][i]
                start2, end2 = best_alignment.aligned[1][i]
                
                for j in range(end1 - start1):
                    if seq1[start1 + j] == seq2[start2 + j]:
                        matches += 1
                    total_positions += 1
            
            return matches / total_positions if total_positions > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _get_alignment_score(self, seq1: str, seq2: str) -> float:
        """Get alignment score between two sequences."""
        try:
            alignments = self.aligner.align(seq1, seq2)
            return alignments[0].score if alignments else 0.0
        except:
            return 0.0
    
    def store_prediction_results(self, results: List[Dict]):
        """Store prediction results in database."""
        # Implementation for storing ML prediction results
        pass
    
    def get_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data for machine learning models.
        
        Returns:
            Tuple of (positive_examples, negative_examples)
        """
        bacteriocins = self.get_local_bacteriocins()
        
        if bacteriocins.empty:
            bacteriocins = self.fetch_bagel4_data()
        
        # Positive examples (known bacteriocins)
        positive_data = bacteriocins[bacteriocins['sequence'].notna()].copy()
        positive_data['label'] = 1
        
        # For negative examples, you would need non-bacteriocin proteins
        # This is a simplified approach - in practice, you'd want a curated negative dataset
        negative_data = pd.DataFrame()  # Placeholder
        
        return positive_data, negative_data
