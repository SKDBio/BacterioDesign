"""
Advanced feature extraction for bacteriocin sequences.
Computes physicochemical properties, composition features, and structural indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import re

class FeatureExtractor:
    """Extract comprehensive features from bacteriocin sequences."""
    
    def __init__(self):
        self.amino_acid_properties = {
            'A': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0},
            'C': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0},
            'D': {'hydrophobic': 0, 'charge': -1, 'polar': 1, 'aromatic': 0},
            'E': {'hydrophobic': 0, 'charge': -1, 'polar': 1, 'aromatic': 0},
            'F': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 1},
            'G': {'hydrophobic': 0, 'charge': 0, 'polar': 0, 'aromatic': 0},
            'H': {'hydrophobic': 0, 'charge': 1, 'polar': 1, 'aromatic': 1},
            'I': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0},
            'K': {'hydrophobic': 0, 'charge': 1, 'polar': 1, 'aromatic': 0},
            'L': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0},
            'M': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0},
            'N': {'hydrophobic': 0, 'charge': 0, 'polar': 1, 'aromatic': 0},
            'P': {'hydrophobic': 0, 'charge': 0, 'polar': 0, 'aromatic': 0},
            'Q': {'hydrophobic': 0, 'charge': 0, 'polar': 1, 'aromatic': 0},
            'R': {'hydrophobic': 0, 'charge': 1, 'polar': 1, 'aromatic': 0},
            'S': {'hydrophobic': 0, 'charge': 0, 'polar': 1, 'aromatic': 0},
            'T': {'hydrophobic': 0, 'charge': 0, 'polar': 1, 'aromatic': 0},
            'V': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 0},
            'W': {'hydrophobic': 1, 'charge': 0, 'polar': 0, 'aromatic': 1},
            'Y': {'hydrophobic': 1, 'charge': 0, 'polar': 1, 'aromatic': 1}
        }
    
    def extract_all_features(self, sequence: str) -> Dict:
        """
        Extract comprehensive feature set from protein sequence.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic sequence properties
        features.update(self._extract_basic_properties(sequence))
        
        # Composition features
        features.update(self._extract_composition_features(sequence))
        
        # Physicochemical properties
        features.update(self._extract_physicochemical_features(sequence))
        
        # Structural features
        features.update(self._extract_structural_features(sequence))
        
        # Motif features
        features.update(self._extract_motif_features(sequence))
        
        return features
    
    def _extract_basic_properties(self, sequence: str) -> Dict:
        """Extract basic sequence properties."""
        return {
            'length': len(sequence),
            'molecular_weight': self._safe_calculate(lambda: ProteinAnalysis(sequence).molecular_weight()),
            'isoelectric_point': self._safe_calculate(lambda: ProteinAnalysis(sequence).isoelectric_point()),
            'instability_index': self._safe_calculate(lambda: ProteinAnalysis(sequence).instability_index())
        }
    
    def _extract_composition_features(self, sequence: str) -> Dict:
        """Extract amino acid composition features."""
        composition = Counter(sequence)
        total_length = len(sequence)
        
        features = {}
        
        # Individual amino acid frequencies
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            features[f'freq_{aa}'] = composition.get(aa, 0) / total_length
        
        # Property-based compositions
        hydrophobic_count = sum(1 for aa in sequence if self.amino_acid_properties[aa]['hydrophobic'])
        polar_count = sum(1 for aa in sequence if self.amino_acid_properties[aa]['polar'])
        charged_count = sum(1 for aa in sequence if abs(self.amino_acid_properties[aa]['charge']) > 0)
        aromatic_count = sum(1 for aa in sequence if self.amino_acid_properties[aa]['aromatic'])
        
        features.update({
            'hydrophobic_ratio': hydrophobic_count / total_length,
            'polar_ratio': polar_count / total_length,
            'charged_ratio': charged_count / total_length,
            'aromatic_ratio': aromatic_count / total_length
        })
        
        return features
    
    def _extract_physicochemical_features(self, sequence: str) -> Dict:
        """Extract physicochemical properties."""
        try:
            analysis = ProteinAnalysis(sequence)
            secondary_structure = analysis.secondary_structure_fraction()
            
            return {
                'helix_fraction': secondary_structure[0],
                'turn_fraction': secondary_structure[1],
                'sheet_fraction': secondary_structure[2],
                'gravy': analysis.gravy(),  # Grand average of hydropathy
                'aromaticity': analysis.aromaticity(),
                'flexibility': np.mean(analysis.flexibility()) if analysis.flexibility() else 0
            }
        except:
            return {
                'helix_fraction': 0, 'turn_fraction': 0, 'sheet_fraction': 0,
                'gravy': 0, 'aromaticity': 0, 'flexibility': 0
            }
    
    def _extract_structural_features(self, sequence: str) -> Dict:
        """Extract structural indicators."""
        features = {}
        
        # Cysteine content (important for disulfide bonds)
        cys_count = sequence.count('C')
        features['cysteine_count'] = cys_count
        features['cysteine_ratio'] = cys_count / len(sequence)
        
        # Proline content (affects flexibility)
        pro_count = sequence.count('P')
        features['proline_ratio'] = pro_count / len(sequence)
        
        # Glycine content (increases flexibility)
        gly_count = sequence.count('G')
        features['glycine_ratio'] = gly_count / len(sequence)
        
        # Calculate net charge
        positive_charge = sum(1 for aa in sequence if self.amino_acid_properties[aa]['charge'] > 0)
        negative_charge = sum(1 for aa in sequence if self.amino_acid_properties[aa]['charge'] < 0)
        features['net_charge'] = positive_charge - negative_charge
        features['charge_density'] = features['net_charge'] / len(sequence)
        
        return features
    
    def _extract_motif_features(self, sequence: str) -> Dict:
        """Extract bacteriocin-specific motif features."""
        features = {}
        
        # Common bacteriocin motifs
        motifs = {
            'double_glycine': r'GG',
            'cysteine_pattern': r'C.{2,10}C',
            'ygngv_motif': r'YGNGV',
            'lysine_rich': r'K.{0,2}K',
            'leucine_zipper': r'L.{6}L.{6}L'
        }
        
        for motif_name, pattern in motifs.items():
            matches = re.findall(pattern, sequence)
            features[f'{motif_name}_count'] = len(matches)
            features[f'{motif_name}_present'] = int(len(matches) > 0)
        
        return features
    
    def _safe_calculate(self, calculation_func):
        """Safely perform calculation with error handling."""
        try:
            return calculation_func()
        except:
            return 0.0
    
    def create_feature_matrix(self, sequences: List[str]) -> pd.DataFrame:
        """
        Create feature matrix from list of sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            DataFrame with features as columns, sequences as rows
        """
        feature_list = []
        
        for sequence in sequences:
            features = self.extract_all_features(sequence)
            feature_list.append(features)
        
        return pd.DataFrame(feature_list).fillna(0)
