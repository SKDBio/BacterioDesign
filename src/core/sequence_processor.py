"""
Comprehensive sequence processing module for bacteriocin analysis.
Handles sequence validation, cleaning, and basic bioinformatics operations.
"""

from typing import List, Dict, Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import re

class SequenceProcessor:
    """Advanced sequence processing for bacteriocin analysis."""
    
    def __init__(self):
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.bacteriocin_motifs = {
            "class_i": [r"GG.{10,20}C.*C", r"GDPX{3,5}"],
            "class_ii": [r"YGNGV", r"KXXXW", r"CXXXC"],
            "class_iii": [r"[ST]X{2}[ST]", r"GXXXG"]
        }
    
    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate protein sequence for bacteriocin analysis.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not sequence or len(sequence) < 20:
            return False
            
        # Check for valid amino acids only
        if not all(aa in self.amino_acids for aa in sequence.upper()):
            return False
            
        # Check sequence length constraints
        if len(sequence) > 150:
            return False
            
        return True
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean and standardize protein sequence.
        
        Args:
            sequence: Raw protein sequence
            
        Returns:
            str: Cleaned sequence
        """
        # Remove whitespace and convert to uppercase
        cleaned = re.sub(r'\s+', '', sequence.upper())
        
        # Remove invalid characters
        cleaned = ''.join(aa for aa in cleaned if aa in self.amino_acids)
        
        return cleaned
    
    def extract_orfs(self, dna_sequence: str, min_length: int = 60) -> List[str]:
        """
        Extract open reading frames from DNA sequence.
        
        Args:
            dna_sequence: DNA sequence string
            min_length: Minimum ORF length in nucleotides
            
        Returns:
            List of protein sequences from ORFs
        """
        seq = Seq(dna_sequence)
        orfs = []
        
        # Check all three reading frames
        for frame in range(3):
            for strand, nucleotide_seq in [(+1, seq), (-1, seq.reverse_complement())]:
                for i in range(frame, len(nucleotide_seq) - 2, 3):
                    codon = nucleotide_seq[i:i+3]
                    if len(codon) == 3 and str(codon).upper() == 'ATG':  # Start codon
                        # Find stop codon
                        for j in range(i+3, len(nucleotide_seq) - 2, 3):
                            stop_codon = nucleotide_seq[j:j+3]
                            if str(stop_codon).upper() in ['TAA', 'TAG', 'TGA']:
                                if j - i >= min_length:
                                    protein = nucleotide_seq[i:j].translate()
                                    orfs.append(str(protein))
                                break
        
        return orfs
    
    def batch_process_fasta(self, fasta_path: str) -> pd.DataFrame:
        """
        Process FASTA file and extract sequences with metadata.
        
        Args:
            fasta_path: Path to FASTA file
            
        Returns:
            DataFrame with sequence data
        """
        sequences_data = []
        
        with open(fasta_path, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequence = str(record.seq)
                
                if self.validate_sequence(sequence):
                    sequences_data.append({
                        'id': record.id,
                        'description': record.description,
                        'sequence': sequence,
                        'length': len(sequence),
                        'molecular_weight': self._calculate_mw(sequence),
                        'isoelectric_point': self._calculate_pi(sequence)
                    })
        
        return pd.DataFrame(sequences_data)
    
    def _calculate_mw(self, sequence: str) -> float:
        """Calculate molecular weight of protein sequence."""
        try:
            analysis = ProteinAnalysis(sequence)
            return analysis.molecular_weight()
        except:
            return 0.0
    
    def _calculate_pi(self, sequence: str) -> float:
        """Calculate isoelectric point of protein sequence."""
        try:
            analysis = ProteinAnalysis(sequence)
            return analysis.isoelectric_point()
        except:
            return 0.0
