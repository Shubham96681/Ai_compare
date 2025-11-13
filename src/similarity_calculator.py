"""Similarity Calculator - Compares model outputs against master model."""

import re
from typing import Dict, Any, Optional
import math


class SimilarityCalculator:
    """Calculates similarity scores between model outputs and master model output."""
    
    @staticmethod
    def calculate_similarity(
        master_response: str,
        candidate_response: str,
        method: str = "combined"
    ) -> Dict[str, Any]:
        """
        Calculate similarity score between master and candidate responses.
        
        Args:
            master_response: Response from master/reference model
            candidate_response: Response from candidate model to compare
            method: Similarity calculation method ("combined", "cosine", "jaccard", "levenshtein")
        
        Returns:
            Dictionary with similarity score (0-100) and details
        """
        if not master_response or not candidate_response:
            return {
                "similarity_score": 0.0,
                "similarity_percentage": 0.0,
                "method": method,
                "error": "Empty response(s)"
            }
        
        # Normalize responses
        master_normalized = SimilarityCalculator._normalize_text(master_response)
        candidate_normalized = SimilarityCalculator._normalize_text(candidate_response)
        
        if method == "combined":
            # Use multiple methods and average them
            cosine_score = SimilarityCalculator._cosine_similarity(master_normalized, candidate_normalized)
            jaccard_score = SimilarityCalculator._jaccard_similarity(master_normalized, candidate_normalized)
            levenshtein_score = SimilarityCalculator._levenshtein_similarity(master_normalized, candidate_normalized)
            
            # Weighted average (cosine gets more weight as it's generally more reliable)
            combined_score = (cosine_score * 0.5) + (jaccard_score * 0.3) + (levenshtein_score * 0.2)
            
            return {
                "similarity_score": combined_score,
                "similarity_percentage": round(combined_score * 100, 2),
                "method": "combined",
                "cosine_score": round(cosine_score * 100, 2),
                "jaccard_score": round(jaccard_score * 100, 2),
                "levenshtein_score": round(levenshtein_score * 100, 2),
            }
        elif method == "cosine":
            score = SimilarityCalculator._cosine_similarity(master_normalized, candidate_normalized)
            return {
                "similarity_score": score,
                "similarity_percentage": round(score * 100, 2),
                "method": "cosine",
            }
        elif method == "jaccard":
            score = SimilarityCalculator._jaccard_similarity(master_normalized, candidate_normalized)
            return {
                "similarity_score": score,
                "similarity_percentage": round(score * 100, 2),
                "method": "jaccard",
            }
        elif method == "levenshtein":
            score = SimilarityCalculator._levenshtein_similarity(master_normalized, candidate_normalized)
            return {
                "similarity_score": score,
                "similarity_percentage": round(score * 100, 2),
                "method": "levenshtein",
            }
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison (lowercase, remove extra whitespace)."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def _cosine_similarity(text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using word frequency vectors."""
        # Tokenize into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Get all unique words
        all_words = words1.union(words2)
        
        if not all_words:
            return 1.0 if text1 == text2 else 0.0
        
        # Create frequency vectors
        vec1 = [1 if word in words1 else 0 for word in all_words]
        vec2 = [1 if word in words2 else 0 for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity (intersection over union of words)."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def _levenshtein_similarity(text1: str, text2: str) -> float:
        """Calculate similarity based on Levenshtein distance."""
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
        
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)  # Ensure non-negative

