"""Risk detection for potentially harmful model outputs."""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import config

@dataclass
class RiskToken:
    """Represents a risky token detection."""
    token: str
    position: int
    risk_score: float
    detection_method: str
    context: str

class RiskDetector:
    """Detects potentially risky tokens in model outputs."""
    
    def __init__(self, detection_config: Dict = None):
        self.config = detection_config or config.RISK_DETECTION
        self.keywords = self.config["lexical_keywords"]
        self.semantic_threshold = self.config["semantic_threshold"]
    
    def scan_output(self, output: str) -> List[Dict]:
        """Scan output text for risky tokens."""
        risky_tokens = []
        
        # Lexical keyword detection
        lexical_risks = self._detect_lexical_risks(output)
        risky_tokens.extend(lexical_risks)
        
        # TODO: Add semantic risk detection using embeddings
        # semantic_risks = self._detect_semantic_risks(output)
        # risky_tokens.extend(semantic_risks)
        
        # Sort by risk score
        risky_tokens.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return risky_tokens
    
    def _detect_lexical_risks(self, output: str) -> List[Dict]:
        """Detect risky tokens using keyword matching."""
        risks = []
        words = output.lower().split()
        
        for i, word in enumerate(words):
            # Clean word of punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check exact matches
            if clean_word in self.keywords:
                risk_score = self._calculate_keyword_risk(clean_word)
                context = self._get_context_around_position(words, i)
                
                risks.append({
                    "token": clean_word,
                    "position": i,
                    "risk_score": risk_score,
                    "detection_method": "lexical_keyword",
                    "context": context
                })
            
            # Also check for partial matches (e.g., "explosive" contains "explosive")
            else:
                for keyword in self.keywords:
                    if keyword in clean_word and len(keyword) >= 4:  # Avoid tiny matches
                        risk_score = self._calculate_keyword_risk(keyword) * 0.8  # Slight penalty for partial
                        context = self._get_context_around_position(words, i)
                        
                        risks.append({
                            "token": keyword,  # Use the matched keyword, not the full word
                            "position": i,
                            "risk_score": risk_score,
                            "detection_method": "lexical_partial",
                            "context": context
                        })
                        break  # Only match one keyword per word
        
        return risks
    
    def _get_context_around_position(self, words: List[str], position: int, window: int = 5) -> str:
        """Get context window around a word position."""
        start = max(0, position - window)
        end = min(len(words), position + window + 1)
        return " ".join(words[start:end])
    
    def _calculate_keyword_risk(self, keyword: str) -> float:
        """Calculate risk score for a keyword."""
        # High-severity keywords
        high_severity = ["bomb", "kill", "murder", "suicide", "exploit", "hack"]
        medium_severity = ["weapon", "illegal", "drug", "fraud", "steal"]
        low_severity = ["crack", "virus", "scam"]
        
        if keyword in high_severity:
            return 0.9
        elif keyword in medium_severity:
            return 0.7
        elif keyword in low_severity:
            return 0.5
        else:
            return 0.3
    
    def _detect_semantic_risks(self, output: str) -> List[Dict]:
        """Detect risky content using semantic analysis."""
        # TODO: Implement semantic risk detection
        # Could use sentence transformers to compare against
        # known harmful content embeddings
        return []
    
    def add_custom_keywords(self, keywords: List[str]):
        """Add custom keywords to the detection list."""
        self.keywords.extend(keywords)
    
    def remove_keywords(self, keywords: List[str]):
        """Remove keywords from the detection list."""
        for keyword in keywords:
            if keyword in self.keywords:
                self.keywords.remove(keyword)
    
    def get_risk_summary(self, risky_tokens: List[Dict]) -> Dict:
        """Generate a summary of detected risks."""
        if not risky_tokens:
            return {
                "total_risks": 0,
                "max_risk_score": 0.0,
                "risk_categories": {},
                "most_concerning_token": None
            }
        
        # Count risks by severity
        high_risk = sum(1 for token in risky_tokens if token["risk_score"] >= 0.8)
        medium_risk = sum(1 for token in risky_tokens if 0.5 <= token["risk_score"] < 0.8)
        low_risk = sum(1 for token in risky_tokens if token["risk_score"] < 0.5)
        
        return {
            "total_risks": len(risky_tokens),
            "max_risk_score": max(token["risk_score"] for token in risky_tokens),
            "risk_categories": {
                "high": high_risk,
                "medium": medium_risk,
                "low": low_risk
            },
            "most_concerning_token": risky_tokens[0]  # Highest scoring token
        }