"""Context extraction for circuit analysis."""

from typing import Tuple, List
import config

class ContextExtractor:
    """Extracts relevant context windows for circuit analysis."""
    
    def __init__(self, window_size: int = None):
        self.window_size = window_size or config.CONTEXT_WINDOW_SIZE
    
    def extract_context(self, prompt: str, output: str, token_position: int) -> str:
        """
        Extract context window around a target token.
        
        Args:
            prompt: Original input prompt
            output: Model's generated output 
            token_position: Position of risky token in output
            
        Returns:
            Context string suitable for analysis
        """
        # Combine prompt and output
        full_text = prompt + " " + output
        words = full_text.split()
        
        # Calculate window boundaries
        start_pos = max(0, token_position - self.window_size // 2)
        end_pos = min(len(words), token_position + self.window_size // 2)
        
        # Extract context window
        context_words = words[start_pos:end_pos]
        context = " ".join(context_words)
        
        return context
    
    def get_token_position_in_full_text(self, prompt: str, output: str, token_position_in_output: int) -> int:
        """
        Convert token position in output to position in full text (prompt + output).
        
        Args:
            prompt: Original prompt
            output: Generated output
            token_position_in_output: Position in output only
            
        Returns:
            Position in combined prompt + output text
        """
        prompt_length = len(prompt.split())
        return prompt_length + token_position_in_output
    
    def extract_sentence_context(self, text: str, token_position: int) -> str:
        """Extract sentence-level context around a token."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        # Find which sentence contains our token
        word_count = 0
        for sentence in sentences:
            sentence_words = sentence.split()
            if word_count <= token_position < word_count + len(sentence_words):
                return sentence.strip()
            word_count += len(sentence_words)
        
        # Fallback to word-based context
        start = max(0, token_position - 10)
        end = min(len(words), token_position + 10)
        return " ".join(words[start:end])
    
    def prepare_for_tokenizer(self, context: str, tokenizer) -> dict:
        """
        Prepare context for tokenizer input.
        
        Args:
            context: Context string
            tokenizer: HuggingFace tokenizer
            
        Returns:
            Tokenizer inputs ready for model
        """
        inputs = tokenizer(
            context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.window_size
        )
        return inputs