"""Model loading utilities for AgentSentinel."""

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def load_model(model_name: str = "gpt2", device: Optional[str] = None) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a model and tokenizer for circuit analysis.
    
    Args:
        model_name: Name of the model to load (default: "gpt2")
        device: Device to load model on. If None, auto-detects.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading model {model_name} on device {device}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for better precision in analysis
            device_map=device if device != "auto" else "auto"
        )
        
        # Ensure model is in eval mode for analysis
        model.eval()
        
        logger.info(f"Successfully loaded {model_name}")
        logger.info(f"Model has {model.num_parameters():,} parameters")
        logger.info(f"Model config: {model.config}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

def get_model_info(model: torch.nn.Module) -> dict:
    """Get information about a loaded model."""
    return {
        "num_parameters": model.num_parameters(),
        "num_layers": model.config.n_layer if hasattr(model.config, 'n_layer') else "unknown",
        "hidden_size": model.config.hidden_size if hasattr(model.config, 'hidden_size') else "unknown",
        "num_attention_heads": model.config.n_head if hasattr(model.config, 'n_head') else "unknown",
        "vocab_size": model.config.vocab_size if hasattr(model.config, 'vocab_size') else "unknown",
    }

def supported_models():
    """Return list of supported models."""
    return [
        "gpt2",
        "gpt2-medium", 
        "gpt2-large",
        "gpt2-xl",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B"
    ]

def validate_model_for_analysis(model: torch.nn.Module) -> bool:
    """
    Validate that a model is suitable for circuit analysis.
    
    Args:
        model: The loaded model
        
    Returns:
        True if model is suitable for analysis
    """
    required_attrs = ['config', 'transformer']
    
    for attr in required_attrs:
        if not hasattr(model, attr):
            logger.warning(f"Model missing required attribute: {attr}")
            return False
    
    # Check if it's a causal language model
    if not hasattr(model, 'lm_head'):
        logger.warning("Model doesn't appear to be a causal language model")
        return False
        
    return True