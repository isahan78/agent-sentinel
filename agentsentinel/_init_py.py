"""AgentSentinel: Mechanistic AI Security Tool."""

from .core.detector import RiskDetector
from .core.tracer import SimpleCircuitTracer
from .core.policy_engine import PolicyEngine
from .core.context_extractor import ContextExtractor
from .models.model_loader import load_model, get_model_info, supported_models

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Make key classes available at package level
__all__ = [
    "RiskDetector",
    "SimpleCircuitTracer", 
    "PolicyEngine",
    "ContextExtractor",
    "load_model", 
    "get_model_info",
    "supported_models"
]