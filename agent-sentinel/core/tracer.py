"""Core circuit tracing functionality using custom attribution analysis."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import uuid
import numpy as np

from agentsentinel.core.detector import RiskDetector
from agentsentinel.core.context_extractor import ContextExtractor

@dataclass
class CircuitTrace:
    """Result of circuit-level tracing analysis."""
    trace_id: str
    prompt: str
    output: str
    risky_token: str
    token_position: int
    attribution_scores: Dict[str, float]
    dangerous_circuits: List[Tuple[str, int, int]]  # (type, layer, head/neuron)
    policy_violation: bool
    policy_name: str
    alert_level: str
    timestamp: float

class SimpleCircuitTracer:
    """Basic circuit tracer using attention analysis."""
    
    def __init__(self, model, tokenizer, policy_config: str = "strict_safety"):
        self.model = model
        self.tokenizer = tokenizer
        self.policy_config = policy_config
        
        # Initialize components
        self.risk_detector = RiskDetector()
        self.context_extractor = ContextExtractor()
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def analyze_generation(self, prompt: str, output: str) -> Optional[CircuitTrace]:
        """Analyze a model generation for circuit-level policy violations."""
        
        # Step 1: Detect risky tokens in output
        risky_tokens = self.risk_detector.scan_output(output)
        if not risky_tokens:
            return None  # No risky content detected
        
        # Step 2: Select most concerning token for tracing
        target_token, token_position = self._select_target_token(output, risky_tokens)
        
        # Step 3: Extract context window
        context = self.context_extractor.extract_context(
            prompt, output, token_position
        )
        
        # Step 4: Perform circuit tracing
        attribution_result = self._trace_token_circuits(context, target_token)
        if not attribution_result:
            return None
        
        # Step 5: Analyze attribution against policy (simplified for now)
        dangerous_circuits = self._identify_dangerous_circuits(attribution_result["scores"])
        policy_violation = len(dangerous_circuits) > 0
        alert_level = "WARNING" if policy_violation else "INFO"
        
        # Step 6: Create trace result
        trace = CircuitTrace(
            trace_id=str(uuid.uuid4()),
            prompt=prompt,
            output=output,
            risky_token=target_token,
            token_position=token_position,
            attribution_scores=attribution_result["scores"],
            dangerous_circuits=dangerous_circuits,
            policy_violation=policy_violation,
            policy_name=self.policy_config,
            alert_level=alert_level,
            timestamp=time.time()
        )
        
        return trace
    
    def _select_target_token(self, output: str, risky_tokens: List[Dict]) -> Tuple[str, int]:
        """Select the most concerning risky token for detailed tracing."""
        # Sort by risk score and select highest
        sorted_tokens = sorted(risky_tokens, key=lambda x: x["risk_score"], reverse=True)
        target = sorted_tokens[0]
        return target["token"], target["position"]
    
    def _trace_token_circuits(self, context: str, target_token: str) -> Optional[Dict]:
        """Perform basic circuit tracing on target token."""
        try:
            # Tokenize input
            inputs = self.tokenizer(context, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # More robust token finding - try different approaches
            target_token_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
            target_pos = None
            
            # Method 1: Direct token ID match
            for token_id in target_token_ids:
                target_positions = (input_ids == token_id).nonzero(as_tuple=True)
                if len(target_positions[1]) > 0:
                    target_pos = target_positions[1][-1]  # Use last occurrence
                    break
            
            # Method 2: If not found, look for partial matches in decoded text
            if target_pos is None:
                decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[0]]
                for i, decoded_token in enumerate(decoded_tokens):
                    if target_token.lower() in decoded_token.lower():
                        target_pos = torch.tensor(i)
                        break
            
            if target_pos is None:
                print(f"Target token '{target_token}' not found in context: '{context[:100]}...'")
                print(f"Available tokens: {[self.tokenizer.decode([tid]) for tid in input_ids[0][:20]]}")
                return None
            
            # Run forward pass with attention tracking
            with torch.no_grad():
                outputs = self.model(
                    input_ids, 
                    output_attentions=True,
                    output_hidden_states=True
                )
            
            # Extract attribution scores from attention patterns
            attribution_scores = self._extract_attribution_scores(
                outputs.attentions, 
                target_pos,
                input_ids
            )
            
            return {
                "scores": attribution_scores,
                "target_position": target_pos.item(),
                "context_length": input_ids.shape[1]
            }
            
        except Exception as e:
            print(f"Error during circuit tracing: {e}")
            return None
    
    def _extract_attribution_scores(self, attentions: List[torch.Tensor], target_pos: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, float]:
        """Extract attribution scores from attention patterns."""
        scores = {
            "attention_heads": {},
            "total_attribution": 0.0
        }
        
        # Analyze attention patterns for each layer and head
        for layer_idx, layer_attention in enumerate(attentions):
            # layer_attention shape: [batch, heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = layer_attention.shape
            
            for head_idx in range(num_heads):
                # Get attention weights TO our target token
                attention_to_target = layer_attention[0, head_idx, :, target_pos]
                
                # Calculate attribution score (sum of attention from risky context)
                attribution = attention_to_target.sum().item()
                
                scores["attention_heads"][(layer_idx, head_idx)] = attribution
                scores["total_attribution"] += attribution
        
        return scores
    
    def _identify_dangerous_circuits(self, attribution_scores: Dict) -> List[Tuple[str, int, int]]:
        """Identify potentially dangerous circuits based on attribution scores."""
        dangerous_circuits = []
        
        # Simple threshold-based detection
        attention_threshold = 0.15  # Should come from config
        
        attention_scores = attribution_scores.get("attention_heads", {})
        
        for (layer, head), score in attention_scores.items():
            if score > attention_threshold:
                dangerous_circuits.append(("attention_head", layer, head))
        
        return dangerous_circuits
    
    def generate_and_analyze(self, prompt: str, max_length: int = 50) -> Tuple[str, Optional[CircuitTrace]]:
        """Generate text from prompt and analyze for policy violations."""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.get("attention_mask")
            )
        
        # Decode full response
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip()
        
        # Analyze for policy violations
        trace_result = self.analyze_generation(prompt, generated_text)
        
        return generated_text, trace_result