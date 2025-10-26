"""Advanced circuit tracer using TransformerLens for proper mechanistic interpretability."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

try:
    import transformer_lens as tl
    from transformer_lens import HookedTransformer, ActivationCache
    from transformer_lens.utils import get_act_name
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    print("TransformerLens not available. Install with: pip install transformer-lens")

from agentsentinel.core.detector import RiskDetector
from agentsentinel.core.context_extractor import ContextExtractor

@dataclass
class AdvancedCircuitTrace:
    """Result of advanced circuit-level tracing analysis."""
    trace_id: str
    prompt: str
    output: str
    risky_token: str
    token_position: int
    
    # Advanced attribution data
    direct_logit_attribution: Dict[str, float]  # Direct effect on output logits
    attention_attribution: Dict[Tuple[int, int], float]  # (layer, head) -> attribution
    mlp_attribution: Dict[int, float]  # layer -> attribution
    residual_attribution: Dict[int, float]  # layer -> residual stream attribution
    
    # Path analysis
    important_paths: List[Dict]  # Paths through the network
    intervention_effects: Dict[str, float]  # Effect of intervening on components
    
    # Traditional fields
    dangerous_circuits: List[Tuple[str, int, int]]
    policy_violation: bool
    policy_name: str
    alert_level: str
    timestamp: float

class AdvancedCircuitTracer:
    """Advanced circuit tracer using proper mechanistic interpretability techniques."""
    
    def __init__(self, model_name: str = "gpt2-small", policy_config: str = "strict_safety"):
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError("TransformerLens required for advanced tracing. Install with: pip install transformer-lens")
        
        self.model_name = model_name
        self.policy_config = policy_config
        
        # Load model with TransformerLens
        print(f"Loading {model_name} with TransformerLens...")
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.eval()
        
        # Initialize components
        self.risk_detector = RiskDetector()
        self.context_extractor = ContextExtractor()
        
        print(f"✅ Advanced tracer ready with {model_name}")
    
    def analyze_generation(self, prompt: str, output: str) -> Optional[AdvancedCircuitTrace]:
        """Perform advanced circuit-level analysis of model generation."""
        
        # Step 1: Detect risky tokens
        risky_tokens = self.risk_detector.scan_output(output)
        if not risky_tokens:
            return None
        
        # Step 2: Select target token
        target_token, token_position = self._select_target_token(output, risky_tokens)
        
        # Step 3: Prepare full sequence for analysis
        full_text = prompt + " " + output
        tokens = self.model.to_tokens(full_text)
        
        # Find target position in tokenized sequence
        target_pos = self._find_target_position(tokens, target_token, token_position)
        if target_pos is None:
            return None
        
        # Step 4: Run advanced attribution analysis
        attribution_results = self._run_attribution_analysis(tokens, target_pos)
        
        # Step 5: Analyze paths and interventions
        path_results = self._analyze_important_paths(tokens, target_pos, attribution_results)
        
        # Step 6: Evaluate against policy
        dangerous_circuits = self._identify_dangerous_circuits(attribution_results)
        policy_violation = len(dangerous_circuits) > 0
        alert_level = self._determine_alert_level(attribution_results, dangerous_circuits)
        
        # Step 7: Create advanced trace result
        trace = AdvancedCircuitTrace(
            trace_id=f"adv_{int(time.time())}",
            prompt=prompt,
            output=output,
            risky_token=target_token,
            token_position=token_position,
            
            # Advanced attributions
            direct_logit_attribution=attribution_results["direct_logit"],
            attention_attribution=attribution_results["attention"],
            mlp_attribution=attribution_results["mlp"],
            residual_attribution=attribution_results["residual"],
            
            # Path analysis
            important_paths=path_results["paths"],
            intervention_effects=path_results["interventions"],
            
            # Standard fields
            dangerous_circuits=dangerous_circuits,
            policy_violation=policy_violation,
            policy_name=self.policy_config,
            alert_level=alert_level,
            timestamp=time.time()
        )
        
        return trace
    
    def _run_attribution_analysis(self, tokens: torch.Tensor, target_pos: int) -> Dict:
        """Run comprehensive attribution analysis using TransformerLens."""
        
        # Get model activations
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens)
        
        # Target logit for the risky token
        target_logit = logits[0, target_pos]
        
        # 1. Direct logit attribution (residual stream -> logits)
        direct_attribution = self._compute_direct_logit_attribution(cache, target_pos, target_logit)
        
        # 2. Attention head attribution
        attention_attribution = self._compute_attention_attribution(cache, target_pos, target_logit)
        
        # 3. MLP attribution  
        mlp_attribution = self._compute_mlp_attribution(cache, target_pos, target_logit)
        
        # 4. Residual stream attribution
        residual_attribution = self._compute_residual_attribution(cache, target_pos)
        
        return {
            "direct_logit": direct_attribution,
            "attention": attention_attribution,
            "mlp": mlp_attribution,
            "residual": residual_attribution
        }
    
    def _compute_direct_logit_attribution(self, cache: ActivationCache, target_pos: int, target_logit: torch.Tensor) -> Dict[str, float]:
        """Compute direct attribution from residual stream to output logits."""
        
        # Get final residual stream
        final_residual = cache[get_act_name("resid_post", self.model.cfg.n_layers - 1)][0, target_pos]
        
        # Get unembedding weights
        W_U = self.model.W_U  # [d_model, d_vocab]
        
        # Compute contribution of each dimension
        logit_contributions = final_residual @ W_U  # [d_vocab]
        
        # Get top contributing tokens
        top_k = 10
        top_indices = torch.topk(torch.abs(logit_contributions), top_k).indices
        
        direct_attribution = {}
        for idx in top_indices:
            token = self.model.to_string(idx.item())
            direct_attribution[f"token_{token}"] = logit_contributions[idx].item()
        
        return direct_attribution
    
    def _compute_attention_attribution(self, cache: ActivationCache, target_pos: int, target_logit: torch.Tensor) -> Dict[Tuple[int, int], float]:
        """Compute attribution from attention heads."""
        
        attention_attribution = {}
        
        for layer in range(self.model.cfg.n_layers):
            # Get attention pattern for this layer
            attn_pattern = cache[get_act_name("pattern", layer)][0]  # [n_head, seq_len, seq_len]
            
            # Get attention output
            attn_out = cache[get_act_name("attn_out", layer)][0, target_pos]  # [d_model]
            
            for head in range(self.model.cfg.n_heads):
                # Attribution is attention pattern to target position
                head_attribution = attn_pattern[head, :, target_pos].sum().item()
                attention_attribution[(layer, head)] = head_attribution
        
        return attention_attribution
    
    def _compute_mlp_attribution(self, cache: ActivationCache, target_pos: int, target_logit: torch.Tensor) -> Dict[int, float]:
        """Compute attribution from MLP layers."""
        
        mlp_attribution = {}
        
        for layer in range(self.model.cfg.n_layers):
            # Get MLP output
            mlp_out = cache[get_act_name("mlp_out", layer)][0, target_pos]  # [d_model]
            
            # Simple attribution: L2 norm of MLP output
            mlp_attribution[layer] = torch.norm(mlp_out).item()
        
        return mlp_attribution
    
    def _compute_residual_attribution(self, cache: ActivationCache, target_pos: int) -> Dict[int, float]:
        """Compute attribution in residual stream."""
        
        residual_attribution = {}
        
        for layer in range(self.model.cfg.n_layers):
            # Get residual stream at each layer
            resid = cache[get_act_name("resid_post", layer)][0, target_pos]
            residual_attribution[layer] = torch.norm(resid).item()
        
        return residual_attribution
    
    def _analyze_important_paths(self, tokens: torch.Tensor, target_pos: int, attribution_results: Dict) -> Dict:
        """Analyze important paths through the network."""
        
        # Find top attention heads
        top_heads = sorted(attribution_results["attention"].items(), 
                          key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Find top MLP layers
        top_mlps = sorted(attribution_results["mlp"].items(),
                         key=lambda x: abs(x[1]), reverse=True)[:3]
        
        paths = []
        for (layer, head), score in top_heads:
            paths.append({
                "type": "attention_head",
                "layer": layer,
                "head": head,
                "attribution": score,
                "description": f"Attention L{layer}H{head}"
            })
        
        for layer, score in top_mlps:
            paths.append({
                "type": "mlp",
                "layer": layer,
                "attribution": score,
                "description": f"MLP L{layer}"
            })
        
        # TODO: Implement actual intervention experiments
        interventions = {
            "path_patching": 0.0,  # Placeholder
            "activation_patching": 0.0  # Placeholder
        }
        
        return {
            "paths": paths,
            "interventions": interventions
        }
    
    def _select_target_token(self, output: str, risky_tokens: List[Dict]) -> Tuple[str, int]:
        """Select the most concerning risky token for detailed tracing."""
        sorted_tokens = sorted(risky_tokens, key=lambda x: x["risk_score"], reverse=True)
        target = sorted_tokens[0]
        return target["token"], target["position"]
    
    def _find_target_position(self, tokens: torch.Tensor, target_token: str, word_position: int) -> Optional[int]:
        """Find target token position in tokenized sequence."""
        
        # Convert tokens back to strings to find position
        token_strings = [self.model.to_string(token) for token in tokens[0]]
        
        # Simple approach: find the target token
        for i, token_str in enumerate(token_strings):
            if target_token.lower() in token_str.lower():
                return i
        
        return None
    
    def _identify_dangerous_circuits(self, attribution_results: Dict) -> List[Tuple[str, int, int]]:
        """Identify dangerous circuits based on attribution analysis."""
        
        dangerous_circuits = []
        threshold = 0.1  # Lower threshold for more sensitive detection
        
        # Check attention heads
        for (layer, head), score in attribution_results["attention"].items():
            if abs(score) > threshold:
                dangerous_circuits.append(("attention_head", layer, head))
        
        # Check MLPs
        for layer, score in attribution_results["mlp"].items():
            if abs(score) > threshold * 5:  # Higher threshold for MLPs
                dangerous_circuits.append(("mlp", layer, 0))
        
        return dangerous_circuits
    
    def _determine_alert_level(self, attribution_results: Dict, dangerous_circuits: List) -> str:
        """Determine alert level based on attribution analysis."""
        
        total_attention_attr = sum(abs(score) for score in attribution_results["attention"].values())
        
        if total_attention_attr > 2.0:
            return "CRITICAL"
        elif total_attention_attr > 1.0:
            return "WARNING"
        else:
            return "INFO"
    
    def generate_and_analyze(self, prompt: str, max_length: int = 50) -> Tuple[str, Optional[AdvancedCircuitTrace]]:
        """Generate text and perform advanced analysis."""
        
        # Generate using TransformerLens
        tokens = self.model.to_tokens(prompt)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode result
        full_text = self.model.to_string(generated_tokens[0])
        generated_text = full_text[len(prompt):].strip()
        
        # Analyze for policy violations
        trace_result = self.analyze_generation(prompt, generated_text)
        
        return generated_text, trace_result
