"""Activation patching for causal circuit analysis - inspired by Anthropic's methods."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

class ActivationPatcher:
    """Implements activation patching to find causal circuits."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
    
    def patch_attention_head(self, layer: int, head: int, patch_value: torch.Tensor):
        """Patch a specific attention head's output."""
        
        def patch_hook(module, input, output):
            # output shape: [batch, seq_len, num_heads, head_dim]
            if hasattr(output, 'shape') and len(output.shape) == 4:
                output[:, :, head, :] = patch_value
            return output
        
        # Register hook on attention output
        attention_layer = self.model.transformer.h[layer].attn
        hook = attention_layer.register_forward_hook(patch_hook)
        self.hooks.append(hook)
        
        return hook
    
    def patch_mlp_neuron(self, layer: int, neuron_idx: int, patch_value: float):
        """Patch a specific MLP neuron."""
        
        def patch_hook(module, input, output):
            # Patch specific neuron in MLP output
            if hasattr(output, 'shape') and len(output.shape) >= 2:
                output[:, :, neuron_idx] = patch_value
            return output
        
        # Register hook on MLP output
        mlp_layer = self.model.transformer.h[layer].mlp
        hook = mlp_layer.register_forward_hook(patch_hook)
        self.hooks.append(hook)
        
        return hook
    
    def measure_causal_effect(self, 
                            original_input: str, 
                            corrupted_input: str,
                            target_token_pos: int,
                            patch_components: List[Tuple[str, int, int]]) -> Dict[str, float]:
        """
        Measure causal effect using activation patching.
        
        Args:
            original_input: Clean input text
            corrupted_input: Corrupted/baseline input text  
            target_token_pos: Position of token to measure
            patch_components: List of (component_type, layer, head/neuron) to patch
            
        Returns:
            Dictionary of causal effects
        """
        
        # Tokenize inputs
        orig_tokens = self.tokenizer(original_input, return_tensors="pt")
        corr_tokens = self.tokenizer(corrupted_input, return_tensors="pt")
        
        # Get baseline outputs
        with torch.no_grad():
            orig_output = self.model(**orig_tokens)
            corr_output = self.model(**corr_tokens)
        
        # Extract target logits
        orig_logits = orig_output.logits[0, target_token_pos]
        corr_logits = corr_output.logits[0, target_token_pos]
        
        causal_effects = {}
        
        # Test each component
        for component_type, layer, component_idx in patch_components:
            
            # Get activations from corrupted run to use as patch
            with torch.no_grad():
                if component_type == "attention_head":
                    # Extract attention head activation from corrupted input
                    corr_attn = self._extract_attention_head(corr_tokens, layer, component_idx)
                    
                    # Patch this head in original input
                    hook = self.patch_attention_head(layer, component_idx, corr_attn)
                    
                elif component_type == "mlp_neuron":
                    # Extract MLP neuron activation from corrupted input
                    corr_neuron = self._extract_mlp_neuron(corr_tokens, layer, component_idx)
                    
                    # Patch this neuron in original input
                    hook = self.patch_mlp_neuron(layer, component_idx, corr_neuron)
                
                # Run forward pass with patch
                patched_output = self.model(**orig_tokens)
                patched_logits = patched_output.logits[0, target_token_pos]
                
                # Measure causal effect
                causal_effect = torch.norm(orig_logits - patched_logits).item()
                causal_effects[f"{component_type}_L{layer}C{component_idx}"] = causal_effect
                
                # Remove hook
                hook.remove()
        
        return causal_effects
    
    def _extract_attention_head(self, tokens: Dict, layer: int, head: int) -> torch.Tensor:
        """Extract attention head output for patching."""
        
        # This is a simplified version - would need model-specific implementation
        with torch.no_grad():
            outputs = self.model(**tokens, output_attentions=True)
            attention = outputs.attentions[layer]  # [batch, heads, seq, seq]
            
            # Return attention pattern for specific head
            return attention[:, head, :, :]
    
    def _extract_mlp_neuron(self, tokens: Dict, layer: int, neuron_idx: int) -> float:
        """Extract MLP neuron activation for patching."""
        
        # This would need model-specific implementation
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            
            # Return specific neuron activation (simplified)
            return hidden_states[:, :, neuron_idx].mean().item()
    
    def find_important_circuits(self, 
                              prompt: str, 
                              risky_generation: str,
                              safe_generation: str,
                              target_token_pos: int) -> List[Dict]:
        """
        Find circuits important for generating risky vs safe content.
        """
        
        # Test all attention heads
        components_to_test = []
        
        # Add attention heads to test
        for layer in range(self.model.config.n_layer):
            for head in range(self.model.config.n_head):
                components_to_test.append(("attention_head", layer, head))
        
        # Test a subset of MLP neurons (testing all would be expensive)
        for layer in range(self.model.config.n_layer):
            for neuron in range(0, self.model.config.n_embd, 100):  # Sample every 100th neuron
                components_to_test.append(("mlp_neuron", layer, neuron))
        
        # Measure causal effects
        risky_input = prompt + " " + risky_generation
        safe_input = prompt + " " + safe_generation
        
        causal_effects = self.measure_causal_effect(
            risky_input, safe_input, target_token_pos, components_to_test
        )
        
        # Rank by importance
        important_circuits = []
        for component, effect in sorted(causal_effects.items(), key=lambda x: x[1], reverse=True):
            if effect > 0.1:  # Threshold for significance
                important_circuits.append({
                    "component": component,
                    "causal_effect": effect,
                    "importance": "high" if effect > 0.5 else "medium"
                })
        
        return important_circuits[:10]  # Return top 10
    
    def cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
