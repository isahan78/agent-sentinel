"""Policy engine for evaluating circuit attribution against safety policies."""

from typing import Dict, List, Tuple, Any
import config

class PolicyEngine:
    """Evaluates circuit attributions against configured safety policies."""
    
    def __init__(self, policy_name: str):
        self.policy_name = policy_name
        self.policy_config = config.CIRCUIT_POLICIES[policy_name]
    
    def evaluate_attribution(self, attribution_scores: Dict) -> Dict:
        """Evaluate attribution scores against policy rules."""
        dangerous_circuits = []
        violation_detected = False
        alert_level = "INFO"
        
        # Check dangerous attention heads
        attention_violations = self._check_attention_heads(attribution_scores)
        dangerous_circuits.extend(attention_violations)
        
        # Check dangerous MLP neurons (placeholder for now)
        mlp_violations = self._check_mlp_neurons(attribution_scores)
        dangerous_circuits.extend(mlp_violations)
        
        # Determine overall violation status
        if dangerous_circuits:
            violation_detected = True
            alert_level = self._determine_alert_level(dangerous_circuits, attribution_scores)
        
        return {
            "violation_detected": violation_detected,
            "dangerous_circuits": dangerous_circuits,
            "alert_level": alert_level,
            "policy_name": self.policy_name
        }
    
    def _check_attention_heads(self, attribution_scores: Dict) -> List[Tuple[str, int, int]]:
        """Check for dangerous attention head activations."""
        violations = []
        dangerous_heads = self.policy_config["dangerous_attention_heads"]
        threshold = self.policy_config["attribution_thresholds"]["attention_head_threshold"]
        
        attention_scores = attribution_scores.get("attention_heads", {})
        
        for (layer, head) in dangerous_heads:
            if (layer, head) in attention_scores:
                score = attention_scores[(layer, head)]
                if score > threshold:
                    violations.append(("attention_head", layer, head))
        
        return violations
    
    def _check_mlp_neurons(self, attribution_scores: Dict) -> List[Tuple[str, int, int]]:
        """Check for dangerous MLP neuron activations."""
        violations = []
        dangerous_neurons = self.policy_config["dangerous_mlp_neurons"]
        threshold = self.policy_config["attribution_thresholds"]["mlp_neuron_threshold"]
        
        mlp_scores = attribution_scores.get("mlp_neurons", {})
        
        for (layer, neuron_range) in dangerous_neurons:
            start_neuron, end_neuron = neuron_range
            for (l, n) in mlp_scores:
                if l == layer and start_neuron <= n <= end_neuron:
                    score = mlp_scores[(l, n)]
                    if score > threshold:
                        violations.append(("mlp_neuron", layer, n))
        
        return violations
    
    def _determine_alert_level(self, dangerous_circuits: List, attribution_scores: Dict) -> str:
        """Determine appropriate alert level based on violations."""
        total_dangerous_influence = sum(
            attribution_scores.get("attention_heads", {}).get((layer, head), 0)
            for circuit_type, layer, head in dangerous_circuits
            if circuit_type == "attention_head"
        )
        
        total_threshold = self.policy_config["attribution_thresholds"]["total_dangerous_influence"]
        
        if total_dangerous_influence > total_threshold * 1.5:
            return "CRITICAL"
        elif total_dangerous_influence > total_threshold:
            return "WARNING"
        else:
            return "INFO"
    
    def get_policy_summary(self) -> Dict:
        """Get summary of current policy configuration."""
        return {
            "policy_name": self.policy_name,
            "description": self.policy_config["description"],
            "dangerous_heads_count": len(self.policy_config["dangerous_attention_heads"]),
            "dangerous_neurons_count": len(self.policy_config["dangerous_mlp_neurons"]),
            "thresholds": self.policy_config["attribution_thresholds"]
        }
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update policy thresholds dynamically."""
        self.policy_config["attribution_thresholds"].update(new_thresholds)
    
    def add_dangerous_circuit(self, circuit_type: str, layer: int, component: int):
        """Add a new dangerous circuit to the policy."""
        if circuit_type == "attention_head":
            self.policy_config["dangerous_attention_heads"].append((layer, component))
        elif circuit_type == "mlp_neuron":
            # For simplicity, add as single neuron range
            self.policy_config["dangerous_mlp_neurons"].append((layer, (component, component)))
    
    def remove_dangerous_circuit(self, circuit_type: str, layer: int, component: int):
        """Remove a dangerous circuit from the policy."""
        if circuit_type == "attention_head":
            try:
                self.policy_config["dangerous_attention_heads"].remove((layer, component))
            except ValueError:
                pass
        elif circuit_type == "mlp_neuron":
            # Remove any ranges containing this neuron
            self.policy_config["dangerous_mlp_neurons"] = [
                (l, (start, end)) for l, (start, end) in self.policy_config["dangerous_mlp_neurons"]
                if not (l == layer and start <= component <= end)
            ]