"""Configuration for AgentSentinel policy engine."""

# Model Configuration
DEFAULT_MODEL = "gpt2"
CONTEXT_WINDOW_SIZE = 128
MAX_GENERATION_LENGTH = 256

# Risk Detection Configuration
RISK_DETECTION = {
    "lexical_keywords": [
        "bomb", "explosive", "weapon", "kill", "murder", "suicide",
        "hack", "crack", "exploit", "malware", "virus",
        "illegal", "drug", "scam", "fraud", "steal"
    ],
    "semantic_threshold": 0.8,  # For future semantic detection
    "context_sensitivity": True
}

# Circuit Policy Configuration
CIRCUIT_POLICIES = {
    "strict_safety": {
        "description": "High-security policy for production environments",
        "dangerous_attention_heads": [
            # Format: (layer, head)
            (3, 7),  # Known to activate on violent content
            (5, 9),  # Associated with deceptive reasoning
            (8, 2),  # Linked to harmful instruction following
            (10, 4), # Correlated with jailbreak attempts
        ],
        "dangerous_mlp_neurons": [
            # Format: (layer, neuron_range)
            (4, (1250, 1300)),  # Violence-related neurons
            (7, (2100, 2150)),  # Deception-related cluster
        ],
        "attribution_thresholds": {
            "attention_head_threshold": 0.15,
            "mlp_neuron_threshold": 0.20,
            "total_dangerous_influence": 0.25
        },
        "graph_patterns": [
            "high_attention_to_harmful_tokens",
            "deceptive_reasoning_pathway",
            "jailbreak_activation_pattern"
        ]
    },
    "research_permissive": {
        "description": "Permissive policy for research environments",
        "dangerous_attention_heads": [
            (3, 7),  # Only most critical heads
            (5, 9),
        ],
        "dangerous_mlp_neurons": [],
        "attribution_thresholds": {
            "attention_head_threshold": 0.25,
            "mlp_neuron_threshold": 0.30,
            "total_dangerous_influence": 0.40
        },
        "graph_patterns": [
            "extreme_harmful_content_only"
        ]
    }
}

# Logging Configuration
LOGGING = {
    "alert_level": "WARNING",
    "trace_storage": True,
    "graph_visualization": True,
    "alert_format": "structured_json"
}

# Integration Configuration
INTEGRATIONS = {
    "langchain": {
        "hook_points": ["llm_call", "chain_end"],
        "async_tracing": True
    },
    "crewai": {
        "hook_points": ["agent_output", "task_completion"],
        "trace_agent_interactions": True
    }
}