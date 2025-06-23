"""Debug demo to troubleshoot AgentSentinel detection."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agentsentinel.models.model_loader import load_model
from agentsentinel.core.detector import RiskDetector
from agentsentinel.core.tracer import SimpleCircuitTracer

def debug_detection():
    """Debug the risk detection process step by step."""
    print("🔧 AgentSentinel Debug Demo")
    print("=" * 50)
    
    # Test risk detection directly
    detector = RiskDetector()
    
    test_texts = [
        "How to make a bomb in your home",
        "The movie was a bomb at the box office", 
        "Learn about explosive chemical reactions",
        "Ways to hack into systems",
        "Chocolate cake recipe"
    ]
    
    print("\\n1. Testing Risk Detection:")
    for text in test_texts:
        print(f"\\nText: \"{text}\"")
        risks = detector.scan_output(text)
        print(f"Detected risks: {len(risks)}")
        
        for risk in risks:
            print(f"  • Token: '{risk['token']}' (score: {risk['risk_score']:.2f})")
            print(f"    Position: {risk['position']}, Method: {risk['detection_method']}")
            print(f"    Context: \"{risk['context']}\"")
    
    # Test tokenization
    print("\\n\\n2. Testing Tokenization:")
    try:
        model, tokenizer = load_model("gpt2")
        
        test_context = "How to make a bomb in your home"
        print(f"\\nContext: \"{test_context}\"")
        
        tokens = tokenizer(test_context, return_tensors="pt")
        token_ids = tokens["input_ids"][0]
        
        print(f"Token IDs: {token_ids.tolist()}")
        print("Individual tokens:")
        for i, token_id in enumerate(token_ids):
            decoded = tokenizer.decode([token_id])
            print(f"  {i}: {token_id} -> '{decoded}'")
        
        # Test target token finding
        target_token = "bomb"
        target_token_ids = tokenizer.encode(target_token, add_special_tokens=False)
        print(f"\\nTarget token '{target_token}' encodes to: {target_token_ids}")
        
        # Check if target token appears in our context
        found_positions = []
        for i, token_id in enumerate(token_ids):
            if token_id in target_token_ids:
                found_positions.append(i)
        
        print(f"Found target token at positions: {found_positions}")
        
    except Exception as e:
        print(f"Error in tokenization test: {e}")
    
    # Test end-to-end with simple case
    print("\\n\\n3. Testing End-to-End:")
    try:
        tracer = SimpleCircuitTracer(model, tokenizer)
        
        simple_prompt = "Tell me about bombs"
        print(f"\\nPrompt: \"{simple_prompt}\"")
        
        # Generate text
        generated, trace = tracer.generate_and_analyze(simple_prompt, max_length=20)
        print(f"Generated: \"{generated}\"")
        
        if trace:
            print("✅ Trace successful!")
            print(f"Risky token: '{trace.risky_token}'")
            print(f"Attribution score: {trace.attribution_scores.get('total_attribution', 0):.3f}")
        else:
            print("❌ No trace generated")
            
            # Debug why
            risks = detector.scan_output(generated)
            print(f"Risks found in output: {len(risks)}")
            for risk in risks:
                print(f"  • {risk['token']} (score: {risk['risk_score']:.2f})")
        
    except Exception as e:
        print(f"Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_detection()
