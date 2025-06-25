#!/usr/bin/env python3
"""Quick test of our AgentSentinel model loading."""

import sys
import os
sys.path.append('.')

try:
    from agentsentinel.models.model_loader import load_model
    print("✅ AgentSentinel import successful")
    
    print("🚀 Testing model loading...")
    model, tokenizer = load_model("gpt2")
    
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Tokenizer type: {type(tokenizer)}")
    
    # Quick test
    inputs = tokenizer("Hello world", return_tensors="pt")
    print(f"✅ Tokenization works: {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=15, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0])
    print(f"✅ Generation works: '{result}'")
    
    print("🎉 AgentSentinel model loading is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
