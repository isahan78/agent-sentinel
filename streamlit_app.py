"""Streamlit dashboard for AgentSentinel AI safety monitoring."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional
import json

# Import AgentSentinel components
import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import AgentSentinel
try:
    from agentsentinel import SimpleCircuitTracer, RiskDetector, PolicyEngine, load_model
    from agentsentinel.core.context_extractor import ContextExtractor
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)
    
    # Show detailed debugging information
    st.error(f"""
    🚨 **AgentSentinel Import Error**
    
    **Error**: {e}
    
    **Current Working Directory**: `{os.getcwd()}`
    
    **Python Path**: 
    """)
    
    for i, path in enumerate(sys.path[:5]):
        st.code(f"{i}: {path}")
    
    st.markdown("**File Structure Check:**")
    
    # Check what files exist
    files_to_check = [
        'setup.py',
        'config.py', 
        'agentsentinel/__init__.py',
        'agentsentinel/core/__init__.py',
        'agentsentinel/core/detector.py',
        'agentsentinel/core/tracer.py',
        'agentsentinel/core/policy_engine.py',
        'agentsentinel/models/__init__.py',
        'agentsentinel/models/model_loader.py'
    ]
    
    status_list = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            status_list.append(f"✅ {file_path}")
        else:
            status_list.append(f"❌ {file_path} - MISSING")
    
    st.code("\n".join(status_list))
    
    st.markdown("""
    **Solutions to try:**
    
    1. **Install package**: 
       ```bash
       pip install -e .
       ```
    
    2. **Check you're in the right directory**:
       ```bash
       ls -la  # Should see setup.py and agentsentinel/
       ```
    
    3. **Create missing files** if any are marked as MISSING above
    
    4. **Try the standalone version**:
       ```bash
       streamlit run streamlit_standalone.py
       ```
    """)
    
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AgentSentinel Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .safe-card {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
    }
    .warning-card {
        background-color: #fffbeb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
    }
    .critical-card {
        background-color: #fef2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

@st.cache_resource
def load_agentsentinel_model(model_name: str):
    """Load and cache the model."""
    with st.spinner(f'Loading {model_name}...'):
        model, tokenizer = load_model(model_name)
        tracer = SimpleCircuitTracer(model, tokenizer, policy_config="strict_safety")
        detector = RiskDetector()
        policy_engine = PolicyEngine("strict_safety")
    return model, tokenizer, tracer, detector, policy_engine

def create_attention_heatmap(attribution_scores: Dict) -> go.Figure:
    """Create attention heatmap visualization."""
    attention_heads = attribution_scores.get('attention_heads', {})
    
    if not attention_heads:
        fig = go.Figure()
        fig.add_annotation(text="No attention data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Extract layers and heads
    layers = []
    heads = []
    scores = []
    
    for (layer, head), score in attention_heads.items():
        layers.append(layer)
        heads.append(head)
        scores.append(score)
    
    # Create matrix for heatmap
    max_layer = max(layers) if layers else 0
    max_head = max(heads) if heads else 0
    
    matrix = np.zeros((max_layer + 1, max_head + 1))
    
    for (layer, head), score in attention_heads.items():
        matrix[layer, head] = score
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"Head {i}" for i in range(max_head + 1)],
        y=[f"Layer {i}" for i in range(max_layer + 1)],
        colorscale='Reds',
        colorbar=dict(title="Attribution Score")
    ))
    
    fig.update_layout(
        title="Attention Head Attribution Scores",
        xaxis_title="Attention Heads",
        yaxis_title="Layers",
        height=400
    )
    
    return fig

def create_risk_timeline(history: List[Dict]) -> go.Figure:
    """Create timeline of risk detections."""
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="No analysis history yet", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.scatter(df, x='timestamp', y='max_risk_score', 
                     color='alert_level', size='total_attribution',
                     hover_data=['risky_token', 'prompt'],
                     title="Risk Detection Timeline")
    
    fig.update_layout(height=300)
    return fig

def create_circuit_network(dangerous_circuits: List, attribution_scores: Dict) -> go.Figure:
    """Create network graph of dangerous circuits."""
    if not dangerous_circuits:
        fig = go.Figure()
        fig.add_annotation(text="No dangerous circuits detected", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create nodes and edges for circuit visualization
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for i, (circuit_type, layer, component) in enumerate(dangerous_circuits[:10]):  # Limit to 10
        angle = 2 * np.pi * i / len(dangerous_circuits[:10])
        x = np.cos(angle)
        y = np.sin(angle)
        
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{circuit_type}<br>L{layer}C{component}")
        
        # Color by attribution score
        score = attribution_scores.get('attention_heads', {}).get((layer, component), 0)
        node_color.append(score)
    
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=20, color=node_color, colorscale='Reds', showscale=True),
        text=node_text,
        textposition="middle center",
        name="Dangerous Circuits"
    ))
    
    fig.update_layout(
        title="Dangerous Circuit Network",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

# Main app
def main():
    st.title("🔬 AgentSentinel AI Safety Dashboard")
    st.markdown("**Real-time mechanistic analysis and policy enforcement for AI systems**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["gpt2", "gpt2-medium", "gpt2-large"],
            help="Choose the language model for analysis"
        )
        
        # Policy selection
        policy_name = st.selectbox(
            "Safety Policy",
            ["strict_safety", "research_permissive"],
            help="Choose the safety policy configuration"
        )
        
        # Load model
        if st.button("🚀 Load Model"):
            try:
                model, tokenizer, tracer, detector, policy_engine = load_agentsentinel_model(model_name)
                st.session_state.model_loaded = True
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.tracer = tracer
                st.session_state.detector = detector
                st.session_state.policy_engine = policy_engine
                st.success(f"✅ {model_name} loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
        
        # Model status
        if st.session_state.model_loaded:
            st.success("🟢 Model Ready")
            
            # Policy info
            policy_summary = st.session_state.policy_engine.get_policy_summary()
            st.info(f"**Policy**: {policy_summary['policy_name']}\n\n"
                   f"**Dangerous Heads**: {policy_summary['dangerous_heads_count']}\n\n"
                   f"**Thresholds**: {policy_summary['thresholds']['attention_head_threshold']}")
        else:
            st.warning("🟡 Load model to begin")
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("👈 Please load a model from the sidebar to start analyzing AI safety")
        
        # Show example
        st.subheader("🎯 What AgentSentinel Does")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Real-time Risk Detection**
            - Scans model outputs for dangerous content
            - Traces risky tokens to neural circuits
            - Evaluates against safety policies
            """)
        
        with col2:
            st.markdown("""
            **⚡ Circuit-Level Analysis**  
            - Identifies which attention heads caused risks
            - Shows attribution scores for transparency
            - Enables intervention at the source
            """)
        
        return
    
    # Input section
    st.subheader("📝 Text Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter prompt for analysis:",
            value="Tell me about computer security",
            height=100,
            help="The model will generate text from this prompt and analyze it for safety violations"
        )
    
    with col2:
        st.markdown("**Generation Settings**")
        max_length = st.slider("Max Length", 10, 100, 30)
        analyze_button = st.button("🔬 Analyze", type="primary")
    
    # Analysis
    if analyze_button and prompt:
        with st.spinner("Generating and analyzing..."):
            try:
                # Generate and analyze
                generated_text, trace_result = st.session_state.tracer.generate_and_analyze(
                    prompt, max_length=max_length
                )
                
                # Store in history
                analysis_data = {
                    'timestamp': time.time(),
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'trace_result': trace_result,
                    'max_risk_score': 0.0,
                    'alert_level': 'INFO',
                    'total_attribution': 0.0,
                    'risky_token': 'none'
                }
                
                if trace_result:
                    # Map alert levels to risk scores
                    risk_score_map = {"CRITICAL": 0.9, "WARNING": 0.7, "INFO": 0.5}
                    analysis_data.update({
                        'max_risk_score': risk_score_map.get(trace_result.alert_level, 0.5),
                        'alert_level': trace_result.alert_level,
                        'total_attribution': trace_result.attribution_scores.get('total_attribution', 0.0),
                        'risky_token': trace_result.risky_token
                    })
                
                st.session_state.analysis_history.append(analysis_data)
                
                # Display results
                st.subheader("📊 Analysis Results")
                
                # Generated text
                st.markdown("**🤖 Generated Text:**")
                st.text_area("", value=generated_text, height=100, disabled=True)
                
                if trace_result and trace_result.policy_violation:
                    # Alert banner
                    alert_color = {"INFO": "info", "WARNING": "warning", "CRITICAL": "error"}[trace_result.alert_level]
                    st.error(f"🚨 **POLICY VIOLATION DETECTED** - {trace_result.alert_level}")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Risky Token", f"'{trace_result.risky_token}'")
                    
                    with col2:
                        st.metric("Alert Level", trace_result.alert_level)
                    
                    with col3:
                        st.metric("Attribution Score", f"{trace_result.attribution_scores.get('total_attribution', 0):.3f}")
                    
                    with col4:
                        st.metric("Dangerous Circuits", len(trace_result.dangerous_circuits))
                    
                    # Visualizations
                    st.subheader("📈 Circuit Analysis")
                    
                    tab1, tab2, tab3 = st.tabs(["🔥 Attention Heatmap", "🔗 Circuit Network", "📋 Details"])
                    
                    with tab1:
                        fig = create_attention_heatmap(trace_result.attribution_scores)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        fig = create_circuit_network(trace_result.dangerous_circuits, trace_result.attribution_scores)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        st.json({
                            "trace_id": trace_result.trace_id,
                            "policy_name": trace_result.policy_name,
                            "dangerous_circuits": trace_result.dangerous_circuits,
                            "attribution_scores": trace_result.attribution_scores
                        })
                
                else:
                    st.success("✅ **No policy violations detected** - Generated text appears safe")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # History section
    if st.session_state.analysis_history:
        st.subheader("📊 Analysis History")
        
        # Timeline chart
        fig = create_risk_timeline(st.session_state.analysis_history)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent analyses table
        st.markdown("**Recent Analyses:**")
        
        recent_data = []
        for item in st.session_state.analysis_history[-5:]:  # Last 5
            recent_data.append({
                "Timestamp": pd.to_datetime(item['timestamp'], unit='s').strftime('%H:%M:%S'),
                "Prompt": item['prompt'][:50] + "..." if len(item['prompt']) > 50 else item['prompt'],
                "Alert Level": item['alert_level'],
                "Risky Token": item['risky_token'],
                "Attribution": f"{item['total_attribution']:.3f}"
            })
        
        if recent_data:
            df = pd.DataFrame(recent_data)
            st.dataframe(df, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

if __name__ == "__main__":
    main()
    