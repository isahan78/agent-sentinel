#!/usr/bin/env python3
"""Launcher script for AgentSentinel Streamlit dashboard."""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['streamlit', 'plotly', 'pandas', 'torch', 'transformers']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements_streamlit.txt")
        return False
    
    return True

def run_dashboard():
    """Launch the Streamlit dashboard."""
    if not check_dependencies():
        sys.exit(1)
    
    print("🚀 Launching AgentSentinel Dashboard...")
    print("📱 The dashboard will open in your browser")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    run_dashboard()
