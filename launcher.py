"""
Simple Streamlit Dashboard Launcher
"""

import subprocess
import sys

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Streamlit dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard.\n")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "SmartCRM_app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it first using:")
        print("   pip install streamlit")

if __name__ == "__main__":
    launch_dashboard()
