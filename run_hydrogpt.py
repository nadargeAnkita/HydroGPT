"""
HydroGPT — Main Launcher
Starts the FastAPI backend and Streamlit UI together.

Usage:
    python run_hydrogpt.py
"""

import subprocess
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("""
╔══════════════════════════════════════════════╗
║         💧 HydroGPT — Starting Up...         ║
║  AI-Driven Hydropower Forecasting Platform   ║
╚══════════════════════════════════════════════╝
""")


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Check for Groq API key
if os.getenv("GROQ_API_KEY"):
    print("✅ Groq API Key detected — LLM enabled")
else:
    print("⚠️  No GROQ_API_KEY set — using smart fallback mode")
    print("   Get a free key at: https://console.groq.com\n")

# Check vector store exists
vector_store_path = os.path.join("src", "rag", "vector_store.pkl")
if not os.path.exists(vector_store_path):
    print("📚 Building RAG vector store...")
    subprocess.run([sys.executable, "-m", "src.rag.vector_store"])
    print("✅ Vector store ready\n")

# Start FastAPI server
print("🚀 Launching API Server (FastAPI)...")
api_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.app:app", "--reload", "--port", "8000"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# Wait for API to start
time.sleep(3)
print("✅ API Server running at http://127.0.0.1:8000")

# Start Streamlit UI
print("🎨 Launching Streamlit Dashboard...")
ui_process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "src/ui/app.py",
     "--server.port", "8501",
     "--server.headless", "true"]
)

print("""
╔══════════════════════════════════════════════╗
║           HydroGPT is Running! 🎉            ║
╠══════════════════════════════════════════════╣
║  Dashboard : http://localhost:8501           ║
║  API       : http://127.0.0.1:8000           ║
║  API Docs  : http://127.0.0.1:8000/docs      ║
╠══════════════════════════════════════════════╣
║  Press CTRL+C to stop                        ║
╚══════════════════════════════════════════════╝
""")

try:
    api_process.wait()
    ui_process.wait()
except KeyboardInterrupt:
    print("\n⏹ Shutting down HydroGPT...")
    api_process.terminate()
    ui_process.terminate()
    print("✅ HydroGPT stopped. Goodbye!")
    sys.exit(0)
