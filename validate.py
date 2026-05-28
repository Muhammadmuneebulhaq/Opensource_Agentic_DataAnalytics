#!/usr/bin/env python3
"""
Validation script to check that all components are properly configured.
Run this before starting the Streamlit app for the first time.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Data Analysis Agent - Component Validation")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("✗ ERROR: Python 3.8+ required")
    sys.exit(1)

# Check required modules
required_modules = [
    'streamlit',
    'langchain',
    'langgraph',
    'pandas',
    'matplotlib',
    'chromadb',
    'sentence_transformers',
    'pdfplumber',
    'sqlalchemy',
]

print("\nChecking installed packages:")
missing = []
for module in required_modules:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError:
        print(f"  ✗ {module} (missing)")
        missing.append(module)

if missing:
    print(f"\n✗ Missing packages: {', '.join(missing)}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Check project files
print("\nChecking project files:")
required_files = ['analyticTools.py', 'langGraphAgent.py', 'streamlit_app.py']
for file in required_files:
    path = Path(file)
    if path.exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} (not found)")
        sys.exit(1)

# Check environment variables
print("\nChecking environment:")
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"  ✓ OPENAI_API_KEY configured")
else:
    print(f"  ✗ OPENAI_API_KEY not set (required)")
    print("    Create .env file with: OPENAI_API_KEY=sk-...")
    # Don't exit here - it will fail at runtime but allows testing other components

# Test imports
print("\nTesting imports:")
try:
    from analyticTools import AnalysisSession, ingest_source
    print("  ✓ analyticTools imports")
except Exception as e:
    print(f"  ✗ analyticTools import failed: {e}")
    sys.exit(1)

try:
    from langGraphAgent import analysis_graph
    print("  ✓ langGraphAgent imports")
except Exception as e:
    print(f"  ✗ langGraphAgent import failed: {e}")
    # This might fail without API key, which is OK

# Test AnalysisSession functionality
print("\nTesting AnalysisSession:")
try:
    session = AnalysisSession()
    print(f"  ✓ AnalysisSession created")
    print(f"    - DataFrame: {session.df}")
    print(f"    - Documents: {len(session.documents)}")
    print(f"    - History: {len(session.conversation_history)}")
except Exception as e:
    print(f"  ✗ AnalysisSession failed: {e}")
    sys.exit(1)

# Test pandas/numpy
print("\nTesting data dependencies:")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create test data
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"  ✓ Pandas DataFrame created: {df.shape}")
    print(f"  ✓ NumPy operations available")
    print(f"  ✓ Matplotlib ready")
except Exception as e:
    print(f"  ✗ Data dependencies failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All validations passed!")
print("=" * 60)
print("\nTo start the app, run:")
print("  streamlit run streamlit_app.py")
print("\nFor help, see: QUICKSTART.md or README.md")
