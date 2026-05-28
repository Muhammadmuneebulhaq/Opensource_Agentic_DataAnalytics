#!/usr/bin/env python3
"""
Quick fix validation - checks that all imports work with new LangGraph version.
"""

print("=" * 60)
print("LangGraph 0.1.1+ Compatibility Check")
print("=" * 60)

# Test 1: Core LangGraph imports
print("\n✓ Testing LangGraph imports...")
try:
    from langgraph.graph import StateGraph, START, END
    print("  ✓ StateGraph, START, END imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 2: Message types
print("\n✓ Testing LangChain message imports...")
try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    print("  ✓ Message types imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 3: Analysis tools
print("\n✓ Testing analyticTools imports...")
try:
    from analyticTools import (
        AnalysisSession,
        ingest_source,
        load_csv_tool,
        get_dataframe_info_tool,
        execute_code_tool,
        ingest_source_tool,
        embed_documents_tool,
        search_documents_tool
    )
    print("  ✓ All analysis tools imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 4: Agent graph
print("\n✓ Testing agent graph creation...")
try:
    from langGraphAgent import analysis_graph, invoke_agent
    print("  ✓ Agent graph created successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 5: AnalysisSession functionality
print("\n✓ Testing AnalysisSession...")
try:
    session = AnalysisSession()
    assert session.df is None
    assert len(session.documents) == 0
    assert len(session.conversation_history) == 0
    print("  ✓ AnalysisSession initialized correctly")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 6: Streamlit imports (check compatibility)
print("\n✓ Testing Streamlit compatibility...")
try:
    import streamlit as st
    print("  ✓ Streamlit available")
except Exception as e:
    print(f"  ✗ Warning: Streamlit not available: {e}")

# Test 7: Vector DB imports
print("\n✓ Testing vector database imports...")
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    print("  ✓ ChromaDB and sentence-transformers available")
except Exception as e:
    print(f"  ✗ Warning: Vector DB libraries not available: {e}")

print("\n" + "=" * 60)
print("✓ All core compatibility checks passed!")
print("=" * 60)
print("\nYou can now run:")
print("  streamlit run streamlit_app.py")
print("\nOr CLI mode:")
print("  python langGraphAgent.py")
