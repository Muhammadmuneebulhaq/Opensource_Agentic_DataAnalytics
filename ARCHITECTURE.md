# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              STREAMLIT WEB APPLICATION                      │
│  (streamlit_app.py) - Multi-page UI with st.navigation      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Page 1: Connect Data      Page 2: Chat     Page 3: Metrics │
│  ✓ File Upload            ✓ LLM Chat       ✓ Auto-KPIs      │
│  ✓ DB Connection          ✓ Real-time      ✓ Visualizations │
│  ✓ Data Preview             responses      ✓ Analytics      │
└─────────────────────────────────────────────────────────────┘
            │                      │                    │
            │                      │                    │
            └──────────┬───────────┴────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   st.session_state           │
        │ AnalysisSession object       │
        │ • df (DataFrame)             │
        │ • documents (Dict)           │
        │ • conversation_history       │
        │ • chroma_collection          │
        │ • source_files               │
        └──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            ANALYSIS ENGINE                                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AnalysisSession (analyticTools.py)                 │    │
│  │ ════════════════════════════════════════            │    │
│  │ Core state management                               │    │
│  │ • load_csv()          - Load CSV files             │    │
│  │ • get_dataframe_info() - Get DataFrame metadata    │    │
│  │ • execute_code()       - Run Python analysis       │    │
│  │ • _chunk_text()        - Split docs into chunks    │    │
│  │ • embed_documents()    - Index with embeddings     │    │
│  │ • search_documents_in_session() - Semantic search  │    │
│  └─────────────────────────────────────────────────────┘    │
│                       │                                      │
│                       ▼                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ DATA SOURCE INTEGRATION                            │    │
│  │ ════════════════════════════════════════            │    │
│  │ ingest_source() function:                          │    │
│  │ • CSV/Excel → pandas DataFrame                    │    │
│  │ • JSON → json_normalize → DataFrame               │    │
│  │ • PDF → pdfplumber → text blocks                  │    │
│  │ • Database → SQLAlchemy → DataFrame               │    │
│  └─────────────────────────────────────────────────────┘    │
│                       │                                      │
└───────────┬───────────┴───────────┬─────────────────────────┘
            │                       │
            │                       ▼
            │             ┌──────────────────────┐
            │             │ ChromaDB             │
            │             │ Vector Database      │
            │             │ • Stores embeddings  │
            │             │ • Cosine similarity  │
            │             │ • Chunk metadata     │
            │             └──────────────────────┘
            │                       │
            └───────────┬───────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│            LANGGRAPH AGENT (langGraphAgent.py)              │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ LangGraph Workflow                                 │    │
│  │ ════════════════════════════════════════            │    │
│  │                                                     │    │
│  │   START                                            │    │
│  │     │                                              │    │
│  │     ▼                                              │    │
│  │  [Agent Node]                                      │    │
│  │   Claude GPT-4o-mini                              │    │
│  │   • Analyzes user query                           │    │
│  │   • Selects tools to call                         │    │
│  │     │                                              │    │
│  │     ├──→ router                                    │    │
│  │     │      │                                       │    │
│  │     │      ├─→ has_tools?  ──→ Tool Node          │    │
│  │     │      └─→ no_tools?   ──→ END                │    │
│  │     │                                              │    │
│  │     ▼                                              │    │
│  │  [Tool Node]                                       │    │
│  │   Executes:                                        │    │
│  │   • load_csv_tool                                 │    │
│  │   • get_dataframe_info_tool                       │    │
│  │   • execute_code_tool                             │    │
│  │   • ingest_source_tool                            │    │
│  │   • embed_documents_tool                          │    │
│  │   • search_documents_tool                         │    │
│  │     │                                              │    │
│  │     └──→ [Agent Node]  ← (result)                │    │
│  │          (loop until END)                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                       │                                      │
└───────────────────────┼──────────────────────────────────────┘
                        │
                        ▼
            OpenAI API (gpt-4o-mini)
            • Claude reasoning
            • Code generation
            • Query understanding
```

## Data Flow

### 1. File Upload Flow

```
User uploads file
    ↓
streamlit file_uploader widget
    ↓
save to /tmp/
    ↓
ingest_source(path, session)
    ├─ CSV → pd.read_csv() → session.df
    ├─ Excel → pd.read_excel() → session.df
    ├─ JSON → json.load() + json_normalize() → session.df
    ├─ PDF → pdfplumber.open() + extract_text() → session.documents
    └─ DB → SQLAlchemy + pd.read_sql() → session.df
    ↓
Update st.session_state
    ↓
Display in KPI view + Chat available
```

### 2. Chat Query Flow

```
User types question
    ↓
Extract from st.chat_input()
    ↓
Add to st.session_state.messages
    ↓
analysis_graph.invoke({"messages": messages})
    ↓
[Agent Node]
    • Claude analyzes query
    • Determines tools needed
    • Binds analyzed request
    ↓
[Tool Node]
    • execute_code_tool → runs Python
    • search_documents_tool → queries ChromaDB
    • ingest_source_tool → loads new data
    ↓
Returns result to [Agent Node]
    ↓
Claude formats response
    ↓
Display in st.chat_message()
    ↓
If matplotlib chart exists:
    • Extract code block from response
    • Execute code
    • Display with st.pyplot()
```

### 3. Document Search Flow

```
User uploads PDF
    ↓
ingest_source() → session.documents[path] = full_text
    ↓
User clicks "Embed & Index PDF"
    ↓
session.embed_documents()
    ├─ _chunk_text(doc, 500 words, 50 overlap)
    ├─ Load SentenceTransformer('all-MiniLM-L6-v2')
    ├─ For each chunk:
    │   ├─ Generate embedding (384-dim vector)
    │   ├─ Store in ChromaDB:
    │   │   • id: unique chunk ID
    │   │   • embedding: vector
    │   │   • document: text
    │   │   • metadata: source + chunk_idx
    └─ session.chroma_collection ready
    ↓
User asks question
    ↓
Agent calls search_documents(query)
    ├─ Encode query → same model
    ├─ ChromaDB.query(query_embedding, n_results=3)
    ├─ Return top 3 chunks by cosine similarity
    └─ Format for display
    ↓
Display in chat with similarity scores
```

## Component Relationships

```
┌─────────────────────────────────┐
│ streamlit_app.py                │
│ • Page routing                  │
│ • UI components                 │
│ • Session state management      │
└────────────┬────────────────────┘
             │
             ├──────────────────────────────────┐
             │                                  │
             ▼                                  ▼
    ┌───────────────────┐            ┌──────────────────┐
    │ analyticTools.py  │            │ langGraphAgent.py│
    │                   │            │                  │
    │ • AnalysisSession │            │ • analysis_graph │
    │ • ingest_source   │            │ • workflow       │
    │ • Tool wrappers   │            │ • agent_model    │
    └─────────┬─────────┘            └────────┬─────────┘
              │                               │
              │                    Uses       │
              │                   ────────────┘
              │
              ├──────────── imports ─────────────┐
              │                                  │
              ▼                                  ▼
    ┌──────────────────┐              ┌────────────────┐
    │ External APIs    │              │ ChromaDB       │
    │ • pandas         │              │ Vector Store   │
    │ • pdfplumber     │              │                │
    │ • SQLAlchemy     │              └────────────────┘
    └──────────────────┘

              Uses
    ┌──────────────────────────────────┐
    │ OpenAI / LangChain / LangGraph   │
    │ • GPT-4o-mini model              │
    │ • Tool binding                   │
    │ • Message routing                │
    └──────────────────────────────────┘
```

## Key Technologies

| Layer             | Technology             | Purpose                              |
| ----------------- | ---------------------- | ------------------------------------ |
| **Frontend**      | Streamlit              | Web UI with multi-page navigation    |
| **State**         | st.session_state       | Shared AnalysisSession across pages  |
| **Data**          | Pandas, NumPy          | Data manipulation and analysis       |
| **AI/LLM**        | LangGraph, LangChain   | Agent orchestration and tool binding |
| **Model**         | OpenAI (GPT-4o-mini)   | Reasoning and code generation        |
| **Documents**     | ChromaDB               | Vector database for semantic search  |
| **Embeddings**    | Sentence-Transformers  | Generate and search embeddings       |
| **Files**         | pdfplumber, SQLAlchemy | Multi-format data ingestion          |
| **Visualization** | Matplotlib             | Chart generation                     |

## Session State Structure

```python
st.session_state = {
    "session": AnalysisSession {
        "df": DataFrame,                    # Main loaded data
        "documents": {                       # Text from PDFs
            "file.pdf": "extracted text..."
        },
        "source_files": ["file.csv", "file.pdf"],  # Tracking
        "conversation_history": [            # Audit trail
            {
                "timestamp": "2024-01-15T...",
                "type": "tool_call|error",
                "action": "load_csv",
                "details": {...},
                "result": "..."
            }
        ],
        "chroma_collection": ChromaDBCollection,   # Embedded docs
        "document_chunks": [                 # Chunk metadata
            {"id": "file.pdf_0", "text": "...", "source": "...", "chunk_idx": 0}
        ]
    },
    "messages": [
        SystemMessage(...),
        HumanMessage("question"),
        AIMessage("response"),
        ...
    ],
    "agent_responses": []
}
```

## Execution Timeline

1. **App Start** (streamlit_app.py)
   - Initialize AnalysisSession
   - Load system prompt
   - Create navigation
   - Show Page 1

2. **User Uploads File** (Page 1)
   - File saved to /tmp/
   - ingest_source() called
   - session.df or session.documents populated
   - History updated

3. **User Asks Question** (Page 2)
   - Message added to st.session_state.messages
   - analysis_graph.invoke({messages}) called
   - LangGraph routes to tools
   - Tools execute (read df, execute code, search docs)
   - Agent formats response
   - Display in chat + charts

4. **View Metrics** (Page 3)
   - Auto-detect numeric/date columns
   - Generate statistics
   - Create visualizations
   - Update on data load

---

## Why This Architecture?

✓ **Separation of Concerns**: UI (Streamlit), Logic (analyticTools), Agent (LangGraph)  
✓ **Shared State**: AnalysisSession persists across pages via st.session_state  
✓ **Flexible Ingestion**: Support multiple formats with single ingest_source()  
✓ **Semantic Search**: ChromaDB + embeddings for intelligent PDF search  
✓ **Tool-Driven**: Agent dynamically selects tools based on query  
✓ **Transparent History**: All operations logged in conversation_history  
✓ **Extensible**: Easy to add new tools or data sources
