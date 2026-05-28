from langchain_core.tools import Tool
import pandas as pd, matplotlib.pyplot as plt, numpy as np, io, contextlib, sys
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os
from pathlib import Path


class AnalysisSession:
    """
    Manages an analysis session with a DataFrame, loaded source files, and conversation history.
    
    This encapsulates all state related to a single analysis, allowing for:
    - Multiple independent analysis sessions
    - Clear state management and dependencies
    - Easier testing and debugging
    - Thread-safe operations (potential future enhancement)
    """
    
    def __init__(self):
        """Initialize an empty analysis session."""
        self.df: Optional[pd.DataFrame] = None
        self.source_files: List[str] = []
        self.documents: Dict[str, str] = {}  # Store extracted text from PDFs, etc.
        self.conversation_history: List[Dict[str, Any]] = []
        self.chroma_collection = None  # Will be lazily initialized for embedding storage
        self.document_chunks: List[Dict[str, Any]] = []  # Store chunks: {text, source, chunk_id}
    
    def load_csv(self, filename: str) -> str:
        """Load a CSV file into the session's DataFrame."""
        try:
            df = pd.read_csv(filename)
            self.df = df
            self.source_files.append(filename)
            
            # Record action in conversation history
            self._add_to_history(
                "tool_call",
                "load_csv",
                {"filename": filename},
                f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns"
            )
            
            return f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns.\nColumns: {list(df.columns)}"
        except Exception as e:
            self._add_to_history("error", "load_csv", {"filename": filename}, str(e))
            return f"Error loading file: {e}"
    
    def get_dataframe_info(self) -> str:
        """Get information about the session's DataFrame."""
        if self.df is None:
            return "No DataFrame currently loaded. Please load a CSV file first."
        
        try:
            info_parts = []
            info_parts.append(f"DataFrame Shape: {self.df.shape}")
            info_parts.append(f"Columns: {list(self.df.columns)}")
            info_parts.append("\nColumn Data Types:")
            info_parts.append(str(self.df.dtypes))
            info_parts.append(f"\nFirst few rows:")
            info_parts.append(str(self.df.head(3)))
            
            # Check for null values
            null_counts = self.df.isnull().sum()
            if null_counts.sum() > 0:
                info_parts.append(f"\nNull values per column:")
                info_parts.append(str(null_counts[null_counts > 0]))
            else:
                info_parts.append(f"\nNo null values found.")
            
            result = "\n".join(info_parts)
            self._add_to_history("tool_call", "get_dataframe_info", {}, "Retrieved DataFrame info")
            return result
        except Exception as e:
            self._add_to_history("error", "get_dataframe_info", {}, str(e))
            return f"Error getting DataFrame info: {e}"
    
    def execute_code(self, code: str) -> str:
        """Execute Python code with access to the session's DataFrame."""
        if self.df is None:
            return "Error: No DataFrame loaded. Please load a CSV file first."
        
        exec_locals = {
            "pd": pd, 
            "plt": plt, 
            "np": np, 
            "df": self.df
        }
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                # Configure matplotlib for better output
                plt.style.use('default')
                plt.rcParams['figure.figsize'] = (10, 6)
                
                exec(code, {}, exec_locals)
                
                # Show any plots that were created
                if plt.get_fignums():
                    plt.tight_layout()
                    plt.show()
                    
        except Exception as e:
            error_msg = error_buffer.getvalue()
            self._add_to_history("error", "execute_code", {"code": code}, str(e))
            if error_msg:
                return f"Error executing code: {e}\nDetails: {error_msg}"
            else:
                return f"Error executing code: {e}"
        finally:
            plt.close('all')  # Clean up plots
        
        result = output_buffer.getvalue()
        error_output = error_buffer.getvalue()
        
        if error_output:
            result += f"\nWarnings/Errors: {error_output}"
        
        self._add_to_history("tool_call", "execute_code", {"code": code[:100]}, "Code executed")
        return result if result.strip() else "Code executed successfully (no output produced)"
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks of ~chunk_size words.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in words
            overlap: Number of words to overlap between chunks
        
        Returns:
            List of text chunks
        
        Why chunking matters:
        - Embeddings have token limits (sentence-transformers: ~384 tokens max)
        - Large documents lose semantic coherence when embedded as a single vector
        - Overlapping chunks preserve context at chunk boundaries
        - Enables semantic search on specific passages, not whole documents
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def embed_documents(self) -> str:
        """
        Embed all documents in session.documents and store in ChromaDB.
        Call this after loading PDFs/text documents.
        
        Returns:
            Status message with number of chunks created and embedded
        """
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            return f"✗ Missing dependency: {str(e)}\nInstall with: pip install chromadb sentence-transformers"
        
        if not self.documents:
            return "✗ No documents loaded. Load a PDF or text source first."
        
        try:
            # Initialize ChromaDB collection (in-memory, persists for session)
            if self.chroma_collection is None:
                client = chromadb.Client()
                # Use a unique collection name per session
                self.chroma_collection = client.get_or_create_collection(
                    name=f"documents_{id(self)}",
                    metadata={"hnsw:space": "cosine"}
                )
            
            # Load embedding model (lightweight, fast)
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Clear existing chunks and collection
            self.document_chunks = []
            self.chroma_collection.delete_where({"source": {"$ne": ""}})  # Delete all
            
            chunk_count = 0
            
            # Process each document
            for doc_source, doc_text in self.documents.items():
                chunks = self._chunk_text(doc_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_source}_{chunk_idx}"
                    
                    # Generate embedding
                    embedding = model.encode(chunk).tolist()
                    
                    # Store chunk metadata
                    self.document_chunks.append({
                        "id": chunk_id,
                        "text": chunk,
                        "source": doc_source,
                        "chunk_idx": chunk_idx
                    })
                    
                    # Add to ChromaDB
                    self.chroma_collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{"source": doc_source, "chunk_idx": chunk_idx}]
                    )
                    
                    chunk_count += 1
            
            self._add_to_history(
                "tool_call", "embed_documents", {},
                f"Embedded {chunk_count} chunks from {len(self.documents)} documents"
            )
            
            return f"✓ Embedded {chunk_count} chunks from {len(self.documents)} document(s)"
        
        except Exception as e:
            self._add_to_history("error", "embed_documents", {}, str(e))
            return f"✗ Error embedding documents: {str(e)}"
    
    def search_documents_in_session(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search embedded documents for most relevant chunks.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
        
        Returns:
            List of relevant chunk dicts with text, source, similarity score
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return []
        
        if self.chroma_collection is None:
            return []
        
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query).tolist()
            
            # Search ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if not results['ids'] or not results['ids'][0]:
                return []
            
            # Format results
            search_results = []
            for i, chunk_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if 'distances' in results else 0
                # Convert distance to similarity (cosine similarity: 1 - distance)
                similarity = 1 - distance if distance is not None else 0
                
                search_results.append({
                    "text": results['documents'][0][i],
                    "source": results['metadatas'][0][i]['source'],
                    "chunk_idx": results['metadatas'][0][i]['chunk_idx'],
                    "similarity_score": float(similarity)
                })
            
            self._add_to_history(
                "tool_call", "search_documents",
                {"query": query, "top_k": top_k},
                f"Found {len(search_results)} relevant chunks"
            )
            
            return search_results
        
        except Exception as e:
            self._add_to_history("error", "search_documents", {"query": query}, str(e))
            return []
    
    def _add_to_history(self, event_type: str, action: str, details: Dict[str, Any], result: str) -> None:
        """Record an action in the conversation history."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "action": action,
            "details": details,
            "result": result
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Return the conversation history."""
        return self.conversation_history
    
    def get_source_files(self) -> List[str]:
        """Return the list of loaded source files."""
        return self.source_files
    
    def get_documents(self) -> Dict[str, str]:
        """Return the extracted documents."""
        return self.documents


def ingest_source(filepath_or_connection: str, session: AnalysisSession) -> str:
    """
    Ingest data from various sources into an AnalysisSession.
    
    Supports:
    - CSV/Excel files (.csv, .xlsx, .xls) -> loads into session.df
    - JSON files (.json) -> flattens with json_normalize, loads into session.df
    - PDF files (.pdf) -> extracts text, stores in session.documents
    - Database connection strings -> queries database, loads into session.df
    
    Args:
        filepath_or_connection: File path or database connection string
        session: AnalysisSession object to populate
    
    Returns:
        Status message describing what was loaded
    
    Library choices explained:
    - pandas: Standard for tabular data (CSV, Excel, JSON). Efficient, well-documented.
    - pandas.json_normalize: Flattens nested JSON into tabular format automatically.
    - pdfplumber: Lightweight PDF text extraction with good accuracy. Easier than PyPDF2 for text-only use cases.
    - sqlalchemy: Industry-standard Python ORM/connection layer. Supports 20+ database engines without rewriting code.
    """
    try:
        # Normalize path and check if it's a file
        is_file = os.path.isfile(filepath_or_connection)
        
        if is_file:
            file_path = Path(filepath_or_connection)
            suffix = file_path.suffix.lower()
            
            # CSV files
            if suffix == '.csv':
                df = pd.read_csv(filepath_or_connection)
                session.df = df
                session.source_files.append(filepath_or_connection)
                session._add_to_history(
                    "tool_call", "ingest_source",
                    {"source": filepath_or_connection, "type": "csv"},
                    f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns"
                )
                return f"✓ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns\nColumns: {list(df.columns)}"
            
            # Excel files
            elif suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath_or_connection)
                session.df = df
                session.source_files.append(filepath_or_connection)
                session._add_to_history(
                    "tool_call", "ingest_source",
                    {"source": filepath_or_connection, "type": "excel"},
                    f"Loaded Excel: {df.shape[0]} rows, {df.shape[1]} columns"
                )
                return f"✓ Excel loaded: {df.shape[0]} rows, {df.shape[1]} columns\nColumns: {list(df.columns)}"
            
            # JSON files
            elif suffix == '.json':
                with open(filepath_or_connection, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    # List of objects -> normalize directly
                    df = pd.json_normalize(data)
                elif isinstance(data, dict):
                    # Single object -> wrap in list or navigate to data array
                    if 'data' in data and isinstance(data['data'], list):
                        df = pd.json_normalize(data['data'])
                    else:
                        df = pd.json_normalize([data])
                else:
                    return f"✗ JSON file format not supported. Expected list or object, got {type(data)}"
                
                session.df = df
                session.source_files.append(filepath_or_connection)
                session._add_to_history(
                    "tool_call", "ingest_source",
                    {"source": filepath_or_connection, "type": "json"},
                    f"Loaded JSON: {df.shape[0]} rows, {df.shape[1]} columns"
                )
                return f"✓ JSON loaded and flattened: {df.shape[0]} rows, {df.shape[1]} columns\nColumns: {list(df.columns)}"
            
            # PDF files
            elif suffix == '.pdf':
                try:
                    import pdfplumber
                except ImportError:
                    return "✗ pdfplumber not installed. Install with: pip install pdfplumber"
                
                text_content = []
                with pdfplumber.open(filepath_or_connection) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text:
                            text_content.append(f"--- Page {page_num} ---\n{text}")
                
                if not text_content:
                    return f"✗ No text extracted from PDF. File may be image-based or corrupted."
                
                full_text = "\n\n".join(text_content)
                session.documents[filepath_or_connection] = full_text
                session.source_files.append(filepath_or_connection)
                
                session._add_to_history(
                    "tool_call", "ingest_source",
                    {"source": filepath_or_connection, "type": "pdf"},
                    f"Extracted text from PDF: {len(pdf.pages)} pages, {len(full_text)} characters"
                )
                return f"✓ PDF processed: {len(pdf.pages)} pages, {len(full_text)} characters extracted"
            
            else:
                return f"✗ Unsupported file type: {suffix}. Supported: .csv, .xlsx, .xls, .json, .pdf"
        
        else:
            # Assume it's a database connection string
            # Format: "dialect+driver://user:password@host:port/database"
            # Example: "postgresql://user:pass@localhost/mydb"
            # Example: "sqlite:///path/to/database.db"
            
            try:
                from sqlalchemy import create_engine, inspect
            except ImportError:
                return "✗ SQLAlchemy not installed. Install with: pip install sqlalchemy"
            
            try:
                engine = create_engine(filepath_or_connection)
                
                # Test connection
                with engine.connect() as conn:
                    inspector = inspect(engine)
                    tables = inspector.get_table_names()
                
                if not tables:
                    return f"✗ Connected but no tables found in database."
                
                # Load first table (or modify logic to accept table name parameter)
                first_table = tables[0]
                df = pd.read_sql_table(first_table, engine)
                
                session.df = df
                session.source_files.append(filepath_or_connection)
                session._add_to_history(
                    "tool_call", "ingest_source",
                    {"source": filepath_or_connection, "type": "database", "table": first_table},
                    f"Loaded table '{first_table}': {df.shape[0]} rows, {df.shape[1]} columns"
                )
                return f"✓ Database connected and loaded table '{first_table}': {df.shape[0]} rows, {df.shape[1]} columns\nColumns: {list(df.columns)}"
            
            except Exception as db_error:
                return f"✗ Database connection/query failed: {str(db_error)}\nConnection string format: 'dialect+driver://user:password@host:port/database'"
    
    except FileNotFoundError:
        msg = f"✗ File not found: {filepath_or_connection}"
        session._add_to_history("error", "ingest_source", {"source": filepath_or_connection}, msg)
        return msg
    except Exception as e:
        msg = f"✗ Error ingesting source: {str(e)}"
        session._add_to_history("error", "ingest_source", {"source": filepath_or_connection}, msg)
        return msg


# Tool wrapper functions that integrate with LangChain
# These functions create a default session for backward compatibility

_default_session = AnalysisSession()


def load_csv(filename: str) -> str:
    """Loads a CSV file into the default session's DataFrame."""
    return _default_session.load_csv(filename)


def get_dataframe_info(dummy_input: str = "") -> str:
    """Get information about the default session's DataFrame"""
    return _default_session.get_dataframe_info()


def execute_code(code: str) -> str:
    """Executes provided Python code using the default session's DataFrame"""
    return _default_session.execute_code(code)


load_csv_tool = Tool(
    name="load_csv",
    func=load_csv,
    description="Loads a CSV file into a DataFrame.",
)

get_dataframe_info_tool = Tool(
    name="get_dataframe_info", 
    func=get_dataframe_info,
    description="Get detailed information about the currently loaded DataFrame including shape, columns, data types, and sample data.",
)

execute_code_tool = Tool(
    name="execute_code",
    func=execute_code,
    description="Executes provided Python code using pandas/matplotlib. Has access to the DataFrame and common libraries.",
)


def ingest_source_wrapper(filepath_or_connection: str) -> str:
    """Wrapper for LangChain integration - ingests data into the default session."""
    return ingest_source(filepath_or_connection, _default_session)


ingest_source_tool = Tool(
    name="ingest_source",
    func=ingest_source_wrapper,
    description="Ingest data from CSV, Excel, JSON, PDF, or database. Stores tabular data in DataFrame, text from PDFs in documents. Usage: provide file path or database connection string.",
)


def embed_documents_wrapper(dummy_input: str = "") -> str:
    """Wrapper for LangChain integration - embeds documents in the default session."""
    return _default_session.embed_documents()


embed_documents_tool = Tool(
    name="embed_documents",
    func=embed_documents_wrapper,
    description="Embed all loaded documents (PDFs, text) into ChromaDB using sentence-transformers. Must call after loading documents. Returns number of chunks created.",
)


def search_documents_wrapper(query: str) -> str:
    """Wrapper for LangChain integration - searches embedded documents."""
    results = _default_session.search_documents_in_session(query, top_k=3)
    
    if not results:
        return "No documents embedded yet. Load documents and call embed_documents first."
    
    # Format results as readable text
    output = f"Found {len(results)} relevant chunk(s) for query: '{query}'\n\n"
    for i, result in enumerate(results, 1):
        output += f"--- Result {i} (Similarity: {result['similarity_score']:.3f}) ---\n"
        output += f"Source: {result['source']} (Chunk {result['chunk_idx']})\n"
        output += f"Text: {result['text'][:300]}...\n\n"
    
    return output


search_documents_tool = Tool(
    name="search_documents",
    func=search_documents_wrapper,
    description="Search embedded documents for content related to a query. Returns top 3 most relevant chunks with similarity scores. Must call embed_documents after loading PDFs.",
)

tools = [load_csv_tool, get_dataframe_info_tool, execute_code_tool, ingest_source_tool, embed_documents_tool, search_documents_tool]