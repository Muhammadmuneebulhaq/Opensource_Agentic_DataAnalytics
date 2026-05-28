# Data Analysis Agent with Streamlit UI

A powerful multi-page Streamlit application that integrates LangGraph agents with data analysis tools. Upload your data (CSV, Excel, JSON, PDF, or database) and use an AI assistant to analyze it with natural language queries.

## Features

### 📁 Page 1: Connect Data

- **File Upload**: CSV, Excel (.xlsx, .xls), JSON, PDF files
- **Database Connection**: Connect to any SQL database using SQLAlchemy connection strings (PostgreSQL, MySQL, SQLite, etc.)
- **Data Preview**: View loaded data and metadata (row count, column count, memory usage)
- **PDF Indexing**: Embed and index PDF documents for semantic search

### 🤖 Page 2: Ask Your Data

- **Chat Interface**: Natural language queries about your data
- **LangGraph Agent**: Powered by Claude with embedded analysis tools
- **Auto Chart Generation**: Automatically creates visualizations based on analysis code
- **Message History**: Maintains conversation context across multiple queries
- **Tool Integration**: Access to data loading, analysis, and document search tools

### 📈 Page 3: KPI View

- **Auto-Detected Metrics**: Automatically identifies and displays key metrics
  - Row count, column count, null values, memory usage
  - Numeric column statistics (mean, median, min, max, std dev)
  - Data type distribution
  - Time series charts (if date columns exist)
  - Categorical value distributions
- **Interactive Visualizations**: Charts for data exploration
- **Categorical Insights**: Top values and distributions for categorical columns

## Installation

### 1. Clone or download the project

```bash
cd "path/to/Open Source Project"
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Get your OpenAI API key from: https://platform.openai.com/api-keys

## Running the Application

### Option 1: Streamlit Web UI (Recommended)

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Option 2: Command-line Agent

For a simple CLI interface:

```bash
python langGraphAgent.py
```

## Usage Guide

### Page 1: Connect Data

1. **Upload CSV/Excel/JSON**:
   - Click the file uploader
   - Select your file
   - The data will be loaded automatically

2. **Upload PDF**:
   - Click the file uploader and select a PDF
   - Optionally click "Embed & Index PDF" to enable semantic search on the document

3. **Connect to Database**:
   - Enter your SQLAlchemy connection string in the text field
   - Format: `dialect+driver://user:password@host:port/database`
   - Examples:
     - PostgreSQL: `postgresql://user:pass@localhost/mydb`
     - MySQL: `mysql+pymysql://user:pass@localhost/mydb`
     - SQLite: `sqlite:///path/to/database.db`
   - Click "Connect Database"

### Page 2: Ask Your Data

1. Enter your question in the chat box
2. The AI agent will:
   - Analyze your question
   - Write Python code to explore/analyze the data
   - Execute the code and generate visualizations
   - Return a natural language explanation with charts

Example queries:

- "What are the top 5 products by sales?"
- "Show me the trend of revenue over time"
- "What's the correlation between price and quantity sold?"
- "Find the month with highest customer activity"

### Page 3: KPI View

- Automatically displays key metrics from your loaded data
- Shows distribution charts for categorical variables
- Displays time series if your data has date columns
- All metrics update when you load new data

## Architecture

### Core Components

**AnalysisSession** (`analyticTools.py`)

- Manages all analysis state (DataFrame, documents, conversation history)
- Chunk and embed documents for semantic search
- Track source files and analysis history

**Tools Available**:

- `load_csv`: Load CSV files
- `get_dataframe_info`: Get DataFrame metadata
- `execute_code`: Run Python code for analysis
- `ingest_source`: Load multiple file formats and databases
- `embed_documents`: Index documents with sentence-transformers
- `search_documents`: Semantic search over documents

**LangGraph Agent** (`langGraphAgent.py`)

- Multi-tool agent with Claude as the reasoning engine
- Routes queries to appropriate tools
- Maintains conversation context

**Streamlit App** (`streamlit_app.py`)

- Multi-page UI with shared session state
- File upload and database connection
- Chat interface with the agent
- KPI dashboard

## Supported File Formats

| Format   | Details                       | Destination                     |
| -------- | ----------------------------- | ------------------------------- |
| CSV      | Comma-separated values        | DataFrame                       |
| Excel    | .xlsx, .xls files             | DataFrame                       |
| JSON     | Flat or nested JSON           | DataFrame (auto-flattened)      |
| PDF      | Text extraction from PDFs     | Documents (for semantic search) |
| Database | SQLAlchemy connection strings | DataFrame (first table)         |

## Dependencies Overview

| Package                   | Purpose                        |
| ------------------------- | ------------------------------ |
| `streamlit`               | Web UI framework               |
| `langchain` / `langgraph` | LLM and agent framework        |
| `langchain-openai`        | OpenAI integration             |
| `pandas`                  | Data manipulation              |
| `matplotlib`              | Visualization                  |
| `chromadb`                | Vector database for embeddings |
| `sentence-transformers`   | Document embedding model       |
| `pdfplumber`              | PDF text extraction            |
| `sqlalchemy`              | Database connections           |

## Troubleshooting

### "No module named 'analyticTools'"

Make sure you're running streamlit from the correct directory with all Python files present.

### PDF embedding fails

- Ensure pdfplumber is installed: `pip install pdfplumber`
- Ensure sentence-transformers is installed: `pip install sentence-transformers`
- Check that the PDF contains extractable text (not scanned OCR)

### Database connection fails

- Verify connection string format for your database type
- Check that database driver is installed (e.g., `pip install psycopg2` for PostgreSQL)
- Ensure database is accessible from your network

### Agent responses are slow

- First response is slowest (model download + initialization)
- Subsequent responses use cached models
- Increase OpenAI model timeout if queries time out

### "OPENAI_API_KEY not set"

- Create `.env` file with your API key
- Or set environment variable: `export OPENAI_API_KEY=sk-...`

## Advanced Usage

### Custom Analysis

Write Python code in the chat to perform custom analysis:

```
"Run this analysis: df['date']=pd.to_datetime(df['date']); df.groupby(df['date'].dt.month).sum().plot()"
```

### Semantic Document Search

After uploading a PDF:

1. Say "Embed and index the PDF"
2. Ask questions like: "What are the main topics in this document?"
3. The agent will search the document and return relevant excerpts

### Multi-File Analysis

Upload different file types in sequence:

1. Upload a CSV with main data
2. Upload a PDF with background information
3. Ask questions that combine both sources

## Performance Notes

- **First run**: ~30-60 seconds (model initialization)
- **Subsequent runs**: ~5-10 seconds per query
- **Large datasets**: >100k rows may need optimized queries
- **PDFs**: Works best with text-based PDFs (not scanned images)

## Limitations

- Agent has context limits (handles ~5-10k tokens per message)
- Database queries limited to first table (modify code for specific tables)
- Charts auto-generated based on code execution
- Requires internet for LLM calls and model downloads

## Future Enhancements

- [ ] Multi-table database schema exploration
- [ ] Export analysis results and charts
- [ ] Saved analysis history and bookmarks
- [ ] Custom model selection (Anthropic, Llama, etc.)
- [ ] Data profiling and quality reports
- [ ] Advanced filtering and aggregation UI

## License

This project is provided as-is for educational and analytical purposes.
