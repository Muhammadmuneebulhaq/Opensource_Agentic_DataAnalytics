# Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set Up API Key

Create `.env` file in the project directory:

```
OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Run the App

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`

---

## First Time Usage

### Page 1: Load Your Data (2 minutes)

1. Click **📁 Connect Data**
2. Choose one:
   - **Upload file**: CSV, Excel, JSON, or PDF
   - **Database**: E.g., `postgresql://user:pass@localhost/mydb`
3. You'll see a data preview with row/column counts

### Page 2: Ask Questions (3-5 minutes)

1. Click **🤖 Ask Your Data**
2. Type a question: "What are the top 5 items by revenue?"
3. The AI analyzes and generates charts automatically
4. Ask follow-up questions - context is maintained

### Page 3: View Metrics (auto-generated)

1. Click **📈 KPI View**
2. See auto-detected metrics:
   - Row/column counts
   - Numeric summaries (mean, min, max)
   - Time series charts
   - Category breakdowns

---

## Example Workflow

**Scenario**: You have a `sales.csv` file

```
1. Go to "Connect Data" → Upload sales.csv
2. Go to "Ask Your Data" → Type: "What's our total revenue by month?"
3. Agent generates Python code → Creates bar chart
4. Type: "Show the trend" → Agent creates line chart
5. Go to "KPI View" → See all metrics
```

---

## Common Queries

- "What are the top products?"
- "Show me the monthly trend"
- "Find outliers in this column"
- "Compare Q1 vs Q2 sales"
- "What's the correlation between X and Y?"

---

## File Format Examples

**CSV**: `sales.csv` (standard format)  
**Excel**: `data.xlsx` (will be converted to DataFrame)  
**JSON**: `{"data": [{...}, {...}]}` (auto-flattened)  
**PDF**: `report.pdf` (text extracted for search)  
**Database**: `postgresql://user:pass@localhost/mydb`

---

## Troubleshooting

**App won't start?**

- Delete `.venv` folder and reinstall: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.8+)

**API key error?**

- Make sure `.env` file exists in the project folder
- Format: `OPENAI_API_KEY=sk-...` (no quotes)

**Data not loading?**

- CSV/Excel must have headers
- PDF must have extractable text (not scanned)
- Database connection string format is crucial

**Slow responses?**

- First run downloads all models (~1 min)
- Subsequent queries are faster (~5-10 sec)

---

## Next Steps

- Check `README.md` for full documentation
- Try different data formats in "Connect Data"
- Explore "KPI View" auto-generated metrics
