import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import tempfile
import os
from datetime import datetime
from typing import Optional

# Import analysis tools and agent
from analyticTools import (
    AnalysisSession, 
    ingest_source, 
    search_documents_tool
)
from langGraphAgent import analysis_graph
from langchain_core.messages import SystemMessage, HumanMessage

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session():
    """Initialize or retrieve the AnalysisSession from st.session_state."""
    if "session" not in st.session_state:
        st.session_state.session = AnalysisSession()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="""You are a helpful data analysis assistant. You have access to tools for:
- Loading and analyzing CSV, Excel, JSON, and PDF files
- Executing Python data analysis code
- Embedding and searching document content

When users ask questions about their data, help them explore, visualize, and understand it.

IMPORTANT: Always check if data has already been loaded in the session. If the user mentions analyzing data or asks about specific columns, assume the data is available through the execute_code tool using the 'df' variable. Do not ask users to upload data if they've already loaded it. Generate Python code to analyze the data directly.""")
        ]
    
    if "agent_responses" not in st.session_state:
        st.session_state.agent_responses = []

initialize_session()

# ============================================================================
# PAGE 1: Connect Data
# ============================================================================

def page_connect_data():
    """Page 1: File upload and database connection."""
    st.title("📁 Connect Data")
    st.write("Upload your data source (CSV, Excel, PDF, JSON) or connect to a database.")
    
    col1, col2 = st.columns(2)
    
    # File Upload Section
    with col1:
        st.subheader("📤 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "json", "pdf"],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            try:
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest the file
                result = ingest_source(temp_path, st.session_state.session)
                
                # Display result
                if result.startswith("✓"):
                    st.success(result)
                    
                    # If it's a PDF, offer to embed documents
                    if uploaded_file.name.endswith(".pdf"):
                        if st.button("🔍 Embed & Index PDF", key="embed_pdf"):
                            with st.spinner("Embedding document..."):
                                embed_result = st.session_state.session.embed_documents()
                                st.success(embed_result)
                else:
                    st.error(result)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Database Connection Section
    with col2:
        st.subheader("🗄️ Database Connection")
        db_connection = st.text_input(
            "Enter connection string",
            placeholder="postgresql://user:pass@localhost/dbname",
            help="SQLAlchemy format: dialect+driver://user:password@host:port/database"
        )
        
        if st.button("🔗 Connect Database", key="connect_db"):
            if db_connection:
                with st.spinner("Connecting..."):
                    result = ingest_source(db_connection, st.session_state.session)
                    if result.startswith("✓"):
                        st.success(result)
                    else:
                        st.error(result)
            else:
                st.warning("Please enter a connection string.")
    
    # Data Summary
    st.divider()
    st.subheader("📋 Loaded Data Summary")
    
    if st.session_state.session.df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.session.df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.session.df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{st.session_state.session.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.dataframe(
            st.session_state.session.df.head(10),
            use_container_width=True,
            height=300
        )
    else:
        st.info("No data loaded yet. Upload a file or connect to a database above.")
    
    # Loaded Files
    if st.session_state.session.source_files:
        st.subheader("📂 Loaded Sources")
        for source in st.session_state.session.source_files:
            st.text(f"✓ {source}")


# ============================================================================
# PAGE 2: Ask Your Data
# ============================================================================

def page_ask_data():
    """Page 2: Chat interface with LangGraph agent."""
    st.title("🤖 Ask Your Data")
    st.write("Chat with your AI data analysis assistant.")
    
    # Check if data is loaded
    has_dataframe = st.session_state.session.df is not None
    has_documents = st.session_state.session.documents is not None and len(st.session_state.session.documents) > 0
    has_source_files = st.session_state.session.source_files is not None and len(st.session_state.session.source_files) > 0
    
    if not (has_dataframe or has_documents or has_source_files):
        st.warning("⚠️ No data loaded yet. Please go to **Connect Data** page first.")
        return
    
    # Display loaded data summary
    with st.expander("📊 Current Session Data", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if has_source_files:
                st.metric("Loaded Sources", len(st.session_state.session.source_files))
                for source in st.session_state.session.source_files:
                    st.text(f"✓ {source}")
            else:
                st.metric("Loaded Sources", 0)
        
        with col2:
            if has_dataframe:
                st.metric("Rows", st.session_state.session.df.shape[0])
                st.metric("Columns", st.session_state.session.df.shape[1])
            else:
                st.info("No structured data loaded")
        
        with col3:
            if has_documents:
                st.metric("Documents Indexed", len(st.session_state.session.documents))
            else:
                st.info("No documents indexed")
    
    # Display chat history
    st.subheader("💬 Conversation")
    
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            # Skip system message in display
            if msg.__class__.__name__ == "SystemMessage":
                continue
            
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
            
            with st.chat_message(role):
                if isinstance(msg.content, str):
                    st.markdown(msg.content)
                else:
                    st.write(msg.content)
    
    # User input
    user_input = st.chat_input(
        "Ask a question about your data...",
        key="chat_input"
    )
    
    if user_input:
        # Add user message
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get agent response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    # Sync the Streamlit session with the default session used by tools
                    # This ensures the agent's tools have access to the loaded dataframe
                    import analyticTools
                    if has_dataframe:
                        analyticTools._default_session.df = st.session_state.session.df
                        analyticTools._default_session.source_files = st.session_state.session.source_files
                    if has_documents:
                        analyticTools._default_session.documents = st.session_state.session.documents
                        analyticTools._default_session.vectorstore = st.session_state.session.vectorstore
                    
                    # Run the agent with the clean message history
                    # The agent will have access to the synced session data through the tools
                    response = analysis_graph.invoke({"messages": st.session_state.messages})
                    
                    # Extract the response messages
                    response_messages = response.get("messages", [])
                    
                    # Find the new messages added by the agent (everything after what we sent)
                    # The agent response includes all messages, so extract only the new ones
                    new_message_count = max(0, len(response_messages) - len(st.session_state.messages))
                    
                    if new_message_count > 0:
                        new_messages = response_messages[-new_message_count:]
                        
                        # Add new messages to session
                        for msg in new_messages:
                            st.session_state.messages.append(msg)
                        
                        # Extract the assistant text response
                        assistant_response = ""
                        for msg in reversed(new_messages):
                            if msg.__class__.__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
                                assistant_response = msg.content
                                break
                        
                        if assistant_response:
                            # Display response
                            st.markdown(assistant_response)
                            
                            # Try to extract and display charts if mentioned
                            if "plt" in assistant_response.lower() or "chart" in assistant_response.lower():
                                try:
                                    # Extract code from response
                                    if "```python" in assistant_response:
                                        code_start = assistant_response.find("```python") + len("```python")
                                        code_end = assistant_response.find("```", code_start)
                                        code = assistant_response[code_start:code_end].strip()
                                        
                                        # Execute code to generate chart
                                        exec_globals = {
                                            "pd": pd,
                                            "plt": plt,
                                            "df": st.session_state.session.df,
                                            "np": __import__("numpy")
                                        }
                                        exec(code, exec_globals)
                                        
                                        # Display the chart if one was created
                                        if plt.get_fignums():
                                            st.pyplot(plt.gcf())
                                            plt.close('all')
                                except Exception as e:
                                    st.caption(f"Could not display chart: {str(e)}")
                    else:
                        st.warning("No response from agent")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Sidebar: Session Info
    with st.sidebar:
        st.subheader("📊 Session Info")
        if st.session_state.session.df is not None:
            st.write(f"**Rows:** {st.session_state.session.df.shape[0]}")
            st.write(f"**Columns:** {st.session_state.session.df.shape[1]}")
            st.write(f"**Columns:** {', '.join(st.session_state.session.df.columns[:5])}...")


# ============================================================================
# PAGE 3: KPI View
# ============================================================================

def detect_time_columns(df):
    """Detect both datetime and categorical time columns."""
    datetime_cols = []
    categorical_time_cols = []
    
    # Check for datetime columns
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    
    # Check for columns that might be dates
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            if col not in datetime_cols:
                try:
                    pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except:
                    pass
    
    # Look for categorical time patterns (day, month, hour, week, etc.)
    time_keywords = ["day", "month", "year", "week", "hour", "minute", "date", "time", "period"]
    for col in df.columns:
        col_lower = col.lower()
        if col not in datetime_cols and any(keyword in col_lower for keyword in time_keywords):
            categorical_time_cols.append(col)
    
    return datetime_cols, categorical_time_cols


def calculate_trend_metrics(series):
    """Calculate trend metrics for a numeric series."""
    try:
        if len(series) < 2 or series.isnull().all():
            return {}
        
        series_clean = series.dropna()
        if len(series_clean) < 2:
            return {}
        
        # Calculate growth rate
        first_val = series_clean.iloc[0]
        last_val = series_clean.iloc[-1]
        growth_rate = ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else 0
        
        # Volatility (coefficient of variation)
        volatility = (series_clean.std() / series_clean.mean() * 100) if series_clean.mean() != 0 else 0
        
        # Trend direction
        first_half = series_clean.iloc[:len(series_clean)//2].mean()
        second_half = series_clean.iloc[len(series_clean)//2:].mean()
        trend_direction = "📈 Up" if second_half > first_half else "📉 Down" if second_half < first_half else "➡️ Stable"
        
        return {
            "growth_rate": growth_rate,
            "volatility": volatility,
            "trend_direction": trend_direction
        }
    except:
        return {}


def page_kpi_view():
    """Page 3: Auto-detected KPIs and metrics with enhanced time series analysis."""
    st.title("📈 KPI View")
    st.write("Auto-detected key metrics and insights from your data.")
    
    if st.session_state.session.df is None:
        st.warning("⚠️ No data loaded yet. Please go to **Connect Data** page first.")
        return
    
    df = st.session_state.session.df
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols, categorical_time_cols = detect_time_columns(df)
    
    # ==================== SECTION 1: Overview ====================
    st.subheader("📊 Overview & Quality Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        null_count = df.isnull().sum().sum()
        null_pct = (null_count / (len(df) * len(df.columns)) * 100)
        st.metric("Missing Data", f"{null_pct:.1f}%")
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)
    with col5:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.2f} MB")
    
    # ==================== SECTION 2: Numeric KPIs ====================
    st.divider()
    st.subheader("🔢 Numeric Columns - Summary & Trends")
    
    if numeric_cols:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Statistical Summary:**")
            summary_data = []
            for col in numeric_cols:
                trend_metrics = calculate_trend_metrics(df[col])
                summary_data.append({
                    "Column": col,
                    "Count": df[col].count(),
                    "Mean": f"{df[col].mean():.2f}",
                    "Median": f"{df[col].median():.2f}",
                    "Std Dev": f"{df[col].std():.2f}",
                    "Min": f"{df[col].min():.2f}",
                    "Max": f"{df[col].max():.2f}",
                    "Growth %": f"{trend_metrics.get('growth_rate', 0):.1f}%",
                    "Volatility": f"{trend_metrics.get('volatility', 0):.1f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, height=300)
        
        with col2:
            st.write("**Distribution Analysis:**")
            selected_numeric = st.selectbox("Select column for details:", numeric_cols, key="numeric_dist")
            
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df[selected_numeric].dropna(), bins=30, color="steelblue", edgecolor="black", alpha=0.7)
                ax.set_title(f"Distribution: {selected_numeric}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col_b:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.boxplot(df[selected_numeric].dropna(), vert=True)
                ax.set_title(f"Box Plot: {selected_numeric}")
                ax.set_ylabel("Value")
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
    else:
        st.info("No numeric columns found in the dataset.")
    
    # ==================== SECTION 3: Time Series / Trend Analysis ====================
    st.divider()
    st.subheader("📉 Time Series & Trend Analysis")
    
    # Find applicable columns for trend analysis
    has_continuous_time = bool(datetime_cols)
    has_categorical_time = bool(categorical_time_cols)
    
    if (has_continuous_time or has_categorical_time) and numeric_cols:
        st.write("**Select time dimension and metric to analyze:**")
        
        col1, col2 = st.columns([1, 1])
        
        # Continuous Time Series
        if has_continuous_time:
            with col1:
                st.write("**📅 Continuous Time Series:**")
                date_col = st.selectbox("Select date column:", datetime_cols, key="date_col_select")
                numeric_col = st.selectbox("Select metric to plot:", numeric_cols, key="numeric_col_select")
                
                try:
                    # Prepare data
                    temp_df = df[[date_col, numeric_col]].copy()
                    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                    temp_df = temp_df.sort_values(date_col)
                    
                    # Create chart
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(temp_df[date_col], temp_df[numeric_col], marker="o", linewidth=2, color="steelblue")
                    ax.fill_between(temp_df[date_col], temp_df[numeric_col], alpha=0.3, color="steelblue")
                    ax.set_title(f"{numeric_col} over {date_col}")
                    ax.set_xlabel(date_col)
                    ax.set_ylabel(numeric_col)
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show trend metrics
                    trend_metrics = calculate_trend_metrics(temp_df[numeric_col])
                    if trend_metrics:
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Trend Direction", trend_metrics.get("trend_direction", "N/A"))
                        with m2:
                            st.metric("Growth Rate", f"{trend_metrics.get('growth_rate', 0):.1f}%")
                        with m3:
                            st.metric("Volatility", f"{trend_metrics.get('volatility', 0):.1f}%")
                
                except Exception as e:
                    st.warning(f"Could not create continuous time series: {str(e)}")
        
        # Categorical Time Patterns
        if has_categorical_time:
            with col2:
                st.write("**🏷️ Categorical Time Patterns:**")
                cat_time_col = st.selectbox("Select time dimension:", categorical_time_cols, key="cat_time_select")
                numeric_col_cat = st.selectbox("Select metric to analyze:", numeric_cols, key="numeric_col_cat_select")
                
                try:
                    # Create aggregated view by categorical time
                    agg_df = df.groupby(cat_time_col)[numeric_col_cat].agg(['mean', 'count', 'std']).reset_index()
                    
                    # Define order for common time dimensions
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                                 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',
                                 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    # Try to order if it's a day column
                    if cat_time_col.lower() in ['day', 'day_of_week']:
                        existing_days = [d for d in day_order if d in agg_df[cat_time_col].values]
                        if existing_days:
                            agg_df[cat_time_col] = pd.Categorical(agg_df[cat_time_col], categories=existing_days, ordered=True)
                            agg_df = agg_df.sort_values(cat_time_col)
                    
                    # Create chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(range(len(agg_df)), agg_df['mean'], color="coral", alpha=0.7, edgecolor="black")
                    
                    # Add error bars for std dev
                    ax.errorbar(range(len(agg_df)), agg_df['mean'], yerr=agg_df['std'], 
                               fmt='none', ecolor='red', capsize=5, capthick=2, alpha=0.5)
                    
                    ax.set_xticks(range(len(agg_df)))
                    ax.set_xticklabels(agg_df[cat_time_col], rotation=45, ha='right')
                    ax.set_title(f"Average {numeric_col_cat} by {cat_time_col}")
                    ax.set_ylabel(f"Average {numeric_col_cat}")
                    ax.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show statistics table
                    st.write(f"**Breakdown by {cat_time_col}:**")
                    display_agg = agg_df.copy()
                    display_agg.columns = [cat_time_col, f"Avg {numeric_col_cat}", "Count", "Std Dev"]
                    st.dataframe(display_agg, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not create categorical time analysis: {str(e)}")
    else:
        if not (has_continuous_time or has_categorical_time):
            st.info("📅 No time/date columns detected. Trend analysis requires a time dimension.")
        elif not numeric_cols:
            st.info("🔢 No numeric columns found for trend analysis.")
    
    # ==================== SECTION 4: Correlations & Relationships ====================
    if len(numeric_cols) > 1:
        st.divider()
        st.subheader("🔗 Correlations & Relationships")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            import numpy as np
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            cmap = plt.cm.coolwarm
            
            im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax.set_yticklabels(numeric_cols)
            ax.set_title("Correlation Matrix")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation', rotation=270, labelpad=15)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Scatter plot for top correlations
            st.write("**Top Correlations (excluding self):**")
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    corr_pairs.append({
                        'Var1': numeric_cols[i],
                        'Var2': numeric_cols[j],
                        'Correlation': abs(corr_val)
                    })
            
            corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False).head(5)
            
            if not corr_pairs_df.empty:
                st.dataframe(corr_pairs_df, use_container_width=True)
                
                # Create scatter plot for top correlation
                top_corr = corr_pairs_df.iloc[0]
                var1, var2 = top_corr['Var1'], top_corr['Var2']
                
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(df[var1], df[var2], alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
                ax.set_xlabel(var1)
                ax.set_ylabel(var2)
                ax.set_title(f"Top Correlation: {var1} vs {var2}")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # ==================== SECTION 5: Categorical Insights ====================
    st.divider()
    st.subheader("🏷️ Categorical Insights")
    
    if categorical_cols:
        selected_cat = st.selectbox("Select a categorical column:", categorical_cols, key="cat_insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Top 10 values in {selected_cat}:**")
            top_values = df[selected_cat].value_counts().head(10)
            st.dataframe(top_values.rename("Count"), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            top_values.plot(kind="barh", ax=ax, color="seagreen", alpha=0.7, edgecolor="black")
            ax.set_title(f"Top Values: {selected_cat}")
            ax.set_xlabel("Count")
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Cross-tab with numeric if available
        if numeric_cols and categorical_time_cols:
            st.write(f"**Average metrics by {selected_cat}:**")
            
            selected_numeric_for_cat = st.selectbox("Select metric:", numeric_cols, key="metric_by_cat")
            crosstab_df = df.groupby(selected_cat)[selected_numeric_for_cat].agg(['mean', 'count', 'std', 'min', 'max']).reset_index()
            crosstab_df = crosstab_df.sort_values('mean', ascending=False)
            
            st.dataframe(crosstab_df, use_container_width=True)
    else:
        st.info("No categorical columns found in the dataset.")


# ============================================================================
# Navigation Setup
# ============================================================================

pages = [
    st.Page(page_connect_data, title="Connect Data", icon="📁"),
    st.Page(page_ask_data, title="Ask Your Data", icon="🤖"),
    st.Page(page_kpi_view, title="KPI View", icon="📈"),
]

pg = st.navigation(pages)

# ============================================================================
# Run the App
# ============================================================================

if __name__ == "__main__":
    pg.run()
