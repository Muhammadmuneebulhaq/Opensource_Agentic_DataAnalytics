
import pandas as pd
import matplotlib.pyplot as plt

# Work on a copy to avoid mutating the original
df = pd.read_csv("tips.csv")
df2 = df.copy()

# Helper: find a datetime index or column
date_index = isinstance(df2.index, pd.DatetimeIndex)
date_col = None

if not date_index:
    # Prefer columns already typed as datetime
    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            date_col = col
            break

    # If none, try common date-like names
    if date_col is None:
        candidates = ['date', 'time', 'timestamp', 'datetime', 'period', 'year', 'month']
        for col in df2.columns:
            if any(k in str(col).lower() for k in candidates):
                try:
                    converted = pd.to_datetime(df2[col], errors='coerce', infer_datetime_format=True)
                    if converted.notna().sum() > 0:
                        df2[col] = converted
                        date_col = col
                        break
                except Exception:
                    pass

    # As a last resort, try parsing object columns and keep the first with high success
    if date_col is None:
        for col in df2.columns:
            if df2[col].dtype == object:
                converted = pd.to_datetime(df2[col], errors='coerce', infer_datetime_format=True)  
                if len(converted) > 0 and (converted.notna().sum() / len(converted) >= 0.8):       
                    df2[col] = converted
                    date_col = col
                    break

# If no datetime found, stop
if not date_index and date_col is None:
    raise ValueError("No date/time index or column found. Cannot analyze trend over time.")        

# Ensure we have a datetime index for resampling
if date_index:
    df2 = df2.sort_index()
    dt_index = df2.index
else:
    df2 = df2[df2[date_col].notna()].sort_values(date_col)
    dt_index = pd.DatetimeIndex(df2[date_col])
    df2 = df2.set_index(date_col)

# Identify numeric columns to analyze
num_cols = df2.select_dtypes(include='number').columns.tolist()
if not num_cols:
    raise ValueError("No numeric columns available to analyze trend over time.")

# Choose an aggregation frequency based on span
if len(dt_index) > 1:
    span_days = (dt_index.max() - dt_index.min()).days
else:
    span_days = 0

if span_days <= 60:
    freq = 'D'   # daily
elif span_days <= 730:
    freq = 'M'   # monthly
elif span_days <= 3650:
    freq = 'Q'   # quarterly
else:
    freq = 'A'   # yearly

# Aggregate numeric metrics over time
ts = df2[num_cols].resample(freq).sum()

# If too many series to plot, provide a total
plot_cols = ts.columns.tolist()
use_total_only = False
if len(plot_cols) > 6:
    ts['Total'] = ts.sum(axis=1)
    plot_cols = ['Total']
    use_total_only = True

# Plot trends
plt.figure(figsize=(10, 6))
for col in plot_cols:
    plt.plot(ts.index, ts[col], label=col)
title_suffix = "Total" if use_total_only else "by metric"
plt.title(f"Trend over time ({freq} aggregation) - {title_suffix}")
plt.xlabel("Time")
plt.ylabel("Value")
if len(plot_cols) > 1:
    plt.legend()
plt.tight_layout()
plt.show()

# Textual trend summary
def trend_summary(series):
    s = series.dropna()
    if len(s) < 2:
        return "insufficient data"
    first = s.iloc[0]
    last = s.iloc[-1]
    if last > first:
        direction = "increasing"
    elif last < first:
        direction = "decreasing"
    else:
        direction = "flat"
    if first == 0:
        return direction
    pct = (last - first) / abs(first) * 100
    return f"{direction} ({pct:.1f}% change)"

print("Trend summary:")
for col in plot_cols:
    print(f"- {col}: {trend_summary(ts[col])}")