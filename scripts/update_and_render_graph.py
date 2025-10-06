import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

DOCS_DIR = "docs"
HIST_PATH = os.path.join(DOCS_DIR, "history.csv")
INDEX_PATH = os.path.join(DOCS_DIR, "index.html")
TICKER = "SPY"

# Quick re-render uses the latest predictions rows to fill cards
def render_html_cards_from_latest(hist: pd.DataFrame):
    # latest date in history
    if hist.empty:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return ("No predictions yet", "No predictions yet", ts)

    last_date = hist["date"].max()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    subset = hist[hist["date"]==last_date].sort_values("student")
    # choose two students
    if len(subset) >= 1:
        a = subset.iloc[0]
        card_a = {
            "label": a["student"],
            "predicted_date": a["date"],
            "prediction_pct": float(a["prediction_pct"]),
            "norm_source": "training stats",
            "timestamp_utc": ts
        }
    else:
        card_a = {"label":"Student A","predicted_date":last_date,"prediction_pct":0.0,"norm_source":"", "timestamp_utc":ts}
    if len(subset) >= 2:
        b = subset.iloc[1]
        card_b = {
            "label": b["student"],
            "predicted_date": b["date"],
            "prediction_pct": float(b["prediction_pct"]),
            "norm_source": "training stats",
            "timestamp_utc": ts
        }
    else:
        card_b = {"label":"Student B","predicted_date":last_date,"prediction_pct":0.0,"norm_source":"", "timestamp_utc":ts}
    return card_a, card_b, ts

def render_full_page(card_a, card_b, history_df):
    def card(r):
        pct = f"{r['prediction_pct']:+.2f}%"
        return f"""
        <div class="card">
          <div class="title">{r['label']}</div>
          <div class="pred">{pct}</div>
          <div class="meta">for {r['predicted_date']} • (cards from latest entries)</div>
        </div>
        """
    h = history_df.copy()
    h = h.sort_values("date").tail(60)
    rows = "\n".join(
        f"<tr><td>{d}</td><td>{s}</td><td>{float(p):+.2f}%</td><td>{'' if pd.isna(a) else f'{float(a):+.2f}%'}</td></tr>"
        for d,s,p,a in zip(h["date"], h["student"], h["prediction_pct"], h["actual_pct"])
    )

    # Split by student for separate chart lines
    hA = h[h["student"].str.contains("Normal", na=False)]
    hB = h[h["student"].str.contains("Finer", na=False)]
    hAct = h.groupby("date")["actual_pct"].mean().reset_index()

    labels = ",".join([f"'{d}'" for d in sorted(set(h["date"]))])
    def make_series(df):
        vals = []
        for d in sorted(set(h["date"])):
            m = df[df["date"] == d]
            vals.append(f"{float(m['prediction_pct'].iloc[0]):.4f}" if not m.empty else "null")
        return ",".join(vals)
    predsA = make_series(hA)
    predsB = make_series(hB)
    acts = ",".join(["null" if pd.isna(x) else f"{x:.4f}" for x in hAct["actual_pct"].values])

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Daily EOD Return Predictions</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f14; color:#e8eef7; margin:0; padding:2rem; }}
  h1 {{ margin:0 0 1rem 0; font-weight:700; letter-spacing:0.2px; }}
  .time {{ opacity:0.7; margin-bottom:2rem; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:1rem; }}
  .card {{ background:#121824; border:1px solid #1c2433; border-radius:16px; padding:1.25rem; box-shadow:0 6px 24px rgba(0,0,0,0.25); }}
  .title {{ font-size:0.95rem; opacity:0.9; margin-bottom:0.25rem; }}
  .pred {{ font-size:2.2rem; font-weight:800; margin:0.5rem 0 0.25rem 0; }}
  .meta {{ opacity:0.7; font-size:0.9rem; }}
  .footer {{ opacity:0.6; font-size:0.85rem; margin-top:2rem; }}
  table {{ width:100%; border-collapse:collapse; margin-top:2rem; }}
  th, td {{ border-bottom:1px solid #1c2433; padding:0.5rem; text-align:left; }}
  th {{ opacity:0.8; }}
  .chart-wrap {{ background:#121824; border:1px solid #1c2433; border-radius:16px; padding:1rem; margin-top:2rem; }}
</style>
</head>
<body>
  <h1>Daily EOD Return Predictions — {TICKER}</h1>
  <div class="time">Last updated: {card_a['timestamp_utc']}</div>
  <div class="grid">
    {card(card_a)}
    {card(card_b)}
  </div>

  <div class="chart-wrap">
    <canvas id="histChart" height="120"></canvas>
  </div>

  <table>
    <thead><tr><th>Date</th><th>Student</th><th>Pred (%)</th><th>Actual (%)</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <div class="footer">Predictions ~12:00 UTC; Actuals filled ~23:00 UTC (post close).</div>

<script>
const ctx = document.getElementById('histChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: [{labels}],
    datasets: [
      {{ label: 'Student A — Normal', data: [{predsA}], borderWidth: 2, tension: 0.2 }},
      {{ label: 'Student B — Finer', data: [{predsB}], borderWidth: 2, tension: 0.2 }},
      {{ label: 'Actual', data: [{acts}], borderDash: [6,4], borderWidth: 2, tension: 0.2 }}
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      y: {{ title: {{ display:true, text:'Return (%)' }} }}
    }},
    plugins: {{ legend: {{ display: true }} }}
  }}
}});
</script>
</body></html>
"""

def fetch_actual_for_date(date_str: str) -> float | None:
    """
    Close-to-close % return for that US/Eastern date, in percent units.
    """
    # pull small window around date to be safe
    df = yf.download(TICKER, period="10d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    df["DailyReturn"] = df["Close"].pct_change() * 100.0
    df = df.dropna()
    # Match by date string
    df_idx_dates = df.index.tz_localize(None).date
    mask = np.array([str(d) == date_str for d in df_idx_dates])
    if mask.any():
        return float(df.loc[mask, "DailyReturn"].iloc[0])
    return None

def main():
    if not os.path.exists(HIST_PATH):
        print("No history.csv yet, nothing to update.")
        return
    hist = pd.read_csv(HIST_PATH)
    if hist.empty:
        print("Empty history.csv, nothing to update.")
        return

    # Foreach date with any NaN actual, fill it once, although this should never need to be used
    for date_str in sorted(hist["date"].unique()):
        needs = hist[(hist["date"]==date_str) & (hist["actual_pct"].isna())]
        if needs.empty:
            continue
        actual = fetch_actual_for_date(date_str)
        if actual is not None:
            hist.loc[hist["date"]==date_str, "actual_pct"] = actual
            print(f"Filled actual for {date_str}: {actual:+.2f}%")
        else:
            print(f"Could not fetch actual for {date_str} yet.")

    hist.sort_values(["date","student"], inplace=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    hist.to_csv(HIST_PATH, index=False)

    # Re-render the page with latest cards + updated history
    card_a, card_b, _ = render_html_cards_from_latest(hist)
    html = render_full_page(card_a, card_b, hist)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print("Updated docs/index.html with actuals")

if __name__ == "__main__":
    main()
