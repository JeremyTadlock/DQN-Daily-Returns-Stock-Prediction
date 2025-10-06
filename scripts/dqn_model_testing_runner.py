# Imports
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from datetime import datetime, timezone

# Model paths (relative to repo root)
NORMAL_RES_PATH = os.path.join("models", "trained_dqn_predict_daily_return_3yrs_2.pth")
FINER_RES_PATH  = os.path.join("models", "trained_dqn_predict_daily_return_3yrs_2_finer_res.pth")

# Bins (must match each student’s training)
BINS_NORMAL = np.arange(-4.0, 4.0 + 0.5, 0.5)     # ±4.0 in 0.5 steps -> 17 bins
BINS_FINER  = np.linspace(-2.5, 2.5, 51)          # ±2.5 in 0.1 steps -> 51 bins

WINDOW_SIZE = 25
TICKER = "SPY"

# Paths
DOCS_DIR = "docs"
HIST_PATH = os.path.join(DOCS_DIR, "history.csv")
INDEX_PATH = os.path.join(DOCS_DIR, "index.html")

# DQN Network (same as training, but optimizer unused)
class DQN(nn.Module):
    def __init__(self, state_size, action_size, device=None):
        super(DQN, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.value_fc = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)
        self.advantage_fc = nn.Linear(256, 256)
        self.advantage = nn.Linear(256, action_size)
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        value = torch.relu(self.value_fc(x))
        value = self.value(value)
        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# Testing Data Prep
def get_testing_state(ticker, window_size=WINDOW_SIZE, period="60d"):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"No data downloaded for ticker: {ticker}")

    data["DailyReturn"] = data["Close"].pct_change() * 100.0
    data.dropna(inplace=True)

    if len(data) < window_size:
        raise ValueError(f"Not enough data to form a {window_size}-day window; got {len(data)} rows.")

    # predicted_date selection (US/Eastern)
    last_complete_date = pd.Timestamp(data.index[-1]).tz_localize(None)  # trading day from dataset
    now_et = pd.Timestamp.now(tz="America/New_York")
    today_et = now_et.normalize().tz_localize(None)

    if today_et.weekday() < 5:  # Mon–Fri
        # If the dataset already shows today's date (time-zone quirks), predict for TODAY only once.
        if last_complete_date >= today_et:
            predicted_date = (last_complete_date + pd.offsets.BDay(1)).date()
        else:
            predicted_date = today_et.date()
    else:
        # Weekend/holiday → next business day
        predicted_date = (today_et + pd.offsets.BDay(1)).date()

    returns = data["DailyReturn"].iloc[-window_size:].values.astype(np.float32)
    return returns, predicted_date, data

# Load model + training normalization
def load_model_and_stats(model_path, state_size, action_bins, device=None):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(model_path, map_location=dev, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=dev)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # get action bin size
    if "advantage.bias" in state_dict:
        expected_actions = state_dict["advantage.bias"].shape[0]
    else:
        keys = [k for k in state_dict if "advantage" in k and k.endswith("bias")]
        if keys:
            expected_actions = state_dict[keys[0]].shape[0]
        else:
            wkeys = [k for k in state_dict if "advantage" in k and k.endswith("weight")]
            expected_actions = state_dict[wkeys[0]].shape[0]
    if expected_actions != len(action_bins):
        raise ValueError(
            f"Bins length ({len(action_bins)}) does not match checkpoint actions ({expected_actions}). "
            f"Check you selected the correct bin set for {os.path.basename(model_path)}."
        )
    dqn = DQN(state_size=state_size, action_size=expected_actions, device=dev)
    dqn.load_state_dict(state_dict, strict=True)
    dqn.eval()
    # training stats
    if isinstance(ckpt, dict):
        train_mean = float(ckpt.get("train_mean", np.nan))
        train_std  = float(ckpt.get("train_std",  np.nan))
    else:
        train_mean, train_std = np.nan, np.nan
    return dqn, train_mean, train_std

# Predict next-day return
def predict_with_student(model_path, action_bins, label):
    returns_window, predicted_date, _df = get_testing_state(TICKER, WINDOW_SIZE, "60d")
    dqn, train_mean, train_std = load_model_and_stats(model_path, state_size=WINDOW_SIZE, action_bins=action_bins)
    if not (np.isnan(train_mean) or np.isnan(train_std)):
        mean = train_mean
        std = train_std if train_std > 0 else 1e-8
        norm_src = "training stats"
    else:
        mean = returns_window.mean()
        std = returns_window.std() + 1e-8
        norm_src = "window stats (fallback)"
    state_norm = (returns_window - mean) / std
    state_tensor = torch.from_numpy(state_norm).float().to(dqn.device).unsqueeze(0)
    with torch.no_grad():
        q = dqn(state_tensor)
        idx = torch.argmax(q, dim=1).item()
        pred = float(action_bins[idx])
    return {
        "label": label,
        "predicted_date": str(predicted_date),
        "prediction_pct": pred,
        "norm_source": norm_src,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

# --- History helpers ---
def ensure_history():
    os.makedirs(DOCS_DIR, exist_ok=True)
    if not os.path.exists(HIST_PATH):
        pd.DataFrame(columns=["date","student","prediction_pct","actual_pct","timestamp_utc"]).to_csv(HIST_PATH, index=False)

def append_predictions_for_day(res_list):
    ensure_history()
    hist = pd.read_csv(HIST_PATH)
    for r in res_list:
        # upsert by (date, student)
        mask = (hist["date"]==r["predicted_date"]) & (hist["student"]==r["label"])
        if mask.any():
            hist.loc[mask, ["prediction_pct","timestamp_utc"]] = [r["prediction_pct"], r["timestamp_utc"]]
        else:
            hist = pd.concat([hist, pd.DataFrame([{
                "date": r["predicted_date"],
                "student": r["label"],
                "prediction_pct": r["prediction_pct"],
                "actual_pct": np.nan,
                "timestamp_utc": r["timestamp_utc"],
            }])], ignore_index=True)
    hist.sort_values(["date","student"], inplace=True)
    hist.to_csv(HIST_PATH, index=False)
    return hist

# Simple HTML using two cards + history table + chart
def render_html(result_a, result_b, history_df):
    def card(r):
        pct = f"{r['prediction_pct']:+.2f}%"
        return f"""
        <div class="card">
          <div class="title">{r['label']}</div>
          <div class="pred">{pct}</div>
          <div class="meta">for {r['predicted_date']} • normalized with {r['norm_source']}</div>
        </div>
        """
    # prepare compact recent table (last 30 days)
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
<html lang="en">
<head>
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
  <div class="time">Last updated: {result_a['timestamp_utc']}</div>
  <div class="grid">
    {card(result_a)}
    {card(result_b)}
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

  <div class="footer">Runs on weekdays around 12:00 UTC for predictions and 23:00 UTC for actuals via GitHub Actions.</div>

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
      y: {{
        title: {{ display:true, text:'Return (%)' }}
      }}
    }},
    plugins: {{
      legend: {{ display: true }}
    }}
  }}
}});
</script>
</body>
</html>
"""

def main():
    res_normal = predict_with_student(NORMAL_RES_PATH, BINS_NORMAL, "Student A — Normal Resolution (±4%, 0.5%)")
    res_finer  = predict_with_student(FINER_RES_PATH,  BINS_FINER,  "Student B — Finer Resolution (±2.5%, 0.1%)")

    # Save predictions to history (morning job)
    hist = append_predictions_for_day([res_normal, res_finer])

    # Render page
    html = render_html(res_normal, res_finer, hist)
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote docs/index.html")

    # Console summary
    print("\n=== Predictions ===")
    for r in [res_normal, res_finer]:
        print(f"{r['label']}: {r['prediction_pct']:+.2f}% for {r['predicted_date']}")
    print(f"Timestamp (UTC): {res_normal['timestamp_utc']}")
    print("===================\n")

if __name__ == "__main__":
    main()
