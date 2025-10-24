import os
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from string import Template
import requests
from bs4 import BeautifulSoup

# Paths / constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../repo

DOCS_DIR = "docs"
HIST_PATH = os.path.join(DOCS_DIR, "history.csv")
INDEX_PATH = os.path.join(DOCS_DIR, "index.html")
TICKER = "SPY"

# Specific paths
DOCS_DIR = os.path.join(BASE_DIR, DOCS_DIR)
HIST_PATH = os.path.join(BASE_DIR, HIST_PATH)
INDEX_PATH = os.path.join(BASE_DIR, INDEX_PATH)

NEWS_PATH = os.path.join(DOCS_DIR, "news.csv")

# Scraper (Yahoo Finance homepage lead + related)
YF_HOME = "https://finance.yahoo.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _clean_text(t: str | None) -> str:
    return " ".join((t or "").split())

def _extract_title_and_link(a_tag):
    if a_tag is None:
        return None
    href = a_tag.get("href", "")
    if href.startswith("/"):
        href = "https://finance.yahoo.com" + href
    if "https://finance.yahoo.com/news/" not in href:
        return None
    title = a_tag.get("title")
    if not title:
        h = a_tag.select_one('[data-testid="title"], h2, h3')
        title = h.get_text(strip=True) if h else a_tag.get_text(strip=True)
    title = _clean_text(title)
    if not title:
        return None
    return {"title": title, "link": href}

def get_yahoo_home_top3():
    # Scrape the hero lead + two related stories from Yahoo Finance homepage.
    try:
        with requests.Session() as s:
            r = s.get(YF_HOME, headers=HEADERS, timeout=12)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")

        results = []
        # Lead story
        lead_block = soup.select_one('[data-testid="hero-lead-story"]')
        if lead_block:
            lead_anchor = lead_block.select_one('a.titles-link, div.content a[href*="/news/"]')
            lead = _extract_title_and_link(lead_anchor)
            if lead:
                lead["publisher"] = "Yahoo Finance"
                results.append(lead)

        # Two related items
        related_block = soup.select_one(".hero-related")
        if related_block:
            for a in related_block.select('[data-testid="storyitem"] a[href*="/news/"]'):
                item = _extract_title_and_link(a)
                if item:
                    item["publisher"] = "Yahoo Finance"
                    results.append(item)
                if len(results) >= 3:
                    break

        # Deduplicate by link, preserve order
        deduped, seen = [], set()
        for x in results:
            if x["link"] not in seen:
                seen.add(x["link"])
                deduped.append(x)

        return deduped[:3]
    except Exception as e:
        print("Yahoo home scrape failed:", e)
        return []

# News helpers
def ensure_news_csv():
    os.makedirs(DOCS_DIR, exist_ok=True)
    if not os.path.exists(NEWS_PATH):
        pd.DataFrame(columns=["date", "rank", "title", "link", "publisher"]).to_csv(NEWS_PATH, index=False)

def today_et_str():
    return pd.Timestamp.now(tz="America/New_York").normalize().date().isoformat()

def fetch_top3_news_today():
    # Preferred: homepage scrape
    items = get_yahoo_home_top3()
    if items:
        out = []
        for i, it in enumerate(items, start=1):
            out.append({
                "rank": i,
                "title": it["title"],
                "link": it["link"],
                "publisher": it.get("publisher", "Yahoo Finance")[:128],
            })
        return out

    # Fallback: yfinance news
    try:
        tk = yf.Ticker(TICKER)
        news = tk.news or []
    except Exception:
        news = []
    out = []
    for i, it in enumerate(news[:3], start=1):
        title = it.get("title") or ""
        link = it.get("link") or ""
        pub  = (it.get("publisher") or "")[:128] or "Yahoo Finance"
        if title and link:
            out.append({"rank": i, "title": title, "link": link, "publisher": pub})
    return out

def append_today_news_if_missing():
    ensure_news_csv()
    df = pd.read_csv(NEWS_PATH)
    d = today_et_str()
    if (df["date"] == d).any():
        return False
    top3 = fetch_top3_news_today()
    if not top3:
        return False
    rows = [{"date": d, **row} for row in top3]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(NEWS_PATH, index=False)
    return True

# Rendering helpers
def render_html_cards_from_latest(hist: pd.DataFrame):
    if hist.empty:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        card_a = {"label":"Student A — Normal Resolution (±4%, 0.5%)",
                  "predicted_date":"—","prediction_pct":0.0,"norm_source":"", "timestamp_utc":ts}
        card_b = {"label":"Student B — Finer Resolution (±2.5%, 0.1%)",
                  "predicted_date":"—","prediction_pct":0.0,"norm_source":"", "timestamp_utc":ts}
        return card_a, card_b, ts

    last_date = hist["date"].max()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    subset = hist[hist["date"]==last_date].sort_values("student")

    a_row = subset[subset["student"].str.contains("Normal", na=False)].head(1)
    b_row = subset[subset["student"].str.contains("Finer", na=False)].head(1)
    if a_row.empty and not subset.empty:
        a_row = subset.iloc[[0]]
    if b_row.empty and len(subset) >= 2:
        b_row = subset.iloc[[1]]

    def mk_card(row, fallback_label):
        if row.empty:
            return {"label": fallback_label, "predicted_date": last_date,
                    "prediction_pct": 0.0, "norm_source": "", "timestamp_utc": ts}
        r = row.iloc[0]
        return {
            "label": r["student"],
            "predicted_date": r["date"],
            "prediction_pct": float(r["prediction_pct"]),
            "norm_source": "training stats",
            "timestamp_utc": ts
        }

    card_a = mk_card(a_row, "Student A — Normal Resolution (±4%, 0.5%)")
    card_b = mk_card(b_row, "Student B — Finer Resolution (±2.5%, 0.1%)")
    return card_a, card_b, ts

# News map to JS
def build_news_map_for_js(news_df: pd.DataFrame) -> str:
    if news_df is None or news_df.empty:
        return "{}"
    cols = {"date", "rank", "title", "link"}
    have = [c for c in news_df.columns if c in cols]
    df = news_df[have].copy()
    if "rank" in df.columns:
        df = df.sort_values(["date", "rank"])
    else:
        df = df.sort_values(["date"])
    news_map = {}
    for d, grp in df.groupby("date"):
        items = []
        for _, r in grp.iterrows():
            t = str(r.get("title", ""))
            l = str(r.get("link", ""))
            if t and l:
                items.append({"title": t, "link": l})
        if items:
            news_map[d] = items[:3]
    return json.dumps(news_map, ensure_ascii=False)

# Full HTML render 
def render_full_page(card_a, card_b, history_df, news_df):
    news_map_json = build_news_map_for_js(news_df)

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
    rows_html = "\n".join(
        f"<tr><td>{d}</td><td>{s}</td><td>{float(p):+.2f}%</td><td>{'' if pd.isna(a) else f'{float(a):+.2f}%'}</td></tr>"
        for d, s, p, a in zip(h["date"], h["student"], h["prediction_pct"], h["actual_pct"])
    )

    # series
    all_dates = sorted(set(h["date"]))
    hA = h[h["student"].str.contains("Normal", na=False)]
    hB = h[h["student"].str.contains("Finer", na=False)]
    hAct = h.groupby("date", as_index=False)["actual_pct"].mean()

    def make_series(df):
        m = df.set_index("date")["prediction_pct"] if "prediction_pct" in df.columns else df.set_index("date")["actual_pct"]
        out = []
        for d in all_dates:
            if d in m.index:
                val = m.loc[d]
                out.append("null" if pd.isna(val) else f"{float(val):.4f}")
            else:
                out.append("null")
        return ",".join(out)

    labels_js = ",".join([f"'{d}'" for d in all_dates])
    predsA_js = make_series(hA)
    predsB_js = make_series(hB)
    acts_js   = make_series(hAct.rename(columns={"actual_pct": "prediction_pct"}))

    # colors
    colorA = "#4cc9f0"
    colorB = "#f72585"
    colorAct = "#e2e8f0"
    gridCol = "rgba(226, 232, 240, 0.12)"
    tickCol = "rgba(226, 232, 240, 0.75)"

    html_template = Template(r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Daily EOD Return Predictions</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f14; color:#e8eef7; margin:0; padding:2rem; }
  h1 { margin:0 0 1rem 0; font-weight:700; letter-spacing:0.2px; }
  .time { opacity:0.7; margin-bottom:2rem; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:1rem; }
  .card { background:#121824; border:1px solid #1c2433; border-radius:16px; padding:1.25rem; box-shadow:0 6px 24px rgba(0,0,0,0.25); }
  .title { font-size:0.95rem; opacity:0.9; margin-bottom:0.25rem; }
  .pred { font-size:2.2rem; font-weight:800; margin:0.5rem 0 0.25rem 0; }
  .meta { opacity:0.7; font-size:0.9rem; }
  .footer { opacity:0.6; font-size:0.85rem; margin-top:2rem; }
  table { width:100%; border-collapse:collapse; margin-top:2rem; }
  th, td { border-bottom:1px solid #1c2433; padding:0.5rem; text-align:left; }
  th { opacity:0.8; }
  .chart-wrap { background:#121824; border:1px solid #1c2433; border-radius:16px; padding:1rem; margin-top:2rem; position: relative; min-height: 1040px; }
  #histChart { width: 100% !important; height: 1040px !important; }
  #newsTooltip {
    position: absolute;
    background: #0f1624;
    border: 1px solid #25324a;
    border-radius: 10px;
    padding: 10px 12px;
    pointer-events: auto;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    max-width: 420px;
    display: none;
    z-index: 10;
  }
  #newsTooltip .date { font-weight: 700; margin-bottom: 6px; opacity: 0.9; }
  #newsTooltip a { color: #9cd1ff; text-decoration: none; }
  #newsTooltip a:hover { text-decoration: underline; }
  #newsTooltip ul { margin: 6px 0 0 1rem; padding: 0; }
  #newsTooltip li { margin: 4px 0; }
</style>
</head>
<body>
  <h1>Daily EOD Return Predictions — $TICKER</h1>
  <div class="time">Last updated: $LAST_UPDATED</div>
  <div class="grid">
    $CARD_A
    $CARD_B
  </div>

  <div class="chart-wrap">
    <canvas id="histChart"></canvas>
    <div id="newsTooltip"></div>
  </div>

  <table>
    <thead><tr><th>Date</th><th>Student</th><th>Pred (%)</th><th>Actual (%)</th></tr></thead>
    <tbody>
      $ROWS_HTML
    </tbody>
  </table>

  <div class="footer">Predictions ~12:00 UTC; Actuals filled ~23:00 UTC (post close). Hover the "Actual" line for clickable top-3 Yahoo Finance headlines.</div>

<script>
const newsMap = $NEWS_MAP_JSON;  // date -> [{title, link}, ...]

const ctx = document.getElementById('histChart').getContext('2d');
const tooltipEl = document.getElementById('newsTooltip');

// sticky hide logic (only hide after leaving BOTH canvas & tooltip for 1.5s)
let hideTimer = null;
const HIDE_DELAY_MS = 1500;
function cancelHide() { if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; } }
function scheduleHide() { cancelHide(); hideTimer = setTimeout(() => { tooltipEl.style.display = 'none'; }, HIDE_DELAY_MS); }
function hideTooltipNow() { cancelHide(); tooltipEl.style.display = 'none'; }

// clamp tooltip within chart container
function showTooltip(html, x, y, container) {
  cancelHide();
  tooltipEl.innerHTML = html;
  tooltipEl.style.display = 'block';
  tooltipEl.style.left = x + 'px';
  tooltipEl.style.top = y + 'px';

  const ttRect = tooltipEl.getBoundingClientRect();
  const contRect = container.getBoundingClientRect();
  const margin = 12;
  let left = x, top = y;

  if (left + ttRect.width + margin > contRect.width) left = contRect.width - ttRect.width - margin;
  if (left < margin) left = margin;
  if (top + ttRect.height + margin > contRect.height) top = contRect.height - ttRect.height - margin;
  if (top < margin) top = margin;

  tooltipEl.style.left = left + 'px';
  tooltipEl.style.top  = top  + 'px';
}
tooltipEl.addEventListener('mouseenter', () => cancelHide());
tooltipEl.addEventListener('mouseleave', () => scheduleHide());

// colors
const colorA  = '$COLOR_A';
const colorB  = '$COLOR_B';
const colorAct = '$COLOR_ACT';
const gridCol = '$GRID_COL';
const tickCol = '$TICK_COL';

const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [$LABELS],
    datasets: [
      {
        label: 'Student A — Normal',
        data: [$PREDS_A],
        borderColor: colorA,
        backgroundColor: colorA,
        pointBackgroundColor: colorA,
        pointBorderColor: colorA,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5,
        tension: 0.2
      },
      {
        label: 'Student B — Finer',
        data: [$PREDS_B],
        borderColor: colorB,
        backgroundColor: colorB,
        pointBackgroundColor: colorB,
        pointBorderColor: colorB,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5,
        tension: 0.2
      },
      {
        label: 'Actual',
        data: [$ACTS],
        borderColor: colorAct,
        backgroundColor: colorAct,
        pointBackgroundColor: colorAct,
        pointBorderColor: colorAct,
        borderDash: [6,4],
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6,
        tension: 0.2
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    scales: {
      x: {
        ticks: { color: tickCol },
        grid:  { color: gridCol }
      },
      y: {
        ticks: { color: tickCol },
        grid:  { color: gridCol },
        title: { display: true, text: 'Return (%)', color: tickCol }
      }
    },
    plugins: {
      legend: {
        labels: { color: tickCol }
      },
      tooltip: {
        enabled: false,  // use our HTML tooltip
        external: function(context) {
          const tooltip = context.tooltip;
          const containerNode = context.chart.canvas.parentNode;

          // Let mouseleave handlers decide when to hide
          if (!tooltip || tooltip.opacity === 0) return;

          const dp = tooltip.dataPoints && tooltip.dataPoints[0];
          if (!dp) return;

          // Only for "Actual"
          const dsLabel = dp.dataset.label || '';
          if (dsLabel !== 'Actual') return;

          const labelDate = dp.label;
          const items = newsMap[labelDate] || [];
          if (items.length === 0) return;

          let html = '<div class="date">' + labelDate + ' — Top News</div><ul>';
          for (const it of items) {
            const safeTitle = (it.title || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const safeLink  = (it.link  || '#').replace(/"/g, '&quot;');
            html += '<li><a href="' + safeLink + '" target="_blank" rel="noopener noreferrer">' + safeTitle + '</a></li>';
          }
          html += '</ul>';

          const canvasRect = context.chart.canvas.getBoundingClientRect();
          const containerRect = containerNode.getBoundingClientRect();
          const x = tooltip.caretX + (canvasRect.left - containerRect.left) + 12;
          const y = tooltip.caretY + (canvasRect.top  - containerRect.top)  + 12;

          showTooltip(html, x, y, containerNode);
        }
      }
    },
    onHover: (evt) => {
      const point = chart.getElementsAtEventForMode(evt, 'nearest', { intersect: false }, false)[0];
      if (point && chart.data.datasets[point.datasetIndex].label === 'Actual') {
        evt.native.target.style.cursor = 'pointer';
        cancelHide(); // hovering canvas; keep tooltip open
      } else {
        evt.native.target.style.cursor = 'default';
        // do NOT schedule hide here; user may be moving toward the tooltip
      }
    }
  }
});

// Only schedule hide when leaving the canvas (tooltip handles itself)
chart.canvas.addEventListener('mouseleave', () => scheduleHide());
chart.canvas.addEventListener('mouseenter', () => cancelHide());

// Optional tidy-ups
window.addEventListener('scroll', () => scheduleHide());
window.addEventListener('resize', () => scheduleHide());
</script>
</body></html>
""")

    html = html_template.substitute(
        TICKER=TICKER,
        LAST_UPDATED=card_a['timestamp_utc'],
        CARD_A=card(card_a),
        CARD_B=card(card_b),
        ROWS_HTML=rows_html,
        NEWS_MAP_JSON=news_map_json,
        LABELS=labels_js,
        PREDS_A=predsA_js,
        PREDS_B=predsB_js,
        ACTS=acts_js,
        COLOR_A=colorA,
        COLOR_B=colorB,
        COLOR_ACT=colorAct,
        GRID_COL=gridCol,
        TICK_COL=tickCol,
    )
    return html


# Data helpers
def fetch_actual_for_date(date_str: str) -> float | None:
    """
    Close-to-close % return for that US/Eastern date, in percent units.
    """
    df = yf.download(TICKER, period="10d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    df["DailyReturn"] = df["Close"].pct_change() * 100.0
    df = df.dropna()
    df_idx_dates = df.index.tz_localize(None).date
    mask = np.array([str(d) == date_str for d in df_idx_dates])
    if mask.any():
        return float(df.loc[mask, "DailyReturn"].iloc[0])
    return None

# Main
def main():
    if not os.path.exists(HIST_PATH):
        print("No history.csv yet, nothing to update.")
        return
    hist = pd.read_csv(HIST_PATH)
    if hist.empty:
        print("Empty history.csv, nothing to update.")
        return

    # Fill actuals
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

    # News
    try:
        if append_today_news_if_missing():
            print(f"Saved today's news for {today_et_str()}.")
        else:
            print("No new today's news to save.")
    except Exception as e:
        print("News step skipped due to error:", e)

    # Load news for JS
    ensure_news_csv()
    try:
        news_df = pd.read_csv(NEWS_PATH)
    except Exception:
        news_df = pd.DataFrame(columns=["date","rank","title","link","publisher"])

    # Render
    card_a, card_b, _ = render_html_cards_from_latest(hist)
    html = render_full_page(card_a, card_b, hist, news_df)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print("Updated docs/index.html with actuals + clickable news tooltips")

if __name__ == "__main__":
    main()
