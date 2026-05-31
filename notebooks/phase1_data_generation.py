"""
Phase 1 — Synthetic Metric Generation
========================================
Generates 12 months of operational metrics at 15-minute intervals across
four financial platform layers. Eight failure events are injected as ground
truth for the anomaly detection pipeline to find.

System architecture (upstream to downstream)
----------------------------------------------
  L1: Market Data Feed      external prices, rates, indices
  L2: Portfolio Valuation   depends on L1, computes P&L and positions
  L3: Risk Dashboard        depends on L2, shows exposure and limits
  L4: Client Reporting      depends on L3, final downstream consumer

A failure at L1 propagates to L4 in approximately 10-25 minutes depending
on pipeline refresh cycles. The monitoring system must detect L1 degradation
before L4 consumers notice.

Four SLIs (Service Level Indicators) per layer
------------------------------------------------
  availability_pct        is the system up and serving data?
  latency_p95_ms          how fast does data refresh? (P95)
  data_completeness_pct   is the data complete, no dropped records?
  freshness_lag_sec       how current is the data end to end?

Eight injected failure events (ground truth)
---------------------------------------------
  Jan  MDF outage          1 layer   availability ~60%   2h 15min
  Feb  Cascade failure     3 layers  availability ~72%   4.5hr
  Mar  Latency spike       2 layers  latency 5x normal   45min
  Apr  Data quality        1 layer   completeness ~78%   overnight
  Jun  Freshness breach    1 layer   freshness 8x lag    1.5hr
  Aug  Major outage        4 layers  availability ~44%   8.5hr market hours
  Oct  Latency degradation 2 layers  latency 3.5x        2hr
  Dec  EOY data gap        1 layer   completeness ~82%   overnight

Outputs
-------
  data/slo_metrics_raw.csv    140,544 readings (15-min x 4 layers x 8,784 hours)
  outputs/fig1_data_understanding.png

Run
---
  python phase1_data_generation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "../outputs"
DATA_DIR   = "../data"
SEED       = 42
np.random.seed(SEED)

LAYERS = [
    "Market Data Feed",
    "Portfolio Valuation",
    "Risk Dashboard",
    "Client Reporting",
]

LAYER_COLORS = {
    "Market Data Feed":    "#185FA5",
    "Portfolio Valuation": "#534AB7",
    "Risk Dashboard":      "#BA7517",
    "Client Reporting":    "#A32D2D",
}

C_BG = "#FAFAFA"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
})


def base_signal(n, daily_amp=0, weekly_amp=0, noise_std=1.0, base=0):
    """Seasonal signal with daily and weekly components plus noise."""
    t      = np.arange(n)
    daily  = daily_amp  * np.sin(2 * np.pi * t / 96)      # 96 intervals per day
    weekly = weekly_amp * np.sin(2 * np.pi * t / (96 * 7)) # 672 intervals per week
    noise  = np.random.normal(0, noise_std, n)
    return base + daily + weekly + noise


def generate_layer_metrics(n, base_avail, base_lat, base_comp, base_fresh,
                            avail_noise=0.4, lat_noise=8, comp_noise=0.3, fresh_noise=5):
    """Generate 4 SLI streams for one layer with realistic seasonal patterns."""
    hour         = np.tile(np.repeat(np.arange(24), 4), n // 96 + 1)[:n]
    avail        = base_signal(n, daily_amp=0.2, noise_std=avail_noise, base=base_avail)
    avail        = np.clip(avail, 91, 100)
    latency      = 80 + 40 * np.sin(np.pi * (hour - 6) / 12) + base_signal(n, noise_std=lat_noise)
    latency      = latency * (base_lat / 80)
    latency      = np.clip(latency, 20, 2000)
    completeness = base_signal(n, noise_std=comp_noise, base=base_comp)
    completeness = np.clip(completeness, 83, 100)
    freshness    = base_signal(n, daily_amp=15, noise_std=fresh_noise, base=base_fresh)
    freshness    = np.clip(freshness, 5, 1800)
    return avail, latency, completeness, freshness


def build_metrics(idx):
    n = len(idx)
    configs = [
        ("Market Data Feed",    99.7, 1.00, 99.5, 45),
        ("Portfolio Valuation", 99.4, 1.30, 99.1, 81),
        ("Risk Dashboard",      99.1, 1.55, 98.8, 122),
        ("Client Reporting",    98.8, 2.18, 98.3, 243),
    ]
    frames = []
    for layer, base_avail, lat_mult, base_comp, base_fresh in configs:
        avail, lat, comp, fresh = generate_layer_metrics(
            n, base_avail, 80 * lat_mult, base_comp, base_fresh
        )
        frames.append(pd.DataFrame({
            "timestamp":             idx,
            "layer":                 layer,
            "availability_pct":      avail.round(3),
            "latency_p95_ms":        lat.round(1),
            "data_completeness_pct": comp.round(3),
            "freshness_lag_sec":     fresh.round(1),
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "layer"]).reset_index(drop=True)


def inject_failures(df):
    """Apply the 8 ground-truth failure events."""
    failures = [
        ("Jan MDF outage",       "2024-01-15 09:30", "2024-01-15 11:45",
         ["Market Data Feed"],                                           "availability_pct",      0.60),
        ("Feb cascade",          "2024-02-08 10:00", "2024-02-08 14:30",
         ["Market Data Feed","Portfolio Valuation","Risk Dashboard"],    "availability_pct",      0.72),
        ("Mar latency spike",    "2024-03-22 09:15", "2024-03-22 10:00",
         ["Portfolio Valuation","Risk Dashboard"],                       "latency_p95_ms",        5.0),
        ("Apr data quality",     "2024-04-11 00:00", "2024-04-11 06:00",
         ["Market Data Feed"],                                           "data_completeness_pct", 0.78),
        ("Jun freshness breach", "2024-06-03 14:00", "2024-06-03 15:30",
         ["Client Reporting"],                                           "freshness_lag_sec",     8.0),
        ("Aug major outage",     "2024-08-19 09:30", "2024-08-19 18:00",
         ["Market Data Feed","Portfolio Valuation","Risk Dashboard","Client Reporting"],
                                                                         "availability_pct",      0.45),
        ("Oct latency degrade",  "2024-10-07 11:00", "2024-10-07 13:00",
         ["Risk Dashboard","Client Reporting"],                          "latency_p95_ms",        3.5),
        ("Dec data gap",         "2024-12-27 20:00", "2024-12-28 04:00",
         ["Client Reporting"],                                           "data_completeness_pct", 0.82),
    ]

    print("Injected failure events:")
    for name, start, end, layers, metric, mult in failures:
        mask = (
            (df["timestamp"] >= pd.Timestamp(start)) &
            (df["timestamp"] <= pd.Timestamp(end)) &
            (df["layer"].isin(layers))
        )
        if metric in ["availability_pct", "data_completeness_pct"]:
            df.loc[mask, metric] = (df.loc[mask, metric] * mult).clip(0, 100)
        else:
            df.loc[mask, metric] = df.loc[mask, metric] * mult
        print(f"  {name:<25} layers={len(layers)}  metric={metric}  mult=x{mult}  rows={mask.sum():,}")

    return df


def plot_data_understanding(df, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Availability all layers hourly
    ax = axes[0]
    df_h = df.set_index("timestamp").groupby("layer").resample("1h")["availability_pct"].mean().reset_index()
    for layer in LAYERS:
        sub = df_h[df_h["layer"] == layer].sort_values("timestamp")
        ax.plot(sub["timestamp"], sub["availability_pct"],
                color=LAYER_COLORS[layer], linewidth=0.7, alpha=0.85, label=layer[:12])
    ax.axhline(99.5, color="#A32D2D", linestyle="--", linewidth=1, alpha=0.7, label="SLO 99.5%")
    ax.set_title("Availability all layers (hourly)", fontweight="bold")
    ax.set_ylabel("Availability %"); ax.legend(fontsize=7, ncol=2); ax.set_ylim(40, 101)

    # Latency all layers hourly
    ax = axes[1]
    df_lat = df.set_index("timestamp").groupby("layer").resample("1h")["latency_p95_ms"].mean().reset_index()
    for layer in LAYERS:
        sub = df_lat[df_lat["layer"] == layer].sort_values("timestamp")
        ax.plot(sub["timestamp"], sub["latency_p95_ms"],
                color=LAYER_COLORS[layer], linewidth=0.7, alpha=0.85, label=layer[:12])
    ax.axhline(200, color="#A32D2D", linestyle="--", linewidth=1, alpha=0.7, label="SLO 200ms")
    ax.set_title("Latency P95 all layers (hourly)", fontweight="bold")
    ax.set_ylabel("Latency P95 (ms)"); ax.legend(fontsize=7, ncol=2)

    # Feb cascade detail
    ax = axes[2]
    feb_s = pd.Timestamp("2024-02-08 09:00")
    feb_e = pd.Timestamp("2024-02-08 16:00")
    df_feb = df_h[(df_h["timestamp"] >= feb_s) & (df_h["timestamp"] <= feb_e)]
    for layer in LAYERS[:3]:
        sub = df_feb[df_feb["layer"] == layer].sort_values("timestamp")
        ax.plot(sub["timestamp"], sub["availability_pct"],
                color=LAYER_COLORS[layer], linewidth=1.5, marker="o", markersize=3,
                alpha=0.9, label=layer[:12])
    ax.axvline(pd.Timestamp("2024-02-08 10:00"), color="#A32D2D",
               linestyle="--", linewidth=1.5, alpha=0.8, label="Outage start")
    ax.set_title("Feb cascade failure\nupstream failure propagates downstream", fontweight="bold")
    ax.set_ylabel("Availability %"); ax.legend(fontsize=8); ax.set_ylim(65, 101)

    # Hourly latency pattern
    ax = axes[3]
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby(["layer", "hour"])["latency_p95_ms"].mean().reset_index()
    for layer in LAYERS:
        sub = hourly[hourly["layer"] == layer]
        ax.plot(sub["hour"], sub["latency_p95_ms"],
                color=LAYER_COLORS[layer], linewidth=2, marker="o", markersize=4,
                alpha=0.85, label=layer[:12])
    ax.axvspan(9, 16, color="#FAEEDA", alpha=0.3, label="Market hours")
    ax.set_title("Avg latency by hour of day\n(market hours highlighted)", fontweight="bold")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Avg latency P95 (ms)")
    ax.set_xticks(range(0, 24, 3)); ax.legend(fontsize=8)

    # Failure severity
    ax = axes[4]
    events = ["Jan\nOutage","Feb\nCascade","Mar\nLatency","Apr\nQuality",
              "Jun\nFreshness","Aug\nMajor","Oct\nLatency","Dec\nGap"]
    severity = [2, 4, 2, 2, 1, 5, 2, 2]
    layers_n = [1, 3, 2, 1, 1, 4, 2, 1]
    colors_s = ["#3B6D11","#A32D2D","#BA7517","#BA7517","#3B6D11","#A32D2D","#BA7517","#BA7517"]
    bars = ax.bar(events, severity, color=colors_s, alpha=0.85, width=0.6)
    for bar, lc in zip(bars, layers_n):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{lc}L", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Severity (1=Low  5=Critical)")
    ax.set_title("Injected events — severity and layers\n(L = layers affected)", fontweight="bold")
    ax.set_ylim(0, 6.5)

    # Source system overview panel
    ax = axes[5]
    ax.axis("off")
    info = [
        ("140,544", "metric readings", "15-min x 4 layers x 12 months", "#E6F1FB","#185FA5"),
        ("8",       "failure events",  "range: 1-layer outage to 4-layer cascade","#FCEBEB","#A32D2D"),
        ("4",       "SLI dimensions",  "availability, latency, completeness, freshness","#FAEEDA","#854F0B"),
        ("16",      "SLO thresholds",  "calibrated per layer, time-aware for market hours","#EAF3DE","#3B6D11"),
    ]
    for i,(val,lbl,detail,bg,fg) in enumerate(info):
        y = 0.82 - i*0.22
        rect = plt.Rectangle((0.03, y-0.08), 0.94, 0.18, facecolor=bg, edgecolor=fg,
                              linewidth=1.5, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.12, y+0.01, val, transform=ax.transAxes, fontsize=18,
                fontweight="bold", color=fg, va="center")
        ax.text(0.30, y+0.04, lbl, transform=ax.transAxes, fontsize=10,
                fontweight="bold", color=fg, va="center")
        ax.text(0.30, y-0.03, detail, transform=ax.transAxes, fontsize=8,
                color="#555", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Dataset profile", fontweight="bold")

    fig.suptitle("Phase 1 — Operational metric generation and data understanding",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2)
    plt.savefig(f"{output_dir}/fig1_data_understanding.png",
                dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig1_data_understanding.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating 12 months of operational metrics...")
    start = pd.Timestamp("2024-01-01")
    end   = pd.Timestamp("2024-12-31 23:45:00")
    idx   = pd.date_range(start, end, freq="15min")
    print(f"  {len(idx):,} time points x 4 layers = {len(idx)*4:,} total readings")

    df = build_metrics(idx)
    df = inject_failures(df)

    df.to_csv(f"{DATA_DIR}/slo_metrics_raw.csv", index=False)
    print(f"\nSaved: {DATA_DIR}/slo_metrics_raw.csv  ({len(df):,} rows)")

    plot_data_understanding(df, OUTPUT_DIR)
    print("\nPhase 1 complete. Run phase2_slo_design.py next.")
