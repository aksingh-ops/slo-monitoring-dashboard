"""
Phase 2 — SLO Design and Threshold Analysis
=============================================
Defines 16 calibrated SLO thresholds across 4 layers and 4 metrics,
applies Green/Amber/Red classification to every reading, and generates
the EDA charts showing where the portfolio stands against each threshold.

Design principles
------------------
1. Each layer gets its own targets, not a single firm-wide threshold.
   Downstream layers naturally have worse metrics due to accumulated
   processing latency. Client Reporting has looser SLOs than Market Data
   Feed because its position at the end of the chain means more factors
   contribute to its performance.

2. Latency SLOs are time-aware. Market Data Feed runs at 114.8ms average
   during market hours (09:00-16:00 ET) versus 65.7ms overnight, a 1.75x
   ratio driven by peak load. A flat 120ms Green threshold would fire on
   29% of normal market-hours readings, desensitising the ops team to
   alerts. Time-aware thresholds widen by 25-30% during market hours.

3. Error budget tracking. Each layer has a monthly error budget. The
   burn rate tells leadership whether a layer is on track to exhaust
   its budget before the end of the month.

SLO thresholds (Green / Amber / Red)
--------------------------------------
  Market Data Feed    availability >= 99.5% / >= 99.0% / < 98.0%
                      latency P95  <= 120ms / <= 180ms / >  300ms
                      completeness >= 99.5% / >= 99.0% / < 98.0%
                      freshness    <= 60s   / <= 90s   / >  180s

  Portfolio Valuation availability >= 99.0% / >= 98.5% / < 97.5%
                      latency P95  <= 180ms / <= 280ms / >  500ms
                      completeness >= 99.0% / >= 98.5% / < 97.5%
                      freshness    <= 120s  / <= 180s  / >  360s

  Risk Dashboard      availability >= 98.5% / >= 98.0% / < 97.0%
                      latency P95  <= 220ms / <= 350ms / >  600ms
                      completeness >= 98.5% / >= 98.0% / < 97.0%
                      freshness    <= 180s  / <= 270s  / >  540s

  Client Reporting    availability >= 98.0% / >= 97.5% / < 96.5%
                      latency P95  <= 300ms / <= 500ms / >  900ms
                      completeness >= 98.0% / >= 97.5% / < 96.5%
                      freshness    <= 300s  / <= 450s  / >  900s

Outputs
-------
  data/slo_metrics_processed.csv  140,544 rows with SLO status columns
  outputs/fig2_slo_design.png

Run
---
  python phase2_slo_design.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = "../data/slo_metrics_raw.csv"
OUTPUT_DIR = "../outputs"
DATA_DIR   = "../data"

C_BG = "#FAFAFA"
LAYER_COLORS = {
    "Market Data Feed":    "#185FA5",
    "Portfolio Valuation": "#534AB7",
    "Risk Dashboard":      "#BA7517",
    "Client Reporting":    "#A32D2D",
}
LAYERS = ["Market Data Feed","Portfolio Valuation","Risk Dashboard","Client Reporting"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
})

SLO_CONFIG = {
    "Market Data Feed": {
        "availability_pct":      {"green":99.5,"amber":99.0,"red":98.0},
        "latency_p95_ms":        {"green":120, "amber":180, "red":300,
                                  "market_green":150,"market_amber":220,"market_red":400},
        "data_completeness_pct": {"green":99.5,"amber":99.0,"red":98.0},
        "freshness_lag_sec":     {"green":60,  "amber":90,  "red":180},
    },
    "Portfolio Valuation": {
        "availability_pct":      {"green":99.0,"amber":98.5,"red":97.5},
        "latency_p95_ms":        {"green":180, "amber":280, "red":500,
                                  "market_green":220,"market_amber":350,"market_red":600},
        "data_completeness_pct": {"green":99.0,"amber":98.5,"red":97.5},
        "freshness_lag_sec":     {"green":120, "amber":180, "red":360},
    },
    "Risk Dashboard": {
        "availability_pct":      {"green":98.5,"amber":98.0,"red":97.0},
        "latency_p95_ms":        {"green":220, "amber":350, "red":600,
                                  "market_green":280,"market_amber":450,"market_red":750},
        "data_completeness_pct": {"green":98.5,"amber":98.0,"red":97.0},
        "freshness_lag_sec":     {"green":180, "amber":270, "red":540},
    },
    "Client Reporting": {
        "availability_pct":      {"green":98.0,"amber":97.5,"red":96.5},
        "latency_p95_ms":        {"green":300, "amber":500, "red":900,
                                  "market_green":400,"market_amber":650,"market_red":1100},
        "data_completeness_pct": {"green":98.0,"amber":97.5,"red":96.5},
        "freshness_lag_sec":     {"green":300, "amber":450, "red":900},
    },
}


def classify_status(row, metric, config):
    val = row[metric]
    cfg = config[row["layer"]][metric]
    mkt = row["is_market_hrs"]

    if metric in ["latency_p95_ms", "freshness_lag_sec"]:
        g = cfg.get("market_green", cfg["green"]) if mkt else cfg["green"]
        a = cfg.get("market_amber", cfg["amber"]) if mkt else cfg["amber"]
        if   val <= g: return "Green"
        elif val <= a: return "Amber"
        else:          return "Red"
    else:
        if   val >= cfg["green"]: return "Green"
        elif val >= cfg["amber"]: return "Amber"
        else:                     return "Red"


def apply_slo_classifications(df):
    df["hour"]          = df["timestamp"].dt.hour
    df["is_market_hrs"] = df["hour"].between(9, 15)
    status_rank         = {"Green":0,"Amber":1,"Red":2}

    for metric in ["availability_pct","latency_p95_ms","data_completeness_pct","freshness_lag_sec"]:
        df[f"{metric}_status"] = df.apply(
            lambda r: classify_status(r, metric, SLO_CONFIG), axis=1
        )

    status_cols = [c for c in df.columns if c.endswith("_status")]
    df["composite_status"] = df[status_cols].apply(
        lambda row: ["Green","Amber","Red"][max(status_rank[v] for v in row)], axis=1
    )
    return df


def print_slo_summary(df):
    print("SLO status distribution:")
    print(f"  {'Layer':<22} {'Red %':>7} {'Amber %':>8} {'Green %':>9}")
    print("  " + "-"*50)
    for layer in LAYERS:
        sub = df[df["layer"] == layer]
        r = (sub["composite_status"] == "Red").mean()   * 100
        a = (sub["composite_status"] == "Amber").mean() * 100
        g = (sub["composite_status"] == "Green").mean() * 100
        print(f"  {layer:<22} {r:>7.2f}%  {a:>7.2f}%  {g:>8.2f}%")

    # Time-aware impact
    mdf = df[df["layer"] == "Market Data Feed"]
    mkt = mdf[mdf["is_market_hrs"]]["latency_p95_ms"].mean()
    off = mdf[~mdf["is_market_hrs"]]["latency_p95_ms"].mean()
    flat_alarm = (mdf[mdf["is_market_hrs"]]["latency_p95_ms"] > 120).mean() * 100
    print(f"\nTime-aware SLO impact (Market Data Feed latency)")
    print(f"  Market hours avg  : {mkt:.1f}ms")
    print(f"  Off-hours avg     : {off:.1f}ms")
    print(f"  Ratio             : {mkt/off:.2f}x")
    print(f"  False alarms (flat threshold): {flat_alarm:.1f}% of market-hours readings")


def plot_slo_design(df, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    # Monthly availability heatmap
    ax = axes[0]
    df["month"] = df["timestamp"].dt.month
    monthly = df.groupby(["layer","month"])["availability_pct"].mean().unstack()
    monthly = monthly.loc[LAYERS]
    months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    im = ax.imshow(monthly.values, cmap="RdYlGn", aspect="auto", vmin=97, vmax=100)
    ax.set_xticks(range(12)); ax.set_xticklabels(months, fontsize=8)
    ax.set_yticks(range(4));  ax.set_yticklabels(["MDF","Port.Val","Risk","Client"], fontsize=9)
    for i in range(4):
        for j in range(12):
            v = monthly.values[i, j]
            ax.text(j, i, f"{v:.2f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if v < 98.5 else "#222")
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Monthly availability heatmap\nRed = SLO breach", fontweight="bold")
    ax.grid(False)

    # SLO status stacked bars
    ax = axes[1]
    short = ["MDF","Port.Val","Risk","Client"]
    x     = np.arange(4)
    g_p = [(df[df["layer"]==l]["composite_status"]=="Green").mean()*100 for l in LAYERS]
    a_p = [(df[df["layer"]==l]["composite_status"]=="Amber").mean()*100 for l in LAYERS]
    r_p = [(df[df["layer"]==l]["composite_status"]=="Red").mean()*100   for l in LAYERS]
    ax.bar(x, g_p, color="#3B6D11", alpha=0.82, label="Green", width=0.6)
    ax.bar(x, a_p, bottom=g_p, color="#BA7517", alpha=0.82, label="Amber", width=0.6)
    ax.bar(x, r_p, bottom=[g+a for g,a in zip(g_p,a_p)], color="#A32D2D", alpha=0.82, label="Red", width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel("% of readings"); ax.set_ylim(0, 108)
    ax.set_title("SLO status distribution by layer\n(stacked — % of all readings)", fontweight="bold")
    ax.legend(fontsize=9)
    for i, r in enumerate(r_p):
        ax.text(i, 102, f"{r:.1f}%\nRed", ha="center", fontsize=7.5,
                color="#A32D2D", fontweight="bold")

    # Market vs off-hours latency distribution
    ax = axes[2]
    mdf = df[df["layer"] == "Market Data Feed"]
    mkt_lat  = mdf[mdf["is_market_hrs"]]["latency_p95_ms"]
    offh_lat = mdf[~mdf["is_market_hrs"]]["latency_p95_ms"]
    ax.hist(offh_lat, bins=50, alpha=0.65, color="#185FA5", density=True, label="Off-hours")
    ax.hist(mkt_lat,  bins=50, alpha=0.65, color="#A32D2D", density=True, label="Market hours")
    ax.axvline(120, color="#3B6D11", linestyle="--", linewidth=1.5, label="Green SLO 120ms")
    ax.axvline(180, color="#BA7517", linestyle="--", linewidth=1.5, label="Amber SLO 180ms")
    ax.set_xlabel("Latency P95 (ms)")
    ax.set_title("MDF latency: market vs off-hours\nwhy time-aware SLOs are essential", fontweight="bold")
    ax.legend(fontsize=8)

    # Dependency map
    ax = axes[3]; ax.axis("off")
    boxes = [
        ("External\nMarket Feeds", "#F2F2F2","#888780"),
        ("Market Data\nFeed (L1)",  "#E6F1FB","#185FA5"),
        ("Portfolio\nValuation (L2)","#EEEDFE","#534AB7"),
        ("Risk\nDashboard (L3)",    "#FAEEDA","#BA7517"),
        ("Client\nReporting (L4)",  "#FCEBEB","#A32D2D"),
    ]
    y_pos = [0.86,0.69,0.51,0.33,0.15]
    lags  = ["","2-5 min lag","3-8 min lag","5-12 min lag",""]
    for i,(lbl,bg,fg) in enumerate(boxes):
        rect = plt.Rectangle((0.08,y_pos[i]-0.07),0.56,0.13,facecolor=bg,edgecolor=fg,
                              lw=2,transform=ax.transAxes,clip_on=False)
        ax.add_patch(rect)
        ax.text(0.36, y_pos[i]+0.005, lbl, transform=ax.transAxes,
                ha="center",va="center",fontsize=9,fontweight="bold",color=fg)
        if i > 0:
            ax.annotate("",xy=(0.36,y_pos[i]+0.13),xytext=(0.36,y_pos[i-1]-0.07+0.01),
                        xycoords="axes fraction",textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->",color="#555",lw=1.5))
        if i > 0 and i < 4:
            ax.text(0.72,y_pos[i]+0.04,lags[i],transform=ax.transAxes,
                    ha="left",va="center",fontsize=8,color="#A32D2D",style="italic")
    ax.set_title("Upstream to downstream\ndependency chain", fontweight="bold")
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    # Completeness vs availability scatter
    ax = axes[4]
    sample = df[df["layer"].isin(["Market Data Feed","Risk Dashboard"])].sample(3000, random_state=42)
    for layer in ["Market Data Feed","Risk Dashboard"]:
        sub = sample[sample["layer"] == layer]
        ax.scatter(sub["data_completeness_pct"], sub["availability_pct"],
                   alpha=0.25, s=8, color=LAYER_COLORS[layer], label=layer[:12])
    ax.set_xlabel("Data completeness %"); ax.set_ylabel("Availability %")
    ax.set_title("Completeness vs availability\n(correlated degradation pattern)", fontweight="bold")
    ax.legend(fontsize=9)

    # SLO breach by month bar
    ax = axes[5]
    breach_month = df[df["composite_status"]=="Red"].groupby(
        [df["timestamp"].dt.month,"layer"]).size().unstack(fill_value=0)
    breach_month.index = months
    bottom = np.zeros(12)
    for layer, color in zip(LAYERS, ["#185FA5","#534AB7","#BA7517","#A32D2D"]):
        if layer in breach_month.columns:
            vals = breach_month[layer].values
            ax.bar(range(12), vals, bottom=bottom, color=color, alpha=0.82,
                   label=layer[:12], width=0.7)
            bottom += vals
    ax.set_xticks(range(12)); ax.set_xticklabels(months, fontsize=8)
    ax.set_ylabel("Red SLO readings")
    ax.set_title("SLO breaches (Red) by month\nall layers combined", fontweight="bold")
    ax.legend(fontsize=8)
    ax.axvspan(7, 7.7, color="#FCEBEB", alpha=0.5)
    ax.text(7.35, ax.get_ylim()[1]*0.85,"Aug\nOutage",ha="center",fontsize=8,
            color="#A32D2D",fontweight="bold")

    fig.suptitle("Phase 2 — SLO Design and Threshold Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=2)
    plt.savefig(f"{output_dir}/fig2_slo_design.png", dpi=140, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig2_slo_design.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading metrics and applying SLO classifications...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = apply_slo_classifications(df)

    print_slo_summary(df)

    df.to_csv(f"{DATA_DIR}/slo_metrics_processed.csv", index=False)
    print(f"\nSaved: {DATA_DIR}/slo_metrics_processed.csv  ({len(df):,} rows)")

    plot_slo_design(df, OUTPUT_DIR)
    print("\nPhase 2 complete. Run phase3_anomaly_detection.py next.")
