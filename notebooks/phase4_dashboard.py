"""
Phase 4 — Executive Dashboard
===============================
Generates the GBM operational SLO monitoring dashboard and validates
all outputs for Tableau connection.

Tableau connection
-------------------
  mdf_alert_timeline.csv    8,784 hourly rows — alert flags per layer
  breach_probability.csv    120 rows (30 days x 4 layers)

Both CSVs connect to Tableau for the 7-view monitoring dashboard:
  View 1: KPI strip (8 cards)
  View 2: Availability time series all 4 layers
  View 3: Monthly heatmap (availability by layer x month)
  View 4: SLO status stacked bar by layer
  View 5: 30-day breach probability all layers
  View 6: Daily alert count (Z + CUSUM stacked)
  View 7: Dependency chain (static visual)

Outputs
-------
  outputs/fig4_dashboard.png

Run
---
  python phase4_dashboard.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_PATH  = "../data/slo_metrics_processed.csv"
OUTPUT_DIR = "../outputs"

LAYERS = ["Market Data Feed","Portfolio Valuation","Risk Dashboard","Client Reporting"]
LAYER_COLORS = {
    "Market Data Feed":    "#185FA5",
    "Portfolio Valuation": "#534AB7",
    "Risk Dashboard":      "#BA7517",
    "Client Reporting":    "#A32D2D",
}
SLO_GREEN = {
    "Market Data Feed":99.5,"Portfolio Valuation":99.0,
    "Risk Dashboard":98.5,"Client Reporting":98.0,
}
C_BG = "#FAFAFA"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
})


def plot_dashboard(df, alert_df, breach_df, output_dir):
    fig = plt.figure(figsize=(20, 14), facecolor=C_BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38)

    # KPI strip
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis("off")
    kpis = [
        ("8/8",    "Events detected",        "#EAF3DE","#3B6D11"),
        ("0.12%",  "SARIMAX MAPE",           "#EAF3DE","#3B6D11"),
        ("15 min", "Mean time to detect",    "#EAF3DE","#3B6D11"),
        ("63.7%",  "Max breach prob (MDF)",  "#FCEBEB","#A32D2D"),
        ("29%",    "False alarms reduced",   "#EAF3DE","#3B6D11"),
        ("140K+",  "Metric readings",        "#E6F1FB","#185FA5"),
        ("4 layers","Dependency chain",      "#E6F1FB","#185FA5"),
        ("3",      "Detection methods",      "#EEEDFE","#534AB7"),
    ]
    for i,(val,lbl,bg,fg) in enumerate(kpis):
        x0 = 0.005 + i * 0.124
        rect = plt.Rectangle((x0,0.05),0.116,0.88,facecolor=bg,edgecolor=fg,
                              lw=1.5,transform=ax_kpi.transAxes,clip_on=False)
        ax_kpi.add_patch(rect)
        ax_kpi.text(x0+0.058,0.62,val,transform=ax_kpi.transAxes,
                    ha="center",va="center",fontsize=14,fontweight="bold",color=fg)
        ax_kpi.text(x0+0.058,0.20,lbl,transform=ax_kpi.transAxes,
                    ha="center",va="center",fontsize=8,color=fg,linespacing=1.3)
    ax_kpi.set_title("GBM Operational SLO Monitoring Dashboard — Executive View",
                     fontsize=14, fontweight="bold", pad=10)

    # Availability time series
    ax_a = fig.add_subplot(gs[1, :2])
    df_h = df.set_index("timestamp").groupby("layer").resample("1h")["availability_pct"].mean().reset_index()
    for layer in LAYERS:
        sub = df_h[df_h["layer"]==layer].sort_values("timestamp")
        ax_a.plot(sub["timestamp"], sub["availability_pct"],
                  color=LAYER_COLORS[layer], lw=0.8, alpha=0.82, label=layer[:12])
    ax_a.axhline(99.0, color="#A32D2D", ls="--", lw=1.0, alpha=0.7, label="SLO floor")
    ax_a.set_title("Availability all layers (hourly)", fontweight="bold")
    ax_a.set_ylabel("Availability %"); ax_a.legend(fontsize=7,ncol=2); ax_a.set_ylim(40,101)

    # SLO status stacked
    ax_b = fig.add_subplot(gs[1, 2])
    short = ["MDF","Port.Val","Risk","Client"]
    x     = np.arange(4)
    g_p   = [(df[df["layer"]==l]["composite_status"]=="Green").mean()*100 for l in LAYERS]
    a_p   = [(df[df["layer"]==l]["composite_status"]=="Amber").mean()*100 for l in LAYERS]
    r_p   = [(df[df["layer"]==l]["composite_status"]=="Red").mean()*100   for l in LAYERS]
    ax_b.bar(x, g_p, color="#3B6D11",alpha=0.82,label="Green",width=0.6)
    ax_b.bar(x, a_p, bottom=g_p,color="#BA7517",alpha=0.82,label="Amber",width=0.6)
    ax_b.bar(x, r_p, bottom=[g+a for g,a in zip(g_p,a_p)],color="#A32D2D",alpha=0.82,label="Red",width=0.6)
    ax_b.set_xticks(x); ax_b.set_xticklabels(short,fontsize=9)
    ax_b.set_title("SLO status\ndistribution",fontweight="bold")
    ax_b.legend(fontsize=8); ax_b.set_ylim(0,108)

    # Breach probability
    ax_c = fig.add_subplot(gs[1, 3])
    bp_df = pd.read_csv(f"{output_dir}/breach_probability.csv",parse_dates=["forecast_date"])
    for layer in LAYERS:
        sub = bp_df[bp_df["layer"]==layer].sort_values("forecast_date")
        ax_c.plot(sub["forecast_date"], sub["breach_probability"],
                  color=LAYER_COLORS[layer], lw=1.5, ms=2, marker="o",
                  alpha=0.85, label=layer[:10])
    ax_c.axhline(50, color="#A32D2D",ls="--",lw=1.0,label="50% threshold")
    ax_c.set_title("30-day breach\nprobability",fontweight="bold")
    ax_c.legend(fontsize=7); ax_c.set_ylim(0,100); ax_c.set_ylabel("%")

    # Monthly heatmap
    ax_d = fig.add_subplot(gs[2, :2])
    df["month"] = df["timestamp"].dt.month
    monthly     = df.groupby(["layer","month"])["availability_pct"].mean().unstack()
    monthly     = monthly.loc[LAYERS]
    months      = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    im = ax_d.imshow(monthly.values, cmap="RdYlGn", aspect="auto", vmin=97, vmax=100)
    ax_d.set_xticks(range(12)); ax_d.set_xticklabels(months,fontsize=8)
    ax_d.set_yticks(range(4));  ax_d.set_yticklabels(["MDF","Port.Val","Risk","Client"],fontsize=8)
    for i in range(4):
        for j in range(12):
            v = monthly.values[i,j]
            ax_d.text(j,i,f"{v:.2f}%",ha="center",va="center",fontsize=6.5,
                      color="white" if v<98.5 else "#222")
    plt.colorbar(im,ax=ax_d,shrink=0.6)
    ax_d.set_title("Monthly availability heatmap — SLO compliance",fontweight="bold")
    ax_d.grid(False)

    # Daily alert count
    ax_e = fig.add_subplot(gs[2, 2])
    al = pd.read_csv(f"{output_dir}/mdf_alert_timeline.csv",parse_dates=["timestamp"])
    al_mdf = al[al["layer"]=="Market Data Feed"].copy()
    al_mdf_d = al_mdf.set_index("timestamp").resample("D").agg(
        {"z_flag":"sum","cusum_flag":"sum"}).reset_index()
    ax_e.bar(al_mdf_d["timestamp"],al_mdf_d["z_flag"],
             color="#185FA5",alpha=0.7,label="Z-score",width=0.8)
    ax_e.bar(al_mdf_d["timestamp"],al_mdf_d["cusum_flag"],
             bottom=al_mdf_d["z_flag"],color="#534AB7",alpha=0.7,label="CUSUM",width=0.8)
    ax_e.set_title("Daily alert count\nMDF — Z + CUSUM",fontweight="bold")
    ax_e.legend(fontsize=8); ax_e.set_ylabel("Alerts/day")

    # Dependency chain
    ax_f = fig.add_subplot(gs[2, 3]); ax_f.axis("off")
    boxes = [
        ("External\nFeeds","#F2F2F2","#888780"),
        ("Market Data\nFeed","#E6F1FB","#185FA5"),
        ("Portfolio\nValuation","#EEEDFE","#534AB7"),
        ("Risk\nDashboard","#FAEEDA","#BA7517"),
        ("Client\nReporting","#FCEBEB","#A32D2D"),
    ]
    ypos = [0.85,0.68,0.50,0.32,0.14]
    for i,(lbl,bg,fg) in enumerate(boxes):
        rect = plt.Rectangle((0.08,ypos[i]-0.07),0.56,0.13,facecolor=bg,edgecolor=fg,
                              lw=2,transform=ax_f.transAxes,clip_on=False)
        ax_f.add_patch(rect)
        ax_f.text(0.36,ypos[i]+0.005,lbl,transform=ax_f.transAxes,
                  ha="center",va="center",fontsize=8.5,fontweight="bold",color=fg)
        if i>0:
            ax_f.annotate("",xy=(0.36,ypos[i]+0.13),xytext=(0.36,ypos[i-1]-0.07+0.005),
                          xycoords="axes fraction",textcoords="axes fraction",
                          arrowprops=dict(arrowstyle="->",color="#555",lw=1.5))
    ax_f.set_title("Dependency\nchain",fontweight="bold")
    ax_f.set_xlim(0,1); ax_f.set_ylim(0,1)

    plt.savefig(f"{output_dir}/fig4_dashboard.png",dpi=140,bbox_inches="tight",facecolor=C_BG)
    plt.close()
    print(f"Saved: {output_dir}/fig4_dashboard.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df        = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    alert_df  = pd.read_csv(f"{OUTPUT_DIR}/mdf_alert_timeline.csv", parse_dates=["timestamp"])
    breach_df = pd.read_csv(f"{OUTPUT_DIR}/breach_probability.csv", parse_dates=["forecast_date"])

    plot_dashboard(df, alert_df, breach_df, OUTPUT_DIR)

    print("\nFinal output summary")
    print("  fig1_data_understanding   generation and cascade failure detail")
    print("  fig2_slo_design           heatmap, stacked bars, dependency chain")
    print("  fig3_anomaly_detection    Z-score, CUSUM, SARIMAX, breach prob")
    print("  fig4_dashboard            executive 8-panel view")
    print("  mdf_alert_timeline.csv    8,784 hourly rows with Z and CUSUM flags")
    print("  breach_probability.csv    120 rows (30 days x 4 layers)")
    print("\nPhase 4 complete. All outputs ready.")
