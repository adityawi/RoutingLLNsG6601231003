# ============================================================
# DTW-RL MULTI-CSV INTEGRATED ANALYSIS (FULL VERSION)
# - Parse 5 CSV logs
# - Aggregate mean/std across runs
# - Support TC / RL telemetry
# - Support UDP_SUMMARY / FINAL_STATS
# - Support transmission analysis per node
# - Generate full figures + summaries
# ============================================================

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
CSV_FILES = [
    "/content/drive/MyDrive/LOG CONTIKI-NG/DTW-RL/3 Juli/CSV/Mobility5/DTWRL/30/DTWRLmob5Interval301.csv",
    "/content/drive/MyDrive/LOG CONTIKI-NG/DTW-RL/3 Juli/CSV/Mobility5/DTWRL/30/DTWRLmob5Interval302.csv",
    "/content/drive/MyDrive/LOG CONTIKI-NG/DTW-RL/3 Juli/CSV/Mobility5/DTWRL/30/DTWRLmob5Interval303.csv",
    "/content/drive/MyDrive/LOG CONTIKI-NG/DTW-RL/3 Juli/CSV/Mobility5/DTWRL/30/DTWRLmob5Interval304.csv",
    "/content/drive/MyDrive/LOG CONTIKI-NG/DTW-RL/3 Juli/CSV/Mobility5/DTWRL/30/DTWRLmob5Interval305.csv",
]

OUTPUT_DIR = Path("/content/drive/MyDrive/LOG CONTIKI-NG/DTW-RL/3 Juli/CSV/dtw_rl_multirun_integrated_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_SIZE = 30
LAST5_MIN_SECONDS = 300

NORMALIZE_PDR_IN_PAIR = False
NORMALIZE_DELAY_IN_PAIR = True

DELAY_MIN_MS = 1
DELAY_MAX_MS = 10000


# ============================================================
# HELPERS
# ============================================================
def time_to_seconds(t: str) -> float:
    parts = str(t).split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return np.nan
    return np.nan


def extract_float(pattern: str, text: str) -> float:
    m = re.search(pattern, str(text))
    return float(m.group(1)) if m else np.nan


def extract_int(pattern: str, text: str) -> float:
    m = re.search(pattern, str(text))
    return float(m.group(1)) if m else np.nan


def normalize(series: pd.Series) -> pd.Series:
    s = pd.Series(series, dtype=float).ffill().bfill()
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - s_min) / (s_max - s_min)


def save_figure(fig: plt.Figure, outpath: Path, dpi: int = 200) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_output_path(name: str) -> Path:
    safe = name.lower().replace(" ", "_").replace("/", "_")
    return OUTPUT_DIR / f"{safe}.png"


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def safe_min(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.min()) if len(s) else np.nan


def safe_max(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.max()) if len(s) else np.nan


def safe_std(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.std()) if len(s) else np.nan


def extract_node_from_ip(msg: str) -> float:
    m = re.search(r'::.*:([0-9a-fA-F]+)', str(msg))
    if m:
        try:
            return int(m.group(1), 16)
        except Exception:
            return np.nan
    return np.nan


# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "figure.figsize": (10, 4.8),
    "axes.grid": True,
    "grid.alpha": 0.30,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
})

COLORS = {
    "tau":          "tab:blue",
    "HO":           "tab:orange",
    "HO_mean_node": "goldenrod",
    "dE":           "tab:olive",
    "dE_mJ":        "tab:green",
    "dEC":          "tab:brown",
    "invT":         "tab:red",
    "PDR":          "tab:purple",
    "Delay":        "tab:pink",
    "DelayMax":     "hotpink",
    "RSSIAvg":      "teal",
    "I":            "tab:gray",
}

LABELS = {
    "tau":          "tau",
    "HO":           "HO_total_net",
    "HO_mean_node": "HO_mean_node",
    "dE":           "dE",
    "dE_mJ":        "dE_mJ",
    "dEC":          "dEC_CPU_mJ",
    "invT":         "invTrust",
    "PDR":          "PDR_summary",
    "Delay":        "DelayAvg",
    "DelayMax":     "DelayMax",
    "RSSIAvg":      "RSSIAvg",
    "I":            "Instability (I)",
}


# ============================================================
# PARSE ONE RUN
# ============================================================
def parse_single_run(csv_path: str, run_name: str):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    if not {"time", "node", "message"}.issubset(df.columns):
        raise ValueError(f"{csv_path} harus memiliki kolom: time, node, message")

    df["message"] = df["message"].astype(str)
    df["t"] = df["time"].apply(time_to_seconds)

    msgs = df["message"]

    # --------------------------------------------------------
    # RL / TC telemetry
    # --------------------------------------------------------
    df["tau"]    = msgs.apply(lambda x: extract_float(r"tau=([0-9]+(?:\.[0-9]+)?)", x))
    df["tgt"]    = msgs.apply(lambda x: extract_float(r"tgt=([0-9]+(?:\.[0-9]+)?)", x))
    df["action"] = msgs.apply(lambda x: extract_int(r"a=([0-9]+)", x))
    df["invT"]   = msgs.apply(lambda x: extract_float(r"invT=([0-9]+(?:\.[0-9]+)?)", x))
    df["dE"]     = msgs.apply(lambda x: extract_float(r"dE=([0-9]+(?:\.[0-9]+)?)", x))
    df["dE_mJ"]  = msgs.apply(lambda x: extract_float(r"dE_mJ=([0-9]+(?:\.[0-9]+)?)", x))
    df["dEC"]    = msgs.apply(lambda x: extract_float(r"dEC=([0-9]+(?:\.[0-9]+)?)", x))
    df["I"]      = msgs.apply(lambda x: extract_float(r"I=([0-9]+(?:\.[0-9]+)?)", x))
    df["evid"]   = msgs.apply(lambda x: extract_int(r"evid=([0-9]+)", x))
    df["eps"]    = msgs.apply(lambda x: extract_float(r"eps=([0-9]+(?:\.[0-9]+)?)", x))

    df["is_tc"] = msgs.str.contains(r"\bTC,", regex=True, na=False)

    # --------------------------------------------------------
    # UDP logs
    # --------------------------------------------------------
    df["is_udp_server"]  = msgs.str.contains("UDP-Server", na=False)
    df["is_udp_summary"] = msgs.str.contains("UDP_SUMMARY", na=False)
    df["is_final_stats"] = msgs.str.contains("FINAL_STATS", na=False)
    df["is_sent"]        = msgs.str.contains("Sent Seq", na=False)

    df["udp_seq"]       = msgs.apply(lambda x: extract_int(r"Seq=([0-9]+)", x))
    df["udp_delay_pkt"] = msgs.apply(lambda x: extract_float(r"Delay=([0-9]+)ms", x))
    df["udp_rssi_pkt"]  = msgs.apply(lambda x: extract_float(r"RSSI=(-?[0-9]+)dBm", x))

    df["udp_delay_pkt"] = df["udp_delay_pkt"].where(
        (df["udp_delay_pkt"] >= DELAY_MIN_MS) & (df["udp_delay_pkt"] <= DELAY_MAX_MS),
        np.nan
    )

    df["PDR_summary"] = np.where(
        df["is_udp_summary"],
        msgs.apply(lambda x: extract_float(r"PDR=([0-9]+(?:\.[0-9]+)?)%", x)),
        np.nan
    )

    df["DelayAvg"] = np.where(
        df["is_udp_summary"],
        msgs.apply(lambda x: extract_float(r"DelayAvg=([0-9]+(?:\.[0-9]+)?)ms", x)),
        np.nan
    )

    df["DelayMax"] = np.where(
        df["is_udp_summary"],
        msgs.apply(lambda x: extract_float(r"DelayMax=([0-9]+(?:\.[0-9]+)?)ms", x)),
        np.nan
    )

    df["RSSIAvg"] = np.where(
        df["is_udp_summary"],
        msgs.apply(lambda x: extract_float(r"RSSIAvg=(-?[0-9]+(?:\.[0-9]+)?)dBm", x)),
        np.nan
    )

    df["src_node"] = msgs.apply(extract_node_from_ip)

    # --------------------------------------------------------
    # HO
    # --------------------------------------------------------
    df["HO_event"] = msgs.str.contains(r"HO,\d+,delay_ms=\d+", regex=True, na=False).astype(int)
    df["HO_any"] = msgs.str.contains("HO", case=False, na=False).astype(int)

    df = df.dropna(subset=["t"]).copy()

    # --------------------------------------------------------
    # BINNING
    # --------------------------------------------------------
    bins = np.arange(0, df["t"].max() + BIN_SIZE, BIN_SIZE)
    df["bin"] = np.digitize(df["t"], bins) - 1

    ho_sum_bin = df.groupby("bin")["HO_event"].sum()

    ho_rows = df[df["HO_event"] == 1].copy()
    ho_node_bin = ho_rows.groupby(["bin", "node"]).size().reset_index(name="ho_count")
    ho_mean_bin = ho_node_bin.groupby("bin")["ho_count"].mean()

    agg = pd.DataFrame({
        "tau":          df.groupby("bin")["tau"].mean(),
        "invT":         df.groupby("bin")["invT"].mean(),
        "dE":           df.groupby("bin")["dE"].mean(),
        "dE_mJ":        df.groupby("bin")["dE_mJ"].mean(),
        "dEC":          df.groupby("bin")["dEC"].mean(),
        "I":            df.groupby("bin")["I"].mean(),
        "PDR":          df.groupby("bin")["PDR_summary"].mean(),
        "Delay":        df.groupby("bin")["DelayAvg"].mean(),
        "DelayMax":     df.groupby("bin")["DelayMax"].mean(),
        "RSSIAvg":      df.groupby("bin")["RSSIAvg"].mean(),
        "HO":           ho_sum_bin,
        "HO_mean_node": ho_mean_bin,
    }).dropna(how="all")

    agg["time"] = agg.index * BIN_SIZE
    agg["run"] = run_name

    # --------------------------------------------------------
    # FINAL STATS
    # --------------------------------------------------------
    final_stats_rows = df[df["is_final_stats"]].copy()
    if len(final_stats_rows) > 0:
        final_msg = final_stats_rows.iloc[-1]["message"]
        final_stats = {
            "run": run_name,
            "Received": extract_int(r"Received=([0-9]+)", final_msg),
            "Lost": extract_int(r"Lost=([0-9]+)", final_msg),
            "Total": extract_int(r"Total=([0-9]+)", final_msg),
            "PDR_final_pct": extract_float(r"PDR=([0-9]+(?:\.[0-9]+)?)%", final_msg),
            "DelayAvg_final_ms": extract_float(r"DelayAvg=([0-9]+(?:\.[0-9]+)?)ms", final_msg),
            "DelayMax_final_ms": extract_float(r"DelayMax=([0-9]+(?:\.[0-9]+)?)ms", final_msg),
            "RSSIAvg_final_dBm": extract_float(r"RSSIAvg=(-?[0-9]+(?:\.[0-9]+)?)dBm", final_msg),
        }
    else:
        final_stats = {
            "run": run_name,
            "Received": np.nan,
            "Lost": np.nan,
            "Total": np.nan,
            "PDR_final_pct": np.nan,
            "DelayAvg_final_ms": np.nan,
            "DelayMax_final_ms": np.nan,
            "RSSIAvg_final_dBm": np.nan,
        }

    # --------------------------------------------------------
    # TC SUMMARY PER NODE
    # --------------------------------------------------------
    tc = df[df["is_tc"]].copy()
    tc_valid = tc.dropna(subset=["tau", "action", "evid"], how="any").copy()

    tc_results = []
    for node_id in sorted(tc_valid["node"].dropna().unique()):
        sub = tc_valid[tc_valid["node"] == node_id]
        tc_results.append({
            "run": run_name,
            "node": int(node_id),
            "tau_mean": safe_mean(sub["tau"]),
            "tau_last": sub["tau"].iloc[-1] if len(sub) else np.nan,
            "tau_min": safe_min(sub["tau"]),
            "tau_max": safe_max(sub["tau"]),
            "evid_zero": int((sub["evid"] == 0).sum()),
            "total_window": int(len(sub)),
            "a_dec": int((sub["action"] == 0).sum()),
            "a_inc": int((sub["action"] == 1).sum()),
            "instab_mean": safe_mean(sub["I"]),
            "invT_mean": safe_mean(sub["invT"]),
            "dE_mean": safe_mean(sub["dE"]),
            "dE_mJ_mean": safe_mean(sub["dE_mJ"]),
            "dEC_mean": safe_mean(sub["dEC"]),
            "eps_last": sub["eps"].iloc[-1] if len(sub) else np.nan,
        })
    tc_summary = pd.DataFrame(tc_results)

    # --------------------------------------------------------
    # TRANSMISSION ANALYSIS
    # --------------------------------------------------------
    sent = df[df["is_sent"]].copy()
    udp_recv = df[df["udp_seq"].notna()].copy()

    sent_counts = sent.groupby("node").size().rename("sent_count").reset_index()

    recv_counts = (
        udp_recv.dropna(subset=["src_node"])
        .groupby("src_node")
        .size()
        .rename("recv_count")
        .reset_index()
    )

    if len(recv_counts) > 0:
        recv_counts["src_node"] = recv_counts["src_node"].astype(int)

    merged_tx = pd.merge(
        sent_counts,
        recv_counts,
        left_on="node",
        right_on="src_node",
        how="left"
    )

    merged_tx["recv_count"] = merged_tx["recv_count"].fillna(0)
    merged_tx["delivery"] = (merged_tx["recv_count"] / merged_tx["sent_count"] * 100).round(2)
    merged_tx = merged_tx[["node", "sent_count", "recv_count", "delivery"]].sort_values("node")
    merged_tx["run"] = run_name

    # --------------------------------------------------------
    # HO TOTAL PER NODE
    # --------------------------------------------------------
    ho_per_node_total = (
        df[df["HO_event"] == 1]
        .groupby("node")
        .size()
        .reset_index(name="ho_total_per_node")
    )

    all_nodes = pd.DataFrame({"node": sorted(df["node"].dropna().unique())})
    ho_per_node_total = pd.merge(all_nodes, ho_per_node_total, on="node", how="left")
    ho_per_node_total["ho_total_per_node"] = ho_per_node_total["ho_total_per_node"].fillna(0)
    ho_per_node_total["run"] = run_name

    avg_ho_final_per_node = ho_per_node_total["ho_total_per_node"].mean()
    total_ho_network = ho_per_node_total["ho_total_per_node"].sum()

    # --------------------------------------------------------
    # EVENT SUMMARY
    # --------------------------------------------------------
    summary_event = {
        "run": run_name,
        "simulation_start_s": float(df["t"].min()),
        "simulation_end_s": float(df["t"].max()),
        "simulation_duration_s": float(df["t"].max() - df["t"].min()),
        "ho_any_total_lines": int(df["HO_any"].sum()),
        "ho_event_total_lines": int(df["HO_event"].sum()),
        "tau_mean_event": safe_mean(df["tau"]),
        "invT_mean_event": safe_mean(df["invT"]),
        "I_mean_event": safe_mean(df["I"]),
        "dE_mean_event": safe_mean(df["dE"]),
        "dE_mJ_mean_event": safe_mean(df["dE_mJ"]),
        "dEC_mean_event": safe_mean(df["dEC"]),
        "PDR_summary_mean_event": safe_mean(df["PDR_summary"]),
        "DelayAvg_mean_event": safe_mean(df["DelayAvg"]),
        "DelayMax_mean_event": safe_mean(df["DelayMax"]),
        "RSSIAvg_mean_event": safe_mean(df["RSSIAvg"]),
        "HO_total_network": float(total_ho_network),
        "HO_avg_final_per_node": float(avg_ho_final_per_node),
    }

    return {
        "df": df,
        "agg": agg,
        "final_stats": final_stats,
        "tc_summary": tc_summary,
        "merged_tx": merged_tx,
        "ho_per_node_total": ho_per_node_total,
        "summary_event": summary_event,
    }


# ============================================================
# PROCESS ALL RUNS
# ============================================================
all_runs = []
all_aggs = []
all_final_stats = []
all_tc = []
all_tx = []
all_ho_per_node = []
all_event_summary = []

for i, csv_path in enumerate(CSV_FILES, start=1):
    run_name = f"run{i}"
    result = parse_single_run(csv_path, run_name)

    all_runs.append(result)
    all_aggs.append(result["agg"])
    all_final_stats.append(result["final_stats"])
    all_tc.append(result["tc_summary"])
    all_tx.append(result["merged_tx"])
    all_ho_per_node.append(result["ho_per_node_total"])
    all_event_summary.append(result["summary_event"])

agg_all = pd.concat(all_aggs, ignore_index=True)
final_stats_all = pd.DataFrame(all_final_stats)
tc_all = pd.concat(all_tc, ignore_index=True) if len(all_tc) > 0 else pd.DataFrame()
tx_all = pd.concat(all_tx, ignore_index=True) if len(all_tx) > 0 else pd.DataFrame()
ho_per_node_all = pd.concat(all_ho_per_node, ignore_index=True) if len(all_ho_per_node) > 0 else pd.DataFrame()
summary_event_all = pd.DataFrame(all_event_summary)

agg_all.to_csv(OUTPUT_DIR / "agg_binned_all_runs.csv", index=False)
final_stats_all.to_csv(OUTPUT_DIR / "final_stats_all_runs.csv", index=False)
summary_event_all.to_csv(OUTPUT_DIR / "summary_event_all_runs.csv", index=False)

if len(tc_all) > 0:
    tc_all.to_csv(OUTPUT_DIR / "tc_summary_per_node_all_runs.csv", index=False)

if len(tx_all) > 0:
    tx_all.to_csv(OUTPUT_DIR / "transmission_delivery_per_node_all_runs.csv", index=False)

if len(ho_per_node_all) > 0:
    ho_per_node_all.to_csv(OUTPUT_DIR / "ho_total_per_node_all_runs.csv", index=False)


# ============================================================
# MULTI-RUN AGGREGATION (MEAN / STD BY TIME)
# ============================================================
metric_cols = [
    "tau", "invT", "dE", "dE_mJ", "dEC", "I",
    "PDR", "Delay", "DelayMax", "RSSIAvg",
    "HO", "HO_mean_node"
]

agg_mean = agg_all.groupby("time")[metric_cols].mean().reset_index()
agg_std = agg_all.groupby("time")[metric_cols].std().reset_index()

agg_mean.to_csv(OUTPUT_DIR / "agg_mean_multirun.csv", index=False)
agg_std.to_csv(OUTPUT_DIR / "agg_std_multirun.csv", index=False)


# ============================================================
# AGGREGATION: FINAL STATS / EVENT SUMMARY / PER NODE
# ============================================================
final_stats_mean = pd.DataFrame([{
    "Received_mean": safe_mean(final_stats_all["Received"]),
    "Received_std": safe_std(final_stats_all["Received"]),
    "Lost_mean": safe_mean(final_stats_all["Lost"]),
    "Lost_std": safe_std(final_stats_all["Lost"]),
    "Total_mean": safe_mean(final_stats_all["Total"]),
    "Total_std": safe_std(final_stats_all["Total"]),
    "PDR_final_pct_mean": safe_mean(final_stats_all["PDR_final_pct"]),
    "PDR_final_pct_std": safe_std(final_stats_all["PDR_final_pct"]),
    "DelayAvg_final_ms_mean": safe_mean(final_stats_all["DelayAvg_final_ms"]),
    "DelayAvg_final_ms_std": safe_std(final_stats_all["DelayAvg_final_ms"]),
    "DelayMax_final_ms_mean": safe_mean(final_stats_all["DelayMax_final_ms"]),
    "DelayMax_final_ms_std": safe_std(final_stats_all["DelayMax_final_ms"]),
    "RSSIAvg_final_dBm_mean": safe_mean(final_stats_all["RSSIAvg_final_dBm"]),
    "RSSIAvg_final_dBm_std": safe_std(final_stats_all["RSSIAvg_final_dBm"]),
}])
final_stats_mean.to_csv(OUTPUT_DIR / "final_stats_multirun_mean_std.csv", index=False)

summary_event_mean = pd.DataFrame([{
    "simulation_duration_s_mean": safe_mean(summary_event_all["simulation_duration_s"]),
    "simulation_duration_s_std": safe_std(summary_event_all["simulation_duration_s"]),
    "HO_total_network_mean": safe_mean(summary_event_all["HO_total_network"]),
    "HO_total_network_std": safe_std(summary_event_all["HO_total_network"]),
    "HO_avg_final_per_node_mean": safe_mean(summary_event_all["HO_avg_final_per_node"]),
    "HO_avg_final_per_node_std": safe_std(summary_event_all["HO_avg_final_per_node"]),
    "tau_mean_event_mean": safe_mean(summary_event_all["tau_mean_event"]),
    "tau_mean_event_std": safe_std(summary_event_all["tau_mean_event"]),
    "dE_mJ_mean_event_mean": safe_mean(summary_event_all["dE_mJ_mean_event"]),
    "dE_mJ_mean_event_std": safe_std(summary_event_all["dE_mJ_mean_event"]),
    "dEC_mean_event_mean": safe_mean(summary_event_all["dEC_mean_event"]),
    "dEC_mean_event_std": safe_std(summary_event_all["dEC_mean_event"]),
    "PDR_summary_mean_event_mean": safe_mean(summary_event_all["PDR_summary_mean_event"]),
    "PDR_summary_mean_event_std": safe_std(summary_event_all["PDR_summary_mean_event"]),
    "DelayAvg_mean_event_mean": safe_mean(summary_event_all["DelayAvg_mean_event"]),
    "DelayAvg_mean_event_std": safe_std(summary_event_all["DelayAvg_mean_event"]),
    "DelayMax_mean_event_mean": safe_mean(summary_event_all["DelayMax_mean_event"]),
    "DelayMax_mean_event_std": safe_std(summary_event_all["DelayMax_mean_event"]),
    "RSSIAvg_mean_event_mean": safe_mean(summary_event_all["RSSIAvg_mean_event"]),
    "RSSIAvg_mean_event_std": safe_std(summary_event_all["RSSIAvg_mean_event"]),
}])
summary_event_mean.to_csv(OUTPUT_DIR / "summary_event_multirun_mean_std.csv", index=False)

if len(tc_all) > 0:
    tc_mean_per_node = (
        tc_all.groupby("node")[[
            "tau_mean", "tau_last", "tau_min", "tau_max",
            "evid_zero", "total_window", "a_dec", "a_inc",
            "instab_mean", "invT_mean", "dE_mean",
            "dE_mJ_mean", "dEC_mean", "eps_last"
        ]]
        .mean()
        .reset_index()
    )
    tc_std_per_node = (
        tc_all.groupby("node")[[
            "tau_mean", "tau_last", "tau_min", "tau_max",
            "evid_zero", "total_window", "a_dec", "a_inc",
            "instab_mean", "invT_mean", "dE_mean",
            "dE_mJ_mean", "dEC_mean", "eps_last"
        ]]
        .std()
        .reset_index()
    )
    tc_mean_per_node.to_csv(OUTPUT_DIR / "tc_summary_per_node_mean.csv", index=False)
    tc_std_per_node.to_csv(OUTPUT_DIR / "tc_summary_per_node_std.csv", index=False)
else:
    tc_mean_per_node = pd.DataFrame()

if len(tx_all) > 0:
    tx_mean_per_node = (
        tx_all.groupby("node")[["sent_count", "recv_count", "delivery"]]
        .mean()
        .reset_index()
    )
    tx_std_per_node = (
        tx_all.groupby("node")[["sent_count", "recv_count", "delivery"]]
        .std()
        .reset_index()
    )
    tx_mean_per_node.to_csv(OUTPUT_DIR / "transmission_delivery_per_node_mean.csv", index=False)
    tx_std_per_node.to_csv(OUTPUT_DIR / "transmission_delivery_per_node_std.csv", index=False)
else:
    tx_mean_per_node = pd.DataFrame()

if len(ho_per_node_all) > 0:
    ho_mean_per_node = (
        ho_per_node_all.groupby("node")[["ho_total_per_node"]]
        .mean()
        .reset_index()
    )
    ho_std_per_node = (
        ho_per_node_all.groupby("node")[["ho_total_per_node"]]
        .std()
        .reset_index()
    )
    ho_mean_per_node.to_csv(OUTPUT_DIR / "ho_total_per_node_mean.csv", index=False)
    ho_std_per_node.to_csv(OUTPUT_DIR / "ho_total_per_node_std.csv", index=False)
else:
    ho_mean_per_node = pd.DataFrame()


# ============================================================
# LAST 5 MIN SUMMARY (MULTI-RUN MEAN)
# ============================================================
last5 = agg_mean[agg_mean["time"] >= (agg_mean["time"].max() - LAST5_MIN_SECONDS)].copy()
summary_last5 = pd.DataFrame([{
    "tau_mean_last5min": safe_mean(last5["tau"]),
    "HO_total_mean_last5min": safe_mean(last5["HO"]),
    "HO_mean_node_last5min": safe_mean(last5["HO_mean_node"]),
    "dE_mJ_mean_last5min": safe_mean(last5["dE_mJ"]),
    "dEC_mean_last5min": safe_mean(last5["dEC"]),
    "PDR_mean_last5min": safe_mean(last5["PDR"]),
    "DelayAvg_mean_last5min": safe_mean(last5["Delay"]),
    "DelayMax_mean_last5min": safe_mean(last5["DelayMax"]),
    "RSSIAvg_mean_last5min": safe_mean(last5["RSSIAvg"]),
}])
summary_last5.to_csv(OUTPUT_DIR / "summary_last5min_multirun_mean.csv", index=False)


# ============================================================
# PLOT FUNCTIONS
# ============================================================
def pair_plot(y1, y2, title, outname, normalize_y1=True, normalize_y2=True, ylabel="Normalized value"):
    s1 = normalize(agg_mean[y1]) if normalize_y1 else agg_mean[y1].astype(float).ffill().bfill()
    s2 = normalize(agg_mean[y2]) if normalize_y2 else agg_mean[y2].astype(float).ffill().bfill()

    fig, ax = plt.subplots()
    ax.plot(agg_mean["time"], s1, label=LABELS[y1], color=COLORS[y1])
    ax.plot(agg_mean["time"], s2, label=LABELS[y2], color=COLORS[y2])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_figure(fig, build_output_path(outname))


def single_plot_raw(y, title, outname, ylabel=None, last_n_seconds=None):
    plot_df = agg_mean.copy()

    if last_n_seconds is not None:
        t_max = plot_df["time"].max()
        plot_df = plot_df[plot_df["time"] >= (t_max - last_n_seconds)].copy()

    series = plot_df[y].astype(float).ffill().bfill()
    final_ylabel = ylabel or LABELS[y]

    fig, ax = plt.subplots()
    ax.plot(plot_df["time"], series, color=COLORS[y], label=LABELS[y])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(final_ylabel)
    ax.set_title(title)
    ax.legend()
    save_figure(fig, build_output_path(outname))


def combined_plot(cols, title, outname):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for col in cols:
        ax.plot(agg_mean["time"], normalize(agg_mean[col]), label=LABELS[col], color=COLORS[col])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized value")
    ax.set_title(title)
    ax.legend(ncol=min(len(cols), 5))
    save_figure(fig, build_output_path(outname))


def mean_std_plot(y, title, outname, ylabel=None, last_n_seconds=None):
    plot_mean = agg_mean.copy()
    plot_std = agg_std.copy()

    if last_n_seconds is not None:
        t_max = plot_mean["time"].max()
        mask = plot_mean["time"] >= (t_max - last_n_seconds)
        plot_mean = plot_mean[mask].copy()
        plot_std = plot_std[mask].copy()

    m = plot_mean[y].astype(float).ffill().bfill()
    s = plot_std[y].astype(float).fillna(0)

    final_ylabel = ylabel or LABELS[y]

    fig, ax = plt.subplots()
    ax.plot(plot_mean["time"], m, color=COLORS[y], label=f"{LABELS[y]} mean")
    ax.fill_between(plot_mean["time"], m - s, m + s, color=COLORS[y], alpha=0.2, label="± std")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(final_ylabel)
    ax.set_title(title)
    ax.legend()
    save_figure(fig, build_output_path(outname))


# ============================================================
# FIGURES: RL / NETWORK (MULTI-RUN MEAN)
# ============================================================
pair_plot("invT", "HO", "1. invTrust vs Handover (Mean, Normalized)", "01_invtrust_vs_handover_multirun_normalized")
pair_plot("dE_mJ", "HO", "1b. Energy Consumption (dE_mJ) vs Handover (Mean, Normalized)", "01b_demj_vs_handover_multirun_normalized")
pair_plot("tau", "I", "2. Adaptive tau vs Instability I (Mean, Normalized)", "02_tau_vs_instability_multirun_normalized")
pair_plot("tau", "HO", "3. Adaptive tau vs Handover (Mean, Normalized)", "03_tau_vs_handover_multirun_normalized")
pair_plot("tau", "dE_mJ", "4. Adaptive tau vs Energy Consumption dE_mJ (Mean, Normalized)", "04_tau_vs_demj_multirun_normalized")
pair_plot("tau", "dEC", "4b. Adaptive tau vs CPU Energy Consumption dEC (Mean, Normalized)", "04b_tau_vs_dec_multirun_normalized")

pair_plot(
    "tau", "PDR",
    "5. Adaptive tau vs Packet Delivery Ratio (Mean)",
    "05_tau_vs_pdr_multirun",
    normalize_y1=True,
    normalize_y2=NORMALIZE_PDR_IN_PAIR,
    ylabel="Normalized value" if NORMALIZE_PDR_IN_PAIR else "Mixed scale (tau normalized, PDR raw %)"
)

pair_plot(
    "tau", "Delay",
    "6. Adaptive tau vs End-to-End Delay Avg (Mean)",
    "06_tau_vs_delayavg_multirun",
    normalize_y1=True,
    normalize_y2=NORMALIZE_DELAY_IN_PAIR,
    ylabel="Normalized value" if NORMALIZE_DELAY_IN_PAIR else "Mixed scale (tau normalized, DelayAvg raw ms)"
)

pair_plot(
    "tau", "DelayMax",
    "6b. Adaptive tau vs End-to-End Delay Max (Mean)",
    "06b_tau_vs_delaymax_multirun",
    normalize_y1=True,
    normalize_y2=True,
    ylabel="Normalized value"
)

pair_plot(
    "tau", "RSSIAvg",
    "6c. Adaptive tau vs RSSI Average (Mean)",
    "06c_tau_vs_rssi_multirun",
    normalize_y1=True,
    normalize_y2=True,
    ylabel="Normalized value"
)

pair_plot(
    "RSSIAvg", "HO",
    "6d. RSSI Average vs Handover (Mean, Normalized)",
    "06d_rssi_vs_handover_multirun",
    normalize_y1=True,
    normalize_y2=True,
    ylabel="Normalized value"
)

combined_plot(
    ["tau", "HO", "dE_mJ", "invT", "PDR"],
    "7. Combined Normalized Metrics (Mean): tau, HO, dE_mJ, invTrust, PDR",
    "07_combined_main_metrics_multirun"
)

combined_plot(
    ["tau", "HO", "Delay", "RSSIAvg", "dEC"],
    "7b. Combined Normalized Metrics (Mean): tau, HO, DelayAvg, RSSIAvg, dEC",
    "07b_combined_support_metrics_multirun"
)


# ============================================================
# SINGLE-METRIC FIGURES (RAW MEAN)
# ============================================================
single_plot_raw("tau", "A. tau Over Time (Mean of 5 Runs)", "A_tau_over_time_multirun_raw", ylabel="tau")
single_plot_raw("HO_mean_node", "B. Mean HO Per Active Node (Mean of 5 Runs, Last 5 Minutes)", "B_ho_mean_per_node_multirun_raw_last_5min", ylabel="Mean HO events per active node (30s bin)", last_n_seconds=LAST5_MIN_SECONDS)
single_plot_raw("dE_mJ", "C. dE_mJ Over Time (Mean of 5 Runs)", "C_demj_over_time_multirun_raw", ylabel="Energy Consumption (mJ)")
single_plot_raw("dEC", "D. dEC Over Time (Mean of 5 Runs)", "D_dec_over_time_multirun_raw", ylabel="CPU Energy Consumption (mJ)")
single_plot_raw("PDR", "E. PDR Over Time (Mean of 5 Runs)", "E_pdr_over_time_multirun_raw", ylabel="PDR (%)")
single_plot_raw("Delay", "F. DelayAvg Over Time (Mean of 5 Runs)", "F_delayavg_over_time_multirun_raw", ylabel="DelayAvg (ms)")
single_plot_raw("DelayMax", "F2. DelayMax Over Time (Mean of 5 Runs)", "F2_delaymax_over_time_multirun_raw", ylabel="DelayMax (ms)")
single_plot_raw("RSSIAvg", "F3. RSSI Average Over Time (Mean of 5 Runs)", "F3_rssiavg_over_time_multirun_raw", ylabel="RSSIAvg (dBm)")
single_plot_raw("invT", "G. invTrust Over Time (Mean of 5 Runs)", "G_invtrust_over_time_multirun_raw", ylabel="invTrust")
single_plot_raw("I", "H. Instability I Over Time (Mean of 5 Runs)", "H_instability_over_time_multirun_raw", ylabel="Instability (I)")
single_plot_raw("dE", "I. dE Over Time (Mean of 5 Runs, RL State)", "I_de_over_time_multirun_raw", ylabel="dE (normalized RL state)")


# ============================================================
# MEAN ± STD FIGURES
# ============================================================
mean_std_plot("tau", "MS1. tau Over Time (Mean ± Std)", "MS1_tau_mean_std_multirun", ylabel="tau")
mean_std_plot("HO", "MS2. Network-wide HO Over Time (Mean ± Std)", "MS2_ho_mean_std_multirun", ylabel="HO events per 30s bin")
mean_std_plot("dE_mJ", "MS3. dE_mJ Over Time (Mean ± Std)", "MS3_demj_mean_std_multirun", ylabel="Energy Consumption (mJ)")
mean_std_plot("dEC", "MS4. dEC Over Time (Mean ± Std)", "MS4_dec_mean_std_multirun", ylabel="CPU Energy Consumption (mJ)")
mean_std_plot("PDR", "MS5. PDR Over Time (Mean ± Std)", "MS5_pdr_mean_std_multirun", ylabel="PDR (%)")
mean_std_plot("Delay", "MS6. DelayAvg Over Time (Mean ± Std)", "MS6_delayavg_mean_std_multirun", ylabel="DelayAvg (ms)")
mean_std_plot("DelayMax", "MS7. DelayMax Over Time (Mean ± Std)", "MS7_delaymax_mean_std_multirun", ylabel="DelayMax (ms)")
mean_std_plot("RSSIAvg", "MS8. RSSIAvg Over Time (Mean ± Std)", "MS8_rssi_mean_std_multirun", ylabel="RSSIAvg (dBm)")


# ============================================================
# FIGURES: TRANSMISSION
# ============================================================
if len(tc_mean_per_node) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in tc_mean_per_node.iterrows():
        ax.scatter(row["node"], row["tau_mean"], label=f"Node {int(row['node'])}")
    ax.set_title("Mean Tau per Node Across 5 Runs")
    ax.set_xlabel("Node")
    ax.set_ylabel("Mean Tau")
    ax.grid(True)
    save_figure(fig, build_output_path("tx_tau_mean_per_node_multirun"))

if len(tx_mean_per_node) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(tx_mean_per_node["node"], tx_mean_per_node["delivery"])
    ax.set_xlabel("Node")
    ax.set_ylabel("Delivery (%)")
    ax.set_title("Mean Delivery Ratio per Node Across 5 Runs")
    ax.grid(True)
    save_figure(fig, build_output_path("tx_delivery_ratio_per_node_multirun"))

if len(ho_mean_per_node) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ho_mean_per_node["node"], ho_mean_per_node["ho_total_per_node"])
    ax.set_xlabel("Node")
    ax.set_ylabel("Mean HO Total")
    ax.set_title("Mean HO Total per Node Across 5 Runs")
    ax.grid(True)
    save_figure(fig, build_output_path("tx_ho_total_per_node_multirun"))


# ============================================================
# README
# ============================================================
summary_txt = OUTPUT_DIR / "README_multirun_integrated.txt"
summary_txt.write_text(
    "\n".join([
        "DTW-RL Multi-Run Integrated Analysis Generated",
        f"Number of runs: {len(CSV_FILES)}",
        "",
        "Main aggregate outputs:",
        "agg_binned_all_runs.csv",
        "agg_mean_multirun.csv",
        "agg_std_multirun.csv",
        "final_stats_all_runs.csv",
        "final_stats_multirun_mean_std.csv",
        "summary_event_all_runs.csv",
        "summary_event_multirun_mean_std.csv",
        "summary_last5min_multirun_mean.csv",
        "tc_summary_per_node_all_runs.csv",
        "tc_summary_per_node_mean.csv",
        "tc_summary_per_node_std.csv",
        "transmission_delivery_per_node_all_runs.csv",
        "transmission_delivery_per_node_mean.csv",
        "transmission_delivery_per_node_std.csv",
        "ho_total_per_node_all_runs.csv",
        "ho_total_per_node_mean.csv",
        "ho_total_per_node_std.csv",
        "",
        "Figures include:",
        "- tau vs HO",
        "- tau vs PDR",
        "- tau vs DelayAvg",
        "- tau vs DelayMax",
        "- tau vs RSSIAvg",
        "- RSSIAvg vs HO",
        "- tau vs dE_mJ",
        "- tau vs dEC",
        "- combined plots",
        "- raw mean plots",
        "- mean ± std plots",
        "- transmission plots",
    ]),
    encoding="utf-8"
)

print(f"Done. All multi-run outputs saved to: {OUTPUT_DIR}")
print(f"README: {summary_txt}")