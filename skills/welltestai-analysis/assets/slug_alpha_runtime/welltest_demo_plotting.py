from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#6A3D9A"]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_response_fit(result: dict, output_dir: Path) -> Path:
    fit = result["response_fit"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    ax = axes[0]
    x = fit["time"].to_numpy(float)
    order = np.argsort(x)
    x = x[order]
    obs = fit["observed"].to_numpy(float)[order]
    med = fit["median"].to_numpy(float)[order]
    lo = fit["lower_90"].to_numpy(float)[order]
    hi = fit["upper_90"].to_numpy(float)[order]
    ax.fill_between(x, lo, hi, color=PALETTE[0], alpha=0.18, label="90% envelope")
    ax.plot(x, med, color=PALETTE[0], lw=1.8, label="posterior median")
    ax.scatter(x, obs, color="#333333", s=24, label="observed", zorder=3)
    ax.set_xscale("log")
    ax.set_ylabel("transformed response")
    ax.legend(frameon=False)
    axes[1].scatter(x, obs - med, color=PALETTE[1], s=22)
    axes[1].axhline(0, color="#333333", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("residual")
    return _save(fig, Path(output_dir) / f"{result['case_id']}_response_fit")


def plot_parameter_intervals(result: dict, output_dir: Path) -> Path:
    params = result["parameter_summary"].copy()
    y = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hlines(y, params["q05"], params["q95"], color=PALETTE[0], lw=2.2)
    ax.scatter(params["median"], y, color=PALETTE[1], s=35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(params["parameter"])
    ax.set_xlabel("posterior interval")
    ax.set_title("Parameter intervals")
    return _save(fig, Path(output_dir) / f"{result['case_id']}_parameter_intervals")


def plot_model_probabilities(result: dict, output_dir: Path) -> Path:
    probs = result["model_probabilities"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y = np.arange(len(probs))
    ax.barh(y, probs["probability"], color=PALETTE[: len(probs)])
    ax.set_yticks(y)
    ax.set_yticklabels(probs["mode"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title("Model probabilities")
    return _save(fig, Path(output_dir) / f"{result['case_id']}_model_probabilities")


def plot_qc_warnings(result: dict, output_dir: Path) -> Path:
    warnings = result["qc_warnings"]
    fig, ax = plt.subplots(figsize=(7.2, max(2.0, 0.45 * max(len(warnings), 1) + 1.2)))
    ax.axis("off")
    if warnings.empty:
        text = "No high-priority QC warnings in this demo run."
    else:
        text = "\n".join(f"- {row['warning']}: {row['recommended_next_action']}" for _, row in warnings.iterrows())
    ax.text(0.01, 0.98, text, va="top", ha="left", fontsize=9)
    return _save(fig, Path(output_dir) / f"{result['case_id']}_qc_warnings")
