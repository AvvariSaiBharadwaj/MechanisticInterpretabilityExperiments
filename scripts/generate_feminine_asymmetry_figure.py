#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a dedicated asymmetry figure for the GP steering presentation slide")
    parser.add_argument("--causal_json", type=str, required=True)
    parser.add_argument("--smart_json", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def get_best_causal_accuracy(results, target_gender: str):
    entries = results["runs"][target_gender]
    best = min(entries, key=lambda entry: entry["accuracy"])
    return {
        "accuracy": float(best["accuracy"]),
        "sweep_value": float(best["sweep_value"]),
    }


def main():
    args = parse_args()
    causal = load_json(args.causal_json)
    smart = load_json(args.smart_json)

    baseline_he = float(smart["baseline"]["he"]["accuracy"])
    baseline_she = float(smart["baseline"]["she"]["accuracy"])

    best_causal_he = get_best_causal_accuracy(causal, "he")
    best_causal_she = get_best_causal_accuracy(causal, "she")

    smart_he = float(smart["exp1_masc_swap_all"]["accuracy"])
    smart_she = float(smart["exp2_fem_swap_all"]["accuracy"])

    sigma_he = float(smart["sigma_amplification_experiments"]["masc_all_sigma_2.0"]["accuracy"])

    groups = ["he prompts", "she prompts"]
    baseline = np.array([baseline_he, baseline_she])
    strongest_causal = np.array([best_causal_he["accuracy"], best_causal_she["accuracy"]])
    strongest_smart = np.array([smart_he, smart_she])
    sigma_endpoint = np.array([sigma_he, np.nan])

    causal_drop = baseline - strongest_causal
    smart_drop = baseline - strongest_smart

    x = np.arange(len(groups))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    b0 = ax.bar(x - 1.5 * width, baseline, width, label="Baseline", color="#9aa5b1")
    b1 = ax.bar(x - 0.5 * width, strongest_causal, width, label="Strongest causal sweep", color="#4c78a8")
    b2 = ax.bar(x + 0.5 * width, strongest_smart, width, label="Smart ablation", color="#f58518")
    b3 = ax.bar(x + 1.5 * width, np.nan_to_num(sigma_endpoint, nan=0.0), width, label="Sigma x2 endpoint", color="#e45756")
    b3[1].set_alpha(0.25)
    b3[1].set_hatch("//")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy remaining after intervention")
    ax.legend(loc="upper right", fontsize=9)

    for bars in [b0, b1, b2]:
        for rect in bars:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 1.2, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    ax.text(b3[0].get_x() + b3[0].get_width() / 2, sigma_he + 1.2, f"{sigma_he:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(x[1] + 1.5 * width, 6, "n/a", ha="center", va="bottom", fontsize=8, color="#555555")

    ax2 = axes[1]
    d0 = ax2.bar(x - width / 2, causal_drop, width, label="Drop under strongest causal sweep", color="#4c78a8")
    d1 = ax2.bar(x + width / 2, smart_drop, width, label="Drop under smart ablation", color="#f58518")

    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_ylim(0, 80)
    ax2.set_ylabel("Accuracy drop from baseline (points)")
    ax2.set_title("Intervention susceptibility gap")
    ax2.legend(loc="upper right", fontsize=9)

    for bars in [d0, d1]:
        for rect in bars:
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2, height + 1.2, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Feminine asymmetry in GP causal steering", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        (
            f"Best causal sweep: he = {best_causal_he['accuracy']:.1f}% at sweep {best_causal_he['sweep_value']:.3f}, "
            f"she = {best_causal_she['accuracy']:.1f}% at sweep {best_causal_she['sweep_value']:.3f}. "
            "Sigma x2 endpoint is only available for the masculine smart-ablation run."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_dir = os.path.dirname(args.output_png)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output_png, dpi=180)
    plt.close(fig)

    print(f"Saved figure to: {args.output_png}")


if __name__ == "__main__":
    main()
