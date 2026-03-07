#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_optional_json(path: Path):
    if path is None or not path.exists():
        return None
    return load_json(path)


def extract_causal_series(causal_results, gender, group_label):
    runs = causal_results["runs"][gender]
    return {
        "group_label": group_label,
        "sweep_value": [run["sweep_value"] for run in runs],
        "accuracy": [run["accuracy"] for run in runs],
        "target_logit_diff": [run["target_logit_diff"]["mean"] for run in runs],
        "they_logit": [run["they_logit"]["mean"] for run in runs],
        "flip_to_he": [run["flip_to_he"] for run in runs],
        "flip_to_she": [run["flip_to_she"] for run in runs],
        "baseline_target_logit_diff": runs[0]["baseline_target_logit_diff"]["mean"] if runs else 0.0,
        "n_target": runs[0]["n_target"] if runs else 0,
        "runs": runs,
    }


def find_closest_run(runs, target_logit_diff):
    return min(runs, key=lambda run: abs(run["target_logit_diff"]["mean"] - target_logit_diff))


def build_causal_bundle(causal_all_results, causal_masculine_results=None, causal_feminine_results=None):
    all_bundle = {
        "group_name": "all_gender",
        "he": extract_causal_series(causal_all_results, "he", "all_gender"),
        "she": extract_causal_series(causal_all_results, "she", "all_gender"),
    }

    bundle = {
        "all_gender": all_bundle,
        "masculine": {
            "group_name": "masculine",
            "he": extract_causal_series(causal_masculine_results, "he", "masculine") if causal_masculine_results else all_bundle["he"],
            "she": extract_causal_series(causal_masculine_results, "she", "masculine") if causal_masculine_results else all_bundle["she"],
        },
        "feminine": {
            "group_name": "feminine",
            "he": extract_causal_series(causal_feminine_results, "he", "feminine") if causal_feminine_results else all_bundle["he"],
            "she": extract_causal_series(causal_feminine_results, "she", "feminine") if causal_feminine_results else all_bundle["she"],
        },
    }
    return bundle


def build_comparison_rows(smart_results, causal_bundle):
    rows = []
    smart_baseline = smart_results["baseline"]
    experiment_specs = [
        ("he", "exp1_masc_swap_all", "Smart ablation: he swap all directions", "all_gender"),
        ("she", "exp2_fem_swap_all", "Smart ablation: she swap all directions", "all_gender"),
        ("he", "exp3_masc_swap_masc_only", "Smart ablation: he swap masculine only", "masculine"),
        ("she", "exp4_fem_swap_fem_only", "Smart ablation: she swap feminine only", "feminine"),
    ]
    for gender, key, label, causal_group in experiment_specs:
        smart_exp = smart_results[key]
        series = causal_bundle[causal_group][gender]
        closest = find_closest_run(series["runs"], smart_exp["mean_logit_diff"])
        rows.append(
            {
                "label": label,
                "gender": gender,
                "causal_group": causal_group,
                "baseline_accuracy": smart_baseline[gender]["accuracy"],
                "baseline_logit_diff": smart_baseline[gender]["mean_logit_diff"],
                "smart_accuracy": smart_exp["accuracy"],
                "smart_logit_diff": smart_exp["mean_logit_diff"],
                "smart_they_logit": smart_exp["mean_they_logit"],
                "smart_flip_to_he": smart_exp.get("flipped_to_he", 0),
                "smart_flip_to_she": smart_exp.get("flipped_to_she", 0),
                "causal_group_used": series["group_label"],
                "causal_sweep_value": closest["sweep_value"],
                "causal_accuracy": closest["accuracy"],
                "causal_logit_diff": closest["target_logit_diff"]["mean"],
                "causal_they_logit": closest["they_logit"]["mean"],
                "causal_flip_to_he": closest["flip_to_he"],
                "causal_flip_to_she": closest["flip_to_she"],
            }
        )
    return rows


def plot_causal_sweeps(causal_series, comparison_rows, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col")
    genders = ["he", "she"]
    colors = {"he": "#1f77b4", "she": "#d62728"}
    smart_colors = {"he": "#17becf", "she": "#ff9896"}

    for col, gender in enumerate(genders):
        series = causal_series[gender]
        axes[0, col].plot(series["sweep_value"], series["accuracy"], marker="o", color=colors[gender], label=f"Causal steering ({gender})")
        axes[1, col].plot(series["sweep_value"], series["target_logit_diff"], marker="o", color=colors[gender], label=f"Causal steering ({gender})")
        for row in [row for row in comparison_rows if row["gender"] == gender]:
            axes[0, col].scatter(row["causal_sweep_value"], row["causal_accuracy"], color=smart_colors[gender], s=70)
            axes[1, col].scatter(row["causal_sweep_value"], row["causal_logit_diff"], color=smart_colors[gender], s=70)
        axes[0, col].axhline(y=comparison_rows[0]["baseline_accuracy"] if gender == "he" else comparison_rows[1]["baseline_accuracy"], linestyle="--", color="gray", alpha=0.8)
        axes[1, col].axhline(y=series["baseline_target_logit_diff"], linestyle="--", color="gray", alpha=0.8)
        axes[0, col].set_title(f"{gender.upper()} prompts")
        axes[0, col].set_ylabel("Accuracy (%)")
        axes[1, col].set_ylabel("Target logit diff")
        axes[1, col].set_xlabel("Forced scalar value")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)

    fig.suptitle("Causal steering sweep response curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_direction_group_sweeps(causal_bundle, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col")
    genders = ["he", "she"]
    group_styles = {
        "all_gender": ("#1f77b4", "-"),
        "masculine": ("#2ca02c", "--"),
        "feminine": ("#9467bd", ":"),
    }

    for col, gender in enumerate(genders):
        for group_name, group_data in causal_bundle.items():
            series = group_data[gender]
            color, linestyle = group_styles[group_name]
            axes[0, col].plot(
                series["sweep_value"],
                series["accuracy"],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=group_name,
                alpha=0.9,
            )
            axes[1, col].plot(
                series["sweep_value"],
                series["target_logit_diff"],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=group_name,
                alpha=0.9,
            )
        axes[0, col].set_title(f"{gender.upper()} prompts")
        axes[0, col].set_ylabel("Accuracy (%)")
        axes[1, col].set_ylabel("Target logit diff")
        axes[1, col].set_xlabel("Forced scalar value")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
        axes[0, col].legend()

    fig.suptitle("Causal steering by direction group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_smart_vs_causal_bars(comparison_rows, output_path: Path):
    labels = [row["label"].replace("Smart ablation: ", "") for row in comparison_rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    smart_acc = [row["smart_accuracy"] for row in comparison_rows]
    causal_acc = [row["causal_accuracy"] for row in comparison_rows]
    smart_diff = [row["smart_logit_diff"] for row in comparison_rows]
    causal_diff = [row["causal_logit_diff"] for row in comparison_rows]

    axes[0].bar(x - width / 2, smart_acc, width, label="Smart ablation", color="#4c78a8")
    axes[0].bar(x + width / 2, causal_acc, width, label="Closest causal sweep", color="#f58518")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Discrete range-swap vs closest causal sweep")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, smart_diff, width, label="Smart ablation", color="#4c78a8")
    axes[1].bar(x + width / 2, causal_diff, width, label="Closest causal sweep", color="#f58518")
    axes[1].set_ylabel("Target logit diff")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sigma_amplification(smart_results, output_path: Path):
    sigma_results = smart_results["sigma_amplification_experiments"]
    keys = sorted(sigma_results.keys(), key=lambda item: float(item.split("_")[-1]))
    sigma_values = [float(key.split("_")[-1]) for key in keys]
    accuracies = [sigma_results[key]["accuracy"] for key in keys]
    diffs = [sigma_results[key]["mean_logit_diff"] for key in keys]
    flips = [sigma_results[key]["flip_to_she_pct"] for key in keys]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(sigma_values, accuracies, marker="o", color="#4c78a8")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Smart ablation sigma amplification on he prompts")
    axes[0].grid(alpha=0.25)

    axes[1].plot(sigma_values, diffs, marker="o", color="#f58518")
    axes[1].set_ylabel("Target logit diff")
    axes[1].grid(alpha=0.25)

    axes[2].plot(sigma_values, flips, marker="o", color="#54a24b")
    axes[2].set_ylabel("Flip to she (%)")
    axes[2].set_xlabel("Sigma amplification")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_findings(smart_results, comparison_rows, causal_bundle, available_groups):
    primary_series = causal_bundle["all_gender"]
    he_runs = primary_series["he"]["runs"]
    she_runs = primary_series["she"]["runs"]
    he_lowest_acc = min(he_runs, key=lambda run: run["accuracy"])
    she_lowest_acc = min(she_runs, key=lambda run: run["accuracy"])
    sigma_2 = smart_results["sigma_amplification_experiments"]["masc_all_sigma_2.0"]
    exp1 = smart_results["exp1_masc_swap_all"]
    exp2 = smart_results["exp2_fem_swap_all"]
    exp3 = smart_results["exp3_masc_swap_masc_only"]
    exp4 = smart_results["exp4_fem_swap_fem_only"]
    has_masculine = "masculine" in available_groups
    has_feminine = "feminine" in available_groups
    he_masc_series = causal_bundle["masculine"]["he"]
    she_fem_series = causal_bundle["feminine"]["she"]
    he_masc_best = min(he_masc_series["runs"], key=lambda run: abs(run["target_logit_diff"]["mean"] - exp3["mean_logit_diff"]))
    she_fem_best = min(she_fem_series["runs"], key=lambda run: abs(run["target_logit_diff"]["mean"] - exp4["mean_logit_diff"]))

    lines = []
    lines.append("# Paper-Style Report: Causal Steering and Smart Ablation on GP")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("We evaluate two intervention families on the GP task in GPT-2 Small: continuous scalar causal steering over selected OV singular directions and discrete smart range-based ablations that swap directions into empirically measured opposite-gender activation ranges. Across both methods, the dominant empirical pattern is asymmetric control: interventions are substantially stronger on `he` prompts than on `she` prompts. Causal steering provides a smooth dose-response curve, while smart ablation yields stronger endpoint effects and can fully reverse predictions under sigma amplification. The combined evidence supports the claim that the selected OV directions exert real downstream causal control over gender-pronoun behavior, while also indicating that the currently selected directions capture masculine evidence more completely than feminine evidence.")
    lines.append("")
    lines.append("## Experimental Artifacts")
    lines.append("")
    lines.append(f"- **Run directory**: `{smart_results['run_dir']}`")
    lines.append("- **Smart ablation input**: `smart_ablation/smart_ablation_results.json`")
    lines.append("- **Primary causal steering input**: `causal_steering/causal_steering_sweep.json`")
    lines.append(f"- **Additional causal group inputs available**: `{', '.join(sorted(available_groups)) if available_groups else 'none'}`")
    lines.append("")
    lines.append("## Task Setting")
    lines.append("")
    lines.append("We compare interventions on two target subsets: `he` prompts and `she` prompts. The principal metrics are subset accuracy, target logit difference (`he - she` on `he` prompts and `she - he` on `she` prompts), `they` logit, and direct prediction flips. Smart ablation is evaluated as a discrete endpoint intervention, while causal steering is evaluated as a sweep over forced scalar values applied at the final residual stream before `ln_final`.")
    lines.append("")
    lines.append("## Baseline Performance")
    lines.append("")
    for gender in ["he", "she"]:
        baseline = smart_results["baseline"][gender]
        lines.append(f"- **{gender.upper()} prompts**: accuracy `{baseline['accuracy']:.2f}%`, target logit diff `{baseline['mean_logit_diff']:.3f}`, they logit `{baseline['mean_they_logit']:.3f}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### 1. Continuous causal steering exhibits a dose-response effect")
    lines.append("")
    lines.append(f"- **Strongest causal control is on `he` prompts**: the sweep reduces accuracy from `{smart_results['baseline']['he']['accuracy']:.2f}%` baseline to `{he_lowest_acc['accuracy']:.2f}%` at sweep value `{he_lowest_acc['sweep_value']:.3f}`, while shrinking target logit diff from `{smart_results['baseline']['he']['mean_logit_diff']:.3f}` to `{he_lowest_acc['target_logit_diff']['mean']:.3f}`.")
    lines.append(f"- **`she` control is weaker and asymmetric**: the lowest `she` sweep accuracy is `{she_lowest_acc['accuracy']:.2f}%` at sweep value `{she_lowest_acc['sweep_value']:.3f}`, with target logit diff still positive at `{she_lowest_acc['target_logit_diff']['mean']:.3f}`.")
    lines.append("")
    lines.append("![Causal sweep curves](figures/causal_sweep_curves.png)")
    lines.append("")
    lines.append("### 2. Smart ablation produces stronger endpoint effects")
    lines.append("")
    lines.append(f"- **Discrete smart ablation is much stronger on `he` than the smooth sweep**: `exp1_masc_swap_all` drives accuracy to `{exp1['accuracy']:.2f}%` and target logit diff to `{exp1['mean_logit_diff']:.3f}`, with `{exp1['flipped_to_she']}` flips to `she`.")
    lines.append(f"- **For `she`, discrete smart ablation is still moderate rather than catastrophic**: `exp2_fem_swap_all` reaches `{exp2['accuracy']:.2f}%` and target logit diff `{exp2['mean_logit_diff']:.3f}`, with `{exp2['flipped_to_he']}` flips to `he`.")
    lines.append("")
    lines.append("![Smart vs causal comparison](figures/smart_vs_causal_bars.png)")
    lines.append("")
    lines.append("### 3. Direction-selective interventions matter")
    lines.append("")
    lines.append(f"- **Selective direction swaps matter**: on `he`, swapping only masculine directions (`exp3`) leaves accuracy at `{exp3['accuracy']:.2f}%`, much higher than swapping all gender directions; on `she`, swapping only feminine directions (`exp4`) leaves accuracy at `{exp4['accuracy']:.2f}%`.")
    lines.append(f"- **Closest causal match for masculine-only comparison**: the best available `{he_masc_series['group_label']}` causal run reaches accuracy `{he_masc_best['accuracy']:.2f}%` at sweep `{he_masc_best['sweep_value']:.3f}` with target logit diff `{he_masc_best['target_logit_diff']['mean']:.3f}`.")
    lines.append(f"- **Closest causal match for feminine-only comparison**: the best available `{she_fem_series['group_label']}` causal run reaches accuracy `{she_fem_best['accuracy']:.2f}%` at sweep `{she_fem_best['sweep_value']:.3f}` with target logit diff `{she_fem_best['target_logit_diff']['mean']:.3f}`.")
    if not has_masculine or not has_feminine:
        lines.append(f"- **Caveat**: dedicated causal sweep files for `{('masculine' if not has_masculine else '')}{(' and ' if (not has_masculine and not has_feminine) else '')}{('feminine' if not has_feminine else '')}` were not found, so the script currently falls back to the all-gender sweep for missing groups.")
    lines.append("")
    lines.append("![Direction-group causal comparison](figures/direction_group_sweeps.png)")
    lines.append("")
    lines.append("### 4. Sigma amplification reveals a sharp nonlinear regime")
    lines.append("")
    lines.append(f"- **Sigma amplification saturates quickly**: at `sigma x 2.0`, `he` accuracy is already `{sigma_2['accuracy']:.2f}%` and flip-to-`she` reaches `{sigma_2['flip_to_she_pct']:.2f}%`, indicating the range-swap intervention can fully cross the decision boundary when amplified.")
    lines.append("")
    lines.append("![Sigma amplification](figures/sigma_amplification.png)")
    lines.append("")
    lines.append("## Quantitative Comparison")
    lines.append("")
    lines.append("| Experiment | Gender | Causal group used | Smart acc | Closest causal acc | Smart diff | Closest causal diff | Smart they | Closest causal they |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in comparison_rows:
        lines.append(
            f"| {row['label'].replace('Smart ablation: ', '')} | {row['gender']} | {row['causal_group_used']} | {row['smart_accuracy']:.2f} | {row['causal_accuracy']:.2f} | {row['smart_logit_diff']:.3f} | {row['causal_logit_diff']:.3f} | {row['smart_they_logit']:.3f} | {row['causal_they_logit']:.3f} |"
        )
    lines.append("")
    lines.append("## Discussion")
    lines.append("")
    lines.append("- **Causal steering** gives a clean continuous dose-response curve. It is best for showing monotonicity and controllability.")
    lines.append("- **Smart ablation** gives stronger endpoint effects because it swaps to opposite-gender empirical ranges and can be amplified with singular values. It is best for demonstrating maximal reversibility of the behavior.")
    lines.append("- **The asymmetry is consistent across both methods**: masculine-side control is stronger than feminine-side control.")
    lines.append("- **They logits drop sharply under both interventions** relative to baseline, which suggests the intervention is not merely shifting toward uncertainty; it is actively redistributing pronoun evidence.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- **Shared-scalar steering is coarse**: all selected directions are clamped to the same scalar target, which may underfit direction-specific calibration.")
    lines.append("- **Asymmetric capture remains unresolved**: the current direction set appears to capture masculine evidence more completely than feminine evidence.")
    lines.append("- **Direction-group comparison may still be partial** when dedicated masculine-only or feminine-only sweep files are unavailable.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("The two intervention families are complementary. Causal steering demonstrates smooth controllability and exposes the shape of the underlying response curve, while smart ablation demonstrates that opposite-gender range swaps can drive much larger endpoint changes, including full reversal under amplification. Taken together, the current evidence strongly supports a causal role for the selected OV singular directions in gender-pronoun behavior, with a persistent asymmetry that should guide the next round of mechanistic analysis.")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("- **Calibrate per-direction sweep values** instead of forcing one shared scalar across all directions.")
    lines.append("- **Run and save dedicated causal sweeps for `masculine` and `feminine` groups** so the selective comparisons stop relying on fallback behavior.")
    lines.append("- **Add a token-level decomposition** of logit changes to separate targeted `he <-> she` transfer from any residual uncertainty effects.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--smart_results", type=str, default=None)
    parser.add_argument("--causal_results", type=str, default=None)
    parser.add_argument("--causal_masculine_results", type=str, default=None)
    parser.add_argument("--causal_feminine_results", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    smart_path = Path(args.smart_results) if args.smart_results else run_dir / "smart_ablation" / "smart_ablation_results.json"
    causal_path = Path(args.causal_results) if args.causal_results else run_dir / "causal_steering" / "causal_steering_sweep.json"
    causal_masculine_path = Path(args.causal_masculine_results) if args.causal_masculine_results else run_dir / "causal_steering" / "causal_steering_sweep_masculine.json"
    causal_feminine_path = Path(args.causal_feminine_results) if args.causal_feminine_results else run_dir / "causal_steering" / "causal_steering_sweep_feminine.json"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "comparison_report"
    figures_dir = ensure_dir(output_dir / "figures")
    ensure_dir(output_dir)

    smart_results = load_json(smart_path)
    causal_results = load_json(causal_path)
    causal_masculine_results = load_optional_json(causal_masculine_path)
    causal_feminine_results = load_optional_json(causal_feminine_path)
    causal_bundle = build_causal_bundle(causal_results, causal_masculine_results, causal_feminine_results)
    comparison_rows = build_comparison_rows(smart_results, causal_bundle)
    available_groups = [
        group_name
        for group_name, data in (
            ("all_gender", causal_results),
            ("masculine", causal_masculine_results),
            ("feminine", causal_feminine_results),
        )
        if data is not None
    ]

    plot_causal_sweeps(causal_bundle["all_gender"], comparison_rows, figures_dir / "causal_sweep_curves.png")
    plot_direction_group_sweeps(causal_bundle, figures_dir / "direction_group_sweeps.png")
    plot_smart_vs_causal_bars(comparison_rows, figures_dir / "smart_vs_causal_bars.png")
    plot_sigma_amplification(smart_results, figures_dir / "sigma_amplification.png")

    report = summarize_findings(smart_results, comparison_rows, causal_bundle, available_groups)
    report_path = output_dir / "report.md"
    report_path.write_text(report)

    print(f"Report written to: {report_path}")
    print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
