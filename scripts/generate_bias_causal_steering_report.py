#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots and report for occupation-bias causal steering results")
    parser.add_argument("--results_json", type=str, required=True, help="Path to bias causal steering JSON output")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save report and figures")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_results(path: str):
    with open(path, "r") as f:
        return json.load(f)


def extract_condition_values(prompt_records):
    values = []
    for record in prompt_records:
        for intervention in record["interventions"]:
            if intervention["mode"] == "sweep":
                values.append(intervention["value"])
    return sorted(set(values))


def aggregate_by_category_and_value(prompt_records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in prompt_records:
        category = record["category"]
        baseline_bias = record["baseline"]["bias_score"]
        grouped[category]["baseline"].append(baseline_bias)
        for intervention in record["interventions"]:
            grouped[category][intervention["value"]].append(intervention["steered"]["bias_score"])
            grouped[(category, "delta")][intervention["value"]].append(intervention["delta_bias"])
            grouped[(category, "they")][intervention["value"]].append(intervention["delta_they_logit"])
    return grouped


def plot_bias_curves(prompt_records, figures_dir):
    values = extract_condition_values(prompt_records)
    grouped = aggregate_by_category_and_value(prompt_records)
    categories = [category for category in {record["category"] for record in prompt_records}]

    plt.figure(figsize=(9, 6))
    for category in sorted(categories):
        y = [np.mean(grouped[category][value]) if grouped[category][value] else np.nan for value in values]
        baseline = np.mean(grouped[category]["baseline"]) if grouped[category]["baseline"] else 0.0
        plt.plot(values, y, marker="o", label=f"{category} steered")
        plt.axhline(baseline, linestyle="--", linewidth=1, alpha=0.5)
    plt.xlabel("Sweep value")
    plt.ylabel("Mean bias score: logit(he) - logit(she)")
    plt.title("Occupation bias under causal steering")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(figures_dir, "bias_score_curves.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_delta_bias(prompt_records, figures_dir):
    values = extract_condition_values(prompt_records)
    grouped = aggregate_by_category_and_value(prompt_records)
    categories = [category for category in {record["category"] for record in prompt_records}]

    plt.figure(figsize=(9, 6))
    for category in sorted(categories):
        y = [np.mean(grouped[(category, "delta")][value]) if grouped[(category, "delta")][value] else np.nan for value in values]
        plt.plot(values, y, marker="o", label=category)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Sweep value")
    plt.ylabel("Mean delta bias")
    plt.title("Bias shift induced by causal steering")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(figures_dir, "delta_bias_curves.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_flip_rates(results, figures_dir):
    summaries = results["condition_summaries"]
    labels = list(summaries.keys())
    rates = [summaries[label]["preferred_pronoun_flip_rate"] for label in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), rates)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Preferred pronoun flip rate (%)")
    plt.title("Pronoun preference flips by steering condition")
    plt.tight_layout()
    path = os.path.join(figures_dir, "flip_rates.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def summarize_findings(results):
    summaries = results["condition_summaries"]
    best_negative = None
    best_positive = None
    for label, summary in summaries.items():
        mean_delta = summary["delta_bias"]["mean"]
        if best_negative is None or mean_delta < best_negative[1]:
            best_negative = (label, mean_delta)
        if best_positive is None or mean_delta > best_positive[1]:
            best_positive = (label, mean_delta)
    most_feminizing = best_negative if best_negative and best_negative[1] < 0 else None
    most_masculinizing = best_positive if best_positive and best_positive[1] > 0 else None
    return {
        "most_feminizing": most_feminizing,
        "most_masculinizing": most_masculinizing,
        "least_feminizing": best_positive,
        "least_masculinizing": best_negative,
    }


def write_report(results, report_path, figure_paths):
    findings = summarize_findings(results)
    config = results["config"]
    results_json_path = config.get("results_json_path") or config.get("source_results_json")
    lines = []
    lines.append("# Occupation Bias Causal Steering Report")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("This report evaluates whether GP-derived causal steering directions transfer to occupation prompts and modulate stereotyped pronoun preference in a controlled way.")
    lines.append("")
    lines.append("## Experimental Artifacts")
    lines.append("")
    if results_json_path:
        lines.append(f"- **Results JSON**: `{os.path.relpath(results_json_path, os.path.dirname(report_path))}`")
    else:
        lines.append("- **Results JSON**: unavailable")
    lines.append(f"- **Prompt source**: `{config.get('prompts_file', 'inline prompts')}`")
    lines.append(f"- **Mode**: `{config['mode']}`")
    lines.append(f"- **Direction group**: `{config['direction_group']}`")
    lines.append(f"- **Number of prompts**: `{config['n_prompts']}`")
    lines.append("")
    lines.append("## High-level findings")
    lines.append("")
    if findings["most_feminizing"] is not None:
        lines.append(f"- **Most feminizing condition**: `{findings['most_feminizing'][0]}` with mean `delta_bias = {findings['most_feminizing'][1]:.3f}`")
    else:
        lines.append(f"- **Most feminizing condition**: none of the tested conditions produced negative mean `delta_bias`; the closest condition was `{findings['least_feminizing'][0]}` with mean `delta_bias = {findings['least_feminizing'][1]:.3f}`")
    if findings["most_masculinizing"] is not None:
        lines.append(f"- **Most masculinizing condition**: `{findings['most_masculinizing'][0]}` with mean `delta_bias = {findings['most_masculinizing'][1]:.3f}`")
    else:
        lines.append(f"- **Most masculinizing condition**: none of the tested conditions produced positive mean `delta_bias`; the closest condition was `{findings['least_masculinizing'][0]}` with mean `delta_bias = {findings['least_masculinizing'][1]:.3f}`")
    lines.append("")
    lines.append("## Bias score curves")
    lines.append("")
    lines.append(f"![Bias score curves]({os.path.relpath(figure_paths['bias_score_curves'], os.path.dirname(report_path))})")
    lines.append("")
    lines.append("## Delta bias curves")
    lines.append("")
    lines.append(f"![Delta bias curves]({os.path.relpath(figure_paths['delta_bias_curves'], os.path.dirname(report_path))})")
    lines.append("")
    lines.append("## Flip rates")
    lines.append("")
    lines.append(f"![Flip rates]({os.path.relpath(figure_paths['flip_rates'], os.path.dirname(report_path))})")
    lines.append("")
    lines.append("## Condition summary")
    lines.append("")
    lines.append("| Condition | Mean baseline bias | Mean steered bias | Mean delta bias | Flip rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label, summary in results["condition_summaries"].items():
        lines.append(
            f"| {label} | {summary['baseline_bias_score']['mean']:.3f} | {summary['steered_bias_score']['mean']:.3f} | {summary['delta_bias']['mean']:.3f} | {summary['preferred_pronoun_flip_rate']:.2f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **Negative delta bias** means steering moved the model toward `she`.")
    lines.append("- **Positive delta bias** means steering moved the model toward `he`.")
    lines.append("- **Flip rate** measures how often the preferred binary pronoun changed between baseline and steered evaluations.")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("- **Compare direction groups** by running the evaluator for `masculine`, `feminine`, and `all_gender` separately.")
    lines.append("- **Extend prompt coverage** with more occupations and more template variants.")
    lines.append("- **Add prompt-level case studies** for the strongest successful and failed mitigation examples.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    results = load_results(args.results_json)
    results.setdefault("config", {})["results_json_path"] = os.path.abspath(args.results_json)
    base_output_dir = args.output_dir or os.path.join(os.path.dirname(args.results_json), "report")
    figures_dir = os.path.join(base_output_dir, "figures")
    ensure_dir(base_output_dir)
    ensure_dir(figures_dir)

    prompt_records = results["prompts"]
    figure_paths = {
        "bias_score_curves": plot_bias_curves(prompt_records, figures_dir),
        "delta_bias_curves": plot_delta_bias(prompt_records, figures_dir),
        "flip_rates": plot_flip_rates(results, figures_dir),
    }
    report_path = os.path.join(base_output_dir, "report.md")
    write_report(results, report_path, figure_paths)

    print(f"Report written to: {report_path}")
    print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
