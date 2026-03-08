#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.masked_transformer_circuit import MaskedTransformerCircuit


DIRECTION_RANGES = {
    "L9.H7.SV1": {
        "layer": 9,
        "head": 7,
        "sv_idx": 1,
        "type": "masculine",
        "he_mean": 0.115,
        "he_std": 0.180,
        "she_mean": -0.453,
        "she_std": 0.291,
    },
    "L11.H8.SV6": {
        "layer": 11,
        "head": 8,
        "sv_idx": 6,
        "type": "masculine",
        "he_mean": 0.203,
        "he_std": 0.140,
        "she_mean": -0.121,
        "she_std": 0.184,
    },
    "L10.H9.SV0": {
        "layer": 10,
        "head": 9,
        "sv_idx": 0,
        "type": "feminine",
        "he_mean": -0.273,
        "he_std": 0.259,
        "she_mean": 0.652,
        "she_std": 0.420,
    },
    "L11.H8.SV9": {
        "layer": 11,
        "head": 8,
        "sv_idx": 9,
        "type": "feminine",
        "he_mean": -0.159,
        "he_std": 0.171,
        "she_mean": 0.134,
        "she_std": 0.199,
    },
    "L9.H7.SV0": {
        "layer": 9,
        "head": 7,
        "sv_idx": 0,
        "type": "plural",
        "he_mean": -0.334,
        "he_std": 0.137,
        "she_mean": -0.174,
        "she_std": 0.162,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate occupation-bias prompts with GP causal steering")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to a training run directory under svd_logs")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained circuit checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to run summary JSON containing config")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to a .json, .jsonl, .csv, or .txt file of prompts")
    parser.add_argument("--prompt", action="append", default=None, help="Inline prompt; may be passed multiple times")
    parser.add_argument("--mode", type=str, default="sweep", choices=["mean_swap", "additive", "sigma_scaled", "sweep"])
    parser.add_argument("--direction_group", type=str, default="all_gender", choices=["all_gender", "masculine", "feminine", "plural", "all"])
    parser.add_argument("--direction_names", type=str, default=None)
    parser.add_argument("--steering_targets", type=str, default="she,he", help="Comma-separated target poles for mean_swap and sigma_scaled")
    parser.add_argument("--additive_deltas", type=str, default="-1.0,-0.5,0.0,0.5,1.0")
    parser.add_argument("--sigma_scales", type=str, default="-2.0,-1.0,0.0,1.0,2.0")
    parser.add_argument("--sweep_values", type=str, default=None, help="Optional explicit comma-separated sweep values")
    parser.add_argument("--sweep_points", type=int, default=9)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def select_directions(direction_group: str, direction_names: Optional[str]) -> Dict[str, Dict]:
    if direction_names:
        selected = {}
        for name in [item.strip() for item in direction_names.split(",") if item.strip()]:
            if name not in DIRECTION_RANGES:
                raise ValueError(f"Unknown direction name: {name}")
            selected[name] = DIRECTION_RANGES[name]
        return selected

    if direction_group == "masculine":
        return {k: v for k, v in DIRECTION_RANGES.items() if v["type"] == "masculine"}
    if direction_group == "feminine":
        return {k: v for k, v in DIRECTION_RANGES.items() if v["type"] == "feminine"}
    if direction_group == "plural":
        return {k: v for k, v in DIRECTION_RANGES.items() if v["type"] == "plural"}
    if direction_group == "all_gender":
        return {k: v for k, v in DIRECTION_RANGES.items() if v["type"] in {"masculine", "feminine"}}
    return dict(DIRECTION_RANGES)


def get_direction_components(circuit, layer: int, head: int, sv_idx: int, device: torch.device):
    head_key = f"differential_head_{layer}_{head}"
    ov_cache_key = f"{head_key}_ov"
    if ov_cache_key not in circuit.svd_cache:
        circuit._load_or_compute_svd()
    U_ov, S_ov, Vt_ov, _ = circuit.svd_cache[ov_cache_key]
    U_ov = U_ov.to(device)
    S_ov = S_ov.to(device)
    V_ov = Vt_ov.T.to(device)
    u_i = U_ov[:, sv_idx:sv_idx + 1]
    v_i = V_ov[:, sv_idx:sv_idx + 1]
    sigma_i = S_ov[sv_idx]
    return u_i, v_i, sigma_i


def get_context_and_scalar(cache, batch_indices, clean_last_idx, layer, head, u_i, device):
    attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
    attn_in = cache[f"blocks.{layer}.ln1.hook_normalized"]
    attn_weights_last = attn_pattern[batch_indices, :, clean_last_idx, :]
    attn_w = attn_weights_last[:, head, :]
    context_standard = torch.matmul(attn_w.unsqueeze(1), attn_in).squeeze(1)
    ones = torch.ones(context_standard.shape[0], 1, device=device)
    context = torch.cat([ones, context_standard], dim=1)
    current_activation = torch.matmul(context, u_i).squeeze(-1)
    return context, current_activation


def get_sweep_values(mode: str, directions: Dict[str, Dict], target_gender: str, additive_deltas: List[float], sigma_scales: List[float], sweep_points: int):
    if mode == "mean_swap":
        return [0.0]
    if mode == "additive":
        return additive_deltas
    if mode == "sigma_scaled":
        return sigma_scales
    if mode == "sweep":
        low_values = []
        high_values = []
        for dir_info in directions.values():
            low_values.append(min(dir_info["he_mean"] - 2 * dir_info["he_std"], dir_info["she_mean"] - 2 * dir_info["she_std"]))
            high_values.append(max(dir_info["he_mean"] + 2 * dir_info["he_std"], dir_info["she_mean"] + 2 * dir_info["she_std"]))
        global_low = min(low_values)
        global_high = max(high_values)
        return np.linspace(global_low, global_high, sweep_points).tolist()
    raise ValueError(f"Unsupported mode: {mode}")


def load_run_summary(config_path):
    with open(config_path, "r") as f:
        summary = json.load(f)
    return summary, summary["config"]


def resolve_run_paths(run_dir=None, model_path=None, config_path=None, output_dir=None, output_subdir=None):
    if run_dir is not None:
        model_path = model_path or os.path.join(run_dir, "model_final.pt")
        config_path = config_path or os.path.join(run_dir, "run_summary.json")
        if output_dir is None and output_subdir is not None:
            output_dir = os.path.join(run_dir, output_subdir)

    missing = []
    if model_path is None:
        missing.append("model_path")
    if config_path is None:
        missing.append("config_path")
    if missing:
        raise ValueError(f"Missing required path arguments: {', '.join(missing)}")
    return model_path, config_path, output_dir


def load_model_and_circuit(model_path, config):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = HookedTransformer.from_pretrained(
        config["model"]["name"],
        cache_dir=config["model"]["pretrained_cache_dir"],
    )
    model = model.to(device)

    circuit = MaskedTransformerCircuit(
        model=model,
        device=device,
        cache_svd=True,
        mask_init_value=config["masking"]["mask_init_value"],
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    circuit.qk_masks = checkpoint["qk_masks"]
    circuit.ov_masks = checkpoint["ov_masks"]
    circuit.mlp_in_masks = checkpoint["mlp_in_masks"]
    circuit.mlp_out_masks = checkpoint["mlp_out_masks"]
    return model, circuit, device


def load_prompts(prompts_file: Optional[str], inline_prompts: Optional[List[str]]):
    prompts = []
    if prompts_file:
        path = Path(prompts_file)
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    prompts.append(normalize_prompt_record(item, idx))
            else:
                raise ValueError("JSON prompts file must contain a list")
        elif suffix == ".jsonl":
            with path.open("r") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    prompts.append(normalize_prompt_record(json.loads(line), idx))
        elif suffix == ".csv":
            with path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    prompts.append(normalize_prompt_record(row, idx))
        elif suffix == ".txt":
            with path.open("r") as f:
                for idx, line in enumerate(f):
                    text = line.strip()
                    if not text:
                        continue
                    prompts.append(normalize_prompt_record({"prompt": text}, idx))
        else:
            raise ValueError(f"Unsupported prompts file extension: {suffix}")
    if inline_prompts:
        start_idx = len(prompts)
        for offset, prompt in enumerate(inline_prompts):
            prompts.append(normalize_prompt_record({"prompt": prompt}, start_idx + offset))
    if not prompts:
        raise ValueError("No prompts provided. Use --prompts_file or one or more --prompt arguments.")
    return prompts


def normalize_prompt_record(item, idx: int):
    if isinstance(item, str):
        return {
            "id": f"prompt_{idx}",
            "prompt": item,
            "category": "unspecified",
            "occupation": None,
            "metadata": {},
        }
    prompt_text = item.get("prompt") or item.get("prefix")
    if prompt_text is None:
        raise ValueError("Each prompt record must contain a 'prompt' field")
    metadata = {k: v for k, v in item.items() if k not in {"id", "prompt", "prefix", "category", "occupation"}}
    return {
        "id": str(item.get("id", f"prompt_{idx}")),
        "prompt": str(prompt_text),
        "category": str(item.get("category", "unspecified")),
        "occupation": item.get("occupation"),
        "metadata": metadata,
    }


def get_token_ids(model):
    return {
        "he": model.tokenizer.encode(" he", add_special_tokens=False)[0],
        "she": model.tokenizer.encode(" she", add_special_tokens=False)[0],
        "they": model.tokenizer.encode(" they", add_special_tokens=False)[0],
    }


def tokenize_prompts(model, prompts: List[str], device: torch.device):
    encoded = model.tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    clean_lengths = (input_ids != model.tokenizer.pad_token_id).sum(dim=1)
    clean_last_idx = clean_lengths - 1
    attention_mask = torch.arange(input_ids.size(1), device=device)[None, :] < clean_lengths[:, None]
    return input_ids, clean_last_idx, attention_mask


def extract_top_tokens(model, logits: torch.Tensor, top_k: int):
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    result = []
    for token_id, value in zip(indices.tolist(), values.tolist()):
        result.append({
            "token_id": int(token_id),
            "token": model.tokenizer.decode([token_id]),
            "logit": float(value),
        })
    return result


def summarize_metric(values: List[float]):
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(len(values)),
    }


def get_preferred_binary_pronoun(he_logit: float, she_logit: float):
    return "he" if he_logit >= she_logit else "she"


def build_condition_label(mode: str, steering_target: Optional[str], value: float):
    if mode == "mean_swap":
        return f"{mode}:{steering_target}"
    if mode == "sigma_scaled":
        return f"{mode}:{steering_target}:{value}"
    return f"{mode}:{value}"


def get_steering_conditions(args, directions):
    additive_deltas = parse_float_list(args.additive_deltas)
    sigma_scales = parse_float_list(args.sigma_scales)
    steering_targets = [item.strip() for item in args.steering_targets.split(",") if item.strip()]
    conditions = []

    if args.mode == "mean_swap":
        for steering_target in steering_targets:
            conditions.append({"mode": args.mode, "steering_target": steering_target, "value": 0.0})
        return conditions

    if args.mode == "additive":
        for value in additive_deltas:
            conditions.append({"mode": args.mode, "steering_target": None, "value": float(value)})
        return conditions

    if args.mode == "sigma_scaled":
        for steering_target in steering_targets:
            for value in sigma_scales:
                conditions.append({"mode": args.mode, "steering_target": steering_target, "value": float(value)})
        return conditions

    if args.mode == "sweep":
        if args.sweep_values:
            sweep_values = parse_float_list(args.sweep_values)
        else:
            sweep_values = get_sweep_values(args.mode, directions, "he", additive_deltas, sigma_scales, args.sweep_points)
        for value in sweep_values:
            conditions.append({"mode": args.mode, "steering_target": None, "value": float(value)})
        return conditions

    raise ValueError(f"Unsupported mode: {args.mode}")


def get_direction_target_activation(mode: str, dir_info: Dict, current_activation: torch.Tensor, steering_target: Optional[str], value: float):
    if mode == "mean_swap":
        if steering_target not in {"he", "she"}:
            raise ValueError("mean_swap requires steering_target to be 'he' or 'she'")
        return torch.full_like(current_activation, dir_info[f"{steering_target}_mean"])
    if mode == "additive":
        return current_activation + value
    if mode == "sigma_scaled":
        if steering_target not in {"he", "she"}:
            raise ValueError("sigma_scaled requires steering_target to be 'he' or 'she'")
        return current_activation + (value * dir_info[f"{steering_target}_std"])
    if mode == "sweep":
        return torch.full_like(current_activation, value)
    raise ValueError(f"Unsupported mode: {mode}")


def build_total_intervention(cache, batch_indices, clean_last_idx, circuit, device, directions, condition):
    total_intervention = None
    per_direction_before = {}
    per_direction_after = {}

    for dir_name, dir_info in directions.items():
        u_i, v_i, sigma_i = get_direction_components(circuit, dir_info["layer"], dir_info["head"], dir_info["sv_idx"], device)
        _, current_activation = get_context_and_scalar(
            cache,
            batch_indices,
            clean_last_idx,
            dir_info["layer"],
            dir_info["head"],
            u_i,
            device,
        )
        target_activation = get_direction_target_activation(
            condition["mode"],
            dir_info,
            current_activation,
            condition["steering_target"],
            condition["value"],
        )
        delta_activation = target_activation - current_activation
        intervention = delta_activation.unsqueeze(-1) * sigma_i * v_i.T
        total_intervention = intervention if total_intervention is None else total_intervention + intervention
        per_direction_before[dir_name] = current_activation.detach().cpu().tolist()
        per_direction_after[dir_name] = target_activation.detach().cpu().tolist()

    return total_intervention, per_direction_before, per_direction_after


def logits_to_metrics(logits: torch.Tensor, token_ids: Dict[str, int]):
    probs = torch.softmax(logits, dim=-1)
    he_logit = float(logits[token_ids["he"]].item())
    she_logit = float(logits[token_ids["she"]].item())
    they_logit = float(logits[token_ids["they"]].item())
    he_prob = float(probs[token_ids["he"]].item())
    she_prob = float(probs[token_ids["she"]].item())
    they_prob = float(probs[token_ids["they"]].item())
    bias_score = he_logit - she_logit
    preferred_binary_pronoun = get_preferred_binary_pronoun(he_logit, she_logit)
    return {
        "he_logit": he_logit,
        "she_logit": she_logit,
        "they_logit": they_logit,
        "he_prob": he_prob,
        "she_prob": she_prob,
        "they_prob": they_prob,
        "bias_score": bias_score,
        "preferred_binary_pronoun": preferred_binary_pronoun,
    }


def evaluate_prompts(model, circuit, device, prompts, directions, conditions, top_k):
    token_ids = get_token_ids(model)
    prompt_texts = [item["prompt"] for item in prompts]
    input_ids, clean_last_idx, attention_mask = tokenize_prompts(model, prompt_texts, device)

    with torch.no_grad():
        baseline_logits, cache = model.run_with_cache(input_ids, attention_mask=attention_mask)
        batch_indices = torch.arange(input_ids.size(0), device=device)
        baseline_last_logits = baseline_logits[batch_indices, clean_last_idx, :]

        prompt_results = []
        per_prompt_interventions = {item["id"]: [] for item in prompts}

        baseline_predictions = baseline_last_logits.argmax(dim=-1)

        for prompt_idx, prompt_record in enumerate(prompts):
            baseline_metrics = logits_to_metrics(baseline_last_logits[prompt_idx], token_ids)
            baseline_metrics["top_tokens"] = extract_top_tokens(model, baseline_last_logits[prompt_idx], top_k)
            baseline_metrics["top_prediction"] = model.tokenizer.decode([int(baseline_predictions[prompt_idx].item())])
            prompt_results.append(
                {
                    "id": prompt_record["id"],
                    "prompt": prompt_record["prompt"],
                    "category": prompt_record["category"],
                    "occupation": prompt_record["occupation"],
                    "metadata": prompt_record["metadata"],
                    "baseline": baseline_metrics,
                }
            )

        for condition in conditions:
            total_intervention, per_direction_before, per_direction_after = build_total_intervention(
                cache,
                batch_indices,
                clean_last_idx,
                circuit,
                device,
                directions,
                condition,
            )
            final_resid_unnorm = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
            intervened_resid_unnorm = final_resid_unnorm.clone()
            intervened_resid_unnorm[batch_indices, clean_last_idx, :] += total_intervention
            last_token_resid = intervened_resid_unnorm[batch_indices, clean_last_idx, :]
            intervened_resid_norm = model.ln_final(last_token_resid)
            steered_logits = torch.matmul(intervened_resid_norm, model.W_U)
            steered_predictions = steered_logits.argmax(dim=-1)

            for prompt_idx, prompt_record in enumerate(prompts):
                baseline_metrics = prompt_results[prompt_idx]["baseline"]
                steered_metrics = logits_to_metrics(steered_logits[prompt_idx], token_ids)
                steered_metrics["top_tokens"] = extract_top_tokens(model, steered_logits[prompt_idx], top_k)
                steered_metrics["top_prediction"] = model.tokenizer.decode([int(steered_predictions[prompt_idx].item())])
                preferred_pronoun_flipped = baseline_metrics["preferred_binary_pronoun"] != steered_metrics["preferred_binary_pronoun"]
                intervention_record = {
                    "condition_label": build_condition_label(condition["mode"], condition["steering_target"], condition["value"]),
                    "mode": condition["mode"],
                    "steering_target": condition["steering_target"],
                    "value": float(condition["value"]),
                    "steered": steered_metrics,
                    "delta_bias": float(steered_metrics["bias_score"] - baseline_metrics["bias_score"]),
                    "delta_he_logit": float(steered_metrics["he_logit"] - baseline_metrics["he_logit"]),
                    "delta_she_logit": float(steered_metrics["she_logit"] - baseline_metrics["she_logit"]),
                    "delta_they_logit": float(steered_metrics["they_logit"] - baseline_metrics["they_logit"]),
                    "preferred_pronoun_flipped": bool(preferred_pronoun_flipped),
                    "baseline_prediction_changed": bool(int(baseline_predictions[prompt_idx].item()) != int(steered_predictions[prompt_idx].item())),
                    "per_direction_scalar_before": {name: per_direction_before[name][prompt_idx] for name in directions},
                    "per_direction_scalar_after": {name: per_direction_after[name][prompt_idx] for name in directions},
                }
                per_prompt_interventions[prompt_record["id"]].append(intervention_record)

        for prompt_result in prompt_results:
            prompt_result["interventions"] = per_prompt_interventions[prompt_result["id"]]

    return prompt_results


def aggregate_condition(prompt_results, condition_label):
    condition_records = []
    for prompt_result in prompt_results:
        for intervention in prompt_result["interventions"]:
            if intervention["condition_label"] == condition_label:
                condition_records.append((prompt_result, intervention))
                break
    categories = sorted({prompt_result["category"] for prompt_result, _ in condition_records})
    category_summaries = {}
    for category in categories:
        category_records = [(prompt_result, intervention) for prompt_result, intervention in condition_records if prompt_result["category"] == category]
        category_summaries[category] = summarize_condition_records(category_records)
    summary = summarize_condition_records(condition_records)
    summary["by_category"] = category_summaries
    return summary


def summarize_condition_records(records):
    baseline_biases = [prompt_result["baseline"]["bias_score"] for prompt_result, _ in records]
    steered_biases = [intervention["steered"]["bias_score"] for _, intervention in records]
    delta_biases = [intervention["delta_bias"] for _, intervention in records]
    they_deltas = [intervention["delta_they_logit"] for _, intervention in records]
    preference_flips = [intervention["preferred_pronoun_flipped"] for _, intervention in records]
    prediction_changes = [intervention["baseline_prediction_changed"] for _, intervention in records]
    return {
        "n_prompts": len(records),
        "baseline_bias_score": summarize_metric(baseline_biases),
        "steered_bias_score": summarize_metric(steered_biases),
        "delta_bias": summarize_metric(delta_biases),
        "delta_they_logit": summarize_metric(they_deltas),
        "preferred_pronoun_flip_rate": float(np.mean(preference_flips) * 100) if preference_flips else 0.0,
        "prediction_change_rate": float(np.mean(prediction_changes) * 100) if prediction_changes else 0.0,
    }


def build_results(args, run_summary, config, directions, prompts, conditions, prompt_results):
    condition_summaries = {}
    for condition in conditions:
        label = build_condition_label(condition["mode"], condition["steering_target"], condition["value"])
        condition_summaries[label] = aggregate_condition(prompt_results, label)
    return {
        "config": {
            "run_dir": args.run_dir,
            "model_path": args.model_path,
            "config_path": args.config_path,
            "mode": args.mode,
            "direction_group": args.direction_group,
            "direction_names": args.direction_names,
            "steering_targets": [item.strip() for item in args.steering_targets.split(",") if item.strip()],
            "additive_deltas": parse_float_list(args.additive_deltas),
            "sigma_scales": parse_float_list(args.sigma_scales),
            "sweep_values": parse_float_list(args.sweep_values) if args.sweep_values else None,
            "sweep_points": args.sweep_points,
            "top_k": args.top_k,
            "selected_directions": directions,
            "n_prompts": len(prompts),
            "model_name": config["model"]["name"],
            "data_type": config["data_type"],
            "experiment_name": config.get("experiment_name"),
        },
        "run_summary": run_summary,
        "prompt_count": len(prompts),
        "prompts": prompt_results,
        "condition_summaries": condition_summaries,
    }


def write_flat_csv(output_path: str, prompt_results: List[Dict]):
    fieldnames = [
        "prompt_id",
        "prompt",
        "category",
        "occupation",
        "condition_label",
        "mode",
        "steering_target",
        "value",
        "baseline_he_logit",
        "baseline_she_logit",
        "baseline_they_logit",
        "baseline_he_prob",
        "baseline_she_prob",
        "baseline_they_prob",
        "baseline_bias_score",
        "baseline_preferred_binary_pronoun",
        "baseline_top_prediction",
        "steered_he_logit",
        "steered_she_logit",
        "steered_they_logit",
        "steered_he_prob",
        "steered_she_prob",
        "steered_they_prob",
        "steered_bias_score",
        "steered_preferred_binary_pronoun",
        "steered_top_prediction",
        "delta_bias",
        "delta_he_logit",
        "delta_she_logit",
        "delta_they_logit",
        "preferred_pronoun_flipped",
        "baseline_prediction_changed",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt_result in prompt_results:
            baseline = prompt_result["baseline"]
            for intervention in prompt_result["interventions"]:
                steered = intervention["steered"]
                writer.writerow(
                    {
                        "prompt_id": prompt_result["id"],
                        "prompt": prompt_result["prompt"],
                        "category": prompt_result["category"],
                        "occupation": prompt_result["occupation"],
                        "condition_label": intervention["condition_label"],
                        "mode": intervention["mode"],
                        "steering_target": intervention["steering_target"],
                        "value": intervention["value"],
                        "baseline_he_logit": baseline["he_logit"],
                        "baseline_she_logit": baseline["she_logit"],
                        "baseline_they_logit": baseline["they_logit"],
                        "baseline_he_prob": baseline["he_prob"],
                        "baseline_she_prob": baseline["she_prob"],
                        "baseline_they_prob": baseline["they_prob"],
                        "baseline_bias_score": baseline["bias_score"],
                        "baseline_preferred_binary_pronoun": baseline["preferred_binary_pronoun"],
                        "baseline_top_prediction": baseline["top_prediction"],
                        "steered_he_logit": steered["he_logit"],
                        "steered_she_logit": steered["she_logit"],
                        "steered_they_logit": steered["they_logit"],
                        "steered_he_prob": steered["he_prob"],
                        "steered_she_prob": steered["she_prob"],
                        "steered_they_prob": steered["they_prob"],
                        "steered_bias_score": steered["bias_score"],
                        "steered_preferred_binary_pronoun": steered["preferred_binary_pronoun"],
                        "steered_top_prediction": steered["top_prediction"],
                        "delta_bias": intervention["delta_bias"],
                        "delta_he_logit": intervention["delta_he_logit"],
                        "delta_she_logit": intervention["delta_she_logit"],
                        "delta_they_logit": intervention["delta_they_logit"],
                        "preferred_pronoun_flipped": intervention["preferred_pronoun_flipped"],
                        "baseline_prediction_changed": intervention["baseline_prediction_changed"],
                    }
                )


def main():
    args = parse_args()
    model_path, config_path, output_dir = resolve_run_paths(
        run_dir=args.run_dir,
        model_path=args.model_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        output_subdir="bias_causal_steering",
    )
    os.makedirs(output_dir, exist_ok=True)

    run_summary, config = load_run_summary(config_path)
    model, circuit, device = load_model_and_circuit(model_path, config)
    directions = select_directions(args.direction_group, args.direction_names)
    prompts = load_prompts(args.prompts_file, args.prompt)
    conditions = get_steering_conditions(args, directions)
    prompt_results = evaluate_prompts(model, circuit, device, prompts, directions, conditions, args.top_k)
    results = build_results(args, run_summary, config, directions, prompts, conditions, prompt_results)
    results["config"]["model_path"] = model_path
    results["config"]["config_path"] = config_path
    results["config"]["output_dir"] = output_dir
    if args.prompts_file is not None:
        results["config"]["prompts_file"] = args.prompts_file

    output_filename = f"bias_causal_steering_{args.mode}_{args.direction_group}.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_output_path = os.path.join(output_dir, f"bias_causal_steering_{args.mode}_{args.direction_group}.csv")
    write_flat_csv(csv_output_path, prompt_results)

    print(f"Saved bias causal steering evaluation to: {output_path}")
    print(f"Saved flat CSV summary to: {csv_output_path}")


if __name__ == "__main__":
    main()
