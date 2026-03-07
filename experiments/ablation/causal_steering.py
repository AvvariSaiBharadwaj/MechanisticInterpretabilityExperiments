#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiments.ablation.intervention import (
    DIRECTION_RANGES,
    load_model_and_circuit,
    load_run_summary,
    resolve_run_paths,
)
from src.data import data_loader as local_data_loader
from src.utils.utils import get_data_column_names, get_indirect_objects_and_subjects


def parse_args():
    parser = argparse.ArgumentParser(description="Scalar causal steering for GP in GPT-2 Small")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to a training run directory under svd_logs")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained circuit checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to run summary JSON containing config")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save steering results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument(
        "--mode",
        type=str,
        default="sweep",
        choices=["mean_swap", "additive", "sigma_scaled", "sweep"],
    )
    parser.add_argument(
        "--target_genders",
        type=str,
        default="he,she",
        help="Comma-separated genders to evaluate: he,she",
    )
    parser.add_argument(
        "--direction_group",
        type=str,
        default="all_gender",
        choices=["all_gender", "masculine", "feminine", "plural", "all"],
    )
    parser.add_argument(
        "--direction_names",
        type=str,
        default=None,
        help="Optional comma-separated explicit direction names overriding direction_group",
    )
    parser.add_argument(
        "--additive_deltas",
        type=str,
        default="-1.0,-0.5,0.0,0.5,1.0",
        help="Comma-separated additive deltas for additive mode",
    )
    parser.add_argument(
        "--sigma_scales",
        type=str,
        default="-2.0,-1.0,0.0,1.0,2.0",
        help="Comma-separated std multipliers for sigma_scaled mode",
    )
    parser.add_argument(
        "--sweep_points",
        type=int,
        default=9,
        help="Number of scalar target points for sweep mode",
    )
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


def get_token_ids(model):
    return {
        "he": model.tokenizer.encode(" he", add_special_tokens=False)[0],
        "she": model.tokenizer.encode(" she", add_special_tokens=False)[0],
        "they": model.tokenizer.encode(" they", add_special_tokens=False)[0],
    }


def prepare_gp_batch(model, batch, device):
    clean_column_name, _ = get_data_column_names("gp")
    label_column_name, wrong_label_column_name = get_indirect_objects_and_subjects("gp")

    input_ids_clean = model.tokenizer(
        batch[clean_column_name],
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(device)

    correct_labels_raw = batch[label_column_name]
    if isinstance(correct_labels_raw, torch.Tensor):
        correct_labels_raw = correct_labels_raw.tolist()
    correct_text = [" " + str(item) for item in correct_labels_raw]
    correct_labels = model.tokenizer(
        correct_text,
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(device)[:, 0]

    wrong_labels_raw = batch[wrong_label_column_name]
    if isinstance(wrong_labels_raw, torch.Tensor):
        wrong_labels_raw = wrong_labels_raw.tolist()
    wrong_text = [" " + str(item) for item in wrong_labels_raw]
    wrong_labels = model.tokenizer(
        wrong_text,
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(device)[:, 0]

    clean_lengths = (input_ids_clean != model.tokenizer.pad_token_id).sum(dim=1)
    clean_last_idx = clean_lengths - 1
    attention_mask_clean = torch.arange(input_ids_clean.size(1), device=device)[None, :] < clean_lengths[:, None]

    return {
        "input_ids_clean": input_ids_clean,
        "correct_labels": correct_labels,
        "wrong_labels": wrong_labels,
        "clean_last_idx": clean_last_idx,
        "attention_mask_clean": attention_mask_clean,
    }


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


def get_target_mask(correct_labels, token_ids, target_gender: str):
    if target_gender == "he":
        return correct_labels == token_ids["he"]
    if target_gender == "she":
        return correct_labels == token_ids["she"]
    raise ValueError(f"Unsupported target gender: {target_gender}")


def get_direction_target_values(mode: str, dir_info: Dict, current_activation: torch.Tensor, sweep_value: float, target_gender: str):
    if mode == "mean_swap":
        if target_gender == "he":
            return torch.full_like(current_activation, dir_info["she_mean"])
        return torch.full_like(current_activation, dir_info["he_mean"])

    if mode == "additive":
        return current_activation + sweep_value

    if mode == "sigma_scaled":
        std_key = "he_std" if target_gender == "he" else "she_std"
        return current_activation + (sweep_value * dir_info[std_key])

    if mode == "sweep":
        return torch.full_like(current_activation, sweep_value)

    raise ValueError(f"Unsupported mode: {mode}")


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


def summarize_metric(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(len(values)),
    }


def run_steering_setting(
    model,
    circuit,
    data_loader,
    device,
    directions: Dict[str, Dict],
    target_gender: str,
    mode: str,
    sweep_value: float,
    max_batches: int,
):
    token_ids = get_token_ids(model)
    metrics = {
        "correct": [],
        "he_logit": [],
        "she_logit": [],
        "they_logit": [],
        "target_logit_diff": [],
        "baseline_target_logit_diff": [],
        "flip_to_he": 0,
        "flip_to_she": 0,
        "n_target": 0,
        "per_direction_scalar_before": {name: [] for name in directions},
        "per_direction_scalar_after": {name: [] for name in directions},
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"{mode}:{target_gender}:{sweep_value}")):
            if max_batches and batch_idx >= max_batches:
                break

            prepared = prepare_gp_batch(model, batch, device)
            input_ids_clean = prepared["input_ids_clean"]
            correct_labels = prepared["correct_labels"]
            clean_last_idx = prepared["clean_last_idx"]
            attention_mask_clean = prepared["attention_mask_clean"]

            logits, cache = model.run_with_cache(input_ids_clean, attention_mask=attention_mask_clean)
            batch_size = input_ids_clean.size(0)
            batch_indices = torch.arange(batch_size, device=device)
            target_mask = get_target_mask(correct_labels, token_ids, target_gender)

            if target_mask.sum().item() == 0:
                continue

            baseline_last_logits = logits[batch_indices, clean_last_idx, :]
            baseline_predictions = baseline_last_logits.argmax(dim=-1)

            if target_gender == "he":
                baseline_diff = baseline_last_logits[:, token_ids["he"]] - baseline_last_logits[:, token_ids["she"]]
            else:
                baseline_diff = baseline_last_logits[:, token_ids["she"]] - baseline_last_logits[:, token_ids["he"]]

            total_intervention = torch.zeros(batch_size, model.cfg.d_model, device=device)

            for dir_name, dir_info in directions.items():
                layer = dir_info["layer"]
                head = dir_info["head"]
                sv_idx = dir_info["sv_idx"]
                u_i, v_i, sigma_i = get_direction_components(circuit, layer, head, sv_idx, device)
                _, current_activation = get_context_and_scalar(
                    cache,
                    batch_indices,
                    clean_last_idx,
                    layer,
                    head,
                    u_i,
                    device,
                )
                target_activation = get_direction_target_values(
                    mode,
                    dir_info,
                    current_activation,
                    sweep_value,
                    target_gender,
                )
                delta_activation = target_activation - current_activation
                intervention = delta_activation.unsqueeze(-1) * sigma_i * v_i.T
                total_intervention += intervention
                metrics["per_direction_scalar_before"][dir_name].extend(current_activation[target_mask].cpu().tolist())
                metrics["per_direction_scalar_after"][dir_name].extend(target_activation[target_mask].cpu().tolist())

            final_resid_unnorm = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
            intervened_resid_unnorm = final_resid_unnorm.clone()
            intervened_resid_unnorm[batch_indices, clean_last_idx, :] += total_intervention
            last_token_resid = intervened_resid_unnorm[batch_indices, clean_last_idx, :]
            intervened_resid_norm = model.ln_final(last_token_resid)
            steered_logits = torch.matmul(intervened_resid_norm, model.W_U)
            steered_predictions = steered_logits.argmax(dim=-1)

            he_logits = steered_logits[:, token_ids["he"]]
            she_logits = steered_logits[:, token_ids["she"]]
            they_logits = steered_logits[:, token_ids["they"]]

            if target_gender == "he":
                target_logit_diff = he_logits - she_logits
            else:
                target_logit_diff = she_logits - he_logits

            metrics["correct"].extend((steered_predictions[target_mask] == correct_labels[target_mask]).cpu().tolist())
            metrics["he_logit"].extend(he_logits[target_mask].cpu().tolist())
            metrics["she_logit"].extend(she_logits[target_mask].cpu().tolist())
            metrics["they_logit"].extend(they_logits[target_mask].cpu().tolist())
            metrics["target_logit_diff"].extend(target_logit_diff[target_mask].cpu().tolist())
            metrics["baseline_target_logit_diff"].extend(baseline_diff[target_mask].cpu().tolist())
            metrics["n_target"] += int(target_mask.sum().item())

            metrics["flip_to_he"] += int(((baseline_predictions[target_mask] == token_ids["she"]) & (steered_predictions[target_mask] == token_ids["he"])).sum().item())
            metrics["flip_to_she"] += int(((baseline_predictions[target_mask] == token_ids["he"]) & (steered_predictions[target_mask] == token_ids["she"])).sum().item())

    result = {
        "mode": mode,
        "sweep_value": float(sweep_value),
        "target_gender": target_gender,
        "n_target": int(metrics["n_target"]),
        "accuracy": float(np.mean(metrics["correct"]) * 100) if metrics["correct"] else 0.0,
        "he_logit": summarize_metric(metrics["he_logit"]),
        "she_logit": summarize_metric(metrics["she_logit"]),
        "they_logit": summarize_metric(metrics["they_logit"]),
        "target_logit_diff": summarize_metric(metrics["target_logit_diff"]),
        "baseline_target_logit_diff": summarize_metric(metrics["baseline_target_logit_diff"]),
        "delta_target_logit_diff_mean": float(np.mean(np.array(metrics["target_logit_diff"]) - np.array(metrics["baseline_target_logit_diff"]))) if metrics["target_logit_diff"] else 0.0,
        "flip_to_he": int(metrics["flip_to_he"]),
        "flip_to_she": int(metrics["flip_to_she"]),
        "per_direction_scalar_before": {name: summarize_metric(values) for name, values in metrics["per_direction_scalar_before"].items()},
        "per_direction_scalar_after": {name: summarize_metric(values) for name, values in metrics["per_direction_scalar_after"].items()},
    }
    return result


def main():
    args = parse_args()
    model_path, config_path, output_dir = resolve_run_paths(
        run_dir=args.run_dir,
        model_path=args.model_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        output_subdir="causal_steering",
    )
    os.makedirs(output_dir, exist_ok=True)

    summary, config = load_run_summary(config_path)
    model, circuit, device = load_model_and_circuit(model_path, config)
    directions = select_directions(args.direction_group, args.direction_names)
    target_genders = [item.strip() for item in args.target_genders.split(",") if item.strip()]
    additive_deltas = parse_float_list(args.additive_deltas)
    sigma_scales = parse_float_list(args.sigma_scales)

    test_loader = local_data_loader.load_gp_dataset(
        batch_size=args.batch_size,
        train=False,
        validation=False,
        shuffle=False,
    )

    results = {
        "config": {
            "run_dir": args.run_dir,
            "model_path": model_path,
            "config_path": config_path,
            "output_dir": output_dir,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "mode": args.mode,
            "target_genders": target_genders,
            "direction_group": args.direction_group,
            "direction_names": args.direction_names,
            "additive_deltas": additive_deltas,
            "sigma_scales": sigma_scales,
            "sweep_points": args.sweep_points,
            "selected_directions": directions,
            "model_name": config["model"]["name"],
            "data_type": config["data_type"],
            "experiment_name": config.get("experiment_name"),
        },
        "run_summary": summary,
        "runs": {},
    }

    for target_gender in target_genders:
        sweep_values = get_sweep_values(
            args.mode,
            directions,
            target_gender,
            additive_deltas,
            sigma_scales,
            args.sweep_points,
        )
        gender_results = []
        for sweep_value in sweep_values:
            gender_results.append(
                run_steering_setting(
                    model=model,
                    circuit=circuit,
                    data_loader=test_loader,
                    device=device,
                    directions=directions,
                    target_gender=target_gender,
                    mode=args.mode,
                    sweep_value=sweep_value,
                    max_batches=args.max_batches,
                )
            )
        results["runs"][target_gender] = gender_results

    output_path = os.path.join(output_dir, f"causal_steering_{args.mode}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved causal steering results to: {output_path}")


if __name__ == "__main__":
    main()
