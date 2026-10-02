"""Independent multi-image evaluation, calibration, ablations and resource evidence.

Run from the source checkout: python evaluation/vision_eval.py --help.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from strands_decider.data.format import Example, assert_disjoint, group_keys, read_jsonl
from strands_decider.evaluate import (
    Prediction,
    collect_logits,
    fit_temperature,
    fit_temperature_by_kind,
    score_mae,
)
from strands_decider.hf_export import fingerprint
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.prompting import render_content, render_question
from strands_decider.vision import decode_images, move_model_inputs, prepare_vision_batch


def validate_manifests(**splits: list[Example]) -> None:
    assert_disjoint(**splits)
    if any(not rows for rows in splits.values()):
        raise ValueError("qualification manifests must each contain examples")


def metrics(predictions: list[Prediction]) -> dict[str, Any]:
    if not predictions:
        return {}
    n = len(predictions)
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for prediction in predictions:
        confidence = max(prediction.probs)
        bins[min(9, int(confidence * 10))].append((confidence, prediction.correct))
    result = {
        "n": n,
        "accuracy": sum(p.correct for p in predictions) / n,
        "brier": sum(
            sum((q - int(i == p.label)) ** 2 for i, q in enumerate(p.probs)) for p in predictions
        )
        / n,
        "ece_top_label": sum(
            len(rows)
            / n
            * abs(sum(c for c, _ in rows) / len(rows) - sum(ok for _, ok in rows) / len(rows))
            for rows in bins.values()
        ),
        "nll": -sum(math.log(max(p.probs[p.label], 1e-12)) for p in predictions) / n,
    }
    mae = score_mae(predictions)
    if mae is not None:
        result["score_mae_levels"] = mae
    return result


def condition_example(ex: Example, condition: str, donor: Example | None) -> Example | None:
    row = copy.copy(ex)
    row.images = [copy.copy(image) for image in ex.images]
    if condition == "image_only":
        if not row.images:
            return None
        row.state = ""
        for image in row.images:
            image.text = None
    elif condition == "ocr_only":
        row.state = "\n".join(
            [render_content(ex.state), *[image.text or "" for image in ex.images]]
        )
        row.images = []
    elif condition == "removed_page":
        if not row.images:
            return None
        row.images = row.images[:-1]  # Fixed input-order control, independent of gold/prediction.
    elif condition == "unrelated_page":
        if not row.images or donor is None:
            return None
        image = copy.copy(donor.images[0])
        used = {asset.id for asset in row.images}
        image.id = f"unrelated-{image.sha256}"
        while image.id in used:
            image.id += "-extra"
        row.images[-1] = image  # Fixed-position substitution preserves group size.
    return row


@torch.inference_mode()
def predict_example(engine: SystemOneEngine, ex: Example) -> tuple[Prediction, dict[str, Any]]:
    device = engine.device
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rq = render_question(ex.to_question())
    pages = decode_images(ex.to_images(), engine.cfg.image_limits)
    if not pages and not render_content(ex.state).strip():
        raise ValueError("condition removed all evidence")
    prepared = prepare_vision_batch(
        engine.model.processor, [ex.state], [pages], [rq], engine.model.config.vision_budget()
    )
    inputs = move_model_inputs(prepared.tensors, device, engine.model.torso)
    preprocessing_ms = (time.perf_counter() - started) * 1000
    out = engine.model(
        **inputs,
        opt_idx=prepared.opt_idx.to(device),
        answer_positions=prepared.answer_positions.to(device),
        n_slots=torch.tensor([rq.n_slots], device=device),
        temperature=engine._temperatures([ex.kind]),
    )
    if cuda:
        torch.cuda.synchronize(device)
    prediction = Prediction(
        ex.task,
        ex.kind,
        ex.n_options,
        ex.label,
        out["log_probs"][0, : ex.n_options].exp().tolist(),
        engine.model.config.ordinal_smoothing,
    )
    return prediction, {
        "preprocessing_ms": preprocessing_ms,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "input_tokens": prepared.sequence_tokens[0],
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
    }


def evaluate_conditions(
    engine: SystemOneEngine | None, examples: list[Example], *, predictor: Any = None
) -> dict[str, Any]:
    if predictor is None:
        if engine is None or engine.model.config.input_mode != "multimodal":
            raise ValueError("visual evaluation requires a multimodal checkpoint")

        def predictor(ex: Example) -> tuple[Prediction, dict[str, Any]]:
            return predict_example(engine, ex)

    report: dict[str, Any] = {
        "format": "decider-vision-eval/1",
        "conditions": {},
        "controls": {},
        "ece_definition": "10 bins over top-label probability; not API confidence",
        "controls_note": "Gold labels are retained for sensitivity controls; missing evidence can invalidate them.",
        "config": vars(engine.model.config).copy() if engine is not None else {},
    }
    for condition in ("combined", "image_only", "ocr_only", "removed_page", "unrelated_page"):
        predictions = []
        by_count: dict[str, list[Prediction]] = defaultdict(list)
        profiles = []
        failures = []
        skipped = 0
        for index, example in enumerate(examples):
            keys = group_keys(example)
            donor = next(
                (row for row in examples if row.images and not keys & group_keys(row)), None
            )
            row = condition_example(example, condition, donor)
            if row is None:
                skipped += 1
                continue
            try:
                prediction, profile = predictor(row)
            except (ValueError, OSError, RuntimeError) as exc:
                failures.append(
                    {"row": index, "document_id": example.document_id, "error": str(exc)}
                )
                continue
            predictions.append(prediction)
            by_count[str(len(row.images))].append(prediction)
            profiles.append(profile)
        resources = {}
        if profiles:
            resources = {
                "preprocessing_ms_mean": sum(p["preprocessing_ms"] for p in profiles)
                / len(profiles),
                "latency_ms_mean": sum(p["latency_ms"] for p in profiles) / len(profiles),
                "latency_ms_max": max(p["latency_ms"] for p in profiles),
                "input_tokens": sum(p["input_tokens"] for p in profiles),
                "peak_gpu_bytes": max(
                    (p["peak_gpu_bytes"] for p in profiles if p["peak_gpu_bytes"] is not None),
                    default=None,
                ),
            }
        block = {
            "attempted": len(examples) - skipped,
            "succeeded": len(predictions),
            "failed": len(failures),
            "skipped": skipped,
            "failures": failures,
            "metrics": metrics(predictions),
            "resource": resources,
            "by_kind": {
                kind: metrics([p for p in predictions if p.kind == kind])
                for kind in ("choice", "noul", "score")
            },
            "by_image_count": {count: metrics(rows) for count, rows in sorted(by_count.items())},
        }
        report["controls" if condition in ("removed_page", "unrelated_page") else "conditions"][
            condition
        ] = block
    return report


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--train-manifest", action="append", default=[])
    parser.add_argument("--validation-manifest", action="append", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help="Fit NLL/global and per-kind temperatures in memory; checkpoint is unchanged.",
    )
    args = parser.parse_args()
    evaluation = list(read_jsonl(args.evaluation_manifest))
    calibration = list(read_jsonl(args.calibration_manifest))
    splits = {"evaluation": evaluation, "calibration": calibration}
    for name, paths in (("train", args.train_manifest), ("validation", args.validation_manifest)):
        if paths:
            splits[name] = [ex for path in paths for ex in read_jsonl(path)]
    validate_manifests(**splits)
    model = StrandsDeciderModel.load(args.checkpoint)
    engine = SystemOneEngine(model, EngineConfig(device=args.device))
    if model.config.input_mode != "multimodal":
        parser.error("requires a multimodal checkpoint")
    before = evaluate_conditions(engine, calibration)
    if args.fit_calibration:
        logits, labels, slots, examples = collect_logits(
            model, calibration, device=args.device, batch_size=1
        )
        model.config.temperature = fit_temperature(logits, labels, slots)
        model.config.temperature_by_kind = fit_temperature_by_kind(
            logits, labels, slots, examples, objective="nll"
        )
    report = evaluate_conditions(engine, evaluation)
    report["calibration"] = {
        "before": before,
        "after": evaluate_conditions(engine, calibration),
        "fitted_in_memory": args.fit_calibration,
    }
    report["source_manifest_sha256"] = {
        path: file_sha256(path)
        for path in [
            args.evaluation_manifest,
            args.calibration_manifest,
            *args.train_manifest,
            *args.validation_manifest,
        ]
    }
    report["checkpoint_fingerprint"] = fingerprint(args.checkpoint)
    report["training_split_independence_checked"] = bool(
        args.train_manifest and args.validation_manifest
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
