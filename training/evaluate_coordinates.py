"""Evaluate free-generated GUI click coordinates on the saved screenshot holdout."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/huggingface"))

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

COORDINATES = re.compile(r"\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*")


def parse_coordinates(text):
    match = COORDINATES.fullmatch(text)
    if match is None:
        return None
    x, y = map(int, match.groups())
    return [x, y] if 0 <= x <= 1000 and 0 <= y <= 1000 else None


def prepare_examples(run):
    examples = json.loads((run / "heldout.json").read_text())
    sources = {s["id"]: s for s in json.loads((ROOT / "data/guisyn/source_annotations.json").read_text())}
    manifest = json.loads((ROOT / "data/guisyn/examples.json").read_text())
    heldout_ids = {e["source_id"] for e in examples}
    train_ids = {e["source_id"] for e in manifest if e["source_id"] not in heldout_ids}
    config = json.loads((run / "run_config.json").read_text())
    if len(examples) != config["heldout_examples"] or len(train_ids | heldout_ids) != config.get("holdout_source_images", config["images"]):
        raise RuntimeError("Saved holdout does not match the training configuration")
    # Independently reconstruct the exact screenshot split used by the trainer.
    import random
    split_ids = sorted({e["source_id"] for e in manifest})
    random.Random(config["seed"]).shuffle(split_ids)
    if heldout_ids != set(split_ids[:max(1, len(split_ids) // 10)]):
        raise RuntimeError("Holdout screenshots differ from the training split")
    for index, example in enumerate(examples):
        with Image.open(example["image"]) as image:
            w, h = image.size
        target = parse_coordinates(example["answer"])
        matches = [a for a in sources[example["source_id"]]["annotation"]
                   if example["instruction"] in a["intent"] and
                   [round(1000 * a["x_center"] / w), round(1000 * a["y_center"] / h)] == target]
        if len(matches) != 1:
            raise RuntimeError(f"Ambiguous target annotation for example {index}")
        annotation = matches[0]
        if annotation["width"] <= 0 or annotation["height"] <= 0:
            raise RuntimeError(f"Invalid target area for example {index}")
        cx, cy = annotation["x_center"], annotation["y_center"]
        example.update(index=index, image_size=[w, h], target=target,
                       target_name=annotation["name"], target_center_pixels=[cx, cy],
                       bbox_pixels=[cx - annotation["width"] / 2, cy - annotation["height"] / 2,
                                    cx + annotation["width"] / 2, cy + annotation["height"] / 2])
    return examples, config


def score(text, example, tokens, elapsed):
    point = parse_coordinates(text)
    result = {"response": text, "predicted_normalized": point, "valid": point is not None,
              "bbox_hit": False, "exact_center_match": False, "tokens": tokens,
              "elapsed_seconds": round(elapsed, 3)}
    if point is None:
        return result
    w, h = example["image_size"]
    px, py = point[0] * w / 1000, point[1] * h / 1000
    left, top, right, bottom = example["bbox_pixels"]
    cx, cy = example["target_center_pixels"]
    # Use original centers for distances, avoiding rounding the ground truth.
    normalized_distance = math.hypot(point[0] - cx * 1000 / w, point[1] - cy * 1000 / h)
    pixel_distance = math.hypot(px - cx, py - cy)
    result.update(predicted_pixels=[px, py], bbox_hit=left <= px <= right and top <= py <= bottom,
                  exact_center_match=point == example["target"],
                  center_error_normalized=normalized_distance, center_error_pixels=pixel_distance,
                  center_error_diagonal_percent=100 * pixel_distance / math.hypot(w, h))
    return result


def summarize(rows, mode):
    results = [row[mode] for row in rows]
    valid = [r for r in results if r["valid"]]
    n = len(results)
    hits = sum(r["bbox_hit"] for r in results)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_id"], []).append(row[mode]["bbox_hit"])
    summary = {"examples": n, "valid_coordinates": len(valid), "valid_rate": len(valid) / n,
               "bbox_hits": hits, "bbox_accuracy": hits / n,
               "screenshot_macro_bbox_accuracy": statistics.mean(sum(v) / len(v) for v in grouped.values()),
               "exact_center_matches": sum(r["exact_center_match"] for r in results),
               "center_within_normalized_units": {
                   str(t): sum(r.get("center_error_normalized", math.inf) <= t for r in results) / n
                   for t in (10, 25, 50, 100)},
               "invalid_predictions_count_as_misses": True}
    for key in ("center_error_normalized", "center_error_pixels", "center_error_diagonal_percent"):
        values = [r[key] for r in valid]
        summary[f"mean_{key}_valid_only"] = statistics.mean(values) if values else None
        summary[f"median_{key}_valid_only"] = statistics.median(values) if values else None
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/qwen2.5-vl-3b-guisyn")
    args = parser.parse_args()
    run = ROOT / args.run
    output = run / "coordinate_eval"
    output.mkdir(parents=True, exist_ok=True)
    examples, config = prepare_examples(run)
    if config["model"] != "Qwen/Qwen2.5-VL-3B-Instruct" or not config["load_in_4bit"]:
        raise RuntimeError("This evaluator requires the prepared Qwen 3B NF4 run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU evaluation fallback")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(6)
    probe = torch.ones((64, 64), device="cuda")
    if (probe @ probe)[0, 0].item() != 64:
        raise RuntimeError("CUDA computation verification failed")
    torch.cuda.synchronize()
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(run / "final", local_files_only=True,
        min_pixels=64 * 28 * 28, max_pixels=config["max_pixels"])
    print(f"Loading cached base + final LoRA on {torch.cuda.get_device_name(0)}", flush=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["model"], local_files_only=True,
        torch_dtype=dtype, attn_implementation="sdpa", device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["visual", "lm_head"]))
    base.requires_grad_(False)
    # Reproduce the normalization precision used for training.
    for parameter in base.parameters():
        if parameter.ndim == 1 and parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.to(torch.float32)
    model = PeftModel.from_pretrained(base, run / "final", is_trainable=False, local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("Evaluation model is not entirely on CUDA")
    torch.cuda.empty_cache()
    started = time.monotonic()
    rows = []
    with (output / "predictions.jsonl").open("w", buffering=1) as file:
        for example in examples:
            with Image.open(example["image"]) as source_image:
                image = source_image.convert("RGB")
            prompt = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": example["instruction"] +
                 " Return only the click coordinates as (x, y), normalized from 0 to 1000."}]}],
                tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
            row = dict(example)
            # Generate from the prompt alone; never supply the ground-truth response.
            for mode in ("finetuned", "baseline"):
                context = model.disable_adapter() if mode == "baseline" else nullcontext()
                tick = time.monotonic()
                with context, torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    generated = model.generate(**inputs, do_sample=False, num_beams=1,
                        max_new_tokens=32, use_cache=True, logits_to_keep=1)
                continuation = generated[0, inputs["input_ids"].shape[1]:]
                text = processor.tokenizer.decode(continuation, skip_special_tokens=True)
                row[mode] = score(text, example, continuation.numel(), time.monotonic() - tick)
                row[mode]["generation_token_limit_reached"] = continuation.numel() >= 32
            rows.append(row)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"example": example["index"] + 1, "total": len(examples),
                "target": example["target"], "finetuned": row["finetuned"]["response"],
                "finetuned_hit": row["finetuned"]["bbox_hit"],
                "baseline": row["baseline"]["response"], "baseline_hit": row["baseline"]["bbox_hit"]}), flush=True)
    summary = {"model": config["model"], "checkpoint": str(run / "final"),
        "adapter_sha256": hashlib.sha256((run / "final/adapter_model.safetensors").read_bytes()).hexdigest(),
        "base_revision": getattr(base.config, "_commit_hash", None),
        "dataset": config["dataset"], "screenshots": len({e["source_id"] for e in examples}),
        "split": "saved screenshot-level holdout from the 128-image mobile training subset",
        "metric": "Predicted click inside original target rectangle (inclusive edges).",
        "output_contract": "Exactly one integer (x, y) pair in 0..1000; malformed/out-of-range outputs are misses.",
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 32,
                       "min_pixels": 64 * 28 * 28, "max_pixels": config["max_pixels"]},
        "device": "cuda", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "finetuned": summarize(rows, "finetuned"), "baseline": summarize(rows, "baseline")}
    summary["bbox_accuracy_change_percentage_points"] = 100 * (
        summary["finetuned"]["bbox_accuracy"] - summary["baseline"]["bbox_accuracy"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Held-out coordinate generation evaluation", "",
        f"{len(examples)} instructions on {summary['screenshots']} screenshots excluded from training.", "",
        "Predicted clicks are judged against the original target rectangle. Invalid outputs are misses.", "",
        "| Metric | Base (LoRA disabled) | Final LoRA |", "|---|---:|---:|"]
    for label, key in (("Target rectangle hit rate", "bbox_accuracy"),
                       ("Valid coordinate output", "valid_rate"),
                       ("Mean per-screenshot target hit rate", "screenshot_macro_bbox_accuracy")):
        lines.append(f"| {label} | {100*summary['baseline'][key]:.2f}% | {100*summary['finetuned'][key]:.2f}% |")
    for threshold in (10, 25, 50, 100):
        values = [summary[m]["center_within_normalized_units"][str(threshold)] for m in ("baseline", "finetuned")]
        lines.append(f"| Center distance ≤ {threshold} (0–1000 normalized axes) | {100*values[0]:.2f}% | {100*values[1]:.2f}% |")
    for label, key in (("Median center error in pixels (valid outputs)", "median_center_error_pixels_valid_only"),
                       ("Mean center error in pixels (valid outputs)", "mean_center_error_pixels_valid_only")):
        values = [summary[m][key] for m in ("baseline", "finetuned")]
        lines.append(f"| {label} | {values[0]:.2f} | {values[1]:.2f} |" if all(v is not None for v in values) else
                     f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", f"Change in target hit rate: {summary['bbox_accuracy_change_percentage_points']:+.2f} percentage points.",
        "", "Greedy free generation uses the exact training prompt and image preprocessing. The base comparison",
        "disables the adapter on the same NF4 model, using identical generation settings. No target response",
        "is supplied to the model. Pixel errors use original, unrounded annotation centers. Normalized",
        "distance scales x by image width and y by image height; it is not a pixel-diagonal percentage.",
        "", "This is a small local holdout of 12 screenshots, not an official GUISyn or ScreenSpot benchmark.",
        "Per-example predictions, target boxes and errors are saved in predictions.jsonl; full metrics are in summary.json.", ""]
    (output / "report.md").write_text("\n".join(lines))
    # Preserve fresh inference as an annotated gallery as well as raw predictions.
    result_path = run / 'result.json'
    selected_step = json.loads(result_path.read_text()).get('selected_step') if result_path.exists() else None
    gallery_input = output / 'visualization_input.json'
    gallery_input.write_text(json.dumps({'split': 'held-out test',
        'summary': {'step': selected_step, **summary['finetuned']}, 'predictions': rows},
        ensure_ascii=False, indent=2))
    import subprocess
    subprocess.run([sys.executable, str(ROOT / 'training/visualize_predictions.py'),
        '--predictions', str(gallery_input), '--output', str(ROOT / 'output' / (run.name + '_test'))], check=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
