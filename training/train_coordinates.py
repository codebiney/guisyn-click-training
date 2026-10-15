"""Retrain GUI grounding with more targets, sharper images and coordinate-aware loss."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from evaluate_coordinates import parse_coordinates, score, summarize
from hit_training import box_penalty, mining_indices, remove_ambiguous_intents, ReplaySampler, selection_key

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
SUFFIX = " Return only the click coordinates as (x, y), normalized from 0 to 1000."


def coordinate_loss(logits, labels, digit_ids, target, numeric_weight=2.0, auxiliary_weight=3.0,
                    bbox_pixels=None, image_size=None, bbox_weight=0.0):
    """Response-only CE plus expected-digit coordinate regression on teacher-forced logits."""
    # logits[t] predicts labels[t], including the first response token and EOS.
    logits = logits.float()
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
    numeric = (labels[0, :, None] == digit_ids[None, :]).any(dim=1)
    weights = torch.where(numeric, numeric_weight, 1.0)
    ce = (losses * weights).sum() / weights.sum()
    positions = numeric.nonzero().flatten()
    lengths = [len(str(value)) for value in target]
    if positions.numel() != sum(lengths):
        raise RuntimeError("Coordinate digit alignment does not match response labels")
    numeric_logits = logits[0, positions][:, digit_ids]
    expected_digits = numeric_logits.softmax(-1) @ torch.arange(10, device=logits.device, dtype=torch.float32)
    coordinates = []
    offset = 0
    for length in lengths:
        place_values = torch.tensor([10 ** power for power in reversed(range(length))], device=logits.device)
        coordinates.append((expected_digits[offset:offset + length] * place_values).sum() / 1000)
        offset += length
    expected = torch.stack(coordinates)
    truth = torch.tensor(target, device=logits.device, dtype=torch.float32) / 1000
    auxiliary = F.smooth_l1_loss(expected, truth, beta=0.05)
    box = box_penalty(expected, bbox_pixels, image_size) if bbox_weight else expected.sum() * 0
    return ce + auxiliary_weight * auxiliary + bbox_weight * box, ce, auxiliary, box


def prepare_design(previous, output, seed, additional_sources=None):
    test = json.loads((previous / "heldout.json").read_text())
    test_ids = {e["source_id"] for e in test}
    sources = json.loads((ROOT / "data/guisyn/source_annotations.json").read_text())
    if additional_sources:
        sources += json.loads(Path(additional_sources).read_text())
    if len({s["id"] for s in sources}) != len(sources):
        raise RuntimeError("Duplicate source IDs")
    if (previous / "screenshot_split.json").exists():
        prior_split = json.loads((previous / "screenshot_split.json").read_text())
        validation_ids = set(prior_split["validation_ids"])
        if set(prior_split["test_ids"]) != test_ids:
            raise RuntimeError("Previous test split mismatch")
        train_ids = {s["id"] for s in sources} - validation_ids - test_ids
    else:
        candidate_ids = sorted(s["id"] for s in sources if s["id"] not in test_ids)
        random.Random(seed + 2).shuffle(candidate_ids)
        validation_ids = set(candidate_ids[:12])
        train_ids = set(candidate_ids[12:])
    if train_ids & test_ids or validation_ids & test_ids or train_ids & validation_ids:
        raise RuntimeError("Screenshot split overlap")
    train_targets, validation = [], []
    rng = random.Random(seed)
    for source in sources:
        if source["id"] in test_ids:
            continue
        path = Path(source["image"]) if "image" in source else ROOT / "data/guisyn" / f"{source['row_idx']:05d}.png"
        with Image.open(path) as image:
            w, h = image.size
        targets = []
        for index, annotation in enumerate(source["annotation"]):
            intents = sorted({text.strip() for text in annotation.get("intent", []) if text.strip()})
            cx, cy = annotation["x_center"], annotation["y_center"]
            bw, bh = annotation.get("width", 0), annotation.get("height", 0)
            if not intents or not 0 <= cx < w or not 0 <= cy < h or bw <= 0 or bh <= 0:
                continue
            point = [round(1000 * cx / w), round(1000 * cy / h)]
            targets.append({"image": str(path), "source_id": source["id"], "annotation_index": index,
                "intents": intents, "target": point, "answer": f"({point[0]}, {point[1]})",
                "image_size": [w, h], "target_center_pixels": [cx, cy],
                "bbox_pixels": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                "target_name": annotation["name"]})
        if source["id"] in validation_ids:
            for target in rng.sample(targets, min(4, len(targets))):
                validation.append({**target, "instruction": rng.choice(target["intents"])})
        else:
            train_targets.extend(targets)
    if (previous / "validation.json").exists():
        validation = json.loads((previous / "validation.json").read_text())
        if {e["source_id"] for e in validation} != validation_ids:
            raise RuntimeError("Previous validation split mismatch")
    if not train_targets or not validation:
        raise RuntimeError("Empty training/validation split")
    (output / "train_targets.json").write_text(json.dumps(train_targets, ensure_ascii=False, indent=2))
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    (output / "heldout.json").write_text(json.dumps(test, ensure_ascii=False, indent=2))
    split = {"train_ids": sorted(train_ids), "validation_ids": sorted(validation_ids), "test_ids": sorted(test_ids)}
    (output / "screenshot_split.json").write_text(json.dumps(split, indent=2))
    return train_targets, validation, split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/qwen2.5-vl-3b-guisyn-v2")
    parser.add_argument("--previous", default="runs/qwen2.5-vl-3b-guisyn")
    parser.add_argument("--init-adapter", help="Continue from this LoRA with a fresh optimizer/scheduler")
    parser.add_argument("--additional-sources", help="Manifest containing additional screenshots")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--coordinate-loss-weight", type=float, default=3.0)
    parser.add_argument("--bbox-loss-weight", type=float, default=0.0)
    parser.add_argument("--select-by", choices=["error", "bbox"], default="error")
    parser.add_argument("--hard-replay", action="store_true", help="Mine one target per training screen; 50% hard replay")
    parser.add_argument("--filter-ambiguous", action="store_true")
    parser.add_argument("--label-overrides", type=Path)
    parser.add_argument("--near-miss-pixels", type=float, default=None)
    parser.add_argument("--validate-every", type=int, default=128)
    parser.add_argument("--seed", type=int, default=6198)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists() or (output / "result.json").exists():
        raise RuntimeError("This output already has training results; choose a new directory")
    targets, validation, split = prepare_design(ROOT / args.previous, output, args.seed, args.additional_sources)
    corrections = []
    if args.label_overrides:
        overrides = json.loads(args.label_overrides.read_text())
        lookup = {(t['source_id'], t['annotation_index']): t for t in targets}
        for correction in overrides:
            key = (correction['source_id'], correction['annotation_index'])
            if key not in lookup or correction['source_id'] not in split['train_ids']:
                raise RuntimeError('Label override must match a training-only target')
            target = lookup[key]
            if target['bbox_pixels'] != correction['old_bbox_pixels']:
                raise RuntimeError('Label override original box mismatch')
            x0, y0, x1, y1 = correction['bbox_pixels']
            w, h = target['image_size']
            if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
                raise RuntimeError('Invalid corrected box')
            center = [(x0+x1)/2, (y0+y1)/2]
            point = [round(center[0]*1000/w), round(center[1]*1000/h)]
            target.update(bbox_pixels=correction['bbox_pixels'], target_center_pixels=center,
                          target=point, answer=f'({point[0]}, {point[1]})')
            corrections.append(correction)
        (output / 'label_corrections.json').write_text(json.dumps(corrections, indent=2))
        (output / 'train_targets.json').write_text(json.dumps(targets, ensure_ascii=False, indent=2))
    cleaning = {}
    if args.filter_ambiguous:
        targets, cleaning = remove_ambiguous_intents(targets)
        if not targets:
            raise RuntimeError("No unambiguous targets remain")
        (output / "train_targets.json").write_text(json.dumps(targets, ensure_ascii=False, indent=2))
        (output / "data_cleaning.json").write_text(json.dumps(cleaning, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    probe = torch.ones((64, 64), device="cuda")
    if (probe @ probe)[0, 0].item() != 64:
        raise RuntimeError("CUDA computation verification failed")
    torch.cuda.synchronize()
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True,
        min_pixels=64 * 28 * 28, max_pixels=args.max_pixels)
    print(f"Loading cached {MODEL}; NF4, max visual tokens={args.max_pixels // (28*28)}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
        torch_dtype=dtype, attn_implementation="sdpa", device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["visual", "lm_head"]))
    model.requires_grad_(False)
    for parameter in model.parameters():
        if parameter.ndim == 1 and parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.to(torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, ROOT / args.init_adapter, is_trainable=True)
        if model.peft_config["default"].r != args.rank:
            raise RuntimeError("Loaded adapter rank differs from configured rank")
        print(f"Loaded trainable adapter: {args.init_adapter}; fresh optimizer/scheduler", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            task_type="CAUSAL_LM"))
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("All model parameters must remain on CUDA")
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    digit_encoding = [processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)]
    if any(len(ids) != 1 for ids in digit_encoding) or len({ids[0] for ids in digit_encoding}) != 10:
        raise RuntimeError("Coordinate loss requires unique single-token digits")
    digit_ids = torch.tensor([ids[0] for ids in digit_encoding], device="cuda")
    warmup = min(64, max(1, args.steps // 10))
    config = {"model": MODEL, "dataset": "allenai/MolmoPoint-GUISyn", "subset": "mobile", "split": "train",
        "images": sum(len(ids) for ids in split.values()), "holdout_source_images": 128,
        "init_adapter": args.init_adapter, "additional_sources": args.additional_sources,
        "label_corrections": len(corrections), "near_miss_pixels": args.near_miss_pixels,
        "validate_every": args.validate_every,
        "optimizer_initialization": "fresh", "train_screenshots": len(split["train_ids"]),
        "validation_screenshots": len(split["validation_ids"]), "test_screenshots": len(split["test_ids"]),
        "train_targets": len(targets), "train_examples": sum(len(t["intents"]) for t in targets),
        "validation_examples": len(validation), "heldout_examples": 47, "steps": args.steps,
        "device": "cuda", "gradient_accumulation": 4, "learning_rate": args.learning_rate,
        "scheduler": "linear warmup then cosine to 10%", "warmup_steps": warmup,
        "seed": args.seed, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "load_in_4bit": True, "max_pixels": args.max_pixels, "lora_rank": args.rank,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "coordinate_format": "(x, y), normalized integer coordinates 0..1000",
        "loss": "numeric-weighted response-only CE + expected-digit coordinate SmoothL1 + optional size-aware box penalty",
        "numeric_token_weight": 2.0, "coordinate_loss_weight": args.coordinate_loss_weight,
        "bbox_loss_weight": args.bbox_loss_weight,
        "bbox_surrogate": "log1p distance outside inner 80% box / max(box size, 0.01 image axis)",
        "data_cleaning": cleaning, "hard_replay": args.hard_replay,
        "hard_sampling_fraction": 0.5 if args.hard_replay else 0.0,
        "select_by": args.select_by,
        "checkpoint_selection": ("highest validation bbox hits; ties: lowest normalized center error"
            if args.select_by == "bbox" else "lowest mean normalized center error on independent validation screens; invalid=1414.21"),
        "trainable_parameters": sum(p.numel() for p in trainable)}
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)

    def prompt_inputs(example, answer=False):
        with Image.open(example["image"]) as source:
            image = source.convert("RGB")
        prompt = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": example["instruction"] + SUFFIX}]}],
            tokenize=False, add_generation_prompt=True)
        prefix = processor(text=prompt, images=[image], return_tensors="pt")
        if not answer:
            return prefix.to("cuda")
        inputs = processor(text=prompt + example["answer"] + processor.tokenizer.eos_token,
                           images=[image], return_tensors="pt")
        length = prefix["input_ids"].shape[1]
        if not torch.equal(prefix["input_ids"], inputs["input_ids"][:, :length]):
            raise RuntimeError("Prompt/response token boundary mismatch")
        response = inputs["input_ids"][:, length:].to("cuda")
        if not response.numel():
            raise RuntimeError("Empty coordinate response")
        return inputs.to("cuda"), response

    def save(name, step):
        path = output / name
        model.save_pretrained(path)
        processor.save_pretrained(path)
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, path / "training_state.pt")
        print(f"saved {path} at step {step}", flush=True)

    def validate(step):
        model.eval()
        rows = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for example in validation:
                inputs = prompt_inputs(example)
                generated = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                    num_beams=1, use_cache=True, logits_to_keep=1)
                tokens = generated[0, inputs["input_ids"].shape[1]:]
                text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                rows.append({**example, "finetuned": score(text, example, tokens.numel(), 0)})
        summary = summarize(rows, "finetuned")
        distances = []
        for row in rows:
            xy = row['finetuned'].get('predicted_pixels')
            x0, y0, x1, y1 = row['bbox_pixels']
            distances.append(math.hypot(max(x0-xy[0], 0, xy[0]-x1),
                max(y0-xy[1], 0, xy[1]-y1)) if xy else math.inf)
        summary['within_bbox_distance_pixels'] = {str(t): sum(d <= t for d in distances)/len(rows)
                                                   for t in (0, 5, 10)}
        selection_error = sum(row["finetuned"].get("center_error_normalized", math.sqrt(2) * 1000)
                              for row in rows) / len(rows)
        result = {"step": step, "selection_error": selection_error, **summary}
        (output / f"validation-{step}.json").write_text(json.dumps({"summary": result, "predictions": rows},
            ensure_ascii=False, indent=2))
        with (output / "validation_metrics.jsonl").open("a") as file:
            file.write(json.dumps(result) + "\n")
        print("validation " + json.dumps(result), flush=True)
        model.train()
        torch.cuda.empty_cache()
        return result

    def mine_hard_targets():
        model.eval()
        candidates = mining_indices(targets, args.seed + 17)
        corrected_keys = {(c['source_id'], c['annotation_index']) for c in corrections}
        candidates = sorted(set(candidates) | {i for i, t in enumerate(targets)
            if (t['source_id'], t['annotation_index']) in corrected_keys})
        hard = []
        all_misses = 0
        print(f"Mining {len(candidates)} training-only targets for hard replay", flush=True)
        with (output / "hard_mining.jsonl").open("w", buffering=1) as file:
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                for count, index in enumerate(candidates, 1):
                    target = targets[index]
                    example = {**target, "instruction": target["intents"][0]}
                    inputs = prompt_inputs(example)
                    generated = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                        num_beams=1, use_cache=True, logits_to_keep=1)
                    tokens = generated[0, inputs["input_ids"].shape[1]:]
                    prediction = score(processor.tokenizer.decode(tokens, skip_special_tokens=True),
                        example, tokens.numel(), 0)
                    xy = prediction.get('predicted_pixels')
                    x0, y0, x1, y1 = target['bbox_pixels']
                    distance = math.hypot(max(x0-xy[0], 0, xy[0]-x1),
                        max(y0-xy[1], 0, xy[1]-y1)) if xy else math.inf
                    prediction['distance_outside_bbox_pixels'] = distance if xy else None
                    if not prediction['bbox_hit']:
                        all_misses += 1
                        if args.near_miss_pixels is None or distance <= args.near_miss_pixels:
                            hard.append(index)
                    file.write(json.dumps({"target_index": index, "source_id": target["source_id"],
                        "annotation_index": target["annotation_index"], "instruction": example["instruction"],
                        "prediction": prediction}) + "\n")
                    if count % 10 == 0 or count == len(candidates):
                        print(f"hard mining {count}/{len(candidates)}; misses={len(hard)}", flush=True)
        (output / "hard_mining_summary.json").write_text(json.dumps({
            "candidates": len(candidates), "misses": all_misses, "replay_targets": len(hard),
            "near_miss_pixels": args.near_miss_pixels, "hard_target_indices": hard,
            "candidate_selection": "one per train screen, alternating small-target quartile/random",
            "validation_or_test_used": False}, indent=2))
        model.train()
        torch.cuda.empty_cache()
        return hard

    started = time.monotonic()
    best = validate(0)
    save("best", 0)
    hard_indices = mine_hard_targets() if args.hard_replay else []
    sampler = ReplaySampler(targets, hard_indices, args.seed)
    probe_weight = next(p for name, p in model.named_parameters() if p.requires_grad and "lora_B" in name)
    initial = probe_weight.detach().clone()
    with (output / "metrics.jsonl").open("w", buffering=1) as file:
        for step in range(1, args.steps + 1):
            factor = step / warmup if step <= warmup else 0.1 + 0.9 * (1 + math.cos(
                math.pi * (step - warmup) / max(1, args.steps - warmup))) / 2
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            totals = [0.0, 0.0, 0.0, 0.0]
            for microstep in range(4):
                example = sampler.next(hard=args.hard_replay and microstep < 2)
                inputs, labels = prompt_inputs(example, answer=True)
                with torch.autocast("cuda", dtype=dtype):
                    # Keep only response prediction logits, avoiding a huge vocabulary
                    # projection and float32 loss tensor for every image/prompt token.
                    logits = model(**inputs, logits_to_keep=labels.shape[1] + 1).logits[:, :-1, :]
                    loss, ce, auxiliary, box = coordinate_loss(logits, labels, digit_ids, example["target"],
                        auxiliary_weight=args.coordinate_loss_weight, bbox_pixels=example["bbox_pixels"],
                        image_size=example["image_size"], bbox_weight=args.bbox_loss_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                (loss / 4).backward()
                for index, value in enumerate((loss, ce, auxiliary, box)):
                    totals[index] += value.item() / 4
                del inputs, labels, logits, loss, ce, auxiliary, box
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            metric = {"step": step, "loss": totals[0], "cross_entropy": totals[1],
                "coordinate_auxiliary": totals[2], "bbox_auxiliary": totals[3],
                "sampling_counts": dict(sampler.counts), "grad_norm": norm.item(), "learning_rate": args.learning_rate * factor,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "gpu_peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1)}
            file.write(json.dumps(metric) + "\n")
            print(json.dumps(metric), flush=True)
            if step == 1:
                if torch.equal(probe_weight.detach(), initial):
                    raise RuntimeError("Optimizer did not update LoRA weights")
                save("checkpoint-1", step)
            if step % args.validate_every == 0 or step == args.steps:
                score_result = validate(step)
                save(f"checkpoint-{step}", step)
                if selection_key(score_result, args.select_by) < selection_key(best, args.select_by):
                    best = score_result
                    save("best", step)
    (output / "sampling_summary.json").write_text(json.dumps({"counts": sampler.counts,
        "unique_targets_seen": sum(v > 0 for v in sampler.visits)}, indent=2))
    save("last", args.steps)
    weights = load_peft_weights(output / "best", device="cuda")
    set_peft_model_state_dict(model, weights, adapter_name="default")
    final = output / "final"
    model.save_pretrained(final)
    processor.save_pretrained(final)
    shutil.copyfile(output / "best/training_state.pt", final / "training_state.pt")
    result = {"completed_steps": args.steps, "selected_step": best["step"],
        "validation": best, "elapsed_seconds": round(time.monotonic() - started, 1)}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("training complete " + json.dumps(result), flush=True)
    # Evaluate the fixed 47-example test only after validation-based selection.
    # Replace this process so the full CUDA context is released before evaluation.
    with (output / "coordinate_eval.log").open("w") as log:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.execv(sys.executable, [sys.executable, "-u", str(ROOT / "training/evaluate_coordinates.py"),
                                  "--run", args.output])



if __name__ == "__main__":
    main()
