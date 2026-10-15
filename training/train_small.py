"""Download a reproducible GUISyn subset and train coordinate-response LoRA."""
import os
os.environ.setdefault("HF_HOME", os.path.abspath("data/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import argparse
import json
import random
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (AutoProcessor, SmolVLMForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

def prepare(count):
    dest = ROOT / "data/guisyn"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / "examples.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    rng = random.Random(6198)
    examples = []
    sources = []
    for offset in range(0, count, 32):
        response = requests.get("https://datasets-server.huggingface.co/rows", params={
            "dataset": "allenai/MolmoPoint-GUISyn", "config": "mobile",
            "split": "train", "offset": offset, "length": min(32, count-offset)}, timeout=120)
        response.raise_for_status()
        for entry in response.json()["rows"]:
            if entry.get("truncated_cells"):
                raise RuntimeError("Dataset server truncated annotation; refusing incomplete data")
            row = entry["row"]
            path = dest / f'{entry["row_idx"]:05d}.png'
            if not path.exists():
                image_response = requests.get(row["image"]["src"], timeout=120)
                image_response.raise_for_status()
                path.write_bytes(image_response.content)
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            annotations = row["annotation"]
            if isinstance(annotations, str):
                annotations = json.loads(annotations)
            valid = [a for a in annotations if a.get("intent") and
                     0 <= a["x_center"] < width and 0 <= a["y_center"] < height]
            for a in rng.sample(valid, min(4, len(valid))):
                examples.append({"image": str(path), "source_id": row["id"],
                    "instruction": rng.choice(a["intent"]),
                    "answer": f'({round(1000*a["x_center"]/width)}, {round(1000*a["y_center"]/height)})'})
            sources.append({"row_idx": entry["row_idx"], "id": row["id"], "annotation": annotations})
        print(f"downloaded {len(sources)}/{count} screenshots", flush=True)
    manifest.write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    (dest / "source_annotations.json").write_text(json.dumps(sources, ensure_ascii=False))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", default="runs/smolvlm2-guisyn")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default=MODEL,
                        choices=[MODEL, "Qwen/Qwen2.5-VL-3B-Instruct"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=256*28*28)
    args = parser.parse_args()
    torch.manual_seed(6198)
    torch.set_num_threads(6)
    examples = prepare(args.images)
    if not examples:
        raise RuntimeError("No training examples")
    if args.prepare_only:
        return
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    device = args.device
    if args.load_in_4bit and device != "cuda":
        raise RuntimeError("4-bit training requires CUDA; no CPU fallback")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: repair GPU before starting training")
        probe = torch.ones((64, 64), device="cuda")
        probe @ probe
        torch.cuda.synchronize()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    is_qwen = args.model.startswith("Qwen/")
    print(f"loading {args.model}; device={device}; 4bit={args.load_in_4bit}", flush=True)
    processor_options = {"min_pixels": 64*28*28, "max_pixels": args.max_pixels} if is_qwen else {}
    processor = AutoProcessor.from_pretrained(args.model, **processor_options)
    if is_qwen:
        processor.image_processor.min_pixels = 64*28*28
        processor.image_processor.max_pixels = args.max_pixels
    else:
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": 512}
    load_options = {"torch_dtype": dtype, "attn_implementation": "sdpa"}
    if args.load_in_4bit:
        load_options.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"]))
    model_class = Qwen2_5_VLForConditionalGeneration if is_qwen else SmolVLMForConditionalGeneration
    model = model_class.from_pretrained(args.model, **load_options)
    model.config.use_cache = False
    if args.load_in_4bit:
        # PEFT's generic preparation upcasts every non-quantized tensor, including
        # the entire vision tower and large embeddings, causing a transient OOM.
        # Freeze the base and only upcast small normalization/bias parameters.
        model.requires_grad_(False)
        for parameter in model.parameters():
            if parameter.ndim == 1 and parameter.dtype in (torch.float16, torch.bfloat16):
                parameter.data = parameter.data.to(torch.float32)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.cuda.empty_cache()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=(r".*language_model.*\.(q_proj|v_proj)$" if is_qwen else
                        r".*text_model.*\.(q_proj|v_proj)$"), task_type="CAUSAL_LM"))
    if not args.load_in_4bit:
        model.to(device)
    if any(p.device.type != device for p in model.parameters()):
        raise RuntimeError("Model parameters are not entirely on the requested device")
    model.train()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    # LoRA B starts at zero and receives the first update; A can remain unchanged at step 1.
    trainable = next(p for name, p in model.named_parameters() if p.requires_grad and "lora_B" in name)
    initial = trainable.detach().clone()
    split_ids = sorted({e["source_id"] for e in examples})
    random.Random(6198).shuffle(split_ids)
    heldout_ids = set(split_ids[:max(1, len(split_ids)//10)])
    train_examples = [e for e in examples if e["source_id"] not in heldout_ids]
    eval_examples = [e for e in examples if e["source_id"] in heldout_ids]
    random.Random(6198).shuffle(train_examples)
    config = {"model": args.model, "dataset": "allenai/MolmoPoint-GUISyn", "subset": "mobile",
        "split": "train", "images": len(split_ids), "train_examples": len(train_examples),
        "heldout_examples": len(eval_examples), "steps": args.steps, "device": device,
        "gradient_accumulation": 4, "learning_rate": 2e-4, "seed": 6198,
        "coordinate_format": "(x, y), normalized integer coordinates 0..1000",
        "torch": torch.__version__, "load_in_4bit": args.load_in_4bit,
        "max_pixels": args.max_pixels if is_qwen else None,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    (output / "heldout.json").write_text(json.dumps(eval_examples, indent=2))
    print(json.dumps(config), flush=True)

    def batch(example):
        image = Image.open(example["image"]).convert("RGB")
        prompt = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": example["instruction"] +
             " Return only the click coordinates as (x, y), normalized from 0 to 1000."}]}],
            tokenize=False, add_generation_prompt=True)
        prefix = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = processor(text=prompt + example["answer"] + processor.tokenizer.eos_token,
                           images=[image], return_tensors="pt")
        labels = inputs["input_ids"].clone()
        labels[:, :prefix["input_ids"].shape[1]] = -100
        if not (labels != -100).any():
            raise RuntimeError("No supervised response tokens")
        inputs["labels"] = labels
        return {k: v.to(device) for k, v in inputs.items()}

    def save(name):
        checkpoint = output / name
        model.save_pretrained(checkpoint)
        processor.save_pretrained(checkpoint)
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, checkpoint / "training_state.pt")
        print(f"saved {checkpoint}", flush=True)

    started = time.monotonic()
    with (output / "metrics.jsonl").open("a", buffering=1) as log:
        for step in range(1, args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for micro in range(4):
                example = train_examples[((step-1)*4+micro) % len(train_examples)]
                with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                    loss = model(**batch(example)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                (loss / 4).backward()
                total += loss.item()/4
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            metric = {"step": step, "loss": total, "grad_norm": grad_norm.item(),
                      "elapsed_seconds": round(time.monotonic()-started, 1)}
            if device == "cuda":
                metric["gpu_peak_memory_mib"] = round(torch.cuda.max_memory_allocated()/1024**2, 1)
            log.write(json.dumps(metric)+"\n")
            print(json.dumps(metric), flush=True)
            if step == 1:
                if torch.equal(initial, trainable.detach()):
                    raise RuntimeError("Optimizer did not update trainable weights")
                print("verified: optimizer updated LoRA weights", flush=True)
                save("checkpoint-1")
            if step % 32 == 0:
                save(f"checkpoint-{step}")
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            losses = [model(**batch(e)).loss.item() for e in eval_examples[:16]]
        if not all(torch.isfinite(torch.tensor(loss)).item() for loss in losses):
            raise RuntimeError("Non-finite held-out loss")
        result = {"completed_steps": args.steps, "heldout_loss": sum(losses)/len(losses),
                  "heldout_evaluated_examples": len(losses)}
        (output / "result.json").write_text(json.dumps(result, indent=2))
        save("final")
        print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
