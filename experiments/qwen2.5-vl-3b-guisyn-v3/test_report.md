# Held-out coordinate generation evaluation

47 instructions on 12 screenshots excluded from training.

Predicted clicks are judged against the original target rectangle. Invalid outputs are misses.

| Metric | Base (LoRA disabled) | Final LoRA |
|---|---:|---:|
| Target rectangle hit rate | 0.00% | 34.04% |
| Valid coordinate output | 59.57% | 100.00% |
| Mean per-screenshot target hit rate | 0.00% | 33.33% |
| Center distance ≤ 10 (0–1000 normalized axes) | 0.00% | 21.28% |
| Center distance ≤ 25 (0–1000 normalized axes) | 0.00% | 44.68% |
| Center distance ≤ 50 (0–1000 normalized axes) | 2.13% | 61.70% |
| Center distance ≤ 100 (0–1000 normalized axes) | 2.13% | 76.60% |
| Median center error in pixels (valid outputs) | 193.28 | 19.01 |
| Mean center error in pixels (valid outputs) | 240.72 | 91.00 |

Change in target hit rate: +34.04 percentage points.

Greedy free generation uses the exact training prompt and image preprocessing. The base comparison
disables the adapter on the same NF4 model, using identical generation settings. No target response
is supplied to the model. Pixel errors use original, unrounded annotation centers. Normalized
distance scales x by image width and y by image height; it is not a pixel-diagonal percentage.

This is a small local holdout of 12 screenshots, not an official GUISyn or ScreenSpot benchmark.
Per-example predictions, target boxes and errors are saved in predictions.jsonl; full metrics are in summary.json.
