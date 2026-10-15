"""Verify CUDA computation before launching GUI grounding training."""
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("GPU verification failed; training must wait.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
x = torch.ones((64, 64), device="cuda")
value = (x @ x)[0, 0].item()
torch.cuda.synchronize()
if value != 64.0:
    raise SystemExit(f"Unexpected CUDA computation result: {value}")
print(f"CUDA computation verified: {value}")
