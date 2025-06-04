
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except:
    print("Transformers not installed")

try:
    import peft
    print("PEFT: Installed")
except:
    print("PEFT not installed")

try:
    import bitsandbytes as bnb
    print("Bitsandbytes: Installed")
except:
    print("Bitsandbytes not installed")
