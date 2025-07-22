import torch
print("CUDA Available? ", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
