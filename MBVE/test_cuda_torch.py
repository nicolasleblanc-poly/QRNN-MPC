import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
print(torch.__version__)
print(torch.version.cuda)
