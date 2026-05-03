import torch

if torch.cuda.is_available():
    print("CUDA is available.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

print(torch.__version__)
print(torch.version.cuda)
print(torch.randn(1).cuda())