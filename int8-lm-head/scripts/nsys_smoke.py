import torch

print("CUDA:", torch.version.cuda, flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

x = torch.ones(4096, device="cuda")

for i in range(100):
    x = x + 1

torch.cuda.synchronize()

print("RESULT:", x[0].item(), flush=True)
print("NSYS SMOKE COMPLETE", flush=True)
