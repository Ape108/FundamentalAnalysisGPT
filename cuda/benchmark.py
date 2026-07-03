import torch
import time

# Set the size of the matrix. 
# 10,000 x 10,000 will use about 1.2 GB of VRAM, well within your 6GB limit.
MATRIX_SIZE = 15000 

print(f"--- PyTorch Benchmark: {MATRIX_SIZE}x{MATRIX_SIZE} Matrix Multiplication ---")

# ==========================================
# 1. CPU TEST
# ==========================================
print("\n[CPU] Generating matrices...")
a_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE)
b_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE)

print("[CPU] Multiplying... (Grab a coffee, this will take a moment)")
start_time = time.perf_counter()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.perf_counter() - start_time
print(f"[CPU] Time: {cpu_time:.4f} seconds")

# ==========================================
# 2. GPU TEST
# ==========================================
device_name = torch.cuda.get_device_name(0)
print(f"\n[GPU] Moving matrices to {device_name}...")
a_gpu = a_cpu.cuda()
b_gpu = b_cpu.cuda()

# WARMUP: We run a dummy calculation first. 
# GPUs go into a low-power state when idle. The first operation wakes it up,
# which takes a few milliseconds. We don't want that wake-up time in our benchmark.
print("[GPU] Warming up...")
_ = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize() 

print("[GPU] Multiplying...")
start_time = time.perf_counter()
c_gpu = torch.matmul(a_gpu, b_gpu)

# CRITICAL STEP: torch.cuda.synchronize()
# GPU operations in PyTorch are "asynchronous". If we don't include this line, 
# Python will stop the timer the moment it *sends* the instruction to the GPU, 
# not when the GPU actually *finishes* the math, resulting in fake 0.0001s times.
torch.cuda.synchronize() 

gpu_time = time.perf_counter() - start_time
print(f"[GPU] Time: {gpu_time:.4f} seconds")

# ==========================================
# RESULTS
# ==========================================
print("\n" + "="*50)
if gpu_time > 0:
    speedup = cpu_time / gpu_time
    print(f"RESULT: Your GPU was {speedup:.2f}x faster than your CPU!")
else:
    print("RESULT: GPU was too fast to measure accurately.")
print("="*50)