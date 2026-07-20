import csv
import gc
import time
from itertools import product
import torch
from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]

WARMUP_STEPS = 5
BENCHMARK_STEPS = 100
DEVICE = "cuda"
DTYPE = torch.float32
USE_JIT_COMPILER = True
JIT_APPANDIX = "_JIT_version" if USE_JIT_COMPILER else ""


def clear_grads(*tensors):
    for tensor in tensors:
        tensor.grad = None


def attention_forward(q, k, v):
    # 题目没有要求 causal mask，因此这里使用 mask=None

    if USE_JIT_COMPILER:
        compiled_attention = torch.compile(scaled_dot_product_attention)
        return compiled_attention(
            Q=q,
            K=k,
            V=v,
            mask=None, 
        )
    else:
        return scaled_dot_product_attention(
            Q=q,
            K=k,
            V=v,
            mask=None,
        )


def benchmark_one(d_model: int, seq_len: int):
    shape = (BATCH_SIZE, seq_len, d_model)

    q = torch.randn(
        shape, device=DEVICE, dtype=DTYPE, requires_grad=True
    )
    k = torch.randn(
        shape, device=DEVICE, dtype=DTYPE, requires_grad=True
    )
    v = torch.randn(
        shape, device=DEVICE, dtype=DTYPE, requires_grad=True
    )

    # Warmup：同时预热 forward 和 backward
    for _ in range(WARMUP_STEPS):
        clear_grads(q, k, v)

        output = attention_forward(q, k, v)
        torch.cuda.synchronize()

        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

        del output, loss

    forward_times = []
    backward_times = []
    memory_before_backward = []

    for _ in range(BENCHMARK_STEPS):
        clear_grads(q, k, v)

        # Forward
        torch.cuda.synchronize()
        start = time.perf_counter()

        output = attention_forward(q, k, v)

        # CUDA 是异步执行的，必须同步后才能得到正确时间
        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)

        # 构造标量 loss，但不把这部分算入 attention forward
        loss = output.sum()
        torch.cuda.synchronize()

        # 当前仍然存活的 CUDA tensor/计算图占用
        memory_before_backward.append(
            torch.cuda.memory_allocated(DEVICE)
        )

        # Backward
        torch.cuda.synchronize()
        start = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)

        del output, loss

    result = {
        "d_model": d_model,
        "seq_len": seq_len,
        "status": "ok",
        "forward_total_ms": sum(forward_times) * 1000,
        "forward_mean_ms": (
            sum(forward_times) / BENCHMARK_STEPS * 1000
        ),
        "backward_total_ms": sum(backward_times) * 1000,
        "backward_mean_ms": (
            sum(backward_times) / BENCHMARK_STEPS * 1000
        ),
        "memory_before_backward_gib": (
            max(memory_before_backward) / 1024**3
        ),
    }

    del q, k, v
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []

    # 笛卡尔积，共 20 组配置
    for d_model, seq_len in product(D_MODELS, SEQ_LENS):
        print(f"Running d_model={d_model}, seq_len={seq_len}")

        try:
            result = benchmark_one(d_model, seq_len)
        except torch.OutOfMemoryError:
            result = {
                "d_model": d_model,
                "seq_len": seq_len,
                "status": "OOM",
                "forward_total_ms": "",
                "forward_mean_ms": "",
                "backward_total_ms": "",
                "backward_mean_ms": "",
                "memory_before_backward_gib": "",
            }

        results.append(result)
        print(result)

        # 释放上一组配置的缓存，避免影响下一组
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    fieldnames = [
        "d_model",
        "seq_len",
        "status",
        "forward_total_ms",
        "forward_mean_ms",
        "backward_total_ms",
        "backward_mean_ms",
        "memory_before_backward_gib",
    ]

    with open(f"attention_benchmark{JIT_APPANDIX}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()