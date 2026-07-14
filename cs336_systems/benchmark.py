from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import get_cosine_lr, AdamW
import timeit
import argparse
import torch
import numpy as np
import random
import os
import torch.cuda.nvtx as nvtx
import json
from contextlib import nullcontext

# CUDA_VISIBLE_DEVICES=0
# uv run nsys profile -- python benchmark.py
# uv run nsys profile -- python benchmark.py --profile_attn=1
# uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autogradshapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python benchmark.py --profile_attn=1
"""
前置条件：
export CUDA_VISIBLE_DEVICES=0
export PATH="/home/huiwei/fy/zhangshuai/assignment2-systems/nsys/extract/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64:$PATH"


uv run nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  --cudabacktrace=all \
  --python-backtrace=cuda \
  -- python benchmark.py --profile_attn 1

uv run nsys profile \
  -o report_attn \
  -f true \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  --cudabacktrace=kernel,sync \
  --python-backtrace=cuda \
  -- python benchmark.py --profile_attn 1

"""
def parse_args():
    p = argparse.ArgumentParser()
    g_model = p.add_argument_group("model")
    g_model.add_argument("--vocab_size", type=int, default=10000)
    g_model.add_argument("--context_length", type=int, default=256)
    g_model.add_argument("--d_model", type=int, default=512)
    g_model.add_argument("--batch_size", type=int, default=64)
    g_model.add_argument("--num_layers", type=int, default=4)
    g_model.add_argument("--num_heads", type=int, default=16)
    g_model.add_argument("--d_ff", type=int, default=1344)
    g_model.add_argument("--rope_theta", type=float, default=10000.0)
    
    g_optimizer = p.add_argument_group("optimizer")
    g_optimizer.add_argument("--weight_decay", type=float, default=0.1)
    g_optimizer.add_argument("--betas", type=float, default=[0.9, 0.95], nargs=2)
    g_optimizer.add_argument("--eps", type=float, default=1e-8)

    g_training = p.add_argument_group("training")
    g_training.add_argument("--max_learning_rate", type=float, default=1e-3)
    g_training.add_argument("--min_learning_rate", type=float, default=1e-4)
    g_training.add_argument("--warmup_iters", type=int, default=1000)
    g_training.add_argument("--cosine_cycle_iters", type=int, default=19500)
    g_training.add_argument("--end_iter", type=int, default=20000)
    g_training.add_argument("--grad_clip_norm", type=float, default=1.0)
    g_training.add_argument("--val_interval", type=int, default=500)
    g_training.add_argument("--val_batches", type=int, default=50)
    g_training.add_argument("--device", type=str, default="cuda")
    g_training.add_argument("--seed", type=int, default=336)
    g_training.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])

    g_profiling = p.add_argument_group("profiling")
    g_profiling.add_argument("--warmup_steps", type=int, default=5)
    g_profiling.add_argument("--profiling_steps", type=int, default=10)
    g_profiling.add_argument("--profiling_warmup", type=int, default=0)
    g_profiling.add_argument("--profile_attn", type=int, default=0)
    g_profiling.add_argument("--use_mixed_precision", type=int, default=0)

    return p.parse_args()
    

def generate_random_data(batch_size, context_length, vocab_size, device):
    data = torch.randint(low=0, high=vocab_size,
                        size=(batch_size, context_length + 1),
                        device=device, 
                        dtype=torch.long)
    # pytorch 中索引要求为 int64 因此 dtype 使用 torch.long

    x = data[:, :-1]
    y = data[:, 1:]

    return x, y


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    batch_size = inputs.shape[0]
    max_entry = torch.max(inputs, dim=-1, keepdim=True)
    inputs_submax = inputs - max_entry.values
    inputs_exp = torch.exp(inputs_submax)
    inputs_sum = torch.sum(inputs_exp, dim=-1)
    result = torch.log(inputs_sum) - inputs_submax[torch.arange(batch_size, device=inputs.device), targets]
    # -log(exp(target) / sum_exp(i)) = log(sum_exp(i)) - target
    # 如果 inputs_sum 选择了保留维度，那么相减的时候，就是 batch_size, 1 与 batch_size 进行相减
    # 会被 pytorch 广播为 batch_size, batch_size 和 batch_size, batch_size 
    # 前者在第二个维度的数字被广播了 batch_size 遍，后者也是每个数字广播这么多
    # 因此做差后结果是 batch_size, batch_size 但是第二个维度上每个向量内容都一样
    # 取平均值，最后又变成了 所有内容相加除以总数，对于向量内部 n 个内容相加又除以 n 因此不变，最后结果一样
    # 但是却增大了计算量
    result = result.mean()

    return result

@nvtx.range("main process")
def main(args):
    vocab_size = args.vocab_size
    context_length = args.context_length
    d_model = args.d_model
    batch_size = args.batch_size
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_ff = args.d_ff
    rope_theta = args.rope_theta
    weight_decay = args.weight_decay
    [beta1, beta2] = args.betas
    eps =  args.eps
    max_learning_rate = args.max_learning_rate
    device = args.device
    torch.manual_seed(args.seed)
    if  torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    warmup_steps = args.warmup_steps
    profiling_steps = args.profiling_steps
    profile_attn = False if args.profile_attn == 0 else True

    with nvtx.range("define model"):
        transformer_model = BasicsTransformerLM(vocab_size,
                                            context_length,
                                            d_model, num_layers,
                                            num_heads, d_ff,
                                            rope_theta, profile_attn)
        transformer_model.to(device)
    
        optimizer = AdamW(transformer_model.parameters(), args.max_learning_rate)

    log_file = open(os.path.join(os.path.dirname(__file__), "profiling_results/profiling.jsonl"), "a")
    log_file.write(json.dumps({**vars(args)}) + "\n")
    log_file.flush()

    use_bf16 = True if args.use_mixed_precision == 1 else False

    if use_bf16:
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()
    # use nullcontext manager to determine whether use mixed_precision

    
    for t in range(0, args.end_iter):
        nvtx.range_push(f"training_step: {t}")
        step = t + 1
        lr = get_cosine_lr(step, max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = generate_random_data(batch_size, context_length, vocab_size, device)
        
        # forward pass process
        time_start = timeit.default_timer()
        with nvtx.range("forward_pass"):
            logits = transformer_model(x)
        torch.cuda.synchronize()  
        time_after_forward = timeit.default_timer()
        forward_time = time_after_forward - time_start

        # Only forward and loss in the ctx context
        with ctx:
            with nvtx.range("loss_compute"):
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            torch.cuda.synchronize()
            time_after_loss = timeit.default_timer()
            loss_time = time_after_loss - time_after_forward
            # loss = cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            # view(-1, logits.size(-1)) 前一个 -1 表示由 pytorch 自行推断维度，最后一个表示最后一维度的数量
            # 将 batch_size, seq_len, vocab_size 的结果转变为 batch_size * seq_len, vocab_size 的维度
            
            # backward pass process

        with nvtx.range("backward_pass"):
            loss.backward()
        torch.cuda.synchronize()  
        time_after_backward = timeit.default_timer()
        backward_time = time_after_backward - time_after_forward

        # optimizer process
        with nvtx.range("optimizer"):
            optimizer.step()
        torch.cuda.synchronize()  
        time_after_step = timeit.default_timer()
        step_time = time_after_step - time_after_backward
        optimizer.zero_grad()
        torch.cuda.synchronize()  
        # Wait for all kernels in all streams on a CUDA device to complete.
        # In this case, the time we measured is the time of forward plus backward and optimizer update
        nvtx.range_pop()

        # profiling control
        if step > warmup_steps + profiling_steps:
            exit(0)

        if step > warmup_steps or args.profiling_warmup == 1:
            log_file.write(json.dumps({"step": step, "forward_time": forward_time, 
                                    "backward_time": backward_time, "step_time": step_time,
                                    "loss_compute_time": loss_time},) + "\n")
            log_file.flush()


if __name__ == "__main__": main(parse_args())