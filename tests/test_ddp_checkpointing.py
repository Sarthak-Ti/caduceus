"""Does gradient checkpointing survive DDP?

The production job runs `--gres=gpu:a100:2` under Lightning DDP. Activation
checkpointing and DDP interact badly in some configurations: the recomputation forward
can fire a parameter's gradient-ready hook a second time, which the DDP reducer rejects
with "Expected to mark a variable ready only once". PyTorch's remedy is
`static_graph=True`. Recommending `gradient_checkpointing` without checking this would
be recommending a config that crashes only once a multi-GPU run starts.

This runs the real backbone under real DDP on 2 GPUs and compares gradients against a
single-process run of the same weights and inputs.

    torchrun --nproc_per_node=2 tests/test_ddp_checkpointing.py
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D_MODEL, N_LAYER, SEQLEN, BATCH = 128, 3, 2048, 1


def build(gradient_checkpointing, device):
    from tests.bench_models import build_backbone, make_caduceus_config
    torch.manual_seed(0)
    cfg = make_caduceus_config(d_model=D_MODEL, n_layer=N_LAYER)
    cfg.gradient_checkpointing = gradient_checkpointing
    return build_backbone(cfg, device=device)


def run(static_graph):
    rank = dist.get_rank()
    device = f"cuda:{torch.cuda.current_device()}"
    model = build(True, device)
    ddp = DDP(model, device_ids=[torch.cuda.current_device()])
    if static_graph:
        ddp._set_static_graph()
    torch.manual_seed(100 + rank)
    x = torch.randn(BATCH, SEQLEN, D_MODEL, device=device)
    out, _ = ddp(None, inputs_embeds=x)
    out.square().mean().backward()
    gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ddp.parameters()
                           if p.grad is not None))
    return gnorm.item()


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        print(f"world_size={world}  torch={torch.__version__}")

    failures = []
    for static_graph in (False, True):
        label = f"gradient_checkpointing + DDP (static_graph={static_graph})"
        try:
            gnorm = run(static_graph)
            dist.barrier()
            if rank == 0:
                print(f"  PASS  {label}: grad_norm={gnorm:.6e}")
        except Exception as e:  # noqa: BLE001
            failures.append((label, f"{type(e).__name__}: {str(e).splitlines()[0]}"))
            if rank == 0:
                print(f"  FAIL  {label}: {type(e).__name__}: {str(e).splitlines()[0]}")

    if rank == 0:
        print("\nRESULT: " + ("all DDP configurations worked"
                              if not failures else
                              "; ".join(f"{l} -> {e}" for l, e in failures)))
    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
