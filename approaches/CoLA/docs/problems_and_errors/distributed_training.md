Distributed Training Errors (CoLA / HydraLoRA)

Short introduction
This document captures the multi-node rendezvous failures seen when launching CoLA/HydraLoRA training on the otus cluster (SLURM + accelerate + torch.distributed). The failure happens before training starts and is caused by master/worker connection issues. The goal is to provide a checklist to make the rendezvous deterministic and easy to debug.

Symptoms
- `DistStoreError: wait timeout after 900000ms` during rendezvous.
- `c10d` socket `waitForInput` timeouts; ranks never join.
- `torch.distributed.elastic` fails before training starts.

Observed root cause
- Workers try to connect before master is listening or `MASTER_ADDR` is not reachable/resolves incorrectly (IPv6-only or wrong hostname).

Cluster specifics (otus)
- GPU partition, H100 nodes, 4 GPUs per node.
- Hostname resolution works via `getent ahosts`.
- Multi-node runs need a stable master host/IP across nodes.

Required setup (working)
- MASTER_HOST from `scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1`.
- MASTER_ADDR forced to IPv4 via `getent ahosts "$MASTER_HOST"`.
- Non-zero nodes wait for master socket before launching `accelerate`.
- Ensure all nodes use the same `MASTER_ADDR` and `MASTER_PORT`.
- Avoid per-node `hostname -f` as master; always use the first node in `SLURM_JOB_NODELIST`.

Validated debug commands
- Host resolution:
  srun -p gpu --gres=gpu:h100:1 -t 5:00 --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    bash -c 'MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1); echo host=$HOSTNAME master=$MASTER; getent ahosts "$MASTER" | head -n 3'
- TCP reachability (works with a short sleep on clients):
  srun -p gpu --gres=gpu:h100:1 -t 5:00 --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -c '
  MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  PORT=23248
  if [ "$SLURM_NODEID" -eq 0 ]; then
    python3 - <<PY
import socket
s=socket.socket(); s.bind(("0.0.0.0", 23248)); s.listen(8)
for _ in range(3):
    c,_=s.accept(); c.send(b"ok"); c.close()
PY
  else
    sleep 2
    python3 - <<PY
import socket; s=socket.socket(); s.settimeout(5); s.connect(("$MASTER", 23248)); s.send(b"hi"); print("OK")
PY
  fi'

Notes
- This is a startup race; explicit wait_for_master is required for multi-node runs.
- `DIST_DEBUG=1` helps log master/addr/port and `getent` results.
- Prefer IPv4 addresses because some nodes resolve hostnames to IPv6 first.
- A 1–2 second delay on non-zero nodes avoids the race.

Additional checks (run if still failing)
- Confirm master port is reachable from all nodes:
  srun -p gpu --gres=gpu:h100:1 -t 5:00 --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -c '
  MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  PORT=29501
  if [ "$SLURM_NODEID" -eq 0 ]; then
    python3 - <<PY
import socket
s=socket.socket(); s.bind(("0.0.0.0", 29501)); s.listen(8)
for _ in range(3):
    c,_=s.accept(); c.send(b"ok"); c.close()
PY
  else
    sleep 2
    python3 - <<PY
import socket; s=socket.socket(); s.settimeout(5); s.connect(("$MASTER", 29501)); s.send(b"hi"); print("OK")
PY
  fi'
- Verify basic torch.distributed init (Gloo):
  srun -p gpu --gres=gpu:h100:4 -t 5:00 --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    bash -c 'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); \
             export MASTER_PORT=29503; export WORLD_SIZE=$SLURM_NTASKS; export RANK=$SLURM_PROCID; \
             python - <<PY
import torch.distributed as dist
dist.init_process_group("gloo", init_method="env://")
print("rank", dist.get_rank(), "ok")
dist.destroy_process_group()
PY'
- Verify NCCL init:
  srun -p gpu --gres=gpu:h100:4 -t 5:00 --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    bash -c 'MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1); PORT=29504; \
             export NCCL_DEBUG=INFO; torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 \
             --node_rank=$SLURM_NODEID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$PORT - <<PY
import torch.distributed as dist
dist.init_process_group("nccl")
print("nccl rank", dist.get_rank(), "ok")
dist.destroy_process_group()
PY'

Missing info to finalize
- Which job script currently applies the master-wait logic for multi-node runs.
