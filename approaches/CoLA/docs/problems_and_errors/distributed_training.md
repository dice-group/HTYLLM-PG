Distributed Training Errors (CoLA / HydraLoRA)

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

Missing info to finalize
- Which job script currently applies the master-wait logic for multi-node runs.
