# Accessing TensorBoard on VM

## Steps

1. **SSH to VM with port forwarding:**
   ```bash
   ssh -J mounzer@sshgate.cs.upb.de -L 60066:localhost:60066 mounzer@lola.cs.uni-paderborn.de
   ```

2. **Start TensorBoard on VM:**
   ```bash
   tensorboard --logdir checkpoints/pretrain_run_vm --host 0.0.0.0 --port 60066
   ```

3. **Open in browser:**
   Go to: http://localhost:60066/

## Notes
- Keep the SSH session open while using TensorBoard
- The tunnel forwards port 60066 from VM to your local machine 

conda activate htyllm-env
cd HTYLLM-PG

ssh mounzer@sshgate.cs.upb.de
ssh mounzer@lola.cs.uni-paderborn.de

