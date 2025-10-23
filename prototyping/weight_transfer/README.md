# Weight Transfer Between Individual Processes

This prototype explores how to ferry parameters between a training
engine and an inference engine by inserting a lightweight middleware
layer. The data path combines an LMDB-backed command queue for
per-rank communication between engines and their middleware counterpart,
a `torch.distributed.TCPStore` for coordination across middleware ranks,
and NCCL `send`/`recv` calls for the actual tensor transfers.

## Components

- `train.py`  
  Simulates a trainer that updates a single-value tensor every optimizer
  step and announces when new weights are ready.

- `inference.py`  
  Simulates an inference worker that pauses on demand, waits for fresh
  weights, and resumes once the transfer finishes.

- `middleware.py`  
  One process per rank that mediates between the other two worlds: it
  watches the command queues, synchronizes via `TCPStore`, and performs
  the NCCL `send`/`recv` exchanges.

## Launch Topology

| Role        | GPUs             | Command |
|-------------|------------------|---------|
| Middleware  | 0‒7              | `pixi run torchrun --nproc-per-node=8 --master-port=29500 prototyping/weight_transfer/middleware.py` |
| Training    | 0‒3              | `pixi run torchrun --nproc-per-node=4 --master-port=29501 prototyping/weight_transfer/train.py` |
| Inference   | 4‒7              | `pixi run torchrun --nproc-per-node=4 --master-port=29502 prototyping/weight_transfer/inference.py` |

The helper script `launch_all.sh` orchestrates the three commands above
and cleans the LMDB queues before bootstrapping. GPU affinity is handled
inside each entry point by reading `LOCAL_RANK` and mapping it to the
correct physical device.

## Communication Flow

1. **Registration**  
   Each engine serializes its parameter tensor with `ForkingPickler`,
   stores the payload in LMDB, and enqueues a `RegisterTensor` message so
   the matching middleware rank can attach to the tensor’s CUDA storage.

2. **Training loop**  
   The trainer bumps the tensor (`+1.0`) during every optimizer step.
   Once a fresh value is ready it enqueues `Ready`, signalling the
   middleware that a transfer may begin, and then waits on a per-rank
   POSIX semaphore that will be released when the transfer finishes.

3. **Middleware orchestration**  
   - The sending middleware rank requests the peer rank to pause
     inference by updating keys in `TCPStore`.
   - The receiving middleware rank relays a `Stop_Inference` command via
     its queue, after which the inference engine enqueues `Ready` and
     blocks on its semaphore until the transfer completes.

4. **Weight transfer**  
   With both sides ready, the paired middleware ranks call
   `torch.distributed.send`/`recv` to move the tensor directly between
   GPUs.

5. **Resume inference**  
   After the transfer completes the middleware releases the per-rank
   semaphores for both the trainer and inference worker. Once their
   respective `acquire()` calls return, each engine resumes work and the
   inference side prints the updated value. Seeing the number increase by
   one confirms the round-trip succeeded.

## Notes

- The LMDB-backed queues persist between runs. If you terminate
  processes abruptly, consider clearing `prototyping/weight_transfer/dbs`
  before relaunching to avoid replaying stale messages.
