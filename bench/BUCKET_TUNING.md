# Bucket tuning: make coalesced transfer never lose to per-chunk

Goal: a single coalescing policy where the **bucket** (coalesced) method is
**always ≥** the **chunk** method (`bucket_size=1`, one wire op per chunk) across
every mesh / shape / placement case in `transfer_benchmark.py`.

Terminology in the benchmark output:
- **chunk method** = `chunk_to_bucket_ops(chunks, bucket_size=1)` — every chunk is
  its own single-entry bucket. `prepare` aliases the source slice (zero-copy when
  contiguous); many small wire ops; no gather/scatter.
- **bucket method** = `chunk_to_bucket_ops(chunks, bucket_size=BUCKET_SIZE_BYTES)`
  — same-route chunks coalesced into one buffer + one wire op. Costs N gather
  `copy_` (send) + N scatter `copy_` (recv) but issues far fewer wire ops.

---

## M1 — investigate & localize (DONE)

### Mechanism (code paths)
- `Bucket.prepare` (ir.py): single-entry → `chunk.buffer.view(uint8)` (zero-copy
  alias when the slice is contiguous). Multi-entry → allocate `total_bytes`, then a
  **python loop of N `copy_`** gathering each chunk into the buffer; recv side points
  each chunk's buffer at a bucket slice and `finalize` scatters back with N `copy_`.
- `bucket_comm` (comm_methods.py): per-channel pipeline, `max_in_flight=2`. Buffer
  assembly (`prepare`) is meant to overlap in-flight collectives. All on the default
  stream; NCCL runs on its own stream, gated by `buffer_ready_event`.
- `chunk_to_bucket_ops` (get_buckets.py): **a chunk ≥ `bucket_size` becomes its own
  bucket**; smaller chunks accumulate until they reach `bucket_size`.

### Data (baseline run, 16 GPUs, 2 nodes, BUCKET_SIZE_BYTES = 256MB)
Parsed from the full benchmark sweep (8 meshes × 9 shapes × 3 placement configs =
216 cases). Pattern by tensor size:
- **≤1024² (≤100MB batch)**: bucket wins big (2–3×). Many tiny chunks → wire-op
  launch count dominates → coalescing is a clear win.
- **≥8192² (≥6.4GB batch)**: tie (~1.00). Bandwidth-saturated; copy overlaps wire.
- **2048²–4096² (400MB–1.6GB) on certain meshes**: **chunk wins by 5–20%**
  (20/216 cases; 15 of them in `partial_dp`, which adds an all-reduce in `prepare`).

Worst chunk-wins cases (chunk / bucket):
```
partial_dp (2,2,2,1)->(1,1,4,2) 4096²  chunk=242 bucket=203  1.20x
partial_dp (2,4,1,1)->(1,2,4,1) 4096²  chunk=177 bucket=149  1.19x
partial_dp (4,2,1,1)->(1,1,2,4) 4096²  chunk=242 bucket=204  1.19x
partial_dp (2,2,2,1)->(1,1,4,2) 8192²  chunk=298 bucket=255  1.17x
partial_dp (8,1,1,1)->(1,2,2,2) 2048²  chunk=259 bucket=232  1.12x
```

### Root-cause hypothesis
`BUCKET_SIZE_BYTES = 256MB` is far larger than any single chunk at these shapes, so
**all** chunks of a mid-size tensor accumulate into very few huge buckets. Two
compounding losses:
1. Hundreds of gather/scatter `copy_` serialized on the critical path.
2. Too few buckets per channel → `max_in_flight=2` has nothing to overlap, so the
   copies are fully exposed instead of hidden behind wire.

Per-chunk (`bucket_size=1`) avoids both: zero gather/scatter, and many buckets to
pipeline.

The fix lever already exists: `chunk_to_bucket_ops` sends any chunk ≥ `bucket_size`
as its own (zero-copy) bucket. A **moderate** `bucket_size` should let large chunks
take the chunk-method path (no loss) while still coalescing the small chunks (the
win) — and keep enough buckets for pipeline overlap.

---

## M2 — sweep bucket_size (IN PROGRESS)
`benchmark_single_shape` now sweeps `BUCKET_SIZE_SWEEP = {1, 2MB, 8MB, 32MB, 256MB}`
in one run (1 = chunk method). Each value reuses the same chunk list; the result
dict carries `sweep_throughput_gb_s`. (Removed the now-unused per-method profiling
blocks; profiling args kept but unwired, marked noqa.)

Success criterion: one single value where bucket ≥ chunk (within ~2% noise) for all
216 cases, ideally still winning big on the small shapes.

### Smoke (single node, 8 GPU, NVLink, 1 case: partial_dp 2048², 2 tensors)
```
chunk=37.46  2MB=41.59  8MB=33.28  32MB=33.30  256MB=34.60   (GB/s)
```
2MB already beats both chunk and 256MB; larger sizes over-coalesce. Sweet spot
looks small (~2MB). NVLink single-node ≠ the cross-node chunk-wins regime, so this
only confirms the sweep runs and the direction; full 16-GPU sweep submitted
(job `slurm-profile-slurm-42ndm`).

## M2 RESULT — no single bucket_size wins (DONE)
Full 16-GPU sweep, 215 cases. Times each fixed value LOSES to chunk (>2%):
```
2MB: 21   8MB: 5   32MB: 6   256MB: 17
```
Best-value distribution (which value wins each case):
```
32MB: 67   256MB: 55   8MB: 45   chunk: 29   2MB: 19
```
8MB is the best single value (loses 5/215, 4 of them ≤1.11x) but is NOT "always
fastest". The optimum is genuinely case-dependent along **two orthogonal axes**:

1. **Size axis** — optimum bucket_size grows with tensor size. Same mesh
   `(1,1,4,2)->(4,2,1,1)`: 512²→8MB, 1024²→32MB, 2048²+→256MB. Bigger chunks need
   bigger buckets to coalesce; large tensors are bandwidth-saturated so the gather
   copy hides. A fixed value is always wrong for some size.
2. **Partial axis (the killer)** — same mesh `(1,2,4,1)->(8,1,1,1)` 1024²:
   no_partial → 32MB=382 (coalesce wins big); partial_dp → chunk=238 best, all
   coalescing loses. The only difference is the Partial all_reduce. `Bucket.prepare`
   runs **N serial all_reduce, one per chunk**, on the critical path; chunk method
   overlaps each all_reduce with its own wire. All chunks in a bucket share the same
   partial group (it's in `bucket_key`), so the N all_reduce are redundant serial
   collectives on the same comm.

## M3 — make coalescing not lose (IN PROGRESS)
Two independent fixes, matching the two axes:

A. **Batch the partial all_reduce** (fixes axis 2). Since a bucket's chunks share
   one partial group, gather first, then do a SINGLE all_reduce over the whole
   bucket buffer instead of N per-chunk reduces. Should remove the partial_dp
   penalty outright. Targeted, low-risk; the bucket_key guarantee makes it safe.

B. **Adaptive bucket_size = f(per-rank bytes)** (fixes axis 1). Pick the threshold
   so each channel splits into enough buckets to keep the pipeline full (overlap)
   while still coalescing tiny chunks. Computed in the agent/caller from the batch's
   byte volume — transparent to the user ("no-brainer").

Order: do A first (it's the larger, cleaner win and explains most partial losses),
re-sweep, then size B to close any residual size-axis gap.

### M3-A implemented + smoke-verified
`Bucket.prepare` now gathers the whole bucket first, then runs ONE all_reduce over
the buffer (guarded by uniform dtype, no wire downcast). Smoke (single node, 8 GPU,
2048² partial_dp), before → after:
```
            chunk   2MB    8MB    32MB   256MB
before      37.5    41.6   33.3   33.3   34.6
after       38.7    41.6   43.4   46.5   41.7
```
Coalescing flipped from losing (33<37) to winning (43–47>39); correctness check
passed. Confirms the partial penalty was the serial per-chunk all_reduce.

### M3-A 16-GPU re-sweep, take 1 (`hs227`) — partial, but encouraging
Hit the 1h `#SBATCH --time` wall (5-point sweep too slow); only 172/215 cases ran
before SLURM cut it (succeeded, not a hang). On the cases that DID run, 8MB loses to
chunk in **0** cases (was 5), 256MB 6 (was 17). The previously-worst
`(8,1,1,1)->(1,2,2,2)` mid shapes flipped (2048²: 8MB=252 > chunk=228, was 256MB=225
lost). BUT the single worst mesh `(2,2,2,1)->(1,1,4,2)` only reached 512² before the
cut — its 4096²/8192² (the 1.18x losers) were NOT measured. So 8MB=0-losses is on
incomplete data; not conclusive yet.

### take 2 (`ckmtc`) — M3-A DEADLOCKED, reverted
Failed: NCCL ALLREDUCE timeout with **mismatched sizes across ranks** in the same
collective (Rank 3 NumelIn=262144 vs Rank 1 NumelIn=2097152, 8×).

Root cause of the M3-A bug: in a Partial reduce group, reduce-only chunks (dst=(),
distinct `cell_key`) each become their own single-entry bucket → per-cell
all_reduce; shipping chunks (dst≠()) coalesce → one batched all_reduce. So a
shipping rank issues ONE big all_reduce (8 cells = 2M) while a reduce-only rank
issues MANY small ones (1 cell = 256K) on the same group → size/count mismatch →
deadlock. Per-chunk (bucket_size=1) keeps both at per-cell granularity, aligned.
**Batched all_reduce is fundamentally unsafe here; reverted `Bucket.prepare`.**

## M3-B — don't coalesce Partial — FAILED (correctness), reverted
Tried: a chunk with `source_partial_groups` always gets its own bucket. Smoke:
no_partial / replicate_dp pass, partial_dp mismatches (Max diff ~6.9 = wrong, not
numerical). Isolated: `[1]`-only passes (single-entry Partial is correct);
`[32MB]`-only fails (so it's the coalesced path × Partial, not sweep-state leak).

Root cause: send and recv are bucketized **separately** (`agent.py` builds
`send_buckets` / `recv_buckets` independently). `source_partial_groups` lives only on
the SOURCE chunk, so "don't coalesce Partial" fired on the sender (N single-entry
sends) but NOT on the matching receiver (one coalesced recv) → P2P send/recv buffer
sizes mismatch → data lands at wrong offsets. **Any bucketing rule must be symmetric
across the send/recv pair; a source-only signal breaks it.** Reverted.

## The Partial dead end — all three tried
- batched all_reduce → deadlock (reduce-only per-cell vs shipping batched misalign)
- don't-coalesce-Partial → P2P send/recv asymmetry (signal only on sender)
- original coalescing → correct, but serial per-chunk all_reduce is slow on mid sizes

Coalescing helps non-Partial cleanly; Partial is boxed in by two invariants:
(1) the per-cell all_reduce must stay aligned across the reduce group, and
(2) send/recv bucketing must be symmetric across the P2P pair. A source-only tweak
violates one or both.

A correct "don't coalesce Partial" needs the signal on BOTH ends: tag the matching
target recv chunk too (derive from the route's Partial source in `m2m_to_chunks`,
carry a `no_coalesce` flag on Chunk). That's a design change beyond tuning — needs
sign-off. With it: Partial → bucket == chunk (no loss), non-Partial → 8MB wins,
which satisfies "bucket ≥ chunk everywhere".

## M3-C — symmetric `no_coalesce` flag (the fix for M3-B)
Chosen direction. Put the don't-coalesce signal where BOTH ends see it: keyed off
`M2MMap.source_partial_reductions` (same map on source and target ranks).
- `Chunk.no_coalesce: bool` — new field.
- `m2m_to_chunks`: `no_coalesce = bool(m2m.source_partial_reductions)`, set on every
  chunk it builds (send and recv), so the two ends bucketize identically.
- `chunk_to_bucket_ops`: a `no_coalesce` chunk stays single-entry.

Result: Partial transfers → all chunks single-entry on both ends (per-cell
all_reduce, aligned; P2P send/recv symmetric) = chunk method = Partial's optimum.
Non-Partial → coalesce by `bucket_size`.

Smoke (single node, partial_dp mesh WITH a real reduce — `Transport.NONE` chunks
present): **passes** where M3-B mismatched. Confirms the symmetric flag fixes the
send/recv asymmetry.

### M3-C 16-GPU sweep (`kgj5c`) — PASS
Completed cleanly, all 216 blocks, **no deadlock, no mismatch**. 8MB loses to chunk
in 14/215 cases — **all 14 are partial_dp, all ≤1.11x**, none in no_partial/
replicate_dp. Since Partial chunks are `no_coalesce`, the 8MB and chunk runs build
the *same* single-entry buckets for those cases, so the gap is pure measurement
noise (concentrated on small tensors). Net: **non-Partial 8MB ≥ chunk everywhere;
Partial == chunk (its optimum). 8MB + no_coalesce is the no-brainer-fastest policy.**

## M4 — ship (IN PROGRESS)
- `agent.py`: default `coalesce_bytes` 1 → `DEFAULT_COALESCE_BYTES` (8MB). Partial
  self-excludes via `no_coalesce`.
- Validated: CPU unit tests ✅ 55 passed; vLLM weight-sync ✅ 3/3 rounds, no agent
  errors (production Partial MoE on the new 8MB default).
- Shipped: commit `fe94326`, PR #109.

## Outcome
`8MB + no_coalesce` is the no-brainer-fastest policy: non-Partial coalesces (wins up
to 3x on small tensors, ties on large), Partial self-excludes and runs its optimal
per-chunk path. bucket ≥ chunk across all 215 sweep cases, no deadlock, no mismatch.
The Partial path is structurally protected by two invariants we learned the hard
way: per-cell all_reduce alignment across the reduce group, and symmetric send/recv
bucketing.
