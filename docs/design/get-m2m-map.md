# How get_m2m_map works

![get_m2m_map — tensor redistribution](etha_m2m_map.png)

`get_m2m_map` (called the "P2P Map" in the source) computes, for a DTensor
redistributed from a source `(mesh, placement)` to a target `(mesh, placement)`,
the communication map telling which target rank / index every source-shard
element must reach.

> Running example: source shard shape = 4×1 (rows cut into 4, columns into 1),
> target shard shape = 2×2 (rows into 2, columns into 2).

## 1. Align granularity (lcm)

```
   source shard = 4×1 (rows×4, cols×1)   target shard = 2×2 (rows×2, cols×2)
   ┌───┐                                 ┌───┬───┐
   │ 0 │  one shard per row              │ 0 │ 1 │  one shard per block
   ├───┤                                 ├───┼───┤
   │ 1 │                                 │ 2 │ 3 │
   ├───┤                                 └───┴───┘
   │ 2 │
   ├───┤
   │ 3 │
   └───┘

   per-dim lcm aligns the two slicings:
   middle = ( lcm(4,2), lcm(1,2) ) = 4×2   ← finest grid both sides can divide

   ┌────┬────┐
   │    │    │
   ├────┼────┤
   │    │    │     middle tensor  (4 rows × 2 cols = 8 cells)
   ├────┼────┤
   │    │    │
   ├────┼────┤
   │    │    │
   └────┴────┘
```

## 2. Distribute + tag

```
   middle is split by "source layout 4×1" across 4 ranks; each rank holds 1 row
   (1×2), and every cell is stamped with a unique id.

   src rank0 ─► │r0.0│r0.1│      encoding (base-N positional):
   src rank1 ─► │r1.0│r1.1│        base = max(middle_shape) + 1 = 5
   src rank2 ─► │r2.0│r2.1│        enc  = rank
   src rank3 ─► │r3.0│r3.1│        for c in local coords:  enc = enc * base + c
                                        = rank·baseⁿ + c₀·baseⁿ⁻¹ + … + c_{n-1}
   label r{rank}.{local col} = (who · which cell)
   here n=2, base=5, local coord (c₀,c₁) with row c₀=0:  enc = rank·25 + c₀·5 + c₁
```

## 3. full_tensor (gather) → 4. send

```
   each rank's row-shard is gathered back into the whole 4×2 tensor, then sent

   │r0.0│r0.1│ ┐         ┌────┬────┐
   │r1.0│r1.1│ │         │r0.0│r0.1│
   │r2.0│r2.1│ ├gather►  ├────┼────┤
   │r3.0│r3.1│ ┘         │r1.0│r1.1│
                         ├────┼────┤        📦 ╾╾╾╾▶ 📥
                         │r2.0│r2.1│        transfer once
                         ├────┼────┤
                         │r3.0│r3.1│
                         └────┴────┘
```

## 5. Redistribute + decode

```
   receive the whole tensor, re-cut by "target layout 2×2" → 4 blocks (2×1 each),
   cut lines ≠ source

   ┌────┰────┐
   │r0.0┃r0.1│    tgt rank0 (top-left) │ tgt rank1 (top-right)
   │r1.0┃r1.1│
   ┝━━━━╋━━━━┥  ← target cut lines (1 row cut + 1 col cut)
   │r2.0┃r2.1│    tgt rank2 (bot-left) │ tgt rank3 (bot-right)
   │r3.0┃r3.1│
   └────┸────┘
        🔍 take own block, read the label to trace its source

   decode (inverse of encode):  temp = enc
                                for _ in range(n):  coord = temp % base; temp //= base   (coords reversed)
                                reverse coords,  remaining temp = source rank
   e.g. r3.1 → enc = 3·25 + 0·5 + 1 = 76
        76 % 5 = 1, 76 // 5 = 15  │  15 % 5 = 0, 15 // 5 = 3
        → coord (0,1), source rank = 3  →  "source rank3 local (0,1)"  → write to map
```

## 6. all-gather → map

```
   🌍 all_gather → every rank gets the full m2m_map → returns map + slicers
   (each target block is 2×1, so the 2nd coord of every target idx is always 0)

   source rank·idx       target rank·idx
   ───────────────       ───────────────
   0 · (0,0)   ──▶       0 · (0,0)
   0 · (0,1)   ──▶       1 · (0,0)
   1 · (0,0)   ──▶       0 · (1,0)
   1 · (0,1)   ──▶       1 · (1,0)
   2 · (0,0)   ──▶       2 · (0,0)
   2 · (0,1)   ──▶       3 · (0,0)
   3 · (0,0)   ──▶       2 · (1,0)
   3 · (0,1)   ──▶       3 · (1,0)
```

## The pipeline in one line

```
   distribute         full_tensor       send / recv     redistribute       decode         all-gather
   4×1 cut 4 rows ──►  gather to 4×2 ──►  transfer ──►   re-cut by 2×2 ──► read labels ──► 🗺️ global map
   · tag
```

In one sentence: **the shapes don't line up (source 4×1, target 2×2), so cut to
the finest common grid 4×2 and stamp every cell with a "who am I" id;
`full_tensor` gathers the shards back into the whole tensor and sends it once,
the target side `distribute_tensor` re-cuts it by its own 2×2 layout and reads
the stamps to trace origins — reconstructing the full source→target routing map.**
