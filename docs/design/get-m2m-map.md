# get_m2m_map: mark and recapture

![get_m2m_map — tensor redistribution](etha_m2m_map.png)

`get_m2m_map` computes, for a weight resharded from a source `(mesh, placements)`
to a target `(mesh, placements)`, the routing map telling which target rank /
cell every source cell must reach.

The algorithm is mark-recapture, as in field ecology: tag every animal once,
capture the population twice, and join the two capture records by tag to learn
who moved where. Here the animals are cells of a common grid, the tag is a
global cell id (gid), and the two captures are the source and target shardings.

> Running example: source shard counts = 4×1 (rows cut into 4), target shard
> counts = 2×2 (rows into 2, columns into 2).

## Mark: the middle grid

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
   middle = ( lcm(4,2), lcm(1,2) ) = 4×2   ← finest grid both sides divide evenly

   ┌────┬────┐
   │ 0  │ 1  │
   ├────┼────┤
   │ 2  │ 3  │     middle grid, one gid per cell (row-major arange)
   ├────┼────┤
   │ 4  │ 5  │
   ├────┼────┤
   │ 6  │ 7  │
   └────┴────┘
```

The lcm construction guarantees every division below is exact — no uneven
shards, ever.

## Capture twice, locally

For each side, every rank's share of the middle grid is a closed-form box:
torch's own sharding geometry (`_compute_local_shape_and_global_offset`, the
same code path DCP uses for checkpoint resharding) maps any `(mesh coordinate,
placements)` to `(box shape, box offset)` — including `_StridedShard`, the
FSDP2/EP layouts. Enumerating the box yields the rank's `{gid: cell}` record:

```
   source capture (4×1):          target capture (2×2):
   rank0 holds gids {0, 1}        rank0 holds {0, 2}   rank1 holds {1, 3}
   rank1 holds gids {2, 3}        rank2 holds {4, 6}   rank3 holds {5, 7}
   rank2 holds gids {4, 5}
   rank3 holds gids {6, 7}
```

No tensor is touched and nothing is communicated: both captures are pure
arithmetic over the declared shardings, so any rank — or a driver — computes
the identical records. Plans for 512-rank meshes take milliseconds.

## Join by gid

For every gid, the source holders and the target wanters meet:

```
   gid   src holder · cell      tgt wanters · cell
   ───   ─────────────────      ──────────────────
    0       0 · (0,0)    ──▶      0 · (0,0)
    1       0 · (0,1)    ──▶      1 · (0,0)
    2       1 · (0,0)    ──▶      0 · (1,0)
    3       1 · (0,1)    ──▶      1 · (1,0)
    ...
```

When the source side replicates, a gid has several holders; every rank picks
the same one deterministically — holders are sorted and indexed by the
destination's position modulo the holder count, which also spreads multiple
destinations across replicas. The joined records, sorted by (src rank, cell),
are the canonical route list every rank derives identically.

## Correctness

Two guarantees, one watchdog:

- **Exact division by construction**: the lcm middle grid divides evenly under
  both shardings — no uneven shards exist anywhere in the pipeline.
- **Geometry from the source of truth**: boxes come from torch's own sharding
  geometry, the code path DCP relies on for checkpoint resharding; its
  consistency with how tensors are actually sharded is an invariant torch
  itself maintains.
- **A live experiment in CI**: the fuzz test re-runs real mark-recapture on
  random mesh/placement pairs every run (fresh seed each time, printed on
  failure) — a gid-tagged tensor is actually sharded with
  `distribute_tensor` on both sides and the transferred result must match.
  Any drift between the plan and torch's actual sharding fails here first.

## What the boxes cannot say

A box is one contiguous rectangle, so:

- `Partial` is rejected — weights are never partial sums;
- `_StridedShard` is assumed in its canonical FSDP2 form: `split_factor`
  equals the product of same-dim shard counts on **later mesh dims** — the
  ones physically applied **before** it (that inversion of placement order
  vs application order is what "strided" means). FSDP2 always fills it this
  way; a non-canonical value would mean a non-contiguous footprint, outside
  the box model.
