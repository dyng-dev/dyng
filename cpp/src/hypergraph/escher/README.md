# The ESCHER CBST store (`dyng::detail::escher`)

The storage of `hypergraph_storage::escher`: one **CBST** (complete binary search tree) per
adjacency of a dynamic hypergraph (hyperedge to vertices H2V, vertex to hyperedges V2H, the line
graph H2H, the vertex graph V2V). It is internal: no public header exposes a CBST key, offset,
`+1` shift or sentinel (PLAN 4.3, 4.4.2; the test `cbst_source_rules` checks it). This page is for
contributors who change the store. The port and its rules are ADR 0038; the trace replay that
proves it is ADR 0036 / 0037 and `parity/results/M8a.md`.

## Where the code comes from

| Code | Origin |
|---|---|
| `cbst_store.{hpp,cu}`, `build_tree.cu`, `payload.cu`, `insert_reuse.cu`, `delete_avail.cu`, `unfill.cu`, `find.cu`, `kernels.cuh`, `device_utils.cuh`, `row_view.cuh`, `locator.cu`, `integrity.{hpp,cu}`, `flatten.{hpp,cpp}` | ESCHER-GPU@abcf9b2 (A), transplanted in commit c1 (`parity/escher_port/transplant.py` regenerates it) and changed since by one class of mechanical change per commit (errors, memory, streams, registration and counters) |
| the empty tree, `clone()` and `rewire_tree_kernel`, `download()`, `allocated_bytes()`, poisoning, `launch_build_layout` / `launch_extract_keys_from_positions`, `store_policy` (A's numbers) | dynG (M8a step 3) |
| MOSP_ESCHER@4b86159's (B) behaviours: construct validation, the fill clamps, caller row counts, the bounded chain walker (`chain.cuh`), its integrity checks, compaction (`compact.cu`), the presets `mosp_escher()` and dynG's default | M8a steps 4-5 |

ESCHER-GPU's `include/binning.hpp` and `utils/binning.cpp` are dead code (no object of its
Makefile builds them) and are not ported; the degree binning of the store is `degree_bins` in
`cbst_store.cu` (A's `DegreeBins`).

**Rename map.** Every identifier that the port renamed is listed, original to dynG, in
`parity/escher_port/rename_map.toml`; the highlights:

| ESCHER-GPU | dynG |
|---|---|
| `CBSTOperations`, `CBSTContext`, `CBSTNode`, `InsertMapping` | `cbst_store`, `cbst_context`, `cbst_node`, `insert_mapping` |
| `construct`, `rebuildFromDevice`, `insert`, `fill`, `erase`, `unfill`, `deviceBytes` | `construct`, `rebuild_from_device`, `insert`, `append`, `erase`, `remove_values`, `device_bytes` |
| `constructCBST`, `fillCBST`, `insertCBST`, `deleteCBST`, `unfillCBST`, `rebuildCBSTFromDevice` | `construct_cbst`, `fill_cbst`, `insert_cbst`, `delete_cbst`, `unfill_cbst`, `rebuild_cbst_from_device` |
| `insertNode_{thread,warp,block}`, `applyReuse{,_warp,_block}`, `unfill_{thread,warp,block}` | `insert_node_{thread,warp,block}_kernel`, `apply_reuse_{thread,warp,block}_kernel`, `unfill_{thread,warp,block}_kernel` |
| `cbstRankOfPosition`, `nextMultipleOf4`, `unfillRowSerial`, `flatten2DVector` | `cbst_rank_of_position`, `next_multiple_of_4`, `unfill_row_serial`, `flatten_rows` |
| `DegreeBinLaunches` (process-global), `degreeBinLaunches()` | `degree_bin_launches`, per store: `cbst_store::bin_launches()` |
| `RowView`, `RowLocator` | `row_view`, `row_locator` |

## The encoding

**Keys are 1-based.** Row `k` of a tree has key `k >= 1`; the hypergraph container (M8b) maps
its 0-based public ids to keys with a `+1` shift (and back with `-1`), and stores the row values
`+1` as well, because the payload uses 0 and negative numbers as sentinels. Nothing outside this
directory sees a key or a shifted value.

**The tree.** `num_records` nodes in heap order (node `t` has children `2t+1` and `2t+2`); the
node at heap position `t` holds the key of in-order rank `cbst_rank_of_position(t, n)` (computed
in 64 bits: A's original 32-bit product overflowed above 65,535 keys), so a search from the root
by key finds every node (`d_keys` holds the keys sorted). A node (`cbst_node`, 56 bytes) holds
its key (`index`), the offset of its row's first segment (`value`), that segment's slot count
(`length`, terminator included), `size` (the payload size when it was placed), the tail segment
(`tail_base`, `tail_capacity` usable slots, `occupancy` values in it) and its child and parent
pointers. **The pointers are absolute device addresses**: a copy of the node array must be
re-wired (`rewire_tree_kernel`, used by `clone()`); `download()` returns them as heap positions.
`avail[t] = 1` marks a deleted node (its key and row stay until reused); `subtree_avail[t]` counts
the deleted nodes of the subtree of `t` (the order-statistic tree of the best-fit reuse).

**The payload** (`d_flat_payload`, `fixed_size` slots, `initial_payload_size` = the bump pointer:
slots below it are in use, slots above it are 0). A slot holds

| value | meaning |
|---|---|
| `> 0` | a value of the row |
| `0` | an empty slot (padding, or the cleared tail of a row) |
| `INT_MIN` | the terminator: the row ends here |
| `< 0`, not `INT_MIN` | a chain pointer: the row continues at slot `-v` (an overflow segment appended by a fill) |

A row is read from `value` along its chain until 0 or `INT_MIN`.

**The three padding rules.**

1. *Construct* (`flatten_rows`, A's `flatten2DVector`): a row of `n` values takes
   `((n + 4) / 4) * 4` slots: the values, zeros, `INT_MIN` in the last slot (so an empty row takes
   4 slots).
2. *Surplus rows of an insert* (`insert_cbst`): `aligned = ceil(n / alignment) * alignment`
   (0 for `n = 0`) slots for the values and padding, plus one for `INT_MIN`: `length = aligned +
   1`, `tail_capacity = aligned`. The default alignment is 4.
3. *Overflow segments of a fill* (`allocate_space_kernel`): the `k` values that do not fit the
   tail segment get `next_multiple_of_4(k) = ((k + 4) / 4) * 4` slots (`k > 0`) at the bump
   pointer, `INT_MIN` in the last; the old segment's next slot becomes the chain pointer to it.

**Length.** A slot of `length` slots can take a row of up to `length - 1` values (one slot is the
terminator); this capacity decides which items a deleted slot can take.

## The operations

- **construct** (A's `constructCBST`): the tree over a flattened layout; the occupancy of every
  row is scanned from the payload (`init_tail_metadata_kernel`). A capacity below the payload is
  rejected (`invalid_argument_error`) before anything is allocated (A printed "Overflow" and kept
  a tree without nodes). A tree of 0 records is valid (dynG; both originals broke on it).
- **rebuild_from_device** (A's `rebuildCBSTFromDevice`): the tree over row offsets and a payload
  already on the device (keys `1..n`), e.g. a line graph merged on the device.
- **append = fill** (A's `fillCBST`): appends values to rows, binned per item by its value count
  (thread < 32, warp < 1024, block otherwise); what does not fit the tail segment goes to an
  overflow segment (rule 3), growing the payload by the policy when needed.
- **erase** (A's `deleteCBST`): marks nodes available (the node keeps its key, so the BST order
  holds) and recomputes `subtree_avail` level by level.
- **insert** (A's `insertCBST`, the best-fit reuse of ESCHER+):
  - *Eligibility:* with `D` deleted slots, only the first `min(K, D)` items may reuse a slot.
  - *Best fit:* the eligible items, sorted by size, and the deleted slots, sorted by capacity
    `length - 1`, are matched in one pass: `lo[i]` = the first slot that fits sorted item `i`,
    `assigned[i] = i + max_{j <= i}(lo[j] - j)`; item `i` takes slot `assigned[i]` if it is `< D`.
    **Ties:** both sorts are stable: items of equal size keep their input order, slots of equal
    capacity the order of the order-statistic walk, which is key order. A reused slot keeps its
    key, is rewritten from its first slot (the row, `INT_MIN`, zeros up to `length - 1`) and gets
    `occupancy = n`.
  - *Surplus:* every other item (unmatched eligible items and the items beyond `min(K, D)`) is
    appended as a new row (rule 2) and the tree is **rebuilt**: the surviving nodes keep their
    records; **deleted slots that were not reused are dropped** from the tree (their keys
    disappear); the surplus rows get the keys after the last surviving key, in item order.
  - *Requested keys are ignored:* the keys passed to insert do not choose the slots; the result,
    `insert_mapping::item_to_key`, says which key every item got.
- **remove_values = unfill** (A's `unfillCBST`): removes values from rows, binned by the number
  of removals: the thread kernel walks the whole chain (`unfill_row_serial`); the warp and block
  kernels compact the first `occupancy` slots of a one-segment row in parallel (and fall back to
  the serial walk for a chained row).

Every operation runs on the stream of the store's `resources` (kernels, copies, Thrust with
`par_nosync`), allocates from their memory resource, and synchronizes the host only at A's read
points (ADR 0038): fill 1 (the total appended), insert up to 5 (D, the matched count, the
matched pairs, the surviving count, the last surviving key), the others 0; `host_syncs()` counts
them. An operation that fails after it started changing the tree **poisons** the store: every
later operation throws until it is constructed or rebuilt.

## The policies (`store_policy.hpp`)

| preset | initial capacity (the container's choice) | a full payload | compaction |
|---|---|---|---|
| `escher_gpu()` | in-place trees (H2V, V2H) `max(requested, size + size/8 + 1024)`; merged trees (H2H, V2V) their exact size | grows to `max(needed + needed/4, 3 * capacity / 2)`, at most `INT_MAX`, then `capacity_error` | never |
| `mosp_escher()` | (step 4) `flat * 1.3 + extra`, at most 2.0e9 | compacts at B's trigger points, then throws | B's |
| dynG default | (step 5) | compacts first, then grows; `capacity_error` only at the int32 offset limit or on OOM | first |

`device_bytes()` is ESCHER-GPU's `deviceBytes()`: `num_records x (56 + 4 x 4)` bytes for the
nodes, keys, row offsets, avail and subtree_avail, the payload capacity, and the operation scratch
(`5 x scratch_item_capacity + scratch_payload_capacity` ints); `allocated_bytes()` is what the
store holds in that scope, equal to it once the tree is constructed.
