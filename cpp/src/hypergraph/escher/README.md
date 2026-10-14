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
| MOSP_ESCHER@4b86159's (B) behaviours, merged one per commit (ADR 0039 lists every deviation from B's literal code): construct validation (`validate_construct`), the fill clamps (`payload.cu`), caller row counts (`set_initial_occupancy_kernel` in `build_tree.cu`), the bounded chain walk (`chain.cuh`, B's `readFlat` and `warpWalkRow`), its integrity checks (`check_subtree_avail`), the scratch growth rule, `compact()` (`compact.cu`, B's compaction kernels), B's accounting scope in `stats()` | M8a step 4 |
| the control block (the device error word), `check_errors()`, `rows()`, `check_tree_index`, the rule for dead or absent keys (`lookup.cuh`); the staging preset `mosp_escher_staged()` (replaced by `mosp_escher()` in step 5) | dynG (M8a step 4) |
| B's automatic compaction (`fill_impl`, the surplus check of `insert_cbst`), the presets `mosp_escher()` and `dyng_default()`, `store_policy::max_payload`, the mutation hooks `DYNG_MUTATION_CBST_*` (PLAN 8.4) | M8a step 5 (ADR 0040) |
| `set_used_payload()`: the bump pointer of a tree restored from another backend's state | dynG (M8b, ADR 0041) |

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

## The hypergraph container (M8b)

`dyng::hypergraph` (`cpp/src/hypergraph/`, ADR 0041) keeps two trees on the CUDA backend
(`escher_store.cu`): **H2V** (key = hyperedge id + 1, values = member vertices + 1, sorted; only
inserted and erased, so every row is one segment) and **V2H** (key = vertex + 1, values = hyperedge
ids + 1, appended and removed, so rows are chained and unsorted), and the row view of H2V
(`row_locator::build`), rebuilt once per applied batch. The host keeps one byte per id (absent,
live, a deleted slot still in the tree) for validation and the id bound, nothing else; the
members of deleted ids are read back with `rows()`. On the host backends `id_allocator.cpp`
reproduces the H2V tree's ids, slot lengths and payload accounting without a payload (written
from this page, independently of `testing::cbst_host_model`). A container moved between backends
keeps its keys, deleted slots, slot lengths and bump pointer: the device tree is constructed over
those lengths, the deleted slots erased and the bump pointer set with `set_used_payload()`.

## The encoding

**Keys are 1-based.** Row `k` of a tree has key `k >= 1`; the hypergraph container (M8b) maps
its 0-based public ids to keys with a `+1` shift (and back with `-1`), and stores the row values
`+1` as well, because the payload uses 0 and negative numbers as sentinels. Nothing outside this
directory sees a key or a shifted value.

**The tree.** `num_records` nodes in heap order (node `t` has children `2t+1` and `2t+2`); the
node at heap position `t` holds the key of in-order rank `cbst_rank_of_position(t, n)` (computed
in 64 bits, as ESCHER-GPU@abcf9b2 already does; the upstream ESCHER code's 32-bit product
overflowed above 65,535 keys), so a search from the root
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

A row is read from `value` along its chain until 0 or `INT_MIN`. Every reader and the unfill
writer walk it with `chain.cuh`, bounded by the used payload: a walk that reads more slots than
`initial_payload_size` or leaves the used prefix is a broken chain, which raises the device error
word (below); a row is never truncated (B's literal 1,024-hop bound is not ported: under
`escher_gpu()` a legitimate V2H row can have more hops; ADR 0039, section 3).

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
  row is scanned from the payload (`init_tail_metadata_kernel`), or taken from the caller's row
  counts (B's `rowOccupancy`, a fast path with the same nodes; never B's `nullptr`, occupancy 0).
  B's validation runs first: negative sizes, a capacity below the payload and null arrays are
  rejected (`invalid_argument_error`) before anything is allocated (A printed "Overflow" and kept
  a tree without nodes), and so is a malformed layout (dynG, M8a review): the keys must be
  strictly increasing, the row offsets strictly increasing from at least 0 with the last one below
  the payload size (every row has at least one slot, its terminator). A tree of 0 records is
  valid (dynG; both originals broke on it); it drops the arrays of an earlier tree.
- **rebuild_from_device** (A's `rebuildCBSTFromDevice`): the tree over row offsets and a payload
  already on the device (keys `1..n`), e.g. a line graph merged on the device; with 0 records
  (every row deleted) it keeps the payload and builds nothing (dynG).
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
- **compact** (B's `compactCBST`; explicit in M8a step 4): every row rewritten as one segment in
  key order: a live row of `L` values takes `(L + 4) / 4 * 4` slots (construct padding, rule 1), a
  deleted row keeps its first segment of `length` slots, emptied, and drops its chain; keys, node
  positions and avail do not change; the bump pointer moves to the compacted size. A payload too
  small for the compacted rows grows by the policy (or `capacity_error`, nothing changed).
- **rows** (A's and B's `findContents`, which printed them): the rows of keys, read along their
  chains.

**Dead and absent keys.** An append or remove_values item naming a key that is not in the tree, or
whose node is deleted, is skipped and counted, and raises the device error word (dynG's rule; the
originals skipped absent keys silently and wrote to deleted rows); erase ignores absent keys.

**The shape of a batch** (dynG, M8a review; PLAN 4.7.3): insert, append and remove_values take
one prefix size per key, non-decreasing from 0, the last one the payload size; anything else is an
`invalid_argument_error` before anything changes (the kernels read the payload through the prefix
sums), and more than `INT_MAX / 3` items a `capacity_error`. A fill whose overflow segments would
end beyond the int32 offsets is summed exactly on the host first and compacts (where the policy
compacts) or throws `capacity_error` instead of wrapping.

**The device error word** (the control block, PLAN 4.7.3): a broken chain or a rejected item
sets it on the device; every read point copies it back with its synchronization and throws
`internal_error`, poisoning the store; `check_errors()` is the read point of erase and
remove_values, which have none. `clone()` copies the word, so an error the source has not read
yet throws at the copy's next read point too.

Every operation runs on the stream of the store's `resources` (kernels, copies, Thrust with
`par_nosync`), allocates from their memory resource, and synchronizes the host only at A's read
points (ADR 0038): fill 1 (the total appended), insert up to 5 (D, the matched count, the
matched pairs, the surviving count, the last surviving key), the others 0; `host_syncs()` counts
them (a Debug build's check of the caller's row counts adds one counted synchronization to a
construct with counts). An operation that fails after it started changing the tree **poisons**
the store: every later operation, `rows()`, `find()`, `clone()` and `check_errors()` throw
`invalid_argument_error` until it is constructed or rebuilt; `download()` is the one forensic read
it allows. Only the precondition checks, the construct validation, the batch shape and
`compact()`'s `capacity_error` give the strong guarantee (ADR 0040, section 4).

## The policies (`store_policy.hpp`)

| preset | initial capacity (the container's choice) | a full payload | compaction |
|---|---|---|---|
| `escher_gpu()` | in-place trees (H2V, V2H) `max(requested, size + size/8 + 1024)`; merged trees (H2H, V2V) their exact size | grows to `max(needed + needed/4, 3 * capacity / 2)`, at most `INT_MAX`, then `capacity_error` | never |
| `mosp_escher(extra = 2^20)` | `floor(flat * 1.3) + extra`, at most 2.0e9 (B's `constructFromRows`) | `capacity_error` (B's `EscherError`) | at B's trigger points, first |
| `dyng_default(max_payload = INT_MAX)` | as `escher_gpu()` | grows by A's rule, at most `max_payload`, then `capacity_error`; `out_of_memory_error` on an allocation failure | at B's trigger points, first; then grows when fewer than 1/8 of the slots are free (the hysteresis) |

**B's trigger points** (`compaction_rule::mosp_escher`): a fill whose overflow segments do not fit
behind the bump pointer compacts and then appends, once more and without compacting, the values
that did not fit the rows' tails; an insert whose surplus rows do not fit compacts after the
best-fit reuse, before it appends them. What still does not fit grows by the policy (or is a
`capacity_error`, naming the first surplus row that does not fit, before any is written).
`max_payload` (INT_MAX, the int32 offsets) bounds every growth; tests lower it.

**The hysteresis of the default** (`compaction_headroom = 8`; M8a review, ADR 0040 section 1):
after an operation that compacted at a trigger point, `dyng_default()` also grows the payload by
A's rule when fewer than `capacity / 8` slots are free, to the capacity that leaves that many
(at most `max_payload`). Without it, live data that grows slowly toward the capacity compacted
the whole payload (O(payload) each time) more and more often, down to every operation, before it
ever grew. `mosp_escher()` keeps B's literal policy; the host model's `dyng_default()` predicts
the same growths.

The operation scratch grows to an operation's size under `escher_gpu()` (A's `reserveScratch`) and
geometrically (+n/2+1 steps, B's `ensureScratchCapacity`) under `mosp_escher()` and dynG's
default.

`device_bytes()` is ESCHER-GPU's `deviceBytes()`: `num_records x (56 + 4 x 4)` bytes for the
nodes, keys, row offsets, avail and subtree_avail, the payload capacity, and the operation scratch
(`5 x scratch_item_capacity + scratch_payload_capacity` ints); `allocated_bytes()` is what the
store holds in that scope, equal to it once the tree is constructed.
(`stats()` also gives MOSP_ESCHER's `escherDeviceBytes()` scope, without the scratch, the payload
capacity and use, growths, compactions and the path counters; the control block is in neither
scope.)

## The merge log

The store was built in five steps, every commit replayed against both originals' traces
(`parity/results/M8a.md`, sections 11-13; `parity/cbst_merge_steps.sh <commit>...` rebuilds any
of them):

1. **The transplant** of A's store (`477d592`, reproducible with `parity/escher_port/transplant.py
   --check --rev 477d592`), then one class of mechanical change per commit: errors, memory,
   streams, registration and counters (ADR 0038). A's traces match in every recorded field at
   every commit.
2. **The dynG additions**: the empty tree, `clone()`, `download()`, `allocated_bytes()`,
   `store_policy::escher_gpu()`.
3. **B's behaviours that change no trace of A**, one per commit (ADR 0039): validation, the fill
   clamps, caller row counts, `chain.cuh`, integrity, `rows()`, scratch growth, `compact()`, the
   accounting, dead or absent keys.
4. **B's automatic compaction and its preset** `mosp_escher()` (ADR 0040): from this commit B's
   traces match in every recorded field too, apart from the constructs B made without row counts
   (`construct_without_row_counts`: their occupancy only).
5. **dynG's default** `dyng_default()`: every set replayed with the field set `policy` (the rows,
   avail and subtree_avail identical; the points where the policies differ listed).

The mutation hooks (`#if defined(DYNG_MUTATION_CBST_...)`, off in every normal build) put back
the bugs of PLAN 8.4's CBST list; `ctest -L mutation` must see each one fail the store's suites.
