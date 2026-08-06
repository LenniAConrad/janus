# Janus

A deterministic, dependency-free UCI chess engine written in Rust.

Janus is a research engine. It has no third-party crates — the whole workspace
builds against the Rust standard library alone — and `unsafe` is forbidden
everywhere except two SIMD leaf modules whose contract is described below.
Identical inputs and identical depth or node limits reproduce the same nodes,
scores, and principal variations.

## Version

`2026-08-06`. Releases are dated rather than numbered; the version reaches a UCI
controller as `id name Janus 2026-08-06`. Cargo's manifest carries the derived
`2026.8.6`, because a semantic version cannot hold the dated form.

## Building

Janus pins its toolchain in `rust-toolchain.toml` (Rust 1.97.1), which rustup
installs automatically:

```
cargo build --release
```

The binary is `target/release/janus`. Cross-compiling for Windows needs the
mingw-w64 linker:

```
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

## Layout

| Crate | Contents |
| --- | --- |
| `janus-core` | Position state, FEN, move generation, make/unmake, perft. One shared rules implementation. |
| `janus-engine` | Search, evaluation, transposition table, MCTS, and neural model formats. |
| `janus-uci` | The bounded UCI process and its option mapping. |

Rules and state transitions live in `janus-core` and nowhere else; the UCI layer
never duplicates legality logic.

## Search

Iterative-deepening alpha-beta with aspiration windows over a direct-mapped
transposition table. Selectivity comes from null-move pruning with verification,
reverse futility, futility, late-move pruning, late-move reductions driven by
butterfly, capture, and two-ply continuation history, internal iterative
reduction, singular extensions with a multi-cut fail-high, and a conservative
ProbCut. Quiescence searches tactical moves with delta pruning and static
exchange evaluation.

`Threads=1` is the deterministic reference. Above that, Lazy SMP helpers share
one transposition table.

Syzygy endgame tablebases are supported at the root and in the tree when
`SyzygyPath` is set. Chess960 is supported through `UCI_Chess960`.

## Evaluation

The `Eval` option selects the evaluator:

- **Classical** — the hand-crafted evaluator. Tapered material and piece-square
  scoring, pawn structure and passed pawns, mobility, threats, king safety and
  shelter, rook files, and endgame scaling. This is the default and needs no
  files.
- **CompactNNUE** — Janus's own compact network format.
- **UpstreamNNUE** — a supported upstream Stockfish network container.
- **CNN** — an LCZero-style convolutional model.
- **BT4** — the validated `BT4J` v2 transformer family, run through MCTS.
- **JRBT** — a relation-biased transformer backend.

Neural backends require a model file supplied through `Eval File`. No weights
are distributed with this repository.

Janus does not claim bit-exact compatibility with LC0 or with upstream BT4. CNN
inference follows the documented `LC0J` path, and BT4 inference is limited to
the explicitly validated `BT4J` v2 family.

## Safety contract

`unsafe` is forbidden across the workspace with exactly one exception: the
per-instruction-set NNUE SIMD backends under
`crates/janus-engine/src/upstream_nnue/simd/`. Every `unsafe` block there is a
single `std::arch` intrinsic call wrapped in a safe-signature function — no
pointers, no transmutes, no manual memory management. A zero-`unsafe` scalar
backend implements the identical interface and remains the reference
implementation, and backend selection is compile-time `target_feature` dispatch,
so a portable build is pure safe Rust.

## UCI options

| Option | Notes |
| --- | --- |
| `Search` | `AlphaBeta` (default) or `MCTS`. |
| `Eval` | Evaluator selection, as above. |
| `Eval File` | Path to a model for the neural backends. |
| `Threads` | `1..1024`. `1` is the deterministic reference. |
| `Hash` | Transposition table size in MiB. |
| `Clear Hash` | Empties the table. |
| `MultiPV` | `1..8`. |
| `UCI_Chess960` | Chess960 castling and move encoding. |
| `UCI_ShowWDL` | Report win/draw/loss permilles alongside scores. |
| `SyzygyPath`, `SyzygyProbeDepth`, `SyzygyProbeLimit`, `Syzygy50MoveRule` | Endgame tablebases. |
| `Move Overhead` | Milliseconds reserved for transport latency. |
| `BT4 Backend`, `BT4 Device`, `MCTS Tree Nodes`, `Ponder` | Backend and search-mode controls. |

The classical evaluator has no UCI control and cannot be reconfigured at
runtime, so a binary cannot be talked into evaluating differently mid-match.

## Licence

GNU General Public License, version 3. See `LICENSE.txt`.
