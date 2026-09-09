<p align="center">
  <img src="assets/janus-banner.svg" alt="Janus" width="100%">
</p>

Janus is a deterministic, dependency-free UCI chess engine written in Rust.
The three production Rust crates use only the standard library and internal
path crates. An optional, separately built C++17/OpenCL subprocess accelerates
one validated BT4 model family without adding Rust FFI or Cargo dependencies.

## Version

This source snapshot is release `2026-09-09`. Janus releases are
dated; the UCI handshake reports the date while Cargo carries its derived
`year.month.day` semantic-version form.

## Learn

Read the companion book, [Build a Chess Engine](https://lenniaconrad.github.io/build-a-chess-engine/),
online.

## Reproducibility and platform floor

`Threads=1` is the deterministic reference. The same binary, model, options,
position transcript, and depth or node budget reproduce the same nodes, score,
principal variation, and best move. Time-limited searches and multi-threaded
searches are intentionally outside that exact fixed-work claim.

The repository pins Rust 1.97.1 and edition 2021. The source remains compatible
with the documented Rust 1.75 lint floor. On x86-64, the checked-in Cargo
configuration enables `popcnt`; a binary built with the default configuration
therefore requires an SSE4.2-era CPU (Intel Nehalem/2008, AMD Barcelona/2007,
or newer). It does not require the x86-64-v3/AVX2 tier. Setting `RUSTFLAGS`
replaces the checked-in target flags, so custom builds must repeat
`-C target-feature=+popcnt` if they intend to preserve the released profile.

## Building

Build the safe Rust engine with the pinned toolchain:

```text
cargo build --release --workspace --locked
```

The UCI executable is `target/release/janus`.

The optional BT4 OpenCL worker requires a C++17 compiler, Khronos OpenCL
headers, an OpenCL loader, and a vendor ICD already installed on the host:

```text
bash native/opencl/build.sh
```

That writes `target/release/janus-bt4-opencl` beside the engine. Janus never
loads the driver through Rust FFI: it speaks a bounded, versioned pipe protocol
to this subprocess and retains the safe Rust CPU implementation as the
reference and runtime fallback. See
[`native/opencl/README.md`](native/opencl/README.md) and
[`OPENCL_BACKEND_DECISION.md`](OPENCL_BACKEND_DECISION.md).

## Layout

| Path | Contents |
| --- | --- |
| `crates/janus-core` | FEN and Chess960 state, legal move generation, make/unmake, hashing, and perft. |
| `crates/janus-engine` | Alpha-beta, MCTS, evaluation, transposition tables, and neural formats. |
| `crates/janus-uci` | Bounded asynchronous UCI coordinator and option mapping. |
| `native/opencl` | Optional BT4-only OpenCL worker source and build guide. |

All chess rules and state transitions use the one `janus-core` implementation;
the UCI and accelerator boundaries do not duplicate legality logic.

## Search

Classical and supported NNUE evaluators run under iterative-deepening
alpha-beta with PVS, aspiration windows, a transposition table, quiescence,
static exchange evaluation, null-move pruning with verification, reverse
futility, futility and late-move pruning, late-move reductions, internal
iterative reduction, singular extensions, ProbCut, and history-based move
ordering. `Threads>1` uses bounded Lazy SMP helpers sharing one transposition
table; `Threads=1` remains the deterministic local-table reference.

BT4 runs only through PUCT MCTS. A complete transformer prediction supplies
the leaf policy and value, so Janus does not use BT4 inside alpha-beta or
recursive neural quiescence. A deadline can stop between predictions but does
not interrupt a prediction already in progress.

Syzygy tablebases are supported at the root and in the tree when `SyzygyPath`
is configured. Chess960 is selected with `UCI_Chess960`.

## Evaluation and model scope

The shipped default is the model-free hand-crafted `Classical` evaluator. The
`Eval` option can also select `CompactNNUE`, `UpstreamNNUE`, `CNN`, `BT4`, or
`JRBT`; neural backends require an explicit local `Eval File`. No weights are
distributed or downloaded by this repository.

CNN inference is Janus's documented CRTK `LC0J` path, not a claim of bit-exact
upstream LC0 compatibility.

BT4 support is limited to the CRTK `BT4J` v2 conversion of
`BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`: `CLASSICAL_112` input,
`PE_DENSE` embedding, 1024 channels, 15 encoder blocks, and 32 attention
heads. The retained evidence used a 739,840,557-byte converted model with SHA-256
`9359815f5ccb56054e640eb32028e687978a6189cf002530a5198882f1ded6ab`.
That digest identifies the validated evidence artifact; runtime admission is
architectural and tensor-bounded and does not require this digest. The parity
scope is Java float-bit agreement on zero input and a fixed five-position
corpus including sampled top-policy logits; Janus does not claim general or
bit-exact upstream BT4 compatibility.

BT4 retains the newest eight real positions from the UCI move transcript and
extends that bounded window along each MCTS path. History before a supplied
root FEN and LC0 repetition flags are unavailable. A FEN-only fallback can
reconstruct only the immediately preceding double-pawn-push position when a
valid en-passant target proves it; this is not complete upstream LC0 history.

The scalar CPU backend is the reference and fallback. Intel Iris Xe is the only
GPU path with retained end-to-end hardware-corpus evidence. NVIDIA and AMD use
the same compiled OpenCL graph but remain hardware-unverified; successful
compilation or Intel execution is not a vendor-support claim for either.

## UCI options

| Option | Contract |
| --- | --- |
| `Search` | `AlphaBeta` (default) or `MCTS`. |
| `Eval`, `Eval File` | Evaluator family and explicit local model path. |
| `Threads` | `1..1024`; neural implementations may clamp to their documented worker limits. |
| `Hash`, `Clear Hash` | Aggregate transposition-table budget and reset. |
| `MultiPV` | `1..8` ranked alpha-beta lines. |
| `Clock Moves To Go` | Clock-allocation horizon. |
| `Falling Eval Percent` | Extra soft time after a falling completed score. |
| `Root Fail High Reduction` | Root re-search depth reduction. |
| `Move Overhead` | Reserved transport time in milliseconds. |
| `UCI_Chess960`, `UCI_ShowWDL`, `Ponder` | Protocol and reporting controls. |
| `SyzygyPath`, `SyzygyProbeDepth`, `SyzygyProbeLimit`, `Syzygy50MoveRule` | Tablebase controls. |
| `BT4 Backend`, `BT4 Device`, `MCTS Tree Nodes` | BT4/MCTS backend and arena controls. |

The classical evaluator has no mutable experiment option. Build-only research
features, examples, test code, corpora, models, and generated Elo artifacts are
not part of this release tree.

## Safety contract

Rust `unsafe` is forbidden throughout the workspace except the two
maintainer-approved upstream-NNUE SIMD leaf backends in
`crates/janus-engine/src/upstream_nnue/simd/avx2.rs` and `avx512.rs`. Each
unsafe block wraps one `std::arch` intrinsic behind a safe-signature function;
there are no public pointer interfaces, stored pointers, pointer offsets,
transmutes, or manual memory-management operations. The only raw pointer
expressions are immediate intrinsic operands derived from fixed-size array
parameters with an in-block `as_ptr()` or `as_mut_ptr()` and register cast; they
are never returned, stored, or offset. The safe scalar backend implements the
same interface and is the exact-output reference oracle. The optional native
OpenCL implementation is isolated in its own process and never expands the
Rust exception.

## Licence

Janus is licensed under GNU GPL version 3 only. See `LICENSE.txt`.
