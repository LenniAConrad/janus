# Janus BT4 OpenCL worker

This directory contains the optional `janus-bt4-opencl` subprocess. It is a
BT4-only OpenCL 1.2 accelerator, not a general GPU API and not a CUDA, HIP,
SYCL, CNN, or NNUE backend. The safe Rust engine communicates with it through
a bounded, versioned pipe protocol; there is no Rust FFI or vendor SDK crate.

## Supported model contract

The worker accepts only the bounded CRTK `BT4J` v2 architecture family matching
the conversion of `BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`:
`CLASSICAL_112` input, `PE_DENSE` embedding, 1024 channels, 15 encoder blocks,
and 32 attention heads. The retained evidence used a 739,840,557-byte converted
model with SHA-256
`9359815f5ccb56054e640eb32028e687978a6189cf002530a5198882f1ded6ab`.
That digest identifies the validated evidence artifact; runtime admission
validates format, architecture, tensors, dimensions, and bounds and does not
require this digest.

Parity evidence is scoped to Java float-bit agreement on zero input and a
fixed five-position corpus including sampled top-policy logits. It is not a
claim of general or bit-exact upstream BT4 compatibility. The safe scalar Rust
implementation remains the deterministic reference and runtime fallback.

BT4 keeps the newest eight real positions from the UCI move transcript and
extends that bounded window along each MCTS path. History before a supplied
root FEN and LC0 repetition flags are unavailable. A FEN-only fallback recovers
only the immediately preceding double-pawn-push position when a valid
en-passant target proves it; this is not complete upstream LC0 history.

## Build and selection

The helper requires a C++17 compiler, Khronos OpenCL headers, an OpenCL loader,
and a vendor ICD already installed on the host:

```text
bash native/opencl/build.sh
```

The script writes `target/release/janus-bt4-opencl`. The engine discovers that
exact sibling path by default; an installation can instead set
`JANUS_BT4_OPENCL_HELPER` to an exact helper path. Janus does not invoke a
shell or search `PATH` for the worker.

List the stable, memory-sorted device ordinals for a selection with:

```text
target/release/janus-bt4-opencl --list --vendor auto
target/release/janus-bt4-opencl --list --vendor intel
target/release/janus-bt4-opencl --list --vendor nvidia
target/release/janus-bt4-opencl --list --vendor amd
```

`BT4 Device` is the zero-based row in the corresponding list. Explicit vendor
selection is strict during startup and numerical admission: it never silently
selects a different manufacturer. A later runtime failure is bounded, reported
before `bestmove`, and recomputed with the scalar CPU reference.

## Hardware evidence

Intel Iris Xe is the only retained end-to-end hardware-validated OpenCL target.
That run covered enumeration, model admission, corpus parity, repeated
inference, timed UCI search, teardown, and injected CPU fallback. NVIDIA and AMD
are implementation-present but hardware-unverified; successful compilation,
unavailable-device diagnostics, or Intel execution is not validation for those
vendors.

On the retained Intel corpus, maximum compressed-policy absolute error was
`0.000026345` and maximum WDL error was `0.000000536`. Those tolerances and the
recorded latency are device-specific numerical and throughput evidence, not
Elo. A vendor support claim requires the same full model/hardware corpus on a
real device of that vendor, with helper/model hashes, device listing, ICD and
driver versions, parity maxima, and latency retained.

## Runtime boundaries

- Model parameters, kernels, and bounded scratch buffers are uploaded once per
  worker; MCTS sends serial batch-size-one predictions.
- Startup/model upload has a 180-second response limit. Each prediction has a
  five-second watchdog; recovery is bounded and observable.
- Every admission output is checked for finite values and numerical agreement
  before the GPU is accepted.
- `Threads` configures the safe CPU implementation and fallback. OpenCL
  work-group scheduling remains device-managed.
- Diagnostics and protocol frames are bounded; the helper owns no chess rules,
  search, UCI parsing, or time management.

See `../../OPENCL_BACKEND_DECISION.md` for the dependency, maintenance, and
removal decision.
