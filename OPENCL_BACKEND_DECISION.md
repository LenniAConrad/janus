# ADR: isolated OpenCL backend for pinned BT4 inference

- Status: accepted for optional use
- Date: 2026-07-18
- Owner: Janus maintainer
- Scope: BT4 inference only

## Decision

Janus may use one in-tree C++17 OpenCL 1.2 worker for the pinned BT4J v2 model.
The safe Rust engine remains dependency-free and forbids `unsafe`; all driver
and native-library interaction stays in a child process behind a fixed 24-byte
header, bounded payloads, timeouts, numerical admission, and a complete Rust CPU
fallback. No general GPU abstraction or vendor SDK is added.

OpenCL is preferred here because implementing a GPU driver/runtime in-tree would
be unsafe and unmaintainable, while three independent CUDA, HIP, and SYCL graphs
would multiply code and release matrices. A shared OpenCL C 1.2 graph covers
the required vendors without a Rust crate, FFI, dynamic symbol loader, or
vendored binary dependency.

## Dependency inventory and provenance

Production Rust dependency count remains zero. The optional helper dynamically
uses the host OpenCL ABI and C++ runtime. Nothing below is vendored into Janus.

| Role | API/version contract | Validated Ubuntu 24.04 snapshot | License/provenance |
| --- | --- | --- | --- |
| C++ compiler | C++17 | GCC 13.3.0, Ubuntu `13.3.0-6ubuntu2~24.04.1` | GCC system toolchain; build-only |
| OpenCL headers | compile with `CL_TARGET_OPENCL_VERSION=120` | `opencl-c-headers 3.0~2023.12.14-1` | Khronos OpenCL-Headers, Apache-2.0, Ubuntu archive |
| ICD loader/development link | OpenCL 1.2 symbols | `ocl-icd 2.3.2-1build1` | OCL-dev ocl-icd, BSD-2-Clause, Ubuntu archive |
| Validated Intel ICD | OpenCL 3.0 device, OpenCL C 1.2 kernels | `intel-opencl-icd 23.43.27642.40-1ubuntu3`, driver `23.43.027642` | Intel compute-runtime, MIT, Ubuntu archive |
| NVIDIA/AMD ICD | OpenCL 1.2-compatible GPU runtime | not available for hardware validation on this host | supplied and licensed by the installed vendor driver |

The exact Intel ICD package used for the retained run was downloaded from the
Ubuntu archive as
`intel-opencl-icd_23.43.27642.40-1ubuntu3_amd64.deb`, SHA-256
`5b95f0d634fd8ee09610f1f7e2d173474ad06ba9d51de625f064621a6233b8cb`.
Its required runtime packages and checksums belong in the machine/release
artifact rather than this source tree. Releases must record helper/model hashes,
the full `--list` output, and the vendor driver/ICD version.

There are no transitive Cargo dependencies or optional Rust features. The
native source is compiled offline from this repository; system headers, loader,
and the selected vendor ICD must already be installed. CPU-only Cargo builds
remain offline and do not need OpenCL.

## Safety, targets, and failure behavior

Supported source target is a host with a C++17 compiler and OpenCL 1.2 headers,
loader, and GPU ICD. Linux x86-64 is validated. Windows binary pipe mode exists
in the worker but is not release-validated; macOS is not supported because its
OpenCL implementation and hardware matrix are outside this target.

Native unsafety is isolated by process memory, not admitted into Rust. Inputs,
model size, tensor dimensions, strings, frames, and diagnostics are bounded.
Every activated device must match the scalar CPU across every internal policy
and WDL output on zero and start-position inputs. The hardware check adds
different positions and temporal history. `Auto` may use CPU when startup or
admission fails. An explicit vendor fails startup transactionally. A later
device failure is reported and recomputed on CPU to preserve legal play.

## Maintenance and removal

The Janus maintainer owns the worker, protocol, model grammar, hardware matrix,
and license review. Kernel changes require CPU parity and at least one real
device run; vendor support claims require that vendor's real hardware. Helper
and protocol versions change together when shapes or semantics change.

Remove this exception if its release matrix becomes unmaintainable, it is not
faster than the bounded Rust fallback, driver failures compromise UCI timing,
or the pinned BT4 evaluator is retired. Removal is deleting `native/opencl`,
the GPU client/options, and the optional check; the safe Rust CPU path remains
the complete fallback throughout.
