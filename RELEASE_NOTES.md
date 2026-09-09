# Janus 2026-09-09

This dated public release ships the deterministic, dependency-free Rust UCI
engine source and two prebuilt UCI binaries for x86-64 Linux and Windows.

## Changes since 2026-08-06

- Incrementally maintained piece-placement and pawn hashes replace the repeated
  full-board key calculation and support pawn-structure caching.
- Updated maintained search and classical-evaluation source.
- Optional BT4 OpenCL helper source and build documentation.
- Current Janus SVG logo, README banner, and social-preview artwork.

This release reports the checked source update; it makes no new Elo or
measured-speed claim.

## Included

- A safe Rust engine built with Rust 1.97.1 and the checked-in release profile.
- The model-free `Classical` evaluator as the default.
- The authoritative Janus SVG logo and README/social artwork in `assets/`.
- Source build instructions and the GPL-3.0-only license.

No neural-model weights, tablebases, raw game data, generated Elo artifacts,
or experimental build surfaces are included. Neural evaluators require an
explicit local model path. The optional BT4 OpenCL helper is source-built
separately; it is not part of either binary archive.

## Validation

The public source passed formatting, offline compilation, private-item
Rustdoc with warnings denied, Clippy with warnings denied, a no-default-feature
check, a release build, and a Linux UCI handshake. The maintained source
snapshot also passed its release-only depth-5 start-position and depth-4 CPW
perft gates.

The Windows package was built with the installed
`x86_64-pc-windows-gnu` Rust target and MinGW linker, then verified as a PE32+
x86-64 console executable. It was not run on Windows on the release host.

The optional OpenCL helper compiled and listed a device on the release host,
but no model or hardware-corpus parity run was performed. That is not a claim
of GPU model support or of cross-vendor runtime validation.

## Checksums

See `SHA256SUMS.txt` in the release assets for the exact binary archive
digests.
