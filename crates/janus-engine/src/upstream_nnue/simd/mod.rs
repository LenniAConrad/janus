//! Backend interface and shared kernels for upstream-NNUE hot arithmetic.
//!
//! This module is the audit boundary for the single maintainer-approved
//! `unsafe` exception (2026-07-23) in the workspace. It defines the
//! `Backend` trait the hot `upstream_nnue` loops call through the
//! compile-time `Active` alias. The trait has two layers:
//!
//! * Row-level kernels (`add_i8_row`, `dot_i8_row`, ...) with safe
//!   flat-loop default bodies that reproduce the former hand-written scalar
//!   loops exactly. The portable build runs these defaults unchanged, so it
//!   stays pure safe scalar Rust with the loop shapes the auto-vectorizer
//!   already optimized well.
//! * A `LANES`-wide `i32` vector vocabulary (`load_i32`, `widen_i8_i32`,
//!   ...), compiled only for vector builds, in which the `chunked` kernel
//!   bodies and the parity tests are expressed. The lane count is a
//!   compile-time property of the selected backend: eight lanes (one
//!   256-bit register) under AVX2 and sixteen lanes (one 512-bit register)
//!   under AVX-512.
//!
//! Three backends implement the interface:
//!
//! * `scalar::Scalar` contains zero `unsafe` code and inherits every
//!   kernel default, making it the deterministic reference oracle and the
//!   portable build's only backend.
//! * `avx2::Avx2` (compiled only when the build enables the `avx2` target
//!   feature on `x86_64` without the AVX-512 set, for example via
//!   `-C target-cpu=x86-64-v3`) implements the vector vocabulary as
//!   `std::arch::x86_64` wrappers in which every `unsafe` block is a single
//!   intrinsic call, and overrides each kernel with the corresponding
//!   `chunked` body.
//! * `avx512::Avx512` (compiled only when the build enables both the
//!   `avx512f` and `avx512bw` target features on `x86_64`, for example via
//!   `-C target-feature=+avx512f,+avx512bw,+avx512vnni`) implements the
//!   same vocabulary on 512-bit registers; its pairwise multiply-accumulate
//!   additionally uses one AVX-512 VNNI `vpdpwssd` when `avx512vnni` is
//!   enabled, with a bit-identical `avx512bw` `vpmaddwd` form otherwise.
//!
//! For both vector backends, feature gating makes the intrinsics' only
//! precondition - CPU support - a compile-time fact: the module cannot be
//! compiled into a binary whose baseline lacks its instruction set, so each
//! call is sound on every CPU the binary is allowed to run on. Enabling
//! `avx512f` transitively enables `avx2` in rustc's target-feature
//! implication graph, so the shared vector vocabulary stays gated on `avx2`
//! and is present for every vector build.
//!
//! Dispatch is fully static: `Active` is a `#[cfg(target_feature)]` type
//! alias preferring AVX-512 over AVX2 over scalar, and there is no runtime
//! feature detection. All backends compute bit-identical results for every
//! input: all kernel arithmetic is two's complement wrapping, comparisons
//! are signed, downscale shifts are logical, and wrapping addition is
//! associative, so the chunked reduction orders cannot diverge from the
//! sequential defaults. The scalar backend stays compiled under vector
//! builds so the in-crate parity tests can compare the active vector
//! backend against the reference over full-range deterministic inputs.
//!
//! Lint mechanism for the exception: the workspace forbids `unsafe_code`;
//! `janus-engine` alone relaxes that to `deny` in its `Cargo.toml`, and the
//! only `#[allow(unsafe_code)]` attributes in the workspace are the
//! module-level attributes inside the leaf backend files `avx2.rs` and
//! `avx512.rs`. Everything outside those files still rejects `unsafe` at
//! compile time.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(crate) mod avx512;
// The AVX2 backend yields to AVX-512 whenever the wider set is statically
// available, so it compiles only when `avx2` is enabled without the
// `avx512f`/`avx512bw` pair required by the AVX-512 backend.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
))]
pub(crate) mod avx2;
// Under a vector build the scalar backend is not referenced by inference
// code paths, but it must remain compiled as the reference oracle for the
// scalar-vs-vector parity tests, so the dead-code lint is silenced there
// only.
#[cfg_attr(all(target_arch = "x86_64", target_feature = "avx2"), allow(dead_code))]
pub(crate) mod scalar;

/// Backend selected at compile time by the build's target features.
///
/// The AVX-512 backend requires the `avx512f` and `avx512bw` target
/// features on `x86_64` and takes precedence over every narrower backend.
/// There is no runtime detection, so a given binary always evaluates
/// through exactly one backend.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(crate) type Active = avx512::Avx512;
/// Backend selected at compile time by the build's target features.
///
/// The AVX2 backend requires the `avx2` target feature on `x86_64` and is
/// selected only when the AVX-512 feature pair is absent; every other build
/// uses the zero-`unsafe` scalar backend. There is no runtime detection, so
/// a given binary always evaluates through exactly one backend.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
))]
pub(crate) type Active = avx2::Avx2;
/// Backend selected at compile time by the build's target features.
///
/// This portable alias binds the zero-`unsafe` scalar reference backend
/// whenever the `avx2` target feature is absent.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub(crate) type Active = scalar::Scalar;

/// Number of `i32` lanes in one backend vector.
///
/// The AVX-512 backend is sixteen lanes wide (one 512-bit register), so the
/// chunked kernels, the scalar vector model, and the parity tests all share
/// its chunking scheme under an AVX-512 build. Portable builds run the
/// sequential kernel defaults and never chunk, so the constant exists only
/// alongside the vector vocabulary.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(crate) const LANES: usize = 16;
/// Number of `i32` lanes in one backend vector.
///
/// The AVX2 backend is exactly eight lanes wide (one 256-bit register), so
/// the chunked kernels and the parity tests share one chunking scheme.
/// Portable builds run the sequential kernel defaults and never chunk, so
/// the constant exists only alongside the vector vocabulary.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
))]
pub(crate) const LANES: usize = 8;

/// SIMD backend contract for the hot upstream-NNUE integer kernels.
///
/// The row-level kernels carry safe sequential default bodies that are the
/// reference semantics; an implementation may override them only with
/// bit-identical arithmetic (two's-complement wrapping, signed clamps,
/// logical shifts) for every input. The `LANES`-wide vector vocabulary
/// exists on vector builds only and must satisfy the same exactness
/// contract lane by lane; every operation is a pure function of its
/// arguments.
pub(crate) trait Backend {
    /// Backend-native [`LANES`]-wide `i32` vector value.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    type Vector: Copy;

    /// Loads [`LANES`] lanes from a lane-count-sized array.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn load_i32(source: &[i32; LANES]) -> Self::Vector;

    /// Stores [`LANES`] lanes into a lane-count-sized array.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn store_i32(target: &mut [i32; LANES], value: Self::Vector);

    /// Broadcasts one `i32` into every lane.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn splat_i32(value: i32) -> Self::Vector;

    /// Lane-wise wrapping `i32` addition.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn add_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Lane-wise wrapping `i32` subtraction.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn sub_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Lane-wise wrapping low-half `i32` multiplication.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn mul_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Lane-wise signed `i32` minimum.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn min_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Lane-wise signed `i32` maximum.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn max_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Lane-wise fused pairwise `i16` multiply-accumulate.
    ///
    /// Each `i32` lane of `a` and `b` is split into its low and high signed
    /// 16-bit halves; the result lane is `accumulation + lo(a) * lo(b) +
    /// hi(a) * hi(b)` with two's-complement wrapping addition (each 16-bit
    /// product is exact in `i32`). This is the semantics of x86's
    /// `vpmaddwd`/`vpdpwssd` families for every input, including the
    /// wrapped double-overflow lane `0x8000_8000 * 0x8000_8000`.
    ///
    /// The clipped-product kernels apply it to operands previously clamped
    /// to `0..=255`, whose high halves are zero, so there the lane equals
    /// the plain wrapping 32-bit product `a * b` while mapping to one
    /// AVX-512 VNNI `vpdpwssd` (or one AVX2 `vpmaddwd` plus an addition)
    /// instead of the slower full 32-bit multiply.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn madd_pairs_i16_i32(
        accumulation: Self::Vector,
        a: Self::Vector,
        b: Self::Vector,
    ) -> Self::Vector;

    /// Lane-wise logical right shift by nine bits.
    ///
    /// Nine is the fixed pairwise-product downscale (`/ 512`) of the
    /// CURRENT/BIG transformer; the clipped operands are non-negative, so
    /// the logical shift equals the original arithmetic division there.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn shift_right_9_i32(value: Self::Vector) -> Self::Vector;

    /// Sign-extends eight serialized signed bytes to eight `i32` lanes.
    ///
    /// The bytes are `FullThreats`/affine weights kept in their serialized
    /// `u8` storage; each is reinterpreted as `i8` before widening, exactly
    /// like the scalar `i32::from(weight as i8)` in the kernel defaults.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn widen_i8_i32(source: [u8; LANES]) -> Self::Vector;

    /// Sign-extends eight `i16` values to eight `i32` lanes.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn widen_i16_i32(source: &[i16; LANES]) -> Self::Vector;

    /// Bitmask of the nonzero `i32` lanes of one vector.
    ///
    /// Bit `i` of the result is set exactly when lane `i` is nonzero;
    /// every bit at and above [`LANES`] is zero. This is the
    /// movemask-style scan the sparse FC0 lane collection is built on.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn nonzero_mask_i32(value: Self::Vector) -> u32;

    /// Adds one sign-extended `i8` weight row into an `i32` accumulator.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the row length differs from the
    /// accumulator length; model loading sizes every referenced row to the
    /// accumulator width.
    fn add_i8_row(accumulation: &mut [i32], row: &[u8]) {
        debug_assert_eq!(accumulation.len(), row.len());
        for (target, &weight) in accumulation.iter_mut().zip(row) {
            *target = target.wrapping_add(i32::from(weight as i8));
        }
    }

    /// Adds two sign-extended `i8` weight rows into an `i32` accumulator.
    ///
    /// Keeps each destination element live across both ordered additions,
    /// exactly like the fused pairwise threat-row loop it standardizes.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when either row length differs from the
    /// accumulator length; model loading sizes every referenced row to the
    /// accumulator width.
    fn add_i8_row_pair(accumulation: &mut [i32], first: &[u8], second: &[u8]) {
        debug_assert_eq!(accumulation.len(), first.len());
        debug_assert_eq!(accumulation.len(), second.len());
        for ((target, &first_weight), &second_weight) in
            accumulation.iter_mut().zip(first).zip(second)
        {
            *target = target
                .wrapping_add(i32::from(first_weight as i8))
                .wrapping_add(i32::from(second_weight as i8));
        }
    }

    /// Subtracts one sign-extended `i8` weight row from an `i32`
    /// accumulator.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the row length differs from the
    /// accumulator length; model loading sizes every referenced row to the
    /// accumulator width.
    fn sub_i8_row(accumulation: &mut [i32], row: &[u8]) {
        debug_assert_eq!(accumulation.len(), row.len());
        for (target, &weight) in accumulation.iter_mut().zip(row) {
            *target = target.wrapping_sub(i32::from(weight as i8));
        }
    }

    /// Adds one sign-extended `i16` weight row into an `i32` accumulator.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the row length differs from the
    /// accumulator length; model loading sizes every referenced row to the
    /// accumulator width.
    fn add_i16_row(accumulation: &mut [i32], row: &[i16]) {
        debug_assert_eq!(accumulation.len(), row.len());
        for (target, &weight) in accumulation.iter_mut().zip(row) {
            *target = target.wrapping_add(i32::from(weight));
        }
    }

    /// Subtracts one sign-extended `i16` weight row from an `i32`
    /// accumulator.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the row length differs from the
    /// accumulator length; model loading sizes every referenced row to the
    /// accumulator width.
    fn sub_i16_row(accumulation: &mut [i32], row: &[i16]) {
        debug_assert_eq!(accumulation.len(), row.len());
        for (target, &weight) in accumulation.iter_mut().zip(row) {
            *target = target.wrapping_sub(i32::from(weight));
        }
    }

    /// Writes `child = parent - removed + added [- captured]` element-wise.
    ///
    /// One fused pass over the `i16` weight rows of a `HalfKA` child delta;
    /// the per-element operation order is subtract the moved piece, add its
    /// placed form, then subtract an optional capture. The destination must
    /// not alias the parent, which per-ply slot ownership guarantees.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when any slice length differs from the child
    /// length; model loading sizes every referenced row to the accumulator
    /// width.
    fn write_i16_delta(
        child: &mut [i32],
        parent: &[i32],
        removed: &[i16],
        added: &[i16],
        captured: Option<&[i16]>,
    ) {
        debug_assert_eq!(child.len(), parent.len());
        debug_assert_eq!(child.len(), removed.len());
        debug_assert_eq!(child.len(), added.len());
        debug_assert!(captured.map_or(true, |weights| weights.len() == child.len()));
        if let Some(captured) = captured {
            for ((((target, &source), &removed_weight), &added_weight), &captured_weight) in child
                .iter_mut()
                .zip(parent)
                .zip(removed)
                .zip(added)
                .zip(captured)
            {
                *target = source
                    .wrapping_sub(i32::from(removed_weight))
                    .wrapping_add(i32::from(added_weight))
                    .wrapping_sub(i32::from(captured_weight));
            }
        } else {
            for (((target, &source), &removed_weight), &added_weight) in
                child.iter_mut().zip(parent).zip(removed).zip(added)
            {
                *target = source
                    .wrapping_sub(i32::from(removed_weight))
                    .wrapping_add(i32::from(added_weight));
            }
        }
    }

    /// Adds `weight * value` for one sign-extended `i8` row into an
    /// accumulator.
    ///
    /// This is the FC0 column update; `value` is one clipped transformed
    /// activation applied to a whole transposed weight row.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the row length differs from the
    /// accumulator length; FC0's transposed layout stores one whole
    /// output-width row per transformed lane.
    fn madd_i8_row(accumulation: &mut [i32], row: &[u8], value: i32) {
        debug_assert_eq!(accumulation.len(), row.len());
        for (target, &weight) in accumulation.iter_mut().zip(row) {
            *target = target.wrapping_add(i32::from(weight as i8).wrapping_mul(value));
        }
    }

    /// Dot product of one sign-extended `i8` weight row with an `i32`
    /// input.
    ///
    /// Serves the dense FC1/FC2 tail. Wrapping addition is associative and
    /// commutative, so an override may reduce partial sums in any order and
    /// still match this sequential reference bit for bit.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the row length differs from the input
    /// length; callers slice the padded serialized row to the logical
    /// width.
    fn dot_i8_row(row: &[u8], input: &[i32]) -> i32 {
        debug_assert_eq!(row.len(), input.len());
        let mut sum = 0_i32;
        for (&weight, &value) in row.iter().zip(input) {
            sum = sum.wrapping_add(i32::from(weight as i8).wrapping_mul(value));
        }
        sum
    }

    /// Writes the pairwise-clipped transformer products for one
    /// perspective.
    ///
    /// For every index `output[i] = (clamp(low[i], 0, 255) *
    /// clamp(high[i], 0, 255)) >> 9`, exactly the CURRENT/BIG clipped
    /// pairwise product: both factors are non-negative after clamping, so
    /// the logical shift equals the original `/ 512`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the slice lengths differ; callers pass
    /// the two fixed 512-lane halves of one perspective accumulator.
    fn clipped_products_into(low: &[i32], high: &[i32], output: &mut [i32]) {
        debug_assert_eq!(low.len(), high.len());
        debug_assert_eq!(low.len(), output.len());
        for ((target, &low_value), &high_value) in output.iter_mut().zip(low).zip(high) {
            *target = clipped_product(low_value, high_value);
        }
    }

    /// Writes the clipped products of two summed perspective accumulators.
    ///
    /// Identical to [`Backend::clipped_products_into`] with each factor
    /// first formed as the wrapping sum of the `HalfKA` and `FullThreats`
    /// lanes, matching the combined incremental path's
    /// `(psq + threats).clamp(0, 255)` arithmetic.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the slice lengths differ; callers pass
    /// the fixed 512-lane halves of the two perspective accumulators.
    fn combined_clipped_products_into(
        psq_low: &[i32],
        threat_low: &[i32],
        psq_high: &[i32],
        threat_high: &[i32],
        output: &mut [i32],
    ) {
        debug_assert_eq!(psq_low.len(), threat_low.len());
        debug_assert_eq!(psq_low.len(), psq_high.len());
        debug_assert_eq!(psq_low.len(), threat_high.len());
        debug_assert_eq!(psq_low.len(), output.len());
        for (
            (((target, &psq_low_value), &threat_low_value), &psq_high_value),
            &threat_high_value,
        ) in output
            .iter_mut()
            .zip(psq_low)
            .zip(threat_low)
            .zip(psq_high)
            .zip(threat_high)
        {
            *target = clipped_product(
                psq_low_value.wrapping_add(threat_low_value),
                psq_high_value.wrapping_add(threat_high_value),
            );
        }
    }

    /// Writes the indices of the nonzero `values` lanes in ascending order.
    ///
    /// Returns the number of indices written; only that prefix of
    /// `indices` is initialized. This is the sparse-FC0 lane scan: the
    /// caller propagates exactly the returned lanes through the FC0
    /// columns, skipping the zero clipped products that dominate real
    /// positions, and the ascending order preserves the scalar sparse
    /// path's FC0 addition order bit for bit.
    ///
    /// # Panics
    ///
    /// Panics when a nonzero lane's index does not fit `u16`, and in debug
    /// builds when `indices` is shorter than `values`; callers pass one
    /// fixed 512-lane clipped-product half with an equally sized index
    /// buffer.
    fn nonzero_indices_into(values: &[i32], indices: &mut [u16]) -> usize {
        debug_assert!(indices.len() >= values.len());
        let mut count = 0;
        for (index, &value) in values.iter().enumerate() {
            if value != 0 {
                indices[count] =
                    u16::try_from(index).expect("nonzero clipped-product lane index fits u16");
                count += 1;
            }
        }
        count
    }
}

/// Scalar clipped pairwise product shared by kernel defaults and remainders.
///
/// Uses the same wrapping multiply and logical shift as the vector lanes so
/// every code path is bit-identical for every input; on the valid clamped
/// domain it equals the original `(left * right) / 512`.
fn clipped_product(low: i32, high: i32) -> i32 {
    let left = low.clamp(0, 255);
    let right = high.clamp(0, 255);
    ((left.wrapping_mul(right) as u32) >> 9) as i32
}

/// Generic [`LANES`]-wide kernel bodies used by the vector backend
/// overrides.
///
/// Each function is the chunked counterpart of one `Backend` kernel
/// default: whole vectors go through the backend's lane vocabulary and any
/// trailing lanes use the identical scalar arithmetic, so results are
/// bit-identical to the sequential defaults for every input (all arithmetic
/// wraps, and the only reduction - `dot_i8_row` - is over associative
/// wrapping addition). The clipped-product bodies form each product with
/// [`Backend::madd_pairs_i16_i32`] on a zero accumulator: both factors are
/// clamped to `0..=255` first, so their high 16-bit halves are zero and the
/// pairwise multiply-accumulate equals the plain wrapping 32-bit product
/// for every input while using the cheaper `vpmaddwd`/`vpdpwssd` hardware
/// forms. The module is compiled only for vector builds, whose active
/// backend is its sole production instantiation; the parity tests provide
/// the cross-backend evidence.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod chunked {
    use super::{clipped_product, Backend, LANES};

    /// Reborrows one `chunks_exact` chunk as a lane-count-sized array.
    ///
    /// # Panics
    ///
    /// Panics if `chunk` is not exactly [`LANES`] long; every caller passes
    /// chunks produced by `chunks_exact(LANES)`.
    fn lane_array<T>(chunk: &[T]) -> &[T; LANES] {
        chunk
            .try_into()
            .expect("chunks_exact yields lane-width chunks")
    }

    /// Reborrows one `chunks_exact_mut` chunk as a mutable lane-count array.
    ///
    /// # Panics
    ///
    /// Panics if `chunk` is not exactly [`LANES`] long; every caller passes
    /// chunks produced by `chunks_exact_mut(LANES)`.
    fn lane_array_mut<T>(chunk: &mut [T]) -> &mut [T; LANES] {
        chunk
            .try_into()
            .expect("chunks_exact_mut yields lane-width chunks")
    }

    /// Sums a vector's lanes with wrapping `i32` addition.
    fn reduce_add_i32<B: Backend>(value: B::Vector) -> i32 {
        let mut lanes = [0_i32; LANES];
        B::store_i32(&mut lanes, value);
        lanes
            .iter()
            .fold(0_i32, |sum, &lane| sum.wrapping_add(lane))
    }

    /// Chunked body of [`Backend::add_i8_row`].
    pub(super) fn add_i8_row<B: Backend>(accumulation: &mut [i32], row: &[u8]) {
        debug_assert_eq!(accumulation.len(), row.len());
        let mut targets = accumulation.chunks_exact_mut(LANES);
        let mut weights = row.chunks_exact(LANES);
        for (target, weight) in (&mut targets).zip(&mut weights) {
            let target = lane_array_mut(target);
            let sum = B::add_i32(B::load_i32(target), B::widen_i8_i32(*lane_array(weight)));
            B::store_i32(target, sum);
        }
        for (target, &weight) in targets.into_remainder().iter_mut().zip(weights.remainder()) {
            *target = target.wrapping_add(i32::from(weight as i8));
        }
    }

    /// Chunked body of [`Backend::add_i8_row_pair`].
    pub(super) fn add_i8_row_pair<B: Backend>(
        accumulation: &mut [i32],
        first: &[u8],
        second: &[u8],
    ) {
        debug_assert_eq!(accumulation.len(), first.len());
        debug_assert_eq!(accumulation.len(), second.len());
        let mut targets = accumulation.chunks_exact_mut(LANES);
        let mut first_weights = first.chunks_exact(LANES);
        let mut second_weights = second.chunks_exact(LANES);
        for ((target, first_weight), second_weight) in (&mut targets)
            .zip(&mut first_weights)
            .zip(&mut second_weights)
        {
            let target = lane_array_mut(target);
            let once = B::add_i32(
                B::load_i32(target),
                B::widen_i8_i32(*lane_array(first_weight)),
            );
            let twice = B::add_i32(once, B::widen_i8_i32(*lane_array(second_weight)));
            B::store_i32(target, twice);
        }
        for ((target, &first_weight), &second_weight) in targets
            .into_remainder()
            .iter_mut()
            .zip(first_weights.remainder())
            .zip(second_weights.remainder())
        {
            *target = target
                .wrapping_add(i32::from(first_weight as i8))
                .wrapping_add(i32::from(second_weight as i8));
        }
    }

    /// Chunked body of [`Backend::sub_i8_row`].
    pub(super) fn sub_i8_row<B: Backend>(accumulation: &mut [i32], row: &[u8]) {
        debug_assert_eq!(accumulation.len(), row.len());
        let mut targets = accumulation.chunks_exact_mut(LANES);
        let mut weights = row.chunks_exact(LANES);
        for (target, weight) in (&mut targets).zip(&mut weights) {
            let target = lane_array_mut(target);
            let difference = B::sub_i32(B::load_i32(target), B::widen_i8_i32(*lane_array(weight)));
            B::store_i32(target, difference);
        }
        for (target, &weight) in targets.into_remainder().iter_mut().zip(weights.remainder()) {
            *target = target.wrapping_sub(i32::from(weight as i8));
        }
    }

    /// Chunked body of [`Backend::add_i16_row`].
    pub(super) fn add_i16_row<B: Backend>(accumulation: &mut [i32], row: &[i16]) {
        debug_assert_eq!(accumulation.len(), row.len());
        let mut targets = accumulation.chunks_exact_mut(LANES);
        let mut weights = row.chunks_exact(LANES);
        for (target, weight) in (&mut targets).zip(&mut weights) {
            let target = lane_array_mut(target);
            let sum = B::add_i32(B::load_i32(target), B::widen_i16_i32(lane_array(weight)));
            B::store_i32(target, sum);
        }
        for (target, &weight) in targets.into_remainder().iter_mut().zip(weights.remainder()) {
            *target = target.wrapping_add(i32::from(weight));
        }
    }

    /// Chunked body of [`Backend::sub_i16_row`].
    pub(super) fn sub_i16_row<B: Backend>(accumulation: &mut [i32], row: &[i16]) {
        debug_assert_eq!(accumulation.len(), row.len());
        let mut targets = accumulation.chunks_exact_mut(LANES);
        let mut weights = row.chunks_exact(LANES);
        for (target, weight) in (&mut targets).zip(&mut weights) {
            let target = lane_array_mut(target);
            let difference = B::sub_i32(B::load_i32(target), B::widen_i16_i32(lane_array(weight)));
            B::store_i32(target, difference);
        }
        for (target, &weight) in targets.into_remainder().iter_mut().zip(weights.remainder()) {
            *target = target.wrapping_sub(i32::from(weight));
        }
    }

    /// Chunked body of [`Backend::write_i16_delta`].
    pub(super) fn write_i16_delta<B: Backend>(
        child: &mut [i32],
        parent: &[i32],
        removed: &[i16],
        added: &[i16],
        captured: Option<&[i16]>,
    ) {
        debug_assert_eq!(child.len(), parent.len());
        debug_assert_eq!(child.len(), removed.len());
        debug_assert_eq!(child.len(), added.len());
        debug_assert!(captured.map_or(true, |weights| weights.len() == child.len()));
        let mut targets = child.chunks_exact_mut(LANES);
        let mut sources = parent.chunks_exact(LANES);
        let mut removed_rows = removed.chunks_exact(LANES);
        let mut added_rows = added.chunks_exact(LANES);
        if let Some(captured) = captured {
            let mut captured_rows = captured.chunks_exact(LANES);
            for ((((target, source), removed_row), added_row), captured_row) in (&mut targets)
                .zip(&mut sources)
                .zip(&mut removed_rows)
                .zip(&mut added_rows)
                .zip(&mut captured_rows)
            {
                let value = B::sub_i32(
                    B::add_i32(
                        B::sub_i32(
                            B::load_i32(lane_array(source)),
                            B::widen_i16_i32(lane_array(removed_row)),
                        ),
                        B::widen_i16_i32(lane_array(added_row)),
                    ),
                    B::widen_i16_i32(lane_array(captured_row)),
                );
                B::store_i32(lane_array_mut(target), value);
            }
            for ((((target, &source), &removed_weight), &added_weight), &captured_weight) in targets
                .into_remainder()
                .iter_mut()
                .zip(sources.remainder())
                .zip(removed_rows.remainder())
                .zip(added_rows.remainder())
                .zip(captured_rows.remainder())
            {
                *target = source
                    .wrapping_sub(i32::from(removed_weight))
                    .wrapping_add(i32::from(added_weight))
                    .wrapping_sub(i32::from(captured_weight));
            }
        } else {
            for (((target, source), removed_row), added_row) in (&mut targets)
                .zip(&mut sources)
                .zip(&mut removed_rows)
                .zip(&mut added_rows)
            {
                let value = B::add_i32(
                    B::sub_i32(
                        B::load_i32(lane_array(source)),
                        B::widen_i16_i32(lane_array(removed_row)),
                    ),
                    B::widen_i16_i32(lane_array(added_row)),
                );
                B::store_i32(lane_array_mut(target), value);
            }
            for (((target, &source), &removed_weight), &added_weight) in targets
                .into_remainder()
                .iter_mut()
                .zip(sources.remainder())
                .zip(removed_rows.remainder())
                .zip(added_rows.remainder())
            {
                *target = source
                    .wrapping_sub(i32::from(removed_weight))
                    .wrapping_add(i32::from(added_weight));
            }
        }
    }

    /// Chunked body of [`Backend::madd_i8_row`].
    pub(super) fn madd_i8_row<B: Backend>(accumulation: &mut [i32], row: &[u8], value: i32) {
        debug_assert_eq!(accumulation.len(), row.len());
        let factor = B::splat_i32(value);
        let mut targets = accumulation.chunks_exact_mut(LANES);
        let mut weights = row.chunks_exact(LANES);
        for (target, weight) in (&mut targets).zip(&mut weights) {
            let target = lane_array_mut(target);
            let sum = B::add_i32(
                B::load_i32(target),
                B::mul_i32(B::widen_i8_i32(*lane_array(weight)), factor),
            );
            B::store_i32(target, sum);
        }
        for (target, &weight) in targets.into_remainder().iter_mut().zip(weights.remainder()) {
            *target = target.wrapping_add(i32::from(weight as i8).wrapping_mul(value));
        }
    }

    /// Chunked body of [`Backend::dot_i8_row`].
    pub(super) fn dot_i8_row<B: Backend>(row: &[u8], input: &[i32]) -> i32 {
        debug_assert_eq!(row.len(), input.len());
        let mut partial = B::splat_i32(0);
        let mut weights = row.chunks_exact(LANES);
        let mut values = input.chunks_exact(LANES);
        for (weight, value) in (&mut weights).zip(&mut values) {
            partial = B::add_i32(
                partial,
                B::mul_i32(
                    B::widen_i8_i32(*lane_array(weight)),
                    B::load_i32(lane_array(value)),
                ),
            );
        }
        let mut sum = reduce_add_i32::<B>(partial);
        for (&weight, &value) in weights.remainder().iter().zip(values.remainder()) {
            sum = sum.wrapping_add(i32::from(weight as i8).wrapping_mul(value));
        }
        sum
    }

    /// Chunked body of [`Backend::clipped_products_into`].
    pub(super) fn clipped_products_into<B: Backend>(low: &[i32], high: &[i32], output: &mut [i32]) {
        debug_assert_eq!(low.len(), high.len());
        debug_assert_eq!(low.len(), output.len());
        let zero = B::splat_i32(0);
        let ceiling = B::splat_i32(255);
        let mut targets = output.chunks_exact_mut(LANES);
        let mut lows = low.chunks_exact(LANES);
        let mut highs = high.chunks_exact(LANES);
        for ((target, low_chunk), high_chunk) in (&mut targets).zip(&mut lows).zip(&mut highs) {
            let left = B::max_i32(
                B::min_i32(B::load_i32(lane_array(low_chunk)), ceiling),
                zero,
            );
            let right = B::max_i32(
                B::min_i32(B::load_i32(lane_array(high_chunk)), ceiling),
                zero,
            );
            // Both factors are clamped to `0..=255`, so the pairwise
            // multiply-accumulate on a zero accumulator equals the plain
            // wrapping 32-bit product.
            B::store_i32(
                lane_array_mut(target),
                B::shift_right_9_i32(B::madd_pairs_i16_i32(zero, left, right)),
            );
        }
        for ((target, &low_value), &high_value) in targets
            .into_remainder()
            .iter_mut()
            .zip(lows.remainder())
            .zip(highs.remainder())
        {
            *target = clipped_product(low_value, high_value);
        }
    }

    /// Chunked body of [`Backend::combined_clipped_products_into`].
    pub(super) fn combined_clipped_products_into<B: Backend>(
        psq_low: &[i32],
        threat_low: &[i32],
        psq_high: &[i32],
        threat_high: &[i32],
        output: &mut [i32],
    ) {
        debug_assert_eq!(psq_low.len(), threat_low.len());
        debug_assert_eq!(psq_low.len(), psq_high.len());
        debug_assert_eq!(psq_low.len(), threat_high.len());
        debug_assert_eq!(psq_low.len(), output.len());
        let zero = B::splat_i32(0);
        let ceiling = B::splat_i32(255);
        let mut targets = output.chunks_exact_mut(LANES);
        let mut psq_lows = psq_low.chunks_exact(LANES);
        let mut threat_lows = threat_low.chunks_exact(LANES);
        let mut psq_highs = psq_high.chunks_exact(LANES);
        let mut threat_highs = threat_high.chunks_exact(LANES);
        for ((((target, psq_low_chunk), threat_low_chunk), psq_high_chunk), threat_high_chunk) in
            (&mut targets)
                .zip(&mut psq_lows)
                .zip(&mut threat_lows)
                .zip(&mut psq_highs)
                .zip(&mut threat_highs)
        {
            let low_sum = B::add_i32(
                B::load_i32(lane_array(psq_low_chunk)),
                B::load_i32(lane_array(threat_low_chunk)),
            );
            let high_sum = B::add_i32(
                B::load_i32(lane_array(psq_high_chunk)),
                B::load_i32(lane_array(threat_high_chunk)),
            );
            let left = B::max_i32(B::min_i32(low_sum, ceiling), zero);
            let right = B::max_i32(B::min_i32(high_sum, ceiling), zero);
            // Both factors are clamped to `0..=255`, so the pairwise
            // multiply-accumulate on a zero accumulator equals the plain
            // wrapping 32-bit product.
            B::store_i32(
                lane_array_mut(target),
                B::shift_right_9_i32(B::madd_pairs_i16_i32(zero, left, right)),
            );
        }
        for (
            (((target, &psq_low_value), &threat_low_value), &psq_high_value),
            &threat_high_value,
        ) in targets
            .into_remainder()
            .iter_mut()
            .zip(psq_lows.remainder())
            .zip(threat_lows.remainder())
            .zip(psq_highs.remainder())
            .zip(threat_highs.remainder())
        {
            *target = clipped_product(
                psq_low_value.wrapping_add(threat_low_value),
                psq_high_value.wrapping_add(threat_high_value),
            );
        }
    }

    /// Chunked body of [`Backend::nonzero_indices_into`].
    ///
    /// Whole vectors are scanned with one [`Backend::nonzero_mask_i32`]
    /// each; the set bits of the mask are then drained in ascending order,
    /// so the emitted index sequence is identical to the sequential
    /// default's for every input.
    pub(super) fn nonzero_indices_into<B: Backend>(values: &[i32], indices: &mut [u16]) -> usize {
        debug_assert!(indices.len() >= values.len());
        let mut count = 0;
        let mut chunks = values.chunks_exact(LANES);
        for (chunk_index, chunk) in (&mut chunks).enumerate() {
            let base = chunk_index * LANES;
            let mut mask = B::nonzero_mask_i32(B::load_i32(lane_array(chunk)));
            while mask != 0 {
                indices[count] = u16::try_from(base + mask.trailing_zeros() as usize)
                    .expect("nonzero clipped-product lane index fits u16");
                count += 1;
                mask &= mask - 1;
            }
        }
        let base = values.len() - chunks.remainder().len();
        for (offset, &value) in chunks.remainder().iter().enumerate() {
            if value != 0 {
                indices[count] = u16::try_from(base + offset)
                    .expect("nonzero clipped-product lane index fits u16");
                count += 1;
            }
        }
        count
    }
}

