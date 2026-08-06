//! AVX2 leaf backend: audited one-intrinsic wrappers for the NNUE interface.
//!
//! This file and the sibling `avx512.rs` are together the entire `unsafe`
//! surface of the workspace, per the maintainer-approved exception of
//! 2026-07-23. The workspace-wide `unsafe_code = "forbid"` lint is relaxed
//! to `deny` for `janus-engine` alone (see the crate's `Cargo.toml`), and
//! the module-level `#![allow(unsafe_code)]` attributes in these two leaf
//! files are the only places that override it, so any `unsafe` outside
//! them still fails the build. At most one of the two leaf backends is
//! compiled into a given binary: this module yields to `avx512.rs`
//! whenever the build statically enables `avx512f` and `avx512bw`.
//!
//! Audit contract, checkable line by line:
//!
//! * Every `unsafe` block contains exactly one `std::arch::x86_64` intrinsic
//!   call and nothing else; each wrapper documents the intrinsic and the
//!   instruction it maps to.
//! * The module compiles only when the build enables the `avx2` target
//!   feature on `x86_64` (and yields to the AVX-512 backend when that
//!   wider set is statically available), so the intrinsics' sole safety
//!   precondition - AVX2/SSE2 CPU support - is a compile-time fact of the
//!   build baseline rather than a runtime check.
//! * No public interface takes or returns raw pointers, and no pointer is
//!   stored or offset. The only pointer expressions are `as_ptr()` /
//!   `as_mut_ptr()` casts on fixed-size array references whose size exactly
//!   equals the referenced load/store width, evaluated immediately as the
//!   intrinsic's operand; the unaligned-load/store intrinsics carry no
//!   alignment requirement.
//! * There are no transmutes and no memory management; every value is a
//!   `Copy` register type or a caller-owned array.
//! * The kernel overrides at the end of the impl contain no `unsafe` at
//!   all: each is a one-line delegation to the shared generic chunked body
//!   in the parent module, instantiated with this backend's wrappers.
//!
//! Semantics are bit-identical to `super::scalar::Scalar` for every input
//! (wrapping adds/multiplies, signed min/max, logical shift, sign
//! extension), which the in-crate parity tests assert over deterministic
//! full-range inputs.
#![allow(unsafe_code)]
#![allow(
    // The unaligned load/store intrinsics take `__m256i`/`__m128i` pointers
    // but explicitly permit arbitrary alignment; the casts below never
    // create a reference or a dereferenceable-by-Rust pointer.
    clippy::cast_ptr_alignment
)]

use super::{Backend, LANES};
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_castsi256_ps, _mm256_cmpeq_epi32,
    _mm256_cvtepi16_epi32, _mm256_cvtepi8_epi32, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_max_epi32, _mm256_min_epi32, _mm256_movemask_ps, _mm256_mullo_epi32, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_srli_epi32, _mm256_storeu_si256, _mm256_sub_epi32,
    _mm_cvtsi64_si128, _mm_loadu_si128,
};

/// AVX2 backend whose vector is one 256-bit integer register.
pub(crate) struct Avx2;

impl Backend for Avx2 {
    /// Native 256-bit integer register holding eight `i32` lanes.
    type Vector = __m256i;

    /// Unaligned 256-bit load; `_mm256_loadu_si256` (`vmovdqu`).
    fn load_i32(source: &[i32; LANES]) -> Self::Vector {
        // SAFETY: single intrinsic call. AVX2 is statically enabled for this
        // module, and the pointer is derived from a live `&[i32; 8]`, whose
        // 32 bytes are exactly the unaligned load width.
        unsafe { _mm256_loadu_si256(source.as_ptr().cast::<__m256i>()) }
    }

    /// Unaligned 256-bit store; `_mm256_storeu_si256` (`vmovdqu`).
    fn store_i32(target: &mut [i32; LANES], value: Self::Vector) {
        // SAFETY: single intrinsic call. AVX2 is statically enabled for this
        // module, and the pointer is derived from a live `&mut [i32; 8]`,
        // whose 32 bytes are exactly the unaligned store width.
        unsafe { _mm256_storeu_si256(target.as_mut_ptr().cast::<__m256i>(), value) };
    }

    /// Broadcast of one `i32` to all lanes; `_mm256_set1_epi32`
    /// (`vpbroadcastd`).
    fn splat_i32(value: i32) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_set1_epi32(value) }
    }

    /// Lane-wise wrapping addition; `_mm256_add_epi32` (`vpaddd`).
    fn add_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_add_epi32(a, b) }
    }

    /// Lane-wise wrapping subtraction; `_mm256_sub_epi32` (`vpsubd`).
    fn sub_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_sub_epi32(a, b) }
    }

    /// Lane-wise low-half multiplication; `_mm256_mullo_epi32`
    /// (`vpmulld`).
    fn mul_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_mullo_epi32(a, b) }
    }

    /// Lane-wise signed minimum; `_mm256_min_epi32` (`vpminsd`).
    fn min_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_min_epi32(a, b) }
    }

    /// Lane-wise signed maximum; `_mm256_max_epi32` (`vpmaxsd`).
    fn max_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_max_epi32(a, b) }
    }

    /// Pairwise `i16` multiply-accumulate; `_mm256_madd_epi16`
    /// (`vpmaddwd`) then `_mm256_add_epi32` (`vpaddd`).
    ///
    /// `vpmaddwd` adds the two adjacent signed 16-bit products of each
    /// `i32` lane without saturation - its only overflowing lane,
    /// `0x8000_8000 * 0x8000_8000`, wraps - and the lane addition wraps,
    /// so the composition matches the scalar wrapping model bit for bit.
    fn madd_pairs_i16_i32(
        accumulation: Self::Vector,
        a: Self::Vector,
        b: Self::Vector,
    ) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        let products = unsafe { _mm256_madd_epi16(a, b) };
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_add_epi32(accumulation, products) }
    }

    /// Lane-wise logical right shift by nine; `_mm256_srli_epi32`
    /// (`vpsrld`).
    fn shift_right_9_i32(value: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_srli_epi32::<9>(value) }
    }

    /// Sign extension of eight packed `i8` to `i32`;
    /// `_mm_cvtsi64_si128` (`vmovq`) then `_mm256_cvtepi8_epi32`
    /// (`vpmovsxbd`).
    ///
    /// The eight weight bytes are first assembled into an `i64` with safe
    /// little-endian code, so no pointer is involved at all.
    fn widen_i8_i32(source: [u8; LANES]) -> Self::Vector {
        let packed = i64::from_le_bytes(source);
        // SAFETY: single intrinsic call on a register operand; SSE2 is part
        // of the x86-64 baseline and AVX2 is statically enabled.
        let lanes: __m128i = unsafe { _mm_cvtsi64_si128(packed) };
        // SAFETY: single intrinsic call on a register operand; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_cvtepi8_epi32(lanes) }
    }

    /// Sign extension of eight `i16` to `i32`; `_mm_loadu_si128`
    /// (`movdqu`) then `_mm256_cvtepi16_epi32` (`vpmovsxwd`).
    fn widen_i16_i32(source: &[i16; LANES]) -> Self::Vector {
        // SAFETY: single intrinsic call. The pointer is derived from a live
        // `&[i16; 8]`, whose 16 bytes are exactly the unaligned load width;
        // SSE2 is part of the x86-64 baseline.
        let lanes: __m128i = unsafe { _mm_loadu_si128(source.as_ptr().cast::<__m128i>()) };
        // SAFETY: single intrinsic call on a register operand; AVX2 is
        // statically enabled for this module.
        unsafe { _mm256_cvtepi16_epi32(lanes) }
    }

    /// Nonzero-lane bitmask; `_mm256_setzero_si256` (`vpxor`),
    /// `_mm256_cmpeq_epi32` (`vpcmpeqd`), `_mm256_castsi256_ps` (no
    /// instruction), then `_mm256_movemask_ps` (`vmovmskps`).
    ///
    /// `vpcmpeqd` sets a lane to all ones exactly when the lane is zero,
    /// so each lane's sign bit collected by `vmovmskps` marks a zero lane;
    /// complementing the eight collected bits yields the nonzero mask with
    /// zeros above [`LANES`].
    fn nonzero_mask_i32(value: Self::Vector) -> u32 {
        // SAFETY: single intrinsic call producing a zero register; AVX is
        // statically enabled through the AVX2 baseline.
        let zero = unsafe { _mm256_setzero_si256() };
        // SAFETY: single intrinsic call on register operands; AVX2 is
        // statically enabled for this module.
        let zero_lanes = unsafe { _mm256_cmpeq_epi32(value, zero) };
        // SAFETY: single intrinsic call reinterpreting register bits
        // without any instruction; AVX is statically enabled through the
        // AVX2 baseline.
        let as_floats = unsafe { _mm256_castsi256_ps(zero_lanes) };
        // SAFETY: single intrinsic call on a register operand; AVX is
        // statically enabled through the AVX2 baseline.
        let zero_mask = unsafe { _mm256_movemask_ps(as_floats) };
        !u32::try_from(zero_mask).expect("vmovmskps yields eight mask bits") & 0xff
    }

    /// Chunked override of [`Backend::add_i8_row`]; safe delegation.
    fn add_i8_row(accumulation: &mut [i32], row: &[u8]) {
        super::chunked::add_i8_row::<Self>(accumulation, row);
    }

    /// Chunked override of [`Backend::add_i8_row_pair`]; safe delegation.
    fn add_i8_row_pair(accumulation: &mut [i32], first: &[u8], second: &[u8]) {
        super::chunked::add_i8_row_pair::<Self>(accumulation, first, second);
    }

    /// Chunked override of [`Backend::sub_i8_row`]; safe delegation.
    fn sub_i8_row(accumulation: &mut [i32], row: &[u8]) {
        super::chunked::sub_i8_row::<Self>(accumulation, row);
    }

    /// Chunked override of [`Backend::add_i16_row`]; safe delegation.
    fn add_i16_row(accumulation: &mut [i32], row: &[i16]) {
        super::chunked::add_i16_row::<Self>(accumulation, row);
    }

    /// Chunked override of [`Backend::sub_i16_row`]; safe delegation.
    fn sub_i16_row(accumulation: &mut [i32], row: &[i16]) {
        super::chunked::sub_i16_row::<Self>(accumulation, row);
    }

    /// Chunked override of [`Backend::write_i16_delta`]; safe delegation.
    fn write_i16_delta(
        child: &mut [i32],
        parent: &[i32],
        removed: &[i16],
        added: &[i16],
        captured: Option<&[i16]>,
    ) {
        super::chunked::write_i16_delta::<Self>(child, parent, removed, added, captured);
    }

    /// Chunked override of [`Backend::madd_i8_row`]; safe delegation.
    fn madd_i8_row(accumulation: &mut [i32], row: &[u8], value: i32) {
        super::chunked::madd_i8_row::<Self>(accumulation, row, value);
    }

    /// Chunked override of [`Backend::dot_i8_row`]; safe delegation.
    fn dot_i8_row(row: &[u8], input: &[i32]) -> i32 {
        super::chunked::dot_i8_row::<Self>(row, input)
    }

    /// Chunked override of [`Backend::clipped_products_into`]; safe
    /// delegation.
    fn clipped_products_into(low: &[i32], high: &[i32], output: &mut [i32]) {
        super::chunked::clipped_products_into::<Self>(low, high, output);
    }

    /// Chunked override of [`Backend::combined_clipped_products_into`];
    /// safe delegation.
    fn combined_clipped_products_into(
        psq_low: &[i32],
        threat_low: &[i32],
        psq_high: &[i32],
        threat_high: &[i32],
        output: &mut [i32],
    ) {
        super::chunked::combined_clipped_products_into::<Self>(
            psq_low,
            threat_low,
            psq_high,
            threat_high,
            output,
        );
    }

    /// Chunked override of [`Backend::nonzero_indices_into`]; safe
    /// delegation.
    fn nonzero_indices_into(values: &[i32], indices: &mut [u16]) -> usize {
        super::chunked::nonzero_indices_into::<Self>(values, indices)
    }
}
