//! AVX-512 leaf backend: audited one-intrinsic wrappers for the NNUE
//! interface.
//!
//! Public implementation detail retained for the released engine.
//!
//! Audit contract, checkable line by line:
//!
//! * Every `unsafe` block contains exactly one `std::arch::x86_64` intrinsic
//!   call and nothing else; each wrapper documents the intrinsic and the
//!   instruction it maps to.
//! * The module compiles only when the build enables both the `avx512f`
//!   and `avx512bw` target features on `x86_64` (for example via
//!   `-C target-feature=+avx512f,+avx512bw,+avx512vnni`), so the
//!   intrinsics' sole safety precondition - CPU support - is a
//!   compile-time fact of the build baseline rather than a runtime check.
//!   The one wrapper whose instruction choice depends on `avx512vnni` is
//!   additionally gated per configuration below.
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
//! VNNI use and its sign convention: AVX-512 VNNI's byte instruction
//! `vpdpbusd` multiplies unsigned bytes by signed bytes and reduces four
//! adjacent products into each `i32` lane. Under this interface it is
//! deliberately unused: the `i8`-weight kernels take arbitrary full-range
//! `i32` activations under a bit-exact-for-every-input contract, so
//! narrowing those activations to bytes (with or without the customary
//! `+128` bias compensation, which is exact only on an 8-bit activation
//! domain) would be lossy, and the element-wise kernels do not match its
//! four-to-one reduction shape in the first place. Those kernels instead
//! use the exact 512-bit AVX512F widen/multiply/add forms. The VNNI word
//! instruction `vpdpwssd` (signed 16-bit pairs, non-saturating, modular)
//! is exact for every input of the pairwise multiply-accumulate operation
//! and implements [`Backend::madd_pairs_i16_i32`], which the
//! clipped-product kernels use on operands clamped to `0..=255`; when
//! `avx512vnni` is absent, the bit-identical `avx512bw` pair
//! `vpmaddwd` + `vpaddd` is used instead.
//!
//! Semantics are bit-identical to `super::scalar::Scalar` for every input
//! (wrapping adds/multiplies, signed min/max, logical shift, sign
//! extension, modular pairwise multiply-accumulate), which the in-crate
//! parity tests assert over deterministic full-range inputs.
#![allow(unsafe_code)]
#![allow(
    // The unaligned load/store intrinsics take `__m512i`/`__m256i`/`__m128i`
    // pointers but explicitly permit arbitrary alignment; the casts below
    // never create a reference or a dereferenceable-by-Rust pointer.
    clippy::cast_ptr_alignment,
    // The workspace MSRV (1.75) governs the portable build, which never
    // compiles this module. The AVX-512 intrinsics used here were
    // stabilized in Rust 1.89, and this module is reachable only through
    // an explicit `-C target-feature` opt-in on the pinned 1.97 toolchain.
    clippy::incompatible_msrv
)]

use super::{Backend, LANES};
#[cfg(target_feature = "avx512vnni")]
use std::arch::x86_64::_mm512_dpwssd_epi32;
#[cfg(not(target_feature = "avx512vnni"))]
use std::arch::x86_64::_mm512_madd_epi16;
use std::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm256_loadu_si256, _mm512_add_epi32, _mm512_cvtepi16_epi32,
    _mm512_cvtepi8_epi32, _mm512_loadu_si512, _mm512_max_epi32, _mm512_min_epi32,
    _mm512_mullo_epi32, _mm512_set1_epi32, _mm512_srli_epi32, _mm512_storeu_si512,
    _mm512_sub_epi32, _mm512_test_epi32_mask, _mm_loadu_si128,
};

/// AVX-512 backend whose vector is one 512-bit integer register.
pub(crate) struct Avx512;

impl Backend for Avx512 {
    /// Native 512-bit integer register holding sixteen `i32` lanes.
    type Vector = __m512i;

    /// Unaligned 512-bit load; `_mm512_loadu_si512` (`vmovdqu64`).
    fn load_i32(source: &[i32; LANES]) -> Self::Vector {
        // SAFETY: single intrinsic call. AVX512F is statically enabled for
        // this module, and the pointer is derived from a live `&[i32; 16]`,
        // whose 64 bytes are exactly the unaligned load width.
        unsafe { _mm512_loadu_si512(source.as_ptr().cast::<__m512i>()) }
    }

    /// Unaligned 512-bit store; `_mm512_storeu_si512` (`vmovdqu64`).
    fn store_i32(target: &mut [i32; LANES], value: Self::Vector) {
        // SAFETY: single intrinsic call. AVX512F is statically enabled for
        // this module, and the pointer is derived from a live
        // `&mut [i32; 16]`, whose 64 bytes are exactly the unaligned store
        // width.
        unsafe { _mm512_storeu_si512(target.as_mut_ptr().cast::<__m512i>(), value) };
    }

    /// Broadcast of one `i32` to all lanes; `_mm512_set1_epi32`
    /// (`vpbroadcastd`).
    fn splat_i32(value: i32) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_set1_epi32(value) }
    }

    /// Lane-wise wrapping addition; `_mm512_add_epi32` (`vpaddd`).
    fn add_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_add_epi32(a, b) }
    }

    /// Lane-wise wrapping subtraction; `_mm512_sub_epi32` (`vpsubd`).
    fn sub_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_sub_epi32(a, b) }
    }

    /// Lane-wise low-half multiplication; `_mm512_mullo_epi32`
    /// (`vpmulld`).
    fn mul_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_mullo_epi32(a, b) }
    }

    /// Lane-wise signed minimum; `_mm512_min_epi32` (`vpminsd`).
    fn min_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_min_epi32(a, b) }
    }

    /// Lane-wise signed maximum; `_mm512_max_epi32` (`vpmaxsd`).
    fn max_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_max_epi32(a, b) }
    }

    /// Pairwise `i16` multiply-accumulate; `_mm512_dpwssd_epi32`
    /// (AVX-512 VNNI `vpdpwssd`).
    ///
    /// `vpdpwssd` adds the two adjacent signed 16-bit products of each
    /// `i32` lane into the accumulator lane without saturation - all
    /// arithmetic is modular, including the double-overflow lane
    /// `0x8000_8000 * 0x8000_8000` - so the single fused instruction
    /// matches the scalar wrapping model bit for bit.
    #[cfg(target_feature = "avx512vnni")]
    fn madd_pairs_i16_i32(
        accumulation: Self::Vector,
        a: Self::Vector,
        b: Self::Vector,
    ) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512VNNI is
        // statically enabled for this configuration of the module.
        unsafe { _mm512_dpwssd_epi32(accumulation, a, b) }
    }

    /// Pairwise `i16` multiply-accumulate; `_mm512_madd_epi16`
    /// (AVX512BW `vpmaddwd`) then `_mm512_add_epi32` (`vpaddd`).
    ///
    /// Non-VNNI form for builds enabling `avx512f`/`avx512bw` without
    /// `avx512vnni`. `vpmaddwd` adds the two adjacent signed 16-bit
    /// products of each `i32` lane without saturation - its only
    /// overflowing lane, `0x8000_8000 * 0x8000_8000`, wraps - and the lane
    /// addition wraps, so the composition matches both the scalar wrapping
    /// model and the fused `vpdpwssd` form bit for bit.
    #[cfg(not(target_feature = "avx512vnni"))]
    fn madd_pairs_i16_i32(
        accumulation: Self::Vector,
        a: Self::Vector,
        b: Self::Vector,
    ) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512BW is
        // statically enabled for this module.
        let products = unsafe { _mm512_madd_epi16(a, b) };
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_add_epi32(accumulation, products) }
    }

    /// Lane-wise logical right shift by nine; `_mm512_srli_epi32`
    /// (`vpsrld`).
    fn shift_right_9_i32(value: Self::Vector) -> Self::Vector {
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_srli_epi32::<9>(value) }
    }

    /// Sign extension of sixteen packed `i8` to `i32`; `_mm_loadu_si128`
    /// (`movdqu`) then `_mm512_cvtepi8_epi32` (`vpmovsxbd`).
    fn widen_i8_i32(source: [u8; LANES]) -> Self::Vector {
        // SAFETY: single intrinsic call. The pointer is derived from the
        // live owned `[u8; 16]` parameter, whose 16 bytes are exactly the
        // unaligned load width; SSE2 is part of the x86-64 baseline.
        let lanes: __m128i = unsafe { _mm_loadu_si128(source.as_ptr().cast::<__m128i>()) };
        // SAFETY: single intrinsic call on a register operand; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_cvtepi8_epi32(lanes) }
    }

    /// Sign extension of sixteen `i16` to `i32`; `_mm256_loadu_si256`
    /// (`vmovdqu`) then `_mm512_cvtepi16_epi32` (`vpmovsxwd`).
    fn widen_i16_i32(source: &[i16; LANES]) -> Self::Vector {
        // SAFETY: single intrinsic call. The pointer is derived from a live
        // `&[i16; 16]`, whose 32 bytes are exactly the unaligned load
        // width; AVX is statically enabled through the AVX512F baseline.
        let lanes: __m256i = unsafe { _mm256_loadu_si256(source.as_ptr().cast::<__m256i>()) };
        // SAFETY: single intrinsic call on a register operand; AVX512F is
        // statically enabled for this module.
        unsafe { _mm512_cvtepi16_epi32(lanes) }
    }

    /// Nonzero-lane bitmask; `_mm512_test_epi32_mask` (`vptestmd`).
    ///
    /// `vptestmd` sets mask bit `i` exactly when `value AND value` has a
    /// nonzero lane `i`, i.e. when the lane itself is nonzero; the
    /// sixteen-bit hardware mask widens to `u32` with zeros above
    /// [`LANES`].
    fn nonzero_mask_i32(value: Self::Vector) -> u32 {
        // SAFETY: single intrinsic call on register operands; AVX512F is
        // statically enabled for this module.
        let mask = unsafe { _mm512_test_epi32_mask(value, value) };
        u32::from(mask)
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
