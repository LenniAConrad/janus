//! Zero-`unsafe` scalar reference backend for the NNUE SIMD interface.
//!
//! [`Scalar`] deliberately overrides nothing: every row kernel runs the
//! [`Backend`] trait's sequential flat-loop default, which is exactly the
//! arithmetic of the former hand-written scalar loops (wrapping
//! two's-complement adds and multiplies, signed clamps, logical shifts).
//! The portable build therefore keeps both its behavior and its
//! auto-vectorization-friendly loop shapes, and this backend is the
//! deterministic parity oracle for the vector backends.
//!
//! On vector builds the `LANES`-wide vector vocabulary is additionally
//! implemented on plain `[i32; LANES]` arrays with the same wrapping
//! semantics, so the operation-level parity tests can compare each scalar
//! lane operation directly against its intrinsic wrapper.

use super::Backend;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::LANES;

/// Zero-`unsafe` reference backend running the sequential kernel defaults.
pub(crate) struct Scalar;

impl Backend for Scalar {
    /// Plain lane array; copying it is the whole register model.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    type Vector = [i32; LANES];

    /// Copies the source array; the array itself is the vector value.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn load_i32(source: &[i32; LANES]) -> Self::Vector {
        *source
    }

    /// Copies the vector value into the destination array.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn store_i32(target: &mut [i32; LANES], value: Self::Vector) {
        *target = value;
    }

    /// Repeats one value across all eight lanes.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn splat_i32(value: i32) -> Self::Vector {
        [value; LANES]
    }

    /// Adds lane-wise with two's-complement wrapping.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn add_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut lanes = a;
        for (lane, addend) in lanes.iter_mut().zip(b) {
            *lane = lane.wrapping_add(addend);
        }
        lanes
    }

    /// Subtracts lane-wise with two's-complement wrapping.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn sub_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut lanes = a;
        for (lane, subtrahend) in lanes.iter_mut().zip(b) {
            *lane = lane.wrapping_sub(subtrahend);
        }
        lanes
    }

    /// Multiplies lane-wise keeping the wrapped low 32 bits.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn mul_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut lanes = a;
        for (lane, factor) in lanes.iter_mut().zip(b) {
            *lane = lane.wrapping_mul(factor);
        }
        lanes
    }

    /// Takes the signed lane-wise minimum.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn min_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut lanes = a;
        for (lane, other) in lanes.iter_mut().zip(b) {
            *lane = (*lane).min(other);
        }
        lanes
    }

    /// Takes the signed lane-wise maximum.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn max_i32(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut lanes = a;
        for (lane, other) in lanes.iter_mut().zip(b) {
            *lane = (*lane).max(other);
        }
        lanes
    }

    /// Adds both signed 16-bit half products per lane with wrapping sums.
    ///
    /// Each 16-bit half product is exact in `i32`, and the wrapping
    /// additions model the modular lane arithmetic of the hardware
    /// `vpmaddwd`/`vpdpwssd` forms bit for bit, including the wrapped
    /// double-overflow lane `0x8000_8000 * 0x8000_8000`.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn madd_pairs_i16_i32(
        accumulation: Self::Vector,
        a: Self::Vector,
        b: Self::Vector,
    ) -> Self::Vector {
        let mut lanes = accumulation;
        for ((lane, left), right) in lanes.iter_mut().zip(a).zip(b) {
            let low = i32::from(left as i16).wrapping_mul(i32::from(right as i16));
            let high = i32::from((left >> 16) as i16).wrapping_mul(i32::from((right >> 16) as i16));
            *lane = lane.wrapping_add(low).wrapping_add(high);
        }
        lanes
    }

    /// Shifts each lane right by nine bits, shifting in zeros.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn shift_right_9_i32(value: Self::Vector) -> Self::Vector {
        let mut lanes = value;
        for lane in &mut lanes {
            *lane = ((*lane as u32) >> 9) as i32;
        }
        lanes
    }

    /// Reinterprets each byte as `i8` and sign-extends it to `i32`.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn widen_i8_i32(source: [u8; LANES]) -> Self::Vector {
        let mut lanes = [0_i32; LANES];
        for (lane, byte) in lanes.iter_mut().zip(source) {
            *lane = i32::from(byte as i8);
        }
        lanes
    }

    /// Sign-extends each `i16` lane to `i32`.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn widen_i16_i32(source: &[i16; LANES]) -> Self::Vector {
        let mut lanes = [0_i32; LANES];
        for (lane, &half) in lanes.iter_mut().zip(source) {
            *lane = i32::from(half);
        }
        lanes
    }

    /// Sets bit `i` exactly when lane `i` is nonzero; higher bits stay
    /// zero.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn nonzero_mask_i32(value: Self::Vector) -> u32 {
        let mut mask = 0_u32;
        for (lane, &element) in value.iter().enumerate() {
            if element != 0 {
                mask |= 1 << lane;
            }
        }
        mask
    }
}
