//! Compact CRTK NNUE v1 loading and scalar inference.
//!
//! This module intentionally supports only the small, documented `NNUE` v1
//! format.  Stockfish upstream NNUE, LC0 CNN, and BT4 files are detected and
//! rejected explicitly until their independent parity gates are implemented.
//!
//! The little-endian container consists of the four-byte `NNUE` magic, signed
//! 32-bit version/feature-count/hidden-width fields, an output scale, three
//! length-checked tensors — hidden biases, `[feature][hidden]` feature
//! weights, and the two concatenated perspective output rows — and one scalar
//! output bias. Parsing requires finite floating-point values and exact EOF.
//! Inference rebuilds one hidden accumulator per perspective and reports a
//! side-to-move score.

use crate::evaluator::Evaluator;
use janus_core::{Color, PieceKind, Position, Square};
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Used for sizing the sparse HalfKP-style input layer of compact CRTK NNUE
/// v1.
///
/// The product covers 64 king squares, 10 piece planes, and 64 piece squares.
pub const FEATURE_COUNT: usize = 64 * 10 * 64;
/// Used for matching the serialized signed-32 representation of
/// [`FEATURE_COUNT`].
const FEATURE_COUNT_I32: i32 = 40_960;

/// Used for capping the largest file accepted by the dependency-free loader.
///
/// Files above 512 MiB are rejected before any tensor allocation.
pub const MAX_MODEL_BYTES: usize = 512 * 1024 * 1024;

/// Used for identifying the compact CRTK format by its four-byte signature.
const MAGIC: &[u8; 4] = b"NNUE";
/// Used for identifying the only compact format version implemented by this
/// module.
const VERSION: i32 = 1;
/// Used for bounding the accumulator width accepted before allocating
/// tensors.
const MAX_HIDDEN_SIZE: usize = 4_096;
/// Used for recognizing the upstream Stockfish signature so its files receive
/// an explicit family diagnostic.
const STOCKFISH_VERSION: u32 = 0x7AF3_2F20;

/// Broad failure category for loading a compact network.
///
/// Each variant is a stable classification that callers can match on while
/// [`NnueError::message`] carries the human-readable detail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NnueErrorKind {
    /// Used for indicating a filesystem or stream failure.
    Io,
    /// Used for indicating malformed compact data.
    InvalidFormat,
    /// Used for indicating a recognized model family that this module does
    /// not yet implement.
    UnsupportedFormat,
    /// Used for indicating that a declared or actual allocation would exceed
    /// a fixed safety bound.
    ResourceLimit,
}

/// Error returned by compact NNUE loading.
///
/// Pairs a stable [`NnueErrorKind`] category with a context-rich diagnostic
/// message.
#[derive(Debug)]
pub struct NnueError {
    /// Used for the stable category suitable for programmatic handling.
    kind: NnueErrorKind,
    /// Used for the context-rich diagnostic intended for users and logs.
    message: String,
}

impl NnueError {
    /// Used for creating a categorized loader error with owned diagnostic
    /// text.
    ///
    /// # Arguments
    ///
    /// * `kind` - stable failure category
    /// * `message` - human-readable diagnostic detail
    ///
    /// # Returns
    ///
    /// A loader error owning its diagnostic string.
    fn new(kind: NnueErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Used for retrieving the stable error category.
    ///
    /// # Returns
    ///
    /// The [`NnueErrorKind`] classification of this failure.
    #[must_use]
    pub const fn kind(&self) -> NnueErrorKind {
        self.kind
    }

    /// Used for retrieving the human-readable detail.
    ///
    /// # Returns
    ///
    /// The diagnostic message text.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for NnueError {
    /// Used for writing the loader diagnostic.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Errors
    ///
    /// Propagates any failure reported by the underlying formatter.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for NnueError {}

impl From<io::Error> for NnueError {
    /// Used for converting a raw I/O failure into the compact-loader error
    /// vocabulary.
    ///
    /// # Arguments
    ///
    /// * `error` - underlying filesystem or stream failure
    ///
    /// # Returns
    ///
    /// An [`NnueErrorKind::Io`] error preserving the source diagnostic.
    fn from(error: io::Error) -> Self {
        Self::new(NnueErrorKind::Io, format!("NNUE I/O failed: {error}"))
    }
}

/// Shape metadata for a loaded compact model.
///
/// Reported by [`CompactNnue::info`] so callers can log or validate the
/// network's dimensions without touching its tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NnueInfo {
    /// Used for the sparse input width; always [`FEATURE_COUNT`] for v1.
    pub input_features: usize,
    /// Used for the accumulator width of each perspective.
    pub hidden_size: usize,
    /// Used for the total scalar parameter count.
    pub parameter_count: usize,
}

/// Pure-Rust scalar compact NNUE evaluator.
///
/// Holds the validated tensors of one compact CRTK v1 network and evaluates
/// positions by rebuilding both perspective accumulators from scratch.
#[derive(Clone, Debug)]
pub struct CompactNnue {
    /// Used for the number of neurons in each perspective accumulator.
    hidden_size: usize,
    /// Used for the initial value of every accumulator neuron.
    feature_bias: Vec<f32>,
    /// Used for the row-major `[feature][hidden]` sparse-feature weights.
    feature_weights: Vec<f32>,
    /// Used for the concatenated side-to-move and opponent output rows.
    output_weights: Vec<f32>,
    /// Used for the scalar added after the two perspective dot products.
    output_bias: f32,
    /// Used for the multiplier converting the raw network output to
    /// centipawns.
    output_scale: f32,
}

impl CompactNnue {
    /// Used for loading and validating a compact network from disk with a
    /// hard size cap.
    ///
    /// The file length is checked against [`MAX_MODEL_BYTES`] both before and
    /// after reading, then the bytes are parsed by [`Self::from_bytes`].
    ///
    /// # Arguments
    ///
    /// * `path` - filesystem location of the compact model file
    ///
    /// # Returns
    ///
    /// A fully validated evaluator ready for inference.
    ///
    /// # Errors
    ///
    /// Returns [`NnueError`] for I/O failures, oversized files, unsupported
    /// model families, malformed fields, invalid tensor shapes, or trailing
    /// bytes.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, NnueError> {
        let file = File::open(path.as_ref())?;
        let declared = file.metadata()?.len();
        if declared > MAX_MODEL_BYTES as u64 {
            return Err(NnueError::new(
                NnueErrorKind::ResourceLimit,
                format!("NNUE file is {declared} bytes; limit is {MAX_MODEL_BYTES}"),
            ));
        }

        let capacity = usize::try_from(declared).map_err(|_| {
            NnueError::new(
                NnueErrorKind::ResourceLimit,
                "NNUE file length does not fit this platform",
            )
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|_| {
            NnueError::new(
                NnueErrorKind::ResourceLimit,
                format!("could not reserve {capacity} bytes for NNUE file"),
            )
        })?;
        file.take((MAX_MODEL_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_MODEL_BYTES {
            return Err(NnueError::new(
                NnueErrorKind::ResourceLimit,
                "NNUE file grew beyond the loader size limit while reading",
            ));
        }
        Self::from_bytes(&bytes)
    }

    /// Used for parsing compact v1 bytes using exact shape checks and exact
    /// EOF.
    ///
    /// Known foreign model families are rejected with a dedicated diagnostic
    /// before the compact header is interpreted.
    ///
    /// # Arguments
    ///
    /// * `bytes` - complete little-endian compact v1 container
    ///
    /// # Returns
    ///
    /// A fully validated evaluator ready for inference.
    ///
    /// # Errors
    ///
    /// Returns [`NnueError`] for oversized input, unsupported model families,
    /// malformed fields, invalid tensor shapes, non-finite values, or trailing
    /// bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NnueError> {
        if bytes.len() > MAX_MODEL_BYTES {
            return Err(NnueError::new(
                NnueErrorKind::ResourceLimit,
                format!(
                    "NNUE buffer is {} bytes; limit is {MAX_MODEL_BYTES}",
                    bytes.len()
                ),
            ));
        }
        reject_unsupported_family(bytes)?;

        let mut reader = LeReader::new(bytes);
        let magic = reader.read_array::<4>("magic")?;
        if &magic != MAGIC {
            return Err(NnueError::new(
                NnueErrorKind::InvalidFormat,
                "invalid compact NNUE magic",
            ));
        }
        let version = reader.read_i32("version")?;
        if version != VERSION {
            return Err(NnueError::new(
                NnueErrorKind::UnsupportedFormat,
                format!("unsupported compact NNUE version {version}"),
            ));
        }
        let feature_count = reader.read_i32("feature count")?;
        if feature_count != FEATURE_COUNT_I32 {
            return Err(NnueError::new(
                NnueErrorKind::InvalidFormat,
                format!("compact NNUE feature count is {feature_count}, expected {FEATURE_COUNT}"),
            ));
        }
        let hidden_size = positive_usize(reader.read_i32("hidden size")?, "hidden size")?;
        if hidden_size > MAX_HIDDEN_SIZE {
            return Err(NnueError::new(
                NnueErrorKind::ResourceLimit,
                format!("compact NNUE hidden size is {hidden_size}, limit is {MAX_HIDDEN_SIZE}"),
            ));
        }
        let output_scale = reader.read_f32("output scale")?;
        require_finite(output_scale, "output scale")?;

        let feature_weight_count = FEATURE_COUNT.checked_mul(hidden_size).ok_or_else(|| {
            NnueError::new(
                NnueErrorKind::ResourceLimit,
                "compact NNUE feature tensor length overflow",
            )
        })?;
        let output_weight_count = hidden_size.checked_mul(2).ok_or_else(|| {
            NnueError::new(
                NnueErrorKind::ResourceLimit,
                "compact NNUE output tensor length overflow",
            )
        })?;
        let feature_bias = reader.read_f32_vector(hidden_size, "feature bias")?;
        let feature_weights = reader.read_f32_vector(feature_weight_count, "feature weights")?;
        let output_weights = reader.read_f32_vector(output_weight_count, "output weights")?;
        let output_bias = reader.read_f32("output bias")?;
        require_finite(output_bias, "output bias")?;
        if !reader.is_at_end() {
            return Err(NnueError::new(
                NnueErrorKind::InvalidFormat,
                format!(
                    "unexpected {} trailing bytes in compact NNUE",
                    reader.remaining()
                ),
            ));
        }

        Ok(Self {
            hidden_size,
            feature_bias,
            feature_weights,
            output_weights,
            output_bias,
            output_scale,
        })
    }

    /// Used for retrieving immutable model shape metadata.
    ///
    /// # Returns
    ///
    /// The input width, hidden width, and total scalar parameter count.
    #[must_use]
    pub fn info(&self) -> NnueInfo {
        NnueInfo {
            input_features: FEATURE_COUNT,
            hidden_size: self.hidden_size,
            parameter_count: self.feature_bias.len()
                + self.feature_weights.len()
                + self.output_weights.len()
                + 1,
        }
    }

    /// Used for evaluating one position without rounding.
    ///
    /// Both perspective accumulators are rebuilt, passed through the clipped
    /// `ReLU` activation, and combined with the side-to-move and opponent
    /// output rows before scaling.
    ///
    /// # Arguments
    ///
    /// * `position` - validated position to evaluate
    ///
    /// # Returns
    ///
    /// Side-to-move score in centipawns as an unrounded `f32`.
    ///
    /// # Panics
    ///
    /// Panics only if `position` violates `janus-core`'s invariant that both
    /// kings are present. Public position constructors reject such states.
    #[must_use]
    pub fn evaluate_centipawns(&self, position: &Position) -> f32 {
        let white = self.accumulate(position, Color::White);
        let black = self.accumulate(position, Color::Black);
        let (us, them) = match position.side_to_move() {
            Color::White => (&white, &black),
            Color::Black => (&black, &white),
        };

        let mut raw = self.output_bias;
        for index in 0..self.hidden_size {
            let us_activation = clipped_relu(us[index]);
            let them_activation = clipped_relu(them[index]);
            raw += self.output_weights[index] * us_activation;
            raw += self.output_weights[self.hidden_size + index] * them_activation;
        }
        raw * self.output_scale
    }

    /// Used for rebuilding one perspective accumulator from its bias and
    /// active features.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose pieces provide the active features
    /// * `perspective` - color whose king anchors the feature indices
    ///
    /// # Returns
    ///
    /// Hidden accumulator of `hidden_size` pre-activation values.
    fn accumulate(&self, position: &Position, perspective: Color) -> Vec<f32> {
        let mut accumulator = self.feature_bias.clone();
        for feature in active_features(position, perspective) {
            let base = feature * self.hidden_size;
            let weights = &self.feature_weights[base..base + self.hidden_size];
            for (value, weight) in accumulator.iter_mut().zip(weights) {
                *value += *weight;
            }
        }
        accumulator
    }
}

impl Evaluator for CompactNnue {
    /// Used for evaluating and rounding with Java-compatible
    /// `Math.round(float)` behavior.
    ///
    /// # Arguments
    ///
    /// * `position` - validated position to evaluate
    ///
    /// # Returns
    ///
    /// Rounded side-to-move score in centipawns.
    fn evaluate(&mut self, position: &Position) -> i32 {
        java_round(self.evaluate_centipawns(position))
    }
}

impl crate::evaluator::SearchEvaluator for CompactNnue {
    /// Used for delegating search evaluation to the [`Evaluator`]
    /// implementation.
    ///
    /// # Arguments
    ///
    /// * `position` - validated position to evaluate
    ///
    /// # Returns
    ///
    /// Rounded side-to-move score in centipawns.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Evaluator::evaluate(self, position)
    }
}

/// Used for computing sparse compact `HalfKP` feature indices for one
/// perspective.
///
/// Non-king pieces are combined with the perspective's oriented king square:
/// own pieces use planes `0..5` and opposing pieces use planes `5..10`.
///
/// # Arguments
///
/// * `position` - validated position to featurize
/// * `perspective` - color whose king anchors the features
///
/// # Returns
///
/// Feature indices in `0..`[`FEATURE_COUNT`], one per non-king piece.
///
/// # Panics
///
/// Panics only if a `Position` violates `janus-core`'s invariant that both
/// kings exist. Public position constructors reject such states.
#[must_use]
pub fn active_features(position: &Position, perspective: Color) -> Vec<usize> {
    let king = position
        .king_square(perspective)
        .expect("validated positions contain both kings");
    let king_square = orient_square(model_square(king), perspective);
    let mut features = Vec::with_capacity(30);
    for index in 0_u8..64 {
        let square = Square::new(index).expect("loop index is a square");
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        if piece.kind == PieceKind::King {
            continue;
        }
        let own = piece.color == perspective;
        let plane = piece.kind.index() + if own { 0 } else { 5 };
        let piece_square = orient_square(model_square(square), perspective);
        features.push(encode_feature(king_square, plane, piece_square));
    }
    features
}

/// Used for encoding already-oriented compact feature components.
///
/// This low-level arithmetic helper deliberately does not clamp or validate
/// components so it remains usable in constant contexts. Callers supplying
/// the documented ranges receive an index in `0..`[`FEATURE_COUNT`].
///
/// # Arguments
///
/// * `king_square` - oriented king square in model coordinates `0..64`
/// * `piece_plane` - piece plane in `0..10` (five non-king piece kinds for
///   each relative color)
/// * `piece_square` - oriented piece square in model coordinates `0..64`
///
/// # Returns
///
/// Flat sparse-feature index `((king_square * 10 + piece_plane) * 64) +
/// piece_square`.
#[must_use]
pub const fn encode_feature(king_square: usize, piece_plane: usize, piece_square: usize) -> usize {
    ((king_square * 10 + piece_plane) * 64) + piece_square
}

/// Used for converting Janus's `A8 = 0` square numbering to the compact
/// model's `a1 = 0`.
///
/// # Arguments
///
/// * `square` - square in Janus's row/file coordinates
///
/// # Returns
///
/// Equivalent model square index in `0..64`.
fn model_square(square: Square) -> usize {
    usize::from((7 - square.row()) * 8 + square.file())
}

/// Used for mirroring ranks for Black so both feature perspectives advance
/// toward rank 8.
///
/// # Arguments
///
/// * `square` - model square index in `0..64`
/// * `perspective` - color whose viewpoint is requested
///
/// # Returns
///
/// The unchanged square for White, or the rank-mirrored square (`square ^
/// 0x38`) for Black.
fn orient_square(square: usize, perspective: Color) -> usize {
    match perspective {
        Color::White => square,
        Color::Black => square ^ 0x38,
    }
}

/// Used for clamping one accumulator value to the compact network's `[0, 1]`
/// activation.
///
/// # Arguments
///
/// * `value` - pre-activation accumulator value
///
/// # Returns
///
/// `value` clamped into `[0.0, 1.0]`.
fn clipped_relu(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

/// Used for reproducing Java `Math.round(float)`, including NaN and integer
/// saturation.
///
/// NaN maps to zero; every other input is computed as `floor(value + 0.5)`
/// clamped into the `i32` range.
///
/// # Arguments
///
/// * `value` - unrounded score
///
/// # Returns
///
/// The Java-compatible rounded integer.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn java_round(value: f32) -> i32 {
    if value.is_nan() {
        return 0;
    }
    (value + 0.5)
        .floor()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

/// Used for converting a serialized positive dimension into the platform
/// index type.
///
/// # Arguments
///
/// * `value` - serialized signed 32-bit dimension
/// * `label` - field name used in diagnostics
///
/// # Returns
///
/// The dimension as a `usize`.
///
/// # Errors
///
/// Returns an [`NnueErrorKind::InvalidFormat`] error when `value` is zero or
/// negative, or when it does not fit the platform's `usize`.
fn positive_usize(value: i32, label: &str) -> Result<usize, NnueError> {
    if value <= 0 {
        return Err(NnueError::new(
            NnueErrorKind::InvalidFormat,
            format!("compact NNUE {label} must be positive, got {value}"),
        ));
    }
    usize::try_from(value).map_err(|_| {
        NnueError::new(
            NnueErrorKind::InvalidFormat,
            format!("compact NNUE {label} does not fit this platform"),
        )
    })
}

/// Used for rejecting NaN and infinite scalar parameters.
///
/// # Arguments
///
/// * `value` - scalar to validate
/// * `label` - field name used in diagnostics
///
/// # Errors
///
/// Returns an [`NnueErrorKind::InvalidFormat`] error when `value` is NaN or
/// infinite.
fn require_finite(value: f32, label: &str) -> Result<(), NnueError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(NnueError::new(
            NnueErrorKind::InvalidFormat,
            format!("compact NNUE {label} is not finite"),
        ))
    }
}

/// Used for identifying other known model magics before reporting generic
/// compact corruption.
///
/// Recognizes the Stockfish upstream version word and the `LC0J` and `BT4J`
/// signatures; buffers shorter than four bytes are left for the compact
/// parser to diagnose.
///
/// # Arguments
///
/// * `bytes` - candidate model bytes
///
/// # Errors
///
/// Returns an [`NnueErrorKind::UnsupportedFormat`] error naming the
/// recognized foreign model family.
fn reject_unsupported_family(bytes: &[u8]) -> Result<(), NnueError> {
    let Some(prefix) = bytes.get(..4) else {
        return Ok(());
    };
    let prefix: [u8; 4] = prefix.try_into().expect("length checked");
    let family = if u32::from_le_bytes(prefix) == STOCKFISH_VERSION {
        Some("Stockfish upstream NNUE")
    } else if &prefix == b"LC0J" {
        Some("LC0 CNN")
    } else if &prefix == b"BT4J" {
        Some("LC0 BT4")
    } else {
        None
    };
    if let Some(name) = family {
        Err(NnueError::new(
            NnueErrorKind::UnsupportedFormat,
            format!("{name} models are not supported by the compact NNUE loader"),
        ))
    } else {
        Ok(())
    }
}

/// Bounds-checked little-endian reader for the compact in-memory container.
///
/// The cursor only advances after a complete field is available. Tensor
/// readers additionally enforce the declared shape and finite-value contract.
struct LeReader<'a> {
    /// Used for the complete model bytes; the loader never reads beyond this
    /// slice.
    bytes: &'a [u8],
    /// Used for tracking the offset of the next unread byte.
    offset: usize,
}

impl<'a> LeReader<'a> {
    /// Used for creating a reader positioned at byte zero.
    ///
    /// # Arguments
    ///
    /// * `bytes` - complete in-memory model container
    ///
    /// # Returns
    ///
    /// A reader whose cursor starts at the first byte.
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    /// Used for reading one fixed-size field without advancing on truncation.
    ///
    /// # Arguments
    ///
    /// * `label` - field name used in diagnostics
    ///
    /// # Returns
    ///
    /// The next `SIZE` bytes as a fixed array.
    ///
    /// # Errors
    ///
    /// Returns an [`NnueErrorKind::InvalidFormat`] error when the cursor
    /// arithmetic would overflow or the buffer ends before `SIZE` bytes are
    /// available.
    fn read_array<const SIZE: usize>(&mut self, label: &str) -> Result<[u8; SIZE], NnueError> {
        let end = self.offset.checked_add(SIZE).ok_or_else(|| {
            NnueError::new(
                NnueErrorKind::InvalidFormat,
                format!("compact NNUE offset overflow while reading {label}"),
            )
        })?;
        let source = self.bytes.get(self.offset..end).ok_or_else(|| {
            NnueError::new(
                NnueErrorKind::InvalidFormat,
                format!("compact NNUE ended while reading {label}"),
            )
        })?;
        self.offset = end;
        Ok(source.try_into().expect("slice has requested fixed length"))
    }

    /// Used for reading one little-endian signed 32-bit field.
    ///
    /// # Arguments
    ///
    /// * `label` - field name used in diagnostics
    ///
    /// # Returns
    ///
    /// The decoded signed 32-bit value.
    ///
    /// # Errors
    ///
    /// Returns an [`NnueErrorKind::InvalidFormat`] error when the buffer ends
    /// before four bytes are available.
    fn read_i32(&mut self, label: &str) -> Result<i32, NnueError> {
        Ok(i32::from_le_bytes(self.read_array(label)?))
    }

    /// Used for reading one little-endian IEEE-754 scalar without
    /// interpreting finiteness.
    ///
    /// # Arguments
    ///
    /// * `label` - field name used in diagnostics
    ///
    /// # Returns
    ///
    /// The decoded 32-bit floating-point value.
    ///
    /// # Errors
    ///
    /// Returns an [`NnueErrorKind::InvalidFormat`] error when the buffer ends
    /// before four bytes are available.
    fn read_f32(&mut self, label: &str) -> Result<f32, NnueError> {
        Ok(f32::from_le_bytes(self.read_array(label)?))
    }

    /// Used for reading a length-prefixed finite `f32` tensor with an exact
    /// expected size.
    ///
    /// The serialized length prefix must equal `expected`, the remaining
    /// buffer must cover the whole tensor, and every element must be finite.
    ///
    /// # Arguments
    ///
    /// * `expected` - exact number of elements the tensor must declare
    /// * `label` - tensor name used in diagnostics
    ///
    /// # Returns
    ///
    /// The tensor as a vector of `expected` finite values.
    ///
    /// # Errors
    ///
    /// Returns an [`NnueError`] when the declared length is negative or
    /// differs from `expected`, the byte length overflows, the buffer is
    /// truncated, the allocation fails, or an element is not finite.
    fn read_f32_vector(&mut self, expected: usize, label: &str) -> Result<Vec<f32>, NnueError> {
        let declared = self.read_i32(&format!("{label} length"))?;
        let declared_size = usize::try_from(declared).map_err(|_| {
            NnueError::new(
                NnueErrorKind::InvalidFormat,
                format!("compact NNUE {label} length is negative: {declared}"),
            )
        })?;
        if declared_size != expected {
            return Err(NnueError::new(
                NnueErrorKind::InvalidFormat,
                format!("compact NNUE {label} length is {declared}, expected {expected}"),
            ));
        }
        let byte_count = expected.checked_mul(4).ok_or_else(|| {
            NnueError::new(
                NnueErrorKind::ResourceLimit,
                format!("compact NNUE {label} byte length overflow"),
            )
        })?;
        if self.remaining() < byte_count {
            return Err(NnueError::new(
                NnueErrorKind::InvalidFormat,
                format!("compact NNUE ended while reading {label}"),
            ));
        }
        let mut values = Vec::new();
        values.try_reserve_exact(expected).map_err(|_| {
            NnueError::new(
                NnueErrorKind::ResourceLimit,
                format!("could not allocate compact NNUE {label}"),
            )
        })?;
        for _ in 0..expected {
            let value = self.read_f32(label)?;
            require_finite(value, label)?;
            values.push(value);
        }
        Ok(values)
    }

    /// Used for retrieving the number of unread bytes.
    ///
    /// # Returns
    ///
    /// Count of bytes between the cursor and the end of the buffer.
    const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    /// Used for checking whether every byte has been consumed exactly.
    ///
    /// # Returns
    ///
    /// `true` when the cursor sits at the end of the buffer.
    const fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
