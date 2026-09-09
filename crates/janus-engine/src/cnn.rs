//! Dependency-free scalar inference for `ChessRTK`'s `LC0J` residual CNN.
//!
//! This is deliberately a loader for the repository's documented little-
//! endian `LC0J` v1 format, not a loader for arbitrary upstream Leela files.
//! It implements the classical 112-plane encoder with bounded real-position
//! history for MCTS, residual and squeeze/excitation blocks, the mapped
//! 73-plane policy, and a WDL value head. FEN-only calls retain the documented
//! repeated-current fallback, and unavailable LC0 repetition flags remain
//! zero. All tensors are shape checked before allocation or use.
//!
//! The `LC0J` v1 container is little-endian and dimension-first. Every
//! convolution, dense layer, squeeze/excitation unit, and policy map is stored
//! with explicit sizes followed by length-prefixed finite `f32` tensors. The
//! parser validates those declarations against the residual architecture and
//! requires exact EOF before constructing reusable inference buffers.

use crate::evaluator::{Evaluator, PolicyValue};
use crate::threading::spawn_scoped_or_run;
use janus_core::{CastlingRights, Color, Move, Piece, PieceKind, Position};
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Used for sizing LC0's classical input representation at 112 planes.
///
/// The classical encoder produces eight history slots of thirteen planes each
/// plus eight auxiliary planes, giving `8 * 13 + 8 = 112` channels of 64
/// squares apiece.
pub const INPUT_CHANNELS: usize = 112;
/// Used for sizing the uncompressed 73-plane policy representation.
///
/// Every one of the 64 origin squares owns 73 move planes (56 queen-style
/// slides, 8 knight jumps, and 9 underpromotions), giving 4,672 entries.
pub const RAW_POLICY_SIZE: usize = 73 * 64;
/// Used for bounding the largest `LC0J` file accepted by this scalar backend.
///
/// Both the file loader and the in-memory parser reject inputs above this
/// 256 MiB ceiling before any tensor allocation takes place.
pub const MAX_MODEL_BYTES: usize = 256 * 1024 * 1024;
/// Used for bounding the dependency-free CPU worker team accepted for one
/// CNN inference.
///
/// The pinned 10x128 model scales at four workers; larger scoped teams lose
/// throughput to thread creation and heterogeneous-core imbalance.
pub const MAX_INFERENCE_THREADS: usize = 4;

/// Used for identifying the four-byte signature of the compact
/// residual-network format.
///
/// The parser rejects any input whose first four bytes differ from `LC0J`.
const MAGIC: &[u8; 4] = b"LC0J";
/// Used for pinning the only `LC0J` container version implemented by this
/// module.
///
/// Files declaring any other version are rejected as unsupported.
const VERSION: i32 = 1;
/// Used for sizing every channel-major board plane at 64 cells.
///
/// All convolution, plane-fill, and bias loops in this module iterate over
/// exactly one 8x8 board of squares per channel.
const BOARD_SQUARES: usize = 64;
/// Used for sizing the placement-history window of the classical encoder.
///
/// The encoder always writes eight history slots; missing older positions
/// repeat the oldest supplied placement.
const HISTORY: usize = 8;
/// Used for sizing one history slot of the classical encoder.
///
/// Each slot stores twelve piece planes plus one repetition plane.
const PLANES_PER_HISTORY: usize = 13;
/// Used for locating the first auxiliary plane after all placement-history
/// planes.
///
/// Castling rights, side to move, the halfmove clock, and the constant plane
/// are written at offsets relative to this base.
const AUX_BASE: usize = HISTORY * PLANES_PER_HISTORY;
/// Used for fixing the number of win/draw/loss value-head logits at three.
///
/// The parser rejects models declaring any other value-head output count.
const WDL_OUTPUTS: usize = 3;
/// Used for bounding the accepted input, trunk, policy, or value channel
/// count.
///
/// Any declared channel dimension above this limit is rejected as a resource
/// violation before allocation.
const MAX_CHANNELS: usize = 2_048;
/// Used for bounding the accepted number of residual blocks.
///
/// Any declared block count above this limit is rejected as a resource
/// violation before allocation.
const MAX_BLOCKS: usize = 256;
/// Used for bounding the accepted logical width of a dense layer.
///
/// Any declared dense dimension above this limit is rejected as a resource
/// violation before allocation.
const MAX_DENSE_DIM: usize = 65_536;
/// Used for bounding the accepted squeeze/excitation hidden width.
///
/// Any declared SE hidden dimension above this limit is rejected as a
/// resource violation before allocation.
const MAX_SE_HIDDEN: usize = 16_384;
/// Used for capping the global parameter count, derived from the file-size
/// limit.
///
/// A valid file cannot contain more finite `f32` parameters than
/// [`MAX_MODEL_BYTES`] divided by four bytes per float.
const MAX_TENSOR_FLOATS: usize = MAX_MODEL_BYTES / std::mem::size_of::<f32>();
/// Used for ordering LC0 knight-policy deltas from the side-to-move
/// perspective.
///
/// Each `(file delta, rank delta)` pair occupies one of the eight knight
/// policy planes 56..64, in this exact index order.
const KNIGHT_DELTAS: [(i32, i32); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

/// Broad failure category for loading or invoking an LC0J model.
///
/// Every [`CnnError`] carries exactly one of these stable categories so
/// callers can branch on the failure class without parsing the diagnostic
/// message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CnnErrorKind {
    /// Used for indicating a filesystem or stream failure.
    Io,
    /// Used for indicating malformed LC0J data or invalid encoded input.
    InvalidFormat,
    /// Used for indicating a recognized magic/version outside the implemented
    /// LC0J v1 contract.
    UnsupportedFormat,
    /// Used for indicating that a fixed safety limit is exceeded or a
    /// validated model allocation cannot be admitted.
    ResourceLimit,
}

/// Error returned by the LC0J loader or encoded-input API.
///
/// Pairs a stable [`CnnErrorKind`] category with a human-readable diagnostic
/// naming the rejected field or operation.
#[derive(Debug)]
pub struct CnnError {
    /// Used for storing the stable category suitable for programmatic
    /// handling.
    kind: CnnErrorKind,
    /// Used for storing human-readable context for the rejected field or
    /// operation.
    message: String,
}

impl CnnError {
    /// Used for creating a categorized model or encoded-input error.
    ///
    /// # Arguments
    ///
    /// * `kind` - stable failure category
    /// * `message` - human-readable diagnostic context
    ///
    /// # Returns
    ///
    /// A new error combining the category and diagnostic.
    fn new(kind: CnnErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Used for retrieving the stable error category.
    ///
    /// # Returns
    ///
    /// The [`CnnErrorKind`] attached at construction.
    #[must_use]
    pub const fn kind(&self) -> CnnErrorKind {
        self.kind
    }

    /// Used for retrieving the human-readable diagnostic.
    ///
    /// # Returns
    ///
    /// The message text attached at construction.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CnnError {
    /// Used for writing the context-rich diagnostic.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter for the message text
    ///
    /// # Returns
    ///
    /// The formatter's success or failure result.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CnnError {}

impl From<io::Error> for CnnError {
    /// Used for converting a raw I/O error into the CNN loader's error
    /// vocabulary.
    ///
    /// # Arguments
    ///
    /// * `error` - underlying I/O failure
    ///
    /// # Returns
    ///
    /// A [`CnnErrorKind::Io`] error wrapping the original diagnostic.
    fn from(error: io::Error) -> Self {
        Self::new(CnnErrorKind::Io, format!("LC0J I/O failed: {error}"))
    }
}

/// Shape and size metadata for a validated model.
///
/// Produced by the parser after every dimension has been checked against the
/// residual architecture; the copy stored inside [`CnnModel`] is returned by
/// [`CnnModel::info`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CnnInfo {
    /// Used for reporting the input plane count; v1 models usable here always
    /// contain 112.
    pub input_channels: usize,
    /// Used for reporting the width of the residual trunk.
    pub trunk_channels: usize,
    /// Used for reporting the number of residual blocks.
    pub residual_blocks: usize,
    /// Used for reporting the number of output planes before policy-map
    /// compression.
    pub policy_channels: usize,
    /// Used for reporting the number of channels entering the dense value
    /// head.
    pub value_channels: usize,
    /// Used for reporting the number of mapped policy entries; always 4,672
    /// here.
    pub policy_size: usize,
    /// Used for reporting the total number of floating-point parameters.
    pub parameter_count: usize,
}

/// Policy and value prediction from the side-to-move perspective.
///
/// Returned by [`CnnModel::predict`] and [`CnnModel::predict_encoded`] after
/// running the trunk, the mapped policy head, and the WDL value head.
#[derive(Clone, Debug, PartialEq)]
pub struct CnnPrediction {
    /// Used for storing mapped logits indexed by [`raw_policy_index`].
    pub policy: Vec<f32>,
    /// Used for storing probabilities ordered win, draw, loss.
    pub wdl: [f32; WDL_OUTPUTS],
    /// Used for storing the expected result `win - loss`.
    pub value: f32,
}

/// Precomputed valid input squares and kernel cells for padded convolution.
///
/// For output square `s`, entries in `start[s]..start[s + 1]` pair an input
/// square with its row-major kernel index. Off-board padding has no entry and
/// therefore contributes zero without a branch in the innermost channel loop.
#[derive(Debug)]
struct KernelNeighbors {
    /// Used for storing prefix offsets into `entries` for all 64 outputs.
    start: [usize; BOARD_SQUARES + 1],
    /// Used for storing paired in-bounds input squares and row-major kernel
    /// cells.
    entries: Vec<KernelNeighbor>,
}

/// One already-validated padded-convolution input lookup.
///
/// Pairs the channel-local input square that contributes to an output square
/// with the row-major kernel cell whose weight scales that contribution.
#[derive(Clone, Copy, Debug)]
struct KernelNeighbor {
    /// Used for storing the in-bounds channel-local input square.
    square: usize,
    /// Used for storing the row-major kernel cell paired with the input
    /// square.
    kernel_index: usize,
}

impl KernelNeighbors {
    /// Used for building the zero-padding lookup for an odd square kernel.
    ///
    /// Walks all 64 output squares and every kernel cell, recording only the
    /// input squares that stay on the 8x8 board so that off-board padding
    /// needs no branch during inference.
    ///
    /// # Arguments
    ///
    /// * `kernel` - odd square kernel width (three in practice)
    ///
    /// # Returns
    ///
    /// A lookup table covering all 64 output squares.
    ///
    /// # Panics
    ///
    /// Panics only if a board or kernel coordinate fails to fit `i32` or a
    /// validated square is negative, which cannot happen for the 8x8 board
    /// and the accepted kernel widths.
    fn new(kernel: usize) -> Self {
        let pad = kernel / 2;
        let mut start = [0_usize; BOARD_SQUARES + 1];
        let mut entries = Vec::with_capacity(BOARD_SQUARES * kernel * kernel);
        for (output_square, start_entry) in start.iter_mut().take(BOARD_SQUARES).enumerate() {
            *start_entry = entries.len();
            let row = i32::try_from(output_square / 8).expect("board row fits i32");
            let file = i32::try_from(output_square % 8).expect("board file fits i32");
            for kernel_row in 0..kernel {
                let input_row = row + i32::try_from(kernel_row).expect("kernel row fits i32")
                    - i32::try_from(pad).expect("kernel padding fits i32");
                if !(0..8).contains(&input_row) {
                    continue;
                }
                for kernel_file in 0..kernel {
                    let input_file = file
                        + i32::try_from(kernel_file).expect("kernel file fits i32")
                        - i32::try_from(pad).expect("kernel padding fits i32");
                    if !(0..8).contains(&input_file) {
                        continue;
                    }
                    let input_square = input_row * 8 + input_file;
                    entries.push(KernelNeighbor {
                        square: usize::try_from(input_square)
                            .expect("validated board square is nonnegative"),
                        kernel_index: kernel_row * kernel + kernel_file,
                    });
                }
            }
        }
        start[BOARD_SQUARES] = entries.len();
        Self { start, entries }
    }
}

/// Validated channel-major convolution layer.
///
/// Weights are stored row-major as `[output][input][kernel_row][kernel_file]`
/// with one bias per output channel. Kernel widths of one and three are the
/// only shapes accepted by the v1 parser.
#[derive(Debug)]
struct ConvLayer {
    /// Used for recording the number of `[64]` planes consumed by the layer.
    input_channels: usize,
    /// Used for recording the number of `[64]` planes produced by the layer.
    output_channels: usize,
    /// Used for recording the square kernel width; the v1 backend accepts one
    /// or three.
    kernel: usize,
    /// Used for storing row-major `[output][input][kernel_row][kernel_file]`
    /// coefficients.
    weights: Vec<f32>,
    /// Used for storing one additive value per output channel.
    bias: Vec<f32>,
    /// Used for storing the padding lookup for 3x3 kernels; absent for the
    /// direct 1x1 path.
    neighbors: Option<KernelNeighbors>,
}

impl ConvLayer {
    /// Used for computing the convolution only; callers apply bias and
    /// activation.
    ///
    /// The parser and workspace constructor establish all slice-size
    /// preconditions. Debug builds assert them at the call boundary. The 1x1
    /// path and single-worker configurations run entirely on the caller
    /// thread; larger teams split disjoint contiguous output-channel ranges
    /// across scoped worker threads while the caller computes the first
    /// chunk, preserving per-output summation order.
    ///
    /// # Arguments
    ///
    /// * `input` - channel-major input planes of exactly
    ///   `input_channels * 64` values
    /// * `output` - destination holding at least `output_channels * 64`
    ///   values
    /// * `inference_threads` - bounded worker-team size requested by the
    ///   model
    ///
    /// # Panics
    ///
    /// Panics only if the layer has zero output channels, which the parser
    /// rejects before construction.
    fn forward_no_bias(&self, input: &[f32], output: &mut [f32], inference_threads: usize) {
        debug_assert_eq!(input.len(), self.input_channels * BOARD_SQUARES);
        debug_assert!(output.len() >= self.output_channels * BOARD_SQUARES);
        let output = &mut output[..self.output_channels * BOARD_SQUARES];
        if self.kernel == 1 {
            self.forward_output_channels(input, output, 0);
            return;
        }
        let worker_count = inference_threads.min(self.output_channels);
        if worker_count <= 1 {
            self.forward_output_channels(input, output, 0);
            return;
        }
        let channels_per_worker = self.output_channels.div_ceil(worker_count);
        let values_per_worker = channels_per_worker * BOARD_SQUARES;
        std::thread::scope(|scope| {
            let mut chunks = output.chunks_mut(values_per_worker).enumerate();
            let (_, caller_output) = chunks.next().expect("convolution has output channels");
            for (chunk_index, output_chunk) in chunks {
                let first_output_channel = chunk_index * channels_per_worker;
                spawn_scoped_or_run(scope, "janus-cnn-convolution", move || {
                    self.forward_output_channels(input, output_chunk, first_output_channel);
                });
            }
            self.forward_output_channels(input, caller_output, 0);
        });
    }

    /// Used for computing a disjoint contiguous range of output channels.
    ///
    /// The 1x1 path performs a direct per-square dot product over input
    /// channels; the padded path walks the precomputed [`KernelNeighbors`]
    /// lookup so off-board cells contribute zero without branching.
    ///
    /// # Arguments
    ///
    /// * `input` - complete channel-major input planes
    /// * `output` - destination slice for this worker's whole-plane channels
    /// * `first_output_channel` - global index of the first channel written
    ///   into `output`
    ///
    /// # Panics
    ///
    /// Panics only if a non-unit convolution lacks its padding lookup, which
    /// the parser always constructs for 3x3 kernels.
    fn forward_output_channels(
        &self,
        input: &[f32],
        output: &mut [f32],
        first_output_channel: usize,
    ) {
        debug_assert_eq!(output.len() % BOARD_SQUARES, 0);
        let output_channels = output.len() / BOARD_SQUARES;
        if self.kernel == 1 {
            for local_output_channel in 0..output_channels {
                let output_channel = first_output_channel + local_output_channel;
                let weight_base = output_channel * self.input_channels;
                let output_base = local_output_channel * BOARD_SQUARES;
                for square in 0..BOARD_SQUARES {
                    let mut sum = 0.0_f32;
                    for input_channel in 0..self.input_channels {
                        sum += input[input_channel * BOARD_SQUARES + square]
                            * self.weights[weight_base + input_channel];
                    }
                    output[output_base + square] = sum;
                }
            }
            return;
        }

        let neighbors = self
            .neighbors
            .as_ref()
            .expect("non-unit convolution has neighbors");
        let kernel_area = self.kernel * self.kernel;
        for local_output_channel in 0..output_channels {
            let output_channel = first_output_channel + local_output_channel;
            let weight_base = output_channel * self.input_channels * kernel_area;
            let output_base = local_output_channel * BOARD_SQUARES;
            for output_square in 0..BOARD_SQUARES {
                let mut sum = 0.0_f32;
                let start = neighbors.start[output_square];
                let end = neighbors.start[output_square + 1];
                for input_channel in 0..self.input_channels {
                    let input_base = input_channel * BOARD_SQUARES;
                    let kernel_base = weight_base + input_channel * kernel_area;
                    for neighbor in &neighbors.entries[start..end] {
                        sum += input[input_base + neighbor.square]
                            * self.weights[kernel_base + neighbor.kernel_index];
                    }
                }
                output[output_base + output_square] = sum;
            }
        }
    }
}

/// Validated row-major affine layer.
///
/// Weights are stored as `[output][input]` rows with one bias value per
/// output. Used by the dense value head after the value convolution is
/// flattened.
#[derive(Debug)]
struct DenseLayer {
    /// Used for recording the number of scalar inputs in each row.
    input_dimension: usize,
    /// Used for recording the number of output rows and bias values.
    output_dimension: usize,
    /// Used for storing row-major `[output][input]` coefficients.
    weights: Vec<f32>,
    /// Used for storing one additive value per output.
    bias: Vec<f32>,
}

impl DenseLayer {
    /// Used for applying the affine transform and an optional `ReLU`.
    ///
    /// Each output is the bias plus the dot product of one weight row with
    /// the input; with `relu` enabled negative sums clamp to zero.
    ///
    /// # Arguments
    ///
    /// * `input` - exactly `input_dimension` scalar values
    /// * `output` - destination holding at least `output_dimension` values
    /// * `relu` - whether to clamp negative outputs to zero
    fn forward(&self, input: &[f32], output: &mut [f32], relu: bool) {
        debug_assert_eq!(input.len(), self.input_dimension);
        debug_assert!(output.len() >= self.output_dimension);
        for (output_index, output_value) in
            output.iter_mut().take(self.output_dimension).enumerate()
        {
            let mut sum = self.bias[output_index];
            let weight_base = output_index * self.input_dimension;
            for (input_index, input_value) in input.iter().enumerate() {
                sum += self.weights[weight_base + input_index] * *input_value;
            }
            *output_value = if relu && sum < 0.0 { 0.0 } else { sum };
        }
    }
}

/// Squeeze/excitation parameters attached to a residual block.
///
/// Stores the two dense projections that turn spatially pooled channel means
/// into a per-channel multiplicative gamma gate and additive beta shift, as
/// applied by [`apply_se`].
#[derive(Debug)]
struct SeUnit {
    /// Used for recording the number of trunk channels pooled and gated.
    channels: usize,
    /// Used for recording the width of the squeezed hidden representation.
    hidden: usize,
    /// Used for storing the row-major `[hidden][channels]` first projection.
    first_weights: Vec<f32>,
    /// Used for storing the bias of the first projection.
    first_bias: Vec<f32>,
    /// Used for storing the row-major `[2 * channels][hidden]` gate
    /// projection.
    second_weights: Vec<f32>,
    /// Used for storing biases for multiplicative gamma followed by additive
    /// beta outputs.
    second_bias: Vec<f32>,
}

/// Two-convolution residual unit with an optional squeeze/excitation branch.
///
/// The forward pass applies the first convolution with bias and `ReLU`, the
/// second convolution, then either the SE gamma/beta combination or a plain
/// residual addition before the final `ReLU`.
#[derive(Debug)]
struct ResidualBlock {
    /// Used for storing the first convolution, followed by bias and `ReLU`.
    first: ConvLayer,
    /// Used for storing the second convolution, combined with the residual
    /// before final `ReLU`.
    second: ConvLayer,
    /// Used for storing the optional channel-wise gamma/beta generator.
    se: Option<SeUnit>,
}

/// Immutable validated parameters for the complete residual network.
///
/// Constructed only by [`parse_weights`] after every tensor and cross-head
/// shape has been checked, so inference code can rely on all dimensions
/// without re-validation.
#[derive(Debug)]
struct Weights {
    /// Used for storing the public shape and parameter-count summary.
    info: CnnInfo,
    /// Used for storing the input projection from 112 planes into the trunk
    /// width.
    input: ConvLayer,
    /// Used for storing the ordered residual trunk blocks.
    blocks: Vec<ResidualBlock>,
    /// Used for storing the first policy-head projection.
    policy_stem: ConvLayer,
    /// Used for storing the projection into mapped policy planes.
    policy_output: ConvLayer,
    /// Used for storing the spatial projection for the value head.
    value_conv: ConvLayer,
    /// Used for storing the dense hidden value layer after flattening.
    value_first: DenseLayer,
    /// Used for storing the three-logit WDL output layer.
    value_output: DenseLayer,
    /// Used for mapping raw 73-plane indices to policy-plane offsets; `-1`
    /// marks no output.
    policy_map: Vec<i32>,
    /// Used for recording the largest SE hidden width, sizing one shared
    /// scratch buffer.
    max_se_hidden: usize,
}

/// Reusable buffers sized exactly from a validated [`Weights`] instance.
///
/// Keeping these allocations with the model makes repeated scalar inference
/// allocation-free apart from the returned policy vector. Buffers are
/// channel-major and may be overwritten by every prediction.
#[derive(Debug)]
struct Workspace {
    /// Used for storing the classical `[input_channel][square]` input planes.
    encoded: Vec<f32>,
    /// Used for storing the current trunk activation.
    current: Vec<f32>,
    /// Used for storing the next trunk activation before the residual-buffer
    /// swap.
    next: Vec<f32>,
    /// Used for storing the first-convolution temporary for a residual block.
    temporary: Vec<f32>,
    /// Used for storing the second-convolution temporary for a residual
    /// block.
    scratch: Vec<f32>,
    /// Used for storing the activated policy-stem planes.
    policy_hidden: Vec<f32>,
    /// Used for storing the raw policy-head output addressed by the model's
    /// map.
    policy_planes: Vec<f32>,
    /// Used for storing the flattened output of the value convolution.
    value_input: Vec<f32>,
    /// Used for storing the dense value-head hidden activation.
    value_hidden: Vec<f32>,
    /// Used for storing the win/draw/loss logits before softmax.
    value_logits: [f32; WDL_OUTPUTS],
    /// Used for storing the spatially pooled SE channel values.
    se_pooled: Vec<f32>,
    /// Used for storing the SE hidden-layer scratch values.
    se_hidden: Vec<f32>,
    /// Used for storing the concatenated SE gamma and beta outputs.
    se_gates: Vec<f32>,
}

/// Used for fallibly reserving a bounded vector before any caller-visible
/// state is published.
///
/// # Arguments
///
/// * `capacity` - exact number of elements the caller may later append
/// * `label` - diagnostic name of the admitted buffer
///
/// # Returns
///
/// An empty vector whose capacity is at least `capacity`.
///
/// # Errors
///
/// Returns [`CnnErrorKind::ResourceLimit`] for capacity overflow or allocator
/// refusal.
fn try_reserved_vec<T>(capacity: usize, label: &str) -> Result<Vec<T>, CnnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|_| {
        CnnError::new(
            CnnErrorKind::ResourceLimit,
            format!("could not reserve {capacity} elements for LC0J {label}"),
        )
    })?;
    Ok(values)
}

/// Used for fallibly allocating one exact zeroed floating-point buffer.
///
/// # Arguments
///
/// * `length` - number of zeroed lanes required
/// * `label` - diagnostic name of the admitted buffer
///
/// # Returns
///
/// A vector containing exactly `length` positive zero values.
///
/// # Errors
///
/// Returns [`CnnErrorKind::ResourceLimit`] for capacity overflow or allocator
/// refusal.
fn try_zeroed_f32(length: usize, label: &str) -> Result<Vec<f32>, CnnError> {
    let mut values = try_reserved_vec(length, label)?;
    values.resize(length, 0.0);
    Ok(values)
}

impl Workspace {
    /// Used for fallibly allocating all inference buffers from
    /// already-validated model dimensions.
    ///
    /// Every buffer is zero-initialized and sized exactly for the trunk,
    /// policy, value, and squeeze/excitation shapes recorded in the parsed
    /// [`Weights`].
    ///
    /// # Arguments
    ///
    /// * `weights` - validated network parameters providing all dimensions
    ///
    /// # Returns
    ///
    /// A workspace ready for repeated allocation-free inference.
    ///
    /// # Errors
    ///
    /// Returns [`CnnErrorKind::ResourceLimit`] when any complete buffer cannot
    /// be admitted.
    fn try_new(weights: &Weights) -> Result<Self, CnnError> {
        let trunk_size = weights.info.trunk_channels * BOARD_SQUARES;
        Ok(Self {
            encoded: try_zeroed_f32(
                weights.info.input_channels * BOARD_SQUARES,
                "encoded input workspace",
            )?,
            current: try_zeroed_f32(trunk_size, "current trunk workspace")?,
            next: try_zeroed_f32(trunk_size, "next trunk workspace")?,
            temporary: try_zeroed_f32(trunk_size, "temporary trunk workspace")?,
            scratch: try_zeroed_f32(trunk_size, "scratch trunk workspace")?,
            policy_hidden: try_zeroed_f32(
                weights.policy_stem.output_channels * BOARD_SQUARES,
                "policy hidden workspace",
            )?,
            policy_planes: try_zeroed_f32(
                weights.info.policy_channels * BOARD_SQUARES,
                "policy plane workspace",
            )?,
            value_input: try_zeroed_f32(
                weights.info.value_channels * BOARD_SQUARES,
                "value input workspace",
            )?,
            value_hidden: try_zeroed_f32(
                weights.value_first.output_dimension,
                "value hidden workspace",
            )?,
            value_logits: [0.0; WDL_OUTPUTS],
            se_pooled: try_zeroed_f32(
                weights.info.trunk_channels,
                "squeeze/excitation pooling workspace",
            )?,
            se_hidden: try_zeroed_f32(
                weights.max_se_hidden,
                "squeeze/excitation hidden workspace",
            )?,
            se_gates: try_zeroed_f32(
                weights.info.trunk_channels * 2,
                "squeeze/excitation gate workspace",
            )?,
        })
    }
}

/// Loaded CPU LC0J policy/value network with reusable inference buffers.
///
/// Convolution arithmetic remains scalar and preserves per-output summation
/// order; an optional bounded worker team computes disjoint output channels.
#[derive(Debug)]
pub struct CnnModel {
    /// Used for storing the immutable network tensors and policy map.
    weights: Weights,
    /// Used for storing model-sized mutable scratch state, requiring
    /// `&mut self` for inference.
    workspace: Workspace,
    /// Used for storing the output-channel workers used inside each
    /// convolution.
    inference_threads: usize,
    /// Used for storing the newest eight real UCI positions ending at the
    /// configured MCTS root.
    root_history: Vec<Position>,
    /// Used for storing the root history extended by descendants selected in
    /// one MCTS simulation.
    active_history: Vec<Position>,
}

impl CnnModel {
    /// Used for loading and validating a little-endian `LC0J` v1 model.
    ///
    /// The file size is checked against [`MAX_MODEL_BYTES`] before the bytes
    /// are read, the read is capped at one byte past the limit to detect
    /// growth, and the buffer is then handed to [`CnnModel::from_bytes`].
    ///
    /// # Arguments
    ///
    /// * `path` - filesystem location of the `LC0J` model file
    ///
    /// # Returns
    ///
    /// A fully validated model with freshly allocated inference buffers.
    ///
    /// # Errors
    ///
    /// Returns [`CnnError`] for I/O failures, an unsupported file family or
    /// version, malformed tensors, or a fixed resource-limit violation.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CnnError> {
        let file = File::open(path.as_ref())?;
        let declared = file.metadata()?.len();
        if declared > MAX_MODEL_BYTES as u64 {
            return Err(CnnError::new(
                CnnErrorKind::ResourceLimit,
                format!("LC0J file is {declared} bytes; limit is {MAX_MODEL_BYTES}"),
            ));
        }
        let capacity = usize::try_from(declared).map_err(|_| {
            CnnError::new(
                CnnErrorKind::ResourceLimit,
                "LC0J file length does not fit this platform",
            )
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|_| {
            CnnError::new(
                CnnErrorKind::ResourceLimit,
                format!("could not reserve {capacity} bytes for LC0J file"),
            )
        })?;
        file.take((MAX_MODEL_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_MODEL_BYTES {
            return Err(CnnError::new(
                CnnErrorKind::ResourceLimit,
                "LC0J file grew beyond the loader limit while reading",
            ));
        }
        Self::from_bytes(&bytes)
    }

    /// Used for parsing an exact LC0J v1 byte sequence.
    ///
    /// On success the parsed `Weights` are paired with a matching
    /// `Workspace`, one inference thread, and empty root and active MCTS
    /// histories.
    ///
    /// # Arguments
    ///
    /// * `bytes` - complete little-endian `LC0J` v1 container
    ///
    /// # Returns
    ///
    /// A fully validated model with freshly allocated inference buffers.
    ///
    /// # Errors
    ///
    /// Returns [`CnnError`] when the byte sequence is oversized, unsupported,
    /// truncated, non-finite, shape-inconsistent, or has trailing data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CnnError> {
        if bytes.len() > MAX_MODEL_BYTES {
            return Err(CnnError::new(
                CnnErrorKind::ResourceLimit,
                format!(
                    "LC0J buffer is {} bytes; limit is {MAX_MODEL_BYTES}",
                    bytes.len()
                ),
            ));
        }
        let weights = parse_weights(bytes)?;
        let workspace = Workspace::try_new(&weights)?;
        Ok(Self {
            weights,
            workspace,
            inference_threads: 1,
            root_history: try_reserved_vec(HISTORY, "root history")?,
            active_history: try_reserved_vec(HISTORY, "active history")?,
        })
    }

    /// Used for retrieving validated model metadata.
    ///
    /// # Returns
    ///
    /// The [`CnnInfo`] shape and parameter-count summary recorded at parse
    /// time.
    #[must_use]
    pub const fn info(&self) -> CnnInfo {
        self.weights.info
    }

    /// Used for selecting the bounded output-channel team used by each
    /// convolution.
    ///
    /// Zero is treated as one and values above [`MAX_INFERENCE_THREADS`] are
    /// capped. Requested teams of two or three retain the scalar path because
    /// their scoped-thread setup cost exceeds their measured compute saving
    /// on the pinned 10x128 network.
    ///
    /// # Arguments
    ///
    /// * `threads` - requested worker-team size
    pub fn set_inference_threads(&mut self, threads: usize) {
        let bounded = threads.clamp(1, MAX_INFERENCE_THREADS);
        self.inference_threads = if bounded < MAX_INFERENCE_THREADS {
            1
        } else {
            MAX_INFERENCE_THREADS
        };
    }

    /// Used for retrieving the configured convolution worker count.
    ///
    /// # Returns
    ///
    /// Either one or [`MAX_INFERENCE_THREADS`], as decided by
    /// [`CnnModel::set_inference_threads`].
    #[must_use]
    pub const fn inference_threads(&self) -> usize {
        self.inference_threads
    }

    /// Used for retaining the newest eight real game positions ending at an
    /// MCTS root.
    ///
    /// Passing an empty slice clears temporal context. A later path whose root
    /// differs from the final retained position falls back to that supplied
    /// root, preventing stale UCI history from crossing games or FEN commands.
    ///
    /// # Arguments
    ///
    /// * `history_oldest_to_newest` - real game positions, oldest first,
    ///   ending at the intended MCTS root
    pub fn set_root_history(&mut self, history_oldest_to_newest: &[Position]) {
        self.root_history.clear();
        let first = history_oldest_to_newest.len().saturating_sub(HISTORY);
        self.root_history
            .extend(history_oldest_to_newest[first..].iter().cloned());
        self.active_history.clear();
    }

    /// Used for encoding and evaluating one position, including the full
    /// mapped policy.
    ///
    /// The position is encoded through the FEN-only classical entry point
    /// (repeated-current history) before both heads run.
    ///
    /// # Arguments
    ///
    /// * `position` - position evaluated from the side-to-move perspective
    ///
    /// # Returns
    ///
    /// The mapped policy logits, WDL probabilities, and expected value.
    #[must_use]
    pub fn predict(&mut self, position: &Position) -> CnnPrediction {
        encode_position_into(position, &mut self.workspace.encoded);
        self.predict_current_encoded()
    }

    /// Used for evaluating caller-provided channel-major `[112][64]` planes.
    ///
    /// The input is validated for exact length and finiteness before being
    /// copied into the workspace and run through both heads.
    ///
    /// # Arguments
    ///
    /// * `encoded` - channel-major classical input planes
    ///
    /// # Returns
    ///
    /// The mapped policy logits, WDL probabilities, and expected value.
    ///
    /// # Errors
    ///
    /// Returns [`CnnError`] when the input has the wrong length or contains a
    /// non-finite value.
    pub fn predict_encoded(&mut self, encoded: &[f32]) -> Result<CnnPrediction, CnnError> {
        let expected = self.weights.info.input_channels * BOARD_SQUARES;
        if encoded.len() != expected {
            return Err(CnnError::new(
                CnnErrorKind::InvalidFormat,
                format!(
                    "encoded LC0J input contains {} floats; expected {expected}",
                    encoded.len()
                ),
            ));
        }
        if encoded.iter().any(|value| !value.is_finite()) {
            return Err(CnnError::new(
                CnnErrorKind::InvalidFormat,
                "encoded LC0J input contains a non-finite value",
            ));
        }
        self.workspace.encoded.copy_from_slice(encoded);
        Ok(self.predict_current_encoded())
    }

    /// Used for running both heads for the planes already stored in
    /// `workspace.encoded`.
    ///
    /// Executes the trunk, policy head, and value head in order, expands the
    /// mapped policy, and derives the expected value as `win - loss`.
    ///
    /// # Returns
    ///
    /// The complete prediction for the currently encoded planes.
    fn predict_current_encoded(&mut self) -> CnnPrediction {
        run_trunk(&self.weights, &mut self.workspace, self.inference_threads);
        run_policy_head(&self.weights, &mut self.workspace, self.inference_threads);
        let wdl = run_value_head(&self.weights, &mut self.workspace, self.inference_threads);
        let policy = map_policy(&self.workspace.policy_planes, &self.weights.policy_map);
        CnnPrediction {
            policy,
            wdl,
            value: wdl[0] - wdl[2],
        }
    }

    /// Used for running only the value path for alpha-beta evaluation.
    ///
    /// Encodes through the FEN-only entry point and skips the policy head
    /// entirely.
    ///
    /// # Arguments
    ///
    /// * `position` - position evaluated from the side-to-move perspective
    ///
    /// # Returns
    ///
    /// Normalized win/draw/loss probabilities.
    fn evaluate_value(&mut self, position: &Position) -> [f32; WDL_OUTPUTS] {
        encode_position_into(position, &mut self.workspace.encoded);
        run_trunk(&self.weights, &mut self.workspace, self.inference_threads);
        run_value_head(&self.weights, &mut self.workspace, self.inference_threads)
    }

    /// Used for running both heads and extracting finite logits for
    /// representable legal moves.
    ///
    /// When the active MCTS history ends at the queried position its bounded
    /// real-position window feeds the encoder; otherwise the position alone
    /// is encoded with repeated-current history. Moves without a classical
    /// policy representation, moves the model's map marks with `-1`, and
    /// non-finite logits are all skipped.
    ///
    /// # Arguments
    ///
    /// * `position` - position evaluated from the side-to-move perspective
    /// * `legal_moves` - legal moves whose logits should be extracted
    ///
    /// # Returns
    ///
    /// A [`PolicyValue`] holding `win - loss`, the draw probability, and the
    /// per-move logits.
    ///
    /// # Panics
    ///
    /// Panics only if a negative policy-map entry survives the preceding
    /// skip, which the explicit `mapped < 0` check prevents.
    fn evaluate_policy_and_value(
        &mut self,
        position: &Position,
        legal_moves: &[Move],
    ) -> PolicyValue {
        if self.active_history.last() == Some(position) {
            encode_history_into(&self.active_history, &mut self.workspace.encoded);
        } else {
            encode_position_into(position, &mut self.workspace.encoded);
        }
        run_trunk(&self.weights, &mut self.workspace, self.inference_threads);
        run_policy_head(&self.weights, &mut self.workspace, self.inference_threads);
        let wdl = run_value_head(&self.weights, &mut self.workspace, self.inference_threads);
        let mut logits = Vec::with_capacity(legal_moves.len());
        for &mv in legal_moves {
            let Some(raw_index) = raw_policy_index(position, mv) else {
                continue;
            };
            let mapped = self.weights.policy_map[raw_index];
            if mapped < 0 {
                continue;
            }
            let mapped = usize::try_from(mapped).expect("negative policy map was skipped");
            let logit = self.workspace.policy_planes[mapped];
            if logit.is_finite() {
                logits.push((mv, logit));
            }
        }
        PolicyValue::with_logits(wdl[0] - wdl[2], wdl[1], logits)
    }

    /// Used for resetting the active MCTS window to a matching configured
    /// root history.
    ///
    /// When the supplied root equals the newest retained root-history
    /// position the whole retained window is copied; otherwise the window
    /// restarts from the supplied root alone, discarding stale history.
    ///
    /// # Arguments
    ///
    /// * `root` - root position of the MCTS path about to be walked
    fn begin_path(&mut self, root: &Position) {
        self.active_history.clear();
        if self.root_history.last() == Some(root) {
            self.active_history
                .extend(self.root_history.iter().cloned());
        } else {
            self.active_history.push(root.clone());
        }
    }

    /// Used for appending one descendant while retaining the network's
    /// eight-slot window.
    ///
    /// A position equal to the current newest entry is ignored; otherwise the
    /// oldest entry is dropped once the window holds eight positions.
    ///
    /// # Arguments
    ///
    /// * `position` - descendant position selected by the current simulation
    fn append_path_position(&mut self, position: &Position) {
        if self.active_history.last() == Some(position) {
            return;
        }
        if self.active_history.len() == HISTORY {
            self.active_history.remove(0);
        }
        self.active_history.push(position.clone());
    }
}

impl Evaluator for CnnModel {
    /// Used for converting the value head's WDL distribution to a centipawn
    /// score.
    ///
    /// # Arguments
    ///
    /// * `position` - position evaluated from the side-to-move perspective
    ///
    /// # Returns
    ///
    /// A bounded Elo-logit centipawn score.
    fn evaluate(&mut self, position: &Position) -> i32 {
        wdl_to_centipawns(self.evaluate_value(position))
    }

    /// Used for returning a normalized value plus logits for representable
    /// legal moves.
    ///
    /// # Arguments
    ///
    /// * `position` - position evaluated from the side-to-move perspective
    /// * `legal_moves` - legal moves whose logits should be extracted
    ///
    /// # Returns
    ///
    /// A [`PolicyValue`] holding `win - loss`, the draw probability, and the
    /// per-move logits.
    fn evaluate_policy_value(&mut self, position: &Position, legal_moves: &[Move]) -> PolicyValue {
        self.evaluate_policy_and_value(position, legal_moves)
    }

    /// Used for rebuilding the bounded temporal window at the start of one
    /// MCTS path.
    ///
    /// # Arguments
    ///
    /// * `root` - root position of the MCTS path about to be walked
    fn begin_mcts_path(&mut self, root: &Position) {
        self.begin_path(root);
    }

    /// Used for adding one selected descendant to the active temporal window.
    ///
    /// # Arguments
    ///
    /// * `position` - descendant position selected by the current simulation
    fn mcts_path_position(&mut self, position: &Position) {
        self.append_path_position(position);
    }
}

/// Used for encoding a position as classical LC0 `[112][64]` channel-major
/// planes.
///
/// Janus, like the Java core, numbers board bits from `A8 = 0`; the model
/// planes number them from `a1 = 0`. With black to move ranks are mirrored and
/// colors swapped. This FEN-only entry point repeats the current placement in
/// all eight history slots; MCTS uses its bounded real-position transcript.
/// Repetition planes remain zero because LC0 repetition flags are unavailable.
///
/// # Arguments
///
/// * `position` - position encoded from the side-to-move perspective
///
/// # Returns
///
/// A freshly allocated vector of `112 * 64` plane values.
#[must_use]
pub fn encode_position(position: &Position) -> Vec<f32> {
    let mut encoded = vec![0.0; INPUT_CHANNELS * BOARD_SQUARES];
    encode_position_into(position, &mut encoded);
    encoded
}

/// Used for writing the classical encoder into an existing exact-size
/// workspace slice.
///
/// Delegates to [`encode_history_into`] with a single-position window, which
/// repeats the current placement across all eight history slots.
///
/// # Arguments
///
/// * `position` - position encoded from the side-to-move perspective
/// * `encoded` - destination slice of exactly `112 * 64` values
fn encode_position_into(position: &Position, encoded: &mut [f32]) {
    encode_history_into(std::slice::from_ref(position), encoded);
}

/// Used for writing newest-first placement planes from an oldest-to-newest
/// position window.
///
/// Missing older slots repeat the oldest supplied placement. Every placement
/// uses the newest position's side-to-move perspective; auxiliary planes also
/// describe only that newest position. LC0 repetition flags remain unavailable
/// and therefore retain the zeroes installed before encoding.
///
/// # Arguments
///
/// * `history_oldest_to_newest` - non-empty position window, oldest first
/// * `encoded` - destination slice of exactly `112 * 64` values
///
/// # Panics
///
/// Panics only if the history window is empty, which the debug assertion and
/// all callers rule out.
fn encode_history_into(history_oldest_to_newest: &[Position], encoded: &mut [f32]) {
    debug_assert_eq!(encoded.len(), INPUT_CHANNELS * BOARD_SQUARES);
    debug_assert!(!history_oldest_to_newest.is_empty());
    encoded.fill(0.0);
    let current = history_oldest_to_newest
        .last()
        .expect("CNN history contains at least one position");
    let black_to_move = current.side_to_move() == Color::Black;
    for slot in 0..HISTORY {
        let history_index = history_oldest_to_newest.len().saturating_sub(slot + 1);
        let position = &history_oldest_to_newest[history_index];
        let base = slot * PLANES_PER_HISTORY;
        for kind in PieceKind::ALL {
            let (ours, theirs) = if black_to_move {
                (
                    position.piece_bitboard(Piece::new(Color::Black, kind)),
                    position.piece_bitboard(Piece::new(Color::White, kind)),
                )
            } else {
                (
                    position
                        .piece_bitboard(Piece::new(Color::White, kind))
                        .swap_bytes(),
                    position
                        .piece_bitboard(Piece::new(Color::Black, kind))
                        .swap_bytes(),
                )
            };
            write_bits(encoded, base + kind.index(), ours);
            write_bits(encoded, base + 6 + kind.index(), theirs);
        }
    }

    let rights = current.castling_rights();
    let (our_queenside, our_kingside, their_queenside, their_kingside) = if black_to_move {
        (
            CastlingRights::BLACK_QUEENSIDE,
            CastlingRights::BLACK_KINGSIDE,
            CastlingRights::WHITE_QUEENSIDE,
            CastlingRights::WHITE_KINGSIDE,
        )
    } else {
        (
            CastlingRights::WHITE_QUEENSIDE,
            CastlingRights::WHITE_KINGSIDE,
            CastlingRights::BLACK_QUEENSIDE,
            CastlingRights::BLACK_KINGSIDE,
        )
    };
    for (plane, right) in [
        (AUX_BASE, our_queenside),
        (AUX_BASE + 1, our_kingside),
        (AUX_BASE + 2, their_queenside),
        (AUX_BASE + 3, their_kingside),
    ] {
        if rights.contains(right) {
            fill_plane(encoded, plane, 1.0);
        }
    }
    if black_to_move {
        fill_plane(encoded, AUX_BASE + 4, 1.0);
    }
    fill_plane(encoded, AUX_BASE + 5, f32::from(current.halfmove_clock()));
    // Plane 110 is unused. Plane 111 marks the board edge/interior constant.
    fill_plane(encoded, AUX_BASE + 7, 1.0);
}

/// Used for writing all set squares of one bitboard as unit values in a
/// plane.
///
/// Clears the lowest set bit each iteration until the bitboard is exhausted.
///
/// # Arguments
///
/// * `encoded` - channel-major destination planes
/// * `plane` - plane index receiving the unit values
/// * `bits` - bitboard whose set squares are written
fn write_bits(encoded: &mut [f32], plane: usize, mut bits: u64) {
    let base = plane * BOARD_SQUARES;
    while bits != 0 {
        let square = bits.trailing_zeros() as usize;
        encoded[base + square] = 1.0;
        bits &= bits - 1;
    }
}

/// Used for filling one channel-major 64-square plane with a scalar feature
/// value.
///
/// # Arguments
///
/// * `encoded` - channel-major destination planes
/// * `plane` - plane index to fill
/// * `value` - scalar written to all 64 squares of the plane
fn fill_plane(encoded: &mut [f32], plane: usize, value: f32) {
    let start = plane * BOARD_SQUARES;
    encoded[start..start + BOARD_SQUARES].fill(value);
}

/// Used for computing the classical 73-plane index for a move, from the
/// side-to-move perspective, or `None` when that move has no representation.
///
/// Ranks are flipped for black so both sides share one orientation.
/// Underpromotions to knight, bishop, or rook occupy planes 64..73; queen
/// promotions fall through to the ordinary slide encoding. Knight jumps use
/// planes 56..64 in `KNIGHT_DELTAS` order, and the remaining queen-style
/// moves use planes 0..56 keyed by direction and distance one through seven.
///
/// # Arguments
///
/// * `position` - position providing the side to move
/// * `mv` - move whose classical index is requested
///
/// # Returns
///
/// The raw policy index in `0..4_672`, or `None` for unrepresentable moves.
#[must_use]
pub fn raw_policy_index(position: &Position, mv: Move) -> Option<usize> {
    let from = mv.from();
    let to = mv.to();
    let from_file = i32::from(from.file());
    let to_file = i32::from(to.file());
    let mut from_rank = i32::from(from.rank()) - 1;
    let mut to_rank = i32::from(to.rank()) - 1;
    if position.side_to_move() == Color::Black {
        from_rank = 7 - from_rank;
        to_rank = 7 - to_rank;
    }
    let file_delta = to_file - from_file;
    let rank_delta = to_rank - from_rank;
    let from_square = usize::try_from(from_rank * 8 + from_file).ok()?;

    if let Some(promotion) = mv.promotion() {
        if promotion != PieceKind::Queen {
            let piece_index = match promotion {
                PieceKind::Knight => 0,
                PieceKind::Bishop => 1,
                PieceKind::Rook => 2,
                _ => return None,
            };
            let direction_index = match (file_delta, rank_delta) {
                (0, 1) => 0,
                (-1, 1) => 1,
                (1, 1) => 2,
                _ => return None,
            };
            let plane = 64 + piece_index * 3 + direction_index;
            return Some(plane * BOARD_SQUARES + from_square);
        }
    }

    if let Some(knight) = KNIGHT_DELTAS
        .iter()
        .position(|delta| *delta == (file_delta, rank_delta))
    {
        return Some((56 + knight) * BOARD_SQUARES + from_square);
    }

    let (direction, distance) = if file_delta == 0 && rank_delta != 0 {
        (usize::from(rank_delta < 0), rank_delta.abs())
    } else if rank_delta == 0 && file_delta != 0 {
        (2 + usize::from(file_delta < 0), file_delta.abs())
    } else if file_delta != 0 && file_delta.abs() == rank_delta.abs() {
        let direction = match (file_delta > 0, rank_delta > 0) {
            (true, true) => 4,
            (false, true) => 5,
            (true, false) => 6,
            (false, false) => 7,
        };
        (direction, file_delta.abs())
    } else {
        return None;
    };
    if !(1..=7).contains(&distance) {
        return None;
    }
    let plane = direction * 7 + usize::try_from(distance - 1).ok()?;
    Some(plane * BOARD_SQUARES + from_square)
}

/// Used for executing the input projection and every residual block in
/// serialization order.
///
/// Each block applies its first convolution with bias and `ReLU`, its second
/// convolution, then either the squeeze/excitation combination or the plain
/// residual addition, before the current and next trunk buffers swap.
///
/// # Arguments
///
/// * `weights` - validated network parameters
/// * `workspace` - reusable buffers holding the encoded input and trunk
///   activations
/// * `inference_threads` - bounded worker-team size for each convolution
fn run_trunk(weights: &Weights, workspace: &mut Workspace, inference_threads: usize) {
    weights.input.forward_no_bias(
        &workspace.encoded,
        &mut workspace.current,
        inference_threads,
    );
    add_bias_relu(
        &mut workspace.current,
        &weights.input.bias,
        weights.input.output_channels,
    );
    for block in &weights.blocks {
        block.first.forward_no_bias(
            &workspace.current,
            &mut workspace.temporary,
            inference_threads,
        );
        add_bias_relu(
            &mut workspace.temporary,
            &block.first.bias,
            block.first.output_channels,
        );
        block.second.forward_no_bias(
            &workspace.temporary,
            &mut workspace.scratch,
            inference_threads,
        );
        if let Some(se) = &block.se {
            apply_se(
                se,
                &block.second.bias,
                &workspace.current,
                &workspace.scratch,
                &mut workspace.next,
                &mut workspace.se_pooled,
                &mut workspace.se_hidden,
                &mut workspace.se_gates,
            );
        } else {
            add_residual_relu(
                &workspace.scratch,
                &block.second.bias,
                &workspace.current,
                &mut workspace.next,
            );
        }
        std::mem::swap(&mut workspace.current, &mut workspace.next);
    }
}

/// Used for executing the two-convolution policy head into `policy_planes`.
///
/// The policy stem applies bias and `ReLU`; the policy output applies bias
/// without an activation so the planes remain raw logits for the map.
///
/// # Arguments
///
/// * `weights` - validated network parameters
/// * `workspace` - reusable buffers holding the trunk output and policy
///   planes
/// * `inference_threads` - bounded worker-team size for each convolution
fn run_policy_head(weights: &Weights, workspace: &mut Workspace, inference_threads: usize) {
    weights.policy_stem.forward_no_bias(
        &workspace.current,
        &mut workspace.policy_hidden,
        inference_threads,
    );
    add_bias_relu(
        &mut workspace.policy_hidden,
        &weights.policy_stem.bias,
        weights.policy_stem.output_channels,
    );
    weights.policy_output.forward_no_bias(
        &workspace.policy_hidden,
        &mut workspace.policy_planes,
        inference_threads,
    );
    add_bias(
        &mut workspace.policy_planes,
        &weights.policy_output.bias,
        weights.policy_output.output_channels,
    );
}

/// Used for executing the value head and returning normalized win/draw/loss
/// probabilities.
///
/// The value convolution output is flattened into the first dense layer with
/// `ReLU`, projected onto three logits, and normalized by a stable softmax.
///
/// # Arguments
///
/// * `weights` - validated network parameters
/// * `workspace` - reusable buffers holding the trunk output and value
///   activations
/// * `inference_threads` - bounded worker-team size for the value convolution
///
/// # Returns
///
/// Probabilities ordered win, draw, loss that sum to one.
fn run_value_head(
    weights: &Weights,
    workspace: &mut Workspace,
    inference_threads: usize,
) -> [f32; WDL_OUTPUTS] {
    weights.value_conv.forward_no_bias(
        &workspace.current,
        &mut workspace.value_input,
        inference_threads,
    );
    add_bias_relu(
        &mut workspace.value_input,
        &weights.value_conv.bias,
        weights.value_conv.output_channels,
    );
    weights
        .value_first
        .forward(&workspace.value_input, &mut workspace.value_hidden, true);
    weights
        .value_output
        .forward(&workspace.value_hidden, &mut workspace.value_logits, false);
    softmax_three(workspace.value_logits)
}

/// Used for adding a per-channel bias and applying `ReLU` in place.
///
/// # Arguments
///
/// * `values` - channel-major planes updated in place
/// * `bias` - one additive value per channel
/// * `channels` - number of channels to process
fn add_bias_relu(values: &mut [f32], bias: &[f32], channels: usize) {
    for (channel, &channel_bias) in bias.iter().take(channels).enumerate() {
        let base = channel * BOARD_SQUARES;
        for value in &mut values[base..base + BOARD_SQUARES] {
            let sum = *value + channel_bias;
            *value = if sum > 0.0 { sum } else { 0.0 };
        }
    }
}

/// Used for adding a per-channel bias without applying an activation.
///
/// # Arguments
///
/// * `values` - channel-major planes updated in place
/// * `bias` - one additive value per channel
/// * `channels` - number of channels to process
fn add_bias(values: &mut [f32], bias: &[f32], channels: usize) {
    for (channel, &channel_bias) in bias.iter().take(channels).enumerate() {
        let base = channel * BOARD_SQUARES;
        for value in &mut values[base..base + BOARD_SQUARES] {
            *value += channel_bias;
        }
    }
}

/// Used for combining convolution, bias, and residual tensors before a final
/// `ReLU`.
///
/// # Arguments
///
/// * `conv` - biasless second-convolution output of a residual block
/// * `bias` - one additive value per channel
/// * `residual` - block input added back per square
/// * `output` - destination for the activated combined planes
fn add_residual_relu(conv: &[f32], bias: &[f32], residual: &[f32], output: &mut [f32]) {
    for (channel, &channel_bias) in bias.iter().enumerate() {
        let base = channel * BOARD_SQUARES;
        for square in 0..BOARD_SQUARES {
            let value = conv[base + square] + channel_bias + residual[base + square];
            output[base + square] = if value > 0.0 { value } else { 0.0 };
        }
    }
}

/// Used for applying LC0's squeeze/excitation gamma/beta residual formula and
/// the final `ReLU`.
///
/// Channel means of the biased convolution output are squeezed through two
/// dense projections; the first `channels` gate outputs pass through a
/// sigmoid as the multiplicative gamma and the remaining outputs become the
/// additive beta. Each square then computes
/// `relu(gamma * (conv + bias) + residual + beta)`.
///
/// # Arguments
///
/// * `se` - squeeze/excitation parameters for the block
/// * `conv_bias` - second-convolution bias, one value per channel
/// * `residual` - block input added back per square
/// * `conv` - biasless second-convolution output
/// * `output` - destination for the activated combined planes
/// * `pooled` - scratch for spatially pooled channel means
/// * `hidden` - scratch for the squeezed hidden representation
/// * `gates` - scratch for the concatenated gamma and beta outputs
#[allow(clippy::too_many_arguments)]
fn apply_se(
    se: &SeUnit,
    conv_bias: &[f32],
    residual: &[f32],
    conv: &[f32],
    output: &mut [f32],
    pooled: &mut [f32],
    hidden: &mut [f32],
    gates: &mut [f32],
) {
    for channel in 0..se.channels {
        let base = channel * BOARD_SQUARES;
        let mut sum = 0.0_f32;
        for value in &conv[base..base + BOARD_SQUARES] {
            sum += *value;
        }
        pooled[channel] = sum * (1.0 / 64.0) + conv_bias[channel];
    }
    dense_raw(
        &pooled[..se.channels],
        &se.first_weights,
        &se.first_bias,
        &mut hidden[..se.hidden],
        true,
    );
    dense_raw(
        &hidden[..se.hidden],
        &se.second_weights,
        &se.second_bias,
        &mut gates[..se.channels * 2],
        false,
    );
    for channel in 0..se.channels {
        let gamma = sigmoid(gates[channel]);
        let beta = gates[channel + se.channels];
        let base = channel * BOARD_SQUARES;
        for square in 0..BOARD_SQUARES {
            let activated =
                gamma * (conv[base + square] + conv_bias[channel]) + residual[base + square] + beta;
            output[base + square] = if activated > 0.0 { activated } else { 0.0 };
        }
    }
}

/// Used for applying an unwrapped row-major dense transform to caller-sized
/// buffers.
///
/// Unlike [`DenseLayer::forward`], the dimensions come entirely from the
/// slice lengths supplied by the caller.
///
/// # Arguments
///
/// * `input` - scalar input values
/// * `weights` - row-major `[output][input]` coefficients
/// * `bias` - one additive value per output
/// * `output` - destination whose length fixes the output dimension
/// * `relu` - whether to clamp negative outputs to zero
fn dense_raw(input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32], relu: bool) {
    for output_index in 0..output.len() {
        let mut sum = bias[output_index];
        let weight_base = output_index * input.len();
        for input_index in 0..input.len() {
            sum += weights[weight_base + input_index] * input[input_index];
        }
        output[output_index] = if relu && sum < 0.0 { 0.0 } else { sum };
    }
}

/// Used for computing the logistic gate used by squeeze/excitation.
///
/// # Arguments
///
/// * `value` - raw gate logit
///
/// # Returns
///
/// `1 / (1 + e^-value)` between zero and one; large-magnitude inputs
/// saturate to exactly zero or one in `f32` arithmetic.
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Used for computing a stable three-way softmax, falling back to uniform on
/// bad arithmetic.
///
/// The maximum logit is subtracted before exponentiation. A non-finite
/// maximum, a non-positive exponent sum, or a non-finite sum all yield the
/// uniform one-third distribution instead of propagating bad values.
///
/// # Arguments
///
/// * `logits` - raw win/draw/loss logits
///
/// # Returns
///
/// Probabilities that sum to one, or the uniform fallback.
fn softmax_three(logits: [f32; WDL_OUTPUTS]) -> [f32; WDL_OUTPUTS] {
    let maximum = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return [1.0 / 3.0; WDL_OUTPUTS];
    }
    let mut result = [0.0_f32; WDL_OUTPUTS];
    let mut sum = 0.0_f32;
    for index in 0..WDL_OUTPUTS {
        let value = (logits[index] - maximum).exp();
        if value.is_finite() {
            result[index] = value;
            sum += value;
        }
    }
    if !sum.is_finite() || sum <= 0.0 {
        return [1.0 / 3.0; WDL_OUTPUTS];
    }
    for value in &mut result {
        *value /= sum;
    }
    result
}

/// Used for expanding model policy planes into the fixed 4,672-entry raw
/// move space.
///
/// Entries the map marks with `-1` stay zero, and non-finite plane values are
/// replaced with zero rather than propagated.
///
/// # Arguments
///
/// * `planes` - raw policy-head output planes
/// * `policy_map` - validated raw-index to plane-offset map
///
/// # Returns
///
/// A vector with one logit per raw policy index.
///
/// # Panics
///
/// Panics only if a negative map entry reaches the conversion, which the
/// `mapped >= 0` guard prevents.
fn map_policy(planes: &[f32], policy_map: &[i32]) -> Vec<f32> {
    let mut policy = vec![0.0; policy_map.len()];
    for (output, &mapped) in policy.iter_mut().zip(policy_map) {
        if mapped >= 0 {
            let mapped = usize::try_from(mapped).expect("negative policy map was skipped");
            let value = planes[mapped];
            *output = if value.is_finite() { value } else { 0.0 };
        }
    }
    policy
}

/// Used for converting WDL probabilities to Elo-logit centipawns with bounded
/// endpoints.
///
/// The expected score `(win + draw / 2) / sum` is clamped away from zero and
/// one before the 400-per-decade logistic conversion, so the result stays
/// finite; a degenerate distribution maps to zero.
///
/// # Arguments
///
/// * `wdl` - win/draw/loss probabilities, negatives treated as zero
///
/// # Returns
///
/// The rounded centipawn score from the side-to-move perspective.
#[allow(clippy::cast_possible_truncation)]
fn wdl_to_centipawns(wdl: [f32; WDL_OUTPUTS]) -> i32 {
    let win = f64::from(wdl[0].max(0.0));
    let draw = f64::from(wdl[1].max(0.0));
    let loss = f64::from(wdl[2].max(0.0));
    let sum = win + draw + loss;
    if !sum.is_finite() || sum <= 0.0 {
        return 0;
    }
    let expected = ((win + draw * 0.5) / sum).clamp(1.0e-6, 1.0 - 1.0e-6);
    (400.0 * (expected / (1.0 - expected)).log10()).round() as i32
}

/// Used for parsing the complete `LC0J` tensor grammar and enforcing
/// cross-head shapes.
///
/// Reads the magic, version, and header dimensions, then the input
/// convolution, every residual block with its optional squeeze/excitation
/// unit, the policy stem and output, the value convolution and dense layers,
/// and finally the bounds-checked policy map. Parameter counts accumulate
/// under the global limit, and any trailing bytes fail the parse.
///
/// # Arguments
///
/// * `bytes` - complete little-endian `LC0J` v1 container
///
/// # Returns
///
/// Fully validated [`Weights`] ready for inference.
///
/// # Errors
///
/// Returns [`CnnError`] for a wrong magic or version, invalid or oversized
/// dimensions, tensors that mismatch the declared architecture, out-of-range
/// policy-map entries, trailing bytes, or a parameter-count limit violation.
#[allow(clippy::too_many_lines)]
fn parse_weights(bytes: &[u8]) -> Result<Weights, CnnError> {
    let mut reader = LeReader::new(bytes);
    let magic = reader.read_array::<4>("magic")?;
    if &magic != MAGIC {
        return Err(CnnError::new(
            CnnErrorKind::UnsupportedFormat,
            "file is not an LC0J model",
        ));
    }
    let version = reader.read_i32("version")?;
    if version != VERSION {
        return Err(CnnError::new(
            CnnErrorKind::UnsupportedFormat,
            format!("unsupported LC0J version {version}"),
        ));
    }
    let input_channels = reader.read_dimension("input channels", MAX_CHANNELS, false)?;
    if input_channels != INPUT_CHANNELS {
        return Err(CnnError::new(
            CnnErrorKind::UnsupportedFormat,
            format!(
                "LC0J input has {input_channels} channels; classical encoder requires {INPUT_CHANNELS}"
            ),
        ));
    }
    let trunk_channels = reader.read_dimension("trunk channels", MAX_CHANNELS, false)?;
    let residual_blocks = reader.read_dimension("residual blocks", MAX_BLOCKS, true)?;
    let policy_channels = reader.read_dimension("policy channels", MAX_CHANNELS, false)?;
    let value_channels = reader.read_dimension("value channels", MAX_CHANNELS, false)?;
    let value_hidden = reader.read_dimension("value hidden", MAX_DENSE_DIM, false)?;
    let policy_map_length = reader.read_dimension("policy map length", RAW_POLICY_SIZE, false)?;
    if policy_map_length != RAW_POLICY_SIZE {
        return Err(CnnError::new(
            CnnErrorKind::UnsupportedFormat,
            format!("LC0J policy map has {policy_map_length} entries; expected {RAW_POLICY_SIZE}"),
        ));
    }
    let wdl_outputs = reader.read_dimension("WDL outputs", WDL_OUTPUTS, false)?;
    if wdl_outputs != WDL_OUTPUTS {
        return Err(CnnError::new(
            CnnErrorKind::UnsupportedFormat,
            format!("LC0J value head has {wdl_outputs} outputs; expected {WDL_OUTPUTS}"),
        ));
    }

    let input = reader.read_conv("input convolution")?;
    require_conv_shape(&input, input_channels, trunk_channels, "input convolution")?;
    let mut parameter_count = conv_parameter_count(&input)?;
    let mut blocks = Vec::new();
    blocks.try_reserve_exact(residual_blocks).map_err(|_| {
        CnnError::new(
            CnnErrorKind::ResourceLimit,
            format!("could not reserve {residual_blocks} residual blocks"),
        )
    })?;
    let mut max_se_hidden = 0_usize;
    for block_index in 0..residual_blocks {
        let first = reader.read_conv(&format!("residual block {block_index} first convolution"))?;
        require_conv_shape(
            &first,
            trunk_channels,
            trunk_channels,
            "residual first convolution",
        )?;
        let second =
            reader.read_conv(&format!("residual block {block_index} second convolution"))?;
        require_conv_shape(
            &second,
            trunk_channels,
            trunk_channels,
            "residual second convolution",
        )?;
        let se = reader.read_se(trunk_channels, block_index)?;
        parameter_count = checked_parameter_add(parameter_count, conv_parameter_count(&first)?)?;
        parameter_count = checked_parameter_add(parameter_count, conv_parameter_count(&second)?)?;
        if let Some(unit) = &se {
            max_se_hidden = max_se_hidden.max(unit.hidden);
            parameter_count = checked_parameter_add(parameter_count, se_parameter_count(unit)?)?;
        }
        blocks.push(ResidualBlock { first, second, se });
    }

    let policy_stem = reader.read_conv("policy stem")?;
    if policy_stem.input_channels != trunk_channels {
        return Err(shape_error(format!(
            "policy stem input is {}, expected {trunk_channels}",
            policy_stem.input_channels
        )));
    }
    let policy_output = reader.read_conv("policy output")?;
    require_conv_shape(
        &policy_output,
        policy_stem.output_channels,
        policy_channels,
        "policy output",
    )?;
    let value_conv = reader.read_conv("value convolution")?;
    require_conv_shape(
        &value_conv,
        trunk_channels,
        value_channels,
        "value convolution",
    )?;
    let value_first = reader.read_dense("first value dense")?;
    require_dense_shape(
        &value_first,
        checked_product(value_channels, BOARD_SQUARES, "value flatten size")?,
        value_hidden,
        "first value dense",
    )?;
    let value_output = reader.read_dense("WDL dense")?;
    require_dense_shape(&value_output, value_hidden, WDL_OUTPUTS, "WDL dense")?;
    for count in [
        conv_parameter_count(&policy_stem)?,
        conv_parameter_count(&policy_output)?,
        conv_parameter_count(&value_conv)?,
        dense_parameter_count(&value_first)?,
        dense_parameter_count(&value_output)?,
    ] {
        parameter_count = checked_parameter_add(parameter_count, count)?;
    }

    let map_entries = reader.read_dimension("policy map entries", RAW_POLICY_SIZE, false)?;
    if map_entries != policy_map_length {
        return Err(shape_error(format!(
            "policy map payload has {map_entries} entries; header declares {policy_map_length}"
        )));
    }
    let policy_plane_size = checked_product(policy_channels, BOARD_SQUARES, "policy plane size")?;
    let policy_plane_size_i32 = i32::try_from(policy_plane_size).map_err(|_| {
        CnnError::new(
            CnnErrorKind::ResourceLimit,
            "policy plane size does not fit the LC0J index format",
        )
    })?;
    let mut policy_map = Vec::new();
    policy_map.try_reserve_exact(map_entries).map_err(|_| {
        CnnError::new(
            CnnErrorKind::ResourceLimit,
            format!("could not reserve {map_entries} policy-map entries"),
        )
    })?;
    for index in 0..map_entries {
        let mapped = reader.read_i32("policy map value")?;
        if mapped < -1 || mapped >= policy_plane_size_i32 {
            return Err(shape_error(format!(
                "policy map entry {index} is {mapped}; expected -1 or 0..{}",
                policy_plane_size - 1
            )));
        }
        policy_map.push(mapped);
    }
    if !reader.is_at_end() {
        return Err(shape_error(format!(
            "unexpected {} trailing bytes in LC0J model",
            reader.remaining()
        )));
    }
    if parameter_count > MAX_TENSOR_FLOATS {
        return Err(CnnError::new(
            CnnErrorKind::ResourceLimit,
            format!("LC0J contains {parameter_count} parameters; limit is {MAX_TENSOR_FLOATS}"),
        ));
    }
    let info = CnnInfo {
        input_channels,
        trunk_channels,
        residual_blocks,
        policy_channels,
        value_channels,
        policy_size: policy_map_length,
        parameter_count,
    };
    Ok(Weights {
        info,
        input,
        blocks,
        policy_stem,
        policy_output,
        value_conv,
        value_first,
        value_output,
        policy_map,
        max_se_hidden,
    })
}

/// Used for checking a parsed convolution against its required channel
/// dimensions.
///
/// # Arguments
///
/// * `layer` - parsed convolution to check
/// * `expected_input` - required input channel count
/// * `expected_output` - required output channel count
/// * `label` - human-readable layer name for diagnostics
///
/// # Errors
///
/// Returns a shape error when either channel count differs from its
/// expectation.
fn require_conv_shape(
    layer: &ConvLayer,
    expected_input: usize,
    expected_output: usize,
    label: &str,
) -> Result<(), CnnError> {
    if layer.input_channels != expected_input || layer.output_channels != expected_output {
        return Err(shape_error(format!(
            "{label} shape is {}x{}, expected {expected_output}x{expected_input}",
            layer.output_channels, layer.input_channels
        )));
    }
    Ok(())
}

/// Used for checking a parsed dense layer against its required logical
/// dimensions.
///
/// # Arguments
///
/// * `layer` - parsed dense layer to check
/// * `expected_input` - required input dimension
/// * `expected_output` - required output dimension
/// * `label` - human-readable layer name for diagnostics
///
/// # Errors
///
/// Returns a shape error when either dimension differs from its expectation.
fn require_dense_shape(
    layer: &DenseLayer,
    expected_input: usize,
    expected_output: usize,
    label: &str,
) -> Result<(), CnnError> {
    if layer.input_dimension != expected_input || layer.output_dimension != expected_output {
        return Err(shape_error(format!(
            "{label} shape is {}x{}, expected {expected_output}x{expected_input}",
            layer.output_dimension, layer.input_dimension
        )));
    }
    Ok(())
}

/// Used for computing the checked weight-plus-bias count of a convolution.
///
/// # Arguments
///
/// * `layer` - convolution whose parameters are counted
///
/// # Returns
///
/// The combined weight and bias count.
///
/// # Errors
///
/// Returns a resource-limit error when the sum overflows or exceeds
/// [`MAX_TENSOR_FLOATS`].
fn conv_parameter_count(layer: &ConvLayer) -> Result<usize, CnnError> {
    checked_parameter_add(layer.weights.len(), layer.bias.len())
}

/// Used for computing the checked weight-plus-bias count of a dense layer.
///
/// # Arguments
///
/// * `layer` - dense layer whose parameters are counted
///
/// # Returns
///
/// The combined weight and bias count.
///
/// # Errors
///
/// Returns a resource-limit error when the sum overflows or exceeds
/// [`MAX_TENSOR_FLOATS`].
fn dense_parameter_count(layer: &DenseLayer) -> Result<usize, CnnError> {
    checked_parameter_add(layer.weights.len(), layer.bias.len())
}

/// Used for computing the checked total parameter count of a
/// squeeze/excitation unit.
///
/// # Arguments
///
/// * `unit` - squeeze/excitation unit whose parameters are counted
///
/// # Returns
///
/// The combined count of both projections' weights and biases.
///
/// # Errors
///
/// Returns a resource-limit error when any partial sum overflows or exceeds
/// [`MAX_TENSOR_FLOATS`].
fn se_parameter_count(unit: &SeUnit) -> Result<usize, CnnError> {
    let mut count = checked_parameter_add(unit.first_weights.len(), unit.first_bias.len())?;
    count = checked_parameter_add(count, unit.second_weights.len())?;
    checked_parameter_add(count, unit.second_bias.len())
}

/// Used for adding parameter counts while enforcing the global model limit.
///
/// # Arguments
///
/// * `left` - first parameter count
/// * `right` - second parameter count
///
/// # Returns
///
/// The checked sum of both counts.
///
/// # Errors
///
/// Returns a resource-limit error when the addition overflows or the sum
/// exceeds [`MAX_TENSOR_FLOATS`].
fn checked_parameter_add(left: usize, right: usize) -> Result<usize, CnnError> {
    let sum = left.checked_add(right).ok_or_else(|| {
        CnnError::new(CnnErrorKind::ResourceLimit, "LC0J parameter count overflow")
    })?;
    if sum > MAX_TENSOR_FLOATS {
        return Err(CnnError::new(
            CnnErrorKind::ResourceLimit,
            format!("LC0J parameter count {sum} exceeds limit {MAX_TENSOR_FLOATS}"),
        ));
    }
    Ok(sum)
}

/// Used for multiplying tensor dimensions with overflow and resource-limit
/// checks.
///
/// # Arguments
///
/// * `left` - first dimension
/// * `right` - second dimension
/// * `label` - human-readable quantity name for diagnostics
///
/// # Returns
///
/// The checked product of both dimensions.
///
/// # Errors
///
/// Returns a resource-limit error when the multiplication overflows or the
/// product exceeds [`MAX_TENSOR_FLOATS`].
fn checked_product(left: usize, right: usize, label: &str) -> Result<usize, CnnError> {
    let product = left.checked_mul(right).ok_or_else(|| {
        CnnError::new(
            CnnErrorKind::ResourceLimit,
            format!("{label} overflows address space"),
        )
    })?;
    if product > MAX_TENSOR_FLOATS {
        return Err(CnnError::new(
            CnnErrorKind::ResourceLimit,
            format!("{label} is {product}; limit is {MAX_TENSOR_FLOATS}"),
        ));
    }
    Ok(product)
}

/// Used for constructing a consistent malformed-shape error.
///
/// # Arguments
///
/// * `message` - human-readable diagnostic context
///
/// # Returns
///
/// A [`CnnErrorKind::InvalidFormat`] error carrying the message.
fn shape_error(message: impl Into<String>) -> CnnError {
    CnnError::new(CnnErrorKind::InvalidFormat, message)
}

/// Bounds-checked little-endian reader for an in-memory `LC0J` container.
///
/// All tensor readers validate declared lengths before allocation and reject
/// non-finite parameters. A successful top-level parse also checks exact EOF.
struct LeReader<'a> {
    /// Used for holding the complete model bytes.
    bytes: &'a [u8],
    /// Used for tracking the offset of the next unread byte.
    offset: usize,
}

impl<'a> LeReader<'a> {
    /// Used for creating a reader positioned at the model header.
    ///
    /// # Arguments
    ///
    /// * `bytes` - complete model bytes to read from offset zero
    ///
    /// # Returns
    ///
    /// A reader with its offset at the start of the buffer.
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    /// Used for retrieving the number of unread model bytes.
    ///
    /// # Returns
    ///
    /// The count of bytes between the current offset and the buffer end.
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    /// Used for checking whether every model byte was consumed exactly.
    ///
    /// # Returns
    ///
    /// `true` when the offset equals the buffer length.
    fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }

    /// Used for reading one fixed-size field without advancing on truncation.
    ///
    /// # Arguments
    ///
    /// * `label` - human-readable field name for diagnostics
    ///
    /// # Returns
    ///
    /// The next `N` bytes as a fixed array.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the offset arithmetic overflows or fewer
    /// than `N` bytes remain.
    fn read_array<const N: usize>(&mut self, label: &str) -> Result<[u8; N], CnnError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or_else(|| shape_error(format!("offset overflow while reading LC0J {label}")))?;
        let source = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| shape_error(format!("truncated LC0J while reading {label}")))?;
        let mut value = [0_u8; N];
        value.copy_from_slice(source);
        self.offset = end;
        Ok(value)
    }

    /// Used for reading one unsigned byte field.
    ///
    /// # Arguments
    ///
    /// * `label` - human-readable field name for diagnostics
    ///
    /// # Returns
    ///
    /// The next byte.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the buffer is exhausted.
    fn read_u8(&mut self, label: &str) -> Result<u8, CnnError> {
        Ok(self.read_array::<1>(label)?[0])
    }

    /// Used for reading one little-endian signed 32-bit field.
    ///
    /// # Arguments
    ///
    /// * `label` - human-readable field name for diagnostics
    ///
    /// # Returns
    ///
    /// The decoded `i32` value.
    ///
    /// # Errors
    ///
    /// Returns a shape error when fewer than four bytes remain.
    fn read_i32(&mut self, label: &str) -> Result<i32, CnnError> {
        Ok(i32::from_le_bytes(self.read_array::<4>(label)?))
    }

    /// Used for reading one little-endian finite IEEE-754 scalar.
    ///
    /// # Arguments
    ///
    /// * `label` - human-readable field name for diagnostics
    ///
    /// # Returns
    ///
    /// The decoded finite `f32` value.
    ///
    /// # Errors
    ///
    /// Returns a shape error when fewer than four bytes remain or the value
    /// is NaN or infinite.
    fn read_f32(&mut self, label: &str) -> Result<f32, CnnError> {
        let value = f32::from_le_bytes(self.read_array::<4>(label)?);
        if !value.is_finite() {
            return Err(shape_error(format!("non-finite LC0J value in {label}")));
        }
        Ok(value)
    }

    /// Used for reading a nonnegative bounded dimension, optionally
    /// permitting zero.
    ///
    /// # Arguments
    ///
    /// * `label` - human-readable field name for diagnostics
    /// * `maximum` - largest accepted dimension value
    /// * `allow_zero` - whether zero is a valid dimension
    ///
    /// # Returns
    ///
    /// The validated dimension as `usize`.
    ///
    /// # Errors
    ///
    /// Returns a shape error for negative or disallowed-zero values and a
    /// resource-limit error for values above `maximum`.
    fn read_dimension(
        &mut self,
        label: &str,
        maximum: usize,
        allow_zero: bool,
    ) -> Result<usize, CnnError> {
        let raw = self.read_i32(label)?;
        if raw < 0 || (!allow_zero && raw == 0) {
            return Err(shape_error(format!("invalid {label}: {raw}")));
        }
        let value =
            usize::try_from(raw).map_err(|_| shape_error(format!("invalid {label}: {raw}")))?;
        if value > maximum {
            return Err(CnnError::new(
                CnnErrorKind::ResourceLimit,
                format!("LC0J {label} is {value}; limit is {maximum}"),
            ));
        }
        Ok(value)
    }

    /// Used for reading an exactly sized length-prefixed finite-`f32` tensor.
    ///
    /// The declared length must equal the expected count, the remaining bytes
    /// must cover the payload, and every scalar must be finite.
    ///
    /// # Arguments
    ///
    /// * `expected` - exact number of floats the tensor must contain
    /// * `label` - human-readable tensor name for diagnostics
    ///
    /// # Returns
    ///
    /// The decoded tensor values.
    ///
    /// # Errors
    ///
    /// Returns a shape error for a length mismatch, truncation, or a
    /// non-finite scalar, and a resource-limit error for byte-size overflow
    /// or a failed reservation.
    fn read_float_vector(&mut self, expected: usize, label: &str) -> Result<Vec<f32>, CnnError> {
        let declared = self.read_dimension(&format!("{label} length"), MAX_TENSOR_FLOATS, true)?;
        if declared != expected {
            return Err(shape_error(format!(
                "{label} contains {declared} floats; expected {expected}"
            )));
        }
        let byte_count = expected.checked_mul(4).ok_or_else(|| {
            CnnError::new(
                CnnErrorKind::ResourceLimit,
                format!("{label} byte size overflow"),
            )
        })?;
        if byte_count > self.remaining() {
            return Err(shape_error(format!(
                "truncated LC0J while reading {label}: need {byte_count} bytes, have {}",
                self.remaining()
            )));
        }
        let mut values = Vec::new();
        values.try_reserve_exact(expected).map_err(|_| {
            CnnError::new(
                CnnErrorKind::ResourceLimit,
                format!("could not reserve {expected} floats for {label}"),
            )
        })?;
        for _ in 0..expected {
            values.push(self.read_f32(label)?);
        }
        Ok(values)
    }

    /// Used for reading one 1x1 or 3x3 convolution and building its padding
    /// lookup.
    ///
    /// Reads the output channels, input channels, and kernel width, then the
    /// weight and bias tensors sized from those dimensions. The
    /// [`KernelNeighbors`] lookup is built only for non-unit kernels.
    ///
    /// # Arguments
    ///
    /// * `label` - human-readable layer name for diagnostics
    ///
    /// # Returns
    ///
    /// A validated [`ConvLayer`].
    ///
    /// # Errors
    ///
    /// Returns [`CnnError`] for invalid dimensions, an unsupported kernel
    /// width, size overflow, or a malformed weight or bias tensor.
    fn read_conv(&mut self, label: &str) -> Result<ConvLayer, CnnError> {
        let output_channels =
            self.read_dimension(&format!("{label} output channels"), MAX_CHANNELS, false)?;
        let input_channels =
            self.read_dimension(&format!("{label} input channels"), MAX_CHANNELS, false)?;
        let kernel = self.read_dimension(&format!("{label} kernel"), 3, false)?;
        if kernel != 1 && kernel != 3 {
            return Err(CnnError::new(
                CnnErrorKind::UnsupportedFormat,
                format!("{label} uses unsupported {kernel}x{kernel} kernel"),
            ));
        }
        let kernel_area = checked_product(kernel, kernel, &format!("{label} kernel area"))?;
        let weights_count = checked_product(
            checked_product(
                output_channels,
                input_channels,
                &format!("{label} channels"),
            )?,
            kernel_area,
            &format!("{label} weights"),
        )?;
        let weights = self.read_float_vector(weights_count, &format!("{label} weights"))?;
        let bias = self.read_float_vector(output_channels, &format!("{label} bias"))?;
        let neighbors = (kernel != 1).then(|| KernelNeighbors::new(kernel));
        Ok(ConvLayer {
            input_channels,
            output_channels,
            kernel,
            weights,
            bias,
            neighbors,
        })
    }

    /// Used for reading one row-major dense matrix and bias vector.
    ///
    /// Reads the output and input dimensions followed by the weight and bias
    /// tensors sized from them.
    ///
    /// # Arguments
    ///
    /// * `label` - human-readable layer name for diagnostics
    ///
    /// # Returns
    ///
    /// A validated [`DenseLayer`].
    ///
    /// # Errors
    ///
    /// Returns [`CnnError`] for invalid dimensions, size overflow, or a
    /// malformed weight or bias tensor.
    fn read_dense(&mut self, label: &str) -> Result<DenseLayer, CnnError> {
        let output_dimension =
            self.read_dimension(&format!("{label} output dimension"), MAX_DENSE_DIM, false)?;
        let input_dimension =
            self.read_dimension(&format!("{label} input dimension"), MAX_DENSE_DIM, false)?;
        let weights_count = checked_product(
            output_dimension,
            input_dimension,
            &format!("{label} weights"),
        )?;
        let weights = self.read_float_vector(weights_count, &format!("{label} weights"))?;
        let bias = self.read_float_vector(output_dimension, &format!("{label} bias"))?;
        Ok(DenseLayer {
            input_dimension,
            output_dimension,
            weights,
            bias,
        })
    }

    /// Used for reading an optional squeeze/excitation unit for one residual
    /// block.
    ///
    /// A presence byte of zero yields `None`; one introduces the hidden and
    /// channel dimensions followed by both projections' weight and bias
    /// tensors. The declared channel count must match the trunk width.
    ///
    /// # Arguments
    ///
    /// * `channels` - trunk channel count the unit must match
    /// * `block` - residual block index for diagnostics
    ///
    /// # Returns
    ///
    /// The parsed unit, or `None` when the block carries no SE branch.
    ///
    /// # Errors
    ///
    /// Returns [`CnnError`] for an invalid presence byte, invalid dimensions,
    /// a channel mismatch, size overflow, or a malformed tensor.
    fn read_se(&mut self, channels: usize, block: usize) -> Result<Option<SeUnit>, CnnError> {
        let present = self.read_u8(&format!("residual block {block} SE presence"))?;
        if present == 0 {
            return Ok(None);
        }
        if present != 1 {
            return Err(shape_error(format!(
                "residual block {block} has invalid SE presence byte {present}"
            )));
        }
        let hidden = self.read_dimension(
            &format!("residual block {block} SE hidden"),
            MAX_SE_HIDDEN,
            false,
        )?;
        let declared_channels = self.read_dimension(
            &format!("residual block {block} SE channels"),
            MAX_CHANNELS,
            false,
        )?;
        if declared_channels != channels {
            return Err(shape_error(format!(
                "residual block {block} SE has {declared_channels} channels; expected {channels}"
            )));
        }
        let first_count = checked_product(hidden, channels, "SE first weights")?;
        let second_outputs = checked_product(channels, 2, "SE gate count")?;
        let second_count = checked_product(second_outputs, hidden, "SE second weights")?;
        let first_weights = self.read_float_vector(
            first_count,
            &format!("residual block {block} SE first weights"),
        )?;
        let first_bias =
            self.read_float_vector(hidden, &format!("residual block {block} SE first bias"))?;
        let second_weights = self.read_float_vector(
            second_count,
            &format!("residual block {block} SE second weights"),
        )?;
        let second_bias = self.read_float_vector(
            second_outputs,
            &format!("residual block {block} SE second bias"),
        )?;
        Ok(Some(SeUnit {
            channels,
            hidden,
            first_weights,
            first_bias,
            second_weights,
            second_bias,
        }))
    }
}
