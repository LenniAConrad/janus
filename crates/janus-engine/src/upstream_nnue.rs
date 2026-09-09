#![allow(
    // Format dimensions and board coordinates are fixed, range-checked values.
    // Explicit casts keep the scalar port legible and match Java's signed-byte
    // interpretation of serialized int8 weights.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // PSQ/PSQT and attacker/attacked names are intentional NNUE terminology.
    clippy::similar_names
)]

//! Scalar Stockfish `CURRENT/BIG` NNUE loading and inference.
//!
//! This is a deliberately narrow compatibility implementation for the
//! `SFNNv13` `CURRENT/BIG` architecture used by `ChessRTK`'s pinned local model.
//! It is safe, dependency-free, rejects every other architecture hash, and
//! performs complete inference without inference-time allocation. FC0 consumes
//! a sparse list of nonzero transformed features.
//!
//! Alpha-beta workers maintain per-ply `HalfKA` accumulators when a root has at
//! least 12 pieces. `FullThreats` is rebuilt exactly into reusable worker
//! scratch because its occupancy/contact changes are nonlocal; low-material
//! roots use the faster sparse full rebuild. The hot integer kernels run on
//! the compile-time-dispatched `simd` backends: portable builds use the
//! zero-`unsafe` scalar backend, AVX2 and AVX-512 builds the audited
//! intrinsic wrappers, and all are bit-identical. Exact incremental `FullThreats` remains
//! measured future work. These integer-format choices do not
//! describe compact CRTK float NNUE, which remains a separate full-rebuild
//! implementation.
//!
//! The accepted stream is Stockfish's little-endian version/hash/description
//! header followed by one combined `FullThreats + HalfKAv2_hm` feature
//! transformer and eight material-bucket network stacks. Transformer 16- and
//! 32-bit integer arrays use Stockfish's length-delimited signed LEB128
//! blocks; `FullThreats` transform weights and affine weights are signed
//! bytes, the latter stored in padded row-major matrices. Every architecture
//! hash, block boundary, fixed dimension, and final EOF is checked before the
//! model becomes usable.

/// Compile-time-dispatched vector backends for the hot integer kernels.
///
/// Contains the `simd::Backend` interface, the zero-`unsafe` scalar
/// reference backend used by portable builds, and the audited AVX2 and
/// AVX-512 backends selected by compile-time `#[cfg(target_feature)]`
/// dispatch (AVX-512 preferred when `avx512f`/`avx512bw` are enabled); see
/// its module contract for the workspace's single approved `unsafe`
/// boundary.
mod simd;

use crate::evaluator::{Evaluator, SearchEvaluator, SearchStateError};
use janus_core::{Color, Move, Piece, PieceKind, Position, Square, Undo};
use simd::Backend as _;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::Arc;

/// Used for matching the little-endian version field that opens every
/// accepted Stockfish NNUE stream.
///
/// Loading rejects any input whose first four bytes decode to a different
/// value.
pub const UPSTREAM_VERSION: u32 = 0x7af3_2f20;

/// Used for bounding the model bytes this loader will read or parse.
///
/// Files, buffers, and length-delimited blocks beyond this 512 MiB limit
/// are rejected with [`UpstreamNnueErrorKind::ResourceLimit`].
pub const MAX_UPSTREAM_MODEL_BYTES: usize = 512 * 1024 * 1024;

/// Used for converting the network's internal score units to centipawns.
///
/// Both the PSQT and positional halves of a prediction are divided by this
/// value before being reported.
const OUTPUT_SCALE: i32 = 16;
/// Used for shifting fixed-point affine activations back into range.
///
/// Clipped activations and the forward-material lane divide by
/// `1 << WEIGHT_SCALE_BITS` to drop the fractional bits.
const WEIGHT_SCALE_BITS: u32 = 6;
/// Used for counting the material-bucket dense networks serialized in
/// CURRENT/BIG.
///
/// One of the eight stacks is selected per evaluation by [`bucket`].
const LAYER_STACKS: usize = 8;
/// Used for recognizing the marker preceding each length-delimited
/// compressed integer tensor.
///
/// [`Cursor::begin_leb`] requires this exact byte string before every
/// LEB128 block.
const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

/// Used for sizing the sparse `HalfKAv2_hm` feature space.
///
/// Thirty-two horizontally mirrored king buckets, each holding eleven
/// 64-square piece planes.
const HALF_KA_DIMENSIONS: usize = 64 * (11 * 64) / 2;
/// Used for validating the Stockfish type hash of the accepted
/// `HalfKAv2_hm` feature set.
///
/// Enters the combined transformer hash via [`current_big_feature_hash`].
const HALF_KA_HASH: u32 = 0x7f23_4cb8;
/// Used for validating the Stockfish type hash of the accepted
/// `FullThreats` feature set.
///
/// Enters the combined transformer hash via [`current_big_feature_hash`].
const FULL_THREATS_HASH: u32 = 0x8f23_4cb8;
/// Used for sizing the sparse `FullThreats` feature space.
///
/// Also serves as the sentinel index marking unsupported attacker/target
/// combinations in [`ThreatTables`].
const THREAT_DIMENSIONS: usize = 60_720;
/// Used for sizing the combined side-to-move/opponent feature-transform
/// output.
///
/// Every accumulator row and each FC0 input vector has this fixed width.
const TRANSFORMED_DIMENSIONS: usize = 1_024;
/// Used for sizing the half of the transformed features allocated to each
/// perspective.
///
/// The pairwise-clipped product combines lanes `i` and
/// `i + HALF_TRANSFORMED_DIMENSIONS` of one perspective accumulator.
const HALF_TRANSFORMED_DIMENSIONS: usize = TRANSFORMED_DIMENSIONS / 2;
/// Used for counting the non-forward outputs of the first affine layer.
///
/// Index `L2` of the FC0 output is the separate forward-material lane.
const L2: usize = 31;
/// Used for sizing the second affine layer.
///
/// The hidden layer consumes `L2 * 2` activations and produces this many
/// outputs.
const L3: usize = 32;
/// Used for sizing the sparse first affine layer's output, including its
/// forward lane.
const FC0_OUTPUT_DIMENSIONS: usize = L2 + 1;
/// Used for sizing the combined sparse input covered by the serialized
/// PSQT table.
///
/// `FullThreats` rows precede `HalfKA` rows in the combined serialization.
const TOTAL_INPUT_DIMENSIONS: usize = HALF_KA_DIMENSIONS + THREAT_DIMENSIONS;
/// Used for bounding active `HalfKA` features at the maximum occupied
/// squares representable by a chess board.
const MAX_ACTIVE_HALF_KA: usize = 64;
/// Used for bounding occupied contacts emitted by all non-king attackers
/// on one board.
///
/// Each of at most 64 occupied origins can contribute no more than eight
/// contacts (queen rays, knight jumps, or adjacent king squares). Kings are not
/// emitted as `FullThreats` attackers, so 512 is a conservative fixed bound.
const MAX_ACTIVE_THREATS: usize = 64 * 8;
/// Used for gating per-ply `HalfKA` copies on sufficient root material.
///
/// Below twelve pieces, copying two 1,024-lane parent accumulators costs more
/// than rebuilding the few active rows on the measured endgame corpus.
const MIN_INCREMENTAL_PIECES: u32 = 12;

/// Used for selecting White in Stockfish piece-code and perspective
/// arithmetic.
const WHITE: usize = 0;
/// Used for selecting Black in Stockfish piece-code and perspective
/// arithmetic.
const BLACK: usize = 1;
/// Used for encoding the pawn type component of a Stockfish piece code.
const PAWN: usize = 1;
/// Used for encoding the knight type component of a Stockfish piece code.
const KNIGHT: usize = 2;
/// Used for encoding the bishop type component of a Stockfish piece code.
const BISHOP: usize = 3;
/// Used for encoding the rook type component of a Stockfish piece code.
const ROOK: usize = 4;
/// Used for encoding the queen type component of a Stockfish piece code.
const QUEEN: usize = 5;
/// Used for encoding the king type component of a Stockfish piece code.
const KING: usize = 6;

/// Used for iterating every non-empty Stockfish piece code in color/type
/// order.
///
/// Drives the deterministic construction of [`ThreatTables`].
const ALL_PIECES: [usize; 12] = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14];
/// Used for counting the attacked-piece class lanes available to each
/// attacker piece code.
///
/// Indexed by Stockfish piece code; kings and empty codes have no lanes.
const NUM_VALID_TARGETS: [usize; 16] = [0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0];
/// Used for mapping attacker and attacked piece types to `FullThreats`
/// class offsets.
///
/// Rows are indexed by `attacker_type - 1` and columns by
/// `attacked_type - 1`; `-1` excludes an unsupported combination.
const THREAT_MAP: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

/// Used for enumerating knight target deltas in the order that reproduces
/// Stockfish pseudo-attack ordering.
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
/// Used for enumerating all adjacent king target deltas.
const KING_DELTAS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
/// Used for tracing the diagonal ray directions of bishops.
const BISHOP_DIRECTIONS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
/// Used for tracing the orthogonal ray directions of rooks.
const ROOK_DIRECTIONS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
/// Used for tracing queen ray directions, diagonals before orthogonals.
const QUEEN_DIRECTIONS: [(i32, i32); 8] = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
];

/// Stable failure category for upstream model loading.
///
/// Callers can match on this instead of parsing diagnostic text; the
/// variants are deliberately coarse and stable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpstreamNnueErrorKind {
    /// Used for indicating a filesystem or stream failure.
    Io,
    /// Used for indicating truncated, inconsistent, or otherwise malformed
    /// serialization.
    InvalidFormat,
    /// Used for indicating a well-formed Stockfish header for an
    /// architecture not implemented here.
    UnsupportedArchitecture,
    /// Used for indicating a file or requested allocation that exceeds a
    /// fixed bound.
    ResourceLimit,
}

/// Error returned by [`UpstreamNnue`] loading.
///
/// Pairs a machine-stable [`UpstreamNnueErrorKind`] with human-readable
/// diagnostic text naming the rejected field or operation.
#[derive(Debug)]
pub struct UpstreamNnueError {
    /// Used for programmatic handling through the stable category.
    kind: UpstreamNnueErrorKind,
    /// Used for human-readable detail identifying the rejected field or
    /// operation.
    message: String,
}

impl UpstreamNnueError {
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
    /// A loader error carrying both values.
    fn new(kind: UpstreamNnueErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Used for retrieving the machine-stable category.
    ///
    /// # Returns
    ///
    /// The failure category recorded at construction.
    #[must_use]
    pub const fn kind(&self) -> UpstreamNnueErrorKind {
        self.kind
    }

    /// Used for retrieving the diagnostic detail.
    ///
    /// # Returns
    ///
    /// The human-readable message recorded at construction.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for UpstreamNnueError {
    /// Used for writing the context-rich loader diagnostic.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Errors
    ///
    /// Propagates any failure reported by the formatter.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for UpstreamNnueError {}

impl From<io::Error> for UpstreamNnueError {
    /// Used for converting a raw I/O failure into the upstream-loader
    /// vocabulary.
    ///
    /// # Arguments
    ///
    /// * `error` - underlying I/O failure
    ///
    /// # Returns
    ///
    /// An [`UpstreamNnueErrorKind::Io`] error wrapping the original
    /// diagnostic.
    fn from(error: io::Error) -> Self {
        Self::new(
            UpstreamNnueErrorKind::Io,
            format!("Stockfish NNUE I/O failed: {error}"),
        )
    }
}

/// Metadata for a loaded CURRENT/BIG model.
///
/// Captures the fixed architecture dimensions together with the hash and
/// description read from the serialized header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpstreamNnueInfo {
    /// Used for reporting the format label; currently always `SFNNv13`.
    pub variant: &'static str,
    /// Used for reporting the network size; currently always `BIG`.
    pub size: &'static str,
    /// Used for reporting the total sparse feature dimensions.
    pub input_features: usize,
    /// Used for reporting the feature-transform output width.
    pub transformed_dimensions: usize,
    /// Used for reporting the first dense hidden width (not including the
    /// forward lane).
    pub l2: usize,
    /// Used for reporting the second dense hidden width.
    pub l3: usize,
    /// Used for reporting the architecture hash stored in the file header.
    pub hash: u32,
    /// Used for reporting the UTF-8 model description.
    pub description: String,
}

/// Decomposed Stockfish evaluation, from the side-to-move perspective.
///
/// Both halves are already divided down to centipawns; their sum is the
/// complete score returned by [`UpstreamPrediction::centipawns`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UpstreamPrediction {
    /// Used for the material/PSQT contribution in centipawns.
    pub psqt: i32,
    /// Used for the dense-network contribution in centipawns.
    pub positional: i32,
}

impl UpstreamPrediction {
    /// Used for computing the complete side-to-move score.
    ///
    /// # Returns
    ///
    /// The sum of the PSQT and positional contributions in centipawns.
    #[must_use]
    pub const fn centipawns(self) -> i32 {
        self.psqt + self.positional
    }
}

/// Safe scalar evaluator for Stockfish CURRENT/BIG NNUE files.
///
/// Owns every validated tensor of one loaded model together with the
/// deterministically rebuilt `FullThreats` lookup tables; all inference
/// entry points borrow this state immutably.
pub struct UpstreamNnue {
    /// Used for retaining the UTF-8 description copied from the serialized
    /// header.
    description: String,
    /// Used for retaining the validated combined feature-transformer and
    /// architecture hash.
    network_hash: u32,
    /// Used for the sparse HalfKA/FullThreats transform and PSQT
    /// parameters.
    transformer: FeatureTransformer,
    /// Used for the eight dense networks selected by material count.
    stacks: Vec<Architecture>,
    /// Used for the deterministically rebuilt feature-index lookup tables.
    threat_tables: ThreatTables,
}

/// Search-facing upstream NNUE evaluator with worker-local incremental state.
///
/// Large immutable model tensors are shared through [`Arc`]. Cloning this
/// wrapper deliberately starts with no search state, so every Lazy-SMP worker
/// receives independent accumulator slots when alpha-beta calls
/// [`SearchEvaluator::begin_search`]. Position-only and MCTS evaluation still
/// use the exact full-rebuild path.
pub struct UpstreamNnueEvaluator {
    /// Used for sharing the validated CURRENT/BIG model across workers.
    network: Arc<UpstreamNnue>,
    /// Used as a fallibly admitted zero-or-one state slot containing the
    /// per-search `HalfKA` slots and reusable `FullThreats` scratch.
    search: Vec<UpstreamNnueSearchState>,
}

impl UpstreamNnueEvaluator {
    /// Used for wrapping one loaded model for use by alpha-beta or MCTS
    /// workers.
    ///
    /// # Arguments
    ///
    /// * `network` - shared validated model
    ///
    /// # Returns
    ///
    /// An evaluator with no open search state.
    #[must_use]
    pub const fn new(network: Arc<UpstreamNnue>) -> Self {
        Self {
            network,
            search: Vec::new(),
        }
    }

    /// Used for retrieving the immutable shared model.
    ///
    /// # Returns
    ///
    /// A borrow of the wrapped [`UpstreamNnue`].
    #[must_use]
    pub fn network(&self) -> &UpstreamNnue {
        &self.network
    }
}

impl Clone for UpstreamNnueEvaluator {
    /// Used for sharing model tensors while discarding another worker's
    /// search path.
    ///
    /// # Returns
    ///
    /// A new evaluator over the same [`Arc`]-shared model with no open
    /// search state.
    fn clone(&self) -> Self {
        Self::new(Arc::clone(&self.network))
    }
}

impl fmt::Debug for UpstreamNnueEvaluator {
    /// Used for printing model metadata and whether a search state is
    /// currently open.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Errors
    ///
    /// Propagates any failure reported by the formatter.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UpstreamNnueEvaluator")
            .field("network", &self.network)
            .field("search_open", &!self.search.is_empty())
            .finish()
    }
}

/// One per-ply `HalfKA` accumulator owned by an alpha-beta worker.
///
/// Holds the bias-seeded piece accumulation and PSQT lanes for both king
/// perspectives; `FullThreats` contributions live in separate scratch.
struct HalfKaSlot {
    /// Used for the bias plus active `HalfKA` rows for the White and Black
    /// king perspectives.
    accumulation: [[i32; TRANSFORMED_DIMENSIONS]; 2],
    /// Used for the eight material-bucket PSQT lanes of both perspectives.
    psqt: [[i32; LAYER_STACKS]; 2],
}

impl HalfKaSlot {
    /// Used for allocating zeroed fixed-size accumulator storage.
    ///
    /// # Returns
    ///
    /// A slot with every accumulation and PSQT lane set to zero.
    const fn new() -> Self {
        Self {
            accumulation: [[0; TRANSFORMED_DIMENSIONS]; 2],
            psqt: [[0; LAYER_STACKS]; 2],
        }
    }

    /// Used for copying the parent position before applying a small child
    /// delta.
    ///
    /// # Arguments
    ///
    /// * `parent` - slot holding the already-aligned parent accumulators
    fn copy_from(&mut self, parent: &Self) {
        for perspective in [WHITE, BLACK] {
            self.accumulation[perspective].copy_from_slice(&parent.accumulation[perspective]);
            self.psqt[perspective] = parent.psqt[perspective];
        }
    }
}

/// Number of `HalfKA` refresh cache entries: one per perspective and anchor
/// king square.
///
/// [`half_ka_index`] depends on the anchoring king only through its square
/// (king bucket plus file-half mirror), so one entry per perspective and
/// king square keeps every cached row index valid for the entry's whole
/// lifetime.
const HALF_KA_REFRESH_CACHE_ENTRIES: usize = 2 * 64;

/// One `HalfKA` refresh cache entry.
///
/// Stores the last bias-seeded accumulator built for one perspective
/// anchored at one king square, together with the Stockfish-coded piece
/// placement those sums reflect. A later refresh with the same anchor
/// applies only the placement difference instead of re-adding every piece.
struct HalfKaRefreshEntry {
    /// Used for the cached bias-plus-active-row sums of the anchored
    /// perspective.
    accumulation: [i32; TRANSFORMED_DIMENSIONS],
    /// Used for the cached material-bucket PSQT lanes of the anchored
    /// perspective.
    psqt: [i32; LAYER_STACKS],
    /// Used for the Stockfish piece codes the cached sums reflect; `0`
    /// marks an empty square.
    board: [u8; 64],
}

/// Per-search `HalfKA` accumulator refresh cache over all anchor king
/// squares.
///
/// Owned by [`UpstreamNnueSearchState`], so its lifetime and reset
/// discipline match the per-ply slots exactly: every
/// [`SearchEvaluator::begin_search`] starts from bias-seeded empty-board
/// entries, and no cached sums leak across searches.
struct HalfKaRefreshCache {
    /// Used for the entries stored row-major as `[perspective][king_square]`.
    entries: Vec<HalfKaRefreshEntry>,
}

/// Used for admitting one fully initialized bounded vector without invoking
/// the process-aborting allocation path.
///
/// # Arguments
///
/// * `length` - exact number of elements required
/// * `initializer` - constructor called once for each admitted element
///
/// # Returns
///
/// The completely initialized vector.
///
/// # Errors
///
/// Returns [`SearchStateError::ResourceExhausted`] for capacity overflow or an
/// allocator refusal.
fn try_initialized_vec<T, F>(length: usize, mut initializer: F) -> Result<Vec<T>, SearchStateError>
where
    F: FnMut() -> T,
{
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| SearchStateError::ResourceExhausted)?;
    for _ in 0..length {
        values.push(initializer());
    }
    Ok(values)
}

impl HalfKaRefreshCache {
    /// Used for fallibly allocating a cold cache whose entries reflect empty
    /// boards.
    ///
    /// # Arguments
    ///
    /// * `biases` - transformer biases seeding every cached accumulator
    ///
    /// # Returns
    ///
    /// A cache in which every entry equals a bias-seeded rebuild of an empty
    /// board, so an entry's first use reproduces a cold rebuild exactly.
    ///
    /// # Errors
    ///
    /// Returns [`SearchStateError::ResourceExhausted`] when the complete cache
    /// cannot be admitted.
    fn try_new(biases: &[i16]) -> Result<Self, SearchStateError> {
        let mut seeded = [0_i32; TRANSFORMED_DIMENSIONS];
        for (target, &bias) in seeded.iter_mut().zip(biases) {
            *target = i32::from(bias);
        }
        let entries = try_initialized_vec(HALF_KA_REFRESH_CACHE_ENTRIES, || HalfKaRefreshEntry {
            accumulation: seeded,
            psqt: [0; LAYER_STACKS],
            board: [0; 64],
        })?;
        Ok(Self { entries })
    }

    /// Used for selecting the entry anchored at one perspective and king
    /// square.
    ///
    /// # Arguments
    ///
    /// * `perspective` - Stockfish color selector of the anchoring king
    /// * `king_square` - Stockfish square of the anchoring king
    ///
    /// # Returns
    ///
    /// The mutable cache entry.
    ///
    /// # Panics
    ///
    /// Panics when `king_square` is not a valid square index; callers pass
    /// squares located by [`find_king`].
    fn entry_mut(&mut self, perspective: usize, king_square: usize) -> &mut HalfKaRefreshEntry {
        &mut self.entries[perspective * 64 + king_square]
    }
}

/// One perspective-independent occupied contact in the `FullThreats` feature set.
///
/// The fixed packing orders contacts without allocating and retains every field
/// needed to project the same board relation through either king perspective.
/// Six bits each hold origin and target squares, followed by four bits each for
/// the Stockfish attacker and victim codes.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ThreatInstance(u32);

impl ThreatInstance {
    /// Zero-valued placeholder used outside the initialized contact prefix.
    const EMPTY: Self = Self(0);

    /// Packs one occupied attack relation into its canonical sortable form.
    fn new(attacker: usize, from: usize, victim: usize, to: usize) -> Self {
        debug_assert!(attacker < 16 && victim < 16 && from < 64 && to < 64);
        Self(
            u32::try_from(from).expect("a square fits six bits")
                | (u32::try_from(to).expect("a square fits six bits") << 6)
                | (u32::try_from(attacker).expect("a piece code fits four bits") << 12)
                | (u32::try_from(victim).expect("a piece code fits four bits") << 16),
        )
    }

    /// Returns the attacker origin square.
    fn from(self) -> usize {
        (self.0 & 63) as usize
    }

    /// Returns the occupied target square.
    fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }

    /// Returns the attacking Stockfish piece code.
    fn attacker(self) -> usize {
        ((self.0 >> 12) & 15) as usize
    }

    /// Returns the attacked Stockfish piece code.
    fn victim(self) -> usize {
        ((self.0 >> 16) & 15) as usize
    }

    /// Projects the contact into one king-oriented feature index or sentinel.
    fn feature(self, perspective: usize, king_square: usize, tables: &ThreatTables) -> usize {
        tables.threat_index(
            perspective,
            self.attacker(),
            self.from(),
            self.to(),
            self.victim(),
            king_square,
        )
    }
}

/// Fixed-capacity sorted contacts stored with each materialized search slot.
#[derive(Clone, Copy)]
struct ActiveThreatInstances {
    /// Packed storage whose initialized prefix is kept in ascending order.
    values: [ThreatInstance; MAX_ACTIVE_THREATS],
    /// Number of initialized packed contacts.
    len: usize,
}

impl ActiveThreatInstances {
    /// Creates an empty contact collection without heap allocation.
    const fn new() -> Self {
        Self {
            values: [ThreatInstance::EMPTY; MAX_ACTIVE_THREATS],
            len: 0,
        }
    }

    /// Adds one board-derived contact within the conservative chess bound.
    fn push(&mut self, instance: ThreatInstance) {
        assert!(
            self.len < MAX_ACTIVE_THREATS,
            "NNUE threat contact count exceeded its board-derived bound"
        );
        self.values[self.len] = instance;
        self.len += 1;
    }

    /// Sorts the initialized prefix and checks the contact emitter is unique.
    fn sort(&mut self) {
        self.values[..self.len].sort_unstable();
        debug_assert!(self.values[..self.len]
            .windows(2)
            .all(|pair| pair[0] != pair[1]));
    }

    /// Returns the canonical initialized contact prefix.
    fn as_slice(&self) -> &[ThreatInstance] {
        &self.values[..self.len]
    }
}

/// Lazily materialized `FullThreats` state for one physical search ply.
struct ThreatSlot {
    /// Active threat-row sums for White and Black king perspectives.
    accumulation: [[i32; TRANSFORMED_DIMENSIONS]; 2],
    /// Material-bucket threat PSQT sums for both perspectives.
    psqt: [[i32; LAYER_STACKS]; 2],
    /// Canonical occupied contacts from which both perspective sums derive.
    instances: ActiveThreatInstances,
    /// File-half orientation of each perspective's king when sums were built.
    orientations: [u8; 2],
    /// Physical parent slot eligible to seed the next materialization.
    parent: Option<usize>,
    /// Whether all fields describe the slot's current position.
    accurate: bool,
}

impl ThreatSlot {
    /// Allocates an inaccurate zeroed slot with no parent dependency.
    const fn new() -> Self {
        Self {
            accumulation: [[0; TRANSFORMED_DIMENSIONS]; 2],
            psqt: [[0; LAYER_STACKS]; 2],
            instances: ActiveThreatInstances::new(),
            orientations: [0; 2],
            parent: None,
            accurate: false,
        }
    }

    /// Marks a reused physical child inaccurate and records its active parent.
    fn invalidate_from(&mut self, parent: usize) {
        self.parent = Some(parent);
        self.accurate = false;
    }
}

/// Worker-local incremental `HalfKA` and lazy `FullThreats` state.
struct UpstreamNnueSearchState {
    /// Dedicated `HalfKA` child storage indexed by physical ply.
    slots: Vec<HalfKaSlot>,
    /// Dedicated lazy threat storage indexed by the same physical ply.
    threat_slots: Vec<ThreatSlot>,
    /// Used for the active physical slot per logical ply; null moves alias
    /// their parent.
    active: Vec<usize>,
    /// Used for bias-seeded FC0 scratch rebuilt in transformed-lane order
    /// per evaluation.
    fc0: [i32; FC0_OUTPUT_DIMENSIONS],
    /// Used for the per-search `HalfKA` refresh cache anchored per
    /// perspective and king square.
    refresh_cache: HalfKaRefreshCache,
}

impl fmt::Debug for UpstreamNnue {
    /// Used for printing metadata without dumping the model's large
    /// parameter tensors.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Errors
    ///
    /// Propagates any failure reported by the formatter.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UpstreamNnue")
            .field("description", &self.description)
            .field("network_hash", &format_args!("0x{:08x}", self.network_hash))
            .field("variant", &"SFNNv13")
            .field("size", &"BIG")
            .finish_non_exhaustive()
    }
}

impl UpstreamNnue {
    /// Used for loading a bounded CURRENT/BIG model from disk.
    ///
    /// The declared file length is checked before reading, and the read
    /// itself is capped so a file growing mid-read cannot exceed the loader
    /// limit.
    ///
    /// # Arguments
    ///
    /// * `path` - filesystem location of the serialized model
    ///
    /// # Returns
    ///
    /// The fully validated model.
    ///
    /// # Errors
    ///
    /// Returns an error for I/O failures, oversized files, malformed data, or
    /// a Stockfish architecture other than the explicitly supported one.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, UpstreamNnueError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let declared = file.metadata()?.len();
        if declared > MAX_UPSTREAM_MODEL_BYTES as u64 {
            return Err(UpstreamNnueError::new(
                UpstreamNnueErrorKind::ResourceLimit,
                format!(
                    "Stockfish NNUE file is {declared} bytes; limit is {MAX_UPSTREAM_MODEL_BYTES}"
                ),
            ));
        }
        let capacity = usize::try_from(declared).map_err(|_| {
            UpstreamNnueError::new(
                UpstreamNnueErrorKind::ResourceLimit,
                "Stockfish NNUE file length does not fit this platform",
            )
        })?;
        let mut bytes = Vec::new();
        reserve_exact(&mut bytes, capacity, "model file")?;
        file.take((MAX_UPSTREAM_MODEL_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_UPSTREAM_MODEL_BYTES {
            return Err(UpstreamNnueError::new(
                UpstreamNnueErrorKind::ResourceLimit,
                "Stockfish NNUE file grew beyond the loader limit while reading",
            ));
        }
        Self::from_bytes(&bytes)
    }

    /// Used for parsing a complete CURRENT/BIG model, requiring exact EOF.
    ///
    /// Validates the version, network hash, description, feature-transformer
    /// hash, and all eight layer-stack hashes before the model becomes
    /// usable.
    ///
    /// # Arguments
    ///
    /// * `bytes` - complete serialized model
    ///
    /// # Returns
    ///
    /// The fully validated model.
    ///
    /// # Errors
    ///
    /// Returns an error for oversized input, malformed fields, inconsistent
    /// hashes or tensor shapes, allocation failure, or trailing bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, UpstreamNnueError> {
        if bytes.len() > MAX_UPSTREAM_MODEL_BYTES {
            return Err(UpstreamNnueError::new(
                UpstreamNnueErrorKind::ResourceLimit,
                format!(
                    "Stockfish NNUE buffer is {} bytes; limit is {MAX_UPSTREAM_MODEL_BYTES}",
                    bytes.len()
                ),
            ));
        }
        let mut cursor = Cursor::new(bytes);
        let version = cursor.read_u32("version")?;
        if version != UPSTREAM_VERSION {
            return Err(UpstreamNnueError::new(
                UpstreamNnueErrorKind::InvalidFormat,
                format!("unsupported Stockfish NNUE version 0x{version:08x}"),
            ));
        }
        let network_hash = cursor.read_u32("network hash")?;
        if network_hash != current_big_network_hash() {
            return Err(UpstreamNnueError::new(
                UpstreamNnueErrorKind::UnsupportedArchitecture,
                format!(
                    "unsupported Stockfish NNUE architecture hash 0x{network_hash:08x}; expected CURRENT/BIG 0x{:08x}",
                    current_big_network_hash()
                ),
            ));
        }
        let description_len = cursor.read_length("description length", 1 << 20)?;
        let description_bytes = cursor.read_slice(description_len, "description")?;
        let description_text = std::str::from_utf8(description_bytes)
            .map_err(|_| invalid("Stockfish NNUE description is not valid UTF-8"))?;
        let mut description = String::new();
        description
            .try_reserve_exact(description_text.len())
            .map_err(|_| {
                UpstreamNnueError::new(
                    UpstreamNnueErrorKind::ResourceLimit,
                    "could not reserve Stockfish NNUE description",
                )
            })?;
        description.push_str(description_text);

        let feature_hash = cursor.read_u32("feature-transformer hash")?;
        if feature_hash != current_big_feature_hash() {
            return Err(invalid(format!(
                "Stockfish NNUE feature-transformer hash mismatch: 0x{feature_hash:08x}"
            )));
        }
        let transformer = FeatureTransformer::read(&mut cursor)?;

        let mut stacks = Vec::new();
        reserve_exact(&mut stacks, LAYER_STACKS, "layer stacks")?;
        for bucket in 0..LAYER_STACKS {
            let hash = cursor.read_u32("layer-stack hash")?;
            if hash != current_big_arch_hash() {
                return Err(invalid(format!(
                    "Stockfish NNUE layer-stack hash mismatch at bucket {bucket}: 0x{hash:08x}"
                )));
            }
            stacks.push(Architecture::read(&mut cursor)?);
        }
        if !cursor.is_at_end() {
            return Err(invalid(format!(
                "unexpected {} trailing bytes in Stockfish NNUE file",
                cursor.remaining()
            )));
        }

        Ok(Self {
            description,
            network_hash,
            transformer,
            stacks,
            threat_tables: ThreatTables::current()?,
        })
    }

    /// Used for testing whether a byte prefix is a Stockfish NNUE version
    /// header.
    ///
    /// # Arguments
    ///
    /// * `bytes` - candidate stream prefix
    ///
    /// # Returns
    ///
    /// `true` when the first four bytes decode to [`UPSTREAM_VERSION`].
    #[must_use]
    pub fn has_upstream_header(bytes: &[u8]) -> bool {
        bytes
            .get(..4)
            .and_then(|prefix| prefix.try_into().ok())
            .map(u32::from_le_bytes)
            == Some(UPSTREAM_VERSION)
    }

    /// Used for retrieving immutable model metadata.
    ///
    /// # Returns
    ///
    /// A fresh [`UpstreamNnueInfo`] combining the fixed CURRENT/BIG
    /// dimensions with the loaded hash and description.
    #[must_use]
    pub fn info(&self) -> UpstreamNnueInfo {
        UpstreamNnueInfo {
            variant: "SFNNv13",
            size: "BIG",
            input_features: TOTAL_INPUT_DIMENSIONS,
            transformed_dimensions: TRANSFORMED_DIMENSIONS,
            l2: L2,
            l3: L3,
            hash: self.network_hash,
            description: self.description.clone(),
        }
    }

    /// Used for evaluating a position with full, allocation-free scalar
    /// inference.
    ///
    /// Rebuilds both perspectives from scratch, selects the material
    /// bucket, and propagates the sparse transformed features through that
    /// bucket's dense stack.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// The decomposed side-to-move prediction in centipawns.
    ///
    /// # Panics
    ///
    /// Panics only if `position` violates `janus-core`'s invariant that both
    /// kings are present. Public position constructors reject such states.
    #[must_use]
    pub fn predict(&self, position: &Position) -> UpstreamPrediction {
        let board = stockfish_board(position);
        let piece_count = board.iter().filter(|&&piece| piece != 0).count();
        let bucket = bucket(piece_count);
        let transformed = self
            .transformer
            .transform(position, &board, bucket, &self.threat_tables);
        let positional = self.stacks[bucket].propagate(&transformed.features);
        UpstreamPrediction {
            psqt: transformed.psqt / OUTPUT_SCALE,
            positional: positional / OUTPUT_SCALE,
        }
    }

    /// Used for computing the complete side-to-move centipawn score.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// The summed PSQT and positional contributions of [`Self::predict`].
    ///
    /// # Panics
    ///
    /// Has the same invariant requirement as [`Self::predict`]: the position
    /// must contain both kings.
    #[must_use]
    pub fn evaluate_centipawns(&self, position: &Position) -> i32 {
        self.predict(position).centipawns()
    }

    /// Used for evaluating one position from an already-aligned incremental
    /// search slot.
    ///
    /// `HalfKA` sums come from the ply's slot; the `FullThreats` transform
    /// is rebuilt exactly into the state's reusable scratch, and both are
    /// combined directly into bias-seeded FC0 scratch.
    ///
    /// # Arguments
    ///
    /// * `state` - worker-local incremental state
    /// * `position` - position whose slot is aligned at `ply`
    /// * `ply` - root-relative ply of `position`
    ///
    /// # Returns
    ///
    /// The decomposed side-to-move prediction in centipawns.
    ///
    /// # Panics
    ///
    /// Panics when `ply` exceeds the search bound announced to
    /// [`UpstreamNnueSearchState::try_new`].
    fn evaluate_incremental(
        &self,
        state: &mut UpstreamNnueSearchState,
        position: &Position,
        ply: usize,
    ) -> UpstreamPrediction {
        let slot_index = *state
            .active
            .get(ply)
            .expect("alpha-beta ply stays within the announced search bound");
        let board = stockfish_board(position);
        self.materialize_threat_slot(state, &board, slot_index);

        let piece_count = board.iter().filter(|&&piece| piece != 0).count();
        let bucket = bucket(piece_count);
        let stm = match position.side_to_move() {
            Color::White => WHITE,
            Color::Black => BLACK,
        };
        let opponent = stm ^ 1;
        let slot = &state.slots[slot_index];
        let threats = &state.threat_slots[slot_index];
        let psqt = (slot.psqt[stm][bucket] - slot.psqt[opponent][bucket]
            + threats.psqt[stm][bucket]
            - threats.psqt[opponent][bucket])
            / 2;
        let architecture = &self.stacks[bucket];
        architecture.fc0.initialize_output(&mut state.fc0);
        architecture.fc0.accumulate_combined_perspective(
            &mut state.fc0,
            0,
            &slot.accumulation[stm],
            &threats.accumulation[stm],
        );
        architecture.fc0.accumulate_combined_perspective(
            &mut state.fc0,
            HALF_TRANSFORMED_DIMENSIONS,
            &slot.accumulation[opponent],
            &threats.accumulation[opponent],
        );
        let positional = architecture.propagate_fc0(&state.fc0);
        UpstreamPrediction {
            psqt: psqt / OUTPUT_SCALE,
            positional: positional / OUTPUT_SCALE,
        }
    }

    /// Materializes one inaccurate threat slot from an accurate parent or refresh.
    ///
    /// Contact enumeration happens only at a real incremental evaluation. A
    /// parent may seed a perspective while its king remains in the same file
    /// half; otherwise that perspective is rebuilt from the same canonical
    /// contact set. The completed slot is exact and reusable by later calls.
    fn materialize_threat_slot(
        &self,
        state: &mut UpstreamNnueSearchState,
        board: &[usize; 64],
        slot_index: usize,
    ) {
        if state.threat_slots[slot_index].accurate {
            return;
        }

        let instances = active_threat_instances(board);
        let king_squares = [find_king(board, WHITE), find_king(board, BLACK)];
        let orientations = king_squares.map(threat_orientation);
        let parent_index = state.threat_slots[slot_index]
            .parent
            .filter(|&parent| parent < slot_index && state.threat_slots[parent].accurate);
        let mut parent_seeded = false;
        let mut perspective_refreshes = 0;
        let mut changed_rows = 0;

        if let Some(parent_index) = parent_index {
            let (parents, children) = state.threat_slots.split_at_mut(slot_index);
            let parent = &parents[parent_index];
            let child = &mut children[0];
            let ThreatSlot {
                accumulation,
                psqt,
                instances: child_instances,
                orientations: child_orientations,
                accurate,
                ..
            } = child;
            for perspective in [WHITE, BLACK] {
                if parent.orientations[perspective] == orientations[perspective] {
                    accumulation[perspective].copy_from_slice(&parent.accumulation[perspective]);
                    psqt[perspective] = parent.psqt[perspective];
                    changed_rows += self.apply_threat_instance_diff(
                        &mut accumulation[perspective],
                        &mut psqt[perspective],
                        parent.instances.as_slice(),
                        instances.as_slice(),
                        perspective,
                        king_squares[perspective],
                    );
                    parent_seeded = true;
                } else {
                    self.refresh_threat_perspective(
                        &mut accumulation[perspective],
                        &mut psqt[perspective],
                        instances.as_slice(),
                        perspective,
                        king_squares[perspective],
                    );
                    perspective_refreshes += 1;
                }
            }
            *child_instances = instances;
            *child_orientations = orientations;
            *accurate = true;
        } else {
            let child = &mut state.threat_slots[slot_index];
            for perspective in [WHITE, BLACK] {
                self.refresh_threat_perspective(
                    &mut child.accumulation[perspective],
                    &mut child.psqt[perspective],
                    instances.as_slice(),
                    perspective,
                    king_squares[perspective],
                );
                perspective_refreshes += 1;
            }
            child.instances = instances;
            child.orientations = orientations;
            child.accurate = true;
        }

        {
            let _ = (parent_seeded, perspective_refreshes, changed_rows);
        }
    }

    /// Rebuilds one perspective from a canonical occupied-contact collection.
    fn refresh_threat_perspective(
        &self,
        accumulation: &mut [i32; TRANSFORMED_DIMENSIONS],
        psqt: &mut [i32; LAYER_STACKS],
        instances: &[ThreatInstance],
        perspective: usize,
        king_square: usize,
    ) {
        accumulation.fill(0);
        psqt.fill(0);
        let features =
            project_threat_instances(instances, perspective, king_square, &self.threat_tables);
        add_i8_features(
            accumulation,
            psqt,
            features.as_slice(),
            &self.transformer.threat_weights,
            &self.transformer.threat_psqt_weights,
        );
    }

    /// Applies the sorted multiset difference between parent and child contacts.
    ///
    /// Removed rows are processed before added rows. Feature indices remain
    /// valid because callers reject a parent whose king changed file-half
    /// orientation. The returned count includes only model-valid changed rows.
    fn apply_threat_instance_diff(
        &self,
        accumulation: &mut [i32; TRANSFORMED_DIMENSIONS],
        psqt: &mut [i32; LAYER_STACKS],
        parent: &[ThreatInstance],
        child: &[ThreatInstance],
        perspective: usize,
        king_square: usize,
    ) -> usize {
        let mut changed_rows = 0;
        let mut parent_index = 0;
        let mut child_index = 0;
        while parent_index < parent.len() {
            if child_index < child.len() && child[child_index] < parent[parent_index] {
                child_index += 1;
            } else if child_index < child.len() && child[child_index] == parent[parent_index] {
                parent_index += 1;
                child_index += 1;
            } else {
                let feature =
                    parent[parent_index].feature(perspective, king_square, &self.threat_tables);
                if feature < THREAT_DIMENSIONS {
                    apply_i8_feature_delta(
                        accumulation,
                        psqt,
                        feature,
                        -1,
                        &self.transformer.threat_weights,
                        &self.transformer.threat_psqt_weights,
                    );
                    changed_rows += 1;
                }
                parent_index += 1;
            }
        }

        parent_index = 0;
        child_index = 0;
        while child_index < child.len() {
            if parent_index < parent.len() && parent[parent_index] < child[child_index] {
                parent_index += 1;
            } else if parent_index < parent.len() && parent[parent_index] == child[child_index] {
                parent_index += 1;
                child_index += 1;
            } else {
                let feature =
                    child[child_index].feature(perspective, king_square, &self.threat_tables);
                if feature < THREAT_DIMENSIONS {
                    apply_i8_feature_delta(
                        accumulation,
                        psqt,
                        feature,
                        1,
                        &self.transformer.threat_weights,
                        &self.transformer.threat_psqt_weights,
                    );
                    changed_rows += 1;
                }
                child_index += 1;
            }
        }
        changed_rows
    }

    /// Used for aligning a physical child slot after one already-made legal
    /// move.
    ///
    /// Non-king, non-castling moves, including captures and promotions,
    /// apply signed `HalfKA` row deltas to a copy of the parent's slot. A
    /// moving king refreshes its own perspective, and castling refreshes
    /// both perspectives exactly; both refresh paths reuse the per-search
    /// king-square cache so only changed feature rows are applied.
    ///
    /// # Arguments
    ///
    /// * `state` - worker-local incremental state
    /// * `child` - position after the move was made
    /// * `mv` - move that produced `child`
    /// * `undo` - undo record captured when the move was made
    /// * `ply` - root-relative ply of `child`
    ///
    /// # Panics
    ///
    /// Panics when `ply` is zero or beyond the announced search bound, or
    /// if a validated position is missing a king.
    fn move_incremental(
        &self,
        state: &mut UpstreamNnueSearchState,
        child: &Position,
        mv: Move,
        undo: Undo,
        ply: usize,
    ) {
        assert!(ply > 0 && ply < state.slots.len());
        let parent_index = state.active[ply - 1];
        state.threat_slots[ply].invalidate_from(parent_index);
        let (parents, children) = state.slots.split_at_mut(ply);
        let slot = &mut children[0];
        let parent = &parents[parent_index];
        state.active[ply] = ply;

        // Castling is rare and can contain overlapping Chess960 source/target
        // squares. Rebuilding both 2 Ki-lane HalfKA perspectives here is exact
        // and keeps the public core Undo record deliberately narrow.
        if undo.was_castle() {
            let board = stockfish_board(child);
            let mut refresh_changed_rows = 0;
            for perspective in [WHITE, BLACK] {
                refresh_changed_rows += self.transformer.refresh_half_ka_perspective_cached(
                    slot,
                    &board,
                    perspective,
                    &mut state.refresh_cache,
                );
            }
            state.record_half_ka_refreshes(2, refresh_changed_rows);
            return;
        }

        let moved = undo.moved_piece();
        let placed = Piece::new(moved.color, mv.promotion().unwrap_or(moved.kind));
        if moved.kind != PieceKind::King {
            for perspective in [WHITE, BLACK] {
                let king_square = child
                    .king_square(color_from_perspective(perspective))
                    .expect("validated positions contain both kings");
                let removed = half_ka_feature(perspective, moved, mv.from(), king_square);
                let added = half_ka_feature(perspective, placed, mv.to(), king_square);
                let captured = undo.captured().map(|(piece, square)| {
                    half_ka_feature(perspective, piece, square, king_square)
                });
                self.transformer.write_half_ka_child_delta(
                    slot,
                    parent,
                    perspective,
                    removed,
                    added,
                    captured,
                );
            }
            return;
        }

        slot.copy_from(parent);
        let board = stockfish_board(child);
        let mut mover_refresh_changed_rows = 0;
        for perspective in [WHITE, BLACK] {
            if moved.kind == PieceKind::King && moved.color.index() == perspective {
                mover_refresh_changed_rows += self.transformer.refresh_half_ka_perspective_cached(
                    slot,
                    &board,
                    perspective,
                    &mut state.refresh_cache,
                );
                continue;
            }
            let king_square = child
                .king_square(color_from_perspective(perspective))
                .expect("validated positions contain both kings");
            apply_half_ka_feature(
                slot,
                perspective,
                moved,
                mv.from(),
                king_square,
                -1,
                &self.transformer,
            );
            apply_half_ka_feature(
                slot,
                perspective,
                placed,
                mv.to(),
                king_square,
                1,
                &self.transformer,
            );
            if let Some((captured, square)) = undo.captured() {
                apply_half_ka_feature(
                    slot,
                    perspective,
                    captured,
                    square,
                    king_square,
                    -1,
                    &self.transformer,
                );
            }
        }
        state.record_half_ka_refreshes(1, mover_refresh_changed_rows);
    }
}

impl UpstreamNnueSearchState {
    /// Used for fallibly allocating bounded per-ply slots and initializing the
    /// root accumulator.
    ///
    /// # Arguments
    ///
    /// * `network` - model supplying the transformer parameters
    /// * `root` - search root position
    /// * `max_plies` - number of per-ply slots to allocate
    ///
    /// # Returns
    ///
    /// A state whose slot zero holds the root's refreshed `HalfKA` sums.
    ///
    /// # Errors
    ///
    /// Returns [`SearchStateError::ResourceExhausted`] when any complete
    /// backing vector cannot be admitted.
    ///
    /// # Panics
    ///
    /// Panics when `max_plies` is zero.
    fn try_new(
        network: &UpstreamNnue,
        root: &Position,
        max_plies: usize,
    ) -> Result<Self, SearchStateError> {
        assert!(max_plies > 0);
        let mut slots = try_initialized_vec(max_plies, HalfKaSlot::new)?;
        let threat_slots = try_initialized_vec(max_plies, ThreatSlot::new)?;
        let mut refresh_cache = HalfKaRefreshCache::try_new(&network.transformer.biases)?;
        let board = stockfish_board(root);
        let mut root_changed_rows = 0;
        for perspective in [WHITE, BLACK] {
            root_changed_rows += network.transformer.refresh_half_ka_perspective_cached(
                &mut slots[0],
                &board,
                perspective,
                &mut refresh_cache,
            );
        }

        let _ = root_changed_rows;
        Ok(Self {
            slots,
            threat_slots,
            active: try_initialized_vec(max_plies, || 0)?,
            fc0: [0; FC0_OUTPUT_DIMENSIONS],
            refresh_cache,
        })
    }

    /// Used for making a null child share its parent's unchanged piece
    /// accumulators.
    ///
    /// # Arguments
    ///
    /// * `ply` - root-relative ply of the null-move child
    ///
    /// # Panics
    ///
    /// Panics when `ply` is zero or beyond the announced search bound.
    fn null_move_played(&mut self, ply: usize) {
        assert!(ply > 0 && ply < self.active.len());
        self.active[ply] = self.active[ply - 1];
    }

    /// Used for recording one batch of cached `HalfKA` refreshes in test
    /// builds.
    ///
    /// Release builds keep the call so both configurations share one
    /// control flow; the arguments are simply discarded.
    ///
    /// # Arguments
    ///
    /// * `refreshes` - number of cache-backed perspective refreshes
    /// * `changed_rows` - number of feature rows added or subtracted
    fn record_half_ka_refreshes(&mut self, refreshes: usize, changed_rows: usize) {
        let _ = (self, refreshes, changed_rows);
    }
}

impl Evaluator for UpstreamNnue {
    /// Used for supplying the side-to-move scalar score to the general
    /// evaluator API.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// The complete centipawn score from
    /// [`UpstreamNnue::evaluate_centipawns`].
    fn evaluate(&mut self, position: &Position) -> i32 {
        self.evaluate_centipawns(position)
    }
}

impl SearchEvaluator for UpstreamNnue {
    /// Used for supplying the same deterministic score to the alpha-beta
    /// evaluator API.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// The complete centipawn score from
    /// [`UpstreamNnue::evaluate_centipawns`].
    fn evaluate(&mut self, position: &Position) -> i32 {
        self.evaluate_centipawns(position)
    }
}

impl Evaluator for UpstreamNnueEvaluator {
    /// Used for evaluating with the full-rebuild oracle outside
    /// alpha-beta's lifecycle hooks.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// The complete side-to-move centipawn score.
    fn evaluate(&mut self, position: &Position) -> i32 {
        self.network.evaluate_centipawns(position)
    }
}

impl SearchEvaluator for UpstreamNnueEvaluator {
    /// Used for evaluating with the full-rebuild oracle when no
    /// root-relative ply is available.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// The complete side-to-move centipawn score.
    fn evaluate(&mut self, position: &Position) -> i32 {
        self.network.evaluate_centipawns(position)
    }

    /// Used for opening independent worker-local `HalfKA` slots at the
    /// current root.
    ///
    /// Roots below `MIN_INCREMENTAL_PIECES` pieces skip incremental state
    /// and fall back to sparse full rebuilds.
    ///
    /// # Arguments
    ///
    /// * `root` - position searched from
    /// * `max_plies` - deepest root-relative ply the search may reach
    ///
    /// # Returns
    ///
    /// Success after either opening incremental state or selecting the
    /// low-material full-rebuild path.
    ///
    /// # Errors
    ///
    /// Returns [`SearchStateError::ResourceExhausted`] when the complete
    /// incremental state cannot be admitted.
    fn begin_search(&mut self, root: &Position, max_plies: usize) -> Result<(), SearchStateError> {
        self.search.clear();
        if root.occupancy().count_ones() >= MIN_INCREMENTAL_PIECES {
            self.search
                .try_reserve_exact(1)
                .map_err(|_| SearchStateError::ResourceExhausted)?;
            let state = UpstreamNnueSearchState::try_new(&self.network, root, max_plies)?;
            self.search.push(state);
        }
        Ok(())
    }

    /// Used for reporting whether this root selected per-ply `HalfKA`
    /// accumulation.
    ///
    /// # Returns
    ///
    /// `true` when [`SearchEvaluator::begin_search`] opened incremental
    /// state.
    fn uses_incremental_search_state(&self) -> bool {
        !self.search.is_empty()
    }

    /// Used for applying one exact `HalfKA` child delta after alpha-beta
    /// makes a move.
    ///
    /// # Arguments
    ///
    /// * `child` - position after the move was made
    /// * `mv` - move that produced `child`
    /// * `undo` - undo record captured when the move was made
    /// * `ply` - root-relative ply of `child`
    fn move_played(&mut self, child: &Position, mv: Move, undo: Undo, ply: usize) {
        if let Some(state) = self.search.first_mut() {
            self.network.move_incremental(state, child, mv, undo, ply);
        }
    }

    /// Used for reusing the parent piece accumulators for a null-move
    /// child.
    ///
    /// # Arguments
    ///
    /// * `ply` - root-relative ply of the null-move child
    fn null_move_played(&mut self, ply: usize) {
        if let Some(state) = self.search.first_mut() {
            state.null_move_played(ply);
        }
    }

    /// Used for combining incremental `HalfKA` with a reusable full
    /// `FullThreats` rebuild.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `ply` - root-relative ply of `position`
    ///
    /// # Returns
    ///
    /// The complete side-to-move centipawn score.
    fn evaluate_at(&mut self, position: &Position, ply: usize) -> i32 {
        self.search.first_mut().map_or_else(
            || self.network.evaluate_centipawns(position),
            |state| {
                self.network
                    .evaluate_incremental(state, position, ply)
                    .centipawns()
            },
        )
    }
}

/// Serialized sparse feature transformer shared by all material buckets.
///
/// `HalfKA` weights are signed 16-bit values. `FullThreats` weights are serialized
/// as bytes and reinterpreted as signed 8-bit values during accumulation. Both
/// feature families have a parallel eight-lane PSQT table.
struct FeatureTransformer {
    /// Used for the initial value of each 1,024-wide `HalfKA` accumulator.
    biases: Vec<i16>,
    /// Used for the row-major `[threat_feature][transformed_dimension]`
    /// signed-byte weights.
    threat_weights: Vec<u8>,
    /// Used for the row-major `[half_ka_feature][transformed_dimension]`
    /// signed-16 weights.
    psq_weights: Vec<i16>,
    /// Used for the row-major `[threat_feature][material_bucket]` PSQT
    /// terms.
    threat_psqt_weights: Vec<i32>,
    /// Used for the row-major `[half_ka_feature][material_bucket]` PSQT
    /// terms.
    psqt_weights: Vec<i32>,
}

impl FeatureTransformer {
    /// Used for reading the exact CURRENT/BIG transformer tensor sequence.
    ///
    /// The serialized combined PSQT table stores `FullThreats` rows before
    /// `HalfKA` rows and is split after decoding.
    ///
    /// # Arguments
    ///
    /// * `cursor` - bounded cursor positioned at the transformer tensors
    ///
    /// # Returns
    ///
    /// The fully decoded transformer.
    ///
    /// # Errors
    ///
    /// Returns an error for truncated or malformed tensors, out-of-range
    /// values, dimension overflow, or allocation failure.
    fn read(cursor: &mut Cursor<'_>) -> Result<Self, UpstreamNnueError> {
        let biases = cursor.read_leb_i16_array(TRANSFORMED_DIMENSIONS, "transformer biases")?;
        let threat_weight_count = checked_product(
            THREAT_DIMENSIONS,
            TRANSFORMED_DIMENSIONS,
            "threat weight count",
        )?;
        let threat_weights = cursor.read_byte_vector(threat_weight_count, "threat weights")?;
        let psq_weight_count = checked_product(
            HALF_KA_DIMENSIONS,
            TRANSFORMED_DIMENSIONS,
            "HalfKA weight count",
        )?;
        let psq_weights = cursor.read_leb_i16_array(psq_weight_count, "HalfKA weights")?;
        let combined_count =
            checked_product(TOTAL_INPUT_DIMENSIONS, LAYER_STACKS, "combined PSQT count")?;
        let combined = cursor.read_leb_i32_array(combined_count, "combined PSQT weights")?;
        let threat_count = checked_product(THREAT_DIMENSIONS, LAYER_STACKS, "threat PSQT count")?;
        let (threat_psqt_source, psqt_source) = combined.split_at(threat_count);
        let threat_psqt_weights = copy_slice(threat_psqt_source, "threat PSQT weights")?;
        let psqt_weights = copy_slice(psqt_source, "HalfKA PSQT weights")?;
        Ok(Self {
            biases,
            threat_weights,
            psq_weights,
            threat_psqt_weights,
            psqt_weights,
        })
    }

    /// Used for rebuilding one `HalfKA` perspective through the per-search
    /// refresh cache.
    ///
    /// Diffs `board` against the cache entry anchored at the perspective's
    /// king square, applies only the changed feature rows to the cached
    /// sums, then copies the updated entry into `slot`. The accumulators
    /// are plain integer sums of feature rows, so the result is
    /// bit-identical to a from-scratch rebuild regardless of the entry's
    /// prior contents.
    ///
    /// # Arguments
    ///
    /// * `slot` - accumulator slot whose perspective is overwritten
    /// * `board` - Stockfish-coded piece placement
    /// * `perspective` - Stockfish color selector of the rebuilt side
    /// * `cache` - per-search refresh cache updated in place
    ///
    /// # Returns
    ///
    /// The number of feature rows added or subtracted.
    ///
    /// # Panics
    ///
    /// Panics if the board lacks the perspective's king.
    fn refresh_half_ka_perspective_cached(
        &self,
        slot: &mut HalfKaSlot,
        board: &[usize; 64],
        perspective: usize,
        cache: &mut HalfKaRefreshCache,
    ) -> usize {
        let king_square = find_king(board, perspective);
        let entry = cache.entry_mut(perspective, king_square);
        let mut changed_rows = 0;
        for (square, &piece) in board.iter().enumerate() {
            let cached = usize::from(entry.board[square]);
            if cached == piece {
                continue;
            }
            if cached != 0 {
                apply_i16_feature_delta(
                    &mut entry.accumulation,
                    &mut entry.psqt,
                    half_ka_index(perspective, square, cached, king_square),
                    -1,
                    &self.psq_weights,
                    &self.psqt_weights,
                );
                changed_rows += 1;
            }
            if piece != 0 {
                apply_i16_feature_delta(
                    &mut entry.accumulation,
                    &mut entry.psqt,
                    half_ka_index(perspective, square, piece, king_square),
                    1,
                    &self.psq_weights,
                    &self.psqt_weights,
                );
                changed_rows += 1;
            }
            entry.board[square] = u8::try_from(piece).expect("Stockfish piece codes fit one byte");
        }
        slot.accumulation[perspective].copy_from_slice(&entry.accumulation);
        slot.psqt[perspective] = entry.psqt;
        changed_rows
    }

    /// Writes one ordinary child perspective from its parent and feature delta.
    ///
    /// `removed`, `added`, and `captured` are validated `HalfKA` row indices for
    /// the same king perspective. The destination is written once in the exact
    /// arithmetic order used by the former copy followed by separate row
    /// updates: subtract the moved piece, add its placed form, then subtract an
    /// optional capture. Model loading guarantees each selected row is whole.
    fn write_half_ka_child_delta(
        &self,
        child: &mut HalfKaSlot,
        parent: &HalfKaSlot,
        perspective: usize,
        removed: usize,
        added: usize,
        captured: Option<usize>,
    ) {
        let removed_base = removed * TRANSFORMED_DIMENSIONS;
        let added_base = added * TRANSFORMED_DIMENSIONS;
        let removed_weights =
            &self.psq_weights[removed_base..removed_base + TRANSFORMED_DIMENSIONS];
        let added_weights = &self.psq_weights[added_base..added_base + TRANSFORMED_DIMENSIONS];
        let captured_weights = captured.map(|feature| {
            let base = feature * TRANSFORMED_DIMENSIONS;
            &self.psq_weights[base..base + TRANSFORMED_DIMENSIONS]
        });
        write_half_ka_accumulator_delta(
            &mut child.accumulation[perspective],
            &parent.accumulation[perspective],
            removed_weights,
            added_weights,
            captured_weights,
        );

        let removed_base = removed * LAYER_STACKS;
        let added_base = added * LAYER_STACKS;
        let removed_weights = &self.psqt_weights[removed_base..removed_base + LAYER_STACKS];
        let added_weights = &self.psqt_weights[added_base..added_base + LAYER_STACKS];
        let captured_weights = captured.map(|feature| {
            let base = feature * LAYER_STACKS;
            &self.psqt_weights[base..base + LAYER_STACKS]
        });
        write_half_ka_psqt_delta(
            &mut child.psqt[perspective],
            &parent.psqt[perspective],
            removed_weights,
            added_weights,
            captured_weights,
        );
    }

    /// Used for rebuilding both perspective accumulators and emitting
    /// side-to-move features.
    ///
    /// # Arguments
    ///
    /// * `position` - position supplying the side to move
    /// * `board` - Stockfish-coded piece placement of `position`
    /// * `bucket` - material bucket selecting the PSQT lane
    /// * `threat_tables` - validated `FullThreats` index tables
    ///
    /// # Returns
    ///
    /// The sparse transformed features and the halved PSQT difference in
    /// internal score units.
    fn transform(
        &self,
        position: &Position,
        board: &[usize; 64],
        bucket: usize,
        threat_tables: &ThreatTables,
    ) -> TransformOutput {
        // Search inference is hot enough that even small heap traffic is
        // visible at short UCI time controls. HalfKA and FullThreats are both
        // additive before the clipped product, so one fixed accumulator per
        // perspective preserves the integer result while halving scratch.
        let mut accumulation = [[0_i32; TRANSFORMED_DIMENSIONS]; 2];
        let mut psqt_accumulation = [[0_i32; LAYER_STACKS]; 2];
        let mut threat_psqt_accumulation = [[0_i32; LAYER_STACKS]; 2];

        for perspective in [WHITE, BLACK] {
            for (target, &bias) in accumulation[perspective].iter_mut().zip(&self.biases) {
                *target = i32::from(bias);
            }
            let half_ka = active_half_ka(board, perspective);
            add_i16_features(
                &mut accumulation[perspective],
                &mut psqt_accumulation[perspective],
                half_ka.as_slice(),
                &self.psq_weights,
                &self.psqt_weights,
            );
            let threats = active_threats(board, perspective, threat_tables);
            add_i8_features(
                &mut accumulation[perspective],
                &mut threat_psqt_accumulation[perspective],
                threats.as_slice(),
                &self.threat_weights,
                &self.threat_psqt_weights,
            );
        }

        let stm = match position.side_to_move() {
            Color::White => WHITE,
            Color::Black => BLACK,
        };
        let opponent = stm ^ 1;
        let psqt = (psqt_accumulation[stm][bucket] - psqt_accumulation[opponent][bucket]
            + threat_psqt_accumulation[stm][bucket]
            - threat_psqt_accumulation[opponent][bucket])
            / 2;

        let mut features = SparseTransformedFeatures::new();
        write_perspective_features(&mut features, 0, &accumulation[stm]);
        write_perspective_features(
            &mut features,
            HALF_TRANSFORMED_DIMENSIONS,
            &accumulation[opponent],
        );
        TransformOutput { features, psqt }
    }
}

/// Used for adding active signed-16 `HalfKA` rows and their
/// material-bucket PSQT lanes.
///
/// # Arguments
///
/// * `accumulation` - destination transform accumulator
/// * `psqt_accumulation` - destination material-bucket PSQT lanes
/// * `features` - active `HalfKA` feature indices
/// * `weights` - row-major transform weight table
/// * `psqt` - row-major PSQT weight table
///
/// # Panics
///
/// Panics if a feature indexes a row beyond either weight table; model
/// loading sizes both tables to cover every valid feature.
fn add_i16_features(
    accumulation: &mut [i32],
    psqt_accumulation: &mut [i32; LAYER_STACKS],
    features: &[usize],
    weights: &[i16],
    psqt: &[i32],
) {
    for &feature in features {
        let base = feature * TRANSFORMED_DIMENSIONS;
        simd::Active::add_i16_row(accumulation, &weights[base..base + TRANSFORMED_DIMENSIONS]);
        let psqt_base = feature * LAYER_STACKS;
        for (target, &weight) in psqt_accumulation
            .iter_mut()
            .zip(&psqt[psqt_base..psqt_base + LAYER_STACKS])
        {
            *target += weight;
        }
    }
}

/// Used for adding active signed-byte `FullThreats` rows and their PSQT
/// lanes.
///
/// The caller owns both destination buffers and supplies feature indices from
/// the validated `FullThreats` table. Complete feature pairs share one pass so
/// each 1,024-lane destination stays live across two ordered additions; an odd
/// final row uses the same operation alone. Widening every serialized signed
/// byte before addition retains the original `i32` accumulator semantics, and
/// model loading guarantees all referenced transform and PSQT rows are whole.
///
/// # Arguments
///
/// * `accumulation` - destination transform accumulator
/// * `psqt_accumulation` - destination material-bucket PSQT lanes
/// * `features` - active `FullThreats` feature indices
/// * `weights` - row-major signed-byte transform weight table
/// * `psqt` - row-major PSQT weight table
///
/// # Panics
///
/// Panics if a feature indexes a row beyond either weight table; model
/// loading sizes both tables to cover every valid feature.
fn add_i8_features(
    accumulation: &mut [i32],
    psqt_accumulation: &mut [i32; LAYER_STACKS],
    features: &[usize],
    weights: &[u8],
    psqt: &[i32],
) {
    let mut pairs = features.chunks_exact(2);
    for pair in &mut pairs {
        let first = pair[0];
        let second = pair[1];
        let first_base = first * TRANSFORMED_DIMENSIONS;
        let second_base = second * TRANSFORMED_DIMENSIONS;
        simd::Active::add_i8_row_pair(
            accumulation,
            &weights[first_base..first_base + TRANSFORMED_DIMENSIONS],
            &weights[second_base..second_base + TRANSFORMED_DIMENSIONS],
        );
        let first_psqt_base = first * LAYER_STACKS;
        let second_psqt_base = second * LAYER_STACKS;
        for ((target, &first_weight), &second_weight) in psqt_accumulation
            .iter_mut()
            .zip(&psqt[first_psqt_base..first_psqt_base + LAYER_STACKS])
            .zip(&psqt[second_psqt_base..second_psqt_base + LAYER_STACKS])
        {
            *target += first_weight;
            *target += second_weight;
        }
    }

    for &feature in pairs.remainder() {
        let base = feature * TRANSFORMED_DIMENSIONS;
        simd::Active::add_i8_row(accumulation, &weights[base..base + TRANSFORMED_DIMENSIONS]);
        let psqt_base = feature * LAYER_STACKS;
        for (target, &weight) in psqt_accumulation
            .iter_mut()
            .zip(&psqt[psqt_base..psqt_base + LAYER_STACKS])
        {
            *target += weight;
        }
    }
}

/// Applies one signed `HalfKA` row delta and its PSQT lanes.
///
/// `sign` is exactly `-1` or `1`. Separate add and subtract kernel calls
/// keep the arithmetic branch-free per lane while checked slices protect
/// every model-row bound.
fn apply_i16_feature_delta(
    accumulation: &mut [i32; TRANSFORMED_DIMENSIONS],
    psqt_accumulation: &mut [i32; LAYER_STACKS],
    feature: usize,
    sign: i32,
    weights: &[i16],
    psqt: &[i32],
) {
    debug_assert!(matches!(sign, -1 | 1));
    let base = feature * TRANSFORMED_DIMENSIONS;
    let row = &weights[base..base + TRANSFORMED_DIMENSIONS];
    if sign > 0 {
        simd::Active::add_i16_row(accumulation, row);
    } else {
        simd::Active::sub_i16_row(accumulation, row);
    }

    let psqt_base = feature * LAYER_STACKS;
    if sign > 0 {
        for (target, &weight) in psqt_accumulation
            .iter_mut()
            .zip(&psqt[psqt_base..psqt_base + LAYER_STACKS])
        {
            *target += weight;
        }
    } else {
        for (target, &weight) in psqt_accumulation
            .iter_mut()
            .zip(&psqt[psqt_base..psqt_base + LAYER_STACKS])
        {
            *target -= weight;
        }
    }
}

/// Applies one signed `FullThreats` row delta and its PSQT lanes.
///
/// `sign` is exactly `-1` or `1`. Separate add and subtract kernel calls
/// keep the arithmetic branch-free per lane while checked slices protect
/// every model-row bound.
fn apply_i8_feature_delta(
    accumulation: &mut [i32; TRANSFORMED_DIMENSIONS],
    psqt_accumulation: &mut [i32; LAYER_STACKS],
    feature: usize,
    sign: i32,
    weights: &[u8],
    psqt: &[i32],
) {
    debug_assert!(matches!(sign, -1 | 1));
    let base = feature * TRANSFORMED_DIMENSIONS;
    let row = &weights[base..base + TRANSFORMED_DIMENSIONS];
    if sign > 0 {
        simd::Active::add_i8_row(accumulation, row);
    } else {
        simd::Active::sub_i8_row(accumulation, row);
    }

    let psqt_base = feature * LAYER_STACKS;
    if sign > 0 {
        for (target, &weight) in psqt_accumulation
            .iter_mut()
            .zip(&psqt[psqt_base..psqt_base + LAYER_STACKS])
        {
            *target += weight;
        }
    } else {
        for (target, &weight) in psqt_accumulation
            .iter_mut()
            .zip(&psqt[psqt_base..psqt_base + LAYER_STACKS])
        {
            *target -= weight;
        }
    }
}

/// Used for applying one pairwise-clipped transformer half and recording
/// nonzero lanes.
///
/// Lanes `i` and `i + HALF_TRANSFORMED_DIMENSIONS` are clamped to
/// `0..=255`, multiplied, and divided by 512; a movemask-style backend
/// scan then collects only the nonzero products in ascending lane order.
///
/// # Arguments
///
/// * `output` - sparse destination appended in dense-lane order
/// * `output_offset` - first transformed lane owned by this perspective
/// * `accumulation` - combined perspective accumulator
///
/// # Panics
///
/// Panics if `accumulation` is shorter than the transformed width or if an
/// appended lane violates the sparse vector's ordering invariants.
fn write_perspective_features(
    output: &mut SparseTransformedFeatures,
    output_offset: usize,
    accumulation: &[i32],
) {
    debug_assert!(output_offset + HALF_TRANSFORMED_DIMENSIONS <= TRANSFORMED_DIMENSIONS);
    let mut values = [0_i32; HALF_TRANSFORMED_DIMENSIONS];
    simd::Active::clipped_products_into(
        &accumulation[..HALF_TRANSFORMED_DIMENSIONS],
        &accumulation[HALF_TRANSFORMED_DIMENSIONS..TRANSFORMED_DIMENSIONS],
        &mut values,
    );
    let mut nonzero = [0_u16; HALF_TRANSFORMED_DIMENSIONS];
    let count = simd::Active::nonzero_indices_into(&values, &mut nonzero);
    for &lane in &nonzero[..count] {
        let index = usize::from(lane);
        output.push_nonzero(output_offset + index, values[index]);
    }
}

/// Fixed-capacity sparse view of the transformed 1,024-lane feature vector.
///
/// Indices are appended in increasing dense-lane order. This preserves each
/// FC0 output's scalar addition order while removing zero multiplications and
/// inference-time allocation.
struct SparseTransformedFeatures {
    /// Used for the increasing transformed-lane indices; `u16` covers the
    /// fixed width.
    indices: [u16; TRANSFORMED_DIMENSIONS],
    /// Used for the values corresponding one-for-one with `indices`.
    values: [i32; TRANSFORMED_DIMENSIONS],
    /// Used for counting the initialized entries at the start of both
    /// arrays.
    len: usize,
}

impl SparseTransformedFeatures {
    /// Used for creating an empty sparse vector in caller-owned fixed
    /// storage.
    ///
    /// # Returns
    ///
    /// A zero-length vector backed by fixed arrays.
    const fn new() -> Self {
        Self {
            indices: [0; TRANSFORMED_DIMENSIONS],
            values: [0; TRANSFORMED_DIMENSIONS],
            len: 0,
        }
    }

    /// Used for recording one nonzero lane in increasing transform order.
    ///
    /// Zero values are skipped so FC0 propagation touches only
    /// contributing lanes.
    ///
    /// # Arguments
    ///
    /// * `index` - dense transformed-lane index
    /// * `value` - clipped product for that lane
    ///
    /// # Panics
    ///
    /// Panics when `index` reaches the transformed width or does not
    /// exceed the previously recorded index.
    fn push_nonzero(&mut self, index: usize, value: i32) {
        if value == 0 {
            return;
        }
        assert!(
            index < TRANSFORMED_DIMENSIONS,
            "NNUE transformed feature index exceeds CURRENT/BIG width"
        );
        if self.len != 0 {
            assert!(
                usize::from(self.indices[self.len - 1]) < index,
                "NNUE transformed features must be appended in increasing order"
            );
        }
        self.indices[self.len] =
            u16::try_from(index).expect("CURRENT/BIG transformed indices fit u16");
        self.values[self.len] = value;
        self.len += 1;
    }

    /// Used for iterating initialized `(index, value)` pairs in dense-lane
    /// order.
    ///
    /// # Returns
    ///
    /// An iterator over exactly the recorded nonzero lanes.
    fn iter(&self) -> impl Iterator<Item = (usize, i32)> + '_ {
        self.indices[..self.len]
            .iter()
            .copied()
            .map(usize::from)
            .zip(self.values[..self.len].iter().copied())
    }
}

/// Output of the sparse transformer before one dense material-bucket network.
///
/// Bundles the sparse feature vector consumed by FC0 with the PSQT half of
/// the evaluation.
struct TransformOutput {
    /// Used for the nonzero side-to-move lanes followed by the nonzero
    /// opponent lanes.
    features: SparseTransformedFeatures,
    /// Used for the halved difference of the two perspective PSQT lanes in
    /// internal score units.
    psqt: i32,
}

/// One of eight CURRENT/BIG dense stacks selected by material count.
///
/// Consists of the sparse first affine layer and the two dense layers that
/// produce the positional score.
struct Architecture {
    /// Used for the sparse first affine layer, including the extra
    /// forward-material lane.
    fc0: SparseAffineLayer,
    /// Used for the hidden layer consuming squared and ordinary clipped
    /// activations.
    fc1: AffineLayer,
    /// Used for the scalar positional output layer.
    fc2: AffineLayer,
}

impl Architecture {
    /// Used for reading the fixed `1024 -> 32`, `62 -> 32`, `32 -> 1`
    /// CURRENT/BIG stack.
    ///
    /// # Arguments
    ///
    /// * `cursor` - bounded cursor positioned at the stack tensors
    ///
    /// # Returns
    ///
    /// The decoded stack with FC0 transposed for sparse propagation.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed tensors, an inconsistent FC0 shape,
    /// or allocation failure.
    fn read(cursor: &mut Cursor<'_>) -> Result<Self, UpstreamNnueError> {
        let fc0 = AffineLayer::read(cursor, TRANSFORMED_DIMENSIONS, FC0_OUTPUT_DIMENSIONS)?;
        Ok(Self {
            fc0: SparseAffineLayer::from_row_major(fc0)?,
            fc1: AffineLayer::read(cursor, L2 * 2, L3)?,
            fc2: AffineLayer::read(cursor, L3, 1)?,
        })
    }

    /// Used for propagating transformed features and adding Stockfish's
    /// forward lane.
    ///
    /// # Arguments
    ///
    /// * `transformed` - sparse nonzero transformed features
    ///
    /// # Returns
    ///
    /// The positional score in internal units before output scaling.
    fn propagate(&self, transformed: &SparseTransformedFeatures) -> i32 {
        let mut fc0 = [0_i32; FC0_OUTPUT_DIMENSIONS];
        self.fc0.forward_sparse_into(transformed, &mut fc0);
        self.propagate_fc0(&fc0)
    }

    /// Used for propagating a complete FC0 vector through the two dense
    /// output layers.
    ///
    /// The caller owns the scratch and must initialize its biases before
    /// accumulating every active transformed lane. This split lets incremental
    /// evaluation avoid materializing sparse features without changing the
    /// full-rebuild path or the forward-material lane.
    ///
    /// # Arguments
    ///
    /// * `fc0` - complete bias-seeded FC0 output vector
    ///
    /// # Returns
    ///
    /// The positional score in internal units, including the scaled
    /// forward-material lane.
    fn propagate_fc0(&self, fc0: &[i32; FC0_OUTPUT_DIMENSIONS]) -> i32 {
        let mut fc1_input = [0_i32; L2 * 2];
        for index in 0..L2 {
            fc1_input[index] = squared_clipped_relu(fc0[index]);
            fc1_input[L2 + index] = clipped_relu(fc0[index]);
        }
        let mut fc1 = [0_i32; L3];
        self.fc1.forward_into(&fc1_input, &mut fc1);
        let mut fc2_input = [0_i32; L3];
        for (target, value) in fc2_input.iter_mut().zip(fc1) {
            *target = clipped_relu(value);
        }
        let output = self.fc2.forward_single(&fc2_input);
        let forward = fc0[L2] * (600 * OUTPUT_SCALE) / (127 * (1 << WEIGHT_SCALE_BITS));
        output + forward
    }
}

/// Input-major sparse view of the fixed CURRENT/BIG first affine layer.
///
/// Stockfish serializes affine matrices output-row first. Sparse propagation
/// needs all output weights for one active input contiguously, so loading
/// validates FC0's exact shape and transposes its logical lanes once.
struct SparseAffineLayer {
    /// Used for one signed 32-bit bias per FC0 output.
    biases: Vec<i32>,
    /// Used for the signed-byte weights in `[input][output]` order.
    input_major_weights: Vec<u8>,
}

impl SparseAffineLayer {
    /// Used for validating and transposing the exact `1024 -> 32`
    /// CURRENT/BIG FC0 matrix.
    ///
    /// # Arguments
    ///
    /// * `layer` - row-major layer as decoded from the stream
    ///
    /// # Returns
    ///
    /// The same parameters with weights reordered to `[input][output]`.
    ///
    /// # Errors
    ///
    /// Returns an error when the layer's dimensions, bias count, or weight
    /// count differ from the fixed CURRENT/BIG FC0 shape, or when the
    /// transposed allocation fails.
    fn from_row_major(layer: AffineLayer) -> Result<Self, UpstreamNnueError> {
        if layer.input_dimensions != TRANSFORMED_DIMENSIONS
            || layer.padded_input_dimensions != TRANSFORMED_DIMENSIONS
            || layer.output_dimensions != FC0_OUTPUT_DIMENSIONS
            || layer.biases.len() != FC0_OUTPUT_DIMENSIONS
        {
            return Err(invalid(format!(
                "invalid CURRENT/BIG FC0 shape: inputs={}, padded={}, outputs={}, biases={}",
                layer.input_dimensions,
                layer.padded_input_dimensions,
                layer.output_dimensions,
                layer.biases.len()
            )));
        }
        let serialized_count = checked_product(
            layer.output_dimensions,
            layer.padded_input_dimensions,
            "serialized FC0 weight count",
        )?;
        if layer.weights.len() != serialized_count {
            return Err(invalid(format!(
                "invalid CURRENT/BIG FC0 weight count: expected {serialized_count}, found {}",
                layer.weights.len()
            )));
        }

        let transposed_count = checked_product(
            layer.input_dimensions,
            layer.output_dimensions,
            "transposed FC0 weight count",
        )?;
        let mut input_major_weights = Vec::new();
        reserve_exact(
            &mut input_major_weights,
            transposed_count,
            "transposed FC0 weights",
        )?;
        for input in 0..layer.input_dimensions {
            for output in 0..layer.output_dimensions {
                input_major_weights
                    .push(layer.weights[output * layer.padded_input_dimensions + input]);
            }
        }
        debug_assert_eq!(input_major_weights.len(), transposed_count);
        Ok(Self {
            biases: layer.biases,
            input_major_weights,
        })
    }

    /// Used for initializing caller-owned FC0 scratch before
    /// transformed-lane additions.
    ///
    /// Model loading guarantees one bias per output. Incremental propagation
    /// calls this exactly once per evaluation so the later additions begin from
    /// the same values as the independent sparse path.
    ///
    /// # Arguments
    ///
    /// * `output` - FC0 scratch overwritten with the layer biases
    fn initialize_output(&self, output: &mut [i32; FC0_OUTPUT_DIMENSIONS]) {
        output.copy_from_slice(&self.biases);
    }

    /// Used for applying one combined perspective directly to bias-seeded
    /// FC0 scratch.
    ///
    /// `output_offset` assigns this perspective its contiguous 512-lane half of
    /// the transformed input. A movemask-style backend scan collects the
    /// nonzero clipped-product lanes first, and only those columns are
    /// propagated, in ascending transformed-index order, preserving the scalar
    /// sparse path's integer operation order. Model-shape validation makes
    /// every selected input-major row complete; checked slices retain memory
    /// safety if that invariant is ever violated internally.
    ///
    /// # Arguments
    ///
    /// * `output` - bias-seeded FC0 scratch
    /// * `output_offset` - first transformed lane owned by this perspective
    /// * `psq` - `HalfKA` perspective accumulator
    /// * `threats` - `FullThreats` perspective accumulator
    fn accumulate_combined_perspective(
        &self,
        output: &mut [i32; FC0_OUTPUT_DIMENSIONS],
        output_offset: usize,
        psq: &[i32],
        threats: &[i32],
    ) {
        debug_assert_eq!(psq.len(), TRANSFORMED_DIMENSIONS);
        debug_assert_eq!(threats.len(), TRANSFORMED_DIMENSIONS);
        debug_assert!(output_offset + HALF_TRANSFORMED_DIMENSIONS <= TRANSFORMED_DIMENSIONS);
        let mut values = [0_i32; HALF_TRANSFORMED_DIMENSIONS];
        simd::Active::combined_clipped_products_into(
            &psq[..HALF_TRANSFORMED_DIMENSIONS],
            &threats[..HALF_TRANSFORMED_DIMENSIONS],
            &psq[HALF_TRANSFORMED_DIMENSIONS..TRANSFORMED_DIMENSIONS],
            &threats[HALF_TRANSFORMED_DIMENSIONS..TRANSFORMED_DIMENSIONS],
            &mut values,
        );
        let mut nonzero = [0_u16; HALF_TRANSFORMED_DIMENSIONS];
        let count = simd::Active::nonzero_indices_into(&values, &mut nonzero);
        for &lane in &nonzero[..count] {
            let index = usize::from(lane);
            let base = (output_offset + index) * FC0_OUTPUT_DIMENSIONS;
            simd::Active::madd_i8_row(
                output.as_mut_slice(),
                &self.input_major_weights[base..base + FC0_OUTPUT_DIMENSIONS],
                values[index],
            );
        }
    }

    /// Used for propagating only initialized transformed lanes into fixed
    /// FC0 scratch.
    ///
    /// # Arguments
    ///
    /// * `input` - sparse nonzero transformed features
    /// * `output` - destination reseeded from the layer biases
    fn forward_sparse_into(&self, input: &SparseTransformedFeatures, output: &mut [i32]) {
        debug_assert_eq!(output.len(), FC0_OUTPUT_DIMENSIONS);
        output.copy_from_slice(&self.biases);
        for (index, value) in input.iter() {
            let base = index * FC0_OUTPUT_DIMENSIONS;
            simd::Active::madd_i8_row(
                output,
                &self.input_major_weights[base..base + FC0_OUTPUT_DIMENSIONS],
                value,
            );
        }
    }
}

/// Signed-byte affine layer with Stockfish's 32-element input-row padding.
///
/// Serves as the row-major decoded form for FC1, FC2, and the FC0 matrix
/// before its sparse transposition.
struct AffineLayer {
    /// Used for the logical number of input values consumed during
    /// propagation.
    input_dimensions: usize,
    /// Used for the serialized row stride rounded up to a multiple of 32.
    padded_input_dimensions: usize,
    /// Used for the number of output rows and biases.
    output_dimensions: usize,
    /// Used for one signed 32-bit bias per output row.
    biases: Vec<i32>,
    /// Used for the row-major signed-byte weights, retained as bytes for
    /// exact decoding.
    weights: Vec<u8>,
}

impl AffineLayer {
    /// Used for reading biases and a padded row-major weight matrix.
    ///
    /// # Arguments
    ///
    /// * `cursor` - bounded cursor positioned at the layer tensors
    /// * `input_dimensions` - logical input width
    /// * `output_dimensions` - number of output rows
    ///
    /// # Returns
    ///
    /// The decoded layer with its padded serialized stride recorded.
    ///
    /// # Errors
    ///
    /// Returns an error for truncated data, width overflow, or allocation
    /// failure.
    fn read(
        cursor: &mut Cursor<'_>,
        input_dimensions: usize,
        output_dimensions: usize,
    ) -> Result<Self, UpstreamNnueError> {
        let padded_input_dimensions = ceil_to_multiple(input_dimensions, 32)?;
        let mut biases = Vec::new();
        reserve_exact(&mut biases, output_dimensions, "affine biases")?;
        for _ in 0..output_dimensions {
            biases.push(cursor.read_i32("affine bias")?);
        }
        let count = checked_product(
            output_dimensions,
            padded_input_dimensions,
            "affine weight count",
        )?;
        let weights = cursor.read_byte_vector(count, "affine weights")?;
        Ok(Self {
            input_dimensions,
            padded_input_dimensions,
            output_dimensions,
            biases,
            weights,
        })
    }

    /// Used for applying every affine row into caller-owned inference
    /// scratch.
    ///
    /// # Arguments
    ///
    /// * `input` - dense activations of the logical input width
    /// * `output` - destination receiving one biased dot product per row
    fn forward_into(&self, input: &[i32], output: &mut [i32]) {
        debug_assert_eq!(input.len(), self.input_dimensions);
        debug_assert_eq!(output.len(), self.output_dimensions);
        for (row, target) in output.iter_mut().enumerate() {
            let base = row * self.padded_input_dimensions;
            let row_weights = &self.weights[base..base + input.len()];
            *target = self.biases[row].wrapping_add(simd::Active::dot_i8_row(row_weights, input));
        }
    }

    /// Used for specialized scalar-output propagation in the final layer.
    ///
    /// # Arguments
    ///
    /// * `input` - dense activations of the logical input width
    ///
    /// # Returns
    ///
    /// The single biased dot product in internal units.
    fn forward_single(&self, input: &[i32]) -> i32 {
        debug_assert_eq!(self.output_dimensions, 1);
        debug_assert_eq!(input.len(), self.input_dimensions);
        self.biases[0].wrapping_add(simd::Active::dot_i8_row(
            &self.weights[..input.len()],
            input,
        ))
    }
}

/// Used for applying Stockfish's fixed-point clipped `ReLU`.
///
/// # Arguments
///
/// * `value` - raw affine activation
///
/// # Returns
///
/// The activation shifted by [`WEIGHT_SCALE_BITS`] and clamped to
/// `0..=127`.
fn clipped_relu(value: i32) -> i32 {
    (value >> WEIGHT_SCALE_BITS).clamp(0, 127)
}

/// Used for applying the scaled squared clipped activation before the
/// second layer.
///
/// # Arguments
///
/// * `value` - raw affine activation
///
/// # Returns
///
/// The squared activation rescaled in 64-bit arithmetic and capped at 127.
fn squared_clipped_relu(value: i32) -> i32 {
    let squared = i64::from(value) * i64::from(value);
    i32::try_from((squared >> (2 * WEIGHT_SCALE_BITS + 7)).min(127)).unwrap_or(127)
}

/// Used for converting Janus pieces and `A8 = 0` squares to Stockfish
/// piece/square codes.
///
/// # Arguments
///
/// * `position` - validated source position
///
/// # Returns
///
/// A 64-entry board indexed by Stockfish square, with `0` marking empty
/// squares.
fn stockfish_board(position: &Position) -> [usize; 64] {
    let mut board = [0_usize; 64];
    for index in 0_u8..64 {
        let square = Square::new(index).expect("loop index is a valid square");
        board[stockfish_square(square)] = position.piece_at(square).map_or(0, stockfish_piece);
    }
    board
}

/// Used for converting a Janus `A8 = 0` square to Stockfish's `A1 = 0`
/// numbering.
///
/// # Arguments
///
/// * `square` - Janus square
///
/// # Returns
///
/// The vertically flipped Stockfish square index.
const fn stockfish_square(square: Square) -> usize {
    (square.index() ^ 0x38) as usize
}

/// Used for converting a Janus piece to Stockfish's color/type code.
///
/// # Arguments
///
/// * `piece` - Janus piece
///
/// # Returns
///
/// The one-based type component, with `8` added for Black pieces.
const fn stockfish_piece(piece: Piece) -> usize {
    piece.kind.index()
        + 1
        + if matches!(piece.color, Color::Black) {
            8
        } else {
            0
        }
}

/// Used for converting a Stockfish perspective selector to a Janus color.
///
/// # Arguments
///
/// * `perspective` - [`WHITE`] or [`BLACK`] selector
///
/// # Returns
///
/// The corresponding [`Color`].
const fn color_from_perspective(perspective: usize) -> Color {
    if perspective == WHITE {
        Color::White
    } else {
        Color::Black
    }
}

/// Returns one king-relative `HalfKA` row for a piece-square observation.
fn half_ka_feature(perspective: usize, piece: Piece, square: Square, king_square: Square) -> usize {
    half_ka_index(
        perspective,
        stockfish_square(square),
        stockfish_piece(piece),
        stockfish_square(king_square),
    )
}

/// Writes an `i16`-weighted accumulator child without an intermediate copy.
///
/// Every weight slice must have [`TRANSFORMED_DIMENSIONS`] entries. The
/// destination may not alias the parent; per-ply `HalfKA` ownership enforces that
/// invariant. A capture row, when present, is applied last to preserve the
/// established signed `i32` operation order.
fn write_half_ka_accumulator_delta(
    child: &mut [i32; TRANSFORMED_DIMENSIONS],
    parent: &[i32; TRANSFORMED_DIMENSIONS],
    removed: &[i16],
    added: &[i16],
    captured: Option<&[i16]>,
) {
    debug_assert_eq!(removed.len(), TRANSFORMED_DIMENSIONS);
    debug_assert_eq!(added.len(), TRANSFORMED_DIMENSIONS);
    debug_assert!(captured.map_or(true, |weights| weights.len() == TRANSFORMED_DIMENSIONS));
    simd::Active::write_i16_delta(child, parent, removed, added, captured);
}

/// Writes the eight material-bucket PSQT lanes for one fused child delta.
///
/// All slices must cover exactly [`LAYER_STACKS`] entries. The update order
/// matches [`write_half_ka_accumulator_delta`], keeping every bucket bit-exact
/// with the former copy and sequential row applications.
fn write_half_ka_psqt_delta(
    child: &mut [i32; LAYER_STACKS],
    parent: &[i32; LAYER_STACKS],
    removed: &[i32],
    added: &[i32],
    captured: Option<&[i32]>,
) {
    debug_assert_eq!(removed.len(), LAYER_STACKS);
    debug_assert_eq!(added.len(), LAYER_STACKS);
    debug_assert!(captured.map_or(true, |weights| weights.len() == LAYER_STACKS));
    if let Some(captured) = captured {
        for ((((target, &source), &removed), &added), &captured) in child
            .iter_mut()
            .zip(parent)
            .zip(removed)
            .zip(added)
            .zip(captured)
        {
            *target = source - removed + added - captured;
        }
    } else {
        for (((target, &source), &removed), &added) in
            child.iter_mut().zip(parent).zip(removed).zip(added)
        {
            *target = source - removed + added;
        }
    }
}

/// Used for adding or removing one king-relative `HalfKA` row and its
/// PSQT lanes.
///
/// # Arguments
///
/// * `slot` - accumulator slot updated in place
/// * `perspective` - Stockfish color selector of the updated side
/// * `piece` - Janus piece whose feature changes
/// * `square` - Janus square the piece occupies or leaves
/// * `king_square` - Janus square of the perspective's king
/// * `sign` - `1` to add the feature, `-1` to remove it
/// * `transformer` - source of the transform and PSQT weight rows
#[allow(clippy::too_many_arguments)]
fn apply_half_ka_feature(
    slot: &mut HalfKaSlot,
    perspective: usize,
    piece: Piece,
    square: Square,
    king_square: Square,
    sign: i32,
    transformer: &FeatureTransformer,
) {
    debug_assert!(matches!(sign, -1 | 1));
    let feature = half_ka_feature(perspective, piece, square, king_square);
    let base = feature * TRANSFORMED_DIMENSIONS;
    let row = &transformer.psq_weights[base..base + TRANSFORMED_DIMENSIONS];
    if sign > 0 {
        simd::Active::add_i16_row(&mut slot.accumulation[perspective], row);
    } else {
        simd::Active::sub_i16_row(&mut slot.accumulation[perspective], row);
    }
    let psqt_base = feature * LAYER_STACKS;
    for (target, &weight) in slot.psqt[perspective]
        .iter_mut()
        .zip(&transformer.psqt_weights[psqt_base..psqt_base + LAYER_STACKS])
    {
        *target += sign * weight;
    }
}

/// Fixed-capacity sparse feature list used by hot inference paths.
///
/// The capacity is a board-derived compile-time bound, so collection never
/// allocates.
struct ActiveFeatures<const CAPACITY: usize> {
    /// Used for the preallocated feature slots.
    values: [usize; CAPACITY],
    /// Used for counting the initialized slots at the start of `values`.
    len: usize,
}

impl<const CAPACITY: usize> ActiveFeatures<CAPACITY> {
    /// Used for creating an empty feature list without touching the heap.
    ///
    /// # Returns
    ///
    /// A zero-length list backed by a fixed array.
    const fn new() -> Self {
        Self {
            values: [0; CAPACITY],
            len: 0,
        }
    }

    /// Used for adding one feature within the board-derived capacity
    /// bound.
    ///
    /// # Arguments
    ///
    /// * `feature` - sparse feature index to record
    ///
    /// # Panics
    ///
    /// Panics when the fixed capacity is exhausted.
    fn push(&mut self, feature: usize) {
        assert!(
            self.len < CAPACITY,
            "NNUE active feature count exceeded its board-derived bound"
        );
        self.values[self.len] = feature;
        self.len += 1;
    }

    /// Used for viewing exactly the initialized features in deterministic
    /// scan order.
    ///
    /// # Returns
    ///
    /// The recorded features.
    fn as_slice(&self) -> &[usize] {
        &self.values[..self.len]
    }
}

/// Used for enumerating active king-relative `HalfKA` features for one
/// perspective.
///
/// # Arguments
///
/// * `board` - Stockfish-coded piece placement
/// * `perspective` - Stockfish color selector of the anchoring king
///
/// # Returns
///
/// One feature index per occupied square, in ascending square order.
///
/// # Panics
///
/// Panics if the board lacks the perspective's king.
fn active_half_ka(board: &[usize; 64], perspective: usize) -> ActiveFeatures<MAX_ACTIVE_HALF_KA> {
    let king_square = find_king(board, perspective);
    let mut active = ActiveFeatures::new();
    for (square, &piece) in board.iter().enumerate() {
        if piece != 0 {
            active.push(half_ka_index(perspective, square, piece, king_square));
        }
    }
    active
}

/// Used for mapping one piece to its perspective- and
/// king-bucket-relative `HalfKA` index.
///
/// The square is flipped for the Black perspective and mirrored when the
/// king stands on files A-D.
///
/// # Arguments
///
/// * `perspective` - Stockfish color selector of the anchoring king
/// * `square` - Stockfish square of the piece
/// * `piece` - Stockfish piece code
/// * `king_square` - Stockfish square of the anchoring king
///
/// # Returns
///
/// The sparse `HalfKA` feature index.
fn half_ka_index(perspective: usize, square: usize, piece: usize, king_square: usize) -> usize {
    let flip = 56 * perspective;
    let orientation = if file(king_square) < 4 { 7 } else { 0 };
    (square ^ orientation ^ flip)
        + piece_square_offset(perspective, piece)
        + king_bucket(king_square ^ flip)
}

/// Used for computing the relative-color piece-plane offset within a
/// `HalfKA` king bucket.
///
/// # Arguments
///
/// * `perspective` - Stockfish color selector of the anchoring king
/// * `piece` - Stockfish piece code
///
/// # Returns
///
/// The plane offset in squares; both kings share the final plane.
fn piece_square_offset(perspective: usize, piece: usize) -> usize {
    let kind = type_of(piece);
    if kind == KING {
        return 10 * 64;
    }
    let same_color = color_of(piece) == perspective;
    let plane = 2 * (kind - 1) + usize::from(!same_color);
    plane * 64
}

/// Used for computing the horizontally mirrored `HalfKA` king-bucket
/// base.
///
/// # Arguments
///
/// * `square` - perspective-flipped Stockfish king square
///
/// # Returns
///
/// The bucket base offset, each bucket spanning eleven 64-square planes.
fn king_bucket(square: usize) -> usize {
    let rank_from_top = 7 - rank(square);
    let mirrored_file = file(square).min(7 - file(square));
    ((rank_from_top * 4) + mirrored_file) * 11 * 64
}

/// Used for locating a king in a validated Stockfish-coded board.
///
/// # Arguments
///
/// * `board` - Stockfish-coded piece placement
/// * `color` - Stockfish color selector of the requested king
///
/// # Returns
///
/// The Stockfish square holding the king.
///
/// # Panics
///
/// Panics when the board lacks the requested king; callers derive the board
/// from a validated [`Position`].
fn find_king(board: &[usize; 64], color: usize) -> usize {
    let king = make_piece(color, KING);
    board
        .iter()
        .position(|&piece| piece == king)
        .expect("janus-core positions contain both kings")
}

/// Returns the only king-square property used by `FullThreats` feature indices.
const fn threat_orientation(king_square: usize) -> u8 {
    if file(king_square) < 4 {
        0
    } else {
        1
    }
}

/// Enumerates one sorted perspective-independent occupied-contact collection.
///
/// Kings are not `FullThreats` attackers. Invalid attacker/victim classes stay
/// in the contact collection and are filtered only when projected; retaining
/// them keeps the board relation canonical across both perspectives.
fn active_threat_instances(board: &[usize; 64]) -> ActiveThreatInstances {
    let mut instances = ActiveThreatInstances::new();
    for (from, &attacker) in board.iter().enumerate() {
        if attacker == 0 || type_of(attacker) == KING {
            continue;
        }
        visit_threat_contacts_from(board, attacker, from, |instance| {
            instances.push(instance);
        });
    }
    instances.sort();
    instances
}

/// Projects canonical contacts into one perspective's valid feature indices.
fn project_threat_instances(
    instances: &[ThreatInstance],
    perspective: usize,
    king_square: usize,
    tables: &ThreatTables,
) -> ActiveFeatures<MAX_ACTIVE_THREATS> {
    let mut features = ActiveFeatures::new();
    for &instance in instances {
        let feature = instance.feature(perspective, king_square, tables);
        if feature < THREAT_DIMENSIONS {
            features.push(feature);
        }
    }
    features
}

/// Used for enumerating occupied-target `FullThreats` features for one
/// perspective.
///
/// The validated Stockfish piece codes are indexed once into stack-owned origin
/// bitsets. Traversal retains the format's relative-color, piece-type, and
/// ascending-square order, so the resulting fixed-capacity list remains
/// byte-for-byte compatible with scalar board scanning. Contact ordering within
/// each origin is still owned by [`append_threats_from`].
///
/// # Arguments
///
/// * `board` - Stockfish-coded piece placement
/// * `perspective` - Stockfish color selector of the observing king
/// * `tables` - validated `FullThreats` index tables
///
/// # Returns
///
/// The active feature indices in the format's deterministic order.
///
/// # Panics
///
/// Panics if the board lacks the perspective's king.
fn active_threats(
    board: &[usize; 64],
    perspective: usize,
    tables: &ThreatTables,
) -> ActiveFeatures<MAX_ACTIVE_THREATS> {
    let king_square = find_king(board, perspective);
    let mut origins_by_piece = [0_u64; NUM_VALID_TARGETS.len()];
    for (from, &piece) in board.iter().enumerate() {
        debug_assert!(piece < origins_by_piece.len());
        if piece != 0 {
            origins_by_piece[piece] |= 1_u64 << from;
        }
    }

    let mut active = ActiveFeatures::new();
    for color_selector in [WHITE, BLACK] {
        let color = perspective ^ color_selector;
        for piece_type in PAWN..KING {
            let attacker = make_piece(color, piece_type);
            let mut origins = origins_by_piece[attacker];
            while origins != 0 {
                let from = origins.trailing_zeros() as usize;
                origins &= origins - 1;
                append_threats_from(
                    board,
                    perspective,
                    attacker,
                    from,
                    king_square,
                    tables,
                    &mut active,
                );
            }
        }
    }
    active
}

/// Used for appending occupied targets attacked by one piece, stopping
/// slider rays at first contact.
///
/// # Arguments
///
/// * `board` - Stockfish-coded piece placement
/// * `perspective` - Stockfish color selector of the observing king
/// * `attacker` - Stockfish piece code at `from`
/// * `from` - Stockfish origin square
/// * `king_square` - Stockfish square of the perspective's king
/// * `tables` - validated `FullThreats` index tables
/// * `active` - destination feature list
fn append_threats_from(
    board: &[usize; 64],
    perspective: usize,
    attacker: usize,
    from: usize,
    king_square: usize,
    tables: &ThreatTables,
    active: &mut ActiveFeatures<MAX_ACTIVE_THREATS>,
) {
    visit_threat_contacts_from(board, attacker, from, |instance| {
        let index = instance.feature(perspective, king_square, tables);
        if index < THREAT_DIMENSIONS {
            active.push(index);
        }
    });
}

/// Visits every occupied contact emitted by one `FullThreats` attacker.
///
/// Slider rays stop at their first occupant, leapers emit only occupied
/// targets, and pawns include the format's pawn-on-pawn push contact. The
/// caller chooses whether to project contacts immediately or retain them for a
/// later parent/child diff.
fn visit_threat_contacts_from(
    board: &[usize; 64],
    attacker: usize,
    from: usize,
    mut visit: impl FnMut(ThreatInstance),
) {
    let piece_type = type_of(attacker);
    match piece_type {
        PAWN => {
            let forward = if color_of(attacker) == WHITE { 1 } else { -1 };
            let next_rank = rank(from) as i32 + forward;
            if !(0..8).contains(&next_rank) {
                return;
            }
            for file_delta in [-1, 1] {
                let target_file = file(from) as i32 + file_delta;
                if (0..8).contains(&target_file) {
                    let to = square(target_file as usize, next_rank as usize);
                    if board[to] != 0 {
                        visit(ThreatInstance::new(attacker, from, board[to], to));
                    }
                }
            }
            let to = square(file(from), next_rank as usize);
            if type_of(board[to]) == PAWN {
                visit(ThreatInstance::new(attacker, from, board[to], to));
            }
        }
        KNIGHT | KING => {
            let deltas = if piece_type == KNIGHT {
                &KNIGHT_DELTAS[..]
            } else {
                &KING_DELTAS[..]
            };
            for &(file_delta, rank_delta) in deltas {
                let to_file = file(from) as i32 + file_delta;
                let to_rank = rank(from) as i32 + rank_delta;
                if on_board(to_file, to_rank) {
                    let to = square(to_file as usize, to_rank as usize);
                    if board[to] != 0 {
                        visit(ThreatInstance::new(attacker, from, board[to], to));
                    }
                }
            }
        }
        BISHOP | ROOK | QUEEN => {
            let directions: &[(i32, i32)] = match piece_type {
                BISHOP => &BISHOP_DIRECTIONS,
                ROOK => &ROOK_DIRECTIONS,
                _ => &QUEEN_DIRECTIONS,
            };
            for &(file_delta, rank_delta) in directions {
                let mut to_file = file(from) as i32 + file_delta;
                let mut to_rank = rank(from) as i32 + rank_delta;
                while on_board(to_file, to_rank) {
                    let to = square(to_file as usize, to_rank as usize);
                    if board[to] != 0 {
                        visit(ThreatInstance::new(attacker, from, board[to], to));
                        break;
                    }
                    to_file += file_delta;
                    to_rank += rank_delta;
                }
            }
        }
        _ => {}
    }
}

/// Lookup tables reproducing Stockfish `FullThreats` feature ordering.
///
/// Invalid attacker/target combinations map to the sentinel
/// [`THREAT_DIMENSIONS`]. Valid indices are assembled from an attacker/target
/// class base, an origin-square prefix, and a deterministic target order.
struct ThreatTables {
    /// Used for the per-class feature stride of each attacker piece code:
    /// its pseudo-target total over valid origin squares (pawn back ranks
    /// excluded).
    helper_piece_offsets: [usize; 16],
    /// Used for the global feature-space base per attacker piece code.
    helper_global_offsets: [usize; 16],
    /// Used for the prefix offset for each attacker piece code and origin
    /// square.
    offsets: [[usize; 64]; 16],
    /// Used for the base selected by oriented attacker, attacked piece,
    /// and square ordering.
    index_lut1: [[[usize; 2]; 16]; 16],
    /// Used for the target-order rank indexed by piece code, origin, and
    /// destination.
    index_lut2: Vec<u8>,
}

impl ThreatTables {
    /// Used for reconstructing the immutable CURRENT `FullThreats`
    /// ordering tables.
    ///
    /// Walks every piece code and origin square in serialization order,
    /// recording per-square prefix offsets, target-order ranks, and
    /// attacker/attacked class bases.
    ///
    /// # Returns
    ///
    /// The fully populated tables.
    ///
    /// # Errors
    ///
    /// Returns an error when the target-order table allocation fails.
    fn current() -> Result<Self, UpstreamNnueError> {
        let index_lut2_length = 16 * 64 * 64;
        let mut index_lut2 = Vec::new();
        reserve_exact(
            &mut index_lut2,
            index_lut2_length,
            "FullThreats target-order table",
        )?;
        index_lut2.resize(index_lut2_length, 0);
        let mut tables = Self {
            helper_piece_offsets: [0; 16],
            helper_global_offsets: [0; 16],
            offsets: [[0; 64]; 16],
            index_lut1: [[[THREAT_DIMENSIONS; 2]; 16]; 16],
            index_lut2,
        };
        let mut cumulative_offset = 0;
        for piece in ALL_PIECES {
            let mut cumulative_piece_offset = 0;
            for from in 0..64 {
                tables.offsets[piece][from] = cumulative_piece_offset;
                let targets = pseudo_targets(type_of(piece), color_of(piece), from);
                for (order, &target) in targets.iter().enumerate() {
                    tables.index_lut2[target_order_offset(piece, from, target)] =
                        u8::try_from(order).expect("a chess ray has fewer than 256 targets");
                }
                if type_of(piece) != PAWN || (8..=55).contains(&from) {
                    cumulative_piece_offset += targets.len();
                }
            }
            tables.helper_piece_offsets[piece] = cumulative_piece_offset;
            tables.helper_global_offsets[piece] = cumulative_offset;
            cumulative_offset += NUM_VALID_TARGETS[piece] * cumulative_piece_offset;
        }
        debug_assert_eq!(cumulative_offset, THREAT_DIMENSIONS);

        for attacker in ALL_PIECES {
            for attacked in ALL_PIECES {
                let attacker_type = type_of(attacker);
                let attacked_type = type_of(attacked);
                let mapped = THREAT_MAP[attacker_type - 1][attacked_type - 1];
                if mapped < 0 {
                    continue;
                }
                let feature = tables.helper_global_offsets[attacker]
                    + (color_of(attacked) * (NUM_VALID_TARGETS[attacker] / 2) + mapped as usize)
                        * tables.helper_piece_offsets[attacker];
                tables.index_lut1[attacker][attacked][0] = feature;
                let enemy = (attacker ^ attacked) == 8;
                let semi_excluded =
                    attacker_type == attacked_type && (enemy || attacker_type != PAWN);
                if !semi_excluded {
                    tables.index_lut1[attacker][attacked][1] = feature;
                }
            }
        }
        Ok(tables)
    }

    /// Used for mapping one occupied attack to its king-oriented feature
    /// index or sentinel.
    ///
    /// Squares are mirrored by king file and flipped by perspective; piece
    /// codes swap colors for the Black perspective. The square-ordering
    /// lane resolves semi-excluded same-type attacks.
    ///
    /// # Arguments
    ///
    /// * `perspective` - Stockfish color selector of the observing king
    /// * `attacker` - Stockfish piece code at `from`
    /// * `from` - Stockfish origin square
    /// * `to` - Stockfish destination square
    /// * `attacked` - Stockfish piece code at `to`
    /// * `king_square` - Stockfish square of the perspective's king
    ///
    /// # Returns
    ///
    /// A valid index below [`THREAT_DIMENSIONS`], or a sentinel value no
    /// smaller than it for unsupported combinations.
    fn threat_index(
        &self,
        perspective: usize,
        attacker: usize,
        from: usize,
        to: usize,
        attacked: usize,
        king_square: usize,
    ) -> usize {
        let orientation = (if file(king_square) < 4 { 0 } else { 7 }) ^ (56 * perspective);
        let from_oriented = from ^ orientation;
        let to_oriented = to ^ orientation;
        let swap = 8 * perspective;
        let attacker_oriented = attacker ^ swap;
        let attacked_oriented = attacked ^ swap;
        self.index_lut1[attacker_oriented][attacked_oriented]
            [usize::from(from_oriented < to_oriented)]
            + self.offsets[attacker_oriented][from_oriented]
            + usize::from(
                self.index_lut2[target_order_offset(attacker_oriented, from_oriented, to_oriented)],
            )
    }
}

/// Used for flattening a piece/origin/destination tuple into the
/// target-order table.
///
/// # Arguments
///
/// * `piece` - Stockfish piece code
/// * `from` - Stockfish origin square
/// * `to` - Stockfish destination square
///
/// # Returns
///
/// The flat `index_lut2` offset.
const fn target_order_offset(piece: usize, from: usize, to: usize) -> usize {
    piece * 64 * 64 + from * 64 + to
}

/// Used for producing the sorted pseudo-legal target squares behind
/// Stockfish feature ordering.
///
/// # Arguments
///
/// * `piece_type` - Stockfish type component
/// * `color` - Stockfish color selector, deciding pawn direction
/// * `from` - Stockfish origin square
///
/// # Returns
///
/// Ascending target squares reachable on an empty board.
fn pseudo_targets(piece_type: usize, color: usize, from: usize) -> Vec<usize> {
    let mut targets = Vec::with_capacity(32);
    match piece_type {
        PAWN => {
            let next_rank = rank(from) as i32 + if color == WHITE { 1 } else { -1 };
            if (0..8).contains(&next_rank) {
                targets.push(square(file(from), next_rank as usize));
                if file(from) > 0 {
                    targets.push(square(file(from) - 1, next_rank as usize));
                }
                if file(from) < 7 {
                    targets.push(square(file(from) + 1, next_rank as usize));
                }
            }
        }
        KNIGHT | KING => {
            let deltas = if piece_type == KNIGHT {
                &KNIGHT_DELTAS[..]
            } else {
                &KING_DELTAS[..]
            };
            for &(file_delta, rank_delta) in deltas {
                let target_file = file(from) as i32 + file_delta;
                let target_rank = rank(from) as i32 + rank_delta;
                if on_board(target_file, target_rank) {
                    targets.push(square(target_file as usize, target_rank as usize));
                }
            }
        }
        BISHOP | ROOK | QUEEN => {
            let directions: &[(i32, i32)] = match piece_type {
                BISHOP => &BISHOP_DIRECTIONS,
                ROOK => &ROOK_DIRECTIONS,
                _ => &QUEEN_DIRECTIONS,
            };
            for &(file_delta, rank_delta) in directions {
                let mut target_file = file(from) as i32 + file_delta;
                let mut target_rank = rank(from) as i32 + rank_delta;
                while on_board(target_file, target_rank) {
                    targets.push(square(target_file as usize, target_rank as usize));
                    target_file += file_delta;
                    target_rank += rank_delta;
                }
            }
        }
        _ => {}
    }
    targets.sort_unstable();
    targets
}

/// Used for combining Stockfish color and type components into one piece
/// code.
///
/// # Arguments
///
/// * `color` - Stockfish color selector
/// * `piece_type` - Stockfish type component
///
/// # Returns
///
/// The combined piece code, with `8` added for Black.
const fn make_piece(color: usize, piece_type: usize) -> usize {
    piece_type + if color == BLACK { 8 } else { 0 }
}

/// Used for extracting the three-bit type component from a Stockfish
/// piece code.
///
/// # Arguments
///
/// * `piece` - Stockfish piece code
///
/// # Returns
///
/// The type component in `PAWN..=KING`, or `0` for empty.
const fn type_of(piece: usize) -> usize {
    piece & 7
}

/// Used for extracting the color selector from a Stockfish piece code.
///
/// # Arguments
///
/// * `piece` - Stockfish piece code
///
/// # Returns
///
/// [`WHITE`] or [`BLACK`].
const fn color_of(piece: usize) -> usize {
    piece >> 3
}

/// Used for retrieving the zero-based file of a Stockfish-numbered
/// square.
///
/// # Arguments
///
/// * `square` - Stockfish square
///
/// # Returns
///
/// The file in `0..8`.
const fn file(square: usize) -> usize {
    square & 7
}

/// Used for retrieving the zero-based rank of a Stockfish-numbered
/// square.
///
/// # Arguments
///
/// * `square` - Stockfish square
///
/// # Returns
///
/// The rank in `0..8`.
const fn rank(square: usize) -> usize {
    square >> 3
}

/// Used for combining a zero-based file and rank into Stockfish square
/// numbering.
///
/// # Arguments
///
/// * `file` - zero-based file
/// * `rank` - zero-based rank
///
/// # Returns
///
/// The Stockfish square index.
const fn square(file: usize, rank: usize) -> usize {
    rank * 8 + file
}

/// Used for testing whether signed file/rank coordinates lie on the
/// chessboard.
///
/// # Arguments
///
/// * `file` - candidate file
/// * `rank` - candidate rank
///
/// # Returns
///
/// `true` when both coordinates are within `0..8`.
const fn on_board(file: i32, rank: i32) -> bool {
    file >= 0 && file < 8 && rank >= 0 && rank < 8
}

/// Used for selecting one of eight four-piece material buckets.
///
/// # Arguments
///
/// * `piece_count` - total occupied squares
///
/// # Returns
///
/// `(piece_count - 1) / 4` capped at the last bucket; zero pieces map to
/// bucket zero.
fn bucket(piece_count: usize) -> usize {
    if piece_count == 0 {
        0
    } else {
        ((piece_count - 1) / 4).min(LAYER_STACKS - 1)
    }
}

/// Used for applying the one-bit rotation from Stockfish feature hash
/// composition.
///
/// # Arguments
///
/// * `value` - hash to rotate
///
/// # Returns
///
/// The value rotated left by one bit.
const fn rotate_left_one(value: u32) -> u32 {
    value.rotate_left(1)
}

/// Used for computing the accepted combined FullThreats/HalfKA
/// transformer hash.
///
/// # Returns
///
/// The rotated `FullThreats` hash xored with the `HalfKA` hash and the
/// doubled transform width.
const fn current_big_feature_hash() -> u32 {
    let combined = rotate_left_one(FULL_THREATS_HASH) ^ HALF_KA_HASH;
    combined ^ ((TRANSFORMED_DIMENSIONS as u32) * 2)
}

/// Used for extending a Stockfish architecture hash with one affine
/// layer.
///
/// # Arguments
///
/// * `previous` - hash of the preceding layers
/// * `outputs` - output width of the appended affine layer
///
/// # Returns
///
/// The extended hash.
const fn affine_hash(previous: u32, outputs: usize) -> u32 {
    0xcc03_dae4_u32.wrapping_add(outputs as u32) ^ (previous >> 1) ^ (previous << 31)
}

/// Used for extending a Stockfish architecture hash with a clipped-ReLU
/// layer.
///
/// # Arguments
///
/// * `previous` - hash of the preceding layers
///
/// # Returns
///
/// The extended hash.
const fn clipped_relu_hash(previous: u32) -> u32 {
    0x538d_24c7_u32.wrapping_add(previous)
}

/// Used for computing the accepted CURRENT/BIG dense-stack architecture
/// hash.
///
/// # Returns
///
/// The chained hash of FC0, two clipped-ReLU stages, FC1, and the scalar
/// output layer.
const fn current_big_arch_hash() -> u32 {
    let mut hash = 0xec42_e90d ^ ((TRANSFORMED_DIMENSIONS as u32) * 2);
    hash = affine_hash(hash, L2 + 1);
    hash = clipped_relu_hash(hash);
    hash = affine_hash(hash, L3);
    hash = clipped_relu_hash(hash);
    affine_hash(hash, 1)
}

/// Used for computing the complete accepted network hash from the feature
/// and stack hashes.
///
/// # Returns
///
/// The transformer hash xored with the architecture hash.
const fn current_big_network_hash() -> u32 {
    current_big_feature_hash() ^ current_big_arch_hash()
}

/// Bounds-checked cursor over the complete upstream NNUE byte sequence.
///
/// Fixed-width fields are little-endian. Compressed arrays are delegated to a
/// bounded [`LebReader`], and the top-level parser requires exact EOF.
struct Cursor<'a> {
    /// Used for the complete model bytes.
    bytes: &'a [u8],
    /// Used for the offset of the next unread byte.
    offset: usize,
}

impl<'a> Cursor<'a> {
    /// Used for creating a cursor at the version field.
    ///
    /// # Arguments
    ///
    /// * `bytes` - complete serialized model
    ///
    /// # Returns
    ///
    /// A cursor positioned at offset zero.
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    /// Used for taking one bounded byte range, advancing only on success.
    ///
    /// # Arguments
    ///
    /// * `length` - number of bytes to read
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// The borrowed byte range.
    ///
    /// # Errors
    ///
    /// Returns an error when the range overflows `usize` or runs past the
    /// end of the stream.
    fn read_slice(&mut self, length: usize, label: &str) -> Result<&'a [u8], UpstreamNnueError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| invalid(format!("offset overflow while reading {label}")))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid(format!("Stockfish NNUE ended while reading {label}")))?;
        self.offset = end;
        Ok(value)
    }

    /// Used for reading one little-endian unsigned 32-bit field.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// The decoded value.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than four bytes remain.
    fn read_u32(&mut self, label: &str) -> Result<u32, UpstreamNnueError> {
        let bytes: [u8; 4] = self
            .read_slice(4, label)?
            .try_into()
            .expect("four-byte slice");
        Ok(u32::from_le_bytes(bytes))
    }

    /// Used for reading one little-endian signed 32-bit field.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    ///
    /// # Returns
    ///
    /// The decoded value.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than four bytes remain.
    fn read_i32(&mut self, label: &str) -> Result<i32, UpstreamNnueError> {
        let bytes: [u8; 4] = self
            .read_slice(4, label)?
            .try_into()
            .expect("four-byte slice");
        Ok(i32::from_le_bytes(bytes))
    }

    /// Used for reading a nonnegative signed length bounded by a
    /// caller-supplied maximum.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the field being read
    /// * `maximum` - largest accepted length
    ///
    /// # Returns
    ///
    /// The validated length.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error for truncation or a negative value,
    /// and a resource-limit error when the length exceeds `maximum`.
    fn read_length(&mut self, label: &str, maximum: usize) -> Result<usize, UpstreamNnueError> {
        let signed = self.read_i32(label)?;
        let value = usize::try_from(signed)
            .map_err(|_| invalid(format!("negative Stockfish NNUE {label}: {signed}")))?;
        if value > maximum {
            return Err(UpstreamNnueError::new(
                UpstreamNnueErrorKind::ResourceLimit,
                format!("Stockfish NNUE {label} is {value}; limit is {maximum}"),
            ));
        }
        Ok(value)
    }

    /// Used for copying a fixed number of raw serialized weight bytes.
    ///
    /// # Arguments
    ///
    /// * `count` - number of bytes to copy
    /// * `label` - diagnostic name of the tensor being read
    ///
    /// # Returns
    ///
    /// The copied bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for truncation or allocation failure.
    fn read_byte_vector(
        &mut self,
        count: usize,
        label: &str,
    ) -> Result<Vec<u8>, UpstreamNnueError> {
        let source = self.read_slice(count, label)?;
        let mut output = Vec::new();
        reserve_exact(&mut output, count, label)?;
        output.extend_from_slice(source);
        Ok(output)
    }

    /// Used for decoding an exact-size signed-LEB128 tensor whose values
    /// must fit `i16`.
    ///
    /// # Arguments
    ///
    /// * `count` - number of elements to decode
    /// * `label` - diagnostic name of the tensor being read
    ///
    /// # Returns
    ///
    /// The decoded values.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing marker, truncation, values outside
    /// `i16`, unused block bytes, or allocation failure.
    fn read_leb_i16_array(
        &mut self,
        count: usize,
        label: &str,
    ) -> Result<Vec<i16>, UpstreamNnueError> {
        let block = self.begin_leb(label)?;
        let mut reader = LebReader::new(block);
        let mut output = Vec::new();
        reserve_exact(&mut output, count, label)?;
        for index in 0..count {
            let value = reader.read_signed(label, index)?;
            output.push(i16::try_from(value).map_err(|_| {
                invalid(format!(
                    "Stockfish NNUE {label}[{index}]={value} does not fit i16"
                ))
            })?);
        }
        reader.finish(label)?;
        Ok(output)
    }

    /// Used for decoding an exact-size signed-LEB128 tensor of `i32`
    /// values.
    ///
    /// # Arguments
    ///
    /// * `count` - number of elements to decode
    /// * `label` - diagnostic name of the tensor being read
    ///
    /// # Returns
    ///
    /// The decoded values.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing marker, truncation, out-of-range
    /// values, unused block bytes, or allocation failure.
    fn read_leb_i32_array(
        &mut self,
        count: usize,
        label: &str,
    ) -> Result<Vec<i32>, UpstreamNnueError> {
        let block = self.begin_leb(label)?;
        let mut reader = LebReader::new(block);
        let mut output = Vec::new();
        reserve_exact(&mut output, count, label)?;
        for index in 0..count {
            output.push(reader.read_signed(label, index)?);
        }
        reader.finish(label)?;
        Ok(output)
    }

    /// Used for validating a compressed-block marker and returning its
    /// bounded payload.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic name of the tensor being read
    ///
    /// # Returns
    ///
    /// The block's payload bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing [`LEB128_MAGIC`] marker, an
    /// out-of-bounds block length, or truncation.
    fn begin_leb(&mut self, label: &str) -> Result<&'a [u8], UpstreamNnueError> {
        if self.read_slice(LEB128_MAGIC.len(), "LEB128 marker")? != LEB128_MAGIC {
            return Err(invalid(format!(
                "missing Stockfish LEB128 marker for {label}"
            )));
        }
        let byte_count = self.read_length("LEB128 block length", MAX_UPSTREAM_MODEL_BYTES)?;
        self.read_slice(byte_count, label)
    }

    /// Used for testing whether every model byte was consumed exactly.
    ///
    /// # Returns
    ///
    /// `true` when the cursor stands at the end of the stream.
    const fn is_at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }

    /// Used for counting the unread model bytes.
    ///
    /// # Returns
    ///
    /// The number of bytes after the cursor.
    const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

/// Decoder for one length-delimited signed LEB128 tensor payload.
///
/// Operates on a block already bounded by [`Cursor::begin_leb`]; the block
/// must be consumed exactly by [`LebReader::finish`].
struct LebReader<'a> {
    /// Used for the bytes inside one already-bounded compressed block.
    bytes: &'a [u8],
    /// Used for the offset of the next encoded integer.
    offset: usize,
}

impl<'a> LebReader<'a> {
    /// Used for creating a decoder at the first compressed integer.
    ///
    /// # Arguments
    ///
    /// * `bytes` - bounded compressed-block payload
    ///
    /// # Returns
    ///
    /// A decoder positioned at offset zero.
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    /// Used for decoding one signed 32-bit value using at most five bytes.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic tensor name
    /// * `index` - element index reported in diagnostics
    ///
    /// # Returns
    ///
    /// The sign-extended decoded value.
    ///
    /// # Errors
    ///
    /// Returns an error for truncation, encodings longer than five bytes,
    /// or values outside the signed 32-bit range.
    fn read_signed(&mut self, label: &str, index: usize) -> Result<i32, UpstreamNnueError> {
        let mut result = 0_i64;
        let mut shift = 0_u32;
        for byte_number in 0..5 {
            let byte = *self.bytes.get(self.offset).ok_or_else(|| {
                invalid(format!(
                    "truncated Stockfish LEB128 {label} at element {index}"
                ))
            })?;
            self.offset += 1;
            result |= i64::from(byte & 0x7f) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                if byte & 0x40 != 0 {
                    result |= (!0_i64) << shift;
                }
                return i32::try_from(result).map_err(|_| {
                    invalid(format!(
                        "Stockfish LEB128 {label}[{index}] exceeds signed 32-bit range"
                    ))
                });
            }

            if byte_number == 4 {
                return Err(invalid(format!(
                    "Stockfish LEB128 {label}[{index}] uses more than five bytes"
                )));
            }
        }
        unreachable!("five-byte loop always returns")
    }

    /// Used for requiring the compressed block to contain exactly the
    /// declared elements.
    ///
    /// # Arguments
    ///
    /// * `label` - diagnostic tensor name
    ///
    /// # Errors
    ///
    /// Returns an error when undecoded bytes remain in the block.
    fn finish(&self, label: &str) -> Result<(), UpstreamNnueError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(invalid(format!(
                "{} unused bytes in Stockfish LEB128 {label} block",
                self.bytes.len() - self.offset
            )))
        }
    }
}

/// Used for constructing a consistent malformed-stream error.
///
/// # Arguments
///
/// * `message` - human-readable diagnostic detail
///
/// # Returns
///
/// An [`UpstreamNnueErrorKind::InvalidFormat`] error.
fn invalid(message: impl Into<String>) -> UpstreamNnueError {
    UpstreamNnueError::new(UpstreamNnueErrorKind::InvalidFormat, message)
}

/// Used for multiplying tensor dimensions with platform-overflow
/// checking.
///
/// # Arguments
///
/// * `left` - first dimension
/// * `right` - second dimension
/// * `label` - diagnostic name of the computed count
///
/// # Returns
///
/// The exact product.
///
/// # Errors
///
/// Returns a resource-limit error when the product overflows `usize`.
fn checked_product(left: usize, right: usize, label: &str) -> Result<usize, UpstreamNnueError> {
    left.checked_mul(right).ok_or_else(|| {
        UpstreamNnueError::new(
            UpstreamNnueErrorKind::ResourceLimit,
            format!("Stockfish NNUE {label} overflows this platform"),
        )
    })
}

/// Used for rounding an affine input width upward with checked
/// arithmetic.
///
/// # Arguments
///
/// * `value` - width to round up
/// * `multiple` - required nonzero divisor of the result
///
/// # Returns
///
/// The smallest multiple of `multiple` not below `value`.
///
/// # Errors
///
/// Returns a resource-limit error when the rounding addition overflows.
fn ceil_to_multiple(value: usize, multiple: usize) -> Result<usize, UpstreamNnueError> {
    value
        .checked_add(multiple - 1)
        .map(|sum| (sum / multiple) * multiple)
        .ok_or_else(|| {
            UpstreamNnueError::new(
                UpstreamNnueErrorKind::ResourceLimit,
                "Stockfish NNUE padded affine width overflow",
            )
        })
}

/// Used for reserving a model-sized vector allocation, mapping failure to
/// `ResourceLimit`.
///
/// # Arguments
///
/// * `values` - vector to grow
/// * `additional` - exact number of extra elements to reserve
/// * `label` - diagnostic name of the reserved tensor
///
/// # Errors
///
/// Returns a resource-limit error when the allocation cannot be reserved.
fn reserve_exact<T>(
    values: &mut Vec<T>,
    additional: usize,
    label: &str,
) -> Result<(), UpstreamNnueError> {
    values.try_reserve_exact(additional).map_err(|_| {
        UpstreamNnueError::new(
            UpstreamNnueErrorKind::ResourceLimit,
            format!("could not reserve Stockfish NNUE {label} ({additional} elements)"),
        )
    })
}

/// Used for copying a tensor slice through the allocation-failure-aware
/// reserve path.
///
/// # Arguments
///
/// * `source` - slice to copy
/// * `label` - diagnostic name of the copied tensor
///
/// # Returns
///
/// An owned copy of `source`.
///
/// # Errors
///
/// Returns a resource-limit error when the allocation cannot be reserved.
fn copy_slice<T: Copy>(source: &[T], label: &str) -> Result<Vec<T>, UpstreamNnueError> {
    let mut output = Vec::new();
    reserve_exact(&mut output, source.len(), label)?;
    output.extend_from_slice(source);
    Ok(output)
}

/// Hidden training-tooling surface exposing the engine's own sparse feature
/// enumeration (NNUE-20260724-174 milestone M-A) and decoded-tensor dump
/// (milestone M-B).
///
/// The out-of-workspace `tools/janus-datagen` crate links this module so its
/// LC0 chunk extractor uses the exact `HalfKAv2_hm` and `FullThreats` index
/// functions the engine evaluates with, and so its quantized exporter
/// re-serializes the engine's own decoded tensors and header hashes instead
/// of re-deriving either. The module is `#[doc(hidden)]` and offers no
/// stability guarantees; nothing in the engine itself depends on it.
#[doc(hidden)]
pub mod datagen {
    use super::{
        active_half_ka, active_threats, bucket, reserve_exact, stockfish_board, Color,
        FeatureTransformer, Position, SparseTransformedFeatures, ThreatTables, UpstreamNnue,
        UpstreamNnueError, BLACK, FC0_OUTPUT_DIMENSIONS, HALF_KA_DIMENSIONS,
        HALF_TRANSFORMED_DIMENSIONS, LAYER_STACKS, THREAT_DIMENSIONS, TRANSFORMED_DIMENSIONS,
        WHITE,
    };

    /// Sparse active feature sets of one position for both perspectives.
    ///
    /// Index `0` holds the White-anchored perspective and index `1` the
    /// Black-anchored perspective, matching the engine's Stockfish color
    /// selectors.
    pub struct ActiveFeatureSets {
        /// Used for the active `HalfKAv2_hm` feature indices of each
        /// perspective, in ascending Stockfish-square scan order.
        pub half_ka: [Vec<usize>; 2],
        /// Used for the active `FullThreats` feature indices of each
        /// perspective, in the format's serialization order.
        pub threats: [Vec<usize>; 2],
        /// Used for the material bucket that selects the layer stack.
        pub material_bucket: usize,
    }

    /// Deterministic generator used to fill the synthetic parity transformer.
    ///
    /// Implements `SplitMix64`, whose fixed seed makes every probe identical
    /// across runs and hosts.
    struct SplitMix64(u64);

    impl SplitMix64 {
        /// Used for drawing the next pseudo-random 64-bit value.
        ///
        /// # Returns
        ///
        /// The next value of the deterministic sequence.
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = self.0;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            mixed ^ (mixed >> 31)
        }
    }

    /// Probe pairing the engine's feature enumeration with a synthetic
    /// transformer for full-rebuild parity checks.
    ///
    /// Construction fills roughly 110 MiB of deterministic pseudo-random
    /// weights, so callers should build one probe and reuse it.
    pub struct FeatureProbe {
        /// Used for the deterministically rebuilt `FullThreats` index
        /// tables.
        tables: ThreatTables,
        /// Used for the synthetic transformer driving the production
        /// full-rebuild transform during parity checks.
        transformer: FeatureTransformer,
    }

    impl FeatureProbe {
        /// Used for building the probe's threat tables and synthetic
        /// parity transformer.
        ///
        /// Weight magnitudes are chosen so no accumulation can overflow:
        /// `HalfKA` and `FullThreats` transform weights stay within `-4..=3`
        /// and PSQT weights within one signed 21-bit lane.
        ///
        /// # Returns
        ///
        /// A reusable probe with deterministic contents.
        ///
        /// # Errors
        ///
        /// Returns an error when a weight-table allocation cannot be
        /// reserved.
        ///
        /// # Panics
        ///
        /// Never panics in practice; every drawn value is masked into the
        /// range its narrowing conversion accepts.
        pub fn new() -> Result<Self, UpstreamNnueError> {
            let tables = ThreatTables::current()?;
            let mut generator = SplitMix64(0x6a61_6e75_735f_6d61);

            let mut biases = Vec::new();
            reserve_exact(&mut biases, TRANSFORMED_DIMENSIONS, "parity biases")?;
            for _ in 0..TRANSFORMED_DIMENSIONS {
                biases.push(i16::try_from(generator.next_u64() & 0xff).expect("masked to a byte"));
            }

            let threat_weight_count = THREAT_DIMENSIONS * TRANSFORMED_DIMENSIONS;
            let mut threat_weights = Vec::new();
            reserve_exact(&mut threat_weights, threat_weight_count, "parity threats")?;
            for _ in 0..threat_weight_count {
                let magnitude = u8::try_from(generator.next_u64() & 0x07).expect("three bits");
                threat_weights.push(magnitude.wrapping_sub(4));
            }

            let psq_weight_count = HALF_KA_DIMENSIONS * TRANSFORMED_DIMENSIONS;
            let mut psq_weights = Vec::new();
            reserve_exact(&mut psq_weights, psq_weight_count, "parity HalfKA")?;
            for _ in 0..psq_weight_count {
                let magnitude = i16::try_from(generator.next_u64() & 0x07).expect("three bits");
                psq_weights.push(magnitude - 4);
            }

            let threat_psqt_count = THREAT_DIMENSIONS * LAYER_STACKS;
            let mut threat_psqt_weights = Vec::new();
            reserve_exact(
                &mut threat_psqt_weights,
                threat_psqt_count,
                "parity threat PSQT",
            )?;
            for _ in 0..threat_psqt_count {
                let lane = i32::try_from(generator.next_u64() & 0x1f_ffff).expect("21 bits");
                threat_psqt_weights.push(lane - (1 << 20));
            }

            let psqt_count = HALF_KA_DIMENSIONS * LAYER_STACKS;
            let mut psqt_weights = Vec::new();
            reserve_exact(&mut psqt_weights, psqt_count, "parity HalfKA PSQT")?;
            for _ in 0..psqt_count {
                let lane = i32::try_from(generator.next_u64() & 0x1f_ffff).expect("21 bits");
                psqt_weights.push(lane - (1 << 20));
            }

            Ok(Self {
                tables,
                transformer: FeatureTransformer {
                    biases,
                    threat_weights,
                    psq_weights,
                    threat_psqt_weights,
                    psqt_weights,
                },
            })
        }

        /// Used for enumerating the position's active sparse features with
        /// the engine's own index functions.
        ///
        /// # Arguments
        ///
        /// * `position` - validated position to enumerate
        ///
        /// # Returns
        ///
        /// Both perspectives' `HalfKA` and `FullThreats` index sets plus
        /// the material bucket.
        #[must_use]
        pub fn active_feature_sets(&self, position: &Position) -> ActiveFeatureSets {
            let board = stockfish_board(position);
            let piece_count = board.iter().filter(|&&piece| piece != 0).count();
            let mut half_ka = [Vec::new(), Vec::new()];
            let mut threats = [Vec::new(), Vec::new()];
            for perspective in 0..2 {
                half_ka[perspective]
                    .extend_from_slice(active_half_ka(&board, perspective).as_slice());
                threats[perspective].extend_from_slice(
                    active_threats(&board, perspective, &self.tables).as_slice(),
                );
            }
            ActiveFeatureSets {
                half_ka,
                threats,
                material_bucket: bucket(piece_count),
            }
        }

        /// Used for checking one position's enumerated feature sets against
        /// the production full-rebuild transform.
        ///
        /// The engine's [`FeatureTransformer::transform`] re-derives the
        /// active indices internally and accumulates them through the SIMD
        /// kernels; this check independently sums the same synthetic weight
        /// rows from `sets` with plain scalar loops and requires the sparse
        /// activation and PSQT outputs to match exactly. A single index
        /// differing between the two paths perturbs 1,024 pseudo-random
        /// lanes, so any divergence is detected.
        ///
        /// # Arguments
        ///
        /// * `position` - validated position to check
        /// * `sets` - feature sets claimed for `position`
        ///
        /// # Returns
        ///
        /// `Ok(())` when both paths agree.
        ///
        /// # Errors
        ///
        /// Returns a description of the first observed mismatch.
        pub fn transform_parity(
            &self,
            position: &Position,
            sets: &ActiveFeatureSets,
        ) -> Result<(), String> {
            let board = stockfish_board(position);
            let piece_count = board.iter().filter(|&&piece| piece != 0).count();
            let material_bucket = bucket(piece_count);
            if material_bucket != sets.material_bucket {
                return Err(format!(
                    "material bucket mismatch: engine {material_bucket}, sets {}",
                    sets.material_bucket
                ));
            }

            let actual =
                self.transformer
                    .transform(position, &board, material_bucket, &self.tables);

            let mut accumulation = [[0_i32; TRANSFORMED_DIMENSIONS]; 2];
            let mut half_ka_psqt = [[0_i32; LAYER_STACKS]; 2];
            let mut threat_psqt = [[0_i32; LAYER_STACKS]; 2];
            for perspective in 0..2 {
                for (lane, &bias) in accumulation[perspective]
                    .iter_mut()
                    .zip(&self.transformer.biases)
                {
                    *lane = i32::from(bias);
                }
                for &feature in &sets.half_ka[perspective] {
                    let base = feature * TRANSFORMED_DIMENSIONS;
                    for (lane, &weight) in accumulation[perspective]
                        .iter_mut()
                        .zip(&self.transformer.psq_weights[base..base + TRANSFORMED_DIMENSIONS])
                    {
                        *lane += i32::from(weight);
                    }
                    let psqt_base = feature * LAYER_STACKS;
                    for (lane, &weight) in half_ka_psqt[perspective]
                        .iter_mut()
                        .zip(&self.transformer.psqt_weights[psqt_base..psqt_base + LAYER_STACKS])
                    {
                        *lane += weight;
                    }
                }
                for &feature in &sets.threats[perspective] {
                    let base = feature * TRANSFORMED_DIMENSIONS;
                    for (lane, &weight) in accumulation[perspective]
                        .iter_mut()
                        .zip(&self.transformer.threat_weights[base..base + TRANSFORMED_DIMENSIONS])
                    {
                        *lane += i32::from(weight as i8);
                    }
                    let psqt_base = feature * LAYER_STACKS;
                    for (lane, &weight) in threat_psqt[perspective].iter_mut().zip(
                        &self.transformer.threat_psqt_weights[psqt_base..psqt_base + LAYER_STACKS],
                    ) {
                        *lane += weight;
                    }
                }
            }

            let stm = match position.side_to_move() {
                Color::White => WHITE,
                Color::Black => BLACK,
            };
            let opponent = stm ^ 1;
            let expected_psqt = (half_ka_psqt[stm][material_bucket]
                - half_ka_psqt[opponent][material_bucket]
                + threat_psqt[stm][material_bucket]
                - threat_psqt[opponent][material_bucket])
                / 2;
            if actual.psqt != expected_psqt {
                return Err(format!(
                    "PSQT mismatch: transform {}, independent {expected_psqt}",
                    actual.psqt
                ));
            }

            let mut expected = SparseTransformedFeatures::new();
            for (offset, perspective) in [(0, stm), (HALF_TRANSFORMED_DIMENSIONS, opponent)] {
                for index in 0..HALF_TRANSFORMED_DIMENSIONS {
                    let left = accumulation[perspective][index].clamp(0, 255);
                    let right = accumulation[perspective][index + HALF_TRANSFORMED_DIMENSIONS]
                        .clamp(0, 255);
                    expected.push_nonzero(offset + index, (left * right) / 512);
                }
            }
            if actual.features.len != expected.len {
                return Err(format!(
                    "sparse lane count mismatch: transform {}, independent {}",
                    actual.features.len, expected.len
                ));
            }
            for ((actual_index, actual_value), (expected_index, expected_value)) in
                actual.features.iter().zip(expected.iter())
            {
                if actual_index != expected_index || actual_value != expected_value {
                    return Err(format!(
                        "sparse lane mismatch at index {actual_index}: transform value \
                         {actual_value}, independent lane {expected_index} value \
                         {expected_value}"
                    ));
                }
            }
            Ok(())
        }
    }

    /// Used for re-exporting the exact header hash triple the loader
    /// accepts (NNUE-20260724-174 milestone M-B).
    ///
    /// The out-of-workspace exporter writes these values instead of
    /// re-deriving Stockfish's hash chain, so header parity with
    /// [`UpstreamNnue::from_bytes`] is structural.
    ///
    /// # Returns
    ///
    /// The `(network, feature-transformer, layer-stack)` hashes in the
    /// order their fields appear in the serialized stream.
    #[must_use]
    pub const fn header_hashes() -> (u32, u32, u32) {
        (
            super::current_big_network_hash(),
            super::current_big_feature_hash(),
            super::current_big_arch_hash(),
        )
    }

    /// Complete decoded tensor set of one loaded CURRENT/BIG model.
    ///
    /// Produced by [`dump_tensors`] so the out-of-workspace exporter can
    /// re-serialize a loaded model byte-exactly (NNUE-20260724-174
    /// milestone M-B). Every tensor is returned in serialization order and
    /// in its serialized value domain; only the FC0 matrix is transposed
    /// back from the engine's sparse input-major layout to the serialized
    /// row-major layout.
    pub struct TensorDump {
        /// Used for the UTF-8 description block copied from the header.
        pub description: String,
        /// Used for the 1,024 signed-16 feature-transformer biases.
        pub ft_biases: Vec<i16>,
        /// Used for the row-major `[60720][1024]` signed-byte
        /// `FullThreats` transform weights.
        pub threat_weights: Vec<i8>,
        /// Used for the row-major `[22528][1024]` signed-16 `HalfKA`
        /// transform weights.
        pub half_ka_weights: Vec<i16>,
        /// Used for the row-major `[60720][8]` `FullThreats` PSQT terms,
        /// serialized before the `HalfKA` PSQT rows.
        pub threat_psqt: Vec<i32>,
        /// Used for the row-major `[22528][8]` `HalfKA` PSQT terms.
        pub half_ka_psqt: Vec<i32>,
        /// Used for the eight material-bucket dense stacks in
        /// serialization order.
        pub stacks: Vec<StackDump>,
    }

    /// Decoded dense tensors of one material-bucket layer stack.
    ///
    /// Weight matrices keep the serialized row-major layout, including
    /// FC1's stride-64 input padding bytes exactly as stored.
    pub struct StackDump {
        /// Used for the 32 FC0 output biases.
        pub fc0_biases: Vec<i32>,
        /// Used for the row-major `[32][1024]` FC0 weights, reconstructed
        /// from the engine's input-major transpose.
        pub fc0_weights: Vec<i8>,
        /// Used for the 32 FC1 output biases.
        pub fc1_biases: Vec<i32>,
        /// Used for the row-major `[32][64]` FC1 weights with the two
        /// serialized padding columns retained.
        pub fc1_weights: Vec<i8>,
        /// Used for the single FC2 output bias.
        pub fc2_biases: Vec<i32>,
        /// Used for the row-major `[1][32]` FC2 weights.
        pub fc2_weights: Vec<i8>,
    }

    /// Used for dumping every serialized tensor of a loaded model.
    ///
    /// Allocates fresh owned copies (about 180 MiB for CURRENT/BIG), so
    /// callers should dump once and reuse the result.
    ///
    /// # Arguments
    ///
    /// * `network` - fully validated loaded model
    ///
    /// # Returns
    ///
    /// The complete tensor set in serialization order.
    #[must_use]
    pub fn dump_tensors(network: &UpstreamNnue) -> TensorDump {
        let transformer = &network.transformer;
        let stacks = network
            .stacks
            .iter()
            .map(|stack| {
                let mut fc0_weights = vec![0_i8; FC0_OUTPUT_DIMENSIONS * TRANSFORMED_DIMENSIONS];
                for input in 0..TRANSFORMED_DIMENSIONS {
                    for output in 0..FC0_OUTPUT_DIMENSIONS {
                        fc0_weights[output * TRANSFORMED_DIMENSIONS + input] =
                            stack.fc0.input_major_weights[input * FC0_OUTPUT_DIMENSIONS + output]
                                as i8;
                    }
                }
                StackDump {
                    fc0_biases: stack.fc0.biases.clone(),
                    fc0_weights,
                    fc1_biases: stack.fc1.biases.clone(),
                    fc1_weights: stack.fc1.weights.iter().map(|&byte| byte as i8).collect(),
                    fc2_biases: stack.fc2.biases.clone(),
                    fc2_weights: stack.fc2.weights.iter().map(|&byte| byte as i8).collect(),
                }
            })
            .collect();
        TensorDump {
            description: network.description.clone(),
            ft_biases: transformer.biases.clone(),
            threat_weights: transformer
                .threat_weights
                .iter()
                .map(|&byte| byte as i8)
                .collect(),
            half_ka_weights: transformer.psq_weights.clone(),
            threat_psqt: transformer.threat_psqt_weights.clone(),
            half_ka_psqt: transformer.psqt_weights.clone(),
            stacks,
        }
    }
}
