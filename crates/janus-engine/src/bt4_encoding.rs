//! CRTK-compatible board and attention-policy encoding for BT4 networks.
//!
//! The routines in this module reproduce the Java BT4 encoder's observable
//! board layout: 112 channel-major input planes, side-to-move perspective,
//! optional canonical transforms, and the 1,858-entry attention policy gather.
//! Castling policy uses LCZero's king-to-rook convention, matching the corrected
//! Java path for orthodox and Chess960 positions. This is compatibility with
//! ChessRTK's documented experimental BT4 path, not a claim of bit-exact
//! compatibility with every upstream LCZero input-history mode.
//!
//! Janus stores squares from `a8 = 0`, while the neural representation uses
//! `a1 = 0`. The conversion is intentionally kept here rather than duplicated
//! by an evaluator, so board features and transformed move indices cannot drift
//! apart.

use crate::bt4::Bt4InputFormat;
use janus_core::{CastlingRights, Color, Move, Piece, PieceKind, Position, Square};
use std::fmt;

/// Used for sizing the piece and repetition planes reserved for one
/// historical board.
///
/// Each history slot holds twelve piece planes plus one repetition plane;
/// this encoder writes only the piece planes and leaves every repetition
/// plane zero.
pub const PLANES_PER_BOARD: usize = 13;
/// Used for counting the historical board slots consumed by the 112-plane
/// input format.
pub const HISTORY: usize = 8;
/// Used for locating the first auxiliary channel after the eight historical
/// boards.
///
/// Equal to [`PLANES_PER_BOARD`] times [`HISTORY`].
pub const AUX_BASE: usize = PLANES_PER_BOARD * HISTORY;
/// Used for counting the total number of input feature channels.
pub const INPUT_CHANNELS: usize = 112;
/// Used for counting the square tokens in a chess board.
pub const TOKENS: usize = 64;
/// Used for sizing one channel-major encoded position in scalar values.
///
/// Equal to [`INPUT_CHANNELS`] times [`TOKENS`].
pub const INPUT_VALUES: usize = INPUT_CHANNELS * TOKENS;
/// Used for marking the horizontal file-reflection bit in the canonical
/// transform mask.
pub const FLIP_TRANSFORM: u8 = 1;
/// Used for marking the vertical rank-reflection bit in the canonical
/// transform mask.
pub const MIRROR_TRANSFORM: u8 = 2;
/// Used for marking the a8-h1 anti-diagonal transpose bit in the canonical
/// transform mask.
pub const TRANSPOSE_TRANSFORM: u8 = 4;
/// Used for counting the direct from-square/to-square attention logits.
///
/// Equal to [`TOKENS`] squared.
pub const FROM_TO_POLICY_SIZE: usize = TOKENS * TOKENS;
/// Used for counting the internal policy planes, including three non-knight
/// promotion groups.
pub const INTERNAL_POLICY_PLANES: usize = 67;
/// Used for counting the logits emitted by the uncompressed attention policy
/// head.
///
/// Equal to [`INTERNAL_POLICY_PLANES`] times [`TOKENS`].
pub const INTERNAL_POLICY_SIZE: usize = INTERNAL_POLICY_PLANES * TOKENS;
/// Used for counting the geometrically valid logits in the compressed
/// attention policy.
pub const POLICY_SIZE: usize = 1_858;
/// Used for sizing one token after appending one square-identity feature to
/// every token.
///
/// Equal to [`INPUT_CHANNELS`] plus [`TOKENS`].
pub const POSITION_MAP_WIDTH: usize = INPUT_CHANNELS + TOKENS;
/// Used for sizing a token-major input with an appended position map.
///
/// Equal to [`TOKENS`] times [`POSITION_MAP_WIDTH`].
pub const POSITION_MAP_VALUES: usize = TOKENS * POSITION_MAP_WIDTH;

/// A complete BT4 board input and the canonical transform applied to it.
///
/// The transform is kept with the planes because the same spatial permutation
/// must be applied to policy move indices; separating the two would let board
/// features and move mapping drift apart.
#[derive(Clone, Debug, PartialEq)]
pub struct Bt4EncodedInput {
    /// Used for storing the channel-major `[112][64]` feature values.
    planes: Vec<f32>,
    /// Used for storing the spatial transform that must also be applied to
    /// policy moves.
    transform: u8,
}

impl Bt4EncodedInput {
    /// Used for borrowing the channel-major `[112][64]` input values.
    ///
    /// # Returns
    ///
    /// Slice of exactly [`INPUT_VALUES`] feature values.
    #[must_use]
    pub fn planes(&self) -> &[f32] {
        &self.planes
    }

    /// Used for retrieving the canonical transform used by both board and
    /// policy encoding.
    ///
    /// # Returns
    ///
    /// Bitmask combining [`FLIP_TRANSFORM`], [`MIRROR_TRANSFORM`], and
    /// [`TRANSPOSE_TRANSFORM`].
    #[must_use]
    pub const fn transform(&self) -> u8 {
        self.transform
    }
}

/// Invalid input supplied to a BT4 encoding or policy-gather operation.
///
/// Every validation failure is reported through this enum so callers can
/// distinguish an empty history from a wrongly sized tensor or workspace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Bt4EncodingError {
    /// Used for reporting that the position history did not contain a current
    /// position.
    EmptyHistory,
    /// Used for reporting that a caller-provided tensor or workspace had the
    /// wrong exact length.
    LengthMismatch {
        /// Used for naming the tensor or workspace being validated.
        role: &'static str,
        /// Used for recording the required number of scalar values.
        expected: usize,
        /// Used for recording the number of values supplied by the caller.
        actual: usize,
    },
}

impl fmt::Display for Bt4EncodingError {
    /// Used for writing a stable, context-rich validation diagnostic.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Returns
    ///
    /// Propagated formatter result.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHistory => formatter.write_str("BT4 position history is empty"),
            Self::LengthMismatch {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "BT4 {role} length is {actual}; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for Bt4EncodingError {}

/// Used for encoding one position in the selected 112-plane format.
///
/// Since no earlier positions are supplied, the current placement is repeated
/// in all eight history slots, matching the Java single-position API.
///
/// # Arguments
///
/// * `position` - current position to encode
/// * `format` - input-plane interpretation to produce
///
/// # Returns
///
/// Encoded planes together with the canonical transform applied to them.
#[must_use]
pub fn encode_position(position: &Position, format: Bt4InputFormat) -> Bt4EncodedInput {
    let mut planes = vec![0.0; INPUT_VALUES];
    let transform =
        encode_nonempty_history_into(std::slice::from_ref(position), format, &mut planes);
    Bt4EncodedInput { planes, transform }
}

/// Used for encoding one FEN-derived position with LC0's known en-passant
/// predecessor.
///
/// A FEN containing an en-passant target identifies the preceding double pawn
/// push unambiguously. For non-canonical formats, slot zero stores the current
/// board and all older missing slots restore that pawn to its source square.
/// Canonical input stops history at an en-passant boundary, so its older slots
/// remain zero. Positions without an en-passant target retain the normal
/// repeated-position fallback from [`encode_position`].
///
/// This API is separate so explicit Java-compatible histories and existing
/// single-position callers do not silently change representation.
///
/// # Arguments
///
/// * `position` - current FEN-derived position to encode
/// * `format` - input-plane interpretation to produce
///
/// # Returns
///
/// Encoded planes together with the canonical transform applied to them.
#[must_use]
pub fn encode_lc0_fen_position(position: &Position, format: Bt4InputFormat) -> Bt4EncodedInput {
    let mut planes = vec![0.0; INPUT_VALUES];
    let transform =
        encode_nonempty_history_into(std::slice::from_ref(position), format, &mut planes);
    reconstruct_fen_en_passant_history(position, format, transform, &mut planes);
    Bt4EncodedInput { planes, transform }
}

/// Used for encoding positions ordered from oldest to current.
///
/// When fewer than eight positions are supplied, the oldest available
/// placement is reused for the missing history slots. Every slot is oriented
/// from the current position's side-to-move perspective.
///
/// # Arguments
///
/// * `history_oldest_to_newest` - positions with the current one last
/// * `format` - input-plane interpretation to produce
///
/// # Returns
///
/// Encoded planes together with the canonical transform applied to them.
///
/// # Errors
///
/// Returns [`Bt4EncodingError::EmptyHistory`] when no current position exists.
pub fn encode_history(
    history_oldest_to_newest: &[Position],
    format: Bt4InputFormat,
) -> Result<Bt4EncodedInput, Bt4EncodingError> {
    if history_oldest_to_newest.is_empty() {
        return Err(Bt4EncodingError::EmptyHistory);
    }
    let mut planes = vec![0.0; INPUT_VALUES];
    let transform = encode_nonempty_history_into(history_oldest_to_newest, format, &mut planes);
    Ok(Bt4EncodedInput { planes, transform })
}

/// Used for writing one repeated-position encoding into an existing
/// workspace.
///
/// The exact-size destination permits allocation-free evaluator calls. The
/// returned transform must be retained while mapping that prediction's moves.
///
/// # Arguments
///
/// * `position` - current position to encode
/// * `format` - input-plane interpretation to produce
/// * `planes` - destination workspace of exactly [`INPUT_VALUES`] values
///
/// # Returns
///
/// Canonical transform mask applied to the written planes.
///
/// # Errors
///
/// Returns [`Bt4EncodingError::LengthMismatch`] unless `planes` contains
/// exactly [`INPUT_VALUES`] elements.
pub fn encode_position_into(
    position: &Position,
    format: Bt4InputFormat,
    planes: &mut [f32],
) -> Result<u8, Bt4EncodingError> {
    validate_length("input workspace", planes.len(), INPUT_VALUES)?;
    Ok(encode_nonempty_history_into(
        std::slice::from_ref(position),
        format,
        planes,
    ))
}

/// Used for writing FEN-derived input with LC0's en-passant predecessor
/// reconstruction.
///
/// This is the allocation-free counterpart to [`encode_lc0_fen_position`].
/// The returned transform must be used for the prediction's policy moves.
///
/// # Arguments
///
/// * `position` - current FEN-derived position to encode
/// * `format` - input-plane interpretation to produce
/// * `planes` - destination workspace of exactly [`INPUT_VALUES`] values
///
/// # Returns
///
/// Canonical transform mask applied to the written planes.
///
/// # Errors
///
/// Returns [`Bt4EncodingError::LengthMismatch`] unless `planes` contains
/// exactly [`INPUT_VALUES`] elements.
pub fn encode_lc0_fen_position_into(
    position: &Position,
    format: Bt4InputFormat,
    planes: &mut [f32],
) -> Result<u8, Bt4EncodingError> {
    validate_length("input workspace", planes.len(), INPUT_VALUES)?;
    let transform = encode_nonempty_history_into(std::slice::from_ref(position), format, planes);
    reconstruct_fen_en_passant_history(position, format, transform, planes);
    Ok(transform)
}

/// Used for writing an oldest-to-current position history into an existing
/// workspace.
///
/// This is the allocation-free counterpart to [`encode_history`], sharing its
/// padding and perspective rules.
///
/// # Arguments
///
/// * `history_oldest_to_newest` - positions with the current one last
/// * `format` - input-plane interpretation to produce
/// * `planes` - destination workspace of exactly [`INPUT_VALUES`] values
///
/// # Returns
///
/// Canonical transform mask applied to the written planes.
///
/// # Errors
///
/// Returns an error for an empty history or unless `planes` contains exactly
/// [`INPUT_VALUES`] elements.
pub fn encode_history_into(
    history_oldest_to_newest: &[Position],
    format: Bt4InputFormat,
    planes: &mut [f32],
) -> Result<u8, Bt4EncodingError> {
    if history_oldest_to_newest.is_empty() {
        return Err(Bt4EncodingError::EmptyHistory);
    }
    validate_length("input workspace", planes.len(), INPUT_VALUES)?;
    Ok(encode_nonempty_history_into(
        history_oldest_to_newest,
        format,
        planes,
    ))
}

/// Used for converting channel-major `[112][64]` input into token-major
/// `[64][112]` data.
///
/// # Arguments
///
/// * `planes` - channel-major input of exactly [`INPUT_VALUES`] values
///
/// # Returns
///
/// Newly allocated token-major copy of the input.
///
/// # Errors
///
/// Returns an error unless `planes` contains exactly [`INPUT_VALUES`] values.
pub fn to_token_major(planes: &[f32]) -> Result<Vec<f32>, Bt4EncodingError> {
    validate_length("channel-major input", planes.len(), INPUT_VALUES)?;
    let mut output = vec![0.0; INPUT_VALUES];
    for token in 0..TOKENS {
        let output_base = token * INPUT_CHANNELS;
        for channel in 0..INPUT_CHANNELS {
            output[output_base + channel] = planes[channel * TOKENS + token];
        }
    }
    Ok(output)
}

/// Used for appending a 64-way square identity map to token-major `[64][112]`
/// input.
///
/// This implements the `PositionMap` embedding used by simplified v1 models;
/// dense-position v2 models consume the original planes through their learned
/// preprocessing layer instead.
///
/// # Arguments
///
/// * `token_major` - token-major input of exactly [`INPUT_VALUES`] values
///
/// # Returns
///
/// Newly allocated `[64][176]` tensor of [`POSITION_MAP_VALUES`] values.
///
/// # Errors
///
/// Returns an error unless `token_major` contains exactly [`INPUT_VALUES`]
/// values.
pub fn append_position_map(token_major: &[f32]) -> Result<Vec<f32>, Bt4EncodingError> {
    validate_length("token-major input", token_major.len(), INPUT_VALUES)?;
    let mut output = vec![0.0; POSITION_MAP_VALUES];
    for token in 0..TOKENS {
        let source = token * INPUT_CHANNELS;
        let destination = token * POSITION_MAP_WIDTH;
        output[destination..destination + INPUT_CHANNELS]
            .copy_from_slice(&token_major[source..source + INPUT_CHANNELS]);
        output[destination + INPUT_CHANNELS + token] = 1.0;
    }
    Ok(output)
}

/// Used for computing the uncompressed attention-policy index for `mv`.
///
/// Ordinary moves and knight promotions use the `64 * 64` from/to matrix.
/// Queen, rook, and bishop promotions use the final three groups, in that
/// order. This is the upstream `LCZero` attention layout and the corrected
/// `ChessRTK` Java promotion layout.
/// `transform` must be the mask returned by the corresponding board encoding;
/// unknown high bits are ignored like the Java reference.
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - move to map into the attention layout
/// * `transform` - canonical transform mask from the board encoding
///
/// # Returns
///
/// Index into the internal `[67][64]` policy tensor, or `None` when the move
/// geometry has no attention representation.
#[must_use]
pub fn internal_policy_index(position: &Position, mv: Move, transform: u8) -> Option<usize> {
    let mut from = perspective_square(position, mv.from());
    let mut to = perspective_square(position, policy_destination(position, mv));
    if transform != 0 {
        from = transform_square(from, transform);
        to = transform_square(to, transform);
    }

    if let Some(promotion) = mv.promotion() {
        if promotion != PieceKind::Knight {
            let promotion_index = match promotion {
                PieceKind::Queen => 0,
                PieceKind::Rook => 1,
                PieceKind::Bishop => 2,
                PieceKind::Pawn | PieceKind::Knight | PieceKind::King => return None,
            };
            let from_file = from & 7;
            let from_rank = from >> 3;
            let to_file = to & 7;
            let to_rank = to >> 3;
            if from_rank != 6 || to_rank != 7 || absolute_difference(from_file, to_file) > 1 {
                return None;
            }
            return Some(FROM_TO_POLICY_SIZE + from_file * 24 + to_file * 3 + promotion_index);
        }
    }

    if from == to || !is_queen_like_or_knight(from, to) {
        return None;
    }
    Some(from * TOKENS + to)
}

/// Used for computing the compressed 1,858-entry policy index for `mv`.
///
/// `None` means the move geometry is absent from the attention policy. The
/// function does not perform legality checking; normal callers pass moves from
/// [`Position::legal_moves`].
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - move to map into the compressed layout
/// * `transform` - canonical transform mask from the board encoding
///
/// # Returns
///
/// Index in `0..POLICY_SIZE`, or `None` for unrepresented geometry.
#[must_use]
pub fn compressed_policy_index(position: &Position, mv: Move, transform: u8) -> Option<usize> {
    let internal = internal_policy_index(position, mv, transform)?;
    let compressed = COMPRESSED_BY_INTERNAL[internal];
    usize::try_from(compressed).ok()
}

/// Used for gathering the internal 4,288 logits into LC0's 1,858-entry
/// geometric order.
///
/// Invalid geometric entries are discarded. Values, including non-finite
/// values, are copied without normalization so the evaluator retains control
/// over numeric sanitization.
///
/// # Arguments
///
/// * `internal_logits` - uncompressed policy tensor of exactly
///   [`INTERNAL_POLICY_SIZE`] values
///
/// # Returns
///
/// Newly allocated compressed vector of [`POLICY_SIZE`] values.
///
/// # Errors
///
/// Returns an error unless `internal_logits` has [`INTERNAL_POLICY_SIZE`]
/// values.
pub fn gather_policy(internal_logits: &[f32]) -> Result<Vec<f32>, Bt4EncodingError> {
    validate_length(
        "internal policy tensor",
        internal_logits.len(),
        INTERNAL_POLICY_SIZE,
    )?;
    let mut output = vec![0.0; POLICY_SIZE];
    for (internal, &compressed) in COMPRESSED_BY_INTERNAL.iter().enumerate() {
        if let Ok(compressed) = usize::try_from(compressed) {
            output[compressed] = internal_logits[internal];
        }
    }
    Ok(output)
}

/// Used for extracting compressed logits for a caller-supplied legal move
/// list.
///
/// Output preserves the move order and omits only moves without an attention
/// representation. Values are not normalized or filtered for finiteness.
///
/// # Arguments
///
/// * `position` - position the moves belong to
/// * `legal_moves` - moves whose logits should be extracted
/// * `compressed_logits` - compressed policy tensor of exactly
///   [`POLICY_SIZE`] values
/// * `transform` - canonical transform mask from the board encoding
///
/// # Returns
///
/// Move/logit pairs in the order of `legal_moves`.
///
/// # Errors
///
/// Returns an error unless `compressed_logits` has [`POLICY_SIZE`] values.
pub fn gather_legal_policy_logits(
    position: &Position,
    legal_moves: &[Move],
    compressed_logits: &[f32],
    transform: u8,
) -> Result<Vec<(Move, f32)>, Bt4EncodingError> {
    validate_length(
        "compressed policy tensor",
        compressed_logits.len(),
        POLICY_SIZE,
    )?;
    let mut output = Vec::with_capacity(legal_moves.len());
    for &mv in legal_moves {
        if let Some(index) = compressed_policy_index(position, mv, transform) {
            output.push((mv, compressed_logits[index]));
        }
    }
    Ok(output)
}

/// Used for extracting legal move logits directly from the internal attention
/// tensor.
///
/// This allocation-light path is equivalent to [`gather_policy`] followed by
/// [`gather_legal_policy_logits`], but avoids constructing the complete
/// compressed vector during MCTS evaluation.
///
/// # Arguments
///
/// * `position` - position the moves belong to
/// * `legal_moves` - moves whose logits should be extracted
/// * `internal_logits` - uncompressed policy tensor of exactly
///   [`INTERNAL_POLICY_SIZE`] values
/// * `transform` - canonical transform mask from the board encoding
///
/// # Returns
///
/// Move/logit pairs in the order of `legal_moves`.
///
/// # Errors
///
/// Returns an error unless `internal_logits` has [`INTERNAL_POLICY_SIZE`]
/// values.
pub fn gather_legal_internal_logits(
    position: &Position,
    legal_moves: &[Move],
    internal_logits: &[f32],
    transform: u8,
) -> Result<Vec<(Move, f32)>, Bt4EncodingError> {
    validate_length(
        "internal policy tensor",
        internal_logits.len(),
        INTERNAL_POLICY_SIZE,
    )?;
    let mut output = Vec::with_capacity(legal_moves.len());
    for &mv in legal_moves {
        if let Some(index) = internal_policy_index(position, mv, transform) {
            output.push((mv, internal_logits[index]));
        }
    }
    Ok(output)
}

/// Used for borrowing the immutable internal-to-compressed policy gather
/// table.
///
/// Negative entries are geometrically invalid; every non-negative entry is a
/// unique index in `0..POLICY_SIZE`.
///
/// # Returns
///
/// Reference to the compile-time `COMPRESSED_BY_INTERNAL` map.
#[must_use]
pub const fn compressed_by_internal_map() -> &'static [i16; INTERNAL_POLICY_SIZE] {
    &COMPRESSED_BY_INTERNAL
}

/// Used for applying the canonical transform mask to one a1-origin plane
/// bitboard.
///
/// The flip, mirror, and transpose bits are applied in that fixed order;
/// unknown high bits in the mask are ignored.
///
/// # Arguments
///
/// * `bits` - a1-origin bitboard to permute
/// * `transform` - canonical transform mask
///
/// # Returns
///
/// Permuted bitboard.
#[must_use]
pub const fn transform_plane_bits(bits: u64, transform: u8) -> u64 {
    let mut output = bits;
    if transform & FLIP_TRANSFORM != 0 {
        output = flip_files(output);
    }
    if transform & MIRROR_TRANSFORM != 0 {
        output = mirror_ranks(output);
    }
    if transform & TRANSPOSE_TRANSFORM != 0 {
        output = transpose(output);
    }
    output
}

/// Used for encoding a known non-empty history after public validation.
///
/// The destination is zeroed, the eight history slots are filled newest-first
/// starting from the current position (reusing the oldest placement for
/// missing slots), and the auxiliary planes are written for the current
/// position.
/// Canonical input additionally selects the deterministic spatial transform.
///
/// # Arguments
///
/// * `history_oldest_to_newest` - non-empty positions with the current one
///   last
/// * `format` - input-plane interpretation to produce
/// * `planes` - destination workspace of exactly [`INPUT_VALUES`] values
///
/// # Returns
///
/// Canonical transform mask applied to the written planes.
///
/// # Panics
///
/// Panics when the history is empty or `planes` is too short to hold
/// [`INPUT_VALUES`] values; debug builds additionally assert both conditions
/// up front. Public callers validate them first.
fn encode_nonempty_history_into(
    history_oldest_to_newest: &[Position],
    format: Bt4InputFormat,
    planes: &mut [f32],
) -> u8 {
    debug_assert!(!history_oldest_to_newest.is_empty());
    debug_assert_eq!(planes.len(), INPUT_VALUES);
    planes.fill(0.0);
    let current = history_oldest_to_newest
        .last()
        .expect("non-empty BT4 history has a current position");
    let black_to_move = current.side_to_move() == Color::Black;
    let current_perspective = to_side_perspective(collect_plane_bits(current), black_to_move);
    let transform = if format == Bt4InputFormat::Canonical112 {
        choose_canonical_transform(current, &current_perspective)
    } else {
        0
    };

    for slot in 0..HISTORY {
        let history_index = history_oldest_to_newest.len().saturating_sub(slot + 1);
        let historical = &history_oldest_to_newest[history_index];
        let perspective = to_side_perspective(collect_plane_bits(historical), black_to_move);
        write_history_slot(planes, slot, &perspective, transform);
    }
    write_aux_planes(planes, current, format, black_to_move, transform);
    transform
}

/// Used for rewriting missing slots using the double pawn push encoded by a
/// FEN target.
///
/// Positions without an en-passant target are left untouched. Canonical input
/// zeroes every older history slot instead, matching LC0's history boundary
/// at an en-passant reset. Otherwise, each older slot's opponent pawn plane
/// is rewritten with the pushed pawn moved back to its source square.
///
/// # Arguments
///
/// * `position` - current FEN-derived position
/// * `format` - input-plane interpretation being produced
/// * `transform` - canonical transform mask already applied to `planes`
/// * `planes` - workspace of exactly [`INPUT_VALUES`] values already holding
///   the repeated-position encoding
///
/// # Panics
///
/// Panics in debug builds when the perspective en-passant square falls
/// outside the sixth-rank index window `40..48`.
fn reconstruct_fen_en_passant_history(
    position: &Position,
    format: Bt4InputFormat,
    transform: u8,
    planes: &mut [f32],
) {
    let Some(en_passant) = position.en_passant_square() else {
        return;
    };
    if format == Bt4InputFormat::Canonical112 {
        for slot in 1..HISTORY {
            let start = slot * PLANES_PER_BOARD * TOKENS;
            planes[start..start + PLANES_PER_BOARD * TOKENS].fill(0.0);
        }
        return;
    }

    let black_to_move = position.side_to_move() == Color::Black;
    let perspective = to_side_perspective(collect_plane_bits(position), black_to_move);
    let en_passant_index =
        usize::try_from(perspective_square_bit(en_passant, black_to_move).trailing_zeros())
            .expect("an en-passant square index fits usize");
    debug_assert!((40..48).contains(&en_passant_index));
    let destination = 1_u64 << (en_passant_index - 8);
    let source = 1_u64 << (en_passant_index + 8);
    let predecessor_pawns = (perspective[6 + PieceKind::Pawn.index()] & !destination) | source;
    let predecessor_pawns = transform_plane_bits(predecessor_pawns, transform);
    for slot in 1..HISTORY {
        let plane = slot * PLANES_PER_BOARD + 6 + PieceKind::Pawn.index();
        fill_plane(planes, plane, 0.0);
        write_bits(planes, plane, predecessor_pawns);
    }
}

/// Used for converting Janus piece bitboards to a1-origin neural plane
/// coordinates.
///
/// Janus stores `a8 = 0`, so each bitboard is byte-swapped into the `a1 = 0`
/// orientation expected by the network planes.
///
/// # Arguments
///
/// * `position` - position whose piece bitboards are collected
///
/// # Returns
///
/// Twelve a1-origin bitboards indexed by [`Piece::index`], White pieces
/// first.
fn collect_plane_bits(position: &Position) -> [u64; 12] {
    let mut bits = [0_u64; 12];
    for color in [Color::White, Color::Black] {
        for kind in PieceKind::ALL {
            let piece = Piece::new(color, kind);
            bits[piece.index()] = position.piece_bitboard(piece).swap_bytes();
        }
    }
    bits
}

/// Used for swapping colors and ranks when Black owns the current
/// perspective.
///
/// White-to-move input passes through unchanged; for Black, the two color
/// groups exchange places and every bitboard is rank-mirrored so the side to
/// move always plays "up" the board.
///
/// # Arguments
///
/// * `bits` - twelve a1-origin piece bitboards, White pieces first
/// * `black_to_move` - whether Black owns the current perspective
///
/// # Returns
///
/// Perspective-oriented bitboards with side-to-move pieces in the first six
/// slots.
fn to_side_perspective(bits: [u64; 12], black_to_move: bool) -> [u64; 12] {
    if !black_to_move {
        return bits;
    }
    let mut output = [0_u64; 12];
    for piece in 0..6 {
        output[piece] = bits[6 + piece].swap_bytes();
        output[6 + piece] = bits[piece].swap_bytes();
    }
    output
}

/// Used for writing the twelve piece planes of one historical slot.
///
/// # Arguments
///
/// * `planes` - destination channel-major workspace
/// * `slot` - history slot index, 0 being the current position
/// * `bits` - perspective-oriented piece bitboards for the slot
/// * `transform` - canonical transform mask applied to every plane
fn write_history_slot(planes: &mut [f32], slot: usize, bits: &[u64; 12], transform: u8) {
    let base = slot * PLANES_PER_BOARD;
    for (piece, &piece_bits) in bits.iter().enumerate() {
        write_bits(
            planes,
            base + piece,
            transform_plane_bits(piece_bits, transform),
        );
    }
}

/// Used for writing castling, en-passant, clock, and board-constant auxiliary
/// planes.
///
/// Classical input writes four constant castling-rights planes; the other
/// formats write two rook-identity planes instead. Canonical input encodes
/// the en-passant square as a plane bit and scales the halfmove clock by 100,
/// while the other formats use the side-to-move constant plane and the raw
/// clock. The final constant plane is always filled with ones.
///
/// # Arguments
///
/// * `planes` - destination channel-major workspace
/// * `position` - current position supplying the auxiliary state
/// * `format` - input-plane interpretation being produced
/// * `black_to_move` - whether Black owns the current perspective
/// * `transform` - canonical transform mask applied to spatial planes
fn write_aux_planes(
    planes: &mut [f32],
    position: &Position,
    format: Bt4InputFormat,
    black_to_move: bool,
    transform: u8,
) {
    if format == Bt4InputFormat::Classical112 {
        write_classical_castling_planes(planes, position, black_to_move);
    } else {
        write_castling_rook_plane(planes, AUX_BASE, position, black_to_move, false, transform);
        write_castling_rook_plane(
            planes,
            AUX_BASE + 1,
            position,
            black_to_move,
            true,
            transform,
        );
    }

    if format == Bt4InputFormat::Canonical112 {
        let en_passant = position
            .en_passant_square()
            .map_or(0, |square| perspective_square_bit(square, black_to_move));
        write_bits(
            planes,
            AUX_BASE + 4,
            transform_plane_bits(en_passant, transform),
        );
    } else if black_to_move {
        fill_plane(planes, AUX_BASE + 4, 1.0);
    }

    let halfmove = f32::from(position.halfmove_clock());
    let halfmove = if format == Bt4InputFormat::Canonical112 {
        halfmove / 100.0
    } else {
        halfmove
    };
    fill_plane(planes, AUX_BASE + 5, halfmove);
    fill_plane(planes, AUX_BASE + 7, 1.0);
}

/// Used for writing four constant castling-right planes from side-to-move
/// perspective.
///
/// The plane order is our queenside, our kingside, their queenside, their
/// kingside; each granted right fills its whole plane with ones.
///
/// # Arguments
///
/// * `planes` - destination channel-major workspace
/// * `position` - current position supplying the castling rights
/// * `black_to_move` - whether Black owns the current perspective
fn write_classical_castling_planes(planes: &mut [f32], position: &Position, black_to_move: bool) {
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
    let rights = position.castling_rights();
    for (plane, right) in [
        (AUX_BASE, our_queenside),
        (AUX_BASE + 1, our_kingside),
        (AUX_BASE + 2, their_queenside),
        (AUX_BASE + 3, their_kingside),
    ] {
        if rights.contains(right) {
            fill_plane(planes, plane, 1.0);
        }
    }
}

/// Used for writing the two rook identities that share one castling-side
/// plane.
///
/// Both sides' rooks for the selected board side are marked as perspective
/// square bits on the same plane, so the network sees which rooks still
/// participate in castling.
///
/// # Arguments
///
/// * `planes` - destination channel-major workspace
/// * `plane` - auxiliary plane index receiving the rook bits
/// * `position` - current position supplying rights and rook squares
/// * `black_to_move` - whether Black owns the current perspective
/// * `kingside` - whether the kingside rather than queenside rooks are
///   written
/// * `transform` - canonical transform mask applied to the plane bits
fn write_castling_rook_plane(
    planes: &mut [f32],
    plane: usize,
    position: &Position,
    black_to_move: bool,
    kingside: bool,
    transform: u8,
) {
    let (our_right, their_right) = match (black_to_move, kingside) {
        (false, false) => (
            CastlingRights::WHITE_QUEENSIDE,
            CastlingRights::BLACK_QUEENSIDE,
        ),
        (false, true) => (
            CastlingRights::WHITE_KINGSIDE,
            CastlingRights::BLACK_KINGSIDE,
        ),
        (true, false) => (
            CastlingRights::BLACK_QUEENSIDE,
            CastlingRights::WHITE_QUEENSIDE,
        ),
        (true, true) => (
            CastlingRights::BLACK_KINGSIDE,
            CastlingRights::WHITE_KINGSIDE,
        ),
    };
    let bits = [our_right, their_right]
        .into_iter()
        .filter_map(|right| position.castling_rook_square(right))
        .fold(0_u64, |bits, square| {
            bits | perspective_square_bit(square, black_to_move)
        });
    write_bits(planes, plane, transform_plane_bits(bits, transform));
}

/// Used for selecting the deterministic spatial canonicalization used by the
/// Java path.
///
/// Positions with any castling right keep the identity transform. Otherwise
/// the side-to-move king is flipped into the right half of the board; when no
/// pawns remain it is additionally mirrored into the lower half, a king
/// beyond the a8-h1 diagonal forces the transpose bit, and a king exactly on
/// that diagonal decides it through [`compare_transposing`]'s stable
/// tie-break.
///
/// # Arguments
///
/// * `position` - current position supplying castling rights
/// * `perspective` - perspective-oriented piece bitboards of the position
///
/// # Returns
///
/// Canonical transform mask for the position.
fn choose_canonical_transform(position: &Position, perspective: &[u64; 12]) -> u8 {
    if position.castling_rights().bits() != 0 {
        return 0;
    }
    let mut our_king = perspective[PieceKind::King.index()];
    let mut transform = 0;
    if our_king & 0x0F0F_0F0F_0F0F_0F0F != 0 {
        transform |= FLIP_TRANSFORM;
        our_king = flip_files(our_king);
    }
    let pawns = perspective[PieceKind::Pawn.index()] | perspective[6 + PieceKind::Pawn.index()];
    if pawns != 0 {
        return transform;
    }
    if our_king & 0xFFFF_FFFF_0000_0000 != 0 {
        transform |= MIRROR_TRANSFORM;
        our_king = mirror_ranks(our_king);
    }
    if our_king & 0x0000_0000_E0C0_8000 != 0 {
        return transform | TRANSPOSE_TRANSFORM;
    }
    if our_king & 0x0000_0000_1020_4080 == 0 {
        return transform;
    }
    if compare_transposing(perspective, transform).is_gt() {
        transform | TRANSPOSE_TRANSFORM
    } else {
        transform
    }
}

/// Used for tie-breaking diagonal canonicalization with stable unsigned
/// bitboard ordering.
///
/// A fixed sequence of combined piece bitboards is compared against its own
/// transpose; the first unequal comparison decides whether transposing yields
/// the canonically smaller board.
///
/// # Arguments
///
/// * `perspective` - perspective-oriented piece bitboards of the position
/// * `transform` - flip/mirror mask already chosen for the position
///
/// # Returns
///
/// Ordering of the transformed bitboards relative to their transposes.
fn compare_transposing(perspective: &[u64; 12], transform: u8) -> std::cmp::Ordering {
    let tests = [
        union(perspective),
        own_union(perspective),
        perspective[PieceKind::King.index()] | perspective[6 + PieceKind::King.index()],
        perspective[PieceKind::Queen.index()] | perspective[6 + PieceKind::Queen.index()],
        perspective[PieceKind::Rook.index()] | perspective[6 + PieceKind::Rook.index()],
        perspective[PieceKind::Knight.index()] | perspective[6 + PieceKind::Knight.index()],
        perspective[PieceKind::Bishop.index()] | perspective[6 + PieceKind::Bishop.index()],
    ];
    for bits in tests {
        let value = transform_plane_bits(bits, transform);
        let comparison = value.cmp(&transpose(value));
        if !comparison.is_eq() {
            return comparison;
        }
    }
    std::cmp::Ordering::Equal
}

/// Used for combining all twelve perspective piece planes.
///
/// # Arguments
///
/// * `bits` - perspective-oriented piece bitboards
///
/// # Returns
///
/// Bitwise union of every plane.
fn union(bits: &[u64; 12]) -> u64 {
    bits.iter().copied().fold(0, std::ops::BitOr::bitor)
}

/// Used for combining the six side-to-move piece planes.
///
/// # Arguments
///
/// * `bits` - perspective-oriented piece bitboards
///
/// # Returns
///
/// Bitwise union of the first six planes.
fn own_union(bits: &[u64; 12]) -> u64 {
    bits[..6].iter().copied().fold(0, std::ops::BitOr::bitor)
}

/// Used for converting one core square to an a1-origin side-to-move plane
/// bit.
///
/// Janus squares are a8-origin, so the White perspective byte-swaps the
/// square bit while the Black perspective uses it directly.
///
/// # Arguments
///
/// * `square` - core square to convert
/// * `black_to_move` - whether Black owns the current perspective
///
/// # Returns
///
/// Single-bit bitboard in perspective plane coordinates.
fn perspective_square_bit(square: Square, black_to_move: bool) -> u64 {
    if black_to_move {
        square.bit()
    } else {
        square.bit().swap_bytes()
    }
}

/// Used for converting one core square to an a1-origin side-to-move token
/// index.
///
/// # Arguments
///
/// * `position` - position supplying the side to move
/// * `square` - core square to convert
///
/// # Returns
///
/// Token index in `0..TOKENS` from the side-to-move perspective.
fn perspective_square(position: &Position, square: Square) -> usize {
    let file = usize::from(square.file());
    let white_rank = usize::from(square.rank() - 1);
    let rank = if position.side_to_move() == Color::Black {
        7 - white_rank
    } else {
        white_rank
    };
    rank * 8 + file
}

/// Used for converting an internally encoded castle to the attention-policy
/// destination.
///
/// `LCZero` policy always represents castling as king-to-participating-rook.
/// Janus uses king-to-c/g for orthodox chess and king-to-rook for Chess960, so
/// only the orthodox form requires translation.
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - move whose policy destination is requested
///
/// # Returns
///
/// The participating rook square for a recognized castle, otherwise the
/// move's ordinary destination.
fn policy_destination(position: &Position, mv: Move) -> Square {
    let Some(piece) = position.piece_at(mv.from()) else {
        return mv.to();
    };
    if piece.kind != PieceKind::King || piece.color != position.side_to_move() {
        return mv.to();
    }
    for kingside in [true, false] {
        let right = castling_right(piece.color, kingside);
        let Some(rook) = position.castling_rook_square(right) else {
            continue;
        };
        let internal_target = if position.is_chess960() {
            rook
        } else {
            castling_king_target(piece.color, kingside)
        };
        if mv.to() == internal_target {
            return rook;
        }
    }
    mv.to()
}

/// Used for retrieving one castling permission bit for a color and board
/// side.
///
/// # Arguments
///
/// * `color` - side owning the right
/// * `kingside` - whether the kingside rather than queenside right is
///   requested
///
/// # Returns
///
/// Matching [`CastlingRights`] bit constant.
const fn castling_right(color: Color, kingside: bool) -> u8 {
    match (color, kingside) {
        (Color::White, true) => CastlingRights::WHITE_KINGSIDE,
        (Color::White, false) => CastlingRights::WHITE_QUEENSIDE,
        (Color::Black, true) => CastlingRights::BLACK_KINGSIDE,
        (Color::Black, false) => CastlingRights::BLACK_QUEENSIDE,
    }
}

/// Used for retrieving the orthodox internal king destination for one castle.
///
/// # Arguments
///
/// * `color` - side performing the castle
/// * `kingside` - whether the kingside rather than queenside castle is
///   requested
///
/// # Returns
///
/// The g- or c-file king destination on the side's back rank.
const fn castling_king_target(color: Color, kingside: bool) -> Square {
    match (color, kingside) {
        (Color::White, true) => Square::G1,
        (Color::White, false) => Square::C1,
        (Color::Black, true) => Square::G8,
        (Color::Black, false) => Square::C8,
    }
}

/// Used for applying a canonical bitboard permutation to one token index.
///
/// # Arguments
///
/// * `square` - token index in `0..TOKENS`
/// * `transform` - canonical transform mask
///
/// # Returns
///
/// Permuted token index.
fn transform_square(square: usize, transform: u8) -> usize {
    usize::try_from(transform_plane_bits(1_u64 << square, transform).trailing_zeros())
        .expect("a transformed single square fits usize")
}

/// Used for writing set bits into one channel-major plane.
///
/// # Arguments
///
/// * `planes` - destination channel-major workspace
/// * `plane` - channel index receiving the bits
/// * `bits` - a1-origin bitboard whose set squares become `1.0`
///
/// # Panics
///
/// Panics when `plane` addresses values beyond the end of `planes`.
fn write_bits(planes: &mut [f32], plane: usize, mut bits: u64) {
    let base = plane * TOKENS;
    while bits != 0 {
        let square = usize::try_from(bits.trailing_zeros()).expect("a square index fits usize");
        planes[base + square] = 1.0;
        bits &= bits - 1;
    }
}

/// Used for filling one channel-major plane with a scalar feature.
///
/// # Arguments
///
/// * `planes` - destination channel-major workspace
/// * `plane` - channel index to fill
/// * `value` - scalar written to all 64 squares of the plane
///
/// # Panics
///
/// Panics when `plane` addresses values beyond the end of `planes`.
fn fill_plane(planes: &mut [f32], plane: usize, value: f32) {
    let start = plane * TOKENS;
    planes[start..start + TOKENS].fill(value);
}

/// Used for validating one exact tensor or workspace length.
///
/// # Arguments
///
/// * `role` - stable name of the tensor or workspace being validated
/// * `actual` - number of values supplied by the caller
/// * `expected` - required number of values
///
/// # Errors
///
/// Returns [`Bt4EncodingError::LengthMismatch`] when the lengths differ.
fn validate_length(
    role: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), Bt4EncodingError> {
    if actual == expected {
        Ok(())
    } else {
        Err(Bt4EncodingError::LengthMismatch {
            role,
            expected,
            actual,
        })
    }
}

/// Used for computing the unsigned distance between two board coordinates.
///
/// # Arguments
///
/// * `left` - first coordinate
/// * `right` - second coordinate
///
/// # Returns
///
/// Absolute difference of the coordinates.
const fn absolute_difference(left: usize, right: usize) -> usize {
    left.abs_diff(right)
}

/// Used for reporting whether one from/to pair is represented by attention
/// geometry.
///
/// The attention policy covers queen-like moves (same file, same rank, or
/// same diagonal) and knight jumps; every other pair has no logit.
///
/// # Arguments
///
/// * `from` - source token index
/// * `to` - destination token index
///
/// # Returns
///
/// `true` when the pair is queen-like or a knight jump.
const fn is_queen_like_or_knight(from: usize, to: usize) -> bool {
    let file_difference = absolute_difference(from & 7, to & 7);
    let rank_difference = absolute_difference(from >> 3, to >> 3);
    file_difference == 0
        || rank_difference == 0
        || file_difference == rank_difference
        || (file_difference == 1 && rank_difference == 2)
        || (file_difference == 2 && rank_difference == 1)
}

/// Used for mirroring all files of one a1-origin bitboard.
///
/// Implemented as a three-step bit-parallel swap of adjacent files, file
/// pairs, and file quads.
///
/// # Arguments
///
/// * `bits` - bitboard to reflect horizontally
///
/// # Returns
///
/// File-reflected bitboard.
const fn flip_files(mut bits: u64) -> u64 {
    bits = ((bits >> 1) & 0x5555_5555_5555_5555) | ((bits & 0x5555_5555_5555_5555) << 1);
    bits = ((bits >> 2) & 0x3333_3333_3333_3333) | ((bits & 0x3333_3333_3333_3333) << 2);
    ((bits >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((bits & 0x0F0F_0F0F_0F0F_0F0F) << 4)
}

/// Used for mirroring all ranks of one a1-origin bitboard.
///
/// One rank occupies one byte, so a byte swap reverses the rank order.
///
/// # Arguments
///
/// * `bits` - bitboard to reflect vertically
///
/// # Returns
///
/// Rank-reflected bitboard.
const fn mirror_ranks(bits: u64) -> u64 {
    bits.swap_bytes()
}

/// Used for transposing one bitboard across the a8-h1 anti-diagonal.
///
/// Implemented as the classical three-step delta-swap exchanging 1x1, 2x2,
/// and 4x4 blocks around the anti-diagonal, mapping `a1` to `h8`.
///
/// # Arguments
///
/// * `bits` - bitboard to transpose
///
/// # Returns
///
/// Transposed bitboard.
const fn transpose(mut bits: u64) -> u64 {
    bits = ((bits & 0xAA00_AA00_AA00_AA00) >> 9)
        | ((bits & 0x0055_0055_0055_0055) << 9)
        | (bits & 0x55AA_55AA_55AA_55AA);
    bits = ((bits & 0xCCCC_0000_CCCC_0000) >> 18)
        | ((bits & 0x0000_3333_0000_3333) << 18)
        | (bits & 0x3333_CCCC_3333_CCCC);
    ((bits & 0xF0F0_F0F0_0000_0000) >> 36)
        | ((bits & 0x0000_0000_0F0F_0F0F) << 36)
        | (bits & 0x0F0F_0F0F_F0F0_F0F0)
}

/// Used for building the immutable geometric attention-policy gather map.
///
/// Valid from/to pairs are numbered first in row-major order, followed by the
/// queen, rook, and bishop promotion groups for each legal from-file/to-file
/// pair; unrepresented entries stay `-1`. The final counter equals
/// [`POLICY_SIZE`].
///
/// # Returns
///
/// Complete internal-to-compressed index map.
const fn build_compressed_by_internal() -> [i16; INTERNAL_POLICY_SIZE] {
    let mut map = [-1_i16; INTERNAL_POLICY_SIZE];
    let mut next = 0_i16;
    let mut from = 0;
    while from < TOKENS {
        let mut to = 0;
        while to < TOKENS {
            if from != to && is_queen_like_or_knight(from, to) {
                map[from * TOKENS + to] = next;
                next += 1;
            }
            to += 1;
        }
        from += 1;
    }
    let mut from_file = 0;
    while from_file < 8 {
        let minimum_to = if from_file == 0 { 0 } else { from_file - 1 };
        let maximum_to = if from_file == 7 { 7 } else { from_file + 1 };
        let mut to_file = minimum_to;
        while to_file <= maximum_to {
            let mut promotion = 0;
            while promotion < 3 {
                map[FROM_TO_POLICY_SIZE + from_file * 24 + to_file * 3 + promotion] = next;
                next += 1;
                promotion += 1;
            }
            to_file += 1;
        }
        from_file += 1;
    }
    map
}

/// Used for mapping uncompressed attention logits to compressed policy
/// indices at compile time.
///
/// Negative entries mark geometrically invalid logits; the table is built by
/// [`build_compressed_by_internal`] and exposed through
/// [`compressed_by_internal_map`].
const COMPRESSED_BY_INTERNAL: [i16; INTERNAL_POLICY_SIZE] = build_compressed_by_internal();

