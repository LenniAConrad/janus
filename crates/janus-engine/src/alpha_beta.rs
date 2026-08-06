//! Deterministic iterative-deepening alpha-beta search.
//!
//! The module hosts the reusable [`AlphaBeta`] searcher, its per-search
//! `SearchContext`, and the pruning, ordering, quiescence, and
//! static-exchange helpers shared by the main worker, Lazy-SMP helpers, and
//! `MultiPV` batches. All decisions are deterministic: identical inputs and
//! depth or node limits reproduce the same nodes, scores, and principal
//! variations.

use crate::classical::Classical;
use crate::evaluator::SearchEvaluator;
use crate::limits::SearchLimits;
use crate::score::{
    clamp_static_score, score_from_tt, score_to_tt, terminal_score, INFINITY, MATE_SCORE,
    MATE_THRESHOLD, TB_WIN_SCORE,
};
use crate::search_math::{
    late_move_prune_threshold, lmr_reduction, lmr_reduction_units, null_move_reduction,
    stability_budget_millis, LMR_UNIT,
};
use crate::syzygy::{Prober, SyzygyConfig};
use crate::tt::{Bound, SharedTranspositionTable, TranspositionTable, TtHit, TtPayload};
use janus_core::{Color, Move, Piece, PieceKind, Position, Square, NO_MOVE};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Used for selecting the first iterative-deepening depth that narrows the
/// initial alpha-beta window.
const ASPIRATION_START_DEPTH: u8 = 4;
/// Used for sizing the initial half-width, in centipawns, of an aspiration
/// window around the previous iteration's score.
const ASPIRATION_WINDOW: i32 = 50;
/// Used for bounding the number of tactical plies searched after entering
/// quiescence.
const QUIESCENCE_MAX_PLY: u8 = 8;
/// Used for bounding the stack depth and principal-variation length of every
/// search path.
const MAX_SEARCH_PLY: usize = 128;
/// Used for spacing periodic time-limit checks, in nodes, after the search
/// warms up.
const STOP_CHECK_INTERVAL: u64 = 1_024;
/// Used for counting the opening nodes during which the hard time limit is
/// checked on every node.
const STOP_CHECK_EVERY_NODE_UNTIL: u64 = 512;
/// Used for capping the soft clock target, in milliseconds, when the root has
/// exactly one legal reply.
///
/// A forced reply gains nothing from a full soft allocation, so the soft
/// target shrinks to at most this cap (2026-07-24, TIME-178). The hard
/// deadline and all fixed-node and fixed-depth budgets stay untouched, so the
/// cap only applies where a soft budget exists.
const SINGLE_REPLY_SOFT_CAP_MILLIS: u64 = 50;
/// Used for sizing a standalone searcher's default transposition table in
/// entries.
const DEFAULT_TT_ENTRIES: usize = 1 << 20;
/// Used for sizing the exact-key static-evaluation cache of one search.
const EVAL_CACHE_ENTRIES: usize = 1 << 16;
/// Used as the evaluation-cache size when memory is not the binding
/// constraint.
///
/// Measured hit rates over 3M-node searches: at the historical 2^16 buckets
/// the cache hits 12.2% in a middlegame and **collides on 85.3%** of probes —
/// it thrashes. At 2^20 the same position hits 20.7% with 45.8% collisions,
/// and 2^22 adds little beyond that. Growing the cache is output-preserving:
/// it stores exact static scores, so a miss merely recomputes the identical
/// value and no search decision changes.
const EVAL_CACHE_ENTRIES_LARGE: usize = 1 << 20;
/// Used for the thread count above which the evaluation cache is shrunk.
///
/// The cache is **per worker**, so a flat large size becomes gigabytes at
/// competition thread counts: 2^20 buckets is about 13 MiB, which is 13 GiB
/// across 1024 threads. Above this many workers the size is halved for each
/// doubling of the thread count, down to [`EVAL_CACHE_ENTRIES`].
const EVAL_CACHE_FULL_SIZE_THREADS: usize = 16;
/// Used for the process-wide evaluation-cache size chosen from the configured
/// thread count.
///
/// Set once by the UCI layer before any search starts, so every worker in a
/// run agrees and behaviour stays deterministic.
static EVAL_CACHE_SIZE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(EVAL_CACHE_ENTRIES_LARGE);

/// Used for choosing the evaluation-cache size for a worker pool.
///
/// # Arguments
///
/// * `threads` - configured worker count sharing this machine
///
/// # Returns
///
/// Bucket count for each worker's evaluation cache.
#[must_use]
pub fn eval_cache_size_for_threads(threads: usize) -> usize {
    let workers = threads.max(1);
    if workers <= EVAL_CACHE_FULL_SIZE_THREADS {
        return EVAL_CACHE_ENTRIES_LARGE;
    }
    let shrink = (workers / EVAL_CACHE_FULL_SIZE_THREADS).next_power_of_two();
    (EVAL_CACHE_ENTRIES_LARGE / shrink).max(EVAL_CACHE_ENTRIES)
}

/// Used for publishing the evaluation-cache size before a run starts.
///
/// # Arguments
///
/// * `entries` - bucket count each worker's cache should use
pub fn set_eval_cache_size(entries: usize) {
    EVAL_CACHE_SIZE.store(
        entries.max(EVAL_CACHE_ENTRIES).next_power_of_two(),
        std::sync::atomic::Ordering::Relaxed,
    );
}
/// Used for saturating history and continuation-history values at an
/// absolute bound.
const HISTORY_MAX: i32 = 1 << 14;
/// Used for sizing the capture-history table over colored-mover, destination,
/// and victim-kind cells.
const CAPTURE_HISTORY_BUCKETS: usize = 12 * 64 * 6;
/// Used for saturating capture history while preserving Janus's fixed capture
/// and killer base bands.
///
/// This adjustment alone cannot move the lowest non-losing capture below the
/// 70,000 primary-killer base or a losing capture above the 69,000 secondary
/// base. Quiet history, continuation, and evaluator priors remain independent
/// ordering inputs and are deliberately not constrained by this bound.
const CAPTURE_HISTORY_MAX: i32 = 1 << 13;
/// Used for marking a ply at which no static evaluation was available.
const NO_STATIC_SCORE: i32 = i32::MIN;
/// Used for sizing one correction-history table over structural hash cells.
///
/// The index is a mixed structural key reduced modulo this power of two, so
/// distinct structures may share a cell. Correction history tolerates that:
/// a collision blends two unrelated residual estimates toward each other and
/// the gravity update in [`CorrectionEntry::update`] decays the error away
/// again, whereas a transposition collision would return a wrong bound.
const CORRECTION_ENTRIES: usize = 1 << 14;
/// Used for bounding the magnitude a correction-history cell may reach.
///
/// The stored unit is a centipawn scaled by [`CORRECTION_GRAIN`], so this
/// bound caps any single correction at `CORRECTION_LIMIT / CORRECTION_GRAIN`
/// centipawns. Keeping the cap small is deliberate: the mechanism is meant to
/// remove a systematic evaluator bias for a pawn structure, not to override
/// the evaluation.
const CORRECTION_LIMIT: i32 = 1_024;
/// Used for bounding one correction-history update.
///
/// A quarter of [`CORRECTION_LIMIT`] means no single node can move a cell more
/// than a quarter of the way to saturation, so a cell reflects a pattern seen
/// repeatedly rather than one deep search that happened to disagree.
const CORRECTION_MAX_BONUS: i32 = CORRECTION_LIMIT / 4;
/// Used for scaling stored correction units back into centipawns.
const CORRECTION_GRAIN: i32 = 32;
/// Used for scaling the search-versus-static residual into a stored bonus.
///
/// The residual is multiplied by the remaining depth and divided by this
/// constant, so a deep disagreement moves the cell further than a shallow one.
const CORRECTION_BONUS_DIVISOR: i32 = 4;
/// Correction-history flavor selected for one searcher.
///
/// Correction history records how far a completed search moved away from the
/// static evaluation for a given pawn structure, then applies the learned
/// residual to later static evaluations of positions sharing that structure.
/// It is a search-side repair for a systematically biased evaluator, which is
/// exactly Janus's measured situation: `INFRA-20260805-318` attributes about
/// `793` Elo of an equal-node contrast to the evaluator alone.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CorrectionHistoryMode {
    /// Released behavior: static evaluations are used exactly as produced.
    #[default]
    Off,
    /// Research candidate: correct by pawn structure alone.
    Pawn,
    /// Research candidate: correct by pawn structure and by each side's
    /// non-pawn material placement.
    PawnAndNonPawn,
}

impl CorrectionHistoryMode {
    /// Used for deciding whether any correction table is consulted.
    ///
    /// # Returns
    ///
    /// `true` when the flavor applies a correction, `false` for
    /// [`CorrectionHistoryMode::Off`].
    const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Used for sizing the per-searcher correction workspace.
    ///
    /// # Returns
    ///
    /// The number of structural tables the flavor indexes, per side to move.
    const fn table_count(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Pawn => 1,
            Self::PawnAndNonPawn => 3,
        }
    }
}

/// Research search flavors selected for one searcher.
///
/// Each field defaults to the released behavior, so
/// `ResearchSearch::default()` leaves the search byte-identical to a build
/// without any research flavor. The flavors are grouped in one value rather
/// than spread across the searcher so adding the next one does not widen every
/// construction site again.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResearchSearch {
    /// Used for selecting which structural correction tables the search
    /// consults and updates.
    pub correction: CorrectionHistoryMode,
    /// Used for selecting `STR-20260806-335`'s evaluation-independent
    /// late-move pruning.
    ///
    /// `false` is the released behavior. Setting it removes the static-
    /// evaluation precondition from the late-move prune, leaving the move
    /// count and history conditions in force.
    pub eval_free_late_move_pruning: bool,
}

/// One correction-history cell holding a scaled centipawn residual.
///
/// Stored as [`i16`] so a full flavor's workspace stays inside a few tens of
/// kilobytes and shares cache with the ordering histories it sits beside.
/// [`CORRECTION_LIMIT`] bounds the value far below the type's range.
#[derive(Clone, Copy, Debug, Default)]
struct CorrectionEntry(i16);

impl CorrectionEntry {
    /// Used for blending one observed residual into the cell.
    ///
    /// The update is the standard history "gravity" form: the increment is
    /// damped in proportion to how close the cell already sits to
    /// [`CORRECTION_LIMIT`], so the value converges instead of saturating and
    /// old evidence decays without an explicit ageing pass.
    ///
    /// # Arguments
    ///
    /// * `bonus` - clamped scaled residual observed at one node
    fn update(&mut self, bonus: i32) {
        let value = i32::from(self.0);
        let blended = value + bonus - value * bonus.abs() / CORRECTION_LIMIT;
        self.0 = i16::try_from(blended.clamp(-CORRECTION_LIMIT, CORRECTION_LIMIT))
            .expect("a clamped correction value fits i16");
    }

    /// Used for reading the cell as a scaled centipawn residual.
    ///
    /// # Returns
    ///
    /// The stored value widened to [`i32`].
    const fn value(self) -> i32 {
        self.0 as i32
    }
}

/// Used for mixing a bitboard into a well-distributed 64-bit hash.
///
/// Correction history indexes by piece placement rather than by the
/// transposition key, and Janus's position keys are not maintained per piece
/// class, so the structural keys are derived here instead of being carried
/// incrementally through make/unmake. This is the `SplitMix64` finalizer: five
/// dependent operations with no table lookup, chosen because the index needs
/// avalanche rather than any algebraic property.
///
/// # Arguments
///
/// * `value` - bitboard or salted bitboard to mix
///
/// # Returns
///
/// The avalanched 64-bit hash of `value`.
const fn mix_bits(value: u64) -> u64 {
    let mut mixed = value;
    mixed ^= mixed >> 30;
    mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed ^= mixed >> 27;
    mixed = mixed.wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

/// Used for deriving the pawn-placement key of a position.
///
/// The two pawn bitboards are mixed independently and one is rotated before
/// they are combined, so a position and its color-swapped pawn skeleton do not
/// share a cell.
///
/// # Arguments
///
/// * `position` - position whose pawn placement is hashed
///
/// # Returns
///
/// A 64-bit key determined by pawn placement alone.
fn pawn_structure_key(position: &Position) -> u64 {
    let white = position.piece_bitboard(Piece::new(Color::White, PieceKind::Pawn));
    let black = position.piece_bitboard(Piece::new(Color::Black, PieceKind::Pawn));
    mix_bits(white).rotate_left(1) ^ mix_bits(black)
}

/// Used for deriving one side's non-pawn placement key.
///
/// Each kind is rotated by a distinct amount so that, for example, a knight on
/// `c3` and a bishop on `c3` do not cancel. An absent kind hashes to zero and
/// therefore contributes nothing, which is the intended behavior: two
/// positions that both lack queens should agree on that dimension.
///
/// # Arguments
///
/// * `position` - position whose piece placement is hashed
/// * `color` - side whose non-pawn pieces are hashed
///
/// # Returns
///
/// A 64-bit key determined by that side's non-pawn placement alone.
fn non_pawn_structure_key(position: &Position, color: Color) -> u64 {
    let kinds = [
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
        PieceKind::King,
    ];
    let mut key = 0;
    for (index, kind) in kinds.into_iter().enumerate() {
        let board = position.piece_bitboard(Piece::new(color, kind));
        key ^= mix_bits(board).rotate_left(u32::try_from(index).expect("five kinds fit u32") * 11);
    }
    key
}
/// Used for marking a missing piece-square continuation context.
const NO_CONTINUATION: u16 = u16::MAX;
/// Used for sizing one continuation-history dimension in piece-square
/// values.
const CONTINUATION_BUCKETS: usize = 12 * 64;
/// Used for thresholding the continuation history at or above which
/// late-move reduction shrinks by one ply.
const CONTINUATION_GOOD: i32 = 1 << 12;
/// Used for thresholding the continuation history at or below which
/// late-move reduction grows by one ply.
const CONTINUATION_BAD: i32 = -(1 << 12);
/// Used for bounding how far the fractional late-move reduction may extend a
/// late quiet move.
///
/// Only the STR-316 flavor can reach a negative reduction. Capping it at one
/// ply keeps the deepening strictly smaller than the depth consumed by the
/// recursion itself, so an extended late move cannot grow the tree without
/// bound; the released flavor keeps its `0` floor.
const LMR_MAX_EXTENSION_PLIES: i32 = 1;
/// Used for gating null-move pruning at a minimum remaining depth.
const NULL_MOVE_MIN_DEPTH: i32 = 3;
/// Used for selecting the minimum remaining depth at which a null fail-high
/// requires verification.
///
/// Shallower null searches retain Janus's established pruning policy. At deep
/// nodes, re-searching the original position prevents a zugzwang-like null
/// result from becoming an unverified cutoff.
const NULL_MOVE_VERIFICATION_MIN_DEPTH: i32 = 16;
/// Used for bounding the deepest node at which reverse futility pruning is
/// allowed.
const REVERSE_FUTILITY_MAX_DEPTH: i32 = 4;
/// Used for scaling the per-ply centipawn margin of reverse futility
/// pruning.
const REVERSE_FUTILITY_MARGIN: i32 = 120;
/// Used for bounding the deepest node at which quiet-move futility pruning is
/// allowed.
const FUTILITY_MAX_DEPTH: i32 = 3;
/// Used for scaling the per-ply centipawn margin of quiet-move futility
/// pruning.
const FUTILITY_MARGIN: i32 = 150;
/// Used for bounding the deepest node at which late quiet moves may be pruned
/// outright.
const LATE_MOVE_PRUNING_MAX_DEPTH: i32 = 5;
/// Used for retaining a material margin when delta-pruning quiescence
/// captures.
const DELTA_MARGIN: i32 = 200;
/// Used for gating internal iterative reduction at the minimum depth from
/// which it may remove one ply.
const IIR_MIN_DEPTH: i32 = 4;
/// Used for gating singular-extension checks at a minimum remaining depth.
const SINGULAR_MIN_DEPTH: i32 = 8;
/// Used for bounding how much shallower than the current remaining depth a
/// transposition entry may be while still seeding a singular check.
const SINGULAR_TT_DEPTH_MARGIN: i32 = 3;
/// Used for scaling the depth-proportional margin subtracted from the
/// transposition score when forming the singular verification bound
/// `singular_beta = tt_score - SINGULAR_MARGIN * depth / 64`.
const SINGULAR_MARGIN: i32 = 96;
/// Used for gating conservative `ProbCut` at a minimum remaining depth.
///
/// `ProbCut` is only attempted with at least this much full-width depth left, so
/// the reduced verification search still has meaningful horizon after the
/// [`PROBCUT_REDUCTION`] cut. The value is deliberately more cautious than the
/// depth-3 gate used by Stockfish, Reckless, and Viridithas; conservative v1
/// only prunes where the verification is comparatively trustworthy.
const PROBCUT_MIN_DEPTH: i32 = 5;
/// Used for the centipawn margin added to `beta` to form the `ProbCut` bound.
///
/// `probcut_beta = beta + PROBCUT_MARGIN`. A capture must plausibly reach this
/// raised bound before it is worth verifying. The value sits inside the range
/// of the reference engines (Stockfish 241, Reckless 254, Viridithas 176 base);
/// 180 is a fixed, improving-agnostic choice appropriate for a conservative
/// first cut.
const PROBCUT_MARGIN: i32 = 180;
/// Used for the depth reduction applied to the `ProbCut` verification search.
///
/// The reduced search runs at `depth - PROBCUT_REDUCTION`. With the depth-5 gate
/// this leaves at least one full-width ply. The reduction matches the middle of
/// the reference range (Stockfish reduces by 3-5 depending on `improving`);
/// conservative v1 uses the fixed, deeper reduction of 4.
const PROBCUT_REDUCTION: i32 = 4;
/// Used for setting the width of the alternating searched and skipped depth
/// bands in each deterministic helper worker's schedule.
///
/// Paired with [`HELPER_SKIP_PHASE`] to give 64 distinct schedules. **The
/// first sixteen entries are the originals, unchanged**, so every helper index
/// that existing measurements used behaves exactly as before; only indices
/// beyond sixteen are new. Sixteen sufficed while `Threads` was capped there,
/// but the schedule table is indexed modulo its length, so a competition thread
/// count previously gave several helpers identical schedules — each duplicate
/// searching the same depths as an earlier helper and contributing no
/// Lazy-SMP diversity.
const HELPER_SKIP_SIZE: [usize; 64] = [
    1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8,
    8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11,
    11, 11, 11, 11, 11,
];
/// Used for setting the phase offset paired with [`HELPER_SKIP_SIZE`] for
/// each helper worker.
const HELPER_SKIP_PHASE: [usize; 64] = [
    0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1,
    2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6,
];

/// Outcome of a conservative `ProbCut` attempt at one search node.
///
/// The three cases are distinct because the caller must react differently to
/// each: propagate an interruption, return a pruned score, or fall through to
/// the ordinary move loop. A three-state result is clearer here than an
/// `Option<Option<i32>>`.
enum ProbcutOutcome {
    /// Used for signalling that the search was interrupted mid-verification and
    /// the caller must return `None`.
    Interrupted,
    /// Used for signalling that no cutoff applied and the ordinary search must
    /// continue.
    Continue,
    /// Used for carrying the score of a confirmed `ProbCut` cutoff, already
    /// mapped back into the caller's window.
    Cutoff(i32),
}

/// Failure returned before alpha-beta tree work begins.
///
/// Input errors are detected during validation. Resource exhaustion is
/// reported while admitting freshly zeroed per-search workspace, before any
/// node is visited.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchError {
    /// Used for indicating that the requested depth or time limits violate
    /// [`SearchLimits`] invariants.
    InvalidLimits,
    /// Used for indicating that a restricted search was requested without any
    /// root moves.
    EmptyRootMoves,
    /// Used for indicating that a restricted root move is not legal in the
    /// searched position.
    IllegalRootMove(Move),
    /// Used for indicating that `MultiPV` was requested with a zero line
    /// count.
    InvalidMultiPv,
    /// Used for indicating that freshly zeroed per-search workspace could not
    /// be allocated.
    ResourceExhausted,
}

impl fmt::Display for SearchError {
    /// Used for writing a concise diagnostic suitable for protocol and test
    /// output.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination for the rendered diagnostic text
    ///
    /// # Returns
    ///
    /// Propagated success or failure of the underlying formatter writes.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("invalid search limits"),
            Self::EmptyRootMoves => formatter.write_str("root move whitelist is empty"),
            Self::IllegalRootMove(mv) => write!(formatter, "illegal root move: {mv}"),
            Self::InvalidMultiPv => formatter.write_str("MultiPV line count must be positive"),
            Self::ResourceExhausted => {
                formatter.write_str("alpha-beta search workspace allocation failed")
            }
        }
    }
}

impl Error for SearchError {}

/// Progress report emitted after a fully completed iterative-deepening pass.
///
/// One value is delivered to a search observer for every finished depth, so
/// callers can stream protocol-style progress output while the search runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchInfo {
    /// Used for reporting the completed depth in plies.
    pub depth: u8,
    /// Used for reporting the greatest ply entered during this search.
    pub selective_depth: u16,
    /// Used for reporting the root-side score in centipawns or mate-score
    /// encoding.
    pub score: i32,
    /// Used for reporting the number of nodes visited so far.
    pub nodes: u64,
    /// Used for reporting the wall-clock time elapsed so far.
    pub elapsed: Duration,
    /// Used for reporting current-generation transposition-table occupancy in
    /// per mille.
    pub hashfull_per_mille: u16,
    /// Used for reporting successful tablebase accesses so far; zero
    /// whenever no tablebases are configured.
    pub tb_hits: u64,
    /// Used for reporting the current principal variation.
    pub principal_variation: Vec<Move>,
}

/// Final result of an alpha-beta search.
///
/// The result reflects the last fully completed iteration together with the
/// total resources consumed, including any incomplete final work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchResult {
    /// Used for returning the best legal root move, or `None` for a terminal
    /// position.
    pub best_move: Option<Move>,
    /// Used for returning the root-side centipawn or mate score.
    pub score: i32,
    /// Used for returning the last fully completed depth.
    pub depth: u8,
    /// Used for returning the greatest ply entered during the search.
    pub selective_depth: u16,
    /// Used for returning the total visited nodes, including incomplete final
    /// work.
    pub nodes: u64,
    /// Used for returning the wall-clock search duration.
    pub elapsed: Duration,
    /// Used for returning current-generation transposition-table occupancy in
    /// per mille.
    pub hashfull_per_mille: u16,
    /// Used for reporting whether a node, time, or external-stop budget ended
    /// the search.
    pub stopped: bool,
    /// Used for returning the total successful tablebase accesses; zero
    /// whenever no tablebases are configured.
    pub tb_hits: u64,
    /// Used for returning the principal variation from the last fully
    /// completed iteration.
    pub principal_variation: Vec<Move>,
}

/// Exact-key, direct-mapped cache for static evaluations within one search.
///
/// Alpha-beta treats [`SearchEvaluator::evaluate`] as position-pure during a
/// search; its mutable receiver may hold inference scratch, but the returned
/// score must depend only on the position. Every bucket carries an explicit
/// occupancy bit and the complete key, so key zero and every `i32` score remain
/// representable and cache-index collisions are detected by exact key checks.
struct EvaluationCache {
    /// Used for storing the complete position key of each direct-mapped
    /// bucket.
    keys: Vec<u64>,
    /// Used for storing the static centipawn score paired with each stored
    /// key.
    scores: Vec<i32>,
    /// Used for tracking explicit occupancy, allowing key zero to be cached
    /// safely.
    occupied: Vec<bool>,
    /// Used for masking a mixed key into the power-of-two bucket count.
    mask: usize,
}

/// Used for admitting one fully initialized heap array without invoking the
/// process-aborting allocation path.
///
/// Capacity is reserved before initialization, so `resize` cannot grow the
/// allocation and a refusal is returned through [`SearchError`].
///
/// # Arguments
///
/// * `length` - number of initialized elements required
/// * `value` - value cloned into every element
///
/// # Returns
///
/// The initialized vector on success.
///
/// # Errors
///
/// Returns [`SearchError::ResourceExhausted`] for capacity overflow or an
/// allocator refusal.
fn try_filled_vec<T: Clone>(length: usize, value: T) -> Result<Vec<T>, SearchError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| SearchError::ResourceExhausted)?;
    values.resize(length, value);
    Ok(values)
}

/// Used for admitting the per-ply principal-variation rows without an
/// infallible intermediate allocation.
///
/// Every row preserves the historical initial capacity of sixteen moves, and
/// an incomplete table is dropped before an error reaches the caller.
///
/// # Returns
///
/// One empty PV row for every searchable ply.
///
/// # Errors
///
/// Returns [`SearchError::ResourceExhausted`] when the outer table or any row
/// cannot reserve its complete initial capacity.
fn try_pv_rows() -> Result<Vec<Vec<Move>>, SearchError> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(MAX_SEARCH_PLY)
        .map_err(|_| SearchError::ResourceExhausted)?;
    for _ in 0..MAX_SEARCH_PLY {
        let mut row = Vec::new();
        row.try_reserve_exact(16)
            .map_err(|_| SearchError::ResourceExhausted)?;
        rows.push(row);
    }
    Ok(rows)
}

/// Local reference storage or one atomically shared Lazy-SMP allocation.
///
/// The wrapper lets one search implementation run against either the exact
/// single-thread [`TranspositionTable`] or an [`Arc`]-shared
/// [`SharedTranspositionTable`] without duplicating call sites.
#[derive(Debug)]
enum TableStorage {
    /// Used for exact, non-atomic storage retained by every one-thread
    /// search.
    Local(TranspositionTable),
    /// Used for one allocation shared by all workers in a parallel search
    /// pool.
    Shared(Arc<SharedTranspositionTable>),
}

impl TableStorage {
    /// Used for clearing every entry in the selected table implementation.
    fn clear(&mut self) {
        match self {
            Self::Local(table) => table.clear(),
            Self::Shared(table) => table.clear(),
        }
    }

    /// Used for probing the selected table implementation.
    ///
    /// # Arguments
    ///
    /// * `key` - full transposition key identifying the position
    ///
    /// # Returns
    ///
    /// The stored hit for `key`, or `None` when no entry matches.
    fn probe(&self, key: u64) -> Option<TtHit> {
        match self {
            Self::Local(table) => table.probe(key),
            Self::Shared(table) => table.probe(key),
        }
    }

    /// Used for storing a validated search payload in the selected
    /// implementation.
    ///
    /// # Arguments
    ///
    /// * `key` - full transposition key identifying the position
    /// * `payload` - depth, score, bound, and best-move payload to record
    /// * `generation` - age tag of the storing search
    fn store(&mut self, key: u64, payload: TtPayload, generation: u16) {
        match self {
            Self::Local(table) => table.store(key, payload, generation),
            Self::Shared(table) => table.store(key, payload, generation),
        }
    }

    /// Used for sampling current-generation occupancy in per mille.
    ///
    /// # Arguments
    ///
    /// * `generation` - age tag whose entries count as current
    ///
    /// # Returns
    ///
    /// Sampled occupancy in per mille for the given generation.
    fn hashfull_per_mille(&self, generation: u16) -> u16 {
        match self {
            Self::Local(table) => table.hashfull_per_mille(generation),
            Self::Shared(table) => table.hashfull_per_mille(generation),
        }
    }

    /// Used for returning and advancing the shared generation, if storage is
    /// shared.
    ///
    /// # Returns
    ///
    /// The newly advanced shared generation, or `None` for local storage.
    fn advance_shared_generation(&self) -> Option<u16> {
        match self {
            Self::Local(_) => None,
            Self::Shared(table) => Some(table.advance_generation()),
        }
    }

    /// Used for reading the current shared generation, if storage is shared.
    ///
    /// # Returns
    ///
    /// The current shared generation, or `None` for local storage.
    fn shared_generation(&self) -> Option<u16> {
        match self {
            Self::Local(_) => None,
            Self::Shared(table) => Some(table.generation()),
        }
    }

    /// Used for counting the buckets in either storage implementation.
    ///
    /// # Returns
    ///
    /// The number of buckets in the selected table.
    fn len(&self) -> usize {
        match self {
            Self::Local(table) => table.len(),
            Self::Shared(table) => table.len(),
        }
    }

    /// Used for testing whether this storage references an atomic shared
    /// allocation.
    ///
    /// # Returns
    ///
    /// `true` when the storage wraps a [`SharedTranspositionTable`].
    fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }
}

impl EvaluationCache {
    /// Used for fallibly allocating an empty cache with a power-of-two number
    /// of buckets.
    ///
    /// # Arguments
    ///
    /// * `entry_count` - number of direct-mapped buckets; must be a power of
    ///   two
    ///
    /// # Returns
    ///
    /// An empty cache whose buckets are all unoccupied.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::ResourceExhausted`] when any backing array
    /// cannot be admitted.
    ///
    /// # Panics
    ///
    /// Panics when `entry_count` is not a power of two.
    fn try_new(entry_count: usize) -> Result<Self, SearchError> {
        assert!(
            entry_count.is_power_of_two(),
            "evaluation-cache size must be a power of two"
        );
        Ok(Self {
            keys: try_filled_vec(entry_count, 0)?,
            scores: try_filled_vec(entry_count, 0)?,
            occupied: try_filled_vec(entry_count, false)?,
            mask: entry_count - 1,
        })
    }


    /// Used for retrieving the cached score only when the bucket contains the
    /// exact key.
    ///
    /// # Arguments
    ///
    /// * `key` - clock-free position key to look up
    ///
    /// # Returns
    ///
    /// The cached score, or `None` on an empty or colliding bucket.
    fn get(&self, key: u64) -> Option<i32> {
        let index = self.index(key);
        (self.occupied[index] && self.keys[index] == key).then_some(self.scores[index])
    }

    /// Used for replacing the key and score in the key's direct-mapped
    /// bucket.
    ///
    /// # Arguments
    ///
    /// * `key` - clock-free position key to store
    /// * `score` - static score associated with `key`
    fn put(&mut self, key: u64, score: i32) {
        let index = self.index(key);
        self.keys[index] = key;
        self.scores[index] = score;
        self.occupied[index] = true;
    }

    /// Used for mixing a 64-bit key and mapping it into the cache's bucket
    /// range.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit key to fold and mask
    ///
    /// # Returns
    ///
    /// A bucket index below the cache's power-of-two size.
    ///
    /// # Panics
    ///
    /// Panics on targets whose `usize` cannot hold a 32-bit value.
    #[inline]
    fn index(&self, key: u64) -> usize {
        let mixed = key ^ (key >> 32);
        usize::try_from(mixed & u64::from(u32::MAX)).expect("u32 fits supported usize") & self.mask
    }
}

/// Reusable alpha-beta searcher with a persistent transposition table.
///
/// One instance owns its evaluator and table storage, so repeated searches
/// reuse learned transposition entries. The evaluator type defaults to the
/// dependency-free [`Classical`] evaluation.
///
/// `FRACTIONAL_LMR` selects the late-move-reduction representation at compile
/// time and defaults to the released whole-ply formula, so every ordinary
/// construction is unchanged. Setting it selects the STR-316 candidate that
/// accumulates the same signals in `1/1024`-ply units. `LMR_BIAS_UNITS` then
/// shifts that accumulation by a fixed fraction of a ply and is meaningful
/// only for the fractional flavor; it exists because the whole-ply
/// representation could not express a sub-ply aggression step at all, which is
/// why every previous aggression retune had to move a large move population by
/// a full ply at once. Keeping both choices in the type rather than in fields
/// means no flavor pays a runtime branch and several can be instantiated side
/// by side inside one deterministic fixed-node harness process.
pub struct AlphaBeta<
    E = Classical,
    const FRACTIONAL_LMR: bool = false,
    const LMR_BIAS_UNITS: i32 = 0,
> {
    /// Used for evaluating quiet leaves and informing pruning decisions.
    evaluator: E,
    /// Used for storing the persistent transposition table reused between
    /// searches.
    table: TableStorage,
    /// Used for tagging table entries with a wrapping age advanced once per
    /// search.
    generation: u16,
    /// Used for optional Syzygy probing; `None` leaves search behavior
    /// byte-identical to a tablebase-free build.
    syzygy: Option<Prober>,
    /// Used for selecting the structural correction-history flavor of every
    /// search this instance runs.
    ///
    /// [`CorrectionHistoryMode::Off`] is the released behavior and leaves the
    /// search byte-identical to a build without correction history, so the two
    /// flavors can be instantiated side by side in one deterministic
    /// fixed-node harness process.
    research: ResearchSearch,
}

impl AlphaBeta<Classical> {
    /// Used for creating the standard dependency-free classical searcher.
    ///
    /// # Returns
    ///
    /// A searcher over the [`Classical`] evaluator with the default
    /// transposition-table capacity.
    #[must_use]
    pub fn classical() -> Self {
        Self::new(Classical)
    }
}

impl Default for AlphaBeta<Classical> {
    /// Used for creating the default classical alpha-beta searcher.
    ///
    /// # Returns
    ///
    /// The same searcher as [`AlphaBeta::classical`].
    fn default() -> Self {
        Self::classical()
    }
}

impl<E: SearchEvaluator, const LMR_BIAS_UNITS: i32> AlphaBeta<E, true, LMR_BIAS_UNITS> {
    /// Used for creating the STR-316 fractional late-move-reduction searcher.
    ///
    /// This is the only constructor of the candidate flavor, so the ordinary
    /// engine cannot reach it by accident and the two flavors can still be
    /// instantiated side by side inside one deterministic research process.
    /// Everything except the reduction representation is shared with the
    /// released searcher.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator driving leaf scores and priors
    /// * `entries` - number of local transposition-table entries
    ///
    /// # Returns
    ///
    /// A candidate-flavor searcher owning a freshly allocated local table.
    #[doc(hidden)]
    #[must_use]
    pub fn research_fractional_lmr(evaluator: E, entries: usize) -> Self {
        Self::from_local_entries(evaluator, entries)
    }
}

impl<E: SearchEvaluator> AlphaBeta<E, false> {
    /// Used for creating the `STR-20260806-331` correction-history searcher.
    ///
    /// The flavor is a field rather than a type parameter because the released
    /// mode allocates no tables and takes one predictable branch per node,
    /// while a type parameter would have to be threaded through every search
    /// entry point for a mechanism that is expected to become the default if
    /// it screens positively.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator driving leaf scores and priors
    /// * `entries` - number of local transposition-table entries
    /// * `mode` - correction-history flavor to apply
    ///
    /// # Returns
    ///
    /// A searcher owning a freshly allocated local table and running the
    /// requested correction flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_correction_history(
        evaluator: E,
        entries: usize,
        mode: CorrectionHistoryMode,
    ) -> Self {
        let mut searcher = Self::from_local_entries(evaluator, entries);
        searcher.research.correction = mode;
        searcher
    }

    /// Used for creating the `STR-20260806-335` evaluation-independent
    /// late-move-pruning searcher.
    ///
    /// Janus's released late-move prune fires only when the static evaluation
    /// is already far below alpha, which makes an eval-independent mechanism
    /// depend on the one component `INFRA-20260805-318` measured as the
    /// binding constraint. This flavor keeps the move-count and history
    /// conditions and drops the evaluation precondition.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator driving leaf scores and priors
    /// * `entries` - number of local transposition-table entries
    ///
    /// # Returns
    ///
    /// A searcher owning a freshly allocated local table and pruning without
    /// consulting the static evaluation.
    #[doc(hidden)]
    #[must_use]
    pub fn research_eval_free_late_move_pruning(evaluator: E, entries: usize) -> Self {
        let mut searcher = Self::from_local_entries(evaluator, entries);
        searcher.research.eval_free_late_move_pruning = true;
        searcher
    }
}

impl<E: SearchEvaluator> AlphaBeta<E, false> {
    /// Used for creating a searcher with the default transposition-table
    /// capacity.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator driving leaf scores and priors
    ///
    /// # Returns
    ///
    /// A searcher with `DEFAULT_TT_ENTRIES` local table entries.
    #[must_use]
    pub fn new(evaluator: E) -> Self {
        Self::with_tt_entries(evaluator, DEFAULT_TT_ENTRIES)
    }

    /// Used for creating a searcher with a power-of-two number of TT entries.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator driving leaf scores and priors
    /// * `entries` - number of local transposition-table entries
    ///
    /// # Returns
    ///
    /// A searcher owning a freshly allocated local table.
    #[must_use]
    pub fn with_tt_entries(evaluator: E, entries: usize) -> Self {
        Self::from_local_entries(evaluator, entries)
    }

    /// Used for creating a searcher around a caller-allocated local table.
    ///
    /// Callers that must tolerate an allocation failure allocate through
    /// [`TranspositionTable::try_new`] and hand the result here, rather than
    /// letting [`Self::with_tt_entries`] allocate infallibly.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator driving leaf scores and priors
    /// * `table` - already-allocated local transposition table
    ///
    /// # Returns
    ///
    /// A searcher owning the supplied table.
    #[must_use]
    pub fn with_table(evaluator: E, table: TranspositionTable) -> Self {
        Self {
            evaluator,
            table: TableStorage::Local(table),
            generation: 0,
            syzygy: None,
            research: ResearchSearch::default(),
        }
    }

    /// Used for creating a searcher backed by a caller-owned shared
    /// transposition table.
    ///
    /// Every helper in one Lazy-SMP pool should receive a clone of the same
    /// [`Arc`]. Before starting the workers, the coordinator must call
    /// [`Self::advance_shared_generation`] exactly once on one pool member.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator driving leaf scores and priors
    /// * `table` - shared allocation cloned into every pool member
    ///
    /// # Returns
    ///
    /// A searcher publishing and probing through the shared table.
    #[must_use]
    pub fn with_shared_tt(evaluator: E, table: Arc<SharedTranspositionTable>) -> Self {
        Self {
            evaluator,
            table: TableStorage::Shared(table),
            generation: 0,
            syzygy: None,
            research: ResearchSearch::default(),
        }
    }
}

impl<E: SearchEvaluator, const FRACTIONAL_LMR: bool, const LMR_BIAS_UNITS: i32>
    AlphaBeta<E, FRACTIONAL_LMR, LMR_BIAS_UNITS>
{
    /// Used for building either reduction flavor around a fresh local table.
    ///
    /// Shared by the released constructors and by
    /// [`AlphaBeta::research_fractional_lmr`] so both flavors start from
    /// identical storage, generation, and tablebase state.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator driving leaf scores and priors
    /// * `entries` - number of local transposition-table entries
    ///
    /// # Returns
    ///
    /// A searcher owning a freshly allocated local table.
    fn from_local_entries(evaluator: E, entries: usize) -> Self {
        Self {
            evaluator,
            table: TableStorage::Local(TranspositionTable::new(entries)),
            generation: 0,
            syzygy: None,
            research: ResearchSearch::default(),
        }
    }

    /// Used for installing or removing the optional Syzygy tablebases.
    ///
    /// Each worker builds its own [`Prober`] around the shared registry so
    /// block caching stays worker-local and deterministic. Passing `None`
    /// disables probing and restores tablebase-free behavior exactly. The
    /// coordinator must only call this between searches.
    ///
    /// # Arguments
    ///
    /// * `config` - shared registry and clamped UCI option values, or
    ///   `None` to disable probing
    pub fn set_syzygy(&mut self, config: Option<&SyzygyConfig>) {
        self.syzygy = config.map(Prober::new);
    }

    /// Used for advancing a shared table's generation before one parallel
    /// search.
    ///
    /// Returns `None` for the exact local one-thread implementation. Every
    /// worker sharing the allocation observes the returned generation when its
    /// search begins.
    ///
    /// # Returns
    ///
    /// The newly advanced shared generation, or `None` for local storage.
    #[must_use]
    pub fn advance_shared_generation(&self) -> Option<u16> {
        self.table.advance_shared_generation()
    }

    /// Used for sampling occupancy for this searcher's current generation.
    ///
    /// # Returns
    ///
    /// Sampled transposition-table occupancy in per mille.
    #[must_use]
    pub fn current_hashfull_per_mille(&self) -> u16 {
        self.table.hashfull_per_mille(self.generation)
    }

    /// Used for counting the transposition-table buckets visible to this
    /// worker.
    ///
    /// # Returns
    ///
    /// The bucket count of the local or shared table.
    #[must_use]
    pub fn transposition_entries(&self) -> usize {
        self.table.len()
    }

    /// Used for testing whether this worker uses an atomic shared table
    /// allocation.
    ///
    /// # Returns
    ///
    /// `true` when the searcher probes a shared Lazy-SMP table.
    #[must_use]
    pub fn uses_shared_transposition_table(&self) -> bool {
        self.table.is_shared()
    }

    /// Used for clearing all position-dependent search state.
    ///
    /// The transposition table is emptied and the generation counter returns
    /// to zero, matching a freshly constructed searcher.
    pub fn clear(&mut self) {
        self.table.clear();
        self.generation = 0;
    }

    /// Used for selecting the table generation of a newly started search.
    ///
    /// Local storage preserves the historical wrapping counter exactly. Shared
    /// storage consumes the coordinator-assigned generation without advancing
    /// it independently in each helper.
    fn begin_search_generation(&mut self) {
        let local_next = self.generation.wrapping_add(1);
        self.generation = self.table.shared_generation().unwrap_or(local_next);
    }

    /// Used for searching without an externally shared stop token or
    /// iteration callback.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `limits` - depth, node, and time budgets for the search
    ///
    /// # Returns
    ///
    /// The completed search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] when `limits` is inconsistent or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    pub fn search(
        &mut self,
        position: &Position,
        limits: SearchLimits,
    ) -> Result<SearchResult, SearchError> {
        let stop = AtomicBool::new(false);
        self.search_with_history_and_observer(position, &[], limits, &stop, |_| {})
    }

    /// Used for searching with repetition keys for every game position
    /// preceding `position`.
    ///
    /// The keys must be chronological and use [`Position::key`]. Only entries
    /// inside the root's reversible halfmove window can affect adjudication.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    ///
    /// # Returns
    ///
    /// The completed search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] when `limits` is inconsistent or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    pub fn search_with_history(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
    ) -> Result<SearchResult, SearchError> {
        let stop = AtomicBool::new(false);
        self.search_with_history_and_observer(position, history, limits, &stop, |_| {})
    }

    /// Used for searching while polling a caller-owned atomic stop token.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `limits` - depth, node, and time budgets for the search
    /// * `stop` - cancellation flag polled at safe search boundaries
    ///
    /// # Returns
    ///
    /// The completed search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] when `limits` is inconsistent or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    pub fn search_with_stop(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        stop: &AtomicBool,
    ) -> Result<SearchResult, SearchError> {
        self.search_with_history_and_observer(position, &[], limits, stop, |_| {})
    }

    /// Used for searching and reporting every fully completed depth to
    /// `observer`.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `limits` - depth, node, and time budgets for the search
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `observer` - callback receiving one report per completed depth
    ///
    /// # Returns
    ///
    /// The completed search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] when `limits` is inconsistent or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    #[allow(clippy::too_many_lines)]
    pub fn search_with_observer<F>(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        stop: &AtomicBool,
        observer: F,
    ) -> Result<SearchResult, SearchError>
    where
        F: FnMut(&SearchInfo),
    {
        self.search_with_history_and_observer(position, &[], limits, stop, observer)
    }

    /// Used for searching with game history and reporting every completed
    /// depth.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `observer` - callback receiving one report per completed depth
    ///
    /// # Returns
    ///
    /// The completed search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] when `limits` is inconsistent or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn search_with_history_and_observer<F>(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
        stop: &AtomicBool,
        observer: F,
    ) -> Result<SearchResult, SearchError>
    where
        F: FnMut(&SearchInfo),
    {
        self.search_with_observer_worker(position, history, limits, stop, 0, None, observer)
    }

    /// Used for searching as an independent parallel helper with a
    /// diversified iterative-deepening schedule and no iteration callback.
    ///
    /// `worker_index` is one-based. Helpers skip deterministic depth bands so a
    /// parallel pool covers different horizons instead of repeating the main
    /// worker exactly. Shared-table helpers publish bounds but never replace
    /// the main worker's authoritative root result.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `limits` - depth, node, and time budgets for the search
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `worker_index` - one-based helper index selecting a skip schedule
    ///
    /// # Returns
    ///
    /// The completed helper search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] when `limits` is inconsistent or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    pub fn search_as_helper(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        stop: &AtomicBool,
        worker_index: usize,
    ) -> Result<SearchResult, SearchError> {
        self.search_as_helper_with_history(position, &[], limits, stop, worker_index)
    }

    /// Used for searching as a diversified helper with game repetition
    /// history.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `worker_index` - one-based helper index selecting a skip schedule
    ///
    /// # Returns
    ///
    /// The completed helper search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] when `limits` is inconsistent or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    pub fn search_as_helper_with_history(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
        stop: &AtomicBool,
        worker_index: usize,
    ) -> Result<SearchResult, SearchError> {
        self.search_with_observer_worker(
            position,
            history,
            limits,
            stop,
            worker_index.max(1),
            None,
            |_| {},
        )
    }

    /// Used for searching only moves in `root_moves` without an external stop
    /// token.
    ///
    /// Whitelist order and duplicates do not affect search order: legal moves
    /// retain the deterministic order produced by `janus-core`.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `limits` - depth, node, and time budgets for the search
    /// * `root_moves` - whitelist restricting the searched root moves
    ///
    /// # Returns
    ///
    /// The completed restricted search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] for inconsistent limits,
    /// [`SearchError::EmptyRootMoves`] for an empty whitelist, or
    /// [`SearchError::IllegalRootMove`] when a listed move is not legal, or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    pub fn search_root_moves(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        root_moves: &[Move],
    ) -> Result<SearchResult, SearchError> {
        let stop = AtomicBool::new(false);
        self.search_root_moves_with_observer(position, limits, root_moves, &stop, |_| {})
    }

    /// Used for searching only moves in `root_moves` and reporting completed
    /// iterations.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `limits` - depth, node, and time budgets for the search
    /// * `root_moves` - whitelist restricting the searched root moves
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `observer` - callback receiving one report per completed depth
    ///
    /// # Returns
    ///
    /// The completed restricted search result.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::search_root_moves`].
    pub fn search_root_moves_with_observer<F>(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        root_moves: &[Move],
        stop: &AtomicBool,
        observer: F,
    ) -> Result<SearchResult, SearchError>
    where
        F: FnMut(&SearchInfo),
    {
        self.search_root_moves_with_history_and_observer(
            position,
            &[],
            limits,
            root_moves,
            stop,
            observer,
        )
    }

    /// Used for searching a root whitelist with game history and iteration
    /// reporting.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `root_moves` - whitelist restricting the searched root moves
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `observer` - callback receiving one report per completed depth
    ///
    /// # Returns
    ///
    /// The completed restricted search result.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::search_root_moves`].
    #[allow(clippy::too_many_arguments)]
    pub fn search_root_moves_with_history_and_observer<F>(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
        root_moves: &[Move],
        stop: &AtomicBool,
        observer: F,
    ) -> Result<SearchResult, SearchError>
    where
        F: FnMut(&SearchInfo),
    {
        self.search_with_observer_worker(
            position,
            history,
            limits,
            stop,
            0,
            Some(root_moves),
            observer,
        )
    }

    /// Used for searching only moves in `root_moves` as an independent
    /// parallel helper.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `limits` - depth, node, and time budgets for the search
    /// * `root_moves` - whitelist restricting the searched root moves
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `worker_index` - one-based helper index selecting a skip schedule
    ///
    /// # Returns
    ///
    /// The completed restricted helper search result.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::search_root_moves`].
    pub fn search_root_moves_as_helper(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        root_moves: &[Move],
        stop: &AtomicBool,
        worker_index: usize,
    ) -> Result<SearchResult, SearchError> {
        self.search_root_moves_as_helper_with_history(
            position,
            &[],
            limits,
            root_moves,
            stop,
            worker_index,
        )
    }

    /// Used for searching a root whitelist as a helper with game repetition
    /// history.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `root_moves` - whitelist restricting the searched root moves
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `worker_index` - one-based helper index selecting a skip schedule
    ///
    /// # Returns
    ///
    /// The completed restricted helper search result.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::search_root_moves`].
    pub fn search_root_moves_as_helper_with_history(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
        root_moves: &[Move],
        stop: &AtomicBool,
        worker_index: usize,
    ) -> Result<SearchResult, SearchError> {
        self.search_with_observer_worker(
            position,
            history,
            limits,
            stop,
            worker_index.max(1),
            Some(root_moves),
            |_| {},
        )
    }

    /// Used for searching a complete, rank-ordered `MultiPV` batch at every
    /// finished depth.
    ///
    /// All lines share one node counter, clock, stop token, and transposition
    /// table. A depth is reported only after every requested line at that depth
    /// finishes, so observers never receive a mixture of old and new ranks.
    /// `requested_root_moves` uses the same deterministic whitelist semantics as
    /// [`Self::search_root_moves`]. When more lines are requested than there are
    /// candidate moves, the result is capped to the available move count.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `limits` - depth, node, and time budgets for the search
    /// * `requested_root_moves` - optional whitelist restricting root moves
    /// * `multi_pv` - number of ranked lines to search per depth
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `observer` - callback receiving one complete ranked batch per depth
    ///
    /// # Returns
    ///
    /// One completed search result per ranked line.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidMultiPv`] when `multi_pv` is zero and the
    /// same failures as [`Self::search_root_moves`] for limits, resources, and an
    /// optional root whitelist.
    pub fn search_multi_pv_with_observer<F>(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        requested_root_moves: Option<&[Move]>,
        multi_pv: usize,
        stop: &AtomicBool,
        observer: F,
    ) -> Result<Vec<SearchResult>, SearchError>
    where
        F: FnMut(&[SearchInfo]),
    {
        self.search_multi_pv_with_history_and_observer(
            position,
            &[],
            limits,
            requested_root_moves,
            multi_pv,
            stop,
            observer,
        )
    }

    /// Used for searching a complete ranked batch with game repetition
    /// history.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `requested_root_moves` - optional whitelist restricting root moves
    /// * `multi_pv` - number of ranked lines to search per depth
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `observer` - callback receiving one complete ranked batch per depth
    ///
    /// # Returns
    ///
    /// One completed search result per ranked line.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::search_multi_pv_with_observer`].
    #[allow(clippy::too_many_arguments)]
    pub fn search_multi_pv_with_history_and_observer<F>(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
        requested_root_moves: Option<&[Move]>,
        multi_pv: usize,
        stop: &AtomicBool,
        mut observer: F,
    ) -> Result<Vec<SearchResult>, SearchError>
    where
        F: FnMut(&[SearchInfo]),
    {
        if multi_pv == 0 {
            return Err(SearchError::InvalidMultiPv);
        }
        if multi_pv == 1 {
            return self
                .search_with_observer_worker(
                    position,
                    history,
                    limits,
                    stop,
                    0,
                    requested_root_moves,
                    |info| observer(std::slice::from_ref(info)),
                )
                .map(|result| vec![result]);
        }
        self.search_multi_pv_with_observer_worker(
            position,
            history,
            limits,
            stop,
            0,
            requested_root_moves,
            multi_pv,
            observer,
        )
    }

    /// Used for searching a ranked helper batch with game repetition
    /// history.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `requested_root_moves` - optional whitelist restricting root moves
    /// * `multi_pv` - number of ranked lines to search per depth
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `worker_index` - one-based helper index selecting a skip schedule
    ///
    /// # Returns
    ///
    /// One completed search result per ranked line.
    ///
    /// # Errors
    ///
    /// Returns the same failures as
    /// [`Self::search_multi_pv_with_observer`].
    #[allow(clippy::too_many_arguments)]
    pub fn search_multi_pv_as_helper_with_history(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
        requested_root_moves: Option<&[Move]>,
        multi_pv: usize,
        stop: &AtomicBool,
        worker_index: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        if multi_pv == 0 {
            return Err(SearchError::InvalidMultiPv);
        }
        if multi_pv == 1 {
            return self
                .search_with_observer_worker(
                    position,
                    history,
                    limits,
                    stop,
                    worker_index.max(1),
                    requested_root_moves,
                    |_| {},
                )
                .map(|result| vec![result]);
        }
        self.search_multi_pv_with_observer_worker(
            position,
            history,
            limits,
            stop,
            worker_index.max(1),
            requested_root_moves,
            multi_pv,
            |_| {},
        )
    }

    /// Used for implementing a complete `MultiPV` iterative-deepening search
    /// for one worker.
    ///
    /// Each reported depth is a transactional batch: every rank is searched
    /// from the same root context before the observer sees any line from that
    /// depth. Later ranks exclude the root moves already selected above them.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `worker_index` - zero for the main worker, one-based for helpers
    /// * `requested_root_moves` - optional whitelist restricting root moves
    /// * `multi_pv` - number of ranked lines to search per depth
    /// * `observer` - callback receiving one complete ranked batch per depth
    ///
    /// # Returns
    ///
    /// One completed search result per ranked line.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] for inconsistent limits, the
    /// root-whitelist failures produced by [`select_root_moves`], or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn search_multi_pv_with_observer_worker<F>(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
        stop: &AtomicBool,
        worker_index: usize,
        requested_root_moves: Option<&[Move]>,
        multi_pv: usize,
        mut observer: F,
    ) -> Result<Vec<SearchResult>, SearchError>
    where
        F: FnMut(&[SearchInfo]),
    {
        if !limits.is_valid() {
            return Err(SearchError::InvalidLimits);
        }

        self.begin_search_generation();
        let started = Instant::now();
        let legal_moves = select_root_moves(position.legal_moves(), requested_root_moves)?;
        if legal_moves.is_empty() {
            return Ok(vec![SearchResult {
                best_move: None,
                score: terminal_score(position.in_check(position.side_to_move()), 0),
                depth: 0,
                selective_depth: 0,
                nodes: 0,
                elapsed: started.elapsed(),
                hashfull_per_mille: self.table.hashfull_per_mille(self.generation),
                stopped: false,
                tb_hits: 0,
                principal_variation: Vec::new(),
            }]);
        }

        let single_reply_root = legal_moves.len() == 1;
        let line_count = multi_pv.min(legal_moves.len());
        let mut completed_lines: Vec<(Move, i32, Vec<Move>)> = legal_moves
            .iter()
            .copied()
            .take(line_count)
            .map(|mv| (mv, 0, vec![mv]))
            .collect();
        let mut previous_scores = vec![0_i32; line_count];
        let mut completed_depth = 0_u8;
        let mut stable_iterations = 0_u8;
        let max_depth = limits.depth;
        self.evaluator
            .begin_search(position, MAX_SEARCH_PLY)
            .map_err(|_| SearchError::ResourceExhausted)?;
        // `MultiPV` keeps every requested line searchable, so the root is
        // never tablebase-filtered here; interior probing stays active.
        let mut tb_cardinality = 0;
        if let Some(prober) = self.syzygy.as_mut() {
            prober.reset_hits();
            tb_cardinality = prober.cardinality();
        }
        let mut context = SearchContext::<E, FRACTIONAL_LMR, LMR_BIAS_UNITS>::try_new(
            &mut self.evaluator,
            &mut self.table,
            self.generation,
            history,
            limits,
            stop,
            started,
            self.syzygy.as_mut(),
            tb_cardinality,
            self.research,
        )?;
        context.single_reply_root = single_reply_root;

        'iterations: for depth in 1..=max_depth {
            if context.should_stop_before_iteration(stable_iterations) {
                break;
            }
            if worker_index > 0 && helper_skips_iteration(worker_index, depth) {
                continue;
            }

            let mut remaining = legal_moves.clone();
            let mut iteration_lines = Vec::with_capacity(line_count);
            for (rank, previous_score) in previous_scores.iter().copied().enumerate() {
                context.root_moves = Some(remaining.clone().into_boxed_slice());
                let fallback = remaining[0];
                let mut window = ASPIRATION_WINDOW;
                let (mut alpha, mut beta) =
                    if depth >= ASPIRATION_START_DEPTH && completed_depth > 0 {
                        (
                            previous_score.saturating_sub(window).max(-INFINITY),
                            previous_score.saturating_add(window).min(INFINITY),
                        )
                    } else {
                        (-INFINITY, INFINITY)
                    };

                let completed = loop {
                    context.clear_root_pv();
                    let mut working = position.clone();
                    let Some(score) = context.negamax(
                        &mut working,
                        i32::from(depth),
                        alpha,
                        beta,
                        0,
                        true,
                        false,
                    ) else {
                        break None;
                    };

                    if score <= alpha && alpha > -INFINITY {
                        window = window.saturating_mul(2).min(INFINITY);
                        alpha = score.saturating_sub(window).max(-INFINITY);
                        beta = previous_score.saturating_add(window).min(INFINITY);
                        continue;
                    }
                    if score >= beta && beta < INFINITY {
                        window = window.saturating_mul(2).min(INFINITY);
                        alpha = previous_score.saturating_sub(window).max(-INFINITY);
                        beta = score.saturating_add(window).min(INFINITY);
                        continue;
                    }
                    let pv = context.root_pv();
                    let iteration_move = pv.first().copied().unwrap_or(fallback);
                    break Some((iteration_move, score, pv));
                };

                let Some((iteration_move, score, pv)) = completed else {
                    break 'iterations;
                };
                debug_assert!(remaining.contains(&iteration_move));
                iteration_lines.push((iteration_move, score, pv));
                remaining.retain(|mv| *mv != iteration_move);
                debug_assert_eq!(iteration_lines.len(), rank + 1);
            }

            let previous_best = completed_lines[0].0;
            stable_iterations = if iteration_lines[0].0 == previous_best {
                stable_iterations.saturating_add(1)
            } else {
                0
            };
            for (target, (_, score, _)) in previous_scores.iter_mut().zip(&iteration_lines) {
                *target = *score;
            }
            completed_depth = depth;
            completed_lines = iteration_lines;

            let hashfull = context.table.hashfull_per_mille(context.generation);
            let elapsed = started.elapsed();
            let infos: Vec<SearchInfo> = completed_lines
                .iter()
                .map(|(_, score, pv)| SearchInfo {
                    depth,
                    selective_depth: context.selective_depth,
                    score: *score,
                    nodes: context.nodes,
                    elapsed,
                    hashfull_per_mille: hashfull,
                    tb_hits: context.tb_hits(),
                    principal_variation: pv.clone(),
                })
                .collect();
            observer(&infos);
        }

        let hashfull = context.table.hashfull_per_mille(context.generation);
        let elapsed = started.elapsed();
        let tb_hits = context.tb_hits();
        Ok(completed_lines
            .into_iter()
            .map(|(best_move, score, principal_variation)| SearchResult {
                best_move: Some(best_move),
                score,
                depth: completed_depth,
                selective_depth: context.selective_depth,
                nodes: context.nodes,
                elapsed,
                hashfull_per_mille: hashfull,
                stopped: context.stopped,
                tb_hits,
                principal_variation,
            })
            .collect())
    }

    /// Used for implementing one principal-variation search for a main or
    /// helper worker.
    ///
    /// A zero `worker_index` uses every depth and reports completed iterations;
    /// positive indices follow deterministic helper depth schedules.
    ///
    /// # Arguments
    ///
    /// * `position` - root position to search
    /// * `history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `stop` - cancellation flag polled at safe search boundaries
    /// * `worker_index` - zero for the main worker, one-based for helpers
    /// * `requested_root_moves` - optional whitelist restricting root moves
    /// * `observer` - callback receiving one report per completed depth
    ///
    /// # Returns
    ///
    /// The completed search result.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidLimits`] for inconsistent limits, the
    /// root-whitelist failures produced by [`select_root_moves`], or
    /// [`SearchError::ResourceExhausted`] when workspace admission fails.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn search_with_observer_worker<F>(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
        stop: &AtomicBool,
        worker_index: usize,
        requested_root_moves: Option<&[Move]>,
        mut observer: F,
    ) -> Result<SearchResult, SearchError>
    where
        F: FnMut(&SearchInfo),
    {
        if !limits.is_valid() {
            return Err(SearchError::InvalidLimits);
        }

        self.begin_search_generation();
        let started = Instant::now();
        let mut legal_moves = select_root_moves(position.legal_moves(), requested_root_moves)?;
        if legal_moves.is_empty() {
            return Ok(SearchResult {
                best_move: None,
                score: terminal_score(position.in_check(position.side_to_move()), 0),
                depth: 0,
                selective_depth: 0,
                nodes: 0,
                elapsed: started.elapsed(),
                hashfull_per_mille: self.table.hashfull_per_mille(self.generation),
                stopped: false,
                tb_hits: 0,
                principal_variation: Vec::new(),
            });
        }

        // Time management's single-reply detection precedes tablebase
        // filtering: it reflects the position's own legal-move count.
        let single_reply_root = legal_moves.len() == 1;

        // Root tablebase ranking: inside the tables, restrict the root to
        // the outcome-preserving moves and adjust interior probing.
        let mut tb_cardinality = 0;
        let mut tb_filtered = false;
        if let Some(prober) = self.syzygy.as_mut() {
            prober.reset_hits();
            tb_cardinality = prober.cardinality();
            if let Some(filter) = prober.filter_root_moves(position, &legal_moves) {
                legal_moves = filter.moves;
                tb_cardinality = filter.search_cardinality;
                tb_filtered = true;
            }
        }

        let fallback = legal_moves[0];
        let root_probe_allowed = !stop.load(Ordering::Relaxed) && !limits.hard_expired(started);
        if let Some(mating_move) = root_probe_allowed
            .then(|| find_mate_in_one(position, &legal_moves))
            .flatten()
        {
            return Ok(SearchResult {
                best_move: Some(mating_move),
                score: MATE_SCORE - 1,
                depth: 1,
                selective_depth: 1,
                nodes: 0,
                elapsed: started.elapsed(),
                hashfull_per_mille: self.table.hashfull_per_mille(self.generation),
                stopped: false,
                tb_hits: self.syzygy.as_ref().map_or(0, Prober::hits),
                principal_variation: vec![mating_move],
            });
        }
        let mut best_move = fallback;
        let mut best_score: i32 = 0;
        let mut completed_depth = 0;
        let mut completed_pv = vec![fallback];
        let mut stable_iterations = 0_u8;
        let max_depth = limits.depth;
        self.evaluator
            .begin_search(position, MAX_SEARCH_PLY)
            .map_err(|_| SearchError::ResourceExhausted)?;
        let mut context = SearchContext::<E, FRACTIONAL_LMR, LMR_BIAS_UNITS>::try_new(
            &mut self.evaluator,
            &mut self.table,
            self.generation,
            history,
            limits,
            stop,
            started,
            self.syzygy.as_mut(),
            tb_cardinality,
            self.research,
        )?;
        context.single_reply_root = single_reply_root;
        if requested_root_moves.is_some() || tb_filtered {
            context.root_moves = Some(legal_moves.into_boxed_slice());
        }

        'iterations: for depth in 1..=max_depth {
            if context.should_stop_before_iteration(stable_iterations) {
                break;
            }
            if worker_index > 0 && helper_skips_iteration(worker_index, depth) {
                continue;
            }

            let mut window = ASPIRATION_WINDOW;
            let (mut alpha, mut beta) = if depth >= ASPIRATION_START_DEPTH && completed_depth > 0 {
                (
                    best_score.saturating_sub(window).max(-INFINITY),
                    best_score.saturating_add(window).min(INFINITY),
                )
            } else {
                (-INFINITY, INFINITY)
            };

            let completed = loop {
                context.clear_root_pv();
                let mut working = position.clone();
                let Some(score) =
                    context.negamax(&mut working, i32::from(depth), alpha, beta, 0, true, false)
                else {
                    break None;
                };

                if score <= alpha && alpha > -INFINITY {
                    window = window.saturating_mul(2).min(INFINITY);
                    alpha = score.saturating_sub(window).max(-INFINITY);
                    beta = best_score.saturating_add(window).min(INFINITY);
                    continue;
                }
                if score >= beta && beta < INFINITY {
                    window = window.saturating_mul(2).min(INFINITY);
                    alpha = best_score.saturating_sub(window).max(-INFINITY);
                    beta = score.saturating_add(window).min(INFINITY);
                    continue;
                }
                let pv = context.root_pv();
                let iteration_move = pv.first().copied().unwrap_or(best_move);
                break Some((score, iteration_move, pv));
            };

            let Some((score, iteration_move, pv)) = completed else {
                break 'iterations;
            };
            stable_iterations = if iteration_move == best_move {
                stable_iterations.saturating_add(1)
            } else {
                0
            };
            best_move = iteration_move;
            best_score = score;
            completed_depth = depth;
            completed_pv = pv;
            observer(&SearchInfo {
                depth,
                selective_depth: context.selective_depth,
                score,
                nodes: context.nodes,
                elapsed: started.elapsed(),
                hashfull_per_mille: context.table.hashfull_per_mille(context.generation),
                tb_hits: context.tb_hits(),
                principal_variation: completed_pv.clone(),
            });
        }

        Ok(SearchResult {
            best_move: Some(best_move),
            score: best_score,
            depth: completed_depth,
            selective_depth: context.selective_depth,
            nodes: context.nodes,
            elapsed: started.elapsed(),
            hashfull_per_mille: context.table.hashfull_per_mille(context.generation),
            stopped: context.stopped,
            tb_hits: context.tb_hits(),
            principal_variation: completed_pv,
        })
    }
}

/// Used for applying an optional whitelist while retaining canonical
/// legal-move order.
///
/// # Arguments
///
/// * `legal_moves` - canonical legal moves generated for the root
/// * `requested` - optional whitelist restricting the root moves
///
/// # Returns
///
/// The legal moves, filtered to the whitelist when one is present.
///
/// # Errors
///
/// Returns [`SearchError::EmptyRootMoves`] for an empty whitelist and
/// [`SearchError::IllegalRootMove`] when a listed move is not legal.
fn select_root_moves(
    mut legal_moves: Vec<Move>,
    requested: Option<&[Move]>,
) -> Result<Vec<Move>, SearchError> {
    let Some(requested) = requested else {
        return Ok(legal_moves);
    };
    if requested.is_empty() {
        return Err(SearchError::EmptyRootMoves);
    }
    if let Some(illegal) = requested
        .iter()
        .copied()
        .find(|mv| !legal_moves.contains(mv))
    {
        return Err(SearchError::IllegalRootMove(illegal));
    }
    legal_moves.retain(|mv| requested.contains(mv));
    Ok(legal_moves)
}

/// Used for deciding whether a diversified helper omits the given iterative
/// depth.
///
/// The decision follows the deterministic [`HELPER_SKIP_SIZE`] and
/// [`HELPER_SKIP_PHASE`] schedules, cycling over helper indices.
///
/// # Arguments
///
/// * `worker_index` - one-based helper index
/// * `depth` - iterative-deepening depth under consideration
///
/// # Returns
///
/// `true` when the helper skips this depth.
fn helper_skips_iteration(worker_index: usize, depth: u8) -> bool {
    debug_assert!(worker_index > 0);
    let index = (worker_index - 1) % HELPER_SKIP_SIZE.len();
    ((usize::from(depth) + HELPER_SKIP_PHASE[index]) / HELPER_SKIP_SIZE[index]) % 2 == 1
}

/// Mutable state shared by every node and `MultiPV` line in one search.
///
/// The context borrows the reusable searcher's evaluator and transposition
/// storage and adds per-search ordering, history, and path-tracking state
/// sized by [`MAX_SEARCH_PLY`].
///
/// `FRACTIONAL_LMR` is inherited from the owning [`AlphaBeta`] so the
/// late-move-reduction flavor is fixed at compile time for the whole search
/// tree; it defaults to the released whole-ply formula.
struct SearchContext<'a, E, const FRACTIONAL_LMR: bool = false, const LMR_BIAS_UNITS: i32 = 0> {
    /// Used for evaluating positions; borrowed from the reusable searcher.
    evaluator: &'a mut E,
    /// Used for probing and storing bounds; borrowed from the reusable
    /// searcher.
    table: &'a mut TableStorage,
    /// Used for tagging table stores with the current search's generation.
    generation: u16,
    /// Used for holding the chronological repetition keys preceding the
    /// searched root.
    game_history: &'a [u64],
    /// Used for holding the depth, node, and time budgets governing this
    /// search.
    limits: SearchLimits,
    /// Used for polling the caller-owned cancellation flag at safe search
    /// boundaries.
    stop: &'a AtomicBool,
    /// Used for anchoring all elapsed-time decisions to one monotonic start
    /// instant.
    started: Instant,
    /// Used for counting the alpha-beta and quiescence nodes entered so far.
    nodes: u64,
    /// Used for tracking the greatest ply entered during the current search.
    selective_depth: u16,
    /// Used for recording whether a resource limit or external cancellation
    /// interrupted work.
    stopped: bool,
    /// Used for bounding the earliest ply at which null-move pruning is
    /// allowed during verification.
    ///
    /// Zero means that no verification search is active. A positive boundary
    /// suppresses recursive null pruning near the verification root while
    /// permitting it again in sufficiently deep descendants.
    null_move_pruning_min_ply: usize,
    /// Used for probing Syzygy tables inside the tree; borrowed from the
    /// reusable searcher, `None` when tablebases are disabled.
    syzygy: Option<&'a mut Prober>,
    /// Used for gating interior probes by piece count after root
    /// adjustment; zero disables interior probing entirely.
    tb_cardinality: usize,
    /// Used for caching static evaluator results by exact key within one
    /// search.
    eval_cache: EvaluationCache,
    /// Used for recording whether evaluator lifecycle hooks are active for
    /// this root.
    incremental_evaluator: bool,
    /// Used for storing butterfly history indexed by source and destination
    /// squares.
    history: Vec<i32>,
    /// Used for storing capture ordering history indexed by colored mover,
    /// target, and victim kind.
    capture_history: Vec<i16>,
    /// Used for storing the one-ply piece-square continuation-history table.
    continuation_one: Vec<i16>,
    /// Used for storing the two-ply piece-square continuation-history table.
    continuation_two: Vec<i16>,
    /// Used for selecting the research flavors active in this search.
    research: ResearchSearch,
    /// Used for storing the structural correction tables, laid out as side to
    /// move, then table, then structural cell.
    ///
    /// Empty when [`CorrectionHistoryMode::Off`] is selected, so the released
    /// searcher allocates nothing and reads nothing.
    correction_history: Vec<CorrectionEntry>,
    /// Used for retaining two quiet cutoff moves at each search ply.
    killers: [[u16; 2]; MAX_SEARCH_PLY],
    /// Used for holding the transposition move excluded by an active
    /// singular verification at each ply, or [`NO_MOVE`].
    ///
    /// The state is one slot per ply rather than a scalar because the
    /// verification search recurses at the same ply while every outer ply
    /// keeps its own independent exclusion.
    excluded_moves: [u16; MAX_SEARCH_PLY],
    /// Used for storing the repetition keys along the current search path.
    path_keys: [u64; MAX_SEARCH_PLY],
    /// Used for storing the piece-square continuation context at each search
    /// ply.
    path_context: [u16; MAX_SEARCH_PLY],
    /// Whether the main-search move entering each ply was its parent's first searched quiet.
    first_quiet_predecessor: [bool; MAX_SEARCH_PLY],
    /// Used for storing the static evaluation at each ply, or
    /// [`NO_STATIC_SCORE`].
    static_evals: [i32; MAX_SEARCH_PLY],
    /// Used for assembling an independent principal-variation suffix at each
    /// ply.
    pv: Vec<Vec<Move>>,
    /// Used for holding the optional legal root subset applied by searchmoves
    /// and `MultiPV` exclusion.
    root_moves: Option<Box<[Move]>>,
    /// Used for recording that the root has exactly one legal reply.
    ///
    /// A forced reply caps the soft clock target at
    /// [`SINGLE_REPLY_SOFT_CAP_MILLIS`]; hard deadlines and fixed-node or
    /// fixed-depth budgets are unaffected.
    single_reply_root: bool,
}

impl<'a, E: SearchEvaluator, const FRACTIONAL_LMR: bool, const LMR_BIAS_UNITS: i32>
    SearchContext<'a, E, FRACTIONAL_LMR, LMR_BIAS_UNITS>
{
    /// Used for initializing empty node-local state around borrowed engine
    /// resources.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - evaluator borrowed from the reusable searcher
    /// * `table` - transposition storage borrowed from the searcher
    /// * `generation` - table generation assigned to this search
    /// * `game_history` - chronological repetition keys preceding the root
    /// * `limits` - depth, node, and time budgets for the search
    /// * `stop` - caller-owned cancellation flag
    /// * `started` - monotonic instant marking the start of the search
    /// * `syzygy` - optional worker-local tablebase prober
    /// * `tb_cardinality` - piece-count gate for interior probes, already
    ///   adjusted by any root-ranking outcome; zero disables probing
    /// * `research` - research flavors active for this search; the default
    ///   value allocates no correction workspace and changes no behavior
    ///
    /// # Returns
    ///
    /// A context with cleared history, killer, path, and PV state.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::ResourceExhausted`] when any heap-owned
    /// per-search table cannot be admitted.
    #[allow(clippy::too_many_arguments)]
    fn try_new(
        evaluator: &'a mut E,
        table: &'a mut TableStorage,
        generation: u16,
        game_history: &'a [u64],
        limits: SearchLimits,
        stop: &'a AtomicBool,
        started: Instant,
        syzygy: Option<&'a mut Prober>,
        tb_cardinality: usize,
        research: ResearchSearch,
    ) -> Result<Self, SearchError> {
        let eval_cache =
            EvaluationCache::try_new(EVAL_CACHE_SIZE.load(std::sync::atomic::Ordering::Relaxed))?;
        let history = try_filled_vec(64 * 64, 0)?;
        let capture_history = try_filled_vec(CAPTURE_HISTORY_BUCKETS, 0)?;
        let continuation_one = try_filled_vec(CONTINUATION_BUCKETS * CONTINUATION_BUCKETS, 0)?;
        let continuation_two = try_filled_vec(CONTINUATION_BUCKETS * CONTINUATION_BUCKETS, 0)?;
        let correction_history = try_filled_vec(
            2 * research.correction.table_count() * CORRECTION_ENTRIES,
            CorrectionEntry::default(),
        )?;
        let pv = try_pv_rows()?;
        let incremental_evaluator = evaluator.uses_incremental_search_state();
        Ok(Self {
            evaluator,
            table,
            generation,
            game_history,
            limits,
            stop,
            started,
            nodes: 0,
            selective_depth: 0,
            stopped: false,
            null_move_pruning_min_ply: 0,
            syzygy,
            tb_cardinality,
            eval_cache,
            incremental_evaluator,
            history,
            capture_history,
            continuation_one,
            continuation_two,
            research,
            correction_history,
            killers: [[NO_MOVE; 2]; MAX_SEARCH_PLY],
            excluded_moves: [NO_MOVE; MAX_SEARCH_PLY],
            path_keys: [0; MAX_SEARCH_PLY],
            path_context: [NO_CONTINUATION; MAX_SEARCH_PLY],
            first_quiet_predecessor: [false; MAX_SEARCH_PLY],
            static_evals: [NO_STATIC_SCORE; MAX_SEARCH_PLY],
            pv,
            root_moves: None,
            single_reply_root: false,
        })
    }


    /// Used for discarding the previous root PV before a new aspiration
    /// attempt.
    fn clear_root_pv(&mut self) {
        self.pv[0].clear();
    }

    /// Used for cloning the principal variation assembled at the root.
    ///
    /// # Returns
    ///
    /// The root principal variation as an owned move list.
    fn root_pv(&self) -> Vec<Move> {
        self.pv[0].clone()
    }

    /// Used for checking cancellation and resource budgets between completed
    /// depths.
    ///
    /// Soft time limits are scaled through [`stability_budget_millis`] with
    /// the count of consecutive stable iterations before comparison.
    ///
    /// # Arguments
    ///
    /// * `stable_iterations` - consecutive iterations with an unchanged best
    ///   move
    ///
    /// # Returns
    ///
    /// `true` when the search must stop before starting another iteration.
    fn should_stop_before_iteration(&mut self, stable_iterations: u8) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            self.stopped = true;
            return true;
        }
        if self.limits.max_nodes > 0 && self.nodes >= self.limits.max_nodes {
            self.stopped = true;
            return true;
        }
        if self.limits.hard_expired(self.started) {
            self.stopped = true;
            return true;
        }
        if let Some(soft) = self.limits.soft_time {
            let scaled = self.scaled_soft_budget(soft, stable_iterations);
            if self.started.elapsed() >= scaled {
                self.stopped = true;
                return true;
            }
        }
        if let Some((soft, elapsed)) = self.limits.live_soft_state() {
            let scaled = self.scaled_soft_budget(soft, stable_iterations);
            if elapsed >= scaled {
                self.stopped = true;
                return true;
            }
        }
        false
    }

    /// Used for scaling a nominal soft budget before an elapsed-time
    /// comparison.
    ///
    /// A single-reply root first caps the nominal target at
    /// [`SINGLE_REPLY_SOFT_CAP_MILLIS`]; the capped target is then scaled
    /// through [`stability_budget_millis`]. The math is pure, so fixed-node
    /// and fixed-depth searches, which carry no soft budget, never reach it.
    ///
    /// # Arguments
    ///
    /// * `soft` - nominal soft budget published by the limits
    /// * `stable_iterations` - completed iterations with an unchanged best
    ///   move
    ///
    /// # Returns
    ///
    /// The scaled soft budget to compare against elapsed time.
    fn scaled_soft_budget(&self, soft: Duration, stable_iterations: u8) -> Duration {
        let mut soft_millis = duration_millis(soft);
        if self.single_reply_root {
            soft_millis = soft_millis.min(SINGLE_REPLY_SOFT_CAP_MILLIS);
        }
        Duration::from_millis(stability_budget_millis(soft_millis, stable_iterations))
    }

    /// Used for accounting one node and enforcing hard limits at bounded
    /// intervals.
    ///
    /// The stop token and node budget are checked on every entry; the hard
    /// clock is consulted for each of the first
    /// [`STOP_CHECK_EVERY_NODE_UNTIL`] nodes and once every
    /// [`STOP_CHECK_INTERVAL`] nodes thereafter.
    ///
    /// # Arguments
    ///
    /// * `ply` - distance from the root, used to track selective depth
    ///
    /// # Returns
    ///
    /// `true` when the search may continue at this node.
    fn enter_node(&mut self, ply: usize) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            self.stopped = true;
            return false;
        }
        if self.limits.max_nodes > 0 && self.nodes >= self.limits.max_nodes {
            self.stopped = true;
            return false;
        }
        self.nodes = self.nodes.saturating_add(1);
        self.selective_depth = self.selective_depth.max(ply_u16(ply));

        if (self.nodes <= STOP_CHECK_EVERY_NODE_UNTIL || self.nodes % STOP_CHECK_INTERVAL == 0)
            && self.limits.hard_expired(self.started)
        {
            self.stopped = true;
            return false;
        }
        true
    }

    /// Used for reading the tablebase hit counter of this search.
    ///
    /// # Returns
    ///
    /// Successful table accesses since the worker reset the counter; zero
    /// when no tablebases are configured.
    fn tb_hits(&self) -> u64 {
        self.syzygy.as_ref().map_or(0, |prober| prober.hits())
    }

    /// Used for probing the Syzygy tables at an interior node.
    ///
    /// The probe runs only when every gate holds: probing enabled, the
    /// piece count within the (root-adjusted) cardinality with sufficient
    /// remaining depth at the boundary, a zeroed halfmove clock, and no
    /// castling rights. A successful verdict maps to a score and bound
    /// honoring the `Syzygy50MoveRule` option; when that bound already
    /// decides this node it is stored in the transposition table and
    /// returned. Mate-distance interplay is preserved because tablebase
    /// scores live strictly below the mate band, so a provable mate found
    /// by the search still outranks them.
    ///
    /// # Arguments
    ///
    /// * `position` - node position; mutated and restored during probing
    /// * `depth` - remaining full-width depth in plies
    /// * `ply` - distance from the search root (positive here)
    /// * `alpha` - current lower bound
    /// * `beta` - current upper bound
    /// * `tt_key` - transposition key for storing a decisive verdict
    ///
    /// # Returns
    ///
    /// `Some(score)` when the tablebase verdict decides this node, `None`
    /// otherwise.
    fn tablebase_cutoff(
        &mut self,
        position: &mut Position,
        depth: i32,
        ply: usize,
        alpha: i32,
        beta: i32,
        tt_key: u64,
    ) -> Option<i32> {
        if self.tb_cardinality == 0 {
            return None;
        }
        let piece_count =
            usize::try_from(position.occupancy().count_ones()).expect("piece count fits usize");
        let probe_depth = self
            .syzygy
            .as_ref()
            .map_or(0, |prober| prober.probe_depth());
        // The depth gate applies at every supported piece count, not only at
        // the largest one. Restricting it to `piece_count == tb_cardinality`
        // made `SyzygyProbeDepth` unable to reduce probing cost at all: three-
        // and four-man positions were probed at every depth regardless of the
        // option, and PERF-219 measured raising it from 1 to 8 changing node
        // throughput by nothing.
        if piece_count > self.tb_cardinality
            || depth < probe_depth
            || position.halfmove_clock() != 0
            || position.castling_rights().bits() != 0
        {
            return None;
        }
        let prober = self.syzygy.as_mut()?;
        let rule50 = prober.rule50();
        let wdl = prober.probe_wdl(position)?;

        // With the fifty-move rule active, cursed wins and blessed losses
        // count as draws with a slight nudge toward the real outcome.
        let draw_band = i32::from(rule50);
        let verdict = wdl.value();
        let (score, bound) = if verdict < -draw_band {
            (-TB_WIN_SCORE + ply_i32(ply), Bound::Upper)
        } else if verdict > draw_band {
            (TB_WIN_SCORE - ply_i32(ply), Bound::Lower)
        } else {
            (2 * verdict * draw_band, Bound::Exact)
        };
        let decisive = match bound {
            Bound::Exact => true,
            Bound::Lower => score >= beta,
            Bound::Upper => score <= alpha,
        };
        if !decisive {
            return None;
        }
        // Record the verdict at a strengthened depth so shallower probes
        // reuse it instead of touching the tables again.
        let stored_depth = u16::try_from((depth + 6).clamp(0, i32::from(u16::MAX)))
            .expect("clamped depth fits u16");
        self.table.store(
            tt_key,
            TtPayload::new(
                stored_depth,
                score_to_tt(score, ply_u16(ply)),
                bound,
                NO_MOVE,
            ),
            self.generation,
        );
        Some(score)
    }

    /// Used for searching a position with negamax alpha-beta and selective
    /// pruning.
    ///
    /// Returns `None` when a hard resource limit interrupts the subtree before
    /// it produces a usable bound. Mate scores are encoded relative to `ply`.
    ///
    /// # Arguments
    ///
    /// * `position` - position to search, restored before returning
    /// * `depth` - remaining full-width depth in plies
    /// * `alpha` - lower search bound from the moving side's view
    /// * `beta` - upper search bound from the moving side's view
    /// * `ply` - distance from the search root
    /// * `pv_node` - whether this node lies on the principal variation
    /// * `null_node` - whether the parent made a null move
    ///
    /// # Returns
    ///
    /// The node's negamax score, or `None` when the search was interrupted.
    ///
    /// # Panics
    ///
    /// Panics only when internal invariants break, such as a legal move that
    /// cannot be applied or a transposition hit without a decoded score.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn negamax(
        &mut self,
        position: &mut Position,
        mut depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        pv_node: bool,
        null_node: bool,
    ) -> Option<i32> {
        if !self.enter_node(ply) {
            return None;
        }
        if ply + 2 >= MAX_SEARCH_PLY {
            return Some(self.static_score(position, ply));
        }
        self.pv[ply].clear();

        alpha = alpha.max(-MATE_SCORE + ply_i32(ply));
        beta = beta.min(MATE_SCORE - ply_i32(ply) - 1);
        if alpha >= beta {
            return Some(alpha);
        }

        let repetition_key = position.key();
        self.path_keys[ply] = if null_node { 0 } else { repetition_key };
        if !null_node && self.is_search_draw(position, ply, repetition_key) {
            let moves = position.legal_moves();
            if moves.is_empty() {
                return Some(terminal_score(
                    position.in_check(position.side_to_move()),
                    ply_u16(ply),
                ));
            }
            return Some(0);
        }

        if depth <= 0 {
            return self.quiescence(
                position,
                alpha,
                beta,
                ply,
                0,
                false,
                null_node,
                Some(repetition_key),
            );
        }

        let alpha_original = alpha;
        // A non-sentinel slot marks this node as the root of an active
        // singular verification: the stored move is skipped, and the TT
        // cutoff, Syzygy probe, TT store, and static-pruning shortcuts below
        // stay disabled so the verification measures the remaining moves.
        let excluded = self.excluded_moves[ply];
        let exclusion_active = excluded != NO_MOVE;
        let tt_key = position.tt_key_from_repetition_key(repetition_key);
        let tt_hit = self.table.probe(tt_key);
        let tt_move = tt_hit.and_then(|hit| hit.payload.best_move());
        let tt_score = tt_hit.map(|hit| score_from_tt(hit.payload.score(), ply_u16(ply)));
        if ply > 0 && !exclusion_active {
            if let Some(hit) = tt_hit.filter(|hit| i32::from(hit.payload.depth()) >= depth) {
                let score = tt_score.expect("a transposition hit has a decoded score");
                let cutoff = match hit.payload.bound() {
                    Bound::Exact => true,
                    Bound::Lower => score >= beta,
                    Bound::Upper => score <= alpha,
                };
                if cutoff {
                    if score >= beta {
                        if let Some(mv) =
                            tt_move.filter(|mv| position.piece_at(mv.from()).is_some())
                        {
                            self.reward_quiet_tt_cutoff(position, mv, depth, ply);
                            self.penalize_first_quiet_predecessor(depth, ply);
                        }
                    }
                    return Some(score);
                }
            }

            // Step 6: Syzygy tablebase probe. Runs only away from the root and
            // only outside an active singular verification (the enclosing
            // `!exclusion_active` guard keeps the probe from deciding the
            // verification node), and only when the position is inside the
            // probe gates; with no tablebases configured this is a no-op.
            if let Some(score) = self.tablebase_cutoff(position, depth, ply, alpha, beta, tt_key) {
                return Some(score);
            }
        }

        let in_check = position.in_check(position.side_to_move());
        // Correction keys are structural, so they are derived once per node
        // and reused by both the static-evaluation read below and the
        // end-of-node update. A checked node has no static evaluation and can
        // neither read nor teach the tables.
        let correction_keys = if self.research.correction.is_enabled() && !in_check {
            self.correction_keys(position)
        } else {
            [0; 3]
        };
        let static_eval = if in_check {
            self.static_evals[ply] = NO_STATIC_SCORE;
            NO_STATIC_SCORE
        } else {
            let raw = self.static_score_with_key(position, repetition_key, ply);
            let value = self.apply_correction(raw, position.side_to_move(), correction_keys);
            self.static_evals[ply] = value;
            value
        };
        let improving = self.is_improving(ply, static_eval);
        let estimated_eval = tt_hit.zip(tt_score).map_or(static_eval, |(hit, score)| {
            bound_aligned_tt_estimate(static_eval, score, hit.payload.bound())
        });

        if !pv_node && !in_check && !exclusion_active {
            if depth <= REVERSE_FUTILITY_MAX_DEPTH
                && beta.abs() < MATE_THRESHOLD
                && estimated_eval - REVERSE_FUTILITY_MARGIN * (depth - i32::from(improving)) >= beta
            {
                return Some(estimated_eval);
            }

            if !null_node
                && depth >= NULL_MOVE_MIN_DEPTH
                && ply >= self.null_move_pruning_min_ply
                && estimated_eval >= beta
                && has_null_move_material(position)
            {
                let reduction = i32::from(null_move_reduction(
                    u8::try_from(depth.min(63)).expect("positive search depth fits u8"),
                    estimated_eval,
                    beta,
                ));
                let reduced_depth = (depth - 1 - reduction).max(0);
                let undo = position
                    .make_null()
                    .expect("a non-check node accepts a null move");
                if self.incremental_evaluator {
                    self.evaluator.null_move_played(ply + 1);
                }
                self.path_context[ply + 1] = NO_CONTINUATION;
                self.first_quiet_predecessor[ply + 1] = false;
                let score = self
                    .negamax(
                        position,
                        reduced_depth,
                        -beta,
                        -beta + 1,
                        ply + 1,
                        false,
                        true,
                    )
                    .map(|value| -value);
                position.unmake_null(undo);
                let score = score?;
                if score >= beta
                    && score.abs() < MATE_THRESHOLD
                    && (!requires_null_move_verification(depth, self.null_move_pruning_min_ply)
                        || self.verify_null_move(position, reduced_depth, beta, ply)?)
                {
                    return Some(score);
                }
            }
        }

        // Conservative ProbCut. Attempted before internal iterative reduction so
        // it reasons about the same full-width depth the reference engines use.
        // The gate mirrors reverse futility and null move: non-PV, out of check,
        // and disabled while a singular verification excludes a move at this ply.
        if probcut_trigger_allowed(ply, pv_node, in_check, exclusion_active, depth, beta) {
            match self.probcut(
                position,
                depth,
                beta,
                ply,
                static_eval,
                tt_move,
                tt_score,
                tt_key,
            ) {
                ProbcutOutcome::Interrupted => return None,
                ProbcutOutcome::Cutoff(score) => return Some(score),
                ProbcutOutcome::Continue => {}
            }
        }

        if depth >= IIR_MIN_DEPTH && !in_check && tt_move.is_none() {
            depth -= 1;
        }

        let mut moves = if ply == 0 {
            self.root_moves
                .as_ref()
                .map_or_else(|| position.legal_moves(), |moves| moves.to_vec())
        } else {
            position.legal_moves()
        };
        if moves.is_empty() {
            return Some(terminal_score(in_check, ply_u16(ply)));
        }
        self.order_moves(position, &mut moves, tt_move, ply);

        let mut singular_extension = 0;
        if let (Some(hit), Some(mv), Some(score)) = (tt_hit, tt_move, tt_score) {
            if singular_trigger_allowed(
                ply,
                exclusion_active,
                depth,
                i32::from(hit.payload.depth()),
                hit.payload.bound(),
                score,
            ) {
                let singular_beta = score - SINGULAR_MARGIN * depth / 64;
                let value =
                    self.singular_verification_value(position, depth, ply, mv, singular_beta)?;
                // Multi-cut: even with the stored move excluded the remaining
                // moves reach beta, so at least two moves fail high and the
                // node fails high immediately, without searching any move and
                // without storing a transposition bound for the full move
                // set. Mate-range values stay excluded because the
                // verification searched a reduced depth.
                if value >= beta && value.abs() < MATE_THRESHOLD {
                    return Some(value);
                }
                if value < singular_beta {
                    singular_extension = 1;
                }
            }
        }

        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut best_capture = None;
        let mut searched = 0_u16;
        let mut tried_quiets = Vec::with_capacity(32);
        let mut tried_captures = Vec::with_capacity(32);
        for mv in moves {
            if exclusion_active && mv.raw() == excluded {
                continue;
            }
            let tactical = position.is_capture(mv) || mv.promotion().is_some();
            let capture_index = capture_history_index(position, mv);
            let quiet_target = (!tactical).then(|| continuation_target(position, mv));
            let history_score = self.history[history_index(mv)];
            let continuation_score =
                quiet_target.map_or(0, |target| self.continuation_score(ply, target));
            let futility = static_eval != NO_STATIC_SCORE
                && !pv_node
                && !in_check
                && !tactical
                && searched > 0
                && depth <= FUTILITY_MAX_DEPTH
                && !self.is_killer(ply, mv)
                && alpha.abs() < MATE_THRESHOLD
                && static_eval + FUTILITY_MARGIN * depth <= alpha;
            let late_move_prune = static_eval != NO_STATIC_SCORE
                && !pv_node
                && !in_check
                && !tactical
                && depth <= LATE_MOVE_PRUNING_MAX_DEPTH
                && !self.is_killer(ply, mv)
                && alpha.abs() < MATE_THRESHOLD
                && searched
                    >= late_move_prune_threshold(
                        u8::try_from(depth.max(0)).expect("shallow depth fits u8"),
                    )
                && history_score <= depth * depth
                && (self.research.eval_free_late_move_pruning
                    || static_eval + FUTILITY_MARGIN * depth <= alpha);
            if futility || late_move_prune {
                continue;
            }

            let move_context = continuation_target(position, mv);
            let prune_losing_capture = position.is_capture(mv)
                && main_search_losing_capture_pruning_allowed(
                    depth, ply, pv_node, in_check, searched, alpha, beta,
                )
                && is_losing_capture(position, mv);
            let undo = position
                .make_move(mv)
                .expect("legal generator emitted an applicable move");
            let gives_check = position.in_check(position.side_to_move());
            if prune_losing_capture && !gives_check {
                position.unmake_move(mv, undo);
                continue;
            }
            if let Some(target) = quiet_target {
                tried_quiets.push(target);
            }
            if let Some(index) = capture_index {
                if tried_captures.len() < 32 && !tried_captures.contains(&index) {
                    tried_captures.push(index);
                }
            }
            if self.incremental_evaluator {
                self.evaluator.move_played(position, mv, undo, ply + 1);
            }
            self.path_context[ply + 1] = move_context;
            self.first_quiet_predecessor[ply + 1] = !tactical && searched == 0;
            let extension = i32::from(gives_check)
                + if Some(mv) == tt_move {
                    singular_extension
                } else {
                    0
                };
            let child_depth = depth - 1 + extension;

            let score = if searched == 0 {
                self.negamax(
                    position,
                    child_depth,
                    -beta,
                    -alpha,
                    ply + 1,
                    pv_node,
                    false,
                )
                .map(|value| -value)
            } else {
                let reduction = late_move_reduction_for::<FRACTIONAL_LMR, LMR_BIAS_UNITS>(
                    LateMoveReductionContext {
                        node: if pv_node {
                            LmrNode::PrincipalVariation
                        } else {
                            LmrNode::Cut
                        },
                        in_check,
                        move_kind: if tactical {
                            LmrMove::Tactical
                        } else {
                            LmrMove::Quiet
                        },
                        depth,
                        searched,
                        history_score,
                        improving,
                        continuation_score,
                        child_depth,
                    },
                );
                let reduced_depth = (child_depth - reduction).max(0);
                let mut candidate = self
                    .negamax(
                        position,
                        reduced_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        false,
                        false,
                    )
                    .map(|value| -value);
                if let Some(value) = candidate {
                    if reduction > 0 && value > alpha {
                        candidate = self
                            .negamax(
                                position,
                                child_depth,
                                -alpha - 1,
                                -alpha,
                                ply + 1,
                                false,
                                false,
                            )
                            .map(|full| -full);
                    }
                }
                if let Some(value) = candidate {
                    if value > alpha && value < beta {
                        candidate = self
                            .negamax(position, child_depth, -beta, -alpha, ply + 1, true, false)
                            .map(|full| -full);
                    }
                }
                candidate
            };
            position.unmake_move(mv, undo);
            let score = score?;
            searched = searched.saturating_add(1);

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
                best_capture = capture_index;
            }
            if score > alpha {
                alpha = score;
                self.update_pv(ply, mv);
            }
            if alpha >= beta {
                if !tactical {
                    self.record_killer(ply, mv);
                    self.update_history(mv, depth);
                    self.update_continuation(ply, quiet_target, &tried_quiets, depth);
                }
                break;
            }
        }

        // A verification root whose only candidate was the excluded move has
        // no searched alternative: fail low at alpha so the singular check
        // treats the forced transposition move as clearly best.
        if exclusion_active && searched == 0 {
            return Some(alpha);
        }
        if best_score > alpha_original {
            self.update_capture_history(best_capture, &tried_captures, depth);
        }
        let best_move = best_move.expect("nonterminal node searched at least one move");
        let bound = if best_score <= alpha_original {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        // A restricted root score is not a valid bound for the unrestricted
        // position, and a verification score under an active exclusion is not
        // a bound for the full move set. Child entries remain reusable because
        // only ply zero is filtered and only this ply excludes a move.
        if !exclusion_active && (ply > 0 || self.root_moves.is_none()) {
            self.table.store(
                tt_key,
                TtPayload::new(
                    u16::try_from(depth.clamp(0, i32::from(u16::MAX)))
                        .expect("clamped depth fits u16"),
                    score_to_tt(best_score, ply_u16(ply)),
                    bound,
                    best_move.raw(),
                ),
                self.generation,
            );
        }
        // Step 15: correction history. The residual between what the search
        // proved and what the evaluator claimed is evidence about this pawn
        // structure rather than about this node, so it is recorded for later
        // static evaluations that share the structure.
        if self.correction_update_allowed(
            in_check,
            exclusion_active,
            best_move,
            best_score,
            static_eval,
            bound,
            depth,
            position,
        ) {
            self.update_correction_history(
                position.side_to_move(),
                correction_keys,
                depth,
                best_score - static_eval,
            );
        }
        Some(best_score)
    }

    /// Used for re-searching a deep null fail-high with nearby null pruning
    /// disabled.
    ///
    /// The original position is searched at the same reduced depth and through
    /// the same null window used to establish the cutoff. The guard is restored
    /// before this method propagates cancellation, so a stopped verification
    /// cannot leave later work in a partially disabled state.
    ///
    /// # Arguments
    ///
    /// * `position` - original position whose null fail-high is re-checked
    /// * `reduced_depth` - depth of the null search that produced the cutoff
    /// * `beta` - bound the verification must reach to confirm the cutoff
    /// * `ply` - distance from the search root
    ///
    /// # Returns
    ///
    /// `Some(true)` when the cutoff is confirmed, `Some(false)` when it is
    /// rejected, or `None` when the verification was interrupted.
    fn verify_null_move(
        &mut self,
        position: &mut Position,
        reduced_depth: i32,
        beta: i32,
        ply: usize,
    ) -> Option<bool> {
        debug_assert_eq!(self.null_move_pruning_min_ply, 0);
        let previous_min_ply = self.null_move_pruning_min_ply;
        self.null_move_pruning_min_ply = null_move_verification_min_ply(ply, reduced_depth);

        let verification = self.negamax(position, reduced_depth, beta - 1, beta, ply, false, false);
        self.null_move_pruning_min_ply = previous_min_ply;
        verification.map(|value| value >= beta)
    }

    /// Used for measuring how the moves other than the transposition move
    /// compare against the singular verification bound at this node.
    ///
    /// The same position is re-searched at the same ply with the stored move
    /// excluded, through the null window `[singular_beta - 1, singular_beta]`
    /// at depth `(depth - 1) / 2`. A value below `singular_beta` proves that
    /// every alternative is clearly worse, so the stored move deserves one
    /// extra ply; a value at or above the caller's beta instead supports a
    /// multi-cut fail high. The exclusion slot is restored before this method
    /// propagates cancellation, so a stopped verification cannot leave later
    /// work at this ply excluded.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose transposition move is tested
    /// * `depth` - remaining full-width depth at the node
    /// * `ply` - distance from the search root
    /// * `tt_move` - stored transposition move to exclude
    /// * `singular_beta` - verification bound
    ///   `tt_score - SINGULAR_MARGIN * depth / 64`
    ///
    /// # Returns
    ///
    /// `Some(value)` with the verification search value, or `None` when the
    /// verification was interrupted.
    fn singular_verification_value(
        &mut self,
        position: &mut Position,
        depth: i32,
        ply: usize,
        tt_move: Move,
        singular_beta: i32,
    ) -> Option<i32> {
        debug_assert_eq!(self.excluded_moves[ply], NO_MOVE);
        let verification_depth = (depth - 1) / 2;
        let previous_excluded = self.excluded_moves[ply];
        self.excluded_moves[ply] = tt_move.raw();

        let verification = self.negamax(
            position,
            verification_depth,
            singular_beta - 1,
            singular_beta,
            ply,
            false,
            false,
        );
        self.excluded_moves[ply] = previous_excluded;
        verification
    }

    /// Used for attempting a conservative `ProbCut` prune at one node.
    ///
    /// When a good tactical move (a capture or promotion whose static exchange
    /// evaluation plausibly reaches `probcut_beta = beta + PROBCUT_MARGIN`)
    /// survives a reduced zero-window verification at that raised bound, the
    /// node almost certainly fails high and can be pruned without the full-width
    /// move loop. Each candidate is first checked with a quiescence search at the
    /// raised bound; only if that holds is the reduced
    /// `depth - PROBCUT_REDUCTION` search run, exactly as the reference engines
    /// (Stockfish, Reckless, Viridithas) sequence the two searches.
    ///
    /// A confirmed cutoff stores a lower-bound transposition entry keyed to the
    /// verification depth, mirroring the references, and returns the value mapped
    /// back into the caller's window as `value - (probcut_beta - beta)`, which is
    /// a sound lower bound at least `beta`. Mate-range verification values are
    /// never used to cut because the reduced search cannot prove a mate.
    ///
    /// The caller guarantees the node is eligible (non-PV, not in check, no
    /// active singular exclusion, deep enough, and a non-mate `beta`) via
    /// [`probcut_trigger_allowed`]. A transposition score already below
    /// `probcut_beta` suppresses the attempt, since the node is unlikely to reach
    /// the raised bound.
    ///
    /// # Arguments
    ///
    /// * `position` - position to probe, restored before returning
    /// * `depth` - remaining full-width depth at the node
    /// * `beta` - upper search bound from the moving side's view
    /// * `ply` - distance from the search root
    /// * `static_eval` - static evaluation at the node (never a sentinel here)
    /// * `tt_move` - transposition best move used only for candidate ordering
    /// * `tt_score` - decoded transposition score, if any
    /// * `tt_key` - transposition key used to store a confirmed cutoff bound
    ///
    /// # Returns
    ///
    /// [`ProbcutOutcome::Cutoff`] when a `ProbCut` cutoff prunes the node,
    /// [`ProbcutOutcome::Continue`] when no cutoff applies and the ordinary
    /// search must continue, or [`ProbcutOutcome::Interrupted`] when the search
    /// was interrupted.
    #[allow(clippy::too_many_arguments)]
    fn probcut(
        &mut self,
        position: &mut Position,
        depth: i32,
        beta: i32,
        ply: usize,
        static_eval: i32,
        tt_move: Option<Move>,
        tt_score: Option<i32>,
        tt_key: u64,
    ) -> ProbcutOutcome {
        debug_assert_ne!(static_eval, NO_STATIC_SCORE);
        let probcut_beta = beta + PROBCUT_MARGIN;
        // A trusted transposition score below the raised bound means the node is
        // unlikely to reach probcut_beta, so skip the attempt entirely.
        if tt_score.is_some_and(|score| score < probcut_beta) {
            return ProbcutOutcome::Continue;
        }
        let probcut_depth = (depth - PROBCUT_REDUCTION).max(0);
        // A capture must plausibly recover probcut_beta from the static eval.
        let see_threshold = probcut_beta - static_eval;
        let mut candidates = position.legal_tactical_moves();
        self.order_moves(position, &mut candidates, tt_move, ply);
        for mv in candidates {
            if static_exchange_eval(position, mv) < see_threshold {
                continue;
            }
            let move_context = continuation_target(position, mv);
            let undo = position
                .make_move(mv)
                .expect("legal generator emitted an applicable move");
            if self.incremental_evaluator {
                self.evaluator.move_played(position, mv, undo, ply + 1);
            }
            self.path_context[ply + 1] = move_context;
            self.first_quiet_predecessor[ply + 1] = false;
            let mut value = self
                .quiescence(
                    position,
                    -probcut_beta,
                    -probcut_beta + 1,
                    ply + 1,
                    0,
                    true,
                    false,
                    None,
                )
                .map(|score| -score);
            if let Some(score) = value {
                if score >= probcut_beta && probcut_depth > 0 {
                    value = self
                        .negamax(
                            position,
                            probcut_depth,
                            -probcut_beta,
                            -probcut_beta + 1,
                            ply + 1,
                            false,
                            false,
                        )
                        .map(|score| -score);
                }
            }
            position.unmake_move(mv, undo);
            let Some(value) = value else {
                return ProbcutOutcome::Interrupted;
            };
            if value >= probcut_beta && value.abs() < MATE_THRESHOLD {
                let store_depth = u16::try_from((probcut_depth + 1).clamp(0, i32::from(u16::MAX)))
                    .expect("clamped depth fits u16");
                self.table.store(
                    tt_key,
                    TtPayload::new(
                        store_depth,
                        score_to_tt(value, ply_u16(ply)),
                        Bound::Lower,
                        mv.raw(),
                    ),
                    self.generation,
                );
                return ProbcutOutcome::Cutoff(value - (probcut_beta - beta));
            }
        }
        ProbcutOutcome::Continue
    }

    /// Used for extending a nominal leaf through forcing captures,
    /// promotions, and checks.
    ///
    /// The search uses stand-pat, delta pruning, and static exchange evaluation
    /// while preserving every legal check evasion. `known_repetition_key`
    /// avoids recomputing a key already obtained by the caller.
    ///
    /// # Arguments
    ///
    /// * `position` - position to search, restored before returning
    /// * `alpha` - lower search bound from the moving side's view
    /// * `beta` - upper search bound from the moving side's view
    /// * `ply` - distance from the search root
    /// * `qply` - number of quiescence plies already entered
    /// * `count_node` - whether this call accounts a node on entry
    /// * `null_node` - whether the parent made a null move
    /// * `known_repetition_key` - caller-computed repetition key, if any
    ///
    /// # Returns
    ///
    /// The quiescence score, or `None` when the search was interrupted.
    #[allow(
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn quiescence(
        &mut self,
        position: &mut Position,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        qply: u8,
        count_node: bool,
        null_node: bool,
        known_repetition_key: Option<u64>,
    ) -> Option<i32> {
        if count_node && !self.enter_node(ply) {
            return None;
        }
        if ply + 2 >= MAX_SEARCH_PLY {
            return Some(self.static_score(position, ply));
        }
        self.pv[ply].clear();
        alpha = alpha.max(-MATE_SCORE + ply_i32(ply));
        beta = beta.min(MATE_SCORE - ply_i32(ply) - 1);
        if alpha >= beta {
            return Some(alpha);
        }
        let alpha_original = alpha;

        let key = known_repetition_key.unwrap_or_else(|| position.key());
        self.path_keys[ply] = if null_node { 0 } else { key };
        let in_check = position.in_check(position.side_to_move());
        let mut check_moves = in_check.then(|| position.legal_moves());
        if !null_node && self.is_search_draw(position, ply, key) {
            if check_moves.as_ref().is_some_and(Vec::is_empty) {
                return Some(terminal_score(true, ply_u16(ply)));
            }
            return Some(0);
        }
        if check_moves.as_ref().is_some_and(Vec::is_empty) {
            return Some(terminal_score(true, ply_u16(ply)));
        }

        let tt_key = position.tt_key_from_repetition_key(key);
        let tt_hit = (qply == 0).then(|| self.table.probe(tt_key)).flatten();
        let tt_move = tt_hit.and_then(|hit| hit.payload.best_move());
        if let Some(hit) = tt_hit {
            let score = score_from_tt(hit.payload.score(), ply_u16(ply));
            match hit.payload.bound() {
                Bound::Exact => return Some(score),
                Bound::Lower if score >= beta => return Some(score),
                Bound::Upper if score <= alpha => return Some(score),
                _ => {}
            }
        }

        let horizon_legal = if !in_check && qply == 0 {
            let legal = position.legal_moves();
            if legal.is_empty() {
                self.store_quiescence(tt_key, 0, alpha_original, beta, None, ply, qply);
                return Some(0);
            }
            if let Some(mating_move) = find_mate_in_one(position, &legal) {
                let score = MATE_SCORE - ply_i32(ply) - 1;
                self.store_quiescence(
                    tt_key,
                    score,
                    alpha_original,
                    beta,
                    Some(mating_move),
                    ply,
                    qply,
                );
                return Some(score);
            }
            Some(legal)
        } else {
            None
        };
        let horizon_checked = horizon_legal.is_some();
        let stand_pat = if in_check {
            -INFINITY
        } else {
            // Quiescence reads the same learned structural residual as the
            // main search but never teaches it: a stand-pat is the evaluator's
            // own claim, so treating it as evidence about the evaluator would
            // let the tables reinforce themselves.
            let raw = self.static_score_with_key(position, key, ply);
            let value = if self.research.correction.is_enabled() {
                let keys = self.correction_keys(position);
                self.apply_correction(raw, position.side_to_move(), keys)
            } else {
                raw
            };
            if value >= beta {
                if horizon_legal.is_none() && position.legal_moves().is_empty() {
                    return Some(0);
                }
                self.store_quiescence(tt_key, value, alpha_original, beta, None, ply, qply);
                return Some(value);
            }
            alpha = alpha.max(value);
            value
        };
        if qply >= QUIESCENCE_MAX_PLY {
            if !in_check && horizon_legal.is_none() && position.legal_moves().is_empty() {
                return Some(0);
            }
            return Some(if in_check {
                self.static_score_with_key(position, key, ply)
            } else {
                stand_pat
            });
        }

        let mut moves = if let Some(moves) = check_moves.take() {
            moves
        } else if let Some(mut moves) = horizon_legal {
            retain_tactical_moves(position, &mut moves);
            moves
        } else {
            position.legal_tactical_moves()
        };
        if moves.is_empty() {
            let score = if in_check {
                terminal_score(true, ply_u16(ply))
            } else if !horizon_checked && position.legal_moves().is_empty() {
                0
            } else {
                stand_pat
            };
            self.store_quiescence(tt_key, score, alpha_original, beta, None, ply, qply);
            return Some(score);
        }
        self.order_moves(position, &mut moves, tt_move, ply);

        let mut best = stand_pat;
        let mut best_move = None;
        for mv in moves {
            if !in_check {
                if mv.promotion().is_none()
                    && alpha.abs() < MATE_THRESHOLD
                    && stand_pat + captured_value(position, mv) + DELTA_MARGIN <= alpha
                {
                    continue;
                }
                if is_losing_capture(position, mv) {
                    continue;
                }
            }
            let move_context = continuation_target(position, mv);
            let undo = position
                .make_move(mv)
                .expect("legal generator emitted an applicable move");
            if self.incremental_evaluator {
                self.evaluator.move_played(position, mv, undo, ply + 1);
            }
            self.path_context[ply + 1] = move_context;
            self.first_quiet_predecessor[ply + 1] = false;
            let score = self
                .quiescence(
                    position,
                    -beta,
                    -alpha,
                    ply + 1,
                    qply + 1,
                    true,
                    false,
                    None,
                )
                .map(|value| -value);
            position.unmake_move(mv, undo);
            let score = score?;
            if score > best {
                best = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
                self.update_pv(ply, mv);
            }
            if alpha >= beta {
                break;
            }
        }
        self.store_quiescence(tt_key, best, alpha_original, beta, best_move, ply, qply);
        Some(best)
    }

    /// Used for storing one complete top-level quiescence bound in the shared
    /// TT format.
    ///
    /// Descendant quiescence calls are excluded because their remaining
    /// eight-ply tactical horizon depends on `qply`. The ordinary replacement
    /// policy prevents a depth-zero entry from displacing deeper same-generation
    /// search work.
    ///
    /// # Arguments
    ///
    /// * `tt_key` - full transposition key of the stored position
    /// * `score` - quiescence score to encode relative to the root
    /// * `alpha_original` - lower bound in force before the quiescence search
    /// * `beta` - upper bound in force during the quiescence search
    /// * `best_move` - best tactical move found, if any
    /// * `ply` - distance from the search root
    /// * `qply` - quiescence ply; only zero is stored
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn store_quiescence(
        &mut self,
        tt_key: u64,
        score: i32,
        alpha_original: i32,
        beta: i32,
        best_move: Option<Move>,
        ply: usize,
        qply: u8,
    ) {
        if qply != 0 {
            return;
        }
        let bound = if score <= alpha_original {
            Bound::Upper
        } else if score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.table.store(
            tt_key,
            TtPayload::new(
                0,
                score_to_tt(score, ply_u16(ply)),
                bound,
                best_move.map_or(NO_MOVE, Move::raw),
            ),
            self.generation,
        );
    }

    /// Used for computing a clamped, cached static score for `position`.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `ply` - distance from the root, forwarded to incremental evaluators
    ///
    /// # Returns
    ///
    /// The clamped static score in centipawns.
    fn static_score(&mut self, position: &Position, ply: usize) -> i32 {
        let key = position.key();
        self.static_score_with_key(position, key, ply)
    }

    /// Used for computing a clamped static score with a caller-supplied
    /// repetition key.
    ///
    /// Static evaluators intentionally ignore the FEN clocks. Reusing their
    /// result across clock variants avoids duplicate neural inference while the
    /// transposition table continues to use the clock-sensitive TT key.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `key` - clock-free repetition key identifying the position
    /// * `ply` - distance from the root, forwarded to incremental evaluators
    ///
    /// # Returns
    ///
    /// The clamped static score in centipawns.
    fn static_score_with_key(&mut self, position: &Position, key: u64, ply: usize) -> i32 {
        if let Some(score) = self.eval_cache.get(key) {
            return score;
        }
        let score = clamp_static_score(if self.incremental_evaluator {
            self.evaluator.evaluate_at(position, ply)
        } else {
            self.evaluator.evaluate(position)
        });
        self.eval_cache.put(key, score);
        score
    }

    /// Used for deriving the structural correction keys of one position.
    ///
    /// Unused table slots are left zero. The keys are computed once per node
    /// and reused by both the read and the update so the hashing is not paid
    /// twice.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose structure is hashed
    ///
    /// # Returns
    ///
    /// One key per active correction table, in table order.
    fn correction_keys(&self, position: &Position) -> [u64; 3] {
        match self.research.correction {
            CorrectionHistoryMode::Off => [0; 3],
            CorrectionHistoryMode::Pawn => [pawn_structure_key(position), 0, 0],
            CorrectionHistoryMode::PawnAndNonPawn => [
                pawn_structure_key(position),
                non_pawn_structure_key(position, Color::White),
                non_pawn_structure_key(position, Color::Black),
            ],
        }
    }

    /// Used for locating one correction cell.
    ///
    /// # Arguments
    ///
    /// * `side` - side to move at the corrected node
    /// * `table` - index of the structural table within the active flavor
    /// * `key` - structural key for that table
    ///
    /// # Returns
    ///
    /// Index into [`Self::correction_history`].
    fn correction_slot(&self, side: Color, table: usize, key: u64) -> usize {
        let mixed = key ^ (key >> 32);
        let cell = usize::try_from(mixed & u64::from(u32::MAX)).expect("u32 fits supported usize")
            & (CORRECTION_ENTRIES - 1);
        (side.index() * self.research.correction.table_count() + table) * CORRECTION_ENTRIES + cell
    }

    /// Used for correcting one raw static evaluation by learned structural
    /// residuals.
    ///
    /// Every active table contributes its stored residual; the sum is scaled
    /// back into centipawns and clamped so a correction can never manufacture
    /// a mate-range score.
    ///
    /// # Arguments
    ///
    /// * `raw` - clamped static evaluation as produced by the evaluator
    /// * `side` - side to move at the corrected node
    /// * `keys` - structural keys from [`Self::correction_keys`]
    ///
    /// # Returns
    ///
    /// The corrected static evaluation, or `raw` unchanged when correction is
    /// disabled.
    fn apply_correction(&self, raw: i32, side: Color, keys: [u64; 3]) -> i32 {
        if !self.research.correction.is_enabled() {
            return raw;
        }
        let mut total = 0;
        for (table, key) in keys
            .iter()
            .copied()
            .take(self.research.correction.table_count())
            .enumerate()
        {
            total += self.correction_history[self.correction_slot(side, table, key)].value();
        }
        clamp_static_score(raw + total / CORRECTION_GRAIN)
    }

    /// Used for blending one node's search-versus-static residual into every
    /// active correction table.
    ///
    /// The caller has already established that the node is eligible: it was
    /// not in check, it was not a singular verification, its best move was
    /// quiet, its score is outside the mate range, and its bound points in the
    /// same direction as the residual. Those conditions are what keep the
    /// tables measuring evaluator bias rather than search noise.
    ///
    /// # Arguments
    ///
    /// * `side` - side to move at the node
    /// * `keys` - structural keys from [`Self::correction_keys`]
    /// * `depth` - remaining depth searched at the node
    /// * `residual` - search score minus corrected static evaluation
    fn update_correction_history(
        &mut self,
        side: Color,
        keys: [u64; 3],
        depth: i32,
        residual: i32,
    ) {
        let bonus = (residual * CORRECTION_GRAIN * depth / CORRECTION_BONUS_DIVISOR)
            .clamp(-CORRECTION_MAX_BONUS, CORRECTION_MAX_BONUS);
        for (table, key) in keys
            .iter()
            .copied()
            .take(self.research.correction.table_count())
            .enumerate()
        {
            let slot = self.correction_slot(side, table, key);
            self.correction_history[slot].update(bonus);
        }
    }

    /// Used for deciding whether one completed node may teach the correction
    /// tables.
    ///
    /// # Arguments
    ///
    /// * `in_check` - whether the node's side to move was in check
    /// * `exclusion_active` - whether a singular verification excluded a move
    /// * `best_move` - best move found at the node
    /// * `best_score` - score returned by the node
    /// * `static_eval` - corrected static evaluation at the node
    /// * `bound` - transposition bound the score carries
    /// * `depth` - remaining depth searched at the node
    /// * `position` - position at the node, used to classify the best move
    ///
    /// # Returns
    ///
    /// `true` when the residual is a usable observation of evaluator bias.
    #[allow(clippy::too_many_arguments)]
    fn correction_update_allowed(
        &self,
        in_check: bool,
        exclusion_active: bool,
        best_move: Move,
        best_score: i32,
        static_eval: i32,
        bound: Bound,
        depth: i32,
        position: &Position,
    ) -> bool {
        if !self.research.correction.is_enabled()
            || in_check
            || exclusion_active
            || depth <= 0
            || static_eval == NO_STATIC_SCORE
            || best_score.abs() >= MATE_THRESHOLD
        {
            return false;
        }
        if position.is_capture(best_move) || best_move.promotion().is_some() {
            return false;
        }
        match bound {
            Bound::Exact => true,
            Bound::Lower => best_score > static_eval,
            Bound::Upper => best_score < static_eval,
        }
    }

    /// Used for detecting rule draws, true game-history threefolds, and
    /// search cycles.
    ///
    /// Historical positions require two matching prior occurrences because the
    /// current node is the third. A matching position already on the current
    /// search path is deliberately treated as a draw on its first recurrence;
    /// this standard engine cycle rule prevents reversible loops from consuming
    /// the remaining search. Both scans step by two for side-to-move parity and
    /// stop at the current position's reversible halfmove boundary.
    ///
    /// # Arguments
    ///
    /// * `position` - position tested for draw adjudication
    /// * `ply` - distance from the search root
    /// * `key` - repetition key of `position`
    ///
    /// # Returns
    ///
    /// `true` when the node adjudicates as a draw.
    fn is_search_draw(&self, position: &Position, ply: usize, key: u64) -> bool {
        if position.halfmove_clock() >= 100 || position.is_insufficient_material() {
            return true;
        }
        let preceding = self.game_history.len().saturating_add(ply);
        let reversible = usize::from(position.halfmove_clock()).min(preceding);
        if reversible < 2 {
            return false;
        }

        let mut historical_occurrences = 0_u8;
        for distance in (2..=reversible).step_by(2) {
            if distance <= ply {
                if self.path_keys[ply - distance] == key {
                    return true;
                }
                continue;
            }
            let history_index = preceding - distance;
            if self.game_history[history_index] == key {
                historical_occurrences += 1;
                if historical_occurrences >= 2 {
                    return true;
                }
            }
        }
        false
    }

    /// Used for sorting moves by transposition, tactical, killer, and history
    /// priorities.
    ///
    /// # Arguments
    ///
    /// * `position` - position the moves belong to
    /// * `moves` - legal moves reordered in place
    /// * `tt_move` - transposition-table move ordered first, if any
    /// * `ply` - distance from the root, selecting the killer slots
    fn order_moves(
        &self,
        position: &Position,
        moves: &mut [Move],
        tt_move: Option<Move>,
        ply: usize,
    ) {
        moves.sort_by_cached_key(|mv| {
            std::cmp::Reverse(self.move_order_score(position, *mv, tt_move, ply))
        });
    }

    /// Used for computing the deterministic ordering score of one legal move.
    ///
    /// The transposition move ranks highest, followed by the overlapping
    /// non-losing capture and promotion bands, killer moves, and
    /// history-scored quiets; losing captures rank below neutral quiets.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move belongs to
    /// * `mv` - legal move to score
    /// * `tt_move` - transposition-table move ordered first, if any
    /// * `ply` - distance from the root, selecting the killer slots
    ///
    /// # Returns
    ///
    /// The move's ordering score; larger values are searched earlier.
    ///
    /// # Panics
    ///
    /// Panics when `mv` has no moving piece on its source square.
    fn move_order_score(
        &self,
        position: &Position,
        mv: Move,
        tt_move: Option<Move>,
        ply: usize,
    ) -> i32 {
        if Some(mv) == tt_move {
            return 1_000_000;
        }
        let moving = position
            .piece_at(mv.from())
            .expect("legal move has a moving piece");
        if position.is_capture(mv) {
            let victim = captured_value(position, mv);
            let base = if is_losing_capture(position, mv) {
                -20_000
            } else {
                100_000
            };
            let promotion = mv.promotion().map_or(0, material_value);
            let capture_history = capture_history_index(position, mv)
                .map_or(0, |index| i32::from(self.capture_history[index]));
            return base + victim * 16 - material_value(moving.kind) + promotion + capture_history;
        }
        if let Some(promotion) = mv.promotion() {
            return 80_000 + material_value(promotion);
        }
        let quiet_score = self.history[history_index(mv)]
            + self.continuation_score(ply, continuation_target(position, mv))
            + self.evaluator.quiet_move_order_prior(position, mv);
        if ply < MAX_SEARCH_PLY {
            if self.killers[ply][0] == mv.raw() {
                return 70_000 + quiet_score;
            }
            if self.killers[ply][1] == mv.raw() {
                return 69_000 + quiet_score;
            }
        }
        quiet_score
    }

    /// Used for promoting a quiet beta-cutoff move into the ply's killer
    /// slots.
    ///
    /// # Arguments
    ///
    /// * `ply` - distance from the root, selecting the killer slots
    /// * `mv` - quiet move that caused the beta cutoff
    fn record_killer(&mut self, ply: usize, mv: Move) {
        if ply >= MAX_SEARCH_PLY || self.killers[ply][0] == mv.raw() {
            return;
        }
        self.killers[ply][1] = self.killers[ply][0];
        self.killers[ply][0] = mv.raw();
    }

    /// Used for applying a depth-scaled gravity update to butterfly history.
    ///
    /// # Arguments
    ///
    /// * `mv` - quiet move whose source-destination cell is rewarded
    /// * `depth` - remaining depth scaling the depth-squared bonus
    fn update_history(&mut self, mv: Move, depth: i32) {
        let bonus = depth.saturating_mul(depth).clamp(1, 2_048);
        let slot = &mut self.history[history_index(mv)];
        *slot = (*slot + bonus - *slot * bonus / HISTORY_MAX).clamp(-HISTORY_MAX, HISTORY_MAX);
    }

    /// Used for rewarding a quiet move whose depth-qualified TT score proves
    /// a beta cutoff.
    ///
    /// A table cutoff bypasses the ordinary move loop, so without this hook its
    /// best move cannot teach butterfly or continuation ordering. Captures,
    /// promotions, and entries whose source square has no piece remain outside
    /// quiet history.
    ///
    /// # Arguments
    ///
    /// * `position` - position the stored move belongs to
    /// * `mv` - stored transposition move that proved the cutoff
    /// * `depth` - remaining depth scaling the history bonus
    /// * `ply` - distance from the root, selecting continuation contexts
    fn reward_quiet_tt_cutoff(&mut self, position: &Position, mv: Move, depth: i32, ply: usize) {
        if position.is_capture(mv) || mv.promotion().is_some() {
            return;
        }
        let target = continuation_target(position, mv);
        if target == NO_CONTINUATION {
            return;
        }
        self.update_history(mv, depth);
        self.update_continuation(ply, Some(target), &[target], depth);
    }

    /// Penalizes a first quiet predecessor when its child has a stored refutation.
    ///
    /// The move entering `ply` supplies the target, while the preceding one- and
    /// two-ply path contexts select the same continuation cells that would have
    /// learned from an ordinary search. Root moves without an earlier context,
    /// later moves, tactical moves, and null moves leave the tables unchanged.
    fn penalize_first_quiet_predecessor(&mut self, depth: i32, ply: usize) {
        if ply == 0 || !self.first_quiet_predecessor[ply] {
            return;
        }
        let target = self.path_context[ply];
        if target == NO_CONTINUATION {
            return;
        }
        let malus = -depth.saturating_mul(depth).clamp(1, HISTORY_MAX);
        if self.path_context[ply - 1] != NO_CONTINUATION {
            let index = continuation_index(self.path_context[ply - 1], target);
            self.continuation_one[index] = gravity_i16(self.continuation_one[index], malus);
        }
        if ply >= 2 && self.path_context[ply - 2] != NO_CONTINUATION {
            let index = continuation_index(self.path_context[ply - 2], target);
            self.continuation_two[index] = gravity_i16(self.continuation_two[index], malus);
        }
    }

    /// Used for rewarding a best capture and penalizing searched captures it
    /// surpassed.
    ///
    /// Updates use Janus's depth-squared signal and bounded gravity. The caller
    /// records unique buckets, and the best bucket is excluded from penalties,
    /// so promotion variants cannot multiply or cancel one ordering signal.
    ///
    /// # Arguments
    ///
    /// * `best` - capture-history bucket of the best move, if it captured
    /// * `tried` - unique buckets of the captures searched at this node
    /// * `depth` - remaining depth scaling the depth-squared signal
    fn update_capture_history(&mut self, best: Option<usize>, tried: &[usize], depth: i32) {
        let bonus = depth.saturating_mul(depth).clamp(1, 2_048);
        if let Some(index) = best {
            self.capture_history[index] = capture_gravity_i16(self.capture_history[index], bonus);
        }
        for index in tried.iter().copied().filter(|index| Some(*index) != best) {
            self.capture_history[index] = capture_gravity_i16(self.capture_history[index], -bonus);
        }
    }

    /// Used for testing whether `mv` occupies either killer slot at `ply`.
    ///
    /// # Arguments
    ///
    /// * `ply` - distance from the root, selecting the killer slots
    /// * `mv` - move compared against both stored killers
    ///
    /// # Returns
    ///
    /// `true` when `mv` matches a killer stored at `ply`.
    fn is_killer(&self, ply: usize, mv: Move) -> bool {
        ply < MAX_SEARCH_PLY
            && (self.killers[ply][0] == mv.raw() || self.killers[ply][1] == mv.raw())
    }

    /// Used for comparing a static score with the nearest valid same-side
    /// ancestor.
    ///
    /// # Arguments
    ///
    /// * `ply` - distance from the root; the ancestor sits two plies above
    /// * `static_eval` - current static evaluation, or [`NO_STATIC_SCORE`]
    ///
    /// # Returns
    ///
    /// `true` when both evaluations exist and the current one is greater.
    fn is_improving(&self, ply: usize, static_eval: i32) -> bool {
        ply >= 2
            && static_eval != NO_STATIC_SCORE
            && self.static_evals[ply - 2] != NO_STATIC_SCORE
            && static_eval > self.static_evals[ply - 2]
    }

    /// Used for combining one- and two-ply continuation history for a move
    /// target.
    ///
    /// # Arguments
    ///
    /// * `ply` - distance from the root, selecting the prior contexts
    /// * `target` - piece-square target of the candidate move
    ///
    /// # Returns
    ///
    /// The summed continuation score, or zero without a valid context.
    fn continuation_score(&self, ply: usize, target: u16) -> i32 {
        if target == NO_CONTINUATION {
            return 0;
        }
        let mut score = 0;
        if self.path_context[ply] != NO_CONTINUATION {
            score += i32::from(
                self.continuation_one[continuation_index(self.path_context[ply], target)],
            );
        }
        if ply >= 1 && self.path_context[ply - 1] != NO_CONTINUATION {
            score += i32::from(
                self.continuation_two[continuation_index(self.path_context[ply - 1], target)],
            );
        }
        score
    }

    /// Used for rewarding the continuation causing a cutoff and penalizing
    /// alternatives.
    ///
    /// # Arguments
    ///
    /// * `ply` - distance from the root, selecting the prior contexts
    /// * `cutoff` - piece-square target of the cutoff move, if quiet
    /// * `tried` - piece-square targets of the quiets searched at this node
    /// * `depth` - remaining depth scaling the depth-squared signal
    fn update_continuation(&mut self, ply: usize, cutoff: Option<u16>, tried: &[u16], depth: i32) {
        let Some(cutoff) = cutoff else {
            return;
        };
        let bonus = depth.saturating_mul(depth).clamp(1, HISTORY_MAX);
        for target in tried {
            let delta = if *target == cutoff { bonus } else { -bonus };
            if self.path_context[ply] != NO_CONTINUATION {
                let index = continuation_index(self.path_context[ply], *target);
                self.continuation_one[index] = gravity_i16(self.continuation_one[index], delta);
            }
            if ply >= 1 && self.path_context[ply - 1] != NO_CONTINUATION {
                let index = continuation_index(self.path_context[ply - 1], *target);
                self.continuation_two[index] = gravity_i16(self.continuation_two[index], delta);
            }
        }
    }

    /// Used for prepending `mv` to the child PV and storing the resulting
    /// suffix at `ply`.
    ///
    /// # Arguments
    ///
    /// * `ply` - ply whose principal-variation suffix is replaced
    /// * `mv` - move that raised alpha at this ply
    fn update_pv(&mut self, ply: usize, mv: Move) {
        let child = self.pv[ply + 1].clone();
        self.pv[ply].clear();
        self.pv[ply].push(mv);
        self.pv[ply].extend(child);
    }
}

/// Used for tightening a static evaluation with a directionally compatible
/// TT score.
///
/// A lower bound is useful only above the evaluator's score, and an upper
/// bound only below it. Exact entries support either direction. The returned
/// value is a pruning estimate rather than a depth-qualified TT cutoff, so
/// mate-range scores are deliberately ignored: selective static pruning must
/// never turn a stored mate claim into an unverified mate result.
///
/// # Arguments
///
/// * `static_eval` - clamped static evaluation of the node
/// * `tt_score` - decoded score stored in the transposition table
/// * `bound` - bound type qualifying `tt_score`
///
/// # Returns
///
/// The tightened pruning estimate, or `static_eval` when the stored score
/// cannot help.
fn bound_aligned_tt_estimate(static_eval: i32, tt_score: i32, bound: Bound) -> i32 {
    if tt_score <= -MATE_THRESHOLD || tt_score >= MATE_THRESHOLD {
        return static_eval;
    }
    match bound {
        Bound::Exact => tt_score,
        Bound::Lower if tt_score > static_eval => tt_score,
        Bound::Upper if tt_score < static_eval => tt_score,
        Bound::Lower | Bound::Upper => static_eval,
    }
}

/// Used for deciding whether a null fail-high needs a verification search.
///
/// A nonzero minimum ply means that an outer verification is already active;
/// nested verification is suppressed until that outer call restores its guard.
///
/// # Arguments
///
/// * `depth` - remaining depth at the null-pruning node
/// * `null_move_pruning_min_ply` - active verification boundary, zero when
///   none
///
/// # Returns
///
/// `true` when the fail-high must be re-searched before it may cut off.
fn requires_null_move_verification(depth: i32, null_move_pruning_min_ply: usize) -> bool {
    depth >= NULL_MOVE_VERIFICATION_MIN_DEPTH && null_move_pruning_min_ply == 0
}

/// Used for gating the singular-extension check at one node.
///
/// The check requires a non-root node without an active exclusion, enough
/// remaining depth, a transposition entry deep enough to trust, a lower-bound
/// entry whose stored move failed high, and a non-mate stored score. The
/// caller separately requires the stored move itself to be present.
///
/// # Arguments
///
/// * `ply` - distance from the search root
/// * `exclusion_active` - whether this node already excludes a move
/// * `depth` - remaining full-width depth at the node
/// * `tt_depth` - depth recorded by the transposition entry
/// * `bound` - bound type recorded by the transposition entry
/// * `tt_score` - decoded transposition score
///
/// # Returns
///
/// `true` when the node may run a singular verification search.
fn singular_trigger_allowed(
    ply: usize,
    exclusion_active: bool,
    depth: i32,
    tt_depth: i32,
    bound: Bound,
    tt_score: i32,
) -> bool {
    ply > 0
        && !exclusion_active
        && depth >= SINGULAR_MIN_DEPTH
        && tt_depth >= depth - SINGULAR_TT_DEPTH_MARGIN
        && bound == Bound::Lower
        && tt_score.abs() < MATE_THRESHOLD
}

/// Used for gating the conservative `ProbCut` attempt at one node.
///
/// The attempt requires a non-root, non-PV node that is not in check, has no
/// active singular exclusion, retains at least [`PROBCUT_MIN_DEPTH`] of
/// full-width depth, and carries a non-mate `beta`. These mirror the gates the
/// reference engines place on `ProbCut`, and the exclusion guard disables it
/// during a singular verification exactly as reverse futility and null-move
/// pruning are disabled.
///
/// # Arguments
///
/// * `ply` - distance from the search root
/// * `pv_node` - whether this node lies on the principal variation
/// * `in_check` - whether the side to move is in check
/// * `exclusion_active` - whether this node already excludes a move
/// * `depth` - remaining full-width depth at the node
/// * `beta` - upper search bound from the moving side's view
///
/// # Returns
///
/// `true` when the node may attempt a `ProbCut` prune.
fn probcut_trigger_allowed(
    ply: usize,
    pv_node: bool,
    in_check: bool,
    exclusion_active: bool,
    depth: i32,
    beta: i32,
) -> bool {
    ply > 0
        && !pv_node
        && !in_check
        && !exclusion_active
        && depth >= PROBCUT_MIN_DEPTH
        && beta.abs() < MATE_THRESHOLD
}

/// Used for computing the descendant ply at which verification may resume
/// null pruning.
///
/// Three quarters of the reduced search depth is protected. The boundary lies
/// strictly beyond the verification root's ply for every bounded input,
/// including a zero reduced depth. Saturating arithmetic returns
/// [`usize::MAX`] for synthetic inputs at the integer limit, where no strictly
/// greater ply is representable.
///
/// # Arguments
///
/// * `ply` - ply of the verification root
/// * `reduced_depth` - reduced depth of the verified null search
///
/// # Returns
///
/// The earliest ply at which descendants may null-prune again.
fn null_move_verification_min_ply(ply: usize, reduced_depth: i32) -> usize {
    let reduced_depth = usize::try_from(reduced_depth.max(0)).unwrap_or(0);
    let protected_plies = reduced_depth.saturating_mul(3) / 4;
    ply.saturating_add(protected_plies)
        .max(ply.saturating_add(1))
}

/// Used for converting a duration to milliseconds, saturating values wider
/// than `u64`.
///
/// # Arguments
///
/// * `duration` - duration to convert
///
/// # Returns
///
/// The duration in whole milliseconds, capped at [`u64::MAX`].
fn duration_millis(duration: Duration) -> u64 {
    let millis = duration.as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}

/// Used for mapping a move's source and destination to the butterfly-history
/// array.
///
/// # Arguments
///
/// * `mv` - move whose source and destination squares are combined
///
/// # Returns
///
/// A flat index into the 64x64 butterfly-history table.
fn history_index(mv: Move) -> usize {
    usize::from(mv.from().index()) * 64 + usize::from(mv.to().index())
}

/// Search-window class used by late-move reduction.
///
/// Principal-variation nodes receive one ply less reduction than cut nodes,
/// so the classification directly shapes the reduction formula.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LmrNode {
    /// Used for indicating a full-window node on the current principal
    /// variation.
    PrincipalVariation,
    /// Used for indicating a null-window node searching for a beta cutoff.
    Cut,
}

/// Tactical class used by late-move reduction.
///
/// Tactical moves are exempt from reduction so captures and promotions always
/// retain their full search depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LmrMove {
    /// Used for indicating a non-capture without promotion.
    Quiet,
    /// Used for indicating a capture or promotion that must retain full
    /// depth.
    Tactical,
}

/// Inputs that determine the reduction of one late, already-ordered move.
///
/// The context bundles the node, move, and history signals consumed by
/// [`late_move_reduction_for`] into a single copyable value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LateMoveReductionContext {
    /// Used for recording the search-window class of the parent node.
    node: LmrNode,
    /// Used for recording whether the side to move at the parent is in
    /// check.
    in_check: bool,
    /// Used for recording the tactical class of the candidate move.
    move_kind: LmrMove,
    /// Used for recording the remaining full search depth at the parent.
    depth: i32,
    /// Used for counting the moves already searched at the parent.
    searched: u16,
    /// Used for recording the butterfly-history value of the candidate move.
    history_score: i32,
    /// Used for recording whether static evaluation improved over the
    /// same-side ancestor.
    improving: bool,
    /// Used for recording the combined one- and two-ply
    /// continuation-history value.
    continuation_score: i32,
    /// Used for recording the unreduced depth available to the candidate's
    /// child.
    child_depth: i32,
}

/// Used for selecting one late-move-reduction representation at compile time.
///
/// `FRACTIONAL` is threaded from [`AlphaBeta`]'s const parameter, so the
/// released build monomorphizes to exactly the whole-ply body it had before
/// STR-316 and the candidate build to the fractional one. Both share the same
/// full-depth exemptions, so a move exempt in one flavor is exempt in the
/// other.
///
/// # Arguments
///
/// * `context` - node, move, and history signals for the reduction decision
///
/// # Returns
///
/// The reduction in plies for the selected flavor.
fn late_move_reduction_for<const FRACTIONAL: bool, const BIAS_UNITS: i32>(
    context: LateMoveReductionContext,
) -> i32 {
    if context.in_check
        || context.move_kind == LmrMove::Tactical
        || context.depth < 3
        || context.searched < 3
    {
        return 0;
    }
    if FRACTIONAL {
        fractional_late_move_reduction(context, BIAS_UNITS)
    } else {
        released_late_move_reduction(context)
    }
}

/// Used for computing the released whole-ply late-move reduction of one
/// already-ordered quiet move.
///
/// Principal-variation nodes use one ply less reduction than cut nodes. Any
/// reduced move that raises alpha is still verified at full depth by the caller
/// before it may enter the principal variation or cause a beta cutoff. Every
/// adjustment is a whole ply, so the smallest expressible change is one ply
/// and the result can never extend.
///
/// # Arguments
///
/// * `context` - node, move, and history signals for the reduction decision
///
/// # Returns
///
/// The reduction in plies, clamped to `0..=child_depth`.
fn released_late_move_reduction(context: LateMoveReductionContext) -> i32 {
    let lmr_depth = u8::try_from(context.depth.min(63)).expect("positive depth fits u8");
    let mut reduction = i32::from(lmr_reduction(lmr_depth, context.searched));
    if context.node == LmrNode::PrincipalVariation {
        reduction -= 1;
    }
    if context.history_score >= context.depth * context.depth {
        reduction -= 1;
    }
    if !context.improving {
        reduction += 1;
    }
    if context.continuation_score >= CONTINUATION_GOOD {
        reduction -= 1;
    } else if context.continuation_score <= CONTINUATION_BAD {
        reduction += 1;
    }
    reduction.clamp(0, context.child_depth.max(0))
}

/// Used for computing the STR-316 fractional late-move reduction of one
/// already-ordered quiet move.
///
/// The signals are exactly the released ones and they are combined in the same
/// directions; only the arithmetic changes. The base curve keeps its own
/// fractional value instead of being rounded to a whole ply first, the two
/// thresholded signals become proportional terms that cross `1` ply of relief
/// exactly where the released binary terms switched, and the accumulated sum
/// is converted to plies once at the end. The result may reach
/// `-LMR_MAX_EXTENSION_PLIES`, so a late move with strong history and
/// continuation evidence can be searched deeper instead of merely unreduced;
/// the caller's re-search contract treats a non-positive reduction as an
/// already-verified search.
///
/// # Arguments
///
/// * `context` - node, move, and history signals for the reduction decision
/// * `bias_units` - fixed aggression shift in `1/1024`-ply units, positive to
///   reduce more
///
/// # Returns
///
/// The reduction in plies, clamped to
/// `-LMR_MAX_EXTENSION_PLIES..=child_depth`.
fn fractional_late_move_reduction(context: LateMoveReductionContext, bias_units: i32) -> i32 {
    let lmr_depth = u8::try_from(context.depth.min(63)).expect("positive depth fits u8");
    let mut units = lmr_reduction_units(lmr_depth, context.searched).saturating_add(bias_units);
    if context.node == LmrNode::PrincipalVariation {
        units -= LMR_UNIT;
    }
    if !context.improving {
        units += LMR_UNIT;
    }
    units -= proportional_reduction_relief(
        context.history_score,
        i64::from(context.depth) * i64::from(context.depth),
    );
    units -=
        proportional_reduction_relief(context.continuation_score, i64::from(CONTINUATION_GOOD));
    let plies = units_to_plies(units);
    plies.clamp(-LMR_MAX_EXTENSION_PLIES, context.child_depth.max(0))
}

/// Used for scaling one signed ordering statistic into reduction relief.
///
/// The result is `LMR_UNIT` when `score` reaches `pivot`, `-LMR_UNIT` when it
/// reaches `-pivot`, and proportional in between, so the term crosses one ply
/// exactly where the released binary test switched while remaining continuous
/// on both sides. Saturating at a single ply keeps any one statistic from
/// dominating the accumulated sum.
///
/// # Arguments
///
/// * `score` - signed history or continuation statistic
/// * `pivot` - statistic value worth exactly one ply of relief
///
/// # Returns
///
/// Relief in `1/1024`-ply units, within `-LMR_UNIT..=LMR_UNIT`.
fn proportional_reduction_relief(score: i32, pivot: i64) -> i32 {
    if pivot <= 0 {
        return 0;
    }
    let scaled = i64::from(score) * i64::from(LMR_UNIT) / pivot;
    i32::try_from(scaled.clamp(-i64::from(LMR_UNIT), i64::from(LMR_UNIT)))
        .expect("clamped relief fits i32")
}

/// Used for converting accumulated reduction units to whole plies.
///
/// Rounds to the nearest ply with ties away from zero, which reproduces the
/// released formula's own `round` behavior and keeps the conversion symmetric
/// for the extension side.
///
/// # Arguments
///
/// * `units` - accumulated reduction in `1/1024`-ply units
///
/// # Returns
///
/// The reduction in whole plies, before clamping.
fn units_to_plies(units: i32) -> i32 {
    let half = LMR_UNIT / 2;
    if units >= 0 {
        (units + half) / LMR_UNIT
    } else {
        (units - half) / LMR_UNIT
    }
}

/// Used for converting a bounded search ply to the mate-score
/// representation.
///
/// # Arguments
///
/// * `ply` - search ply, bounded below [`MAX_SEARCH_PLY`]
///
/// # Returns
///
/// The ply as a `u16`.
///
/// # Panics
///
/// Panics when `ply` exceeds `u16::MAX`, which bounded search plies never do.
fn ply_u16(ply: usize) -> u16 {
    u16::try_from(ply).expect("search ply is bounded below 128")
}

/// Used for converting a bounded search ply to signed score arithmetic.
///
/// # Arguments
///
/// * `ply` - search ply, bounded below [`MAX_SEARCH_PLY`]
///
/// # Returns
///
/// The ply as an `i32`.
///
/// # Panics
///
/// Panics when `ply` exceeds `i32::MAX`, which bounded search plies never do.
fn ply_i32(ply: usize) -> i32 {
    i32::try_from(ply).expect("search ply is bounded below 128")
}

/// Used for retrieving the coarse centipawn value consumed by ordering and
/// exchange search.
///
/// # Arguments
///
/// * `kind` - piece kind to value
///
/// # Returns
///
/// The kind's coarse material value in centipawns.
const fn material_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight | PieceKind::Bishop => 300,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 20_000,
    }
}

/// Used for encoding the moving piece and destination as a
/// continuation-history target.
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - move whose piece-square target is encoded
///
/// # Returns
///
/// The piece-square target, or [`NO_CONTINUATION`] when the source square is
/// empty.
fn continuation_target(position: &Position, mv: Move) -> u16 {
    position
        .piece_at(mv.from())
        .map_or(NO_CONTINUATION, |piece| {
            u16::try_from(piece.index() * 64 + usize::from(mv.to().index()))
                .expect("piece-square continuation index fits u16")
        })
}

/// Used for mapping a prior context and current target into a continuation
/// table index.
///
/// # Arguments
///
/// * `context` - piece-square context of the earlier move
/// * `target` - piece-square target of the current move
///
/// # Returns
///
/// A flat index into one continuation-history table.
fn continuation_index(context: u16, target: u16) -> usize {
    usize::from(context) * CONTINUATION_BUCKETS + usize::from(target)
}

/// Used for retaining captures and promotions while preserving generator
/// order.
///
/// # Arguments
///
/// * `position` - position the moves belong to
/// * `moves` - legal moves filtered in place to tactical moves
fn retain_tactical_moves(position: &Position, moves: &mut Vec<Move>) {
    moves.retain(|mv| position.is_capture(*mv) || mv.promotion().is_some());
}

/// Used for applying a bounded gravity update to one continuation-history
/// value.
///
/// # Arguments
///
/// * `current` - current continuation-history value
/// * `delta` - signed depth-squared update signal
///
/// # Returns
///
/// The updated value clamped to `-HISTORY_MAX..=HISTORY_MAX`.
///
/// # Panics
///
/// Panics when the clamped result does not fit `i16`, which the
/// [`HISTORY_MAX`] bound prevents.
fn gravity_i16(current: i16, delta: i32) -> i16 {
    let current = i32::from(current);
    let updated = current + delta - current * delta.abs() / HISTORY_MAX;
    i16::try_from(updated.clamp(-HISTORY_MAX, HISTORY_MAX))
        .expect("bounded continuation history fits i16")
}

/// Used for applying bounded gravity to one capture-ordering history cell.
///
/// Clamping the signal before the update keeps hostile test inputs and future
/// tuning changes within the ordering-stage invariant expressed by
/// [`CAPTURE_HISTORY_MAX`].
///
/// # Arguments
///
/// * `current` - current capture-history value
/// * `delta` - signed depth-squared update signal, clamped before use
///
/// # Returns
///
/// The updated value clamped to `-CAPTURE_HISTORY_MAX..=CAPTURE_HISTORY_MAX`.
///
/// # Panics
///
/// Panics when the clamped result does not fit `i16`, which the
/// [`CAPTURE_HISTORY_MAX`] bound prevents.
fn capture_gravity_i16(current: i16, delta: i32) -> i16 {
    let current = i32::from(current);
    let delta = delta.clamp(-CAPTURE_HISTORY_MAX, CAPTURE_HISTORY_MAX);
    let updated = current + delta - current * delta.abs() / CAPTURE_HISTORY_MAX;
    i16::try_from(updated.clamp(-CAPTURE_HISTORY_MAX, CAPTURE_HISTORY_MAX))
        .expect("bounded capture history fits i16")
}

/// Used for finding the first canonical legal move that immediately
/// checkmates.
///
/// # Arguments
///
/// * `position` - position whose moves are probed
/// * `legal_moves` - canonical legal moves to test in order
///
/// # Returns
///
/// The first mating move, or `None` when no move mates in one.
///
/// # Panics
///
/// Panics when a listed move cannot be applied to `position`.
fn find_mate_in_one(position: &Position, legal_moves: &[Move]) -> Option<Move> {
    let mut child = position.clone();
    for mv in legal_moves {
        let undo = child
            .make_move(*mv)
            .expect("root legal move is applicable to its position");
        let checkmate = child.in_check(child.side_to_move()) && child.legal_moves().is_empty();
        child.unmake_move(*mv, undo);
        if checkmate {
            return Some(*mv);
        }
    }
    None
}

/// Used for testing whether the side to move has non-pawn material for null
/// pruning.
///
/// # Arguments
///
/// * `position` - position whose side to move is inspected
///
/// # Returns
///
/// `true` when the moving side owns at least one knight, bishop, rook, or
/// queen.
fn has_null_move_material(position: &Position) -> bool {
    let color = position.side_to_move();
    [
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
    ]
    .into_iter()
    .any(|kind| position.piece_bitboard(Piece::new(color, kind)) != 0)
}

/// Used for resolving the captured piece, including an en-passant victim.
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - move whose victim is resolved
///
/// # Returns
///
/// The captured piece, or `None` when `mv` captures nothing.
fn captured_piece(position: &Position, mv: Move) -> Option<Piece> {
    if let Some(piece) = position.piece_at(mv.to()) {
        return Some(piece);
    }
    let moving = position.piece_at(mv.from())?;
    if moving.kind != PieceKind::Pawn
        || position.en_passant_square() != Some(mv.to())
        || mv.from().file() == mv.to().file()
    {
        return None;
    }
    let captured = Square::from_file_row(mv.to().file(), mv.from().row())?;
    position.piece_at(captured)
}

/// Used for mapping a legal capture to its colored-mover, destination, and
/// victim bucket.
///
/// The victim lookup delegates to [`captured_piece`], so ordinary captures and
/// en passant use the same stable six-kind victim dimension. Non-captures and
/// malformed move/position pairs return `None` instead of touching history.
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - candidate capture to bucket
///
/// # Returns
///
/// The capture-history bucket index, or `None` for a non-capture.
fn capture_history_index(position: &Position, mv: Move) -> Option<usize> {
    let moving = position.piece_at(mv.from())?;
    let victim = captured_piece(position, mv)?;
    Some((moving.index() * 64 + usize::from(mv.to().index())) * 6 + victim.kind.index())
}

/// Used for retrieving the coarse material value captured by `mv`, or zero.
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - move whose victim is valued
///
/// # Returns
///
/// The victim's coarse centipawn value, or zero for a non-capture.
fn captured_value(position: &Position, mv: Move) -> i32 {
    captured_piece(position, mv).map_or(0, |piece| material_value(piece.kind))
}

/// Returns whether a main-search node may discard a proven losing capture.
///
/// The move-specific SEE and checking-move tests remain outside this policy:
/// SEE must inspect the parent, while check status is known only after the
/// legal move is made. Root/PV work, check evasions, the first searched move,
/// deeper nodes, and mate-score windows are deliberately protected.
fn main_search_losing_capture_pruning_allowed(
    depth: i32,
    ply: usize,
    pv_node: bool,
    in_check: bool,
    searched: u16,
    alpha: i32,
    beta: i32,
) -> bool {
    depth > 0
        && depth <= FUTILITY_MAX_DEPTH
        && ply > 0
        && !pv_node
        && !in_check
        && searched > 0
        && alpha.abs() < MATE_THRESHOLD
        && beta.abs() < MATE_THRESHOLD
}

/// Used for gating losing captures with Java-parity semantics.
///
/// Promotions and en-passant are never discarded as losing captures. For an
/// ordinary capture, SEE is only needed when the victim is cheaper than the
/// attacker; every capture of an equal or more valuable piece is non-losing by
/// construction under the standing-pat exchange rule.
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - candidate capture to classify
///
/// # Returns
///
/// `true` when static exchange evaluation proves the capture loses material.
fn is_losing_capture(position: &Position, mv: Move) -> bool {
    if mv.promotion().is_some() {
        return false;
    }
    let Some(moving) = position.piece_at(mv.from()) else {
        return false;
    };
    let Some(victim) = position.piece_at(mv.to()) else {
        // En-passant lands on an empty square.
        return false;
    };
    if victim.color == moving.color || material_value(victim.kind) >= material_value(moving.kind) {
        return false;
    }
    static_exchange_eval(position, mv) < 0
}

/// Used for computing conventional pin-agnostic static exchange evaluation in
/// centipawns.
///
/// The exchange is simulated on a mailbox copy of the board with least
/// valuable attackers recapturing first, including pawn promotions to queens
/// on the back ranks, and the gain sequence is minimaxed backwards. An
/// occupancy bitboard is carried alongside the mailbox so each recapture scan
/// visits only occupied squares that can geometrically reach the contested
/// square, rather than all sixty-four.
///
/// # Arguments
///
/// * `position` - position the capture belongs to
/// * `mv` - capture whose exchange balance is evaluated
///
/// # Returns
///
/// The exchange balance in centipawns from the capturer's view.
///
/// # Panics
///
/// Panics when an en-passant victim square cannot be reconstructed, which
/// legal en-passant moves prevent.
fn static_exchange_eval(position: &Position, mv: Move) -> i32 {
    let Some(moving) = position.piece_at(mv.from()) else {
        return 0;
    };
    let Some(victim) = captured_piece(position, mv) else {
        return 0;
    };
    let capture_square = if position.piece_at(mv.to()).is_some() {
        mv.to()
    } else {
        Square::from_file_row(mv.to().file(), mv.from().row())
            .expect("en-passant captured square is on board")
    };

    // Only occupied squares can hold an attacker, so the exchange walks the
    // occupancy bitboard rather than all sixty-four squares. Empty squares
    // contributed nothing to either loop, so the simulated exchange is
    // unchanged.
    let mut board = [None; 64];
    let mut occupied = position.occupancy();
    let mut remaining = occupied;
    while remaining != 0 {
        let index = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let square = Square::new(u8::try_from(index).expect("bit index is a square"))
            .expect("occupancy bit is a square");
        board[index as usize] = position.piece_at(square);
    }
    board[mv.from().index() as usize] = None;
    occupied &= !(1_u64 << mv.from().index());
    board[capture_square.index() as usize] = None;
    occupied &= !(1_u64 << capture_square.index());
    let placed_kind = mv.promotion().unwrap_or(moving.kind);
    board[mv.to().index() as usize] = Some(Piece::new(moving.color, placed_kind));
    occupied |= 1_u64 << mv.to().index();

    let mut gains = [0_i32; 64];
    let mut gain_count = 1_usize;
    let promotion_gain = mv.promotion().map_or(0, |kind| {
        material_value(kind) - material_value(PieceKind::Pawn)
    });
    gains[0] = material_value(victim.kind) + promotion_gain;
    let mut occupant = placed_kind;
    let mut side = moving.color.opposite();

    while let Some((attacker_square, attacker)) = least_valuable_attacker(
        &board,
        occupied & potential_attacker_mask(mv.to()),
        mv.to(),
        side,
    ) {
        let next = material_value(occupant) - gains[gain_count - 1];
        let promoted = attacker.kind == PieceKind::Pawn
            && ((side == Color::White && mv.to().row() == 0)
                || (side == Color::Black && mv.to().row() == 7));
        let next_occupant = if promoted {
            PieceKind::Queen
        } else {
            attacker.kind
        };
        let next = if promoted {
            next + material_value(PieceKind::Queen) - material_value(PieceKind::Pawn)
        } else {
            next
        };
        gains[gain_count] = next;
        gain_count += 1;
        board[attacker_square.index() as usize] = None;
        occupied &= !(1_u64 << attacker_square.index());
        board[mv.to().index() as usize] = Some(Piece::new(side, next_occupant));
        occupied |= 1_u64 << mv.to().index();
        occupant = next_occupant;
        side = side.opposite();
    }

    for index in (1..gain_count).rev() {
        gains[index - 1] = -(-gains[index - 1]).max(gains[index]);
    }
    gains[0]
}

/// Used for masking the squares from which any piece could geometrically
/// reach a target square.
///
/// A square off every rook ray, bishop ray, knight jump and king step from the
/// target can never attack it under any occupancy, so the static exchange
/// simulation can skip it without consulting [`piece_attacks`]. Pawn attack
/// squares are a subset of the king pattern, so they need no separate term.
///
/// The table is built once on first use and indexed by target square.
///
/// # Arguments
///
/// * `target` - square being fought over
///
/// # Returns
///
/// Bitboard of every square that could hold an attacker of `target`.
#[inline]
fn potential_attacker_mask(target: Square) -> u64 {
    static MASKS: std::sync::OnceLock<[u64; 64]> = std::sync::OnceLock::new();
    let masks = MASKS.get_or_init(|| {
        let mut table = [0_u64; 64];
        for (index, slot) in table.iter_mut().enumerate() {
            let file = i16::try_from(index % 8).expect("file fits i16");
            let row = i16::try_from(index / 8).expect("row fits i16");
            let mut mask = 0_u64;
            for other in 0_usize..64 {
                if other == index {
                    continue;
                }
                let other_file = i16::try_from(other % 8).expect("file fits i16");
                let other_row = i16::try_from(other / 8).expect("row fits i16");
                let file_delta = file - other_file;
                let row_delta = row - other_row;
                let straight = file_delta == 0 || row_delta == 0;
                let diagonal = file_delta.abs() == row_delta.abs();
                let knight = matches!((file_delta.abs(), row_delta.abs()), (1, 2) | (2, 1));
                let king = file_delta.abs().max(row_delta.abs()) == 1;
                if straight || diagonal || knight || king {
                    mask |= 1_u64 << other;
                }
            }
            *slot = mask;
        }
        table
    });
    masks[target.index() as usize]
}

/// Used for finding the least valuable geometrical attacker of `target` for
/// `side`.
///
/// Ties in material value resolve to the lowest square index, keeping the
/// exchange simulation deterministic.
///
/// # Arguments
///
/// * `board` - mailbox board state of the simulated exchange
/// * `occupied` - squares still holding a piece, already narrowed to those
///   that can geometrically reach `target`; empty squares cannot hold an
///   attacker, so the scan visits only set bits
/// * `target` - square being fought over
/// * `side` - color whose attackers are considered
///
/// # Returns
///
/// The attacker's square and piece, or `None` when no attacker remains.
fn least_valuable_attacker(
    board: &[Option<Piece>; 64],
    occupied: u64,
    target: Square,
    side: Color,
) -> Option<(Square, Piece)> {
    let mut best: Option<(Square, Piece, i32)> = None;
    let mut remaining = occupied;
    while remaining != 0 {
        let index = u8::try_from(remaining.trailing_zeros()).expect("bit index is a square");
        remaining &= remaining - 1;
        let square = Square::new(index).expect("occupancy bit is a square");
        let Some(piece) = board[index as usize] else {
            continue;
        };
        if piece.color != side || !piece_attacks(board, square, target, piece) {
            continue;
        }
        let value = material_value(piece.kind);
        if best.map_or(true, |(best_square, _, best_value)| {
            value < best_value || (value == best_value && square < best_square)
        }) {
            best = Some((square, piece, value));
        }
    }
    best.map(|(square, piece, _)| (square, piece))
}

/// Used for testing a piece's geometric attack on a mailbox board without
/// pin filtering.
///
/// # Arguments
///
/// * `board` - mailbox board state of the simulated exchange
/// * `from` - square the piece attacks from
/// * `target` - square being attacked
/// * `piece` - piece whose movement pattern is applied
///
/// # Returns
///
/// `true` when the piece geometrically attacks `target` from `from`.
fn piece_attacks(board: &[Option<Piece>; 64], from: Square, target: Square, piece: Piece) -> bool {
    let file_delta = i16::from(target.file()) - i16::from(from.file());
    let row_delta = i16::from(target.row()) - i16::from(from.row());
    match piece.kind {
        PieceKind::Pawn => {
            row_delta == if piece.color == Color::White { -1 } else { 1 } && file_delta.abs() == 1
        }
        PieceKind::Knight => matches!((file_delta.abs(), row_delta.abs()), (1, 2) | (2, 1)),
        PieceKind::King => file_delta.abs().max(row_delta.abs()) == 1,
        PieceKind::Bishop => {
            file_delta.abs() == row_delta.abs()
                && ray_is_clear(board, from, target, file_delta.signum(), row_delta.signum())
        }
        PieceKind::Rook => {
            (file_delta == 0 || row_delta == 0)
                && ray_is_clear(board, from, target, file_delta.signum(), row_delta.signum())
        }
        PieceKind::Queen => {
            (file_delta == 0 || row_delta == 0 || file_delta.abs() == row_delta.abs())
                && ray_is_clear(board, from, target, file_delta.signum(), row_delta.signum())
        }
    }
}

/// Used for testing whether every intermediate square on a slider ray is
/// empty.
///
/// # Arguments
///
/// * `board` - mailbox board state of the simulated exchange
/// * `from` - square the slider starts from
/// * `target` - square the ray must reach
/// * `file_step` - per-square file increment of the ray
/// * `row_step` - per-square row increment of the ray
///
/// # Returns
///
/// `true` when the ray reaches `target` without hitting a piece or leaving
/// the board.
fn ray_is_clear(
    board: &[Option<Piece>; 64],
    from: Square,
    target: Square,
    file_step: i16,
    row_step: i16,
) -> bool {
    if file_step == 0 && row_step == 0 {
        return false;
    }
    let mut file = i16::from(from.file()) + file_step;
    let mut row = i16::from(from.row()) + row_step;
    while file != i16::from(target.file()) || row != i16::from(target.row()) {
        if !(0..8).contains(&file) || !(0..8).contains(&row) {
            return false;
        }
        let index = usize::try_from(row * 8 + file).expect("checked ray square is non-negative");
        if board[index].is_some() {
            return false;
        }
        file += file_step;
        row += row_step;
    }
    true
}

