//! Shared value and policy evaluation contracts.
//!
//! Scores exposed through [`Evaluator::evaluate`] are centipawns from the
//! side-to-move perspective. MCTS uses [`PolicyValue`] so neural backends can
//! supply policy logits without changing the search interface. All floating
//! outputs are sanitized at the public constructor boundary.

use janus_core::{Color, Move, Piece, PieceKind, Position, Undo};
use std::error::Error;
use std::fmt;

/// A combined policy/value prediction from the side-to-move perspective.
///
/// Both public constructors keep `value` and `draw` inside their documented
/// ranges, so downstream MCTS code never sees an unbounded or non-finite
/// prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyValue {
    /// Used for reporting the expected game result in `[-1, 1]`.
    ///
    /// A value of `1` is a win for the side to move.
    pub value: f32,
    /// Used for reporting the estimated draw probability in `[0, 1]`.
    pub draw: f32,
    /// Used for supplying optional raw policy logits keyed by legal moves.
    ///
    /// Missing legal moves receive a small floor probability.  An empty list
    /// asks MCTS to construct deterministic handcrafted priors instead.
    pub policy_logits: Vec<(Move, f32)>,
}

impl PolicyValue {
    /// Used for converting a centipawn score to the bounded value convention
    /// used by MCTS.
    ///
    /// Inputs are clamped to `[-2000, 2000]` before a smooth `tanh`
    /// conversion. The heuristic draw estimate decreases as the absolute
    /// value grows, and no policy logits are supplied.
    ///
    /// # Arguments
    ///
    /// * `centipawns` - score in centipawns from the side to move
    ///
    /// # Returns
    ///
    /// A prediction with a `tanh`-bounded value, a heuristic draw estimate,
    /// and an empty logit list.
    #[must_use]
    pub fn from_centipawns(centipawns: i32) -> Self {
        let bounded = centipawns.clamp(-2_000, 2_000);
        let bounded = i16::try_from(bounded).unwrap_or_default();
        let value = (f32::from(bounded) / 600.0).tanh();
        let draw = (0.35_f32 - value.abs() * 0.25).max(0.0);
        Self {
            value,
            draw,
            policy_logits: Vec::new(),
        }
    }

    /// Used for building a prediction with explicit policy logits.
    ///
    /// Value and draw outputs are clamped to their documented ranges.
    /// Non-finite values become zero. Logits remain unmodified because MCTS
    /// filters invalid entries while matching them to the current legal move
    /// list.
    ///
    /// # Arguments
    ///
    /// * `value` - expected game result, sanitized into `[-1, 1]`
    /// * `draw` - draw probability, sanitized into `[0, 1]`
    /// * `policy_logits` - raw policy logits keyed by moves
    ///
    /// # Returns
    ///
    /// A prediction with sanitized value and draw and the given logits.
    #[must_use]
    pub fn with_logits(value: f32, draw: f32, policy_logits: Vec<(Move, f32)>) -> Self {
        Self {
            value: finite_clamp(value, -1.0, 1.0, 0.0),
            draw: finite_clamp(draw, 0.0, 1.0, 0.0),
            policy_logits,
        }
    }
}

/// Position evaluation contract for MCTS and general position scoring.
///
/// Implementations may keep reusable scratch state, hence the mutable
/// receiver. The trait is `Send` so the search coordinator can own one
/// evaluator per worker.
pub trait Evaluator: Send {
    /// Used for scoring a position in centipawns from the side-to-move
    /// perspective.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Centipawn score from the side to move.
    fn evaluate(&mut self, position: &Position) -> i32;

    /// Used for returning an optional move-ordering term consumed by
    /// handcrafted MCTS priors.
    ///
    /// The default contributes no evaluator-specific ordering information.
    ///
    /// # Arguments
    ///
    /// * `_position` - position the move would be played in
    /// * `_mv` - move to score for ordering
    ///
    /// # Returns
    ///
    /// Evaluator-specific ordering term; the default returns zero.
    fn move_ordering_score(&mut self, _position: &Position, _mv: Move) -> i32 {
        0
    }

    /// Used for resetting evaluator-owned temporal state at the start of one
    /// MCTS path.
    ///
    /// History-aware networks can retain the root's pre-search positions and
    /// rebuild their active eight-position window here. Stateless evaluators
    /// need no work.
    ///
    /// # Arguments
    ///
    /// * `_root` - root position of the simulation about to descend
    fn begin_mcts_path(&mut self, _root: &Position) {}

    /// Used for appending a position reached while descending the current
    /// MCTS path.
    ///
    /// Calls are ordered from the root's first child to the leaf and are
    /// preceded by exactly one [`Self::begin_mcts_path`] per simulation.
    ///
    /// # Arguments
    ///
    /// * `_position` - position just reached on the current path
    fn mcts_path_position(&mut self, _position: &Position) {}

    /// Used for indicating whether MCTS may spend additional evaluator calls
    /// on leaf tactics.
    ///
    /// Cheap handcrafted and compact evaluators normally benefit from the
    /// bounded quiescence extension. Very large policy/value networks can
    /// return `false` so their already-computed leaf value is backed up
    /// directly after terminal and mate-in-one checks. This is a latency
    /// capability, not an evaluation-quality claim.
    ///
    /// # Returns
    ///
    /// `true` when the bounded MCTS quiescence extension is worthwhile; the
    /// default allows it.
    fn allows_mcts_quiescence(&self) -> bool {
        true
    }

    /// Used for returning a combined value and optional neural policy
    /// prediction.
    ///
    /// The default derives a bounded value from [`Self::evaluate`] and
    /// supplies no policy logits.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `_legal_moves` - legal moves available in `position`
    ///
    /// # Returns
    ///
    /// A [`PolicyValue`] prediction for the position.
    fn evaluate_policy_value(&mut self, position: &Position, _legal_moves: &[Move]) -> PolicyValue {
        PolicyValue::from_centipawns(self.evaluate(position))
    }
}

/// Failure to initialize evaluator-owned state for one alpha-beta search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchStateError {
    /// Used for indicating that bounded evaluator scratch could not be
    /// allocated.
    ResourceExhausted,
}

impl fmt::Display for SearchStateError {
    /// Used for writing the bounded evaluator lifecycle diagnostic.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination for the rendered message
    ///
    /// # Returns
    ///
    /// Propagated formatter success or failure.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceExhausted => {
                formatter.write_str("evaluator search-state allocation failed")
            }
        }
    }
}

impl Error for SearchStateError {}

/// Narrow evaluator contract consumed by alpha-beta search.
///
/// Implementations must be position-pure for the duration of a search: calling
/// [`SearchEvaluator::evaluate`] again for the same complete position must
/// return the same score. The mutable receiver may own scratch buffers or
/// internal caches, but [`SearchEvaluator::quiet_move_order_prior`] must not
/// depend on side effects from the most recent evaluation. Alpha-beta relies on
/// this contract when it reuses exact-key static evaluations.
pub trait SearchEvaluator: Send {
    /// Used for scoring a position in centipawns from the side-to-move
    /// perspective.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Centipawn score from the side to move.
    fn evaluate(&mut self, position: &Position) -> i32;

    /// Used for initializing optional evaluator-owned state for one root
    /// search.
    ///
    /// Stateless evaluators retain the default no-op implementation.
    ///
    /// # Arguments
    ///
    /// * `_root` - root position of the search
    /// * `_max_plies` - bound on every `ply` later supplied to the lifecycle
    ///   hooks
    ///
    /// # Returns
    ///
    /// Success after any evaluator-owned state is ready.
    ///
    /// # Errors
    ///
    /// Returns [`SearchStateError::ResourceExhausted`] when bounded scratch
    /// cannot be admitted.
    fn begin_search(
        &mut self,
        _root: &Position,
        _max_plies: usize,
    ) -> Result<(), SearchStateError> {
        Ok(())
    }

    /// Used for reporting whether the just-opened search uses move-aligned
    /// evaluator state.
    ///
    /// Alpha-beta still calls [`Self::begin_search`], but skips the per-move
    /// lifecycle hooks and the ply-aware dispatch when this is false, keeping
    /// simple evaluators and sparse low-material NNUE on their minimal
    /// position-only path.
    ///
    /// # Returns
    ///
    /// `true` when the lifecycle hooks must be called; the default is
    /// `false`.
    fn uses_incremental_search_state(&self) -> bool {
        false
    }

    /// Used for advancing optional evaluator-owned state after one normal
    /// move.
    ///
    /// The position is the already-made child and `undo` is the exact record
    /// returned by [`Position::make_move`]. Search calls this for every move,
    /// including nodes whose static score is subsequently served from cache.
    ///
    /// # Arguments
    ///
    /// * `_child` - position after the move has been made
    /// * `_mv` - move that was played
    /// * `_undo` - undo record returned by [`Position::make_move`]
    /// * `_ply` - root-relative ply of the child position
    fn move_played(&mut self, _child: &Position, _mv: Move, _undo: Undo, _ply: usize) {}

    /// Used for advancing optional evaluator-owned state after a search-only
    /// null move.
    ///
    /// # Arguments
    ///
    /// * `_ply` - root-relative ply reached by the null move
    fn null_move_played(&mut self, _ply: usize) {}

    /// Used for evaluating the current search position at an explicit
    /// root-relative ply.
    ///
    /// Incremental evaluators override this method. The default preserves the
    /// original position-only evaluator contract.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `_ply` - root-relative ply of the position
    ///
    /// # Returns
    ///
    /// Centipawn score from the side to move.
    fn evaluate_at(&mut self, position: &Position, _ply: usize) -> i32 {
        self.evaluate(position)
    }

    /// Used for returning an evaluator-owned ordering prior for one quiet
    /// move.
    ///
    /// Implementations must return zero for captures and promotions. The
    /// search combines this small positional hint with its
    /// transposition-table, killer, and history scores; evaluators without a
    /// useful prior keep the default.
    ///
    /// # Arguments
    ///
    /// * `_position` - position the quiet move would be played in
    /// * `_mv` - quiet move to score
    ///
    /// # Returns
    ///
    /// Small positional ordering prior; the default returns zero.
    fn quiet_move_order_prior(&self, _position: &Position, _mv: Move) -> i32 {
        0
    }
}

/// A stateless dependency-free material-only evaluator.
///
/// This deliberately simple implementation is useful for tests and as a
/// predictable fallback; [`crate::Classical`] is the normal handcrafted engine
/// evaluator.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialEvaluator;

impl Evaluator for MaterialEvaluator {
    /// Used for scoring the position as a plain material count in centipawns.
    ///
    /// Piece counts are weighted by [`piece_value`] and the White-minus-Black
    /// balance is negated when Black is to move. The `expect` calls cannot
    /// fail because a bitboard population never exceeds 64.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Material balance in centipawns from the side to move.
    fn evaluate(&mut self, position: &Position) -> i32 {
        let mut white = 0_i32;
        let mut black = 0_i32;
        for kind in PieceKind::ALL {
            let value = piece_value(kind);
            let white_count = i32::try_from(
                position
                    .piece_bitboard(Piece::new(Color::White, kind))
                    .count_ones(),
            )
            .expect("a bitboard population fits i32");
            let black_count = i32::try_from(
                position
                    .piece_bitboard(Piece::new(Color::Black, kind))
                    .count_ones(),
            )
            .expect("a bitboard population fits i32");
            white += value * white_count;
            black += value * black_count;
        }
        let white_score = white - black;
        match position.side_to_move() {
            Color::White => white_score,
            Color::Black => -white_score,
        }
    }
}

/// Used for returning the standard material value consumed by the fallback
/// evaluator and priors.
///
/// Kings have value zero here because they are not capturable. Static
/// exchange code that requires a finite king value defines that search-local
/// convention separately.
///
/// # Arguments
///
/// * `kind` - piece kind to look up
///
/// # Returns
///
/// Material value in centipawns.
#[must_use]
pub const fn piece_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight | PieceKind::Bishop => 300,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 0,
    }
}

/// Used for clamping finite values and substituting `fallback` for NaN or
/// infinity.
///
/// # Arguments
///
/// * `value` - value to sanitize
/// * `minimum` - lower clamp bound
/// * `maximum` - upper clamp bound
/// * `fallback` - replacement for non-finite input
///
/// # Returns
///
/// The clamped value, or `fallback` when the input is not finite.
fn finite_clamp(value: f32, minimum: f32, maximum: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(minimum, maximum)
    } else {
        fallback
    }
}

