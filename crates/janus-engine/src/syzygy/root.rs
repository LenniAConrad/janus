//! Root-move ranking and filtering with DTZ tables and a WDL fallback.
//!
//! When the root position is inside the available tables the engine should
//! never leave the proven result: every root move is ranked by the
//! distance-to-zeroing behind it (respecting the fifty-move counter), and
//! only the moves sharing the best rank are searched. When one or more DTZ
//! files are missing the ranking falls back to pure WDL, which preserves
//! the game-theoretic outcome but not the fastest conversion. When even a
//! WDL probe fails the filter reports `None` and the search runs
//! unchanged.

use super::probe::{dtz_before_zeroing, Prober};
use super::Wdl;
use janus_core::{Move, Position};

/// Rank magnitude of a proven win, mirroring the conventional ranking
/// scale so cursed and real results never interleave.
const MAX_DTZ_RANK: i32 = 1 << 18;

/// Result of ranking every root move against the tables.
///
/// Carries the outcome-preserving root subset in canonical legal-move
/// order plus the cardinality interior probing should continue with.
pub struct RootFilter {
    /// Used for restricting the root search to outcome-preserving moves.
    pub moves: Vec<Move>,
    /// Used for continuing interior probing: zero when the root ranking
    /// already pins the game course (DTZ present, or not winning).
    pub search_cardinality: usize,
}

impl Prober {
    /// Used for ranking and filtering the root moves of an in-table
    /// position.
    ///
    /// Probing is attempted when the position fits the configured
    /// cardinality and has no castling rights. Each move is ranked by DTZ
    /// counted from the root (zeroing moves take the WDL verdict behind
    /// them); if any DTZ probe fails the whole ranking falls back to WDL.
    /// Only the moves sharing the best rank are kept.
    ///
    /// A known bound of this implementation: repetitions since the last
    /// zeroing move are not consulted, so with a high halfmove clock the
    /// ranking trusts the fifty-move arithmetic alone. This is safe with
    /// `Syzygy50MoveRule` enabled because ranks already discount the
    /// clock.
    ///
    /// # Arguments
    ///
    /// * `root` - root position (cloned internally for probing)
    /// * `legal` - canonical legal root moves, possibly pre-restricted
    ///
    /// # Returns
    ///
    /// The filter when every move was ranked, or `None` when the position
    /// is outside the tables or a fallback probe failed.
    #[must_use]
    pub fn filter_root_moves(&mut self, root: &Position, legal: &[Move]) -> Option<RootFilter> {
        let piece_count = usize::try_from(root.occupancy().count_ones()).unwrap_or(usize::MAX);
        if legal.is_empty()
            || self.cardinality() == 0
            || root.castling_rights().bits() != 0
            || piece_count > self.cardinality()
        {
            return None;
        }

        let (ranks, dtz_available) = match self.rank_with_dtz(root, legal) {
            Some(ranks) => (ranks, true),
            None => (self.rank_with_wdl(root, legal)?, false),
        };

        let best = ranks.iter().copied().max()?;
        let moves: Vec<Move> = legal
            .iter()
            .zip(&ranks)
            .filter(|(_, rank)| **rank == best)
            .map(|(mv, _)| *mv)
            .collect();
        // Keep probing inside the search only when the win must still be
        // steered home without DTZ guidance.
        let search_cardinality = if dtz_available || best <= 0 {
            0
        } else {
            self.cardinality()
        };
        Some(RootFilter {
            moves,
            search_cardinality,
        })
    }

    /// Used for ranking every root move by DTZ counted from the root.
    ///
    /// # Arguments
    ///
    /// * `root` - root position
    /// * `legal` - legal root moves to rank
    ///
    /// # Returns
    ///
    /// One rank per move, or `None` when any probe failed.
    fn rank_with_dtz(&mut self, root: &Position, legal: &[Move]) -> Option<Vec<i32>> {
        let cnt50 = i32::from(root.halfmove_clock());
        let mut position = root.clone();
        let mut ranks = Vec::with_capacity(legal.len());
        for mv in legal {
            let undo = position.make_move(*mv).ok()?;
            let dtz = self.move_dtz(&mut position);
            position.unmake_move(*mv, undo);
            let dtz = dtz?;
            ranks.push(rank_from_dtz(dtz, cnt50));
        }
        Some(ranks)
    }

    /// Used for computing one root move's DTZ from the root's viewpoint.
    ///
    /// # Arguments
    ///
    /// * `position` - position after the root move; restored by the caller
    ///
    /// # Returns
    ///
    /// The signed DTZ, or `None` when a probe failed.
    fn move_dtz(&mut self, position: &mut Position) -> Option<i32> {
        let mut dtz = if position.halfmove_clock() == 0 {
            // The move zeroed the clock: its DTZ is decided by the verdict
            // it reached, one of -101/-1/0/1/101.
            let (wdl, _) = self.wdl_search(position, false).ok()?;
            dtz_before_zeroing(wdl.negate())
        } else if let Some(outcome) = rule_boundary_outcome(position) {
            dtz_before_zeroing(outcome.negate())
        } else {
            let below = self.dtz_search(position).ok()?;
            let dtz = -below;
            // One ply passed since the root.
            match dtz.cmp(&0) {
                std::cmp::Ordering::Greater => dtz + 1,
                std::cmp::Ordering::Less => dtz - 1,
                std::cmp::Ordering::Equal => 0,
            }
        };
        // A checkmating move always ranks as distance one.
        if dtz == 2
            && position.in_check(position.side_to_move())
            && position.legal_moves().is_empty()
        {
            dtz = 1;
        }
        Some(dtz)
    }

    /// Used for ranking every root move by pure WDL when DTZ is missing.
    ///
    /// # Arguments
    ///
    /// * `root` - root position
    /// * `legal` - legal root moves to rank
    ///
    /// # Returns
    ///
    /// One rank per move, or `None` when any probe failed.
    pub(super) fn rank_with_wdl(&mut self, root: &Position, legal: &[Move]) -> Option<Vec<i32>> {
        let mut position = root.clone();
        let mut ranks = Vec::with_capacity(legal.len());
        for mv in legal {
            let undo = position.make_move(*mv).ok()?;
            let wdl = if let Some(outcome) = rule_boundary_outcome(&position) {
                Some(outcome.negate())
            } else {
                self.wdl_search(&mut position, false)
                    .ok()
                    .map(|(wdl, _)| wdl.negate())
            };
            position.unmake_move(*mv, undo);
            ranks.push(rank_from_wdl(wdl?));
        }
        Some(ranks)
    }
}

/// Used for testing whether a position is already drawn on the board.
///
/// Covers the fifty-move rule and insufficient material; repetition
/// history is intentionally out of scope here (see
/// [`Prober::filter_root_moves`]).
///
/// # Arguments
///
/// * `position` - position reached by a candidate root move
///
/// # Returns
///
/// `true` when the position is a rule draw regardless of the tables.
fn drawn_on_the_board(position: &Position) -> bool {
    position.halfmove_clock() >= 100 || position.is_insufficient_material()
}

/// Used for adjudicating a child caught by the rule-draw predicate without
/// allowing that predicate to override checkmate.
///
/// The inexpensive clock/material test runs first, so ordinary root children
/// do not generate legal moves here. At the boundary, a checked side with no
/// legal reply has already lost; every other admitted position is drawn.
///
/// # Arguments
///
/// * `position` - position after one candidate root move
///
/// # Returns
///
/// The side-to-move outcome at a rule boundary, or `None` when table probing
/// should decide the position.
fn rule_boundary_outcome(position: &Position) -> Option<Wdl> {
    if !drawn_on_the_board(position) {
        return None;
    }
    if position.in_check(position.side_to_move()) && position.legal_moves().is_empty() {
        Some(Wdl::Loss)
    } else {
        Some(Wdl::Draw)
    }
}

/// Used for converting a root move's DTZ into its rank.
///
/// Certain wins rank at the top band; wins that risk the fifty-move rule
/// rank by how much clock margin remains; losses mirror the same scheme
/// below zero and draws rank zero.
///
/// # Arguments
///
/// * `dtz` - move's signed DTZ from the root
/// * `cnt50` - root position's halfmove clock
///
/// # Returns
///
/// The move's rank; higher is better.
pub(crate) fn rank_from_dtz(dtz: i32, cnt50: i32) -> i32 {
    match dtz.cmp(&0) {
        std::cmp::Ordering::Greater => {
            if dtz + cnt50 <= 99 {
                MAX_DTZ_RANK
            } else {
                MAX_DTZ_RANK / 2 - (dtz + cnt50)
            }
        }
        std::cmp::Ordering::Less => {
            if -dtz * 2 + cnt50 < 100 {
                -MAX_DTZ_RANK
            } else {
                -MAX_DTZ_RANK / 2 + (-dtz + cnt50)
            }
        }
        std::cmp::Ordering::Equal => 0,
    }
}

/// Used for converting a root move's WDL verdict into its rank.
///
/// # Arguments
///
/// * `wdl` - verdict from the root mover's point of view
///
/// # Returns
///
/// The move's rank on the same scale as [`rank_from_dtz`].
pub(crate) fn rank_from_wdl(wdl: Wdl) -> i32 {
    match wdl {
        Wdl::Loss => -MAX_DTZ_RANK,
        Wdl::BlessedLoss => -MAX_DTZ_RANK + 101,
        Wdl::Draw => 0,
        Wdl::CursedWin => MAX_DTZ_RANK - 101,
        Wdl::Win => MAX_DTZ_RANK,
    }
}
