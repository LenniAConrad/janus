//! Search score conventions shared by all Janus searchers.
//!
//! Scores are centipawns from the side to move. Forced mates are encoded near
//! [`MATE_SCORE`], static evaluations are clamped strictly below the mate
//! band, and transposition-table storage rebases mate distances between the
//! root-relative and position-relative frames.

/// Used for bounding the supported search depth in plies.
///
/// Search limits are validated against this bound before a search starts.
pub const MAX_DEPTH: u8 = 64;

/// Used for encoding a forced mate as a base score.
///
/// A mate in `n` plies from the current frame is scored `MATE_SCORE - n` for
/// the winning side and `-(MATE_SCORE - n)` for the side being mated.
pub const MATE_SCORE: i32 = 900_000;

/// Used for classifying scores whose absolute value marks a forced mate.
///
/// The 1,000-point band below [`MATE_SCORE`] leaves room for mate distances
/// while remaining far above [`MAX_STATIC_SCORE`].
pub const MATE_THRESHOLD: i32 = MATE_SCORE - 1_000;

/// Used for initializing search windows as an unattainable bound.
///
/// Static and mate scores remain strictly inside it, so `-INFINITY` and
/// `INFINITY` can serve as fully open alpha-beta window bounds.
pub const INFINITY: i32 = 1_000_000;

/// Used for encoding a proven tablebase win as a base score.
///
/// A tablebase win discovered at ply `n` scores `TB_WIN_SCORE - n`, so
/// nearer conversions order first while every tablebase score stays
/// strictly below [`MATE_THRESHOLD`]: a concrete forced mate found by the
/// search always outranks a tablebase verdict, preserving mate-distance
/// pruning.
pub const TB_WIN_SCORE: i32 = 800_000;

/// Used for classifying scores whose absolute value marks a tablebase
/// verdict or better.
///
/// The 1,000-point band below [`TB_WIN_SCORE`] mirrors the mate band and
/// comfortably covers every reachable search ply, while staying far above
/// [`MAX_STATIC_SCORE`].
pub const TB_WIN_THRESHOLD: i32 = TB_WIN_SCORE - 1_000;

/// Used for bounding the largest accepted non-mate evaluator score.
///
/// [`clamp_static_score`] enforces this bound so no static evaluation can
/// reach the mate band at [`MATE_THRESHOLD`].
pub const MAX_STATIC_SCORE: i32 = 100_000;

/// Used for testing whether `score` encodes a forced mate.
///
/// A score is a mate when its absolute value reaches [`MATE_THRESHOLD`].
///
/// # Arguments
///
/// * `score` - search score in centipawns
///
/// # Returns
///
/// `true` when the score lies in the positive or negative mate band.
#[inline]
#[must_use]
pub const fn is_mate_score(score: i32) -> bool {
    score >= MATE_THRESHOLD || score <= -MATE_THRESHOLD
}

/// Used for clamping an evaluator result to the non-mate score range.
///
/// Values outside `[-MAX_STATIC_SCORE, MAX_STATIC_SCORE]` saturate to the
/// nearer bound; values inside the range pass through unchanged.
///
/// # Arguments
///
/// * `score` - raw evaluator score in centipawns
///
/// # Returns
///
/// Score clamped into the non-mate range.
#[inline]
#[must_use]
pub const fn clamp_static_score(score: i32) -> i32 {
    if score > MAX_STATIC_SCORE {
        MAX_STATIC_SCORE
    } else if score < -MAX_STATIC_SCORE {
        -MAX_STATIC_SCORE
    } else {
        score
    }
}

/// Used for converting a root-relative mate or tablebase score to a
/// position-relative transposition-table score.
///
/// Ordinary scores are unchanged. Mate and tablebase scores move away from
/// zero by `ply` so their packed value describes distance from the stored
/// position rather than from the root that happened to discover it.
///
/// # Arguments
///
/// * `score` - root-relative search score
/// * `ply` - root-relative ply of the position being stored
///
/// # Returns
///
/// Score suitable for transposition-table storage.
#[inline]
#[must_use]
pub const fn score_to_tt(score: i32, ply: u16) -> i32 {
    if score >= TB_WIN_THRESHOLD {
        score + ply as i32
    } else if score <= -TB_WIN_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

/// Used for converting a position-relative TT mate or tablebase score to
/// the current root frame.
///
/// This is the inverse of [`score_to_tt`] when called with the same ply:
/// mate and tablebase scores move toward zero by `ply` and ordinary scores
/// are unchanged.
///
/// # Arguments
///
/// * `score` - score loaded from the transposition table
/// * `ply` - root-relative ply of the probing position
///
/// # Returns
///
/// Root-relative search score.
#[inline]
#[must_use]
pub const fn score_from_tt(score: i32, ply: u16) -> i32 {
    if score >= TB_WIN_THRESHOLD {
        score - ply as i32
    } else if score <= -TB_WIN_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

/// Used for converting a mate score to a signed mate distance in full moves.
///
/// Positive values mean the side to move can force mate; negative values mean
/// it is being mated. The ply distance encoded relative to [`MATE_SCORE`] is
/// rounded up to full moves. Scores outside the mate band return `None`.
///
/// # Arguments
///
/// * `score` - search score in centipawns
///
/// # Returns
///
/// `Some` signed mate distance in full moves, or `None` for non-mate scores.
#[must_use]
pub const fn mate_moves(score: i32) -> Option<i32> {
    if !is_mate_score(score) {
        return None;
    }
    let magnitude = if score < 0 { -score } else { score };
    let plies = if magnitude >= MATE_SCORE {
        0
    } else {
        MATE_SCORE - magnitude
    };
    let moves = (plies + 1) / 2;
    Some(if score < 0 { -moves } else { moves })
}

/// Used for returning the terminal score of a node with no legal moves.
///
/// A checked node is checkmate, and mates delivered sooner (at a smaller ply)
/// score more strongly against the side to move. An unchecked node is
/// stalemate and scores zero.
///
/// # Arguments
///
/// * `in_check` - whether the side to move is in check
/// * `ply` - root-relative ply of the terminal node
///
/// # Returns
///
/// `-MATE_SCORE + ply` for checkmate, or zero for stalemate.
#[inline]
#[must_use]
pub const fn terminal_score(in_check: bool, ply: u16) -> i32 {
    if in_check {
        -MATE_SCORE + ply as i32
    } else {
        0
    }
}

