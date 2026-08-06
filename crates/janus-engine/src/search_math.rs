//! Small, testable search formulas kept separate from tree traversal.
//!
//! Keeping these deterministic policies free of search state makes strength
//! changes directly testable and keeps their boundary behavior explicit. All
//! formulas clamp or saturate hostile inputs instead of overflowing.

/// Used for expressing one ply of late-move reduction in fixed-point units.
///
/// The STR-316 candidate accumulates every reduction signal in these
/// `1/1024`-ply units and converts to whole plies once, so a single signal can
/// nudge the reduction by a fraction of a ply instead of only by a whole one.
pub const LMR_UNIT: i32 = 1_024;

/// Used for computing the base log-log late-move reduction in fixed-point
/// units.
///
/// This is the same `0.75 + ln(depth) * ln(move_number) / 2.25` curve as
/// [`lmr_reduction`] evaluated on the finer grid: the value is scaled by
/// [`LMR_UNIT`] before rounding, so the base itself keeps its fractional part
/// instead of being quantized to a whole ply before any context term is
/// applied. Inputs above 63 are clamped, the result is never negative, and it
/// is capped at `depth - 1` plies exactly as the whole-ply curve is.
///
/// # Arguments
///
/// * `depth` - remaining search depth in plies
/// * `move_number` - index of the move in the ordered move list
///
/// # Returns
///
/// Reduction in `1/1024`-ply units, at most `(depth - 1) * LMR_UNIT`.
#[must_use]
pub fn lmr_reduction_units(depth: u8, move_number: u16) -> i32 {
    if depth == 0 || move_number == 0 {
        return 0;
    }
    let depth = depth.min(63);
    let move_number = move_number.min(63);
    let scaled =
        (0.75 + f64::from(depth).ln() * f64::from(move_number).ln() / 2.25) * f64::from(LMR_UNIT);
    // The clamped inputs keep the rounded product inside a few thousand units.
    #[allow(clippy::cast_possible_truncation)]
    let units = scaled.round() as i32;
    units
        .max(0)
        .min(i32::from(depth.saturating_sub(1)) * LMR_UNIT)
}

/// Used for computing the base log-log late-move reduction in plies.
///
/// Depth or move number zero disables reduction. Inputs above 63 are clamped
/// before the `0.75 + ln(depth) * ln(move_number) / 2.25` term is rounded,
/// and the result never removes the final full-depth ply because it is capped
/// at `depth - 1`. The clamped inputs bound the reduction by eight plies.
///
/// # Arguments
///
/// * `depth` - remaining search depth in plies
/// * `move_number` - index of the move in the ordered move list
///
/// # Returns
///
/// Reduction in plies, at most `depth - 1`.
#[must_use]
pub fn lmr_reduction(depth: u8, move_number: u16) -> u8 {
    if depth == 0 || move_number == 0 {
        return 0;
    }
    let depth = depth.min(63);
    let move_number = move_number.min(63);
    let reduction = (0.75 + f64::from(depth).ln() * f64::from(move_number).ln() / 2.25).round();
    // The clamped inputs make the rounded result finite and bounded by eight.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let reduction = reduction as u8;
    reduction.min(depth.saturating_sub(1))
}

/// Used for computing the Java-compatible null-move reduction in plies.
///
/// The base of two plies grows by one for every six plies of depth and
/// receives up to three extra plies when `static_eval` exceeds `beta` by
/// large margins, at a rate of one ply per 200 centipawns. Negative margins
/// contribute nothing, and the additions saturate.
///
/// # Arguments
///
/// * `depth` - remaining search depth in plies
/// * `static_eval` - static evaluation of the node in centipawns
/// * `beta` - current beta bound in centipawns
///
/// # Returns
///
/// Null-move depth reduction in plies.
#[must_use]
pub fn null_move_reduction(depth: u8, static_eval: i32, beta: i32) -> u8 {
    let margin_reduction = static_eval.saturating_sub(beta) / 200;
    let margin_reduction = u8::try_from(margin_reduction.clamp(0, 3)).unwrap_or(0);
    2_u8.saturating_add(depth / 6)
        .saturating_add(margin_reduction)
}

/// Used for returning the searched-quiet count after which late-move pruning
/// may begin.
///
/// The quadratic threshold is `3 + depth^2`, so shallow nodes may prune after
/// only a handful of quiet moves while deeper nodes examine many more.
///
/// # Arguments
///
/// * `depth` - remaining search depth in plies
///
/// # Returns
///
/// Number of searched quiet moves after which pruning may begin.
#[must_use]
pub fn late_move_prune_threshold(depth: u8) -> u16 {
    3 + u16::from(depth) * u16::from(depth)
}

/// Used for scaling a soft clock budget according to root best-move stability.
///
/// A newly unstable root (zero stable iterations) may spend 170% of its
/// nominal target. One stable iteration leaves the target unchanged, two or
/// three stable iterations reduce it to 75%, and four or more reduce it to
/// 55%. The multiplication saturates before the division is applied.
///
/// # Arguments
///
/// * `soft_millis` - nominal soft budget in milliseconds
/// * `stable_iterations` - completed iterations with an unchanged best move
///
/// # Returns
///
/// Scaled soft budget in milliseconds.
#[must_use]
pub const fn stability_budget_millis(soft_millis: u64, stable_iterations: u8) -> u64 {
    let (numerator, denominator) = if stable_iterations == 0 {
        (17, 10)
    } else if stable_iterations >= 4 {
        (55, 100)
    } else if stable_iterations >= 2 {
        (75, 100)
    } else {
        (1, 1)
    };
    soft_millis
        .saturating_mul(numerator)
        .saturating_div(denominator)
}

