//! Small, testable search formulas kept separate from tree traversal.
//!
//! Keeping these deterministic policies free of search state makes strength
//! changes directly testable and keeps their boundary behavior explicit. All
//! formulas clamp or saturate hostile inputs instead of overflowing.

/// Used for expressing one ply of late-move reduction in fixed-point units.
///
/// Alternative configuration retained for controlled evaluation.
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

/// Used for computing the released null-move reduction in plies.
///
/// Alternative configuration retained for controlled evaluation.
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
    null_move_reduction_tuned(
        depth,
        static_eval,
        beta,
        RELEASED_NULL_MOVE_BASE,
        RELEASED_NULL_MOVE_DEPTH_DIVISOR,
    )
}

/// Used for the released null-move reduction's constant base in plies.
///
/// Alternative configuration retained for controlled evaluation.
pub const RELEASED_NULL_MOVE_BASE: u8 = 3;
/// Used for the released null-move reduction's depth divisor.
///
/// Alternative configuration retained for controlled evaluation.
pub const RELEASED_NULL_MOVE_DEPTH_DIVISOR: u8 = 4;

/// Used for computing the null-move reduction under tunable growth.
///
/// The reduction is `base + depth / divisor`, plus up to three further plies
/// when `static_eval` exceeds `beta` by large margins, at a rate of one ply
/// per 200 centipawns. Negative margins contribute nothing, and every addition
/// saturates.
///
/// Alternative configuration retained for controlled evaluation.
///
/// # Arguments
///
/// * `depth` - remaining search depth in plies
/// * `static_eval` - static evaluation of the node in centipawns
/// * `beta` - current beta bound in centipawns
/// * `base` - constant reduction in plies before the depth term
/// * `divisor` - plies of depth worth one further ply of reduction
///
/// # Returns
///
/// Null-move depth reduction in plies.
#[must_use]
pub fn null_move_reduction_tuned(
    depth: u8,
    static_eval: i32,
    beta: i32,
    base: u8,
    divisor: u8,
) -> u8 {
    let margin_reduction = static_eval.saturating_sub(beta) / 200;
    let margin_reduction = u8::try_from(margin_reduction.clamp(0, 3)).unwrap_or(0);
    let divisor = if divisor == 0 {
        RELEASED_NULL_MOVE_DEPTH_DIVISOR
    } else {
        divisor
    };
    base.saturating_add(depth / divisor)
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
    late_move_prune_threshold_scaled(depth, 100)
}

/// Used for returning the searched-quiet count after which late-move pruning
/// may begin, under a tunable scale.
///
/// The released threshold is `3 + depth^2` at `percent = 100`. That quadratic
/// grows fast enough to make the mechanism's *depth cap* inert: at depth six a
/// node must already have searched 39 quiet moves before pruning may begin,
/// and at depth ten it must have searched 103, while a typical position offers
/// about 35 legal moves in total. Late-move pruning therefore cannot fire
/// above roughly depth five whatever the cap permits, which is why raising the
/// cap alone reproduces the release exactly.
///
/// Stockfish 11's equivalent is `(5 + depth^2) * (1 + improving) / 2`, roughly
/// half this at a non-improving node, so the scale is the knob that actually
/// binds.
///
/// # Arguments
///
/// * `depth` - remaining search depth in plies
/// * `percent` - scale applied to the released threshold, `100` for release
///
/// # Returns
///
/// Number of searched quiet moves after which pruning may begin, at least one
/// so the first quiet move is never pruned.
#[must_use]
pub fn late_move_prune_threshold_scaled(depth: u8, percent: i32) -> u16 {
    let released = 3 + u32::from(depth) * u32::from(depth);
    let percent = percent.clamp(1, 1000);
    // The released threshold is bounded by `3 + 63^2` and the scale by 1000,
    // so the product stays far inside `u32`.
    #[allow(clippy::cast_sign_loss)]
    let scaled = released * (percent as u32) / 100;
    u16::try_from(scaled.max(1)).unwrap_or(u16::MAX)
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

/// Used for the largest score drop, in centipawns, that lengthens the budget.
///
/// Beyond this the position is lost or won outright and more thinking does not
/// help, so the extension saturates rather than growing without bound.
pub const FALLING_EVAL_CAP_CENTIPAWNS: i32 = 200;

/// Used for lengthening the soft budget when the score is falling.
///
/// A completed iteration that scores *worse* than the one before it is the
/// clearest in-search signal that the position just turned, and it is exactly
/// when the engine most wants another iteration. Janus's budget has until now
/// been scaled only by best-move stability, which is the opposite signal: it
/// measures when the engine is *confident*, not when it is in trouble. Both
/// matter and they are independent -- a move can be stable while the score
/// collapses under it.
///
/// `percent` is thousandths of the soft budget added per centipawn dropped, so
/// `0` is an exact no-op and the released default. The drop is clamped to
/// [`FALLING_EVAL_CAP_CENTIPAWNS`] before scaling.
///
/// # Arguments
///
/// * `soft_millis` - budget already scaled for stability
/// * `dropped_centipawns` - how far this iteration scored below the previous
///   one; zero or negative when the score held or improved
/// * `percent` - thousandths of the budget added per centipawn
///
/// # Returns
///
/// The lengthened budget in milliseconds, unchanged when `percent` is zero or
/// the score did not fall.
/// Not `const`: the widening conversions it needs are not const-callable on the
/// pinned 1.75 toolchain, and the function is only ever called at run time.
pub fn falling_eval_budget_millis(soft_millis: u64, dropped_centipawns: i32, percent: i32) -> u64 {
    if percent <= 0 || dropped_centipawns <= 0 {
        return soft_millis;
    }
    let capped = if dropped_centipawns > FALLING_EVAL_CAP_CENTIPAWNS {
        FALLING_EVAL_CAP_CENTIPAWNS
    } else {
        dropped_centipawns
    };
    // `1000 + capped * percent` thousandths, so percent = 0 is exactly 1x and
    // the arithmetic never leaves integer space.
    // Both are strictly positive here -- the early return covers the rest --
    // so `unsigned_abs` widens exactly and avoids a sign-losing cast.
    let scale = 1000_u64.saturating_add(
        u64::from(capped.unsigned_abs()).saturating_mul(u64::from(percent.unsigned_abs())),
    );
    soft_millis.saturating_mul(scale).saturating_div(1000)
}
