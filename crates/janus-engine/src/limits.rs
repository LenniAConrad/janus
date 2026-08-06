//! Search resource limits and deterministic clock allocation.
//!
//! [`SearchLimits`] bundles the depth, node, and wall-clock budgets accepted
//! by the searchers, [`SearchClock`] provides a one-shot live deadline shared
//! with a UCI coordinator, and the `clock_*` helpers turn a remaining game
//! clock into per-move soft and hard budgets.

use crate::score::MAX_DEPTH;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Used for supplying the default planning horizon when UCI does not provide
/// `movestogo`.
///
/// Callers hand this thirty-move horizon to the clock allocators when the
/// protocol omits an explicit count.
pub const CLOCK_MOVES_TO_GO: u32 = 30;

/// Used for retaining clock time that tolerates scheduling and protocol
/// overhead.
///
/// The budget allocators avoid planning to spend the final 50 ms of the
/// remaining clock whenever the clock is large enough to preserve them.
pub const CLOCK_RESERVE_MILLIS: u64 = 50;

/// One-shot wall clock that can be activated after a search has started.
///
/// Ponder searches share this clock with their UCI coordinator. Activation
/// publishes the soft and hard budgets atomically from the search's point of
/// view, and elapsed time begins at activation rather than construction. The
/// clock cannot be reset or extended, which prevents repeated protocol events
/// from silently changing an accepted search deadline.
#[derive(Debug)]
pub struct SearchClock {
    /// Used for encoding the activation instant without a lock.
    ///
    /// All atomic timestamps are nanosecond offsets from this monotonic
    /// epoch captured at construction.
    epoch: Instant,
    /// Used for tracking the atomic publication state.
    ///
    /// Zero is inactive, one is publishing, and two is fully active.
    state: AtomicU8,
    /// Used for recording the nanoseconds from `epoch` at which the live
    /// budget became active.
    ///
    /// Meaningful only once `state` has reached the active value.
    activation_nanos: AtomicU64,
    /// Used for storing the soft budget in nanoseconds.
    ///
    /// Holds [`NO_SOFT_BUDGET`] when no soft target was published.
    soft_nanos: AtomicU64,
    /// Used for storing the hard budget in nanoseconds measured from
    /// activation.
    hard_nanos: AtomicU64,
}

/// Used for distinguishing an absent soft target from a zero-duration target.
///
/// [`SearchClock::activate`] rejects budgets that would collide with this
/// sentinel value.
const NO_SOFT_BUDGET: u64 = u64::MAX;

impl SearchClock {
    /// Used for creating an inactive one-shot clock.
    ///
    /// The clock records its monotonic epoch at construction but publishes no
    /// budget until [`Self::activate`] succeeds.
    ///
    /// # Returns
    ///
    /// An inactive clock ready for its single permitted activation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
            state: AtomicU8::new(0),
            activation_nanos: AtomicU64::new(0),
            soft_nanos: AtomicU64::new(NO_SOFT_BUDGET),
            hard_nanos: AtomicU64::new(0),
        }
    }

    /// Used for activating the clock with budgets measured from this call.
    ///
    /// Returns `true` only for the caller that performs the one permitted
    /// activation. A soft budget greater than the hard budget is rejected, as
    /// is a duration too large to represent without colliding with the absent
    /// soft-budget sentinel. Publication is a two-step atomic handshake, so
    /// readers never observe a partially written budget.
    ///
    /// # Arguments
    ///
    /// * `soft` - optional soft target measured from this call
    /// * `hard` - hard deadline measured from this call
    ///
    /// # Returns
    ///
    /// `true` when this call performed the single permitted activation.
    pub fn activate(&self, soft: Option<Duration>, hard: Duration) -> bool {
        let hard_nanos = duration_nanos(hard);
        let soft_nanos = match soft {
            Some(value) if value <= hard => duration_nanos(value),
            Some(_) => return false,
            None => NO_SOFT_BUDGET,
        };
        if hard_nanos == NO_SOFT_BUDGET || soft_nanos == NO_SOFT_BUDGET && soft.is_some() {
            return false;
        }
        if self
            .state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        self.activation_nanos
            .store(duration_nanos(self.epoch.elapsed()), Ordering::Relaxed);
        self.soft_nanos.store(soft_nanos, Ordering::Relaxed);
        self.hard_nanos.store(hard_nanos, Ordering::Relaxed);
        self.state.store(2, Ordering::Release);
        true
    }

    /// Used for checking whether a complete budget publication has occurred.
    ///
    /// # Returns
    ///
    /// `true` once [`Self::activate`] has fully published a budget.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state.load(Ordering::Acquire) == 2
    }

    /// Used for measuring elapsed time since activation.
    ///
    /// # Returns
    ///
    /// Elapsed time since activation, or `None` while the clock is inactive.
    fn elapsed(&self) -> Option<Duration> {
        if !self.is_active() {
            return None;
        }
        let activated = self.activation_nanos.load(Ordering::Relaxed);
        let now = duration_nanos(self.epoch.elapsed());
        Some(Duration::from_nanos(now.saturating_sub(activated)))
    }

    /// Used for reading the published soft budget together with the live
    /// elapsed time.
    ///
    /// # Returns
    ///
    /// The soft budget paired with elapsed time since activation, or `None`
    /// when the clock is inactive or was activated without a soft target.
    pub(crate) fn soft_state(&self) -> Option<(Duration, Duration)> {
        let elapsed = self.elapsed()?;
        let soft = self.soft_nanos.load(Ordering::Relaxed);
        (soft != NO_SOFT_BUDGET).then(|| (Duration::from_nanos(soft), elapsed))
    }

    /// Used for checking whether the active hard budget has elapsed.
    ///
    /// # Returns
    ///
    /// `true` when the clock is active and no hard-deadline room remains.
    pub(crate) fn hard_expired(&self) -> bool {
        self.hard_remaining() == Some(Duration::ZERO)
    }

    /// Used for measuring the remaining hard-deadline room.
    ///
    /// # Returns
    ///
    /// Time left before the hard deadline, saturating at zero, or `None`
    /// while the clock is inactive.
    pub(crate) fn hard_remaining(&self) -> Option<Duration> {
        let elapsed = self.elapsed()?;
        let hard = Duration::from_nanos(self.hard_nanos.load(Ordering::Relaxed));
        Some(hard.saturating_sub(elapsed))
    }
}

impl Default for SearchClock {
    /// Used for creating an inactive one-shot clock.
    ///
    /// Equivalent to [`SearchClock::new`].
    ///
    /// # Returns
    ///
    /// An inactive clock ready for its single permitted activation.
    fn default() -> Self {
        Self::new()
    }
}

/// Used for converting a duration into the atomic clock's bounded nanosecond
/// domain.
///
/// Durations beyond `u64::MAX` nanoseconds saturate to `u64::MAX`, which
/// [`SearchClock::activate`] then rejects as unrepresentable.
///
/// # Arguments
///
/// * `duration` - duration to convert
///
/// # Returns
///
/// Whole nanoseconds, saturated to `u64::MAX`.
fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Resource limits for one search.
///
/// Fixed depth, node, and wall-clock budgets combine with an optional shared
/// [`SearchClock`] whose deadline may be activated while the search is
/// already running. Every limit that is present binds independently.
#[derive(Clone, Debug)]
pub struct SearchLimits {
    /// Used for bounding the completed iterative-deepening depth.
    ///
    /// Must lie in `1..=`[`MAX_DEPTH`] for [`Self::is_valid`] to accept the
    /// limits.
    pub depth: u8,
    /// Used for capping the number of visited nodes.
    ///
    /// Zero means unlimited.
    pub max_nodes: u64,
    /// Used for enforcing a hard wall-clock limit.
    ///
    /// `None` means unlimited.
    pub hard_time: Option<Duration>,
    /// Used for targeting a soft clock budget consulted between completed
    /// iterations.
    ///
    /// Valid only alongside a hard deadline that is at least as large.
    pub soft_time: Option<Duration>,
    /// Used for holding an optional one-shot deadline activated while the
    /// search is already live.
    ///
    /// Shared with the UCI coordinator through an [`Arc`].
    live_clock: Option<Arc<SearchClock>>,
}

impl PartialEq for SearchLimits {
    /// Used for comparing fixed budgets and the identity of an optional live
    /// clock.
    ///
    /// Limits carrying live clocks are equal only when both share the same
    /// [`SearchClock`] allocation via [`Arc::ptr_eq`].
    ///
    /// # Arguments
    ///
    /// * `other` - limits to compare against
    ///
    /// # Returns
    ///
    /// `true` when all fixed budgets match and any live clocks are the same
    /// allocation.
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth
            && self.max_nodes == other.max_nodes
            && self.hard_time == other.hard_time
            && self.soft_time == other.soft_time
            && match (&self.live_clock, &other.live_clock) {
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for SearchLimits {}

impl SearchLimits {
    /// Used for creating depth-only limits.
    ///
    /// Call [`Self::is_valid`] before passing user-derived values to a
    /// searcher.
    ///
    /// # Arguments
    ///
    /// * `depth` - maximum completed iterative-deepening depth
    ///
    /// # Returns
    ///
    /// Limits with only the depth bound set.
    #[must_use]
    pub const fn depth(depth: u8) -> Self {
        Self {
            depth,
            max_nodes: 0,
            hard_time: None,
            soft_time: None,
            live_clock: None,
        }
    }

    /// Used for adding a visited-node limit.
    ///
    /// Zero disables the limit.
    ///
    /// # Arguments
    ///
    /// * `max_nodes` - node cap, or zero for unlimited
    ///
    /// # Returns
    ///
    /// The limits with the node cap applied.
    #[must_use]
    pub const fn with_nodes(mut self, max_nodes: u64) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Used for adding a hard per-move time limit.
    ///
    /// This clears neither a node cap nor an existing soft target.
    ///
    /// # Arguments
    ///
    /// * `hard_time` - hard wall-clock deadline for the move
    ///
    /// # Returns
    ///
    /// The limits with the hard deadline applied.
    #[must_use]
    pub const fn with_time(mut self, hard_time: Duration) -> Self {
        self.hard_time = Some(hard_time);
        self
    }

    /// Used for adding paired soft and hard clock limits.
    ///
    /// A valid soft target must not exceed the hard deadline;
    /// [`Self::is_valid`] rejects the pair otherwise.
    ///
    /// # Arguments
    ///
    /// * `soft_time` - soft clock target consulted between iterations
    /// * `hard_time` - hard wall-clock deadline for the move
    ///
    /// # Returns
    ///
    /// The limits with both clock budgets applied.
    #[must_use]
    pub const fn with_clock(mut self, soft_time: Duration, hard_time: Duration) -> Self {
        self.soft_time = Some(soft_time);
        self.hard_time = Some(hard_time);
        self
    }

    /// Used for adding a shared one-shot clock for a later live deadline
    /// transition.
    ///
    /// Fixed time limits, when also present, remain independently binding.
    ///
    /// # Arguments
    ///
    /// * `clock` - shared clock that a coordinator may activate later
    ///
    /// # Returns
    ///
    /// The limits with the live clock attached.
    #[must_use]
    pub fn with_live_clock(mut self, clock: Arc<SearchClock>) -> Self {
        self.live_clock = Some(clock);
        self
    }

    /// Used for checking whether all invariants required by the searcher
    /// hold.
    ///
    /// Depth must be in `1..=`[`MAX_DEPTH`]. A soft target is valid only when
    /// a hard deadline exists and is at least as large.
    ///
    /// # Returns
    ///
    /// `true` when the limits satisfy every searcher invariant.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.depth == 0 || self.depth > MAX_DEPTH {
            return false;
        }
        match (self.soft_time, self.hard_time) {
            (Some(soft), Some(hard)) => soft <= hard,
            (Some(_), None) => false,
            _ => true,
        }
    }

    /// Used for checking whether either the fixed or the activated hard
    /// deadline has elapsed.
    ///
    /// # Arguments
    ///
    /// * `started` - instant at which the search began, anchoring the fixed
    ///   deadline
    ///
    /// # Returns
    ///
    /// `true` when the fixed hard limit has elapsed since `started` or an
    /// attached live clock has expired.
    pub(crate) fn hard_expired(&self, started: Instant) -> bool {
        self.hard_time.is_some_and(|hard| started.elapsed() >= hard)
            || self
                .live_clock
                .as_ref()
                .is_some_and(|clock| clock.hard_expired())
    }

    /// Used for reading the activated clock's optional soft budget and
    /// elapsed time.
    ///
    /// # Returns
    ///
    /// The live clock's soft budget paired with its elapsed time, or `None`
    /// when no clock is attached, the clock is inactive, or no soft target
    /// was published.
    pub(crate) fn live_soft_state(&self) -> Option<(Duration, Duration)> {
        self.live_clock.as_ref()?.soft_state()
    }
}

impl Default for SearchLimits {
    /// Used for returning the bounded default limits.
    ///
    /// The default is depth 3, 250,000 nodes, and a five-second hard limit.
    ///
    /// # Returns
    ///
    /// The bounded default limits.
    fn default() -> Self {
        Self::depth(3)
            .with_nodes(250_000)
            .with_time(Duration::from_secs(5))
    }
}

/// Used for computing the increment share spent by both clock allocators.
///
/// The share is three quarters of the increment, computed as
/// `increment * 3 / 4` in integer math with a saturating multiplication so
/// hostile protocol values bound the result instead of overflowing. Retuned
/// on 2026-07-24 from the previous `increment / 2` coefficient.
///
/// # Arguments
///
/// * `increment_millis` - per-move increment in milliseconds
///
/// # Returns
///
/// The increment share in milliseconds, saturating at `u64::MAX / 4`.
const fn increment_share_millis(increment_millis: u64) -> u64 {
    increment_millis.saturating_mul(3) / 4
}

/// Used for computing the normal per-move target from a remaining game clock.
///
/// The allocator spends one planning-horizon share plus three quarters of the
/// increment (`inc * 3 / 4` in saturating integer math, retuned 2026-07-24
/// from `inc / 2` toward the 0.75..1.0 range used by reference allocators),
/// while retaining [`CLOCK_RESERVE_MILLIS`] whenever the remaining clock
/// permits. Zero `moves_to_go` means one move rather than division by zero.
/// Arithmetic saturates for hostile protocol values.
///
/// # Arguments
///
/// * `remaining_millis` - remaining clock time in milliseconds
/// * `increment_millis` - per-move increment in milliseconds
/// * `moves_to_go` - moves until the next time control; zero is treated as one
///
/// # Returns
///
/// Soft per-move budget in milliseconds; zero only for an empty clock.
#[must_use]
pub fn clock_budget_millis(remaining_millis: u64, increment_millis: u64, moves_to_go: u32) -> u64 {
    if remaining_millis == 0 {
        return 0;
    }
    let divisor = u64::from(if moves_to_go == 0 { 1 } else { moves_to_go });
    let budget = remaining_millis
        .saturating_div(divisor)
        .saturating_add(increment_share_millis(increment_millis))
        .max(1);
    let reserve_adjusted = if remaining_millis > CLOCK_RESERVE_MILLIS {
        remaining_millis - CLOCK_RESERVE_MILLIS
    } else {
        1
    };
    if budget < reserve_adjusted {
        budget
    } else {
        reserve_adjusted
    }
}

/// Used for computing the hard deadline paired with [`clock_budget_millis`].
///
/// The extension is at most four times the soft target, one quarter of the
/// remaining clock plus three quarters of the increment, and the
/// reserve-adjusted clock; the smallest of the three wins and never falls
/// below the soft target.
///
/// # Arguments
///
/// * `remaining_millis` - remaining clock time in milliseconds
/// * `increment_millis` - per-move increment in milliseconds
/// * `moves_to_go` - moves until the next time control; zero is treated as one
///
/// # Returns
///
/// Hard per-move deadline in milliseconds, at least the soft budget; zero
/// only for an empty clock.
#[must_use]
pub fn clock_hard_budget_millis(
    remaining_millis: u64,
    increment_millis: u64,
    moves_to_go: u32,
) -> u64 {
    let soft = clock_budget_millis(remaining_millis, increment_millis, moves_to_go);
    if soft == 0 {
        return 0;
    }
    let quarter_clock = (remaining_millis / 4)
        .saturating_add(increment_share_millis(increment_millis))
        .max(1);
    let reserve_adjusted = if remaining_millis > CLOCK_RESERVE_MILLIS {
        remaining_millis - CLOCK_RESERVE_MILLIS
    } else {
        1
    };
    let extension = soft
        .saturating_mul(4)
        .min(quarter_clock)
        .min(reserve_adjusted);
    soft.max(extension)
}

/// Used for producing `(soft, hard)` durations for a clock-based search.
///
/// Combines [`clock_budget_millis`] and [`clock_hard_budget_millis`] on the
/// same inputs.
///
/// # Arguments
///
/// * `remaining_millis` - remaining clock time in milliseconds
/// * `increment_millis` - per-move increment in milliseconds
/// * `moves_to_go` - moves until the next time control; zero is treated as one
///
/// # Returns
///
/// Soft and hard budgets as durations, in that order.
#[must_use]
pub fn clock_budgets(
    remaining_millis: u64,
    increment_millis: u64,
    moves_to_go: u32,
) -> (Duration, Duration) {
    let soft = clock_budget_millis(remaining_millis, increment_millis, moves_to_go);
    let hard = clock_hard_budget_millis(remaining_millis, increment_millis, moves_to_go);
    (Duration::from_millis(soft), Duration::from_millis(hard))
}

