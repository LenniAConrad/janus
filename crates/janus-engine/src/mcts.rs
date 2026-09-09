//! Deterministic single-thread PUCT Monte Carlo tree search.
//!
//! The implementation deliberately favors a small, auditable search: one
//! arena, one evaluator, and limits checked before arena growth, between
//! simulations, and inside recursive leaf quiescence.
//! Exact simulation budgets remain deterministic; optional wall-clock limits
//! make the same search usable through timed UCI commands. Neural evaluators
//! can replace handcrafted move priors through the shared
//! [`Evaluator`](crate::evaluator::Evaluator) contract.

use crate::evaluator::{piece_value, Evaluator, PolicyValue};
use crate::limits::SearchClock;
use janus_core::{Color, Move, Piece, PieceKind, Position, Square};
use std::cmp::Ordering;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Used for converting handcrafted integer ordering terms into softmax-scale
/// logits.
///
/// Handcrafted move-ordering scores are centipawn-magnitude integers; the
/// combined ordering and tactical terms are divided by this scale before
/// entering the prior softmax.
const POLICY_LOGIT_SCALE: f64 = 18_000.0;
/// Used for reserving total prior probability for legal moves missing a
/// supplied logit.
///
/// When at least one legal move lacks a finite supplied logit, this mass is
/// split uniformly across the missing moves and the remainder is distributed
/// over the supplied ones.
const MISSING_POLICY_MASS: f64 = 0.05;
/// Used for capping the number of tactical plies evaluated below an expanded
/// leaf.
///
/// Quiescence returns a static evaluation once this recursion depth is
/// reached.
const QUIESCENCE_MAX_PLY: u8 = 4;
/// Used for capping the non-evasion tactical candidates searched at one
/// quiescence node.
///
/// Check evasions are exempt: when the side to move is in check, every legal
/// evasion is searched.
const QUIESCENCE_MAX_MOVES: usize = 12;

/// Production ceiling for arena nodes allocated by one MCTS invocation.
///
/// The root counts as one node. UCI may lower this limit for constrained
/// environments, but never raises it above this default. A complete child
/// batch that would cross the ceiling is refused before the arena changes.
pub const DEFAULT_MCTS_TREE_NODE_LIMIT: usize = 1_048_576;

/// Tunable PUCT and handcrafted-prior settings.
///
/// Values are validated by [`Mcts::with_config`]: `cpuct` must be finite and
/// positive, `fpu_reduction` finite and non-negative, and every handcrafted
/// prior term non-negative. [`Default`] supplies the current deterministic
/// Janus settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig {
    /// Used for weighting the PUCT exploration term during child selection.
    ///
    /// Larger values favor low-visit, high-prior children over exploitation
    /// of the current mean value.
    pub cpuct: f64,
    /// Used for reducing the first-play urgency of an unvisited child.
    ///
    /// The unvisited-child value starts from the parent mean and is lowered
    /// by this amount scaled by the square root of the prior mass already
    /// visited.
    pub fpu_reduction: f64,
    /// Used for adding a policy-logit bonus to a move that gives check.
    ///
    /// Applied only when handcrafted priors are in use, i.e. when the
    /// evaluator supplies no finite policy logits.
    pub check_prior_bonus: i32,
    /// Used for penalizing the policy logit of a capture whose static
    /// exchange value is negative.
    ///
    /// Applied only when handcrafted priors are in use.
    pub losing_capture_prior_penalty: i32,
    /// Used for scaling a positive static exchange value, capped at 900
    /// centipawns, into the handcrafted policy logit.
    ///
    /// Applied only when handcrafted priors are in use.
    pub winning_capture_prior_scale: i32,
}

impl Default for MctsConfig {
    /// Used for returning the current deterministic Janus search settings.
    ///
    /// # Returns
    ///
    /// Configuration with `cpuct` 2.0, `fpu_reduction` 0.05, and the tuned
    /// handcrafted-prior terms.
    fn default() -> Self {
        Self {
            cpuct: 2.0,
            fpu_reduction: 0.05,
            check_prior_bonus: 4_000,
            losing_capture_prior_penalty: 8_000,
            winning_capture_prior_scale: 8,
        }
    }
}

/// Resource limits for one MCTS invocation.
///
/// The search checks `hard_time` between simulations and during recursive leaf
/// quiescence, and admits arena expansions only below `max_tree_nodes`. One
/// evaluator call already in progress is allowed to finish, so evaluators
/// should keep an individual evaluation bounded.
#[derive(Clone, Debug)]
pub struct MctsLimits {
    /// Used for capping the number of completed simulations.
    ///
    /// A tree-proven root result may stop the search before the cap is
    /// consumed.
    pub max_simulations: u64,
    /// Used for bounding elapsed wall-clock time; `None` means unlimited.
    ///
    /// Checked between simulations and inside recursive leaf quiescence.
    pub hard_time: Option<Duration>,
    /// Used for capping arena nodes allocated by this invocation, including
    /// the root.
    ///
    /// Expansion is atomic with respect to this limit: every legal child of a
    /// leaf fits, or none are appended.
    max_tree_nodes: usize,
    /// Used for an optional one-shot deadline activated while the tree
    /// remains live.
    ///
    /// Shared through [`SearchClock`]; clock identity, not clock state,
    /// participates in equality.
    live_clock: Option<Arc<SearchClock>>,
    /// Used for an optional canonical root subset; `None` permits every legal
    /// root move.
    ///
    /// Stored sorted by UCI order and de-duplicated by
    /// [`MctsLimits::with_root_moves`].
    root_moves: Option<Vec<Move>>,
}

impl PartialEq for MctsLimits {
    /// Used for comparing fixed budgets and the identity of an optional live
    /// clock.
    ///
    /// Two limits holding distinct [`SearchClock`] allocations compare
    /// unequal even when both clocks carry the same state.
    ///
    /// # Arguments
    ///
    /// * `other` - limits compared against `self`
    ///
    /// # Returns
    ///
    /// `true` when budgets, root subsets, and clock identity all match.
    fn eq(&self, other: &Self) -> bool {
        self.max_simulations == other.max_simulations
            && self.hard_time == other.hard_time
            && self.max_tree_nodes == other.max_tree_nodes
            && match (&self.live_clock, &other.live_clock) {
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
            && self.root_moves == other.root_moves
    }
}

impl Eq for MctsLimits {}

impl MctsLimits {
    /// Used for creating an exact simulation limit with no wall-clock
    /// deadline.
    ///
    /// # Arguments
    ///
    /// * `max_simulations` - maximum number of completed simulations
    ///
    /// # Returns
    ///
    /// Limits with the production tree-node ceiling and no time bound, live
    /// clock, or root-move restriction.
    #[must_use]
    pub const fn simulations(max_simulations: u64) -> Self {
        Self {
            max_simulations,
            hard_time: None,
            max_tree_nodes: DEFAULT_MCTS_TREE_NODE_LIMIT,
            live_clock: None,
            root_moves: None,
        }
    }

    /// Used for lowering or explicitly replacing the arena-node ceiling.
    ///
    /// A value of zero is normalized to one because every result owns a root
    /// node. Library callers may explicitly select a value above
    /// [`DEFAULT_MCTS_TREE_NODE_LIMIT`]; the UCI shell enforces that production
    /// value as its hard maximum.
    ///
    /// # Arguments
    ///
    /// * `max_tree_nodes` - maximum arena nodes including the root
    ///
    /// # Returns
    ///
    /// The limits with a node ceiling of at least one installed.
    #[must_use]
    pub const fn with_tree_node_limit(mut self, max_tree_nodes: usize) -> Self {
        self.max_tree_nodes = if max_tree_nodes == 0 {
            1
        } else {
            max_tree_nodes
        };
        self
    }

    /// Used for adding a hard elapsed-time limit.
    ///
    /// # Arguments
    ///
    /// * `hard_time` - maximum elapsed wall-clock time for the invocation
    ///
    /// # Returns
    ///
    /// The limits with the hard deadline installed.
    #[must_use]
    pub const fn with_time(mut self, hard_time: Duration) -> Self {
        self.hard_time = Some(hard_time);
        self
    }

    /// Used for adding a shared one-shot clock for a later live deadline
    /// transition.
    ///
    /// A fixed hard limit, when also present, remains independently binding.
    ///
    /// # Arguments
    ///
    /// * `clock` - shared clock that may later activate a hard deadline
    ///
    /// # Returns
    ///
    /// The limits with the live clock installed.
    #[must_use]
    pub fn with_live_clock(mut self, clock: Arc<SearchClock>) -> Self {
        self.live_clock = Some(clock);
        self
    }

    /// Used for restricting the root to a canonical, de-duplicated move
    /// subset.
    ///
    /// The search intersects this subset with the root's complete legal moves.
    /// An explicit empty or wholly non-legal subset fails closed with no move;
    /// it never falls back to an unrestricted search. Evaluators still receive
    /// the complete legal root policy request before the permitted prior mass
    /// is conditioned.
    ///
    /// # Arguments
    ///
    /// * `root_moves` - requested root moves; sorted and de-duplicated here
    ///
    /// # Returns
    ///
    /// The limits with the canonical root subset installed.
    #[must_use]
    pub fn with_root_moves(mut self, mut root_moves: Vec<Move>) -> Self {
        root_moves.sort_by_key(|mv| mv.uci_order_key());
        root_moves.dedup();
        self.root_moves = Some(root_moves);
        self
    }

    /// Used for computing remaining room under the earliest active hard
    /// deadline.
    ///
    /// # Arguments
    ///
    /// * `started` - instant the invocation began, anchoring the fixed limit
    ///
    /// # Returns
    ///
    /// The smaller remaining duration of the fixed and live deadlines, or
    /// `None` when neither is active.
    fn hard_remaining(&self, started: Instant) -> Option<Duration> {
        let fixed = self
            .hard_time
            .map(|hard| hard.saturating_sub(started.elapsed()));
        let live = self
            .live_clock
            .as_ref()
            .and_then(|clock| clock.hard_remaining());
        match (fixed, live) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (Some(remaining), None) | (None, Some(remaining)) => Some(remaining),
            (None, None) => None,
        }
    }
}

/// Invalid MCTS configuration.
///
/// Produced by [`Mcts::with_config`] when a numeric setting fails validation.
/// The wrapped string carries the human-readable diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MctsError(String);

impl MctsError {
    /// Used for retrieving the configuration diagnostic.
    ///
    /// # Returns
    ///
    /// The human-readable validation message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MctsError {
    /// Used for writing the configuration diagnostic to the formatter.
    ///
    /// # Arguments
    ///
    /// * `formatter` - output sink receiving the diagnostic text
    ///
    /// # Returns
    ///
    /// The result of writing the message to `formatter`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MctsError {}

/// Terminal or tree-proven result of the root position.
///
/// Reported from the root side-to-move perspective, either because the root
/// is intrinsically terminal or because the explored tree proves an exact
/// minimax result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameOutcome {
    /// Used for indicating the root side to move has a proven win in the
    /// explored tree.
    ///
    /// Corresponds to a root value of `1.0`.
    Win,
    /// Used for indicating the position is drawn.
    ///
    /// Covers intrinsic rule draws and tree-proven forced draws; corresponds
    /// to a root value of `0.0`.
    Draw,
    /// Used for indicating the root side to move has a proven loss.
    ///
    /// Corresponds to a root value of `-1.0`.
    Loss,
}

/// Public statistics for one root move.
///
/// Entries appear in [`MctsResult::root_moves`] sorted by proof, visits,
/// value, prior, then UCI text.
#[derive(Clone, Debug, PartialEq)]
pub struct RootMove {
    /// Used for identifying the legal root move these statistics describe.
    ///
    /// Always a member of the (possibly restricted) legal root move set.
    pub mv: Move,
    /// Used for counting the simulations that traversed the move.
    ///
    /// Zero for children that were expanded but never selected.
    pub visits: u64,
    /// Used for reporting the mean result from the root side-to-move
    /// perspective.
    ///
    /// Proven children report their exact proof value; unvisited unproven
    /// children report `0.0`.
    pub value: f64,
    /// Used for reporting the normalized expansion prior of the move.
    ///
    /// Zero-simulation fallback results report `0.0` for every move.
    pub prior: f64,
}

/// Result of one bounded MCTS invocation.
///
/// Snapshots of this structure are also reported through
/// [`Mcts::search_with_progress`]; the final return value remains
/// authoritative.
#[derive(Clone, Debug, PartialEq)]
pub struct MctsResult {
    /// Used for reporting the highest-ranked legal move after proof, visit,
    /// value, and prior ordering.
    ///
    /// `None` only for a terminal root or an empty permitted root move set.
    pub best_move: Option<Move>,
    /// Used for reporting the principal variation formed by following the
    /// same child ordering.
    ///
    /// The line stops at the first unvisited or unexpanded node.
    pub principal_variation: Vec<Move>,
    /// Used for counting completed simulations; may be below the request
    /// after a tree proof or capacity exhaustion.
    ///
    /// Zero when the root is terminal or no simulation could begin.
    pub simulations: u64,
    /// Used for counting nodes allocated in the search arena, including the
    /// root.
    ///
    /// Terminal and zero-simulation results report exactly one node.
    pub tree_nodes: usize,
    /// Used for reporting that the next complete leaf expansion could not be
    /// admitted.
    ///
    /// This is `true` for either the configured arena-node ceiling or a
    /// fallible reservation refusal. The returned tree remains complete and
    /// internally consistent.
    pub tree_capacity_exhausted: bool,
    /// Used for reporting the root visit count; equal to `simulations` for a
    /// non-terminal search.
    ///
    /// Zero when no simulation completed.
    pub root_visits: u64,
    /// Used for reporting the mean root value from the root side-to-move
    /// perspective.
    ///
    /// A proven root reports its exact proof value instead of the visit mean.
    pub value: f64,
    /// Used for reporting an intrinsic or tree-proven root result.
    ///
    /// `None` while the explored tree has not established an exact result.
    pub outcome: Option<GameOutcome>,
    /// Used for reporting root moves sorted by proof, visits, value, prior,
    /// then UCI text.
    ///
    /// Empty for terminal roots and empty permitted root move sets.
    pub root_moves: Vec<RootMove>,
}

/// Reusable board and path storage for one sequence of MCTS simulations.
///
/// `keys` begins with the immutable pre-root history plus the root key, while
/// `path` begins with the arena root. A completed simulation may append to both
/// vectors but must never modify those prefixes; [`Self::reset`] restores them
/// without reallocating or recopying the game history.
struct SimulationScratch {
    /// Used for the mutable position rebuilt to the search root before every
    /// simulation.
    ///
    /// Selection pushes moves onto it; quiescence makes and unmakes moves on
    /// it in place.
    position: Position,
    /// Used for repetition keys covering the immutable game prefix and the
    /// current search path.
    ///
    /// The first `history_len + 1` entries (prefix plus root key) are never
    /// modified by a simulation.
    keys: Vec<u64>,
    /// Used for arena node indices from the root through the selected leaf.
    ///
    /// Index `0` always refers to the arena root node.
    path: Vec<usize>,
    /// Used for counting repetition keys preceding the root position.
    ///
    /// Splits true game history from search-path keys in repetition checks.
    history_len: usize,
}

impl SimulationScratch {
    /// Used for creating scratch with the complete immutable root-key prefix
    /// installed.
    ///
    /// # Arguments
    ///
    /// * `root` - search root cloned into the mutable position
    /// * `root_keys` - pre-root history keys followed by the root key
    ///
    /// # Returns
    ///
    /// Scratch whose path contains only the arena root index.
    ///
    /// # Panics
    ///
    /// Panics when `root_keys` is empty, because the prefix must contain the
    /// root key.
    fn new(root: &Position, root_keys: &[u64]) -> Self {
        let history_len = root_keys
            .len()
            .checked_sub(1)
            .expect("MCTS root-key prefix contains the root");
        Self {
            position: root.clone(),
            keys: root_keys.to_vec(),
            path: vec![0],
            history_len,
        }
    }

    /// Used for restoring root state while retaining the largest allocated
    /// capacities.
    ///
    /// Truncates the key and path buffers back to their immutable prefixes
    /// without reallocating or recopying the game history.
    ///
    /// # Arguments
    ///
    /// * `root` - search root cloned into the mutable position
    fn reset(&mut self, root: &Position) {
        self.position.clone_from(root);
        let root_key_count = self.history_len + 1;
        debug_assert!(self.keys.len() >= root_key_count);
        self.keys.truncate(root_key_count);
        debug_assert!(!self.path.is_empty());
        self.path.truncate(1);
        debug_assert_eq!(self.path[0], 0);
    }
}

/// Immutable root and cancellation context shared by every simulation.
///
/// Bundles the borrowed inputs of one invocation so the simulation loop can
/// pass them as a unit.
struct SimulationControl<'a> {
    /// Used for the search root restored before each simulation.
    ///
    /// Never mutated; simulations clone from it into scratch storage.
    root: &'a Position,
    /// Used for the canonical legal moves admitted only at the root.
    ///
    /// Deeper nodes always expand their complete legal move set.
    permitted_root_moves: &'a [Move],
    /// Used for the external cancellation token.
    ///
    /// Checked between simulations and inside recursive leaf quiescence.
    stop: &'a AtomicBool,
    /// Used for the start instant anchoring fixed elapsed limits.
    ///
    /// Shared by every deadline check throughout one invocation.
    started: Instant,
    /// Used for the complete per-invocation resource and live-clock limits.
    ///
    /// Consulted together with `started` for wall-clock exhaustion checks.
    limits: &'a MctsLimits,
}

/// Completion state of one attempted MCTS simulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SimulationStatus {
    /// Used when selection, optional expansion, evaluation, and backup all
    /// completed.
    Completed,
    /// Used when the next complete child batch exceeded the node ceiling or
    /// its fallible reservation was refused.
    TreeCapacityExhausted,
}

/// Reusable deterministic MCTS instance owning one evaluator.
///
/// One instance can run any number of sequential searches; every invocation
/// builds a fresh arena while the evaluator persists across calls.
pub struct Mcts<E: Evaluator> {
    /// Used for the position evaluator and optional policy provider owned by
    /// this searcher.
    ///
    /// Borrowed through [`Mcts::evaluator`].
    evaluator: E,
    /// Used for the validated exploration and handcrafted-prior parameters.
    ///
    /// Set at construction and never modified afterwards.
    config: MctsConfig,
}

impl<E: Evaluator> Mcts<E> {
    /// Used for creating a search with the current Janus PUCT settings.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator owned by the new searcher
    ///
    /// # Returns
    ///
    /// A searcher configured with [`MctsConfig::default`].
    #[must_use]
    pub fn new(evaluator: E) -> Self {
        Self {
            evaluator,
            config: MctsConfig::default(),
        }
    }

    /// Used for creating a search after validating all numeric settings.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - position evaluator owned by the new searcher
    /// * `config` - candidate configuration validated before use
    ///
    /// # Returns
    ///
    /// A searcher using `config` when validation succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`MctsError`] when a setting is non-finite, out of range, or a
    /// handcrafted-prior bonus is negative.
    pub fn with_config(evaluator: E, config: MctsConfig) -> Result<Self, MctsError> {
        validate_config(config)?;
        Ok(Self { evaluator, config })
    }

    /// Used for retrieving the active search configuration.
    ///
    /// # Returns
    ///
    /// A copy of the validated configuration.
    #[must_use]
    pub const fn config(&self) -> MctsConfig {
        self.config
    }

    /// Used for borrowing the owned evaluator.
    ///
    /// # Returns
    ///
    /// A shared reference to the evaluator.
    #[must_use]
    pub const fn evaluator(&self) -> &E {
        &self.evaluator
    }

    /// Used for running at most `node_budget` simulations for a non-terminal
    /// root.
    ///
    /// Search stops early when the explored tree proves the root result.
    ///
    /// # Arguments
    ///
    /// * `root` - position searched from
    /// * `node_budget` - maximum number of completed simulations
    ///
    /// # Returns
    ///
    /// Complete root statistics for the bounded invocation.
    #[must_use]
    pub fn search(&mut self, root: &Position, node_budget: u64) -> MctsResult {
        self.search_with_limits(root, MctsLimits::simulations(node_budget))
    }

    /// Used for searching with both a simulation cap and an optional
    /// elapsed-time limit.
    ///
    /// Search stops early when either limit is reached or the explored tree
    /// proves the root result.
    ///
    /// # Arguments
    ///
    /// * `root` - position searched from
    /// * `limits` - per-invocation simulation, time, and root-move limits
    ///
    /// # Returns
    ///
    /// Complete root statistics for the bounded invocation.
    #[must_use]
    pub fn search_with_limits(&mut self, root: &Position, limits: MctsLimits) -> MctsResult {
        self.search_with_history_limits(root, &[], limits)
    }

    /// Used for searching with repetition keys for positions occurring before
    /// `root`.
    ///
    /// `previous_keys` must be a dense oldest-to-newest sequence containing
    /// exactly one [`Position::key`] for every played ply immediately preceding
    /// `root`; it must not include the root key itself. Sparse repetition
    /// candidates are invalid because search uses sequence indices for
    /// side-to-move parity and reversible-cycle boundaries. The search appends
    /// the root once, requires two pre-root matches for a rule threefold, and
    /// treats the first recurrence wholly inside the search path as a cycle draw
    /// estimate.
    ///
    /// # Arguments
    ///
    /// * `root` - position searched from
    /// * `previous_keys` - dense oldest-to-newest pre-root repetition keys
    /// * `node_budget` - maximum number of completed simulations
    ///
    /// # Returns
    ///
    /// Complete root statistics for the bounded invocation.
    #[must_use]
    pub fn search_with_history(
        &mut self,
        root: &Position,
        previous_keys: &[u64],
        node_budget: u64,
    ) -> MctsResult {
        self.search_with_history_limits(root, previous_keys, MctsLimits::simulations(node_budget))
    }

    /// Used for searching with repetition history and bounded simulation/time
    /// resources.
    ///
    /// `previous_keys` must be a dense oldest-to-newest sequence containing
    /// exactly one [`Position::key`] for every played ply immediately preceding
    /// `root`; it must not include the root key itself. Sparse repetition
    /// candidates are invalid because search uses sequence indices for
    /// side-to-move parity and reversible-cycle boundaries. The search appends
    /// the root once, requires two pre-root matches for a rule threefold, and
    /// treats the first recurrence wholly inside the search path as a cycle draw
    /// estimate. Terminal root outcomes take precedence over exhausted limits.
    ///
    /// # Arguments
    ///
    /// * `root` - position searched from
    /// * `previous_keys` - dense oldest-to-newest pre-root repetition keys
    /// * `limits` - per-invocation simulation, time, and root-move limits
    ///
    /// # Returns
    ///
    /// Complete root statistics for the bounded invocation.
    #[must_use]
    pub fn search_with_history_limits(
        &mut self,
        root: &Position,
        previous_keys: &[u64],
        limits: MctsLimits,
    ) -> MctsResult {
        let stop = AtomicBool::new(false);
        self.search_with_history_limits_and_stop(root, previous_keys, limits, &stop)
    }

    /// Used for searching with repetition history, bounded resources, and an
    /// external stop token checked between simulations and inside leaf
    /// quiescence.
    ///
    /// `previous_keys` must be a dense oldest-to-newest sequence containing
    /// exactly one [`Position::key`] for every played ply immediately preceding
    /// `root`; it must not include the root key itself. Sparse repetition
    /// candidates are invalid because search uses sequence indices for
    /// side-to-move parity and reversible-cycle boundaries. Terminal root
    /// outcomes take precedence over an already-set stop token. An interrupted
    /// leaf backs up its latest bounded estimate, ensuring the returned tree
    /// remains internally consistent; one evaluator call in progress may still
    /// finish.
    ///
    /// # Arguments
    ///
    /// * `root` - position searched from
    /// * `previous_keys` - dense oldest-to-newest pre-root repetition keys
    /// * `limits` - per-invocation simulation, time, and root-move limits
    /// * `stop` - external cancellation token
    ///
    /// # Returns
    ///
    /// Complete root statistics for the bounded invocation.
    #[must_use]
    // Resource-limit entry points consistently take ownership of one invocation's
    // configuration, including the shared live-clock handle.
    #[allow(clippy::needless_pass_by_value)]
    pub fn search_with_history_limits_and_stop(
        &mut self,
        root: &Position,
        previous_keys: &[u64],
        limits: MctsLimits,
        stop: &AtomicBool,
    ) -> MctsResult {
        self.search_with_history_limits_and_stop_observer::<_, false>(
            root,
            previous_keys,
            &limits,
            stop,
            Duration::MAX,
            |_| {},
        )
    }

    /// Used for searching with external stopping while periodically reporting
    /// complete tree snapshots to `observer`.
    ///
    /// `previous_keys` must be a dense oldest-to-newest sequence containing
    /// exactly one [`Position::key`] for every played ply immediately preceding
    /// `root`; it must not include the root key itself. Sparse repetition
    /// candidates are invalid because search uses sequence indices for
    /// side-to-move parity and reversible-cycle boundaries.
    ///
    /// The callback runs only between simulations and therefore always sees a
    /// consistent [`MctsResult`]. `progress_interval` is a minimum wall-clock
    /// interval; a long-running simulation can delay a report. Terminal roots
    /// and searches stopped before their first simulation return normally
    /// without a progress callback. The final return value remains authoritative
    /// and may be newer than the last reported snapshot.
    ///
    /// # Arguments
    ///
    /// * `root` - position searched from
    /// * `previous_keys` - dense oldest-to-newest pre-root repetition keys
    /// * `limits` - per-invocation simulation, time, and root-move limits
    /// * `stop` - external cancellation token
    /// * `progress_interval` - minimum wall-clock time between reports
    /// * `observer` - callback receiving consistent tree snapshots
    ///
    /// # Returns
    ///
    /// Complete root statistics for the bounded invocation.
    #[must_use]
    // Keep the same owned per-invocation contract as the other limit entry points.
    #[allow(clippy::needless_pass_by_value)]
    pub fn search_with_progress<F>(
        &mut self,
        root: &Position,
        previous_keys: &[u64],
        limits: MctsLimits,
        stop: &AtomicBool,
        progress_interval: Duration,
        observer: F,
    ) -> MctsResult
    where
        F: FnMut(&MctsResult),
    {
        self.search_with_history_limits_and_stop_observer::<_, true>(
            root,
            previous_keys,
            &limits,
            stop,
            progress_interval,
            observer,
        )
    }

    /// Used for implementing stopped searches with a compile-time removable
    /// progress path.
    ///
    /// `REPORT_PROGRESS` selects at compile time whether the observer and
    /// interval are consulted. Terminal roots, empty permitted root sets, and
    /// pre-exhausted limits return deterministic fallbacks without running a
    /// simulation.
    ///
    /// # Arguments
    ///
    /// * `root` - position searched from
    /// * `previous_keys` - dense oldest-to-newest pre-root repetition keys
    /// * `limits` - per-invocation simulation, time, and root-move limits
    /// * `stop` - external cancellation token
    /// * `progress_interval` - minimum wall-clock time between reports
    /// * `observer` - callback receiving consistent tree snapshots
    ///
    /// # Returns
    ///
    /// Complete root statistics for the bounded invocation.
    fn search_with_history_limits_and_stop_observer<F, const REPORT_PROGRESS: bool>(
        &mut self,
        root: &Position,
        previous_keys: &[u64],
        limits: &MctsLimits,
        stop: &AtomicBool,
        progress_interval: Duration,
        mut observer: F,
    ) -> MctsResult
    where
        F: FnMut(&MctsResult),
    {
        let started = Instant::now();
        let mut last_progress = started;
        let mut root_keys = previous_keys.to_vec();
        root_keys.push(root.key());
        let complete_root_legal = sorted_legal_moves(root);
        if let Some((outcome, value)) =
            root_terminal(root, &root_keys, previous_keys.len(), &complete_root_legal)
        {
            return MctsResult {
                best_move: None,
                principal_variation: Vec::new(),
                simulations: 0,
                tree_nodes: 1,
                tree_capacity_exhausted: false,
                root_visits: 0,
                value,
                outcome: Some(outcome),
                root_moves: Vec::new(),
            };
        }
        let root_legal = permitted_root_moves(&complete_root_legal, limits.root_moves.as_deref());
        if root_legal.is_empty() {
            return zero_simulation_result(root_legal, false);
        }

        if limits.max_simulations == 0
            || stop.load(AtomicOrdering::Relaxed)
            || time_exhausted(started, limits)
        {
            return zero_simulation_result(root_legal, false);
        }

        let mut tree = vec![Node::root()];
        let mut scratch = SimulationScratch::new(root, &root_keys);
        let mut simulations = 0_u64;
        let mut tree_capacity_exhausted = false;
        let mut longest_simulation = Duration::ZERO;
        let simulation_control = SimulationControl {
            root,
            permitted_root_moves: &root_legal,
            stop,
            started,
            limits,
        };
        while simulations < limits.max_simulations
            && tree[0].proof == ProofState::Unknown
            && !stop.load(AtomicOrdering::Relaxed)
            && !time_exhausted(started, limits)
            && next_simulation_fits(started, limits, simulations, longest_simulation)
        {
            let simulation_started = Instant::now();
            let status = self.simulate(&simulation_control, &mut tree, &mut scratch);
            longest_simulation = longest_simulation.max(simulation_started.elapsed());
            match status {
                SimulationStatus::Completed => {
                    simulations += 1;
                    if REPORT_PROGRESS && last_progress.elapsed() >= progress_interval {
                        observer(&build_result(&tree, simulations, false));
                        last_progress = Instant::now();
                    }
                }
                SimulationStatus::TreeCapacityExhausted => {
                    tree_capacity_exhausted = true;
                    break;
                }
            }
        }
        if simulations == 0 {
            return zero_simulation_result(root_legal, tree_capacity_exhausted);
        }
        build_result(&tree, simulations, tree_capacity_exhausted)
    }

    /// Used for performing selection, expansion/evaluation, backup, and proof
    /// propagation.
    ///
    /// The position is rebuilt from the root while repetition and arena-path
    /// buffers retain their allocations between simulations.
    ///
    /// # Arguments
    ///
    /// * `control` - immutable root and cancellation context
    /// * `tree` - arena extended in place with newly expanded children
    /// * `scratch` - reusable board, key, and path storage
    ///
    /// # Panics
    ///
    /// Panics if arena invariants are violated: an empty simulation path, a
    /// non-root path node without an incoming move, or a generated child move
    /// that fails to apply.
    fn simulate(
        &mut self,
        control: &SimulationControl<'_>,
        tree: &mut Vec<Node>,
        scratch: &mut SimulationScratch,
    ) -> SimulationStatus {
        scratch.reset(control.root);
        self.evaluator.begin_mcts_path(control.root);
        let history_len = scratch.history_len;
        let position = &mut scratch.position;
        let keys = &mut scratch.keys;
        let path = &mut scratch.path;

        let leaf_value = loop {
            let node_index = *path.last().expect("simulation path contains root");
            if tree[node_index].proof != ProofState::Unknown {
                break tree[node_index].proof.value();
            }
            if let Some(value) = tree[node_index].terminal_value {
                break value;
            }
            if is_repetition_draw(keys, history_len, position.halfmove_clock()) {
                break 0.0;
            }

            if !tree[node_index].expanded {
                let legal = sorted_legal_moves(position);
                if legal.is_empty() {
                    let proof = if position.in_check(position.side_to_move()) {
                        ProofState::Loss
                    } else {
                        ProofState::Draw
                    };
                    mark_terminal(&mut tree[node_index], proof);
                    break proof.value();
                }
                if is_static_draw(position) {
                    mark_terminal(&mut tree[node_index], ProofState::Draw);
                    break 0.0;
                }

                let child_count = if node_index == 0 {
                    control.permitted_root_moves.len()
                } else {
                    legal.len()
                };
                let Some(children) =
                    reserve_tree_expansion(tree, child_count, control.limits.max_tree_nodes)
                else {
                    return SimulationStatus::TreeCapacityExhausted;
                };

                let prediction = self.evaluator.evaluate_policy_value(position, &legal);
                let priors = self.normalized_priors(position, &legal, &prediction);
                let child_policy = if node_index == 0 {
                    conditioned_root_policy(&legal, &priors, control.permitted_root_moves)
                } else {
                    legal.iter().copied().zip(priors).collect()
                };
                debug_assert_eq!(child_policy.len(), child_count);
                commit_tree_expansion(tree, node_index, position, child_policy, children);
                if tree[node_index].proof != ProofState::Unknown {
                    break tree[node_index].proof.value();
                }
                break self.quiescence_value(
                    position,
                    keys,
                    history_len,
                    if node_index == 0 {
                        control.permitted_root_moves
                    } else {
                        &legal
                    },
                    Some(&prediction),
                    0,
                    -1.0,
                    1.0,
                    control.stop,
                    control.started,
                    control.limits,
                );
            }

            let child_index = select_child(tree, node_index, self.config);
            let mv = tree[child_index]
                .incoming_move
                .expect("non-root node has an incoming move");
            position
                .make_move(mv)
                .expect("MCTS child was generated as legal");
            self.evaluator.mcts_path_position(position);
            keys.push(position.key());
            path.push(child_index);
        };

        backup(tree, path, leaf_value);
        propagate_proof(tree, path);
        SimulationStatus::Completed
    }

    /// Used for resolving forcing leaf tactics with a bounded negamax
    /// quiescence search.
    ///
    /// Checks search every legal evasion. Other nodes search tactical moves in
    /// deterministic order, discard losing non-promotion captures, and cap
    /// branching at `QUIESCENCE_MAX_MOVES`; recursion stops at
    /// `QUIESCENCE_MAX_PLY`.
    ///
    /// # Arguments
    ///
    /// * `position` - node position, mutated and restored around each move
    /// * `keys` - repetition keys extended and popped around each move
    /// * `history_len` - number of keys preceding the search root
    /// * `legal` - legal moves searched at `position`; the permitted subset
    ///   at a restricted root, the complete legal set elsewhere
    /// * `initial` - optional already-computed prediction for this node
    /// * `qply` - current quiescence depth below the expanded leaf
    /// * `alpha` - lower search bound for the side to move
    /// * `beta` - upper search bound for the side to move
    /// * `stop` - external cancellation token
    /// * `started` - invocation start instant for deadline checks
    /// * `limits` - per-invocation resource limits
    ///
    /// # Returns
    ///
    /// A bounded value in `[-1, 1]` from the side to move at `position`.
    ///
    /// # Panics
    ///
    /// Panics if a generated legal move fails to apply.
    #[allow(clippy::too_many_arguments)]
    fn quiescence_value(
        &mut self,
        position: &mut Position,
        keys: &mut Vec<u64>,
        history_len: usize,
        legal: &[Move],
        initial: Option<&PolicyValue>,
        qply: u8,
        mut alpha: f64,
        beta: f64,
        stop: &AtomicBool,
        started: Instant,
        limits: &MctsLimits,
    ) -> f64 {
        if legal.is_empty() {
            return if position.in_check(position.side_to_move()) {
                -1.0
            } else {
                0.0
            };
        }
        if is_search_draw(position, keys, history_len) {
            return 0.0;
        }
        if has_mate_in_one(position, legal) {
            return 1.0;
        }
        if let Some(prediction) = initial {
            if !self.evaluator.allows_mcts_quiescence() {
                return finite_value(prediction.value);
            }
        }
        if search_interrupted(stop, started, limits) {
            return initial.map_or(0.0, |prediction| finite_value(prediction.value));
        }

        let in_check = position.in_check(position.side_to_move());
        if qply >= QUIESCENCE_MAX_PLY {
            return initial.map_or_else(
                || finite_value(self.evaluator.evaluate_policy_value(position, legal).value),
                |prediction| finite_value(prediction.value),
            );
        }
        let stand_pat = if in_check {
            -1.0
        } else if let Some(prediction) = initial {
            finite_value(prediction.value)
        } else {
            finite_value(self.evaluator.evaluate_policy_value(position, legal).value)
        };
        if search_interrupted(stop, started, limits) {
            return stand_pat;
        }
        if !in_check {
            if stand_pat >= beta {
                return stand_pat;
            }
            alpha = alpha.max(stand_pat);
        }

        let mut moves = if in_check {
            legal.to_vec()
        } else {
            position.legal_tactical_moves()
        };
        if moves.is_empty() {
            return stand_pat;
        }
        order_quiescence_moves(position, &mut moves);
        if !in_check && moves.len() > QUIESCENCE_MAX_MOVES {
            moves.truncate(QUIESCENCE_MAX_MOVES);
        }

        let mut best = stand_pat;
        for mv in moves {
            if search_interrupted(stop, started, limits) {
                break;
            }
            if !in_check
                && mv.promotion().is_none()
                && position.is_capture(mv)
                && static_exchange_eval(position, mv) < 0
            {
                continue;
            }
            let undo = position
                .make_move(mv)
                .expect("quiescence move was generated as legal");
            keys.push(position.key());
            let child_legal = sorted_legal_moves(position);
            let value = -self.quiescence_value(
                position,
                keys,
                history_len,
                &child_legal,
                None,
                qply + 1,
                -beta,
                -alpha,
                stop,
                started,
                limits,
            );
            keys.pop();
            position.unmake_move(mv, undo);
            if value > best {
                best = value;
            }
            if value > alpha {
                alpha = value;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    /// Used for producing one normalized prior per legal move in the same
    /// order.
    ///
    /// Any supplied finite policy logits take precedence. Otherwise the method
    /// combines evaluator ordering and handcrafted tactical terms scaled by
    /// `POLICY_LOGIT_SCALE`.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose legal moves receive priors
    /// * `legal` - complete legal moves in stable order
    /// * `prediction` - evaluator output that may carry policy logits
    ///
    /// # Returns
    ///
    /// Normalized prior probabilities aligned with `legal`.
    fn normalized_priors(
        &mut self,
        position: &Position,
        legal: &[Move],
        prediction: &PolicyValue,
    ) -> Vec<f64> {
        let supplied = match_policy_logits(legal, &prediction.policy_logits);
        if supplied.iter().any(Option::is_some) {
            return softmax_with_missing_floor(&supplied);
        }

        let mut working = position.clone();
        let mut logits = Vec::with_capacity(legal.len());
        for mv in legal {
            let ordering = self.evaluator.move_ordering_score(position, *mv);
            let tactical = tactical_prior(position, &mut working, *mv, self.config);
            logits.push(Some(f64::from(ordering + tactical) / POLICY_LOGIT_SCALE));
        }
        softmax_with_missing_floor(&logits)
    }
}

/// Used for matching finite policy logits to legal moves without changing
/// duplicate rules.
///
/// The built-in neural evaluators preserve the supplied legal-move order while
/// omitting moves they cannot represent. That common subsequence uses a linear
/// pass. Arbitrary evaluator output retains the complete scan fallback,
/// including maximum selection for duplicate moves and omission of non-finite
/// values.
///
/// # Arguments
///
/// * `legal` - complete legal moves in stable order
/// * `policy` - evaluator-supplied move/logit pairs
///
/// # Returns
///
/// One optional finite logit per legal move, aligned with `legal`.
fn match_policy_logits(legal: &[Move], policy: &[(Move, f32)]) -> Vec<Option<f64>> {
    if policy.is_empty() {
        return vec![None; legal.len()];
    }
    let mut supplied = Vec::with_capacity(legal.len());
    let mut policy_index = 0_usize;
    for mv in legal {
        if policy_index < policy.len() && policy[policy_index].0 == *mv {
            let logit = policy[policy_index].1;
            supplied.push(logit.is_finite().then(|| f64::from(logit)));
            policy_index += 1;
        } else {
            supplied.push(None);
        }
    }
    if policy_index == policy.len() {
        supplied
    } else {
        match_unordered_policy_logits(legal, policy)
    }
}

/// Used for scanning arbitrary policy order while preserving the public
/// duplicate contract.
///
/// Duplicate entries for one move keep the maximum finite logit; non-finite
/// values are omitted entirely.
///
/// # Arguments
///
/// * `legal` - complete legal moves in stable order
/// * `policy` - evaluator-supplied move/logit pairs in any order
///
/// # Returns
///
/// One optional finite logit per legal move, aligned with `legal`.
fn match_unordered_policy_logits(legal: &[Move], policy: &[(Move, f32)]) -> Vec<Option<f64>> {
    legal
        .iter()
        .map(|mv| {
            policy
                .iter()
                .filter(|(candidate, logit)| candidate == mv && logit.is_finite())
                .map(|(_, logit)| f64::from(*logit))
                .max_by(f64::total_cmp)
        })
        .collect()
}

/// Used for reserving one complete child expansion without partial mutation.
///
/// The child-index vector is exact-sized. Arena capacity grows geometrically
/// up to the invocation ceiling so repeated expansions do not reallocate on
/// every simulation. Neither vector's length changes here, and both
/// reservations are fallible.
///
/// # Arguments
///
/// * `tree` - flat arena whose length remains unchanged on every outcome
/// * `child_count` - complete number of children appended by the expansion
/// * `max_tree_nodes` - invocation ceiling including the existing root
///
/// # Returns
///
/// An empty, fully reserved child-index vector when the batch fits and both
/// allocations succeed; `None` on arithmetic overflow, limit exhaustion, or
/// allocation refusal.
fn reserve_tree_expansion(
    tree: &mut Vec<Node>,
    child_count: usize,
    max_tree_nodes: usize,
) -> Option<Vec<usize>> {
    let required = tree.len().checked_add(child_count)?;
    if required > max_tree_nodes {
        return None;
    }

    let mut children = Vec::new();
    children.try_reserve_exact(child_count).ok()?;
    if tree.capacity() < required {
        let doubled = tree.capacity().max(1).saturating_mul(2);
        let target = required.max(doubled).min(max_tree_nodes);
        tree.try_reserve_exact(target.checked_sub(tree.len())?)
            .ok()?;
    }
    Some(children)
}

/// Used for atomically appending a previously reserved child batch.
///
/// Proof metadata is established before the parent publishes its child
/// indices and expanded flag. The caller has already reserved both vectors for
/// exactly `child_policy.len()` additions, so this commit cannot encounter a
/// growth allocation.
///
/// # Arguments
///
/// * `tree` - flat arena with capacity for every supplied child
/// * `node_index` - valid parent index, with zero identifying the root
/// * `position` - board represented by the parent, mutated and restored by
///   direct proof probes
/// * `child_policy` - complete move/prior batch in deterministic order
/// * `children` - empty index vector reserved for the complete batch
///
/// # Panics
///
/// Panics if `node_index` is outside the arena. Internal selection guarantees
/// that invariant.
fn commit_tree_expansion(
    tree: &mut Vec<Node>,
    node_index: usize,
    position: &mut Position,
    child_policy: Vec<(Move, f64)>,
    mut children: Vec<usize>,
) {
    for (mv, prior) in child_policy {
        let child = tree.len();
        let (proof, proof_plies) = if node_index == 0 {
            root_child_proof(position, mv)
        } else {
            let proof = direct_child_proof(position, mv);
            let plies = if proof == ProofState::Unknown {
                u16::MAX
            } else {
                0
            };
            (proof, plies)
        };
        tree.push(Node::child(mv, prior, proof, proof_plies));
        children.push(child);
    }
    tree[node_index].children = children;
    tree[node_index].expanded = true;
    refresh_proof(tree, node_index);
}

/// One arena node whose stored value is from that node's side to move.
///
/// Nodes live in a single flat `Vec` arena; children are referenced by index
/// and never removed, so indices stay stable for the whole search.
#[derive(Clone, Debug)]
struct Node {
    /// Used for the move from the parent, absent only for the root.
    ///
    /// Every non-root node stores the move that reached it.
    incoming_move: Option<Move>,
    /// Used for the normalized prior assigned during parent expansion.
    ///
    /// The root carries a unit prior.
    prior: f64,
    /// Used for counting simulations backed up through this node.
    ///
    /// Incremented once per backup pass.
    visits: u64,
    /// Used for accumulating backed-up values from this node's perspective.
    ///
    /// Divided by `visits` to obtain the mean value.
    value_sum: f64,
    /// Used for indices of children in the stable arena.
    ///
    /// Empty until the node is expanded with legal moves.
    children: Vec<usize>,
    /// Used for recording whether legal children or a terminal value have
    /// been established.
    ///
    /// Terminal children are created already expanded.
    expanded: bool,
    /// Used for the terminal or proven result, from this node's perspective.
    ///
    /// `Some` only for intrinsic terminals and proof-completed nodes.
    terminal_value: Option<f64>,
    /// Used for the exact minimax result known for the explored subtree.
    ///
    /// Remains unknown until children establish an exact result.
    proof: ProofState,
    /// Used for the plies to the chosen proof; unknown nodes use
    /// [`u16::MAX`].
    ///
    /// Distances saturate rather than overflow while propagating upward.
    proof_plies: u16,
}

/// Exact result known for a node from its own side-to-move perspective.
///
/// A child loss is a parent win; sibling ordering prefers fast wins and
/// delayed forced losses through the stored proof distances.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProofState {
    /// Used for indicating the explored subtree has not established an exact
    /// result.
    ///
    /// Maps to a neutral value of `0.0`.
    Unknown,
    /// Used for indicating the side to move can force a win.
    ///
    /// Maps to a value of `1.0`.
    Win,
    /// Used for indicating the side to move can force a draw but no known
    /// win.
    ///
    /// Maps to a value of `0.0`.
    Draw,
    /// Used for indicating every legal continuation loses.
    ///
    /// Maps to a value of `-1.0`.
    Loss,
}

impl ProofState {
    /// Used for mapping an exact proof to the MCTS value convention.
    ///
    /// # Returns
    ///
    /// `1.0` for a win, `-1.0` for a loss, and `0.0` for draws and unknown
    /// states.
    const fn value(self) -> f64 {
        match self {
            Self::Win => 1.0,
            Self::Draw | Self::Unknown => 0.0,
            Self::Loss => -1.0,
        }
    }
}

impl Node {
    /// Used for creating the unexpanded root with unit prior and no incoming
    /// move.
    ///
    /// # Returns
    ///
    /// A fresh root node with no visits, children, or proof.
    fn root() -> Self {
        Self {
            incoming_move: None,
            prior: 1.0,
            visits: 0,
            value_sum: 0.0,
            children: Vec::new(),
            expanded: false,
            terminal_value: None,
            proof: ProofState::Unknown,
            proof_plies: u16::MAX,
        }
    }

    /// Used for creating a child, marking immediately detected terminal
    /// results expanded.
    ///
    /// # Arguments
    ///
    /// * `mv` - move from the parent reaching this node
    /// * `prior` - normalized expansion prior
    /// * `proof` - exact result detected during expansion, if any
    /// * `proof_plies` - distance to the proof; `u16::MAX` when unknown
    ///
    /// # Returns
    ///
    /// A node pre-expanded with a terminal value when `proof` is known.
    fn child(mv: Move, prior: f64, proof: ProofState, proof_plies: u16) -> Self {
        debug_assert_eq!(proof == ProofState::Unknown, proof_plies == u16::MAX);
        Self {
            incoming_move: Some(mv),
            prior,
            visits: 0,
            value_sum: 0.0,
            children: Vec::new(),
            expanded: proof != ProofState::Unknown,
            terminal_value: (proof != ProofState::Unknown).then(|| proof.value()),
            proof,
            proof_plies,
        }
    }

    /// Used for computing the mean backed-up value, or zero before the first
    /// visit.
    ///
    /// # Returns
    ///
    /// `value_sum / visits`, or `0.0` when unvisited.
    fn q(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / visits_as_f64(self.visits)
        }
    }
}

/// Used for marking one expanded leaf as an intrinsic loss or draw.
///
/// Sets the terminal value, proof state, and zero proof distance in place.
///
/// # Arguments
///
/// * `node` - leaf node marked terminal
/// * `proof` - intrinsic result; must be a loss or draw
fn mark_terminal(node: &mut Node, proof: ProofState) {
    debug_assert!(matches!(proof, ProofState::Loss | ProofState::Draw));
    node.expanded = true;
    node.terminal_value = Some(proof.value());
    node.proof = proof;
    node.proof_plies = 0;
}

/// Used for validating all tunable parameters before they enter the selection
/// formulas.
///
/// # Arguments
///
/// * `config` - candidate configuration
///
/// # Errors
///
/// Returns [`MctsError`] when `cpuct` is non-finite or non-positive, when
/// `fpu_reduction` is non-finite or negative, or when any handcrafted prior
/// term is negative.
fn validate_config(config: MctsConfig) -> Result<(), MctsError> {
    if !config.cpuct.is_finite() || config.cpuct <= 0.0 {
        return Err(MctsError("cpuct must be finite and positive".into()));
    }
    if !config.fpu_reduction.is_finite() || config.fpu_reduction < 0.0 {
        return Err(MctsError(
            "fpu_reduction must be finite and non-negative".into(),
        ));
    }
    if config.check_prior_bonus < 0
        || config.losing_capture_prior_penalty < 0
        || config.winning_capture_prior_scale < 0
    {
        return Err(MctsError(
            "handcrafted prior bonuses and scales must be non-negative".into(),
        ));
    }
    Ok(())
}

/// Used for generating legal moves in stable UCI-text order for reproducible
/// searches.
///
/// # Arguments
///
/// * `position` - position whose legal moves are generated
///
/// # Returns
///
/// Legal moves sorted by their UCI order key.
fn sorted_legal_moves(position: &Position) -> Vec<Move> {
    let mut legal = position.legal_moves();
    legal.sort_by_key(|mv| mv.uci_order_key());
    legal
}

/// Used for intersecting an optional canonical root subset with complete
/// legal moves.
///
/// # Arguments
///
/// * `legal` - complete legal root moves
/// * `requested` - optional canonical subset from the caller
///
/// # Returns
///
/// Every legal move when unrestricted, otherwise the legal moves contained in
/// the requested subset; possibly empty.
fn permitted_root_moves(legal: &[Move], requested: Option<&[Move]>) -> Vec<Move> {
    requested.map_or_else(
        || legal.to_vec(),
        |requested| {
            legal
                .iter()
                .filter(|mv| requested.contains(mv))
                .copied()
                .collect()
        },
    )
}

/// Used for conditioning complete root priors over only the permitted child
/// moves.
///
/// The full legal policy has already been evaluated and normalized. If every
/// selected weight underflows to zero, the permitted subset receives a uniform
/// finite fallback instead of producing invalid PUCT arithmetic.
///
/// # Arguments
///
/// * `legal` - complete legal root moves in stable order
/// * `priors` - normalized priors aligned with `legal`
/// * `permitted` - non-empty permitted root subset
///
/// # Returns
///
/// Permitted moves paired with priors renormalized to unit mass.
fn conditioned_root_policy(legal: &[Move], priors: &[f64], permitted: &[Move]) -> Vec<(Move, f64)> {
    debug_assert_eq!(legal.len(), priors.len());
    debug_assert!(!permitted.is_empty());
    let mut selected: Vec<(Move, f64)> = legal
        .iter()
        .copied()
        .zip(priors.iter().copied())
        .filter(|(mv, _)| permitted.contains(mv))
        .collect();
    debug_assert_eq!(selected.len(), permitted.len());
    if selected.len() == legal.len() {
        return selected;
    }
    let mass = selected.iter().map(|(_, prior)| *prior).sum::<f64>();
    if mass.is_finite() && mass > 0.0 {
        for (_, prior) in &mut selected {
            *prior /= mass;
        }
    } else {
        let uniform = 1.0 / count_as_f64(selected.len());
        for (_, prior) in &mut selected {
            *prior = uniform;
        }
    }
    selected
}

/// Used for testing whether the earliest active wall-clock deadline has
/// elapsed.
///
/// # Arguments
///
/// * `started` - invocation start instant
/// * `limits` - per-invocation resource limits
///
/// # Returns
///
/// `true` when a hard deadline is active and fully consumed.
fn time_exhausted(started: Instant, limits: &MctsLimits) -> bool {
    limits.hard_remaining(started) == Some(Duration::ZERO)
}

/// Used for predicting whether another complete simulation fits its hard
/// deadline.
///
/// The first simulation is always admitted so a non-terminal search can obtain
/// one real policy/value result. Later simulations reserve the slowest observed
/// simulation plus a one-eighth jitter margin and one millisecond. This avoids
/// repeatedly starting an uninterruptible neural call just before its deadline
/// while remaining a no-op for fixed-work searches.
///
/// # Arguments
///
/// * `started` - invocation start instant
/// * `limits` - per-invocation resource limits
/// * `completed` - number of simulations already completed
/// * `longest_simulation` - slowest observed simulation duration
///
/// # Returns
///
/// `true` when the reserved cost fits inside the remaining deadline or no
/// deadline is active.
fn next_simulation_fits(
    started: Instant,
    limits: &MctsLimits,
    completed: u64,
    longest_simulation: Duration,
) -> bool {
    if completed == 0 || longest_simulation.is_zero() {
        return true;
    }
    let Some(remaining) = limits.hard_remaining(started) else {
        return true;
    };
    let reserve = longest_simulation
        .saturating_add(longest_simulation / 8)
        .saturating_add(Duration::from_millis(1));
    reserve < remaining
}

/// Used for testing whether cancellation or an elapsed deadline should stop
/// leaf work.
///
/// # Arguments
///
/// * `stop` - external cancellation token
/// * `started` - invocation start instant
/// * `limits` - per-invocation resource limits
///
/// # Returns
///
/// `true` when the stop token is set or a hard deadline has elapsed.
fn search_interrupted(stop: &AtomicBool, started: Instant, limits: &MctsLimits) -> bool {
    stop.load(AtomicOrdering::Relaxed) || time_exhausted(started, limits)
}

/// Used for building a deterministic legal fallback when no simulation can
/// begin.
///
/// The first permitted legal move, when present, becomes the best move and a
/// single-ply principal variation with zeroed statistics.
///
/// # Arguments
///
/// * `root_legal` - permitted legal root moves in stable order
/// * `tree_capacity_exhausted` - whether capacity refusal prevented all work
///
/// # Returns
///
/// A zero-simulation result reporting one arena node.
fn zero_simulation_result(root_legal: Vec<Move>, tree_capacity_exhausted: bool) -> MctsResult {
    MctsResult {
        best_move: root_legal.first().copied(),
        principal_variation: root_legal.first().copied().into_iter().collect(),
        simulations: 0,
        tree_nodes: 1,
        tree_capacity_exhausted,
        root_visits: 0,
        value: 0.0,
        outcome: None,
        root_moves: root_legal
            .into_iter()
            .map(|mv| RootMove {
                mv,
                visits: 0,
                value: 0.0,
                prior: 0.0,
            })
            .collect(),
    }
}

/// Used for detecting checkmate, stalemate, repetition, and static draws at
/// the root.
///
/// # Arguments
///
/// * `position` - root position
/// * `keys` - pre-root history keys followed by the root key
/// * `history_len` - number of keys preceding the root
/// * `legal` - complete legal root moves
///
/// # Returns
///
/// The outcome and root value when the root is terminal, otherwise `None`.
fn root_terminal(
    position: &Position,
    keys: &[u64],
    history_len: usize,
    legal: &[Move],
) -> Option<(GameOutcome, f64)> {
    if legal.is_empty() {
        if position.in_check(position.side_to_move()) {
            Some((GameOutcome::Loss, -1.0))
        } else {
            Some((GameOutcome::Draw, 0.0))
        }
    } else if is_search_draw(position, keys, history_len) {
        Some((GameOutcome::Draw, 0.0))
    } else {
        None
    }
}

/// Used for selecting a child using exact proofs first and PUCT otherwise.
///
/// Unvisited nodes use a first-play urgency below the parent mean, scaled by
/// prior mass already explored. Equal floating scores retain arena order.
///
/// # Arguments
///
/// * `tree` - arena containing the parent and its children
/// * `parent_index` - index of the expanded parent node
/// * `config` - PUCT exploration parameters
///
/// # Returns
///
/// The arena index of the selected child.
///
/// # Panics
///
/// Panics when the parent has no children.
fn select_child(tree: &[Node], parent_index: usize, config: MctsConfig) -> usize {
    let parent = &tree[parent_index];
    if let Some(proven) = proof_preferred_child(tree, parent) {
        return proven;
    }
    let parent_q = parent.q();
    let visit_scale = visits_as_f64(parent.visits.max(1)).sqrt();
    let visited_prior = parent
        .children
        .iter()
        .filter(|index| tree[**index].visits > 0)
        .map(|index| tree[*index].prior)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let first_play = parent_q - config.fpu_reduction * visited_prior.sqrt();
    let mut best = parent.children[0];
    let mut best_score = f64::NEG_INFINITY;
    for child_index in &parent.children {
        let child = &tree[*child_index];
        let exploitation = if child.proof != ProofState::Unknown {
            -child.proof.value()
        } else if child.visits == 0 {
            first_play
        } else {
            -child.q()
        };
        let exploration = if child.proof == ProofState::Unknown {
            config.cpuct * child.prior * visit_scale / (1.0 + visits_as_f64(child.visits))
        } else {
            0.0
        };
        let score = exploitation + exploration;
        if score > best_score {
            best = *child_index;
            best_score = score;
        }
    }
    best
}

/// Used for backing a leaf value through `path`, negating at each change of
/// side.
///
/// Visits increment and value sums accumulate from the leaf back to the root.
///
/// # Arguments
///
/// * `tree` - arena containing every node on the path
/// * `path` - node indices from the root through the leaf
/// * `leaf_value` - value from the leaf side-to-move perspective
fn backup(tree: &mut [Node], path: &[usize], leaf_value: f64) {
    let mut value = leaf_value;
    for index in path.iter().rev() {
        tree[*index].visits += 1;
        tree[*index].value_sum += value;
        value = -value;
    }
}

/// Used for recomputing exact minimax proofs from the leaf to the root.
///
/// Each node on the path is refreshed against the current exact states of its
/// children.
///
/// # Arguments
///
/// * `tree` - arena containing every node on the path
/// * `path` - node indices from the root through the leaf
fn propagate_proof(tree: &mut [Node], path: &[usize]) {
    for index in path.iter().rev() {
        refresh_proof(tree, *index);
    }
}

/// Used for updating one parent proof from the exact states of its children.
///
/// A losing child proves the parent win. If all children are known, any draw
/// proves a draw and otherwise every child win proves a loss. Proof distance
/// chooses the fastest win/draw and the slowest forced loss.
///
/// # Arguments
///
/// * `tree` - arena containing the parent and its children
/// * `node_index` - index of the parent; childless nodes are left unchanged
fn refresh_proof(tree: &mut [Node], node_index: usize) {
    if tree[node_index].children.is_empty() {
        return;
    }
    let mut winning_child: Option<u16> = None;
    let mut drawing_child: Option<u16> = None;
    let mut longest_losing_child: Option<u16> = None;
    let mut all_known = true;
    for child_index in tree[node_index].children.iter().copied() {
        let child = &tree[child_index];
        match child.proof {
            ProofState::Loss => {
                winning_child = Some(
                    winning_child.map_or(child.proof_plies, |plies| plies.min(child.proof_plies)),
                );
            }
            ProofState::Draw => {
                drawing_child = Some(
                    drawing_child.map_or(child.proof_plies, |plies| plies.min(child.proof_plies)),
                );
            }
            ProofState::Win => {
                longest_losing_child = Some(
                    longest_losing_child
                        .map_or(child.proof_plies, |plies| plies.max(child.proof_plies)),
                );
            }
            ProofState::Unknown => all_known = false,
        }
    }
    let proof = if let Some(plies) = winning_child {
        Some((ProofState::Win, plies.saturating_add(1)))
    } else if all_known {
        if let Some(plies) = drawing_child {
            Some((ProofState::Draw, plies.saturating_add(1)))
        } else {
            longest_losing_child.map(|plies| (ProofState::Loss, plies.saturating_add(1)))
        }
    } else {
        None
    };
    if let Some((proof, plies)) = proof {
        tree[node_index].proof = proof;
        tree[node_index].proof_plies = plies;
        tree[node_index].terminal_value = Some(proof.value());
    }
}

/// Used for choosing the distance-optimal exact continuation when proofs
/// permit one.
///
/// A proven winning child is always taken with the fastest proof. A drawing
/// child is taken only when the parent itself is proven drawn, and the
/// slowest losing child only when every child is exactly known.
///
/// # Arguments
///
/// * `tree` - arena containing the children
/// * `parent` - node whose children are inspected
///
/// # Returns
///
/// The preferred child index, or `None` when no exact choice applies.
fn proof_preferred_child(tree: &[Node], parent: &Node) -> Option<usize> {
    let mut winning: Option<usize> = None;
    let mut drawing: Option<usize> = None;
    let mut delaying_loss: Option<usize> = None;
    let mut all_known = true;
    for index in parent.children.iter().copied() {
        let child = &tree[index];
        match child.proof {
            ProofState::Loss => {
                if winning.map_or(true, |best| child.proof_plies < tree[best].proof_plies) {
                    winning = Some(index);
                }
            }
            ProofState::Draw => {
                if drawing.map_or(true, |best| child.proof_plies < tree[best].proof_plies) {
                    drawing = Some(index);
                }
            }
            ProofState::Win => {
                if delaying_loss.map_or(true, |best| child.proof_plies > tree[best].proof_plies) {
                    delaying_loss = Some(index);
                }
            }
            ProofState::Unknown => all_known = false,
        }
    }
    winning
        .or_else(|| {
            (parent.proof == ProofState::Draw)
                .then_some(drawing)
                .flatten()
        })
        .or_else(|| all_known.then_some(delaying_loss).flatten())
}

/// Used for sanitizing an evaluator value into the finite `[-1, 1]` search
/// interval.
///
/// # Arguments
///
/// * `value` - raw evaluator value
///
/// # Returns
///
/// The clamped finite value, or `0.0` for NaN and infinities.
fn finite_value(value: f32) -> f64 {
    if value.is_finite() {
        f64::from(value.clamp(-1.0, 1.0))
    } else {
        0.0
    }
}

/// Used for converting a visit count for PUCT floating-point arithmetic.
///
/// # Arguments
///
/// * `visits` - visit count
///
/// # Returns
///
/// The count as `f64`; values above 2^53 lose exact integer precision but
/// keep monotonic magnitude.
#[allow(clippy::cast_precision_loss)]
fn visits_as_f64(visits: u64) -> f64 {
    // PUCT is inherently floating-point. Counts above 2^53 retain monotonic
    // magnitude even though individual integers can no longer be represented.
    visits as f64
}

/// Used for converting a bounded move count for probability normalization.
///
/// # Arguments
///
/// * `count` - number of moves; must fit in `u32`
///
/// # Returns
///
/// The count as `f64`.
///
/// # Panics
///
/// Panics when `count` exceeds `u32::MAX`.
fn count_as_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("a legal move list fits u32"))
}

/// Used for normalizing optional logits while reserving probability for
/// missing moves.
///
/// Invalid or wholly absent supplied logits fall back to a uniform prior.
/// When at least one logit is missing, `MISSING_POLICY_MASS` is split
/// uniformly across the missing moves and the remainder across the supplied
/// ones.
///
/// # Arguments
///
/// * `logits` - one optional logit per legal move
///
/// # Returns
///
/// Normalized probabilities aligned with `logits`; empty input yields an
/// empty vector.
fn softmax_with_missing_floor(logits: &[Option<f64>]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let maximum = logits
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return vec![1.0 / count_as_f64(logits.len()); logits.len()];
    }

    let mut weights = Vec::with_capacity(logits.len());
    let mut valid_sum = 0.0;
    let mut missing_count = 0_usize;
    for logit in logits {
        if let Some(logit) = logit {
            let weight = (*logit - maximum).exp();
            weights.push(Some(weight));
            valid_sum += weight;
        } else {
            weights.push(None);
            missing_count += 1;
        }
    }
    if !valid_sum.is_finite() || valid_sum <= 0.0 {
        return vec![1.0 / count_as_f64(logits.len()); logits.len()];
    }
    let missing_mass = if missing_count == 0 {
        0.0
    } else {
        MISSING_POLICY_MASS
    };
    let supplied_mass = 1.0 - missing_mass;
    weights
        .into_iter()
        .map(|weight| {
            weight.map_or_else(
                || missing_mass / count_as_f64(missing_count),
                |weight| supplied_mass * weight / valid_sum,
            )
        })
        .collect()
}

/// Used for proving terminal root children and root moves allowing mate in
/// one.
///
/// The returned proof is from the opponent-to-move child perspective. This
/// exact root-wide safety pass prevents an unvisited neural-policy favorite
/// from outranking a safe move when the favorite permits mate on the reply.
/// The temporary root move and every tested reply are unmade before returning.
///
/// # Arguments
///
/// * `position` - root position; restored before returning
/// * `mv` - legal root move under test
///
/// # Returns
///
/// The child proof state and its ply distance; `u16::MAX` when unknown.
///
/// # Panics
///
/// Panics when `mv` is not legal in `position`.
fn root_child_proof(position: &mut Position, mv: Move) -> (ProofState, u16) {
    let undo = position
        .make_move(mv)
        .expect("MCTS root child was generated as legal");
    let replies = position.legal_moves();
    let result = if replies.is_empty() {
        if position.in_check(position.side_to_move()) {
            (ProofState::Loss, 0)
        } else {
            (ProofState::Draw, 0)
        }
    } else if is_static_draw(position) {
        (ProofState::Draw, 0)
    } else if has_mate_in_one(position, &replies) {
        (ProofState::Win, 1)
    } else {
        (ProofState::Unknown, u16::MAX)
    };
    position.unmake_move(mv, undo);
    result
}

/// Used for detecting immediate checkmate or static draw after one legal
/// move.
///
/// The temporary move is always unmade before returning.
///
/// # Arguments
///
/// * `position` - parent position; restored before returning
/// * `mv` - legal move under test
///
/// # Returns
///
/// The child proof state, or unknown when neither detection applies;
/// stalemate is not probed here and is detected later at child expansion.
///
/// # Panics
///
/// Panics when `mv` is not legal in `position`.
fn direct_child_proof(position: &mut Position, mv: Move) -> ProofState {
    let undo = position
        .make_move(mv)
        .expect("MCTS child was generated as legal");
    let proof = if position.in_check(position.side_to_move()) && position.legal_moves().is_empty() {
        ProofState::Loss
    } else if is_static_draw(position) {
        ProofState::Draw
    } else {
        ProofState::Unknown
    };
    position.unmake_move(mv, undo);
    proof
}

/// Used for testing whether any supplied legal move checkmates immediately.
///
/// Each temporary move is unmade before testing the next candidate.
///
/// # Arguments
///
/// * `position` - position to move from; restored before returning
/// * `legal` - candidate legal moves
///
/// # Returns
///
/// `true` when at least one candidate delivers checkmate.
///
/// # Panics
///
/// Panics when a candidate move is not legal in `position`.
fn has_mate_in_one(position: &mut Position, legal: &[Move]) -> bool {
    for mv in legal {
        let undo = position
            .make_move(*mv)
            .expect("mate candidate was generated as legal");
        let mate = position.in_check(position.side_to_move()) && position.legal_moves().is_empty();
        position.unmake_move(*mv, undo);
        if mate {
            return true;
        }
    }
    false
}

/// Used for ordering tactical moves by promotion, victim, attacker, and
/// exchange value.
///
/// Ties fall back to stable UCI-text order for determinism.
///
/// # Arguments
///
/// * `position` - position the moves belong to
/// * `moves` - tactical moves sorted in place, best first
fn order_quiescence_moves(position: &Position, moves: &mut [Move]) {
    moves.sort_by(|left, right| {
        quiescence_move_score(position, *right)
            .cmp(&quiescence_move_score(position, *left))
            .then_with(|| left.uci_order_key().cmp(&right.uci_order_key()))
    });
}

/// Used for computing the deterministic tactical ordering score for one legal
/// move.
///
/// Promotions score by piece value; captures combine victim value, attacker
/// value, and a clamped static exchange term.
///
/// # Arguments
///
/// * `position` - position the move belongs to
/// * `mv` - legal tactical move
///
/// # Returns
///
/// A larger score for more promising tactical moves.
///
/// # Panics
///
/// Panics when a capturing move has no piece on its origin square.
fn quiescence_move_score(position: &Position, mv: Move) -> i32 {
    let promotion = mv.promotion().map_or(0, |kind| piece_value(kind) * 32);
    let capture = captured_piece(position, mv).map_or(0, |victim| {
        let attacker = position
            .piece_at(mv.from())
            .expect("legal tactical move has a moving piece");
        piece_value(victim.kind) * 32 - piece_value(attacker.kind)
            + static_exchange_eval(position, mv).clamp(-2_000, 2_000) * 4
    });
    promotion + capture
}

/// Used for producing a handcrafted policy-logit term for tactical and
/// castling moves.
///
/// Captures gain victim value and lose attacker value, adjusted by static
/// exchange; promotions gain a bonus scaled by the promotion piece value,
/// castling a fixed bonus; a move that gives check earns the configured check
/// bonus. The temporary move on `working` is always unmade before returning.
///
/// # Arguments
///
/// * `position` - immutable position used for capture and castling tests
/// * `working` - scratch clone of `position` used to test for check
/// * `mv` - legal move receiving the prior term
/// * `config` - handcrafted-prior settings
///
/// # Returns
///
/// The combined logit bonus in ordering-score units.
///
/// # Panics
///
/// Panics when `mv` is not legal in `working` or a capture has no victim.
#[allow(clippy::comparison_chain)]
fn tactical_prior(
    position: &Position,
    working: &mut Position,
    mv: Move,
    config: MctsConfig,
) -> i32 {
    let mut bonus = 0_i32;
    if position.is_capture(mv) {
        let moving = position
            .piece_at(mv.from())
            .expect("legal move has a moving piece");
        let victim =
            captured_piece(position, mv).expect("capture has a direct or en-passant victim");
        bonus += piece_value(victim.kind) * 40;
        bonus -= piece_value(moving.kind) * 6;
        let exchange = static_exchange_eval(position, mv);
        if exchange < 0 {
            bonus -= config.losing_capture_prior_penalty;
        } else if exchange > 0 {
            bonus += exchange.min(900) * config.winning_capture_prior_scale;
        }
    }
    if let Some(promotion) = mv.promotion() {
        bonus += piece_value(promotion) * 35;
    }
    if position.is_castling_move(mv) {
        bonus += 12_000;
    }

    let undo = working
        .make_move(mv)
        .expect("prior move was generated as legal");
    if working.in_check(working.side_to_move()) {
        bonus += config.check_prior_bonus;
    }
    working.unmake_move(mv, undo);
    bonus
}

/// Used for retrieving the captured piece for ordinary and en-passant
/// captures.
///
/// A pawn moving diagonally onto the en-passant square captures the opposing
/// pawn even though the destination square is empty.
///
/// # Arguments
///
/// * `position` - position before the move
/// * `mv` - candidate capturing move
///
/// # Returns
///
/// The victim piece, or `None` for non-captures.
fn captured_piece(position: &Position, mv: Move) -> Option<Piece> {
    if let Some(piece) = position.piece_at(mv.to()) {
        return Some(piece);
    }
    let moving = position.piece_at(mv.from())?;
    if moving.kind == PieceKind::Pawn
        && mv.from().file() != mv.to().file()
        && Some(mv.to()) == position.en_passant_square()
    {
        Some(Piece::new(moving.color.opposite(), PieceKind::Pawn))
    } else {
        None
    }
}

/// Used for detecting true game-history threefolds and search-local
/// reversible cycles.
///
/// `history_len` splits the keys preceding the root from the root and its
/// descendants. Two matching keys are required before that boundary because
/// the current position is the third rule occurrence. One same-side match at
/// or after the boundary is deliberately scored as a draw to stop a reversible
/// search cycle. The result is path-dependent and must not be stored as an
/// intrinsic terminal proof in the tree.
///
/// # Arguments
///
/// * `keys` - repetition keys ending with the current position key
/// * `history_len` - number of keys preceding the search root
/// * `halfmove_clock` - reversible-ply count bounding the scan distance
///
/// # Returns
///
/// `true` when the current position should be scored as a repetition draw.
fn is_repetition_draw(keys: &[u64], history_len: usize, halfmove_clock: u16) -> bool {
    let Some((&current, preceding)) = keys.split_last() else {
        return false;
    };
    debug_assert!(history_len <= preceding.len());
    let reversible = usize::from(halfmove_clock).min(preceding.len());
    let current_index = preceding.len();
    let mut historical_occurrences = 0_u8;
    for distance in (2..=reversible).step_by(2) {
        let index = current_index - distance;
        if keys[index] != current {
            continue;
        }
        if index >= history_len {
            return true;
        }
        historical_occurrences += 1;
        if historical_occurrences >= 2 {
            return true;
        }
    }
    false
}

/// Used for detecting a repetition/cycle estimate or a position-intrinsic
/// rule draw.
///
/// # Arguments
///
/// * `position` - current position
/// * `keys` - repetition keys ending with the current position key
/// * `history_len` - number of keys preceding the search root
///
/// # Returns
///
/// `true` when either the repetition or the static draw detection fires.
fn is_search_draw(position: &Position, keys: &[u64], history_len: usize) -> bool {
    is_repetition_draw(keys, history_len, position.halfmove_clock()) || is_static_draw(position)
}

/// Used for detecting the fifty-move rule and supported
/// insufficient-material cases.
///
/// # Arguments
///
/// * `position` - position tested for an intrinsic rule draw
///
/// # Returns
///
/// `true` at a halfmove clock of one hundred or more, or with insufficient
/// material.
fn is_static_draw(position: &Position) -> bool {
    position.halfmove_clock() >= 100 || position.is_insufficient_material()
}

/// Used for converting the arena into stable public root statistics and a PV.
///
/// Root children are sorted with the deterministic sibling ordering; a proven
/// root reports its exact value and outcome.
///
/// # Arguments
///
/// * `tree` - complete search arena with the root at index zero
/// * `simulations` - number of completed simulations
/// * `tree_capacity_exhausted` - whether the next expansion was refused
///
/// # Returns
///
/// The public result mirroring the current tree state.
///
/// # Panics
///
/// Panics if a root child lacks an incoming move.
fn build_result(tree: &[Node], simulations: u64, tree_capacity_exhausted: bool) -> MctsResult {
    let root = &tree[0];
    let mut root_children = root.children.clone();
    root_children.sort_by(|left, right| compare_tree_children(tree, *left, *right));
    let root_moves: Vec<RootMove> = root_children
        .iter()
        .map(|index| {
            let child = &tree[*index];
            RootMove {
                mv: child.incoming_move.expect("root child has a move"),
                visits: child.visits,
                value: root_child_value(child),
                prior: child.prior,
            }
        })
        .collect();
    let best_move = root_moves.first().map(|entry| entry.mv);
    let principal_variation = principal_variation(tree);
    MctsResult {
        best_move,
        principal_variation,
        simulations,
        tree_nodes: tree.len(),
        tree_capacity_exhausted,
        root_visits: root.visits,
        value: if root.proof == ProofState::Unknown {
            root.q()
        } else {
            root.proof.value()
        },
        outcome: match root.proof {
            ProofState::Unknown => None,
            ProofState::Win => Some(GameOutcome::Win),
            ProofState::Draw => Some(GameOutcome::Draw),
            ProofState::Loss => Some(GameOutcome::Loss),
        },
        root_moves,
    }
}

/// Used for converting a child's node-relative value to the root parent's
/// perspective.
///
/// Proven children report their negated exact proof value; unvisited unproven
/// children report `0.0`.
///
/// # Arguments
///
/// * `child` - root child node
///
/// # Returns
///
/// The child value from the parent side-to-move perspective.
fn root_child_value(child: &Node) -> f64 {
    if child.proof == ProofState::Unknown {
        if child.visits == 0 {
            0.0
        } else {
            -child.q()
        }
    } else {
        -child.proof.value()
    }
}

/// Used for following the strongest deterministically ordered child until an
/// unvisited leaf.
///
/// # Arguments
///
/// * `tree` - complete search arena with the root at index zero
///
/// # Returns
///
/// The move sequence of the principal variation, possibly empty.
///
/// # Panics
///
/// Panics if a node on the variation lacks an incoming move.
fn principal_variation(tree: &[Node]) -> Vec<Move> {
    let mut pv = Vec::new();
    let mut node_index = 0_usize;
    while !tree[node_index].children.is_empty() {
        let mut children = tree[node_index].children.clone();
        children.sort_by(|left, right| compare_tree_children(tree, *left, *right));
        node_index = children[0];
        pv.push(
            tree[node_index]
                .incoming_move
                .expect("PV child has an incoming move"),
        );
        if tree[node_index].visits == 0 {
            break;
        }
    }
    pv
}

/// Used for ordering siblings by proof, distance, visits, value, prior, then
/// move text.
///
/// The final UCI-text comparison makes the ordering total and deterministic.
///
/// # Arguments
///
/// * `tree` - arena containing both siblings
/// * `left` - arena index of the first sibling
/// * `right` - arena index of the second sibling
///
/// # Returns
///
/// The ordering that ranks the stronger sibling first.
///
/// # Panics
///
/// Panics if either sibling lacks an incoming move.
fn compare_tree_children(tree: &[Node], left: usize, right: usize) -> Ordering {
    let left = &tree[left];
    let right = &tree[right];
    proof_rank(right.proof)
        .cmp(&proof_rank(left.proof))
        .then_with(|| compare_proof_distance(left, right))
        .then_with(|| right.visits.cmp(&left.visits))
        .then_with(|| root_child_value(right).total_cmp(&root_child_value(left)))
        .then_with(|| right.prior.total_cmp(&left.prior))
        .then_with(|| {
            let left_key = left.incoming_move.expect("child has move").uci_order_key();
            let right_key = right.incoming_move.expect("child has move").uci_order_key();
            left_key.cmp(&right_key)
        })
}

/// Used for ranking child proofs from the parent's perspective.
///
/// A child loss (a parent win) ranks highest, then draw, unknown, and a child
/// win last.
///
/// # Arguments
///
/// * `proof` - child proof state
///
/// # Returns
///
/// A rank where larger values sort earlier for the parent.
const fn proof_rank(proof: ProofState) -> u8 {
    match proof {
        // Child loss is a proven win for the parent.
        ProofState::Loss => 3,
        ProofState::Draw => 2,
        ProofState::Unknown => 1,
        ProofState::Win => 0,
    }
}

/// Used for preferring fast wins/draws and delaying forced losses between
/// equal proof types.
///
/// Child losses and draws compare fastest-first even when mixed; pairs
/// involving an unknown proof, or a child win paired with any other type,
/// compare equal.
///
/// # Arguments
///
/// * `left` - first sibling node
/// * `right` - second sibling node
///
/// # Returns
///
/// The ordering that ranks the preferred proof distance first.
fn compare_proof_distance(left: &Node, right: &Node) -> Ordering {
    match (left.proof, right.proof) {
        // Win quickly, draw quickly, and delay a forced loss.
        (ProofState::Loss | ProofState::Draw, ProofState::Loss | ProofState::Draw) => {
            left.proof_plies.cmp(&right.proof_plies)
        }
        (ProofState::Win, ProofState::Win) => right.proof_plies.cmp(&left.proof_plies),
        _ => Ordering::Equal,
    }
}

/// Used for conventional pin-agnostic static exchange evaluation in
/// centipawns.
///
/// The exchange is simulated on a local 64-square board copy: each side
/// recaptures with its least valuable attacker, pawn recaptures on the back
/// rank promote to queens, and the gain sequence is minimaxed from the tail.
/// Non-captures score zero.
///
/// # Arguments
///
/// * `position` - position before the capture
/// * `mv` - candidate capture, possibly en passant
///
/// # Returns
///
/// The expected material balance of the capture sequence in centipawns.
///
/// # Panics
///
/// Panics if a derived square index is invalid; inputs built from legal moves
/// cannot trigger this.
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

    let mut board = [None; 64];
    for index in 0_u8..64 {
        let square = Square::new(index).expect("loop index is a square");
        board[index as usize] = position.piece_at(square);
    }
    board[mv.from().index() as usize] = None;
    board[capture_square.index() as usize] = None;
    let placed_kind = mv.promotion().unwrap_or(moving.kind);
    board[mv.to().index() as usize] = Some(Piece::new(moving.color, placed_kind));

    let mut gains = Vec::with_capacity(32);
    let promotion_gain = mv.promotion().map_or(0, |kind| {
        see_piece_value(kind) - see_piece_value(PieceKind::Pawn)
    });
    gains.push(see_piece_value(victim.kind) + promotion_gain);
    let mut occupant = placed_kind;
    let mut side = moving.color.opposite();

    while let Some((attacker_square, attacker)) = least_valuable_attacker(&board, mv.to(), side) {
        let next = see_piece_value(occupant) - *gains.last().expect("SEE has initial gain");
        let promoted = attacker.kind == PieceKind::Pawn
            && ((side == Color::White && mv.to().row() == 0)
                || (side == Color::Black && mv.to().row() == 7));
        let next_occupant = if promoted {
            PieceKind::Queen
        } else {
            attacker.kind
        };
        let next = if promoted {
            next + see_piece_value(PieceKind::Queen) - see_piece_value(PieceKind::Pawn)
        } else {
            next
        };
        gains.push(next);
        board[attacker_square.index() as usize] = None;
        board[mv.to().index() as usize] = Some(Piece::new(side, next_occupant));
        occupant = next_occupant;
        side = side.opposite();
    }

    for index in (1..gains.len()).rev() {
        gains[index - 1] = -(-gains[index - 1]).max(gains[index]);
    }
    gains[0]
}

/// Used for finding the least valuable geometric attacker, breaking ties by
/// square index.
///
/// # Arguments
///
/// * `board` - local SEE board
/// * `target` - square being contested
/// * `side` - color of the attacking side
///
/// # Returns
///
/// The attacker square and piece, or `None` when the side has no geometric
/// attacker.
fn least_valuable_attacker(
    board: &[Option<Piece>; 64],
    target: Square,
    side: Color,
) -> Option<(Square, Piece)> {
    let mut best: Option<(Square, Piece, i32)> = None;
    for index in 0_u8..64 {
        let square = Square::new(index).expect("loop index is a square");
        let Some(piece) = board[index as usize] else {
            continue;
        };
        if piece.color != side || !piece_attacks(board, square, target, piece) {
            continue;
        }
        let value = see_piece_value(piece.kind);
        if best.map_or(true, |(best_square, _, best_value)| {
            value < best_value || (value == best_value && square < best_square)
        }) {
            best = Some((square, piece, value));
        }
    }
    best.map(|(square, piece, _)| (square, piece))
}

/// Used for testing whether `piece` geometrically attacks `target` on the
/// local SEE board.
///
/// Sliding pieces additionally require a clear ray; pins and full legality
/// are deliberately ignored.
///
/// # Arguments
///
/// * `board` - local SEE board
/// * `from` - square the piece stands on
/// * `target` - square tested for attack
/// * `piece` - attacking piece and color
///
/// # Returns
///
/// `true` when the geometry (and any required ray) permits the attack.
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

/// Used for testing that every square strictly between two aligned endpoints
/// is empty.
///
/// A zero step pair is rejected outright; walking off the board also fails.
///
/// # Arguments
///
/// * `board` - local SEE board
/// * `from` - ray origin, excluded from the test
/// * `target` - ray destination, excluded from the test
/// * `file_step` - per-square file increment
/// * `row_step` - per-square row increment
///
/// # Returns
///
/// `true` when every intermediate square is empty.
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

/// Used for retrieving exchange values with a finite, dominant value for the
/// king.
///
/// # Arguments
///
/// * `kind` - piece kind exchanged
///
/// # Returns
///
/// The centipawn exchange value; kings use `20_000`.
const fn see_piece_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::King => 20_000,
        _ => piece_value(kind),
    }
}
