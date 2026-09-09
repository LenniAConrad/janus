#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

//! Universal Chess Interface process boundary for Janus.
//!
//! A coordinator thread remains responsive to protocol commands while one
//! owned search or developer-perft job runs in the background. Only the coordinator writes to
//! standard output, which preserves line ordering and guarantees one final
//! `bestmove` per accepted search. The binary exposes dependency-free
//! alpha-beta and MCTS search, bounded helper threads, `MultiPV`, evaluator
//! selection, clock management, checked perft diagnostics, and Chess960 wire
//! formatting.
//!
//! # Evaluator selection
//!
//! Two options interact and their contract is load-bearing:
//!
//! * `Eval` chooses which evaluator plays: `Classical` (the hand-crafted
//!   evaluation, and the shipped default), `CompactNNUE`, `UpstreamNNUE`,
//!   `CNN`, `BT4`, or `JRBT`.
//! * `Eval File` loads a network from disk.
//!
//! **`Eval File` never changes which evaluator is active once `Eval` has been
//! set explicitly.** Loading a network while `Eval=Classical` stands keeps the
//! hand-crafted evaluator playing and reports so on an info string; the network
//! is retained and a later `Eval` switch activates it without reloading. When
//! no explicit `Eval` has been given, a successful load activates the network,
//! which keeps the common single-option configuration working.
//!
//! This ordering independence matters because match harnesses send both options
//! at engine start and do not guarantee an order. Before this contract was
//! enforced, `Eval File` installed the loaded network unconditionally, so every
//! match configured as `Eval=Classical` **silently played the network instead**
//! and no evaluation experiment on the hand-crafted evaluator measured
//! anything. See `eval_file_does_not_override_an_explicit_classical_selection`
//! for the regression test that pins both directions.

/// Bounded parser for commands received from a UCI controller.
///
/// Supplies the `Command` grammar together with the `go`, `perft`, and
/// `position` option types, plus the bounded line reader used by the
/// dedicated input thread.
mod protocol;

use janus_core::{
    checked_divide, detailed_divide, detailed_perft, Move, PerftDivideResult, PerftError,
    PerftNodeResult, PerftStats, Position, MAX_PERFT_DEPTH,
};
use janus_engine::jrbt::JrbtNetwork;
use janus_engine::syzygy::{SyzygyConfig, Tablebases};
use janus_engine::{
    clock_budgets, eval_cache_size_for_threads, mate_moves, set_eval_cache_size, AlphaBeta,
    Bt4Backend, Bt4BackendStatus, Bt4Network, Classical, CnnModel, CompactNnue, Evaluator,
    PolicyValue, SearchClock, SearchError, SearchEvaluator, SearchInfo, SearchLimits, SearchResult,
    SearchStateError, SharedTranspositionTable, TranspositionTable, UpstreamNnue,
    UpstreamNnueEvaluator, MAX_DEPTH,
};
#[cfg(feature = "mcts")]
use janus_engine::{Mcts, MctsLimits, MctsResult, DEFAULT_MCTS_TREE_NODE_LIMIT};
use protocol::{Command, GoOptions, PerftFormat, PerftOptions, PositionSource, PositionSpec};
use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender as Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Used for the dated version reported by the UCI handshake.
const ENGINE_VERSION: &str = "2026-09-09";
/// Used for bounding the number of independent alpha-beta workers exposed
/// through UCI.
///
/// The `Threads` spin option advertises this value as its maximum, and
/// [`alpha_beta_workers`] clamps every requested pool size to it.
const MAX_SEARCH_THREADS: usize = 1_024;
/// Used for bounding the number of ranked principal variations reported by
/// alpha-beta.
///
/// The `MultiPV` spin option advertises this value as its maximum.
const MAX_MULTI_PV: usize = 8;
/// Used for bounding parsed input and progress events waiting for the UCI
/// coordinator.
///
/// Backpressure at this boundary prevents a fast stdin producer from turning
/// bounded 16-KiB lines into unbounded process memory while a search worker is
/// completing a non-interruptible evaluation.
const COORDINATOR_QUEUE_CAPACITY: usize = 128;
/// Used for bounding state-changing commands deferred behind one active job.
///
/// Normal GUIs need only a handful of entries. The separate cap is necessary
/// even with a bounded coordinator channel because the coordinator can drain
/// that channel into this queue faster than a stopped neural prediction can
/// return.
const MAX_PENDING_COMMANDS: usize = 64;
/// Used for sizing the aggregate transposition-table budget in mebibytes
/// before any `Hash` option arrives.
///
/// Advertised as the `Hash` spin default and applied by
/// [`EngineState::new`].
const DEFAULT_HASH_MB: usize = 32;
/// Used for bounding the accepted aggregate transposition-table budget in
/// mebibytes.
///
/// The `Hash` spin option advertises this value as its maximum, and
/// [`alpha_beta_workers`] clamps the configured budget to it.
const MAX_HASH_MB: usize = 1_048_576;
/// Used as the smallest transposition allocation the engine will run with.
///
/// Reached only when a requested hash cannot be allocated at a larger size.
/// If even this allocation fails, the UCI pool remains empty and a search
/// fails closed with a null move instead of retrying through an infallible
/// allocator.
const MIN_TT_ENTRIES: usize = TT_ENTRIES_PER_MB;

/// Used for converting a mebibyte hash budget into an approximate number of
/// compact transposition entries.
///
/// [`alpha_beta_workers`] multiplies the configured `Hash` value by this
/// factor before rounding the entry count down to a power of two.
const TT_ENTRIES_PER_MB: usize = 1 << 15;
/// Used for reserving safety time subtracted from clock-derived move
/// budgets before any `Move Overhead` option arrives.
///
/// Advertised as the `Move Overhead` spin default.
const DEFAULT_MOVE_OVERHEAD_MS: u64 = 10;
/// Used for bounding the accepted `Clock Moves To Go` setting.
///
/// Alternative configuration retained for controlled evaluation.
const MAX_CLOCK_MOVES_TO_GO: u32 = 200;
/// Used for bounding the accepted `Falling Eval Percent` setting.
///
/// Alternative configuration retained for controlled evaluation.
const MAX_FALLING_EVAL_PERCENT: i32 = 50;
/// Used for bounding the accepted `Root Fail High Reduction` setting.
///
/// Alternative configuration retained for controlled evaluation.
const MAX_ROOT_FAIL_HIGH_REDUCTION: i32 = 8;
/// Used for bounding the accepted `Move Overhead` setting in milliseconds.
///
/// The `Move Overhead` spin option advertises this value as its maximum.
const MAX_MOVE_OVERHEAD_MS: u64 = 5_000;
/// Used for bounding the intended silence between complete MCTS progress
/// snapshots in milliseconds.
///
/// Unbounded and ponder MCTS jobs pass this interval to the progress-aware
/// search so periodic `info` lines keep flowing.
#[cfg(feature = "mcts")]
const MCTS_INFO_INTERVAL_MS: u64 = 250;
/// Used for bounding the GPU ordinal accepted through the UCI `BT4 Device`
/// selector.
///
/// The `BT4 Device` spin option advertises this value as its maximum.
const MAX_BT4_DEVICE_INDEX: usize = 15;
/// Used for defaulting the minimum remaining depth at which boundary
/// tablebase positions are probed.
///
/// The `SyzygyProbeDepth` spin option advertises this value as its default.
///
/// Measured in PERF-219 on endgame positions with the 3-4-5 set: probing at
/// every depth costs **-7.2%** node throughput, while a depth-4 gate turns the
/// same tables into **+5.1%** — a tablebase cutoff returns without evaluating
/// or generating moves, so gated probing pays for itself. Depth 8 is
/// comparable at +4.7% but lands far fewer hits (3,895 against 32,209), and
/// depth 12 recovers almost nothing (111 hits).
const DEFAULT_SYZYGY_PROBE_DEPTH: u8 = 4;
/// Used for bounding the accepted `SyzygyProbeDepth` setting.
///
/// The `SyzygyProbeDepth` spin option advertises this value as its maximum.
const MAX_SYZYGY_PROBE_DEPTH: u8 = 100;
/// Used for defaulting and bounding the probed piece count.
///
/// Syzygy tables exist up to seven men, so the `SyzygyProbeLimit` spin
/// option advertises this value as both its default and maximum.
const MAX_SYZYGY_PROBE_LIMIT: u8 = 7;

/// Search algorithm selected for the next UCI `go` command.
///
/// Stored in [`EngineState::search_mode`] and changed through the `Search`
/// combo option; [`ActiveSearch::mode`] records the algorithm actually
/// running.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchMode {
    /// Used for selecting iterative-deepening negamax alpha-beta search.
    AlphaBeta,
    /// Used for selecting single-thread PUCT Monte Carlo tree search.
    #[cfg(feature = "mcts")]
    Mcts,
}

/// Concrete evaluator families exposed through the UCI `Eval` option.
///
/// Each variant names one loadable backend; [`JanusEvaluator::kind`] reports
/// which family a constructed evaluator belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvaluatorKind {
    /// Used for naming the dependency-free handcrafted evaluation.
    Classical,
    /// Used for naming Janus' compact floating-point NNUE format.
    CompactNnue,
    /// Used for naming supported Stockfish `CURRENT/BIG` NNUE containers.
    UpstreamNnue,
    /// Used for naming CRTK's `LC0J` residual convolutional network.
    Cnn,
    /// Used for naming CRTK's executable `BT4J` attention policy/value
    /// network.
    Bt4,
    /// Used for naming the E04 `JRBT` relation-biased transformer.
    Jrbt,
}

impl EvaluatorKind {
    /// Used for retrieving the stable UCI spelling of this family in
    /// diagnostics.
    ///
    /// # Returns
    ///
    /// Static family name, e.g. `"CompactNNUE"` for [`Self::CompactNnue`].
    const fn uci_name(self) -> &'static str {
        match self {
            Self::Classical => "Classical",
            Self::CompactNnue => "CompactNNUE",
            Self::UpstreamNnue => "UpstreamNNUE",
            Self::Cnn => "CNN",
            Self::Bt4 => "BT4",
            Self::Jrbt => "JRBT",
        }
    }

    /// Used for checking whether this evaluator can execute the selected
    /// search safely.
    ///
    /// Only the BT4 family is restricted: it supports MCTS but not
    /// alpha-beta.
    ///
    /// # Arguments
    ///
    /// * `mode` - search algorithm requested for the next `go`
    ///
    /// # Returns
    ///
    /// `true` unless `self` is [`Self::Bt4`] and `mode` is
    /// [`SearchMode::AlphaBeta`].
    const fn supports_search(self, mode: SearchMode) -> bool {
        !matches!(
            (self, mode),
            (Self::Bt4 | Self::Jrbt, SearchMode::AlphaBeta)
        )
    }
}

/// Parsed evaluator request, including the backwards-compatible generic NNUE
/// alias.
///
/// Produced by [`parse_evaluator_selection`] from the UCI `Eval` combo value
/// and matched against the family of the currently loaded model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvaluatorSelection {
    /// Used for selecting one exact model family.
    Exact(EvaluatorKind),
    /// Used for selecting either compact or upstream NNUE according to the
    /// loaded file.
    AnyNnue,
}

impl EvaluatorSelection {
    /// Used for checking whether a loaded evaluator implements this
    /// selection.
    ///
    /// # Arguments
    ///
    /// * `kind` - family of the loaded evaluator
    ///
    /// # Returns
    ///
    /// `true` when `kind` matches exactly, or when the selection is
    /// [`Self::AnyNnue`] and `kind` is either NNUE family.
    fn accepts(self, kind: EvaluatorKind) -> bool {
        match self {
            Self::Exact(expected) => expected == kind,
            Self::AnyNnue => matches!(
                kind,
                EvaluatorKind::CompactNnue | EvaluatorKind::UpstreamNnue
            ),
        }
    }

    /// Used for retrieving the user-facing family name for a mismatch
    /// diagnostic.
    ///
    /// # Returns
    ///
    /// The exact family's UCI spelling, or `"NNUE"` for [`Self::AnyNnue`].
    const fn uci_name(self) -> &'static str {
        match self {
            Self::Exact(kind) => kind.uci_name(),
            Self::AnyNnue => "NNUE",
        }
    }
}

/// Cloneable evaluator configuration shared when constructing search workers.
///
/// Immutable network weights are shared through [`Arc`]; backends that keep
/// mutable inference scratch (CNN and BT4) are additionally serialized behind
/// a [`Mutex`] so clones stay safe across worker threads.
#[derive(Clone)]
enum JanusEvaluator {
    /// Used for evaluating with the dependency-free handcrafted evaluation.
    Classical,
    /// Used for evaluating with the Janus compact NNUE network behind
    /// immutable shared weights.
    CompactNnue(Arc<CompactNnue>),
    /// Used for evaluating with a supported upstream NNUE network behind
    /// immutable shared weights.
    UpstreamNnue(UpstreamNnueEvaluator),
    /// Used for evaluating with the CNN model serialized behind a mutex for
    /// mutable inference scratch.
    Cnn(Arc<Mutex<CnnModel>>),
    /// Used for evaluating with the BT4 network serialized behind a mutex
    /// for mutable inference scratch.
    Bt4(Arc<Mutex<Bt4Network>>),
    /// Used for evaluating with the JRBT relation-biased transformer behind a
    /// mutex for mutable inference scratch.
    Jrbt(Arc<Mutex<JrbtNetwork>>),
}

impl JanusEvaluator {
    /// Used for retrieving the concrete model family represented by this
    /// evaluator.
    ///
    /// # Returns
    ///
    /// Matching [`EvaluatorKind`] variant.
    const fn kind(&self) -> EvaluatorKind {
        match self {
            Self::Classical => EvaluatorKind::Classical,
            Self::CompactNnue(_) => EvaluatorKind::CompactNnue,
            Self::UpstreamNnue(_) => EvaluatorKind::UpstreamNnue,
            Self::Cnn(_) => EvaluatorKind::Cnn,
            Self::Bt4(_) => EvaluatorKind::Bt4,
            Self::Jrbt(_) => EvaluatorKind::Jrbt,
        }
    }
}

impl SearchEvaluator for JanusEvaluator {
    /// Used for evaluating a position with the currently selected backend.
    ///
    /// CNN and BT4 lock their shared network, recovering from a poisoned
    /// mutex by taking the inner value; compact NNUE output is rounded
    /// through [`compact_score`].
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Backend evaluation score as an `i32`.
    fn evaluate(&mut self, position: &Position) -> i32 {
        match self {
            Self::Classical => classical_evaluate_position(position),
            Self::CompactNnue(network) => compact_score(network.evaluate_centipawns(position)),
            Self::UpstreamNnue(evaluator) => SearchEvaluator::evaluate(evaluator, position),
            Self::Cnn(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::evaluate(&mut *network, position)
            }
            Self::Bt4(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::evaluate(&mut *network, position)
            }
            Self::Jrbt(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::evaluate(&mut *network, position)
            }
        }
    }

    /// Used for initializing evaluator-owned alpha-beta state at the current
    /// root.
    ///
    /// Only the upstream NNUE evaluator maintains such state; every other
    /// backend ignores the call.
    ///
    /// # Arguments
    ///
    /// * `root` - root position of the upcoming search
    /// * `max_plies` - deepest ply the search may reach
    ///
    /// # Returns
    ///
    /// Success after any upstream state is ready.
    ///
    /// # Errors
    ///
    /// Propagates an upstream search-state allocation refusal.
    fn begin_search(&mut self, root: &Position, max_plies: usize) -> Result<(), SearchStateError> {
        if let Self::UpstreamNnue(evaluator) = self {
            evaluator.begin_search(root, max_plies)?;
        }
        Ok(())
    }

    /// Used for reporting whether the selected evaluator opened move-aligned
    /// state.
    ///
    /// # Returns
    ///
    /// `true` only when the upstream NNUE evaluator reports incremental
    /// state.
    fn uses_incremental_search_state(&self) -> bool {
        matches!(self, Self::UpstreamNnue(evaluator) if evaluator.uses_incremental_search_state())
    }

    /// Used for forwarding already-made moves to the incremental upstream
    /// evaluator.
    ///
    /// Backends other than upstream NNUE ignore the notification.
    ///
    /// # Arguments
    ///
    /// * `child` - resulting child position
    /// * `mv` - move that was played
    /// * `undo` - undo record produced by making the move
    /// * `ply` - search ply at which the move was made
    fn move_played(&mut self, child: &Position, mv: Move, undo: janus_core::Undo, ply: usize) {
        if let Self::UpstreamNnue(evaluator) = self {
            evaluator.move_played(child, mv, undo, ply);
        }
    }

    /// Used for forwarding search-only null moves to the incremental
    /// upstream evaluator.
    ///
    /// Backends other than upstream NNUE ignore the notification.
    ///
    /// # Arguments
    ///
    /// * `ply` - search ply at which the null move was made
    fn null_move_played(&mut self, ply: usize) {
        if let Self::UpstreamNnue(evaluator) = self {
            evaluator.null_move_played(ply);
        }
    }

    /// Used for evaluating with root-relative incremental inference when
    /// upstream NNUE is active.
    ///
    /// Every other backend falls back to the plain
    /// [`SearchEvaluator::evaluate`] path.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `ply` - search ply of the position relative to the root
    ///
    /// # Returns
    ///
    /// Backend evaluation score as an `i32`.
    fn evaluate_at(&mut self, position: &Position, ply: usize) -> i32 {
        match self {
            Self::UpstreamNnue(evaluator) => evaluator.evaluate_at(position, ply),
            _ => SearchEvaluator::evaluate(self, position),
        }
    }

    /// Used for supplying classical quiet ordering priors only for the
    /// classical backend.
    ///
    /// Neural backends return zero so move ordering does not consult them.
    ///
    /// # Arguments
    ///
    /// * `position` - position containing the quiet move
    /// * `mv` - quiet move to rank
    ///
    /// # Returns
    ///
    /// Classical ordering prior, or `0` for every non-classical backend.
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        match self {
            Self::Classical => SearchEvaluator::quiet_move_order_prior(&Classical, position, mv),
            Self::CompactNnue(_)
            | Self::UpstreamNnue(_)
            | Self::Cnn(_)
            | Self::Bt4(_)
            | Self::Jrbt(_) => 0,
        }
    }
}

/// Used for completing worker construction without mutable build flavors.
///
/// The public engine has one immutable released search configuration.
///
/// # Arguments
///
/// * `workers` - freshly constructed search workers
fn prepare_workers(workers: &mut [AlphaBeta<JanusEvaluator>]) {
    let _ = workers;
}

/// Used for evaluating a position with the immutable released classical evaluator.
///
/// # Arguments
///
/// * `position` - position to evaluate
///
/// # Returns
///
/// Calibrated side-to-move classical score.
fn classical_evaluate_position(position: &Position) -> i32 {
    Classical::evaluate_position(position)
}

impl Evaluator for JanusEvaluator {
    /// Used for adapting the search evaluator to the generic value-evaluator
    /// contract.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Score from [`SearchEvaluator::evaluate`].
    fn evaluate(&mut self, position: &Position) -> i32 {
        SearchEvaluator::evaluate(self, position)
    }

    /// Used for resetting a temporal neural evaluator at the start of one
    /// MCTS path.
    ///
    /// Only CNN and BT4 maintain per-path state; other backends ignore the
    /// call.
    ///
    /// # Arguments
    ///
    /// * `root` - root position of the new selection path
    fn begin_mcts_path(&mut self, root: &Position) {
        match self {
            Self::Cnn(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::begin_mcts_path(&mut *network, root);
            }
            Self::Bt4(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::begin_mcts_path(&mut *network, root);
            }
            Self::Jrbt(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::begin_mcts_path(&mut *network, root);
            }
            Self::Classical | Self::CompactNnue(_) | Self::UpstreamNnue(_) => {}
        }
    }

    /// Used for appending one selected descendant to a temporal neural
    /// evaluator.
    ///
    /// Only CNN and BT4 maintain per-path state; other backends ignore the
    /// call.
    ///
    /// # Arguments
    ///
    /// * `position` - descendant position appended to the current path
    fn mcts_path_position(&mut self, position: &Position) {
        match self {
            Self::Cnn(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::mcts_path_position(&mut *network, position);
            }
            Self::Bt4(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::mcts_path_position(&mut *network, position);
            }
            Self::Jrbt(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                Evaluator::mcts_path_position(&mut *network, position);
            }
            Self::Classical | Self::CompactNnue(_) | Self::UpstreamNnue(_) => {}
        }
    }

    /// Used for retrieving neural policy/value output or a value-only
    /// fallback for other backends.
    ///
    /// CNN and BT4 run full policy/value inference; every other backend
    /// wraps its scalar evaluation through
    /// [`PolicyValue::from_centipawns`].
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `legal_moves` - legal moves receiving policy mass
    ///
    /// # Returns
    ///
    /// Policy/value pair for the position.
    fn evaluate_policy_value(&mut self, position: &Position, legal_moves: &[Move]) -> PolicyValue {
        match self {
            Self::Cnn(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                network.evaluate_policy_value(position, legal_moves)
            }
            Self::Bt4(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                network.evaluate_policy_value(position, legal_moves)
            }
            Self::Jrbt(network) => {
                let mut network = network
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                network.evaluate_policy_value(position, legal_moves)
            }
            _ => PolicyValue::from_centipawns(SearchEvaluator::evaluate(self, position)),
        }
    }

    /// Used for skipping recursive network quiescence for the full BT4
    /// evaluator.
    ///
    /// # Returns
    ///
    /// `false` only for the BT4 backend.
    fn allows_mcts_quiescence(&self) -> bool {
        !matches!(self, Self::Bt4(_) | Self::Jrbt(_))
    }
}

/// Messages serialized through the coordinator's multi-producer channel.
///
/// The input thread, search workers, and perft workers all send these events
/// into one [`mpsc`] channel; the coordinator loop in [`run`] is the only
/// consumer.
enum CoordinatorEvent {
    /// Used for delivering a successfully parsed command from the input
    /// thread.
    Command(Command),
    /// Used for reporting a recoverable syntax failure for one bounded input
    /// line.
    ParseError(String),
    /// Used for signalling end of input with an optional fatal reader error.
    InputClosed(Option<io::Error>),
    /// Used for delivering one completed alpha-beta principal-variation
    /// line.
    AlphaBetaInfo {
        /// Used for matching the line to the search that produced it.
        search_id: u64,
        /// Used for reporting the one-based rank of this line in its
        /// completed `MultiPV` batch.
        multipv: usize,
        /// Used for carrying completed-depth search statistics and the
        /// principal variation.
        info: SearchInfo,
    },
    /// Used for delivering a complete MCTS progress or final snapshot.
    #[cfg(feature = "mcts")]
    MctsInfo {
        /// Used for matching the snapshot to the search that produced it.
        search_id: u64,
        /// Used for carrying an internally consistent MCTS tree summary.
        result: MctsResult,
        /// Used for reporting wall-clock time elapsed since the job began.
        elapsed: Duration,
    },
    /// Used for returning a background search job and its owned searcher on
    /// success.
    SearchFinished {
        /// Used for matching the completion to the identifier assigned when
        /// the job started.
        search_id: u64,
        /// Used for carrying the algorithm-specific result and reusable
        /// engine state.
        search: FinishedSearch,
    },
    /// Used for reporting a panic caught at the background worker boundary.
    SearchPanicked {
        /// Used for matching the panic to the identifier assigned when the
        /// job started.
        search_id: u64,
        /// Used for naming the algorithm whose state must be rebuilt after
        /// the panic.
        mode: SearchMode,
    },
    /// Used for returning one background perft job on success.
    PerftFinished {
        /// Used for matching the completion to the identifier assigned when
        /// the job started.
        job_id: u64,
        /// Used for carrying complete counters or a controlled
        /// cancellation/overflow result.
        result: Result<FinishedPerft, PerftError>,
        /// Used for reporting monotonic wall time observed entirely inside
        /// the worker.
        elapsed: Duration,
    },
    /// Used for reporting a panic caught at the background perft worker
    /// boundary.
    PerftPanicked {
        /// Used for matching the panic to the identifier assigned when the
        /// job started.
        job_id: u64,
    },
}

/// Algorithm-specific state returned by a completed background job.
///
/// Ownership of the searcher travels with the result so the coordinator can
/// restore it into [`EngineState`] before emitting `bestmove`.
enum FinishedSearch {
    /// Used for returning the reusable alpha-beta worker pool and ranked
    /// search result.
    AlphaBeta {
        /// Used for restoring Lazy-SMP workers retaining local heuristics
        /// and one shared table.
        workers: Vec<AlphaBeta<JanusEvaluator>>,
        /// Used for carrying ranked lines or a validated search-start
        /// failure.
        result: Result<Vec<SearchResult>, SearchError>,
        /// Used for reporting the first helper that the operating system
        /// refused to start, if any.
        helper_spawn_failure: Option<HelperSpawnFailure>,
        /// Used for reporting helpers that started but could not admit their
        /// per-search workspace.
        helper_workspace_refusals: usize,
    },
    /// Used for returning the reusable MCTS searcher and its final tree
    /// summary.
    #[cfg(feature = "mcts")]
    Mcts {
        /// Used for restoring the searcher retaining the selected evaluator
        /// and configuration.
        searcher: Mcts<JanusEvaluator>,
        /// Used for carrying the final complete search snapshot.
        result: MctsResult,
    },
}

/// Complete output payload returned by one developer perft job.
///
/// The variant mirrors the [`PerftFormat`] requested when the job started.
enum FinishedPerft {
    /// Used for carrying one detailed aggregate.
    Detail(PerftStats),
    /// Used for carrying detailed counters split by legal root move.
    Divide(PerftDivideResult),
    /// Used for carrying node-only counts split by legal root move.
    Stockfish(PerftNodeResult),
}

/// Coordinator-owned metadata for one background developer perft job.
///
/// Created by [`start_perft_with_spawner`] and consumed by [`finish_perft`] or
/// [`finish_panicked_perft`] when the matching completion event arrives.
struct ActivePerft {
    /// Used for discarding stale job events; monotonic and nonzero.
    id: u64,
    /// Used for formatting FEN, status, and move-text output; immutable
    /// root.
    root: Position,
    /// Used for reporting the requested enumeration depth.
    depth: u32,
    /// Used for checking the requested output family at completion.
    format: PerftFormat,
    /// Used for deciding whether castling rows use Chess960 rook-source
    /// notation.
    uci_chess960: bool,
    /// Used for cancelling the worker; token shared with it.
    stop: Arc<AtomicBool>,
    /// Used for reclaiming the worker on every terminal path.
    handle: JoinHandle<()>,
}

/// The single background owner admitted by the responsive coordinator.
///
/// At most one job runs at a time; every other state-changing command
/// received meanwhile is queued behind a stop request.
enum ActiveJob {
    /// Used for tracking a normal alpha-beta or MCTS search.
    Search(Box<ActiveSearch>),
    /// Used for tracking a checked CPU-only developer perft enumeration.
    Perft(Box<ActivePerft>),
}

impl ActiveJob {
    /// Used for retrieving the identifier shared with completion events.
    ///
    /// # Returns
    ///
    /// Monotonic nonzero job identifier.
    fn id(&self) -> u64 {
        match self {
            Self::Search(search) => search.id,
            Self::Perft(perft) => perft.id,
        }
    }

    /// Used for requesting bounded cancellation and authorizing any deferred
    /// search result.
    ///
    /// A search additionally records that release was requested so a
    /// naturally completed infinite result may be emitted; perft only raises
    /// its stop token.
    fn request_stop(&mut self) {
        match self {
            Self::Search(search) => {
                search.release_requested = true;
                search.stop.store(true, Ordering::Relaxed);
            }
            Self::Perft(perft) => perft.stop.store(true, Ordering::Relaxed),
        }
    }

    /// Used for retrieving mutable search metadata when this job is not
    /// perft.
    ///
    /// # Returns
    ///
    /// Mutable [`ActiveSearch`] reference, or [`None`] for a perft job.
    fn search_mut(&mut self) -> Option<&mut ActiveSearch> {
        match self {
            Self::Search(search) => Some(search),
            Self::Perft(_) => None,
        }
    }

    /// Used for retrieving shared search metadata when this job is not
    /// perft.
    ///
    /// # Returns
    ///
    /// Shared [`ActiveSearch`] reference, or [`None`] for a perft job.
    fn search_ref(&self) -> Option<&ActiveSearch> {
        match self {
            Self::Search(search) => Some(search),
            Self::Perft(_) => None,
        }
    }
}

/// Coordinator-owned metadata for the single active search job.
///
/// Its booleans represent independent output capabilities and lifecycle facts,
/// rather than mutually exclusive states that would form one useful enum.
#[allow(clippy::struct_excessive_bools)]
struct ActiveSearch {
    /// Used for discarding stale job events; monotonic and nonzero.
    id: u64,
    /// Used for recording the algorithm executing in the job.
    mode: SearchMode,
    /// Used for formatting PV and best-move output; immutable root.
    root: Position,
    /// Used for cancelling the job; token shared with every worker.
    stop: Arc<AtomicBool>,
    /// Used for reclaiming background resources at completion.
    handle: JoinHandle<()>,
    /// Used for deciding whether UCI info lines include approximate WDL
    /// values.
    show_wdl: bool,
    /// Used for deciding whether castling moves use Chess960 rook-source
    /// notation.
    uci_chess960: bool,
    /// Used for holding back a naturally completed result until an explicit
    /// stop.
    defer_until_stop: bool,
    /// Used for recording that a command authorized release of a deferred
    /// result.
    release_requested: bool,
    /// Used for retaining the live ponder transition until the first
    /// matching `ponderhit`.
    ponder: Option<PonderState>,
    /// Used for retaining a naturally completed infinite or ponder result
    /// until release.
    completion: Option<FinishedSearch>,
}

/// Coordinator-owned transition state for one accepted ponder search.
///
/// Built by [`ponder_state`] and consumed in place by
/// [`transition_ponderhit`] when the matching `ponderhit` arrives.
struct PonderState {
    /// Used for publishing an optional clock budget without replacing the
    /// search job.
    deadline: Option<PonderDeadline>,
    /// Used for deciding whether ordinary finite-search completion is legal
    /// after `ponderhit`.
    bounded_after_hit: bool,
}

/// One pending activation of the engine's shared live search clock.
///
/// The budgets are measured from the moment `ponderhit` activates the clock,
/// not from search start.
struct PonderDeadline {
    /// Used for activating the clock shared by every worker participating
    /// in the ponder search.
    clock: Arc<SearchClock>,
    /// Used for publishing the optional alpha-beta soft target measured
    /// from `ponderhit`.
    soft_time: Option<Duration>,
    /// Used for publishing the binding deadline measured from `ponderhit`.
    hard_time: Duration,
}

/// Mutable engine configuration and reusable search state while idle.
///
/// The boolean fields are independent UCI check options and protocol flags,
/// not competing lifecycle states.
#[allow(clippy::struct_excessive_bools)]
struct EngineState {
    /// Used for rooting the next search; established by the latest valid
    /// `position` command.
    position: Position,
    /// Used for repetition detection; keys preceding the current position.
    previous_keys: Vec<u64>,
    /// Used for temporal evaluator history; real game positions from the
    /// UCI source position through the current root.
    position_history: Vec<Position>,
    /// Used for constructing newly rebuilt searchers.
    active_evaluator: JanusEvaluator,
    /// Used for caching the last successfully loaded neural evaluator, if
    /// any.
    loaded_evaluator: Option<JanusEvaluator>,
    /// Used for remembering the last explicit `Eval` selection.
    ///
    /// A later `Eval File` load must not silently replace an evaluator the
    /// operator asked for by name; without this, `Eval=Classical` followed by
    /// `Eval File` activates the network instead.
    requested_evaluator: Option<EvaluatorSelection>,
    /// Used for transactional reloads; source path of the cached evaluator.
    loaded_evaluator_path: Option<String>,
    /// Used for selecting the BT4 execution backend on the next model load
    /// or reload.
    bt4_backend: Bt4Backend,
    /// Used for selecting the zero-based vendor device ordinal of BT4 GPU
    /// backends.
    bt4_device: usize,
    /// Used for de-duplicating diagnostics; last active BT4 backend state
    /// emitted through the UCI stream.
    reported_bt4_status: Option<Bt4BackendStatus>,
    /// Used for alpha-beta search; workers with a local table at one thread
    /// or one shared table.
    alpha_beta: Vec<AlphaBeta<JanusEvaluator>>,
    /// Used for MCTS search; absent only while owned by a background job.
    #[cfg(feature = "mcts")]
    mcts: Option<Mcts<JanusEvaluator>>,
    /// Used for capping arena nodes allocated by one MCTS invocation.
    #[cfg(feature = "mcts")]
    mcts_tree_nodes: usize,
    /// Used for selecting the algorithm of the next search.
    search_mode: SearchMode,
    /// Used for recording the requested alpha-beta worker count before
    /// evaluator-specific caps.
    search_threads: usize,
    /// Used for sizing the aggregate transposition-table budget in
    /// mebibytes.
    hash_mb: usize,
    /// Used for recording the requested number of ranked alpha-beta lines.
    multi_pv: usize,
    /// Used for deciding whether score output includes approximate WDL
    /// triplets.
    show_wdl: bool,
    /// Used for deciding whether UCI castling notation follows the Chess960
    /// convention.
    uci_chess960: bool,
    /// Used for recording GUI-advertised intent to use pondering;
    /// `go ponder` remains authoritative.
    ponder_enabled: bool,
    /// Used for reserving milliseconds from clock-derived hard and soft
    /// budgets.
    move_overhead_ms: u64,
    /// Used for extending the soft budget when the score is falling.
    ///
    /// `0` is the released no-op. Applied to every worker, and re-applied
    /// whenever the pool is rebuilt, so a `Hash` or `Threads` change cannot
    /// silently drop it.
    falling_eval_percent: i32,
    /// Used for the root fail-high reduction; `1` is the released value.
    root_fail_high_reduction: i32,
    /// Used as the fallback divisor when `go` carries no `movestogo`.
    ///
    /// Alternative configuration retained for controlled evaluation.
    clock_moves_to_go: u32,
    /// Used for deciding whether optional protocol diagnostics are enabled.
    debug: bool,
    /// Used for sharing the scanned Syzygy registry with alpha-beta
    /// workers; `None` while `SyzygyPath` is unset.
    syzygy: Option<Arc<Tablebases>>,
    /// Used for gating boundary-cardinality probes by remaining depth.
    syzygy_probe_depth: u8,
    /// Used for capping the probed piece count.
    syzygy_probe_limit: u8,
    /// Used for honoring the fifty-move rule in tablebase score mapping.
    syzygy_rule50: bool,
}

impl EngineState {
    /// Used for constructing the default classical, single-thread engine
    /// state.
    ///
    /// The initial configuration mirrors the advertised UCI option defaults:
    /// classical evaluation, alpha-beta search, one thread, and the default
    /// hash and overhead budgets.
    ///
    /// # Returns
    ///
    /// Fresh engine state rooted at the standard start position.
    fn new() -> Self {
        let evaluator = JanusEvaluator::Classical;
        let position = Position::start();
        Self {
            position: position.clone(),
            previous_keys: Vec::new(),
            position_history: vec![position],
            active_evaluator: evaluator.clone(),
            loaded_evaluator: None,
            requested_evaluator: None,
            loaded_evaluator_path: None,
            bt4_backend: Bt4Backend::Auto,
            bt4_device: 0,
            reported_bt4_status: None,
            alpha_beta: alpha_beta_workers(&evaluator, 1, DEFAULT_HASH_MB),
            #[cfg(feature = "mcts")]
            mcts: Some(Mcts::new(evaluator)),
            #[cfg(feature = "mcts")]
            mcts_tree_nodes: DEFAULT_MCTS_TREE_NODE_LIMIT,
            search_mode: SearchMode::AlphaBeta,
            search_threads: 1,
            hash_mb: DEFAULT_HASH_MB,
            multi_pv: 1,
            show_wdl: true,
            uci_chess960: false,
            ponder_enabled: false,
            move_overhead_ms: DEFAULT_MOVE_OVERHEAD_MS,
            clock_moves_to_go: janus_engine::CLOCK_MOVES_TO_GO,
            falling_eval_percent: 0,
            root_fail_high_reduction: 1,
            debug: false,
            syzygy: None,
            syzygy_probe_depth: DEFAULT_SYZYGY_PROBE_DEPTH,
            syzygy_probe_limit: MAX_SYZYGY_PROBE_LIMIT,
            syzygy_rule50: true,
        }
    }

    /// Used for recreating alpha-beta workers after evaluator, thread, or
    /// hash changes.
    ///
    /// CNN and BT4 evaluators cap the pool at one worker; every other
    /// backend uses the configured thread count.
    fn rebuild_alpha_beta(&mut self) {
        let threads = if matches!(
            self.active_evaluator,
            JanusEvaluator::Cnn(_) | JanusEvaluator::Bt4(_) | JanusEvaluator::Jrbt(_)
        ) {
            1
        } else {
            self.search_threads
        };
        self.alpha_beta = alpha_beta_workers(&self.active_evaluator, threads, self.hash_mb);
        self.apply_falling_eval_to_workers();
        self.apply_syzygy_to_workers();
    }

    /// Used for pushing the falling-eval extension onto every live worker.
    ///
    /// Mirrors [`Self::apply_syzygy_to_workers`]. Must be called both when the
    /// option arrives and whenever the pool is rebuilt, or a `Hash` or
    /// `Threads` change would silently drop the setting.
    fn apply_falling_eval_to_workers(&mut self) {
        for worker in &mut self.alpha_beta {
            worker.set_falling_eval_percent(self.falling_eval_percent);
            worker.set_root_fail_high_reduction(self.root_fail_high_reduction);
        }
    }

    /// Used for pushing the current Syzygy configuration onto every
    /// alpha-beta worker.
    ///
    /// Runs only between searches (the coordinator defers `setoption`
    /// while a job is active), so probers are never replaced mid-search.
    /// Each worker builds its own prober, keeping block caching
    /// worker-local and deterministic.
    fn apply_syzygy_to_workers(&mut self) {
        let config = self.syzygy.as_ref().map(|tables| SyzygyConfig {
            tables: Arc::clone(tables),
            probe_limit: self.syzygy_probe_limit,
            probe_depth: self.syzygy_probe_depth,
            rule50: self.syzygy_rule50,
        });
        for worker in &mut self.alpha_beta {
            worker.set_syzygy(config.as_ref());
        }
    }

    /// Used for recreating MCTS around the active evaluator.
    ///
    /// Any previously owned searcher is replaced.
    #[cfg(feature = "mcts")]
    fn rebuild_mcts(&mut self) {
        self.mcts = Some(Mcts::new(self.active_evaluator.clone()));
    }
}

/// Heap-owned top-level worker body accepted by the fallible spawn boundary.
type WorkerTask = Box<dyn FnOnce() + Send + 'static>;

/// Function used for starting one named top-level UCI worker.
///
/// Keeping the boundary as a function pointer lets unit tests inject a
/// deterministic OS-refusal result without exhausting host resources.
type WorkerSpawner = fn(&str, WorkerTask) -> io::Result<JoinHandle<()>>;

/// Used for constructing the one bounded event channel shared by input and
/// background jobs.
///
/// # Returns
///
/// Synchronous sender and matching receiver with exactly
/// [`COORDINATOR_QUEUE_CAPACITY`] waiting slots.
fn coordinator_channel() -> (Sender<CoordinatorEvent>, Receiver<CoordinatorEvent>) {
    mpsc::sync_channel(COORDINATOR_QUEUE_CAPACITY)
}

/// Used for starting a named top-level worker without panicking on OS refusal.
///
/// # Arguments
///
/// * `name` - diagnostic thread name visible to host tooling
/// * `task` - owned worker body
///
/// # Returns
///
/// Join handle for a started worker.
///
/// # Errors
///
/// Returns the operating-system spawn error without running `task`.
fn spawn_named_worker(name: &str, task: WorkerTask) -> io::Result<JoinHandle<()>> {
    thread::Builder::new().name(name.to_owned()).spawn(task)
}

/// Used for running the UCI coordinator over standard input and output.
///
/// Exits the process with status `2` after printing a diagnostic when the
/// coordinator returns an input/output failure.
fn main() {
    if let Err(error) = run(
        BufReader::new(io::stdin()),
        BufWriter::new(io::stdout().lock()),
    ) {
        eprintln!("janus: UCI input/output failed: {error}");
        std::process::exit(2);
    }
}

/// Used for coordinating input, search events, output ordering, and
/// engine-state ownership.
///
/// The input reader may block independently, while this loop stays responsive
/// to `isready`, `stop`, and `quit`. State-changing commands received during a
/// search are queued after requesting cancellation of the active job.
///
/// # Arguments
///
/// * `input` - buffered command source handed to the reader thread
/// * `output` - sink receiving every UCI response line
///
/// # Returns
///
/// `Ok(())` after `quit` or clean end of input.
///
/// # Errors
///
/// Returns an [`io::Error`] when writing or flushing `output` fails, when
/// the coordinator channel closes unexpectedly, or when the input reader
/// ended with a fatal error.
///
/// # Panics
///
/// Panics only if internal job bookkeeping is violated; every `expect`
/// follows a check that the matching active job exists.
#[allow(clippy::too_many_lines)]
fn run<R, W>(input: R, mut output: W) -> io::Result<()>
where
    R: BufRead + Send + 'static,
    W: Write,
{
    run_with_input_spawner(input, &mut output, spawn_named_worker)
}

/// Used for running the coordinator with an injectable input-thread boundary.
///
/// Production calls this through [`run`] with [`spawn_named_worker`]. Tests
/// pass a deterministic refusal function to verify that input-thread
/// exhaustion becomes an ordinary I/O error rather than a panic.
///
/// # Arguments
///
/// * `input` - buffered command source handed to the reader thread
/// * `output` - sink receiving every UCI response line
/// * `input_spawner` - fallible boundary used only for the input reader
///
/// # Returns
///
/// `Ok(())` after `quit` or clean end of input.
///
/// # Errors
///
/// Returns an [`io::Error`] for input-thread refusal, output failures, a
/// closed coordinator channel, or a fatal input read error.
#[allow(clippy::too_many_lines)]
fn run_with_input_spawner<R, W>(
    input: R,
    mut output: W,
    input_spawner: WorkerSpawner,
) -> io::Result<()>
where
    R: BufRead + Send + 'static,
    W: Write,
{
    let (sender, receiver) = coordinator_channel();
    let input_sender = sender.clone();
    let _input_handle = input_spawner(
        "janus-uci-input",
        Box::new(move || read_commands(input, &input_sender)),
    )?;

    let mut state = EngineState::new();
    write_alpha_beta_admission(&state.alpha_beta, state.hash_mb, &mut output)?;
    let mut active: Option<ActiveJob> = None;
    let mut pending = VecDeque::new();
    let mut next_search_id = 1_u64;
    let mut quitting = false;
    let mut input_error = None;

    loop {
        let event = receiver.recv().map_err(|_| {
            io::Error::new(io::ErrorKind::BrokenPipe, "UCI coordinator channel closed")
        })?;
        match event {
            CoordinatorEvent::Command(Command::Quit) => {
                quitting = true;
                pending.clear();
                if let Some(job) = &mut active {
                    job.request_stop();
                } else {
                    break;
                }
                if release_deferred_completion(&mut active, &mut state, &mut output)? {
                    break;
                }
            }
            CoordinatorEvent::Command(Command::Stop) => {
                if let Some(job) = &mut active {
                    if pending_job_needs_stop(&pending) {
                        let _ = enqueue_priority_pending(&mut pending, Command::Stop, &mut output)?;
                    }
                    job.request_stop();
                }
                if release_deferred_completion(&mut active, &mut state, &mut output)? {
                    start_pending_commands(
                        &mut pending,
                        &mut active,
                        &mut state,
                        &sender,
                        &mut next_search_id,
                        &mut output,
                    )?;
                }
            }
            CoordinatorEvent::Command(Command::IsReady) => {
                if active.is_some() && !pending.is_empty() {
                    let queued =
                        enqueue_priority_pending(&mut pending, Command::IsReady, &mut output)?;
                    if queued {
                        active.as_mut().expect("checked active job").request_stop();
                    } else {
                        writeln!(output, "readyok")?;
                    }
                    if release_deferred_completion(&mut active, &mut state, &mut output)? {
                        start_pending_commands(
                            &mut pending,
                            &mut active,
                            &mut state,
                            &sender,
                            &mut next_search_id,
                            &mut output,
                        )?;
                    }
                } else {
                    writeln!(output, "readyok")?;
                }
            }
            CoordinatorEvent::Command(Command::Uci) => write_uci_identity(&mut output)?,
            CoordinatorEvent::Command(Command::Debug(enabled)) => {
                state.debug = enabled;
                writeln!(
                    output,
                    "info string debug {}",
                    if state.debug { "on" } else { "off" }
                )?;
            }
            CoordinatorEvent::Command(Command::PonderHit) => {
                let (recognized, activation_failed) = active
                    .as_mut()
                    .and_then(ActiveJob::search_mut)
                    .map_or((false, false), transition_ponderhit);
                if activation_failed {
                    writeln!(
                        output,
                        "info string ponderhit clock activation failed; search stopped safely"
                    )?;
                }
                if state.debug {
                    writeln!(
                        output,
                        "info string ponderhit {}",
                        if recognized {
                            "acknowledged"
                        } else {
                            "ignored"
                        }
                    )?;
                }
                if recognized && release_deferred_completion(&mut active, &mut state, &mut output)?
                {
                    start_pending_commands(
                        &mut pending,
                        &mut active,
                        &mut state,
                        &sender,
                        &mut next_search_id,
                        &mut output,
                    )?;
                }
            }
            CoordinatorEvent::Command(Command::Unknown(name)) => {
                writeln!(output, "info string unknown command: {name}")?;
            }
            CoordinatorEvent::Command(command) => {
                if let Some(job) = &mut active {
                    if enqueue_pending(&mut pending, command, &mut output)? {
                        job.request_stop();
                    }
                    if release_deferred_completion(&mut active, &mut state, &mut output)? {
                        start_pending_commands(
                            &mut pending,
                            &mut active,
                            &mut state,
                            &sender,
                            &mut next_search_id,
                            &mut output,
                        )?;
                    }
                } else {
                    active = handle_idle_command(
                        command,
                        &mut state,
                        &sender,
                        &mut next_search_id,
                        &mut output,
                    )?;
                }
            }
            CoordinatorEvent::ParseError(error) => {
                writeln!(output, "info string invalid command: {error}")?;
            }
            CoordinatorEvent::InputClosed(error) => {
                input_error = error;
                quitting = true;
                pending.clear();
                if let Some(job) = &mut active {
                    job.request_stop();
                } else {
                    break;
                }
                if release_deferred_completion(&mut active, &mut state, &mut output)? {
                    break;
                }
            }
            CoordinatorEvent::AlphaBetaInfo {
                search_id,
                multipv,
                info,
            } => {
                if let Some(search) = active
                    .as_ref()
                    .filter(|job| job.id() == search_id)
                    .and_then(ActiveJob::search_ref)
                {
                    write_search_info(
                        &mut output,
                        &info,
                        multipv,
                        &search.root,
                        search.uci_chess960,
                        search.show_wdl,
                    )?;
                }
            }
            #[cfg(feature = "mcts")]
            CoordinatorEvent::MctsInfo {
                search_id,
                result,
                elapsed,
            } => {
                if let Some(search) = active
                    .as_ref()
                    .filter(|job| job.id() == search_id)
                    .and_then(ActiveJob::search_ref)
                {
                    write_mcts_info(
                        &mut output,
                        &result,
                        elapsed,
                        &search.root,
                        search.uci_chess960,
                        search.show_wdl,
                    )?;
                }
            }
            CoordinatorEvent::SearchFinished { search_id, search } => {
                if active.as_ref().is_some_and(|job| job.id() == search_id) {
                    let defer = active.as_ref().is_some_and(|job| {
                        job.search_ref().is_some_and(|search| {
                            search.defer_until_stop && !search.release_requested
                        })
                    });
                    if defer {
                        active
                            .as_mut()
                            .and_then(ActiveJob::search_mut)
                            .expect("search completion matched an active search")
                            .completion = Some(search);
                    } else {
                        let ActiveJob::Search(active_search) =
                            active.take().expect("matched active job")
                        else {
                            unreachable!("search completion matched perft");
                        };
                        finish_search(&mut state, *active_search, search, &mut output)?;
                        if quitting {
                            break;
                        }
                        start_pending_commands(
                            &mut pending,
                            &mut active,
                            &mut state,
                            &sender,
                            &mut next_search_id,
                            &mut output,
                        )?;
                    }
                }
            }
            CoordinatorEvent::SearchPanicked { search_id, mode } => {
                if active.as_ref().is_some_and(|job| job.id() == search_id) {
                    let ActiveJob::Search(active_search) =
                        active.take().expect("matched active job")
                    else {
                        unreachable!("search panic matched perft");
                    };
                    finish_panicked_search(&mut state, *active_search, mode, &mut output)?;
                    if quitting {
                        break;
                    }
                    start_pending_commands(
                        &mut pending,
                        &mut active,
                        &mut state,
                        &sender,
                        &mut next_search_id,
                        &mut output,
                    )?;
                }
            }
            CoordinatorEvent::PerftFinished {
                job_id,
                result,
                elapsed,
            } => {
                if active.as_ref().is_some_and(|job| job.id() == job_id) {
                    let ActiveJob::Perft(perft) = active.take().expect("matched active job") else {
                        unreachable!("perft completion matched search");
                    };
                    finish_perft(*perft, result, elapsed, &mut output)?;
                    if quitting {
                        break;
                    }
                    start_pending_commands(
                        &mut pending,
                        &mut active,
                        &mut state,
                        &sender,
                        &mut next_search_id,
                        &mut output,
                    )?;
                }
            }
            CoordinatorEvent::PerftPanicked { job_id } => {
                if active.as_ref().is_some_and(|job| job.id() == job_id) {
                    let ActiveJob::Perft(perft) = active.take().expect("matched active job") else {
                        unreachable!("perft panic matched search");
                    };
                    finish_panicked_perft(*perft, &mut output)?;
                    if quitting {
                        break;
                    }
                    start_pending_commands(
                        &mut pending,
                        &mut active,
                        &mut state,
                        &sender,
                        &mut next_search_id,
                        &mut output,
                    )?;
                }
            }
        }
        output.flush()?;
    }

    if let Some(error) = input_error {
        Err(error)
    } else {
        Ok(())
    }
}

/// Used for releasing an already completed infinite result after stop or
/// shutdown.
///
/// When the active job is a search holding a deferred completion, the job is
/// consumed and finalized through [`finish_search`].
///
/// # Arguments
///
/// * `active` - slot owning the single background job
/// * `state` - engine state receiving the reclaimed searcher
/// * `output` - sink receiving final info and `bestmove` lines
///
/// # Returns
///
/// `true` when an active search was consumed and finalized.
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
///
/// # Panics
///
/// Panics only if internal bookkeeping is violated; the checks preceding
/// each `expect` guarantee the job and completion exist.
fn release_deferred_completion<W: Write>(
    active: &mut Option<ActiveJob>,
    state: &mut EngineState,
    output: &mut W,
) -> io::Result<bool> {
    if !active
        .as_ref()
        .and_then(ActiveJob::search_ref)
        .is_some_and(|search| search.completion.is_some())
    {
        return Ok(false);
    }
    let ActiveJob::Search(mut active_search) = active.take().expect("checked active job") else {
        unreachable!("perft cannot retain a deferred search completion");
    };
    let completion = active_search
        .completion
        .take()
        .expect("checked deferred completion");
    finish_search(state, *active_search, completion, output)?;
    Ok(true)
}

/// Used for applying the first `ponderhit` to an active ponder search in
/// place.
///
/// The returned tuple reports whether the command matched and whether the
/// shared clock unexpectedly rejected its one-shot publication. A failed
/// publication cancels the job so a clock-only ponder cannot run forever.
///
/// # Arguments
///
/// * `search` - active search possibly holding a ponder transition
///
/// # Returns
///
/// `(recognized, activation_failed)` pair as described above.
fn transition_ponderhit(search: &mut ActiveSearch) -> (bool, bool) {
    let Some(ponder) = search.ponder.take() else {
        return (false, false);
    };
    let activation_failed = ponder.deadline.is_some_and(|deadline| {
        !deadline
            .clock
            .activate(deadline.soft_time, deadline.hard_time)
    });
    if ponder.bounded_after_hit || activation_failed {
        search.release_requested = true;
    }
    if activation_failed {
        search.stop.store(true, Ordering::Relaxed);
    }
    (true, activation_failed)
}

/// Used for reading, bounding, and parsing input lines on the dedicated
/// reader thread.
///
/// Every parsed command, parse error, and the final end-of-input condition
/// is forwarded as a [`CoordinatorEvent`]; the thread exits after `quit`, a
/// closed channel, or end of input.
///
/// # Arguments
///
/// * `input` - buffered command source
/// * `sender` - coordinator channel receiving reader events
fn read_commands<R: BufRead>(mut input: R, sender: &Sender<CoordinatorEvent>) {
    let mut line = String::new();
    loop {
        match protocol::read_line_bounded(&mut input, &mut line) {
            Ok(true) => match protocol::parse(&line) {
                Ok(Some(command)) => {
                    let quitting = matches!(command, Command::Quit);
                    if sender.send(CoordinatorEvent::Command(command)).is_err() || quitting {
                        return;
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    if sender
                        .send(CoordinatorEvent::ParseError(error.to_string()))
                        .is_err()
                    {
                        return;
                    }
                }
            },
            Ok(false) => {
                let _ = sender.send(CoordinatorEvent::InputClosed(None));
                return;
            }
            Err(error) => {
                let _ = sender.send(CoordinatorEvent::InputClosed(Some(error)));
                return;
            }
        }
    }
}

/// Used for writing engine identity, supported options, and the terminating
/// `uciok`.
///
/// # Arguments
///
/// * `output` - sink receiving the identity block
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn write_uci_identity<W: Write>(output: &mut W) -> io::Result<()> {
    writeln!(output, "id name Janus {ENGINE_VERSION}")?;
    writeln!(output, "id author Lennart A. Conrad")?;
    #[cfg(feature = "mcts")]
    writeln!(
        output,
        "option name Search type combo default AlphaBeta var AlphaBeta var MCTS"
    )?;
    #[cfg(not(feature = "mcts"))]
    writeln!(
        output,
        "option name Search type combo default AlphaBeta var AlphaBeta"
    )?;
    writeln!(
        output,
        "option name Eval type combo default Classical var Classical var CompactNNUE var UpstreamNNUE var CNN var BT4 var JRBT"
    )?;
    writeln!(output, "option name Eval File type string default <empty>")?;
    writeln!(
        output,
        "option name BT4 Backend type combo default Auto var Auto var CPU var Nvidia var AMD var Intel"
    )?;
    writeln!(
        output,
        "option name BT4 Device type spin default 0 min 0 max {MAX_BT4_DEVICE_INDEX}"
    )?;
    writeln!(
        output,
        "option name Threads type spin default 1 min 1 max {MAX_SEARCH_THREADS}"
    )?;
    writeln!(
        output,
        "option name Hash type spin default {DEFAULT_HASH_MB} min 1 max {MAX_HASH_MB}"
    )?;
    writeln!(output, "option name Clear Hash type button")?;
    writeln!(
        output,
        "option name Clock Moves To Go type spin default {} min 1 max {MAX_CLOCK_MOVES_TO_GO}",
        janus_engine::CLOCK_MOVES_TO_GO
    )?;
    writeln!(
        output,
        "option name Falling Eval Percent type spin default 0 min 0 max {MAX_FALLING_EVAL_PERCENT}"
    )?;
    writeln!(
        output,
        "option name Root Fail High Reduction type spin default 1 min 0 max {MAX_ROOT_FAIL_HIGH_REDUCTION}"
    )?;
    #[cfg(feature = "mcts")]
    writeln!(
        output,
        "option name MCTS Tree Nodes type spin default {DEFAULT_MCTS_TREE_NODE_LIMIT} min 1 max {DEFAULT_MCTS_TREE_NODE_LIMIT}"
    )?;
    writeln!(
        output,
        "option name MultiPV type spin default 1 min 1 max {MAX_MULTI_PV}"
    )?;
    writeln!(output, "option name UCI_Chess960 type check default false")?;
    writeln!(output, "option name UCI_ShowWDL type check default true")?;
    writeln!(output, "option name Ponder type check default false")?;
    writeln!(output, "option name SyzygyPath type string default <empty>")?;
    writeln!(
        output,
        "option name SyzygyProbeDepth type spin default {DEFAULT_SYZYGY_PROBE_DEPTH} min 1 max {MAX_SYZYGY_PROBE_DEPTH}"
    )?;
    writeln!(
        output,
        "option name SyzygyProbeLimit type spin default {MAX_SYZYGY_PROBE_LIMIT} min 0 max {MAX_SYZYGY_PROBE_LIMIT}"
    )?;
    writeln!(
        output,
        "option name Syzygy50MoveRule type check default true"
    )?;
    writeln!(
        output,
        "option name Move Overhead type spin default {DEFAULT_MOVE_OVERHEAD_MS} min 0 max {MAX_MOVE_OVERHEAD_MS}"
    )?;
    writeln!(output, "uciok")
}

/// Used for applying one command while no search owns the engine state.
///
/// Accepted `go` and `go perft` commands start a background job; every other
/// command mutates state or writes diagnostics directly.
///
/// # Arguments
///
/// * `command` - parsed protocol command
/// * `state` - engine state to read and mutate
/// * `sender` - coordinator channel handed to new background jobs
/// * `next_search_id` - monotonic identifier source for new jobs
/// * `output` - sink receiving diagnostics and responses
///
/// # Returns
///
/// Newly started job for an accepted `go` or `go perft` command, [`None`]
/// otherwise.
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
#[allow(clippy::too_many_lines)]
fn handle_idle_command<W: Write>(
    command: Command,
    state: &mut EngineState,
    sender: &Sender<CoordinatorEvent>,
    next_search_id: &mut u64,
    output: &mut W,
) -> io::Result<Option<ActiveJob>> {
    handle_idle_command_with_spawner(
        command,
        state,
        sender,
        next_search_id,
        output,
        spawn_named_worker,
    )
}

/// Used for applying one idle command with an injectable job-spawn boundary.
///
/// Production reaches this through [`handle_idle_command`]. Tests inject a
/// deterministic refusal to prove that accepted searches retain their owned
/// engine state and emit one null result rather than panicking.
///
/// # Arguments
///
/// * `command` - parsed protocol command
/// * `state` - engine state to read and mutate
/// * `sender` - coordinator channel handed to a successfully started job
/// * `next_search_id` - monotonic identifier source
/// * `output` - sink receiving diagnostics and responses
/// * `worker_spawner` - fallible top-level job boundary
///
/// # Returns
///
/// Newly started job, or [`None`] for non-job commands and refused workers.
///
/// # Errors
///
/// Returns an [`io::Error`] only when writing protocol output fails. Spawn
/// refusal is reported inside the UCI stream and returned as `Ok(None)`.
#[allow(clippy::too_many_lines)]
fn handle_idle_command_with_spawner<W: Write>(
    command: Command,
    state: &mut EngineState,
    sender: &Sender<CoordinatorEvent>,
    next_search_id: &mut u64,
    output: &mut W,
    worker_spawner: WorkerSpawner,
) -> io::Result<Option<ActiveJob>> {
    match command {
        Command::NewGame => {
            state.position = Position::start();
            state.previous_keys.clear();
            state.position_history.clear();
            state.position_history.push(state.position.clone());
            clear_alpha_beta_workers(&mut state.alpha_beta);
        }
        Command::Position(spec) => match replay_position(&spec) {
            Ok((position, keys, positions)) => {
                state.position = position;
                state.previous_keys = keys;
                state.position_history = positions;
            }
            Err(error) => writeln!(output, "info string invalid position: {error}")?,
        },
        Command::SetOption { name, value } => handle_set_option(state, &name, &value, output)?,
        Command::Go(options) => {
            if !state
                .active_evaluator
                .kind()
                .supports_search(state.search_mode)
            {
                writeln!(
                    output,
                    "info string BT4 evaluation supports MCTS only; select Search MCTS"
                )?;
                writeln!(output, "bestmove 0000")?;
                return Ok(None);
            }
            #[cfg(feature = "mcts")]
            if state.search_mode == SearchMode::Mcts && state.search_threads > 1 {
                note_mcts_threading(&state.active_evaluator, output)?;
            }
            let root_moves = match resolve_search_moves(&state.position, &options.search_moves) {
                Ok(root_moves) => root_moves,
                Err(error) => {
                    writeln!(output, "info string invalid searchmoves: {error}")?;
                    writeln!(output, "bestmove 0000")?;
                    return Ok(None);
                }
            };
            #[cfg(feature = "mcts")]
            if state.search_mode == SearchMode::Mcts && state.multi_pv > 1 {
                writeln!(
                    output,
                    "info string MultiPV greater than 1 is currently supported only by AlphaBeta; MCTS uses 1"
                )?;
            }
            if options.ponder && state.debug && !state.ponder_enabled {
                writeln!(
                    output,
                    "info string go ponder accepted despite Ponder=false; command is authoritative"
                )?;
            }
            let search_id = *next_search_id;
            *next_search_id = next_search_id.wrapping_add(1).max(1);
            let search = start_search_with_spawner(
                state,
                options,
                root_moves,
                sender.clone(),
                search_id,
                worker_spawner,
            );
            return match search {
                Ok(search) => Ok(Some(ActiveJob::Search(Box::new(search)))),
                Err(error) => {
                    writeln!(output, "info string search worker could not start: {error}")?;
                    writeln!(output, "bestmove 0000")?;
                    Ok(None)
                }
            };
        }
        Command::Perft(options) => {
            if options.depth > MAX_PERFT_DEPTH {
                writeln!(
                    output,
                    "info string invalid go perft depth: {} (expected 0..={MAX_PERFT_DEPTH})",
                    options.depth
                )?;
                return Ok(None);
            }
            if state.search_threads != 1 {
                writeln!(
                    output,
                    "info string go perft requires Threads=1 (configured {})",
                    state.search_threads
                )?;
                return Ok(None);
            }
            let job_id = *next_search_id;
            *next_search_id = next_search_id.wrapping_add(1).max(1);
            let perft = start_perft_with_spawner(
                &state.position,
                options,
                state.uci_chess960,
                sender.clone(),
                job_id,
                worker_spawner,
            );
            return match perft {
                Ok(perft) => Ok(Some(ActiveJob::Perft(Box::new(perft)))),
                Err(error) => {
                    writeln!(output, "info string perft worker could not start: {error}")?;
                    Ok(None)
                }
            };
        }
        Command::IsReady => writeln!(output, "readyok")?,
        Command::Uci
        | Command::Stop
        | Command::PonderHit
        | Command::Quit
        | Command::Debug(_)
        | Command::Unknown(_) => {}
    }
    Ok(None)
}

/// Used for appending an ordinary deferred command without exceeding the
/// coordinator's explicit memory budget.
///
/// # Arguments
///
/// * `pending` - state/job commands waiting for the active worker to return
/// * `command` - newest parsed command
/// * `output` - sink receiving the overload diagnostic
///
/// # Returns
///
/// `true` when the command was retained, `false` when overload discarded it.
///
/// # Errors
///
/// Returns an [`io::Error`] when writing the overload diagnostic fails.
fn enqueue_pending<W: Write>(
    pending: &mut VecDeque<Command>,
    command: Command,
    output: &mut W,
) -> io::Result<bool> {
    if pending.len() >= MAX_PENDING_COMMANDS {
        writeln!(
            output,
            "info string pending command queue full; newest deferred command ignored"
        )?;
        return Ok(false);
    }
    pending.push_back(command);
    Ok(true)
}

/// Used for admitting a stop or readiness barrier when the deferred queue is
/// already full.
///
/// Control commands are never displaced. At overload, the newest ordinary
/// state/job command is removed to make room. If the queue contains only
/// readiness/stop controls, no entry can be removed and the caller may answer
/// the new readiness probe immediately.
///
/// # Arguments
///
/// * `pending` - commands waiting for the active worker to return
/// * `command` - [`Command::Stop`] or [`Command::IsReady`]
/// * `output` - sink receiving an overload diagnostic when state is displaced
///
/// # Returns
///
/// `true` when the control command was queued, `false` when a controls-only
/// queue left no replaceable entry.
///
/// # Errors
///
/// Returns an [`io::Error`] when writing the overload diagnostic fails.
fn enqueue_priority_pending<W: Write>(
    pending: &mut VecDeque<Command>,
    command: Command,
    output: &mut W,
) -> io::Result<bool> {
    debug_assert!(matches!(command, Command::Stop | Command::IsReady));
    if pending.len() < MAX_PENDING_COMMANDS {
        pending.push_back(command);
        return Ok(true);
    }
    let Some(index) = pending
        .iter()
        .rposition(|queued| !matches!(queued, Command::Stop | Command::IsReady))
    else {
        return Ok(false);
    };
    let _ = pending.remove(index);
    writeln!(
        output,
        "info string pending command queue full; discarded one deferred state command to preserve a control command"
    )?;
    pending.push_back(command);
    Ok(true)
}

/// Used for replaying queued commands until the queue empties or a search
/// starts.
///
/// A leading readiness barrier behind a newly started search is emitted without
/// cancelling that search: every earlier command has already been applied at
/// that point. Only a remaining command that needs the owned engine state asks
/// the new search to stop.
///
/// # Arguments
///
/// * `pending` - queued commands awaiting the idle engine state
/// * `active` - slot owning the single background job
/// * `state` - engine state to read and mutate
/// * `sender` - coordinator channel handed to new background jobs
/// * `next_search_id` - monotonic identifier source for new jobs
/// * `output` - sink receiving diagnostics and responses
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
///
/// # Panics
///
/// Panics only if the checked active job disappears between the emptiness
/// test and the `expect`.
fn start_pending_commands<W: Write>(
    pending: &mut VecDeque<Command>,
    active: &mut Option<ActiveJob>,
    state: &mut EngineState,
    sender: &Sender<CoordinatorEvent>,
    next_search_id: &mut u64,
    output: &mut W,
) -> io::Result<()> {
    while active.is_none() {
        let Some(command) = pending.pop_front() else {
            break;
        };
        *active = handle_idle_command(command, state, sender, next_search_id, output)?;
    }
    while active.is_some() && matches!(pending.front(), Some(Command::IsReady)) {
        let _ = pending.pop_front();
        writeln!(output, "readyok")?;
    }
    if active.is_some() && !pending.is_empty() {
        active.as_mut().expect("checked active job").request_stop();
    }
    Ok(())
}

/// Used for checking whether the newest unstopped queue segment contains a
/// background job.
///
/// Scanning walks backwards and stops at the most recent queued `Stop`, so
/// only `go` or perft commands queued after it matter.
///
/// # Arguments
///
/// * `pending` - queued commands awaiting the idle engine state
///
/// # Returns
///
/// `true` when a queued `Go` or `Perft` follows the last queued `Stop`.
fn pending_job_needs_stop(pending: &VecDeque<Command>) -> bool {
    pending
        .iter()
        .rev()
        .take_while(|command| !matches!(command, Command::Stop))
        .any(|command| matches!(command, Command::Go(_) | Command::Perft(_)))
}

/// Used for validating and applying one UCI option without partially
/// mutating on failure.
///
/// Option names are matched case-insensitively. Invalid values leave the
/// existing configuration untouched and emit one `info string` diagnostic.
///
/// TODO: the invalid-value diagnostics print inclusive ranges with
/// exclusive-looking `..` notation (for example "expected 1..16" although 16
/// is accepted); switch them to `..=` like the perft depth message for
/// consistency.
///
/// # Arguments
///
/// * `state` - engine state receiving the accepted option
/// * `name` - option name from `setoption`
/// * `value` - option value from `setoption`
/// * `output` - sink receiving diagnostics
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
#[allow(clippy::too_many_lines)]
fn handle_set_option<W: Write>(
    state: &mut EngineState,
    name: &str,
    value: &str,
    output: &mut W,
) -> io::Result<()> {
    #[cfg(feature = "mcts")]
    if name.eq_ignore_ascii_case("mcts tree nodes") {
        match value.parse::<usize>() {
            Ok(nodes) if (1..=DEFAULT_MCTS_TREE_NODE_LIMIT).contains(&nodes) => {
                state.mcts_tree_nodes = nodes;
            }
            _ => writeln!(
                output,
                "info string invalid MCTS Tree Nodes value: {value} (expected 1..={DEFAULT_MCTS_TREE_NODE_LIMIT})"
            )?,
        }
        return Ok(());
    }

    if name.eq_ignore_ascii_case("search") {
        if value.eq_ignore_ascii_case("alphabeta")
            || value.eq_ignore_ascii_case("alpha-beta")
            || value.eq_ignore_ascii_case("ab")
        {
            if state
                .active_evaluator
                .kind()
                .supports_search(SearchMode::AlphaBeta)
            {
                state.search_mode = SearchMode::AlphaBeta;
            } else {
                #[cfg(feature = "mcts")]
                {
                    state.search_mode = SearchMode::Mcts;
                    writeln!(
                        output,
                        "info string BT4 evaluation supports MCTS only; Search remains MCTS"
                    )?;
                }
                #[cfg(not(feature = "mcts"))]
                writeln!(
                    output,
                    "info string BT4 evaluation supports MCTS only, which was not compiled into this build"
                )?;
            }
        } else if value.eq_ignore_ascii_case("mcts") || value.eq_ignore_ascii_case("puct") {
            #[cfg(feature = "mcts")]
            {
                state.search_mode = SearchMode::Mcts;
            }
            #[cfg(not(feature = "mcts"))]
            writeln!(
                output,
                "info string MCTS search was not compiled into this build; Search remains AlphaBeta"
            )?;
        } else {
            writeln!(output, "info string invalid Search value: {value}")?;
        }
    } else if name.eq_ignore_ascii_case("eval file") || name.eq_ignore_ascii_case("evalfile") {
        match load_evaluator(value, state.bt4_backend, state.bt4_device) {
            Ok(evaluator) => {
                state.loaded_evaluator = Some(evaluator.clone());
                state.loaded_evaluator_path = Some(value.to_owned());
                // Loading a network must not override an evaluator the operator
                // selected by name. Cute Chess sends `Eval` and `Eval File`
                // together at startup, so installing unconditionally here
                // silently replaced `Eval=Classical` with the network.
                let keeps_classical = matches!(
                    state.requested_evaluator,
                    Some(EvaluatorSelection::Exact(EvaluatorKind::Classical))
                );
                if keeps_classical {
                    writeln!(
                        output,
                        "info string loaded Eval File: {value}; Eval=Classical remains active"
                    )?;
                } else {
                    install_evaluator(state, evaluator, output)?;
                    writeln!(output, "info string loaded evaluator: {value}")?;
                }
            }
            Err(error) => {
                writeln!(output, "info string failed to load Eval File: {error}")?;
            }
        }
    } else if name.eq_ignore_ascii_case("eval") {
        if let Some(selection) = parse_evaluator_selection(value) {
            state.requested_evaluator = Some(selection);
        }
        match parse_evaluator_selection(value) {
            Some(EvaluatorSelection::Exact(EvaluatorKind::Classical)) => {
                install_evaluator(state, JanusEvaluator::Classical, output)?;
            }
            Some(selection) => match state.loaded_evaluator.clone() {
                Some(evaluator) if selection.accepts(evaluator.kind()) => {
                    install_evaluator(state, evaluator, output)?;
                }
                Some(evaluator) => {
                    writeln!(
                        output,
                        "info string loaded Eval File is {}, not {}; load a compatible model first",
                        evaluator.kind().uci_name(),
                        selection.uci_name()
                    )?;
                }
                None => {
                    writeln!(
                        output,
                        "info string neural evaluation requires setoption name Eval File value PATH"
                    )?;
                }
            },
            None => writeln!(output, "info string invalid Eval value: {value}")?,
        }
    } else if name.eq_ignore_ascii_case("bt4 backend") {
        match parse_bt4_backend(value) {
            Some(backend) => {
                let device = state.bt4_device;
                apply_bt4_selection(
                    state,
                    backend,
                    device,
                    "BT4 Backend",
                    value,
                    output,
                )?;
            }
            None => writeln!(
                output,
                "info string invalid BT4 Backend value: {value} (expected Auto, CPU, Nvidia, AMD, or Intel)"
            )?,
        }
    } else if name.eq_ignore_ascii_case("bt4 device") {
        match value.parse::<usize>() {
            Ok(device) if device <= MAX_BT4_DEVICE_INDEX => {
                let backend = state.bt4_backend;
                apply_bt4_selection(state, backend, device, "BT4 Device", value, output)?;
            }
            _ => writeln!(
                output,
                "info string invalid BT4 Device value: {value} (expected 0..{MAX_BT4_DEVICE_INDEX})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("threads") {
        match value.parse::<usize>() {
            Ok(threads) if (1..=MAX_SEARCH_THREADS).contains(&threads) => {
                state.search_threads = threads;
                configure_inference_threads(&state.active_evaluator, threads);
                state.rebuild_alpha_beta();
                write_alpha_beta_admission(&state.alpha_beta, state.hash_mb, output)?;
                note_configured_threading(state, output)?;
            }
            _ => writeln!(
                output,
                "info string invalid Threads value: {value} (expected 1..{MAX_SEARCH_THREADS})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("hash") {
        match value.parse::<usize>() {
            Ok(megabytes) if (1..=MAX_HASH_MB).contains(&megabytes) => {
                state.hash_mb = megabytes;
                state.rebuild_alpha_beta();
                write_alpha_beta_admission(&state.alpha_beta, state.hash_mb, output)?;
            }
            _ => writeln!(
                output,
                "info string invalid Hash value: {value} (expected 1..{MAX_HASH_MB})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("clear hash") {
        clear_alpha_beta_workers(&mut state.alpha_beta);
    } else if name.eq_ignore_ascii_case("multipv") {
        match value.parse::<usize>() {
            Ok(multi_pv) if (1..=MAX_MULTI_PV).contains(&multi_pv) => {
                state.multi_pv = multi_pv;
            }
            _ => writeln!(
                output,
                "info string invalid MultiPV value: {value} (expected 1..{MAX_MULTI_PV})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("uci_chess960") {
        match parse_check_option(value) {
            Some(enabled) => state.uci_chess960 = enabled,
            None => writeln!(output, "info string invalid UCI_Chess960 value: {value}")?,
        }
    } else if name.eq_ignore_ascii_case("uci_showwdl") {
        match parse_check_option(value) {
            Some(enabled) => state.show_wdl = enabled,
            None => writeln!(output, "info string invalid UCI_ShowWDL value: {value}")?,
        }
    } else if name.eq_ignore_ascii_case("ponder") {
        match parse_check_option(value) {
            Some(enabled) => state.ponder_enabled = enabled,
            None => writeln!(output, "info string invalid Ponder value: {value}")?,
        }
    } else if name.eq_ignore_ascii_case("syzygypath") {
        let path = value.trim();
        if path.is_empty() || path.eq_ignore_ascii_case("<empty>") {
            state.syzygy = None;
            state.apply_syzygy_to_workers();
            writeln!(output, "info string Syzygy tablebases disabled")?;
        } else {
            match Tablebases::new(path) {
                Ok(tables) => {
                    writeln!(
                        output,
                        "info string Syzygy: found {} WDL and {} DTZ files (up to {}-man)",
                        tables.wdl_file_count(),
                        tables.dtz_file_count(),
                        tables.max_cardinality()
                    )?;
                    state.syzygy = Some(Arc::new(tables));
                    state.apply_syzygy_to_workers();
                }
                Err(error) => {
                    writeln!(output, "info string failed to set SyzygyPath: {error}")?;
                }
            }
        }
    } else if name.eq_ignore_ascii_case("syzygyprobedepth") {
        match value.parse::<u8>() {
            Ok(depth) if (1..=MAX_SYZYGY_PROBE_DEPTH).contains(&depth) => {
                state.syzygy_probe_depth = depth;
                state.apply_syzygy_to_workers();
            }
            _ => writeln!(
                output,
                "info string invalid SyzygyProbeDepth value: {value} (expected 1..{MAX_SYZYGY_PROBE_DEPTH})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("syzygyprobelimit") {
        match value.parse::<u8>() {
            Ok(limit) if limit <= MAX_SYZYGY_PROBE_LIMIT => {
                state.syzygy_probe_limit = limit;
                state.apply_syzygy_to_workers();
            }
            _ => writeln!(
                output,
                "info string invalid SyzygyProbeLimit value: {value} (expected 0..{MAX_SYZYGY_PROBE_LIMIT})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("syzygy50moverule") {
        match parse_check_option(value) {
            Some(enabled) => {
                state.syzygy_rule50 = enabled;
                state.apply_syzygy_to_workers();
            }
            None => writeln!(
                output,
                "info string invalid Syzygy50MoveRule value: {value}"
            )?,
        }
    } else if name.eq_ignore_ascii_case("root fail high reduction") {
        match value.parse::<i32>() {
            Ok(reduction) if (0..=MAX_ROOT_FAIL_HIGH_REDUCTION).contains(&reduction) => {
                state.root_fail_high_reduction = reduction;
                state.apply_falling_eval_to_workers();
            }
            _ => writeln!(
                output,
                "info string invalid Root Fail High Reduction value: {value} (expected 0..{MAX_ROOT_FAIL_HIGH_REDUCTION})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("falling eval percent") {
        match value.parse::<i32>() {
            Ok(percent) if (0..=MAX_FALLING_EVAL_PERCENT).contains(&percent) => {
                state.falling_eval_percent = percent;
                state.apply_falling_eval_to_workers();
            }
            _ => writeln!(
                output,
                "info string invalid Falling Eval Percent value: {value} (expected 0..{MAX_FALLING_EVAL_PERCENT})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("clock moves to go") {
        match value.parse::<u32>() {
            Ok(moves) if (1..=MAX_CLOCK_MOVES_TO_GO).contains(&moves) => {
                state.clock_moves_to_go = moves;
            }
            _ => writeln!(
                output,
                "info string invalid Clock Moves To Go value: {value} (expected 1..{MAX_CLOCK_MOVES_TO_GO})"
            )?,
        }
    } else if name.eq_ignore_ascii_case("move overhead") {
        match value.parse::<u64>() {
            Ok(milliseconds) if milliseconds <= MAX_MOVE_OVERHEAD_MS => {
                state.move_overhead_ms = milliseconds;
            }
            _ => writeln!(
                output,
                "info string invalid Move Overhead value: {value} (expected 0..{MAX_MOVE_OVERHEAD_MS})"
            )?,
        }
    } else {
        writeln!(output, "info string unknown option: {name}")?;
    }
    Ok(())
}

/// Used for parsing documented evaluator names and narrow
/// backwards-compatible aliases.
///
/// Matching is case-insensitive; `nnue` maps to the generic
/// [`EvaluatorSelection::AnyNnue`] alias.
///
/// # Arguments
///
/// * `value` - UCI `Eval` combo value
///
/// # Returns
///
/// Parsed selection, or [`None`] for an unrecognized name.
fn parse_evaluator_selection(value: &str) -> Option<EvaluatorSelection> {
    if value.eq_ignore_ascii_case("classical") {
        Some(EvaluatorSelection::Exact(EvaluatorKind::Classical))
    } else if value.eq_ignore_ascii_case("compactnnue")
        || value.eq_ignore_ascii_case("compact-nnue")
    {
        Some(EvaluatorSelection::Exact(EvaluatorKind::CompactNnue))
    } else if value.eq_ignore_ascii_case("upstreamnnue")
        || value.eq_ignore_ascii_case("upstream-nnue")
    {
        Some(EvaluatorSelection::Exact(EvaluatorKind::UpstreamNnue))
    } else if value.eq_ignore_ascii_case("cnn") {
        Some(EvaluatorSelection::Exact(EvaluatorKind::Cnn))
    } else if value.eq_ignore_ascii_case("bt4") {
        Some(EvaluatorSelection::Exact(EvaluatorKind::Bt4))
    } else if value.eq_ignore_ascii_case("jrbt") {
        Some(EvaluatorSelection::Exact(EvaluatorKind::Jrbt))
    } else if value.eq_ignore_ascii_case("nnue") {
        Some(EvaluatorSelection::AnyNnue)
    } else {
        None
    }
}

/// Used for parsing the exact documented BT4 backend selections, covering
/// the `Auto` and `CPU` policies plus the `OpenCL` vendor names.
///
/// # Arguments
///
/// * `value` - UCI `BT4 Backend` combo value, matched case-insensitively
///
/// # Returns
///
/// Parsed backend, or [`None`] for an unrecognized name.
fn parse_bt4_backend(value: &str) -> Option<Bt4Backend> {
    if value.eq_ignore_ascii_case("auto") {
        Some(Bt4Backend::Auto)
    } else if value.eq_ignore_ascii_case("cpu") {
        Some(Bt4Backend::Cpu)
    } else if value.eq_ignore_ascii_case("nvidia") {
        Some(Bt4Backend::Nvidia)
    } else if value.eq_ignore_ascii_case("amd") {
        Some(Bt4Backend::Amd)
    } else if value.eq_ignore_ascii_case("intel") {
        Some(Bt4Backend::Intel)
    } else {
        None
    }
}

/// Used for retrieving the stable spelling of one BT4 backend in UCI
/// diagnostics.
///
/// # Arguments
///
/// * `backend` - backend to name
///
/// # Returns
///
/// Static spelling such as `"Nvidia"`.
const fn bt4_backend_name(backend: Bt4Backend) -> &'static str {
    match backend {
        Bt4Backend::Auto => "Auto",
        Bt4Backend::Cpu => "CPU",
        Bt4Backend::Nvidia => "Nvidia",
        Bt4Backend::Amd => "AMD",
        Bt4Backend::Intel => "Intel",
    }
}

/// Used for applying a BT4 backend/device pair, reloading a cached BT4
/// model atomically.
///
/// Explicit GPU requests are intentionally strict in the engine loader. If a
/// requested helper or device cannot initialize, this function retains the
/// previous requested selection, cached model, and active evaluator.
///
/// # Arguments
///
/// * `state` - engine state holding the BT4 selection and cached model
/// * `backend` - requested execution backend
/// * `device` - requested zero-based device ordinal
/// * `option_name` - option spelling used in failure diagnostics
/// * `option_value` - raw value used in failure diagnostics
/// * `output` - sink receiving diagnostics
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn apply_bt4_selection<W: Write>(
    state: &mut EngineState,
    backend: Bt4Backend,
    device: usize,
    option_name: &str,
    option_value: &str,
    output: &mut W,
) -> io::Result<()> {
    let cached_bt4_path = if matches!(&state.loaded_evaluator, Some(JanusEvaluator::Bt4(_))) {
        state.loaded_evaluator_path.clone()
    } else {
        None
    };
    let active_bt4 = matches!(&state.active_evaluator, JanusEvaluator::Bt4(_));

    if let Some(path) = cached_bt4_path {
        let evaluator = match load_evaluator(&path, backend, device) {
            Ok(evaluator) => evaluator,
            Err(error) => {
                writeln!(
                    output,
                    "info string failed to set {option_name} to {option_value}: {}; previous BT4 selection retained",
                    sanitize_uci_text(&error)
                )?;
                return Ok(());
            }
        };

        state.bt4_backend = backend;
        state.bt4_device = device;
        state.loaded_evaluator = Some(evaluator.clone());
        if active_bt4 {
            install_evaluator(state, evaluator, output)?;
        } else {
            note_bt4_backend_status(&evaluator, output)?;
        }
        return Ok(());
    }

    state.bt4_backend = backend;
    state.bt4_device = device;
    writeln!(
        output,
        "info string BT4 backend configured requested {} device {device}; applies on next BT4 load",
        bt4_backend_name(backend)
    )
}

/// Used for installing an evaluator and rebuilding every evaluator-owning
/// searcher.
///
/// Installing a BT4 evaluator forces the search mode to MCTS, and the
/// resulting backend status and threading notes are reported through
/// `output`.
///
/// # Arguments
///
/// * `state` - engine state receiving the evaluator
/// * `evaluator` - evaluator to install
/// * `output` - sink receiving diagnostics
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn install_evaluator<W: Write>(
    state: &mut EngineState,
    evaluator: JanusEvaluator,
    output: &mut W,
) -> io::Result<()> {
    configure_inference_threads(&evaluator, state.search_threads);
    if !evaluator.kind().supports_search(state.search_mode) {
        #[cfg(feature = "mcts")]
        {
            state.search_mode = SearchMode::Mcts;
            writeln!(
                output,
                "info string BT4 evaluation requires MCTS; Search set to MCTS"
            )?;
        }
        #[cfg(not(feature = "mcts"))]
        writeln!(
            output,
            "info string BT4 evaluation requires MCTS, which was not compiled into this build"
        )?;
    }
    state.active_evaluator = evaluator;
    state.rebuild_alpha_beta();
    write_alpha_beta_admission(&state.alpha_beta, state.hash_mb, output)?;
    #[cfg(feature = "mcts")]
    state.rebuild_mcts();
    report_installed_bt4_backend(state, output)?;
    note_configured_threading(state, output)
}

/// Used for reporting a freshly installed BT4 backend and recording its
/// observable state.
///
/// # Arguments
///
/// * `state` - engine state whose reported status is refreshed
/// * `output` - sink receiving the diagnostic lines
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn report_installed_bt4_backend<W: Write>(
    state: &mut EngineState,
    output: &mut W,
) -> io::Result<()> {
    let status = bt4_backend_status(&state.active_evaluator);
    if let Some(status) = &status {
        write_bt4_backend_status(status, output)?;
    }
    state.reported_bt4_status = status;
    Ok(())
}

/// Used for reporting the selected BT4 execution backend when a BT4
/// evaluator is present.
///
/// # Arguments
///
/// * `evaluator` - evaluator possibly carrying a BT4 network
/// * `output` - sink receiving the diagnostic lines
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn note_bt4_backend_status<W: Write>(evaluator: &JanusEvaluator, output: &mut W) -> io::Result<()> {
    if let Some(status) = bt4_backend_status(evaluator) {
        write_bt4_backend_status(&status, output)?;
    }
    Ok(())
}

/// Used for taking an owned snapshot without retaining the BT4 network lock
/// during I/O.
///
/// # Arguments
///
/// * `evaluator` - evaluator possibly carrying a BT4 network
///
/// # Returns
///
/// Cloned backend status, or [`None`] for non-BT4 evaluators.
fn bt4_backend_status(evaluator: &JanusEvaluator) -> Option<Bt4BackendStatus> {
    let JanusEvaluator::Bt4(network) = evaluator else {
        return None;
    };
    Some(
        network
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .backend_status()
            .clone(),
    )
}

/// Used for emitting a runtime GPU-to-CPU transition exactly once after
/// search reclamation.
///
/// The current snapshot is compared to the last reported status; only a
/// change is written and recorded.
///
/// # Arguments
///
/// * `state` - engine state carrying the last reported status
/// * `output` - sink receiving the diagnostic lines
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
#[cfg(feature = "mcts")]
fn report_bt4_backend_change<W: Write>(state: &mut EngineState, output: &mut W) -> io::Result<()> {
    let current = bt4_backend_status(&state.active_evaluator);
    if current != state.reported_bt4_status {
        if let Some(status) = &current {
            write_bt4_backend_status(status, output)?;
        }
        state.reported_bt4_status = current;
    }
    Ok(())
}

/// Used for writing stable single-line UCI diagnostics for a resolved BT4
/// backend.
///
/// A fallback reason, when present, is sanitized and written on its own
/// line.
///
/// # Arguments
///
/// * `status` - resolved backend status snapshot
/// * `output` - sink receiving the diagnostic lines
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn write_bt4_backend_status<W: Write>(status: &Bt4BackendStatus, output: &mut W) -> io::Result<()> {
    let device_name = status
        .device_name()
        .map_or_else(|| "<none>".to_owned(), sanitize_uci_text);
    writeln!(
        output,
        "info string BT4 backend requested {} active {} device {} name {device_name}",
        bt4_backend_name(status.requested()),
        bt4_backend_name(status.active()),
        status.device_index()
    )?;
    if let Some(reason) = status.fallback_reason() {
        writeln!(
            output,
            "info string BT4 backend fallback {}",
            sanitize_uci_text(reason)
        )?;
    }
    Ok(())
}

/// Used for replacing UCI line separators and control whitespace in
/// external diagnostics.
///
/// Every control character is replaced with a space so untrusted text
/// cannot break a single `info string` line.
///
/// # Arguments
///
/// * `text` - external diagnostic text
///
/// # Returns
///
/// Sanitized copy safe to embed in one UCI line.
fn sanitize_uci_text(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

/// Used for applying UCI `Threads` to a neural evaluator's bounded internal
/// team.
///
/// Only CNN and BT4 own inference teams; other backends ignore the call.
///
/// # Arguments
///
/// * `evaluator` - evaluator receiving the thread count
/// * `threads` - configured UCI `Threads` value
fn configure_inference_threads(evaluator: &JanusEvaluator, threads: usize) {
    match evaluator {
        JanusEvaluator::Cnn(network) => network
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_inference_threads(threads),
        JanusEvaluator::Bt4(network) => network
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_inference_threads(threads),
        JanusEvaluator::Classical
        | JanusEvaluator::CompactNnue(_)
        | JanusEvaluator::UpstreamNnue(_)
        | JanusEvaluator::Jrbt(_) => {}
    }
}

/// Used for supplying temporal neural evaluators with UCI positions ending
/// at the root.
///
/// Only CNN and BT4 consume history; other backends ignore the call.
///
/// # Arguments
///
/// * `evaluator` - evaluator receiving the history
/// * `positions` - real game positions ending at the search root
#[cfg(feature = "mcts")]
fn configure_mcts_root_history(evaluator: &JanusEvaluator, positions: &[Position]) {
    match evaluator {
        JanusEvaluator::Cnn(network) => network
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_root_history(positions),
        JanusEvaluator::Bt4(network) => network
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_root_history(positions),
        JanusEvaluator::Classical
        | JanusEvaluator::CompactNnue(_)
        | JanusEvaluator::UpstreamNnue(_)
        | JanusEvaluator::Jrbt(_) => {}
    }
}

/// Used for explaining whether `Threads` accelerates neural inference
/// inside serial MCTS.
///
/// BT4 running on a GPU backend receives the dedicated `OpenCL` note; CPU
/// neural backends report their inference thread count; classical and NNUE
/// backends receive the generic single-threaded MCTS note.
///
/// # Arguments
///
/// * `evaluator` - active evaluator to describe
/// * `output` - sink receiving the diagnostic line
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
#[cfg(feature = "mcts")]
fn note_mcts_threading<W: Write>(evaluator: &JanusEvaluator, output: &mut W) -> io::Result<()> {
    let inference = match evaluator {
        JanusEvaluator::Cnn(network) => Some((
            EvaluatorKind::Cnn,
            network
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .inference_threads(),
        )),
        JanusEvaluator::Bt4(network) => {
            let network = network
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let threads = network.inference_threads();
            let active = network.backend_status().active();
            if active != Bt4Backend::Cpu {
                return write_bt4_gpu_threading_note(active, threads, output);
            }
            Some((EvaluatorKind::Bt4, threads))
        }
        JanusEvaluator::Classical
        | JanusEvaluator::CompactNnue(_)
        | JanusEvaluator::UpstreamNnue(_)
        | JanusEvaluator::Jrbt(_) => None,
    };
    write_mcts_threading_note(inference, output)
}

/// Used for explaining that the `OpenCL` runtime owns GPU parallelism and
/// CPU fallback uses the configured Rust worker count.
///
/// # Arguments
///
/// * `backend` - active GPU backend named in the note
/// * `fallback_threads` - configured CPU fallback thread count
/// * `output` - sink receiving the diagnostic line
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
#[cfg(feature = "mcts")]
fn write_bt4_gpu_threading_note<W: Write>(
    backend: Bt4Backend,
    fallback_threads: usize,
    output: &mut W,
) -> io::Result<()> {
    writeln!(
        output,
        "info string MCTS tree search is single-threaded; {backend} OpenCL inference is device-managed; Threads {fallback_threads} configures CPU fallback"
    )
}

/// Used for writing the stable UCI diagnostic for one MCTS/evaluator
/// combination.
///
/// # Arguments
///
/// * `inference` - evaluator family and inference thread count, when the
///   active evaluator owns an internal team
/// * `output` - sink receiving the diagnostic line
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
#[cfg(feature = "mcts")]
fn write_mcts_threading_note<W: Write>(
    inference: Option<(EvaluatorKind, usize)>,
    output: &mut W,
) -> io::Result<()> {
    if let Some((kind, inference_threads)) = inference {
        writeln!(
            output,
            "info string MCTS tree search is single-threaded; {} uses {inference_threads} inference threads",
            kind.uci_name()
        )
    } else {
        writeln!(
            output,
            "info string Threads applies to AlphaBeta; MCTS is single-threaded"
        )
    }
}

/// Used for reporting the selected search mode's effective use of
/// `Threads`.
///
/// Nothing is written for a single-thread configuration.
///
/// # Arguments
///
/// * `state` - engine state carrying the thread count and search mode
/// * `output` - sink receiving the diagnostic line
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn note_configured_threading<W: Write>(state: &EngineState, output: &mut W) -> io::Result<()> {
    if state.search_threads <= 1 {
        return Ok(());
    }
    match state.search_mode {
        #[cfg(feature = "mcts")]
        SearchMode::Mcts => note_mcts_threading(&state.active_evaluator, output),
        SearchMode::AlphaBeta => note_cnn_thread_cap(state, output),
    }
}

/// Used for reporting how CNN inference uses `Threads` inside one
/// alpha-beta worker.
///
/// The note is written only when more than one thread is configured and the
/// active evaluator is the CNN backend.
///
/// # Arguments
///
/// * `state` - engine state carrying the thread count and evaluator
/// * `output` - sink receiving the diagnostic line
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn note_cnn_thread_cap<W: Write>(state: &EngineState, output: &mut W) -> io::Result<()> {
    if state.search_threads > 1 {
        if let JanusEvaluator::Cnn(network) = &state.active_evaluator {
            let inference_threads = network
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .inference_threads();
            writeln!(
                output,
                "info string CNN uses {inference_threads} inference threads; AlphaBeta search is capped to 1 worker"
            )?;
        }
    }
    Ok(())
}

/// Used for starting one checked CPU-only perft job without transferring
/// engine state.
///
/// The worker clones the root, runs the requested enumeration behind a
/// panic boundary, and reports completion or panic through the coordinator
/// channel together with the wall time measured inside the worker.
///
/// # Arguments
///
/// * `root` - position to enumerate
/// * `options` - requested depth and output format
/// * `uci_chess960` - whether move text uses Chess960 rook-source notation
/// * `sender` - coordinator channel receiving the completion event
/// * `job_id` - identifier echoed by every event of this job
/// * `worker_spawner` - fallible top-level worker boundary
///
/// # Returns
///
/// Coordinator-side metadata for the running job.
///
/// # Errors
///
/// Returns the operating-system spawn error without starting enumeration.
fn start_perft_with_spawner(
    root: &Position,
    options: PerftOptions,
    uci_chess960: bool,
    sender: Sender<CoordinatorEvent>,
    job_id: u64,
    worker_spawner: WorkerSpawner,
) -> io::Result<ActivePerft> {
    let root = root.clone();
    let mut worker_root = root.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let format = options.format;
    let handle = worker_spawner(
        "janus-perft",
        Box::new(move || {
            let started = Instant::now();
            let outcome = catch_unwind(AssertUnwindSafe(|| match format {
                PerftFormat::Detail => {
                    detailed_perft(&mut worker_root, options.depth, &worker_stop)
                        .map(FinishedPerft::Detail)
                }
                PerftFormat::Divide => {
                    detailed_divide(&mut worker_root, options.depth, &worker_stop)
                        .map(FinishedPerft::Divide)
                }
                PerftFormat::Stockfish => {
                    checked_divide(&mut worker_root, options.depth, &worker_stop)
                        .map(FinishedPerft::Stockfish)
                }
            }));
            let elapsed = started.elapsed();
            match outcome {
                Ok(result) => {
                    let _ = sender.send(CoordinatorEvent::PerftFinished {
                        job_id,
                        result,
                        elapsed,
                    });
                }
                Err(_) => {
                    let _ = sender.send(CoordinatorEvent::PerftPanicked { job_id });
                }
            }
        }),
    )?;
    Ok(ActivePerft {
        id: job_id,
        root,
        depth: options.depth,
        format: options.format,
        uci_chess960,
        stop,
        handle,
    })
}

/// Used for reclaiming a completed perft job and emitting only complete or
/// explicit terminal output.
///
/// Cancellation and counter overflow produce a diagnostic instead of
/// counters so no partial totals are ever reported.
///
/// # Arguments
///
/// * `active` - metadata of the finished job
/// * `result` - complete payload or controlled failure
/// * `elapsed` - wall time measured inside the worker
/// * `output` - sink receiving the report
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn finish_perft<W: Write>(
    active: ActivePerft,
    result: Result<FinishedPerft, PerftError>,
    elapsed: Duration,
    output: &mut W,
) -> io::Result<()> {
    let _ = active.handle.join();
    match result {
        Ok(FinishedPerft::Detail(stats)) => {
            debug_assert_eq!(active.format, PerftFormat::Detail);
            write_perft_detail(output, &active.root, active.depth, stats, elapsed)
        }
        Ok(FinishedPerft::Divide(result)) => {
            debug_assert_eq!(active.format, PerftFormat::Divide);
            write_perft_divide(
                output,
                &active.root,
                active.depth,
                active.uci_chess960,
                &result,
                elapsed,
            )
        }
        Ok(FinishedPerft::Stockfish(result)) => {
            debug_assert_eq!(active.format, PerftFormat::Stockfish);
            write_stockfish_perft(output, &active.root, active.uci_chess960, &result)
        }
        Err(PerftError::Cancelled) => {
            writeln!(output, "info string perft cancelled; no complete totals")
        }
        Err(PerftError::CounterOverflow) => {
            writeln!(
                output,
                "info string perft failed: counter overflow; no complete totals"
            )
        }
    }
}

/// Used for reclaiming a panicked perft worker while emitting no misleading
/// counters or move.
///
/// # Arguments
///
/// * `active` - metadata of the panicked job
/// * `output` - sink receiving the diagnostic
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn finish_panicked_perft<W: Write>(active: ActivePerft, output: &mut W) -> io::Result<()> {
    let _ = active.handle.join();
    writeln!(
        output,
        "info string perft worker panicked; no complete totals"
    )
}

/// Root state reported before a detailed positive-depth enumeration.
///
/// Classified by [`perft_root_status`] and written as one lowercase token
/// in the detailed perft header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PerftRootStatus {
    /// Used for indicating the side to move is neither checked nor
    /// terminal.
    Normal,
    /// Used for indicating the side to move is checked and has a legal
    /// evasion.
    Check,
    /// Used for indicating the side to move is checked with no legal move.
    Checkmate,
    /// Used for indicating the side to move is not checked and has no legal
    /// move.
    Stalemate,
}

impl PerftRootStatus {
    /// Used for retrieving the stable lowercase protocol spelling.
    ///
    /// # Returns
    ///
    /// Static token such as `"checkmate"`.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Check => "check",
            Self::Checkmate => "checkmate",
            Self::Stalemate => "stalemate",
        }
    }
}

/// Used for classifying check and terminal state without duplicating core
/// legality.
///
/// # Arguments
///
/// * `root` - position to classify
///
/// # Returns
///
/// Root status derived from check state and legal-move availability.
fn perft_root_status(root: &Position) -> PerftRootStatus {
    let checked = root.in_check(root.side_to_move());
    let terminal = root.legal_moves().is_empty();
    match (checked, terminal) {
        (true, true) => PerftRootStatus::Checkmate,
        (true, false) => PerftRootStatus::Check,
        (false, true) => PerftRootStatus::Stalemate,
        (false, false) => PerftRootStatus::Normal,
    }
}

/// Used for writing one detailed aggregate with deterministic fields before
/// timing fields.
///
/// # Arguments
///
/// * `output` - sink receiving the report
/// * `root` - enumerated root position
/// * `depth` - requested enumeration depth
/// * `stats` - complete aggregate counters
/// * `elapsed` - wall time measured inside the worker
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn write_perft_detail<W: Write>(
    output: &mut W,
    root: &Position,
    depth: u32,
    stats: PerftStats,
    elapsed: Duration,
) -> io::Result<()> {
    writeln!(output, "info string perft fen {}", root.to_fen())?;
    writeln!(
        output,
        "info string perft root {}",
        perft_root_status(root).as_str()
    )?;
    writeln!(output, "info string perft depth {depth}")?;
    writeln!(output, "info string perft nodes {}", stats.nodes)?;
    writeln!(output, "info string perft captures {}", stats.captures)?;
    writeln!(output, "info string perft en_passant {}", stats.en_passant)?;
    writeln!(output, "info string perft castles {}", stats.castles)?;
    writeln!(output, "info string perft promotions {}", stats.promotions)?;
    writeln!(output, "info string perft checks {}", stats.checks)?;
    writeln!(output, "info string perft checkmates {}", stats.checkmates)?;
    writeln!(output, "info string perft time_ms {}", elapsed.as_millis())?;
    writeln!(
        output,
        "info string perft nps {}",
        perft_nps(stats.nodes, elapsed)
    )?;
    writeln!(output, "info string perft complete")
}

/// Used for writing detailed divide rows in the actual configured wire-text
/// order.
///
/// Rows are sorted by their formatted move text, followed by the aggregate
/// total and one summary line.
///
/// # Arguments
///
/// * `output` - sink receiving the report
/// * `root` - enumerated root position
/// * `depth` - requested enumeration depth
/// * `uci_chess960` - whether move text uses Chess960 rook-source notation
/// * `result` - per-root-move counters and total
/// * `elapsed` - wall time measured inside the worker
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn write_perft_divide<W: Write>(
    output: &mut W,
    root: &Position,
    depth: u32,
    uci_chess960: bool,
    result: &PerftDivideResult,
    elapsed: Duration,
) -> io::Result<()> {
    let mut rows: Vec<(String, PerftStats)> = result
        .entries
        .iter()
        .map(|entry| {
            (
                root.uci_move_string(entry.root_move, uci_chess960),
                entry.stats,
            )
        })
        .collect();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    for (mv, stats) in &rows {
        writeln!(
            output,
            "info string perft divide {mv} nodes {} captures {} en_passant {} castles {} promotions {} checks {} checkmates {}",
            stats.nodes,
            stats.captures,
            stats.en_passant,
            stats.castles,
            stats.promotions,
            stats.checks,
            stats.checkmates
        )?;
    }
    let total = result.total;
    writeln!(
        output,
        "info string perft total nodes {} captures {} en_passant {} castles {} promotions {} checks {} checkmates {}",
        total.nodes,
        total.captures,
        total.en_passant,
        total.castles,
        total.promotions,
        total.checks,
        total.checkmates
    )?;
    writeln!(
        output,
        "info string perft summary depth {depth} moves {} nodes {} time_ms {} nps {}",
        rows.len(),
        total.nodes,
        elapsed.as_millis(),
        perft_nps(total.nodes, elapsed)
    )?;
    writeln!(output, "info string perft complete")
}

/// Used for writing node-only rows and the exact Stockfish-compatible final
/// total shape.
///
/// Rows are sorted by their formatted move text and written without the
/// `info string` prefix.
///
/// # Arguments
///
/// * `output` - sink receiving the report
/// * `root` - enumerated root position
/// * `uci_chess960` - whether move text uses Chess960 rook-source notation
/// * `result` - per-root-move node counts and total
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn write_stockfish_perft<W: Write>(
    output: &mut W,
    root: &Position,
    uci_chess960: bool,
    result: &PerftNodeResult,
) -> io::Result<()> {
    let mut rows: Vec<(String, u64)> = result
        .entries
        .iter()
        .map(|entry| {
            (
                root.uci_move_string(entry.root_move, uci_chess960),
                entry.nodes,
            )
        })
        .collect();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    for (mv, nodes) in rows {
        writeln!(output, "{mv}: {nodes}")?;
    }
    writeln!(output, "Nodes searched: {}", result.total_nodes)
}

/// Used for converting monotonic elapsed time and complete nodes into a
/// saturated rate.
///
/// The elapsed time is clamped to at least one nanosecond, and rates wider
/// than `u64` saturate at [`u64::MAX`].
///
/// # Arguments
///
/// * `nodes` - complete node total
/// * `elapsed` - wall time of the enumeration
///
/// # Returns
///
/// Nodes per second.
fn perft_nps(nodes: u64, elapsed: Duration) -> u64 {
    let nanos = elapsed.as_nanos().max(1);
    let nps = u128::from(nodes).saturating_mul(1_000_000_000) / nanos;
    u64::try_from(nps).unwrap_or(u64::MAX)
}

/// Used for moving the selected searcher into a background job and
/// returning its metadata.
///
/// Infinite jobs may continue searching indefinitely, but a naturally
/// completed proof is retained until `stop` so UCI never emits an early move.
///
/// # Arguments
///
/// * `state` - engine state lending its searcher to the job
/// * `options` - parsed `go` fields
/// * `root_moves` - optional validated `searchmoves` restriction
/// * `sender` - coordinator channel receiving job events
/// * `search_id` - identifier echoed by every event of this job
/// * `worker_spawner` - fallible top-level worker boundary
///
/// # Returns
///
/// Coordinator-side metadata for the running search.
///
/// # Errors
///
/// Returns the operating-system spawn error after restoring the alpha-beta
/// worker pool or MCTS searcher to `state` exactly.
///
/// # Panics
///
/// Panics if the idle engine state does not own its MCTS searcher when an
/// MCTS job starts.
#[allow(clippy::too_many_lines)]
#[cfg_attr(not(feature = "mcts"), allow(clippy::needless_pass_by_value))]
fn start_search_with_spawner(
    state: &mut EngineState,
    options: GoOptions,
    root_moves: Option<Vec<Move>>,
    sender: Sender<CoordinatorEvent>,
    search_id: u64,
    worker_spawner: WorkerSpawner,
) -> io::Result<ActiveSearch> {
    let mode = state.search_mode;
    let defer_until_stop = is_unbounded_uci_search(&options) || options.ponder;
    let root = state.position.clone();
    let ponder = options
        .ponder
        .then(|| ponder_state(&options, &root, state.move_overhead_ms, mode));
    let live_clock = ponder
        .as_ref()
        .and_then(|ponder| ponder.deadline.as_ref())
        .map(|deadline| Arc::clone(&deadline.clock));
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let worker_root = root.clone();
    let history = state.previous_keys.clone();

    #[cfg(feature = "mcts")]
    if mode == SearchMode::Mcts {
        configure_mcts_root_history(&state.active_evaluator, &state.position_history);
    }

    let handle = match mode {
        SearchMode::AlphaBeta => {
            let resources = Arc::new(Mutex::new(Some(std::mem::take(&mut state.alpha_beta))));
            let worker_resources = Arc::clone(&resources);
            let mut limits = alpha_beta_limits(
                &options,
                &root,
                state.move_overhead_ms,
                state.clock_moves_to_go,
            );
            if let Some(clock) = live_clock {
                limits = limits.with_live_clock(clock);
            }
            let multi_pv = if root.legal_moves().is_empty() {
                1
            } else {
                state.multi_pv
            };
            let task = Box::new(move || {
                let mut workers = worker_resources
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .expect("a started alpha-beta job owns its worker pool");
                let event_sender = sender.clone();
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    let info_sender = event_sender.clone();
                    let outcome = if multi_pv == 1 {
                        let outcome = search_alpha_beta_parallel(
                            &mut workers,
                            &worker_root,
                            &history,
                            &limits,
                            &worker_stop,
                            root_moves.as_deref(),
                            |info| {
                                let _ = info_sender.send(CoordinatorEvent::AlphaBetaInfo {
                                    search_id,
                                    multipv: 1,
                                    info: info.clone(),
                                });
                            },
                        );
                        ParallelSearchOutcome {
                            result: outcome.result.map(|result| vec![result]),
                            helper_spawn_failure: outcome.helper_spawn_failure,
                            helper_workspace_refusals: outcome.helper_workspace_refusals,
                        }
                    } else {
                        search_alpha_beta_parallel_multi(
                            &mut workers,
                            &worker_root,
                            &history,
                            &limits,
                            &worker_stop,
                            root_moves.as_deref(),
                            multi_pv,
                            |infos| send_alpha_beta_infos(&info_sender, search_id, infos),
                        )
                    };
                    if let Ok(lines) = &outcome.result {
                        send_final_alpha_beta_infos(&event_sender, search_id, lines);
                    }
                    (workers, outcome)
                }));
                match outcome {
                    Ok((workers, outcome)) => {
                        let _ = event_sender.send(CoordinatorEvent::SearchFinished {
                            search_id,
                            search: FinishedSearch::AlphaBeta {
                                workers,
                                result: outcome.result,
                                helper_spawn_failure: outcome.helper_spawn_failure,
                                helper_workspace_refusals: outcome.helper_workspace_refusals,
                            },
                        });
                    }
                    Err(_) => {
                        let _ = event_sender.send(CoordinatorEvent::SearchPanicked {
                            search_id,
                            mode: SearchMode::AlphaBeta,
                        });
                    }
                }
            });
            match worker_spawner("janus-alpha-beta", task) {
                Ok(handle) => handle,
                Err(error) => {
                    state.alpha_beta = resources
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take()
                        .expect("a refused alpha-beta spawn retains its worker pool");
                    return Err(error);
                }
            }
        }
        #[cfg(feature = "mcts")]
        SearchMode::Mcts => {
            let searcher = state
                .mcts
                .take()
                .expect("an idle UCI state owns its MCTS searcher");
            let resources = Arc::new(Mutex::new(Some(searcher)));
            let worker_resources = Arc::clone(&resources);
            let move_overhead_ms = state.move_overhead_ms;
            let mcts_tree_nodes = state.mcts_tree_nodes;
            let task = Box::new(move || {
                let mut searcher = worker_resources
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .expect("a started MCTS job owns its searcher");
                let event_sender = sender.clone();
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    let result = run_mcts_job(
                        &mut searcher,
                        &worker_root,
                        &history,
                        &options,
                        root_moves,
                        move_overhead_ms,
                        mcts_tree_nodes,
                        live_clock,
                        &worker_stop,
                        &event_sender,
                        search_id,
                    );
                    (searcher, result)
                }));
                match outcome {
                    Ok((searcher, result)) => {
                        let _ = event_sender.send(CoordinatorEvent::SearchFinished {
                            search_id,
                            search: FinishedSearch::Mcts { searcher, result },
                        });
                    }
                    Err(_) => {
                        let _ = event_sender.send(CoordinatorEvent::SearchPanicked {
                            search_id,
                            mode: SearchMode::Mcts,
                        });
                    }
                }
            });
            match worker_spawner("janus-mcts", task) {
                Ok(handle) => handle,
                Err(error) => {
                    state.mcts = Some(
                        resources
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .take()
                            .expect("a refused MCTS spawn retains its searcher"),
                    );
                    return Err(error);
                }
            }
        }
    };

    Ok(ActiveSearch {
        id: search_id,
        mode,
        root,
        stop,
        handle,
        show_wdl: state.show_wdl,
        uci_chess960: state.uci_chess960,
        defer_until_stop,
        release_requested: false,
        ponder,
        completion: None,
    })
}

/// Used for running MCTS with resolved limits and sending periodic and
/// final snapshots.
///
/// Unbounded and ponder jobs report progress roughly every
/// [`MCTS_INFO_INTERVAL_MS`] milliseconds; every job sends one final
/// snapshot before returning.
///
/// # Arguments
///
/// * `searcher` - MCTS searcher borrowed from the engine state
/// * `root` - search root position
/// * `history` - repetition keys preceding the root
/// * `options` - parsed `go` fields
/// * `root_moves` - optional validated `searchmoves` restriction
/// * `move_overhead_ms` - configured clock overhead in milliseconds
/// * `max_tree_nodes` - configured arena-node ceiling including the root
/// * `live_clock` - shared clock awaiting `ponderhit` activation, if any
/// * `stop` - cancellation token shared with the coordinator
/// * `sender` - coordinator channel receiving snapshots
/// * `search_id` - identifier echoed by every event of this job
///
/// # Returns
///
/// Final complete search snapshot.
#[cfg(feature = "mcts")]
#[allow(clippy::too_many_arguments)]
fn run_mcts_job(
    searcher: &mut Mcts<JanusEvaluator>,
    root: &Position,
    history: &[u64],
    options: &GoOptions,
    root_moves: Option<Vec<Move>>,
    move_overhead_ms: u64,
    max_tree_nodes: usize,
    live_clock: Option<Arc<SearchClock>>,
    stop: &AtomicBool,
    sender: &Sender<CoordinatorEvent>,
    search_id: u64,
) -> MctsResult {
    let started = Instant::now();
    let unbounded = is_unbounded_uci_search(options);
    let mut limits =
        mcts_limits(options, root, move_overhead_ms).with_tree_node_limit(max_tree_nodes);
    if let Some(root_moves) = root_moves {
        limits = limits.with_root_moves(root_moves);
    }
    if let Some(clock) = live_clock {
        limits = limits.with_live_clock(clock);
    }
    let result = if unbounded || options.ponder {
        searcher.search_with_progress(
            root,
            history,
            limits,
            stop,
            Duration::from_millis(MCTS_INFO_INTERVAL_MS),
            |result| {
                let _ = sender.send(CoordinatorEvent::MctsInfo {
                    search_id,
                    result: result.clone(),
                    elapsed: started.elapsed(),
                });
            },
        )
    } else {
        searcher.search_with_history_limits_and_stop(root, history, limits, stop)
    };
    let _ = sender.send(CoordinatorEvent::MctsInfo {
        search_id,
        result: result.clone(),
        elapsed: started.elapsed(),
    });
    result
}

/// Used for reclaiming a completed job, restoring its searcher, and
/// emitting one best move.
///
/// Alpha-beta failures produce a diagnostic and a null move; MCTS
/// completion additionally reports any runtime BT4 backend change.
///
/// # Arguments
///
/// * `state` - engine state receiving the returned searcher
/// * `active` - metadata of the finished job
/// * `search` - algorithm-specific result and searcher
/// * `output` - sink receiving `bestmove` and diagnostics
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn finish_search<W: Write>(
    state: &mut EngineState,
    active: ActiveSearch,
    search: FinishedSearch,
    output: &mut W,
) -> io::Result<()> {
    let _ = active.handle.join();
    match search {
        FinishedSearch::AlphaBeta {
            workers,
            result,
            helper_spawn_failure,
            helper_workspace_refusals,
        } => {
            debug_assert_eq!(active.mode, SearchMode::AlphaBeta);
            state.alpha_beta = workers;
            if let Some(failure) = helper_spawn_failure {
                writeln!(
                    output,
                    "info string helper {} could not start ({}); continuing with {} helpers",
                    failure.helper_index + 1,
                    failure.error,
                    failure.admitted_helpers
                )?;
            }
            if helper_workspace_refusals > 0 {
                writeln!(
                    output,
                    "info string {helper_workspace_refusals} alpha-beta helpers could not admit search workspace; continuing with the authoritative main result"
                )?;
            }
            match result {
                Ok(lines) => {
                    if let Some(result) = lines.first() {
                        write_bestmove(output, result, &active.root, active.uci_chess960)?;
                    } else {
                        writeln!(output, "bestmove 0000")?;
                    }
                }
                Err(error) => {
                    writeln!(output, "info string search failed: {error}")?;
                    writeln!(output, "bestmove 0000")?;
                }
            }
        }
        #[cfg(feature = "mcts")]
        FinishedSearch::Mcts { searcher, result } => {
            debug_assert_eq!(active.mode, SearchMode::Mcts);
            state.mcts = Some(searcher);
            report_bt4_backend_change(state, output)?;
            if result.tree_capacity_exhausted {
                writeln!(
                    output,
                    "info string MCTS tree capacity exhausted at {} nodes",
                    result.tree_nodes
                )?;
            }
            write_mcts_bestmove(output, &result, &active.root, active.uci_chess960)?;
        }
    }
    Ok(())
}

/// Used for rebuilding algorithm state after a caught panic and emitting a
/// null best move.
///
/// # Arguments
///
/// * `state` - engine state whose searcher is rebuilt
/// * `active` - metadata of the panicked job
/// * `mode` - algorithm whose state must be rebuilt
/// * `output` - sink receiving the diagnostic and `bestmove 0000`
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn finish_panicked_search<W: Write>(
    state: &mut EngineState,
    active: ActiveSearch,
    mode: SearchMode,
    output: &mut W,
) -> io::Result<()> {
    let _ = active.handle.join();
    debug_assert_eq!(active.mode, mode);
    match mode {
        SearchMode::AlphaBeta => {
            state.rebuild_alpha_beta();
            write_alpha_beta_admission(&state.alpha_beta, state.hash_mb, output)?;
        }
        #[cfg(feature = "mcts")]
        SearchMode::Mcts => state.rebuild_mcts(),
    }
    writeln!(output, "info string search worker panicked")?;
    writeln!(output, "bestmove 0000")
}

/// Used for detecting and loading a supported evaluator model from its
/// binary header.
///
/// Backend and device selection are consulted only for BT4 files; all other
/// evaluator families preserve their format-specific CPU implementation.
///
/// # Arguments
///
/// * `path` - file-system path of the model
/// * `bt4_backend` - requested BT4 execution backend
/// * `bt4_device` - requested zero-based BT4 device ordinal
///
/// # Returns
///
/// Loaded evaluator matching the detected header.
///
/// # Errors
///
/// Returns a formatted message when the file cannot be opened or read, when
/// the header is unsupported, or when the format-specific loader fails.
fn load_evaluator(
    path: &str,
    bt4_backend: Bt4Backend,
    bt4_device: usize,
) -> Result<JanusEvaluator, String> {
    let mut file = std::fs::File::open(path).map_err(|error| format!("{path}: {error}"))?;
    let mut header = [0_u8; 4];
    file.read_exact(&mut header)
        .map_err(|error| format!("{path}: cannot read model header: {error}"))?;
    if UpstreamNnue::has_upstream_header(&header) {
        return UpstreamNnue::load(path)
            .map(|network| {
                JanusEvaluator::UpstreamNnue(UpstreamNnueEvaluator::new(Arc::new(network)))
            })
            .map_err(|error| error.to_string());
    }
    if &header == b"LC0J" {
        return CnnModel::load(path)
            .map(|network| JanusEvaluator::Cnn(Arc::new(Mutex::new(network))))
            .map_err(|error| error.to_string());
    }
    if &header == b"BT4J" {
        return Bt4Network::load_with_backend(path, bt4_backend, bt4_device)
            .map(|network| JanusEvaluator::Bt4(Arc::new(Mutex::new(network))))
            .map_err(|error| error.to_string());
    }
    if &header == b"JRBT" {
        return JrbtNetwork::load(path)
            .map(|network| JanusEvaluator::Jrbt(Arc::new(Mutex::new(network))))
            .map_err(|error| error.to_string());
    }
    if &header == b"NNUE" {
        return CompactNnue::load(path)
            .map(|network| JanusEvaluator::CompactNnue(Arc::new(network)))
            .map_err(|error| error.to_string());
    }
    Err(format!("{path}: unsupported model header"))
}

/// Used for building a bounded worker pool with one coherent aggregate TT
/// allocation.
///
/// The one-thread control retains the non-atomic local table exactly. Two or
/// more workers clone one [`Arc`] around the full configured Hash allocation;
/// only worker-local evaluator and search heuristics remain independent.
///
/// # Arguments
///
/// * `evaluator` - evaluator cloned into every worker
/// * `threads` - requested worker count, clamped to
///   `1..=MAX_SEARCH_THREADS`
/// * `hash_mb` - aggregate table budget in mebibytes, clamped to
///   `1..=MAX_HASH_MB`
///
/// # Returns
///
/// Worker pool sized to the clamped thread count, or empty when the minimum
/// table or worker-vector allocation cannot be admitted.
fn alpha_beta_workers(
    evaluator: &JanusEvaluator,
    threads: usize,
    hash_mb: usize,
) -> Vec<AlphaBeta<JanusEvaluator>> {
    alpha_beta_workers_with_allocators(
        evaluator,
        threads,
        hash_mb,
        TranspositionTable::try_new,
        SharedTranspositionTable::try_new,
    )
}

/// Used for building a worker pool through injectable fallible table
/// allocators.
///
/// Production supplies both table types' `try_new` constructors. Tests can
/// refuse every size deterministically and verify that the pool becomes empty
/// instead of reaching an infallible minimum-table retry.
///
/// # Arguments
///
/// * `evaluator` - evaluator cloned into admitted workers
/// * `threads` - requested worker count
/// * `hash_mb` - requested aggregate table budget
/// * `local_allocator` - fallible one-thread table constructor
/// * `shared_allocator` - fallible shared-table constructor
///
/// # Returns
///
/// Admitted pool, or an empty pool when even the minimum table or worker-vector
/// allocation fails.
fn alpha_beta_workers_with_allocators<L, S>(
    evaluator: &JanusEvaluator,
    threads: usize,
    hash_mb: usize,
    mut local_allocator: L,
    mut shared_allocator: S,
) -> Vec<AlphaBeta<JanusEvaluator>>
where
    L: FnMut(usize) -> Option<TranspositionTable>,
    S: FnMut(usize) -> Option<SharedTranspositionTable>,
{
    let threads = threads.clamp(1, MAX_SEARCH_THREADS);
    // The evaluation cache is per worker, so its size is chosen from the
    // thread count before any worker is built.
    set_eval_cache_size(eval_cache_size_for_threads(threads));
    let entry_budget = hash_mb
        .clamp(1, MAX_HASH_MB)
        .saturating_mul(TT_ENTRIES_PER_MB);
    let requested = floor_power_of_two(entry_budget);
    if threads == 1 {
        let mut entries = requested;
        loop {
            if let Some(table) = local_allocator(entries) {
                let mut workers = Vec::new();
                if workers.try_reserve_exact(1).is_err() {
                    return workers;
                }
                workers.push(AlphaBeta::with_table(evaluator.clone(), table));
                prepare_workers(&mut workers);
                return workers;
            }
            match shrink_entries(entries) {
                Some(smaller) => entries = smaller,
                None => return Vec::new(),
            }
        }
    }
    let mut entries = requested;
    loop {
        let table = match shared_allocator(entries) {
            Some(table) => Arc::new(table),
            None => match shrink_entries(entries) {
                Some(smaller) => {
                    entries = smaller;
                    continue;
                }
                None => return Vec::new(),
            },
        };
        let mut workers = Vec::new();
        if workers.try_reserve_exact(threads).is_err() {
            return workers;
        }
        for _ in 0..threads {
            workers.push(AlphaBeta::with_shared_tt(
                evaluator.clone(),
                Arc::clone(&table),
            ));
        }
        prepare_workers(&mut workers);
        return workers;
    }
}

/// Used for reporting one completed alpha-beta pool admission through the
/// coordinator-owned output sink.
///
/// Full-size admission is silent. A smaller usable table produces one bounded
/// summary, while an empty pool reports that alpha-beta search is disabled.
/// The allocation helpers themselves perform no process-global I/O.
///
/// # Arguments
///
/// * `workers` - admitted worker pool, possibly empty after resource refusal
/// * `hash_mb` - configured aggregate hash request in mebibytes
/// * `output` - coordinator sink receiving at most one diagnostic
///
/// # Errors
///
/// Returns the sink's [`io::Error`] without panicking when the diagnostic
/// cannot be written.
fn write_alpha_beta_admission<W: Write>(
    workers: &[AlphaBeta<JanusEvaluator>],
    hash_mb: usize,
    output: &mut W,
) -> io::Result<()> {
    let requested = floor_power_of_two(
        hash_mb
            .clamp(1, MAX_HASH_MB)
            .saturating_mul(TT_ENTRIES_PER_MB),
    );
    let Some(first) = workers.first() else {
        return writeln!(
            output,
            "info string hash or alpha-beta worker allocation failed at {} MiB including the minimum table; search pool disabled",
            requested / TT_ENTRIES_PER_MB
        );
    };
    let admitted = first.transposition_entries();
    if admitted < requested {
        writeln!(
            output,
            "info string hash allocation of {} MiB failed; using {} MiB",
            requested / TT_ENTRIES_PER_MB,
            admitted / TT_ENTRIES_PER_MB
        )?;
    }
    Ok(())
}

/// Used for halving a failed transposition request.
///
/// A competition host may configure a hash the machine cannot supply. Halving
/// and retrying keeps the engine playing on a smaller table instead of
/// aborting the process. The coordinator reports the final admission once the
/// pure allocation loop returns.
///
/// # Arguments
///
/// * `entries` - bucket count that just failed to allocate
/// # Returns
///
/// `Some(next)` when a smaller power-of-two attempt remains, `None` at the
/// floor.
fn shrink_entries(entries: usize) -> Option<usize> {
    let next = entries / 2;
    if next < MIN_TT_ENTRIES {
        return None;
    }
    Some(next)
}

/// Used for clearing an alpha-beta pool exactly once per physical table
/// allocation.
///
/// With a shared table only the first worker clears it; local tables are
/// cleared per worker.
///
/// # Arguments
///
/// * `workers` - pool whose tables are cleared
fn clear_alpha_beta_workers(workers: &mut [AlphaBeta<JanusEvaluator>]) {
    let Some((first, rest)) = workers.split_first_mut() else {
        return;
    };
    let shared = first.uses_shared_transposition_table();
    first.clear();
    if !shared {
        for worker in rest {
            worker.clear();
        }
    }
}

/// Used for retrieving the greatest power of two no larger than a positive
/// value.
///
/// Zero is treated as one before rounding.
///
/// # Arguments
///
/// * `value` - value to round down
///
/// # Returns
///
/// Largest power of two less than or equal to `max(value, 1)`.
fn floor_power_of_two(value: usize) -> usize {
    let shift = usize::BITS - 1 - value.max(1).leading_zeros();
    1_usize << shift
}

/// Used for partitioning an exact node budget across workers using
/// remainder distribution.
///
/// Workers with an index below the remainder receive one extra node so the
/// per-worker budgets sum exactly to the requested total. Limits without a
/// node budget, or single-worker pools, pass through unchanged.
///
/// # Arguments
///
/// * `limits` - limits carrying the aggregate node budget
/// * `worker` - zero-based worker index
/// * `workers` - total worker count
///
/// # Returns
///
/// Limits with this worker's share of the node budget.
///
/// # Panics
///
/// Panics if a worker index or count does not fit `u64`.
fn per_worker_limits(mut limits: SearchLimits, worker: usize, workers: usize) -> SearchLimits {
    if limits.max_nodes > 0 && workers > 1 {
        let worker = u64::try_from(worker).expect("bounded worker index fits u64");
        let workers = u64::try_from(workers).expect("bounded worker count fits u64");
        let base = limits.max_nodes / workers;
        let remainder = limits.max_nodes % workers;
        limits.max_nodes = base + u64::from(worker < remainder);
    }
    limits
}

/// Split of an active worker pool into its main worker, its helpers, and the
/// active worker count.
type ActivePool<'pool> = (
    &'pool mut AlphaBeta<JanusEvaluator>,
    &'pool mut [AlphaBeta<JanusEvaluator>],
    usize,
);

/// First operating-system refusal encountered while admitting alpha-beta
/// helper threads for one search.
///
/// The value travels with the completed search so only the UCI coordinator
/// writes its diagnostic to the long-lived stdout lock.
struct HelperSpawnFailure {
    /// Used for identifying the zero-based helper whose spawn was refused.
    helper_index: usize,
    /// Used for reporting how many earlier helpers were successfully started.
    admitted_helpers: usize,
    /// Used for preserving the operating-system admission diagnostic.
    error: io::Error,
}

/// Search result paired with at most one bounded helper-admission diagnostic.
struct ParallelSearchOutcome<T> {
    /// Used for carrying the authoritative search result or search-start
    /// failure.
    result: Result<T, SearchError>,
    /// Used for retaining only the first helper spawn refusal.
    helper_spawn_failure: Option<HelperSpawnFailure>,
    /// Used for counting started helpers whose per-search workspace was
    /// refused.
    helper_workspace_refusals: usize,
}

/// Aggregate telemetry and first fatal error from joined helper searches.
#[derive(Default)]
struct HelperSearchSummary {
    /// Used for retaining the first non-resource helper error in worker order.
    fatal_error: Option<SearchError>,
    /// Used for summing nodes from successfully completed helpers.
    nodes: u64,
    /// Used for summing tablebase hits from successfully completed helpers.
    tb_hits: u64,
    /// Used for counting helpers whose workspace admission failed.
    workspace_refusals: usize,
}

/// Used for selecting the active workers and advancing their shared
/// generation before a parallel search.
///
/// A node-limited search never needs more workers than nodes, and a shared
/// table's generation must be advanced exactly once per search by one pool
/// member.
///
/// # Arguments
///
/// * `workers` - configured worker pool
/// * `limits` - search limits supplying any node cap
///
/// # Returns
///
/// `Some((main, helpers, count))` when at least one worker is active, `None`
/// when the pool is empty.
fn prepare_worker_pool<'pool>(
    workers: &'pool mut [AlphaBeta<JanusEvaluator>],
    limits: &SearchLimits,
) -> Option<ActivePool<'pool>> {
    let active_workers = if limits.max_nodes == 0 {
        workers.len()
    } else {
        workers
            .len()
            .min(usize::try_from(limits.max_nodes).unwrap_or(usize::MAX))
    };
    let (active, _) = workers.split_at_mut(active_workers);
    let worker_count = active.len();
    if active
        .first()
        .is_some_and(AlphaBeta::uses_shared_transposition_table)
    {
        active[0]
            .advance_shared_generation()
            .expect("configured SMP workers share one transposition table");
    }
    let (main, helpers) = active.split_first_mut()?;
    Some((main, helpers, worker_count))
}

/// Used for recording a helper thread that started, or reporting one that
/// could not.
///
/// A competition host can refuse further threads once the engine is configured
/// near its `Threads` ceiling. Continuing with the helpers already running is
/// far better than unwinding out of the scope, which would abandon the move and
/// forfeit the game, so a refusal degrades the pool instead of failing.
///
/// # Arguments
///
/// * `handles` - accumulated helper handles for this search
/// * `spawned` - result of the scoped spawn attempt
/// * `index` - zero-based helper index, for the report
///
/// # Returns
///
/// `None` when the helper started. A populated failure identifies the refused
/// helper and the number already admitted, after which the pool must stop
/// growing.
fn push_helper_handle<'scope, T>(
    handles: &mut Vec<thread::ScopedJoinHandle<'scope, T>>,
    spawned: std::io::Result<thread::ScopedJoinHandle<'scope, T>>,
    index: usize,
) -> Option<HelperSpawnFailure> {
    match spawned {
        Ok(handle) => {
            handles.push(handle);
            None
        }
        Err(error) => Some(HelperSpawnFailure {
            helper_index: index,
            admitted_helpers: handles.len(),
            error,
        }),
    }
}

/// Used for classifying a completed helper's search error without allowing
/// optional workspace pressure to discard the authoritative main result.
///
/// # Arguments
///
/// * `error` - helper-local search failure
/// * `workspace_refusals` - saturating count of resource-only helper failures
/// * `fatal_error` - first non-resource helper failure, retained in worker
///   order
fn record_helper_search_error(
    error: SearchError,
    workspace_refusals: &mut usize,
    fatal_error: &mut Option<SearchError>,
) {
    if error == SearchError::ResourceExhausted {
        *workspace_refusals = workspace_refusals.saturating_add(1);
    } else if fatal_error.is_none() {
        *fatal_error = Some(error);
    }
}

/// Used for joining every admitted helper without allocating a result vector.
///
/// Resource-only failures are counted; the first other error is retained, and
/// telemetry from successful helpers is accumulated in worker order.
///
/// # Arguments
///
/// * `handles` - admitted scoped helpers in deterministic worker order
/// * `telemetry` - projection from one successful helper result to its shared
///   node and tablebase-hit counts
///
/// # Returns
///
/// Bounded aggregate helper status.
///
/// # Panics
///
/// Re-raises any panic produced by a helper thread.
fn join_helper_searches<T, F>(
    handles: Vec<thread::ScopedJoinHandle<'_, Result<T, SearchError>>>,
    telemetry: F,
) -> HelperSearchSummary
where
    F: Fn(&T) -> (u64, u64),
{
    let mut summary = HelperSearchSummary::default();
    for handle in handles {
        let result = match handle.join() {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        };
        match result {
            Ok(helper) => {
                let (nodes, tb_hits) = telemetry(&helper);
                summary.nodes = summary.nodes.saturating_add(nodes);
                summary.tb_hits = summary.tb_hits.saturating_add(tb_hits);
            }
            Err(error) => record_helper_search_error(
                error,
                &mut summary.workspace_refusals,
                &mut summary.fatal_error,
            ),
        }
    }
    summary
}

/// Used for running one authoritative main search plus shared-table
/// diversified helpers.
///
/// Helpers exchange bounds and ordering moves through one atomic table, but
/// never replace the main worker's move, score, depth, or PV. Reported nodes
/// aggregate every successfully completed worker. A helper-only workspace
/// refusal is counted in the outcome and leaves the main result valid. With
/// one worker this returns the exact local reference search unchanged.
///
/// # Arguments
///
/// * `workers` - pool whose first worker is the authoritative main
/// * `position` - search root position
/// * `history` - repetition keys preceding the root
/// * `limits` - resolved aggregate search limits
/// * `stop` - cancellation token shared with the coordinator
/// * `root_moves` - optional validated `searchmoves` restriction
/// * `observer` - callback receiving completed-depth progress lines
///
/// # Returns
///
/// Main worker's result with aggregate telemetry and bounded helper-admission
/// metadata.
///
/// # Errors
///
/// The outcome contains [`SearchError::InvalidLimits`] when no worker is
/// active, a main-worker error, or the first non-resource helper error.
///
/// # Panics
///
/// Panics if advancing the shared table generation fails despite a shared
/// configuration, and re-raises any helper thread panic.
fn search_alpha_beta_parallel<F>(
    workers: &mut [AlphaBeta<JanusEvaluator>],
    position: &Position,
    history: &[u64],
    limits: &SearchLimits,
    stop: &AtomicBool,
    root_moves: Option<&[Move]>,
    mut observer: F,
) -> ParallelSearchOutcome<SearchResult>
where
    F: FnMut(&SearchInfo),
{
    let Some((main, helpers, worker_count)) = prepare_worker_pool(workers, limits) else {
        return ParallelSearchOutcome {
            result: Err(SearchError::InvalidLimits),
            helper_spawn_failure: None,
            helper_workspace_refusals: 0,
        };
    };
    let started = Instant::now();
    let main_limits = per_worker_limits(limits.clone(), 0, worker_count);

    let (main_result, helper_summary, helper_spawn_failure) = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(helpers.len());
        let mut helper_spawn_failure = None;
        for (index, helper) in helpers.iter_mut().enumerate() {
            let worker_stop = stop;
            let helper_limits = per_worker_limits(limits.clone(), index + 1, worker_count);
            let spawned = thread::Builder::new().spawn_scoped(scope, move || {
                if let Some(root_moves) = root_moves {
                    helper.search_root_moves_as_helper_with_history(
                        position,
                        history,
                        helper_limits,
                        root_moves,
                        worker_stop,
                        index + 1,
                    )
                } else {
                    helper.search_as_helper_with_history(
                        position,
                        history,
                        helper_limits,
                        worker_stop,
                        index + 1,
                    )
                }
            });
            if let Some(failure) = push_helper_handle(&mut handles, spawned, index) {
                helper_spawn_failure = Some(failure);
                break;
            }
        }

        let main_result = if let Some(root_moves) = root_moves {
            main.search_root_moves_with_history_and_observer(
                position,
                history,
                main_limits,
                root_moves,
                stop,
                |info| observer(info),
            )
        } else {
            main.search_with_history_and_observer(position, history, main_limits, stop, |info| {
                observer(info);
            })
        };

        let helper_summary = join_helper_searches(handles, |helper| (helper.nodes, helper.tb_hits));
        (main_result, helper_summary, helper_spawn_failure)
    });

    let result = match (main_result, helper_summary.fatal_error) {
        (Err(error), _) | (_, Some(error)) => Err(error),
        (Ok(main_result), None) if worker_count == 1 => Ok(main_result),
        (Ok(main_result), None) => {
            let mut selected = main_result;
            selected.nodes = selected.nodes.saturating_add(helper_summary.nodes);
            selected.tb_hits = selected.tb_hits.saturating_add(helper_summary.tb_hits);
            selected.elapsed = started.elapsed();
            selected.hashfull_per_mille = main.current_hashfull_per_mille();
            observer(&SearchInfo {
                depth: selected.depth,
                selective_depth: selected.selective_depth,
                score: selected.score,
                nodes: selected.nodes,
                elapsed: selected.elapsed,
                hashfull_per_mille: selected.hashfull_per_mille,
                tb_hits: selected.tb_hits,
                principal_variation: selected.principal_variation.clone(),
            });
            Ok(selected)
        }
    };
    ParallelSearchOutcome {
        result,
        helper_spawn_failure,
        helper_workspace_refusals: helper_summary.workspace_refusals,
    }
}

/// Used for running authoritative ranked `MultiPV` batches with
/// shared-table helpers.
///
/// The main worker's complete batch remains authoritative. Helpers diversify
/// the table contents only; their node counts are included in telemetry.
///
/// # Arguments
///
/// * `workers` - pool whose first worker is the authoritative main
/// * `position` - search root position
/// * `history` - repetition keys preceding the root
/// * `limits` - resolved aggregate search limits
/// * `stop` - cancellation token shared with the coordinator
/// * `root_moves` - optional validated `searchmoves` restriction
/// * `multi_pv` - requested number of ranked lines
/// * `observer` - callback receiving complete ranked batches
///
/// # Returns
///
/// Main worker's ranked batch with aggregate telemetry and bounded
/// helper-admission metadata.
///
/// # Errors
///
/// The outcome contains [`SearchError::InvalidLimits`] when no worker is
/// active, a main-worker error, or the first non-resource helper error.
///
/// # Panics
///
/// Panics if advancing the shared table generation fails despite a shared
/// configuration, and re-raises any helper thread panic.
#[allow(clippy::too_many_arguments)]
fn search_alpha_beta_parallel_multi<F>(
    workers: &mut [AlphaBeta<JanusEvaluator>],
    position: &Position,
    history: &[u64],
    limits: &SearchLimits,
    stop: &AtomicBool,
    root_moves: Option<&[Move]>,
    multi_pv: usize,
    mut observer: F,
) -> ParallelSearchOutcome<Vec<SearchResult>>
where
    F: FnMut(&[SearchInfo]),
{
    let Some((main, helpers, worker_count)) = prepare_worker_pool(workers, limits) else {
        return ParallelSearchOutcome {
            result: Err(SearchError::InvalidLimits),
            helper_spawn_failure: None,
            helper_workspace_refusals: 0,
        };
    };
    let started = Instant::now();
    let main_limits = per_worker_limits(limits.clone(), 0, worker_count);

    let (main_result, helper_summary, helper_spawn_failure) = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(helpers.len());
        let mut helper_spawn_failure = None;
        for (index, helper) in helpers.iter_mut().enumerate() {
            let helper_limits = per_worker_limits(limits.clone(), index + 1, worker_count);
            let spawned = thread::Builder::new().spawn_scoped(scope, move || {
                helper.search_multi_pv_as_helper_with_history(
                    position,
                    history,
                    helper_limits,
                    root_moves,
                    multi_pv,
                    stop,
                    index + 1,
                )
            });
            if let Some(failure) = push_helper_handle(&mut handles, spawned, index) {
                helper_spawn_failure = Some(failure);
                break;
            }
        }

        let main_result = main.search_multi_pv_with_history_and_observer(
            position,
            history,
            main_limits,
            root_moves,
            multi_pv,
            stop,
            |infos| observer(infos),
        );
        let helper_summary = join_helper_searches(handles, |helper| {
            (search_batch_nodes(helper), search_batch_tb_hits(helper))
        });
        (main_result, helper_summary, helper_spawn_failure)
    });

    let result = match (main_result, helper_summary.fatal_error) {
        (Err(error), _) | (_, Some(error)) => Err(error),
        (Ok(main_result), None) if worker_count == 1 => Ok(main_result),
        (Ok(main_result), None) => {
            let aggregate_nodes =
                search_batch_nodes(&main_result).saturating_add(helper_summary.nodes);
            let aggregate_tb_hits =
                search_batch_tb_hits(&main_result).saturating_add(helper_summary.tb_hits);
            let main_stopped = main_result.first().is_some_and(|line| line.stopped);
            let mut selected = main_result;
            let elapsed = started.elapsed();
            let hashfull = main.current_hashfull_per_mille();
            for line in &mut selected {
                line.nodes = aggregate_nodes;
                line.tb_hits = aggregate_tb_hits;
                line.elapsed = elapsed;
                line.stopped = main_stopped;
                line.hashfull_per_mille = hashfull;
            }
            let infos: Vec<SearchInfo> = selected.iter().map(search_info_from_result).collect();
            observer(&infos);
            Ok(selected)
        }
    };
    ParallelSearchOutcome {
        result,
        helper_spawn_failure,
        helper_workspace_refusals: helper_summary.workspace_refusals,
    }
}

/// Used for retrieving the shared node count carried by a complete
/// `MultiPV` batch.
///
/// # Arguments
///
/// * `lines` - ranked batch whose first line carries the shared count
///
/// # Returns
///
/// First line's node count, or `0` for an empty batch.
fn search_batch_nodes(lines: &[SearchResult]) -> u64 {
    lines.first().map_or(0, |line| line.nodes)
}

/// Used for retrieving the shared tablebase-hit count carried by a complete
/// `MultiPV` batch.
///
/// # Arguments
///
/// * `lines` - ranked batch whose first line carries the shared count
///
/// # Returns
///
/// First line's tablebase-hit count, or `0` for an empty batch.
fn search_batch_tb_hits(lines: &[SearchResult]) -> u64 {
    lines.first().map_or(0, |line| line.tb_hits)
}

/// Used for converting a final alpha-beta result into a protocol progress
/// snapshot.
///
/// # Arguments
///
/// * `result` - final ranked search result
///
/// # Returns
///
/// Snapshot carrying the result's depth, score, telemetry, and PV.
fn search_info_from_result(result: &SearchResult) -> SearchInfo {
    SearchInfo {
        depth: result.depth,
        selective_depth: result.selective_depth,
        score: result.score,
        nodes: result.nodes,
        elapsed: result.elapsed,
        hashfull_per_mille: result.hashfull_per_mille,
        tb_hits: result.tb_hits,
        principal_variation: result.principal_variation.clone(),
    }
}

/// Used for sending a complete `MultiPV` batch as ordered one-based
/// coordinator events.
///
/// # Arguments
///
/// * `sender` - coordinator channel receiving the events
/// * `search_id` - identifier echoed by every event
/// * `infos` - ranked snapshots in `MultiPV` order
fn send_alpha_beta_infos(sender: &Sender<CoordinatorEvent>, search_id: u64, infos: &[SearchInfo]) {
    for (index, info) in infos.iter().enumerate() {
        let _ = sender.send(CoordinatorEvent::AlphaBetaInfo {
            search_id,
            multipv: index + 1,
            info: info.clone(),
        });
    }
}

/// Used for emitting authoritative final info lines before the matching
/// best-move event.
///
/// # Arguments
///
/// * `sender` - coordinator channel receiving the events
/// * `search_id` - identifier echoed by every event
/// * `lines` - final ranked search results
fn send_final_alpha_beta_infos(
    sender: &Sender<CoordinatorEvent>,
    search_id: u64,
    lines: &[SearchResult],
) {
    let infos: Vec<SearchInfo> = lines.iter().map(search_info_from_result).collect();
    send_alpha_beta_infos(sender, search_id, &infos);
}

/// Used for parsing common textual and numeric forms of a UCI check option.
///
/// Accepts `true`/`false`, `1`/`0`, and `on`/`off`, matching text
/// case-insensitively.
///
/// # Arguments
///
/// * `value` - check option value
///
/// # Returns
///
/// Parsed flag, or [`None`] for an unrecognized value.
fn parse_check_option(value: &str) -> Option<bool> {
    if value.eq_ignore_ascii_case("true") || value == "1" || value.eq_ignore_ascii_case("on") {
        Some(true)
    } else if value.eq_ignore_ascii_case("false")
        || value == "0"
        || value.eq_ignore_ascii_case("off")
    {
        Some(false)
    } else {
        None
    }
}

/// Used for converting a compact floating-point NNUE score using the Java
/// rounding contract.
///
/// NaN maps to zero; other values add one half and floor, saturating at the
/// `i32` bounds through the float-to-int cast.
///
/// # Arguments
///
/// * `value` - raw floating-point score from the compact network
///
/// # Returns
///
/// Rounded integer score.
#[allow(clippy::cast_possible_truncation)]
fn compact_score(value: f32) -> i32 {
    if value.is_nan() {
        0
    } else {
        // Float-to-int casts saturate at the integer bounds. The explicit
        // floor preserves the compact Java NNUE rounding contract.
        (value + 0.5).floor() as i32
    }
}

/// Used for checking whether UCI requires an explicit `stop` before
/// releasing a move.
///
/// A bare `go` has no resource field and is therefore equivalent to
/// `go infinite`. Root filtering and clock increments alone do not bound work.
///
/// # Arguments
///
/// * `options` - parsed `go` fields
///
/// # Returns
///
/// `true` for `go infinite` or a `go` without any bounding resource field.
fn is_unbounded_uci_search(options: &GoOptions) -> bool {
    options.infinite
        || (options.depth.is_none()
            && options.nodes.is_none()
            && options.move_time_ms.is_none()
            && options.white_time_ms.is_none()
            && options.black_time_ms.is_none()
            && options.mate.is_none())
}

/// Used for building the coordinator half of one accepted `go ponder`
/// transition.
///
/// The options are re-examined with `ponder` cleared to decide whether the
/// search remains bounded after `ponderhit`, and clock budgets become a
/// pending [`PonderDeadline`] when the fields provide one.
///
/// # Arguments
///
/// * `options` - parsed `go` fields including the ponder flag
/// * `position` - search root deciding whose clock applies
/// * `move_overhead_ms` - configured clock overhead in milliseconds
/// * `mode` - algorithm the ponder search runs
///
/// # Returns
///
/// Transition state consumed by the first matching `ponderhit`.
fn ponder_state(
    options: &GoOptions,
    position: &Position,
    move_overhead_ms: u64,
    mode: SearchMode,
) -> PonderState {
    let mut live_options = options.clone();
    live_options.ponder = false;
    let bounded_after_hit = !is_unbounded_uci_search(&live_options);
    let deadline = ponder_deadline_budgets(options, position, move_overhead_ms, mode).map(
        |(soft_time, hard_time)| PonderDeadline {
            clock: Arc::new(SearchClock::new()),
            soft_time,
            hard_time,
        },
    );
    PonderState {
        deadline,
        bounded_after_hit,
    }
}

/// Used for resolving a ponder search's clock budget without starting it
/// prematurely.
///
/// Alpha-beta retains its paired soft/hard allocation. MCTS preserves the
/// existing conservative contract that treats the ordinary soft allocation as
/// its hard admission deadline between complete neural predictions.
///
/// # Arguments
///
/// * `options` - parsed `go` fields
/// * `position` - search root deciding whose clock applies
/// * `move_overhead_ms` - configured clock overhead in milliseconds
/// * `mode` - algorithm the ponder search runs
///
/// # Returns
///
/// `(soft, hard)` budget pair, or [`None`] when the fields provide no
/// deadline.
fn ponder_deadline_budgets(
    options: &GoOptions,
    position: &Position,
    move_overhead_ms: u64,
    mode: SearchMode,
) -> Option<(Option<Duration>, Duration)> {
    if options.infinite {
        return None;
    }
    if let Some(milliseconds) = options.move_time_ms {
        return Some((None, Duration::from_millis(milliseconds.max(1))));
    }
    let (remaining, increment) = match position.side_to_move() {
        janus_core::Color::White => (options.white_time_ms, options.white_increment_ms),
        janus_core::Color::Black => (options.black_time_ms, options.black_increment_ms),
    };
    let remaining = remaining?;
    let (soft, hard) = clock_budgets(
        remaining,
        increment.unwrap_or(0),
        options
            .moves_to_go
            .unwrap_or(janus_engine::CLOCK_MOVES_TO_GO),
    );
    let overhead = Duration::from_millis(move_overhead_ms);
    let adjusted_soft = soft.saturating_sub(overhead);
    match mode {
        #[cfg(feature = "mcts")]
        SearchMode::Mcts => Some((None, adjusted_soft)),
        SearchMode::AlphaBeta => {
            let adjusted_hard = hard.saturating_sub(overhead);
            Some((Some(adjusted_soft.min(adjusted_hard)), adjusted_hard))
        }
    }
}

/// Used for resolving UCI search and clock fields into MCTS resource
/// limits.
///
/// `nodes` maps directly to simulations and `depth` to `depth * 1024`
/// simulations. Infinite and ponder searches stay unbounded in time;
/// otherwise `movetime` supplies the budget unchanged, or the side-to-move
/// clock supplies its soft allocation minus the configured overhead.
///
/// # Arguments
///
/// * `options` - parsed `go` fields
/// * `position` - search root deciding whose clock applies
/// * `move_overhead_ms` - configured clock overhead in milliseconds
///
/// # Returns
///
/// Resolved MCTS limits.
#[cfg(feature = "mcts")]
fn mcts_limits(options: &GoOptions, position: &Position, move_overhead_ms: u64) -> MctsLimits {
    let (remaining, increment) = match position.side_to_move() {
        janus_core::Color::White => (options.white_time_ms, options.white_increment_ms),
        janus_core::Color::Black => (options.black_time_ms, options.black_increment_ms),
    };
    let max_simulations = options.nodes.or_else(|| {
        options
            .depth
            .map(|depth| u64::from(depth).saturating_mul(1_024))
    });
    let mut limits = MctsLimits::simulations(max_simulations.unwrap_or(u64::MAX));

    if options.infinite || options.ponder {
        return limits;
    }
    if let Some(milliseconds) = options.move_time_ms {
        return limits.with_time(Duration::from_millis(milliseconds.max(1)));
    }
    if let Some(remaining) = remaining {
        let (soft, _) = clock_budgets(
            remaining,
            increment.unwrap_or(0),
            options
                .moves_to_go
                .unwrap_or(janus_engine::CLOCK_MOVES_TO_GO),
        );
        limits = limits.with_time(soft.saturating_sub(Duration::from_millis(move_overhead_ms)));
    }
    limits
}

/// Used for resolving UCI search and clock fields into alpha-beta resource
/// limits.
///
/// `mate N` converts to an odd ply depth of `2 * N - 1` and caps any
/// explicit depth. Infinite and ponder searches carry no time budget;
/// otherwise `movetime` supplies a single unadjusted budget, or the
/// side-to-move clock supplies overhead-adjusted soft and hard budgets.
///
/// # Arguments
///
/// * `options` - parsed `go` fields
/// * `position` - search root deciding whose clock applies
/// * `move_overhead_ms` - configured clock overhead in milliseconds
/// * `clock_moves_to_go` - fallback divisor when `go` carries no `movestogo`
///
/// # Returns
///
/// Resolved alpha-beta limits.
fn alpha_beta_limits(
    options: &GoOptions,
    position: &Position,
    move_overhead_ms: u64,
    clock_moves_to_go: u32,
) -> SearchLimits {
    let mate_depth = options
        .mate
        .map(|moves| moves.saturating_mul(2).saturating_sub(1).max(1));
    let depth = options
        .depth
        .or(mate_depth)
        .unwrap_or(MAX_DEPTH)
        .min(mate_depth.unwrap_or(MAX_DEPTH))
        .clamp(1, MAX_DEPTH);
    let mut limits = SearchLimits::depth(depth);
    if let Some(nodes) = options.nodes {
        limits = limits.with_nodes(nodes.max(1));
    }
    if options.infinite || options.ponder {
        return limits;
    }
    if let Some(milliseconds) = options.move_time_ms {
        return limits.with_time(Duration::from_millis(milliseconds.max(1)));
    }

    let (remaining, increment) = match position.side_to_move() {
        janus_core::Color::White => (options.white_time_ms, options.white_increment_ms),
        janus_core::Color::Black => (options.black_time_ms, options.black_increment_ms),
    };
    if let Some(remaining) = remaining {
        // An explicit `movestogo` from the GUI still wins; the option only
        // supplies the fallback for sudden-death-plus-increment controls,
        // which is what every match in this project actually plays.
        let (soft, hard) = clock_budgets(
            remaining,
            increment.unwrap_or(0),
            options.moves_to_go.unwrap_or(clock_moves_to_go),
        );
        let overhead = Duration::from_millis(move_overhead_ms);
        let adjusted_hard = hard.saturating_sub(overhead);
        let adjusted_soft = soft.saturating_sub(overhead).min(adjusted_hard);
        return limits.with_clock(adjusted_soft, adjusted_hard);
    }

    limits
}

/// Used for formatting one alpha-beta UCI `info` line with score, work, and
/// PV fields.
///
/// Mate scores are written as `mate <moves>`; other scores as centipawns
/// with an optional WDL triplet.
///
/// # Arguments
///
/// * `output` - sink receiving the line
/// * `info` - completed-depth snapshot to format
/// * `multipv` - one-based rank of the line
/// * `root` - root position used to format the PV
/// * `uci_chess960` - whether castling uses Chess960 rook-source notation
/// * `show_wdl` - whether the WDL triplet is included
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn write_search_info<W: Write>(
    output: &mut W,
    info: &SearchInfo,
    multipv: usize,
    root: &Position,
    uci_chess960: bool,
    show_wdl: bool,
) -> io::Result<()> {
    write!(
        output,
        "info depth {} seldepth {} multipv {} score ",
        info.depth, info.selective_depth, multipv
    )?;
    if let Some(moves) = mate_moves(info.score) {
        write!(output, "mate {moves}")?;
    } else {
        write!(output, "cp {}", info.score)?;
    }
    if show_wdl {
        let [win, draw, loss] = score_wdl(info.score);
        write!(output, " wdl {win} {draw} {loss}")?;
    }
    let elapsed_ms = duration_millis(info.elapsed);
    let nps = info.nodes.saturating_mul(1_000) / elapsed_ms.max(1);
    write!(
        output,
        " nodes {} nps {} hashfull {} tbhits {} time {}",
        info.nodes, nps, info.hashfull_per_mille, info.tb_hits, elapsed_ms
    )?;
    if !info.principal_variation.is_empty() {
        write!(output, " pv")?;
        write_pv(output, root, &info.principal_variation, uci_chess960)?;
    }
    writeln!(output)
}

/// Used for writing the final alpha-beta move in standard or Chess960 UCI
/// notation.
///
/// A result without a best move produces `bestmove 0000`.
///
/// # Arguments
///
/// * `output` - sink receiving the line
/// * `result` - final search result carrying the move
/// * `root` - root position used to format the move
/// * `uci_chess960` - whether castling uses Chess960 rook-source notation
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn write_bestmove<W: Write>(
    output: &mut W,
    result: &SearchResult,
    root: &Position,
    uci_chess960: bool,
) -> io::Result<()> {
    writeln!(
        output,
        "bestmove {}",
        result.best_move.map_or_else(
            || "0000".to_owned(),
            |mv| { root.uci_move_string(mv, uci_chess960) }
        )
    )
}

/// Used for formatting one internally consistent MCTS snapshot as a UCI
/// `info` line.
///
/// Depth reports the PV length, the score converts the tree value to
/// centipawns, and nodes report completed simulations.
///
/// # Arguments
///
/// * `output` - sink receiving the line
/// * `result` - complete tree snapshot to format
/// * `elapsed` - wall-clock time since the job began
/// * `root` - root position used to format the PV
/// * `uci_chess960` - whether castling uses Chess960 rook-source notation
/// * `show_wdl` - whether the WDL triplet is included
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
#[cfg(feature = "mcts")]
fn write_mcts_info<W: Write>(
    output: &mut W,
    result: &MctsResult,
    elapsed: Duration,
    root: &Position,
    uci_chess960: bool,
    show_wdl: bool,
) -> io::Result<()> {
    let elapsed_ms = duration_millis(elapsed);
    let nps = result.simulations.saturating_mul(1_000) / elapsed_ms.max(1);
    write!(
        output,
        "info depth {} multipv 1 score cp {} nodes {} nps {} hashfull 0 tbhits 0 time {}",
        result.principal_variation.len(),
        value_to_centipawns(result.value),
        result.simulations,
        nps,
        elapsed_ms
    )?;
    if show_wdl {
        let [win, draw, loss] = score_wdl(value_to_centipawns(result.value));
        write!(output, " wdl {win} {draw} {loss}")?;
    }
    if !result.principal_variation.is_empty() {
        write!(output, " pv")?;
        write_pv(output, root, &result.principal_variation, uci_chess960)?;
    }
    writeln!(output)
}

/// Used for writing the final MCTS move in standard or Chess960 UCI
/// notation.
///
/// A snapshot without a best move produces `bestmove 0000`.
///
/// # Arguments
///
/// * `output` - sink receiving the line
/// * `result` - final tree snapshot carrying the move
/// * `root` - root position used to format the move
/// * `uci_chess960` - whether castling uses Chess960 rook-source notation
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
#[cfg(feature = "mcts")]
fn write_mcts_bestmove<W: Write>(
    output: &mut W,
    result: &MctsResult,
    root: &Position,
    uci_chess960: bool,
) -> io::Result<()> {
    // A root drawn by repetition, the fifty-move rule or insufficient material
    // is deliberately not searched -- the engine returns immediately rather
    // than spending budget on it -- so `best_move` is `None` even though the
    // position has legal moves. Emitting `bestmove 0000` there tells the
    // arbiter the engine has no move, which is read as a forfeit. `0000` is
    // only correct when no legal move exists, so fall back to a legal move and
    // let the arbiter adjudicate the draw (BUG-20260803-270).
    let chosen = result
        .best_move
        .or_else(|| root.legal_moves().first().copied());
    writeln!(
        output,
        "bestmove {}",
        chosen.map_or_else(
            || "0000".to_owned(),
            |mv| { root.uci_move_string(mv, uci_chess960) }
        )
    )
}

/// Used for replaying and formatting a legal PV so castling notation uses
/// each ply's state.
///
/// Formatting stops early when a replayed move fails to apply, leaving the
/// already-written prefix intact.
///
/// # Arguments
///
/// * `output` - sink receiving the space-prefixed move list
/// * `root` - position the variation starts from
/// * `moves` - principal variation to format
/// * `uci_chess960` - whether castling uses Chess960 rook-source notation
///
/// # Errors
///
/// Returns an [`io::Error`] when writing to `output` fails.
fn write_pv<W: Write>(
    output: &mut W,
    root: &Position,
    moves: &[Move],
    uci_chess960: bool,
) -> io::Result<()> {
    let mut position = root.clone();
    for mv in moves {
        write!(output, " {}", position.uci_move_string(*mv, uci_chess960))?;
        if position.make_move(*mv).is_err() {
            break;
        }
    }
    Ok(())
}

/// Used for mapping a bounded MCTS value to a logistic centipawn
/// approximation.
///
/// The value is clamped just inside the open interval `(-1, 1)` before the
/// transform `300 * ln((1 + v) / (1 - v))` is rounded to an integer.
///
/// # Arguments
///
/// * `value` - tree value in `[-1, 1]`
///
/// # Returns
///
/// Approximate centipawn score.
#[cfg(feature = "mcts")]
#[allow(clippy::cast_possible_truncation)]
fn value_to_centipawns(value: f64) -> i32 {
    let bounded = value.clamp(-0.999_999, 0.999_999);
    (300.0 * ((1.0 + bounded) / (1.0 - bounded)).ln()).round() as i32
}

/// Used for converting a score into an approximate win/draw/loss triplet
/// summing to 1000.
///
/// Mate scores collapse to a certain win or loss. Other scores pass through
/// `tanh(score / 600)` and quadratic shaping, with clamping that keeps the
/// three parts non-negative and summing to exactly 1000.
///
/// # Arguments
///
/// * `score` - centipawn or mate-encoded score
///
/// # Returns
///
/// `[win, draw, loss]` in per-mille.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn score_wdl(score: i32) -> [u16; 3] {
    if mate_moves(score).is_some() {
        return if score > 0 {
            [1_000, 0, 0]
        } else {
            [0, 0, 1_000]
        };
    }
    let value = (f64::from(score) / 600.0).tanh();
    let win = (((1.0 + value) * (1.0 + value) * 250.0).round() as i32).clamp(0, 1_000);
    let loss = (((1.0 - value) * (1.0 - value) * 250.0).round() as i32).clamp(0, 1_000 - win);
    let draw = 1_000 - win - loss;
    [win as u16, draw as u16, loss as u16]
}

/// Used for converting elapsed time to milliseconds, saturating values
/// wider than `u64`.
///
/// # Arguments
///
/// * `duration` - elapsed wall-clock time
///
/// # Returns
///
/// Milliseconds, saturating at [`u64::MAX`].
fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Used for parsing, validating, and de-duplicating `go searchmoves` at the
/// current root.
///
/// # Arguments
///
/// * `position` - root against which move text is parsed
/// * `requested` - raw `searchmoves` tokens
///
/// # Returns
///
/// De-duplicated legal moves, or [`None`] when no restriction was
/// requested.
///
/// # Errors
///
/// Returns the parse error text for the first token that is not a legal UCI
/// move at the root.
fn resolve_search_moves(
    position: &Position,
    requested: &[String],
) -> Result<Option<Vec<Move>>, String> {
    if requested.is_empty() {
        return Ok(None);
    }
    let mut moves = Vec::with_capacity(requested.len());
    for text in requested {
        let mv = position
            .parse_uci_move(text)
            .map_err(|error| error.to_string())?;
        if !moves.contains(&mv) {
            moves.push(mv);
        }
    }
    Ok(Some(moves))
}

/// Used for replaying a UCI position and returning its root, repetition
/// keys, and real positions.
///
/// The returned key list contains the key of every position before its move
/// was applied, and the position list runs from the source position through
/// the final root inclusive.
///
/// # Arguments
///
/// * `spec` - parsed `position` command
///
/// # Returns
///
/// `(root, previous_keys, positions)` triple described above.
///
/// # Errors
///
/// Returns the error text when the FEN is invalid or any move fails to
/// parse or apply.
fn replay_position(spec: &PositionSpec) -> Result<(Position, Vec<u64>, Vec<Position>), String> {
    let mut position = match &spec.source {
        PositionSource::StartPos => Position::start(),
        PositionSource::Fen(fen) => Position::from_fen(fen).map_err(|error| error.to_string())?,
    };
    let mut previous_keys = Vec::with_capacity(spec.moves.len());
    let mut positions = Vec::with_capacity(spec.moves.len().saturating_add(1));
    positions.push(position.clone());
    for text in &spec.moves {
        previous_keys.push(position.key());
        let mv: Move = position
            .parse_uci_move(text)
            .map_err(|error| error.to_string())?;
        position.make_move(mv).map_err(|error| error.to_string())?;
        positions.push(position.clone());
    }
    Ok((position, previous_keys, positions))
}
