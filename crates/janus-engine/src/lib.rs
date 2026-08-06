#![allow(
    // Public engine types retain their domain prefix when re-exported at the
    // crate root, where names such as MctsConfig and TtPayload are unambiguous.
    clippy::module_name_repetitions
)]
#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

//! Search and evaluation for the Janus chess engine.
//!
//! The crate is intentionally dependency-free. Searchers consume the single
//! rules implementation in [`janus_core`] and share explicit evaluator, score,
//! resource-limit, and transposition-table contracts. This keeps alpha-beta,
//! MCTS, classical evaluation, and neural backends interchangeable without
//! duplicating chess rules.
//!
//! Scores returned by alpha-beta evaluators are centipawns from the side to move.
//! MCTS values use the bounded `[-1, 1]` convention described by [`PolicyValue`].
//! Fixed node or simulation limits are deterministic; wall-clock limits are
//! intentionally machine dependent.

/// Iterative-deepening alpha-beta search and its result types.
///
/// Re-exported at the crate root as [`AlphaBeta`], [`SearchError`],
/// [`SearchInfo`], and [`SearchResult`].
mod alpha_beta;
/// Safe parser for CRTK's dependency-free BT4 containers.
///
/// Key items such as [`Bt4Model`] and [`Bt4Error`] are also re-exported at
/// the crate root.
pub mod bt4;
/// Java-compatible BT4 board and attention-policy encoding.
///
/// Its encoding helpers are re-exported at the crate root under
/// `bt4`-prefixed names such as [`encode_bt4_position`].
pub mod bt4_encoding;
/// Safe process boundary for optional BT4 OpenCL acceleration.
///
/// Re-exported at the crate root as [`Bt4Backend`] and [`Bt4BackendStatus`].
pub mod bt4_gpu;
/// Safe scalar inference for the pinned full BT4 v2 model.
///
/// Provides [`Bt4Network`] and its [`Bt4Prediction`] output, both re-exported
/// at the crate root.
pub mod bt4_inference;
/// Dependency-free tapered classical evaluation.
///
/// Re-exported at the crate root as [`Classical`] and [`ClassicalBreakdown`].
mod classical;
/// LCZero-style convolutional model reader and evaluator.
///
/// Provides [`CnnModel`] and [`CnnPrediction`], both re-exported at the crate
/// root.
pub mod cnn;
/// Shared value and policy evaluation contracts.
///
/// Defines [`Evaluator`], [`SearchEvaluator`], [`SearchStateError`],
/// [`PolicyValue`], and the fallback [`MaterialEvaluator`].
pub mod evaluator;
pub mod jrbt;
/// Search resource limits and clock allocation.
///
/// Defines [`SearchLimits`], the one-shot [`SearchClock`], and the `clock_*`
/// budget allocators re-exported at the crate root.
mod limits;
/// Deterministic single-thread PUCT Monte Carlo tree search.
///
/// Re-exported at the crate root as [`Mcts`], [`MctsConfig`], [`MctsLimits`],
/// and their related result types. Compiled only with the default `mcts`
/// feature.
#[cfg(feature = "mcts")]
pub mod mcts;
/// Compact Janus NNUE model reader and evaluator.
///
/// Provides [`CompactNnue`], re-exported at the crate root alongside its
/// feature helpers.
pub mod nnue;
/// Shared centipawn and mate-score conventions.
///
/// Defines the mate encoding around [`MATE_SCORE`] and the score-conversion
/// helpers re-exported at the crate root.
mod score;
/// Small search-reduction and time-stability formulas.
///
/// Provides [`lmr_reduction`], [`null_move_reduction`],
/// [`late_move_prune_threshold`], and [`stability_budget_millis`].
mod search_math;
/// Safe, deterministic Syzygy endgame tablebase probing.
///
/// Key items such as [`syzygy::Tablebases`], [`syzygy::Prober`], and
/// [`syzygy::SyzygyConfig`] configure alpha-beta's optional endgame-table
/// integration.
pub mod syzygy;
/// Fallible scoped-worker admission with exact caller-thread fallback.
mod threading;
/// Compact direct-mapped transposition table.
///
/// Re-exported at the crate root as [`TranspositionTable`],
/// [`SharedTranspositionTable`], and their support types.
mod tt;
/// Reader and evaluator for supported upstream NNUE containers.
///
/// Provides [`UpstreamNnue`] and [`UpstreamNnueEvaluator`], both re-exported
/// at the crate root.
pub mod upstream_nnue;

pub use alpha_beta::{
    eval_cache_size_for_threads, set_eval_cache_size, AlphaBeta, CorrectionHistoryMode,
    SearchError, SearchInfo, SearchResult,
};
pub use bt4::{
    Bt4Activation, Bt4Architecture, Bt4Error, Bt4ErrorKind, Bt4InputEmbedding, Bt4InputFormat,
    Bt4Model, Bt4TensorInfo, Bt4V2Extensions, MAX_MODEL_BYTES as MAX_BT4_MODEL_BYTES,
    MAX_MODEL_PARAMETERS as MAX_BT4_MODEL_PARAMETERS,
};
pub use bt4_encoding::{
    append_position_map as append_bt4_position_map,
    compressed_by_internal_map as bt4_compressed_by_internal_map,
    compressed_policy_index as bt4_compressed_policy_index, encode_history as encode_bt4_history,
    encode_history_into as encode_bt4_history_into,
    encode_lc0_fen_position as encode_bt4_lc0_fen_position,
    encode_lc0_fen_position_into as encode_bt4_lc0_fen_position_into,
    encode_position as encode_bt4_position, encode_position_into as encode_bt4_position_into,
    gather_legal_internal_logits as gather_bt4_legal_internal_logits,
    gather_legal_policy_logits as gather_bt4_legal_policy_logits,
    gather_policy as gather_bt4_policy, internal_policy_index as bt4_internal_policy_index,
    to_token_major as bt4_to_token_major, Bt4EncodedInput, Bt4EncodingError,
};
pub use bt4_gpu::{Bt4Backend, Bt4BackendStatus};
pub use bt4_inference::{
    Bt4Info, Bt4Network, Bt4Prediction, INPUT_CHANNELS as BT4_INPUT_CHANNELS,
    MAX_INFERENCE_THREADS as MAX_BT4_INFERENCE_THREADS, POLICY_SIZE as BT4_POLICY_SIZE,
    TOKENS as BT4_TOKENS,
};
/// Research-only parameterized twin of the classical evaluation for offline
/// HCE tuning; hidden from the documented API surface.
#[doc(hidden)]
pub use classical::tuning as classical_tuning;
pub use classical::{Classical, ClassicalBreakdown, PieceSquareDelta};
pub use cnn::{
    encode_position as encode_cnn_position, raw_policy_index, CnnError, CnnErrorKind, CnnInfo,
    CnnModel, CnnPrediction, INPUT_CHANNELS as CNN_INPUT_CHANNELS,
    MAX_INFERENCE_THREADS as MAX_CNN_INFERENCE_THREADS, RAW_POLICY_SIZE as CNN_RAW_POLICY_SIZE,
};
pub use evaluator::{Evaluator, MaterialEvaluator, PolicyValue, SearchEvaluator, SearchStateError};
pub use limits::{
    clock_budget_millis, clock_budgets, clock_hard_budget_millis, SearchClock, SearchLimits,
    CLOCK_MOVES_TO_GO, CLOCK_RESERVE_MILLIS,
};
#[cfg(feature = "mcts")]
pub use mcts::{
    GameOutcome, Mcts, MctsConfig, MctsError, MctsLimits, MctsResult, RootMove,
    DEFAULT_MCTS_TREE_NODE_LIMIT,
};
pub use nnue::{active_features, CompactNnue, NnueError, NnueErrorKind, NnueInfo, FEATURE_COUNT};
pub use score::{
    clamp_static_score, is_mate_score, mate_moves, score_from_tt, score_to_tt, terminal_score,
    INFINITY, MATE_SCORE, MATE_THRESHOLD, MAX_DEPTH, MAX_STATIC_SCORE, TB_WIN_SCORE,
    TB_WIN_THRESHOLD,
};
pub use search_math::{
    late_move_prune_threshold, lmr_reduction, null_move_reduction, stability_budget_millis,
};
pub use tt::{Bound, SharedTranspositionTable, TranspositionTable, TtHit, TtPayload};
pub use upstream_nnue::{
    UpstreamNnue, UpstreamNnueError, UpstreamNnueErrorKind, UpstreamNnueEvaluator,
    UpstreamNnueInfo, UpstreamPrediction, UPSTREAM_VERSION,
};
