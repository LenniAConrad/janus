//! Safe, deterministic Syzygy endgame tablebase probing.
//!
//! The module reads the standard `.rtbw` (win/draw/loss) and `.rtbz`
//! (distance-to-zeroing) tablebase files through bounded positional reads
//! only: no memory mapping and no `unsafe` code anywhere. Table metadata is
//! parsed once, lazily, into owned buffers with strict offset validation
//! against the file length, and compressed data blocks are fetched through a
//! per-worker deterministic least-recently-used `BlockCache`, so probing is
//! reproducible byte for byte across runs and across workers.
//!
//! Layer map over the private submodules:
//!
//! - `file` - bounded positional file access (`BoundedFile`).
//! - `cache` - deterministic per-worker block cache.
//! - `encode` - deterministic position-encoding tables built at init.
//! - `material` - material keys, table naming, and the `Tablebases`
//!   registry re-exported here.
//! - `pairs` - RE-Pair plus canonical-Huffman symbol decoding.
//! - `table` - `.rtbw`/`.rtbz` header parsing into owned metadata.
//! - `probe` - WDL and DTZ probing over a position, including the
//!   capture/en-passant resolution search.
//! - `root` - root-move DTZ ranking with a WDL fallback filter.
//!
//! The abstract file format follows the public Syzygy specification as
//! implemented by the reference probers; this implementation is an
//! independent safe-Rust translation of the format, not a source port.

mod cache;
mod encode;
mod file;
mod material;
mod pairs;
mod probe;
mod root;
mod table;

pub use material::Tablebases;
pub use probe::Prober;
pub use root::RootFilter;

use std::fmt;
use std::sync::Arc;

/// Bound on the number of pieces any supported table may hold.
///
/// Syzygy tables exist up to seven men; every fixed-size piece array in this
/// module is sized by this constant.
pub(crate) const TB_PIECES: usize = 7;

/// Error raised while opening, parsing, or decoding a tablebase file.
///
/// A probe that encounters this error is reported as a failed probe; the
/// search then continues without tablebase information, so corrupt or
/// truncated files can never abort the engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyzygyError {
    /// Used for carrying the single human-readable diagnostic line.
    message: String,
}

impl SyzygyError {
    /// Used for creating an error from owned or borrowed diagnostic text.
    ///
    /// # Arguments
    ///
    /// * `message` - human-readable description of the failure
    ///
    /// # Returns
    ///
    /// A new error wrapping the message.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SyzygyError {
    /// Used for writing the wrapped diagnostic text.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter for the diagnostic
    ///
    /// # Returns
    ///
    /// Result of writing the message to `formatter`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SyzygyError {}

/// Win/draw/loss verdict from the probed side to move's point of view.
///
/// The variant order is ascending in desirability for the side to move, so
/// the derived [`Ord`] matches the natural comparison of outcomes. The
/// "cursed" and "blessed" variants mark decisive results that the fifty-move
/// rule converts into draws with best play.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Wdl {
    /// Used for indicating a loss for the side to move.
    Loss,
    /// Used for indicating a loss saved by the fifty-move rule.
    BlessedLoss,
    /// Used for indicating a draw.
    Draw,
    /// Used for indicating a win spoiled by the fifty-move rule.
    CursedWin,
    /// Used for indicating a win for the side to move.
    Win,
}

impl Wdl {
    /// Used for converting the verdict to the standard signed value in
    /// `-2..=2`.
    ///
    /// # Returns
    ///
    /// `-2` for loss through `2` for win, matching the on-disk WDL encoding
    /// minus two.
    #[must_use]
    pub const fn value(self) -> i32 {
        match self {
            Self::Loss => -2,
            Self::BlessedLoss => -1,
            Self::Draw => 0,
            Self::CursedWin => 1,
            Self::Win => 2,
        }
    }

    /// Used for decoding a signed WDL value in `-2..=2`.
    ///
    /// # Arguments
    ///
    /// * `value` - candidate signed verdict value
    ///
    /// # Returns
    ///
    /// `Some` verdict for values `-2..=2`, `None` otherwise.
    #[must_use]
    pub const fn from_value(value: i32) -> Option<Self> {
        match value {
            -2 => Some(Self::Loss),
            -1 => Some(Self::BlessedLoss),
            0 => Some(Self::Draw),
            1 => Some(Self::CursedWin),
            2 => Some(Self::Win),
            _ => None,
        }
    }

    /// Used for flipping the verdict to the opponent's point of view.
    ///
    /// # Returns
    ///
    /// The verdict with its sign negated; draws map to draws.
    #[must_use]
    pub const fn negate(self) -> Self {
        match self {
            Self::Loss => Self::Win,
            Self::BlessedLoss => Self::CursedWin,
            Self::Draw => Self::Draw,
            Self::CursedWin => Self::BlessedLoss,
            Self::Win => Self::Loss,
        }
    }
}

/// Search-facing tablebase configuration installed on one alpha-beta worker.
///
/// The registry is shared immutably between workers through the [`Arc`];
/// every worker builds its own [`Prober`] around it so block caching stays
/// worker-local and deterministic.
#[derive(Clone)]
pub struct SyzygyConfig {
    /// Used for sharing the immutable table registry across workers.
    pub tables: Arc<Tablebases>,
    /// Used for capping the probed piece count (UCI `SyzygyProbeLimit`).
    pub probe_limit: u8,
    /// Used for requiring a minimum remaining depth before probing positions
    /// at the cardinality boundary (UCI `SyzygyProbeDepth`).
    pub probe_depth: u8,
    /// Used for honoring the fifty-move rule when mapping WDL verdicts to
    /// scores (UCI `Syzygy50MoveRule`).
    pub rule50: bool,
}
