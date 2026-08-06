#![allow(clippy::module_name_repetitions)]

//! Deterministic perft utilities for validating the legal move generator.
//!
//! Perft counts the legal leaves an exact number of plies below a root
//! position. Unchecked helpers serve callers that own their execution
//! budget; checked variants add bounded cancellation and overflow-safe
//! aggregation and never return partial counters.

use crate::position::movegen::{self, MoveBuffer};
use crate::{Move, Position};
use core::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

/// Used for bounding the depth accepted by Janus's developer perft command.
///
/// The legacy [`perft`] and [`divide`] helpers remain unrestricted for callers
/// that own their execution budget. Checked command paths enforce this limit
/// before starting enumeration.
pub const MAX_PERFT_DEPTH: u32 = 12;

/// Used for fixing the number of recursive entries between cancellation-token
/// observations.
///
/// Polling is deliberately amortized so the responsive node-only path retains
/// useful comparison throughput without allowing cancellation latency to grow
/// with the requested depth.
const CANCELLATION_POLL_INTERVAL: u16 = 1_024;

/// Detailed counters for leaves exactly one requested depth below a root.
///
/// Move-event fields describe only the move entering a counted leaf. They do
/// not include events on earlier plies. A depth-zero result therefore contains
/// one node and zero move events.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PerftStats {
    /// Used for counting legal leaves reached at the requested depth.
    pub nodes: u64,
    /// Used for counting leaves entered by a capture, including en passant.
    pub captures: u64,
    /// Used for counting leaves entered specifically by an en-passant capture.
    pub en_passant: u64,
    /// Used for counting leaves entered by an orthodox or Chess960 castle.
    pub castles: u64,
    /// Used for counting leaves entered by any promotion choice.
    pub promotions: u64,
    /// Used for counting leaves whose resulting side to move is in check.
    pub checks: u64,
    /// Used for counting leaves whose resulting side to move is checkmated.
    pub checkmates: u64,
}

impl PerftStats {
    /// Used for constructing the event-free result for a depth-zero root.
    ///
    /// A depth-zero perft counts the current position itself as the single
    /// leaf, so no move enters it and no move event can be recorded.
    ///
    /// # Returns
    ///
    /// Counters holding one node and zero move events.
    #[must_use]
    pub const fn depth_zero() -> Self {
        Self {
            nodes: 1,
            captures: 0,
            en_passant: 0,
            castles: 0,
            promotions: 0,
            checks: 0,
            checkmates: 0,
        }
    }

    /// Used for adding every counter without permitting release-mode wrapping.
    ///
    /// # Arguments
    ///
    /// * `other` - counters added to `self` field by field
    ///
    /// # Returns
    ///
    /// The field-wise sum of both counter sets.
    ///
    /// # Errors
    ///
    /// Returns [`PerftError::CounterOverflow`] if any field exceeds `u64`.
    pub fn checked_add(self, other: Self) -> Result<Self, PerftError> {
        Ok(Self {
            nodes: checked_sum(self.nodes, other.nodes)?,
            captures: checked_sum(self.captures, other.captures)?,
            en_passant: checked_sum(self.en_passant, other.en_passant)?,
            castles: checked_sum(self.castles, other.castles)?,
            promotions: checked_sum(self.promotions, other.promotions)?,
            checks: checked_sum(self.checks, other.checks)?,
            checkmates: checked_sum(self.checkmates, other.checkmates)?,
        })
    }
}

/// Detailed counters attributed to one legal root move.
///
/// One entry forms one row of the detailed divide report produced by
/// [`detailed_divide`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PerftDivideEntry {
    /// Used for identifying the internally encoded legal root move.
    pub root_move: Move,
    /// Used for holding the leaf counters below `root_move`.
    pub stats: PerftStats,
}

/// Deterministic detailed divide rows and their checked aggregate.
///
/// Produced by [`detailed_divide`]. Rows are sorted in context-free
/// UCI-lexicographic move order, so equal roots and depths always yield an
/// identical report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PerftDivideResult {
    /// Used for listing root rows in context-free UCI-lexicographic move
    /// order.
    pub entries: Vec<PerftDivideEntry>,
    /// Used for holding the checked sum of every row, or the depth-zero
    /// singleton at depth zero.
    pub total: PerftStats,
}

/// Node-only count attributed to one legal root move.
///
/// One entry forms one row of the node-only divide report produced by
/// [`checked_divide`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PerftNodeEntry {
    /// Used for identifying the internally encoded legal root move.
    pub root_move: Move,
    /// Used for counting the legal leaves below `root_move`.
    pub nodes: u64,
}

/// Deterministic node-only divide rows and their checked aggregate.
///
/// Produced by [`checked_divide`]. Rows are sorted in context-free
/// UCI-lexicographic move order, so equal roots and depths always yield an
/// identical report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PerftNodeResult {
    /// Used for listing root rows in context-free UCI-lexicographic move
    /// order.
    pub entries: Vec<PerftNodeEntry>,
    /// Used for holding the checked sum of row nodes, or one for depth zero.
    pub total_nodes: u64,
}

/// Controlled perft termination that never returns partial counters.
///
/// Checked entry points return this error instead of surfacing incomplete or
/// wrapped totals, and they restore the traversed position on every exit
/// path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerftError {
    /// Used for indicating that the caller's shared stop token requested
    /// cancellation.
    Cancelled,
    /// Used for indicating that a node or detailed event counter exceeded
    /// `u64`.
    CounterOverflow,
}

impl fmt::Display for PerftError {
    /// Used for writing a stable user-facing termination reason.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter receiving the reason text
    ///
    /// # Returns
    ///
    /// The formatter's write result.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("cancelled"),
            Self::CounterOverflow => formatter.write_str("counter overflow"),
        }
    }
}

impl std::error::Error for PerftError {}

/// Amortized observer for one shared cancellation token.
///
/// Loads the token only once every [`CANCELLATION_POLL_INTERVAL`] recursive
/// entries so cancellation checks stay off the hot counting path.
struct PerftControl<'a> {
    /// Used for referencing the token owned by the UCI coordinator or a
    /// focused caller.
    stop: &'a AtomicBool,
    /// Used for tracking the recursive entries remaining before the next
    /// atomic load.
    remaining: u16,
}

impl<'a> PerftControl<'a> {
    /// Used for creating an observer that starts with an immediate
    /// observation before any position mutation.
    ///
    /// # Arguments
    ///
    /// * `stop` - shared cancellation token observed during the traversal
    ///
    /// # Returns
    ///
    /// Observer whose first [`poll`](Self::poll) performs an atomic load.
    fn new(stop: &'a AtomicBool) -> Self {
        Self { stop, remaining: 0 }
    }

    /// Used for observing cancellation at a fixed bounded recursive interval.
    ///
    /// Performs an atomic load only when the amortization counter reaches
    /// zero, then rearms the counter with [`CANCELLATION_POLL_INTERVAL`].
    ///
    /// # Errors
    ///
    /// Returns [`PerftError::Cancelled`] when the shared token was observed
    /// set.
    fn poll(&mut self) -> Result<(), PerftError> {
        if self.remaining == 0 {
            if self.stop.load(Ordering::Relaxed) {
                return Err(PerftError::Cancelled);
            }
            self.remaining = CANCELLATION_POLL_INTERVAL;
        }
        self.remaining -= 1;
        Ok(())
    }
}

/// Per-ply direct legal-move scratch shared by one perft traversal.
///
/// Each recursion level owns one legal buffer. The vector allocates only when
/// the traversal starts, then reuses fixed-capacity storage through every node.
struct MovegenContext {
    /// Used for holding the legal output scratch indexed by recursion ply.
    legal: Vec<MoveBuffer>,
}

impl MovegenContext {
    /// Used for allocating enough scratch for every requested ply and one
    /// leaf probe.
    ///
    /// # Arguments
    ///
    /// * `depth` - requested perft depth in plies
    ///
    /// # Returns
    ///
    /// Context owning `depth + 1` reusable legal move buffers.
    ///
    /// # Panics
    ///
    /// Panics if `depth` does not fit `usize` or the extra leaf-probe slot
    /// overflows `usize`.
    fn new(depth: u32) -> Self {
        let slots = usize::try_from(depth)
            .expect("u32 perft depth fits usize")
            .checked_add(1)
            .expect("perft scratch depth overflow");
        Self {
            legal: vec![MoveBuffer::default(); slots],
        }
    }

    /// Used for generating legal moves into the scratch owned by `ply`.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose legal moves are generated
    /// * `ply` - recursion level selecting the scratch buffer
    /// * `tactical` - tactical-only filter flag forwarded to the legal
    ///   generator; every perft call site passes `false`
    ///
    /// # Returns
    ///
    /// Number of legal moves written into the selected buffer.
    ///
    /// # Panics
    ///
    /// Panics if `ply` exceeds the scratch depth allocated by
    /// [`MovegenContext::new`].
    fn generate(&mut self, position: &mut Position, ply: usize, tactical: bool) -> usize {
        movegen::generate_legal_moves(position, &mut self.legal[ply], tactical);
        self.legal[ply].as_slice().len()
    }

    /// Used for counting legal moves through the bulk path with the scratch
    /// owned by `ply`.
    ///
    /// Delegates to [`movegen::count_legal_moves`], which popcounts the
    /// common-case pawn, knight, and slider target masks and materializes
    /// only king moves, castles, and fallback cases into the selected
    /// buffer.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose legal moves are counted
    /// * `ply` - recursion level selecting the scratch buffer
    ///
    /// # Returns
    ///
    /// Number of legal moves for the side to move.
    ///
    /// # Panics
    ///
    /// Panics if `ply` exceeds the scratch depth allocated by
    /// [`MovegenContext::new`].
    fn count_legal(&mut self, position: &Position, ply: usize) -> usize {
        movegen::count_legal_moves(position, &mut self.legal[ply])
    }

    /// Used for returning one copied move from a previously generated ply.
    ///
    /// # Arguments
    ///
    /// * `ply` - recursion level whose scratch buffer is read
    /// * `index` - zero-based slot within that buffer
    ///
    /// # Returns
    ///
    /// The requested move by value.
    ///
    /// # Panics
    ///
    /// Panics if `ply` or `index` lies outside the previously generated
    /// scratch contents.
    fn move_at(&self, ply: usize, index: usize) -> Move {
        self.legal[ply].as_slice()[index]
    }
}

/// Used for counting legal leaf nodes exactly `depth` plies below `position`.
///
/// The supplied position is restored before this function returns. At depth
/// zero, the current position itself is the single leaf.
///
/// # Arguments
///
/// * `position` - root position; mutated during traversal and restored
/// * `depth` - number of plies between the root and the counted leaves
///
/// # Returns
///
/// Number of legal leaves at the requested depth.
///
/// # Panics
///
/// Panics only if the legal move generator produces a move that
/// [`Position::make_move`] rejects, which indicates an internal core invariant
/// violation. Callers requiring cancellation and checked overflow should use
/// [`checked_perft`].
pub fn perft(position: &mut Position, depth: u32) -> u64 {
    node_perft_unchecked(position, depth, &mut MovegenContext::new(depth), 0)
}

/// Used for returning root move counts in deterministic UCI-lexicographic
/// order.
///
/// The supplied position is restored before this function returns. Depth zero
/// has no root moves and therefore returns an empty vector.
///
/// # Arguments
///
/// * `position` - root position; mutated during traversal and restored
/// * `depth` - number of plies between the root and the counted leaves
///
/// # Returns
///
/// One `(move, leaf count)` pair per legal root move.
///
/// # Panics
///
/// Panics only if the legal move generator produces a move that
/// [`Position::make_move`] rejects, which indicates an internal core invariant
/// violation. Callers requiring cancellation, a depth-zero total, and checked
/// overflow should use [`checked_divide`].
pub fn divide(position: &mut Position, depth: u32) -> Vec<(Move, u64)> {
    if depth == 0 {
        return Vec::new();
    }
    let mut context = MovegenContext::new(depth);
    let move_count = context.generate(position, 0, false);
    let mut result = Vec::new();
    for index in 0..move_count {
        let mv = context.move_at(0, index);
        let undo = position
            .make_move(mv)
            .expect("legal generator emitted an applicable move");
        let nodes = node_perft_unchecked(position, depth - 1, &mut context, 1);
        position.unmake_move(mv, undo);
        result.push((mv, nodes));
    }
    result.sort_by_key(|(mv, _)| mv.uci_order_key());
    result
}

/// Used for counting nodes with bounded cancellation and checked aggregation.
///
/// The supplied position is restored on success, cancellation, and overflow.
/// This function does not enforce [`MAX_PERFT_DEPTH`]; command owners must
/// validate their requested depth before starting work.
///
/// # Arguments
///
/// * `position` - root position; mutated during traversal and restored
/// * `depth` - number of plies between the root and the counted leaves
/// * `stop` - shared token observed at a bounded recursive interval
///
/// # Returns
///
/// Number of legal leaves at the requested depth.
///
/// # Errors
///
/// Returns [`PerftError::Cancelled`] after observing `stop`, or
/// [`PerftError::CounterOverflow`] instead of wrapping a node total.
///
/// # Panics
///
/// Panics only for an internal legal-move/make-move invariant violation.
pub fn checked_perft(
    position: &mut Position,
    depth: u32,
    stop: &AtomicBool,
) -> Result<u64, PerftError> {
    node_perft(
        position,
        depth,
        &mut PerftControl::new(stop),
        &mut MovegenContext::new(depth),
        0,
    )
}

/// Used for returning checked node-only root rows with bounded cancellation.
///
/// Depth zero returns no rows and a total of one. The supplied position is
/// restored on every exit path.
///
/// # Arguments
///
/// * `position` - root position; mutated during traversal and restored
/// * `depth` - number of plies between the root and the counted leaves
/// * `stop` - shared token observed at a bounded recursive interval
///
/// # Returns
///
/// Node-only rows in UCI-lexicographic order plus their checked total.
///
/// # Errors
///
/// Returns [`PerftError::Cancelled`] after observing `stop`, or
/// [`PerftError::CounterOverflow`] instead of wrapping the total.
///
/// # Panics
///
/// Panics only for an internal legal-move/make-move invariant violation.
pub fn checked_divide(
    position: &mut Position,
    depth: u32,
    stop: &AtomicBool,
) -> Result<PerftNodeResult, PerftError> {
    let mut control = PerftControl::new(stop);
    control.poll()?;
    if depth == 0 {
        return Ok(PerftNodeResult {
            entries: Vec::new(),
            total_nodes: 1,
        });
    }

    let mut context = MovegenContext::new(depth);
    let move_count = context.generate(position, 0, false);
    let mut entries = Vec::new();
    let mut total_nodes = 0_u64;
    for index in 0..move_count {
        let mv = context.move_at(0, index);
        let undo = position
            .make_move(mv)
            .expect("legal generator emitted an applicable move");
        let result = node_perft(position, depth - 1, &mut control, &mut context, 1);
        position.unmake_move(mv, undo);
        let nodes = result?;
        total_nodes = checked_sum(total_nodes, nodes)?;
        entries.push(PerftNodeEntry {
            root_move: mv,
            nodes,
        });
    }
    entries.sort_by_key(|entry| entry.root_move.uci_order_key());
    Ok(PerftNodeResult {
        entries,
        total_nodes,
    })
}

/// Used for counting detailed leaf events with bounded cancellation and
/// checked totals.
///
/// The supplied position is restored on success, cancellation, and overflow.
/// Event fields classify only the final move into each counted leaf.
///
/// # Arguments
///
/// * `position` - root position; mutated during traversal and restored
/// * `depth` - number of plies between the root and the counted leaves
/// * `stop` - shared token observed at a bounded recursive interval
///
/// # Returns
///
/// Detailed counters aggregated over every leaf at the requested depth.
///
/// # Errors
///
/// Returns [`PerftError::Cancelled`] after observing `stop`, or
/// [`PerftError::CounterOverflow`] instead of wrapping any counter.
///
/// # Panics
///
/// Panics only for an internal legal-move/make-move invariant violation.
pub fn detailed_perft(
    position: &mut Position,
    depth: u32,
    stop: &AtomicBool,
) -> Result<PerftStats, PerftError> {
    detailed_perft_inner(
        position,
        depth,
        &mut PerftControl::new(stop),
        &mut MovegenContext::new(depth),
        0,
    )
}

/// Used for returning checked detailed root rows with bounded cancellation.
///
/// Depth zero returns no rows and [`PerftStats::depth_zero`]. The supplied
/// position is restored on every exit path.
///
/// # Arguments
///
/// * `position` - root position; mutated during traversal and restored
/// * `depth` - number of plies between the root and the counted leaves
/// * `stop` - shared token observed at a bounded recursive interval
///
/// # Returns
///
/// Detailed rows in UCI-lexicographic order plus their checked aggregate.
///
/// # Errors
///
/// Returns [`PerftError::Cancelled`] after observing `stop`, or
/// [`PerftError::CounterOverflow`] instead of wrapping any counter.
///
/// # Panics
///
/// Panics only for an internal legal-move/make-move invariant violation.
pub fn detailed_divide(
    position: &mut Position,
    depth: u32,
    stop: &AtomicBool,
) -> Result<PerftDivideResult, PerftError> {
    let mut control = PerftControl::new(stop);
    control.poll()?;
    if depth == 0 {
        return Ok(PerftDivideResult {
            entries: Vec::new(),
            total: PerftStats::depth_zero(),
        });
    }

    let mut context = MovegenContext::new(depth);
    let move_count = context.generate(position, 0, false);
    let mut entries = Vec::new();
    let mut total = PerftStats::default();
    for index in 0..move_count {
        let mv = context.move_at(0, index);
        let undo = position
            .make_move(mv)
            .expect("legal generator emitted an applicable move");
        let result = if depth == 1 {
            Ok(detailed_leaf(position, mv, undo, &mut context, 1))
        } else {
            detailed_perft_inner(position, depth - 1, &mut control, &mut context, 1)
        };
        position.unmake_move(mv, undo);
        let stats = result?;
        total = total.checked_add(stats)?;
        entries.push(PerftDivideEntry {
            root_move: mv,
            stats,
        });
    }
    entries.sort_by_key(|entry| entry.root_move.uci_order_key());
    Ok(PerftDivideResult { entries, total })
}

/// Used for recursively counting nodes without cancellation or overflow
/// checks.
///
/// Uses the shared per-ply scratch path and skips the final make/unmake
/// pair by returning the bulk legal-move count directly at depth one,
/// without materializing the common-case move list.
///
/// # Arguments
///
/// * `position` - current position; mutated and restored around each child
/// * `depth` - remaining plies below the current node
/// * `context` - per-ply legal move scratch shared by the traversal
/// * `ply` - current recursion level selecting the scratch buffer
///
/// # Returns
///
/// Number of legal leaves `depth` plies below the current position.
///
/// # Panics
///
/// Panics only for an internal legal-move/make-move invariant violation.
fn node_perft_unchecked(
    position: &mut Position,
    depth: u32,
    context: &mut MovegenContext,
    ply: usize,
) -> u64 {
    if depth == 0 {
        return 1;
    }
    if depth == 1 {
        return context.count_legal(position, ply) as u64;
    }
    let move_count = context.generate(position, ply, false);

    let mut nodes = 0_u64;
    for index in 0..move_count {
        let mv = context.move_at(ply, index);
        let undo = position
            .make_move(mv)
            .expect("legal generator emitted an applicable move");
        nodes += node_perft_unchecked(position, depth - 1, context, ply + 1);
        position.unmake_move(mv, undo);
    }
    nodes
}

/// Used for recursively counting nodes while sharing one amortized stop
/// observer.
///
/// Like [`node_perft_unchecked`], the final make/unmake pair is skipped by
/// returning the bulk legal-move count at depth one.
///
/// # Arguments
///
/// * `position` - current position; mutated and restored around each child
/// * `depth` - remaining plies below the current node
/// * `control` - amortized cancellation observer shared by the traversal
/// * `context` - per-ply legal move scratch shared by the traversal
/// * `ply` - current recursion level selecting the scratch buffer
///
/// # Returns
///
/// Number of legal leaves `depth` plies below the current position.
///
/// # Errors
///
/// Returns [`PerftError::Cancelled`] after observing the stop token, or
/// [`PerftError::CounterOverflow`] instead of wrapping the total.
///
/// # Panics
///
/// Panics only for an internal legal-move/make-move invariant violation.
fn node_perft(
    position: &mut Position,
    depth: u32,
    control: &mut PerftControl<'_>,
    context: &mut MovegenContext,
    ply: usize,
) -> Result<u64, PerftError> {
    control.poll()?;
    if depth == 0 {
        return Ok(1);
    }
    if depth == 1 {
        let move_count = context.count_legal(position, ply);
        return u64::try_from(move_count).map_err(|_| PerftError::CounterOverflow);
    }
    let move_count = context.generate(position, ply, false);

    let mut nodes = 0_u64;
    for index in 0..move_count {
        let mv = context.move_at(ply, index);
        let undo = position
            .make_move(mv)
            .expect("legal generator emitted an applicable move");
        let result = node_perft(position, depth - 1, control, context, ply + 1);
        position.unmake_move(mv, undo);
        nodes = checked_sum(nodes, result?)?;
    }
    Ok(nodes)
}

/// Used for recursively aggregating detailed counters while sharing one
/// amortized stop observer.
///
/// At depth one every child is classified through [`detailed_leaf`] while
/// the child position is still made.
///
/// # Arguments
///
/// * `position` - current position; mutated and restored around each child
/// * `depth` - remaining plies below the current node
/// * `control` - amortized cancellation observer shared by the traversal
/// * `context` - per-ply legal move scratch shared by the traversal
/// * `ply` - current recursion level selecting the scratch buffer
///
/// # Returns
///
/// Detailed counters aggregated over every leaf below the current position.
///
/// # Errors
///
/// Returns [`PerftError::Cancelled`] after observing the stop token, or
/// [`PerftError::CounterOverflow`] instead of wrapping any counter.
///
/// # Panics
///
/// Panics only for an internal legal-move/make-move invariant violation.
fn detailed_perft_inner(
    position: &mut Position,
    depth: u32,
    control: &mut PerftControl<'_>,
    context: &mut MovegenContext,
    ply: usize,
) -> Result<PerftStats, PerftError> {
    control.poll()?;
    if depth == 0 {
        return Ok(PerftStats::depth_zero());
    }

    let move_count = context.generate(position, ply, false);
    let mut total = PerftStats::default();
    for index in 0..move_count {
        let mv = context.move_at(ply, index);
        let undo = position
            .make_move(mv)
            .expect("legal generator emitted an applicable move");
        let result = if depth == 1 {
            Ok(detailed_leaf(position, mv, undo, context, ply + 1))
        } else {
            detailed_perft_inner(position, depth - 1, control, context, ply + 1)
        };
        position.unmake_move(mv, undo);
        total = total.checked_add(result?)?;
    }
    Ok(total)
}

/// Used for classifying the move that entered one already-made leaf position.
///
/// En-passant captures are recognized by a captured square differing from
/// the move destination. Checkmate is detected by generating the leaf's
/// legal replies into the supplied scratch ply while the side to move is in
/// check.
///
/// # Arguments
///
/// * `position` - leaf position after `mv` has been made
/// * `mv` - move that entered the leaf
/// * `undo` - undo record produced when `mv` was made
/// * `context` - per-ply legal move scratch shared by the traversal
/// * `scratch_ply` - scratch level used for the checkmate reply probe
///
/// # Returns
///
/// Single-node counters describing the entering move's events.
fn detailed_leaf(
    position: &mut Position,
    mv: Move,
    undo: crate::Undo,
    context: &mut MovegenContext,
    scratch_ply: usize,
) -> PerftStats {
    let captured = undo.captured();
    let checked = position.in_check(position.side_to_move());
    let checkmated = checked && context.generate(position, scratch_ply, false) == 0;
    PerftStats {
        nodes: 1,
        captures: u64::from(captured.is_some()),
        en_passant: u64::from(captured.is_some_and(|(_, square)| square != mv.to())),
        castles: u64::from(undo.was_castle()),
        promotions: u64::from(mv.promotion().is_some()),
        checks: u64::from(checked),
        checkmates: u64::from(checkmated),
    }
}

/// Used for adding two counters or converting arithmetic overflow into a
/// controlled error.
///
/// # Arguments
///
/// * `left` - first addend
/// * `right` - second addend
///
/// # Returns
///
/// The exact sum when it fits `u64`.
///
/// # Errors
///
/// Returns [`PerftError::CounterOverflow`] when the sum exceeds `u64`.
fn checked_sum(left: u64, right: u64) -> Result<u64, PerftError> {
    left.checked_add(right).ok_or(PerftError::CounterOverflow)
}

