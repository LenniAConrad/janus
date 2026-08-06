//! WDL and DTZ probing of a position against the loaded tables.
//!
//! The probing entry point is the per-worker [`Prober`]: it owns the
//! worker's deterministic block cache and shares the immutable
//! [`Tablebases`] registry. Probing follows the standard Syzygy contract:
//! because generators store "don't care" values for positions with a
//! winning capture, [`Prober::probe_wdl`] resolves captures (and, for DTZ
//! purposes, pawn pushes) with a bounded recursive search before trusting a
//! table value, and en-passant positions are resolved the same way because
//! tables never encode en-passant rights. DTZ tables store one side to
//! move only; [`Prober::probe_dtz`] recovers the other side through a
//! bounded one-ply search.

use super::cache::{BlockCache, BlockKey, DEFAULT_CACHE_BYTES};
use super::encode::{edge_distance, flip_file, flip_rank, off_a1h8};
use super::material::{position_material_key, TableEntry, Tablebases};
use super::pairs::{FLAG_LOSS_PLIES, FLAG_MAPPED, FLAG_STM, FLAG_WIDE, FLAG_WIN_PLIES};
use super::table::{LoadedTable, TableKind};
use super::{SyzygyConfig, SyzygyError, Wdl, TB_PIECES};
use janus_core::{Color, Piece, PieceKind, Position, Square};
use std::sync::Arc;

/// Resolution-search verdict attached to a WDL probe.
///
/// Distinguishes plain table values from values established by a zeroing
/// move, which DTZ probing must handle specially because DTZ tables store
/// "don't care" values in that case.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WdlState {
    /// Used for values read from a table or dominated by one.
    Table,
    /// Used for values established by a winning or forced zeroing move.
    ZeroingBestMove,
}

/// Outcome of one raw DTZ table access.
pub(crate) enum DtzOutcome {
    /// Used for a stored distance-to-zeroing value.
    Value(i32),
    /// Used for tables storing only the other side to move.
    ChangeStm,
}

/// Used for recovering the DTZ of the move that reached a just-zeroed
/// position.
///
/// # Arguments
///
/// * `wdl` - verdict of the position after the zeroing move
///
/// # Returns
///
/// `1`/`-1` for decisive verdicts, `101`/`-101` for cursed ones, and `0`
/// for draws, all from the mover's point of view.
pub(crate) fn dtz_before_zeroing(wdl: Wdl) -> i32 {
    match wdl {
        Wdl::Win => 1,
        Wdl::CursedWin => 101,
        Wdl::BlessedLoss => -101,
        Wdl::Loss => -1,
        Wdl::Draw => 0,
    }
}

/// Used for taking the sign of a distance value.
///
/// # Arguments
///
/// * `value` - signed distance
///
/// # Returns
///
/// `-1`, `0`, or `1`.
pub(crate) fn sign(value: i32) -> i32 {
    match value.cmp(&0) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Less => -1,
    }
}

/// Used for converting a `janus_core` square index to the Syzygy layout.
///
/// `janus_core` counts squares from `a8`; the tablebase encoding counts
/// from `a1` with `square = 8 * rank + file`.
///
/// # Arguments
///
/// * `janus_index` - square index in `0..64` with `a8 = 0`
///
/// # Returns
///
/// Square index in `0..64` with `a1 = 0`.
pub(crate) fn syzygy_square(janus_index: u8) -> u8 {
    ((7 - janus_index / 8) << 3) | (janus_index % 8)
}

/// Used for encoding a piece in the tables' nibble convention.
///
/// # Arguments
///
/// * `piece` - piece to encode
///
/// # Returns
///
/// Piece type in bits `0..3` (pawn `1` through king `6`) and color in bit
/// `3` (set for black).
pub(crate) fn tb_piece(piece: Piece) -> u8 {
    let kind = u8::try_from(piece.kind.index() + 1).expect("kind index below seven");
    match piece.color {
        Color::White => kind,
        Color::Black => kind | 8,
    }
}

/// Per-worker tablebase prober.
///
/// Owns this worker's block cache and probe statistics while sharing the
/// immutable registry, so identical probe sequences behave identically on
/// every worker. The gating fields mirror the UCI options after clamping
/// against the registry's actual cardinality.
pub struct Prober {
    /// Used for sharing the immutable registry and encode tables.
    shared: Arc<Tablebases>,
    /// Used for caching compressed blocks deterministically per worker.
    cache: BlockCache,
    /// Used for counting successful table accesses for `info ... tbhits`.
    hits: u64,
    /// Used for gating probes by piece count; zero disables probing.
    cardinality: usize,
    /// Used for gating boundary-cardinality probes by remaining depth.
    probe_depth: i32,
    /// Used for honoring the fifty-move rule in score mapping.
    rule50: bool,
}

impl Prober {
    /// Used for building a worker-local prober from the shared
    /// configuration.
    ///
    /// A probe limit above the registry's largest available table clamps to
    /// that size and drops the depth gate to zero, mirroring the standard
    /// UCI semantics of `SyzygyProbeLimit`/`SyzygyProbeDepth`.
    ///
    /// # Arguments
    ///
    /// * `config` - shared registry and clamped UCI option values
    ///
    /// # Returns
    ///
    /// A prober with an empty cache and zeroed statistics.
    #[must_use]
    pub fn new(config: &SyzygyConfig) -> Self {
        let max_cardinality = config.tables.max_cardinality();
        let mut cardinality = usize::from(config.probe_limit);
        let probe_depth = i32::from(config.probe_depth);
        // Clamping the requested piece limit to what the installed set actually
        // provides is correct. Zeroing the configured probe depth alongside it
        // was not: `SyzygyProbeLimit` defaults to seven, so any set smaller
        // than seven men made the clamp fire and silently discarded the
        // operator's `SyzygyProbeDepth`. With the common 3-4-5 set the option
        // was inert, and PERF-219 measured identical tbhits (48,285) at probe
        // depths 1, 8, and 20 because of it.
        if cardinality > max_cardinality {
            cardinality = max_cardinality;
        }
        Self {
            shared: Arc::clone(&config.tables),
            cache: BlockCache::new(DEFAULT_CACHE_BYTES),
            hits: 0,
            cardinality,
            probe_depth,
            rule50: config.rule50,
        }
    }

    /// Used for reading the number of successful table accesses so far.
    ///
    /// # Returns
    ///
    /// Monotonic per-search hit count.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Used for zeroing the hit counter at the start of a search.
    pub fn reset_hits(&mut self) {
        self.hits = 0;
    }

    /// Used for reading the effective probe cardinality.
    ///
    /// # Returns
    ///
    /// Largest piece count this prober will probe; zero disables probing.
    #[must_use]
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Used for reading the effective probe-depth gate.
    ///
    /// # Returns
    ///
    /// Minimum remaining depth required at the cardinality boundary.
    #[must_use]
    pub fn probe_depth(&self) -> i32 {
        self.probe_depth
    }

    /// Used for reading whether the fifty-move rule applies to mapping.
    ///
    /// # Returns
    ///
    /// `true` when cursed wins and blessed losses count as draws.
    #[must_use]
    pub fn rule50(&self) -> bool {
        self.rule50
    }

    /// Used for probing the WDL verdict of a position.
    ///
    /// The position must have no castling rights; the halfmove clock is
    /// irrelevant to WDL. Captures and en-passant possibilities are
    /// resolved with a bounded recursive search so generator "don't care"
    /// values can never leak out. The position is restored before
    /// returning.
    ///
    /// # Arguments
    ///
    /// * `position` - position to probe; mutated and restored internally
    ///
    /// # Returns
    ///
    /// The verdict for the side to move, or `None` when a required table
    /// is missing or unreadable.
    pub fn probe_wdl(&mut self, position: &mut Position) -> Option<Wdl> {
        self.wdl_search(position, false).ok().map(|(wdl, _)| wdl)
    }

    /// Used for probing the distance to zeroing of a position.
    ///
    /// Follows the standard convention: positive values are winning
    /// distances in plies (values above `100` are wins spoiled by the
    /// fifty-move rule), negative values are losing distances, zero is a
    /// draw, and results may be off by one ply toward safety.
    ///
    /// # Arguments
    ///
    /// * `position` - position to probe; mutated and restored internally
    ///
    /// # Returns
    ///
    /// The signed distance, or `None` when a required WDL or DTZ table is
    /// missing or unreadable.
    pub fn probe_dtz(&mut self, position: &mut Position) -> Option<i32> {
        self.dtz_search(position).ok()
    }

    /// Used for resolving captures before trusting a WDL table value.
    ///
    /// Recursion is bounded by the number of men: every searched move is a
    /// capture (or, at the first level of DTZ probing, a pawn move whose
    /// children only search captures), so depth shrinks with material.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate; mutated and restored
    /// * `check_zeroing` - whether pawn moves must also be resolved (DTZ)
    ///
    /// # Returns
    ///
    /// The verdict and whether a zeroing move established it.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when any required table is missing or
    /// unreadable.
    pub(crate) fn wdl_search(
        &mut self,
        position: &mut Position,
        check_zeroing: bool,
    ) -> Result<(Wdl, WdlState), SyzygyError> {
        let moves = position.legal_moves();
        let total_moves = moves.len();
        let mut tried = 0usize;
        let mut best = Wdl::Loss;

        for mv in moves {
            let capture = position.is_capture(mv);
            let pawn_move = position
                .piece_at(mv.from())
                .is_some_and(|piece| piece.kind == PieceKind::Pawn);
            if !capture && (!check_zeroing || !pawn_move) {
                continue;
            }
            tried += 1;
            let undo = position
                .make_move(mv)
                .map_err(|error| SyzygyError::new(format!("probe make_move: {error}")))?;
            let child = self.wdl_search(position, false);
            position.unmake_move(mv, undo);
            let value = child?.0.negate();
            if value > best {
                best = value;
                if value == Wdl::Win {
                    return Ok((value, WdlState::ZeroingBestMove));
                }
            }
        }

        // With only zeroing moves available the table value may be a
        // "don't care" (and en-passant-only positions are not encoded at
        // all), so the resolved best value is authoritative.
        let no_more_moves = tried > 0 && tried == total_moves;
        let value = if no_more_moves {
            best
        } else {
            self.probe_wdl_table(position)?
        };
        if best >= value {
            let state = if best > Wdl::Draw || no_more_moves {
                WdlState::ZeroingBestMove
            } else {
                WdlState::Table
            };
            return Ok((best, state));
        }
        Ok((value, WdlState::Table))
    }

    /// Used for computing the signed DTZ of a position recursively.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate; mutated and restored
    ///
    /// # Returns
    ///
    /// The signed distance to zeroing in plies.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when any required table is missing or
    /// unreadable.
    pub(crate) fn dtz_search(&mut self, position: &mut Position) -> Result<i32, SyzygyError> {
        let (wdl, state) = self.wdl_search(position, true)?;
        if wdl == Wdl::Draw {
            return Ok(0); // DTZ tables do not store draws.
        }
        if state == WdlState::ZeroingBestMove {
            return Ok(dtz_before_zeroing(wdl));
        }
        match self.probe_dtz_table(position, wdl)? {
            DtzOutcome::Value(dtz) => {
                let cursed = i32::from(matches!(wdl, Wdl::BlessedLoss | Wdl::CursedWin));
                Ok((dtz + 100 * cursed) * sign(wdl.value()))
            }
            DtzOutcome::ChangeStm => self.dtz_change_stm(position, wdl),
        }
    }

    /// Used for recovering DTZ through a one-ply search when the table
    /// stores only the other side to move.
    ///
    /// After any move the side to move matches the stored side, so the
    /// recursion terminates after one ply. Zeroing moves take their value
    /// from the position before the zeroing per [`dtz_before_zeroing`].
    ///
    /// # Arguments
    ///
    /// * `position` - position whose DTZ table stores the other side
    /// * `wdl` - already-resolved verdict of the position
    ///
    /// # Returns
    ///
    /// The signed distance, minimized over the winning side's moves.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when any child probe fails.
    fn dtz_change_stm(&mut self, position: &mut Position, wdl: Wdl) -> Result<i32, SyzygyError> {
        const NO_DTZ: i32 = 0xFFFF;
        let mut min_dtz = NO_DTZ;
        for mv in position.legal_moves() {
            let zeroing = position.is_capture(mv)
                || position
                    .piece_at(mv.from())
                    .is_some_and(|piece| piece.kind == PieceKind::Pawn);
            let undo = position
                .make_move(mv)
                .map_err(|error| SyzygyError::new(format!("probe make_move: {error}")))?;
            let child = if zeroing {
                self.wdl_search(position, false)
                    .map(|(child_wdl, _)| -dtz_before_zeroing(child_wdl))
            } else {
                self.dtz_search(position).map(|dtz| -dtz)
            };
            let mates = child.as_ref().is_ok_and(|dtz| *dtz == 1)
                && position.in_check(position.side_to_move())
                && position.legal_moves().is_empty();
            position.unmake_move(mv, undo);
            let mut dtz = child?;
            if mates {
                min_dtz = 1;
            }
            if !zeroing {
                dtz += sign(dtz);
            }
            if dtz < min_dtz && sign(dtz) == sign(wdl.value()) {
                min_dtz = dtz;
            }
        }
        // No legal moves means the side to move is mated.
        Ok(if min_dtz == NO_DTZ { -1 } else { min_dtz })
    }

    /// Used for reading one WDL value straight from its table.
    ///
    /// # Arguments
    ///
    /// * `position` - position to look up
    ///
    /// # Returns
    ///
    /// The stored verdict (which may be a generator "don't care" that the
    /// caller must dominate with resolved captures).
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when no table covers the material or the
    /// table cannot be read.
    pub(crate) fn probe_wdl_table(&mut self, position: &Position) -> Result<Wdl, SyzygyError> {
        if position.occupancy().count_ones() == 2 {
            return Ok(Wdl::Draw); // Bare kings.
        }
        let shared = Arc::clone(&self.shared);
        let key = position_material_key(position);
        let entry = shared
            .entry_for(key)
            .ok_or_else(|| SyzygyError::new("no WDL table for this material"))?;
        let loaded = entry
            .wdl_table(&shared.encode)
            .ok_or_else(|| SyzygyError::new("WDL table failed to load"))?
            .clone();
        let (data_index, idx) = encode_position(position, entry, &loaded, &shared)?
            .ok_or_else(|| SyzygyError::new("WDL tables store both sides"))?;
        let raw = self.decompress(&loaded, entry.index * 2, data_index, idx)?;
        let wdl = Wdl::from_value(i32::from(raw) - 2)
            .ok_or_else(|| SyzygyError::new("WDL value out of range"))?;
        self.hits += 1;
        Ok(wdl)
    }

    /// Used for reading one DTZ value straight from its table.
    ///
    /// # Arguments
    ///
    /// * `position` - position to look up
    /// * `wdl` - resolved verdict selecting the value map and ply doubling
    ///
    /// # Returns
    ///
    /// The mapped unsigned distance, or [`DtzOutcome::ChangeStm`] when the
    /// table stores only the other side to move.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when no DTZ table covers the material or
    /// the table cannot be read.
    fn probe_dtz_table(
        &mut self,
        position: &Position,
        wdl: Wdl,
    ) -> Result<DtzOutcome, SyzygyError> {
        let shared = Arc::clone(&self.shared);
        let key = position_material_key(position);
        let entry = shared
            .entry_for(key)
            .ok_or_else(|| SyzygyError::new("no DTZ table for this material"))?;
        let loaded = entry
            .dtz_table(&shared.encode)
            .ok_or_else(|| SyzygyError::new("DTZ table missing or failed to load"))?
            .clone();
        let Some((data_index, idx)) = encode_position(position, entry, &loaded, &shared)? else {
            return Ok(DtzOutcome::ChangeStm);
        };
        let raw = self.decompress(&loaded, entry.index * 2 + 1, data_index, idx)?;
        let value = map_dtz_value(&loaded, data_index, wdl, raw)?;
        self.hits += 1;
        Ok(DtzOutcome::Value(value))
    }

    /// Used for decoding one value from a stream, via the block cache for
    /// disk-backed tables and directly for memory-backed ones.
    ///
    /// # Arguments
    ///
    /// * `loaded` - parsed table owning the stream
    /// * `file_id` - registry-assigned cache identity of the physical file
    /// * `data_index` - index of the stream inside `loaded.pairs`
    /// * `idx` - encoded position index
    ///
    /// # Returns
    ///
    /// The raw decoded value.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] on any read or decode failure.
    fn decompress(
        &mut self,
        loaded: &LoadedTable,
        file_id: u32,
        data_index: usize,
        idx: u64,
    ) -> Result<u16, SyzygyError> {
        let data = &loaded.pairs[data_index];
        if data.is_single_value() {
            return Ok(u16::from(data.min_sym_len));
        }
        let (block, offset) = data.locate(idx)?;
        let start = data
            .data_offset
            .checked_add(u64::from(block).saturating_mul(data.block_size as u64))
            .ok_or_else(|| SyzygyError::new("block offset overflows"))?;
        if let Some(slice) = loaded.file.memory_slice(start, data.block_size) {
            return data.decode(slice, offset);
        }
        let key = BlockKey {
            file: file_id,
            block,
        };
        let bytes = self
            .cache
            .get_or_load(key, || loaded.file.read_vec_at(start, data.block_size))?;
        data.decode(bytes, offset)
    }
}

/// Used for mapping a raw DTZ table value to a distance in plies.
///
/// Applies the optional per-outcome value map and doubles move-based
/// storage into plies, then adds the conventional one.
///
/// # Arguments
///
/// * `loaded` - parsed DTZ table owning the value map
/// * `data_index` - index of the probed stream inside `loaded.pairs`
/// * `wdl` - resolved verdict selecting the map sequence
/// * `raw` - raw decoded value
///
/// # Returns
///
/// The unsigned distance in plies (before the caller applies the sign).
///
/// # Errors
///
/// Returns [`SyzygyError`] when the map lookup leaves the map region.
fn map_dtz_value(
    loaded: &LoadedTable,
    data_index: usize,
    wdl: Wdl,
    raw: u16,
) -> Result<i32, SyzygyError> {
    let data = &loaded.pairs[data_index];
    let flags = data.flags;
    let mut value = i32::from(raw);
    if flags & FLAG_MAPPED != 0 {
        // Map sequence order inside the file: win, loss, cursed win,
        // blessed loss; draws are never stored.
        let sequence = match wdl {
            Wdl::Win | Wdl::Draw => 0,
            Wdl::Loss => 1,
            Wdl::CursedWin => 2,
            Wdl::BlessedLoss => 3,
        };
        let base = usize::from(data.map_idx[sequence]);
        let element = base
            .checked_add(
                usize::try_from(value).map_err(|_| SyzygyError::new("negative raw DTZ value"))?,
            )
            .ok_or_else(|| SyzygyError::new("DTZ map index overflows"))?;
        if flags & FLAG_WIDE != 0 {
            let bytes = loaded
                .map
                .get(element * 2..element * 2 + 2)
                .ok_or_else(|| SyzygyError::new("wide DTZ map lookup out of range"))?;
            value = i32::from(u16::from_le_bytes(bytes.try_into().expect("two map bytes")));
        } else {
            value = i32::from(
                *loaded
                    .map
                    .get(element)
                    .ok_or_else(|| SyzygyError::new("DTZ map lookup out of range"))?,
            );
        }
    }
    // Values stored in moves are converted to plies; cursed outcomes are
    // always stored in moves.
    let stored_in_moves = match wdl {
        Wdl::Win => flags & FLAG_WIN_PLIES == 0,
        Wdl::Loss => flags & FLAG_LOSS_PLIES == 0,
        Wdl::CursedWin | Wdl::BlessedLoss => true,
        Wdl::Draw => false,
    };
    if stored_in_moves {
        value *= 2;
    }
    Ok(value + 1)
}

/// Used for encoding a position into a stream index.
///
/// Selects the stream matching the side to move and leading-pawn file,
/// applies the color, file, rank, and diagonal normalizations, and folds
/// the piece groups into the final index.
///
/// # Arguments
///
/// * `position` - position to encode
/// * `entry` - registry entry covering the material
/// * `loaded` - parsed table supplying piece sequences and multipliers
/// * `shared` - registry owning the encode tables
///
/// # Returns
///
/// The stream's index inside `loaded.pairs` and the encoded position
/// index, or `None` when a one-sided DTZ table stores the other side.
///
/// # Errors
///
/// Returns [`SyzygyError`] when the position is inconsistent with the
/// table's piece sequence.
#[allow(clippy::too_many_lines)]
fn encode_position(
    position: &Position,
    entry: &TableEntry,
    loaded: &LoadedTable,
    shared: &Tablebases,
) -> Result<Option<(usize, u64)>, SyzygyError> {
    let encode = &shared.encode;
    let meta = entry.meta;
    let piece_count = usize::from(meta.piece_count);

    // Symmetric tables store white to move only; a stronger black side
    // means the position's colors are swapped relative to the file.
    let symmetric_black = meta.symmetric && position.side_to_move() == Color::Black;
    let black_stronger = position_material_key(position) != entry.key;
    let flip = symmetric_black || black_stronger;
    let color_mask = if flip { 8u8 } else { 0 };
    let square_mask = if flip { 0x38u8 } else { 0 };
    let stm = usize::from(flip) ^ usize::from(position.side_to_move() == Color::Black);

    let mut squares = [0u8; TB_PIECES];
    let mut pieces = [0u8; TB_PIECES];
    let mut size = 0usize;
    let mut lead_pawns_cnt = 0usize;
    let mut lead_pawns_bb = 0u64;
    let mut tb_file = 0usize;

    if meta.has_pawns {
        // The leading pawns' color in the current position follows the
        // table's first piece after undoing the color flip.
        let reference = loaded.pairs_for(0, 0).pieces[0] ^ color_mask;
        let color = if reference & 8 != 0 {
            Color::Black
        } else {
            Color::White
        };
        lead_pawns_bb = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
        let mut bits = lead_pawns_bb;
        while bits != 0 {
            let janus_index = u8::try_from(bits.trailing_zeros()).expect("bit index below 64");
            bits &= bits - 1;
            squares[size] = syzygy_square(janus_index) ^ square_mask;
            size += 1;
        }
        if size == 0 {
            return Err(SyzygyError::new("pawn table probed without lead pawns"));
        }
        lead_pawns_cnt = size;
        // The leading pawn maximizes the pawn map (nearest the edge, then
        // the lowest rank).
        let mut lead = 0usize;
        for i in 1..size {
            if encode.map_pawns[usize::from(squares[i])]
                > encode.map_pawns[usize::from(squares[lead])]
            {
                lead = i;
            }
        }
        squares.swap(0, lead);
        tb_file = edge_distance(usize::from(squares[0]) % 8);
    }

    // One-sided DTZ tables can only answer for their stored side.
    if loaded.kind == TableKind::Dtz {
        let flags = loaded.pairs_for(stm, tb_file).flags;
        let stm_matches = usize::from(flags & FLAG_STM) == stm;
        if !(stm_matches || (meta.symmetric && !meta.has_pawns)) {
            return Ok(None);
        }
    }

    // Collect the remaining pieces with their table encodings.
    let mut bits = position.occupancy() ^ lead_pawns_bb;
    while bits != 0 {
        let janus_index = u8::try_from(bits.trailing_zeros()).expect("bit index below 64");
        bits &= bits - 1;
        let square = Square::new(janus_index).expect("bit index is a square");
        let piece = position
            .piece_at(square)
            .ok_or_else(|| SyzygyError::new("occupancy bit without a piece"))?;
        if size >= piece_count {
            return Err(SyzygyError::new("position has more men than the table"));
        }
        squares[size] = syzygy_square(janus_index) ^ square_mask;
        pieces[size] = tb_piece(piece) ^ color_mask;
        size += 1;
    }
    if size != piece_count {
        return Err(SyzygyError::new("position has fewer men than the table"));
    }

    let data_index = pairs_index(loaded, stm, tb_file);
    let data = &loaded.pairs[data_index];

    // Bring the non-pawn pieces into the table's storage sequence.
    for i in lead_pawns_cnt..size.saturating_sub(1) {
        for j in i + 1..size {
            if data.pieces[i] == pieces[j] {
                pieces.swap(i, j);
                squares.swap(i, j);
                break;
            }
        }
    }

    // Normalize the leading square into the left half of the board.
    if usize::from(squares[0]) % 8 > 3 {
        for square in squares.iter_mut().take(size) {
            *square = u8::try_from(flip_file(usize::from(*square))).expect("square in range");
        }
    }

    let mut idx: u64;
    if meta.has_pawns {
        idx = encode.lead_pawn_idx[lead_pawns_cnt][usize::from(squares[0])];
        // Remaining lead pawns encode in ascending pawn-map order.
        squares[1..lead_pawns_cnt].sort_by_key(|square| encode.map_pawns[usize::from(*square)]);
        for (i, square) in squares.iter().enumerate().take(lead_pawns_cnt).skip(1) {
            let pawn_rank = usize::try_from(encode.map_pawns[usize::from(*square)])
                .expect("pawn map codes stay below 48");
            idx += encode.binomial[i][pawn_rank];
        }
    } else {
        // Normalize the leading square below the horizontal midline and
        // the leading group onto or below the long diagonal.
        if usize::from(squares[0]) / 8 > 3 {
            for square in squares.iter_mut().take(size) {
                *square = u8::try_from(flip_rank(usize::from(*square))).expect("square in range");
            }
        }
        for i in 0..usize::from(data.group_len[0]) {
            if off_a1h8(usize::from(squares[i])) == 0 {
                continue;
            }
            if off_a1h8(usize::from(squares[i])) > 0 {
                for square in squares.iter_mut().take(size).skip(i) {
                    let value = usize::from(*square);
                    *square =
                        u8::try_from(((value >> 3) | (value << 3)) & 63).expect("square in range");
                }
            }
            break;
        }

        if meta.has_unique_pieces {
            let lead = u64::from(squares[0]);
            let second = u64::from(squares[1]);
            let third = u64::from(squares[2]);
            let adjust1 = u64::from(squares[1] > squares[0]);
            let adjust2 = u64::from(squares[2] > squares[0]) + u64::from(squares[2] > squares[1]);
            idx = if off_a1h8(usize::from(squares[0])) != 0 {
                (encode.map_a1d1d4[usize::from(squares[0])] * 63 + (second - adjust1)) * 62 + third
                    - adjust2
            } else if off_a1h8(usize::from(squares[1])) != 0 {
                (6 * 63 + (lead / 8) * 28 + encode.map_b1h1h7[usize::from(squares[1])]) * 62 + third
                    - adjust2
            } else if off_a1h8(usize::from(squares[2])) != 0 {
                6 * 63 * 62
                    + 4 * 28 * 62
                    + (lead / 8) * 7 * 28
                    + (second / 8 - adjust1) * 28
                    + encode.map_b1h1h7[usize::from(squares[2])]
            } else {
                6 * 63 * 62
                    + 4 * 28 * 62
                    + 4 * 7 * 28
                    + (lead / 8) * 7 * 6
                    + (second / 8 - adjust1) * 6
                    + (third / 8 - adjust2)
            };
        } else {
            let triangle = usize::try_from(encode.map_a1d1d4[usize::from(squares[0])])
                .expect("triangle codes stay below ten");
            idx = encode.map_kk[triangle][usize::from(squares[1])];
        }
    }

    // Fold the remaining groups into the index, smallest multiplier first.
    idx = idx
        .checked_mul(data.group_idx[0])
        .ok_or_else(|| SyzygyError::new("group index overflows"))?;
    let mut group_start = usize::from(data.group_len[0]);
    let mut remaining_pawns = meta.has_pawns && meta.pawn_count[1] > 0;
    let mut next = 1usize;
    while data.group_len[next] != 0 {
        let group_len = usize::from(data.group_len[next]);
        let group_end = group_start + group_len;
        if group_end > size {
            return Err(SyzygyError::new("group walks past the piece list"));
        }
        squares[group_start..group_end].sort_unstable();
        let mut group_index = 0u64;
        for i in 0..group_len {
            let current = squares[group_start + i];
            let mut adjust = 0usize;
            for earlier in &squares[..group_start] {
                adjust += usize::from(current > *earlier);
            }
            let shifted = usize::from(current)
                .checked_sub(adjust)
                .and_then(|value| value.checked_sub(if remaining_pawns { 8 } else { 0 }))
                .filter(|value| *value < 64)
                .ok_or_else(|| SyzygyError::new("group square underflows"))?;
            group_index += encode.binomial[i + 1][shifted];
        }
        remaining_pawns = false;
        idx = group_index
            .checked_mul(data.group_idx[next])
            .and_then(|value| idx.checked_add(value))
            .ok_or_else(|| SyzygyError::new("group index overflows"))?;
        group_start = group_end;
        next += 1;
    }

    Ok(Some((data_index, idx)))
}

/// Used for computing a stream's index inside a table's `pairs` vector.
///
/// # Arguments
///
/// * `loaded` - parsed table
/// * `stm` - side to move in the table's frame
/// * `tb_file` - leading-pawn file (ignored for pawnless tables)
///
/// # Returns
///
/// Index usable with `loaded.pairs`.
fn pairs_index(loaded: &LoadedTable, stm: usize, tb_file: usize) -> usize {
    let file_index = if loaded.files == 4 { tb_file } else { 0 };
    file_index * loaded.sides + (stm % loaded.sides)
}
