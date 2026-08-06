//! Bounded parsing of `.rtbw` and `.rtbz` headers into owned metadata.
//!
//! A table file holds, per pawn file (`a..=d`, pawnless tables use one) and
//! per stored side, one compressed stream described by a
//! [`PairsData`] record. This module walks the header exactly once per
//! table, validating every offset against the file length before reading
//! and every structural size against fixed bounds, and leaves behind owned
//! vectors (sparse index, block lengths, Huffman shape, pairing tree, DTZ
//! value map) plus the absolute offset of each stream's data blocks. The
//! compressed blocks themselves are not read here.

use super::encode::EncodeTables;
use super::file::BoundedFile;
use super::pairs::{build_base64, build_symlen, LrEntry, PairsData, FLAG_MAPPED, FLAG_WIDE};
use super::{SyzygyError, TB_PIECES};
use janus_core::PieceKind;

/// Magic prefix of every WDL (`.rtbw`) file.
const WDL_MAGIC: [u8; 4] = [0x71, 0xE8, 0x23, 0x5D];
/// Magic prefix of every DTZ (`.rtbz`) file.
const DTZ_MAGIC: [u8; 4] = [0xD7, 0x66, 0x0C, 0xA5];
/// Layout bit: the file stores both sides to move.
const LAYOUT_SPLIT: u8 = 1;
/// Layout bit: the material configuration contains pawns.
const LAYOUT_HAS_PAWNS: u8 = 2;
/// Largest accepted block-size exponent; reference generators use at most
/// 1024-byte blocks, so this bound only rejects corrupt headers.
const MAX_BLOCK_SIZE_LOG: u8 = 20;
/// Largest accepted span exponent, bounding the sparse-index stride.
const MAX_SPAN_LOG: u8 = 40;
/// Largest accepted Huffman code length in bits; the 32-bit refill decoder
/// requires this bound.
const MAX_SYM_LEN: u8 = 32;
/// Largest accepted symbol count; symbols are 12-bit values.
const MAX_SYMBOLS: usize = 1 << 12;

/// Table family selecting the magic prefix and per-side layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TableKind {
    /// Used for win/draw/loss tables, storing up to two sides.
    Wdl,
    /// Used for distance-to-zeroing tables, storing exactly one side.
    Dtz,
}

/// Shape facts about one material configuration, derived from its piece
/// lists at registration time and consumed by header parsing and probing.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TableMeta {
    /// Used for counting all men including both kings.
    pub piece_count: u8,
    /// Used for recording whether either side has pawns.
    pub has_pawns: bool,
    /// Used for recording whether any non-king piece class has exactly one
    /// piece.
    pub has_unique_pieces: bool,
    /// Used for counting pawns: leading color first, other color second.
    pub pawn_count: [u8; 2],
    /// Used for recording whether both sides hold identical material.
    pub symmetric: bool,
}

impl TableMeta {
    /// Used for deriving the shape facts of a configuration.
    ///
    /// The first list plays the table's "white"; the leading pawn color is
    /// the side with fewer pawns, preferring the first side on ties, which
    /// matches how generators choose the better-compressing orientation.
    ///
    /// # Arguments
    ///
    /// * `first` - first-named side's non-king pieces
    /// * `second` - second-named side's non-king pieces
    ///
    /// # Returns
    ///
    /// The derived shape facts.
    pub fn from_sides(first: &[PieceKind], second: &[PieceKind]) -> Self {
        let count_of = |side: &[PieceKind], kind: PieceKind| {
            u8::try_from(side.iter().filter(|k| **k == kind).count())
                .expect("side holds at most five pieces")
        };
        let mut has_unique_pieces = false;
        for side in [first, second] {
            for kind in [
                PieceKind::Pawn,
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
            ] {
                if count_of(side, kind) == 1 {
                    has_unique_pieces = true;
                }
            }
        }
        let first_pawns = count_of(first, PieceKind::Pawn);
        let second_pawns = count_of(second, PieceKind::Pawn);
        let first_leads = second_pawns == 0 || (first_pawns > 0 && second_pawns >= first_pawns);
        let pawn_count = if first_leads {
            [first_pawns, second_pawns]
        } else {
            [second_pawns, first_pawns]
        };
        let mut first_sorted: Vec<PieceKind> = first.to_vec();
        let mut second_sorted: Vec<PieceKind> = second.to_vec();
        first_sorted.sort_by_key(|kind| kind.index());
        second_sorted.sort_by_key(|kind| kind.index());
        Self {
            piece_count: u8::try_from(first.len() + second.len() + 2)
                .expect("configurations hold at most seven men"),
            has_pawns: first_pawns + second_pawns > 0,
            has_unique_pieces,
            pawn_count,
            symmetric: first_sorted == second_sorted,
        }
    }
}

/// Fully parsed metadata of one physical table file.
///
/// Owns the bounded file handle for later block reads plus one
/// [`PairsData`] per (pawn file, stored side) stream and the raw DTZ value
/// map. Shared read-only between workers behind an `Arc`.
pub(crate) struct LoadedTable {
    /// Used for reading compressed blocks later.
    pub file: BoundedFile,
    /// Used for recording whether this is a WDL or DTZ table.
    pub kind: TableKind,
    /// Used for counting stored sides (two only for asymmetric WDL).
    pub sides: usize,
    /// Used for counting pawn-file sub-tables (four with pawns, else one).
    pub files: usize,
    /// Used for storing one stream description per (file, side), indexed
    /// `file * sides + side`.
    pub pairs: Vec<PairsData>,
    /// Used for storing the raw DTZ value-map region; empty for WDL.
    pub map: Vec<u8>,
}

impl LoadedTable {
    /// Used for selecting the stream covering a side to move and pawn file.
    ///
    /// # Arguments
    ///
    /// * `stm` - side to move in the table's frame (`0` or `1`)
    /// * `file` - pawn file index `0..4`; ignored for pawnless tables
    ///
    /// # Returns
    ///
    /// The matching stream description.
    pub fn pairs_for(&self, stm: usize, file: usize) -> &PairsData {
        let file_index = if self.files == 4 { file } else { 0 };
        &self.pairs[file_index * self.sides + (stm % self.sides)]
    }
}

/// Cursor performing bounded sequential reads over a table file.
struct Reader<'a> {
    /// Used for issuing validated positional reads.
    file: &'a BoundedFile,
    /// Used for tracking the absolute offset of the next read.
    pos: u64,
}

impl Reader<'_> {
    /// Used for reading one byte and advancing.
    ///
    /// # Returns
    ///
    /// The byte at the cursor.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when the cursor is at or past end of file.
    fn u8(&mut self) -> Result<u8, SyzygyError> {
        let mut byte = [0u8; 1];
        self.file.read_exact_at(self.pos, &mut byte)?;
        self.pos += 1;
        Ok(byte[0])
    }

    /// Used for reading one little-endian 16-bit word and advancing.
    ///
    /// # Returns
    ///
    /// The decoded word.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when fewer than two bytes remain.
    fn u16_le(&mut self) -> Result<u16, SyzygyError> {
        let mut bytes = [0u8; 2];
        self.file.read_exact_at(self.pos, &mut bytes)?;
        self.pos += 2;
        Ok(u16::from_le_bytes(bytes))
    }

    /// Used for reading one little-endian 32-bit word and advancing.
    ///
    /// # Returns
    ///
    /// The decoded word.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when fewer than four bytes remain.
    fn u32_le(&mut self) -> Result<u32, SyzygyError> {
        let mut bytes = [0u8; 4];
        self.file.read_exact_at(self.pos, &mut bytes)?;
        self.pos += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Used for reading an owned byte run and advancing.
    ///
    /// # Arguments
    ///
    /// * `length` - exact number of bytes to read
    ///
    /// # Returns
    ///
    /// The owned bytes.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when the run extends past end of file.
    fn bytes(&mut self, length: usize) -> Result<Vec<u8>, SyzygyError> {
        let bytes = self.file.read_vec_at(self.pos, length)?;
        self.pos += length as u64;
        Ok(bytes)
    }

    /// Used for skipping to the next even file offset.
    fn align_even(&mut self) {
        self.pos += self.pos & 1;
    }

    /// Used for skipping to the next 64-byte-aligned file offset, matching
    /// the cache-line alignment of data blocks.
    fn align64(&mut self) {
        self.pos = (self.pos + 0x3F) & !0x3F;
    }
}

/// Used for parsing one complete table header into owned metadata.
///
/// # Arguments
///
/// * `file` - opened table file
/// * `kind` - expected table family (checked against the magic)
/// * `meta` - registration-time shape facts (checked against the layout)
/// * `encode` - shared encoding tables for group-size computation
///
/// # Returns
///
/// The fully parsed table.
///
/// # Errors
///
/// Returns [`SyzygyError`] on a wrong magic, a layout contradicting the
/// registered material, any offset leaving the file, or any structural
/// size outside its documented bound.
#[allow(clippy::too_many_lines)]
pub(crate) fn load_table(
    file: BoundedFile,
    kind: TableKind,
    meta: TableMeta,
    encode: &EncodeTables,
) -> Result<LoadedTable, SyzygyError> {
    if file.byte_len() % 64 != 16 {
        return Err(SyzygyError::new("table length is not 64k+16 bytes"));
    }
    let mut reader = Reader {
        file: &file,
        pos: 0,
    };
    let mut magic = [0u8; 4];
    reader.file.read_exact_at(0, &mut magic)?;
    reader.pos = 4;
    let expected = match kind {
        TableKind::Wdl => WDL_MAGIC,
        TableKind::Dtz => DTZ_MAGIC,
    };
    if magic != expected {
        return Err(SyzygyError::new("table magic mismatch"));
    }

    let layout = reader.u8()?;
    if (layout & LAYOUT_HAS_PAWNS != 0) != meta.has_pawns {
        return Err(SyzygyError::new("table pawn layout contradicts material"));
    }
    if (layout & LAYOUT_SPLIT != 0) == meta.symmetric {
        return Err(SyzygyError::new("table split layout contradicts material"));
    }

    let sides = if kind == TableKind::Wdl && !meta.symmetric {
        2
    } else {
        1
    };
    let files = if meta.has_pawns { 4 } else { 1 };
    let pawns_on_both = meta.has_pawns && meta.pawn_count[1] > 0;
    let piece_count = usize::from(meta.piece_count);

    let mut pairs: Vec<PairsData> = (0..files * sides).map(|_| PairsData::empty()).collect();

    // Piece sequences and group orders, one block per pawn file.
    for file_index in 0..files {
        let order_low = reader.u8()?;
        let order_high = if pawns_on_both { reader.u8()? } else { 0xFF };
        let orders = [
            [order_low & 0x0F, order_high & 0x0F],
            [order_low >> 4, order_high >> 4],
        ];
        let mut piece_bytes = [0u8; TB_PIECES];
        for byte in piece_bytes.iter_mut().take(piece_count) {
            *byte = reader.u8()?;
        }
        for side in 0..sides {
            let data = &mut pairs[file_index * sides + side];
            for (piece, byte) in data.pieces.iter_mut().zip(&piece_bytes).take(piece_count) {
                *piece = if side == 1 { byte >> 4 } else { byte & 0x0F };
            }
            set_groups(data, meta, encode, orders[side], file_index)?;
        }
    }
    reader.align_even();

    // Compression headers per stream. `counts[i]` records how many sparse
    // and block-length entries stream `i` owns in the shared regions below.
    let mut counts = vec![(0usize, 0usize); pairs.len()];
    for file_index in 0..files {
        for side in 0..sides {
            let stream = file_index * sides + side;
            counts[stream] = read_sizes(&mut reader, &mut pairs[stream])?;
        }
    }

    // DTZ value map.
    let mut map = Vec::new();
    if kind == TableKind::Dtz {
        let map_start = reader.pos;
        for file_index in 0..files {
            let flags = pairs[file_index * sides].flags;
            if flags & FLAG_MAPPED == 0 {
                continue;
            }
            let mut map_idx = [0u16; 4];
            if flags & FLAG_WIDE != 0 {
                reader.align_even();
                for slot in &mut map_idx {
                    let relative = reader.pos - map_start;
                    if relative % 2 != 0 {
                        return Err(SyzygyError::new("wide DTZ map is misaligned"));
                    }
                    *slot = u16::try_from(relative / 2 + 1)
                        .map_err(|_| SyzygyError::new("DTZ map index exceeds 16 bits"))?;
                    let count = u64::from(reader.u16_le()?);
                    reader.pos = reader
                        .pos
                        .checked_add(count * 2)
                        .ok_or_else(|| SyzygyError::new("DTZ map length overflows"))?;
                }
            } else {
                for slot in &mut map_idx {
                    *slot = u16::try_from(reader.pos - map_start + 1)
                        .map_err(|_| SyzygyError::new("DTZ map index exceeds 16 bits"))?;
                    let count = u64::from(reader.u8()?);
                    reader.pos += count;
                }
            }
            pairs[file_index * sides].map_idx = map_idx;
        }
        let map_length = usize::try_from(reader.pos - map_start)
            .map_err(|_| SyzygyError::new("DTZ map exceeds address space"))?;
        map = file.read_vec_at(map_start, map_length)?;
        reader.align_even();
    }

    // Sparse indices, then block lengths, then 64-byte-aligned data blocks.
    for file_index in 0..files {
        for side in 0..sides {
            let stream = file_index * sides + side;
            let data = &mut pairs[stream];
            if data.is_single_value() {
                continue;
            }
            let bytes = counts[stream]
                .0
                .checked_mul(6)
                .ok_or_else(|| SyzygyError::new("sparse index region overflows"))?;
            let raw = reader.bytes(bytes)?;
            data.sparse_index = raw
                .chunks_exact(6)
                .map(|entry| {
                    (
                        u32::from_le_bytes(entry[0..4].try_into().expect("four bytes")),
                        u16::from_le_bytes(entry[4..6].try_into().expect("two bytes")),
                    )
                })
                .collect();
        }
    }
    for file_index in 0..files {
        for side in 0..sides {
            let stream = file_index * sides + side;
            let data = &mut pairs[stream];
            if data.is_single_value() {
                continue;
            }
            let bytes = counts[stream]
                .1
                .checked_mul(2)
                .ok_or_else(|| SyzygyError::new("block length region overflows"))?;
            let raw = reader.bytes(bytes)?;
            data.block_length = raw
                .chunks_exact(2)
                .map(|entry| u16::from_le_bytes(entry.try_into().expect("two bytes")))
                .collect();
        }
    }
    for file_index in 0..files {
        for side in 0..sides {
            let data = &mut pairs[file_index * sides + side];
            if data.is_single_value() {
                continue;
            }
            reader.align64();
            data.data_offset = reader.pos;
            let span = u64::from(data.blocks_num)
                .checked_mul(data.block_size as u64)
                .ok_or_else(|| SyzygyError::new("data-block region overflows"))?;
            reader.pos = reader
                .pos
                .checked_add(span)
                .ok_or_else(|| SyzygyError::new("data-block region overflows"))?;
        }
    }
    if reader.pos > file.byte_len() {
        return Err(SyzygyError::new("table data region exceeds file length"));
    }

    Ok(LoadedTable {
        file,
        kind,
        sides,
        files,
        pairs,
        map,
    })
}

/// Used for computing one stream's piece groups and index multipliers.
///
/// Groups collect adjacent identical pieces; the leading group is the king
/// pair or the unique-piece triple for pawnless tables and the leading
/// pawns otherwise. The header's `order` nibbles say where the leading
/// group and the second-side pawns sit in the multiplier chain.
///
/// # Arguments
///
/// * `data` - stream record whose `pieces` are already filled
/// * `meta` - shape facts of the configuration
/// * `encode` - shared encoding tables
/// * `order` - the stream's two order nibbles
/// * `file_index` - pawn file `0..4` selecting the leading-pawn size
///
/// # Returns
///
/// `Ok(())` with `group_len`/`group_idx` populated.
///
/// # Errors
///
/// Returns [`SyzygyError`] when a group is larger than the encoding tables
/// support or a multiplier overflows.
fn set_groups(
    data: &mut PairsData,
    meta: TableMeta,
    encode: &EncodeTables,
    order: [u8; 2],
    file_index: usize,
) -> Result<(), SyzygyError> {
    let piece_count = usize::from(meta.piece_count);
    let mut n = 0usize;
    let mut first_len: i32 = if meta.has_pawns {
        0
    } else if meta.has_unique_pieces {
        3
    } else {
        2
    };
    data.group_len = [0; TB_PIECES + 1];
    data.group_len[0] = 1;
    for i in 1..piece_count {
        first_len -= 1;
        if first_len > 0 || data.pieces[i] == data.pieces[i - 1] {
            data.group_len[n] += 1;
        } else {
            n += 1;
            data.group_len[n] = 1;
        }
    }
    n += 1;
    data.group_len[n] = 0;

    let pawns_on_both = meta.has_pawns && meta.pawn_count[1] > 0;
    let mut next = if pawns_on_both { 2 } else { 1 };
    let mut free_squares = 64usize
        .saturating_sub(usize::from(data.group_len[0]))
        .saturating_sub(if pawns_on_both {
            usize::from(data.group_len[1])
        } else {
            0
        });
    let mut idx = 1u64;
    let mut k = 0u8;
    while next < n || k == order[0] || k == order[1] {
        let factor = if k == order[0] {
            data.group_idx[0] = idx;
            let lead_len = usize::from(data.group_len[0]);
            if meta.has_pawns {
                if lead_len >= 6 {
                    return Err(SyzygyError::new("leading pawn group too large"));
                }
                encode.lead_pawns_size[lead_len][file_index]
            } else if meta.has_unique_pieces {
                31_332
            } else {
                462
            }
        } else if k == order[1] {
            data.group_idx[1] = idx;
            let len = usize::from(data.group_len[1]);
            if len >= 6 || usize::from(data.group_len[0]) > 48 {
                return Err(SyzygyError::new("second pawn group too large"));
            }
            encode.binomial[len][48 - usize::from(data.group_len[0])]
        } else {
            data.group_idx[next] = idx;
            let len = usize::from(data.group_len[next]);
            if len >= 6 || free_squares >= 64 {
                return Err(SyzygyError::new("piece group too large"));
            }
            let factor = encode.binomial[len][free_squares];
            free_squares -= len.min(free_squares);
            next += 1;
            factor
        };
        idx = idx
            .checked_mul(factor)
            .ok_or_else(|| SyzygyError::new("group multiplier overflows"))?;
        k += 1;
        if k > 16 {
            return Err(SyzygyError::new("group order nibbles are inconsistent"));
        }
    }
    data.group_idx[n] = idx;
    Ok(())
}

/// Used for parsing one stream's compression header.
///
/// Fills the Huffman shape, pairing tree, and expansion counts, and
/// reserves exact capacities for the sparse index and block lengths that
/// the caller reads afterwards.
///
/// # Arguments
///
/// * `reader` - cursor positioned at the stream's flags byte
/// * `data` - stream record whose groups are already computed
///
/// # Returns
///
/// The stream's sparse-index and block-length entry counts, with the
/// record's compression fields populated.
///
/// # Errors
///
/// Returns [`SyzygyError`] on structural sizes outside their bounds or a
/// malformed pairing tree.
fn read_sizes(
    reader: &mut Reader<'_>,
    data: &mut PairsData,
) -> Result<(usize, usize), SyzygyError> {
    data.flags = reader.u8()?;
    if data.is_single_value() {
        data.min_sym_len = reader.u8()?;
        data.blocks_num = 0;
        data.span = 0;
        return Ok((0, 0));
    }

    let table_size = data.group_idx[data
        .group_len
        .iter()
        .position(|len| *len == 0)
        .unwrap_or(TB_PIECES)];

    let block_size_log = reader.u8()?;
    if block_size_log > MAX_BLOCK_SIZE_LOG {
        return Err(SyzygyError::new("block size exponent out of range"));
    }
    data.block_size = 1usize << block_size_log;
    let span_log = reader.u8()?;
    if span_log > MAX_SPAN_LOG {
        return Err(SyzygyError::new("span exponent out of range"));
    }
    data.span = 1u64 << span_log;
    let sparse_count = usize::try_from(table_size.div_ceil(data.span))
        .map_err(|_| SyzygyError::new("sparse index exceeds address space"))?;
    let padding = reader.u8()?;
    data.blocks_num = reader.u32_le()?;
    let block_length_count = usize::try_from(data.blocks_num)
        .map_err(|_| SyzygyError::new("block count exceeds address space"))?
        + usize::from(padding);

    data.max_sym_len = reader.u8()?;
    data.min_sym_len = reader.u8()?;
    if data.max_sym_len < data.min_sym_len
        || data.max_sym_len > MAX_SYM_LEN
        || data.min_sym_len == 0
    {
        return Err(SyzygyError::new("Huffman code lengths out of range"));
    }
    let length_count = usize::from(data.max_sym_len - data.min_sym_len) + 1;
    let raw_lowest = reader.bytes(length_count * 2)?;
    data.lowest_sym = raw_lowest
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes(pair.try_into().expect("two bytes")))
        .collect();
    data.base64 = build_base64(&data.lowest_sym, data.min_sym_len)?;

    let symbol_count = usize::from(reader.u16_le()?);
    if symbol_count > MAX_SYMBOLS {
        return Err(SyzygyError::new("symbol count out of range"));
    }
    let raw_tree = reader.bytes(symbol_count * 3)?;
    data.btree = raw_tree
        .chunks_exact(3)
        .map(|entry| LrEntry([entry[0], entry[1], entry[2]]))
        .collect();
    if symbol_count % 2 == 1 {
        reader.pos += 1;
    }
    data.symlen = build_symlen(&data.btree)?;

    Ok((sparse_count, block_length_count))
}
