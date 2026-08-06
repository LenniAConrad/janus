//! RE-Pair plus canonical-Huffman decoding of tablebase values.
//!
//! Tablebase values are compressed in two layers: Recursive Pairing maps
//! runs of values onto a binary tree of symbols, and a canonical Huffman
//! code stores the resulting symbol stream in fixed-size blocks. This
//! module owns the per-table decoding metadata ([`PairsData`]) and the
//! decoder itself; block bytes are supplied by the caller so that cache
//! policy stays outside the decoding layer. All loops are explicitly
//! bounded so corrupt data yields a [`SyzygyError`] rather than divergence.

use super::{SyzygyError, TB_PIECES};

/// Table flag: the stored side to move (DTZ tables only).
pub(crate) const FLAG_STM: u8 = 1;
/// Table flag: DTZ values pass through an explicit value map.
pub(crate) const FLAG_MAPPED: u8 = 2;
/// Table flag: DTZ win values are stored in plies rather than moves.
pub(crate) const FLAG_WIN_PLIES: u8 = 4;
/// Table flag: DTZ loss values are stored in plies rather than moves.
pub(crate) const FLAG_LOSS_PLIES: u8 = 8;
/// Table flag: the DTZ value map holds 16-bit entries.
pub(crate) const FLAG_WIDE: u8 = 16;
/// Table flag: every position of this table stores one identical value.
pub(crate) const FLAG_SINGLE_VALUE: u8 = 128;

/// One three-byte Recursive Pairing tree entry.
///
/// Twelve bits name the left child symbol and twelve bits the right child;
/// a right child of `0xFFF` marks a leaf whose left field stores the value.
#[derive(Clone, Copy)]
pub(crate) struct LrEntry(pub [u8; 3]);

impl LrEntry {
    /// Used for extracting the left child symbol or leaf value.
    ///
    /// # Returns
    ///
    /// The low twelve bits of the packed entry.
    pub fn left(self) -> u16 {
        (u16::from(self.0[1] & 0x0F) << 8) | u16::from(self.0[0])
    }

    /// Used for extracting the right child symbol.
    ///
    /// # Returns
    ///
    /// The high twelve bits of the packed entry; `0xFFF` marks a leaf.
    pub fn right(self) -> u16 {
        (u16::from(self.0[2]) << 4) | u16::from(self.0[1] >> 4)
    }
}

/// Owned decoding metadata for one (side, pawn-file) sub-table.
///
/// Mirrors the on-disk header of one compressed stream: the canonical
/// Huffman shape (`base64`, `lowest_sym`, symbol lengths), the Recursive
/// Pairing tree, the sparse block index, and the piece grouping used by the
/// position encoder. Populated once at table load and immutable afterwards.
pub(crate) struct PairsData {
    /// Used for storing the sub-table flag byte (`FLAG_*`).
    pub flags: u8,
    /// Used for storing the shortest Huffman code length in bits; for
    /// single-value tables this field stores the value itself.
    pub min_sym_len: u8,
    /// Used for storing the longest Huffman code length in bits.
    pub max_sym_len: u8,
    /// Used for counting the compressed blocks of this sub-table.
    pub blocks_num: u32,
    /// Used for sizing each compressed block in bytes.
    pub block_size: usize,
    /// Used for spacing the sparse index: one entry per `span` values.
    pub span: u64,
    /// Used for jumping near a value index: `(block, offset)` pairs.
    pub sparse_index: Vec<(u32, u16)>,
    /// Used for storing the number of values (minus one) in each block.
    pub block_length: Vec<u16>,
    /// Used for storing the 64-bit padded lowest code of each length.
    pub base64: Vec<u64>,
    /// Used for storing the lowest symbol of each code length.
    pub lowest_sym: Vec<u16>,
    /// Used for storing the number of values (minus one) each symbol
    /// expands to.
    pub symlen: Vec<u8>,
    /// Used for storing the Recursive Pairing expansion tree.
    pub btree: Vec<LrEntry>,
    /// Used for storing the table's piece sequence in on-disk nibble
    /// encoding (type in bits `0..3`, color in bit `3`).
    pub pieces: [u8; TB_PIECES],
    /// Used for storing each encoding group's index multiplier.
    pub group_idx: [u64; TB_PIECES + 1],
    /// Used for storing each encoding group's piece count, zero terminated.
    pub group_len: [u8; TB_PIECES + 1],
    /// Used for locating the four per-outcome DTZ map sequences.
    pub map_idx: [u16; 4],
    /// Used for locating the first compressed block in the file.
    pub data_offset: u64,
}

impl PairsData {
    /// Used for creating an all-zero record before header parsing fills it.
    ///
    /// # Returns
    ///
    /// A record whose vectors are empty and whose scalars are zero.
    pub fn empty() -> Self {
        Self {
            flags: 0,
            min_sym_len: 0,
            max_sym_len: 0,
            blocks_num: 0,
            block_size: 0,
            span: 0,
            sparse_index: Vec::new(),
            block_length: Vec::new(),
            base64: Vec::new(),
            lowest_sym: Vec::new(),
            symlen: Vec::new(),
            btree: Vec::new(),
            pieces: [0; TB_PIECES],
            group_idx: [0; TB_PIECES + 1],
            group_len: [0; TB_PIECES + 1],
            map_idx: [0; 4],
            data_offset: 0,
        }
    }

    /// Used for testing whether every position stores one identical value.
    ///
    /// # Returns
    ///
    /// `true` when the single-value flag is set.
    pub fn is_single_value(&self) -> bool {
        self.flags & FLAG_SINGLE_VALUE != 0
    }

    /// Used for locating the block and in-block offset of a value index.
    ///
    /// Starts from the sparse index entry nearest the requested index and
    /// walks the per-block value counts to the exact block.
    ///
    /// # Arguments
    ///
    /// * `idx` - position index produced by the position encoder
    ///
    /// # Returns
    ///
    /// The block number and the value offset inside that block.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when the sparse index or block lengths are
    /// inconsistent with the requested index.
    pub fn locate(&self, idx: u64) -> Result<(u32, u32), SyzygyError> {
        if self.span == 0 {
            return Err(SyzygyError::new("zero span in pairs data"));
        }
        let sparse = usize::try_from(idx / self.span)
            .map_err(|_| SyzygyError::new("sparse index exceeds address space"))?;
        let &(block, offset) = self
            .sparse_index
            .get(sparse)
            .ok_or_else(|| SyzygyError::new("sparse index out of range"))?;
        let mut block = i64::from(block);
        // Signed distance from the sparse entry's reference value index.
        let in_span = i64::try_from(idx % self.span)
            .map_err(|_| SyzygyError::new("span residue exceeds i64"))?;
        let half_span =
            i64::try_from(self.span / 2).map_err(|_| SyzygyError::new("span exceeds i64"))?;
        let mut offset = i64::from(offset) + in_span - half_span;

        // Walk backward or forward until the offset lies inside the block.
        // Each step consumes at least one value, so the walk is bounded by
        // the number of blocks.
        loop {
            if offset < 0 {
                block -= 1;
                let length = self.block_value_count(block)?;
                offset += length;
            } else {
                let length = self.block_value_count(block)?;
                if offset >= length {
                    offset -= length;
                    block += 1;
                } else {
                    break;
                }
            }
        }
        let block =
            u32::try_from(block).map_err(|_| SyzygyError::new("block walk left the table"))?;
        let offset =
            u32::try_from(offset).map_err(|_| SyzygyError::new("negative block offset"))?;
        Ok((block, offset))
    }

    /// Used for reading the value count of one block during location.
    ///
    /// # Arguments
    ///
    /// * `block` - candidate block number from the sparse-index walk
    ///
    /// # Returns
    ///
    /// The number of values stored in the block (`block_length + 1`).
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when the block number leaves the table.
    fn block_value_count(&self, block: i64) -> Result<i64, SyzygyError> {
        let index =
            usize::try_from(block).map_err(|_| SyzygyError::new("block walk went below zero"))?;
        let length = self
            .block_length
            .get(index)
            .ok_or_else(|| SyzygyError::new("block walk ran past the last block"))?;
        Ok(i64::from(*length) + 1)
    }

    /// Used for decoding the value at a given offset inside one block.
    ///
    /// Reads the canonical Huffman stream big-endian from the start of the
    /// block, skips whole symbols until the requested offset falls inside
    /// one, and then expands that symbol through the Recursive Pairing tree
    /// down to the stored leaf value.
    ///
    /// # Arguments
    ///
    /// * `block` - raw bytes of exactly one compressed block
    /// * `offset` - value offset inside the block from [`Self::locate`]
    ///
    /// # Returns
    ///
    /// The decoded raw value (a WDL code or an unmapped DTZ value).
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when the stream is exhausted or inconsistent
    /// with the declared Huffman shape.
    pub fn decode(&self, block: &[u8], offset: u32) -> Result<u16, SyzygyError> {
        let mut offset = i64::from(offset);
        let mut position = 0usize;
        let mut buffer = read_be_u64(block, &mut position)?;
        let mut buffer_bits = 64i32;
        let min_len = u32::from(self.min_sym_len);

        // Skip whole symbols until `offset` falls inside the next one. Each
        // iteration consumes at least one stream bit, so the loop is bounded
        // by the bit length of the block.
        let max_iterations = block.len().saturating_mul(8) + 64;
        let mut symbol = 0u16;
        let mut found = false;
        for _ in 0..=max_iterations {
            let mut len = 0usize;
            while len < self.base64.len() && buffer < self.base64[len] {
                len += 1;
            }
            let shift = 64u32
                .checked_sub(u32::try_from(len).map_err(|_| SyzygyError::new("length overflow"))?)
                .and_then(|value| value.checked_sub(min_len))
                .filter(|value| *value < 64 && len < self.base64.len())
                .ok_or_else(|| SyzygyError::new("symbol length exceeds Huffman table"))?;
            symbol = u16::try_from((buffer.wrapping_sub(self.base64[len])) >> shift)
                .map_err(|_| SyzygyError::new("decoded symbol exceeds 16 bits"))?;
            symbol = symbol.wrapping_add(
                *self
                    .lowest_sym
                    .get(len)
                    .ok_or_else(|| SyzygyError::new("lowest-symbol table too short"))?,
            );
            let expansion = i64::from(
                *self
                    .symlen
                    .get(usize::from(symbol))
                    .ok_or_else(|| SyzygyError::new("symbol outside symlen table"))?,
            ) + 1;
            if offset < expansion {
                found = true;
                break;
            }
            offset -= expansion;
            let consumed = i32::try_from(len).expect("length bounded by Huffman table")
                + i32::from(self.min_sym_len);
            buffer <<= consumed;
            buffer_bits -= consumed;
            if buffer_bits <= 32 {
                buffer_bits += 32;
                let refill = read_be_u32(block, &mut position)?;
                buffer |= u64::from(refill)
                    << (64 - u32::try_from(buffer_bits).expect("bits stay positive"));
            }
        }
        if !found {
            return Err(SyzygyError::new("block ended before the requested value"));
        }

        // Expand the symbol down the pairing tree; each step strictly
        // reduces the symbol's expansion count, so the walk is bounded.
        for _ in 0..=usize::from(u8::MAX) + 1 {
            let entry = *self
                .btree
                .get(usize::from(symbol))
                .ok_or_else(|| SyzygyError::new("symbol outside pairing tree"))?;
            if self.symlen[usize::from(symbol)] == 0 {
                return Ok(entry.left());
            }
            let left = entry.left();
            let left_expansion = i64::from(
                *self
                    .symlen
                    .get(usize::from(left))
                    .ok_or_else(|| SyzygyError::new("left child outside symlen table"))?,
            ) + 1;
            if offset < left_expansion {
                symbol = left;
            } else {
                offset -= left_expansion;
                symbol = entry.right();
            }
        }
        Err(SyzygyError::new("pairing tree walk did not terminate"))
    }
}

/// Used for reading a big-endian 64-bit word from a block stream.
///
/// # Arguments
///
/// * `block` - raw block bytes
/// * `position` - stream cursor, advanced by eight on success
///
/// # Returns
///
/// The decoded word.
///
/// # Errors
///
/// Returns [`SyzygyError`] when fewer than eight bytes remain.
fn read_be_u64(block: &[u8], position: &mut usize) -> Result<u64, SyzygyError> {
    let bytes = block
        .get(*position..*position + 8)
        .ok_or_else(|| SyzygyError::new("block stream exhausted"))?;
    *position += 8;
    Ok(u64::from_be_bytes(
        bytes.try_into().expect("slice has eight bytes"),
    ))
}

/// Used for reading a big-endian 32-bit refill word from a block stream.
///
/// # Arguments
///
/// * `block` - raw block bytes
/// * `position` - stream cursor, advanced by four on success
///
/// # Returns
///
/// The decoded word.
///
/// # Errors
///
/// Returns [`SyzygyError`] when fewer than four bytes remain.
fn read_be_u32(block: &[u8], position: &mut usize) -> Result<u32, SyzygyError> {
    let bytes = block
        .get(*position..*position + 4)
        .ok_or_else(|| SyzygyError::new("block stream exhausted"))?;
    *position += 4;
    Ok(u32::from_be_bytes(
        bytes.try_into().expect("slice has four bytes"),
    ))
}

/// Used for computing the padded canonical-code boundaries per code length.
///
/// Rebuilds the `base64` table from the per-length lowest symbols: shorter
/// codes receive numerically higher 64-bit padded boundaries, so the decoder
/// can find a code's length by comparing the padded stream against the
/// boundaries in ascending length order.
///
/// # Arguments
///
/// * `lowest_sym` - lowest symbol per code length, shortest first
/// * `min_sym_len` - length in bits of the shortest code
///
/// # Returns
///
/// One padded boundary per code length.
///
/// # Errors
///
/// Returns [`SyzygyError`] when the declared lengths cannot form a
/// canonical code (boundary underflow or shift overflow).
pub(crate) fn build_base64(lowest_sym: &[u16], min_sym_len: u8) -> Result<Vec<u64>, SyzygyError> {
    let mut base64 = vec![0u64; lowest_sym.len()];
    if base64.is_empty() {
        return Ok(base64);
    }
    for i in (0..base64.len().saturating_sub(1)).rev() {
        let sum = base64[i + 1]
            .checked_add(u64::from(lowest_sym[i]))
            .and_then(|value| value.checked_sub(u64::from(lowest_sym[i + 1])))
            .ok_or_else(|| SyzygyError::new("canonical code boundaries underflow"))?;
        base64[i] = sum / 2;
        if base64[i].checked_mul(2).map_or(true, |x| x < base64[i + 1]) {
            return Err(SyzygyError::new("canonical code boundaries not monotone"));
        }
    }
    for (i, boundary) in base64.iter_mut().enumerate() {
        let shift = 64u32
            .checked_sub(u32::try_from(i).map_err(|_| SyzygyError::new("code length overflow"))?)
            .and_then(|value| value.checked_sub(u32::from(min_sym_len)))
            .ok_or_else(|| SyzygyError::new("code length exceeds 64 bits"))?;
        if shift >= 64 {
            return Err(SyzygyError::new("code length underflows the pad"));
        }
        *boundary <<= shift;
    }
    Ok(base64)
}

/// Used for computing every symbol's expansion count iteratively.
///
/// Walks the Recursive Pairing tree with an explicit stack instead of
/// recursion, so hostile trees cannot overflow the call stack; cycles and
/// out-of-range children surface as errors.
///
/// # Arguments
///
/// * `btree` - packed pairing-tree entries, one per symbol
///
/// # Returns
///
/// The number of values (minus one) each symbol expands to.
///
/// # Errors
///
/// Returns [`SyzygyError`] on child indices outside the tree, on cyclic
/// trees, or when an expansion count exceeds 255.
pub(crate) fn build_symlen(btree: &[LrEntry]) -> Result<Vec<u8>, SyzygyError> {
    let count = btree.len();
    let mut symlen = vec![0u8; count];
    let mut state = vec![WalkState::Unvisited; count];

    for root in 0..count {
        if state[root] != WalkState::Unvisited {
            continue;
        }
        let mut stack = vec![root];
        while let Some(&symbol) = stack.last() {
            match state[symbol] {
                WalkState::Done => {
                    stack.pop();
                }
                WalkState::InProgress => {
                    // Both children resolved; combine them.
                    stack.pop();
                    let entry = btree[symbol];
                    let left = usize::from(entry.left());
                    let right = usize::from(entry.right());
                    let combined = u16::from(symlen[left]) + u16::from(symlen[right]) + 1;
                    symlen[symbol] = u8::try_from(combined)
                        .map_err(|_| SyzygyError::new("symbol expansion exceeds 255"))?;
                    state[symbol] = WalkState::Done;
                }
                WalkState::Unvisited => {
                    let entry = btree[symbol];
                    if entry.right() == 0x0FFF {
                        symlen[symbol] = 0;
                        state[symbol] = WalkState::Done;
                        stack.pop();
                        continue;
                    }
                    let left = usize::from(entry.left());
                    let right = usize::from(entry.right());
                    if left >= count || right >= count {
                        return Err(SyzygyError::new("pairing tree child out of range"));
                    }
                    state[symbol] = WalkState::InProgress;
                    for child in [left, right] {
                        match state[child] {
                            WalkState::Unvisited => stack.push(child),
                            WalkState::InProgress => {
                                return Err(SyzygyError::new("pairing tree contains a cycle"));
                            }
                            WalkState::Done => {}
                        }
                    }
                }
            }
        }
    }
    Ok(symlen)
}

/// Traversal state of one symbol during the iterative expansion walk.
#[derive(Clone, Copy, Eq, PartialEq)]
enum WalkState {
    /// Used for marking symbols not yet reached.
    Unvisited,
    /// Used for marking symbols whose children are being resolved; seeing
    /// one as a child again proves a cycle.
    InProgress,
    /// Used for marking symbols with a final expansion count.
    Done,
}
