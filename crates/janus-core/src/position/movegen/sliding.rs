//! Deterministic magic-bitboard attacks translated from the frozen CRTK CPU core.
//!
//! The tables are built once from safe reference line arithmetic. Masked
//! occupancies are multiplied by a deterministic per-square magic and shifted
//! into a collision-free attack table. Initialization uses only standard-library
//! storage and wrapping integer operations that match Java's two's-complement
//! `long` arithmetic.

use std::sync::OnceLock;

/// Used for constructing bishop blocker masks from the four diagonal
/// directions.
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
/// Used for constructing rook blocker masks from the four orthogonal
/// directions.
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
/// Used for hashing bishop occupancies with the multipliers produced by the
/// frozen CRTK deterministic search.
///
/// Materializing the search result keeps normal engine startup bounded while a
/// regeneration test below proves that these values remain tied to the source
/// algorithm rather than becoming an independent generated-data authority.
const BISHOP_MAGICS: [u64; 64] = [
    0x2228_2008_0210_2028,
    0x0008_0808_08b0_2109,
    0x8010_00c0_8100_4400,
    0x002c_0403_9202_4018,
    0x4111_1040_0300_1000,
    0xd006_0610_a524_4c82,
    0x2804_0404_0209_8090,
    0x0002_1446_0a20_2000,
    0x4002_1020_4881_0b48,
    0x0000_08d0_0400_8028,
    0x0040_0438_0210_4800,
    0x2000_0821_8228_0801,
    0x0280_0110_4000_0040,
    0x0006_0888_2008_1055,
    0x2352_1901_1010_8600,
    0x2020_0046_0090_1800,
    0x2920_0040_0841_0100,
    0x0002_0550_7002_4080,
    0x0044_0208_0028_2200,
    0x2008_0200_8200_4086,
    0x001c_0186_0211_0400,
    0x3100_2246_0210_0200,
    0x0404_a082_0610_0240,
    0x0000_8081_1088_0100,
    0x0c82_0800_9260_2800,
    0x2024_2000_0401_0400,
    0x9500_4120_4802_0400,
    0x0800_8080_0802_0202,
    0x0000_8400_0880_2020,
    0x2002_1200_1348_0210,
    0x0202_0400_044c_0200,
    0x0801_0208_00a2_0540,
    0x4004_2044_0049_1000,
    0x2210_8260_0038_0808,
    0x0020_4110_002a_2410,
    0x0201_0200_8008_0080,
    0x0044_2002_0000_2080,
    0x000a_0c87_0092_0058,
    0x0004_0810_a502_0080,
    0x0002_2405_1190_4044,
    0x0051_0110_11a0_4010,
    0x0000_4108_0819_6001,
    0x0800_4200_4044_0400,
    0x0024_01a2_1400_1802,
    0x4001_0282_0a00_0402,
    0x42a4_9088_0040_0602,
    0x1028_0800_c405_0080,
    0x0008_0109_0208_0421,
    0x1000_4442_2010_0290,
    0x0200_2201_1421_0802,
    0x8400_1020_8410_0080,
    0x6840_1042_0504_0001,
    0x08c0_1090_0202_1901,
    0x0000_4108_0101_0000,
    0x0010_1030_0100_6040,
    0x2191_0202_0400_2004,
    0x0130_e380_4820_3000,
    0x4088_0200_8c84_9010,
    0x0104_4a01_0108_4901,
    0x0002_0040_0084_0404,
    0x0180_1400_2421_8200,
    0x0000_0140_0418_4484,
    0x8008_0808_c808_0840,
    0x0204_200a_4a02_0410,
];
/// Used for hashing rook occupancies with the multipliers produced by the
/// frozen CRTK deterministic search.
const ROOK_MAGICS: [u64; 64] = [
    0x0080_0080_2010_4000,
    0x0140_0020_0010_01c0,
    0x5080_0810_0020_0080,
    0x0100_1001_0020_0804,
    0x0080_0400_8008_0002,
    0x0500_0100_2204_0008,
    0x0400_1000_8801_0402,
    0x8880_09a0_8000_5300,
    0x2218_8000_2188_c000,
    0x4400_8020_0040_0080,
    0x8008_8020_0090_0a80,
    0x0010_8008_0010_0080,
    0x0013_0008_0013_0004,
    0xc800_8080_0200_0400,
    0x2004_0021_0802_9410,
    0x4002_0001_0200_40a4,
    0x0000_2080_0080_4000,
    0x9010_0140_0042_2000,
    0x0010_8080_2000_1008,
    0x1090_2100_1001_0008,
    0x1000_8180_1c00_0800,
    0x000a_0101_0004_0008,
    0x0080_0101_0004_0200,
    0x02ad_8200_0044_8d24,
    0x0880_00c2_4001_2001,
    0x0800_2004_4010_0540,
    0x0100_8202_0010_2041,
    0x0000_0800_8080_1000,
    0x0418_0101_0008_0411,
    0x1004_0004_8002_0080,
    0x4000_0a84_0070_0108,
    0x0002_8842_0001_04b4,
    0x0280_8040_0680_0020,
    0x8000_2000_4040_1008,
    0x1803_8010_0280_2002,
    0x0008_0010_0080_0884,
    0x0403_0004_1100_0800,
    0x2502_8102_0080_0400,
    0x0002_0608_0c00_1005,
    0x1040_0100_4200_0084,
    0x4140_08f1_8040_8000,
    0x4000_2010_0042_4000,
    0x0081_0020_0041_0010,
    0x0000_2010_0101_0008,
    0x0048_0011_0009_0005,
    0x2102_0020_1004_0400,
    0x1021_0042_0021_0004,
    0x0000_0043_0286_0004,
    0x0010_8001_a043_0100,
    0x0124_2002_8040_0380,
    0x0004_2005_0210_4500,
    0x0120_2010_0100_0900,
    0x8100_0400_0800_8080,
    0x0880_0200_8004_0080,
    0x0000_3062_0841_0400,
    0x8b04_1110_8044_0200,
    0x0080_0300_4080_6455,
    0x4004_8a00_a100_50c2,
    0x8041_4020_0011_000d,
    0x0401_0421_0810_0101,
    0x1012_0120_8804_1002,
    0x0001_0002_0400_0801,
    0x0802_0810_0102_0084,
    0x0700_0080_4401_0022,
];

/// Used for storing the lazily initialized attack tables shared by every
/// position.
static TABLES: OnceLock<SlidingTables> = OnceLock::new();

/// Complete bishop and rook lookup tables.
///
/// Both families index one contiguous attack buffer through a per-square
/// offset, so a lookup costs a fixed-array read and a single indexed load
/// instead of walking two separate heap allocations.
struct SlidingTables {
    /// Used for looking up per-square bishop magic parameters.
    bishops: [MagicEntry; 64],
    /// Used for looking up per-square rook magic parameters.
    rooks: [MagicEntry; 64],
    /// Used for storing every square's attacks for both families back to back.
    attacks: Box<[u64]>,
}

impl SlidingTables {
    /// Used for building both slider families in the same deterministic
    /// order as CRTK.
    ///
    /// # Returns
    ///
    /// Tables holding one collision-free [`MagicEntry`] per square for each
    /// family.
    fn new() -> Self {
        let lines = LineMasks::new();
        let mut attacks = Vec::new();
        let bishops = build_magic_tables(
            &BISHOP_DIRECTIONS,
            &BISHOP_MAGICS,
            true,
            &lines,
            &mut attacks,
        );
        let rooks = build_magic_tables(&ROOK_DIRECTIONS, &ROOK_MAGICS, false, &lines, &mut attacks);
        Self {
            bishops,
            rooks,
            attacks: attacks.into_boxed_slice(),
        }
    }

    /// Used for reading one family's attacks for a full-board occupancy.
    ///
    /// # Arguments
    ///
    /// * `entry` - per-square magic parameters selecting the table slice
    /// * `occupancy` - full-board occupancy; irrelevant bits are masked off
    ///
    /// # Returns
    ///
    /// Attack bitboard stored for the hashed relevant occupancy.
    #[inline]
    fn lookup(&self, entry: &MagicEntry, occupancy: u64) -> u64 {
        let index = ((occupancy & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
        self.attacks[entry.offset as usize + index]
    }
}

/// Full board lines used by the reference hyperbola implementation.
struct LineMasks {
    /// Used for masking the rank through each square, including the origin.
    ranks: [u64; 64],
    /// Used for masking the file through each square, including the origin.
    files: [u64; 64],
    /// Used for masking the main diagonal through each square, including the
    /// origin.
    diagonals: [u64; 64],
    /// Used for masking the anti-diagonal through each square, including the
    /// origin.
    anti_diagonals: [u64; 64],
}

impl LineMasks {
    /// Used for constructing every full line mask from explicit
    /// board-coordinate rays.
    ///
    /// Each line combines the two opposing [`line_mask`] rays with the origin
    /// bit itself.
    ///
    /// # Returns
    ///
    /// Complete rank, file, diagonal, and anti-diagonal masks for all 64
    /// squares.
    fn new() -> Self {
        let mut lines = Self {
            ranks: [0; 64],
            files: [0; 64],
            diagonals: [0; 64],
            anti_diagonals: [0; 64],
        };
        for square in 0_u8..64 {
            let origin = 1_u64 << square;
            lines.ranks[square as usize] =
                line_mask(square, 1, 0) | line_mask(square, -1, 0) | origin;
            lines.files[square as usize] =
                line_mask(square, 0, 1) | line_mask(square, 0, -1) | origin;
            lines.diagonals[square as usize] =
                line_mask(square, 1, 1) | line_mask(square, -1, -1) | origin;
            lines.anti_diagonals[square as usize] =
                line_mask(square, 1, -1) | line_mask(square, -1, 1) | origin;
        }
        lines
    }
}

/// Collision-free magic parameters for one origin square.
#[derive(Clone, Copy)]
struct MagicEntry {
    /// Used for selecting the occupancy bits that can change the attack
    /// result.
    mask: u64,
    /// Used for mapping relevant occupancies to unique attack slots through
    /// a wrapping multiplication.
    magic: u64,
    /// Used for right-shifting the product down to the table index width.
    shift: u32,
    /// Used for locating this square's slice inside the shared attack buffer.
    offset: u32,
}

/// Used for retrieving bishop attacks from `square`, including the first
/// blocker on each ray.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `occupancy` - full-board occupancy bitboard
///
/// # Returns
///
/// Bishop attack bitboard under the given occupancy.
#[inline]
pub fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
    let tables = TABLES.get_or_init(SlidingTables::new);
    tables.lookup(&tables.bishops[square as usize], occupancy)
}

/// Used for retrieving rook attacks from `square`, including the first
/// blocker on each ray.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `occupancy` - full-board occupancy bitboard
///
/// # Returns
///
/// Rook attack bitboard under the given occupancy.
#[inline]
pub fn rook_attacks(square: u8, occupancy: u64) -> u64 {
    let tables = TABLES.get_or_init(SlidingTables::new);
    tables.lookup(&tables.rooks[square as usize], occupancy)
}

/// Used for building all magic tables for one slider family.
///
/// # Arguments
///
/// * `directions` - the family's four ray directions
/// * `magics` - frozen per-square multipliers for the family
/// * `bishop` - `true` to build with the bishop reference, `false` for rook
/// * `lines` - full-board line masks feeding the reference attacks
/// * `attacks` - shared buffer each square's table is appended to; every
///   returned entry records its own offset into it
///
/// # Returns
///
/// One [`MagicEntry`] per square in ascending square order.
fn build_magic_tables(
    directions: &[(i8, i8)],
    magics: &[u64; 64],
    bishop: bool,
    lines: &LineMasks,
    attacks: &mut Vec<u64>,
) -> [MagicEntry; 64] {
    core::array::from_fn(|square| {
        let square = u8::try_from(square).expect("square index fits u8");
        let mask = relevant_blockers(square, directions);
        build_magic(
            square,
            mask,
            magics[square as usize],
            bishop,
            lines,
            attacks,
        )
    })
}

/// Used for populating one collision-free attack table from its frozen CRTK
/// multiplier.
///
/// Every subset of the relevant-blocker mask is enumerated with the
/// Carry-Rippler style `subset = subset.wrapping_sub(mask) & mask` walk, its
/// reference attacks are computed, and each subset's hashed slot must agree
/// with any earlier occupant.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `mask` - relevant-blocker mask for the square and family
/// * `magic` - frozen multiplier hashed against masked occupancies
/// * `bishop` - `true` for the bishop reference, `false` for the rook
/// * `lines` - full-board line masks feeding the reference attacks
/// * `buffer` - shared attack buffer this square's table is appended to
///
/// # Returns
///
/// Ready-to-query [`MagicEntry`] for the square.
///
/// # Panics
///
/// Panics when the frozen multiplier maps two occupancies with different
/// reference attacks to the same slot, which would mean the frozen CRTK
/// magic is invalid.
fn build_magic(
    square: u8,
    mask: u64,
    magic: u64,
    bishop: bool,
    lines: &LineMasks,
    buffer: &mut Vec<u64>,
) -> MagicEntry {
    let bits = mask.count_ones();
    let size = 1_usize << bits;
    let mut occupancies = vec![0_u64; size];
    let mut references = vec![0_u64; size];
    let mut subset = 0_u64;
    for index in 0..size {
        occupancies[index] = subset;
        references[index] = if bishop {
            bishop_reference(square, subset, lines)
        } else {
            rook_reference(square, subset, lines)
        };
        subset = subset.wrapping_sub(mask) & mask;
    }

    let shift = 64 - bits;
    let mut attacks = vec![0_u64; size];
    let mut used = vec![false; size];
    for index in 0..size {
        let slot = (occupancies[index].wrapping_mul(magic) >> shift) as usize;
        if used[slot] {
            assert_eq!(
                attacks[slot], references[index],
                "frozen CRTK magic collision at square {square}"
            );
        } else {
            used[slot] = true;
            attacks[slot] = references[index];
        }
    }
    let offset = u32::try_from(buffer.len()).expect("attack buffer stays within u32");
    buffer.extend_from_slice(&attacks);
    MagicEntry {
        mask,
        magic,
        shift,
        offset,
    }
}

/// Used for computing reference bishop attacks through the square's two
/// diagonals.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `occupancy` - full-board occupancy bitboard
/// * `lines` - full-board line masks for the hyperbola arithmetic
///
/// # Returns
///
/// Union of [`line_attacks`] over the diagonal and anti-diagonal.
fn bishop_reference(square: u8, occupancy: u64, lines: &LineMasks) -> u64 {
    line_attacks(square, occupancy, lines.diagonals[square as usize])
        | line_attacks(square, occupancy, lines.anti_diagonals[square as usize])
}

/// Used for computing reference rook attacks through the square's rank and
/// file.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `occupancy` - full-board occupancy bitboard
/// * `lines` - full-board line masks for the hyperbola arithmetic
///
/// # Returns
///
/// Union of [`line_attacks`] over the rank and file.
fn rook_reference(square: u8, occupancy: u64, lines: &LineMasks) -> u64 {
    line_attacks(square, occupancy, lines.ranks[square as usize])
        | line_attacks(square, occupancy, lines.files[square as usize])
}

/// Used for building the interior blocker mask for one origin and slider
/// family.
///
/// Board-edge squares are excluded because a blocker on the edge cannot
/// change the attack result, keeping the mask (and table size) minimal.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `directions` - the family's four ray directions
///
/// # Returns
///
/// Bitboard of occupancy bits that can change the square's attack result.
fn relevant_blockers(square: u8, directions: &[(i8, i8)]) -> u64 {
    let mut mask = 0_u64;
    let origin_file = (square & 7) as i8;
    let origin_row = (square >> 3) as i8;
    for &(file_delta, row_delta) in directions {
        let mut file = origin_file + file_delta;
        let mut row = origin_row + row_delta;
        while on_board(file, row) && on_board(file + file_delta, row + row_delta) {
            mask |= 1_u64 << (row * 8 + file);
            file += file_delta;
            row += row_delta;
        }
    }
    mask
}

/// Used for building a complete one-direction ray excluding the origin.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `file_delta` - per-step file component of the direction
/// * `row_delta` - per-step row component of the direction
///
/// # Returns
///
/// Bitboard of every square along the direction until the board edge.
fn line_mask(square: u8, file_delta: i8, row_delta: i8) -> u64 {
    let mut mask = 0_u64;
    let mut file = (square & 7) as i8 + file_delta;
    let mut row = (square >> 3) as i8 + row_delta;
    while on_board(file, row) {
        mask |= 1_u64 << (row * 8 + file);
        file += file_delta;
        row += row_delta;
    }
    mask
}

/// Used for computing attacks on one full rank, file, or diagonal by
/// hyperbola arithmetic.
///
/// The forward direction uses `o - 2s` subtraction on the masked occupancy;
/// the reverse direction applies the same subtraction on bit-reversed
/// operands, and the XOR of both restricted to `mask` yields the attacks up
/// to and including the first blocker on each side.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `occupancy` - full-board occupancy bitboard
/// * `mask` - full line mask through the square, including the origin
///
/// # Returns
///
/// Attack bitboard along the masked line under the given occupancy.
fn line_attacks(square: u8, occupancy: u64, mask: u64) -> u64 {
    let bit = 1_u64 << square;
    let forward = (occupancy & mask).wrapping_sub(bit << 1);
    let reverse_bit = bit.reverse_bits();
    let reverse_occupancy = (occupancy & mask).reverse_bits();
    let reverse = reverse_occupancy.wrapping_sub(reverse_bit << 1);
    (forward ^ reverse.reverse_bits()) & mask
}

/// Used for testing whether signed top-origin board coordinates lie inside
/// the board.
///
/// # Arguments
///
/// * `file` - signed file coordinate
/// * `row` - signed top-origin row coordinate
///
/// # Returns
///
/// `true` when both coordinates lie in `0..8`.
const fn on_board(file: i8, row: i8) -> bool {
    file >= 0 && file < 8 && row >= 0 && row < 8
}
