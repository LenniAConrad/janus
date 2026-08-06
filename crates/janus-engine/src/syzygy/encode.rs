//! Deterministic Syzygy position-encoding tables.
//!
//! All tables are rebuilt from first principles whenever a [`EncodeTables`]
//! value is constructed, so their content depends only on this source file.
//! Squares in this module use the Syzygy convention: `a1 = 0`, `b1 = 1`, ...
//! `h8 = 63`, i.e. `square = 8 * rank + file` with rank `0` at White's back
//! rank. This differs from `janus_core`'s `a8 = 0` layout; the probing layer
//! converts between the two.

/// Position-encoding lookup tables shared by every probe.
///
/// The tables mirror the mathematical definitions of the Syzygy index
/// encoding: binomial coefficients, the king-pair table over the `a1-d1-d4`
/// triangle, the below-diagonal square maps, and the leading-pawn tables per
/// file. All fields are populated once by [`EncodeTables::new`] and never
/// mutated afterwards.
pub(crate) struct EncodeTables {
    /// Used for choosing `k` squares out of `n`: `binomial[k][n]` is `C(n, k)`
    /// for `k` in `0..6` following Pascal's rule.
    pub binomial: [[u64; 64]; 6],
    /// Used for ranking pawn squares `a2..h7`: higher values are nearer the
    /// board edge and lower ranks, selecting the leading pawn.
    pub map_pawns: [u64; 64],
    /// Used for encoding a square strictly below the `a1-h8` diagonal to
    /// `0..28`.
    pub map_b1h1h7: [u64; 64],
    /// Used for encoding a square of the `a1-d1-d4` triangle to `0..10`, with
    /// the diagonal squares mapped last.
    pub map_a1d1d4: [u64; 64],
    /// Used for encoding the 462 legal, canonical two-king placements,
    /// indexed by the first king's [`Self::map_a1d1d4`] code and the second
    /// king's square.
    pub map_kk: [[u64; 64]; 10],
    /// Used for indexing the leading-pawn configuration by pawn count `1..=5`
    /// and the leading pawn's square.
    pub lead_pawn_idx: [[u64; 64]; 6],
    /// Used for sizing the leading-pawn index space by pawn count `1..=5`
    /// and the leading pawn's file `a..=d`.
    pub lead_pawns_size: [[u64; 4]; 6],
}

/// Used for computing a square's offset from the `a1-h8` diagonal.
///
/// # Arguments
///
/// * `square` - Syzygy square index in `0..64`
///
/// # Returns
///
/// `rank - file`; zero on the diagonal, negative below it.
pub(crate) fn off_a1h8(square: usize) -> i32 {
    let rank = i32::try_from(square / 8).expect("square rank fits i32");
    let file = i32::try_from(square % 8).expect("square file fits i32");
    rank - file
}

/// Used for mirroring a square across the vertical board axis.
///
/// # Arguments
///
/// * `square` - Syzygy square index in `0..64`
///
/// # Returns
///
/// The square with its file replaced by `7 - file`.
pub(crate) fn flip_file(square: usize) -> usize {
    square ^ 7
}

/// Used for mirroring a square across the horizontal board axis.
///
/// # Arguments
///
/// * `square` - Syzygy square index in `0..64`
///
/// # Returns
///
/// The square with its rank replaced by `7 - rank`.
pub(crate) fn flip_rank(square: usize) -> usize {
    square ^ 0x38
}

/// Used for computing a file's distance from the nearest board edge.
///
/// # Arguments
///
/// * `file` - file index in `0..8`
///
/// # Returns
///
/// `min(file, 7 - file)`, always in `0..4`.
pub(crate) fn edge_distance(file: usize) -> usize {
    file.min(7 - file)
}

/// Used for computing the king-move target set of one square.
///
/// # Arguments
///
/// * `square` - Syzygy square index in `0..64`
///
/// # Returns
///
/// Bitboard over Syzygy square indices holding every adjacent square.
fn king_attacks(square: usize) -> u64 {
    let rank = square / 8;
    let file = square % 8;
    let mut attacks = 0u64;
    for rank_delta in -1i32..=1 {
        for file_delta in -1i32..=1 {
            if rank_delta == 0 && file_delta == 0 {
                continue;
            }
            let to_rank = i32::try_from(rank).expect("rank fits i32") + rank_delta;
            let to_file = i32::try_from(file).expect("file fits i32") + file_delta;
            if (0..8).contains(&to_rank) && (0..8).contains(&to_file) {
                let to = usize::try_from(to_rank * 8 + to_file).expect("square is non-negative");
                attacks |= 1u64 << to;
            }
        }
    }
    attacks
}

impl EncodeTables {
    /// Used for building every encoding table deterministically.
    ///
    /// The construction follows the published Syzygy index definitions; unit
    /// tests pin golden values such as the 462 king-pair codes and the
    /// per-file leading-pawn sizes.
    ///
    /// # Returns
    ///
    /// Fully populated, immutable encoding tables.
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let mut tables = Self {
            binomial: [[0; 64]; 6],
            map_pawns: [0; 64],
            map_b1h1h7: [0; 64],
            map_a1d1d4: [0; 64],
            map_kk: [[0; 64]; 10],
            lead_pawn_idx: [[0; 64]; 6],
            lead_pawns_size: [[0; 4]; 6],
        };

        // Squares strictly below the a1-h8 diagonal enumerate to 0..28.
        let mut code = 0u64;
        for square in 0..64 {
            if off_a1h8(square) < 0 {
                tables.map_b1h1h7[square] = code;
                code += 1;
            }
        }

        // The a1-d1-d4 triangle enumerates its six below-diagonal squares
        // first and its four diagonal squares last.
        let mut diagonal = Vec::new();
        code = 0;
        for square in 0..=27 {
            if square % 8 <= 3 {
                if off_a1h8(square) < 0 {
                    tables.map_a1d1d4[square] = code;
                    code += 1;
                } else if off_a1h8(square) == 0 {
                    diagonal.push(square);
                }
            }
        }
        for square in diagonal {
            tables.map_a1d1d4[square] = code;
            code += 1;
        }

        // All 462 legal canonical king pairs: the first king inside the
        // triangle, and when it sits on the long diagonal the second king may
        // not be above that diagonal. Pairs with both kings on the diagonal
        // are enumerated last.
        let mut both_on_diagonal = Vec::new();
        code = 0;
        for index in 0..10 {
            for first in 0..=27usize {
                // Only the square actually mapped to `index`; square b1 is
                // the unique holder of code zero.
                if tables.map_a1d1d4[first] != index as u64 || (index == 0 && first != 1) {
                    continue;
                }
                if first % 8 > 3 {
                    continue;
                }
                for second in 0..64usize {
                    if (king_attacks(first) | (1u64 << first)) & (1u64 << second) != 0 {
                        continue; // Kings touch or coincide.
                    }
                    if off_a1h8(first) == 0 && off_a1h8(second) > 0 {
                        continue; // First on the diagonal, second above it.
                    }
                    if off_a1h8(first) == 0 && off_a1h8(second) == 0 {
                        both_on_diagonal.push((index, second));
                    } else {
                        tables.map_kk[index][second] = code;
                        code += 1;
                    }
                }
            }
        }
        for (index, second) in both_on_diagonal {
            tables.map_kk[index][second] = code;
            code += 1;
        }

        // Binomial coefficients by Pascal's rule, for up to five chosen
        // squares out of sixty-three.
        tables.binomial[0][0] = 1;
        for n in 1..64 {
            for k in 0..6usize.min(n + 1) {
                let up = if k > 0 {
                    tables.binomial[k - 1][n - 1]
                } else {
                    0
                };
                let left = if k < n { tables.binomial[k][n - 1] } else { 0 };
                tables.binomial[k][n] = up + left;
            }
        }

        // Pawn ranking plus leading-pawn index and size tables. The pawn on
        // the highest `map_pawns` value leads; remaining squares available to
        // other pawns shrink by two per rank step because of file mirroring.
        let mut available: u64 = 47;
        for lead_count in 1..=5usize {
            for file in 0..4usize {
                let mut index = 0u64;
                for rank in 1..=6usize {
                    let square = 8 * rank + file;
                    if lead_count == 1 {
                        tables.map_pawns[square] = available;
                        available -= 1;
                        tables.map_pawns[flip_file(square)] = available;
                        available = available.wrapping_sub(1);
                    }
                    tables.lead_pawn_idx[lead_count][square] = index;
                    let rank_pawns = usize::try_from(tables.map_pawns[square])
                        .expect("pawn map codes stay below 48");
                    index += tables.binomial[lead_count - 1][rank_pawns];
                }
                tables.lead_pawns_size[lead_count][file] = index;
            }
        }

        tables
    }
}
