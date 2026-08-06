//! Validated board squares and algebraic-coordinate conversion.
//!
//! Squares use the `ChessRTK` layout: index `0` is `a8`, indices grow along
//! each rank toward the h-file, and index `63` is `h1`. Every constructor
//! validates its coordinates, so a [`Square`] value always refers to a real
//! board coordinate and its accessors never need range checks.

use crate::CoreError;
use core::fmt;
use core::str::FromStr;

/// A validated board square in `ChessRTK` order (`A8 = 0`, `H1 = 63`).
///
/// The wrapped index is always inside `0..64`: each constructor rejects
/// out-of-range coordinates, so files, rows, ranks, and bitboard masks can be
/// decoded without further checks.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Square(u8);

impl Square {
    /// Used for referring to square `a8`, the first square in `ChessRTK`
    /// index order.
    pub const A8: Self = Self(0);
    /// Used for referring to square `e8`, the orthodox black king home
    /// square.
    pub const E8: Self = Self(4);
    /// Used for referring to square `h8`, the orthodox black kingside-rook
    /// home square.
    pub const H8: Self = Self(7);
    /// Used for referring to square `a1`, the orthodox white queenside-rook
    /// home square.
    pub const A1: Self = Self(56);
    /// Used for referring to square `c1`, the white queenside-castling king
    /// destination.
    pub const C1: Self = Self(58);
    /// Used for referring to square `d1`, the white queenside-castling rook
    /// destination.
    pub const D1: Self = Self(59);
    /// Used for referring to square `e1`, the orthodox white king home
    /// square.
    pub const E1: Self = Self(60);
    /// Used for referring to square `f1`, the white kingside-castling rook
    /// destination.
    pub const F1: Self = Self(61);
    /// Used for referring to square `g1`, the white kingside-castling king
    /// destination.
    pub const G1: Self = Self(62);
    /// Used for referring to square `h1`, the last square in `ChessRTK` index
    /// order.
    pub const H1: Self = Self(63);
    /// Used for referring to square `c8`, the black queenside-castling king
    /// destination.
    pub const C8: Self = Self(2);
    /// Used for referring to square `d8`, the black queenside-castling rook
    /// destination.
    pub const D8: Self = Self(3);
    /// Used for referring to square `f8`, the black kingside-castling rook
    /// destination.
    pub const F8: Self = Self(5);
    /// Used for referring to square `g8`, the black kingside-castling king
    /// destination.
    pub const G8: Self = Self(6);

    /// Used for creating a square from a `ChessRTK` board index.
    ///
    /// # Arguments
    ///
    /// * `index` - candidate board index in `ChessRTK` order
    ///
    /// # Returns
    ///
    /// `Some(square)` when `index` is inside `0..64`, `None` otherwise.
    #[inline]
    pub const fn new(index: u8) -> Option<Self> {
        if index < 64 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// Used for creating a square from a zero-based file and top-origin row.
    ///
    /// Both coordinates must be inside `0..8`; row `0` is rank eight,
    /// matching the top-origin `ChessRTK` layout.
    ///
    /// # Arguments
    ///
    /// * `file` - zero-based file, where `a` is `0` and `h` is `7`
    /// * `row_from_eighth` - zero-based row counted downwards from rank eight
    ///
    /// # Returns
    ///
    /// `Some(square)` when both coordinates are inside `0..8`, `None`
    /// otherwise.
    #[inline]
    pub const fn from_file_row(file: u8, row_from_eighth: u8) -> Option<Self> {
        if file < 8 && row_from_eighth < 8 {
            Some(Self(row_from_eighth * 8 + file))
        } else {
            None
        }
    }

    /// Used for creating a square from a zero-based file and human chess
    /// rank.
    ///
    /// `file` must be inside `0..8` and `rank` inside `1..=8`; rank `1` is
    /// the bottom rank from white's point of view.
    ///
    /// # Arguments
    ///
    /// * `file` - zero-based file, where `a` is `0` and `h` is `7`
    /// * `rank` - human chess rank in `1..=8`
    ///
    /// # Returns
    ///
    /// `Some(square)` when both coordinates are in range, `None` otherwise.
    #[inline]
    pub const fn from_file_rank(file: u8, rank: u8) -> Option<Self> {
        if file < 8 && rank >= 1 && rank <= 8 {
            Some(Self((8 - rank) * 8 + file))
        } else {
            None
        }
    }

    /// Used for retrieving the `ChessRTK` board index in `0..64`.
    ///
    /// # Returns
    ///
    /// Wrapped board index of this square.
    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Used for retrieving the zero-based file, where `a` is `0` and `h` is
    /// `7`.
    ///
    /// # Returns
    ///
    /// File component of the board index, in `0..8`.
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    /// Used for retrieving the zero-based row counted downwards from rank
    /// eight.
    ///
    /// Row `0` is rank eight and row `7` is rank one.
    ///
    /// # Returns
    ///
    /// Top-origin row component of the board index, in `0..8`.
    #[inline]
    pub const fn row(self) -> u8 {
        self.0 >> 3
    }

    /// Used for retrieving the human chess rank in `1..=8`.
    ///
    /// The rank is the top-origin row mirrored, so rank `8` corresponds to
    /// row `0`.
    ///
    /// # Returns
    ///
    /// Rank component of this square, in `1..=8`.
    #[inline]
    pub const fn rank(self) -> u8 {
        8 - self.row()
    }

    /// Used for producing a bitboard containing only this square.
    ///
    /// The set bit sits at this square's `ChessRTK` index inside the `u64`
    /// mask.
    ///
    /// # Returns
    ///
    /// Single-bit bitboard mask for this square.
    #[inline]
    pub const fn bit(self) -> u64 {
        1_u64 << self.0
    }

    /// Used for offsetting this square in file/top-origin-row coordinates.
    ///
    /// # Arguments
    ///
    /// * `file_delta` - signed file displacement, positive toward the h-file
    /// * `row_delta` - signed row displacement, positive toward rank one
    ///
    /// # Returns
    ///
    /// `Some(square)` for an on-board destination, `None` when the offset
    /// would leave the board.
    pub(crate) fn offset(self, file_delta: i8, row_delta: i8) -> Option<Self> {
        let file = self.file() as i8 + file_delta;
        let row = self.row() as i8 + row_delta;
        if (0..8).contains(&file) && (0..8).contains(&row) {
            Some(Self((row * 8 + file) as u8))
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    /// Used for writing the lowercase algebraic coordinate, such as `e4`.
    ///
    /// Both output bytes are always ASCII, so the internal UTF-8 conversion
    /// cannot fail.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter for the coordinate text
    ///
    /// # Returns
    ///
    /// Result of writing the two-character coordinate to `formatter`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = [b'a' + self.file(), b'0' + self.rank()];
        // Both bytes are guaranteed ASCII.
        let text = core::str::from_utf8(&bytes).expect("square is ASCII");
        formatter.write_str(text)
    }
}

impl FromStr for Square {
    /// Error type produced when the coordinate text is malformed.
    type Err = CoreError;

    /// Used for parsing a lowercase algebraic coordinate in `a1..=h8`.
    ///
    /// # Arguments
    ///
    /// * `text` - candidate two-character coordinate text
    ///
    /// # Returns
    ///
    /// Parsed square on success.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] unless `text` is exactly a lowercase file letter
    /// followed by a rank digit.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let bytes = text.as_bytes();
        if bytes.len() != 2
            || !(b'a'..=b'h').contains(&bytes[0])
            || !(b'1'..=b'8').contains(&bytes[1])
        {
            return Err(CoreError::new(format!("invalid square: {text}")));
        }
        Self::from_file_rank(bytes[0] - b'a', bytes[1] - b'0')
            .ok_or_else(|| CoreError::new(format!("invalid square: {text}")))
    }
}
