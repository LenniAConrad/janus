//! Color and piece value types used by the board and engine layers.
//!
//! [`Color`], [`PieceKind`], and [`Piece`] define the stable index contracts
//! (colors `0..2`, kinds `0..6`, combined pieces `0..12`) that the position's
//! bitboard and mailbox storage rely on, alongside the FEN and UCI character
//! conversions for pieces and promotions.

use core::fmt;

/// The two players of a chess game.
///
/// Variant order fixes the stable index contract exposed by
/// [`Color::index`]: white first, black second.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Color {
    /// Used for indicating the player whose pawns advance toward rank eight.
    White,
    /// Used for indicating the player whose pawns advance toward rank one.
    Black,
}

impl Color {
    /// Used for retrieving the other player.
    ///
    /// # Returns
    ///
    /// [`Color::Black`] for white and [`Color::White`] for black.
    #[inline]
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    /// Used for retrieving the stable array index for this color.
    ///
    /// White maps to `0` and black maps to `1`. Position storage relies on this
    /// ordering, so it is part of the crate's representation contract.
    ///
    /// # Returns
    ///
    /// `0` for white, `1` for black.
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Self::White => 0,
            Self::Black => 1,
        }
    }
}

impl fmt::Display for Color {
    /// Used for writing `white` or `black`.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter for the color name
    ///
    /// # Returns
    ///
    /// Result of writing the lowercase color name to `formatter`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::White => "white",
            Self::Black => "black",
        })
    }
}

/// A color-independent chessman type.
///
/// The discriminants `0..=5` run pawn through king and double as the stable
/// per-color bitboard indices returned by [`PieceKind::index`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum PieceKind {
    /// Used for indicating a pawn (discriminant `0`).
    Pawn = 0,
    /// Used for indicating a knight (discriminant `1`).
    Knight = 1,
    /// Used for indicating a bishop (discriminant `2`).
    Bishop = 2,
    /// Used for indicating a rook (discriminant `3`).
    Rook = 3,
    /// Used for indicating a queen (discriminant `4`).
    Queen = 4,
    /// Used for indicating a king (discriminant `5`).
    King = 5,
}

impl PieceKind {
    /// Used for iterating all piece kinds in their stable bitboard-index
    /// order.
    ///
    /// The order matches the discriminants: pawn, knight, bishop, rook,
    /// queen, king.
    pub const ALL: [Self; 6] = [
        Self::Pawn,
        Self::Knight,
        Self::Bishop,
        Self::Rook,
        Self::Queen,
        Self::King,
    ];

    /// Used for retrieving the stable per-color bitboard index in `0..6`.
    ///
    /// # Returns
    ///
    /// This kind's discriminant widened to `usize`.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Used for decoding the three-bit promotion payload stored in a
    /// [`crate::Move`].
    ///
    /// # Arguments
    ///
    /// * `code` - three-bit promotion payload extracted from a move
    ///
    /// # Returns
    ///
    /// `Some` knight, bishop, rook, or queen for codes `1..=4`; `None` for
    /// any other code, including `0`, which encodes the absence of a
    /// promotion.
    pub(crate) const fn from_promotion_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Knight),
            2 => Some(Self::Bishop),
            3 => Some(Self::Rook),
            4 => Some(Self::Queen),
            _ => None,
        }
    }

    /// Used for retrieving the lowercase UCI suffix for a promotable piece
    /// kind.
    ///
    /// # Returns
    ///
    /// `Some('n')`, `Some('b')`, `Some('r')`, or `Some('q')` for knight,
    /// bishop, rook, or queen; `None` for pawn and king, which are not valid
    /// promotion targets.
    pub(crate) const fn promotion_char(self) -> Option<char> {
        match self {
            Self::Knight => Some('n'),
            Self::Bishop => Some('b'),
            Self::Rook => Some('r'),
            Self::Queen => Some('q'),
            _ => None,
        }
    }

    /// Used for parsing an ASCII UCI promotion suffix, accepting either case.
    ///
    /// # Arguments
    ///
    /// * `ch` - candidate promotion suffix character
    ///
    /// # Returns
    ///
    /// `Some` knight, bishop, rook, or queen for `n`, `b`, `r`, or `q` in
    /// either case; `None` for any other character.
    pub(crate) const fn from_promotion_char(ch: char) -> Option<Self> {
        match ch {
            'n' | 'N' => Some(Self::Knight),
            'b' | 'B' => Some(Self::Bishop),
            'r' | 'R' => Some(Self::Rook),
            'q' | 'Q' => Some(Self::Queen),
            _ => None,
        }
    }
}

/// A piece combining a color and a color-independent kind.
///
/// The pair maps onto the twelve-bitboard layout through [`Piece::index`]:
/// white pawn through king occupy `0..6` and black pawn through king occupy
/// `6..12`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Piece {
    /// Used for identifying the player that owns this piece.
    pub color: Color,
    /// Used for identifying the piece's color-independent kind.
    pub kind: PieceKind,
}

impl Piece {
    /// Used for combining a color and kind into a piece value.
    ///
    /// # Arguments
    ///
    /// * `color` - owning player
    /// * `kind` - color-independent piece kind
    ///
    /// # Returns
    ///
    /// Piece value holding both components.
    #[inline]
    pub const fn new(color: Color, kind: PieceKind) -> Self {
        Self { color, kind }
    }

    /// Used for retrieving the index in the twelve bitboards: white pawn
    /// through king, then black pawn through king.
    ///
    /// # Returns
    ///
    /// Index in `0..12`, computed as the color index times six plus the kind
    /// index.
    #[inline]
    pub const fn index(self) -> usize {
        self.color.index() * 6 + self.kind.index()
    }

    /// Used for decoding a mailbox value in white-then-black,
    /// pawn-through-king order.
    ///
    /// # Arguments
    ///
    /// * `index` - candidate mailbox piece value
    ///
    /// # Returns
    ///
    /// `Some(piece)` for indices `0..12`, where `0..6` are the white pieces
    /// and `6..12` the black pieces; `None` otherwise.
    pub(crate) const fn from_index(index: u8) -> Option<Self> {
        if index >= 12 {
            return None;
        }
        let color = if index < 6 {
            Color::White
        } else {
            Color::Black
        };
        let kind = match index % 6 {
            0 => PieceKind::Pawn,
            1 => PieceKind::Knight,
            2 => PieceKind::Bishop,
            3 => PieceKind::Rook,
            4 => PieceKind::Queen,
            _ => PieceKind::King,
        };
        Some(Self { color, kind })
    }

    /// Used for decoding one FEN piece-placement character.
    ///
    /// Uppercase letters produce white pieces and all other accepted
    /// characters produce black pieces.
    ///
    /// # Arguments
    ///
    /// * `ch` - candidate FEN piece letter
    ///
    /// # Returns
    ///
    /// `Some(piece)` for `p`, `n`, `b`, `r`, `q`, or `k` in either case;
    /// `None` for any other character.
    pub(crate) const fn from_fen(ch: char) -> Option<Self> {
        let color = if ch.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let kind = match ch.to_ascii_lowercase() {
            'p' => PieceKind::Pawn,
            'n' => PieceKind::Knight,
            'b' => PieceKind::Bishop,
            'r' => PieceKind::Rook,
            'q' => PieceKind::Queen,
            'k' => PieceKind::King,
            _ => return None,
        };
        Some(Self { color, kind })
    }

    /// Used for encoding this piece as its case-sensitive FEN character.
    ///
    /// # Returns
    ///
    /// One of `p`, `n`, `b`, `r`, `q`, or `k` chosen by kind, uppercased for
    /// white pieces and lowercase for black pieces.
    pub(crate) const fn fen(self) -> char {
        let ch = match self.kind {
            PieceKind::Pawn => 'p',
            PieceKind::Knight => 'n',
            PieceKind::Bishop => 'b',
            PieceKind::Rook => 'r',
            PieceKind::Queen => 'q',
            PieceKind::King => 'k',
        };
        if matches!(self.color, Color::White) {
            ch.to_ascii_uppercase()
        } else {
            ch
        }
    }
}
