//! Compact move encoding and UCI long-algebraic conversion.
//!
//! [`Move`] packs source square, destination square, and an optional
//! promotion kind into a single `u16` using the same bit layout as the
//! `ChessRTK` Java core, and converts to and from UCI long-algebraic text.
//! The [`NO_MOVE`] sentinel covers boundaries that need a primitive
//! "no move" value.

use crate::{CoreError, PieceKind, Square};
use core::fmt;
use core::str::FromStr;

/// Used for marking the absence of a move with a raw sentinel bit pattern.
///
/// This value is deliberately not a valid [`Move`]; [`Move::from_raw`]
/// rejects it. Use it only at serialized or table boundaries that require a
/// primitive sentinel.
pub const NO_MOVE: u16 = u16::MAX;

/// A compact `ChessRTK`-compatible move value stored in a `u16`.
///
/// Bits `0..=5` encode the source square, bits `6..=11` the destination, and
/// bits `12..=14` the optional promotion kind; bit `15` is reserved and always
/// clear in a valid move. Whether a move is legal depends on a
/// [`crate::Position`]; this type validates only its binary shape.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Move(u16);

impl Move {
    /// Used for encoding source, destination, and optional promotion into a
    /// compact 16-bit move value.
    ///
    /// Knight, bishop, rook, and queen promotions are preserved. Pawn and king
    /// values are not valid promotion targets and are encoded as no promotion.
    ///
    /// # Arguments
    ///
    /// * `from` - source square
    /// * `to` - destination square
    /// * `promotion` - optional promotion piece kind
    ///
    /// # Returns
    ///
    /// Encoded move value.
    #[inline]
    pub const fn new(from: Square, to: Square, promotion: Option<PieceKind>) -> Self {
        let promotion = match promotion {
            Some(PieceKind::Knight) => 1,
            Some(PieceKind::Bishop) => 2,
            Some(PieceKind::Rook) => 3,
            Some(PieceKind::Queen) => 4,
            _ => 0,
        };
        Self(from.index() as u16 | ((to.index() as u16) << 6) | ((promotion as u16) << 12))
    }

    /// Used for retrieving the encoded source square.
    ///
    /// The six masked low bits are always inside `0..64`, so the internal
    /// unwrap cannot fail.
    ///
    /// # Returns
    ///
    /// Source square decoded from bits `0..=5`.
    #[inline]
    pub fn from(self) -> Square {
        // Six masked bits are always in range.
        Square::new((self.0 & 0x3f) as u8).unwrap()
    }

    /// Used for retrieving the encoded destination square.
    ///
    /// The six masked bits are always inside `0..64`, so the internal unwrap
    /// cannot fail.
    ///
    /// # Returns
    ///
    /// Destination square decoded from bits `6..=11`.
    #[inline]
    pub fn to(self) -> Square {
        Square::new(((self.0 >> 6) & 0x3f) as u8).unwrap()
    }

    /// Used for retrieving the encoded promotion kind, if present.
    ///
    /// # Returns
    ///
    /// `Some` knight, bishop, rook, or queen when bits `12..=14` hold a
    /// promotion code in `1..=4`; `None` otherwise.
    #[inline]
    pub const fn promotion(self) -> Option<PieceKind> {
        PieceKind::from_promotion_code(((self.0 >> 12) & 7) as u8)
    }

    /// Used for retrieving the underlying 16-bit representation.
    ///
    /// # Returns
    ///
    /// Raw `u16` bit pattern of this move.
    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Used for producing a numeric key with the same order as [`Self`]'s UCI
    /// text.
    ///
    /// The five ASCII positions are packed most-significant first. A missing
    /// promotion suffix uses zero, so the four-character form sorts before a
    /// five-character form with the same squares, exactly as string ordering
    /// does. This key lets deterministic move lists avoid allocating temporary
    /// [`String`] values solely for comparison.
    ///
    /// # Returns
    ///
    /// Packed `u64` sort key that orders moves identically to their UCI text.
    #[inline]
    #[must_use]
    pub fn uci_order_key(self) -> u64 {
        let from = self.from();
        let to = self.to();
        let promotion = self
            .promotion()
            .and_then(PieceKind::promotion_char)
            .map_or(0, |ch| ch as u8);
        (u64::from(b'a' + from.file()) << 32)
            | (u64::from(b'0' + from.rank()) << 24)
            | (u64::from(b'a' + to.file()) << 16)
            | (u64::from(b'0' + to.rank()) << 8)
            | u64::from(promotion)
    }

    /// Used for validating and decoding a raw 16-bit move representation.
    ///
    /// # Arguments
    ///
    /// * `raw` - candidate 16-bit move encoding
    ///
    /// # Returns
    ///
    /// Decoded move sharing `raw`'s bit pattern.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] for [`NO_MOVE`], a set reserved high bit, or a
    /// promotion payload outside `0..=4`.
    pub fn from_raw(raw: u16) -> Result<Self, CoreError> {
        if raw == NO_MOVE || ((raw >> 12) & 7) > 4 || raw & 0x8000 != 0 {
            Err(CoreError::new(format!("invalid encoded move: {raw}")))
        } else {
            Ok(Self(raw))
        }
    }
}

impl fmt::Display for Move {
    /// Used for writing UCI long algebraic notation, including a promotion
    /// suffix.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter for the UCI text
    ///
    /// # Returns
    ///
    /// Result of writing the four- or five-character move to `formatter`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{}", self.from(), self.to())?;
        if let Some(ch) = self.promotion().and_then(PieceKind::promotion_char) {
            write!(formatter, "{ch}")?;
        }
        Ok(())
    }
}

impl FromStr for Move {
    /// Error type produced when the UCI move text is malformed.
    type Err = CoreError;

    /// Used for parsing the shape of a UCI long-algebraic move.
    ///
    /// This parser does not check the move against a position. Use
    /// [`crate::Position::parse_uci_move`] when legality is required.
    ///
    /// # Arguments
    ///
    /// * `text` - candidate UCI long-algebraic move text
    ///
    /// # Returns
    ///
    /// Encoded move on success.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] for non-ASCII text, invalid coordinates, identical
    /// source and destination squares, an unsupported promotion suffix, or a
    /// length other than four or five bytes.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if (text.len() != 4 && text.len() != 5) || !text.is_ascii() {
            return Err(CoreError::new(format!("invalid UCI move: {text}")));
        }
        let from = text[0..2].parse::<Square>()?;
        let to = text[2..4].parse::<Square>()?;
        if from == to {
            return Err(CoreError::new(format!(
                "move has identical squares: {text}"
            )));
        }
        let promotion =
            if text.len() == 5 {
                let ch = text.as_bytes()[4] as char;
                Some(PieceKind::from_promotion_char(ch).ok_or_else(|| {
                    CoreError::new(format!("invalid promotion in UCI move: {text}"))
                })?)
            } else {
                None
            };
        Ok(Self::new(from, to, promotion))
    }
}
