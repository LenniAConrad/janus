//! Position storage, FEN parsing, move generation, and reversible transitions.
//!
//! [`Position`] is the authoritative mutable state for chess rules. Its mailbox,
//! piece bitboards, color occupancy, and cached king squares are updated as one
//! invariant by the private `put` and `take` primitives. Public mutation returns
//! opaque undo records so search can restore state without cloning at each ply.

use crate::{Color, CoreError, Move, Piece, PieceKind, Square};
use core::fmt;

/// Shared CRTK-derived attack computation and move generation.
///
/// Backs the public attack queries and the pseudo-legal, legal, and tactical
/// move generators exposed by [`Position`].
pub(crate) mod movegen;
/// Shared CRTK-derived reversible position transitions.
///
/// Backs [`Position::make_move`], [`Position::unmake_move`],
/// [`Position::make_null`], and [`Position::unmake_null`].
mod transition;

/// Used for marking an unoccupied square in the mailbox board array.
///
/// The twelve piece indices occupy the values below `12`, leaving `12` as the
/// reserved out-of-band vacancy marker.
const EMPTY: u8 = 12;
/// Used for bounding each color's army to the number of pieces reachable from
/// an orthodox or Chess960 initial position.
const MAX_ARMY_PIECES: u32 = 16;
/// Used for deriving how many promotions a color's missing pawns can explain.
const INITIAL_PAWNS: u32 = 8;
/// Used for counting non-pawn material beyond the unpromoted initial army.
///
/// Each excess knight, bishop, rook, or queen consumes one missing pawn's
/// promotion allowance.
const INITIAL_PROMOTION_TARGETS: [(PieceKind, u32); 4] = [
    (PieceKind::Knight, 2),
    (PieceKind::Bishop, 2),
    (PieceKind::Rook, 2),
    (PieceKind::Queen, 1),
];
/// Used for seeding the deterministic FNV-1a-style position hashes.
///
/// Serves as the initial accumulator value of [`Position::key`].
///
/// TODO: this value is the canonical 64-bit FNV offset basis
/// `14_695_981_039_346_656_037` with its leading digit truncated (a likely
/// transcription slip). Determinism and hash quality are unaffected, and
/// every persisted key and parity test depends on the current value, so
/// changing it would invalidate existing digests. Either adopt the canonical
/// basis in a coordinated migration or document the truncation as permanent.
const FNV_OFFSET: u64 = 1_469_598_103_934_665_603;
/// Used for mixing folded state into the deterministic FNV-1a-style hashes.
///
/// Every folded value in [`Position::key`] and
/// [`Position::tt_key_from_repetition_key`] is multiplied by this 64-bit FNV
/// prime.
const FNV_PRIME: u64 = 1_099_511_628_211;
/// Used for enumerating the eight file/row deltas reachable by a knight.
///
/// Consumed by the shared movegen attack tables and jump generators as well
/// as the test-only legacy scans.
const KNIGHT_OFFSETS: [(i8, i8); 8] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];
/// Used for enumerating the eight file/row deltas reachable by a king.
///
/// Consumed by the shared movegen attack tables and jump generators as well
/// as the test-only legacy scans.
const KING_OFFSETS: [(i8, i8); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];
/// Used for tracing the four diagonal ray directions of bishops and queens.
///
/// Consumed by the shared movegen pin detection and the test-only legacy
/// generators.
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
/// Used for tracing the four orthogonal ray directions of rooks and queens.
///
/// Consumed by the shared movegen pin detection and the test-only legacy
/// generators.
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
/// Used for tracing every ray direction in the same order as full queen move
/// generation.
///
/// Concatenates [`BISHOP_DIRECTIONS`] then [`ROOK_DIRECTIONS`]; the shared
/// movegen ray tables and the test-only legacy queen generators both walk
/// rays in this order.
const QUEEN_DIRECTIONS: [(i8, i8); 8] = [
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
];

/// Used for constructing the orthodox initial chess position.
///
/// Six-field FEN consumed by [`Position::start`] and available to callers
/// that need the canonical starting string.
pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// A compact mask of the four independent castling permissions.
///
/// The low four bits store the white kingside, white queenside, black
/// kingside, and black queenside permissions in that order. Rook identity is
/// stored separately by [`Position`] because Chess960 rights identify a rook
/// file as well as a side.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct CastlingRights(u8);

impl CastlingRights {
    /// Used for indicating that white may castle toward the kingside rook.
    pub const WHITE_KINGSIDE: u8 = 1;
    /// Used for indicating that white may castle toward the queenside rook.
    pub const WHITE_QUEENSIDE: u8 = 2;
    /// Used for indicating that black may castle toward the kingside rook.
    pub const BLACK_KINGSIDE: u8 = 4;
    /// Used for indicating that black may castle toward the queenside rook.
    pub const BLACK_QUEENSIDE: u8 = 8;

    /// Used for retrieving the raw four-bit permission mask.
    ///
    /// # Returns
    ///
    /// The underlying byte with one bit per castling permission.
    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Used for testing whether at least one bit in `right` is active.
    ///
    /// Pass one of the four associated permission constants to test a single
    /// castling side.
    ///
    /// # Arguments
    ///
    /// * `right` - permission bit mask to test
    ///
    /// # Returns
    ///
    /// `true` when the mask shares at least one active bit with `right`.
    #[inline]
    pub const fn contains(self, right: u8) -> bool {
        self.0 & right != 0
    }

    /// Used for activating every permission bit present in `right`.
    ///
    /// # Arguments
    ///
    /// * `right` - permission bits to switch on
    #[inline]
    fn insert(&mut self, right: u8) {
        self.0 |= right;
    }
}

/// Extra piece locations required to reverse a castling move.
///
/// Stored inside [`Undo`] only when the recorded move castled, so ordinary
/// moves carry no castle bookkeeping.
#[derive(Clone, Copy, Debug)]
struct CastleUndo {
    /// Used for restoring the rook value moved during castling.
    rook: Piece,
    /// Used for restoring the rook to its square before castling.
    rook_from: Square,
    /// Used for locating the rook square after castling.
    rook_to: Square,
    /// Used for locating the king square after castling.
    king_to: Square,
}

/// Opaque state required to reverse one successful [`Position::make_move`].
///
/// An undo record belongs to the exact position and move that produced it. It
/// intentionally exposes only search-relevant facts; restoration is performed
/// by [`Position::unmake_move`].
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    /// Used for restoring the original value of the piece that moved, before
    /// promotion if applicable.
    moved: Piece,
    /// Used for restoring the captured piece and its original square,
    /// including en-passant captures.
    captured: Option<(Piece, Square)>,
    /// Used for restoring the additional king and rook locations of a castle.
    castle: Option<CastleUndo>,
    /// Used for restoring the castling permissions active before the move.
    castling_rights: CastlingRights,
    /// Used for restoring the en-passant target present before the move.
    en_passant: Option<Square>,
    /// Used for restoring the halfmove clock value before the move.
    halfmove_clock: u16,
    /// Used for restoring the fullmove number before the move.
    fullmove_number: u16,
}

/// Opaque state required to reverse one successful [`Position::make_null`].
///
/// A null-undo record belongs to the exact position that produced it;
/// restoration is performed by [`Position::unmake_null`].
#[derive(Clone, Copy, Debug)]
pub struct NullUndo {
    /// Used for restoring the side to move before the null move.
    side_to_move: Color,
    /// Used for restoring the en-passant target before the null move expired
    /// it.
    en_passant: Option<Square>,
    /// Used for restoring the halfmove clock retained across the null move.
    halfmove_clock: u16,
    /// Used for restoring the fullmove number retained across the null move.
    fullmove_number: u16,
}

impl Undo {
    /// Used for retrieving the piece value before it moved or promoted.
    ///
    /// # Returns
    ///
    /// The moving piece as it stood on its source square.
    pub fn moved_piece(self) -> Piece {
        self.moved
    }

    /// Used for retrieving the captured piece and its original square, if
    /// any.
    ///
    /// For en passant, the square differs from the move's destination.
    ///
    /// # Returns
    ///
    /// The captured piece and the square it stood on, or `None` when the
    /// move captured nothing.
    pub fn captured(self) -> Option<(Piece, Square)> {
        self.captured
    }

    /// Used for testing whether this record represents a castling move.
    ///
    /// # Returns
    ///
    /// `true` when castle-specific undo data was recorded.
    pub fn was_castle(self) -> bool {
        self.castle.is_some()
    }
}

/// A complete mutable orthodox or Chess960 position.
///
/// The mailbox, twelve piece bitboards, two color-occupancy masks, and cached
/// king squares always describe the same board. Construction is restricted to
/// validated FEN or the built-in start position; mutation is reversible through
/// [`Undo`] and [`NullUndo`].
#[derive(Clone, Eq, PartialEq)]
pub struct Position {
    /// Used for storing the twelve piece bitboards in white then black
    /// pawn-through-king order.
    pieces: [u64; 12],
    /// Used for mailbox lookup of per-square piece indices, with [`EMPTY`]
    /// marking vacant squares.
    board: [u8; 64],
    /// Used for the incrementally maintained piece-placement hash.
    ///
    /// [`Position::key`] previously walked all sixty-four mailbox squares and
    /// folded them with FNV on every call, costing about `37` nanoseconds.
    /// Search calls it at least once per node, so a million-node search spent
    /// tens of milliseconds rehashing a board that changed by two squares.
    /// Every engine in the reference tree instead keeps a hash updated by
    /// exclusive-or inside piece placement and removal, which is what
    /// [`Position::put`] and [`Position::take`] now do.
    ///
    /// This covers piece placement only; side to move, castling and en passant
    /// are still folded in by `key`, because they are a handful of cheap words
    /// rather than a sixty-four iteration loop.
    piece_key: u64,
    /// Used for keying a pawn-structure cache.
    ///
    /// Incrementally maintained over **pawns only**. That is the exact input
    /// set of the terms a cache can hold: doubled and isolated penalties come
    /// from per-file pawn counts, and which pawns are passers is decided by
    /// intersecting the enemy pawn bitboard with a precomputed mask.
    ///
    /// Kings are deliberately excluded even though passer realization reads
    /// them. Folding them in would invalidate the entry on every king move,
    /// and in the endgames where this cache matters most the kings move
    /// constantly — the king-dependent geometry is cheap and is recomputed
    /// outside the cache instead.
    ///
    /// Every classical engine in the reference tree keeps one — Stockfish 11
    /// probes `Pawns::probe`, Ethereal a `PKTable` of
    /// `PKEntry { pkhash, passed, eval, safetyw, safetyb }`.
    ///
    /// What this key must **not** be used to cache is anything reading a
    /// non-pawn piece — most importantly whether a passer's front square is
    /// occupied, which any piece can decide. Stockfish computes that outside
    /// its pawn hash for the same reason.
    pawn_key: u64,
    /// Used for caching aggregate occupancy for white and black.
    occupancy: [u64; 2],
    /// Used for caching the king square of each color.
    kings: [Option<Square>; 2],
    /// Used for tracking the player that must make the next real move.
    side_to_move: Color,
    /// Used for tracking the currently active castling permissions.
    castling_rights: CastlingRights,
    /// Used for recording castling-rook identity in white K/Q, black K/Q
    /// order.
    ///
    /// Entries remain as metadata after a right is lost, but only active
    /// entries participate in [`Position::key`].
    castling_rooks: [Option<Square>; 4],
    /// Used for selecting Chess960 king-to-rook internal castling move
    /// notation.
    chess960: bool,
    /// Used for tracking the current FEN en-passant target, if the preceding
    /// move exposed one.
    en_passant: Option<Square>,
    /// Used for counting reversible plies toward the fifty-move rule.
    halfmove_clock: u16,
    /// Used for tracking the one-based FEN move number, incremented after
    /// black moves.
    fullmove_number: u16,
}

impl Default for Position {
    /// Used for creating the orthodox initial position.
    ///
    /// # Returns
    ///
    /// The result of [`Position::start`].
    fn default() -> Self {
        Self::start()
    }
}

impl fmt::Debug for Position {
    /// Used for formatting the position through its canonical six-field FEN.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter
    ///
    /// # Returns
    ///
    /// The outcome of writing the single-field debug representation.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Position")
            .field("fen", &self.to_fen())
            .finish()
    }
}

/// Used for one piece-on-square hash contribution.
///
/// The table is a deterministic xorshift sequence rather than stored random
/// data, so the crate embeds no blob and the values are reproducible from the
/// source alone.
///
/// # Arguments
///
/// * `piece` - piece occupying the square
/// * `square` - square the piece sits on
///
/// # Returns
///
/// The hash contribution to exclusive-or into the running key.
#[inline]
fn piece_square_key(piece: Piece, square: Square) -> u64 {
    PIECE_SQUARE_KEYS[piece.index()][square.index() as usize]
}

/// Used for the Zobrist piece-square words, built at compile time.
///
/// The chain has no runtime input -- it is a fixed xorshift64 seeded with a
/// literal -- so a `const fn` walking the same nesting order emits the same
/// 768 words as the previous `OnceLock`. Only the moment of evaluation moves.
///
/// The win is not the atomic load. Removing the cold initialiser lets the
/// caller inline, which is what took make and unmake from `29.3` to `24.4` ns
/// on Kiwipete. Sizing this with perft would hide it: make/unmake is only
/// about `0.6` ns of a `3.9` ns perft node.
static PIECE_SQUARE_KEYS: [[u64; 64]; 12] = build_piece_square_keys();

/// Used for materialising [`PIECE_SQUARE_KEYS`] at compile time.
///
/// # Returns
///
/// The Zobrist words, in piece-major then square order.
const fn build_piece_square_keys() -> [[u64; 64]; 12] {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut built = [[0_u64; 64]; 12];
    let mut piece = 0;
    while piece < 12 {
        let mut square = 0;
        while square < 64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            built[piece][square] = state;
            square += 1;
        }
        piece += 1;
    }
    built
}

impl Position {
    /// Used for creating the orthodox initial position.
    ///
    /// # Returns
    ///
    /// A fresh [`Position`] parsed from [`START_FEN`].
    ///
    /// # Panics
    ///
    /// Panics if the built-in starting FEN fails to parse, which cannot
    /// happen for the shipped constant.
    pub fn start() -> Self {
        Self::from_fen(START_FEN).expect("built-in starting FEN is valid")
    }

    /// Used for creating a synchronized but kingless board for staged FEN
    /// construction.
    ///
    /// # Returns
    ///
    /// A position with empty bitboards, an all-[`EMPTY`] mailbox, white to
    /// move, no castling rights, no en-passant target, and clocks `0 1`.
    fn empty() -> Self {
        Self {
            pieces: [0; 12],
            board: [EMPTY; 64],
            piece_key: 0,
            pawn_key: 0,
            occupancy: [0; 2],
            kings: [None; 2],
            side_to_move: Color::White,
            castling_rights: CastlingRights::default(),
            castling_rooks: [None; 4],
            chess960: false,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    /// Used for parsing a strict four- or six-field FEN.
    ///
    /// Castling accepts canonical `KQkq` notation for orthodox chess and
    /// Shredder-FEN rook-file letters for Chess960. Four-field input defaults
    /// the clocks to `0 1`. The resulting position must contain exactly one
    /// king per color and must not leave the side that just moved in check.
    ///
    /// # Arguments
    ///
    /// * `fen` - whitespace-separated FEN text with four or six fields
    ///
    /// # Returns
    ///
    /// The fully validated parsed position.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when any field is malformed or violates a
    /// validated position invariant, including piece counts, pawn placement,
    /// castling-rook identity, and en-passant consistency.
    pub fn from_fen(fen: &str) -> Result<Self, CoreError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 4 && fields.len() != 6 {
            return Err(CoreError::new(format!(
                "FEN must have four or six fields, got {}: {fen}",
                fields.len()
            )));
        }

        let mut position = Self::empty();
        position.parse_placement(fields[0])?;
        position.side_to_move = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            value => return Err(CoreError::new(format!("invalid FEN active color: {value}"))),
        };
        position.parse_castling(fields[2])?;
        position.en_passant = if fields[3] == "-" {
            None
        } else {
            Some(fields[3].parse::<Square>()?)
        };
        if fields.len() == 6 {
            position.halfmove_clock = fields[4]
                .parse::<u16>()
                .map_err(|_| CoreError::new(format!("invalid halfmove clock: {}", fields[4])))?;
            position.fullmove_number = fields[5]
                .parse::<u16>()
                .map_err(|_| CoreError::new(format!("invalid fullmove number: {}", fields[5])))?;
            if position.fullmove_number == 0 {
                return Err(CoreError::new("fullmove number must be positive"));
            }
        }
        position.validate_fen()?;
        Ok(position)
    }

    /// Used for parsing the eight slash-separated FEN ranks into synchronized
    /// storage.
    ///
    /// Digits encode runs of empty squares and may not repeat back to back;
    /// piece letters are placed through the `put` primitive so every board
    /// index stays synchronized.
    ///
    /// # Arguments
    ///
    /// * `placement` - first FEN field describing the board contents
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when the rank count, a digit run, a piece
    /// letter, or a rank width is malformed.
    fn parse_placement(&mut self, placement: &str) -> Result<(), CoreError> {
        let ranks: Vec<&str> = placement.split('/').collect();
        if ranks.len() != 8 {
            return Err(CoreError::new("FEN placement must contain eight ranks"));
        }
        for (row, rank) in ranks.iter().enumerate() {
            let mut file = 0_u8;
            let mut previous_digit = false;
            for ch in rank.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    if !(1..=8).contains(&empty) || previous_digit {
                        return Err(CoreError::new(format!("invalid FEN rank: {rank}")));
                    }
                    file = file
                        .checked_add(empty as u8)
                        .ok_or_else(|| CoreError::new(format!("invalid FEN rank: {rank}")))?;
                    previous_digit = true;
                    continue;
                }
                let piece = Piece::from_fen(ch)
                    .ok_or_else(|| CoreError::new(format!("invalid FEN piece: {ch}")))?;
                if file >= 8 {
                    return Err(CoreError::new(format!("FEN rank is too wide: {rank}")));
                }
                let square = Square::from_file_row(file, row as u8).unwrap();
                self.put(piece, square);
                file += 1;
                previous_digit = false;
            }
            if file != 8 {
                return Err(CoreError::new(format!(
                    "FEN rank width is {file}, expected 8: {rank}"
                )));
            }
        }
        Ok(())
    }

    /// Used for parsing orthodox or Shredder-FEN castling rights and rook
    /// identities.
    ///
    /// Canonical `KQkq` subsets keep orthodox semantics with fixed corner
    /// rooks. Rook-file letters switch the position to Chess960: white
    /// entries must precede black entries, a kingside entry must precede its
    /// queenside partner, and each side of each color may appear only once.
    ///
    /// # Arguments
    ///
    /// * `text` - third FEN field, with `-` meaning no rights
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] for empty or non-canonical fields, unknown
    /// characters, misordered or duplicate Chess960 entries, a missing king,
    /// or a rook sharing the king's file.
    fn parse_castling(&mut self, text: &str) -> Result<(), CoreError> {
        if text == "-" {
            return Ok(());
        }
        if text.is_empty() || text.contains('-') {
            return Err(CoreError::new(format!("invalid castling field: {text}")));
        }

        if text.chars().all(|ch| matches!(ch, 'K' | 'Q' | 'k' | 'q')) {
            let canonical: String = ['K', 'Q', 'k', 'q']
                .iter()
                .filter(|candidate| text.contains(**candidate))
                .collect();
            if canonical != text {
                return Err(CoreError::new(format!(
                    "non-canonical castling field: {text}"
                )));
            }
            for ch in text.chars() {
                let (right, rook) = match ch {
                    'K' => (CastlingRights::WHITE_KINGSIDE, Square::H1),
                    'Q' => (CastlingRights::WHITE_QUEENSIDE, Square::A1),
                    'k' => (CastlingRights::BLACK_KINGSIDE, Square::H8),
                    'q' => (CastlingRights::BLACK_QUEENSIDE, Square::A8),
                    _ => unreachable!(),
                };
                self.castling_rights.insert(right);
                self.castling_rooks[Self::right_index(right).unwrap()] = Some(rook);
            }
            return Ok(());
        }

        self.chess960 = true;
        let mut seen_black = false;
        let mut seen_white_queenside = false;
        let mut seen_black_queenside = false;
        for ch in text.chars() {
            let (color, file) = if ('A'..='H').contains(&ch) {
                if seen_black {
                    return Err(CoreError::new(format!(
                        "invalid Chess960 castling order: {text}"
                    )));
                }
                (Color::White, ch as u8 - b'A')
            } else if ('a'..='h').contains(&ch) {
                seen_black = true;
                (Color::Black, ch as u8 - b'a')
            } else {
                return Err(CoreError::new(format!("invalid castling character: {ch}")));
            };
            let king = self.kings[color.index()]
                .ok_or_else(|| CoreError::new("Chess960 castling requires a king"))?;
            if file == king.file() {
                return Err(CoreError::new("Chess960 rook cannot share the king file"));
            }
            let kingside = file > king.file();
            if color == Color::White && !kingside {
                seen_white_queenside = true;
            } else if color == Color::White && seen_white_queenside {
                return Err(CoreError::new(format!(
                    "invalid Chess960 castling order: {text}"
                )));
            } else if color == Color::Black && !kingside {
                seen_black_queenside = true;
            } else if color == Color::Black && seen_black_queenside {
                return Err(CoreError::new(format!(
                    "invalid Chess960 castling order: {text}"
                )));
            }
            let right = Self::right_for(color, kingside);
            if self.castling_rights.contains(right) {
                return Err(CoreError::new(format!(
                    "duplicate Chess960 castling side: {text}"
                )));
            }
            let row = if color == Color::White { 7 } else { 0 };
            self.castling_rights.insert(right);
            self.castling_rooks[Self::right_index(right).unwrap()] =
                Square::from_file_row(file, row);
        }
        Ok(())
    }

    /// Used for validating cross-field rules after all FEN fields have been
    /// parsed.
    ///
    /// Checks exactly one king per color, reachable army and promotion counts,
    /// at most eight pawns per color, no pawns on rank one or eight, and that
    /// the side not to move is not in check, then delegates to the castling and
    /// en-passant validators.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when any validated invariant fails.
    fn validate_fen(&self) -> Result<(), CoreError> {
        for color in [Color::White, Color::Black] {
            if self
                .piece_bitboard(Piece::new(color, PieceKind::King))
                .count_ones()
                != 1
            {
                return Err(CoreError::new(format!("FEN must contain one {color} king")));
            }
            let piece_count = self.color_occupancy(color).count_ones();
            if piece_count > MAX_ARMY_PIECES {
                return Err(CoreError::new(format!(
                    "FEN has {piece_count} {color} pieces; limit is {MAX_ARMY_PIECES}"
                )));
            }
            let pawn_count = self
                .piece_bitboard(Piece::new(color, PieceKind::Pawn))
                .count_ones();
            if pawn_count > INITIAL_PAWNS {
                return Err(CoreError::new(format!("FEN has too many {color} pawns")));
            }
            let promotion_count = INITIAL_PROMOTION_TARGETS
                .into_iter()
                .map(|(kind, initial)| {
                    self.piece_bitboard(Piece::new(color, kind))
                        .count_ones()
                        .saturating_sub(initial)
                })
                .sum::<u32>();
            let promotion_budget = INITIAL_PAWNS - pawn_count;
            if promotion_count > promotion_budget {
                return Err(CoreError::new(format!(
                    "FEN has {promotion_count} excess {color} pieces but only {promotion_budget} missing pawns"
                )));
            }
        }
        let back_ranks = 0xff_u64 | (0xff_u64 << 56);
        let pawns = self.piece_bitboard(Piece::new(Color::White, PieceKind::Pawn))
            | self.piece_bitboard(Piece::new(Color::Black, PieceKind::Pawn));
        if pawns & back_ranks != 0 {
            return Err(CoreError::new("pawns may not occupy rank one or eight"));
        }
        // A legal FEN cannot leave the player who just moved in check.
        if self.in_check(self.side_to_move.opposite()) {
            return Err(CoreError::new("side not to move is in check"));
        }
        self.validate_castling()?;
        self.validate_en_passant()?;
        Ok(())
    }

    /// Used for validating active castling rights against the king and
    /// recorded rooks.
    ///
    /// Every active right must name a rook square that holds a same-colored
    /// rook on the shared home rank, on the correct side of the king;
    /// orthodox positions additionally require the king on its e-file home
    /// square.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] describing the first violated castling
    /// invariant.
    fn validate_castling(&self) -> Result<(), CoreError> {
        for right in [
            CastlingRights::WHITE_KINGSIDE,
            CastlingRights::WHITE_QUEENSIDE,
            CastlingRights::BLACK_KINGSIDE,
            CastlingRights::BLACK_QUEENSIDE,
        ] {
            if !self.castling_rights.contains(right) {
                continue;
            }
            let index = Self::right_index(right).unwrap();
            let rook_square = self.castling_rooks[index]
                .ok_or_else(|| CoreError::new("castling right has no rook square"))?;
            let color = if right <= CastlingRights::WHITE_QUEENSIDE {
                Color::White
            } else {
                Color::Black
            };
            if self.piece_at(rook_square) != Some(Piece::new(color, PieceKind::Rook)) {
                return Err(CoreError::new(format!(
                    "castling rook missing on {rook_square}"
                )));
            }
            let king = self.kings[color.index()].unwrap();
            let home_row = if color == Color::White { 7 } else { 0 };
            if king.row() != home_row || rook_square.row() != home_row {
                return Err(CoreError::new(
                    "castling king and rook must be on their home rank",
                ));
            }
            let kingside = matches!(right, 1 | 4);
            if (kingside && rook_square.file() <= king.file())
                || (!kingside && rook_square.file() >= king.file())
            {
                return Err(CoreError::new(
                    "castling rook is on the wrong side of its king",
                ));
            }
            if !self.chess960 {
                let expected = if color == Color::White {
                    Square::E1
                } else {
                    Square::E8
                };
                if king != expected {
                    return Err(CoreError::new(
                        "orthodox castling king is not on the e-file",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Used for validating that the en-passant target describes a
    /// just-passed pawn.
    ///
    /// The target must be empty and on rank six when white is to move or
    /// rank three when black is to move, and the enemy pawn that just made
    /// the double push must stand on the square beyond the target. The pawn's
    /// two-rank origin on the other side of the target must be empty.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when the target is occupied, lies on the wrong
    /// rank, points off the board, has no passed pawn behind it, or has an
    /// occupied pawn origin.
    fn validate_en_passant(&self) -> Result<(), CoreError> {
        let Some(target) = self.en_passant else {
            return Ok(());
        };
        if self.piece_at(target).is_some() {
            return Err(CoreError::new("en-passant target is occupied"));
        }
        let (rank, pawn_offset, pawn_color) = match self.side_to_move {
            Color::White => (6, 1_i8, Color::Black),
            Color::Black => (3, -1_i8, Color::White),
        };
        if target.rank() != rank {
            return Err(CoreError::new("en-passant target is on the wrong rank"));
        }
        let pawn_square = target
            .offset(0, pawn_offset)
            .ok_or_else(|| CoreError::new("invalid en-passant target"))?;
        if self.piece_at(pawn_square) != Some(Piece::new(pawn_color, PieceKind::Pawn)) {
            return Err(CoreError::new("en-passant target has no passed pawn"));
        }
        let source_square = target
            .offset(0, -pawn_offset)
            .ok_or_else(|| CoreError::new("invalid en-passant target"))?;
        if self.piece_at(source_square).is_some() {
            return Err(CoreError::new("en-passant pawn source is occupied"));
        }
        Ok(())
    }

    /// Used for formatting the position as a canonical six-field FEN.
    ///
    /// Chess960 castling rights use Shredder-FEN rook-file letters. Clock
    /// fields are always included, even when the position was parsed from
    /// four fields.
    ///
    /// # Returns
    ///
    /// The complete six-field FEN string.
    pub fn to_fen(&self) -> String {
        let mut output = String::with_capacity(88);
        for row in 0..8_u8 {
            let mut empty = 0_u8;
            for file in 0..8_u8 {
                let square = Square::from_file_row(file, row).unwrap();
                if let Some(piece) = self.piece_at(square) {
                    if empty != 0 {
                        output.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    output.push(piece.fen());
                } else {
                    empty += 1;
                }
            }
            if empty != 0 {
                output.push((b'0' + empty) as char);
            }
            if row != 7 {
                output.push('/');
            }
        }
        output.push(' ');
        output.push(if self.side_to_move == Color::White {
            'w'
        } else {
            'b'
        });
        output.push(' ');
        let before = output.len();
        for (right, orthodox) in [
            (CastlingRights::WHITE_KINGSIDE, 'K'),
            (CastlingRights::WHITE_QUEENSIDE, 'Q'),
            (CastlingRights::BLACK_KINGSIDE, 'k'),
            (CastlingRights::BLACK_QUEENSIDE, 'q'),
        ] {
            if !self.castling_rights.contains(right) {
                continue;
            }
            if self.chess960 {
                let square = self.castling_rooks[Self::right_index(right).unwrap()].unwrap();
                let base = if right <= CastlingRights::WHITE_QUEENSIDE {
                    b'A'
                } else {
                    b'a'
                };
                output.push((base + square.file()) as char);
            } else {
                output.push(orthodox);
            }
        }
        if output.len() == before {
            output.push('-');
        }
        output.push(' ');
        if let Some(square) = self.en_passant {
            output.push_str(&square.to_string());
        } else {
            output.push('-');
        }
        output.push(' ');
        output.push_str(&self.halfmove_clock.to_string());
        output.push(' ');
        output.push_str(&self.fullmove_number.to_string());
        output
    }

    /// Used for retrieving the player that must make the next move.
    #[inline]
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// Used for retrieving the active castling-permission mask.
    #[inline]
    pub const fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    /// Used for retrieving the rook square identified by one active castling
    /// permission.
    ///
    /// `right` must be one of the four [`CastlingRights`] constants. The
    /// returned identity is the rook's source square, including for Chess960.
    ///
    /// # Arguments
    ///
    /// * `right` - single castling permission bit to look up
    ///
    /// # Returns
    ///
    /// The rook's source square, or `None` when the permission is inactive
    /// or the bit is not a single recognized right.
    #[inline]
    pub fn castling_rook_square(&self, right: u8) -> Option<Square> {
        if self.castling_rights.contains(right) {
            self.castling_rook(right)
        } else {
            None
        }
    }

    /// Used for retrieving the FEN en-passant target, if present.
    ///
    /// The target may be geometrically valid without enabling a legal
    /// capture; [`Self::key`] excludes it in that case.
    #[inline]
    pub const fn en_passant_square(&self) -> Option<Square> {
        self.en_passant
    }

    /// Used for retrieving the reversible-ply count of the fifty-move rule.
    #[inline]
    pub const fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    /// Used for retrieving the one-based FEN move number.
    #[inline]
    pub const fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    /// Used for testing whether the position uses Chess960 castling
    /// semantics.
    #[inline]
    pub const fn is_chess960(&self) -> bool {
        self.chess960
    }

    /// Used for retrieving the piece occupying `square`, or `None` when it
    /// is empty.
    ///
    /// # Arguments
    ///
    /// * `square` - board square to inspect
    #[inline]
    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        Piece::from_index(self.board[square.index() as usize])
    }

    /// Used for retrieving the bitboard occupied by one exact color-and-kind
    /// combination.
    ///
    /// # Arguments
    ///
    /// * `piece` - color-and-kind combination to look up
    #[inline]
    pub const fn piece_bitboard(&self, piece: Piece) -> u64 {
        self.pieces[piece.index()]
    }

    /// Used for retrieving all squares occupied by `color`.
    ///
    /// # Arguments
    ///
    /// * `color` - side whose occupancy mask is requested
    #[inline]
    pub const fn color_occupancy(&self, color: Color) -> u64 {
        self.occupancy[color.index()]
    }

    /// Used for retrieving all occupied board squares of both colors.
    #[inline]
    pub const fn occupancy(&self) -> u64 {
        self.occupancy[0] | self.occupancy[1]
    }

    /// Used for testing whether material alone proves that no mating
    /// sequence exists.
    ///
    /// This conservative dead-position predicate recognizes bare kings, a
    /// lone bishop or knight, and bishop-only positions in which every bishop
    /// stays on the same square color. Pawns, rooks, queens, knights
    /// alongside other minors, or bishops spanning both square colors remain
    /// playable because material alone cannot rule out a legal mating
    /// sequence.
    ///
    /// # Returns
    ///
    /// `true` when the remaining material can never produce checkmate.
    #[must_use]
    pub fn is_insufficient_material(&self) -> bool {
        for color in [Color::White, Color::Black] {
            for kind in [PieceKind::Pawn, PieceKind::Rook, PieceKind::Queen] {
                if self.piece_bitboard(Piece::new(color, kind)) != 0 {
                    return false;
                }
            }
        }

        let white_knights = self.piece_bitboard(Piece::new(Color::White, PieceKind::Knight));
        let black_knights = self.piece_bitboard(Piece::new(Color::Black, PieceKind::Knight));
        let white_bishops = self.piece_bitboard(Piece::new(Color::White, PieceKind::Bishop));
        let black_bishops = self.piece_bitboard(Piece::new(Color::Black, PieceKind::Bishop));
        let minor_count =
            (white_knights | black_knights | white_bishops | black_bishops).count_ones();
        if minor_count <= 1 {
            return true;
        }
        if white_knights | black_knights != 0 {
            return false;
        }

        let mut bishops = white_bishops | black_bishops;
        let first_index = bishops.trailing_zeros();
        let first_color = ((first_index & 7) + (first_index >> 3)) & 1;
        while bishops != 0 {
            let index = bishops.trailing_zeros();
            if ((index & 7) + (index >> 3)) & 1 != first_color {
                return false;
            }
            bishops &= bishops - 1;
        }
        true
    }

    /// Used for retrieving the cached king square for `color`.
    ///
    /// Valid public positions always contain both kings; the optional shape
    /// is retained because staged internal FEN construction starts empty.
    ///
    /// # Arguments
    ///
    /// * `color` - side whose king square is requested
    #[inline]
    pub const fn king_square(&self, color: Color) -> Option<Square> {
        self.kings[color.index()]
    }

    /// Used for keying a pawn-structure cache over pawns.
    ///
    /// Maintained incrementally by piece placement and removal, so this is a
    /// field read rather than a board walk.
    ///
    /// Two positions sharing this key have identical pawn placement, and
    /// therefore identical doubled and isolated structure and identical
    /// passers. They may differ in every other piece — including the kings —
    /// which is precisely why the cache is worth having: a search re-evaluates
    /// the same pawn skeleton across most of a subtree.
    ///
    /// # Returns
    ///
    /// The incremental pawn placement hash.
    #[must_use]
    pub const fn pawn_key(&self) -> u64 {
        self.pawn_key
    }

    /// Used for computing the deterministic repetition key.
    ///
    /// This correctness-first implementation computes the key on demand in
    /// O(64), rather than maintaining it incrementally. Clocks are excluded.
    /// Castling rook identity is included for active rights, and an
    /// en-passant square is included only when it enables at least one legal
    /// capture.
    ///
    /// # Returns
    ///
    /// The clock-free FNV-1a-style hash of the repetition-relevant state.
    pub fn key(&self) -> u64 {
        let mut hash = FNV_OFFSET ^ self.piece_key;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= self.side_to_move.index() as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u64::from(self.castling_rights.bits());
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u64::from(self.chess960);
        hash = hash.wrapping_mul(FNV_PRIME);
        for right in [1_u8, 2, 4, 8] {
            let value = if self.castling_rights.contains(right) {
                self.castling_rooks[Self::right_index(right).unwrap()].map_or(64, Square::index)
            } else {
                64
            };
            hash ^= u64::from(value);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(self.effective_en_passant().map_or(64, Square::index));
        hash.wrapping_mul(FNV_PRIME)
    }

    /// Used for computing the deterministic transposition-table key
    /// including both FEN clocks.
    ///
    /// Search results can depend on the halfmove clock through the
    /// fifty-move rule. A persistent table must therefore distinguish
    /// otherwise identical boards at different clock states. The clock-free
    /// [`Self::key`] is Janus's semantic repetition key; extending it with
    /// both FEN clocks keeps TT entries safe without claiming byte-for-byte
    /// parity with `ChessRTK`'s Java `Position.signature()` implementation.
    ///
    /// # Returns
    ///
    /// The repetition key extended with the halfmove clock and fullmove
    /// number.
    #[must_use]
    pub fn tt_key(&self) -> u64 {
        self.tt_key_from_repetition_key(self.key())
    }

    /// Used for extending a previously computed repetition key with both FEN
    /// clocks.
    ///
    /// `repetition_key` must be the value returned by [`Self::key`] for this
    /// position. Search code that already needs the repetition key can use
    /// this helper to avoid hashing the board a second time for the
    /// transposition table.
    ///
    /// # Arguments
    ///
    /// * `repetition_key` - value of [`Self::key`] for this exact position
    ///
    /// # Returns
    ///
    /// The transposition-table key equal to [`Self::tt_key`].
    #[must_use]
    pub fn tt_key_from_repetition_key(&self, repetition_key: u64) -> u64 {
        let mut hash = repetition_key;
        hash ^= u64::from(self.halfmove_clock);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= u64::from(self.fullmove_number);
        hash.wrapping_mul(FNV_PRIME)
    }

    /// Used for placing a piece on an empty square while updating every
    /// board index.
    ///
    /// The mailbox, piece bitboard, color occupancy, and cached king square
    /// are updated together so the synchronized-board invariant holds.
    ///
    /// # Arguments
    ///
    /// * `piece` - piece value to place
    /// * `square` - destination square, which must currently be empty
    ///
    /// # Panics
    ///
    /// In debug builds, asserts that the destination square is empty.
    fn put(&mut self, piece: Piece, square: Square) {
        debug_assert_eq!(self.board[square.index() as usize], EMPTY);
        self.piece_key ^= piece_square_key(piece, square);
        if piece.kind == PieceKind::Pawn {
            self.pawn_key ^= piece_square_key(piece, square);
        }
        self.board[square.index() as usize] = piece.index() as u8;
        self.pieces[piece.index()] |= square.bit();
        self.occupancy[piece.color.index()] |= square.bit();
        if piece.kind == PieceKind::King {
            self.kings[piece.color.index()] = Some(square);
        }
    }

    /// Used for removing and returning a piece while updating every board
    /// index.
    ///
    /// The mailbox, piece bitboard, color occupancy, and cached king square
    /// are updated together so the synchronized-board invariant holds.
    ///
    /// # Arguments
    ///
    /// * `square` - square to clear
    ///
    /// # Returns
    ///
    /// The removed piece, or `None` when the square was already empty.
    fn take(&mut self, square: Square) -> Option<Piece> {
        let piece = self.piece_at(square)?;
        self.piece_key ^= piece_square_key(piece, square);
        if piece.kind == PieceKind::Pawn {
            self.pawn_key ^= piece_square_key(piece, square);
        }
        self.board[square.index() as usize] = EMPTY;
        self.pieces[piece.index()] &= !square.bit();
        self.occupancy[piece.color.index()] &= !square.bit();
        if piece.kind == PieceKind::King {
            self.kings[piece.color.index()] = None;
        }
        Some(piece)
    }

    /// Used for mapping a single castling permission bit to its
    /// rook-identity slot.
    ///
    /// # Arguments
    ///
    /// * `right` - candidate permission bit
    ///
    /// # Returns
    ///
    /// Index `0..4` in white K/Q, black K/Q order, or `None` when `right`
    /// is not exactly one recognized permission bit.
    fn right_index(right: u8) -> Option<usize> {
        match right {
            1 => Some(0),
            2 => Some(1),
            4 => Some(2),
            8 => Some(3),
            _ => None,
        }
    }

    /// Used for retrieving the permission bit for a color and castling side.
    ///
    /// # Arguments
    ///
    /// * `color` - castling player
    /// * `kingside` - `true` for kingside, `false` for queenside
    ///
    /// # Returns
    ///
    /// The matching [`CastlingRights`] permission constant.
    const fn right_for(color: Color, kingside: bool) -> u8 {
        match (color, kingside) {
            (Color::White, true) => CastlingRights::WHITE_KINGSIDE,
            (Color::White, false) => CastlingRights::WHITE_QUEENSIDE,
            (Color::Black, true) => CastlingRights::BLACK_KINGSIDE,
            (Color::Black, false) => CastlingRights::BLACK_QUEENSIDE,
        }
    }

    /// Used for retrieving the FIDE-defined final king square for a castle.
    ///
    /// # Arguments
    ///
    /// * `color` - castling player
    /// * `kingside` - `true` for kingside, `false` for queenside
    ///
    /// # Returns
    ///
    /// The g- or c-file destination on the player's home rank.
    fn castle_king_target(color: Color, kingside: bool) -> Square {
        match (color, kingside) {
            (Color::White, true) => Square::G1,
            (Color::White, false) => Square::C1,
            (Color::Black, true) => Square::G8,
            (Color::Black, false) => Square::C8,
        }
    }

    /// Used for retrieving the FIDE-defined final rook square for a castle.
    ///
    /// # Arguments
    ///
    /// * `color` - castling player
    /// * `kingside` - `true` for kingside, `false` for queenside
    ///
    /// # Returns
    ///
    /// The f- or d-file destination on the player's home rank.
    fn castle_rook_target(color: Color, kingside: bool) -> Square {
        match (color, kingside) {
            (Color::White, true) => Square::F1,
            (Color::White, false) => Square::D1,
            (Color::Black, true) => Square::F8,
            (Color::Black, false) => Square::D8,
        }
    }

    /// Used for retrieving the source square of the rook identified by an
    /// active or lost right.
    ///
    /// # Arguments
    ///
    /// * `right` - single castling permission bit
    ///
    /// # Returns
    ///
    /// The recorded rook square, or `None` when the bit is unrecognized or
    /// no rook identity was ever recorded for it.
    fn castling_rook(&self, right: u8) -> Option<Square> {
        self.castling_rooks[Self::right_index(right)?]
    }

    /// Used for retrieving the internal move destination for the requested
    /// castling right.
    ///
    /// Orthodox moves target the king destination; Chess960 moves target the
    /// participating rook so even stationary-king castles remain
    /// representable.
    ///
    /// # Arguments
    ///
    /// * `right` - single castling permission bit
    ///
    /// # Returns
    ///
    /// The internal destination square, or `None` when a Chess960 right has
    /// no recorded rook.
    fn castle_move_target(&self, right: u8) -> Option<Square> {
        let color = if right <= 2 {
            Color::White
        } else {
            Color::Black
        };
        let kingside = matches!(right, 1 | 4);
        if self.chess960 {
            self.castling_rook(right)
        } else {
            Some(Self::castle_king_target(color, kingside))
        }
    }

    /// Used for deriving the final king square and post-castle occupancy of a
    /// castle without mutating or cloning the position.
    ///
    /// Both castling participants are lifted from their source squares and
    /// placed on their FIDE destinations, yielding the exact occupancy a
    /// completed castle produces. The returned occupancy lets a caller probe
    /// the king's final-square safety with an attack scan instead of a full
    /// make/unmake on a working clone. Overlapping source and destination
    /// squares — reachable only in Chess960 — resolve correctly because the
    /// sources are cleared before the destinations are set.
    ///
    /// # Arguments
    ///
    /// * `right` - single castling permission bit identifying the castle
    ///
    /// # Returns
    ///
    /// `Some((king_to, occupancy))` where `king_to` is the FIDE king
    /// destination and `occupancy` is the full-board occupancy after the
    /// castle, or `None` when the king or rook source is absent.
    fn castle_destination_and_occupancy(&self, right: u8) -> Option<(Square, u64)> {
        let color = if right <= 2 {
            Color::White
        } else {
            Color::Black
        };
        let kingside = matches!(right, 1 | 4);
        let king_from = self.kings[color.index()]?;
        let rook_from = self.castling_rook(right)?;
        let king_to = Self::castle_king_target(color, kingside);
        let rook_to = Self::castle_rook_target(color, kingside);
        let occupancy = (self.occupancy() & !king_from.bit() & !rook_from.bit())
            | king_to.bit()
            | rook_to.bit();
        Some((king_to, occupancy))
    }

    /// Used for identifying an internally encoded castle under the active
    /// rights.
    ///
    /// # Arguments
    ///
    /// * `piece` - piece occupying the move's source square
    /// * `target` - move destination square
    ///
    /// # Returns
    ///
    /// The matched permission bit, or `None` when the mover is not a king or
    /// no active right targets `target`.
    fn castle_right_for_move(&self, piece: Piece, target: Square) -> Option<u8> {
        if piece.kind != PieceKind::King {
            return None;
        }
        for kingside in [true, false] {
            let right = Self::right_for(piece.color, kingside);
            if self.castling_rights.contains(right)
                && self.castle_move_target(right) == Some(target)
            {
                return Some(right);
            }
        }
        None
    }
}

impl Position {
    /// Used for retrieving the geometric attack mask produced by the piece
    /// on `square`.
    ///
    /// Pawn attacks are diagonal regardless of occupancy. Sliding attacks
    /// stop at, and include, the first occupied square, so friendly blockers
    /// remain represented as defended squares. Empty source squares return
    /// zero.
    ///
    /// # Arguments
    ///
    /// * `square` - source square whose piece is queried
    ///
    /// # Returns
    ///
    /// Bitboard of every square attacked by the occupying piece.
    pub fn attacks_from(&self, square: Square) -> u64 {
        movegen::attacks_from(self, square)
    }

    /// Used for retrieving geometric attacks when a typed-bitboard caller
    /// already knows the piece occupying `square`.
    ///
    /// The supplied piece must match the position's occupant. Debug builds
    /// verify that contract. Pawn attacks remain occupancy-independent while
    /// sliding attacks stop at, and include, the first occupied square.
    ///
    /// # Arguments
    ///
    /// * `square` - occupied source square to query
    /// * `piece` - piece known to occupy `square`
    ///
    /// # Returns
    ///
    /// Bitboard of every square attacked by the supplied occupying piece.
    #[inline]
    pub fn attacks_for_piece(&self, square: Square, piece: Piece) -> u64 {
        movegen::attacks_for_piece(self, square, piece)
    }

    /// Used for retrieving the pieces absolutely pinned against `color`'s king.
    ///
    /// A pinned piece cannot leave its ray without exposing the king, so its
    /// apparent mobility overstates what it can actually do. That is a
    /// *relational* fact -- it depends on the king and the enemy sliders, not
    /// on the piece's square -- which is why a per-square evaluation table
    /// cannot represent it at any weighting.
    ///
    /// This is the same set the legal-move generator computes and relies on,
    /// so its geometry is already proven by every perft fixture rather than by
    /// a separate oracle.
    ///
    /// # Arguments
    ///
    /// * `color` - side whose pinned pieces are collected
    ///
    /// # Returns
    ///
    /// Bitboard of `color`'s absolutely pinned pieces; zero without a king.
    #[inline]
    #[must_use]
    pub fn absolute_pins(&self, color: Color) -> u64 {
        movegen::pinned_pieces(self, color)
    }

    /// Used for testing whether a color attacks a square under a supplied
    /// occupancy.
    ///
    /// Slider rays are traced through `occupancy` rather than the position's
    /// own, so a caller simulating a move can pass the post-move occupancy and
    /// see attackers that the departure discovers. That exactness is the point:
    /// a test against the current occupancy would miss a slider whose line the
    /// moving piece was blocking.
    ///
    /// # Arguments
    ///
    /// * `square` - square under test
    /// * `by` - attacking color
    /// * `occupancy` - full-board occupancy used for slider lookups
    ///
    /// # Returns
    ///
    /// `true` when any `by` piece attacks `square` under `occupancy`.
    #[inline]
    #[must_use]
    pub fn is_attacked_under(&self, square: Square, by: Color, occupancy: u64) -> bool {
        movegen::attackers_to_with(self, square, by, occupancy) != 0
    }

    /// Used for testing whether `square` is attacked by `by` in the current
    /// position.
    ///
    /// Friendly blockers still count as defended squares, while sliding rays
    /// end at the first occupied square.
    ///
    /// # Arguments
    ///
    /// * `square` - square being probed
    /// * `by` - attacking side
    ///
    /// # Returns
    ///
    /// `true` when at least one piece of `by` attacks `square`.
    pub fn is_square_attacked(&self, square: Square, by: Color) -> bool {
        movegen::is_square_attacked(self, square, by)
    }

    /// Used for testing whether `color`'s king is attacked.
    ///
    /// A temporarily kingless internal construction state returns `false`.
    ///
    /// # Arguments
    ///
    /// * `color` - side whose king is examined
    ///
    /// # Returns
    ///
    /// `true` when the opposing side attacks the king's square.
    pub fn in_check(&self, color: Color) -> bool {
        movegen::is_king_attacked(self, color)
    }

    /// Used for testing whether `mv` captures in this position, including en
    /// passant.
    ///
    /// Castling is never classified as a capture, including Chess960's
    /// internal king-to-rook encoding.
    ///
    /// # Arguments
    ///
    /// * `mv` - candidate move to classify
    ///
    /// # Returns
    ///
    /// `true` when the destination is occupied or the move is a pawn's
    /// en-passant capture; `false` for empty sources and castles.
    pub fn is_capture(&self, mv: Move) -> bool {
        if let Some(piece) = self.piece_at(mv.from()) {
            if self.castle_right_for_move(piece, mv.to()).is_some() {
                return false;
            }
            self.piece_at(mv.to()).is_some()
                || (piece.kind == PieceKind::Pawn
                    && Some(mv.to()) == self.en_passant
                    && mv.from().file() != mv.to().file())
        } else {
            false
        }
    }

    /// Used for testing whether `mv` encodes castling under the active
    /// rights.
    ///
    /// # Arguments
    ///
    /// * `mv` - candidate move to classify
    ///
    /// # Returns
    ///
    /// `true` when the source holds a king whose destination matches an
    /// active castling right's internal target.
    pub fn is_castling_move(&self, mv: Move) -> bool {
        self.piece_at(mv.from())
            .and_then(|piece| self.castle_right_for_move(piece, mv.to()))
            .is_some()
    }

    /// Used for formatting one legal move using orthodox or UCI Chess960
    /// castling notation.
    ///
    /// The internal encoding follows the position's FEN-derived variant.
    /// UCI's `UCI_Chess960` option is an output convention instead: when
    /// enabled, a castle targets the rook source square; otherwise it
    /// targets the king's final square.
    ///
    /// Non-castling moves, and moves whose source is empty, use [`Move`]'s
    /// ordinary long-algebraic formatting.
    ///
    /// # Arguments
    ///
    /// * `mv` - move in the position's internal encoding
    /// * `uci_chess960` - whether the consumer expects king-to-rook castles
    ///
    /// # Returns
    ///
    /// The long-algebraic move text in the requested convention.
    #[must_use]
    pub fn uci_move_string(&self, mv: Move, uci_chess960: bool) -> String {
        let Some(piece) = self.piece_at(mv.from()) else {
            return mv.to_string();
        };
        let Some(right) = self.castle_right_for_move(piece, mv.to()) else {
            return mv.to_string();
        };
        let kingside = matches!(right, 1 | 4);
        let target = if uci_chess960 {
            self.castling_rook(right).unwrap_or(mv.to())
        } else {
            Self::castle_king_target(piece.color, kingside)
        };
        Move::new(mv.from(), target, None).to_string()
    }

    /// Used for generating legal captures and promotions for quiescence
    /// search.
    ///
    /// Quiet promotions are included because promotion materially changes
    /// the position even without a capture. The generator emits the same
    /// ordered subsequence that filtering [`Self::legal_moves`] would
    /// produce, without first constructing or validating ordinary quiet
    /// moves.
    ///
    /// # Returns
    ///
    /// The ordered list of legal tactical moves for the side to move.
    pub fn legal_tactical_moves(&self) -> Vec<Move> {
        movegen::legal_tactical_moves(self)
    }

    /// Used for generating movement-correct moves that may expose the moving
    /// king.
    ///
    /// The result excludes friendly captures and king captures. Castling
    /// paths must be empty, but attacks on the king's source, transit, and
    /// destination are checked only by [`Self::legal_moves`].
    ///
    /// # Returns
    ///
    /// The ordered pseudo-legal move list for the side to move.
    pub fn pseudo_legal_moves(&self) -> Vec<Move> {
        movegen::pseudo_legal_moves(self)
    }

    /// Used for generating every legal move for the side to move.
    ///
    /// Output order is deterministic and follows the crate's
    /// piece-generation order; callers must not treat it as a strength-based
    /// move ordering.
    ///
    /// # Returns
    ///
    /// The complete ordered legal move list; empty for mate or stalemate.
    pub fn legal_moves(&self) -> Vec<Move> {
        movegen::legal_moves(self)
    }

    /// Used for generating every legal move into a caller-owned buffer.
    ///
    /// Identical output to [`Self::legal_moves`], but the caller keeps the
    /// allocation across calls. A search visiting a million nodes a second
    /// otherwise pays a heap allocation and free per node purely to hand back
    /// a vector it discards immediately.
    ///
    /// # Arguments
    ///
    /// * `out` - buffer cleared and refilled with the legal moves
    pub fn legal_moves_into(&self, out: &mut Vec<Move>) {
        movegen::legal_moves_into(self, out);
    }

    /// Used for testing whether the side to move has any legal move at all.
    ///
    /// This answers the mate-or-stalemate question that search asks on its
    /// hottest exits, where building the move list only to measure its length
    /// discards every move it just generated. The result equals
    /// `!self.legal_moves().is_empty()` exactly, so it is a drop-in
    /// replacement wherever emptiness is the only thing consulted.
    ///
    /// # Returns
    ///
    /// `true` when at least one legal move exists; `false` for mate or
    /// stalemate.
    #[must_use]
    pub fn has_legal_move(&self) -> bool {
        movegen::has_legal_move(self)
    }

    /// Used for testing whether `mv` is an en-passant capture by `moving`.
    ///
    /// En passant always receives a full legality probe because removing a
    /// pawn from a square other than the destination can uncover a rook or
    /// bishop ray.
    ///
    /// # Arguments
    ///
    /// * `moving` - piece on the move's source square
    /// * `mv` - candidate move to classify
    ///
    /// # Returns
    ///
    /// `true` for a diagonal pawn move onto the empty en-passant target.
    fn is_en_passant_move(&self, moving: Piece, mv: Move) -> bool {
        moving.kind == PieceKind::Pawn
            && Some(mv.to()) == self.en_passant
            && self.piece_at(mv.to()).is_none()
            && mv.from().file() != mv.to().file()
    }

    /// Used for resolving a UCI move against this position while requiring
    /// it to be legal.
    ///
    /// For Chess960, both king-to-rook UCI notation and king-to-final-square
    /// notation are accepted; the returned value uses king-to-rook encoding.
    ///
    /// # Arguments
    ///
    /// * `text` - long-algebraic UCI move text
    ///
    /// # Returns
    ///
    /// The matching legal move in the internal encoding.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when the text is not a well-formed
    /// long-algebraic move or does not identify a legal move in this
    /// position.
    pub fn parse_uci_move(&self, text: &str) -> Result<Move, CoreError> {
        let parsed = text.parse::<Move>()?;
        let legal = self.legal_moves();
        if legal.contains(&parsed) {
            return Ok(parsed);
        }
        if self.chess960 {
            let piece = self.piece_at(parsed.from());
            if piece == Some(Piece::new(self.side_to_move, PieceKind::King)) {
                for mv in legal {
                    if mv.from() != parsed.from() || mv.promotion().is_some() {
                        continue;
                    }
                    let Some(right) = self.castle_right_for_move(piece.unwrap(), mv.to()) else {
                        continue;
                    };
                    let kingside = matches!(right, 1 | 4);
                    if parsed.to() == Self::castle_king_target(self.side_to_move, kingside) {
                        return Ok(mv);
                    }
                }
            }
        }
        Err(CoreError::new(format!("illegal move in position: {text}")))
    }

    /// Used for applying a pseudo-legal move and returning exact undo data.
    ///
    /// This is the low-level search transition: it enforces structural state
    /// constraints but does not independently prove the piece's movement
    /// pattern or protect its king. Normal callers should obtain `mv` from
    /// [`Self::pseudo_legal_moves`] or [`Self::legal_moves`]. On success,
    /// pass the returned record and the same move to [`Self::unmake_move`]
    /// in LIFO order.
    ///
    /// # Arguments
    ///
    /// * `mv` - pseudo-legal move in the internal encoding
    ///
    /// # Returns
    ///
    /// The [`Undo`] record required to reverse this exact transition.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when the source is empty, the piece is not
    /// owned by the side to move, promotion or capture metadata is
    /// inconsistent, or a requested castle cannot be applied. An error
    /// leaves the position usable and does not advance the side to move.
    #[allow(clippy::too_many_lines)]
    pub fn make_move(&mut self, mv: Move) -> Result<Undo, CoreError> {
        transition::make_move(self, mv)
    }

    /// Used for reversing the corresponding successful [`Self::make_move`]
    /// call.
    ///
    /// `mv` and `undo` must be the exact pair produced for the current
    /// position, and nested moves must be unwound in reverse order.
    ///
    /// # Arguments
    ///
    /// * `mv` - move originally passed to [`Self::make_move`]
    /// * `undo` - record returned by that same call
    ///
    /// # Panics
    ///
    /// Panics in any build when incompatible or out-of-order undo data
    /// leaves a recorded square unexpectedly empty; debug builds can
    /// additionally trip the synchronized-board assertions.
    pub fn unmake_move(&mut self, mv: Move, undo: Undo) {
        transition::unmake_move(self, mv, undo);
    }

    /// Used for applying a reversible search-only null move.
    ///
    /// Castling and clocks remain unchanged; en-passant expires and the side
    /// toggles. A null move while checked is rejected because null-move
    /// pruning is unsound there.
    ///
    /// # Returns
    ///
    /// The [`NullUndo`] record required to reverse this transition.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when the side to move is in check.
    pub fn make_null(&mut self) -> Result<NullUndo, CoreError> {
        transition::make_null(self)
    }

    /// Used for restoring a corresponding successful [`Self::make_null`]
    /// call exactly.
    ///
    /// The record must belong to the most recent outstanding null move.
    ///
    /// # Arguments
    ///
    /// * `undo` - record returned by the matching [`Self::make_null`] call
    pub fn unmake_null(&mut self, undo: NullUndo) {
        transition::unmake_null(self, undo);
    }

    /// Used for checking Chess960-aware king and rook travel paths for
    /// blockers.
    ///
    /// # Arguments
    ///
    /// * `right` - single castling permission bit to examine
    ///
    /// # Returns
    ///
    /// `true` when the recorded rook is present and both inclusive travel
    /// paths hold no piece other than the castling king and rook.
    fn castle_path_is_clear(&self, right: u8) -> bool {
        let color = if right <= 2 {
            Color::White
        } else {
            Color::Black
        };
        let kingside = matches!(right, 1 | 4);
        let Some(king_from) = self.kings[color.index()] else {
            return false;
        };
        let Some(rook_from) = self.castling_rook(right) else {
            return false;
        };
        if self.piece_at(rook_from) != Some(Piece::new(color, PieceKind::Rook)) {
            return false;
        }
        let king_to = Self::castle_king_target(color, kingside);
        let rook_to = Self::castle_rook_target(color, kingside);
        self.horizontal_path_empty(king_from, king_to, king_from, rook_from)
            && self.horizontal_path_empty(rook_from, rook_to, king_from, rook_from)
    }

    /// Used for checking an inclusive same-rank path while ignoring two
    /// participating pieces.
    ///
    /// # Arguments
    ///
    /// * `from` - one end of the path
    /// * `to` - other end of the path
    /// * `allowed_one` - first square whose occupant is ignored
    /// * `allowed_two` - second square whose occupant is ignored
    ///
    /// # Returns
    ///
    /// `true` when both ends share a row and no non-ignored square in the
    /// inclusive file range is occupied.
    ///
    /// # Panics
    ///
    /// Panics if a file/row pair fails to form a square, which the inclusive
    /// on-board range never produces.
    fn horizontal_path_empty(
        &self,
        from: Square,
        to: Square,
        allowed_one: Square,
        allowed_two: Square,
    ) -> bool {
        if from.row() != to.row() {
            return false;
        }
        let low = from.file().min(to.file());
        let high = from.file().max(to.file());
        for file in low..=high {
            let square = Square::from_file_row(file, from.row()).unwrap();
            if square != allowed_one && square != allowed_two && self.piece_at(square).is_some() {
                return false;
            }
        }
        true
    }

    /// Used for checking that a castling king is not in, through, or finally
    /// left in check.
    ///
    /// The final-square test is performed by the normal post-move legality
    /// check; this helper probes the source and genuinely traversed
    /// intermediate squares.
    ///
    /// # Arguments
    ///
    /// * `mv` - candidate castling move in the internal encoding
    ///
    /// # Returns
    ///
    /// `true` when the move is not a castle, or when the king starts out of
    /// check and every traversed intermediate square is empty after lifting
    /// the participants and is unattacked with the king placed there.
    ///
    /// # Panics
    ///
    /// Panics when the move's source square holds no piece; callers pass
    /// generated moves whose source is occupied.
    fn castle_transit_is_safe(&self, mv: Move) -> bool {
        let moving = self.piece_at(mv.from()).unwrap();
        let Some(right) = self.castle_right_for_move(moving, mv.to()) else {
            return true;
        };
        if self.in_check(moving.color) {
            return false;
        }
        let kingside = matches!(right, 1 | 4);
        let target = Self::castle_king_target(moving.color, kingside);
        if target.file() == mv.from().file() {
            return true;
        }
        let delta = if target.file() > mv.from().file() {
            1
        } else {
            -1
        };
        let mut file = mv.from().file() as i8 + delta;
        // The final square is checked after the complete castle is made. Here
        // we test only genuinely traversed intermediate squares with the king's
        // source vacated and the rook still on its source when possible.
        while file != target.file() as i8 {
            let transit = Square::from_file_row(file as u8, mv.from().row()).unwrap();
            let mut probe = self.clone();
            probe.take(mv.from());
            if probe.piece_at(transit) == Some(Piece::new(moving.color, PieceKind::Rook)) {
                probe.take(transit);
            }
            if probe.piece_at(transit).is_some() {
                return false;
            }
            probe.put(moving, transit);
            if probe.is_square_attacked(transit, moving.color.opposite()) {
                return false;
            }
            file += delta;
        }
        true
    }

    /// Used for deriving the next en-passant target after a pawn double
    /// push.
    ///
    /// Internally generated positions retain the target only when an
    /// opposing pawn attacks it; legality against pins is considered later
    /// by the key.
    ///
    /// # Arguments
    ///
    /// * `moving` - piece that just moved
    /// * `from` - move source square
    /// * `to` - move destination square
    ///
    /// # Returns
    ///
    /// The skipped square when a pawn double push is geometrically
    /// capturable by an enemy pawn, otherwise `None`.
    ///
    /// # Panics
    ///
    /// Panics if the midpoint square fails to form, which a two-row pawn
    /// push on one file never produces.
    fn next_en_passant(&self, moving: Piece, from: Square, to: Square) -> Option<Square> {
        if moving.kind != PieceKind::Pawn || from.row().abs_diff(to.row()) != 2 {
            return None;
        }
        let target = Square::from_file_row(from.file(), (from.row() + to.row()) / 2).unwrap();
        if self.pawn_attackers(target, moving.color.opposite()) != 0 {
            Some(target)
        } else {
            None
        }
    }

    /// Used for retrieving `color` pawns that attack `target` geometrically.
    ///
    /// # Arguments
    ///
    /// * `target` - square being attacked
    /// * `color` - side owning the attacking pawns
    ///
    /// # Returns
    ///
    /// Bitboard of pawns of `color` attacking `target`.
    fn pawn_attackers(&self, target: Square, color: Color) -> u64 {
        movegen::pawn_attackers(self, target, color)
    }

    /// Used for retrieving the en-passant target only when at least one
    /// legal capture exists.
    ///
    /// This normalization prevents irrelevant FEN targets from splitting
    /// repetition-equivalent positions.
    ///
    /// # Returns
    ///
    /// The stored target when some pawn can legally capture onto it,
    /// otherwise `None`.
    fn effective_en_passant(&self) -> Option<Square> {
        let target = self.en_passant?;
        let mut attackers = self.pawn_attackers(target, self.side_to_move);
        while attackers != 0 {
            let from = Square::new(attackers.trailing_zeros() as u8).unwrap();
            attackers &= attackers - 1;
            let mv = Move::new(from, target, None);
            let mut probe = self.clone();
            if let Ok(undo) = probe.make_move(mv) {
                let legal = !probe.in_check(self.side_to_move);
                probe.unmake_move(mv, undo);
                if legal {
                    return Some(target);
                }
            }
        }
        None
    }
}
