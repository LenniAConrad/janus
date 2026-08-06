//! Reversible position transitions translated from the frozen CRTK CPU core.
//!
//! One move updates the mailbox, piece bitboards, color occupancies, cached king
//! squares, castling permissions, en-passant metadata, clocks, and side to move
//! as a single invariant. Public validation is completed before mutation; the
//! returned typed undo record owns exactly the state needed for LIFO reversal.

use super::{CastleUndo, NullUndo, Position, Undo};
use crate::{Color, CoreError, Move, Piece, PieceKind, Square};

/// Used for applying one legal or pseudo-legal move through the shared
/// transition path.
///
/// All validation (side to move, promotion metadata, castling preconditions,
/// and capture targets) is completed before any board mutation. The update
/// then strips castling rights via [`castling_keep_mask`], moves the pieces
/// through the mailbox/bitboard accessors, recomputes the en-passant target,
/// updates both clocks, and flips the side to move as one invariant.
///
/// # Arguments
///
/// * `position` - position mutated in place
/// * `mv` - encoded move to apply
///
/// # Returns
///
/// Typed undo record owning exactly the state needed for LIFO reversal by
/// [`unmake_move`].
///
/// # Errors
///
/// Returns a [`CoreError`] when the source square is empty, the moving piece
/// does not belong to the side to move, promotion metadata is inconsistent,
/// an en-passant capture leaves the board or finds no capturable pawn, the
/// capture target is not capturable, or castling preconditions fail.
///
/// # Panics
///
/// Panics only if a square validated as occupied is unexpectedly empty while
/// pieces are removed, which would indicate a violated internal invariant.
pub(super) fn make_move(position: &mut Position, mv: Move) -> Result<Undo, CoreError> {
    let from = mv.from();
    let encoded_to = mv.to();
    let moving = position
        .piece_at(from)
        .ok_or_else(|| CoreError::new(format!("no piece on {from}")))?;
    if moving.color != position.side_to_move {
        return Err(CoreError::new(format!(
            "piece on {from} is not side to move"
        )));
    }
    validate_promotion(moving, encoded_to, mv)?;

    let castle_right = position.castle_right_for_move(moving, encoded_to);
    let castle = prepare_castle(position, moving, from, castle_right)?;
    let actual_to = castle.map_or(encoded_to, |record| record.king_to);
    let en_passant_capture = castle.is_none()
        && moving.kind == PieceKind::Pawn
        && position.en_passant == Some(encoded_to)
        && position.piece_at(encoded_to).is_none()
        && from.file() != encoded_to.file();
    let capture_square = if en_passant_capture {
        encoded_to
            .offset(0, if moving.color == Color::White { 1 } else { -1 })
            .ok_or_else(|| CoreError::new("en-passant capture leaves the board"))?
    } else {
        actual_to
    };
    let captured = if castle.is_none() {
        validate_capture(position, moving, capture_square, en_passant_capture)?
    } else {
        None
    };

    let undo = Undo {
        moved: moving,
        captured,
        castle,
        castling_rights: position.castling_rights,
        en_passant: position.en_passant,
        halfmove_clock: position.halfmove_clock,
        fullmove_number: position.fullmove_number,
    };

    if position.castling_rights.bits() != 0 {
        position.castling_rights.0 &= castling_keep_mask(position, from, encoded_to);
    }
    position
        .take(from)
        .expect("validated move source remains occupied");
    if let Some((_, square)) = captured {
        position
            .take(square)
            .expect("validated captured square remains occupied");
    }
    if let Some(record) = castle {
        position
            .take(record.rook_from)
            .expect("validated castling rook remains occupied");
        position.put(record.rook, record.rook_to);
    }
    let placed = Piece::new(moving.color, mv.promotion().unwrap_or(moving.kind));
    position.put(placed, actual_to);

    position.en_passant = position.next_en_passant(moving, from, actual_to);
    if moving.kind == PieceKind::Pawn || captured.is_some() {
        position.halfmove_clock = 0;
    } else {
        position.halfmove_clock = position.halfmove_clock.saturating_add(1);
    }
    if position.side_to_move == Color::Black {
        position.fullmove_number = position.fullmove_number.saturating_add(1);
    }
    position.side_to_move = position.side_to_move.opposite();
    Ok(undo)
}

/// Used for reversing the exact LIFO move/undo pair produced by [`make_move`].
///
/// Restores the side to move, castling rights, en-passant target, and both
/// clocks from the undo record, then returns the moved piece (undoing any
/// promotion) plus any captured piece or castling rook to their original
/// squares.
///
/// # Arguments
///
/// * `position` - position mutated back to its exact pre-move state
/// * `mv` - move previously applied by [`make_move`]
/// * `undo` - undo record returned by the matching [`make_move`] call
///
/// # Panics
///
/// Panics if a square recorded by the undo pair is unexpectedly empty, which
/// would indicate the LIFO move/undo discipline was violated.
pub(super) fn unmake_move(position: &mut Position, mv: Move, undo: Undo) {
    position.side_to_move = position.side_to_move.opposite();
    position.castling_rights = undo.castling_rights;
    position.en_passant = undo.en_passant;
    position.halfmove_clock = undo.halfmove_clock;
    position.fullmove_number = undo.fullmove_number;
    if let Some(castle) = undo.castle {
        position
            .take(castle.king_to)
            .expect("castling king remains on its undo square");
        position
            .take(castle.rook_to)
            .expect("castling rook remains on its undo square");
        position.put(undo.moved, mv.from());
        position.put(castle.rook, castle.rook_from);
    } else {
        position
            .take(mv.to())
            .expect("moved piece remains on its undo square");
        position.put(undo.moved, mv.from());
        if let Some((piece, square)) = undo.captured {
            position.put(piece, square);
        }
    }
}

/// Used for applying a reversible search-only null move after the normal
/// check guard.
///
/// Clears the en-passant target and flips the side to move without touching
/// any pieces. Both clocks are captured in the undo record but intentionally
/// left unchanged on the position.
///
/// # Arguments
///
/// * `position` - position mutated in place
///
/// # Returns
///
/// Undo record consumed by [`unmake_null`] to restore the saved state.
///
/// # Errors
///
/// Returns a [`CoreError`] when the side to move is in check, where a null
/// move is illegal.
pub(super) fn make_null(position: &mut Position) -> Result<NullUndo, CoreError> {
    if position.in_check(position.side_to_move) {
        return Err(CoreError::new("cannot make a null move while in check"));
    }
    let undo = NullUndo {
        side_to_move: position.side_to_move,
        en_passant: position.en_passant,
        halfmove_clock: position.halfmove_clock,
        fullmove_number: position.fullmove_number,
    };
    position.en_passant = None;
    position.side_to_move = position.side_to_move.opposite();
    Ok(undo)
}

/// Used for restoring the exact state saved by [`make_null`].
///
/// Writes the recorded side to move, en-passant target, and clocks straight
/// back onto the position; no piece state was changed by the null move.
///
/// # Arguments
///
/// * `position` - position mutated back to its pre-null-move state
/// * `undo` - undo record returned by the matching [`make_null`] call
pub(super) fn unmake_null(position: &mut Position, undo: NullUndo) {
    position.side_to_move = undo.side_to_move;
    position.en_passant = undo.en_passant;
    position.halfmove_clock = undo.halfmove_clock;
    position.fullmove_number = undo.fullmove_number;
}

/// Used for validating promotion metadata before any state is changed.
///
/// Only a pawn may carry a promotion. A pawn reaching the back rank (rank
/// eight for White, rank one for Black) must promote, and a pawn move to any
/// other rank must not carry a promotion.
///
/// # Arguments
///
/// * `moving` - piece found on the move's source square
/// * `target` - encoded destination square
/// * `mv` - move supplying the optional promotion kind
///
/// # Errors
///
/// Returns a [`CoreError`] for a non-pawn promotion, a missing required
/// back-rank promotion, or a promotion away from the back rank.
fn validate_promotion(moving: Piece, target: Square, mv: Move) -> Result<(), CoreError> {
    if mv.promotion().is_some() && moving.kind != PieceKind::Pawn {
        return Err(CoreError::new("only a pawn can promote"));
    }
    if moving.kind == PieceKind::Pawn {
        let promotion_rank = if moving.color == Color::White { 8 } else { 1 };
        if target.rank() == promotion_rank && mv.promotion().is_none() {
            return Err(CoreError::new("pawn move to back rank requires promotion"));
        }
        if target.rank() != promotion_rank && mv.promotion().is_some() {
            return Err(CoreError::new("pawn can promote only on the back rank"));
        }
    }
    Ok(())
}

/// Used for validating one optional captured piece and returning undo
/// metadata.
///
/// # Arguments
///
/// * `position` - position queried before any mutation
/// * `moving` - piece performing the move
/// * `square` - square any captured piece must occupy (the en-passant pawn
///   square for en-passant captures, the destination otherwise)
/// * `en_passant` - whether the move was classified as an en-passant capture
///
/// # Returns
///
/// `Some((piece, square))` describing the captured piece for a capture, or
/// `None` for a quiet move.
///
/// # Errors
///
/// Returns a [`CoreError`] when the target square holds a friendly piece or
/// a king, or when an en-passant capture finds no pawn on `square`, whether
/// it is empty or holds another piece kind.
fn validate_capture(
    position: &Position,
    moving: Piece,
    square: Square,
    en_passant: bool,
) -> Result<Option<(Piece, Square)>, CoreError> {
    let captured = position.piece_at(square);
    if let Some(piece) = captured {
        if piece.color == moving.color || piece.kind == PieceKind::King {
            return Err(CoreError::new("move target is not capturable"));
        }
        if en_passant && piece.kind != PieceKind::Pawn {
            return Err(CoreError::new("en-passant target has no capturable pawn"));
        }
        Ok(Some((piece, square)))
    } else if en_passant {
        Err(CoreError::new("en-passant target has no capturable pawn"))
    } else {
        Ok(None)
    }
}

/// Used for resolving and validating castling rook/target metadata before
/// mutation.
///
/// Right values `1` and `4` select the kingside king/rook targets; the other
/// rights castle queenside. The rook bound to the right must be a friendly
/// rook on its recorded square, and both final squares must be empty unless
/// they coincide with the king's or rook's current square.
///
/// # Arguments
///
/// * `position` - position queried before any mutation
/// * `moving` - castling king piece
/// * `king_from` - king's current square
/// * `right` - resolved castling-right bit, or `None` for a non-castling move
///
/// # Returns
///
/// `Some(CastleUndo)` carrying the rook piece, both rook squares, and the
/// king's destination when the move castles, or `None` when `right` is
/// `None`.
///
/// # Errors
///
/// Returns a [`CoreError`] when the right has no recorded rook square, the
/// rook is missing from that square, or a castling destination is occupied by
/// an uninvolved piece.
fn prepare_castle(
    position: &Position,
    moving: Piece,
    king_from: Square,
    right: Option<u8>,
) -> Result<Option<CastleUndo>, CoreError> {
    let Some(right) = right else {
        return Ok(None);
    };
    let kingside = matches!(right, 1 | 4);
    let rook_from = position
        .castling_rook(right)
        .ok_or_else(|| CoreError::new("castling right has no rook"))?;
    let rook = Piece::new(moving.color, PieceKind::Rook);
    if position.piece_at(rook_from) != Some(rook) {
        return Err(CoreError::new("castling rook is missing"));
    }
    let king_to = Position::castle_king_target(moving.color, kingside);
    let rook_to = Position::castle_rook_target(moving.color, kingside);
    for target in [king_to, rook_to] {
        if target != king_from && target != rook_from && position.piece_at(target).is_some() {
            return Err(CoreError::new("castling destination is occupied"));
        }
    }
    Ok(Some(CastleUndo {
        rook,
        rook_from,
        rook_to,
        king_to,
    }))
}

/// Used for computing the castling permissions retained after two relevant
/// squares change.
///
/// Intersects [`castling_square_keep_mask`] for the move's source and encoded
/// destination so rights are stripped whether a king or rook departs or a
/// rook is captured on its identity square.
///
/// # Arguments
///
/// * `position` - position supplying king and castling-rook identity squares
/// * `first` - move source square
/// * `second` - encoded move destination square
///
/// # Returns
///
/// Bit mask to AND with the current castling-rights bits.
fn castling_keep_mask(position: &Position, first: Square, second: Square) -> u8 {
    castling_square_keep_mask(position, first) & castling_square_keep_mask(position, second)
}

/// Used for computing the permissions retained after one king/rook identity
/// square changes.
///
/// A change on a king's square strips both rights of that color (bits `1 | 2`
/// for White, `4 | 8` for Black); a change on a castling rook's square strips
/// only the single right bound to that rook.
///
/// # Arguments
///
/// * `position` - position supplying king and castling-rook identity squares
/// * `square` - square whose occupant changes during the move
///
/// # Returns
///
/// Bit mask of castling rights that survive the change on `square`.
fn castling_square_keep_mask(position: &Position, square: Square) -> u8 {
    let mut keep = 1 | 2 | 4 | 8;
    if position.kings[Color::White.index()] == Some(square) {
        keep &= !(1 | 2);
    }
    if position.kings[Color::Black.index()] == Some(square) {
        keep &= !(4 | 8);
    }
    for (right, mask) in [(1, !1), (2, !2), (4, !4), (8, !8)] {
        if position.castling_rook(right) == Some(square) {
            keep &= mask;
        }
    }
    keep
}
