//! Shared safe-Rust move generation translated from the frozen CRTK CPU core.
//!
//! This module owns Janus's sole production pseudo/legal generator, attack
//! queries, pin classification, and reusable move scratch. CRTK's bitboard
//! masks, deterministic lookup tables, and legality control flow are adapted to
//! Janus's typed state and pre-existing observable move order.

mod sliding;

use super::{
    Position, BISHOP_DIRECTIONS, KING_OFFSETS, KNIGHT_OFFSETS, QUEEN_DIRECTIONS, ROOK_DIRECTIONS,
};
use crate::{Color, Move, Piece, PieceKind, Square};
pub use sliding::{bishop_attacks, rook_attacks};
use std::sync::OnceLock;

/// Used for sizing the fixed per-ply move scratch, a capacity inherited from
/// CRTK's default that no reachable chess position exceeds.
const MAX_MOVES: usize = 256;
/// Used for initializing safe fixed-capacity move arrays with a placeholder
/// value that never escapes the buffer's logical length.
const EMPTY_MOVE: Move = Move::new(Square::A8, Square::A8, None);
/// Used for masking every square on file A in Janus/CRTK index order.
const FILE_A: u64 = 0x0101_0101_0101_0101;
/// Used for masking every square on file H in Janus/CRTK index order.
const FILE_H: u64 = 0x8080_8080_8080_8080;
/// Used for masking every square on rank eight.
const RANK_8: u64 = 0x0000_0000_0000_00ff;
/// Used for masking every square on rank six.
const RANK_6: u64 = 0x0000_0000_00ff_0000;
/// Used for masking every square on rank three.
const RANK_3: u64 = 0x0000_ff00_0000_0000;
/// Used for masking every square on rank one.
const RANK_1: u64 = 0xff00_0000_0000_0000;
/// Used for indicating whether numeric square indices increase from source to
/// target on each [`QUEEN_DIRECTIONS`] ray, selecting the bit-scan direction
/// that emits slider targets in stable near-to-far order.
const SLIDER_RAY_ASCENDING: [bool; 8] = [false, true, false, true, false, true, false, true];

/// Used for storing the lazily initialized non-slider attack tables and
/// empty-board line guards shared by every position.
static ATTACKS: OnceLock<AttackTables> = OnceLock::new();

/// Reusable bounded move storage for one generation ply.
///
/// Valid moves occupy `moves[..len]`. Chess positions cannot reach the
/// translated 256-move bound; an invariant failure panics safely instead of
/// writing beyond the array.
#[derive(Clone)]
pub(crate) struct MoveBuffer {
    /// Used for storing encoded moves, with unused slots holding
    /// [`EMPTY_MOVE`].
    moves: [Move; MAX_MOVES],
    /// Used for tracking the number of initialized logical entries.
    len: usize,
}

impl Default for MoveBuffer {
    /// Used for creating an empty buffer without heap allocation.
    ///
    /// # Returns
    ///
    /// Buffer with every slot holding [`EMPTY_MOVE`] and a logical length of
    /// zero.
    fn default() -> Self {
        Self {
            moves: [EMPTY_MOVE; MAX_MOVES],
            len: 0,
        }
    }
}

impl MoveBuffer {
    /// Used for removing all logical entries while retaining the fixed
    /// backing storage.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    /// Used for appending one generated move within the proven
    /// chess-position bound.
    ///
    /// # Arguments
    ///
    /// * `mv` - encoded move to append after the current logical entries
    ///
    /// # Panics
    ///
    /// Panics when the buffer already holds [`MAX_MOVES`] entries, which no
    /// reachable chess position produces.
    #[inline]
    fn push(&mut self, mv: Move) {
        assert!(self.len < MAX_MOVES, "move buffer capacity exceeded");
        self.moves[self.len] = mv;
        self.len += 1;
    }

    /// Used for retrieving the initialized move slice in deterministic
    /// generation order.
    ///
    /// # Returns
    ///
    /// Slice covering exactly the logical entries `moves[..len]`.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    /// Used for copying the initialized moves into the stable public vector
    /// representation.
    ///
    /// # Returns
    ///
    /// Newly allocated vector holding the logical entries in buffer order.
    fn to_vec(&self) -> Vec<Move> {
        self.as_slice().to_vec()
    }
}

/// Precomputed pawn, leaper, and empty-board slider-line masks.
///
/// Every table is indexed by the 0..64 square index. The empty-board line
/// masks guard slider lookups and pin scans so magic-table queries only run
/// when a relevant slider actually shares a line with the origin.
struct AttackTables {
    /// Used for looking up knight attacks indexed by origin square.
    knights: [u64; 64],
    /// Used for looking up king attacks indexed by origin square.
    kings: [u64; 64],
    /// Used for looking up pawn attacks indexed first by color and then by
    /// origin square.
    pawns: [[u64; 64]; 2],
    /// Used for guarding rook-slider lookups with empty-board rook lines,
    /// avoiding unnecessary magic-table queries.
    rook_lines: [u64; 64],
    /// Used for guarding bishop-slider lookups with empty-board bishop lines,
    /// avoiding unnecessary magic-table queries.
    bishop_lines: [u64; 64],
    /// Used for emitting slider targets from empty-board rays kept in exact
    /// [`QUEEN_DIRECTIONS`] order for stable emission.
    slider_rays: [[u64; 8]; 64],
}

impl AttackTables {
    /// Used for building the direct translation of CRTK's static
    /// attack-table initializer.
    ///
    /// For each square this fills the knight, king, and per-color pawn jump
    /// masks, the combined empty-board rook and bishop lines, and one
    /// empty-board ray per [`QUEEN_DIRECTIONS`] entry.
    ///
    /// # Returns
    ///
    /// Fully populated table set for all 64 squares.
    fn new() -> Self {
        let mut tables = Self {
            knights: [0; 64],
            kings: [0; 64],
            pawns: [[0; 64]; 2],
            rook_lines: [0; 64],
            bishop_lines: [0; 64],
            slider_rays: [[0; 8]; 64],
        };
        for square in 0_u8..64 {
            let index = square as usize;
            tables.knights[index] = jump_attacks(square, &KNIGHT_OFFSETS);
            tables.kings[index] = jump_attacks(square, &KING_OFFSETS);
            tables.pawns[Color::White.index()][index] = pawn_attacks(square, Color::White);
            tables.pawns[Color::Black.index()][index] = pawn_attacks(square, Color::Black);
            tables.rook_lines[index] =
                line(square, 1, 0) | line(square, -1, 0) | line(square, 0, 1) | line(square, 0, -1);
            tables.bishop_lines[index] = line(square, 1, 1)
                | line(square, 1, -1)
                | line(square, -1, 1)
                | line(square, -1, -1);
            for (direction, &(file_delta, row_delta)) in QUEEN_DIRECTIONS.iter().enumerate() {
                tables.slider_rays[index][direction] = line(square, file_delta, row_delta);
            }
        }
        tables
    }
}

/// Used for retrieving the shared non-slider attack tables, initializing them
/// exactly once on first use.
///
/// # Returns
///
/// Reference to the process-wide [`AttackTables`] singleton.
#[inline]
fn attacks() -> &'static AttackTables {
    ATTACKS.get_or_init(AttackTables::new)
}

/// Used for producing the translated pseudo-legal sequence in pre-entry Janus
/// order.
///
/// # Arguments
///
/// * `position` - position whose side to move is generated for
///
/// # Returns
///
/// Vector of movement-correct moves without any legality filtering.
pub(crate) fn pseudo_legal_moves(position: &Position) -> Vec<Move> {
    let mut moves = MoveBuffer::default();
    generate_pseudo_legal_moves(position, &mut moves);
    moves.to_vec()
}

/// Used for producing every legal move while leaving the caller's immutable
/// position unchanged.
///
/// # Arguments
///
/// * `position` - position whose side to move is generated for
///
/// # Returns
///
/// Vector of fully legal moves in stable Janus order.
pub(crate) fn legal_moves_into(position: &Position, out: &mut Vec<Move>) {
    let mut legal = MoveBuffer::default();
    generate_legal_moves(position, &mut legal, false);
    out.clear();
    out.extend_from_slice(legal.as_slice());
}

/// Used for producing every legal move as a freshly allocated vector.
///
/// # Arguments
///
/// * `position` - position whose side to move is generated for
///
/// # Returns
///
/// The complete ordered legal move list.
pub(crate) fn legal_moves(position: &Position) -> Vec<Move> {
    let mut legal = MoveBuffer::default();
    generate_legal_moves(position, &mut legal, false);
    legal.to_vec()
}

/// Used for producing legal captures and promotions while leaving the input
/// position unchanged.
///
/// # Arguments
///
/// * `position` - position whose side to move is generated for
///
/// # Returns
///
/// Vector of legal tactical moves in stable Janus order.
pub(crate) fn legal_tactical_moves(position: &Position) -> Vec<Move> {
    let mut legal = MoveBuffer::default();
    generate_legal_moves(position, &mut legal, true);
    legal.to_vec()
}

/// Used for filling caller-owned scratch directly with legal moves in stable
/// Janus order.
///
/// `tactical` selects the capture/promotion subsequence used by quiescence.
/// The buffer is cleared before generation so it can be reused across plies.
///
/// # Arguments
///
/// * `position` - position whose side to move is generated for
/// * `legal` - reusable output buffer receiving the legal moves
/// * `tactical` - `true` to restrict output to captures and promotions
pub(crate) fn generate_legal_moves(position: &Position, legal: &mut MoveBuffer, tactical: bool) {
    legal.clear();
    let mut emitter = LegalEmitter::new(position, legal);
    generate_ordered_moves(position, tactical, &mut emitter);
}

/// Used for counting legal moves without materializing the full move list.
///
/// In the common case — a present unchecked king, no absolutely pinned
/// friendly piece, and no usable en-passant target — every movement-correct
/// pawn, knight, and slider move is legal, so those families are counted by
/// popcounting exactly the target masks the ordered walk would emit, with
/// promotions expanded arithmetically. Only king moves and castles pass
/// through the shared legality constraints individually, using `scratch` for
/// their emission. Any other case falls back to exact full generation into
/// `scratch`, so the returned count always equals the length of
/// [`generate_legal_moves`] output.
///
/// # Arguments
///
/// * `position` - position whose side to move is counted for
/// * `scratch` - reusable buffer receiving king/castle or fallback emission
///
/// # Returns
///
/// Number of fully legal moves for the side to move.
pub(crate) fn count_legal_moves(position: &Position, scratch: &mut MoveBuffer) -> usize {
    let color = position.side_to_move;
    let constraints = LegalConstraints::new(position, color);
    let common_case = constraints.king.is_some()
        && constraints.non_king_targets == u64::MAX
        && constraints.pinned == 0
        && en_passant_mask(position, color) == 0;
    scratch.clear();
    let mut emitter = LegalEmitter {
        position,
        color,
        constraints,
        moves: scratch,
        probe: None,
    };
    if !common_case {
        generate_ordered_moves(position, false, &mut emitter);
        return emitter.moves.as_slice().len();
    }

    let bulk = count_pawn_moves(position, color) + count_piece_moves(position, color);
    generate_jumps(
        position,
        color,
        PieceKind::King,
        &KING_OFFSETS,
        &mut emitter,
        false,
    );
    generate_castles(position, color, &mut emitter);
    bulk + emitter.moves.as_slice().len()
}

/// Used for deciding only whether the side to move has any legal move.
///
/// Callers that ask this question are testing for mate or stalemate, and the
/// answer is overwhelmingly `true`, so the cost that matters is the cost of
/// the affirmative case. In the same common case [`count_legal_moves`]
/// identifies — a present unchecked king, no absolutely pinned friendly
/// piece, and no usable en-passant target — every movement-correct pawn,
/// knight, and slider move is legal, so a non-zero bulk popcount already
/// proves existence and no move is emitted, tested, or stored at all. Only
/// when that shortcut does not apply does the query fall back to the exact
/// count.
///
/// The result is identical to `!legal_moves(position).is_empty()` by
/// construction, since the fallback is [`count_legal_moves`], whose count is
/// pinned to the generated list length by the perft cross-checks.
///
/// # Arguments
///
/// * `position` - position whose side to move is queried
///
/// # Returns
///
/// `true` when at least one fully legal move exists.
pub(crate) fn has_legal_move(position: &Position) -> bool {
    let color = position.side_to_move;
    let constraints = LegalConstraints::new(position, color);
    let common_case = constraints.king.is_some()
        && constraints.non_king_targets == u64::MAX
        && constraints.pinned == 0
        && en_passant_mask(position, color) == 0;
    // Both operands are non-negative popcount sums, so `a + b > 0` is exactly
    // `a > 0 || b > 0` -- but `||` stops at the first non-zero instead of
    // always paying for both. Pawns answer first because they do so most
    // often. `count_legal_moves` still backs the uncommon cases, and the sum
    // at the other call site is left alone: there the value IS the count.
    if common_case
        && (count_pawn_moves(position, color) > 0 || count_piece_moves(position, color) > 0)
    {
        return true;
    }
    let mut scratch = MoveBuffer::default();
    count_legal_moves(position, &mut scratch) > 0
}

/// Used for computing the geometric attacks produced by the piece occupying
/// `square`.
///
/// Pawn, knight, and king attacks come from the precomputed tables; sliders
/// consult the magic tables under the current full-board occupancy.
///
/// # Arguments
///
/// * `position` - position supplying the occupant and occupancy
/// * `square` - origin square to query
///
/// # Returns
///
/// Attack bitboard of the occupying piece, or `0` for an empty square.
pub(crate) fn attacks_from(position: &Position, square: Square) -> u64 {
    let Some(piece) = position.piece_at(square) else {
        return 0;
    };
    attacks_for_piece(position, square, piece)
}

/// Used for computing geometric attacks when the caller already knows the
/// piece occupying `square`.
///
/// The supplied piece must match the position's occupant. Keeping that
/// invariant at typed-bitboard callers avoids a redundant mailbox lookup while
/// preserving the same attack tables and occupancy-sensitive slider geometry
/// as [`attacks_from`]. Debug builds verify the caller contract.
///
/// # Arguments
///
/// * `position` - position supplying the matching occupant and occupancy
/// * `square` - occupied origin square to query
/// * `piece` - piece known to occupy `square`
///
/// # Returns
///
/// Attack bitboard of `piece` under the position's current occupancy.
#[inline]
pub(crate) fn attacks_for_piece(position: &Position, square: Square, piece: Piece) -> u64 {
    debug_assert_eq!(position.piece_at(square), Some(piece));
    let index = square.index() as usize;
    match piece.kind {
        PieceKind::Pawn => attacks().pawns[piece.color.index()][index],
        PieceKind::Knight => attacks().knights[index],
        PieceKind::Bishop => bishop_attacks(square.index(), position.occupancy()),
        PieceKind::Rook => rook_attacks(square.index(), position.occupancy()),
        PieceKind::Queen => {
            bishop_attacks(square.index(), position.occupancy())
                | rook_attacks(square.index(), position.occupancy())
        }
        PieceKind::King => attacks().kings[index],
    }
}

/// Used for testing whether `square` is attacked by `by` in the current
/// occupancy.
///
/// # Arguments
///
/// * `position` - position supplying piece bitboards and occupancy
/// * `square` - square under test
/// * `by` - attacking color
///
/// # Returns
///
/// `true` when at least one piece of `by` attacks `square`.
pub(crate) fn is_square_attacked(position: &Position, square: Square, by: Color) -> bool {
    is_square_attacked_with(position, square, by, position.occupancy(), 0)
}

/// Used for testing whether `color`'s cached king square is attacked.
///
/// # Arguments
///
/// * `position` - position supplying the cached king square
/// * `color` - side whose king is tested
///
/// # Returns
///
/// `true` when the king exists and its square is attacked by the opponent;
/// `false` when the king is absent.
pub(crate) fn is_king_attacked(position: &Position, color: Color) -> bool {
    position
        .king_square(color)
        .is_some_and(|king| is_square_attacked(position, king, color.opposite()))
}

/// Used for finding the `color` pawns that attack `target` geometrically.
///
/// # Arguments
///
/// * `position` - position supplying the pawn bitboard
/// * `target` - attacked square
/// * `color` - color of the attacking pawns
///
/// # Returns
///
/// Bitboard of `color` pawns whose attack masks include `target`.
pub(crate) fn pawn_attackers(position: &Position, target: Square, color: Color) -> u64 {
    attacks().pawns[color.opposite().index()][target.index() as usize]
        & position.piece_bitboard(Piece::new(color, PieceKind::Pawn))
}

/// Sink receiving movement-correct moves from the one stable ordered walk.
///
/// Implementations decide whether an emitted move is stored unfiltered
/// (pseudo-legal) or passed through the shared legality constraints first.
trait MoveEmitter {
    /// Used for observing one move together with its moving piece family.
    ///
    /// # Arguments
    ///
    /// * `kind` - piece family performing the move
    /// * `mv` - encoded movement-correct move
    fn emit(&mut self, kind: PieceKind, mv: Move);
}

/// Unfiltered sink backing the public pseudo-legal contracts.
struct PseudoEmitter<'a> {
    /// Used for storing emitted moves in caller-owned fixed-capacity output.
    moves: &'a mut MoveBuffer,
}

impl MoveEmitter for PseudoEmitter<'_> {
    /// Used for appending one movement-correct move without a legality
    /// constraint.
    ///
    /// # Arguments
    ///
    /// * `_kind` - moving piece family, unused by the pseudo-legal sink
    /// * `mv` - encoded move appended to the output buffer
    #[inline]
    fn emit(&mut self, _kind: PieceKind, mv: Move) {
        self.moves.push(mv);
    }
}

/// Immutable geometry shared by every move in one direct legal generation.
struct LegalConstraints {
    /// Used for anchoring absolute-pin rays at the friendly king, when one
    /// exists.
    king: Option<Square>,
    /// Used for restricting non-king destinations to squares that resolve
    /// zero or one checks; zero denotes double check.
    non_king_targets: u64,
    /// Used for marking friendly sources whose departure can uncover an
    /// enemy slider.
    pinned: u64,
}

impl LegalConstraints {
    /// Used for deriving checker, evasion, and pin geometry once for one
    /// side to move.
    ///
    /// Without a friendly king every destination is allowed and nothing is
    /// pinned. With one checker the non-king targets shrink to that
    /// checker's capture-or-block squares; with two or more checkers they
    /// become empty so only king moves survive.
    ///
    /// # Arguments
    ///
    /// * `position` - position supplying attack and pin state
    /// * `color` - side to move whose constraints are derived
    ///
    /// # Returns
    ///
    /// Constraint set valid for every move of one generation pass.
    fn new(position: &Position, color: Color) -> Self {
        let Some(king) = position.king_square(color) else {
            return Self {
                king: None,
                non_king_targets: u64::MAX,
                pinned: 0,
            };
        };
        let checkers = attackers_to(position, king, color.opposite());
        let non_king_targets = match checkers.count_ones() {
            0 => u64::MAX,
            1 => evasion_targets(position, king, square(checkers.trailing_zeros() as u8)),
            _ => 0,
        };
        Self {
            king: Some(king),
            non_king_targets,
            pinned: pinned_pieces(position, color),
        }
    }
}

/// Direct legal sink shared by immutable search and mutable perft callers.
struct LegalEmitter<'a> {
    /// Used for reading rules and attack state from the unchanged root
    /// position.
    position: &'a Position,
    /// Used for identifying the side whose moves are being generated.
    color: Color,
    /// Used for applying the generation-wide legality geometry.
    constraints: LegalConstraints,
    /// Used for storing accepted moves in caller-owned fixed-capacity legal
    /// output.
    moves: &'a mut MoveBuffer,
    /// Used for probing en passant and castling on one lazily cloned state,
    /// reused across probes within the generation.
    probe: Option<Position>,
}

impl<'a> LegalEmitter<'a> {
    /// Used for creating one direct sink after deriving the immutable
    /// constraints.
    ///
    /// # Arguments
    ///
    /// * `position` - unchanged root position for the whole generation
    /// * `moves` - caller-owned output buffer for accepted legal moves
    ///
    /// # Returns
    ///
    /// Emitter bound to the position's side to move with no probe clone yet.
    fn new(position: &'a Position, moves: &'a mut MoveBuffer) -> Self {
        let color = position.side_to_move;
        Self {
            position,
            color,
            constraints: LegalConstraints::new(position, color),
            moves,
            probe: None,
        }
    }

    /// Used for testing a non-castling king destination under its final
    /// occupancy.
    ///
    /// Removes the king from its source square, adds it to the destination,
    /// and excludes any captured piece from the attacker set so sliders see
    /// through neither stale square.
    ///
    /// # Arguments
    ///
    /// * `mv` - candidate king move
    ///
    /// # Returns
    ///
    /// `true` when the destination is not attacked under the adjusted
    /// occupancy.
    fn king_destination_is_safe(&self, mv: Move) -> bool {
        let captured = self.position.color_occupancy(self.color.opposite()) & mv.to().bit();
        let occupancy = (self.position.occupancy() & !mv.from().bit()) | mv.to().bit();
        !is_square_attacked_with(
            self.position,
            mv.to(),
            self.color.opposite(),
            occupancy,
            captured,
        )
    }

    /// Used for testing a castle's final king square under the post-castle
    /// occupancy without cloning or mutating the position.
    ///
    /// Both castle participants are relocated to their FIDE destinations in a
    /// derived occupancy, then the king's final square is scanned for enemy
    /// attackers through the shared occupancy-parameterized attack query. This
    /// reproduces exactly the make/unmake destination probe it replaces —
    /// castling never captures, so the enemy attacker sets are unchanged — at
    /// the cost of one attack scan instead of a full working clone.
    ///
    /// # Arguments
    ///
    /// * `right` - single castling permission bit identifying the castle
    ///
    /// # Returns
    ///
    /// `true` when the king's final square is unattacked after the castle, and
    /// `false` when the castle geometry cannot be resolved.
    fn castle_destination_is_safe(&self, right: u8) -> bool {
        match self.position.castle_destination_and_occupancy(right) {
            Some((king_to, occupancy)) => !is_square_attacked_with(
                self.position,
                king_to,
                self.color.opposite(),
                occupancy,
                0,
            ),
            None => false,
        }
    }

    /// Used for performing a bounded exact special-move probe on one reused
    /// clone.
    ///
    /// The clone is created on first use, mutated with the candidate move,
    /// checked for a safe king, and restored, so the root position is never
    /// touched.
    ///
    /// # Arguments
    ///
    /// * `mv` - candidate en-passant or castling move
    ///
    /// # Returns
    ///
    /// `true` when the move applies cleanly and leaves the friendly king
    /// unattacked.
    fn probe_is_legal(&mut self, mv: Move) -> bool {
        let probe = self.probe.get_or_insert_with(|| self.position.clone());
        let Ok(undo) = probe.make_move(mv) else {
            return false;
        };
        let legal = !is_king_attacked(probe, self.color);
        probe.unmake_move(mv, undo);
        legal
    }

    /// Used for applying the direct ordinary constraints or the exact
    /// special-move boundary.
    ///
    /// Castling requires a safe transit plus an exact probe; other king
    /// moves use the adjusted-occupancy destination test; en passant always
    /// probes exactly. Remaining moves must land inside the evasion targets
    /// and, when pinned, stay on their king ray.
    ///
    /// # Arguments
    ///
    /// * `moving_kind` - piece family performing the move
    /// * `mv` - candidate movement-correct move
    ///
    /// # Returns
    ///
    /// `true` when the move is fully legal for the side to move.
    fn move_is_legal(&mut self, moving_kind: PieceKind, mv: Move) -> bool {
        if moving_kind == PieceKind::King {
            let king_piece = Piece::new(self.color, PieceKind::King);
            if let Some(right) = self.position.castle_right_for_move(king_piece, mv.to()) {
                return self.position.castle_transit_is_safe(mv)
                    && self.castle_destination_is_safe(right);
            }
            return self.king_destination_is_safe(mv);
        }

        if moving_kind == PieceKind::Pawn
            && self
                .position
                .is_en_passant_move(Piece::new(self.color, PieceKind::Pawn), mv)
        {
            return self.probe_is_legal(mv);
        }

        if self.constraints.non_king_targets & mv.to().bit() == 0 {
            return false;
        }
        if self.constraints.pinned & mv.from().bit() != 0 {
            let Some(king_square) = self.constraints.king else {
                return false;
            };
            if !same_king_ray(king_square, mv.from(), mv.to()) {
                return false;
            }
        }
        true
    }
}

impl MoveEmitter for LegalEmitter<'_> {
    /// Used for appending one move only when the direct shared constraints
    /// accept it.
    ///
    /// # Arguments
    ///
    /// * `kind` - piece family performing the move
    /// * `mv` - encoded movement-correct candidate move
    #[inline]
    fn emit(&mut self, kind: PieceKind, mv: Move) {
        if self.move_is_legal(kind, mv) {
            self.moves.push(mv);
        }
    }
}

/// Used for emitting all pseudo-legal moves through the translated bitboard
/// path.
///
/// # Arguments
///
/// * `position` - position whose side to move is generated for
/// * `moves` - reusable output buffer, cleared before generation
fn generate_pseudo_legal_moves(position: &Position, moves: &mut MoveBuffer) {
    moves.clear();
    let mut emitter = PseudoEmitter { moves };
    generate_ordered_moves(position, false, &mut emitter);
}

/// Used for walking every piece family once in the stable observable Janus
/// order.
///
/// The order is pawns, knights, bishops, rooks, queens, kings, and finally
/// castles; castles are skipped entirely in tactical mode.
///
/// # Arguments
///
/// * `position` - position whose side to move is generated for
/// * `tactical` - `true` to restrict emission to captures and promotions
/// * `emitter` - sink deciding whether emitted moves are kept
fn generate_ordered_moves(position: &Position, tactical: bool, emitter: &mut impl MoveEmitter) {
    let color = position.side_to_move;
    generate_pawns(position, color, emitter, tactical);
    generate_jumps(
        position,
        color,
        PieceKind::Knight,
        &KNIGHT_OFFSETS,
        emitter,
        tactical,
    );
    generate_sliders(position, color, PieceKind::Bishop, emitter, tactical);
    generate_sliders(position, color, PieceKind::Rook, emitter, tactical);
    generate_sliders(position, color, PieceKind::Queen, emitter, tactical);
    generate_jumps(
        position,
        color,
        PieceKind::King,
        &KING_OFFSETS,
        emitter,
        tactical,
    );
    if !tactical {
        generate_castles(position, color, emitter);
    }
}

/// Used for emitting one color's pawn moves from CRTK target masks in Janus
/// origin order.
///
/// Single pushes, double pushes, and both capture directions are computed as
/// whole-board shift masks first, then walked per origin pawn so emission
/// order follows ascending source indices. Tactical mode restricts single
/// pushes to promotions and drops double pushes; the enemy king is never a
/// capture target.
///
/// # Arguments
///
/// * `position` - position supplying pawn, occupancy, and en-passant state
/// * `color` - side whose pawns are generated
/// * `emitter` - sink deciding whether emitted moves are kept
/// * `tactical` - `true` to emit only captures, en passant, and promotions
fn generate_pawns(
    position: &Position,
    color: Color,
    emitter: &mut impl MoveEmitter,
    tactical: bool,
) {
    let pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
    let PawnTargets {
        single,
        double,
        left: left_capture,
        right: right_capture,
    } = pawn_target_masks(position, color);

    let (single_sources, double_sources, left_sources, right_sources, promotion_sources) =
        if color == Color::White {
            (
                single << 8,
                double << 16,
                left_capture << 9,
                right_capture << 7,
                (single & RANK_8) << 8,
            )
        } else {
            (
                single >> 8,
                double >> 16,
                left_capture >> 7,
                right_capture >> 9,
                (single & RANK_1) >> 8,
            )
        };
    let emitted_single_sources = if tactical {
        promotion_sources
    } else {
        single_sources
    };
    let emitted_double_sources = if tactical { 0 } else { double_sources };
    let mut sources = pawns;
    while sources != 0 {
        let from_index = sources.trailing_zeros() as u8;
        sources &= sources - 1;
        let from = square(from_index);
        let from_bit = 1_u64 << from_index;

        if emitted_single_sources & from_bit != 0 {
            let target = if color == Color::White {
                from_index - 8
            } else {
                from_index + 8
            };
            emit_pawn_move(from, square(target), emitter);
        }

        if emitted_double_sources & from_bit != 0 {
            let target = if color == Color::White {
                from_index - 16
            } else {
                from_index + 16
            };
            emitter.emit(PieceKind::Pawn, Move::new(from, square(target), None));
        }

        if left_sources & from_bit != 0 {
            let target = if color == Color::White {
                from_index - 9
            } else {
                from_index + 7
            };
            emit_pawn_move(from, square(target), emitter);
        }

        if right_sources & from_bit != 0 {
            let target = if color == Color::White {
                from_index - 7
            } else {
                from_index + 9
            };
            emit_pawn_move(from, square(target), emitter);
        }
    }
}

/// Whole-board pawn destination masks shared by emission and bulk counting.
///
/// Each mask maps one pawn source to at most one destination, so the mask
/// population equals the number of distinct pawn moves before promotion
/// expansion.
struct PawnTargets {
    /// Used for holding single-push destinations onto empty squares.
    single: u64,
    /// Used for holding double-push destinations through empty squares.
    double: u64,
    /// Used for holding lower-file-diagonal capture destinations.
    left: u64,
    /// Used for holding higher-file-diagonal capture destinations.
    right: u64,
}

/// Used for computing the translated CRTK pawn destination masks once per
/// side.
///
/// Captures target enemy pieces except the enemy king plus the usable
/// en-passant square; pushes require empty destinations, and double pushes
/// additionally require an empty intermediate square.
///
/// # Arguments
///
/// * `position` - position supplying pawn, occupancy, and en-passant state
/// * `color` - side whose pawn destinations are computed
///
/// # Returns
///
/// The four whole-board destination masks for `color`'s pawns.
fn pawn_target_masks(position: &Position, color: Color) -> PawnTargets {
    let pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
    let empty = !position.occupancy();
    let enemies = position.color_occupancy(color.opposite())
        & !position.piece_bitboard(Piece::new(color.opposite(), PieceKind::King));
    let capturable = enemies | en_passant_mask(position, color);

    if color == Color::White {
        let single = (pawns >> 8) & empty;
        PawnTargets {
            single,
            double: ((single & RANK_3) >> 8) & empty,
            left: ((pawns & !FILE_A) >> 9) & capturable,
            right: ((pawns & !FILE_H) >> 7) & capturable,
        }
    } else {
        let single = (pawns << 8) & empty;
        PawnTargets {
            single,
            double: ((single & RANK_6) << 8) & empty,
            left: ((pawns & !FILE_A) << 7) & capturable,
            right: ((pawns & !FILE_H) << 9) & capturable,
        }
    }
}

/// Used for counting one side's pawn moves directly from the destination
/// masks.
///
/// Valid only in the bulk common case: an unchecked king, no pinned friendly
/// piece, and no usable en-passant target, where every movement-correct pawn
/// move is legal. Each destination inside a mask stems from exactly one
/// source pawn, so popcounts equal move counts; back-rank destinations add
/// three extra moves each for the four-way promotion expansion.
///
/// # Arguments
///
/// * `position` - position supplying pawn, occupancy, and en-passant state
/// * `color` - side whose pawn moves are counted
///
/// # Returns
///
/// Number of legal pawn moves including every promotion choice.
fn count_pawn_moves(position: &Position, color: Color) -> usize {
    let targets = pawn_target_masks(position, color);
    let promotion_rank = if color == Color::White {
        RANK_8
    } else {
        RANK_1
    };
    let ordinary = targets.single.count_ones()
        + targets.double.count_ones()
        + targets.left.count_ones()
        + targets.right.count_ones();
    let promoting = (targets.single & promotion_rank).count_ones()
        + (targets.left & promotion_rank).count_ones()
        + (targets.right & promotion_rank).count_ones();
    (ordinary + 3 * promoting) as usize
}

/// Used for counting one side's knight and slider moves directly from attack
/// masks.
///
/// Valid only in the bulk common case: an unchecked king, no pinned friendly
/// piece, and no usable en-passant target, where every movement-correct
/// non-king move is legal. Queen counts split into disjoint diagonal and
/// orthogonal attack sets, so combined bishop/queen and rook/queen scans sum
/// exactly.
///
/// # Arguments
///
/// * `position` - position supplying piece bitboards and occupancy
/// * `color` - side whose knight and slider moves are counted
///
/// # Returns
///
/// Number of legal knight, bishop, rook, and queen moves.
fn count_piece_moves(position: &Position, color: Color) -> usize {
    let occupancy = position.occupancy();
    let allowed = !position.color_occupancy(color)
        & !position.piece_bitboard(Piece::new(color.opposite(), PieceKind::King));
    let mut count = 0_u32;
    let mut knights = position.piece_bitboard(Piece::new(color, PieceKind::Knight));
    while knights != 0 {
        let from = knights.trailing_zeros() as usize;
        knights &= knights - 1;
        count += (attacks().knights[from] & allowed).count_ones();
    }
    let queens = position.piece_bitboard(Piece::new(color, PieceKind::Queen));
    let mut diagonal = position.piece_bitboard(Piece::new(color, PieceKind::Bishop)) | queens;
    while diagonal != 0 {
        let from = diagonal.trailing_zeros() as u8;
        diagonal &= diagonal - 1;
        count += (bishop_attacks(from, occupancy) & allowed).count_ones();
    }
    let mut orthogonal = position.piece_bitboard(Piece::new(color, PieceKind::Rook)) | queens;
    while orthogonal != 0 {
        let from = orthogonal.trailing_zeros() as u8;
        orthogonal &= orthogonal - 1;
        count += (rook_attacks(from, occupancy) & allowed).count_ones();
    }
    count as usize
}

/// Used for emitting one ordinary pawn move or its four promotion variants.
///
/// Back-rank destinations expand into knight, bishop, rook, and queen
/// promotions in that fixed order; every other destination emits one plain
/// pawn move.
///
/// # Arguments
///
/// * `from` - pawn origin square
/// * `to` - pawn destination square
/// * `emitter` - sink deciding whether emitted moves are kept
fn emit_pawn_move(from: Square, to: Square, emitter: &mut impl MoveEmitter) {
    if to.bit() & (RANK_8 | RANK_1) != 0 {
        for promotion in [
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
        ] {
            emitter.emit(PieceKind::Pawn, Move::new(from, to, Some(promotion)));
        }
    } else {
        emitter.emit(PieceKind::Pawn, Move::new(from, to, None));
    }
}

/// Used for emitting leaper targets through translated masks and the stable
/// Janus delta order.
///
/// Attack masks come from the shared tables, but targets are walked in the
/// caller's offset order so emission matches the pre-existing observable
/// sequence. The enemy king is never a target.
///
/// # Arguments
///
/// * `position` - position supplying piece bitboards and occupancies
/// * `color` - side whose leapers are generated
/// * `kind` - leaper family, either knight or king
/// * `offsets` - file/row deltas fixing the per-origin emission order
/// * `emitter` - sink deciding whether emitted moves are kept
/// * `captures_only` - `true` to restrict targets to enemy pieces
///
/// # Panics
///
/// Panics via `unreachable!` when `kind` is neither
/// [`PieceKind::Knight`] nor [`PieceKind::King`].
fn generate_jumps(
    position: &Position,
    color: Color,
    kind: PieceKind,
    offsets: &[(i8, i8)],
    emitter: &mut impl MoveEmitter,
    captures_only: bool,
) {
    let own = position.color_occupancy(color);
    let enemy_king = position.piece_bitboard(Piece::new(color.opposite(), PieceKind::King));
    let captures = position.color_occupancy(color.opposite()) & !enemy_king;
    let mut pieces = position.piece_bitboard(Piece::new(color, kind));
    while pieces != 0 {
        let from_index = pieces.trailing_zeros() as u8;
        pieces &= pieces - 1;
        let from = square(from_index);
        let attacks = match kind {
            PieceKind::Knight => attacks().knights[from_index as usize],
            PieceKind::King => attacks().kings[from_index as usize],
            _ => unreachable!("jump generation requires a leaper"),
        };
        let targets = if captures_only {
            attacks & captures
        } else {
            attacks & !own & !enemy_king
        };
        for &(file_delta, row_delta) in offsets {
            if let Some(to_index) = offset(from_index, file_delta, row_delta) {
                if targets & (1_u64 << to_index) != 0 {
                    emitter.emit(kind, Move::new(from, square(to_index), None));
                }
            }
        }
    }
}

/// Used for emitting one slider's target mask in exact Janus direction and
/// distance order.
///
/// Directions index into the [`QUEEN_DIRECTIONS`]-ordered rays: bishops use
/// rays 0..4, rooks 4..8, and queens all eight. Within each ray the scan
/// direction from [`SLIDER_RAY_ASCENDING`] emits targets from nearest to
/// farthest.
///
/// # Arguments
///
/// * `kind` - slider family selecting the ray range
/// * `from` - slider origin square
/// * `targets` - precomputed reachable-target bitboard
/// * `rays` - empty-board rays of the origin in [`QUEEN_DIRECTIONS`] order
/// * `emitter` - sink deciding whether emitted moves are kept
///
/// # Panics
///
/// Panics via `unreachable!` when `kind` is not a bishop, rook, or queen.
#[inline]
fn emit_slider_targets(
    kind: PieceKind,
    from: Square,
    targets: u64,
    rays: &[u64; 8],
    emitter: &mut impl MoveEmitter,
) {
    let directions = match kind {
        PieceKind::Bishop => 0..4,
        PieceKind::Rook => 4..8,
        PieceKind::Queen => 0..8,
        _ => unreachable!("slider emission requires bishop, rook, or queen"),
    };
    for direction in directions {
        let mut ray_targets = targets & rays[direction];
        while ray_targets != 0 {
            let to_index = if SLIDER_RAY_ASCENDING[direction] {
                ray_targets.trailing_zeros() as u8
            } else {
                (u64::BITS - 1 - ray_targets.leading_zeros()) as u8
            };
            ray_targets ^= 1_u64 << to_index;
            emitter.emit(kind, Move::new(from, square(to_index), None));
        }
    }
}

/// Used for emitting slider targets through magic attacks and Janus's stable
/// ray adapter.
///
/// Attack sets come from the magic tables under the current occupancy; the
/// enemy king is masked out of every target set before emission through
/// [`emit_slider_targets`].
///
/// # Arguments
///
/// * `position` - position supplying piece bitboards and occupancy
/// * `color` - side whose sliders are generated
/// * `kind` - slider family, one of bishop, rook, or queen
/// * `emitter` - sink deciding whether emitted moves are kept
/// * `captures_only` - `true` to restrict targets to enemy pieces
///
/// # Panics
///
/// Panics via `unreachable!` when `kind` is not a bishop, rook, or queen.
fn generate_sliders(
    position: &Position,
    color: Color,
    kind: PieceKind,
    emitter: &mut impl MoveEmitter,
    captures_only: bool,
) {
    let occupancy = position.occupancy();
    let own = position.color_occupancy(color);
    let enemy_king = position.piece_bitboard(Piece::new(color.opposite(), PieceKind::King));
    let captures = position.color_occupancy(color.opposite()) & !enemy_king;
    let slider_rays = &attacks().slider_rays;
    let mut pieces = position.piece_bitboard(Piece::new(color, kind));
    while pieces != 0 {
        let from_index = pieces.trailing_zeros() as u8;
        pieces &= pieces - 1;
        let from = square(from_index);
        let attacks = match kind {
            PieceKind::Bishop => bishop_attacks(from_index, occupancy),
            PieceKind::Rook => rook_attacks(from_index, occupancy),
            PieceKind::Queen => {
                bishop_attacks(from_index, occupancy) | rook_attacks(from_index, occupancy)
            }
            _ => unreachable!("slider generation requires bishop, rook, or queen"),
        };
        let targets = if captures_only {
            attacks & captures
        } else {
            attacks & !own & !enemy_king
        };
        emit_slider_targets(
            kind,
            from,
            targets,
            &slider_rays[from_index as usize],
            emitter,
        );
    }
}

/// Used for emitting path-clear castles in the pre-entry kingside-then-
/// queenside order.
///
/// Each side is emitted only when its right is held, the path between the
/// involved squares is clear, and a concrete encoded target exists. Legality
/// of the transit itself is left to the emitter.
///
/// # Arguments
///
/// * `position` - position supplying rights, rook squares, and path state
/// * `color` - side whose castles are generated
/// * `emitter` - sink deciding whether emitted moves are kept
fn generate_castles(position: &Position, color: Color, emitter: &mut impl MoveEmitter) {
    let Some(king) = position.kings[color.index()] else {
        return;
    };
    for kingside in [true, false] {
        let right = Position::right_for(color, kingside);
        if position.castling_rights.contains(right) && position.castle_path_is_clear(right) {
            if let Some(target) = position.castle_move_target(right) {
                emitter.emit(PieceKind::King, Move::new(king, target, None));
            }
        }
    }
}

/// Used for computing a safe en-passant target bit for pseudo pawn
/// generation.
///
/// The stored target only contributes when it sits on the capturing color's
/// valid rank (six for White, three for Black) and the target square itself
/// is empty.
///
/// # Arguments
///
/// * `position` - position supplying the stored en-passant target
/// * `color` - capturing side
///
/// # Returns
///
/// Single-bit mask of the usable en-passant target, or `0` when none exists.
fn en_passant_mask(position: &Position, color: Color) -> u64 {
    position.en_passant.map_or(0, |target| {
        let valid_rank = if color == Color::White {
            target.rank() == 6
        } else {
            target.rank() == 3
        };
        if valid_rank && position.piece_at(target).is_none() {
            target.bit()
        } else {
            0
        }
    })
}

/// Used for finding the concrete pieces of `by` that currently attack
/// `target`.
///
/// Pawn, knight, and king attackers come straight from the shared tables;
/// bishop/queen and rook/queen attackers consult the magic tables only when
/// the empty-board line guards show a candidate on a shared line.
///
/// # Arguments
///
/// * `position` - position supplying piece bitboards and occupancy
/// * `target` - attacked square
/// * `by` - attacking color
///
/// # Returns
///
/// Bitboard of every `by` piece attacking `target` under the current
/// occupancy.
fn attackers_to(position: &Position, target: Square, by: Color) -> u64 {
    attackers_to_with(position, target, by, position.occupancy())
}

/// Used for finding every `by` piece attacking `target` under a supplied
/// occupancy.
///
/// Identical to [`attackers_to`] except that slider rays are traced through
/// `occupancy` rather than the position's own. Callers simulating a move —
/// where a piece has left its origin and may have opened a line — must pass
/// the post-move occupancy, because an attacker discovered by the departure is
/// a real attacker and the position's occupancy would hide it.
///
/// # Arguments
///
/// * `position` - position supplying piece bitboards
/// * `target` - square under attack
/// * `by` - attacking color
/// * `occupancy` - full-board occupancy used for slider lookups
///
/// # Returns
///
/// Bitboard of every `by` piece attacking `target` under `occupancy`.
pub(crate) fn attackers_to_with(
    position: &Position,
    target: Square,
    by: Color,
    occupancy: u64,
) -> u64 {
    let index = target.index() as usize;
    let queens = position.piece_bitboard(Piece::new(by, PieceKind::Queen));
    let bishop_queens = position.piece_bitboard(Piece::new(by, PieceKind::Bishop)) | queens;
    let rook_queens = position.piece_bitboard(Piece::new(by, PieceKind::Rook)) | queens;
    let bishop_checkers = if attacks().bishop_lines[index] & bishop_queens != 0 {
        bishop_attacks(target.index(), occupancy) & bishop_queens
    } else {
        0
    };
    let rook_checkers = if attacks().rook_lines[index] & rook_queens != 0 {
        rook_attacks(target.index(), occupancy) & rook_queens
    } else {
        0
    };
    (attacks().pawns[by.opposite().index()][index]
        & position.piece_bitboard(Piece::new(by, PieceKind::Pawn)))
        | (attacks().knights[index] & position.piece_bitboard(Piece::new(by, PieceKind::Knight)))
        | (attacks().kings[index] & position.piece_bitboard(Piece::new(by, PieceKind::King)))
        | bishop_checkers
        | rook_checkers
}

/// Used for computing the capture-or-block destinations that resolve one
/// concrete checker.
///
/// Non-slider checkers (and sliders not aligned with the king) can only be
/// captured, so the result is the checker's own bit. An aligned bishop,
/// rook, or queen additionally allows every square between the king and the
/// checker as a blocking destination.
///
/// # Arguments
///
/// * `position` - position supplying the checker's identity
/// * `king` - checked king's square
/// * `checker` - the single checking piece's square
///
/// # Returns
///
/// Bitboard of destinations that end the check for non-king moves.
fn evasion_targets(position: &Position, king: Square, checker: Square) -> u64 {
    let Some(piece) = position.piece_at(checker) else {
        return checker.bit();
    };
    let file_delta = checker.file() as i8 - king.file() as i8;
    let row_delta = checker.row() as i8 - king.row() as i8;
    let orthogonal = (file_delta == 0 || row_delta == 0) && (file_delta != 0 || row_delta != 0);
    let diagonal = file_delta != 0 && file_delta.abs() == row_delta.abs();
    let aligned_slider = match piece.kind {
        PieceKind::Bishop => diagonal,
        PieceKind::Rook => orthogonal,
        PieceKind::Queen => orthogonal || diagonal,
        _ => false,
    };
    if !aligned_slider {
        return checker.bit();
    }

    let file_step = file_delta.signum();
    let row_step = row_delta.signum();
    let mut targets = 0_u64;
    let mut distance = 1_i8;
    while let Some(index) = offset(king.index(), file_step * distance, row_step * distance) {
        let candidate = square(index);
        targets |= candidate.bit();
        if candidate == checker {
            return targets;
        }
        distance += 1;
    }
    checker.bit()
}

/// Used for testing whether `to` remains on the pinned source's ray from
/// `king`.
///
/// The cross-product equality requires king, source, and destination to be
/// collinear, while the positive dot product keeps the destination on the
/// same side of the king as the source.
///
/// # Arguments
///
/// * `king` - friendly king anchoring the pin ray
/// * `from` - pinned piece's source square
/// * `to` - candidate destination square
///
/// # Returns
///
/// `true` when moving from `from` to `to` stays on the king ray.
fn same_king_ray(king: Square, from: Square, to: Square) -> bool {
    let from_file = i16::from(from.file()) - i16::from(king.file());
    let from_row = i16::from(from.row()) - i16::from(king.row());
    let to_file = i16::from(to.file()) - i16::from(king.file());
    let to_row = i16::from(to.row()) - i16::from(king.row());
    from_file * to_row == from_row * to_file && from_file * to_file + from_row * to_row > 0
}

/// Used for finding friendly pieces absolutely pinned to `color`'s king.
///
/// Rook and bishop directions are scanned from the king outward, but only
/// when the empty-board line guards show a matching enemy slider on a shared
/// line. A missing king pins nothing.
///
/// # Arguments
///
/// * `position` - position supplying piece bitboards and occupancy
/// * `color` - side whose pinned pieces are collected
///
/// # Returns
///
/// Bitboard of `color` pieces whose departure would expose the king to an
/// enemy slider.
pub(crate) fn pinned_pieces(position: &Position, color: Color) -> u64 {
    let Some(king) = position.king_square(color) else {
        return 0;
    };
    let enemy = color.opposite();
    let enemy_rook_queen = position.piece_bitboard(Piece::new(enemy, PieceKind::Rook))
        | position.piece_bitboard(Piece::new(enemy, PieceKind::Queen));
    let enemy_bishop_queen = position.piece_bitboard(Piece::new(enemy, PieceKind::Bishop))
        | position.piece_bitboard(Piece::new(enemy, PieceKind::Queen));
    let own = position.color_occupancy(color);
    let occupied = position.occupancy();
    let mut pinned = 0_u64;
    if attacks().rook_lines[king.index() as usize] & enemy_rook_queen != 0 {
        for &(file_delta, row_delta) in &ROOK_DIRECTIONS {
            pinned |=
                pinned_in_direction(king, own, occupied, enemy_rook_queen, file_delta, row_delta);
        }
    }
    if attacks().bishop_lines[king.index() as usize] & enemy_bishop_queen != 0 {
        for &(file_delta, row_delta) in &BISHOP_DIRECTIONS {
            pinned |= pinned_in_direction(
                king,
                own,
                occupied,
                enemy_bishop_queen,
                file_delta,
                row_delta,
            );
        }
    }
    pinned
}

/// Used for finding the single friendly blocker pinned on one ray, if any.
///
/// Walking outward from the king, the first occupied square must hold a
/// friendly piece and the next occupied square must hold a matching enemy
/// slider for a pin to exist; any other arrangement yields no pin.
///
/// # Arguments
///
/// * `king` - friendly king anchoring the ray
/// * `own` - friendly occupancy bitboard
/// * `occupied` - full-board occupancy bitboard
/// * `enemy_sliders` - enemy sliders able to pin along this ray family
/// * `file_delta` - per-step file component of the ray
/// * `row_delta` - per-step row component of the ray
///
/// # Returns
///
/// Single-bit mask of the pinned friendly piece, or `0` when nothing is
/// pinned on this ray.
fn pinned_in_direction(
    king: Square,
    own: u64,
    occupied: u64,
    enemy_sliders: u64,
    file_delta: i8,
    row_delta: i8,
) -> u64 {
    let mut candidate = 0_u64;
    let mut distance = 1_i8;
    while let Some(index) = offset(king.index(), file_delta * distance, row_delta * distance) {
        let bit = 1_u64 << index;
        if occupied & bit != 0 {
            if candidate == 0 && own & bit != 0 {
                candidate = bit;
            } else {
                return if candidate != 0 && enemy_sliders & bit != 0 {
                    candidate
                } else {
                    0
                };
            }
        }
        distance += 1;
    }
    0
}

/// Used for testing attacks with a caller-supplied occupancy and optional
/// removed attacker.
///
/// Non-slider attackers are checked first from the shared tables; slider
/// checks run only when the empty-board line guards admit a candidate.
/// `excluded_attacker` removes pieces (such as one being captured) from
/// every attacker set without mutating the position.
///
/// # Arguments
///
/// * `position` - position supplying piece bitboards
/// * `square` - square under test
/// * `by` - attacking color
/// * `occupancy` - full-board occupancy used for slider lookups
/// * `excluded_attacker` - bitboard of `by` pieces to ignore
///
/// # Returns
///
/// `true` when a non-excluded `by` piece attacks `square` under `occupancy`.
fn is_square_attacked_with(
    position: &Position,
    square: Square,
    by: Color,
    occupancy: u64,
    excluded_attacker: u64,
) -> bool {
    let index = square.index() as usize;
    let included = !excluded_attacker;
    let pawn_sources = attacks().pawns[by.opposite().index()][index];
    if pawn_sources & position.piece_bitboard(Piece::new(by, PieceKind::Pawn)) & included != 0
        || attacks().knights[index]
            & position.piece_bitboard(Piece::new(by, PieceKind::Knight))
            & included
            != 0
        || attacks().kings[index]
            & position.piece_bitboard(Piece::new(by, PieceKind::King))
            & included
            != 0
    {
        return true;
    }

    let queens = position.piece_bitboard(Piece::new(by, PieceKind::Queen)) & included;
    let bishop_queens =
        (position.piece_bitboard(Piece::new(by, PieceKind::Bishop)) & included) | queens;
    if attacks().bishop_lines[index] & bishop_queens != 0
        && bishop_attacks(square.index(), occupancy) & bishop_queens != 0
    {
        return true;
    }
    let rook_queens =
        (position.piece_bitboard(Piece::new(by, PieceKind::Rook)) & included) | queens;
    attacks().rook_lines[index] & rook_queens != 0
        && rook_attacks(square.index(), occupancy) & rook_queens != 0
}

/// Used for building an edge-clipped jump attack mask for one origin square.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `offsets` - file/row deltas of the leaper family
///
/// # Returns
///
/// Bitboard of every offset target that stays on the board.
fn jump_attacks(square: u8, offsets: &[(i8, i8)]) -> u64 {
    offsets.iter().fold(0_u64, |mask, &(file, row)| {
        mask | offset(square, file, row).map_or(0, |target| 1_u64 << target)
    })
}

/// Used for building a one-direction empty-board line excluding its origin.
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
fn line(square: u8, file_delta: i8, row_delta: i8) -> u64 {
    let mut mask = 0_u64;
    let mut distance = 1_i8;
    while let Some(target) = offset(square, file_delta * distance, row_delta * distance) {
        mask |= 1_u64 << target;
        distance += 1;
    }
    mask
}

/// Used for building pawn attacks from one origin square for the requested
/// color.
///
/// White pawns attack toward decreasing row indices and Black pawns toward
/// increasing row indices, on both diagonal files that stay on the board.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `color` - color of the attacking pawn
///
/// # Returns
///
/// Bitboard of at most two diagonal attack targets.
fn pawn_attacks(square: u8, color: Color) -> u64 {
    let row_delta = if color == Color::White { -1 } else { 1 };
    [-1, 1].into_iter().fold(0_u64, |mask, file_delta| {
        mask | offset(square, file_delta, row_delta).map_or(0, |target| 1_u64 << target)
    })
}

/// Used for applying a signed file/top-origin-row delta to a validated
/// square index.
///
/// # Arguments
///
/// * `square` - origin square index in 0..64
/// * `file_delta` - signed file displacement
/// * `row_delta` - signed top-origin row displacement
///
/// # Returns
///
/// `Some` target index when the displaced coordinates stay on the board,
/// `None` otherwise.
fn offset(square: u8, file_delta: i8, row_delta: i8) -> Option<u8> {
    let file = (square & 7) as i8 + file_delta;
    let row = (square >> 3) as i8 + row_delta;
    if (0..8).contains(&file) && (0..8).contains(&row) {
        Some((row * 8 + file) as u8)
    } else {
        None
    }
}

/// Used for converting a proven in-range board index to its typed square.
///
/// # Arguments
///
/// * `index` - board index proven to lie in 0..64 by the generation logic
///
/// # Returns
///
/// Typed square for the given index.
///
/// # Panics
///
/// Panics when `index` is not a valid board index, which would indicate a
/// violated move-generation invariant.
#[inline]
fn square(index: u8) -> Square {
    Square::new(index).expect("move-generation index is on board")
}
