//! Parameterized twin of the released classical evaluation for HCE tuning.
//!
//! The release evaluator in the parent module bakes every weight into named
//! constants. Texel-style tuning needs three additional capabilities that
//! must never disturb the release path:
//!
//! 1. [`ClassicalParams`](crate::classical::tuning::ClassicalParams): a
//!    runtime struct covering every *linear* tunable scalar, whose
//!    [`Default`] reproduces the released constants exactly.
//! 2. [`params_breakdown`](crate::classical::tuning::params_breakdown): an
//!    integer re-evaluation that consumes the parameter struct and matches
//!    [`Classical::breakdown`](crate::Classical::breakdown) bit-for-bit at
//!    the default parameters (release rich attack-state flavor only).
//! 3. [`linear_model`](crate::classical::tuning::linear_model): a
//!    per-position sparse coefficient extraction such
//!    that the white-relative score is, up to bounded integer-rounding
//!    residue, `sum(param * (mg_coeff * phase + eg_coeff * (MAX_PHASE -
//!    phase)) / MAX_PHASE)` plus a position constant holding the frozen
//!    non-linear terms.
//!
//! Deliberately excluded from the parameter vector (stage 1 of the 2026-07-24
//! blueprint): `TEMPO`, `SEARCH_SCORE_SCALE_PERCENT`, `MAX_PHASE` and the
//! phase weights, king sentinels, the dead compact flavor, the quiet-move
//! ordering prior, and the non-linear interiors (the passed-pawn squared-
//! advance reward,
//! bad-bishop products, king-pressure `flank^2`/`attackers^2` quadratics, the
//! queen-absent percentage, and the single-attacker halving).

use super::{
    activity_term, bad_bishop_penalty, bit_count, board_square, camp_mask, can_castle_either,
    center_squares, color_sign, diagonal_attacks, file_mask, game_phase, is_castled_king_square,
    is_outpost, is_passed, king_distance, king_flank_mask, king_ring, knight_attacks,
    mobility_term, orthogonal_attacks, outpost_mask, pawn_attack_mask, pawn_push_mask, rank_mask,
    relative_rank, same_flank, shelter_file_count, square_offset, tapered_score, AttackMaps,
    ClassicalBreakdown, KING_ATTACKERS_SQUARED, MAX_PHASE, PASSED_PAWN_QUADRATIC, TEMPO,
};
use janus_core::{Color, Piece, PieceKind, Position, Square};

/// Used for sizing the flattened linear parameter vector.
///
/// The layout is frozen by [`ClassicalParams::to_flat`] and documented by
/// [`descriptors`]; every consumer indexes the same 223 slots.
pub const PARAM_COUNT: usize = 223;

/// Used for bounding the integer-rounding residue between the released
/// integer evaluation and the smooth linear model at the default parameters.
///
/// The bound sums the worst cases of every truncation and symmetric-rounding
/// site the smooth model replaces with exact real arithmetic: one truncated
/// taper per piece (up to 32), blocked-passer halving, bad-bishop and
/// trapped-rook endgame halving, per-king shelter/castling truncation, the
/// three tapered symmetric roundings, and per-king pressure truncation.
pub const RESIDUAL_BOUND: f64 = 64.0;

/// Every linear tunable scalar of the release rich attack-state evaluator.
///
/// Field defaults reproduce the released constants in the parent module
/// exactly; [`params_breakdown`] at [`ClassicalParams::default`] is verified
/// to equal [`Classical::breakdown`](super::Classical::breakdown)
/// bit-for-bit. Fields are grouped exactly
/// like the constants they shadow.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub struct ClassicalParams {
    /// Used for the base centipawn material values, pawn through queen.
    ///
    /// Shadows the first five entries of `MATERIAL`; the king entry stays a
    /// structural zero.
    pub material: [i32; 5],
    /// Used for the pawn middlegame advance multiplier (`advance * 4`).
    pub pawn_advance_mg: i32,
    /// Used for the pawn middlegame centralization multiplier (`center * -3`).
    pub pawn_center_mg: i32,
    /// Used for the pawn endgame advance multiplier (`advance * 10`).
    pub pawn_advance_eg: i32,
    /// Used for the pawn endgame centralization multiplier (`center * -3`).
    pub pawn_center_eg: i32,
    /// Used for the knight middlegame centralization multiplier
    /// (`center * 5`).
    pub knight_center_mg: i32,
    /// Used for the knight endgame centralization multiplier (`center * 3`).
    pub knight_center_eg: i32,
    /// Used for the single-edge knight placement penalty (`10`).
    pub knight_edge_penalty_mg: i32,
    /// Used for the corner knight placement penalty (`24`).
    pub knight_corner_penalty_mg: i32,
    /// Used for the bishop middlegame centralization multiplier
    /// (`center * 4`).
    pub bishop_center_mg: i32,
    /// Used for the bishop middlegame advance multiplier (`advance * 1`).
    pub bishop_advance_mg: i32,
    /// Used for the bishop endgame centralization multiplier (`center * 2`).
    pub bishop_center_eg: i32,
    /// Used for the rook middlegame advance multiplier (`advance * 3`).
    pub rook_advance_mg: i32,
    /// Used for the rook endgame centralization multiplier (`center * 0`).
    pub rook_center_eg: i32,
    /// Used for the rook endgame advance multiplier (`advance * 2`).
    pub rook_advance_eg: i32,
    /// Used for the queen middlegame centralization multiplier
    /// (`center * 5`).
    pub queen_center_mg: i32,
    /// Used for the queen endgame centralization multiplier (`center * 3`).
    pub queen_center_eg: i32,
    /// Used for the king middlegame centralization multiplier, applied
    /// negatively (`-center * 8`).
    pub king_center_mg: i32,
    /// Used for the king middlegame advance multiplier, applied negatively
    /// (`- advance * 2`).
    pub king_advance_mg: i32,
    /// Used for the king endgame centralization multiplier (`center * 5`).
    pub king_center_eg: i32,
    /// Used for the king endgame advance multiplier (`advance * 2`).
    pub king_advance_eg: i32,
    /// Used for the middlegame knight safe-mobility table.
    pub knight_mobility_mg: [i32; 9],
    /// Used for the endgame knight safe-mobility table.
    pub knight_mobility_eg: [i32; 9],
    /// Used for the middlegame bishop safe-mobility table.
    pub bishop_mobility_mg: [i32; 14],
    /// Used for the endgame bishop safe-mobility table.
    pub bishop_mobility_eg: [i32; 14],
    /// Used for the middlegame rook safe-mobility table.
    pub rook_mobility_mg: [i32; 15],
    /// Used for the endgame rook safe-mobility table.
    pub rook_mobility_eg: [i32; 15],
    /// Used for the middlegame queen safe-mobility table.
    pub queen_mobility_mg: [i32; 28],
    /// Used for the endgame queen safe-mobility table.
    pub queen_mobility_eg: [i32; 28],
    /// Used for the per-extra-pawn doubled-pawn penalty (`13`).
    pub doubled_pawn_penalty: i32,
    /// Used for the per-pawn isolated-pawn penalty (`12`).
    pub isolated_pawn_penalty: i32,
    /// Used for the flat passed-pawn base bonus (`8`); the quadratic
    /// `5 * advance^2` interior stays constant in stage 1.
    pub passed_pawn_base: i32,
    /// Used for the bishop-pair bonus (`39`).
    pub bishop_pair: i32,
    /// Used for the rook-on-open-file bonus (`20`).
    pub rook_open_file: i32,
    /// Used for the rook-on-semi-open-file bonus (`10`).
    pub rook_semi_open_file: i32,
    /// Used for the phase-scaled castled-king placement bonus (`28`).
    pub king_castled_mg: i32,
    /// Used for the phase-scaled per-file king shelter bonus (`9`).
    pub king_shelter_mg: i32,
    /// Used for the king-zone attacker weights, pawn through queen.
    ///
    /// Shadows the first five entries of `KING_ATTACK_WEIGHT`; the king entry
    /// stays a structural zero.
    pub king_attack_weight: [i32; 5],
    /// Used for the per-ring-hit king pressure (`KING_RING_ATTACK`, `15`).
    pub king_ring_attack: i32,
    /// Used for the per-weak-zone-square king pressure (`KING_WEAK_SQUARE`,
    /// `9`).
    pub king_weak_square: i32,
    /// Used for the per-defended-flank-square relief (`KING_FLANK_DEFENSE`,
    /// `5`).
    pub king_flank_defense: i32,
    /// Used for the pawnless-flank king pressure (`KING_PAWNLESS_FLANK`,
    /// `18`).
    pub king_pawnless_flank: i32,
    /// Used for the flat in-check penalty added after phase scaling (`35`).
    pub king_check_penalty: i32,
    /// Used for the knight outpost middlegame bonus (`KNIGHT_OUTPOST.0`).
    pub knight_outpost_mg: i32,
    /// Used for the knight outpost endgame bonus (`KNIGHT_OUTPOST.1`).
    pub knight_outpost_eg: i32,
    /// Used for the bishop outpost middlegame bonus (`BISHOP_OUTPOST.0`).
    pub bishop_outpost_mg: i32,
    /// Used for the bishop outpost endgame bonus (`BISHOP_OUTPOST.1`).
    pub bishop_outpost_eg: i32,
    /// Used for the reachable knight outpost middlegame bonus (`13`).
    pub knight_reachable_outpost_mg: i32,
    /// Used for the reachable knight outpost endgame bonus (`6`).
    pub knight_reachable_outpost_eg: i32,
    /// Used for the reachable bishop outpost middlegame bonus (`7`).
    pub bishop_reachable_outpost_mg: i32,
    /// Used for the reachable bishop outpost endgame bonus (`3`).
    pub bishop_reachable_outpost_eg: i32,
    /// Used for the minor-behind-own-pawn middlegame bonus (`8`).
    pub minor_behind_pawn_mg: i32,
    /// Used for the minor-behind-own-pawn endgame bonus (`5`).
    pub minor_behind_pawn_eg: i32,
    /// Used for the knight distance-to-own-king middlegame weight (`2`).
    pub knight_king_distance_mg: i32,
    /// Used for the bishop distance-to-own-king middlegame weight (`2`).
    pub bishop_king_distance_mg: i32,
    /// Used for the bishop long-diagonal middlegame bonus (`8`).
    pub bishop_long_diagonal_mg: i32,
    /// Used for the bishop long-diagonal endgame bonus (`4`).
    pub bishop_long_diagonal_eg: i32,
    /// Used for the rook-on-queen-file middlegame bonus (`6`).
    pub rook_queen_file_mg: i32,
    /// Used for the rook-on-seventh middlegame bonus (`18`).
    pub rook_seventh_mg: i32,
    /// Used for the rook-on-seventh endgame bonus (`27`).
    pub rook_seventh_eg: i32,
    /// Used for the trapped-rook penalty while castling stays available
    /// (`10`); the endgame half stays structurally `penalty / 2`.
    pub trapped_rook_castle: i32,
    /// Used for the trapped-rook penalty once castling is gone (`16`); the
    /// endgame half stays structurally `penalty / 2`.
    pub trapped_rook_no_castle: i32,
    /// Used for the advanced-safe-queen middlegame bonus (`7`).
    pub queen_advanced_mg: i32,
    /// Used for the advanced-safe-queen endgame bonus (`10`).
    pub queen_advanced_eg: i32,
    /// Used for the hanging-piece middlegame threat (`HANGING_THREAT.0`).
    pub hanging_threat_mg: i32,
    /// Used for the hanging-piece endgame threat (`HANGING_THREAT.1`).
    pub hanging_threat_eg: i32,
    /// Used for the restricted-square middlegame threat
    /// (`RESTRICTED_THREAT_MG`).
    pub restricted_threat_mg: i32,
    /// Used for the safe-pawn-attack middlegame threat
    /// (`SAFE_PAWN_THREAT.0`).
    pub safe_pawn_threat_mg: i32,
    /// Used for the safe-pawn-attack endgame threat (`SAFE_PAWN_THREAT.1`).
    pub safe_pawn_threat_eg: i32,
    /// Used for the pawn-push-attack middlegame threat
    /// (`PAWN_PUSH_THREAT.0`).
    pub pawn_push_threat_mg: i32,
    /// Used for the pawn-push-attack endgame threat (`PAWN_PUSH_THREAT.1`).
    pub pawn_push_threat_eg: i32,
    /// Used for the safe knight-on-queen middlegame threat
    /// (`QUEEN_KNIGHT_THREAT.0`).
    pub queen_knight_threat_mg: i32,
    /// Used for the safe knight-on-queen endgame threat
    /// (`QUEEN_KNIGHT_THREAT.1`).
    pub queen_knight_threat_eg: i32,
    /// Used for the doubly supported slider-on-queen middlegame threat
    /// (`QUEEN_SLIDER_THREAT.0`).
    pub queen_slider_threat_mg: i32,
    /// Used for the doubly supported slider-on-queen endgame threat
    /// (`QUEEN_SLIDER_THREAT.1`).
    pub queen_slider_threat_eg: i32,
    /// Used for the minor-attacks-pawn middlegame threat (`6`).
    pub minor_threat_pawn_mg: i32,
    /// Used for the minor-attacks-pawn endgame threat (`15`).
    pub minor_threat_pawn_eg: i32,
    /// Used for the minor-attacks-minor middlegame threat (`32`).
    pub minor_threat_minor_mg: i32,
    /// Used for the minor-attacks-minor endgame threat (`25`).
    pub minor_threat_minor_eg: i32,
    /// Used for the minor-attacks-rook middlegame threat (`48`).
    pub minor_threat_rook_mg: i32,
    /// Used for the minor-attacks-rook endgame threat (`37`).
    pub minor_threat_rook_eg: i32,
    /// Used for the minor-attacks-queen middlegame threat (`57`).
    pub minor_threat_queen_mg: i32,
    /// Used for the minor-attacks-queen endgame threat (`70`).
    pub minor_threat_queen_eg: i32,
    /// Used for the rook-attacks-pawn middlegame threat (`4`).
    pub rook_threat_pawn_mg: i32,
    /// Used for the rook-attacks-pawn endgame threat (`28`).
    pub rook_threat_pawn_eg: i32,
    /// Used for the rook-attacks-minor middlegame threat (`24`).
    pub rook_threat_minor_mg: i32,
    /// Used for the rook-attacks-minor endgame threat (`36`).
    pub rook_threat_minor_eg: i32,
    /// Used for the rook-attacks-rook middlegame threat (`0`).
    pub rook_threat_rook_mg: i32,
    /// Used for the rook-attacks-rook endgame threat (`18`).
    pub rook_threat_rook_eg: i32,
    /// Used for the rook-attacks-queen middlegame threat (`41`).
    pub rook_threat_queen_mg: i32,
    /// Used for the rook-attacks-queen endgame threat (`36`).
    pub rook_threat_queen_eg: i32,
}

impl Default for ClassicalParams {
    /// Used for reproducing every released constant exactly.
    ///
    /// # Returns
    ///
    /// Parameters equal to the frozen release values in the parent module.
    fn default() -> Self {
        Self {
            material: [98, 357, 382, 585, 1211],
            pawn_advance_mg: 4,
            pawn_center_mg: -3,
            pawn_advance_eg: 10,
            pawn_center_eg: -3,
            knight_center_mg: 5,
            knight_center_eg: 3,
            knight_edge_penalty_mg: 10,
            knight_corner_penalty_mg: 24,
            bishop_center_mg: 4,
            bishop_advance_mg: 1,
            bishop_center_eg: 2,
            rook_advance_mg: 3,
            rook_center_eg: 0,
            rook_advance_eg: 2,
            queen_center_mg: 5,
            queen_center_eg: 3,
            king_center_mg: 8,
            king_advance_mg: 2,
            king_center_eg: 5,
            king_advance_eg: 2,
            knight_mobility_mg: super::KNIGHT_MOBILITY_MG,
            knight_mobility_eg: super::KNIGHT_MOBILITY_EG,
            bishop_mobility_mg: super::BISHOP_MOBILITY_MG,
            bishop_mobility_eg: super::BISHOP_MOBILITY_EG,
            rook_mobility_mg: super::ROOK_MOBILITY_MG,
            rook_mobility_eg: super::ROOK_MOBILITY_EG,
            queen_mobility_mg: super::QUEEN_MOBILITY_MG,
            queen_mobility_eg: super::QUEEN_MOBILITY_EG,
            doubled_pawn_penalty: 13,
            isolated_pawn_penalty: 12,
            passed_pawn_base: 8,
            bishop_pair: 39,
            rook_open_file: 20,
            rook_semi_open_file: 10,
            king_castled_mg: 28,
            king_shelter_mg: 9,
            king_attack_weight: [-2, 10, 8, 12, 19],
            king_ring_attack: 15,
            king_weak_square: 9,
            king_flank_defense: 5,
            king_pawnless_flank: 18,
            king_check_penalty: 35,
            knight_outpost_mg: 27,
            knight_outpost_eg: 14,
            bishop_outpost_mg: 15,
            bishop_outpost_eg: 8,
            knight_reachable_outpost_mg: 13,
            knight_reachable_outpost_eg: 6,
            bishop_reachable_outpost_mg: 7,
            bishop_reachable_outpost_eg: 3,
            minor_behind_pawn_mg: 8,
            minor_behind_pawn_eg: 5,
            knight_king_distance_mg: 2,
            bishop_king_distance_mg: 2,
            bishop_long_diagonal_mg: 8,
            bishop_long_diagonal_eg: 4,
            rook_queen_file_mg: 6,
            rook_seventh_mg: 18,
            rook_seventh_eg: 27,
            trapped_rook_castle: 10,
            trapped_rook_no_castle: 16,
            queen_advanced_mg: 7,
            queen_advanced_eg: 10,
            hanging_threat_mg: 21,
            hanging_threat_eg: 25,
            restricted_threat_mg: 4,
            safe_pawn_threat_mg: 43,
            safe_pawn_threat_eg: 30,
            pawn_push_threat_mg: 13,
            pawn_push_threat_eg: 8,
            queen_knight_threat_mg: 7,
            queen_knight_threat_eg: 7,
            queen_slider_threat_mg: 15,
            queen_slider_threat_eg: 4,
            minor_threat_pawn_mg: 6,
            minor_threat_pawn_eg: 15,
            minor_threat_minor_mg: 32,
            minor_threat_minor_eg: 25,
            minor_threat_rook_mg: 48,
            minor_threat_rook_eg: 37,
            minor_threat_queen_mg: 57,
            minor_threat_queen_eg: 70,
            rook_threat_pawn_mg: 4,
            rook_threat_pawn_eg: 28,
            rook_threat_minor_mg: 24,
            rook_threat_minor_eg: 36,
            rook_threat_rook_mg: 0,
            rook_threat_rook_eg: 18,
            rook_threat_queen_mg: 41,
            rook_threat_queen_eg: 36,
        }
    }
}

impl ClassicalParams {
    /// Used for exposing every parameter slot in the frozen flat order.
    ///
    /// This is the single definition of the flat layout; [`Self::to_flat`],
    /// [`Self::from_flat`], and (by tested agreement) [`descriptors`] all
    /// derive from it.
    ///
    /// # Returns
    ///
    /// Mutable references to all [`PARAM_COUNT`] scalars in layout order.
    fn slots_mut(&mut self) -> Vec<&mut i32> {
        let mut slots: Vec<&mut i32> = Vec::with_capacity(PARAM_COUNT);
        slots.extend(self.material.iter_mut());
        slots.push(&mut self.pawn_advance_mg);
        slots.push(&mut self.pawn_center_mg);
        slots.push(&mut self.pawn_advance_eg);
        slots.push(&mut self.pawn_center_eg);
        slots.push(&mut self.knight_center_mg);
        slots.push(&mut self.knight_center_eg);
        slots.push(&mut self.knight_edge_penalty_mg);
        slots.push(&mut self.knight_corner_penalty_mg);
        slots.push(&mut self.bishop_center_mg);
        slots.push(&mut self.bishop_advance_mg);
        slots.push(&mut self.bishop_center_eg);
        slots.push(&mut self.rook_advance_mg);
        slots.push(&mut self.rook_center_eg);
        slots.push(&mut self.rook_advance_eg);
        slots.push(&mut self.queen_center_mg);
        slots.push(&mut self.queen_center_eg);
        slots.push(&mut self.king_center_mg);
        slots.push(&mut self.king_advance_mg);
        slots.push(&mut self.king_center_eg);
        slots.push(&mut self.king_advance_eg);
        slots.extend(self.knight_mobility_mg.iter_mut());
        slots.extend(self.knight_mobility_eg.iter_mut());
        slots.extend(self.bishop_mobility_mg.iter_mut());
        slots.extend(self.bishop_mobility_eg.iter_mut());
        slots.extend(self.rook_mobility_mg.iter_mut());
        slots.extend(self.rook_mobility_eg.iter_mut());
        slots.extend(self.queen_mobility_mg.iter_mut());
        slots.extend(self.queen_mobility_eg.iter_mut());
        slots.push(&mut self.doubled_pawn_penalty);
        slots.push(&mut self.isolated_pawn_penalty);
        slots.push(&mut self.passed_pawn_base);
        slots.push(&mut self.bishop_pair);
        slots.push(&mut self.rook_open_file);
        slots.push(&mut self.rook_semi_open_file);
        slots.push(&mut self.king_castled_mg);
        slots.push(&mut self.king_shelter_mg);
        slots.extend(self.king_attack_weight.iter_mut());
        slots.push(&mut self.king_ring_attack);
        slots.push(&mut self.king_weak_square);
        slots.push(&mut self.king_flank_defense);
        slots.push(&mut self.king_pawnless_flank);
        slots.push(&mut self.king_check_penalty);
        slots.push(&mut self.knight_outpost_mg);
        slots.push(&mut self.knight_outpost_eg);
        slots.push(&mut self.bishop_outpost_mg);
        slots.push(&mut self.bishop_outpost_eg);
        slots.push(&mut self.knight_reachable_outpost_mg);
        slots.push(&mut self.knight_reachable_outpost_eg);
        slots.push(&mut self.bishop_reachable_outpost_mg);
        slots.push(&mut self.bishop_reachable_outpost_eg);
        slots.push(&mut self.minor_behind_pawn_mg);
        slots.push(&mut self.minor_behind_pawn_eg);
        slots.push(&mut self.knight_king_distance_mg);
        slots.push(&mut self.bishop_king_distance_mg);
        slots.push(&mut self.bishop_long_diagonal_mg);
        slots.push(&mut self.bishop_long_diagonal_eg);
        slots.push(&mut self.rook_queen_file_mg);
        slots.push(&mut self.rook_seventh_mg);
        slots.push(&mut self.rook_seventh_eg);
        slots.push(&mut self.trapped_rook_castle);
        slots.push(&mut self.trapped_rook_no_castle);
        slots.push(&mut self.queen_advanced_mg);
        slots.push(&mut self.queen_advanced_eg);
        slots.push(&mut self.hanging_threat_mg);
        slots.push(&mut self.hanging_threat_eg);
        slots.push(&mut self.restricted_threat_mg);
        slots.push(&mut self.safe_pawn_threat_mg);
        slots.push(&mut self.safe_pawn_threat_eg);
        slots.push(&mut self.pawn_push_threat_mg);
        slots.push(&mut self.pawn_push_threat_eg);
        slots.push(&mut self.queen_knight_threat_mg);
        slots.push(&mut self.queen_knight_threat_eg);
        slots.push(&mut self.queen_slider_threat_mg);
        slots.push(&mut self.queen_slider_threat_eg);
        slots.push(&mut self.minor_threat_pawn_mg);
        slots.push(&mut self.minor_threat_pawn_eg);
        slots.push(&mut self.minor_threat_minor_mg);
        slots.push(&mut self.minor_threat_minor_eg);
        slots.push(&mut self.minor_threat_rook_mg);
        slots.push(&mut self.minor_threat_rook_eg);
        slots.push(&mut self.minor_threat_queen_mg);
        slots.push(&mut self.minor_threat_queen_eg);
        slots.push(&mut self.rook_threat_pawn_mg);
        slots.push(&mut self.rook_threat_pawn_eg);
        slots.push(&mut self.rook_threat_minor_mg);
        slots.push(&mut self.rook_threat_minor_eg);
        slots.push(&mut self.rook_threat_rook_mg);
        slots.push(&mut self.rook_threat_rook_eg);
        slots.push(&mut self.rook_threat_queen_mg);
        slots.push(&mut self.rook_threat_queen_eg);
        slots
    }

    /// Used for flattening the parameters into the frozen vector layout.
    ///
    /// # Panics
    ///
    /// Panics if the slot layout does not contain exactly [`PARAM_COUNT`]
    /// entries.
    ///
    /// # Returns
    ///
    /// The [`PARAM_COUNT`] parameter values in layout order.
    #[must_use]
    pub fn to_flat(&self) -> Vec<i32> {
        let mut copy = self.clone();
        let flat: Vec<i32> = copy.slots_mut().into_iter().map(|slot| *slot).collect();
        assert_eq!(flat.len(), PARAM_COUNT, "flat layout size is frozen");
        flat
    }

    /// Used for rebuilding parameters from the frozen vector layout.
    ///
    /// # Arguments
    ///
    /// * `flat` - exactly [`PARAM_COUNT`] values in layout order
    ///
    /// # Panics
    ///
    /// Panics when `flat` does not contain exactly [`PARAM_COUNT`] values.
    ///
    /// # Returns
    ///
    /// Parameters whose [`Self::to_flat`] reproduces `flat`.
    #[must_use]
    pub fn from_flat(flat: &[i32]) -> Self {
        assert_eq!(flat.len(), PARAM_COUNT, "flat layout size is frozen");
        let mut params = Self::default();
        for (slot, value) in params.slots_mut().into_iter().zip(flat.iter()) {
            *slot = *value;
        }
        params
    }
}

/// One flat parameter slot's name, release constant site, and default.
///
/// Produced by [`descriptors`] in the exact [`ClassicalParams::to_flat`]
/// order so reports can map fitted values back to `classical.rs`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamDescriptor {
    /// Used for naming the slot, matching the [`ClassicalParams`] field
    /// (array fields carry a bracketed index).
    pub name: String,
    /// Used for locating the released constant or literal in `classical.rs`.
    pub site: String,
    /// Used for holding the released default value.
    pub default: i32,
}

/// Used for describing every flat parameter slot in layout order.
///
/// # Panics
///
/// Panics if the generated list does not contain exactly [`PARAM_COUNT`]
/// entries; a unit test additionally proves the defaults equal
/// [`ClassicalParams::default`] flattened.
///
/// # Returns
///
/// One [`ParamDescriptor`] per flat slot, in [`ClassicalParams::to_flat`]
/// order.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn descriptors() -> Vec<ParamDescriptor> {
    let mut list: Vec<ParamDescriptor> = Vec::with_capacity(PARAM_COUNT);
    let mut push = |name: String, site: &str, default: i32| {
        list.push(ParamDescriptor {
            name,
            site: site.to_owned(),
            default,
        });
    };
    for (index, default) in [98, 357, 382, 585, 1211].into_iter().enumerate() {
        push(
            format!("material[{index}]"),
            "const MATERIAL (entry per PieceKind, king stays 0)",
            default,
        );
    }
    let placement = "fn tapered_piece_square formula multiplier";
    push("pawn_advance_mg".into(), placement, 4);
    push("pawn_center_mg".into(), placement, -3);
    push("pawn_advance_eg".into(), placement, 10);
    push("pawn_center_eg".into(), placement, -3);
    push("knight_center_mg".into(), placement, 5);
    push("knight_center_eg".into(), placement, 3);
    push(
        "knight_edge_penalty_mg".into(),
        "fn edge_penalty single-edge literal",
        10,
    );
    push(
        "knight_corner_penalty_mg".into(),
        "fn edge_penalty corner literal",
        24,
    );
    push("bishop_center_mg".into(), placement, 4);
    push("bishop_advance_mg".into(), placement, 1);
    push("bishop_center_eg".into(), placement, 2);
    push("rook_advance_mg".into(), placement, 3);
    push("rook_center_eg".into(), placement, 0);
    push("rook_advance_eg".into(), placement, 2);
    push("queen_center_mg".into(), placement, 5);
    push("queen_center_eg".into(), placement, 3);
    push("king_center_mg".into(), placement, 8);
    push("king_advance_mg".into(), placement, 2);
    push("king_center_eg".into(), placement, 5);
    push("king_advance_eg".into(), placement, 2);
    let mobility_tables: [(&str, &str, &[i32]); 8] = [
        (
            "knight_mobility_mg",
            "const KNIGHT_MOBILITY_MG",
            &super::KNIGHT_MOBILITY_MG,
        ),
        (
            "knight_mobility_eg",
            "const KNIGHT_MOBILITY_EG",
            &super::KNIGHT_MOBILITY_EG,
        ),
        (
            "bishop_mobility_mg",
            "const BISHOP_MOBILITY_MG",
            &super::BISHOP_MOBILITY_MG,
        ),
        (
            "bishop_mobility_eg",
            "const BISHOP_MOBILITY_EG",
            &super::BISHOP_MOBILITY_EG,
        ),
        (
            "rook_mobility_mg",
            "const ROOK_MOBILITY_MG",
            &super::ROOK_MOBILITY_MG,
        ),
        (
            "rook_mobility_eg",
            "const ROOK_MOBILITY_EG",
            &super::ROOK_MOBILITY_EG,
        ),
        (
            "queen_mobility_mg",
            "const QUEEN_MOBILITY_MG",
            &super::QUEEN_MOBILITY_MG,
        ),
        (
            "queen_mobility_eg",
            "const QUEEN_MOBILITY_EG",
            &super::QUEEN_MOBILITY_EG,
        ),
    ];
    for (name, site, table) in mobility_tables {
        for (index, default) in table.iter().enumerate() {
            push(format!("{name}[{index}]"), site, *default);
        }
    }
    push(
        "doubled_pawn_penalty".into(),
        "fn pawn_structure doubled literal 12",
        13,
    );
    push(
        "isolated_pawn_penalty".into(),
        "fn pawn_structure isolated literal 10",
        12,
    );
    push(
        "passed_pawn_base".into(),
        "fn pawn_structure passed base literal 10",
        8,
    );
    push("bishop_pair".into(), "fn pair_bonus literal 35", 39);
    push("rook_open_file".into(), "fn rook_files literal 18", 20);
    push("rook_semi_open_file".into(), "fn rook_files literal 10", 10);
    push("king_castled_mg".into(), "fn king_term literal 22", 28);
    push("king_shelter_mg".into(), "fn king_term literal 7", 9);
    for (index, default) in [-2, 10, 8, 12, 19].into_iter().enumerate() {
        push(
            format!("king_attack_weight[{index}]"),
            "const KING_ATTACK_WEIGHT (entry per PieceKind, king stays 0)",
            default,
        );
    }
    push("king_ring_attack".into(), "const KING_RING_ATTACK", 15);
    push("king_weak_square".into(), "const KING_WEAK_SQUARE", 9);
    push("king_flank_defense".into(), "const KING_FLANK_DEFENSE", 5);
    push(
        "king_pawnless_flank".into(),
        "const KING_PAWNLESS_FLANK",
        18,
    );
    push(
        "king_check_penalty".into(),
        "fn king_pressure_term in-check literal 35",
        35,
    );
    push("knight_outpost_mg".into(), "const KNIGHT_OUTPOST.0", 27);
    push("knight_outpost_eg".into(), "const KNIGHT_OUTPOST.1", 14);
    push("bishop_outpost_mg".into(), "const BISHOP_OUTPOST.0", 15);
    push("bishop_outpost_eg".into(), "const BISHOP_OUTPOST.1", 8);
    let reachable = "fn piece_activity reachable-outpost literal";
    push("knight_reachable_outpost_mg".into(), reachable, 13);
    push("knight_reachable_outpost_eg".into(), reachable, 6);
    push("bishop_reachable_outpost_mg".into(), reachable, 7);
    push("bishop_reachable_outpost_eg".into(), reachable, 3);
    let behind = "fn piece_activity minor-behind-pawn literal";
    push("minor_behind_pawn_mg".into(), behind, 8);
    push("minor_behind_pawn_eg".into(), behind, 5);
    push(
        "knight_king_distance_mg".into(),
        "fn piece_activity king-distance weight literal 3",
        2,
    );
    push(
        "bishop_king_distance_mg".into(),
        "fn piece_activity king-distance weight literal 2",
        2,
    );
    let diagonal = "fn piece_activity long-diagonal literal";
    push("bishop_long_diagonal_mg".into(), diagonal, 8);
    push("bishop_long_diagonal_eg".into(), diagonal, 4);
    push(
        "rook_queen_file_mg".into(),
        "fn piece_activity queen-file literal 6",
        6,
    );
    let seventh = "fn piece_activity seventh-rank literal";
    push("rook_seventh_mg".into(), seventh, 18);
    push("rook_seventh_eg".into(), seventh, 27);
    push(
        "trapped_rook_castle".into(),
        "fn piece_activity trapped-rook literal 10 (castling available)",
        10,
    );
    push(
        "trapped_rook_no_castle".into(),
        "fn piece_activity trapped-rook literal 22 (castling gone)",
        16,
    );
    let advanced = "fn piece_activity advanced-queen literal";
    push("queen_advanced_mg".into(), advanced, 7);
    push("queen_advanced_eg".into(), advanced, 10);
    push("hanging_threat_mg".into(), "const HANGING_THREAT.0", 21);
    push("hanging_threat_eg".into(), "const HANGING_THREAT.1", 25);
    push(
        "restricted_threat_mg".into(),
        "const RESTRICTED_THREAT_MG",
        4,
    );
    push("safe_pawn_threat_mg".into(), "const SAFE_PAWN_THREAT.0", 43);
    push("safe_pawn_threat_eg".into(), "const SAFE_PAWN_THREAT.1", 30);
    push("pawn_push_threat_mg".into(), "const PAWN_PUSH_THREAT.0", 13);
    push("pawn_push_threat_eg".into(), "const PAWN_PUSH_THREAT.1", 8);
    push(
        "queen_knight_threat_mg".into(),
        "const QUEEN_KNIGHT_THREAT.0",
        7,
    );
    push(
        "queen_knight_threat_eg".into(),
        "const QUEEN_KNIGHT_THREAT.1",
        7,
    );
    push(
        "queen_slider_threat_mg".into(),
        "const QUEEN_SLIDER_THREAT.0",
        15,
    );
    push(
        "queen_slider_threat_eg".into(),
        "const QUEEN_SLIDER_THREAT.1",
        4,
    );
    let minor_threat = "fn minor_threat_scores typed table";
    push("minor_threat_pawn_mg".into(), minor_threat, 6);
    push("minor_threat_pawn_eg".into(), minor_threat, 15);
    push("minor_threat_minor_mg".into(), minor_threat, 32);
    push("minor_threat_minor_eg".into(), minor_threat, 25);
    push("minor_threat_rook_mg".into(), minor_threat, 48);
    push("minor_threat_rook_eg".into(), minor_threat, 37);
    push("minor_threat_queen_mg".into(), minor_threat, 57);
    push("minor_threat_queen_eg".into(), minor_threat, 70);
    let rook_threat = "fn rook_threat_scores typed table";
    push("rook_threat_pawn_mg".into(), rook_threat, 4);
    push("rook_threat_pawn_eg".into(), rook_threat, 28);
    push("rook_threat_minor_mg".into(), rook_threat, 24);
    push("rook_threat_minor_eg".into(), rook_threat, 36);
    push("rook_threat_rook_mg".into(), rook_threat, 0);
    push("rook_threat_rook_eg".into(), rook_threat, 18);
    push("rook_threat_queen_mg".into(), rook_threat, 41);
    push("rook_threat_queen_eg".into(), rook_threat, 36);
    assert_eq!(list.len(), PARAM_COUNT, "descriptor layout size is frozen");
    list
}

/// Flat-vector slot indices matching [`ClassicalParams::to_flat`].
///
/// The coefficient extractor addresses parameters through these constants; a
/// unit test proves each named index points at the expected descriptor.
mod idx {
    /// First material slot (pawn); knight..queen follow contiguously.
    pub const MATERIAL: usize = 0;
    /// Pawn middlegame advance multiplier.
    pub const PAWN_ADVANCE_MG: usize = 5;
    /// Pawn middlegame centralization multiplier.
    pub const PAWN_CENTER_MG: usize = 6;
    /// Pawn endgame advance multiplier.
    pub const PAWN_ADVANCE_EG: usize = 7;
    /// Pawn endgame centralization multiplier.
    pub const PAWN_CENTER_EG: usize = 8;
    /// Knight middlegame centralization multiplier.
    pub const KNIGHT_CENTER_MG: usize = 9;
    /// Knight endgame centralization multiplier.
    pub const KNIGHT_CENTER_EG: usize = 10;
    /// Single-edge knight penalty.
    pub const KNIGHT_EDGE_PENALTY_MG: usize = 11;
    /// Corner knight penalty.
    pub const KNIGHT_CORNER_PENALTY_MG: usize = 12;
    /// Bishop middlegame centralization multiplier.
    pub const BISHOP_CENTER_MG: usize = 13;
    /// Bishop middlegame advance multiplier.
    pub const BISHOP_ADVANCE_MG: usize = 14;
    /// Bishop endgame centralization multiplier.
    pub const BISHOP_CENTER_EG: usize = 15;
    /// Rook middlegame advance multiplier.
    pub const ROOK_ADVANCE_MG: usize = 16;
    /// Rook endgame centralization multiplier.
    pub const ROOK_CENTER_EG: usize = 17;
    /// Rook endgame advance multiplier.
    pub const ROOK_ADVANCE_EG: usize = 18;
    /// Queen middlegame centralization multiplier.
    pub const QUEEN_CENTER_MG: usize = 19;
    /// Queen endgame centralization multiplier.
    pub const QUEEN_CENTER_EG: usize = 20;
    /// King middlegame centralization multiplier (applied negatively).
    pub const KING_CENTER_MG: usize = 21;
    /// King middlegame advance multiplier (applied negatively).
    pub const KING_ADVANCE_MG: usize = 22;
    /// King endgame centralization multiplier.
    pub const KING_CENTER_EG: usize = 23;
    /// King endgame advance multiplier.
    pub const KING_ADVANCE_EG: usize = 24;
    /// First knight middlegame mobility slot.
    pub const KNIGHT_MOBILITY_MG: usize = 25;
    /// First knight endgame mobility slot.
    pub const KNIGHT_MOBILITY_EG: usize = 34;
    /// First bishop middlegame mobility slot.
    pub const BISHOP_MOBILITY_MG: usize = 43;
    /// First bishop endgame mobility slot.
    pub const BISHOP_MOBILITY_EG: usize = 57;
    /// First rook middlegame mobility slot.
    pub const ROOK_MOBILITY_MG: usize = 71;
    /// First rook endgame mobility slot.
    pub const ROOK_MOBILITY_EG: usize = 86;
    /// First queen middlegame mobility slot.
    pub const QUEEN_MOBILITY_MG: usize = 101;
    /// First queen endgame mobility slot.
    pub const QUEEN_MOBILITY_EG: usize = 129;
    /// Doubled-pawn penalty.
    pub const DOUBLED_PAWN_PENALTY: usize = 157;
    /// Isolated-pawn penalty.
    pub const ISOLATED_PAWN_PENALTY: usize = 158;
    /// Passed-pawn base bonus.
    pub const PASSED_PAWN_BASE: usize = 159;
    /// Bishop-pair bonus.
    pub const BISHOP_PAIR: usize = 160;
    /// Rook on a fully open file.
    pub const ROOK_OPEN_FILE: usize = 161;
    /// Rook on a semi-open file.
    pub const ROOK_SEMI_OPEN_FILE: usize = 162;
    /// Castled-king placement bonus.
    pub const KING_CASTLED_MG: usize = 163;
    /// Per-file king shelter bonus.
    pub const KING_SHELTER_MG: usize = 164;
    /// First king-zone attacker weight slot (pawn).
    pub const KING_ATTACK_WEIGHT: usize = 165;
    /// Per-ring-hit king pressure.
    pub const KING_RING_ATTACK: usize = 170;
    /// Per-weak-zone-square king pressure.
    pub const KING_WEAK_SQUARE: usize = 171;
    /// Per-defended-flank-square relief.
    pub const KING_FLANK_DEFENSE: usize = 172;
    /// Pawnless-flank king pressure.
    pub const KING_PAWNLESS_FLANK: usize = 173;
    /// Flat in-check penalty.
    pub const KING_CHECK_PENALTY: usize = 174;
    /// Knight outpost middlegame bonus.
    pub const KNIGHT_OUTPOST_MG: usize = 175;
    /// Knight outpost endgame bonus.
    pub const KNIGHT_OUTPOST_EG: usize = 176;
    /// Bishop outpost middlegame bonus.
    pub const BISHOP_OUTPOST_MG: usize = 177;
    /// Bishop outpost endgame bonus.
    pub const BISHOP_OUTPOST_EG: usize = 178;
    /// Reachable knight outpost middlegame bonus.
    pub const KNIGHT_REACHABLE_OUTPOST_MG: usize = 179;
    /// Reachable knight outpost endgame bonus.
    pub const KNIGHT_REACHABLE_OUTPOST_EG: usize = 180;
    /// Reachable bishop outpost middlegame bonus.
    pub const BISHOP_REACHABLE_OUTPOST_MG: usize = 181;
    /// Reachable bishop outpost endgame bonus.
    pub const BISHOP_REACHABLE_OUTPOST_EG: usize = 182;
    /// Minor-behind-own-pawn middlegame bonus.
    pub const MINOR_BEHIND_PAWN_MG: usize = 183;
    /// Minor-behind-own-pawn endgame bonus.
    pub const MINOR_BEHIND_PAWN_EG: usize = 184;
    /// Knight distance-to-own-king weight.
    pub const KNIGHT_KING_DISTANCE_MG: usize = 185;
    /// Bishop distance-to-own-king weight.
    pub const BISHOP_KING_DISTANCE_MG: usize = 186;
    /// Bishop long-diagonal middlegame bonus.
    pub const BISHOP_LONG_DIAGONAL_MG: usize = 187;
    /// Bishop long-diagonal endgame bonus.
    pub const BISHOP_LONG_DIAGONAL_EG: usize = 188;
    /// Rook-on-queen-file middlegame bonus.
    pub const ROOK_QUEEN_FILE_MG: usize = 189;
    /// Rook-on-seventh middlegame bonus.
    pub const ROOK_SEVENTH_MG: usize = 190;
    /// Rook-on-seventh endgame bonus.
    pub const ROOK_SEVENTH_EG: usize = 191;
    /// Trapped-rook penalty with castling available.
    pub const TRAPPED_ROOK_CASTLE: usize = 192;
    /// Trapped-rook penalty with castling gone.
    pub const TRAPPED_ROOK_NO_CASTLE: usize = 193;
    /// Advanced-safe-queen middlegame bonus.
    pub const QUEEN_ADVANCED_MG: usize = 194;
    /// Advanced-safe-queen endgame bonus.
    pub const QUEEN_ADVANCED_EG: usize = 195;
    /// Hanging-piece middlegame threat.
    pub const HANGING_THREAT_MG: usize = 196;
    /// Hanging-piece endgame threat.
    pub const HANGING_THREAT_EG: usize = 197;
    /// Restricted-square middlegame threat.
    pub const RESTRICTED_THREAT_MG: usize = 198;
    /// Safe-pawn-attack middlegame threat.
    pub const SAFE_PAWN_THREAT_MG: usize = 199;
    /// Safe-pawn-attack endgame threat.
    pub const SAFE_PAWN_THREAT_EG: usize = 200;
    /// Pawn-push-attack middlegame threat.
    pub const PAWN_PUSH_THREAT_MG: usize = 201;
    /// Pawn-push-attack endgame threat.
    pub const PAWN_PUSH_THREAT_EG: usize = 202;
    /// Safe knight-on-queen middlegame threat.
    pub const QUEEN_KNIGHT_THREAT_MG: usize = 203;
    /// Safe knight-on-queen endgame threat.
    pub const QUEEN_KNIGHT_THREAT_EG: usize = 204;
    /// Doubly supported slider-on-queen middlegame threat.
    pub const QUEEN_SLIDER_THREAT_MG: usize = 205;
    /// Doubly supported slider-on-queen endgame threat.
    pub const QUEEN_SLIDER_THREAT_EG: usize = 206;
    /// First typed minor-threat slot (pawn target, middlegame).
    pub const MINOR_THREAT: usize = 207;
    /// First typed rook-threat slot (pawn target, middlegame).
    pub const ROOK_THREAT: usize = 215;
}

/// Used for looking up the parameterized middlegame and endgame safe-mobility
/// scores.
///
/// Mirrors `mobility_scores` with tables read from `params`; counts beyond a
/// table's range clamp to the final entry.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `kind` - piece kind selecting the table pair
/// * `mobility` - number of safe destination squares
///
/// # Returns
///
/// Middlegame and endgame mobility score pair.
fn mobility_scores_params(
    params: &ClassicalParams,
    kind: PieceKind,
    mobility: usize,
) -> (i32, i32) {
    let (middle, ending): (&[i32], &[i32]) = match kind {
        PieceKind::Knight => (&params.knight_mobility_mg, &params.knight_mobility_eg),
        PieceKind::Bishop => (&params.bishop_mobility_mg, &params.bishop_mobility_eg),
        PieceKind::Rook => (&params.rook_mobility_mg, &params.rook_mobility_eg),
        PieceKind::Queen => (&params.queen_mobility_mg, &params.queen_mobility_eg),
        PieceKind::Pawn | PieceKind::King => return (0, 0),
    };
    let index = mobility.min(middle.len() - 1);
    (middle[index], ending[index])
}

/// Used for interpolating the parameterized placement heuristic at `phase`.
///
/// Mirrors `tapered_piece_square` with every formula multiplier read from
/// `params`; the formula structure, signs, and truncating taper stay frozen.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `piece` - colored piece being placed
/// * `square` - square the piece occupies
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Phase-interpolated placement score for the piece's own color.
fn tapered_piece_square_params(
    params: &ClassicalParams,
    piece: Piece,
    square: Square,
    phase: i32,
) -> i32 {
    let file = i32::from(square.file());
    let rank = i32::from(square.rank());
    let relative = if piece.color == Color::White {
        rank
    } else {
        9 - rank
    };
    let center = 14 - ((2 * file - 7).abs() + (2 * (rank - 1) - 7).abs());
    let advance = relative - 1;
    let (middle, ending) = match piece.kind {
        PieceKind::Pawn => (
            advance * params.pawn_advance_mg + center * params.pawn_center_mg,
            advance * params.pawn_advance_eg + center * params.pawn_center_eg,
        ),
        PieceKind::Knight => (
            center * params.knight_center_mg - edge_penalty_params(params, file, rank),
            center * params.knight_center_eg,
        ),
        PieceKind::Bishop => (
            center * params.bishop_center_mg + advance * params.bishop_advance_mg,
            center * params.bishop_center_eg,
        ),
        PieceKind::Rook => (
            advance * params.rook_advance_mg,
            center * params.rook_center_eg + advance * params.rook_advance_eg,
        ),
        PieceKind::Queen => (
            center * params.queen_center_mg,
            center * params.queen_center_eg,
        ),
        PieceKind::King => (
            -center * params.king_center_mg - advance * params.king_advance_mg,
            center * params.king_center_eg + advance * params.king_advance_eg,
        ),
    };
    (middle * phase + ending * (MAX_PHASE - phase)) / MAX_PHASE
}

/// Used for the parameterized knight edge penalty.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `file` - zero-based file in `0..=7`
/// * `rank` - human rank in `1..=8`
///
/// # Returns
///
/// The corner penalty in a corner, the edge penalty on a single edge,
/// otherwise zero.
fn edge_penalty_params(params: &ClassicalParams, file: i32, rank: i32) -> i32 {
    let on_file_edge = file == 0 || file == 7;
    let on_rank_edge = rank == 1 || rank == 8;
    if on_file_edge && on_rank_edge {
        params.knight_corner_penalty_mg
    } else if on_file_edge || on_rank_edge {
        params.knight_edge_penalty_mg
    } else {
        0
    }
}

/// Used for the parameterized doubled/isolated/passed pawn term.
///
/// Mirrors `pawn_structure`; the quadratic passed-pawn interior
/// (`5 * advance^2`) and the blocked halving stay frozen.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position whose pawns are scored
/// * `file_pawns` - per-file pawn counts indexed by [`Color::index`]
///
/// # Returns
///
/// White-relative pawn-structure score.
fn pawn_structure_params(
    params: &ClassicalParams,
    position: &Position,
    file_pawns: [[u8; 8]; 2],
) -> i32 {
    let mut score = 0;
    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        for file in 0..8 {
            let count = file_pawns[color.index()][file];
            if count > 1 {
                score -= sign * params.doubled_pawn_penalty * i32::from(count - 1);
            }
            if count > 0 {
                let left_empty = file == 0 || file_pawns[color.index()][file - 1] == 0;
                let right_empty = file == 7 || file_pawns[color.index()][file + 1] == 0;
                if left_empty && right_empty {
                    score -= sign * params.isolated_pawn_penalty * i32::from(count);
                }
            }
        }

        let mut pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
        while pawns != 0 {
            let index = u8::try_from(pawns.trailing_zeros()).expect("set bit index is below 64");
            pawns &= pawns - 1;
            let square = board_square(index);
            if is_passed(position, square, color) {
                let relative_rank = if color == Color::White {
                    square.rank()
                } else {
                    9 - square.rank()
                };
                let advance = i32::from(relative_rank.saturating_sub(1));
                let bonus = params.passed_pawn_base + PASSED_PAWN_QUADRATIC * advance * advance;
                let push_row = if color == Color::White { -1 } else { 1 };
                let blocked = square_offset(square, 0, push_row)
                    .is_some_and(|front| position.piece_at(front).is_some());
                score += sign * if blocked { bonus / 2 } else { bonus };
            }
        }
    }
    score
}

/// Used for the parameterized rook open/semi-open file term.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position holding the rooks
/// * `file_pawns` - per-file pawn counts indexed by [`Color::index`]
///
/// # Returns
///
/// White-relative rook-file score.
fn rook_files_params(
    params: &ClassicalParams,
    position: &Position,
    file_pawns: [[u8; 8]; 2],
) -> i32 {
    let mut score = 0;
    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        let own = color.index();
        let enemy = color.opposite().index();
        let mut rooks = position.piece_bitboard(Piece::new(color, PieceKind::Rook));
        while rooks != 0 {
            let index = u8::try_from(rooks.trailing_zeros()).expect("set bit index is below 64");
            rooks &= rooks - 1;
            let file = usize::from(board_square(index).file());
            if file_pawns[own][file] == 0 {
                score += sign
                    * if file_pawns[enemy][file] == 0 {
                        params.rook_open_file
                    } else {
                        params.rook_semi_open_file
                    };
            }
        }
    }
    score
}

/// Used for the parameterized castled-placement and shelter term.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position holding the kings and pawns
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative king placement and shelter score.
fn king_term_params(params: &ClassicalParams, position: &Position, phase: i32) -> i32 {
    let mut score = 0;
    for color in [Color::White, Color::Black] {
        let Some(king) = position.king_square(color) else {
            continue;
        };
        let sign = color_sign(color);
        let castled = is_castled_king_square(color, king);
        let castled_bonus = if castled {
            params.king_castled_mg * phase / MAX_PHASE
        } else {
            0
        };
        let shelter =
            shelter_file_count(position, color, king) * params.king_shelter_mg * phase / MAX_PHASE;
        score += sign * (castled_bonus + shelter);
    }
    score
}

/// Used for the parameterized per-piece activity pair.
///
/// Mirrors `piece_activity` with every linear bonus read from `params`; the
/// bad-bishop product interior stays frozen, and the trapped-rook and
/// bad-bishop endgame halves stay structurally `penalty / 2`.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position holding the piece
/// * `color` - color owning the piece
/// * `kind` - kind of the scored piece
/// * `square` - square the piece occupies
/// * `mobility` - safe destination count already measured for the piece
/// * `attacks` - attack bitboard of the piece
/// * `maps` - attack maps built so far, supplying both pawn-attack unions
///
/// # Returns
///
/// Middlegame and endgame activity pair for the piece's own color.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn piece_activity_params(
    params: &ClassicalParams,
    position: &Position,
    color: Color,
    kind: PieceKind,
    square: Square,
    mobility: usize,
    attacks: u64,
    maps: &AttackMaps,
) -> (i32, i32) {
    let side = color.index();
    let enemy = color.opposite();
    let own = position.color_occupancy(color);
    let own_pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
    let enemy_pawns = position.piece_bitboard(Piece::new(enemy, PieceKind::Pawn));
    let own_pawn_attacks = maps.by_kind[side][PieceKind::Pawn.index()];
    let enemy_pawn_attacks = maps.by_kind[enemy.index()][PieceKind::Pawn.index()];
    let mut middle = 0;
    let mut ending = 0;

    if matches!(kind, PieceKind::Knight | PieceKind::Bishop) {
        if is_outpost(color, square, own_pawn_attacks, enemy_pawn_attacks) {
            let bonus = if kind == PieceKind::Knight {
                (params.knight_outpost_mg, params.knight_outpost_eg)
            } else {
                (params.bishop_outpost_mg, params.bishop_outpost_eg)
            };
            middle += bonus.0;
            ending += bonus.1;
        } else if attacks & outpost_mask(color, own_pawn_attacks, enemy_pawn_attacks) & !own != 0 {
            let bonus = if kind == PieceKind::Knight {
                (
                    params.knight_reachable_outpost_mg,
                    params.knight_reachable_outpost_eg,
                )
            } else {
                (
                    params.bishop_reachable_outpost_mg,
                    params.bishop_reachable_outpost_eg,
                )
            };
            middle += bonus.0;
            ending += bonus.1;
        }
        if super::minor_behind_pawn(position, color, square) {
            middle += params.minor_behind_pawn_mg;
            ending += params.minor_behind_pawn_eg;
        }
        if let Some(king) = position.king_square(color) {
            let distance = king_distance(king, square);
            let weight = if kind == PieceKind::Knight {
                params.knight_king_distance_mg
            } else {
                params.bishop_king_distance_mg
            };
            middle -= (distance - 1).max(0) * weight;
        }
    }

    match kind {
        PieceKind::Bishop => {
            let penalty = bad_bishop_penalty(square, mobility, own_pawns);
            middle -= penalty;
            ending -= penalty / 2;
            let pawn_occupancy = own_pawns | enemy_pawns;
            if bit_count(diagonal_attacks(square, pawn_occupancy) & center_squares()) >= 2 {
                middle += params.bishop_long_diagonal_mg;
                ending += params.bishop_long_diagonal_eg;
            }
        }
        PieceKind::Rook => {
            let queens = position.piece_bitboard(Piece::new(Color::White, PieceKind::Queen))
                | position.piece_bitboard(Piece::new(Color::Black, PieceKind::Queen));
            if queens & file_mask(square.file()) != 0 {
                middle += params.rook_queen_file_mg;
            }
            let seventh = if color == Color::White { 7 } else { 2 };
            if square.rank() == seventh && position.color_occupancy(enemy) & rank_mask(seventh) != 0
            {
                middle += params.rook_seventh_mg;
                ending += params.rook_seventh_eg;
            }
            if mobility <= 3 {
                if let Some(king) = position.king_square(color) {
                    if same_flank(square, king) {
                        let penalty = if can_castle_either(position, color) {
                            params.trapped_rook_castle
                        } else {
                            params.trapped_rook_no_castle
                        };
                        middle -= penalty;
                        ending -= penalty / 2;
                    }
                }
            }
        }
        PieceKind::Queen => {
            if relative_rank(color, square) >= 4 && square.bit() & enemy_pawn_attacks == 0 {
                middle += params.queen_advanced_mg;
                ending += params.queen_advanced_eg;
            }
        }
        PieceKind::Pawn | PieceKind::Knight | PieceKind::King => {}
    }
    (middle, ending)
}

/// Used for building parameterized attack maps in the release scan order.
///
/// Mirrors `AttackMaps::build` with the rich attack state always enabled and
/// the mobility, activity, and king-attacker weights read from `params`.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position whose attacks are collected
///
/// # Returns
///
/// Fully populated attack maps for both colors.
fn attack_maps_params(params: &ClassicalParams, position: &Position) -> AttackMaps {
    let mut maps = AttackMaps::default();

    for color in [Color::White, Color::Black] {
        maps.king_zone[color.index()] = position
            .king_square(color)
            .map_or(0, |king| king.bit() | king_ring(king));
    }

    for color in [Color::White, Color::Black] {
        scan_kind_params(params, &mut maps, position, color, PieceKind::Pawn, 0);
    }

    for color in [Color::White, Color::Black] {
        let enemy = color.opposite();
        let enemy_king = position.piece_bitboard(Piece::new(enemy, PieceKind::King));
        let enemy_pawn_attacks = maps.by_kind[enemy.index()][PieceKind::Pawn.index()];
        let mobility_area = !(position.color_occupancy(color) | enemy_king | enemy_pawn_attacks);
        for kind in [
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
            PieceKind::King,
        ] {
            scan_kind_params(params, &mut maps, position, color, kind, mobility_area);
        }
    }
    maps
}

/// Used for the parameterized single-kind attack scan.
///
/// Mirrors `AttackMaps::scan_kind` with the rich attack state always enabled.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `maps` - attack maps being accumulated
/// * `position` - position whose pieces are scanned
/// * `color` - color owning the scanned pieces
/// * `kind` - piece kind to scan
/// * `mobility_area` - squares that count toward safe mobility
fn scan_kind_params(
    params: &ClassicalParams,
    maps: &mut AttackMaps,
    position: &Position,
    color: Color,
    kind: PieceKind,
    mobility_area: u64,
) {
    let mut pieces = position.piece_bitboard(Piece::new(color, kind));
    while pieces != 0 {
        let index = u8::try_from(pieces.trailing_zeros()).expect("set bit index is below 64");
        pieces &= pieces - 1;
        let attacks = position.attacks_from(board_square(index));
        maps.add(color, kind, attacks, true);
        if !matches!(kind, PieceKind::Pawn | PieceKind::King) {
            let mobility = usize::try_from((attacks & mobility_area).count_ones())
                .expect("a bit count fits usize");
            let (middle, ending) = mobility_scores_params(params, kind, mobility);
            maps.mobility_mg[color.index()] += middle;
            maps.mobility_eg[color.index()] += ending;
            let (middle_activity, ending_activity) = piece_activity_params(
                params,
                position,
                color,
                kind,
                board_square(index),
                mobility,
                attacks,
                maps,
            );
            maps.activity_mg[color.index()] += middle_activity;
            maps.activity_eg[color.index()] += ending_activity;
        }
        if kind != PieceKind::King {
            let enemy = color.opposite();
            let king_hits = attacks & maps.king_zone[enemy.index()];
            if king_hits != 0 {
                maps.king_attackers[color.index()] += 1;
                maps.king_attacker_weight[color.index()] += params.king_attack_weight[kind.index()];
                maps.king_ring_hits[color.index()] += bit_count(king_hits);
            }
        }
    }
}

/// Used for the parameterized typed minor-threat pair.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `target` - kind of the threatened enemy piece
///
/// # Returns
///
/// Middlegame and endgame threat values; zero for a king target.
fn minor_threat_scores_params(params: &ClassicalParams, target: PieceKind) -> (i32, i32) {
    match target {
        PieceKind::Pawn => (params.minor_threat_pawn_mg, params.minor_threat_pawn_eg),
        PieceKind::Knight | PieceKind::Bishop => {
            (params.minor_threat_minor_mg, params.minor_threat_minor_eg)
        }
        PieceKind::Rook => (params.minor_threat_rook_mg, params.minor_threat_rook_eg),
        PieceKind::Queen => (params.minor_threat_queen_mg, params.minor_threat_queen_eg),
        PieceKind::King => (0, 0),
    }
}

/// Used for the parameterized typed rook-threat pair.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `target` - kind of the threatened enemy piece
///
/// # Returns
///
/// Middlegame and endgame threat values; zero for a king target.
fn rook_threat_scores_params(params: &ClassicalParams, target: PieceKind) -> (i32, i32) {
    match target {
        PieceKind::Pawn => (params.rook_threat_pawn_mg, params.rook_threat_pawn_eg),
        PieceKind::Knight | PieceKind::Bishop => {
            (params.rook_threat_minor_mg, params.rook_threat_minor_eg)
        }
        PieceKind::Rook => (params.rook_threat_rook_mg, params.rook_threat_rook_eg),
        PieceKind::Queen => (params.rook_threat_queen_mg, params.rook_threat_queen_eg),
        PieceKind::King => (0, 0),
    }
}

/// Used for the parameterized per-color threat pair.
///
/// Mirrors `side_threats` with every threat weight read from `params`.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position whose pieces are inspected
/// * `attacks` - rich attack maps for both colors
/// * `color` - attacking color earning the threats
///
/// # Returns
///
/// Middlegame and endgame threat totals for the attacker.
#[allow(clippy::too_many_lines)]
fn side_threats_params(
    params: &ClassicalParams,
    position: &Position,
    attacks: &AttackMaps,
    color: Color,
) -> (i32, i32) {
    let side = color.index();
    let enemy = color.opposite();
    let enemy_side = enemy.index();
    let enemies = position.color_occupancy(enemy);
    let enemy_pawns = position.piece_bitboard(Piece::new(enemy, PieceKind::Pawn));
    let non_pawn_enemies = enemies & !enemy_pawns;
    let strongly_protected = attacks.by_kind[enemy_side][PieceKind::Pawn.index()]
        | (attacks.attacked_twice[enemy_side] & !attacks.attacked_twice[side]);
    let weak = enemies & !strongly_protected & attacks.all[side];
    let mut middle = 0;
    let mut ending = 0;

    let mut minor_targets = (weak | (non_pawn_enemies & strongly_protected))
        & (attacks.by_kind[side][PieceKind::Knight.index()]
            | attacks.by_kind[side][PieceKind::Bishop.index()]);
    while minor_targets != 0 {
        let target = board_square(
            u8::try_from(minor_targets.trailing_zeros()).expect("set bit index is below 64"),
        );
        minor_targets &= minor_targets - 1;
        if let Some(piece) = position.piece_at(target) {
            let scores = minor_threat_scores_params(params, piece.kind);
            middle += scores.0;
            ending += scores.1;
        }
    }

    let mut rook_targets = weak & attacks.by_kind[side][PieceKind::Rook.index()];
    while rook_targets != 0 {
        let target = board_square(
            u8::try_from(rook_targets.trailing_zeros()).expect("set bit index is below 64"),
        );
        rook_targets &= rook_targets - 1;
        if let Some(piece) = position.piece_at(target) {
            let scores = rook_threat_scores_params(params, piece.kind);
            middle += scores.0;
            ending += scores.1;
        }
    }

    let hanging = weak & (!attacks.all[enemy_side] | attacks.attacked_twice[side]);
    let hanging_count = bit_count(hanging);
    middle += hanging_count * params.hanging_threat_mg;
    ending += hanging_count * params.hanging_threat_eg;

    let restricted = attacks.all[enemy_side] & !strongly_protected & attacks.all[side];
    middle += bit_count(restricted) * params.restricted_threat_mg;

    let safe = !attacks.all[enemy_side] | attacks.all[side];
    let safe_pawn_threats =
        attacks.by_kind[side][PieceKind::Pawn.index()] & non_pawn_enemies & safe;
    middle += bit_count(safe_pawn_threats) * params.safe_pawn_threat_mg;
    ending += bit_count(safe_pawn_threats) * params.safe_pawn_threat_eg;

    let own_pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
    let pushed_pawns = pawn_push_mask(color, own_pawns, !position.occupancy());
    let pawn_push_threats = pawn_attack_mask(color, pushed_pawns)
        & non_pawn_enemies
        & !attacks.by_kind[enemy_side][PieceKind::Pawn.index()]
        & safe;
    middle += bit_count(pawn_push_threats) * params.pawn_push_threat_mg;
    ending += bit_count(pawn_push_threats) * params.pawn_push_threat_eg;

    let enemy_queens = position.piece_bitboard(Piece::new(enemy, PieceKind::Queen));
    if enemy_queens != 0 {
        let queen = board_square(
            u8::try_from(enemy_queens.trailing_zeros()).expect("set bit index is below 64"),
        );
        let safe_knight_pressure =
            attacks.by_kind[side][PieceKind::Knight.index()] & knight_attacks(queen) & safe;
        middle += bit_count(safe_knight_pressure) * params.queen_knight_threat_mg;
        ending += bit_count(safe_knight_pressure) * params.queen_knight_threat_eg;

        let occupancy = position.occupancy();
        let slider_pressure = (attacks.by_kind[side][PieceKind::Bishop.index()]
            & diagonal_attacks(queen, occupancy)
            | attacks.by_kind[side][PieceKind::Rook.index()]
                & orthogonal_attacks(queen, occupancy))
            & attacks.attacked_twice[side]
            & safe;
        middle += bit_count(slider_pressure) * params.queen_slider_threat_mg;
        ending += bit_count(slider_pressure) * params.queen_slider_threat_eg;
    }
    (middle, ending)
}

/// Used for the parameterized white-relative tapered threat term.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position whose threats are scored
/// * `attacks` - rich attack maps for both colors
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative tapered threat score.
fn rich_threat_term_params(
    params: &ClassicalParams,
    position: &Position,
    attacks: &AttackMaps,
    phase: i32,
) -> i32 {
    let (white_middle, white_ending) = side_threats_params(params, position, attacks, Color::White);
    let (black_middle, black_ending) = side_threats_params(params, position, attacks, Color::Black);
    tapered_score(
        white_middle - black_middle,
        white_ending - black_ending,
        phase,
    )
}

/// Used for the parameterized unphased king-zone pressure against one king.
///
/// Mirrors `side_king_pressure`; the `attackers^2` and `flank_attack^2`
/// quadratics, the positive mobility edge, and the single-attacker halving
/// stay structurally frozen.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position supplying the pawn layout
/// * `attacks` - rich attack maps for both colors
/// * `king_color` - color of the defending king
/// * `king` - defending king's square
///
/// # Returns
///
/// Raw pressure total before phase scaling and queen discounts.
fn side_king_pressure_params(
    params: &ClassicalParams,
    position: &Position,
    attacks: &AttackMaps,
    king_color: Color,
    king: Square,
) -> i32 {
    let defender = king_color.index();
    let attacker = king_color.opposite().index();
    let attackers = attacks.king_attackers[attacker];
    if attackers == 0 {
        return 0;
    }
    let weak = attacks.all[attacker]
        & !attacks.attacked_twice[defender]
        & (!attacks.all[defender]
            | attacks.by_kind[defender][PieceKind::King.index()]
            | attacks.by_kind[defender][PieceKind::Queen.index()]);
    let flank = king_flank_mask(king) & camp_mask(king_color);
    let flank_attack = bit_count(attacks.all[attacker] & flank)
        + bit_count(attacks.attacked_twice[attacker] & flank);
    let flank_defense = bit_count(attacks.all[defender] & flank);
    let mut pressure = attacks.king_attacker_weight[attacker]
        + attackers * attackers * KING_ATTACKERS_SQUARED
        + params.king_ring_attack * attacks.king_ring_hits[attacker]
        + params.king_weak_square * bit_count(attacks.king_zone[defender] & weak)
        + flank_attack * flank_attack / 2
        + (attacks.mobility_mg[attacker] - attacks.mobility_mg[defender]).max(0) / 2
        - flank_defense * params.king_flank_defense;
    let pawns = position.piece_bitboard(Piece::new(Color::White, PieceKind::Pawn))
        | position.piece_bitboard(Piece::new(Color::Black, PieceKind::Pawn));
    if pawns & flank == 0 {
        pressure += params.king_pawnless_flank;
    }
    if attackers == 1 {
        pressure / 2
    } else {
        pressure
    }
}

/// Used for the parameterized white-relative king-pressure term.
///
/// # Arguments
///
/// * `params` - runtime parameter set
/// * `position` - position holding the kings
/// * `attacks` - rich attack maps for both colors
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative king-pressure score.
fn king_pressure_term_params(
    params: &ClassicalParams,
    position: &Position,
    attacks: &AttackMaps,
    phase: i32,
) -> i32 {
    let mut score = 0;
    for king_color in [Color::White, Color::Black] {
        let Some(king) = position.king_square(king_color) else {
            continue;
        };
        let mut pressure = side_king_pressure_params(params, position, attacks, king_color, king);
        let enemy = king_color.opposite();
        if position.piece_bitboard(Piece::new(enemy, PieceKind::Queen)) == 0 {
            pressure = pressure * 55 / 100;
        }
        pressure = pressure * phase / MAX_PHASE;
        if position.in_check(king_color) {
            pressure += params.king_check_penalty;
        }
        score -= color_sign(king_color) * pressure;
    }
    score
}

/// Used for evaluating the release rich attack-state flavor with runtime
/// parameters.
///
/// At [`ClassicalParams::default`] this reproduces
/// [`Classical::breakdown`](super::Classical::breakdown) bit-for-bit (unit
/// and corpus tested). Non-tuned scalars keep their frozen released values.
///
/// # Arguments
///
/// * `position` - position to evaluate
/// * `params` - runtime parameter set
///
/// # Panics
///
/// Panics only on internal bit-index conversions that cannot fail for a
/// valid position.
///
/// # Returns
///
/// White-relative decomposition of every classical term.
#[must_use]
pub fn params_breakdown(position: &Position, params: &ClassicalParams) -> ClassicalBreakdown {
    let phase = game_phase(position);
    let mut result = ClassicalBreakdown::default();
    let mut file_pawns = [[0_u8; 8]; 2];
    let mut bishops = [0_u8; 2];

    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        for kind in [
            PieceKind::Pawn,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
            PieceKind::King,
        ] {
            let piece = Piece::new(color, kind);
            let mut pieces = position.piece_bitboard(piece);
            let value = if kind == PieceKind::King {
                0
            } else {
                params.material[kind.index()]
            };
            result.material +=
                sign * value * i32::try_from(pieces.count_ones()).expect("piece count fits i32");
            if kind == PieceKind::Bishop {
                bishops[color.index()] =
                    u8::try_from(pieces.count_ones()).expect("bishop count fits u8");
            }
            while pieces != 0 {
                let index =
                    u8::try_from(pieces.trailing_zeros()).expect("set bit index is below 64");
                pieces &= pieces - 1;
                let square = board_square(index);
                result.piece_square +=
                    sign * tapered_piece_square_params(params, piece, square, phase);
                if kind == PieceKind::Pawn {
                    file_pawns[color.index()][usize::from(square.file())] += 1;
                }
            }
        }
    }

    result.pawns = pawn_structure_params(params, position, file_pawns);
    result.bishops = {
        let white = i32::from(bishops[Color::White.index()] >= 2) * params.bishop_pair;
        let black = i32::from(bishops[Color::Black.index()] >= 2) * params.bishop_pair;
        white - black
    };
    result.rooks = rook_files_params(params, position, file_pawns);
    result.kings = king_term_params(params, position, phase);
    let attacks = attack_maps_params(params, position);
    result.mobility = mobility_term(&attacks, phase);
    result.activity = activity_term(&attacks, phase);
    result.threats = rich_threat_term_params(params, position, &attacks, phase);
    result.king_pressure = king_pressure_term_params(params, position, &attacks, phase)
        .saturating_add(super::king_danger_promoted_delta(position));
    let (connected_pawns, king_pawn_file) = super::promoted_capacity_delta(position);
    result.pawns = result.pawns.saturating_add(connected_pawns);
    result.kings = result.kings.saturating_add(king_pawn_file);
    result.tempo = if position.side_to_move() == Color::White {
        TEMPO
    } else {
        -TEMPO
    };
    result
}

/// One position's sparse linear decomposition of the white-relative score.
///
/// For flat parameters `theta`, the modelled score is
/// `(mg_sum * phase + eg_sum * (MAX_PHASE - phase)) / MAX_PHASE` in exact
/// real arithmetic, where `mg_sum = constant_mg + sum(mg_coeff * theta)` and
/// likewise for the endgame. Frozen non-linear terms (including the tempo)
/// live in the constants. The difference to the released integer score at
/// the default parameters is the bounded integer-rounding residue (see
/// [`RESIDUAL_BOUND`]); king-pressure additionally freezes its positive
/// mobility-edge input at the default mobility tables.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearModel {
    /// Used for the tapering phase of the position, in `0..=MAX_PHASE`.
    pub phase: i32,
    /// Used for the sparse `(flat index, mg coefficient, eg coefficient)`
    /// entries, ordered by ascending index.
    pub coefficients: Vec<(u16, f64, f64)>,
    /// Used for the frozen middlegame constant contribution.
    pub constant_mg: f64,
    /// Used for the frozen endgame constant contribution.
    pub constant_eg: f64,
}

impl LinearModel {
    /// Used for predicting the smooth white-relative score at `flat`.
    ///
    /// # Arguments
    ///
    /// * `flat` - [`PARAM_COUNT`] parameter values in flat-layout order
    ///
    /// # Returns
    ///
    /// Modelled score in centipawns from White's perspective.
    #[must_use]
    pub fn predict(&self, flat: &[f64]) -> f64 {
        let mut middle = self.constant_mg;
        let mut ending = self.constant_eg;
        for &(index, mg_coeff, eg_coeff) in &self.coefficients {
            let value = flat[usize::from(index)];
            middle += mg_coeff * value;
            ending += eg_coeff * value;
        }
        let phase = f64::from(self.phase);
        let max_phase = f64::from(MAX_PHASE);
        (middle * phase + ending * (max_phase - phase)) / max_phase
    }
}

/// Dense coefficient accumulator used while walking one position.
struct CoefficientSink {
    /// Used for the dense middlegame coefficients per flat slot.
    middle: Vec<f64>,
    /// Used for the dense endgame coefficients per flat slot.
    ending: Vec<f64>,
    /// Used for the frozen middlegame constant contribution.
    constant_mg: f64,
    /// Used for the frozen endgame constant contribution.
    constant_eg: f64,
}

impl CoefficientSink {
    /// Used for creating an empty accumulator.
    ///
    /// # Returns
    ///
    /// A sink with all coefficients and constants at zero.
    fn new() -> Self {
        Self {
            middle: vec![0.0; PARAM_COUNT],
            ending: vec![0.0; PARAM_COUNT],
            constant_mg: 0.0,
            constant_eg: 0.0,
        }
    }

    /// Used for adding a tapered coefficient increment to one slot.
    ///
    /// # Arguments
    ///
    /// * `index` - flat slot index
    /// * `middle` - middlegame increment
    /// * `ending` - endgame increment
    fn add(&mut self, index: usize, middle: f64, ending: f64) {
        self.middle[index] += middle;
        self.ending[index] += ending;
    }

    /// Used for adding a phase-independent coefficient increment to one slot.
    ///
    /// # Arguments
    ///
    /// * `index` - flat slot index
    /// * `amount` - flat increment applied to both phases
    fn add_flat(&mut self, index: usize, amount: f64) {
        self.add(index, amount, amount);
    }

    /// Used for adding a frozen (non-tuned) contribution.
    ///
    /// # Arguments
    ///
    /// * `middle` - middlegame increment
    /// * `ending` - endgame increment
    fn add_constant(&mut self, middle: f64, ending: f64) {
        self.constant_mg += middle;
        self.constant_eg += ending;
    }

    /// Used for compressing the dense accumulator into a sparse model.
    ///
    /// # Arguments
    ///
    /// * `phase` - tapering phase of the walked position
    ///
    /// # Panics
    ///
    /// Panics if a slot index does not fit `u16`, which cannot happen for
    /// [`PARAM_COUNT`] slots.
    ///
    /// # Returns
    ///
    /// The sparse [`LinearModel`] with zero entries dropped.
    fn into_model(self, phase: i32) -> LinearModel {
        let mut coefficients = Vec::new();
        for (index, (&middle, &ending)) in self.middle.iter().zip(self.ending.iter()).enumerate() {
            if middle != 0.0 || ending != 0.0 {
                coefficients.push((
                    u16::try_from(index).expect("parameter index fits u16"),
                    middle,
                    ending,
                ));
            }
        }
        LinearModel {
            phase,
            coefficients,
            constant_mg: self.constant_mg,
            constant_eg: self.constant_eg,
        }
    }
}

/// Used for extracting one position's sparse linear coefficients.
///
/// Walks the release evaluation exactly once, crediting every linear
/// parameter with its feature count and folding every frozen term into the
/// constants. Non-dyadic structural factors (the blocked-passer and
/// trapped-rook halves, the queenless `55 / 100` discount, and the
/// single-attacker halving) are applied as exact real factors, so the model
/// differs from the released integer score only by bounded rounding residue.
///
/// # Arguments
///
/// * `position` - position to decompose
///
/// # Panics
///
/// Panics only on internal bit-index conversions that cannot fail for a
/// valid position.
///
/// # Returns
///
/// The position's [`LinearModel`].
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn linear_model(position: &Position) -> LinearModel {
    let phase = game_phase(position);
    let mut sink = CoefficientSink::new();
    let mut file_pawns = [[0_u8; 8]; 2];
    let mut bishops = [0_u8; 2];

    // Material and placement-formula coefficients from one board scan.
    for color in [Color::White, Color::Black] {
        let sign = f64::from(color_sign(color));
        for kind in [
            PieceKind::Pawn,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
            PieceKind::King,
        ] {
            let piece = Piece::new(color, kind);
            let mut pieces = position.piece_bitboard(piece);
            if kind == PieceKind::Bishop {
                bishops[color.index()] =
                    u8::try_from(pieces.count_ones()).expect("bishop count fits u8");
            }
            while pieces != 0 {
                let index =
                    u8::try_from(pieces.trailing_zeros()).expect("set bit index is below 64");
                pieces &= pieces - 1;
                let square = board_square(index);
                if kind != PieceKind::King {
                    sink.add_flat(idx::MATERIAL + kind.index(), sign);
                }
                if kind == PieceKind::Pawn {
                    file_pawns[color.index()][usize::from(square.file())] += 1;
                }
                placement_coefficients(&mut sink, piece, square, sign);
            }
        }
    }

    pawn_coefficients(&mut sink, position, file_pawns);

    // Bishop pair and rook files.
    for color in [Color::White, Color::Black] {
        let sign = f64::from(color_sign(color));
        if bishops[color.index()] >= 2 {
            sink.add_flat(idx::BISHOP_PAIR, sign);
        }
        let own = color.index();
        let enemy = color.opposite().index();
        let mut rooks = position.piece_bitboard(Piece::new(color, PieceKind::Rook));
        while rooks != 0 {
            let index = u8::try_from(rooks.trailing_zeros()).expect("set bit index is below 64");
            rooks &= rooks - 1;
            let file = usize::from(board_square(index).file());
            if file_pawns[own][file] == 0 {
                if file_pawns[enemy][file] == 0 {
                    sink.add_flat(idx::ROOK_OPEN_FILE, sign);
                } else {
                    sink.add_flat(idx::ROOK_SEMI_OPEN_FILE, sign);
                }
            }
        }
    }

    // King placement and shelter: middlegame-only linear scalars.
    for color in [Color::White, Color::Black] {
        let Some(king) = position.king_square(color) else {
            continue;
        };
        let sign = f64::from(color_sign(color));
        if is_castled_king_square(color, king) {
            sink.add(idx::KING_CASTLED_MG, sign, 0.0);
        }
        let shelter = shelter_file_count(position, color, king);
        sink.add(idx::KING_SHELTER_MG, sign * f64::from(shelter), 0.0);
    }

    let maps = AttackMaps::build(position, true);
    mobility_and_activity_coefficients(&mut sink, position, &maps);
    for color in [Color::White, Color::Black] {
        threat_coefficients(&mut sink, position, &maps, color);
    }
    king_pressure_coefficients(&mut sink, position, &maps);

    // The promoted STR-280 king-danger correction is non-linear in the king
    // parameters, so it enters as an already-tapered frozen constant. Adding
    // the same centipawn amount to both phases reproduces it exactly at every
    // phase without pretending it scales with a tuned weight.
    let promoted_king_danger = f64::from(super::king_danger_promoted_delta(position));
    sink.add_constant(promoted_king_danger, promoted_king_danger);

    // STR-319's connected-pawn ramp and STR-304's king-to-pawn-file distance
    // are built from named constants rather than tunable parameters, so they
    // enter as already-tapered frozen constants for the same reason.
    let (connected_pawns, king_pawn_file) = super::promoted_capacity_delta(position);
    let promoted_capacity = f64::from(connected_pawns) + f64::from(king_pawn_file);
    sink.add_constant(promoted_capacity, promoted_capacity);

    // Tempo stays frozen: a flat, phase-independent constant.
    let tempo = if position.side_to_move() == Color::White {
        f64::from(TEMPO)
    } else {
        -f64::from(TEMPO)
    };
    sink.add_constant(tempo, tempo);

    sink.into_model(phase)
}

/// Used for crediting one piece's placement-formula coefficients.
///
/// # Arguments
///
/// * `sink` - coefficient accumulator
/// * `piece` - colored piece being placed
/// * `square` - square the piece occupies
/// * `sign` - `1.0` for White, `-1.0` for Black
fn placement_coefficients(sink: &mut CoefficientSink, piece: Piece, square: Square, sign: f64) {
    let file = i32::from(square.file());
    let rank = i32::from(square.rank());
    let relative = if piece.color == Color::White {
        rank
    } else {
        9 - rank
    };
    let center = f64::from(14 - ((2 * file - 7).abs() + (2 * (rank - 1) - 7).abs()));
    let advance = f64::from(relative - 1);
    match piece.kind {
        PieceKind::Pawn => {
            sink.add(idx::PAWN_ADVANCE_MG, sign * advance, 0.0);
            sink.add(idx::PAWN_CENTER_MG, sign * center, 0.0);
            sink.add(idx::PAWN_ADVANCE_EG, 0.0, sign * advance);
            sink.add(idx::PAWN_CENTER_EG, 0.0, sign * center);
        }
        PieceKind::Knight => {
            sink.add(idx::KNIGHT_CENTER_MG, sign * center, 0.0);
            sink.add(idx::KNIGHT_CENTER_EG, 0.0, sign * center);
            let on_file_edge = file == 0 || file == 7;
            let on_rank_edge = rank == 1 || rank == 8;
            if on_file_edge && on_rank_edge {
                sink.add(idx::KNIGHT_CORNER_PENALTY_MG, -sign, 0.0);
            } else if on_file_edge || on_rank_edge {
                sink.add(idx::KNIGHT_EDGE_PENALTY_MG, -sign, 0.0);
            }
        }
        PieceKind::Bishop => {
            sink.add(idx::BISHOP_CENTER_MG, sign * center, 0.0);
            sink.add(idx::BISHOP_ADVANCE_MG, sign * advance, 0.0);
            sink.add(idx::BISHOP_CENTER_EG, 0.0, sign * center);
        }
        PieceKind::Rook => {
            sink.add(idx::ROOK_ADVANCE_MG, sign * advance, 0.0);
            sink.add(idx::ROOK_CENTER_EG, 0.0, sign * center);
            sink.add(idx::ROOK_ADVANCE_EG, 0.0, sign * advance);
        }
        PieceKind::Queen => {
            sink.add(idx::QUEEN_CENTER_MG, sign * center, 0.0);
            sink.add(idx::QUEEN_CENTER_EG, 0.0, sign * center);
        }
        PieceKind::King => {
            sink.add(idx::KING_CENTER_MG, -sign * center, 0.0);
            sink.add(idx::KING_ADVANCE_MG, -sign * advance, 0.0);
            sink.add(idx::KING_CENTER_EG, 0.0, sign * center);
            sink.add(idx::KING_ADVANCE_EG, 0.0, sign * advance);
        }
    }
}

/// Used for crediting the pawn-structure coefficients.
///
/// The quadratic passed-pawn interior stays frozen: its (possibly halved)
/// value goes into the constants while the base bonus is credited with the
/// exact blocked half-factor.
///
/// # Arguments
///
/// * `sink` - coefficient accumulator
/// * `position` - position whose pawns are scored
/// * `file_pawns` - per-file pawn counts indexed by [`Color::index`]
fn pawn_coefficients(sink: &mut CoefficientSink, position: &Position, file_pawns: [[u8; 8]; 2]) {
    for color in [Color::White, Color::Black] {
        let sign = f64::from(color_sign(color));
        for file in 0..8 {
            let count = file_pawns[color.index()][file];
            if count > 1 {
                sink.add_flat(idx::DOUBLED_PAWN_PENALTY, -sign * f64::from(count - 1));
            }
            if count > 0 {
                let left_empty = file == 0 || file_pawns[color.index()][file - 1] == 0;
                let right_empty = file == 7 || file_pawns[color.index()][file + 1] == 0;
                if left_empty && right_empty {
                    sink.add_flat(idx::ISOLATED_PAWN_PENALTY, -sign * f64::from(count));
                }
            }
        }

        let mut pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
        while pawns != 0 {
            let index = u8::try_from(pawns.trailing_zeros()).expect("set bit index is below 64");
            pawns &= pawns - 1;
            let square = board_square(index);
            if is_passed(position, square, color) {
                let relative_rank = if color == Color::White {
                    square.rank()
                } else {
                    9 - square.rank()
                };
                let advance = f64::from(relative_rank.saturating_sub(1));
                let push_row = if color == Color::White { -1 } else { 1 };
                let blocked = square_offset(square, 0, push_row)
                    .is_some_and(|front| position.piece_at(front).is_some());
                let factor = if blocked { 0.5 } else { 1.0 };
                sink.add_flat(idx::PASSED_PAWN_BASE, sign * factor);
                let quadratic =
                    sign * factor * f64::from(PASSED_PAWN_QUADRATIC) * advance * advance;
                sink.add_constant(quadratic, quadratic);
            }
        }
    }
}

/// Used for crediting mobility-table and piece-activity coefficients.
///
/// Re-walks every non-pawn, non-king piece in the release scan order,
/// re-deriving each piece's safe-mobility count from the finished attack
/// maps (both pawn unions are complete before any non-pawn scan, matching
/// the release build order).
///
/// # Arguments
///
/// * `sink` - coefficient accumulator
/// * `position` - position to decompose
/// * `maps` - finished rich attack maps of the released evaluation
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn mobility_and_activity_coefficients(
    sink: &mut CoefficientSink,
    position: &Position,
    maps: &AttackMaps,
) {
    for color in [Color::White, Color::Black] {
        let sign = f64::from(color_sign(color));
        let enemy = color.opposite();
        let enemy_king = position.piece_bitboard(Piece::new(enemy, PieceKind::King));
        let enemy_pawn_attacks = maps.by_kind[enemy.index()][PieceKind::Pawn.index()];
        let mobility_area = !(position.color_occupancy(color) | enemy_king | enemy_pawn_attacks);
        for kind in [
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
        ] {
            let (base_mg, base_eg, length) = match kind {
                PieceKind::Knight => (idx::KNIGHT_MOBILITY_MG, idx::KNIGHT_MOBILITY_EG, 9),
                PieceKind::Bishop => (idx::BISHOP_MOBILITY_MG, idx::BISHOP_MOBILITY_EG, 14),
                PieceKind::Rook => (idx::ROOK_MOBILITY_MG, idx::ROOK_MOBILITY_EG, 15),
                PieceKind::Queen => (idx::QUEEN_MOBILITY_MG, idx::QUEEN_MOBILITY_EG, 28),
                PieceKind::Pawn | PieceKind::King => unreachable!("scan skips pawns and kings"),
            };
            let mut pieces = position.piece_bitboard(Piece::new(color, kind));
            while pieces != 0 {
                let index =
                    u8::try_from(pieces.trailing_zeros()).expect("set bit index is below 64");
                pieces &= pieces - 1;
                let square = board_square(index);
                let attacks = position.attacks_from(square);
                let mobility = usize::try_from((attacks & mobility_area).count_ones())
                    .expect("a bit count fits usize");
                let bin = mobility.min(length - 1);
                sink.add(base_mg + bin, sign, 0.0);
                sink.add(base_eg + bin, 0.0, sign);
                activity_coefficients(sink, position, maps, color, kind, square, mobility, attacks);
            }
        }
    }
}

/// Used for crediting one piece's activity coefficients.
///
/// Mirrors `piece_activity`; the bad-bishop product stays a frozen constant
/// and the trapped-rook endgame half uses the exact `0.5` factor.
///
/// # Arguments
///
/// * `sink` - coefficient accumulator
/// * `position` - position holding the piece
/// * `maps` - finished rich attack maps
/// * `color` - color owning the piece
/// * `kind` - kind of the scored piece
/// * `square` - square the piece occupies
/// * `mobility` - safe destination count of the piece
/// * `attacks` - attack bitboard of the piece
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn activity_coefficients(
    sink: &mut CoefficientSink,
    position: &Position,
    maps: &AttackMaps,
    color: Color,
    kind: PieceKind,
    square: Square,
    mobility: usize,
    attacks: u64,
) {
    let sign = f64::from(color_sign(color));
    let side = color.index();
    let enemy = color.opposite();
    let own = position.color_occupancy(color);
    let own_pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
    let enemy_pawns = position.piece_bitboard(Piece::new(enemy, PieceKind::Pawn));
    let own_pawn_attacks = maps.by_kind[side][PieceKind::Pawn.index()];
    let enemy_pawn_attacks = maps.by_kind[enemy.index()][PieceKind::Pawn.index()];

    if matches!(kind, PieceKind::Knight | PieceKind::Bishop) {
        if is_outpost(color, square, own_pawn_attacks, enemy_pawn_attacks) {
            if kind == PieceKind::Knight {
                sink.add(idx::KNIGHT_OUTPOST_MG, sign, 0.0);
                sink.add(idx::KNIGHT_OUTPOST_EG, 0.0, sign);
            } else {
                sink.add(idx::BISHOP_OUTPOST_MG, sign, 0.0);
                sink.add(idx::BISHOP_OUTPOST_EG, 0.0, sign);
            }
        } else if attacks & outpost_mask(color, own_pawn_attacks, enemy_pawn_attacks) & !own != 0 {
            if kind == PieceKind::Knight {
                sink.add(idx::KNIGHT_REACHABLE_OUTPOST_MG, sign, 0.0);
                sink.add(idx::KNIGHT_REACHABLE_OUTPOST_EG, 0.0, sign);
            } else {
                sink.add(idx::BISHOP_REACHABLE_OUTPOST_MG, sign, 0.0);
                sink.add(idx::BISHOP_REACHABLE_OUTPOST_EG, 0.0, sign);
            }
        }
        if super::minor_behind_pawn(position, color, square) {
            sink.add(idx::MINOR_BEHIND_PAWN_MG, sign, 0.0);
            sink.add(idx::MINOR_BEHIND_PAWN_EG, 0.0, sign);
        }
        if let Some(king) = position.king_square(color) {
            let distance = f64::from((king_distance(king, square) - 1).max(0));
            let index = if kind == PieceKind::Knight {
                idx::KNIGHT_KING_DISTANCE_MG
            } else {
                idx::BISHOP_KING_DISTANCE_MG
            };
            sink.add(index, -sign * distance, 0.0);
        }
    }

    match kind {
        PieceKind::Bishop => {
            let penalty = f64::from(bad_bishop_penalty(square, mobility, own_pawns));
            sink.add_constant(-sign * penalty, -sign * penalty * 0.5);
            let pawn_occupancy = own_pawns | enemy_pawns;
            if bit_count(diagonal_attacks(square, pawn_occupancy) & center_squares()) >= 2 {
                sink.add(idx::BISHOP_LONG_DIAGONAL_MG, sign, 0.0);
                sink.add(idx::BISHOP_LONG_DIAGONAL_EG, 0.0, sign);
            }
        }
        PieceKind::Rook => {
            let queens = position.piece_bitboard(Piece::new(Color::White, PieceKind::Queen))
                | position.piece_bitboard(Piece::new(Color::Black, PieceKind::Queen));
            if queens & file_mask(square.file()) != 0 {
                sink.add(idx::ROOK_QUEEN_FILE_MG, sign, 0.0);
            }
            let seventh = if color == Color::White { 7 } else { 2 };
            if square.rank() == seventh && position.color_occupancy(enemy) & rank_mask(seventh) != 0
            {
                sink.add(idx::ROOK_SEVENTH_MG, sign, 0.0);
                sink.add(idx::ROOK_SEVENTH_EG, 0.0, sign);
            }
            if mobility <= 3 {
                if let Some(king) = position.king_square(color) {
                    if same_flank(square, king) {
                        let index = if can_castle_either(position, color) {
                            idx::TRAPPED_ROOK_CASTLE
                        } else {
                            idx::TRAPPED_ROOK_NO_CASTLE
                        };
                        sink.add(index, -sign, -sign * 0.5);
                    }
                }
            }
        }
        PieceKind::Queen => {
            if relative_rank(color, square) >= 4 && square.bit() & enemy_pawn_attacks == 0 {
                sink.add(idx::QUEEN_ADVANCED_MG, sign, 0.0);
                sink.add(idx::QUEEN_ADVANCED_EG, 0.0, sign);
            }
        }
        PieceKind::Pawn | PieceKind::Knight | PieceKind::King => {}
    }
}

/// Used for crediting one attacking color's threat coefficients.
///
/// Mirrors `side_threats` with counts instead of scores.
///
/// # Arguments
///
/// * `sink` - coefficient accumulator
/// * `position` - position whose pieces are inspected
/// * `attacks` - finished rich attack maps
/// * `color` - attacking color earning the threats
#[allow(clippy::too_many_lines)]
fn threat_coefficients(
    sink: &mut CoefficientSink,
    position: &Position,
    attacks: &AttackMaps,
    color: Color,
) {
    let sign = f64::from(color_sign(color));
    let side = color.index();
    let enemy = color.opposite();
    let enemy_side = enemy.index();
    let enemies = position.color_occupancy(enemy);
    let enemy_pawns = position.piece_bitboard(Piece::new(enemy, PieceKind::Pawn));
    let non_pawn_enemies = enemies & !enemy_pawns;
    let strongly_protected = attacks.by_kind[enemy_side][PieceKind::Pawn.index()]
        | (attacks.attacked_twice[enemy_side] & !attacks.attacked_twice[side]);
    let weak = enemies & !strongly_protected & attacks.all[side];

    let mut minor_targets = (weak | (non_pawn_enemies & strongly_protected))
        & (attacks.by_kind[side][PieceKind::Knight.index()]
            | attacks.by_kind[side][PieceKind::Bishop.index()]);
    while minor_targets != 0 {
        let target = board_square(
            u8::try_from(minor_targets.trailing_zeros()).expect("set bit index is below 64"),
        );
        minor_targets &= minor_targets - 1;
        if let Some(piece) = position.piece_at(target) {
            if let Some(offset) = typed_threat_offset(piece.kind) {
                sink.add(idx::MINOR_THREAT + offset, sign, 0.0);
                sink.add(idx::MINOR_THREAT + offset + 1, 0.0, sign);
            }
        }
    }

    let mut rook_targets = weak & attacks.by_kind[side][PieceKind::Rook.index()];
    while rook_targets != 0 {
        let target = board_square(
            u8::try_from(rook_targets.trailing_zeros()).expect("set bit index is below 64"),
        );
        rook_targets &= rook_targets - 1;
        if let Some(piece) = position.piece_at(target) {
            if let Some(offset) = typed_threat_offset(piece.kind) {
                sink.add(idx::ROOK_THREAT + offset, sign, 0.0);
                sink.add(idx::ROOK_THREAT + offset + 1, 0.0, sign);
            }
        }
    }

    let hanging = weak & (!attacks.all[enemy_side] | attacks.attacked_twice[side]);
    let hanging_count = f64::from(bit_count(hanging));
    sink.add(idx::HANGING_THREAT_MG, sign * hanging_count, 0.0);
    sink.add(idx::HANGING_THREAT_EG, 0.0, sign * hanging_count);

    let restricted = attacks.all[enemy_side] & !strongly_protected & attacks.all[side];
    sink.add(
        idx::RESTRICTED_THREAT_MG,
        sign * f64::from(bit_count(restricted)),
        0.0,
    );

    let safe = !attacks.all[enemy_side] | attacks.all[side];
    let safe_pawn_threats =
        attacks.by_kind[side][PieceKind::Pawn.index()] & non_pawn_enemies & safe;
    let safe_pawn_count = f64::from(bit_count(safe_pawn_threats));
    sink.add(idx::SAFE_PAWN_THREAT_MG, sign * safe_pawn_count, 0.0);
    sink.add(idx::SAFE_PAWN_THREAT_EG, 0.0, sign * safe_pawn_count);

    let own_pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
    let pushed_pawns = pawn_push_mask(color, own_pawns, !position.occupancy());
    let pawn_push_threats = pawn_attack_mask(color, pushed_pawns)
        & non_pawn_enemies
        & !attacks.by_kind[enemy_side][PieceKind::Pawn.index()]
        & safe;
    let push_count = f64::from(bit_count(pawn_push_threats));
    sink.add(idx::PAWN_PUSH_THREAT_MG, sign * push_count, 0.0);
    sink.add(idx::PAWN_PUSH_THREAT_EG, 0.0, sign * push_count);

    let enemy_queens = position.piece_bitboard(Piece::new(enemy, PieceKind::Queen));
    if enemy_queens != 0 {
        let queen = board_square(
            u8::try_from(enemy_queens.trailing_zeros()).expect("set bit index is below 64"),
        );
        let safe_knight_pressure =
            attacks.by_kind[side][PieceKind::Knight.index()] & knight_attacks(queen) & safe;
        let knight_count = f64::from(bit_count(safe_knight_pressure));
        sink.add(idx::QUEEN_KNIGHT_THREAT_MG, sign * knight_count, 0.0);
        sink.add(idx::QUEEN_KNIGHT_THREAT_EG, 0.0, sign * knight_count);

        let occupancy = position.occupancy();
        let slider_pressure = (attacks.by_kind[side][PieceKind::Bishop.index()]
            & diagonal_attacks(queen, occupancy)
            | attacks.by_kind[side][PieceKind::Rook.index()]
                & orthogonal_attacks(queen, occupancy))
            & attacks.attacked_twice[side]
            & safe;
        let slider_count = f64::from(bit_count(slider_pressure));
        sink.add(idx::QUEEN_SLIDER_THREAT_MG, sign * slider_count, 0.0);
        sink.add(idx::QUEEN_SLIDER_THREAT_EG, 0.0, sign * slider_count);
    }
}

/// Used for mapping a threatened piece kind to its typed-table pair offset.
///
/// # Arguments
///
/// * `target` - kind of the threatened enemy piece
///
/// # Returns
///
/// Offset of the middlegame slot within an eight-slot typed table, or
/// `None` for a king target.
fn typed_threat_offset(target: PieceKind) -> Option<usize> {
    match target {
        PieceKind::Pawn => Some(0),
        PieceKind::Knight | PieceKind::Bishop => Some(2),
        PieceKind::Rook => Some(4),
        PieceKind::Queen => Some(6),
        PieceKind::King => None,
    }
}

/// Used for crediting both kings' pressure coefficients.
///
/// Mirrors `king_pressure_term`/`side_king_pressure`. The `attackers^2` and
/// `flank_attack^2` quadratics and the positive mobility edge (evaluated at
/// the default mobility tables) stay in the constants; the queenless
/// `55 / 100` discount and single-attacker halving are applied as exact real
/// factors to every pressure component.
///
/// # Arguments
///
/// * `sink` - coefficient accumulator
/// * `position` - position holding the kings
/// * `attacks` - finished rich attack maps
#[allow(clippy::too_many_lines)]
fn king_pressure_coefficients(
    sink: &mut CoefficientSink,
    position: &Position,
    attacks: &AttackMaps,
) {
    for king_color in [Color::White, Color::Black] {
        let Some(king) = position.king_square(king_color) else {
            continue;
        };
        let king_sign = f64::from(color_sign(king_color));
        let defender = king_color.index();
        let attacker_color = king_color.opposite();
        let attacker = attacker_color.index();
        let attackers = attacks.king_attackers[attacker];
        if attackers > 0 {
            let queenless =
                position.piece_bitboard(Piece::new(attacker_color, PieceKind::Queen)) == 0;
            let mut factor = -king_sign;
            if queenless {
                factor *= 0.55;
            }
            if attackers == 1 {
                factor *= 0.5;
            }

            // Per-kind attacker counts and ring hits, matching scan_kind.
            let mut ring_hits = 0;
            for kind in [
                PieceKind::Pawn,
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
            ] {
                let mut kind_attackers = 0;
                let mut pieces = position.piece_bitboard(Piece::new(attacker_color, kind));
                while pieces != 0 {
                    let index =
                        u8::try_from(pieces.trailing_zeros()).expect("set bit index is below 64");
                    pieces &= pieces - 1;
                    let hits =
                        position.attacks_from(board_square(index)) & attacks.king_zone[defender];
                    if hits != 0 {
                        kind_attackers += 1;
                        ring_hits += bit_count(hits);
                    }
                }
                if kind_attackers > 0 {
                    sink.add(
                        idx::KING_ATTACK_WEIGHT + kind.index(),
                        factor * f64::from(kind_attackers),
                        0.0,
                    );
                }
            }
            sink.add(idx::KING_RING_ATTACK, factor * f64::from(ring_hits), 0.0);

            let weak = attacks.all[attacker]
                & !attacks.attacked_twice[defender]
                & (!attacks.all[defender]
                    | attacks.by_kind[defender][PieceKind::King.index()]
                    | attacks.by_kind[defender][PieceKind::Queen.index()]);
            let weak_count = f64::from(bit_count(attacks.king_zone[defender] & weak));
            sink.add(idx::KING_WEAK_SQUARE, factor * weak_count, 0.0);

            let flank = king_flank_mask(king) & camp_mask(king_color);
            let flank_attack = f64::from(
                bit_count(attacks.all[attacker] & flank)
                    + bit_count(attacks.attacked_twice[attacker] & flank),
            );
            let flank_defense = f64::from(bit_count(attacks.all[defender] & flank));
            sink.add(idx::KING_FLANK_DEFENSE, -factor * flank_defense, 0.0);

            let pawns = position.piece_bitboard(Piece::new(Color::White, PieceKind::Pawn))
                | position.piece_bitboard(Piece::new(Color::Black, PieceKind::Pawn));
            if pawns & flank == 0 {
                sink.add(idx::KING_PAWNLESS_FLANK, factor, 0.0);
            }

            // Frozen non-linear pressure components at the default weights:
            // squared attackers, squared flank attacks, and the positive
            // mobility edge from the default mobility tables.
            let quadratics = f64::from(attackers * attackers * KING_ATTACKERS_SQUARED)
                + flank_attack * flank_attack * 0.5
                + f64::from((attacks.mobility_mg[attacker] - attacks.mobility_mg[defender]).max(0))
                    * 0.5;
            sink.add_constant(factor * quadratics, 0.0);
        }
        if position.in_check(king_color) {
            sink.add_flat(idx::KING_CHECK_PENALTY, -king_sign);
        }
    }
}

