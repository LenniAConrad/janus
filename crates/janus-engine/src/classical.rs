//! Readable dependency-free classical chess evaluation.
//!
//! The release evaluator combines compact material/placement/pawn terms with a
//! measured `ChessRTK`-compatible attack-state slice. Terms remain separated so
//! later tuning can be measured and reviewed without hiding behavior in a
//! generated parameter blob.
//!
//! [`Classical`] is the stateless evaluator type. [`Classical::breakdown`]
//! exposes the individual white-relative terms so tests and tuning can inspect
//! each contribution independently.
//!
//! # Measured standing
//!
//! This evaluator is the shipped UCI default, and it is **far weaker than the
//! neural evaluators this engine also ships**. Measured on 2026-08-02 against
//! Stockfish 6 at 10+0.10, one thread, 64 MiB hash, over two independent runs
//! totalling 800 games: **0.33% and 0.75%**, roughly 850-990 Elo below that
//! opponent. The `UpstreamNNUE` evaluator scores about **51%** on the same
//! gauge with the same search.
//!
//! Work here is therefore research on a hand-crafted evaluation, not work on
//! the engine's competitive strength, which lives in the neural path. Anyone
//! configuring Janus for play should set `Eval` explicitly rather than accept
//! the default. Note also that these were the first honest measurements of
//! this evaluator: before the `Eval File` contract was fixed, every match
//! configured as `Eval=Classical` silently played the network instead, so
//! earlier evaluation experiments recorded against this module measured
//! nothing.

use crate::evaluator::{Evaluator, SearchEvaluator};
use crate::score::clamp_static_score;
use janus_core::{CastlingRights, Color, Move, Piece, PieceKind, Position, Square};
use std::sync::Arc;

/// Used for scoring base centipawn material, indexed by [`PieceKind::index`].
///
/// Values follow the stage-1 Texel-fitted 98/357/382/585/1211 centipawn
/// scale (STR-20260724-170) with a zero entry for the king, which is never
/// exchanged.
const MATERIAL: [i32; 6] = [98, 357, 382, 585, 1211, 0];
/// Used for capping the tapered-evaluation phase at the opening value
/// produced by the initial non-pawn material.
///
/// [`game_phase`] weighs minors at one, rooks at two, and queens at four, so
/// a full army sums to exactly this value. Middlegame/endgame interpolation
/// divides by it.
const MAX_PHASE: i32 = 24;
/// Used for awarding initiative to the side to move before perspective
/// conversion.
///
/// The bonus enters the white-relative sum as positive when White is to move
/// and negative when Black is to move, so it always favors the mover.
const TEMPO: i32 = 12;
/// Used for calibrating the score after all classical terms have been
/// combined.
///
/// [`side_to_move_score`] multiplies the side-relative total by this
/// percentage before clamping it into the static-score range.
const SEARCH_SCORE_SCALE_PERCENT: i32 = 105;
/// Used as the residual draw floor for a bishops-and-pawns endgame in which
/// the sole bishops stand on opposite square colors (HCE-172).
///
/// [`opposite_colored_bishop_scale_percent`] shrinks the white-relative total
/// to this percentage when neither side has an extra pawn: with only opposite
/// bishops left, the defending bishop permanently guards squares the attacker
/// can never cover, so a small material or positional edge almost never wins.
/// This is a genuinely non-linear drawishness *scaling* — it multiplies the
/// whole evaluation rather than adding a term — so it lives outside the linear
/// Texel basis and has no [`tuning`] lockstep entry. Calibrated against the
/// NNUE-teacher residual over the HCE-168 corpus (HCE-172 residual mining).
const OPPOSITE_BISHOP_DRAW_FLOOR_PERCENT: i32 = 10;
/// Used for restoring winning weight per extra pawn in an opposite-colored
/// bishops endgame (HCE-172).
///
/// Each pawn of material advantage *beyond the first* adds this many
/// percentage points back toward the unscaled evaluation, so a decisive
/// pawn-count edge (roughly three pawns) is scored at full strength while a
/// one-pawn edge stays drawish. Paired with
/// [`OPPOSITE_BISHOP_DRAW_FLOOR_PERCENT`] in
/// [`opposite_colored_bishop_scale_percent`].
const OPPOSITE_BISHOP_PAWN_ADVANTAGE_STEP_PERCENT: i32 = 40;
/// Used for the non-pawn material edge below which a pawnless side cannot win.
///
/// A rook's worth of extra material is the classical threshold: below it, a
/// side with no pawns has no winning method against sensible defence.
const PAWNLESS_WINNING_EDGE: i32 = 500;
/// Used for the residual score retained in a drawn pawnless ending.
///
/// The remainder is kept rather than zeroed so the search still prefers the
/// better side of a drawish position and does not become indifferent between
/// defending well and defending badly.
const PAWNLESS_DRAW_FLOOR_PERCENT: i32 = 20;
/// Used for selecting the squares of one bishop color, matching the parity
/// convention in [`bad_bishop_penalty`].
///
/// A square with `(file + row)` even lies in this set. Two lone bishops stand
/// on opposite colors exactly when one intersects this mask and the other does
/// not, which is how [`opposite_colored_bishop_scale_percent`] detects the
/// opposite-bishop configuration.
const DARK_SQUARE_MASK: u64 = 0xaa55_aa55_aa55_aa55;
/// Used for selecting whether the measured rich attack-state candidate is the
/// released default.
///
/// [`Classical::breakdown`] forwards this flag, so the release evaluator and
/// the `true` research flavor stay identical by construction.
const RELEASE_RICH_ATTACK_STATE: bool = true;
/// Used for looking up forward own/adjacent-file masks indexed by color and
/// pawn square.
///
/// Built at compile time by [`build_passed_pawn_blocker_masks`]; the first
/// table holds White masks and the second holds Black masks, matching
/// [`Color::index`]. Intersecting a mask with the enemy pawn bitboard answers
/// passed-pawn status in a single operation.
const PASSED_PAWN_BLOCKER_MASKS: [[u64; 64]; 2] = build_passed_pawn_blocker_masks();

/// Used for penalizing each extra same-color pawn on a file.
///
/// [`pawn_structure`] applies this released weight to `count - 1` for every
/// occupied file. The broader per-file recognition contract is unchanged by
/// STR-299's research calibration.
const DOUBLED_PAWN_PENALTY: i32 = 13;
/// Used for STR-299's trace-calibrated doubled-pawn research flavor.
///
/// This replaces only [`DOUBLED_PAWN_PENALTY`] in a compile-time evaluator
/// instantiation; isolated and passed-pawn weights and all pawn geometry stay
/// on the release path.
const PAWN_STRUCTURE_CANDIDATE_DOUBLED_PENALTY: i32 = 3;
/// Used for the confirmed STR-319 connected-pawn ramp percentage.
///
/// The `bulk-nodes` magnitude sweep measured `60%` as a null while `150%` was
/// indistinguishable from `100%`, so the smaller of the two equal-strength
/// scales is adopted and the released term needs no runtime scaling.
const CONNECTED_PAWN_SCALE_PERCENT: i32 = 100;

/// Used for the STR-319 connected-pawn bonus, indexed by color-relative rank.
///
/// Before this term the released pawn evaluation carried exactly three — a
/// doubled penalty, an isolated penalty, and a passed bonus — and therefore
/// scored a defended or side-by-side pawn identically to a loose one. Every
/// mature classical evaluator rewards that structure, and the 2026-08-05
/// equal-node decomposition attributed the Stockfish 11 deficit to evaluation
/// capacity rather than calibration. Entries are `(middlegame, endgame)`
/// centipawns; indices `0`, `1`, and `8` are unreachable for a pawn and stay
/// zero. The ramp is Janus's own: the abstract mechanism is learned from the
/// references recorded in the experiment entry, and no third-party table or
/// constant is copied.
const CONNECTED_PAWN_BONUS: [(i32, i32); 9] = [
    (0, 0),
    (0, 0),
    (6, 3),
    (9, 6),
    (14, 11),
    (24, 20),
    (42, 38),
    (65, 62),
    (0, 0),
];

/// Used for the STR-321 knight adjustment per own pawn above five.
///
/// Knights gain value in closed positions and lose it as pawns leave the
/// board; rooks move the other way. Janus's released material values are
/// constant in the pawn count, so it cannot express either. The abstract
/// mechanism is standard classical-engine knowledge and is implemented here
/// with Janus's own coefficients rather than any reference table.
const KNIGHT_PAWN_COUNT_ADJUSTMENT: i32 = 5;

/// Used for the STR-321 rook adjustment per own pawn above five.
const ROOK_PAWN_COUNT_ADJUSTMENT: i32 = -7;


/// Used for the released nonlinear reward per squared passer advance.
///
/// STR-170 fitted the separate flat base while leaving this coefficient
/// frozen. Keeping it named makes the release/tuning parity contract explicit.
const PASSED_PAWN_QUADRATIC: i32 = 5;
/// Used for STR-306's trace-calibrated passed-pawn research flavor.
///
/// This replaces only [`PASSED_PAWN_QUADRATIC`] in one compile-time evaluator
/// instantiation; recognition, rank, flat base, blocker truncation, and scan
/// order remain shared with the release path.
const PASSED_PAWN_CANDIDATE_QUADRATIC: i32 = 2;

/// Used for scoring middlegame knight mobility, indexed by safe destination
/// count.
///
/// Counts beyond the last index reuse the final entry via
/// [`mobility_scores`].
const KNIGHT_MOBILITY_MG: [i32; 9] = [-17, -8, 2, 4, 9, 13, 10, 13, 14];
/// Used for scoring endgame knight mobility, indexed by safe destination
/// count.
///
/// Counts beyond the last index reuse the final entry via
/// [`mobility_scores`].
const KNIGHT_MOBILITY_EG: [i32; 9] = [-24, -18, -8, -2, 6, 9, 17, 10, 9];
/// Used for scoring middlegame bishop mobility, indexed by safe destination
/// count.
///
/// Counts beyond the last index reuse the final entry via
/// [`mobility_scores`].
const BISHOP_MOBILITY_MG: [i32; 14] = [-17, -4, 1, 12, 11, 11, 16, 21, 24, 28, 31, 32, 34, 35];
/// Used for scoring endgame bishop mobility, indexed by safe destination
/// count.
///
/// Counts beyond the last index reuse the final entry via
/// [`mobility_scores`].
const BISHOP_MOBILITY_EG: [i32; 14] = [-18, -10, 0, 6, 13, 22, 25, 28, 36, 38, 39, 39, 42, 41];
/// Used for scoring middlegame rook mobility, indexed by safe destination
/// count.
///
/// Counts beyond the last index reuse the final entry via
/// [`mobility_scores`].
const ROOK_MOBILITY_MG: [i32; 15] = [-15, -8, -2, 2, 5, 9, 12, 15, 19, 19, 24, 27, 29, 32, 31];
/// Used for scoring endgame rook mobility, indexed by safe destination count.
///
/// Counts beyond the last index reuse the final entry via
/// [`mobility_scores`].
const ROOK_MOBILITY_EG: [i32; 15] = [-23, -9, -3, 7, 15, 26, 28, 42, 44, 49, 49, 54, 58, 57, 48];
/// Used for scoring middlegame queen mobility, indexed by safe destination
/// count.
///
/// Counts beyond the last index reuse the final entry via
/// [`mobility_scores`].
const QUEEN_MOBILITY_MG: [i32; 28] = [
    -8, -4, -1, 0, 3, 4, 8, 8, 14, 13, 19, 22, 20, 23, 23, 22, 25, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34,
];
/// Used for scoring endgame queen mobility, indexed by safe destination
/// count.
///
/// Counts beyond the last index reuse the final entry via
/// [`mobility_scores`].
const QUEEN_MOBILITY_EG: [i32; 28] = [
    -12, -8, -4, 0, 5, 10, 14, 18, 22, 26, 30, 34, 41, 42, 44, 47, 51, 53, 56, 58, 59, 62, 65, 65,
    68, 70, 74, 76,
];
/// Used for STR-297's middlegame knight mobility research flavor.
///
/// The table is the frozen affine trace calibration of
/// [`KNIGHT_MOBILITY_MG`]; its length and destination-count indexing are
/// identical to the release table.
const MOBILITY_CANDIDATE_KNIGHT_MG: [i32; 9] = [-27, -19, -10, -8, -3, 0, -2, 0, 1];
/// Used for STR-297's endgame knight mobility research flavor.
///
/// This is the endgame counterpart of [`MOBILITY_CANDIDATE_KNIGHT_MG`].
const MOBILITY_CANDIDATE_KNIGHT_EG: [i32; 9] = [-34, -30, -23, -18, -12, -10, -4, -10, -10];
/// Used for STR-297's middlegame bishop mobility research flavor.
///
/// Counts beyond index thirteen retain the release scorer's last-entry clamp.
const MOBILITY_CANDIDATE_BISHOP_MG: [i32; 14] =
    [-16, -5, 0, 9, 8, 8, 13, 17, 20, 23, 26, 27, 29, 30];
/// Used for STR-297's endgame bishop mobility research flavor.
///
/// Its values are independently calibrated while preserving the release
/// table's domain and lookup contract.
const MOBILITY_CANDIDATE_BISHOP_EG: [i32; 14] =
    [-30, -24, -16, -11, -5, 2, 5, 7, 14, 15, 16, 16, 18, 18];
/// Used for STR-297's middlegame rook mobility research flavor.
///
/// The fifteen entries correspond exactly to safe destination counts zero
/// through fourteen.
const MOBILITY_CANDIDATE_ROOK_MG: [i32; 15] =
    [-28, -22, -17, -14, -12, -9, -6, -4, -1, -1, 3, 5, 7, 9, 8];
/// Used for STR-297's endgame rook mobility research flavor.
///
/// Counts beyond the table reuse its final entry through [`mobility_scores`].
const MOBILITY_CANDIDATE_ROOK_EG: [i32; 15] =
    [-26, -15, -10, -1, 5, 14, 16, 28, 29, 33, 33, 37, 41, 40, 33];
/// Used for STR-297's middlegame queen mobility research flavor.
///
/// The frozen affine model is materialized here so candidate evaluation adds
/// no arithmetic to the existing table lookup.
const MOBILITY_CANDIDATE_QUEEN_MG: [i32; 28] = [
    -26, -22, -20, -19, -17, -16, -13, -13, -8, -9, -4, -2, -3, -1, -1, -2, 1, 0, 1, 2, 2, 3, 4, 5,
    6, 6, 7, 8,
];
/// Used for STR-297's endgame queen mobility research flavor.
///
/// This table retains all twenty-eight release indices and the same final
/// entry clamp.
const MOBILITY_CANDIDATE_QUEEN_EG: [i32; 28] = [
    -43, -40, -37, -35, -31, -28, -25, -23, -20, -17, -15, -12, -7, -7, -5, -3, -1, 0, 2, 4, 4, 6,
    8, 8, 10, 12, 14, 16,
];

/// Used for weighting king-zone attackers with the Java-calibrated values,
/// indexed by [`PieceKind::index`].
///
/// Each non-king piece that reaches the enemy king zone contributes its
/// kind's weight to [`AttackMaps::king_attacker_weight`].
const KING_ATTACK_WEIGHT: [i32; 6] = [-2, 10, 8, 12, 19, 0];
/// Used for assigning king-zone pressure per squared attacker count.
///
/// [`side_king_pressure`] multiplies the squared number of distinct king-zone
/// attackers by this factor.
const KING_ATTACKERS_SQUARED: i32 = 3;
/// Used for assigning king-zone pressure per king-zone square reached by an
/// attacking piece.
///
/// Multiplied by [`AttackMaps::king_ring_hits`] in [`side_king_pressure`];
/// a square reached by several attackers counts once per attacker.
const KING_RING_ATTACK: i32 = 15;
/// Used for assigning king-zone pressure per weak king-zone square.
///
/// A square is weak when the attacker covers it, the defender does not cover
/// it twice, and at most the defending king or queen protects it.
const KING_WEAK_SQUARE: i32 = 9;
/// Used for assigning king-zone relief per defended flank square.
///
/// Subtracted once for every camp-half king-flank square the defender
/// attacks in [`side_king_pressure`].
const KING_FLANK_DEFENSE: i32 = 5;
/// Used for adding king-zone pressure when the king flank contains no pawn.
///
/// Applied once in [`side_king_pressure`] when neither side has a pawn on
/// the camp half of the king's flank.
const KING_PAWNLESS_FLANK: i32 = 18;
/// Used for discounting attack pressure when the attacking army has no queen.
///
/// The same factor is applied to released pressure and research safe-check
/// features so offline fitting cannot accidentally bypass the production gate.
const KING_QUEENLESS_PRESSURE_PERCENT: i32 = 55;
/// Used for adding released-pressure context when a safe check is available.
///
/// This and the following STR-280 values are rounded from the mean of five
/// deterministic reference-trace folds; they are Janus-derived rather than
/// borrowed reference-engine parameters.
const KING_DANGER_CURRENT_DELTA_PERCENT: i32 = 191;
/// Used for valuing queen-gated safe-check destinations by checking piece.
///
/// Entries are knight, bishop, rook, and queen in the order returned by
/// [`safe_check_destinations`].
const KING_DANGER_SAFE_CHECK: [i32; 4] = [156, 89, 189, 9];
/// Used for penalizing each missing nearby king-shelter file.
const KING_DANGER_MISSING_SHELTER: i32 = 42;
/// Used for penalizing each rank a shelter pawn advances beyond first cover.
const KING_DANGER_ADVANCED_SHELTER: i32 = 16;
/// Used for the fitted concave correction to already nonlinear raw pressure.
const KING_DANGER_PRESSURE_SQUARED: i32 = -2;
/// Used for bounding the complete candidate king-pressure term in centipawns.
const KING_DANGER_TERM_LIMIT: i32 = 1_500;
/// Used for scoring the STR-304 nearest-pawn-file balance in each phase.
///
/// The pair is independently fitted on the unsealed development trace and is
/// applied only by the research flavor layered on STR-280.
const KING_PAWN_FILE_PROXIMITY: (i32, i32) = (7, 2);
/// Used for bounding the complete STR-304 white-relative correction.
///
/// Legal file distances make the mathematical maximum smaller than this
/// limit; the explicit clamp keeps malformed research positions bounded too.
const KING_PAWN_FILE_PROXIMITY_TERM_LIMIT: i32 = 96;
/// Used for the STR-295 low-cost passer feature order shared by both phases.
///
/// Entries are blocked advance, unsafe advance, safe advance, defended
/// advance, and outside-file distance, matching the cheap subset of
/// [`PassedPawnFeatures`].
const PASSED_REALIZATION_MG_X100: [i32; 5] = [-240, -973, -391, -34, 40];
/// Used for the endgame half of the STR-295 passer-state vector.
///
/// The entry order is identical to [`PASSED_REALIZATION_MG_X100`].
const PASSED_REALIZATION_EG_X100: [i32; 5] = [-308, -961, -255, -56, 69];
/// Used for converting STR-295's integer-hundredth coefficients to
/// centipawns.
const PASSED_REALIZATION_WEIGHT_SCALE: i64 = 100;
/// Used for bounding the complete phase-tapered STR-295 passer-state delta.
const PASSED_REALIZATION_TERM_LIMIT: i32 = 1_000;
/// Used for the STR-296 endgame-initiative feature order.
///
/// Entries are intercept, total pawn count, king outflanking, king
/// infiltration, pawns on both board flanks, and pure pawn ending. The
/// integer-hundredth values were frozen from the five-fold SF11 Initiative
/// trace fit; its middlegame vector is exactly zero.
const INITIATIVE_EG_X100: [i32; 6] = [-2_392, 224, 255, 540, 1_155, 1_699];
/// Used for converting STR-296's integer-hundredth coefficients to
/// centipawns.
const INITIATIVE_WEIGHT_SCALE: i64 = 100;
/// Used for bounding the complete phase-tapered STR-296 initiative delta.
///
/// Legal values of the frozen six-feature vector remain below half this
/// limit; the extra range is an overflow and future-drift guard.
const INITIATIVE_TERM_LIMIT: i32 = 128;
/// Used for detecting whether any pawn remains on files a through d.
const QUEENSIDE_FLANK_MASK: u64 = 0x0f0f_0f0f_0f0f_0f0f;
/// Used for detecting whether any pawn remains on files e through h.
const KINGSIDE_FLANK_MASK: u64 = 0xf0f0_f0f0_f0f0_f0f0;
/// Used for rewarding a pawn-supported knight outpost with midgame and
/// endgame bonuses.
///
/// Applied by [`piece_activity`] when [`is_outpost`] accepts the knight's
/// square.
const KNIGHT_OUTPOST: (i32, i32) = (27, 14);
/// Used for rewarding a pawn-supported bishop outpost with midgame and
/// endgame bonuses.
///
/// Applied by [`piece_activity`] when [`is_outpost`] accepts the bishop's
/// square.
const BISHOP_OUTPOST: (i32, i32) = (15, 8);
/// Used for weighting a hanging enemy piece with midgame and endgame values.
///
/// [`side_threats`] treats a weak enemy as hanging when it is undefended or
/// attacked at least twice.
const HANGING_THREAT: (i32, i32) = (21, 25);
/// Used for weighting a midgame square that both sides attack and the enemy
/// does not strongly protect.
///
/// Applied per restricted square in [`side_threats`]; the term has no
/// endgame counterpart.
const RESTRICTED_THREAT_MG: i32 = 4;
/// Used for weighting a safe pawn attack on a non-pawn with midgame and
/// endgame values.
///
/// Applied per threatened non-pawn piece in [`side_threats`].
const SAFE_PAWN_THREAT: (i32, i32) = (43, 30);
/// Used for weighting a safe pawn push that creates an attack with midgame
/// and endgame values.
///
/// Applied in [`side_threats`] per non-pawn piece that a one- or two-step
/// pawn push through empty squares would safely attack.
const PAWN_PUSH_THREAT: (i32, i32) = (13, 8);
/// Used for weighting safe knight pressure on the enemy queen with midgame
/// and endgame values.
///
/// Applied in [`side_threats`] per safe knight-reachable square lying a
/// knight's move from the enemy queen.
const QUEEN_KNIGHT_THREAT: (i32, i32) = (7, 7);
/// Used for weighting doubly supported slider pressure on the enemy queen
/// with midgame and endgame values.
///
/// Applied in [`side_threats`] per safe, doubly attacked square from which a
/// bishop or rook bears on the enemy queen.
const QUEEN_SLIDER_THREAT: (i32, i32) = (15, 4);
/// Used for STR-298's hanging-piece research weight.
///
/// The pair is the released value rescaled by the frozen hanging/restricted
/// family coefficients and rounded once before compilation.
const THREAT_CANDIDATE_HANGING: (i32, i32) = (20, 15);
/// Used for STR-298's restricted-square middlegame research weight.
///
/// Restricted squares have no endgame slot in the released evaluator, so the
/// candidate preserves that structural zero.
const THREAT_CANDIDATE_RESTRICTED_MG: i32 = 4;
/// Used for STR-298's safe-pawn-attack research weight.
///
/// The pair materializes the independently fitted pawn-family phase scales.
const THREAT_CANDIDATE_SAFE_PAWN: (i32, i32) = (64, 42);
/// Used for STR-298's safe pawn-push research weight.
///
/// This shares the pawn-family scale with [`THREAT_CANDIDATE_SAFE_PAWN`].
const THREAT_CANDIDATE_PAWN_PUSH: (i32, i32) = (19, 11);
/// Used for STR-298's knight-on-queen research weight.
///
/// The pair materializes the fitted queen-pressure family scale.
const THREAT_CANDIDATE_QUEEN_KNIGHT: (i32, i32) = (6, 3);
/// Used for STR-298's slider-on-queen research weight.
///
/// This shares the queen-pressure scale with
/// [`THREAT_CANDIDATE_QUEEN_KNIGHT`].
const THREAT_CANDIDATE_QUEEN_SLIDER: (i32, i32) = (14, 2);

/// Position-wide geometric attacks and Java-compatible safe mobility totals.
///
/// Built once per evaluation by [`AttackMaps::build`]. Every per-color array
/// is indexed by [`Color::index`]. The rich attack-state accumulators
/// (attacked-twice, king-zone, and activity slices) are only populated when
/// the rich flavor is requested.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AttackMaps {
    /// Used for storing the union of attacks for each color and piece kind.
    ///
    /// The inner array is indexed by [`PieceKind::index`].
    by_kind: [[u64; 6]; 2],
    /// Used for storing the union of every attacked square for each color.
    all: [u64; 2],
    /// Used for storing squares attacked by at least two pieces of each
    /// color.
    ///
    /// Only populated when the rich attack state is enabled.
    attacked_twice: [u64; 2],
    /// Used for storing the king square plus adjacent squares for each color.
    ///
    /// Only populated when the rich attack state is enabled; empty when a
    /// color has no king on the board.
    king_zone: [u64; 2],
    /// Used for counting pieces reaching the enemy king zone for each
    /// attacker color.
    king_attackers: [i32; 2],
    /// Used for accumulating piece-weighted pressure reaching the enemy king
    /// zone.
    ///
    /// Each attacker adds its [`KING_ATTACK_WEIGHT`] entry.
    king_attacker_weight: [i32; 2],
    /// Used for counting distinct per-piece hits into the enemy king zone.
    king_ring_hits: [i32; 2],
    /// Used for accumulating the middlegame safe-mobility score for each
    /// color.
    mobility_mg: [i32; 2],
    /// Used for accumulating the endgame safe-mobility score for each color.
    mobility_eg: [i32; 2],
    /// Used for accumulating the middlegame piece-activity score for each
    /// color.
    ///
    /// Only populated when the rich attack state is enabled.
    activity_mg: [i32; 2],
    /// Used for accumulating the endgame piece-activity score for each color.
    ///
    /// Only populated when the rich attack state is enabled.
    activity_eg: [i32; 2],
}

/// Research-only named inputs to Janus's king-danger capacity experiment.
///
/// Every field is White-relative: positive values describe pressure against
/// Black's king or better cover around White's king. Safe-check fields are
/// scaled by 100 with the release evaluator's existing queen-absence discount
/// already applied, so later fitting cannot accidentally omit that gate.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KingDangerFeatures {
    /// Used for carrying the exact released, phase-tapered king-pressure term.
    pub current_pressure: i32,
    /// Used for counting queen-gated safe knight-check destinations.
    pub safe_knight_checks_x100: i32,
    /// Used for counting queen-gated safe bishop-check destinations.
    pub safe_bishop_checks_x100: i32,
    /// Used for counting queen-gated safe rook-check destinations.
    pub safe_rook_checks_x100: i32,
    /// Used for counting queen-gated safe queen-check destinations.
    pub safe_queen_checks_x100: i32,
    /// Used for counting king-flank files without a nearby shelter pawn.
    pub missing_shelter_files: i32,
    /// Used for summing how far nearby shelter pawns advanced beyond the
    /// closest protective rank.
    pub advanced_shelter_steps: i32,
    /// Used for summing enemy-pawn proximity on the three king-flank files.
    pub pawn_storm_proximity: i32,
    /// Used for the signed convex interaction of each side's existing raw
    /// pressure, normalized by 256 after the queen gate.
    pub pressure_squared_over_256: i32,
    /// Used for reproducing the release material-phase taper in offline fits.
    pub phase: i32,
}

/// Research-only named inputs to Janus's passed-pawn realization experiment.
///
/// Every score-bearing field is White-relative, so reflecting the armies and
/// board vertically negates the field. [`Self::current_bonus`] is the exact
/// released passed-pawn contribution, including immediate-blocker halving.
/// The remaining fields are zero below the color-relative fourth rank and use
/// a rank urgency of one through four, squared where a state is a per-passer
/// event. They describe geometry only and carry no fitted score constants.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PassedPawnFeatures {
    /// Used for carrying the exact released passed-pawn bonus.
    pub current_bonus: i32,
    /// Used for summing squared rank urgency over advanced passers.
    pub rank_pressure: i32,
    /// Used for advanced passers whose immediate front square is occupied.
    pub blocked_advance: i32,
    /// Used for advanced passers whose empty front square is enemy-attacked.
    pub unsafe_advance: i32,
    /// Used for advanced passers whose empty front square is not enemy-attacked.
    pub safe_advance: i32,
    /// Used for advanced passers whose empty front square is defended by their
    /// own army.
    pub defended_advance: i32,
    /// Used for advanced passers with an entirely empty and enemy-unattacked
    /// promotion file ahead.
    pub safe_path: i32,
    /// Used for rank-weighted distance from each passer's king to its front
    /// square.
    pub own_king_distance: i32,
    /// Used for rank-weighted distance from the opposing king to each passer's
    /// front square.
    pub enemy_king_distance: i32,
    /// Used for advanced passers whose first rearward file blocker is an own
    /// rook.
    pub rook_behind: i32,
    /// Used for squared rank urgency multiplied by distance from the central
    /// files.
    pub outside_file: i32,
    /// Used for advanced passers outside the opposing king's promotion-square
    /// catch distance when that army has no non-pawn piece.
    pub square_rule: i32,
    /// Used for reproducing the release material-phase taper in offline fits.
    pub phase: i32,
}

/// Research-only exact score grid for Janus's passed-pawn rank curve.
///
/// The grid reuses the released true-passer traversal and blocker decision.
/// Entry `q` in [`Self::quadratic_deltas`] is the White-relative score change
/// from replacing only `5 * advance^2` with `q * advance^2`, for
/// `q` in `0..=16`. The flat base and integer blocker halving stay exact.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PassedPawnCurveFeatures {
    /// Used for carrying the exact released passed-pawn contribution.
    pub current_bonus: i32,
    /// Used for reproducing the released material phase in offline audits.
    pub phase: i32,
    /// Used for replaying every predeclared quadratic coefficient in `0..=16`.
    pub quadratic_deltas: [i32; 17],
}

/// Research-only named inputs to Janus's endgame-initiative experiment.
///
/// Board-derived fields are color-invariant. [`Self::sign`] records the sign
/// of the white-relative STR-280 candidate score that the correction extends,
/// while [`Self::candidate_delta`] is white-relative and therefore negates
/// under an exact vertical color reflection. The delta is phase tapered,
/// bounded, and capped so it cannot reverse which side the input score favors.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InitiativeFeatures {
    /// Used for carrying the sign of the pre-initiative white score.
    pub sign: i32,
    /// Used for fitting the constant offset of the complexity model.
    pub intercept: i32,
    /// Used for counting all pawns of either color.
    pub total_pawns: i32,
    /// Used for the kings' file distance minus their rank distance.
    pub outflanking: i32,
    /// Used for recording whether either king crossed the central boundary
    /// into the opposing half of the board.
    pub infiltration: i32,
    /// Used for recording pawn presence on both the a--d and e--h flanks.
    pub both_flanks: i32,
    /// Used for recording positions with no knight, bishop, rook, or queen.
    pub pawn_endgame: i32,
    /// Used for reproducing the release material-phase taper in offline fits.
    pub phase: i32,
    /// Used for carrying the exact fitted, bounded, sign-preserving delta.
    pub candidate_delta: i32,
}

/// Minimal accumulator used only by STR-295's score-bearing hot path.
///
/// Keeping the rejected research geometry out of this structure prevents its
/// zero-initialization and register pressure from leaking into evaluation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PassedPawnCandidateFeatures {
    /// Used for advanced passers whose immediate front square is occupied.
    blocked_advance: i32,
    /// Used for advanced passers whose empty front square is enemy-attacked.
    unsafe_advance: i32,
    /// Used for advanced passers whose empty front square is safe.
    safe_advance: i32,
    /// Used for safe or unsafe empty fronts defended by the passer's army.
    defended_advance: i32,
    /// Used for rank urgency weighted by distance from the central files.
    outside_file: i32,
    /// Used for reproducing the release material-phase taper.
    phase: i32,
}

/// Position-wide immutable inputs shared by advanced passed pawns.
///
/// Constructed lazily only when the shared passer traversal reaches relative
/// rank four. Keeping the hot geometry in bitboards avoids repeated board
/// lookups while preserving the exact STR-293 feature semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PassedPawnContext {
    /// Used for testing complete forward paths and rear blockers.
    occupancy: u64,
    /// Used for direct per-color attack-union membership tests.
    attacks: [u64; 2],
    /// Used for identifying whether the nearest rear blocker is an own rook.
    rooks: [u64; 2],
    /// Used for the two king-distance realization features.
    kings: [Option<Square>; 2],
    /// Used for admitting the pawn-race square-rule feature by color.
    has_non_pawn: [bool; 2],
    /// Used for the tempo adjustment in the square-rule catch distance.
    side_to_move: Color,
}

impl PassedPawnContext {
    /// Used for caching the position-wide part of passed-pawn geometry.
    ///
    /// # Arguments
    ///
    /// * `position` - position containing the admitted true passer
    /// * `attacks` - complete rich attack unions already built by evaluation
    ///
    /// # Returns
    ///
    /// Copyable context used by every advanced passer in the position.
    fn build(position: &Position, attacks: &AttackMaps) -> Self {
        let mut rooks = [0_u64; 2];
        let mut kings = [None; 2];
        let mut has_non_pawn = [false; 2];
        for color in [Color::White, Color::Black] {
            let pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
            let king = position.piece_bitboard(Piece::new(color, PieceKind::King));
            rooks[color.index()] = position.piece_bitboard(Piece::new(color, PieceKind::Rook));
            kings[color.index()] = position.king_square(color);
            has_non_pawn[color.index()] = position.color_occupancy(color) & !(pawns | king) != 0;
        }
        Self {
            occupancy: position.occupancy(),
            attacks: attacks.all,
            rooks,
            kings,
            has_non_pawn,
            side_to_move: position.side_to_move(),
        }
    }
}

impl AttackMaps {
    /// Used for building attack unions and safe-mobility totals in
    /// dependency order.
    ///
    /// Pawn control for both colors is collected first because enemy pawn
    /// attacks are excluded from every non-pawn mobility area. When the rich
    /// attack state is requested, both king zones are prepared before any
    /// scan so king-zone attacker accounting sees complete zones.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose attacks are collected
    /// * `rich_attack_state` - whether to also fill the rich attack-state
    ///   accumulators
    ///
    /// # Returns
    ///
    /// Fully populated attack maps for both colors.
    fn build(position: &Position, rich_attack_state: bool) -> Self {
        Self::build_with_mobility::<false>(position, rich_attack_state)
    }

    /// Used for building attack maps with one compile-time mobility-table
    /// flavor.
    ///
    /// Keeping the table choice const-generic lets the optimizer produce the
    /// same scan and lookup shape for release and STR-297. The public engine
    /// path reaches this helper only through [`Self::build`] and therefore
    /// remains locked to the released tables.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose attacks are collected
    /// * `rich_attack_state` - whether to also fill the rich attack-state
    ///   accumulators
    ///
    /// # Returns
    ///
    /// Fully populated attack maps for the selected table flavor.
    fn build_with_mobility<const CANDIDATE_MOBILITY: bool>(
        position: &Position,
        rich_attack_state: bool,
    ) -> Self {
        let mut maps = Self::default();

        if rich_attack_state {
            for color in [Color::White, Color::Black] {
                maps.king_zone[color.index()] = position
                    .king_square(color)
                    .map_or(0, |king| king.bit() | king_ring(king));
            }
        }

        // Both pawn maps must exist before either side's safe-mobility area is
        // formed. Attacks are added one source at a time so attacked-twice
        // tracking uses the same scan contract as every other attack union.
        for color in [Color::White, Color::Black] {
            maps.scan_kind::<CANDIDATE_MOBILITY>(
                position,
                color,
                PieceKind::Pawn,
                0,
                rich_attack_state,
            );
        }

        for color in [Color::White, Color::Black] {
            let enemy = color.opposite();
            let enemy_king = position.piece_bitboard(Piece::new(enemy, PieceKind::King));
            let enemy_pawn_attacks = maps.by_kind[enemy.index()][PieceKind::Pawn.index()];
            let mobility_area =
                !(position.color_occupancy(color) | enemy_king | enemy_pawn_attacks);
            for kind in [
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
                PieceKind::King,
            ] {
                maps.scan_kind::<CANDIDATE_MOBILITY>(
                    position,
                    color,
                    kind,
                    mobility_area,
                    rich_attack_state,
                );
            }
        }
        maps
    }

    /// Used for adding every attack from `kind` and scoring its destinations
    /// in `mobility_area`.
    ///
    /// Each piece of the kind is scanned individually so attacked-twice
    /// tracking, safe-mobility scores, per-piece activity, and king-zone
    /// attacker counts all observe one attack source at a time. Pawns and
    /// kings contribute attacks but no mobility or activity; kings are also
    /// excluded from king-zone attacker accounting.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose pieces are scanned
    /// * `color` - color owning the scanned pieces
    /// * `kind` - piece kind to scan
    /// * `mobility_area` - squares that count toward safe mobility
    /// * `rich_attack_state` - whether to fill the rich attack-state
    ///   accumulators
    fn scan_kind<const CANDIDATE_MOBILITY: bool>(
        &mut self,
        position: &Position,
        color: Color,
        kind: PieceKind,
        mobility_area: u64,
        rich_attack_state: bool,
    ) {
        let piece = Piece::new(color, kind);
        let mut pieces = position.piece_bitboard(piece);
        while pieces != 0 {
            let index = u8::try_from(pieces.trailing_zeros()).expect("set bit index is below 64");
            pieces &= pieces - 1;
            let attacks = position.attacks_for_piece(board_square(index), piece);
            self.add(color, kind, attacks, rich_attack_state);
            if !matches!(kind, PieceKind::Pawn | PieceKind::King) {
                let mobility = usize::try_from((attacks & mobility_area).count_ones())
                    .expect("a bit count fits usize");
                let (middle, ending) = mobility_scores::<CANDIDATE_MOBILITY>(kind, mobility);
                self.mobility_mg[color.index()] += middle;
                self.mobility_eg[color.index()] += ending;
                if rich_attack_state {
                    let (middle_activity, ending_activity) = piece_activity(
                        position,
                        color,
                        kind,
                        board_square(index),
                        mobility,
                        attacks,
                        self,
                    );
                    self.activity_mg[color.index()] += middle_activity;
                    self.activity_eg[color.index()] += ending_activity;
                }
            }
            if rich_attack_state && kind != PieceKind::King {
                let enemy = color.opposite();
                let king_hits = attacks & self.king_zone[enemy.index()];
                if king_hits != 0 {
                    self.king_attackers[color.index()] += 1;
                    self.king_attacker_weight[color.index()] += KING_ATTACK_WEIGHT[kind.index()];
                    self.king_ring_hits[color.index()] += bit_count(king_hits);
                }
            }
        }
    }

    /// Used for merging one piece's attack bitboard into its kind and color
    /// unions.
    ///
    /// When the rich attack state is enabled, squares already covered by the
    /// color are first recorded as attacked twice.
    ///
    /// # Arguments
    ///
    /// * `color` - color owning the attacking piece
    /// * `kind` - kind of the attacking piece
    /// * `attacks` - attack bitboard of the single piece
    /// * `rich_attack_state` - whether to track attacked-twice squares
    fn add(&mut self, color: Color, kind: PieceKind, attacks: u64, rich_attack_state: bool) {
        if rich_attack_state {
            self.attacked_twice[color.index()] |= self.all[color.index()] & attacks;
        }
        self.by_kind[color.index()][kind.index()] |= attacks;
        self.all[color.index()] |= attacks;
    }

    /// Used for querying whether `square` is geometrically attacked by `by`.
    ///
    /// # Arguments
    ///
    /// * `square` - square to test
    /// * `by` - attacking color
    ///
    /// # Returns
    ///
    /// `true` when the color's attack union covers the square.
    fn attacks(&self, square: Square, by: Color) -> bool {
        self.all[by.index()] & square.bit() != 0
    }
}

/// Individual white-relative terms of the classical evaluation.
///
/// Each field is a centipawn contribution from White's perspective; a
/// positive value favors White. [`ClassicalBreakdown::white_score`] sums the
/// terms, and tests compare individual fields to keep every term
/// independently reviewable.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClassicalBreakdown {
    /// Used for holding the material balance.
    pub material: i32,
    /// Used for holding the tapered piece-placement term.
    pub piece_square: i32,
    /// Used for holding the doubled, isolated, and passed-pawn terms.
    pub pawns: i32,
    /// Used for holding the bishop-pair term.
    pub bishops: i32,
    /// Used for holding the open- and semi-open-file rook terms.
    pub rooks: i32,
    /// Used for holding the king placement and shelter term.
    pub kings: i32,
    /// Used for holding the safe piece mobility available to both armies.
    pub mobility: i32,
    /// Used for holding outposts, minor-piece quality, and heavy-piece
    /// activity.
    ///
    /// Zero when the compact attack state is selected.
    pub activity: i32,
    /// Used for holding attacked, weak, hanging, and pawn-threatened
    /// material.
    pub threats: i32,
    /// Used for holding weighted pressure against king rings and flanks.
    ///
    /// Zero when the compact attack state is selected.
    pub king_pressure: i32,
    /// Used for holding the side-to-move initiative.
    pub tempo: i32,
}

impl ClassicalBreakdown {
    /// Used for combining every term into the score from White's
    /// perspective.
    ///
    /// # Returns
    ///
    /// Sum of all breakdown terms in centipawns; positive favors White.
    #[must_use]
    pub const fn white_score(self) -> i32 {
        self.material
            + self.piece_square
            + self.pawns
            + self.bishops
            + self.rooks
            + self.kings
            + self.mobility
            + self.activity
            + self.threats
            + self.king_pressure
            + self.tempo
    }
}

/// Stateless tapered classical evaluator.
///
/// Combines material, placement, pawn-structure, bishop-pair, rook-file,
/// king-shelter, mobility, attack-state, and tempo terms; the placement,
/// king-shelter, mobility, and attack-state terms are tapered between
/// middlegame and endgame weights by the remaining non-pawn material.
/// Implements both [`Evaluator`] and [`SearchEvaluator`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Classical;

impl Classical {
    /// Used for evaluating a position and exposing its independently
    /// testable terms.
    ///
    /// Always evaluates the released attack-state flavor selected by
    /// `RELEASE_RICH_ATTACK_STATE`, including the promoted STR-280
    /// king-danger scorer confirmed on 2026-08-05.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// White-relative decomposition of the classical evaluation.
    #[must_use]
    pub fn breakdown(position: &Position) -> ClassicalBreakdown {
        let mut result =
            Self::breakdown_with_attack_state(position, RELEASE_RICH_ATTACK_STATE, true, false);
        let (connected_pawns, king_pawn_file) = promoted_capacity_delta(position);
        result.pawns = result.pawns.saturating_add(connected_pawns);
        result.kings = result.kings.saturating_add(king_pawn_file);
        result
    }

    /// Used for evaluating either the frozen compact HCE or the released
    /// rich attack state.
    ///
    /// This entry point exists for deterministic engine research. `false`
    /// selects the compact evaluator frozen before the attack-state experiment;
    /// `true` selects the rich path measured by that experiment while keeping
    /// every material, placement, pawn, bishop-pair, rook-file, king-shelter,
    /// mobility, and tempo term identical. Both arms deliberately keep the
    /// pre-STR-280 king-pressure term so the frozen contrast still isolates
    /// the attack state alone; neither arm is production since STR-280 was
    /// promoted. The ordinary engine uses [`Self::breakdown`].
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `rich_attack_state` - `true` for the released rich flavor, `false`
    ///   for the frozen compact flavor
    ///
    /// # Returns
    ///
    /// White-relative decomposition for the selected attack-state flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_breakdown(position: &Position, rich_attack_state: bool) -> ClassicalBreakdown {
        Self::breakdown_with_attack_state(position, rich_attack_state, false, false)
    }

    /// Used for creating an evaluator for paired baseline/candidate HCE
    /// experiments.
    ///
    /// The returned concrete type is intentionally opaque so research controls
    /// do not become a configurable UCI surface. `false` is the frozen compact
    /// attack term; `true` enables the richer shared attack-state slice.
    ///
    /// # Arguments
    ///
    /// * `rich_attack_state` - attack-state flavor the evaluator uses
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the selected attack-state flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_attack_state(rich_attack_state: bool) -> impl SearchEvaluator + Clone {
        ResearchClassical {
            rich_attack_state,
            king_danger_candidate: false,
            passed_pawn_candidate: false,
        }
    }

    /// Used for comparing the released and STR-280 king-danger scorers.
    ///
    /// Both flavors retain the released rich attack state and differ only in
    /// the final king-pressure term. This keeps the static audit and later
    /// search A/B isolated from every other HCE term.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to apply the fitted STR-280 scorer
    ///
    /// # Returns
    ///
    /// White-relative decomposition for the selected king-danger flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_king_danger_breakdown(
        position: &Position,
        candidate: bool,
    ) -> ClassicalBreakdown {
        Self::breakdown_with_attack_state(position, true, candidate, false)
    }

    /// Used for creating a search evaluator for the STR-280 A/B.
    ///
    /// # Arguments
    ///
    /// * `candidate` - whether to apply the fitted king-danger scorer
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested king-danger flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_king_danger(candidate: bool) -> impl SearchEvaluator + Clone {
        ResearchClassical {
            rich_attack_state: true,
            king_danger_candidate: candidate,
            passed_pawn_candidate: false,
        }
    }

    /// Used for evaluating one STR-280 flavor through the exact production
    /// score-calibration path.
    ///
    /// This research entry point exists so a separately built UCI binary can
    /// exercise the fitted candidate without adding a mutable protocol option
    /// or duplicating [`side_to_move_score`] outside this module. Ordinary
    /// builds continue to call [`Self::evaluate_position`].
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to apply the fitted king-danger scorer
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_king_danger_evaluate_position(position: &Position, candidate: bool) -> i32 {
        let white_score = Self::research_king_danger_breakdown(position, candidate).white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for comparing STR-280 with STR-304's pawn-file proximity term.
    ///
    /// Both arms use the fitted STR-280 king-danger scorer. The candidate arm
    /// adds only the bounded, phase-tapered nearest-pawn-file balance to the
    /// king-placement field, leaving every attack map and existing term exact.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to add the fitted proximity correction
    ///
    /// # Returns
    ///
    /// White-relative decomposition for the selected STR-304 flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_king_pawn_proximity_breakdown(
        position: &Position,
        candidate: bool,
    ) -> ClassicalBreakdown {
        let mut breakdown = Self::research_king_danger_breakdown(position, true);
        if candidate {
            breakdown.kings = breakdown
                .kings
                .saturating_add(king_pawn_file_proximity_term(
                    position,
                    game_phase(position),
                ));
        }
        breakdown
    }

    /// Used for creating a search evaluator for the STR-304 A/B.
    ///
    /// # Arguments
    ///
    /// * `candidate` - whether to add the fitted proximity correction
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested pawn-file proximity flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_king_pawn_proximity(candidate: bool) -> impl SearchEvaluator + Clone {
        ResearchKingPawnProximity { candidate }
    }

    /// Used for evaluating one STR-304 flavor through production calibration.
    ///
    /// This research-only entry point composes STR-304 on STR-280 without
    /// changing [`Self::evaluate_position`] or adding a mutable UCI option.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to add the fitted proximity correction
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_king_pawn_proximity_evaluate_position(
        position: &Position,
        candidate: bool,
    ) -> i32 {
        let white_score =
            Self::research_king_pawn_proximity_breakdown(position, candidate).white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for extracting named king-danger inputs without changing play.
    ///
    /// This diagnostic is deliberately computed from the same [`AttackMaps`]
    /// and helper path as the released evaluation. It exists for fitting and
    /// auditing STR-280; production scoring continues to use
    /// [`Self::breakdown`] until a candidate passes its declared gates.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose two kings are inspected
    ///
    /// # Returns
    ///
    /// White-relative feature vector plus the exact release phase.
    #[doc(hidden)]
    #[must_use]
    pub fn research_king_danger_features(position: &Position) -> KingDangerFeatures {
        let phase = game_phase(position);
        let attacks = AttackMaps::build(position, true);
        king_danger_features(position, &attacks, phase)
    }

    /// Used for extracting named passed-pawn realization inputs without
    /// changing play.
    ///
    /// This diagnostic shares [`is_passed`] and the exact released passer
    /// bonus scan with [`pawn_structure`]. The additional geometry is derived
    /// from the already established rich [`AttackMaps`]; production scoring
    /// remains unchanged while STR-293 is in its static-preparation stage.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose true passed pawns are inspected
    ///
    /// # Returns
    ///
    /// White-relative feature vector plus the exact release phase.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_features(position: &Position) -> PassedPawnFeatures {
        let phase = game_phase(position);
        let attacks = AttackMaps::build(position, true);
        passed_pawn_features(position, &attacks, phase)
    }

    /// Used for auditing the frozen passed-pawn quadratic coefficient.
    ///
    /// This diagnostic performs no independent passer recognition. It consumes
    /// the color, rank, released bonus, and blocker state emitted by the one
    /// production traversal, then replays the fixed coefficient grid `0..=16`.
    /// Normal evaluation never calls this path.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose true passed pawns are inspected
    ///
    /// # Returns
    ///
    /// Exact released bonus, phase, and White-relative delta for each grid
    /// coefficient.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_curve_features(position: &Position) -> PassedPawnCurveFeatures {
        passed_pawn_curve_features(position, game_phase(position))
    }

    /// Used for comparing the released and STR-295 passed-pawn state
    /// scorers.
    ///
    /// Both flavors retain the released rich attack state and king-pressure
    /// scorer. The candidate differs only by adding the fitted, bounded
    /// realization delta to the existing pawn term.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to apply the fitted STR-295 passer-state delta
    ///
    /// # Returns
    ///
    /// White-relative decomposition for the selected passed-pawn flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_breakdown(
        position: &Position,
        candidate: bool,
    ) -> ClassicalBreakdown {
        Self::breakdown_with_attack_state(position, true, false, candidate)
    }

    /// Used for creating a search evaluator for the STR-295 A/B.
    ///
    /// # Arguments
    ///
    /// * `candidate` - whether to apply the fitted realization delta
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested passed-pawn flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawns(candidate: bool) -> impl SearchEvaluator + Clone {
        ResearchClassical {
            rich_attack_state: true,
            king_danger_candidate: false,
            passed_pawn_candidate: candidate,
        }
    }

    /// Used for evaluating one STR-295 flavor through the production score
    /// calibration path.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to apply the fitted realization delta
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_evaluate_position(position: &Position, candidate: bool) -> i32 {
        let white_score = Self::research_passed_pawn_breakdown(position, candidate).white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for comparing STR-280 with and without STR-297's mobility tables.
    ///
    /// Both arms use the fitted STR-280 king-danger scorer. The candidate arm
    /// changes only the compile-time mobility-table selection, leaving every
    /// attack mask, destination count, phase rule, and scoring constant
    /// identical. The existing king-pressure formula deliberately consumes
    /// the MG mobility edge, so its induced downstream delta remains part of
    /// this mobility-family experiment.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to use the trace-calibrated mobility tables
    ///
    /// # Returns
    ///
    /// White-relative decomposition for the selected STR-297 flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_mobility_breakdown(position: &Position, candidate: bool) -> ClassicalBreakdown {
        if candidate {
            Self::breakdown_with_mobility::<true>(position, true, true, false)
        } else {
            Self::breakdown_with_mobility::<false>(position, true, true, false)
        }
    }

    /// Used for creating a search evaluator for the STR-297 A/B.
    ///
    /// # Arguments
    ///
    /// * `candidate` - whether to use the trace-calibrated mobility tables
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested mobility flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_mobility(candidate: bool) -> impl SearchEvaluator + Clone {
        ResearchMobility { candidate }
    }

    /// Used for evaluating one STR-297 flavor through production score
    /// calibration.
    ///
    /// This research-only entry point composes STR-297 on STR-280 without
    /// changing [`Self::evaluate_position`] or exposing a mutable UCI option.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to use the trace-calibrated mobility tables
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_mobility_evaluate_position(position: &Position, candidate: bool) -> i32 {
        let white_score = Self::research_mobility_breakdown(position, candidate).white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for comparing STR-280 with and without STR-298's threat weights.
    ///
    /// Both arms use the fitted STR-280 king-danger scorer and the released
    /// mobility tables. The const-specialized candidate changes only the
    /// baked weights consumed by the existing rich threat traversal.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to use the trace-calibrated threat weights
    ///
    /// # Returns
    ///
    /// White-relative decomposition for the selected STR-298 flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_threat_breakdown(position: &Position, candidate: bool) -> ClassicalBreakdown {
        if candidate {
            Self::breakdown_with_research::<false, true, false, false, false>(
                position, true, true, false,
            )
        } else {
            Self::breakdown_with_research::<false, false, false, false, false>(
                position, true, true, false,
            )
        }
    }

    /// Used for creating a search evaluator for the STR-298 A/B.
    ///
    /// # Arguments
    ///
    /// * `candidate` - whether to use the trace-calibrated threat weights
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested threat flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_threats(candidate: bool) -> impl SearchEvaluator + Clone {
        ResearchThreats { candidate }
    }

    /// Used for evaluating one STR-298 flavor through production score
    /// calibration.
    ///
    /// This research-only entry point composes STR-298 on STR-280 without
    /// changing [`Self::evaluate_position`] or adding a mutable UCI option.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to use the trace-calibrated threat weights
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_threat_evaluate_position(position: &Position, candidate: bool) -> i32 {
        let white_score = Self::research_threat_breakdown(position, candidate).white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for comparing STR-280 with and without STR-299's doubled-pawn weight.
    ///
    /// Both arms use the fitted STR-280 king-danger scorer and released
    /// mobility/threat weights. The compile-time candidate changes only the
    /// immediate multiplied by each existing per-file extra-pawn count.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to use the trace-calibrated doubled-pawn weight
    ///
    /// # Returns
    ///
    /// White-relative decomposition for the selected STR-299 flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_pawn_structure_breakdown(
        position: &Position,
        candidate: bool,
    ) -> ClassicalBreakdown {
        if candidate {
            Self::breakdown_with_research::<false, false, true, false, false>(
                position, true, true, false,
            )
        } else {
            Self::breakdown_with_research::<false, false, false, false, false>(
                position, true, true, false,
            )
        }
    }

    /// Used for creating a search evaluator for the STR-299 A/B.
    ///
    /// # Arguments
    ///
    /// * `candidate` - whether to use the trace-calibrated doubled-pawn weight
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested pawn-structure flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_pawn_structure(candidate: bool) -> impl SearchEvaluator + Clone {
        ResearchPawnStructure { candidate }
    }

    /// Used for evaluating one STR-299 flavor through production score calibration.
    ///
    /// This research-only entry point composes STR-299 on STR-280 without
    /// changing [`Self::evaluate_position`] or exposing a mutable UCI option.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to use the trace-calibrated doubled-pawn weight
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_pawn_structure_evaluate_position(position: &Position, candidate: bool) -> i32 {
        let white_score =
            Self::research_pawn_structure_breakdown(position, candidate).white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for comparing STR-280 with STR-306's passed-pawn rank curve.
    ///
    /// Both arms use the fitted STR-280 king-danger scorer. Const
    /// specialization changes only the quadratic multiplication immediate in
    /// the shared true-passer traversal; every other pawn and evaluation term
    /// is identical.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to use the trace-calibrated quadratic
    ///
    /// # Returns
    ///
    /// White-relative decomposition for the selected STR-306 flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_curve_breakdown(
        position: &Position,
        candidate: bool,
    ) -> ClassicalBreakdown {
        if candidate {
            Self::breakdown_with_research::<false, false, false, true, false>(
                position, true, true, false,
            )
        } else {
            Self::breakdown_with_research::<false, false, false, false, false>(
                position, true, true, false,
            )
        }
    }

    /// Used for evaluating one STR-306 flavor through production calibration.
    ///
    /// This immutable research entry point composes the candidate on STR-280
    /// without adding a mutable UCI option or changing [`Self::evaluate_position`].
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to use the trace-calibrated quadratic
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_curve_evaluate_position(
        position: &Position,
        candidate: bool,
    ) -> i32 {
        let white_score =
            Self::research_passed_pawn_curve_breakdown(position, candidate).white_score();
        side_to_move_score(position, white_score)
    }

    /// Used by the immutable STR-306 UCI build without a runtime flavor flag.
    ///
    /// The direct const-specialized call ensures optimized candidate play has
    /// the same passer-loop control flow as release evaluation and changes
    /// only the squared-advance multiplication immediate.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score with STR-280 and coefficient two enabled.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_curve_candidate_evaluate_position(position: &Position) -> i32 {
        let white_score = Self::breakdown_with_research::<false, false, false, true, false>(
            position, true, true, false,
        )
        .white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for creating a const-specialized STR-306 search evaluator.
    ///
    /// # Returns
    ///
    /// Opaque search evaluator with STR-280 and coefficient two enabled.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_curve_candidate() -> impl SearchEvaluator + Clone {
        ResearchPassedPawnCurve
    }

    /// Used by the STR-306 baseline build without a runtime flavor flag.
    ///
    /// This is the exact const-specialized STR-280 plus released-coefficient
    /// counterpart to
    /// [`Self::research_passed_pawn_curve_candidate_evaluate_position`].
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score with STR-280 and coefficient five enabled.
    #[doc(hidden)]
    #[must_use]
    pub fn research_passed_pawn_curve_baseline_evaluate_position(position: &Position) -> i32 {
        let white_score = Self::breakdown_with_research::<false, false, false, false, false>(
            position, true, true, false,
        )
        .white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for extracting STR-296's constant-time endgame-complexity inputs.
    ///
    /// The caller supplies the exact white-relative score being extended so
    /// the emitted sign and capped delta cannot drift from the composed
    /// STR-280 baseline. Missing kings make both king-geometry fields zero;
    /// all material-only fields remain available for bounded diagnostic FENs.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose material and king geometry are inspected
    /// * `pre_initiative_white_score` - white-relative score before STR-296
    ///
    /// # Returns
    ///
    /// Color-invariant raw values plus the exact white-relative candidate
    /// delta and release phase.
    #[doc(hidden)]
    #[must_use]
    pub fn research_initiative_features(
        position: &Position,
        pre_initiative_white_score: i32,
    ) -> InitiativeFeatures {
        initiative_features(position, pre_initiative_white_score, game_phase(position))
    }

    /// Used for creating a search evaluator for the STR-296 A/B.
    ///
    /// Both arms include STR-280's fitted king-danger scorer. The candidate
    /// arm alone adds the fitted initiative delta, keeping the eventual
    /// composition boundary explicit and leaving the release evaluator
    /// unchanged.
    ///
    /// # Arguments
    ///
    /// * `candidate` - whether to apply the fitted initiative correction
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested initiative flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_initiative(candidate: bool) -> impl SearchEvaluator + Clone {
        ResearchInitiative { candidate }
    }

    /// Used for creating a search evaluator that composes STR-304 with
    /// STR-319.
    ///
    /// The two survivors of the 2026-08-05 screening batch touch unrelated
    /// geometry — king-to-nearest-pawn-file distance and pawn connection — so
    /// they are expected to be close to additive. That expectation is exactly
    /// what this evaluator exists to measure: the campaign's own additive audit
    /// already found STR-298 and STR-299 failing to compose, so composition is
    /// tested rather than assumed.
    ///
    /// # Arguments
    ///
    /// * `connected_scale_percent` - connected-pawn ramp percentage; `0`
    ///   disables that feature
    /// * `king_pawn_proximity` - whether to add the STR-304 king term
    /// * `pawn_count_material` - whether to add the STR-321 material term
    /// * `per_king_danger` - whether to convert king danger per defender
    /// * `pawnless_scaling` - whether to scale pawnless drawn endings
    /// * `backward_pawns` - whether to add the STR-323 backward penalty
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested composition.
    #[doc(hidden)]
    #[must_use]
    pub fn research_extra_features(
        backward_penalty: (i32, i32),
        pawn_count_material: bool,
        per_king_danger: bool,
        pawnless_scaling: bool,
    ) -> impl SearchEvaluator + Clone {
        ResearchComposedFeatures {
            backward_penalty,
            pawn_count_material,
            per_king_danger,
            pawnless_scaling,
        }
    }

    /// Used for evaluating one composed-feature flavor through the production
    /// score-calibration path.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `connected_scale_percent` - connected-pawn ramp percentage
    /// * `king_pawn_proximity` - whether to add the STR-304 king term
    /// * `pawn_count_material` - whether to add the STR-321 material term
    /// * `per_king_danger` - whether to convert king danger per defender
    /// * `pawnless_scaling` - whether to scale pawnless drawn endings
    /// * `backward_pawns` - whether to add the STR-323 backward penalty
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_extra_features_evaluate_position(
        position: &Position,
        backward_penalty: (i32, i32),
        pawn_count_material: bool,
        per_king_danger: bool,
        pawnless_scaling: bool,
    ) -> i32 {
        let mut breakdown = if per_king_danger {
            Self::breakdown_with_research::<false, false, false, false, true>(
                position,
                RELEASE_RICH_ATTACK_STATE,
                true,
                false,
            )
        } else {
            Self::breakdown_with_research::<false, false, false, false, false>(
                position,
                RELEASE_RICH_ATTACK_STATE,
                true,
                false,
            )
        };
        let (connected_pawns, king_pawn_file) = promoted_capacity_delta(position);
        breakdown.pawns = breakdown.pawns.saturating_add(connected_pawns);
        breakdown.kings = breakdown.kings.saturating_add(king_pawn_file);
        let phase = game_phase(position);
        if backward_penalty != (0, 0) {
            breakdown.pawns = breakdown.pawns.saturating_add(backward_pawn_term(
                position,
                phase,
                backward_penalty,
            ));
        }
        if pawn_count_material {
            breakdown.material = breakdown
                .material
                .saturating_add(pawn_count_material_adjustment(position));
        }
        let mut white_score = breakdown.white_score();
        if pawnless_scaling {
            white_score = white_score * pawnless_scale_percent(position, white_score) / 100;
        }
        side_to_move_score(position, white_score)
    }

    /// Used for evaluating one candidate parameter vector through the
    /// parameterized twin.
    ///
    /// Exists so a refitted vector can be screened without transcribing
    /// hundreds of constants into `classical.rs` first. The twin reproduces
    /// [`Self::breakdown`] bit-for-bit at the default parameters and already
    /// carries the promoted capacity terms as frozen constants, so the only
    /// difference a screen measures is the parameter change itself. It is a
    /// research path: the twin is slower than the released scorer, so the
    /// reported throughput ratio is meaningless here and the cost must be
    /// measured separately after any transcription.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `params` - candidate parameter vector
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_params_evaluate_position(
        position: &Position,
        params: &tuning::ClassicalParams,
    ) -> i32 {
        let white_score = tuning::params_breakdown(position, params).white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for creating a search evaluator over a candidate parameter vector.
    ///
    /// # Arguments
    ///
    /// * `params` - candidate parameter vector
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the supplied parameters.
    #[doc(hidden)]
    #[must_use]
    pub fn research_params(params: tuning::ClassicalParams) -> impl SearchEvaluator + Clone {
        ResearchParams { params }
    }

    /// Used for creating a search evaluator for the STR-319 A/B.
    ///
    /// The baseline arm is exactly the promoted release. The candidate adds
    /// [`connected_pawn_term`] to the released pawn term and changes nothing
    /// else, so the contrast isolates one added feature rather than a retune
    /// of an existing one.
    ///
    /// # Arguments
    ///
    /// * `scale_percent` - ramp percentage; `0` selects the baseline
    ///
    /// # Returns
    ///
    /// Search evaluator locked to the requested connected-pawn flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_connected_pawns(scale_percent: i32) -> impl SearchEvaluator + Clone {
        ResearchConnectedPawns { scale_percent }
    }

    /// Used for exposing the released tapering phase to offline research
    /// tools.
    ///
    /// Piece-square fitting has to build its features in exactly the phase the
    /// evaluator will taper them with. Re-deriving the rule in a fitting tool
    /// would create a second definition that could drift silently, so the one
    /// definition is published here instead.
    ///
    /// # Arguments
    ///
    /// * `position` - position whose phase is requested
    ///
    /// # Returns
    ///
    /// Phase in `0..=`[`Classical::RESEARCH_MAX_PHASE`], where the maximum is
    /// a full opening army.
    #[doc(hidden)]
    #[must_use]
    pub fn research_phase(position: &Position) -> i32 {
        game_phase(position)
    }

    /// Used for publishing the tapering phase ceiling alongside
    /// [`Classical::research_phase`].
    #[doc(hidden)]
    pub const RESEARCH_MAX_PHASE: i32 = MAX_PHASE;

    /// Used for constructing the `STR-20260806-334` piece-square correction
    /// flavor.
    ///
    /// # Arguments
    ///
    /// * `delta` - fitted per-kind, per-relative-square tapered correction
    ///
    /// # Returns
    ///
    /// A cloneable evaluator adding `delta` to the released piece-square
    /// component. An all-zero table reproduces the release exactly.
    #[doc(hidden)]
    #[must_use]
    pub fn research_piece_square_delta(
        delta: Arc<PieceSquareDelta>,
    ) -> impl SearchEvaluator + Clone {
        ResearchPieceSquareDelta { delta }
    }

    /// Used for evaluating one `STR-20260806-334` flavor through the
    /// production score calibration path.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `delta` - fitted piece-square correction
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_piece_square_delta_evaluate_position(
        position: &Position,
        delta: &PieceSquareDelta,
    ) -> i32 {
        let mut breakdown = Self::breakdown(position);
        breakdown.piece_square = breakdown
            .piece_square
            .saturating_add(piece_square_delta_term(
                position,
                game_phase(position),
                delta,
            ));
        side_to_move_score(position, breakdown.white_score())
    }

    /// Used for constructing one `STR-20260806-332` bad-bishop flavor.
    ///
    /// # Arguments
    ///
    /// * `scale_percent` - ramp percentage; `0` selects the exact released
    ///   baseline
    ///
    /// # Returns
    ///
    /// A cloneable evaluator over the selected flavor.
    #[doc(hidden)]
    #[must_use]
    pub fn research_bishop_pawns(scale_percent: i32) -> impl SearchEvaluator + Clone {
        ResearchBishopPawns { scale_percent }
    }

    /// Used for evaluating one `STR-20260806-332` flavor through the production
    /// score calibration path.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `scale_percent` - ramp percentage; `0` selects the baseline
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_bishop_pawns_evaluate_position(position: &Position, scale_percent: i32) -> i32 {
        let mut breakdown = Self::breakdown(position);
        if scale_percent != 0 {
            breakdown.bishops = breakdown.bishops.saturating_add(bishop_pawn_term(
                position,
                game_phase(position),
                scale_percent,
            ));
        }
        side_to_move_score(position, breakdown.white_score())
    }

    /// Used for evaluating one STR-319 flavor through the production score
    /// calibration path.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `scale_percent` - ramp percentage; `0` selects the baseline
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_connected_pawns_evaluate_position(
        position: &Position,
        scale_percent: i32,
    ) -> i32 {
        let mut breakdown =
            Self::breakdown_with_attack_state(position, RELEASE_RICH_ATTACK_STATE, true, false);
        if scale_percent != 0 {
            breakdown.pawns = breakdown.pawns.saturating_add(connected_pawn_term(
                position,
                game_phase(position),
                scale_percent,
            ));
        }
        side_to_move_score(position, breakdown.white_score())
    }

    /// Used for evaluating one STR-296 flavor through the production score
    /// calibration path.
    ///
    /// Both flavors start from the STR-280 candidate breakdown. This function
    /// is kept separate from [`Self::evaluate_position`] so a research binary
    /// cannot silently enable the candidate in release play.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to apply the fitted initiative correction
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score from the side-to-move perspective.
    #[doc(hidden)]
    #[must_use]
    pub fn research_initiative_evaluate_position(position: &Position, candidate: bool) -> i32 {
        side_to_move_score(
            position,
            Self::research_initiative_white_score(position, candidate),
        )
    }

    /// Used for composing STR-280 and STR-296 before score calibration.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `candidate` - whether to add the initiative correction
    ///
    /// # Returns
    ///
    /// White-relative STR-280 score, optionally extended by STR-296.
    fn research_initiative_white_score(position: &Position, candidate: bool) -> i32 {
        let white_score =
            Self::breakdown_with_attack_state(position, true, true, false).white_score();
        if !candidate {
            return white_score;
        }
        let features = initiative_features(position, white_score, game_phase(position));
        white_score.saturating_add(features.candidate_delta)
    }

    /// Used for building the exact white-relative decomposition for one
    /// attack-state flavor.
    ///
    /// A single board scan accumulates material, tapered placement, per-file
    /// pawn counts, and bishop counts; the structural terms are then derived
    /// from those totals, and the mobility and attack-state terms from
    /// freshly built attack maps. The rich flavor
    /// adds activity, rich threats, and king pressure; the compact flavor
    /// uses the frozen compact threat term instead.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `rich_attack_state` - whether the rich attack-state terms are
    ///   computed
    /// * `king_danger_candidate` - whether rich king pressure uses STR-280
    /// * `passed_pawn_candidate` - whether the pawn term adds STR-295
    ///
    /// # Returns
    ///
    /// White-relative decomposition of every classical term.
    fn breakdown_with_attack_state(
        position: &Position,
        rich_attack_state: bool,
        king_danger_candidate: bool,
        passed_pawn_candidate: bool,
    ) -> ClassicalBreakdown {
        Self::breakdown_with_research::<false, false, false, false, false>(
            position,
            rich_attack_state,
            king_danger_candidate,
            passed_pawn_candidate,
        )
    }

    /// Used for building a decomposition with one compile-time mobility-table
    /// flavor.
    ///
    /// Every term except mobility receives the same inputs in both
    /// instantiations. The const parameter is consumed only by
    /// [`AttackMaps::build_with_mobility`], keeping STR-297 isolated to its
    /// materialized lookup constants.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `rich_attack_state` - whether the rich attack-state terms are
    ///   computed
    /// * `king_danger_candidate` - whether rich king pressure uses STR-280
    /// * `passed_pawn_candidate` - whether the pawn term adds STR-295
    ///
    /// # Returns
    ///
    /// White-relative decomposition of every classical term.
    fn breakdown_with_mobility<const CANDIDATE_MOBILITY: bool>(
        position: &Position,
        rich_attack_state: bool,
        king_danger_candidate: bool,
        passed_pawn_candidate: bool,
    ) -> ClassicalBreakdown {
        Self::breakdown_with_research::<CANDIDATE_MOBILITY, false, false, false, false>(
            position,
            rich_attack_state,
            king_danger_candidate,
            passed_pawn_candidate,
        )
    }

    /// Used for building a decomposition with compile-time research flavors.
    ///
    /// The mobility parameter selects only [`AttackMaps`] lookup constants;
    /// the threat parameter selects only weights in [`rich_threat_term`]; and
    /// the pawn-structure parameter selects only the doubled-pawn immediate in
    /// [`pawn_structure`]; and the passed-curve parameter selects only its
    /// squared-advance immediate. All four are false in release evaluation, so
    /// rejected research candidates cannot become runtime configuration.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    /// * `rich_attack_state` - whether the rich attack-state terms are
    ///   computed
    /// * `king_danger_candidate` - whether rich king pressure uses STR-280
    /// * `passed_pawn_candidate` - whether the pawn term adds STR-295
    ///
    /// # Returns
    ///
    /// White-relative decomposition of every classical term.
    fn breakdown_with_research<
        const CANDIDATE_MOBILITY: bool,
        const CANDIDATE_THREATS: bool,
        const CANDIDATE_PAWN_STRUCTURE: bool,
        const CANDIDATE_PASSED_CURVE: bool,
        const PER_KING_DANGER: bool,
    >(
        position: &Position,
        rich_attack_state: bool,
        king_danger_candidate: bool,
        passed_pawn_candidate: bool,
    ) -> ClassicalBreakdown {
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
                result.material += sign
                    * MATERIAL[kind.index()]
                    * i32::try_from(pieces.count_ones()).expect("piece count fits i32");
                if kind == PieceKind::Bishop {
                    bishops[color.index()] =
                        u8::try_from(pieces.count_ones()).expect("bishop count fits u8");
                }
                while pieces != 0 {
                    let index =
                        u8::try_from(pieces.trailing_zeros()).expect("set bit index is below 64");
                    pieces &= pieces - 1;
                    let square = board_square(index);
                    result.piece_square += sign * tapered_piece_square(piece, square, phase);
                    if kind == PieceKind::Pawn {
                        file_pawns[color.index()][usize::from(square.file())] += 1;
                    }
                }
            }
        }

        let attacks =
            AttackMaps::build_with_mobility::<CANDIDATE_MOBILITY>(position, rich_attack_state);
        result.pawns = if passed_pawn_candidate {
            let (released, features) =
                pawn_structure_and_passed_candidate_features(position, file_pawns, &attacks, phase);
            released.saturating_add(passed_pawn_fast_candidate_delta(features))
        } else {
            pawn_structure::<CANDIDATE_PAWN_STRUCTURE, CANDIDATE_PASSED_CURVE>(position, file_pawns)
        };
        result.bishops = pair_bonus(bishops);
        result.rooks = rook_files(position, file_pawns);
        result.kings = king_term(position, phase);
        result.mobility = mobility_term(&attacks, phase);
        if rich_attack_state {
            result.activity = activity_term(&attacks, phase);
            result.threats = rich_threat_term::<CANDIDATE_THREATS>(position, &attacks, phase);
            result.king_pressure = if PER_KING_DANGER {
                per_king_danger_pressure(position, &attacks, phase)
            } else if king_danger_candidate {
                king_danger_candidate_pressure(king_danger_features(position, &attacks, phase))
            } else {
                king_pressure_term(position, &attacks, phase)
            };
        } else {
            result.threats = compact_threat_term(position, &attacks, phase);
        }
        result.tempo = if position.side_to_move() == Color::White {
            TEMPO
        } else {
            -TEMPO
        };
        result
    }

    /// Used for evaluating from the side-to-move perspective.
    ///
    /// Sums the white-relative breakdown, negates it for Black to move, and
    /// applies the `SEARCH_SCORE_SCALE_PERCENT` calibration and static
    /// clamp via `side_to_move_score`.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated centipawn score, positive when the mover stands better.
    #[must_use]
    pub fn evaluate_position(position: &Position) -> i32 {
        let white_score = Self::breakdown(position).white_score();
        side_to_move_score(position, white_score)
    }
}

/// Research-only evaluator that selects one attack-state implementation.
///
/// Created by [`Classical::research_attack_state`] for paired
/// baseline/candidate experiments; it differs from [`Classical`] only in the
/// attack-state flavor it evaluates.
#[derive(Clone, Copy, Debug)]
struct ResearchClassical {
    /// Used for selecting whether the rich attack-state terms are enabled.
    rich_attack_state: bool,
    /// Used for selecting the fitted STR-280 king-danger scorer.
    king_danger_candidate: bool,
    /// Used for selecting the fitted STR-295 passed-pawn state scorer.
    passed_pawn_candidate: bool,
}

/// Research-only evaluator composing the two 2026-08-05 screening survivors.
///
/// Created only through [`Classical::research_extra_features`]. Each term is
/// added to the promoted release rather than to another candidate's flavour,
/// so a baseline arm stays exactly the release and every contrast is
/// attributable to the one feature it selects.
#[derive(Clone, Copy, Debug)]
struct ResearchComposedFeatures {
    /// Used for the STR-323 backward penalty; `(0, 0)` disables it.
    backward_penalty: (i32, i32),
    /// Used for selecting whether the STR-321 material adjustment is added.
    pawn_count_material: bool,
    /// Used for selecting the STR-325 per-king danger conversion.
    per_king_danger: bool,
    /// Used for selecting the STR-327 pawnless endgame scaling.
    pawnless_scaling: bool,
}

impl SearchEvaluator for ResearchComposedFeatures {
    /// Used for evaluating with the selected composition.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the composed flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_extra_features_evaluate_position(
            position,
            self.backward_penalty,
            self.pawn_count_material,
            self.per_king_danger,
            self.pawnless_scaling,
        )
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Research-only evaluator over an explicit parameter vector.
///
/// Created only through [`Classical::research_params`]; it exists so refitted
/// vectors can be screened before any constant is transcribed.
#[derive(Clone, Debug)]
struct ResearchParams {
    /// Used for the candidate parameter vector under measurement.
    params: tuning::ClassicalParams,
}

impl SearchEvaluator for ResearchParams {
    /// Used for evaluating through the parameterized twin.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the supplied parameters.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_params_evaluate_position(position, &self.params)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Research-only evaluator adding the STR-319 connected-pawn feature.
///
/// Created only through [`Classical::research_connected_pawns`]; the opaque
/// type keeps the added feature outside the mutable UCI surface until a
/// measured result justifies folding it into the released pawn term.
#[derive(Clone, Copy, Debug)]
struct ResearchConnectedPawns {
    /// Used for scaling the connected-pawn ramp; `0` selects the baseline.
    scale_percent: i32,
}

impl SearchEvaluator for ResearchConnectedPawns {
    /// Used for evaluating with the selected STR-319 flavor.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the selected flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_connected_pawns_evaluate_position(position, self.scale_percent)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Fitted per-kind, per-relative-square tapered piece-square correction.
///
/// Indexed by piece kind in `PieceKind::index` order, then by owner-relative
/// square where index zero is the owner's own back rank a-file, then by
/// `(middlegame, endgame)`. The type is a research artifact loaded from a
/// fitted table; the released evaluator does not consult it.
#[doc(hidden)]
pub type PieceSquareDelta = [[(i32, i32); 64]; 6];

/// Used for scoring one position's fitted piece-square correction.
///
/// The two tapering ends are summed across every piece and tapered once,
/// rather than tapering each piece and summing, so the integer result matches
/// the continuous least-squares objective the table was fitted under instead
/// of accumulating one truncation per piece.
///
/// # Arguments
///
/// * `position` - position holding the pieces
/// * `phase` - tapering phase in `0..=MAX_PHASE`
/// * `delta` - fitted correction table
///
/// # Returns
///
/// White-relative correction in centipawns.
fn piece_square_delta_term(position: &Position, phase: i32, delta: &PieceSquareDelta) -> i32 {
    let mut middle = 0;
    let mut ending = 0;
    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        for (kind_index, kind) in [
            PieceKind::Pawn,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
            PieceKind::King,
        ]
        .into_iter()
        .enumerate()
        {
            let mut board = position.piece_bitboard(Piece::new(color, kind));
            while board != 0 {
                let index =
                    usize::try_from(board.trailing_zeros()).expect("set bit index is below 64");
                board &= board - 1;
                let relative = match color {
                    Color::White => index ^ 0b11_1000,
                    Color::Black => index,
                };
                let (piece_middle, piece_ending) = delta[kind_index][relative];
                middle += sign * piece_middle;
                ending += sign * piece_ending;
            }
        }
    }
    tapered_score(middle, ending, phase)
}

/// Research-only evaluator adding the `STR-20260806-334` piece-square
/// correction.
///
/// Created only through [`Classical::research_piece_square_delta`]. The table
/// is shared rather than copied because one match constructs an evaluator per
/// game and the table is three kilobytes.
#[derive(Clone, Debug)]
struct ResearchPieceSquareDelta {
    /// Used for the fitted correction applied on top of the released formula.
    delta: Arc<PieceSquareDelta>,
}

impl SearchEvaluator for ResearchPieceSquareDelta {
    /// Used for evaluating with the fitted piece-square correction.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score including the correction.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_piece_square_delta_evaluate_position(position, &self.delta)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Research-only evaluator adding the `STR-20260806-332` bad-bishop feature.
///
/// Created only through [`Classical::research_bishop_pawns`]. Unlike the
/// earlier research flavors this one starts from [`Classical::breakdown`],
/// which already carries every promoted term, so the contrast measures the
/// added feature against the *current* release rather than against a
/// pre-promotion baseline.
#[derive(Clone, Copy, Debug)]
struct ResearchBishopPawns {
    /// Used for scaling the bad-bishop penalty; `0` selects the baseline.
    scale_percent: i32,
}

impl SearchEvaluator for ResearchBishopPawns {
    /// Used for evaluating with the selected `STR-20260806-332` flavor.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the selected flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_bishop_pawns_evaluate_position(position, self.scale_percent)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Research-only evaluator composing STR-304 with the STR-280 baseline.
///
/// Created only through [`Classical::research_king_pawn_proximity`]; the
/// opaque type keeps the fitted geometry outside the mutable UCI surface.
#[derive(Clone, Copy, Debug)]
struct ResearchKingPawnProximity {
    /// Used for selecting whether the fitted pawn-file correction is applied.
    candidate: bool,
}

impl SearchEvaluator for ResearchKingPawnProximity {
    /// Used for evaluating with the selected STR-304 geometry flavor.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the composed research flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_king_pawn_proximity_evaluate_position(position, self.candidate)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

impl SearchEvaluator for ResearchClassical {
    /// Used for evaluating with the selected attack-state flavor.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the selected flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        let white_score = Classical::breakdown_with_attack_state(
            position,
            self.rich_attack_state,
            self.king_danger_candidate,
            self.passed_pawn_candidate,
        )
        .white_score();
        side_to_move_score(position, white_score)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering
    /// prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from `classical_quiet_move_prior`.
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Research-only evaluator composing STR-297 with the STR-280 baseline.
///
/// Created only through [`Classical::research_mobility`]; the opaque type
/// prevents the static table candidate from becoming a mutable UCI option.
#[derive(Clone, Copy, Debug)]
struct ResearchMobility {
    /// Used for selecting whether the trace-calibrated tables are applied.
    candidate: bool,
}

impl SearchEvaluator for ResearchMobility {
    /// Used for evaluating with the selected STR-297 table flavor.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the composed research flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_mobility_evaluate_position(position, self.candidate)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering
    /// prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Research-only evaluator composing STR-298 with the STR-280 baseline.
///
/// Created only through [`Classical::research_threats`]; the opaque type keeps
/// trace-calibrated weights outside the mutable UCI option surface.
#[derive(Clone, Copy, Debug)]
struct ResearchThreats {
    /// Used for selecting whether the trace-calibrated weights are applied.
    candidate: bool,
}

/// Research-only evaluator composing STR-299 with the STR-280 baseline.
///
/// Created only through [`Classical::research_pawn_structure`]; the opaque
/// type keeps the trace-calibrated weight outside the mutable UCI surface.
#[derive(Clone, Copy, Debug)]
struct ResearchPawnStructure {
    /// Used for selecting whether the reduced doubled-pawn penalty is applied.
    candidate: bool,
}

/// Research-only const-specialized STR-306 search evaluator.
///
/// Created only through [`Classical::research_passed_pawn_curve_candidate`];
/// the unit type exposes no runtime experiment control.
#[derive(Clone, Copy, Debug)]
struct ResearchPassedPawnCurve;

impl SearchEvaluator for ResearchPassedPawnCurve {
    /// Used for evaluating with the fitted passed-pawn quadratic.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the STR-306 candidate.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_passed_pawn_curve_candidate_evaluate_position(position)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

impl SearchEvaluator for ResearchPawnStructure {
    /// Used for evaluating with the selected STR-299 pawn-structure flavor.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the composed research flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_pawn_structure_evaluate_position(position, self.candidate)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

impl SearchEvaluator for ResearchThreats {
    /// Used for evaluating with the selected STR-298 threat flavor.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the composed research flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_threat_evaluate_position(position, self.candidate)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering
    /// prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Research-only evaluator composing STR-296 with the STR-280 baseline.
///
/// Created only through [`Classical::research_initiative`]; the opaque type
/// prevents the static candidate from becoming a mutable UCI option.
#[derive(Clone, Copy, Debug)]
struct ResearchInitiative {
    /// Used for selecting whether the initiative correction is applied.
    candidate: bool,
}

impl SearchEvaluator for ResearchInitiative {
    /// Used for evaluating with the selected STR-296 flavor.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score for the composed research flavor.
    fn evaluate(&mut self, position: &Position) -> i32 {
        Classical::research_initiative_evaluate_position(position, self.candidate)
    }

    /// Used for reusing the release HCE's evaluator-owned quiet ordering
    /// prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from [`classical_quiet_move_prior`].
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

/// Used for converting and calibrating a white-relative HCE score for the
/// side to move.
///
/// Negates the score when Black is to move, scales by
/// [`SEARCH_SCORE_SCALE_PERCENT`], and clamps the result into the static
/// score range.
///
/// # Arguments
///
/// * `position` - position supplying the side to move
/// * `white_score` - white-relative centipawn total
///
/// # Returns
///
/// Clamped, calibrated score relative to the side to move.
fn side_to_move_score(position: &Position, white_score: i32) -> i32 {
    let scaled = white_score * opposite_colored_bishop_scale_percent(position) / 100;
    let relative = if position.side_to_move() == Color::White {
        scaled
    } else {
        -scaled
    };
    clamp_static_score(relative * SEARCH_SCORE_SCALE_PERCENT / 100)
}

/// Used for scaling an opposite-colored bishops endgame toward a draw
/// (HCE-172).
///
/// Returns the percentage by which [`side_to_move_score`] multiplies the
/// white-relative total. The scale drops below `100` only in a pure
/// bishops-and-pawns endgame — no knights, rooks, or queens on the board —
/// where each side has exactly one bishop and the two bishops stand on
/// opposite square colors. Such endings are famously drawish: the defender's
/// bishop guards a fixed color the attacker's bishop can never contest, so a
/// single extra pawn or a placement edge rarely converts.
///
/// The factor starts at [`OPPOSITE_BISHOP_DRAW_FLOOR_PERCENT`] when the pawn
/// count is level (or within one pawn) and climbs by
/// [`OPPOSITE_BISHOP_PAWN_ADVANTAGE_STEP_PERCENT`] for every pawn of advantage
/// beyond the first, reaching a full unscaled `100` once the pawn edge is
/// large enough to win despite the wrong-colored bishop. Because it multiplies
/// the whole score, it is a non-linear scaling that sits outside the linear
/// Texel basis; because it depends only on symmetric material predicates and
/// truncating integer division negates cleanly, army-swap symmetry is
/// preserved.
///
/// # Arguments
///
/// * `position` - position whose material configuration is inspected
///
/// # Returns
///
/// A percentage in `[OPPOSITE_BISHOP_DRAW_FLOOR_PERCENT, 100]`; exactly `100`
/// whenever the opposite-bishop endgame predicate does not hold.
/// Used for scaling down a material edge the stronger side has no pawns to
/// convert.
///
/// This is the most elementary drawing rule in chess and Janus's released
/// evaluator does not express it: with no pawns of its own, a side needs
/// roughly a rook's worth of extra material to force a win, so bishop-versus-
/// knight or rook-versus-rook-and-minor endings are drawn however favourable
/// the running total looks. The released scaling handles only opposite-coloured
/// bishops, so every other pawnless ending is scored at full value and the
/// search happily trades into it.
///
/// The rule is applied to the side the score currently favours, because only
/// that side is trying to win.
///
/// # Arguments
///
/// * `position` - position to classify
/// * `white_score` - white-relative running total in centipawns
///
/// # Returns
///
/// Percentage in `0..=100` applied to the white-relative total.
fn pawnless_scale_percent(position: &Position, white_score: i32) -> i32 {
    if white_score == 0 {
        return 100;
    }
    let stronger = if white_score > 0 {
        Color::White
    } else {
        Color::Black
    };
    if position.piece_bitboard(Piece::new(stronger, PieceKind::Pawn)) != 0 {
        return 100;
    }
    let non_pawn_material = |color: Color| -> i32 {
        [
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
        ]
        .into_iter()
        .map(|kind| {
            let count = i32::try_from(
                position
                    .piece_bitboard(Piece::new(color, kind))
                    .count_ones(),
            )
            .expect("piece count fits i32");
            count * MATERIAL[kind.index()]
        })
        .sum()
    };
    let edge = non_pawn_material(stronger) - non_pawn_material(stronger.opposite());
    if edge >= PAWNLESS_WINNING_EDGE {
        return 100;
    }
    PAWNLESS_DRAW_FLOOR_PERCENT
}

fn opposite_colored_bishop_scale_percent(position: &Position) -> i32 {
    for color in [Color::White, Color::Black] {
        if position.piece_bitboard(Piece::new(color, PieceKind::Knight)) != 0
            || position.piece_bitboard(Piece::new(color, PieceKind::Rook)) != 0
            || position.piece_bitboard(Piece::new(color, PieceKind::Queen)) != 0
        {
            return 100;
        }
    }

    let white_bishops = position.piece_bitboard(Piece::new(Color::White, PieceKind::Bishop));
    let black_bishops = position.piece_bitboard(Piece::new(Color::Black, PieceKind::Bishop));
    if white_bishops.count_ones() != 1 || black_bishops.count_ones() != 1 {
        return 100;
    }
    if (white_bishops & DARK_SQUARE_MASK != 0) == (black_bishops & DARK_SQUARE_MASK != 0) {
        return 100;
    }

    let white_pawns = bit_count(position.piece_bitboard(Piece::new(Color::White, PieceKind::Pawn)));
    let black_pawns = bit_count(position.piece_bitboard(Piece::new(Color::Black, PieceKind::Pawn)));
    let pawn_advantage = (white_pawns - black_pawns).abs();
    let restored = OPPOSITE_BISHOP_PAWN_ADVANTAGE_STEP_PERCENT * (pawn_advantage - 1).max(0);
    (OPPOSITE_BISHOP_DRAW_FLOOR_PERCENT + restored).min(100)
}

impl SearchEvaluator for Classical {
    /// Used for evaluating the released classical HCE inside the search.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score from
    /// [`Classical::evaluate_position`].
    fn evaluate(&mut self, position: &Position) -> i32 {
        Self::evaluate_position(position)
    }

    /// Used for supplying the evaluator-owned quiet move ordering prior.
    ///
    /// # Arguments
    ///
    /// * `position` - position the move is played in
    /// * `mv` - move to rank
    ///
    /// # Returns
    ///
    /// Ordering-only prior from `classical_quiet_move_prior`.
    fn quiet_move_order_prior(&self, position: &Position, mv: Move) -> i32 {
        classical_quiet_move_prior(position, mv)
    }
}

impl Evaluator for Classical {
    /// Used for evaluating the released classical HCE as a plain evaluator.
    ///
    /// # Arguments
    ///
    /// * `position` - position to evaluate
    ///
    /// # Returns
    ///
    /// Calibrated side-to-move score from
    /// [`Classical::evaluate_position`].
    fn evaluate(&mut self, position: &Position) -> i32 {
        Self::evaluate_position(position)
    }
}

/// Used for signing a color's contribution in white-relative sums.
///
/// # Arguments
///
/// * `color` - color whose sign is requested
///
/// # Returns
///
/// `1` for White and `-1` for Black.
const fn color_sign(color: Color) -> i32 {
    match color {
        Color::White => 1,
        Color::Black => -1,
    }
}

/// Used for computing the tapered-evaluation phase from remaining non-pawn
/// material.
///
/// Minors count one, rooks two, and queens four for both colors; the sum is
/// capped at [`MAX_PHASE`], so early promotions cannot push the phase past
/// the opening value.
///
/// # Arguments
///
/// * `position` - position whose material is counted
///
/// # Returns
///
/// Phase in `0..=MAX_PHASE`, where `MAX_PHASE` is a full opening army.
fn game_phase(position: &Position) -> i32 {
    let mut phase = 0;
    for color in [Color::White, Color::Black] {
        phase += bit_count(position.piece_bitboard(Piece::new(color, PieceKind::Knight)));
        phase += bit_count(position.piece_bitboard(Piece::new(color, PieceKind::Bishop)));
        phase += 2 * bit_count(position.piece_bitboard(Piece::new(color, PieceKind::Rook)));
        phase += 4 * bit_count(position.piece_bitboard(Piece::new(color, PieceKind::Queen)));
    }
    phase.min(MAX_PHASE)
}

/// Used for interpolating a compact piece-placement heuristic at `phase`.
///
/// Placement is derived from two formula ingredients rather than tables: a
/// centralization score and the color-relative advance of the square. Each
/// kind mixes them with its own middlegame and endgame weights; knights also
/// take an edge penalty and kings invert centralization in the middlegame.
///
/// # Arguments
///
/// * `piece` - colored piece being placed
/// * `square` - square the piece occupies
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Phase-interpolated placement score for the piece's own color.
fn tapered_piece_square(piece: Piece, square: Square, phase: i32) -> i32 {
    let file = i32::from(square.file());
    let rank = i32::from(square.rank());
    let relative_rank = if piece.color == Color::White {
        rank
    } else {
        9 - rank
    };
    let center = 14 - ((2 * file - 7).abs() + (2 * (rank - 1) - 7).abs());
    let advance = relative_rank - 1;
    let (middle, ending) = match piece.kind {
        PieceKind::Pawn => (advance * 4 - center * 3, advance * 10 - center * 3),
        PieceKind::Knight => (center * 5 - edge_penalty(file, rank), center * 3),
        PieceKind::Bishop => (center * 4 + advance, center * 2),
        // The rook endgame centralization multiplier fitted to zero.
        PieceKind::Rook => (advance * 3, advance * 2),
        PieceKind::Queen => (center * 5, center * 3),
        PieceKind::King => (-center * 8 - advance * 2, center * 5 + advance * 2),
    };
    (middle * phase + ending * (MAX_PHASE - phase)) / MAX_PHASE
}

/// Used for penalizing a knight on an edge, with a larger corner penalty.
///
/// # Arguments
///
/// * `file` - zero-based file in `0..=7`
/// * `rank` - human rank in `1..=8`
///
/// # Returns
///
/// `24` in a corner, `10` on a single edge, otherwise `0`.
const fn edge_penalty(file: i32, rank: i32) -> i32 {
    let on_file_edge = file == 0 || file == 7;
    let on_rank_edge = rank == 1 || rank == 8;
    if on_file_edge && on_rank_edge {
        24
    } else if on_file_edge || on_rank_edge {
        10
    } else {
        0
    }
}

/// Used for computing the white-relative bishop-pair bonus.
///
/// Each side owning two or more bishops earns 39 centipawns.
///
/// # Arguments
///
/// * `bishops` - bishop counts indexed by [`Color::index`]
///
/// # Returns
///
/// White's pair bonus minus Black's pair bonus.
fn pair_bonus(bishops: [u8; 2]) -> i32 {
    let white = i32::from(bishops[Color::White.index()] >= 2) * 39;
    let black = i32::from(bishops[Color::Black.index()] >= 2) * 39;
    white - black
}

/// Used for scoring doubled, isolated, and passed pawns from White's
/// perspective.
///
/// Doubled pawns cost 13 per extra pawn on a file and isolated pawns cost 12
/// per pawn. Passed pawns earn `8 + quadratic * advance^2` by color-relative
/// advance, halved when the square directly in front is occupied.
///
/// # Arguments
///
/// * `position` - position whose pawns are scored
/// * `file_pawns` - per-file pawn counts indexed by [`Color::index`]
///
/// # Returns
///
/// White-relative pawn-structure score.
fn pawn_structure<const CANDIDATE_DOUBLED_PAWNS: bool, const CANDIDATE_PASSED_CURVE: bool>(
    position: &Position,
    file_pawns: [[u8; 8]; 2],
) -> i32 {
    pawn_structure_with_observer::<CANDIDATE_DOUBLED_PAWNS, CANDIDATE_PASSED_CURVE>(
        position,
        file_pawns,
        |_color, _square, _relative_rank, _realized_bonus, _blocked| {},
    )
}

/// Used for scoring the released pawn structure while observing true passers.
///
/// This is the one production passed-pawn traversal. Its no-op observer is
/// compiled into [`pawn_structure`], while research extraction can consume the
/// exact same recognition, rank, blocker, and realized-bonus decisions.
///
/// # Arguments
///
/// * `position` - position whose pawns are scored
/// * `file_pawns` - per-file pawn counts indexed by [`Color::index`]
/// * `observer` - receives color, square, human-relative rank, realized
///   released bonus, and immediate-blocker state for every true passer
///
/// # Returns
///
/// White-relative released pawn-structure score.
fn pawn_structure_with_observer<
    const CANDIDATE_DOUBLED_PAWNS: bool,
    const CANDIDATE_PASSED_CURVE: bool,
>(
    position: &Position,
    file_pawns: [[u8; 8]; 2],
    mut observer: impl FnMut(Color, Square, i32, i32, bool),
) -> i32 {
    let mut score = 0;
    let passed_quadratic = if CANDIDATE_PASSED_CURVE {
        PASSED_PAWN_CANDIDATE_QUADRATIC
    } else {
        PASSED_PAWN_QUADRATIC
    };
    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        let doubled_penalty = if CANDIDATE_DOUBLED_PAWNS {
            PAWN_STRUCTURE_CANDIDATE_DOUBLED_PENALTY
        } else {
            DOUBLED_PAWN_PENALTY
        };
        for file in 0..8 {
            let count = file_pawns[color.index()][file];
            if count > 1 {
                score -= sign * doubled_penalty * i32::from(count - 1);
            }
            if count > 0 {
                let left_empty = file == 0 || file_pawns[color.index()][file - 1] == 0;
                let right_empty = file == 7 || file_pawns[color.index()][file + 1] == 0;
                if left_empty && right_empty {
                    score -= sign * 12 * i32::from(count);
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
                let bonus = 8 + passed_quadratic * advance * advance;
                let push_row = if color == Color::White { -1 } else { 1 };
                let blocked = square_offset(square, 0, push_row)
                    .is_some_and(|front| position.piece_at(front).is_some());
                let realized_bonus = if blocked { bonus / 2 } else { bonus };
                observer(
                    color,
                    square,
                    i32::from(relative_rank),
                    realized_bonus,
                    blocked,
                );
                score += sign * realized_bonus;
            }
        }
    }
    score
}

/// Used for deriving file counts for standalone pawn-feature extraction.
///
/// The normal breakdown obtains these counts during its all-piece scan. The
/// research extractor has no such accumulator, so this bounded pawn-only pass
/// recreates exactly the input required by [`pawn_structure_with_observer`].
///
/// # Arguments
///
/// * `position` - position whose pawns are counted
///
/// # Returns
///
/// Per-color, per-file pawn counts.
fn pawn_file_counts(position: &Position) -> [[u8; 8]; 2] {
    let mut counts = [[0_u8; 8]; 2];
    for color in [Color::White, Color::Black] {
        let mut pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
        while pawns != 0 {
            let index = u8::try_from(pawns.trailing_zeros()).expect("set bit index is below 64");
            pawns &= pawns - 1;
            let square = board_square(index);
            counts[color.index()][usize::from(square.file())] += 1;
        }
    }
    counts
}

/// Used for extracting STR-293 geometry from the released passer traversal.
///
/// # Arguments
///
/// * `position` - position whose true passers are inspected
/// * `attacks` - complete rich attack maps for both armies
/// * `phase` - release material phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative named realization features.
fn passed_pawn_features(
    position: &Position,
    attacks: &AttackMaps,
    phase: i32,
) -> PassedPawnFeatures {
    let file_pawns = pawn_file_counts(position);
    let (_, features) = pawn_structure_and_passed_features(position, file_pawns, attacks, phase);
    features
}

/// Used for replaying the predeclared passed-pawn quadratic grid.
///
/// The observer receives every true passer from the released scorer, so this
/// audit cannot drift in recognition, ordering, relative rank, or immediate
/// blocker ownership. Candidate arithmetic changes only the quadratic
/// coefficient and retains Rust's exact positive-integer division semantics.
///
/// # Arguments
///
/// * `position` - position whose true passers are inspected
/// * `phase` - released material phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Exact released bonus and coefficient-grid deltas.
fn passed_pawn_curve_features(position: &Position, phase: i32) -> PassedPawnCurveFeatures {
    let file_pawns = pawn_file_counts(position);
    let mut features = PassedPawnCurveFeatures {
        phase,
        ..PassedPawnCurveFeatures::default()
    };
    let _ = pawn_structure_with_observer::<false, false>(
        position,
        file_pawns,
        |color, _square, relative_rank, released_bonus, blocked| {
            let sign = color_sign(color);
            features.current_bonus += sign * released_bonus;
            let advance = relative_rank.saturating_sub(1);
            for (quadratic, delta) in features.quadratic_deltas.iter_mut().enumerate() {
                let quadratic = i32::try_from(quadratic).expect("fixed grid index fits i32");
                let bonus = 8 + quadratic * advance * advance;
                let candidate_bonus = if blocked { bonus / 2 } else { bonus };
                *delta += sign * (candidate_bonus - released_bonus);
            }
        },
    );
    features
}

/// Used for scoring the released pawn structure and extracting passer geometry
/// in one true-passer scan.
///
/// This is the intended candidate integration boundary: the returned score is
/// exactly [`pawn_structure`], and the observer accumulates no independent
/// passed-pawn recognition state.
///
/// # Arguments
///
/// * `position` - position whose pawns are scored
/// * `file_pawns` - per-color, per-file pawn counts
/// * `attacks` - complete rich attack maps for both armies
/// * `phase` - release material phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Exact released pawn score and the named research feature vector.
fn pawn_structure_and_passed_features(
    position: &Position,
    file_pawns: [[u8; 8]; 2],
    attacks: &AttackMaps,
    phase: i32,
) -> (i32, PassedPawnFeatures) {
    let mut features = PassedPawnFeatures {
        phase,
        ..PassedPawnFeatures::default()
    };
    let mut context = None;
    let score = pawn_structure_with_observer::<false, false>(
        position,
        file_pawns,
        |color, square, relative_rank, realized_bonus, blocked| {
            accumulate_passed_pawn_features(
                position,
                attacks,
                color,
                square,
                relative_rank,
                realized_bonus,
                blocked,
                &mut context,
                &mut features,
            );
        },
    );
    (score, features)
}

/// Used for scoring released pawns while collecting only STR-295's cheap
/// candidate columns.
///
/// Unlike the complete research extractor, this production-candidate path
/// performs no path, king, rook, or pawn-race work. True-passer recognition,
/// rank, and blocker state still come from [`pawn_structure_with_observer`].
///
/// # Arguments
///
/// * `position` - position whose pawns are scored
/// * `file_pawns` - per-color, per-file pawn counts
/// * `attacks` - already-built attack unions for front-square tests
/// * `phase` - release material phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Exact released pawn score and the five score-bearing candidate features.
fn pawn_structure_and_passed_candidate_features(
    position: &Position,
    file_pawns: [[u8; 8]; 2],
    attacks: &AttackMaps,
    phase: i32,
) -> (i32, PassedPawnCandidateFeatures) {
    let mut features = PassedPawnCandidateFeatures {
        phase,
        ..PassedPawnCandidateFeatures::default()
    };
    let score = pawn_structure_with_observer::<false, false>(
        position,
        file_pawns,
        |color, square, relative_rank, _realized_bonus, blocked| {
            accumulate_passed_pawn_candidate_features(
                attacks,
                color,
                square,
                relative_rank,
                blocked,
                &mut features,
            );
        },
    );
    (score, features)
}

/// Used for adding one passer's low-cost STR-295 state without constructing
/// the complete research geometry.
///
/// # Arguments
///
/// * `attacks` - already-built attack unions for both colors
/// * `color` - color owning the passer
/// * `square` - passer square
/// * `relative_rank` - human-relative rank in `1..=7`
/// * `blocked` - whether the immediate front square is occupied
/// * `features` - cheap candidate accumulator updated in place
fn accumulate_passed_pawn_candidate_features(
    attacks: &AttackMaps,
    color: Color,
    square: Square,
    relative_rank: i32,
    blocked: bool,
    features: &mut PassedPawnCandidateFeatures,
) {
    let urgency = (relative_rank - 3).max(0);
    if urgency == 0 {
        return;
    }
    let front_index = match color {
        Color::White => square.index().checked_sub(8),
        Color::Black => square.index().checked_add(8).filter(|index| *index < 64),
    };
    let Some(front_index) = front_index else {
        return;
    };
    let sign = color_sign(color);
    let rank_weight = urgency * urgency;
    let front = board_square(front_index);
    let enemy_attacks = attacks.all[color.opposite().index()] & front.bit() != 0;
    if blocked {
        features.blocked_advance += sign * rank_weight;
    } else if enemy_attacks {
        features.unsafe_advance += sign * rank_weight;
    } else {
        features.safe_advance += sign * rank_weight;
    }
    if !blocked && attacks.all[color.index()] & front.bit() != 0 {
        features.defended_advance += sign * rank_weight;
    }
    let file_from_center = (2 * i32::from(square.file()) - 7).abs();
    features.outside_file += sign * rank_weight * file_from_center;
}

/// Used for adding one true passer to the STR-293 feature vector.
///
/// Relative rank four starts with urgency one; lower ranks contribute only to
/// the exact released bonus. Advance-state columns are mutually exclusive,
/// while defense, path, rook, file, king, and pawn-race columns are independent
/// context for the later fold-stability test.
///
/// # Arguments
///
/// * `position` - position holding the passer and surrounding geometry
/// * `attacks` - complete rich attack maps for both armies
/// * `color` - color owning the passer
/// * `square` - passer square
/// * `relative_rank` - human-relative rank in `1..=7`
/// * `realized_bonus` - exact released bonus after blocker halving
/// * `blocked` - whether the immediate front square is occupied
/// * `context` - lazily initialized position-wide geometry
/// * `features` - accumulator updated in place
#[allow(clippy::too_many_arguments)]
fn accumulate_passed_pawn_features(
    position: &Position,
    attacks: &AttackMaps,
    color: Color,
    square: Square,
    relative_rank: i32,
    realized_bonus: i32,
    blocked: bool,
    context: &mut Option<PassedPawnContext>,
    features: &mut PassedPawnFeatures,
) {
    let sign = color_sign(color);
    features.current_bonus += sign * realized_bonus;

    let urgency = (relative_rank - 3).max(0);
    if urgency == 0 {
        return;
    }
    let rank_weight = urgency * urgency;
    features.rank_pressure += sign * rank_weight;

    let enemy = color.opposite();
    let front_index = match color {
        Color::White => square.index().checked_sub(8),
        Color::Black => square.index().checked_add(8).filter(|index| *index < 64),
    };
    let Some(front_index) = front_index else {
        return;
    };
    let front = board_square(front_index);
    let context = context.get_or_insert_with(|| PassedPawnContext::build(position, attacks));
    let (forward_path, rear_path) = passed_file_masks(square, color);
    let front_empty = !blocked;
    let front_enemy_attacked = context.attacks[enemy.index()] & front.bit() != 0;
    if blocked {
        features.blocked_advance += sign * rank_weight;
    } else if front_enemy_attacked {
        features.unsafe_advance += sign * rank_weight;
    } else {
        features.safe_advance += sign * rank_weight;
    }
    if front_empty && context.attacks[color.index()] & front.bit() != 0 {
        features.defended_advance += sign * rank_weight;
    }

    if front_empty
        && !front_enemy_attacked
        && forward_path & (context.occupancy | context.attacks[enemy.index()]) == 0
    {
        features.safe_path += sign * rank_weight;
    }

    if let Some(king) = context.kings[color.index()] {
        features.own_king_distance += sign * urgency * king_distance(king, front);
    }
    if let Some(king) = context.kings[enemy.index()] {
        features.enemy_king_distance += sign * urgency * king_distance(king, front);
    }

    let rear_blockers = context.occupancy & rear_path;
    if rear_blockers != 0 {
        let rear_index = match color {
            Color::White => rear_blockers.trailing_zeros(),
            Color::Black => u64::BITS - 1 - rear_blockers.leading_zeros(),
        };
        if context.rooks[color.index()] & (1_u64 << rear_index) != 0 {
            features.rook_behind += sign * rank_weight;
        }
    }

    let file_from_center = (2 * i32::from(square.file()) - 7).abs();
    features.outside_file += sign * rank_weight * file_from_center;

    if !context.has_non_pawn[enemy.index()] {
        let promotion_index = match color {
            Color::White => square.file(),
            Color::Black => 56 + square.file(),
        };
        let promotion = board_square(promotion_index);
        if let Some(enemy_king) = context.kings[enemy.index()] {
            let pushes = 8 - relative_rank;
            let enemy_moves = pushes - i32::from(context.side_to_move == color);
            if king_distance(enemy_king, promotion) > enemy_moves {
                features.square_rule += sign * rank_weight;
            }
        }
    }
}


/// Used for scoring the minimal STR-295 hot-path accumulator.
///
/// # Arguments
///
/// * `features` - five score-bearing values and release phase
///
/// # Returns
///
/// Bounded white-relative delta added to the released pawn term.
fn passed_pawn_fast_candidate_delta(features: PassedPawnCandidateFeatures) -> i32 {
    passed_pawn_candidate_delta_values(
        [
            features.blocked_advance,
            features.unsafe_advance,
            features.safe_advance,
            features.defended_advance,
            features.outside_file,
        ],
        features.phase,
    )
}

/// Used for sharing exact integer arithmetic between research and hot-path
/// feature containers.
///
/// # Arguments
///
/// * `values` - blocked, unsafe, safe, defended, and outside-file values
/// * `phase` - release material phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Symmetrically rounded and bounded phase-tapered delta.
fn passed_pawn_candidate_delta_values(values: [i32; 5], phase: i32) -> i32 {
    let middle_numerator = values
        .iter()
        .zip(PASSED_REALIZATION_MG_X100)
        .map(|(value, weight)| i64::from(*value) * i64::from(weight))
        .sum();
    let ending_numerator = values
        .iter()
        .zip(PASSED_REALIZATION_EG_X100)
        .map(|(value, weight)| i64::from(*value) * i64::from(weight))
        .sum();
    let middle = rounded_feature_division(middle_numerator, PASSED_REALIZATION_WEIGHT_SCALE);
    let ending = rounded_feature_division(ending_numerator, PASSED_REALIZATION_WEIGHT_SCALE);
    tapered_score(middle, ending, phase).clamp(
        -PASSED_REALIZATION_TERM_LIMIT,
        PASSED_REALIZATION_TERM_LIMIT,
    )
}

/// Used for extracting the six constant-time STR-296 initiative inputs.
///
/// King geometry is admitted only when both kings exist, which keeps bounded
/// diagnostic positions deterministic without inventing a coordinate for a
/// missing king. Every board-derived value is unchanged by a vertical color
/// reflection with the armies swapped.
///
/// # Arguments
///
/// * `position` - position supplying pawn mass, material, and king squares
/// * `pre_initiative_white_score` - score whose sign and winner are preserved
/// * `phase` - release material phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Complete raw feature record and exact candidate delta.
fn initiative_features(
    position: &Position,
    pre_initiative_white_score: i32,
    phase: i32,
) -> InitiativeFeatures {
    let pawns = position.piece_bitboard(Piece::new(Color::White, PieceKind::Pawn))
        | position.piece_bitboard(Piece::new(Color::Black, PieceKind::Pawn));
    let kings = position.piece_bitboard(Piece::new(Color::White, PieceKind::King))
        | position.piece_bitboard(Piece::new(Color::Black, PieceKind::King));
    let (outflanking, infiltration) = initiative_king_geometry(
        position.king_square(Color::White),
        position.king_square(Color::Black),
    );
    let mut features = InitiativeFeatures {
        sign: pre_initiative_white_score.signum(),
        intercept: 1,
        total_pawns: bit_count(pawns),
        outflanking,
        infiltration,
        both_flanks: i32::from(
            pawns & QUEENSIDE_FLANK_MASK != 0 && pawns & KINGSIDE_FLANK_MASK != 0,
        ),
        pawn_endgame: i32::from(position.occupancy() & !(pawns | kings) == 0),
        phase,
        candidate_delta: 0,
    };
    features.candidate_delta = initiative_candidate_delta(features, pre_initiative_white_score);
    features
}

/// Used for deriving STR-296's two king-geometry inputs.
///
/// # Arguments
///
/// * `white` - White king square, when available
/// * `black` - Black king square, when available
///
/// # Returns
///
/// File-distance-minus-rank-distance and the symmetric infiltration flag;
/// both zero when either king is absent.
fn initiative_king_geometry(white: Option<Square>, black: Option<Square>) -> (i32, i32) {
    match (white, black) {
        (Some(white), Some(black)) => (
            (i32::from(white.file()) - i32::from(black.file())).abs()
                - (i32::from(white.rank()) - i32::from(black.rank())).abs(),
            i32::from(white.rank() > 4 || black.rank() < 5),
        ),
        _ => (0, 0),
    }
}

/// Used for applying the frozen STR-296 endgame initiative vector.
///
/// The fitted complexity is first rounded in the endgame domain, signed for
/// the side favored by the input score, and tapered from an exactly zero
/// middlegame component. A symmetric implementation bound is followed by a
/// winner-preserving cap: a negative correction may reach equality, but may
/// never make the previously worse side better.
///
/// # Arguments
///
/// * `features` - raw color-invariant complexity values and material phase
/// * `pre_initiative_white_score` - score whose sign must not be reversed
///
/// # Returns
///
/// Bounded, white-relative centipawn correction.
fn initiative_candidate_delta(
    features: InitiativeFeatures,
    pre_initiative_white_score: i32,
) -> i32 {
    if pre_initiative_white_score == 0 {
        return 0;
    }
    let values = [
        features.intercept,
        features.total_pawns,
        features.outflanking,
        features.infiltration,
        features.both_flanks,
        features.pawn_endgame,
    ];
    let ending_numerator: i64 = values
        .iter()
        .zip(INITIATIVE_EG_X100)
        .map(|(value, weight)| i64::from(*value) * i64::from(weight))
        .sum();
    let ending = rounded_feature_division(ending_numerator, INITIATIVE_WEIGHT_SCALE);
    let signed_ending = i64::from(ending) * i64::from(pre_initiative_white_score.signum());
    let ending_weight = i64::from(MAX_PHASE - features.phase.clamp(0, MAX_PHASE));
    let tapered = rounded_feature_division(
        signed_ending.saturating_mul(ending_weight),
        i64::from(MAX_PHASE),
    )
    .clamp(-INITIATIVE_TERM_LIMIT, INITIATIVE_TERM_LIMIT);
    if pre_initiative_white_score > 0 {
        tapered.max(pre_initiative_white_score.saturating_neg())
    } else {
        tapered.min(pre_initiative_white_score.saturating_neg())
    }
}

/// Used for testing whether no enemy pawn blocks this pawn on its own or
/// adjacent files.
///
/// A single intersection with the precomputed
/// [`PASSED_PAWN_BLOCKER_MASKS`] entry answers the question without scanning
/// individual enemy pawns.
///
/// # Arguments
///
/// * `position` - position holding the enemy pawns
/// * `pawn` - candidate passed pawn's square
/// * `color` - color owning the candidate pawn
///
/// # Returns
///
/// `true` when the pawn is passed.
fn is_passed(position: &Position, pawn: Square, color: Color) -> bool {
    let enemy = color.opposite();
    let enemy_pawns = position.piece_bitboard(Piece::new(enemy, PieceKind::Pawn));
    enemy_pawns & PASSED_PAWN_BLOCKER_MASKS[color.index()][usize::from(pawn.index())] == 0
}

/// Used for building every square on the pawn's forward own and adjacent
/// files.
///
/// The mask excludes the pawn's rank and all ranks behind it. Intersecting it
/// with the enemy pawn bitboard therefore answers passed-pawn status without a
/// per-pawn scan of every opposing pawn.
///
/// # Arguments
///
/// * `pawn` - square of the pawn the mask is built for
/// * `color` - color owning the pawn, defining the forward direction
///
/// # Returns
///
/// Bitboard of potential blocker squares ahead of the pawn.
const fn passed_pawn_blocker_mask(pawn: Square, color: Color) -> u64 {
    /// Used for masking the a-file as the shiftable base of every file mask.
    const FILE_A: u64 = 0x0101_0101_0101_0101;

    let file = pawn.file();
    let mut files = FILE_A << file;
    if file > 0 {
        files |= FILE_A << (file - 1);
    }
    if file < 7 {
        files |= FILE_A << (file + 1);
    }

    let row = pawn.row() as u32;
    let forward_ranks = match color {
        Color::White => (1_u64 << (row * 8)).wrapping_sub(1),
        Color::Black if row == 7 => 0,
        Color::Black => u64::MAX << ((row + 1) * 8),
    };
    files & forward_ranks
}

/// Used for building both colors' complete passed-pawn blocker-mask lookup
/// table at compile time.
///
/// Fills [`PASSED_PAWN_BLOCKER_MASKS`] with one
/// [`passed_pawn_blocker_mask`] entry per color and square.
///
/// # Returns
///
/// White and Black mask tables, each indexed by square index.
const fn build_passed_pawn_blocker_masks() -> [[u64; 64]; 2] {
    let mut masks = [[0_u64; 64]; 2];
    let mut square_index = 0_u8;
    let mut table_index = 0_usize;
    while table_index < 64 {
        let Some(pawn) = Square::new(square_index) else {
            return masks;
        };
        masks[0][table_index] = passed_pawn_blocker_mask(pawn, Color::White);
        masks[1][table_index] = passed_pawn_blocker_mask(pawn, Color::Black);
        square_index += 1;
        table_index += 1;
    }
    masks
}

/// Used for scoring rooks on semi-open and fully open files from White's
/// perspective.
///
/// A rook on a file without own pawns earns 20 when the file has no enemy
/// pawns either (open) and 10 otherwise (semi-open).
///
/// # Arguments
///
/// * `position` - position holding the rooks
/// * `file_pawns` - per-file pawn counts indexed by [`Color::index`]
///
/// # Returns
///
/// White-relative rook-file score.
fn rook_files(position: &Position, file_pawns: [[u8; 8]; 2]) -> i32 {
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
                score += sign * if file_pawns[enemy][file] == 0 { 20 } else { 10 };
            }
        }
    }
    score
}

/// Used for scoring how much closer each king is to the nearest occupied pawn
/// file.
///
/// The feature is White-relative: a positive balance means White's king is
/// nearer than Black's king to the union of pawn files. Positions without both
/// kings or without pawns are neutral. The independently fitted MG/EG pair is
/// tapered by the existing material phase and bounded before joining the king
/// placement term.
///
/// # Arguments
///
/// * `position` - position supplying both kings and all pawns
/// * `phase` - release material phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Bounded white-relative STR-304 correction in centipawns.
fn king_pawn_file_proximity_term(position: &Position, phase: i32) -> i32 {
    let (Some(white_king), Some(black_king)) = (
        position.king_square(Color::White),
        position.king_square(Color::Black),
    ) else {
        return 0;
    };
    let pawns = position.piece_bitboard(Piece::new(Color::White, PieceKind::Pawn))
        | position.piece_bitboard(Piece::new(Color::Black, PieceKind::Pawn));
    let occupied_files = occupied_pawn_files(pawns);
    if occupied_files == 0 {
        return 0;
    }

    let white_distance = nearest_occupied_file_distance(occupied_files, white_king.file());
    let black_distance = nearest_occupied_file_distance(occupied_files, black_king.file());
    let balance = black_distance - white_distance;
    tapered_score(
        balance * KING_PAWN_FILE_PROXIMITY.0,
        balance * KING_PAWN_FILE_PROXIMITY.1,
        phase,
    )
    .clamp(
        -KING_PAWN_FILE_PROXIMITY_TERM_LIMIT,
        KING_PAWN_FILE_PROXIMITY_TERM_LIMIT,
    )
}

/// Used for collapsing a pawn bitboard into an eight-bit occupied-file mask.
///
/// # Arguments
///
/// * `pawns` - union of both colors' pawn bitboards
///
/// # Returns
///
/// Bit `f` set exactly when file `f` contains at least one pawn.
fn occupied_pawn_files(pawns: u64) -> u8 {
    let mut files = pawns | (pawns >> 32);
    files |= files >> 16;
    files |= files >> 8;
    u8::try_from(files & 0xff).expect("collapsed pawn files fit u8")
}

/// Used for measuring one king file against a nonempty pawn-file mask.
///
/// # Arguments
///
/// * `occupied_files` - mask produced by [`occupied_pawn_files`]
/// * `king_file` - zero-based king file in `0..=7`
///
/// # Returns
///
/// Smallest absolute file distance in `0..=7`; seven is also the defensive
/// fallback for an empty mask supplied by a research caller.
fn nearest_occupied_file_distance(occupied_files: u8, king_file: u8) -> i32 {
    let king = 1_u8 << king_file;
    for distance in 0_u8..8 {
        if occupied_files & ((king << distance) | (king >> distance)) != 0 {
            return i32::from(distance);
        }
    }
    7
}

/// Used for scoring castled placement and nearby pawn shelter, tapered out
/// in endings.
///
/// A king on a conventional castling destination earns 28 and each sheltered
/// adjacent file earns 9; both bonuses scale linearly with the phase so they
/// vanish in pure endgames. Sides without a king are skipped.
///
/// # Arguments
///
/// * `position` - position holding the kings and pawns
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative king placement and shelter score.
fn king_term(position: &Position, phase: i32) -> i32 {
    let mut score = 0;
    for color in [Color::White, Color::Black] {
        let Some(king) = position.king_square(color) else {
            continue;
        };
        let sign = color_sign(color);
        let castled = is_castled_king_square(color, king);
        let castled_bonus = if castled { 28 * phase / MAX_PHASE } else { 0 };
        let shelter = shelter_file_count(position, color, king) * 9 * phase / MAX_PHASE;
        score += sign * (castled_bonus + shelter);
    }
    score
}

/// Used for recognizing the conventional king destinations after castling.
///
/// # Arguments
///
/// * `color` - color owning the king
/// * `king` - king's current square
///
/// # Returns
///
/// `true` on the color's home rank at file c or g.
fn is_castled_king_square(color: Color, king: Square) -> bool {
    let on_home_rank = match color {
        Color::White => king.rank() == 1,
        Color::Black => king.rank() == 8,
    };
    on_home_rank && matches!(king.file(), 2 | 6)
}

/// Used for counting king-adjacent files containing a plausible forward
/// shelter pawn.
///
/// Considers the king's file and its immediate neighbors; each file counts
/// at most once no matter how many shelter pawns it holds.
///
/// # Arguments
///
/// * `position` - position holding the pawns
/// * `color` - color owning the king and shelter pawns
/// * `king` - king's square
///
/// # Returns
///
/// Number of sheltered files in `0..=3`.
fn shelter_file_count(position: &Position, color: Color, king: Square) -> i32 {
    let king_file = usize::from(king.file());
    let low_file = king_file.saturating_sub(1);
    let high_file = (king_file + 1).min(7);
    let mut sheltered_files = 0_u8;
    let mut pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
    while pawns != 0 {
        let index = u8::try_from(pawns.trailing_zeros()).expect("set bit index is below 64");
        pawns &= pawns - 1;
        let pawn = board_square(index);
        let file = usize::from(pawn.file());
        if (low_file..=high_file).contains(&file) && plausible_shelter_pawn(color, king, pawn) {
            sheltered_files |= 1_u8 << file;
        }
    }
    i32::try_from(sheltered_files.count_ones()).expect("shelter file count fits i32")
}

/// Used for testing whether a pawn lies one to three ranks in front of its
/// king.
///
/// Pawns on, behind, or more than three ranks ahead of the king's rank do
/// not shelter it.
///
/// # Arguments
///
/// * `color` - color owning the king and pawn
/// * `king` - king's square
/// * `pawn` - pawn's square
///
/// # Returns
///
/// `true` when the pawn plausibly shelters the king.
fn plausible_shelter_pawn(color: Color, king: Square, pawn: Square) -> bool {
    let distance = match color {
        Color::White if pawn.rank() > king.rank() => pawn.rank() - king.rank(),
        Color::Black if pawn.rank() < king.rank() => king.rank() - pawn.rank(),
        Color::White | Color::Black => return false,
    };
    distance <= 3
}

/// Used for producing a Java-calibrated ordering hint for non-tactical legal
/// moves.
///
/// The result affects ordering only, never the evaluation. Captures and
/// promotions return zero so their tactical ordering remains authoritative.
/// Castling ranks highest; minors gain from centralization and leaving the
/// back rank, rooks from centralization, and pawns from color-relative
/// advance. Queen and king quiets carry no prior.
///
/// # Arguments
///
/// * `position` - position the move is played in
/// * `mv` - move to rank
///
/// # Returns
///
/// Non-negative prior for castling and pawn advances, signed centralization
/// deltas for minors and rooks, and zero for tactical moves.
fn classical_quiet_move_prior(position: &Position, mv: Move) -> i32 {
    if mv.promotion().is_some() {
        return 0;
    }
    let Some(moving) = position.piece_at(mv.from()) else {
        return 0;
    };
    if position.is_castling_move(mv) {
        return 30_000;
    }
    if position.is_capture(mv) {
        return 0;
    }

    match moving.kind {
        PieceKind::Knight | PieceKind::Bishop => {
            let mut bonus = (move_prior_center(mv.to()) - move_prior_center(mv.from())) * 256;
            if is_back_rank(mv.from(), moving.color) && !is_edge_file(mv.to()) {
                bonus += 8_000;
            }
            bonus
        }
        PieceKind::Rook => (move_prior_center(mv.to()) - move_prior_center(mv.from())) * 96,
        PieceKind::Pawn => {
            let rank = i32::from(mv.to().rank()) - 1;
            let relative_rank = if moving.color == Color::White {
                rank
            } else {
                7 - rank
            };
            let mut bonus = (relative_rank - 2).max(0) * 1_200;
            if relative_rank >= 5 {
                bonus += 3_000;
            }
            bonus
        }
        PieceKind::Queen | PieceKind::King => 0,
    }
}

/// Used for computing a signed centralization coordinate consumed only by
/// quiet-move ordering.
///
/// # Arguments
///
/// * `square` - square to rate
///
/// # Returns
///
/// Centralization value, largest for the four central squares.
fn move_prior_center(square: Square) -> i32 {
    let file = i32::from(square.file());
    let rank = i32::from(square.rank()) - 1;
    14 - 2 * ((2 * file - 7).abs() + (2 * rank - 7).abs())
}

/// Used for testing whether `square` lies on `color`'s home rank.
///
/// # Arguments
///
/// * `square` - square to test
/// * `color` - color whose home rank applies
///
/// # Returns
///
/// `true` on rank 1 for White or rank 8 for Black.
fn is_back_rank(square: Square, color: Color) -> bool {
    match color {
        Color::White => square.rank() == 1,
        Color::Black => square.rank() == 8,
    }
}

/// Used for testing whether `square` lies on the a- or h-file.
///
/// # Arguments
///
/// * `square` - square to test
///
/// # Returns
///
/// `true` for the two outermost files.
fn is_edge_file(square: Square) -> bool {
    matches!(square.file(), 0 | 7)
}

/// Used for looking up middlegame and endgame safe-mobility scores.
///
/// Destination counts beyond a table's calibrated range use its final entry.
/// Pawns and kings have no mobility tables and score zero.
///
/// # Arguments
///
/// * `kind` - piece kind selecting the table pair
/// * `mobility` - number of safe destination squares
///
/// # Returns
///
/// Middlegame and endgame mobility score pair.
fn mobility_scores<const CANDIDATE_MOBILITY: bool>(kind: PieceKind, mobility: usize) -> (i32, i32) {
    let (middle, ending): (&[i32], &[i32]) = match (CANDIDATE_MOBILITY, kind) {
        (false, PieceKind::Knight) => (&KNIGHT_MOBILITY_MG, &KNIGHT_MOBILITY_EG),
        (false, PieceKind::Bishop) => (&BISHOP_MOBILITY_MG, &BISHOP_MOBILITY_EG),
        (false, PieceKind::Rook) => (&ROOK_MOBILITY_MG, &ROOK_MOBILITY_EG),
        (false, PieceKind::Queen) => (&QUEEN_MOBILITY_MG, &QUEEN_MOBILITY_EG),
        (true, PieceKind::Knight) => (&MOBILITY_CANDIDATE_KNIGHT_MG, &MOBILITY_CANDIDATE_KNIGHT_EG),
        (true, PieceKind::Bishop) => (&MOBILITY_CANDIDATE_BISHOP_MG, &MOBILITY_CANDIDATE_BISHOP_EG),
        (true, PieceKind::Rook) => (&MOBILITY_CANDIDATE_ROOK_MG, &MOBILITY_CANDIDATE_ROOK_EG),
        (true, PieceKind::Queen) => (&MOBILITY_CANDIDATE_QUEEN_MG, &MOBILITY_CANDIDATE_QUEEN_EG),
        (_, PieceKind::Pawn | PieceKind::King) => return (0, 0),
    };
    let index = mobility.min(middle.len() - 1);
    (middle[index], ending[index])
}

/// Used for tapering the color-relative safe-mobility totals with symmetric
/// rounding.
///
/// Rounding is symmetric around zero so swapping the armies always negates
/// this term exactly.
///
/// # Arguments
///
/// * `attacks` - attack maps holding both colors' mobility totals
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative tapered mobility score.
fn mobility_term(attacks: &AttackMaps, phase: i32) -> i32 {
    let middle =
        attacks.mobility_mg[Color::White.index()] - attacks.mobility_mg[Color::Black.index()];
    let ending =
        attacks.mobility_eg[Color::White.index()] - attacks.mobility_eg[Color::Black.index()];
    let weighted = middle * phase + ending * (MAX_PHASE - phase);
    // Round symmetrically so swapping the armies always negates this term.
    if weighted >= 0 {
        (weighted + MAX_PHASE / 2) / MAX_PHASE
    } else {
        (weighted - MAX_PHASE / 2) / MAX_PHASE
    }
}

/// Used for scoring attacked material and pressure around each king in the
/// frozen compact attack flavor.
///
/// Each attacked non-king piece contributes a fraction of its material value
/// — one forty-fifth when defended, one eighteenth when not. Each king then
/// concedes phase-scaled pressure per attacked ring square plus a flat check
/// penalty.
///
/// # Arguments
///
/// * `position` - position whose pieces and kings are inspected
/// * `attacks` - geometric attack maps for both colors
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative compact threat score.
fn compact_threat_term(position: &Position, attacks: &AttackMaps, phase: i32) -> i32 {
    let mut score = 0;
    for index in 0_u8..64 {
        let target = board_square(index);
        let Some(piece) = position.piece_at(target) else {
            continue;
        };
        if piece.kind == PieceKind::King {
            continue;
        }
        let enemy = piece.color.opposite();
        if !attacks.attacks(target, enemy) {
            continue;
        }
        let defended = attacks.attacks(target, piece.color);
        let base = MATERIAL[piece.kind.index()] / if defended { 45 } else { 18 };
        score += color_sign(enemy) * base;
    }

    for color in [Color::White, Color::Black] {
        let Some(king) = position.king_square(color) else {
            continue;
        };
        let enemy = color.opposite();
        let mut king_ring = 0_u64;
        for file_delta in -1..=1 {
            for row_delta in -1..=1 {
                if file_delta != 0 || row_delta != 0 {
                    if let Some(square) = square_offset(king, file_delta, row_delta) {
                        king_ring |= square.bit();
                    }
                }
            }
        }
        let attacked_ring = bit_count(king_ring & attacks.all[enemy.index()]);
        let check = i32::from(attacks.attacks(king, enemy)) * 28;
        let pressure = attacked_ring * 5 * phase / MAX_PHASE + check;
        score -= color_sign(color) * pressure;
    }
    score
}

/// Used for blending the per-color Java activity accumulators from White's
/// perspective.
///
/// # Arguments
///
/// * `attacks` - attack maps holding both colors' activity totals
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative tapered activity score.
fn activity_term(attacks: &AttackMaps, phase: i32) -> i32 {
    tapered_score(
        attacks.activity_mg[Color::White.index()] - attacks.activity_mg[Color::Black.index()],
        attacks.activity_eg[Color::White.index()] - attacks.activity_eg[Color::Black.index()],
        phase,
    )
}

/// Used for scoring one non-pawn piece's positional activity in
/// middle/endgame pairs.
///
/// Minors earn outpost and reachable-outpost bonuses, a bonus for standing
/// directly behind an own pawn, and a distance-to-own-king penalty. Bishops
/// additionally take the bad-bishop penalty and earn a long-diagonal bonus;
/// rooks earn queen-file and seventh-rank bonuses and take a trapped-rook
/// penalty; queens earn an advanced-safety bonus.
///
/// # Arguments
///
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
fn piece_activity(
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
                KNIGHT_OUTPOST
            } else {
                BISHOP_OUTPOST
            };
            middle += bonus.0;
            ending += bonus.1;
        } else if attacks & outpost_mask(color, own_pawn_attacks, enemy_pawn_attacks) & !own != 0 {
            let bonus = if kind == PieceKind::Knight {
                (13, 6)
            } else {
                (7, 3)
            };
            middle += bonus.0;
            ending += bonus.1;
        }
        if minor_behind_pawn(position, color, square) {
            middle += 8;
            ending += 5;
        }
        if let Some(king) = position.king_square(color) {
            let distance = king_distance(king, square);
            // Both minor distance weights fitted to the same value.
            let weight = 2;
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
                middle += 8;
                ending += 4;
            }
        }
        PieceKind::Rook => {
            let queens = position.piece_bitboard(Piece::new(Color::White, PieceKind::Queen))
                | position.piece_bitboard(Piece::new(Color::Black, PieceKind::Queen));
            if queens & file_mask(square.file()) != 0 {
                middle += 6;
            }
            let seventh = if color == Color::White { 7 } else { 2 };
            if square.rank() == seventh && position.color_occupancy(enemy) & rank_mask(seventh) != 0
            {
                middle += 18;
                ending += 27;
            }
            if mobility <= 3 {
                if let Some(king) = position.king_square(color) {
                    if same_flank(square, king) {
                        let penalty = if can_castle_either(position, color) {
                            10
                        } else {
                            16
                        };
                        middle -= penalty;
                        ending -= penalty / 2;
                    }
                }
            }
        }
        PieceKind::Queen => {
            if relative_rank(color, square) >= 4 && square.bit() & enemy_pawn_attacks == 0 {
                middle += 7;
                ending += 10;
            }
        }
        PieceKind::Pawn | PieceKind::Knight | PieceKind::King => {}
    }
    (middle, ending)
}

/// Used for scoring Java-style weak, hanging, pawn-push, and queen-pressure
/// threats.
///
/// Computes [`side_threats`] for both colors and tapers the difference.
///
/// # Arguments
///
/// * `position` - position whose threats are scored
/// * `attacks` - rich attack maps for both colors
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative tapered threat score.
fn rich_threat_term<const CANDIDATE_THREATS: bool>(
    position: &Position,
    attacks: &AttackMaps,
    phase: i32,
) -> i32 {
    let (white_middle, white_ending) =
        side_threats::<CANDIDATE_THREATS>(position, attacks, Color::White);
    let (black_middle, black_ending) =
        side_threats::<CANDIDATE_THREATS>(position, attacks, Color::Black);
    tapered_score(
        white_middle - black_middle,
        white_ending - black_ending,
        phase,
    )
}

/// Used for computing the middlegame/endgame threat pair earned by one
/// attacker color.
///
/// Accumulates typed minor attacks on weak enemies and strongly protected
/// non-pawn enemies, typed rook attacks on weak enemies, hanging and
/// restricted squares, safe pawn attacks, threats created by safe pawn
/// pushes, and safe knight or doubly supported slider pressure against the
/// first enemy queen. An enemy is strongly protected when guarded by a pawn
/// or defended at least twice while not attacked at least twice.
///
/// # Arguments
///
/// * `position` - position whose pieces are inspected
/// * `attacks` - rich attack maps for both colors
/// * `color` - attacking color earning the threats
///
/// # Returns
///
/// Middlegame and endgame threat totals for the attacker.
#[allow(clippy::too_many_lines)]
fn side_threats<const CANDIDATE_THREATS: bool>(
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
            let scores = minor_threat_scores::<CANDIDATE_THREATS>(piece.kind);
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
            let scores = rook_threat_scores::<CANDIDATE_THREATS>(piece.kind);
            middle += scores.0;
            ending += scores.1;
        }
    }

    let hanging = weak & (!attacks.all[enemy_side] | attacks.attacked_twice[side]);
    let hanging_count = bit_count(hanging);
    let hanging_weight = if CANDIDATE_THREATS {
        THREAT_CANDIDATE_HANGING
    } else {
        HANGING_THREAT
    };
    middle += hanging_count * hanging_weight.0;
    ending += hanging_count * hanging_weight.1;

    let restricted = attacks.all[enemy_side] & !strongly_protected & attacks.all[side];
    let restricted_weight = if CANDIDATE_THREATS {
        THREAT_CANDIDATE_RESTRICTED_MG
    } else {
        RESTRICTED_THREAT_MG
    };
    middle += bit_count(restricted) * restricted_weight;

    let safe = !attacks.all[enemy_side] | attacks.all[side];
    let safe_pawn_threats =
        attacks.by_kind[side][PieceKind::Pawn.index()] & non_pawn_enemies & safe;
    let safe_pawn_weight = if CANDIDATE_THREATS {
        THREAT_CANDIDATE_SAFE_PAWN
    } else {
        SAFE_PAWN_THREAT
    };
    middle += bit_count(safe_pawn_threats) * safe_pawn_weight.0;
    ending += bit_count(safe_pawn_threats) * safe_pawn_weight.1;

    let own_pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
    let pushed_pawns = pawn_push_mask(color, own_pawns, !position.occupancy());
    let pawn_push_threats = pawn_attack_mask(color, pushed_pawns)
        & non_pawn_enemies
        & !attacks.by_kind[enemy_side][PieceKind::Pawn.index()]
        & safe;
    let pawn_push_weight = if CANDIDATE_THREATS {
        THREAT_CANDIDATE_PAWN_PUSH
    } else {
        PAWN_PUSH_THREAT
    };
    middle += bit_count(pawn_push_threats) * pawn_push_weight.0;
    ending += bit_count(pawn_push_threats) * pawn_push_weight.1;

    let enemy_queens = position.piece_bitboard(Piece::new(enemy, PieceKind::Queen));
    if enemy_queens != 0 {
        let queen = board_square(
            u8::try_from(enemy_queens.trailing_zeros()).expect("set bit index is below 64"),
        );
        let safe_knight_pressure =
            attacks.by_kind[side][PieceKind::Knight.index()] & knight_attacks(queen) & safe;
        let queen_knight_weight = if CANDIDATE_THREATS {
            THREAT_CANDIDATE_QUEEN_KNIGHT
        } else {
            QUEEN_KNIGHT_THREAT
        };
        middle += bit_count(safe_knight_pressure) * queen_knight_weight.0;
        ending += bit_count(safe_knight_pressure) * queen_knight_weight.1;

        let occupancy = position.occupancy();
        let slider_pressure = (attacks.by_kind[side][PieceKind::Bishop.index()]
            & diagonal_attacks(queen, occupancy)
            | attacks.by_kind[side][PieceKind::Rook.index()]
                & orthogonal_attacks(queen, occupancy))
            & attacks.attacked_twice[side]
            & safe;
        let queen_slider_weight = if CANDIDATE_THREATS {
            THREAT_CANDIDATE_QUEEN_SLIDER
        } else {
            QUEEN_SLIDER_THREAT
        };
        middle += bit_count(slider_pressure) * queen_slider_weight.0;
        ending += bit_count(slider_pressure) * queen_slider_weight.1;
    }
    (middle, ending)
}

/// Used for looking up Java's typed minor-piece threat pair for one target
/// kind.
///
/// # Arguments
///
/// * `target` - kind of the threatened enemy piece
///
/// # Returns
///
/// Middlegame and endgame threat values; zero for a king target.
const fn minor_threat_scores<const CANDIDATE_THREATS: bool>(target: PieceKind) -> (i32, i32) {
    if CANDIDATE_THREATS {
        match target {
            PieceKind::Pawn => (5, 13),
            PieceKind::Knight | PieceKind::Bishop => (27, 21),
            PieceKind::Rook => (40, 31),
            PieceKind::Queen => (47, 59),
            PieceKind::King => (0, 0),
        }
    } else {
        match target {
            PieceKind::Pawn => (6, 15),
            PieceKind::Knight | PieceKind::Bishop => (32, 25),
            PieceKind::Rook => (48, 37),
            PieceKind::Queen => (57, 70),
            PieceKind::King => (0, 0),
        }
    }
}

/// Used for looking up Java's typed rook threat pair for one target kind.
///
/// # Arguments
///
/// * `target` - kind of the threatened enemy piece
///
/// # Returns
///
/// Middlegame and endgame threat values; zero for a king target.
const fn rook_threat_scores<const CANDIDATE_THREATS: bool>(target: PieceKind) -> (i32, i32) {
    if CANDIDATE_THREATS {
        match target {
            PieceKind::Pawn => (3, 19),
            PieceKind::Knight | PieceKind::Bishop => (19, 24),
            PieceKind::Rook => (0, 12),
            PieceKind::Queen => (33, 24),
            PieceKind::King => (0, 0),
        }
    } else {
        match target {
            PieceKind::Pawn => (4, 28),
            PieceKind::Knight | PieceKind::Bishop => (24, 36),
            PieceKind::Rook => (0, 18),
            PieceKind::Queen => (41, 36),
            PieceKind::King => (0, 0),
        }
    }
}

/// Used for scoring weighted attacks into each enemy king zone from White's
/// perspective.
///
/// Each king's raw pressure from [`side_king_pressure`] is reduced to 55%
/// when the attacker has no queen, scaled by the phase, and then increased
/// by a flat bonus when the king is actually in check.
///
/// # Arguments
///
/// * `position` - position holding the kings
/// * `attacks` - rich attack maps for both colors
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative king-pressure score.
fn king_pressure_term(position: &Position, attacks: &AttackMaps, phase: i32) -> i32 {
    let mut score = 0;
    for king_color in [Color::White, Color::Black] {
        let Some(king) = position.king_square(king_color) else {
            continue;
        };
        let mut pressure = side_king_pressure(position, attacks, king_color, king);
        let enemy = king_color.opposite();
        if position.piece_bitboard(Piece::new(enemy, PieceKind::Queen)) == 0 {
            pressure = pressure * KING_QUEENLESS_PRESSURE_PERCENT / 100;
        }
        pressure = pressure * phase / MAX_PHASE;
        if position.in_check(king_color) {
            pressure += 35;
        }
        score -= color_sign(king_color) * pressure;
    }
    score
}

/// Used for computing the unphased king-zone pressure against one king.
///
/// Combines the attackers' weight sum, the squared attacker count, ring
/// hits, weak zone squares, quadratic flank attacks, and any positive
/// middlegame mobility edge, minus flank defense; a pawnless flank adds a
/// flat penalty. Pressure is halved when only one piece attacks the zone and
/// is zero with no attackers.
///
/// # Arguments
///
/// * `position` - position supplying the pawn layout
/// * `attacks` - rich attack maps for both colors
/// * `king_color` - color of the defending king
/// * `king` - defending king's square
///
/// # Returns
///
/// Raw pressure total before phase scaling and queen discounts.
fn side_king_pressure(
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
    let weak = weak_king_squares(attacks, defender, attacker);
    let flank = king_flank_mask(king) & camp_mask(king_color);
    let flank_attack = bit_count(attacks.all[attacker] & flank)
        + bit_count(attacks.attacked_twice[attacker] & flank);
    let flank_defense = bit_count(attacks.all[defender] & flank);
    let mut pressure = attacks.king_attacker_weight[attacker]
        + attackers * attackers * KING_ATTACKERS_SQUARED
        + KING_RING_ATTACK * attacks.king_ring_hits[attacker]
        + KING_WEAK_SQUARE * bit_count(attacks.king_zone[defender] & weak)
        + flank_attack * flank_attack / 2
        + (attacks.mobility_mg[attacker] - attacks.mobility_mg[defender]).max(0) / 2
        - flank_defense * KING_FLANK_DEFENSE;
    let pawns = position.piece_bitboard(Piece::new(Color::White, PieceKind::Pawn))
        | position.piece_bitboard(Piece::new(Color::Black, PieceKind::Pawn));
    if pawns & flank == 0 {
        pressure += KING_PAWNLESS_FLANK;
    }
    if attackers == 1 {
        pressure / 2
    } else {
        pressure
    }
}

/// Used for identifying king-area squares the defender cannot hold strongly.
///
/// # Arguments
///
/// * `attacks` - complete rich attack maps
/// * `defender` - index of the king's color
/// * `attacker` - index of the opposing color
///
/// # Returns
///
/// Squares attacked by the opponent and not multiply defended, except for
/// squares whose only available defenders may be the king or queen.
fn weak_king_squares(attacks: &AttackMaps, defender: usize, attacker: usize) -> u64 {
    attacks.all[attacker]
        & !attacks.attacked_twice[defender]
        & (!attacks.all[defender]
            | attacks.by_kind[defender][PieceKind::King.index()]
            | attacks.by_kind[defender][PieceKind::Queen.index()])
}

/// Used for counting relatively safe destinations that would give check.
///
/// A destination must be reachable by an attacking piece of the matching
/// kind, lie on that kind's checking geometry around the king, avoid the
/// attacker's own occupancy, and be either undefended or weak with double
/// attacker support. Counts represent destinations rather than pieces so an
/// overloaded square cannot inflate the feature merely because two sliders
/// share a ray.
///
/// # Arguments
///
/// * `position` - position supplying occupancy and colored pieces
/// * `attacks` - complete rich attack maps
/// * `king_color` - color of the king being threatened
/// * `king` - square occupied by that king
///
/// # Returns
///
/// Counts ordered knight, bishop, rook, queen.
fn safe_check_destinations(
    position: &Position,
    attacks: &AttackMaps,
    king_color: Color,
    king: Square,
) -> [i32; 4] {
    let defender = king_color.index();
    let attacker_color = king_color.opposite();
    let attacker = attacker_color.index();
    let weak = weak_king_squares(attacks, defender, attacker);
    let relatively_safe = !position.color_occupancy(attacker_color)
        & (!attacks.all[defender] | (weak & attacks.attacked_twice[attacker]));
    let occupancy = position.occupancy();
    let diagonal = diagonal_attacks(king, occupancy);
    let orthogonal = orthogonal_attacks(king, occupancy);
    [
        bit_count(
            knight_attacks(king)
                & relatively_safe
                & attacks.by_kind[attacker][PieceKind::Knight.index()],
        ),
        bit_count(
            diagonal & relatively_safe & attacks.by_kind[attacker][PieceKind::Bishop.index()],
        ),
        bit_count(
            orthogonal & relatively_safe & attacks.by_kind[attacker][PieceKind::Rook.index()],
        ),
        bit_count(
            (diagonal | orthogonal)
                & relatively_safe
                & attacks.by_kind[attacker][PieceKind::Queen.index()],
        ),
    ]
}

/// Used for measuring how far a pawn lies in front of a defending king.
///
/// # Arguments
///
/// * `king_color` - color defining the forward half-board direction
/// * `king` - defending king square
/// * `pawn` - own shelter pawn or approaching enemy storm pawn
///
/// # Returns
///
/// Positive rank distance when the pawn lies forward of the king, otherwise
/// `None`.
fn forward_pawn_distance(king_color: Color, king: Square, pawn: Square) -> Option<i32> {
    match king_color {
        Color::White if pawn.rank() > king.rank() => Some(i32::from(pawn.rank() - king.rank())),
        Color::Black if pawn.rank() < king.rank() => Some(i32::from(king.rank() - pawn.rank())),
        Color::White | Color::Black => None,
    }
}

/// Used for finding the nearest forward pawn on one king-flank file.
///
/// # Arguments
///
/// * `pawns` - bitboard containing only the pawn set to inspect
/// * `king_color` - color defining forward from the king
/// * `king` - defending king square
/// * `file` - zero-based file restricted to the king and adjacent files
///
/// # Returns
///
/// Smallest positive rank distance on the file, or `None` when absent.
fn nearest_forward_pawn_distance(
    mut pawns: u64,
    king_color: Color,
    king: Square,
    file: u8,
) -> Option<i32> {
    let mut nearest = None;
    while pawns != 0 {
        let index = u8::try_from(pawns.trailing_zeros()).expect("set bit index is below 64");
        pawns &= pawns - 1;
        let pawn = board_square(index);
        if pawn.file() != file {
            continue;
        }
        if let Some(distance) = forward_pawn_distance(king_color, king, pawn) {
            nearest = Some(nearest.map_or(distance, |value: i32| value.min(distance)));
        }
    }
    nearest
}

/// Used for extracting missing shelter, advanced cover, and pawn-storm inputs.
///
/// Only the king's file and immediate neighbors are inspected. An own pawn
/// within four ranks supplies cover; every step beyond the nearest rank adds
/// advanced-cover exposure. An enemy pawn within four ranks contributes more
/// storm pressure as it approaches the king.
///
/// # Arguments
///
/// * `position` - position supplying both pawn sets
/// * `king_color` - color of the king being inspected
/// * `king` - defending king square
///
/// # Returns
///
/// `(missing_files, advanced_steps, storm_proximity)` for this king.
fn king_pawn_cover_features(
    position: &Position,
    king_color: Color,
    king: Square,
) -> (i32, i32, i32) {
    let own_pawns = position.piece_bitboard(Piece::new(king_color, PieceKind::Pawn));
    let enemy_pawns = position.piece_bitboard(Piece::new(king_color.opposite(), PieceKind::Pawn));
    let low_file = king.file().saturating_sub(1);
    let high_file = king.file().saturating_add(1).min(7);
    let mut missing = 0;
    let mut advanced = 0;
    let mut storm = 0;
    for file in low_file..=high_file {
        match nearest_forward_pawn_distance(own_pawns, king_color, king, file) {
            Some(distance) if distance <= 4 => advanced += (distance - 1).max(0),
            Some(_) | None => missing += 1,
        }
        if let Some(distance) = nearest_forward_pawn_distance(enemy_pawns, king_color, king, file) {
            if distance <= 4 {
                storm += 5 - distance;
            }
        }
    }
    (missing, advanced, storm)
}

/// Used for converting a bounded signed 64-bit feature sum to saturated i32.
///
/// # Arguments
///
/// * `value` - intermediate value whose sign must be preserved
///
/// # Returns
///
/// Exact value when representable, otherwise the nearest i32 endpoint.
fn saturating_feature_i32(value: i64) -> i32 {
    i32::try_from(value).unwrap_or(if value < 0 { i32::MIN } else { i32::MAX })
}

/// Used for extracting the STR-280 king-danger feature vector.
///
/// # Arguments
///
/// * `position` - position whose kings and pawn cover are inspected
/// * `attacks` - complete rich attack maps for the position
/// * `phase` - release material phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative raw features without changing the release score.
/// Used for scoring king danger with the gate and clamp applied per defender.
///
/// The promoted STR-280 scorer gathers both kings into one signed feature
/// vector, tests the *net* safe-check counts for zero, and clamps the combined
/// White-minus-Black term once. Equal checking chances on both sides therefore
/// cancel before the gate, and a read-only audit of the 104,033-position
/// development cohort found 53,451 positions with a zero aggregate check vector
/// but nonzero suppressed shelter or pressure context. This flavour instead
/// builds one unsigned feature record per defended king, converts and clamps
/// each independently with the same fitted coefficients, and only then
/// subtracts the two results. No coefficient is refitted: the contrast isolates
/// where the gate and clamp are applied.
///
/// # Arguments
///
/// * `position` - position holding the kings
/// * `attacks` - rich attack maps for both colors
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// White-relative king-pressure term in centipawns.
fn per_king_danger_pressure(position: &Position, attacks: &AttackMaps, phase: i32) -> i32 {
    let mut score = 0;
    for king_color in [Color::White, Color::Black] {
        if position.king_square(king_color).is_none() {
            continue;
        }
        let defender_features = king_danger_features_for(position, attacks, phase, king_color, 1);
        score -= color_sign(king_color) * king_danger_candidate_pressure(defender_features);
    }
    score
}

/// Used for building one king's danger features at a caller-chosen sign.
///
/// Passing `danger_sign = 1` yields the unsigned record the per-king flavour
/// converts independently; the aggregate flavour passes the defender's own
/// `-color_sign` and accumulates.
///
/// # Arguments
///
/// * `position` - position holding the kings
/// * `attacks` - rich attack maps for both colors
/// * `phase` - tapering phase in `0..=MAX_PHASE`
/// * `king_color` - defending king's color
/// * `danger_sign` - sign applied to every extracted feature
///
/// # Returns
///
/// One defended king's contribution to the danger feature vector.
fn king_danger_features_for(
    position: &Position,
    attacks: &AttackMaps,
    phase: i32,
    king_color: Color,
    danger_sign: i32,
) -> KingDangerFeatures {
    let mut features = KingDangerFeatures {
        phase,
        ..KingDangerFeatures::default()
    };
    let Some(king) = position.king_square(king_color) else {
        return features;
    };
    let attacker = king_color.opposite();
    let queen_scale = if position.piece_bitboard(Piece::new(attacker, PieceKind::Queen)) == 0 {
        KING_QUEENLESS_PRESSURE_PERCENT
    } else {
        100
    };
    let checks = safe_check_destinations(position, attacks, king_color, king);
    features.safe_knight_checks_x100 = danger_sign * queen_scale * checks[0];
    features.safe_bishop_checks_x100 = danger_sign * queen_scale * checks[1];
    features.safe_rook_checks_x100 = danger_sign * queen_scale * checks[2];
    features.safe_queen_checks_x100 = danger_sign * queen_scale * checks[3];
    let (missing, advanced, storm) = king_pawn_cover_features(position, king_color, king);
    features.missing_shelter_files = danger_sign * missing;
    features.advanced_shelter_steps = danger_sign * advanced;
    features.pawn_storm_proximity = danger_sign * storm;

    let raw = side_king_pressure(position, attacks, king_color, king);
    let gated_pressure = raw * queen_scale / 100;
    let mut released_pressure = gated_pressure * phase / MAX_PHASE;
    if position.in_check(king_color) {
        released_pressure += 35;
    }
    features.current_pressure = danger_sign * released_pressure;

    let gated = i64::from(gated_pressure.max(0));
    features.pressure_squared_over_256 =
        saturating_feature_i32(i64::from(danger_sign) * gated * gated / 256);
    features
}

fn king_danger_features(
    position: &Position,
    attacks: &AttackMaps,
    phase: i32,
) -> KingDangerFeatures {
    let mut features = KingDangerFeatures {
        phase,
        ..KingDangerFeatures::default()
    };
    let mut squared = 0_i64;
    for king_color in [Color::White, Color::Black] {
        let Some(king) = position.king_square(king_color) else {
            continue;
        };
        let attacker = king_color.opposite();
        let danger_sign = -color_sign(king_color);
        let queen_scale = if position.piece_bitboard(Piece::new(attacker, PieceKind::Queen)) == 0 {
            KING_QUEENLESS_PRESSURE_PERCENT
        } else {
            100
        };
        let checks = safe_check_destinations(position, attacks, king_color, king);
        features.safe_knight_checks_x100 += danger_sign * queen_scale * checks[0];
        features.safe_bishop_checks_x100 += danger_sign * queen_scale * checks[1];
        features.safe_rook_checks_x100 += danger_sign * queen_scale * checks[2];
        features.safe_queen_checks_x100 += danger_sign * queen_scale * checks[3];
        let (missing, advanced, storm) = king_pawn_cover_features(position, king_color, king);
        features.missing_shelter_files += danger_sign * missing;
        features.advanced_shelter_steps += danger_sign * advanced;
        features.pawn_storm_proximity += danger_sign * storm;

        let raw = side_king_pressure(position, attacks, king_color, king);
        let gated_pressure = raw * queen_scale / 100;
        let mut released_pressure = gated_pressure * phase / MAX_PHASE;
        if position.in_check(king_color) {
            released_pressure += 35;
        }
        features.current_pressure += danger_sign * released_pressure;

        let gated = i64::from(gated_pressure.max(0));
        squared += i64::from(danger_sign) * gated * gated / 256;
    }
    features.pressure_squared_over_256 = saturating_feature_i32(squared);
    features
}

/// Used for rounded symmetric division of a signed feature numerator.
///
/// # Arguments
///
/// * `numerator` - signed numerator
/// * `denominator` - strictly positive scale divisor
///
/// # Returns
///
/// Nearest integer with half ties away from zero, saturated to i32.
fn rounded_feature_division(numerator: i64, denominator: i64) -> i32 {
    debug_assert!(denominator > 0);
    let half = denominator / 2;
    let rounded = if numerator >= 0 {
        numerator.saturating_add(half) / denominator
    } else {
        numerator.saturating_sub(half) / denominator
    };
    saturating_feature_i32(rounded)
}

/// Used for applying the five-fold STR-280 integer king-danger vector.
///
/// The candidate activates only when the signed safe-check feature vector is
/// nonzero. This is the predeclared correction to the first global fit, whose
/// 94% activation violated the experiment's capacity gate. All additions are
/// phase tapered and the complete term is clamped before it rejoins the
/// breakdown.
///
/// # Arguments
///
/// * `features` - named release and candidate king-danger inputs
///
/// # Returns
///
/// White-relative candidate king-pressure term in centipawns.
fn king_danger_candidate_pressure(features: KingDangerFeatures) -> i32 {
    let safe_checks = [
        features.safe_knight_checks_x100,
        features.safe_bishop_checks_x100,
        features.safe_rook_checks_x100,
        features.safe_queen_checks_x100,
    ];
    if safe_checks.iter().all(|value| *value == 0) {
        return features.current_pressure;
    }

    let current_delta = rounded_feature_division(
        i64::from(features.current_pressure) * i64::from(KING_DANGER_CURRENT_DELTA_PERCENT),
        100,
    );
    let safe_numerator = safe_checks
        .iter()
        .zip(KING_DANGER_SAFE_CHECK)
        .map(|(count, weight)| i64::from(*count) * i64::from(weight))
        .sum::<i64>()
        * i64::from(features.phase);
    let safe_delta = rounded_feature_division(safe_numerator, 100 * i64::from(MAX_PHASE));
    let context_numerator = (i64::from(features.missing_shelter_files)
        * i64::from(KING_DANGER_MISSING_SHELTER)
        + i64::from(features.advanced_shelter_steps) * i64::from(KING_DANGER_ADVANCED_SHELTER)
        + i64::from(features.pressure_squared_over_256) * i64::from(KING_DANGER_PRESSURE_SQUARED))
        * i64::from(features.phase);
    let context_delta = rounded_feature_division(context_numerator, i64::from(MAX_PHASE));
    features
        .current_pressure
        .saturating_add(current_delta)
        .saturating_add(safe_delta)
        .saturating_add(context_delta)
        .clamp(-KING_DANGER_TERM_LIMIT, KING_DANGER_TERM_LIMIT)
}

/// Used for the frozen STR-319 and STR-304 additions consumed by the
/// parameterized twin and the sparse linear model.
///
/// Both confirmed terms are built from named constants rather than tunable
/// parameters, so neither belongs in the linear basis. They enter the twin and
/// the sparse model as frozen `(pawn, king)` centipawn constants exactly as
/// STR-280's non-linear correction does.
///
/// # Arguments
///
/// * `position` - position to evaluate
///
/// # Panics
///
/// Panics only on internal bit-index conversions that cannot fail for a valid
/// position.
///
/// # Returns
///
/// White-relative `(connected-pawn, king-pawn-file)` contributions in
/// centipawns.
pub(crate) fn promoted_capacity_delta(position: &Position) -> (i32, i32) {
    let phase = game_phase(position);
    (
        connected_pawn_term(position, phase, CONNECTED_PAWN_SCALE_PERCENT),
        king_pawn_file_proximity_term(position, phase),
    )
}

/// Used for the frozen STR-280 correction over the released linear
/// king-pressure family.
///
/// The promoted scorer is non-linear in the tunable king parameters: it gates
/// on a signed safe-check vector, rescales current pressure by a rounded
/// percentage, adds a squared-pressure interaction, and clamps the sum. The
/// parameterized twin and the sparse linear model therefore keep the released
/// linear pressure family and add this position-local difference as a frozen
/// constant. It is deliberately evaluated at the released attack maps rather
/// than at a caller's perturbed parameters, so the constant stays independent
/// of the tuned vector and cannot be misread as a linear coefficient.
///
/// # Arguments
///
/// * `position` - position to evaluate
///
/// # Panics
///
/// Panics only on internal bit-index conversions that cannot fail for a valid
/// position.
///
/// # Returns
///
/// White-relative `promoted - released` king-pressure difference in
/// centipawns, and exactly zero wherever the promoted gate is inactive.
pub(crate) fn king_danger_promoted_delta(position: &Position) -> i32 {
    let phase = game_phase(position);
    let attacks = AttackMaps::build(position, RELEASE_RICH_ATTACK_STATE);
    let features = king_danger_features(position, &attacks, phase);
    let released = features.current_pressure;
    king_danger_candidate_pressure(features).saturating_sub(released)
}

/// Used for symmetrically tapering a middlegame/endgame pair at the material
/// phase.
///
/// Rounds to the nearest unit with ties away from zero, so negating the
/// inputs exactly negates the result.
///
/// # Arguments
///
/// * `middle` - middlegame component
/// * `ending` - endgame component
/// * `phase` - tapering phase in `0..=MAX_PHASE`
///
/// # Returns
///
/// Phase-weighted blend of the two components.
fn tapered_score(middle: i32, ending: i32, phase: i32) -> i32 {
    let weighted = middle * phase + ending * (MAX_PHASE - phase);
    if weighted >= 0 {
        (weighted + MAX_PHASE / 2) / MAX_PHASE
    } else {
        (weighted - MAX_PHASE / 2) / MAX_PHASE
    }
}

/// Used for testing whether a minor occupies a pawn-backed square immune to
/// enemy pawns.
///
/// # Arguments
///
/// * `color` - color owning the minor
/// * `square` - minor's square
/// * `own_pawn_attacks` - squares defended by the color's pawns
/// * `enemy_pawn_attacks` - squares attacked by enemy pawns
///
/// # Returns
///
/// `true` when the square is pawn-defended, not pawn-attacked, and on the
/// color-relative fourth through sixth ranks.
fn is_outpost(
    color: Color,
    square: Square,
    own_pawn_attacks: u64,
    enemy_pawn_attacks: u64,
) -> bool {
    square.bit() & own_pawn_attacks != 0
        && square.bit() & enemy_pawn_attacks == 0
        && (3..=5).contains(&relative_rank(color, square))
}

/// Used for building every pawn-backed outpost square on the color-relative
/// fourth through sixth ranks.
///
/// # Arguments
///
/// * `color` - color the outposts belong to
/// * `own_pawn_attacks` - squares defended by the color's pawns
/// * `enemy_pawn_attacks` - squares attacked by enemy pawns
///
/// # Returns
///
/// Bitboard of pawn-defended, pawn-safe squares in the outpost ranks.
fn outpost_mask(color: Color, own_pawn_attacks: u64, enemy_pawn_attacks: u64) -> u64 {
    let ranks = match color {
        Color::White => rank_mask(4) | rank_mask(5) | rank_mask(6),
        Color::Black => rank_mask(5) | rank_mask(4) | rank_mask(3),
    };
    ranks & own_pawn_attacks & !enemy_pawn_attacks
}

/// Used for converting a square to a zero-based rank increasing toward
/// `color`'s promotion rank.
///
/// # Arguments
///
/// * `color` - color defining the forward direction
/// * `square` - square whose rank is converted
///
/// # Returns
///
/// Rank in `0..=7`, `0` on the color's home rank.
fn relative_rank(color: Color, square: Square) -> i32 {
    match color {
        Color::White => i32::from(square.rank()) - 1,
        Color::Black => 8 - i32::from(square.rank()),
    }
}

/// Used for testing whether a minor is directly behind one of its own pawns.
///
/// # Arguments
///
/// * `position` - position holding the pawns
/// * `color` - color owning the minor
/// * `square` - minor's square
///
/// # Returns
///
/// `true` when an own pawn stands on the square directly in front.
fn minor_behind_pawn(position: &Position, color: Color, square: Square) -> bool {
    let row_delta = if color == Color::White { -1 } else { 1 };
    square_offset(square, 0, row_delta).is_some_and(|pawn_square| {
        position.piece_at(pawn_square) == Some(Piece::new(color, PieceKind::Pawn))
    })
}

/// Used for measuring the Chebyshev king distance between two board squares.
///
/// # Arguments
///
/// * `first` - first square
/// * `second` - second square
///
/// # Returns
///
/// Maximum of the file and row distances.
fn king_distance(first: Square, second: Square) -> i32 {
    (i32::from(first.file()) - i32::from(second.file()))
        .abs()
        .max((i32::from(first.row()) - i32::from(second.row())).abs())
}

/// Used for penalizing a low-mobility bishop obstructed by same-color
/// friendly pawns.
///
/// Applies only when the bishop has at most four safe destinations and more
/// than four own pawns stand on its square color; the penalty grows with
/// each extra pawn and with each missing point of mobility.
///
/// # Arguments
///
/// * `bishop` - bishop's square, selecting its square-color mask
/// * `mobility` - bishop's safe destination count
/// * `own_pawns` - bitboard of the bishop's own pawns
///
/// # Returns
///
/// Non-negative penalty; zero when the bishop is mobile or unobstructed.
fn bad_bishop_penalty(bishop: Square, mobility: usize, own_pawns: u64) -> i32 {
    if mobility > 4 {
        return 0;
    }
    let same_color = if (bishop.file() + bishop.row()) % 2 == 0 {
        0xaa55_aa55_aa55_aa55_u64
    } else {
        0x55aa_55aa_55aa_55aa_u64
    };
    let pawn_count = bit_count(own_pawns & same_color);
    if pawn_count <= 4 {
        0
    } else {
        (pawn_count - 4) * 4 + (4 - i32::try_from(mobility).expect("mobility is at most four")) * 3
    }
}

/// Used for testing whether two squares lie on the same broad board flank.
///
/// The board splits between the d- and e-files.
///
/// # Arguments
///
/// * `first` - first square
/// * `second` - second square
///
/// # Returns
///
/// `true` when both squares are queenside or both are kingside.
fn same_flank(first: Square, second: Square) -> bool {
    (first.file() <= 3) == (second.file() <= 3)
}

/// Used for testing whether `color` retains either castling permission.
///
/// # Arguments
///
/// * `position` - position holding the castling rights
/// * `color` - color whose rights are queried
///
/// # Returns
///
/// `true` when the kingside or queenside right is still available.
fn can_castle_either(position: &Position, color: Color) -> bool {
    let rights = position.castling_rights();
    let mask = match color {
        Color::White => CastlingRights::WHITE_KINGSIDE | CastlingRights::WHITE_QUEENSIDE,
        Color::Black => CastlingRights::BLACK_KINGSIDE | CastlingRights::BLACK_QUEENSIDE,
    };
    rights.contains(mask)
}

/// Used for building the adjacent ring around a king square.
///
/// The ring contains up to eight neighbors and excludes the king's own
/// square; edge and corner kings have smaller rings.
///
/// # Arguments
///
/// * `king` - square at the ring's center
///
/// # Returns
///
/// Bitboard of the squares adjacent to `king`.
fn king_ring(king: Square) -> u64 {
    let mut adjacent = 0;
    for file_delta in -1..=1 {
        for row_delta in -1..=1 {
            if file_delta != 0 || row_delta != 0 {
                if let Some(square) = square_offset(king, file_delta, row_delta) {
                    adjacent |= square.bit();
                }
            }
        }
    }
    adjacent
}

/// Used for building every pseudo-legal knight destination from one square.
///
/// # Arguments
///
/// * `square` - knight's square
///
/// # Returns
///
/// Bitboard of the up to eight knight-move targets on the board.
fn knight_attacks(square: Square) -> u64 {
    /// Used for enumerating the eight file/row offsets of a knight move.
    const OFFSETS: [(i8, i8); 8] = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ];
    OFFSETS.iter().fold(0, |bits, &(file, row)| {
        bits | square_offset(square, file, row).map_or(0, Square::bit)
    })
}

/// Used for building diagonal slider attacks through the first occupied
/// square per ray.
///
/// # Arguments
///
/// * `square` - slider's square
/// * `occupancy` - bitboard of blocking pieces
///
/// # Returns
///
/// Bitboard of attacked squares on all four diagonals, including each ray's
/// first blocker.
fn diagonal_attacks(square: Square, occupancy: u64) -> u64 {
    ray_attacks(square, occupancy, &[(-1, -1), (-1, 1), (1, -1), (1, 1)])
}

/// Used for building orthogonal slider attacks through the first occupied
/// square per ray.
///
/// # Arguments
///
/// * `square` - slider's square
/// * `occupancy` - bitboard of blocking pieces
///
/// # Returns
///
/// Bitboard of attacked squares on the file and rank, including each ray's
/// first blocker.
fn orthogonal_attacks(square: Square, occupancy: u64) -> u64 {
    ray_attacks(square, occupancy, &[(-1, 0), (1, 0), (0, -1), (0, 1)])
}

/// Used for walking a collection of slider rays until each ray's first
/// blocker.
///
/// # Arguments
///
/// * `square` - ray origin, itself never included
/// * `occupancy` - bitboard of blocking pieces
/// * `directions` - file/row direction pairs to walk
///
/// # Returns
///
/// Bitboard of every visited square, including each ray's first blocker.
fn ray_attacks(square: Square, occupancy: u64, directions: &[(i8, i8)]) -> u64 {
    let mut attacks = 0;
    for &(file_delta, row_delta) in directions {
        let mut cursor = square;
        while let Some(target) = square_offset(cursor, file_delta, row_delta) {
            attacks |= target.bit();
            if occupancy & target.bit() != 0 {
                break;
            }
            cursor = target;
        }
    }
    attacks
}

/// Used for building all pawn attacks for a bitboard of one color's pawns.
///
/// Shifts the whole pawn set at once, masking the a- and h-files so captures
/// never wrap around the board edge.
///
/// # Arguments
///
/// * `color` - color owning the pawns, defining the attack direction
/// * `pawns` - bitboard of the color's pawns
///
/// # Returns
///
/// Bitboard of every square the pawns attack.
/// Used for scoring pawns that are defended by, or abreast of, a friendly pawn.
///
/// A pawn counts as connected when it stands on a square attacked by one of
/// its own pawns (supported) or has a friendly pawn on an adjacent file of the
/// same rank (phalanx). Both are derived with whole-bitboard algebra from the
/// two pawn sets, so the term adds no piece traversal beyond the set bits it
/// scores and no board query at all. The bonus is read from
/// [`CONNECTED_PAWN_BONUS`] at the pawn's color-relative rank and tapered, so
/// the term negates exactly under vertical color reflection.
///
/// # Arguments
///
/// * `position` - position holding the pawns
/// * `phase` - tapering phase in `0..=MAX_PHASE`
/// * `scale_percent` - percentage applied to the ramp before tapering, so one
///   sweep can locate the ramp's magnitude without changing its shape
///
/// # Returns
///
/// White-relative connected-pawn score in centipawns.
fn connected_pawn_term(position: &Position, phase: i32, scale_percent: i32) -> i32 {
    /// Used for masking the a-file so a phalanx never wraps off that edge.
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    /// Used for masking the h-file so a phalanx never wraps off that edge.
    const FILE_H: u64 = 0x8080_8080_8080_8080;

    let mut score = 0;
    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        let pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
        let phalanx = pawns & (((pawns & !FILE_H) << 1) | ((pawns & !FILE_A) >> 1));
        let supported = pawns & pawn_attack_mask(color, pawns);
        let mut connected = phalanx | supported;
        while connected != 0 {
            let index =
                u8::try_from(connected.trailing_zeros()).expect("set bit index is below 64");
            connected &= connected - 1;
            let square = board_square(index);
            let relative_rank = if color == Color::White {
                square.rank()
            } else {
                9 - square.rank()
            };
            let (middle, ending) = CONNECTED_PAWN_BONUS[usize::from(relative_rank.min(8))];
            score += sign
                * tapered_score(
                    middle * scale_percent / 100,
                    ending * scale_percent / 100,
                    phase,
                );
        }
    }
    score
}

/// Used for penalizing a bishop obstructed by its owner's own pawns.
///
/// `STR-20260806-332`. Janus's evaluation has no representation at all of
/// bishop quality: a bishop is scored by material, its piece-square entry, and
/// its mobility, so a bishop whose own pawn chain sits entirely on its own
/// square color is indistinguishable from a bishop with an open board except
/// through whatever mobility happens to survive. Every mature classical
/// evaluator carries some form of this term, and the concept — a bishop is
/// devalued by friendly pawns on its color, more so when those pawns cannot
/// advance — is the abstract mechanism borrowed here. The arithmetic, the
/// blocked-pawn definition, and every constant are Janus's own.
///
/// The multiplier grows with the number of the owner's *blocked* center pawns
/// because an obstructed bishop is only genuinely bad when the structure is
/// fixed; the same pawns in a fluid position will move off the bishop's color.
///
/// # Arguments
///
/// * `position` - position holding the bishops and pawns
/// * `phase` - tapering phase in `0..=MAX_PHASE`
/// * `scale_percent` - ramp percentage applied to both tapered ends
///
/// # Returns
///
/// White-relative penalty in centipawns, negative for the obstructed side.
fn bishop_pawn_term(position: &Position, phase: i32, scale_percent: i32) -> i32 {
    /// Used for masking one of the two square colors.
    ///
    /// Janus's bitboards are top-origin — index zero is `a8` — so this is the
    /// light-square set. Only the partition matters here, not which half is
    /// named, because the bishop selects its own half by testing its own bit.
    const LIGHT_SQUARES: u64 = 0xAA55_AA55_AA55_AA55;
    /// Used for masking the four center files whose pawns decide whether the
    /// structure is fixed.
    const CENTER_FILES: u64 = 0x3C3C_3C3C_3C3C_3C3C;
    /// Used for the penalty per obstructing pawn before the blocked multiplier.
    const BISHOP_PAWN_PENALTY: (i32, i32) = (2, 3);
    /// Used for the multiplier floor, so an unblocked structure still counts.
    const BISHOP_PAWN_BASE_UNITS: i32 = 2;

    let mut score = 0;
    let occupancy = position.occupancy();
    for color in [Color::White, Color::Black] {
        let bishops = position.piece_bitboard(Piece::new(color, PieceKind::Bishop));
        if bishops == 0 {
            continue;
        }
        let sign = color_sign(color);
        let pawns = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
        let blocked = bit_count(pawns & CENTER_FILES & pawn_push_blockers(color, occupancy));
        let multiplier = BISHOP_PAWN_BASE_UNITS + blocked;
        let mut remaining = bishops;
        while remaining != 0 {
            let index =
                u8::try_from(remaining.trailing_zeros()).expect("set bit index is below 64");
            remaining &= remaining - 1;
            let same_color = if (1_u64 << index) & LIGHT_SQUARES == 0 {
                pawns & !LIGHT_SQUARES
            } else {
                pawns & LIGHT_SQUARES
            };
            let units = bit_count(same_color) * multiplier;
            score -= sign
                * tapered_score(
                    units * BISHOP_PAWN_PENALTY.0 * scale_percent / 100,
                    units * BISHOP_PAWN_PENALTY.1 * scale_percent / 100,
                    phase,
                );
        }
    }
    score
}

/// Used for projecting occupancy onto the pawns it obstructs.
///
/// Intersecting the result with a color's pawns marks every pawn whose single
/// push is blocked, without generating a move. Janus's bitboards are
/// top-origin, so White's forward direction subtracts eight from an index and
/// the projection therefore shifts the *occupancy* left by eight.
///
/// # Arguments
///
/// * `color` - color whose pawns are being tested
/// * `squares` - occupancy to project
///
/// # Returns
///
/// `squares` shifted one rank against `color`'s advance.
const fn pawn_push_blockers(color: Color, squares: u64) -> u64 {
    match color {
        Color::White => squares << 8,
        Color::Black => squares >> 8,
    }
}

/// Used for adjusting knight and rook value by the owner's pawn count.
///
/// This is the compact form of the material-imbalance mechanism every mature
/// classical evaluator carries: a knight is worth more with pawns on the board
/// and a rook is worth less. `STR-20260804-301`'s full quadratic table and
/// `STR-20260804-303`'s closedness variant were both rejected at a static
/// teacher-trace gate that `INFRA-20260805-320` has since shown does not track
/// strength in either direction, so this narrow two-coefficient form is
/// screened in play instead of refitted against that trace.
///
/// # Arguments
///
/// * `position` - position holding the pieces
///
/// # Returns
///
/// White-relative material adjustment in centipawns, phase-independent
/// because the pawn count already carries the phase information.
fn pawn_count_material_adjustment(position: &Position) -> i32 {
    let mut score = 0;
    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        let pawns = i32::try_from(
            position
                .piece_bitboard(Piece::new(color, PieceKind::Pawn))
                .count_ones(),
        )
        .expect("pawn count fits i32");
        let knights = i32::try_from(
            position
                .piece_bitboard(Piece::new(color, PieceKind::Knight))
                .count_ones(),
        )
        .expect("knight count fits i32");
        let rooks = i32::try_from(
            position
                .piece_bitboard(Piece::new(color, PieceKind::Rook))
                .count_ones(),
        )
        .expect("rook count fits i32");
        let excess = pawns - 5;
        score += sign
            * (knights * KNIGHT_PAWN_COUNT_ADJUSTMENT + rooks * ROOK_PAWN_COUNT_ADJUSTMENT)
            * excess;
    }
    score
}

/// Used for penalizing pawns that can neither advance safely nor be supported.
///
/// A pawn counts as backward when the square directly ahead of it is attacked
/// by an enemy pawn *and* no friendly pawn on an adjacent file stands at or
/// behind its rank, so no neighbour can ever advance to defend the push. Both
/// conditions are whole-bitboard derivations from the two pawn sets: the
/// forward fill of the sideways-projected own pawns marks every square that
/// has a neighbour at or behind it, and the enemy pawn-attack mask supplies
/// the unsafe stop squares.
///
/// # Arguments
///
/// * `position` - position holding the pawns
/// * `phase` - tapering phase in `0..=MAX_PHASE`
/// * `penalty` - `(middlegame, endgame)` penalty per backward pawn, so one
///   sweep can test the taper's shape rather than only its size
///
/// # Returns
///
/// White-relative backward-pawn score in centipawns, never positive for the
/// side owning the backward pawns.
fn backward_pawn_term(position: &Position, phase: i32, penalty: (i32, i32)) -> i32 {
    /// Used for masking the a-file so a sideways projection never wraps.
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    /// Used for masking the h-file so a sideways projection never wraps.
    const FILE_H: u64 = 0x8080_8080_8080_8080;

    let mut score = 0;
    for color in [Color::White, Color::Black] {
        let sign = color_sign(color);
        let own = position.piece_bitboard(Piece::new(color, PieceKind::Pawn));
        if own == 0 {
            continue;
        }
        let enemy_color = color.opposite();
        let enemy = position.piece_bitboard(Piece::new(enemy_color, PieceKind::Pawn));
        let enemy_attacks = pawn_attack_mask(enemy_color, enemy);
        let neighbours = ((own & !FILE_H) << 1) | ((own & !FILE_A) >> 1);
        let (stops, support_span) = if color == Color::White {
            let mut span = neighbours;
            span |= span >> 8;
            span |= span >> 16;
            span |= span >> 32;
            (own >> 8, span)
        } else {
            let mut span = neighbours;
            span |= span << 8;
            span |= span << 16;
            span |= span << 32;
            (own << 8, span)
        };
        let unsafe_stops = stops & enemy_attacks;
        let blocked = if color == Color::White {
            unsafe_stops << 8
        } else {
            unsafe_stops >> 8
        };
        let backward = own & blocked & !support_span;
        let count = i32::try_from(backward.count_ones()).expect("pawn count fits i32");
        score -= sign * count * tapered_score(penalty.0, penalty.1, phase);
    }
    score
}

fn pawn_attack_mask(color: Color, pawns: u64) -> u64 {
    /// Used for masking the a-file so captures never wrap off that edge.
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    /// Used for masking the h-file so captures never wrap off that edge.
    const FILE_H: u64 = 0x8080_8080_8080_8080;
    match color {
        Color::White => ((pawns & !FILE_A) >> 9) | ((pawns & !FILE_H) >> 7),
        Color::Black => ((pawns & !FILE_A) << 7) | ((pawns & !FILE_H) << 9),
    }
}

/// Used for building empty one- and two-step pawn-push destinations for one
/// color.
///
/// Double pushes are only generated through an empty intermediate square
/// on the color's third rank, mirroring legal pawn movement.
///
/// # Arguments
///
/// * `color` - color owning the pawns, defining the push direction
/// * `pawns` - bitboard of the color's pawns
/// * `empty` - bitboard of unoccupied squares
///
/// # Returns
///
/// Bitboard of reachable single- and double-push destinations.
fn pawn_push_mask(color: Color, pawns: u64, empty: u64) -> u64 {
    match color {
        Color::White => {
            let single = (pawns >> 8) & empty;
            single | (((single & rank_mask(3)) >> 8) & empty)
        }
        Color::Black => {
            let single = (pawns << 8) & empty;
            single | (((single & rank_mask(6)) << 8) & empty)
        }
    }
}

/// Used for building the three-file mask centered as closely as possible on
/// a king file.
///
/// Kings on the a- or h-file use the b- or g-file as center so the mask
/// always spans exactly three files.
///
/// # Arguments
///
/// * `king` - king's square
///
/// # Returns
///
/// Bitboard covering the three flank files.
fn king_flank_mask(king: Square) -> u64 {
    let center = king.file().clamp(1, 6);
    file_mask(center - 1) | file_mask(center) | file_mask(center + 1)
}

/// Used for building the home-side five ranks used for king-flank pressure.
///
/// # Arguments
///
/// * `color` - color whose camp is masked
///
/// # Returns
///
/// Bitboard of the five ranks nearest the color's home rank.
const fn camp_mask(color: Color) -> u64 {
    match color {
        Color::White => 0xffff_ffffff_000000,
        Color::Black => 0x0000_00ff_ffff_ffff,
    }
}

/// Used for building every square on one zero-based file.
///
/// # Arguments
///
/// * `file` - zero-based file in `0..=7`
///
/// # Returns
///
/// Bitboard of the eight squares on the file.
const fn file_mask(file: u8) -> u64 {
    0x0101_0101_0101_0101_u64 << file
}

/// Used for splitting a passer's file into strict forward and rear rays.
///
/// Bit indices increase from the eighth rank toward the first, so White's
/// forward ray is the lower-index half and Black's is the higher-index half.
/// The passer square itself is excluded from both masks.
///
/// # Arguments
///
/// * `square` - passed-pawn square splitting the file
/// * `color` - owner defining forward and rear directions
///
/// # Returns
///
/// `(forward, rear)` file masks in the owner's orientation.
fn passed_file_masks(square: Square, color: Color) -> (u64, u64) {
    let bit = square.bit();
    let lower = file_mask(square.file()) & bit.wrapping_sub(1);
    let higher = file_mask(square.file()) & !(bit | bit.wrapping_sub(1));
    match color {
        Color::White => (lower, higher),
        Color::Black => (higher, lower),
    }
}

/// Used for building every square on one human rank in `1..=8`.
///
/// # Arguments
///
/// * `rank` - human rank in `1..=8`
///
/// # Returns
///
/// Bitboard of the eight squares on the rank.
const fn rank_mask(rank: u8) -> u64 {
    0xff_u64 << ((8 - rank) * 8)
}

/// Used for building the four central squares consumed by the bishop
/// activity oracle.
///
/// # Returns
///
/// Bitboard of the d4, e4, d5, and e5 bit indices.
const fn center_squares() -> u64 {
    (1_u64 << 27) | (1_u64 << 28) | (1_u64 << 35) | (1_u64 << 36)
}

/// Used for applying a board-coordinate offset, returning `None` across an
/// edge.
///
/// # Arguments
///
/// * `square` - starting square
/// * `file_delta` - signed file offset
/// * `row_delta` - signed row offset
///
/// # Returns
///
/// Offset square, or `None` when the result leaves the board.
fn square_offset(square: Square, file_delta: i8, row_delta: i8) -> Option<Square> {
    let file = i16::from(square.file()) + i16::from(file_delta);
    let row = i16::from(square.row()) + i16::from(row_delta);
    if !(0..8).contains(&file) || !(0..8).contains(&row) {
        return None;
    }
    Square::from_file_row(
        u8::try_from(file).expect("checked file fits u8"),
        u8::try_from(row).expect("checked row fits u8"),
    )
}

/// Used for converting a known bitboard index into its square.
///
/// # Arguments
///
/// * `index` - bitboard index in `0..64`
///
/// # Returns
///
/// Square for the index.
///
/// # Panics
///
/// Panics when `index` is 64 or above.
fn board_square(index: u8) -> Square {
    Square::new(index).expect("board index is below 64")
}

/// Used for counting set bits in the signed score type used by evaluation
/// formulas.
///
/// # Arguments
///
/// * `bits` - bitboard to count
///
/// # Returns
///
/// Population count in `0..=64` as an `i32`.
fn bit_count(bits: u64) -> i32 {
    i32::try_from(bits.count_ones()).expect("a bitboard has at most 64 bits")
}

/// Research-only parameterized twin of the released classical evaluation.
///
/// This module exists for offline Texel-style tuning of the linear classical
/// terms. It exposes [`tuning::ClassicalParams`], an exact integer
/// re-evaluation path ([`tuning::params_breakdown`]) whose
/// [`Default`](tuning::ClassicalParams::default) reproduces the released
/// constants bit-for-bit, and a sparse linear-coefficient extractor
/// ([`tuning::linear_model`]) for gradient-based fitting. Nothing here is
/// used by the engine's release evaluation path.
#[doc(hidden)]
pub mod tuning;

