#![allow(renamed_and_removed_lints)]
#![allow(
    // Board indices and bit scans are range-proven by private invariants. These
    // explicit casts keep hot move-generation code readable on Rust 1.75.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    // Public fallible methods have concise contract docs; repeating every
    // malformed-input variant adds noise without changing their CoreError API.
    clippy::missing_errors_doc,
    // Geometry unwraps operate on values already constrained to the 8x8 board.
    clippy::missing_panics_doc,
    // Strict FEN parsing validates the vector length before fixed-field access.
    // (Lint removed in newer clippy; renamed_and_removed_lints below keeps this
    // list valid across the 1.75/1.97 toolchain boundary.)
    clippy::match_on_vec_items,
    // Adding must_use to every cheap query obscures the state-changing methods.
    clippy::must_use_candidate
)]
#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

//! Deterministic, dependency-free chess rules for Janus.
//!
//! This crate is the rules boundary shared by Janus's search and evaluation
//! layers. It owns position parsing, legal move generation, reversible state
//! transitions, repetition keys, and perft. It performs no I/O and depends only
//! on the Rust standard library, so callers receive deterministic results for a
//! given position and move sequence.
//!
//! Squares deliberately match `ChessRTK`'s layout: [`Square::A8`] is index `0`
//! and [`Square::H1`] is index `63`. [`Position`] keeps a mailbox, per-piece
//! bitboards, color occupancy, and cached king squares synchronized. [`Move`]
//! uses the same compact 16-bit representation as the Java core.
//!
//! # Example
//!
//! ```
//! use janus_core::Position;
//!
//! let mut position = Position::start();
//! let mv = position.parse_uci_move("e2e4")?;
//! let undo = position.make_move(mv)?;
//! assert_eq!(position.side_to_move().to_string(), "black");
//! position.unmake_move(mv, undo);
//! assert_eq!(position, Position::start());
//! # Ok::<(), janus_core::CoreError>(())
//! ```

/// Error values returned when encoded chess data is malformed or inapplicable.
///
/// Home of [`CoreError`], the general-purpose error type re-exported at the
/// crate root and shared by the crate's fallible square, move, and position
/// operations.
mod error;
/// Compact, `ChessRTK`-compatible move encoding.
///
/// Home of [`Move`] and the [`NO_MOVE`] sentinel, together with UCI
/// long-algebraic parsing and formatting.
mod mv;
/// Correctness tests for legal move generation.
///
/// Provides perft node counting and divide breakdowns, including the checked
/// and detailed variants re-exported at the crate root.
mod perft;
/// Position representation, state transitions, and move generation.
///
/// Home of [`Position`], [`CastlingRights`], and the [`Undo`]/[`NullUndo`]
/// tokens used to reverse make/unmake state transitions.
mod position;
/// Board-coordinate representation and algebraic conversion.
///
/// Home of [`Square`], which fixes the `ChessRTK` index layout (`a8 = 0`,
/// `h1 = 63`) used throughout the crate.
mod square;
/// Fundamental color and piece value types.
///
/// Home of [`Color`], [`PieceKind`], and [`Piece`], whose stable index
/// contracts underpin the position's storage layout.
mod types;

pub use error::CoreError;
pub use mv::{Move, NO_MOVE};
pub use perft::{
    checked_divide, checked_perft, detailed_divide, detailed_perft, divide, perft,
    PerftDivideEntry, PerftDivideResult, PerftError, PerftNodeEntry, PerftNodeResult, PerftStats,
    MAX_PERFT_DEPTH,
};
pub use position::{CastlingRights, NullUndo, Position, Undo, START_FEN};
pub use square::Square;
pub use types::{Color, Piece, PieceKind};

/// Magic-bitboard slider attacks, shared with evaluation.
///
/// Alternative configuration retained for controlled evaluation.
pub mod sliding_attacks {
    pub use crate::position::movegen::{bishop_attacks, rook_attacks};
}
