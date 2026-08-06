//! Error type shared by fallible chess-core operations.
//!
//! Fallible square, move, and position operations in this crate report
//! failure through the single [`CoreError`] type, which carries one
//! human-readable diagnostic string. Keeping one general-purpose error type
//! keeps the crate's fallible API surface uniform for downstream callers;
//! only the perft helpers use the separate [`crate::PerftError`].

use core::fmt;

/// Error value describing a malformed position, square, or move.
///
/// Instances carry a single stable, human-readable diagnostic string and are
/// returned by the crate's fallible square, move, and position operations.
/// The type is cloneable and comparable so callers can inspect, propagate, or
/// store failures freely.
// The established public name remains explicit when re-exported from
// `janus-core`; shortening it to `Error` would make downstream imports vague.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoreError {
    /// Used for storing the stable human-readable diagnostic owned by the
    /// error.
    ///
    /// The text is written verbatim by the [`fmt::Display`] implementation.
    message: String,
}

impl CoreError {
    /// Used for creating an error from an owned or borrowed diagnostic.
    ///
    /// # Arguments
    ///
    /// * `message` - human-readable diagnostic text, accepted as anything
    ///   convertible into a `String`
    ///
    /// # Returns
    ///
    /// New error owning the converted diagnostic.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Used for retrieving the human-readable error text.
    ///
    /// # Returns
    ///
    /// Borrowed diagnostic string owned by this error.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CoreError {
    /// Used for writing the diagnostic without an additional prefix.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter for the diagnostic text
    ///
    /// # Returns
    ///
    /// Result of writing the message to `formatter`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CoreError {}
