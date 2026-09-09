//! Bounded, allocation-conscious parsing for the UCI command surface.
//!
//! Input lines are read through [`crate::protocol::read_line_bounded`], which
//! enforces the [`crate::protocol::MAX_LINE_BYTES`] limit before any buffer
//! can grow, and are converted into [`Command`] values by
//! [`crate::protocol::parse`]. Unknown command names are preserved as
//! [`Command::Unknown`] rather than rejected so the shell can emit
//! diagnostics without breaking the protocol stream.

use std::fmt;
use std::io::{self, BufRead};

/// Used for bounding the accepted input line length in bytes, excluding the
/// newline terminator.
///
/// Both [`read_line_bounded`] and [`parse`] reject longer input, so an
/// attacker-controlled stream can never force unbounded buffering.
pub const MAX_LINE_BYTES: usize = 16 * 1024;

/// A parsed UCI command.
///
/// One value corresponds to one input line accepted by [`parse`]. Commands
/// carrying structured arguments wrap them in dedicated payload types such as
/// [`PositionSpec`], [`GoOptions`], and [`PerftOptions`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    /// Used for starting UCI identification and option discovery.
    Uci,
    /// Used for requesting an immediate readiness acknowledgement.
    IsReady,
    /// Used for announcing that the GUI has started a new game.
    NewGame,
    /// Used for changing one named engine option.
    SetOption {
        /// Used for carrying the option name after UCI whitespace
        /// normalization.
        name: String,
        /// Used for carrying the option value, or an empty string for button
        /// options.
        value: String,
    },
    /// Used for replacing the current position and move history.
    Position(PositionSpec),
    /// Used for starting a search with the supplied constraints.
    Go(GoOptions),
    /// Used for starting one bounded developer perft enumeration.
    Perft(PerftOptions),
    /// Used for requesting termination of the active search.
    Stop,
    /// Used for converting an active ponder search into a normal search.
    PonderHit,
    /// Used for terminating the engine process.
    Quit,
    /// Used for enabling or disabling protocol diagnostics.
    Debug(bool),
    /// Used for preserving the first token of an unsupported command for
    /// diagnostics.
    Unknown(String),
}

/// Root position and optional move history from a UCI `position` command.
///
/// The board described by [`PositionSpec::source`] is constructed first and
/// the moves in [`PositionSpec::moves`] are then applied in order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionSpec {
    /// Used for describing the initial board, either `startpos` or a
    /// complete FEN string.
    pub source: PositionSource,
    /// Used for holding the UCI moves applied in order after constructing
    /// `source`.
    pub moves: Vec<String>,
}

/// Position source selected by UCI.
///
/// Distinguishes the implicit standard starting position from an explicit
/// FEN description supplied on the `position` command line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PositionSource {
    /// Used for selecting the standard chess starting position.
    StartPos,
    /// Used for carrying complete FEN text exactly as reconstructed from
    /// command tokens.
    Fen(String),
}

/// Search constraints accepted by UCI `go`.
///
/// All limits are optional and compose; the default value places no
/// constraint on the search. Unset numeric limits are `None`, and unset flag
/// fields are `false`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GoOptions {
    /// Used for capping the iterative-deepening depth in plies.
    pub depth: Option<u8>,
    /// Used for capping the aggregate number of visited nodes.
    pub nodes: Option<u64>,
    /// Used for requesting an exact per-move search time in milliseconds.
    pub move_time_ms: Option<u64>,
    /// Used for reporting White's remaining clock time in milliseconds.
    pub white_time_ms: Option<u64>,
    /// Used for reporting Black's remaining clock time in milliseconds.
    pub black_time_ms: Option<u64>,
    /// Used for reporting White's per-move increment in milliseconds.
    pub white_increment_ms: Option<u64>,
    /// Used for reporting Black's per-move increment in milliseconds.
    pub black_increment_ms: Option<u64>,
    /// Used for estimating the moves remaining until the next time control.
    pub moves_to_go: Option<u32>,
    /// Used for requesting a mate-search horizon in full moves.
    pub mate: Option<u8>,
    /// Used for indicating that search continues until an explicit
    /// [`Command::Stop`].
    pub infinite: bool,
    /// Used for indicating that the engine is thinking before the opponent
    /// has moved.
    pub ponder: bool,
    /// Used for restricting the search to an optional legal root-move
    /// whitelist in UCI notation.
    pub search_moves: Vec<String>,
}

/// Output family selected for the developer perft command.
///
/// Chosen by the optional trailing token of `go perft <depth> [format]`;
/// the default is [`PerftFormat::Detail`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerftFormat {
    /// Used for printing one aggregate with all seven deterministic counters
    /// and root status.
    Detail,
    /// Used for printing one all-counter row per root move followed by a
    /// checked total.
    Divide,
    /// Used for printing Stockfish-compatible node-only root rows and a
    /// final total.
    Stockfish,
}

/// Bounded arguments accepted after `go perft`.
///
/// Produced by [`parse`] for the developer perft extension; depth bounds are
/// enforced by the shell rather than the parser.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PerftOptions {
    /// Used for requesting the enumeration depth in plies.
    pub depth: u32,
    /// Used for requesting the deterministic output family.
    pub format: PerftFormat,
}

/// User-facing protocol parse error.
///
/// Wraps a single human-readable diagnostic line describing why a recognized
/// command could not be parsed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError(String);

impl ParseError {
    /// Used for creating a protocol diagnostic from owned or borrowed text.
    ///
    /// # Arguments
    ///
    /// * `message` - human-readable diagnostic text
    ///
    /// # Returns
    ///
    /// A new [`ParseError`] wrapping the message.
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ParseError {
    /// Used for writing the human-readable parser diagnostic.
    ///
    /// # Arguments
    ///
    /// * `formatter` - destination formatter for the diagnostic text
    ///
    /// # Returns
    ///
    /// The formatter result of writing the wrapped message.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

/// Used for reading one UTF-8 command line while never buffering more than
/// [`MAX_LINE_BYTES`] bytes plus an LF or CRLF line terminator.
///
/// The returned string excludes CR/LF terminators. Both LF- and
/// CRLF-terminated lines may carry exactly [`MAX_LINE_BYTES`] content bytes.
/// An oversized or non-UTF-8 command is a fatal protocol input error so
/// callers do not need to drain an attacker-controlled line.
///
/// # Arguments
///
/// * `reader` - buffered byte source to read one line from
/// * `output` - destination string replaced with the line content
///
/// # Returns
///
/// `Ok(true)` when a line was read into `output`, or `Ok(false)` at end of
/// input with no remaining data.
///
/// # Errors
///
/// Returns an I/O error from `reader`, or [`io::ErrorKind::InvalidData`] when
/// the line does not fit the [`MAX_LINE_BYTES`] budget or is not valid
/// UTF-8.
pub fn read_line_bounded<R: BufRead>(reader: &mut R, output: &mut String) -> io::Result<bool> {
    let mut bytes = Vec::with_capacity(256);
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if bytes.is_empty() {
                output.clear();
                return Ok(false);
            }
            break;
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(available.len(), |index| index + 1);
        if bytes.len().saturating_add(take) > MAX_LINE_BYTES + 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "UCI input line exceeds 16384 bytes",
            ));
        }
        bytes.extend_from_slice(&available[..take]);
        reader.consume(take);
        if newline.is_some() {
            break;
        }
    }
    while matches!(bytes.last(), Some(b'\n' | b'\r')) {
        bytes.pop();
    }
    if bytes.len() > MAX_LINE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "UCI input line exceeds 16384 bytes",
        ));
    }
    *output = String::from_utf8(bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "UCI input is not UTF-8"))?;
    Ok(true)
}

/// Used for parsing one line without its trailing newline.
///
/// Empty or whitespace-only input returns `Ok(None)`. Unknown command names
/// are preserved as [`Command::Unknown`] rather than rejected.
///
/// # Arguments
///
/// * `line` - one command line without its trailing newline
///
/// # Returns
///
/// `Ok(Some(command))` for a recognized or unknown command, or `Ok(None)`
/// for blank input.
///
/// # Errors
///
/// Returns [`ParseError`] when a recognized command has malformed structure,
/// an invalid numeric value, an unsupported `go` token, or excessive length.
pub fn parse(line: &str) -> Result<Option<Command>, ParseError> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    if line.len() > MAX_LINE_BYTES {
        return Err(ParseError::new("UCI input line exceeds 16384 bytes"));
    }

    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    let Some(command_name) = tokens.first().copied() else {
        return Ok(None);
    };
    let command = match command_name {
        "uci" => Command::Uci,
        "isready" => Command::IsReady,
        "ucinewgame" => Command::NewGame,
        "stop" => Command::Stop,
        "ponderhit" => Command::PonderHit,
        "quit" => Command::Quit,
        "debug" => Command::Debug(parse_debug(&tokens)?),
        "setoption" => parse_set_option(&tokens)?,
        "position" => Command::Position(parse_position(&tokens)?),
        "go" if tokens.get(1) == Some(&"perft") => Command::Perft(parse_perft(&tokens)?),
        "go" => Command::Go(parse_go(&tokens)?),
        other => Command::Unknown(other.to_owned()),
    };
    Ok(Some(command))
}

/// Used for parsing the exact `go perft <depth> [format]` developer
/// extension.
///
/// The optional format token selects one of the three [`PerftFormat`]
/// families and defaults to `detail`.
///
/// # Arguments
///
/// * `tokens` - whitespace-split tokens of the whole command line
///
/// # Returns
///
/// The parsed perft depth and output format.
///
/// # Errors
///
/// Returns [`ParseError`] when the token count is not three or four, the
/// depth is not a valid `u32`, or the format token is unsupported.
fn parse_perft(tokens: &[&str]) -> Result<PerftOptions, ParseError> {
    if !(tokens.len() == 3 || tokens.len() == 4) {
        return Err(ParseError::new(
            "go perft expects <depth> [detail|divide|stockfish]",
        ));
    }
    let depth = tokens[2]
        .parse::<u32>()
        .map_err(|_| ParseError::new("invalid go perft depth"))?;
    let format = match tokens.get(3).copied().unwrap_or("detail") {
        "detail" => PerftFormat::Detail,
        "divide" => PerftFormat::Divide,
        "stockfish" => PerftFormat::Stockfish,
        other => {
            return Err(ParseError::new(format!(
                "unsupported go perft format: {other}"
            )));
        }
    };
    Ok(PerftOptions { depth, format })
}

/// Used for parsing the required `on` or `off` argument of a `debug`
/// command.
///
/// # Arguments
///
/// * `tokens` - whitespace-split tokens of the whole command line
///
/// # Returns
///
/// `true` for `debug on` and `false` for `debug off`.
///
/// # Errors
///
/// Returns [`ParseError`] when the argument is missing or is neither `on`
/// nor `off`.
fn parse_debug(tokens: &[&str]) -> Result<bool, ParseError> {
    match tokens.get(1).copied() {
        Some("on") => Ok(true),
        Some("off") => Ok(false),
        _ => Err(ParseError::new("debug expects 'on' or 'off'")),
    }
}

/// Used for parsing multiword UCI option names and optional multiword
/// values.
///
/// Tokens between `name` and `value` are joined with single spaces to form
/// the option name; everything after `value` is joined the same way. A
/// missing `value` clause yields an empty value string.
///
/// # Arguments
///
/// * `tokens` - whitespace-split tokens of the whole command line
///
/// # Returns
///
/// A [`Command::SetOption`] carrying the normalized name and value.
///
/// # Errors
///
/// Returns [`ParseError`] when the `name` keyword is missing or the option
/// name is empty.
fn parse_set_option(tokens: &[&str]) -> Result<Command, ParseError> {
    if tokens.get(1) != Some(&"name") {
        return Err(ParseError::new("setoption expects 'name'"));
    }
    let value_index = tokens.iter().position(|token| *token == "value");
    let name_end = value_index.unwrap_or(tokens.len());
    if name_end <= 2 {
        return Err(ParseError::new("setoption has an empty name"));
    }
    let name = tokens[2..name_end].join(" ");
    let value = value_index.map_or_else(String::new, |index| tokens[index + 1..].join(" "));
    Ok(Command::SetOption { name, value })
}

/// Used for parsing a start position or FEN followed by an optional move
/// history.
///
/// FEN fields between `fen` and `moves` are rejoined with single spaces to
/// reconstruct the original FEN text; tokens after `moves` become the move
/// history verbatim.
///
/// # Arguments
///
/// * `tokens` - whitespace-split tokens of the whole command line
///
/// # Returns
///
/// The parsed position source and move list.
///
/// # Errors
///
/// Returns [`ParseError`] when the source keyword is neither `startpos` nor
/// `fen`, when `startpos` is followed by unexpected fields before `moves`,
/// or when `fen` has no FEN fields.
fn parse_position(tokens: &[&str]) -> Result<PositionSpec, ParseError> {
    let moves_index = tokens.iter().position(|token| *token == "moves");
    let source_end = moves_index.unwrap_or(tokens.len());
    let source = match tokens.get(1).copied() {
        Some("startpos") if source_end == 2 => PositionSource::StartPos,
        Some("fen") if source_end > 2 => PositionSource::Fen(tokens[2..source_end].join(" ")),
        Some("startpos") => {
            return Err(ParseError::new(
                "position startpos has unexpected fields before moves",
            ));
        }
        _ => return Err(ParseError::new("position expects startpos or fen")),
    };
    let moves = moves_index.map_or_else(Vec::new, |index| {
        tokens[index + 1..]
            .iter()
            .map(|token| (*token).to_owned())
            .collect()
    });
    Ok(PositionSpec { source, moves })
}

/// Used for parsing supported `go` flags, numeric constraints, and root
/// moves.
///
/// Tokens are consumed left to right; `searchmoves` collects following
/// tokens until the next recognized `go` keyword.
///
/// # Arguments
///
/// * `tokens` - whitespace-split tokens of the whole command line
///
/// # Returns
///
/// The accumulated search constraints; a bare `go` yields the default
/// (unconstrained) options.
///
/// # Errors
///
/// Returns [`ParseError`] on an unsupported token, a missing or invalid
/// numeric value, or a `searchmoves` clause with no moves.
fn parse_go(tokens: &[&str]) -> Result<GoOptions, ParseError> {
    let mut options = GoOptions::default();
    let mut index = 1;
    while index < tokens.len() {
        let token = tokens[index];
        match token {
            "infinite" => options.infinite = true,
            "ponder" => options.ponder = true,
            "depth" => options.depth = Some(parse_number(tokens, &mut index, token)?),
            "nodes" => options.nodes = Some(parse_number(tokens, &mut index, token)?),
            "movetime" => {
                options.move_time_ms = Some(parse_number(tokens, &mut index, token)?);
            }
            "wtime" => options.white_time_ms = Some(parse_number(tokens, &mut index, token)?),
            "btime" => options.black_time_ms = Some(parse_number(tokens, &mut index, token)?),
            "winc" => {
                options.white_increment_ms = Some(parse_number(tokens, &mut index, token)?);
            }
            "binc" => {
                options.black_increment_ms = Some(parse_number(tokens, &mut index, token)?);
            }
            "movestogo" => {
                options.moves_to_go = Some(parse_number(tokens, &mut index, token)?);
            }
            "mate" => options.mate = Some(parse_number(tokens, &mut index, token)?),
            "searchmoves" => {
                index += 1;
                while index < tokens.len() && !is_go_keyword(tokens[index]) {
                    options.search_moves.push(tokens[index].to_owned());
                    index += 1;
                }
                if options.search_moves.is_empty() {
                    return Err(ParseError::new("go searchmoves expects at least one move"));
                }
                continue;
            }
            unknown => {
                return Err(ParseError::new(format!("unsupported go token: {unknown}")));
            }
        }
        index += 1;
    }
    Ok(options)
}

/// Used for consuming and parsing the numeric token following a `go`
/// keyword.
///
/// Advances `index` onto the consumed value token; the caller's loop
/// increment then steps past it.
///
/// # Arguments
///
/// * `tokens` - whitespace-split tokens of the whole command line
/// * `index` - position of the keyword; advanced to the consumed value
/// * `name` - keyword name used in diagnostic messages
///
/// # Returns
///
/// The parsed numeric value.
///
/// # Errors
///
/// Returns [`ParseError`] when the value token is missing or does not parse
/// as `T`.
fn parse_number<T>(tokens: &[&str], index: &mut usize, name: &str) -> Result<T, ParseError>
where
    T: std::str::FromStr,
{
    *index += 1;
    tokens
        .get(*index)
        .ok_or_else(|| ParseError::new(format!("go {name} expects a value")))?
        .parse::<T>()
        .map_err(|_| ParseError::new(format!("invalid go {name} value")))
}

/// Used for deciding whether `token` starts another supported `go` option.
///
/// Terminates the greedy `searchmoves` token collection in [`parse_go`].
///
/// # Arguments
///
/// * `token` - candidate token from the `go` command line
///
/// # Returns
///
/// `true` when the token is one of the recognized `go` keywords.
fn is_go_keyword(token: &str) -> bool {
    matches!(
        token,
        "searchmoves"
            | "ponder"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "depth"
            | "nodes"
            | "mate"
            | "movetime"
            | "infinite"
    )
}
