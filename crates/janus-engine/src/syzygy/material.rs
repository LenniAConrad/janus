//! Material keys, table naming, and the tablebase registry.
//!
//! Every table is identified by a nibble-packed material key counting the
//! pieces of each color and kind. At initialization the registry enumerates
//! every canonical material configuration of three to seven men, checks the
//! configured directories for the corresponding `.rtbw` (and `.rtbz`) file,
//! and records the ones that exist. Table metadata is parsed lazily on
//! first probe through [`std::sync::OnceLock`], so a large set costs no
//! parsing until its tables are actually reached.

use super::encode::EncodeTables;
use super::file::BoundedFile;
use super::table::{load_table, LoadedTable, TableKind, TableMeta};
use super::SyzygyError;
use janus_core::{Color, Piece, PieceKind, Position};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// Maximum number of distinct directories admitted from one tablebase path
/// list.
///
/// Ordinary installations use one or a few roots. This generous ceiling
/// keeps discovery bounded because every admitted root is checked against
/// every canonical three-to-seven-man table name.
pub(crate) const MAX_SYZYGY_DIRECTORIES: usize = 64;

/// Used for computing the nibble-packed material key of a position.
///
/// Each of the twelve (color, kind) piece classes contributes its count to
/// one hexadecimal nibble, so two positions share a key exactly when they
/// share a material distribution.
///
/// # Arguments
///
/// * `position` - position whose material is summarized
///
/// # Returns
///
/// Material key with white piece counts in the low six nibbles.
pub(crate) fn position_material_key(position: &Position) -> u64 {
    let mut key = 0u64;
    for color in [Color::White, Color::Black] {
        for kind in PieceKind::ALL {
            let count = u64::from(
                position
                    .piece_bitboard(Piece::new(color, kind))
                    .count_ones(),
            );
            key += count << nibble_shift(color, kind);
        }
    }
    key
}

/// Used for computing the material key of two explicit piece lists.
///
/// Kings are implicit: each side contributes one king in addition to the
/// listed pieces.
///
/// # Arguments
///
/// * `white` - white's non-king pieces
/// * `black` - black's non-king pieces
///
/// # Returns
///
/// Material key matching [`position_material_key`] for the same material.
pub(crate) fn side_lists_key(white: &[PieceKind], black: &[PieceKind]) -> u64 {
    let mut key = 1u64 << nibble_shift(Color::White, PieceKind::King);
    key += 1u64 << nibble_shift(Color::Black, PieceKind::King);
    for kind in white {
        key += 1u64 << nibble_shift(Color::White, *kind);
    }
    for kind in black {
        key += 1u64 << nibble_shift(Color::Black, *kind);
    }
    key
}

/// Used for locating a piece class's nibble inside the material key.
///
/// # Arguments
///
/// * `color` - piece color
/// * `kind` - piece kind
///
/// # Returns
///
/// Bit shift of the class's four-bit counter.
fn nibble_shift(color: Color, kind: PieceKind) -> u64 {
    u64::try_from((color.index() * 6 + kind.index()) * 4).expect("class index fits u64")
}

/// Used for rendering one side's pieces as table-name letters.
///
/// # Arguments
///
/// * `pieces` - the side's non-king pieces sorted by descending strength
///
/// # Returns
///
/// `K` followed by one uppercase letter per piece.
pub(crate) fn side_code(pieces: &[PieceKind]) -> String {
    let mut code = String::from("K");
    for kind in pieces {
        code.push(match kind {
            PieceKind::Queen => 'Q',
            PieceKind::Rook => 'R',
            PieceKind::Bishop => 'B',
            PieceKind::Knight => 'N',
            _ => 'P',
        });
    }
    code
}

/// Used for ordering piece kinds by conventional table-name strength.
///
/// # Arguments
///
/// * `kind` - piece kind to weigh
///
/// # Returns
///
/// Higher values for stronger pieces (queen five down to pawn one).
fn strength(kind: PieceKind) -> u8 {
    match kind {
        PieceKind::Queen => 5,
        PieceKind::Rook => 4,
        PieceKind::Bishop => 3,
        PieceKind::Knight => 2,
        PieceKind::Pawn => 1,
        PieceKind::King => 6,
    }
}

/// Used for enumerating every canonical material configuration up to seven
/// men.
///
/// Each configuration pairs the two sides' non-king piece multisets, both
/// sorted by descending strength. The pair is canonical when the first side
/// has more pieces, or equally many pieces and the lexicographically
/// stronger multiset; symmetric configurations appear once.
///
/// # Returns
///
/// Deterministically ordered list of canonical `(first, second)` piece
/// lists.
pub(crate) fn canonical_material_sets() -> Vec<(Vec<PieceKind>, Vec<PieceKind>)> {
    let kinds = [
        PieceKind::Queen,
        PieceKind::Rook,
        PieceKind::Bishop,
        PieceKind::Knight,
        PieceKind::Pawn,
    ];
    // Multisets sorted by descending strength, grouped by size 0..=5.
    let mut multisets: Vec<Vec<PieceKind>> = vec![Vec::new()];
    let mut frontier: Vec<Vec<PieceKind>> = vec![Vec::new()];
    for _ in 0..5 {
        let mut next = Vec::new();
        for prefix in &frontier {
            let start = prefix.last().map_or(0, |last| {
                kinds.iter().position(|k| k == last).expect("known")
            });
            for kind in &kinds[start..] {
                let mut extended = prefix.clone();
                extended.push(*kind);
                next.push(extended);
            }
        }
        multisets.extend(next.iter().cloned());
        frontier = next;
    }

    let mut sets = Vec::new();
    for first in &multisets {
        for second in &multisets {
            let men = first.len() + second.len();
            if men == 0 || men > 5 {
                continue;
            }
            let first_ranks: Vec<u8> = first.iter().map(|kind| strength(*kind)).collect();
            let second_ranks: Vec<u8> = second.iter().map(|kind| strength(*kind)).collect();
            let canonical = match first.len().cmp(&second.len()) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => first_ranks >= second_ranks,
            };
            if canonical {
                sets.push((first.clone(), second.clone()));
            }
        }
    }
    sets
}

/// Source of one physical table's bytes.
///
/// Disk paths cover the normal case; the memory variant backs synthetic
/// tables assembled by unit tests without touching the filesystem.
pub(crate) enum TableSource {
    /// Used for on-disk `.rtbw`/`.rtbz` files.
    Path(PathBuf),
}

impl TableSource {
    /// Used for opening the source as a bounded file.
    ///
    /// # Returns
    ///
    /// The opened bounded file.
    ///
    /// # Errors
    ///
    /// Returns [`super::SyzygyError`] when a disk source cannot be opened.
    fn open(&self) -> Result<BoundedFile, super::SyzygyError> {
        match self {
            Self::Path(path) => BoundedFile::open(path),
        }
    }
}

/// One registered material configuration and its lazily loaded tables.
pub(crate) struct TableEntry {
    /// Used for identifying the configuration with the first-named side as
    /// white; the color-swapped key exists only in the registry's lookup
    /// map.
    pub key: u64,
    /// Used for storing shared shape facts consumed by header parsing and
    /// probing.
    pub meta: TableMeta,
    /// Used for assigning unique block-cache file identifiers: the WDL file
    /// uses `2 * index` and the DTZ file `2 * index + 1`.
    pub index: u32,
    /// Used for locating the WDL file bytes.
    wdl_source: TableSource,
    /// Used for locating the DTZ file bytes, when the file exists.
    dtz_source: Option<TableSource>,
    /// Used for caching the parsed WDL metadata; `None` inside marks a
    /// failed parse so it is never retried.
    wdl: OnceLock<Option<Arc<LoadedTable>>>,
    /// Used for caching the parsed DTZ metadata.
    dtz: OnceLock<Option<Arc<LoadedTable>>>,
}

impl TableEntry {
    /// Used for retrieving the lazily parsed WDL table.
    ///
    /// # Arguments
    ///
    /// * `encode` - shared encoding tables needed by header parsing
    ///
    /// # Returns
    ///
    /// The loaded table, or `None` when parsing failed.
    pub fn wdl_table(&self, encode: &EncodeTables) -> Option<&Arc<LoadedTable>> {
        self.wdl
            .get_or_init(|| {
                self.wdl_source
                    .open()
                    .and_then(|file| load_table(file, TableKind::Wdl, self.meta, encode))
                    .ok()
                    .map(Arc::new)
            })
            .as_ref()
    }

    /// Used for retrieving the lazily parsed DTZ table.
    ///
    /// # Arguments
    ///
    /// * `encode` - shared encoding tables needed by header parsing
    ///
    /// # Returns
    ///
    /// The loaded table, or `None` when the file is absent or failed to
    /// parse.
    pub fn dtz_table(&self, encode: &EncodeTables) -> Option<&Arc<LoadedTable>> {
        self.dtz
            .get_or_init(|| {
                self.dtz_source.as_ref().and_then(|source| {
                    source
                        .open()
                        .and_then(|file| load_table(file, TableKind::Dtz, self.meta, encode))
                        .ok()
                        .map(Arc::new)
                })
            })
            .as_ref()
    }
}

/// Immutable registry of every discovered tablebase file.
///
/// Built once from the UCI `SyzygyPath` value and shared read-only across
/// all search workers; per-worker probing state lives in
/// [`super::Prober`].
pub struct Tablebases {
    /// Used for sharing the deterministic position-encoding tables.
    pub(crate) encode: EncodeTables,
    /// Used for owning every registered material configuration.
    pub(crate) entries: Vec<TableEntry>,
    /// Used for point lookups from a material key to its entry.
    by_key: HashMap<u64, usize>,
    /// Used for reporting the largest man count with a WDL file present.
    max_cardinality: usize,
    /// Used for reporting the number of WDL files found.
    wdl_files: usize,
    /// Used for reporting the number of DTZ files found.
    dtz_files: usize,
}

impl Tablebases {
    /// Used for scanning the configured directories for tablebase files.
    ///
    /// The path list uses the platform's conventional separator (`:` on
    /// Unix, `;` on Windows). Every canonical material configuration up to
    /// seven men is checked; a configuration is registered when its `.rtbw`
    /// file exists in any directory, and its `.rtbz` file is recorded when
    /// present. Files are only stat-checked here; parsing is lazy.
    ///
    /// # Arguments
    ///
    /// * `paths` - separator-joined list of directories
    ///
    /// # Returns
    ///
    /// The populated registry; a path without any table files yields an
    /// empty registry with zero cardinality.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when the directory list contains more than 64
    /// distinct entries or its bounded directory vector or a candidate path
    /// cannot be reserved.
    pub fn new(paths: &str) -> Result<Self, SyzygyError> {
        let directories = syzygy_directories(paths)?;
        let mut tablebases = Self {
            encode: EncodeTables::new(),
            entries: Vec::new(),
            by_key: HashMap::new(),
            max_cardinality: 0,
            wdl_files: 0,
            dtz_files: 0,
        };
        for (first, second) in canonical_material_sets() {
            let code = format!("{}v{}", side_code(&first), side_code(&second));
            let Some(wdl_path) = find_in(&directories, &format!("{code}.rtbw"))? else {
                continue;
            };
            let dtz_path = find_in(&directories, &format!("{code}.rtbz"))?;
            tablebases.wdl_files += 1;
            if dtz_path.is_some() {
                tablebases.dtz_files += 1;
            }
            tablebases.register(
                &first,
                &second,
                TableSource::Path(wdl_path),
                dtz_path.map(TableSource::Path),
            );
        }
        Ok(tablebases)
    }

    /// Used for registering one material configuration with its sources.
    ///
    /// Both material keys (as-is and color-swapped) map to the entry;
    /// symmetric configurations register a single key.
    ///
    /// # Arguments
    ///
    /// * `first` - first-named side's non-king pieces
    /// * `second` - second-named side's non-king pieces
    /// * `wdl_source` - WDL byte source
    /// * `dtz_source` - optional DTZ byte source
    pub(crate) fn register(
        &mut self,
        first: &[PieceKind],
        second: &[PieceKind],
        wdl_source: TableSource,
        dtz_source: Option<TableSource>,
    ) {
        let key = side_lists_key(first, second);
        let key2 = side_lists_key(second, first);
        let meta = TableMeta::from_sides(first, second);
        self.max_cardinality = self.max_cardinality.max(usize::from(meta.piece_count));
        let index = u32::try_from(self.entries.len()).expect("bounded table count");
        let entry_index = self.entries.len();
        self.entries.push(TableEntry {
            key,
            meta,
            index,
            wdl_source,
            dtz_source,
            wdl: OnceLock::new(),
            dtz: OnceLock::new(),
        });
        self.by_key.insert(key, entry_index);
        self.by_key.insert(key2, entry_index);
    }

    /// Used for looking up the entry covering a material key.
    ///
    /// # Arguments
    ///
    /// * `key` - material key from [`position_material_key`]
    ///
    /// # Returns
    ///
    /// The matching entry, or `None` when no table covers the material.
    pub(crate) fn entry_for(&self, key: u64) -> Option<&TableEntry> {
        self.by_key.get(&key).map(|index| &self.entries[*index])
    }

    /// Used for reporting the largest man count with a WDL file present.
    ///
    /// # Returns
    ///
    /// Maximum piece count across registered configurations; zero when the
    /// registry is empty.
    #[must_use]
    pub fn max_cardinality(&self) -> usize {
        self.max_cardinality
    }

    /// Used for reporting the number of WDL files found at scan time.
    ///
    /// # Returns
    ///
    /// Count of registered `.rtbw` files.
    #[must_use]
    pub fn wdl_file_count(&self) -> usize {
        self.wdl_files
    }

    /// Used for reporting the number of DTZ files found at scan time.
    ///
    /// # Returns
    ///
    /// Count of registered `.rtbz` files.
    #[must_use]
    pub fn dtz_file_count(&self) -> usize {
        self.dtz_files
    }

}

/// Used for parsing and de-duplicating one separator-delimited tablebase path
/// list before filesystem discovery begins.
///
/// The first occurrence of each exact path is retained, preserving lookup
/// precedence. Empty components are ignored as before. Borrowing the path
/// components from the option string avoids copying attacker-controlled path
/// bytes.
///
/// # Arguments
///
/// * `paths` - separator-delimited directory list
///
/// # Returns
///
/// At most [`MAX_SYZYGY_DIRECTORIES`] distinct borrowed paths in input order.
///
/// # Errors
///
/// Returns [`SyzygyError`] when the bounded directory vector cannot be
/// reserved or the distinct-entry ceiling would be exceeded.
pub(crate) fn syzygy_directories(paths: &str) -> Result<Vec<&Path>, SyzygyError> {
    let separator = if cfg!(windows) { ';' } else { ':' };
    let mut directories = Vec::new();
    directories
        .try_reserve_exact(MAX_SYZYGY_DIRECTORIES)
        .map_err(|error| SyzygyError::new(format!("reserve SyzygyPath directories: {error}")))?;
    for part in paths.split(separator).filter(|part| !part.is_empty()) {
        let path = Path::new(part);
        if directories.contains(&path) {
            continue;
        }
        if directories.len() == MAX_SYZYGY_DIRECTORIES {
            return Err(SyzygyError::new(format!(
                "SyzygyPath contains more than {MAX_SYZYGY_DIRECTORIES} unique directories"
            )));
        }
        directories.push(path);
    }
    Ok(directories)
}

/// Used for locating one file name inside the configured directories.
///
/// Directories are checked in configuration order and the first match
/// wins, keeping discovery deterministic.
///
/// # Arguments
///
/// * `directories` - configured search directories in order
/// * `name` - bare file name to look for
///
/// # Returns
///
/// Full path of the first existing match, or `None`.
///
/// # Errors
///
/// Returns [`SyzygyError`] when a complete candidate path cannot be
/// represented or reserved.
fn find_in(directories: &[&Path], name: &str) -> Result<Option<PathBuf>, SyzygyError> {
    for directory in directories {
        let capacity = directory
            .as_os_str()
            .len()
            .checked_add(name.len())
            .and_then(|length| length.checked_add(2))
            .ok_or_else(|| SyzygyError::new("Syzygy candidate path length overflows"))?;
        let mut candidate = PathBuf::new();
        candidate
            .try_reserve_exact(capacity)
            .map_err(|error| SyzygyError::new(format!("reserve Syzygy candidate path: {error}")))?;
        candidate.push(directory);
        candidate.push(name);
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}
