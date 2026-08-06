//! Bounded positional access to tablebase bytes.
//!
//! Every read names an absolute offset and an exact length, both validated
//! against the recorded file length before any I/O happens, so a truncated
//! or corrupt file surfaces as a [`SyzygyError`] instead of unbounded
//! buffering or a panic. Small files, and every synthetic test table, are
//! held fully in memory; larger files stay on disk behind a mutex-protected
//! seek-and-read handle. No memory mapping and no `unsafe` code is used.

use super::SyzygyError;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

/// Files at most this large are buffered fully in memory at open time.
///
/// The three-to-five-men set contains many tiny tables; buffering them
/// avoids both syscalls and block-cache traffic entirely.
pub(crate) const FULL_BUFFER_LIMIT: u64 = 8 * 1024 * 1024;

/// Used for reading one metadata-classified small file with an exact cap.
///
/// Reserving one sentinel byte beyond the recorded length lets the read
/// distinguish exact input from concurrent growth without buffering an
/// unbounded changing file. The reservation is fallible, and the caller's
/// reader is never asked for more than the recorded length plus that byte.
///
/// # Arguments
///
/// * `reader` - byte source positioned at the start of the file
/// * `expected_len` - file length captured from metadata
/// * `path` - source path used only in diagnostics
///
/// # Returns
///
/// Exactly `expected_len` bytes when the source remained stable.
///
/// # Errors
///
/// Returns [`SyzygyError`] when the classified length exceeds the buffering
/// limit, capacity cannot be reserved, the read fails, or the source is
/// shorter or longer than the captured length.
pub(super) fn read_buffered_exact(
    reader: &mut impl Read,
    expected_len: u64,
    path: &Path,
) -> Result<Vec<u8>, SyzygyError> {
    if expected_len > FULL_BUFFER_LIMIT {
        return Err(SyzygyError::new(format!(
            "buffered table {} exceeds the {FULL_BUFFER_LIMIT}-byte limit",
            path.display()
        )));
    }
    let read_limit = expected_len
        .checked_add(1)
        .ok_or_else(|| SyzygyError::new("buffered table read limit overflows"))?;
    let capacity = usize::try_from(read_limit)
        .map_err(|_| SyzygyError::new("buffered table exceeds address space"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|error| {
        SyzygyError::new(format!(
            "reserve buffered table {}: {error}",
            path.display()
        ))
    })?;
    reader
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| SyzygyError::new(format!("read {}: {error}", path.display())))?;
    let actual_len = u64::try_from(bytes.len())
        .map_err(|_| SyzygyError::new("buffered table length exceeds address space"))?;
    if actual_len != expected_len {
        return Err(SyzygyError::new(format!(
            "table {} changed length while opening: expected {expected_len} bytes, read {actual_len}",
            path.display()
        )));
    }
    Ok(bytes)
}

/// Used for creating a zeroed positional-read destination fallibly.
///
/// A successful exact reservation ensures the subsequent resize performs no
/// allocation, so allocation refusal is returned as a tablebase error rather
/// than surfacing from `vec![0; length]`.
///
/// # Arguments
///
/// * `length` - exact destination length in bytes
///
/// # Returns
///
/// A zero-filled vector of `length` bytes.
///
/// # Errors
///
/// Returns [`SyzygyError`] when the requested capacity cannot be represented
/// or allocated.
pub(super) fn zeroed_read_buffer(length: usize) -> Result<Vec<u8>, SyzygyError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|error| {
        SyzygyError::new(format!(
            "reserve table read buffer of {length} bytes: {error}"
        ))
    })?;
    bytes.resize(length, 0);
    Ok(bytes)
}

/// Backing storage of one opened table.
///
/// The memory variant serves fully buffered small files and synthetic
/// in-memory test tables; the disk variant serves large files through
/// bounded positional reads.
enum Backing {
    /// Used for fully buffered file content.
    Memory(Vec<u8>),
    /// Used for on-demand reads from an open file handle.
    ///
    /// The mutex serializes the seek-then-read pair so concurrent workers
    /// each observe their own consistent positional read.
    Disk(Mutex<fs::File>),
}

/// One opened tablebase file with strictly bounded positional reads.
///
/// The length is captured at open time and every access is validated
/// against it; the file is treated as immutable for the lifetime of this
/// value, matching how tablebase sets are deployed.
pub(crate) struct BoundedFile {
    /// Used for storing the memory or disk backing.
    backing: Backing,
    /// Used for validating every read against the byte length at open time.
    len: u64,
}

impl BoundedFile {
    /// Used for opening a table file from disk.
    ///
    /// Files no larger than [`FULL_BUFFER_LIMIT`] are read fully into
    /// memory; larger files keep an open handle for positional reads.
    ///
    /// # Arguments
    ///
    /// * `path` - table file location
    ///
    /// # Returns
    ///
    /// The opened file wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when the file cannot be opened or read.
    pub fn open(path: &Path) -> Result<Self, SyzygyError> {
        let mut file = fs::File::open(path)
            .map_err(|error| SyzygyError::new(format!("open {}: {error}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|error| SyzygyError::new(format!("stat {}: {error}", path.display())))?
            .len();
        if len <= FULL_BUFFER_LIMIT {
            let bytes = read_buffered_exact(&mut file, len, path)?;
            return Ok(Self::from_bytes(bytes));
        }
        Ok(Self {
            backing: Backing::Disk(Mutex::new(file)),
            len,
        })
    }

    /// Used for wrapping fully in-memory table bytes.
    ///
    /// This constructor backs both fully buffered small files and the
    /// synthetic tables used by unit tests.
    ///
    /// # Arguments
    ///
    /// * `bytes` - complete table content
    ///
    /// # Returns
    ///
    /// A memory-backed file wrapper.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let len = bytes.len() as u64;
        Self {
            backing: Backing::Memory(bytes),
            len,
        }
    }

    /// Used for reading the recorded file length.
    ///
    /// # Returns
    ///
    /// Total byte length captured at open time.
    pub fn byte_len(&self) -> u64 {
        self.len
    }

    /// Used for borrowing a validated in-memory range when the backing is
    /// memory.
    ///
    /// # Arguments
    ///
    /// * `offset` - absolute byte offset of the range start
    /// * `length` - exact number of bytes requested
    ///
    /// # Returns
    ///
    /// `Some` slice for a memory backing whose range is fully inside the
    /// file, `None` for a disk backing or an out-of-range request.
    pub fn memory_slice(&self, offset: u64, length: usize) -> Option<&[u8]> {
        match &self.backing {
            Backing::Memory(bytes) => {
                let start = usize::try_from(offset).ok()?;
                let end = start.checked_add(length)?;
                bytes.get(start..end)
            }
            Backing::Disk(_) => None,
        }
    }

    /// Used for reading an exact byte range at an absolute offset.
    ///
    /// The range is validated against the recorded length before any I/O;
    /// disk reads serialize a seek-then-read pair under the handle's mutex.
    ///
    /// # Arguments
    ///
    /// * `offset` - absolute byte offset of the range start
    /// * `buffer` - destination filled completely on success
    ///
    /// # Returns
    ///
    /// `Ok(())` once `buffer` holds exactly the requested range.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] when the range extends past the end of the
    /// file, when the handle mutex is poisoned, or on any I/O failure.
    pub fn read_exact_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), SyzygyError> {
        let length = buffer.len() as u64;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| SyzygyError::new("table read range overflows"))?;
        if end > self.len {
            return Err(SyzygyError::new(format!(
                "table read of {length} bytes at offset {offset} exceeds file length {}",
                self.len
            )));
        }
        match &self.backing {
            Backing::Memory(bytes) => {
                let start = usize::try_from(offset)
                    .map_err(|_| SyzygyError::new("table offset exceeds address space"))?;
                buffer.copy_from_slice(&bytes[start..start + buffer.len()]);
                Ok(())
            }
            Backing::Disk(handle) => {
                let mut file = handle
                    .lock()
                    .map_err(|_| SyzygyError::new("table file mutex poisoned"))?;
                file.seek(SeekFrom::Start(offset))
                    .map_err(|error| SyzygyError::new(format!("table seek: {error}")))?;
                file.read_exact(buffer)
                    .map_err(|error| SyzygyError::new(format!("table read: {error}")))?;
                Ok(())
            }
        }
    }

    /// Used for reading an owned byte vector at an absolute offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - absolute byte offset of the range start
    /// * `length` - exact number of bytes requested
    ///
    /// # Returns
    ///
    /// Owned bytes of exactly `length` on success.
    ///
    /// # Errors
    ///
    /// Returns [`SyzygyError`] under the same conditions as
    /// [`Self::read_exact_at`]; the destination is only allocated after the
    /// range has been validated against the file length, and allocation
    /// refusal is returned as an error.
    pub fn read_vec_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, SyzygyError> {
        let length_u64 = u64::try_from(length)
            .map_err(|_| SyzygyError::new("table read length exceeds address space"))?;
        let end = offset
            .checked_add(length_u64)
            .ok_or_else(|| SyzygyError::new("table read range overflows"))?;
        if end > self.len {
            return Err(SyzygyError::new(format!(
                "table read of {length} bytes at offset {offset} exceeds file length {}",
                self.len
            )));
        }
        let mut bytes = zeroed_read_buffer(length)?;
        self.read_exact_at(offset, &mut bytes)?;
        Ok(bytes)
    }
}
