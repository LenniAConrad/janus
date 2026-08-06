//! Compact local and race-safe shared direct-mapped transposition tables.
//!
//! The payload layout intentionally matches `ChessRTK`'s Java table so parity
//! tests can compare packed values directly. [`TranspositionTable`] is the
//! exact single-thread reference path. [`SharedTranspositionTable`] uses a
//! bounded per-bucket sequence lock built only from standard-library atomics,
//! allowing Lazy-SMP workers to exchange bounds without accepting a torn key,
//! payload, or generation.

use crate::threading::spawn_scoped_or_run;
use janus_core::{Move, NO_MOVE};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Used for locating the two-bit [`Bound`] field in a packed payload word.
///
/// The bound bits sit immediately above the 16-bit raw move field.
const FLAG_SHIFT: u32 = 16;
/// Used for locating the remaining-depth field in a packed payload word.
///
/// The depth bits sit immediately above the two-bit [`Bound`] field.
const DEPTH_SHIFT: u32 = 18;
/// Used for locating the signed 32-bit score field in a packed payload word.
///
/// The score occupies the upper half of the 64-bit payload.
const SCORE_SHIFT: u32 = 32;
/// Used for sizing the remaining-depth field of a packed payload.
///
/// Depth values are stored modulo `2^14`.
const DEPTH_BITS: u32 = 14;
/// Used for bounding the prefix length inspected by UCI `hashfull` telemetry.
///
/// Sampling at most 1,000 buckets keeps the occupancy estimate's cost
/// independent of the configured hash size.
const HASHFULL_SAMPLE_BUCKETS: usize = 1_000;
/// Used for marking occupancy in a shared entry's generation metadata word.
///
/// The flag sits above the low 16 generation bits, so an all-zero metadata
/// word always reads as unoccupied.
const SHARED_OCCUPIED: u32 = 1 << 16;
/// Used for limiting non-blocking writer-lock attempts on the search hot
/// path.
///
/// After this many failed acquisition attempts a store is dropped instead of
/// blocking the searching thread.
const SHARED_STORE_ATTEMPTS: usize = 4;

/// Alpha-beta bound represented by a table entry.
///
/// The variant is packed into two bits of a [`TtPayload`]; the fourth bit
/// pattern (`3`) is reserved and rejected during decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Bound {
    /// Used for indicating an exact score inside the original search window.
    Exact = 0,
    /// Used for indicating a lower bound produced by a beta cutoff.
    Lower = 1,
    /// Used for indicating an upper bound produced by a fail-low.
    Upper = 2,
}

impl Bound {
    /// Used for obtaining the two-bit packed representation of the bound.
    ///
    /// # Returns
    ///
    /// The value `0`, `1`, or `2` matching the payload encoding.
    const fn bits(self) -> u8 {
        match self {
            Self::Exact => 0,
            Self::Lower => 1,
            Self::Upper => 2,
        }
    }

    /// Used for decoding a two-bit bound, rejecting the reserved value `3`.
    ///
    /// # Arguments
    ///
    /// * `bits` - two-bit packed bound representation
    ///
    /// # Returns
    ///
    /// The decoded bound, or `None` for the reserved bit pattern.
    const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0 => Some(Self::Exact),
            1 => Some(Self::Lower),
            2 => Some(Self::Upper),
            _ => None,
        }
    }
}

/// Packed transposition payload compatible with `ChessRTK`'s Java table.
///
/// The low 16 bits hold a raw move, followed by a two-bit [`Bound`], a 14-bit
/// remaining depth, and a signed 32-bit score. Position keys and replacement
/// generations live in the surrounding [`TranspositionTable`] entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct TtPayload(u64);

impl TtPayload {
    /// Used for packing move, bound, depth, and signed score into one 64-bit
    /// word.
    ///
    /// Depth is stored modulo 2^14. `best_move` may be [`NO_MOVE`] or the raw
    /// representation of a valid [`Move`].
    ///
    /// # Arguments
    ///
    /// * `depth` - remaining search depth in plies, stored modulo `2^14`
    /// * `score` - signed position-relative search score
    /// * `bound` - bound classification of `score`
    /// * `best_move` - raw move bits, or [`NO_MOVE`] when no move is known
    ///
    /// # Returns
    ///
    /// Packed 64-bit payload word.
    #[must_use]
    pub fn new(depth: u16, score: i32, bound: Bound, best_move: u16) -> Self {
        let depth_mask = (1_u64 << DEPTH_BITS) - 1;
        let score_bits = u32::from_ne_bytes(score.to_ne_bytes());
        Self(
            u64::from(best_move)
                | (u64::from(bound.bits()) << FLAG_SHIFT)
                | ((u64::from(depth) & depth_mask) << DEPTH_SHIFT)
                | (u64::from(score_bits) << SCORE_SHIFT),
        )
    }

    /// Used for reconstructing a payload from its wire representation.
    ///
    /// Invalid move bits remain representable and are filtered later by
    /// [`Self::best_move`].
    ///
    /// # Arguments
    ///
    /// * `raw` - packed 64-bit payload word
    ///
    /// # Returns
    ///
    /// The wrapped payload; `None` only when the bound field contains its
    /// reserved bit pattern.
    #[must_use]
    pub fn from_raw(raw: u64) -> Option<Self> {
        let bits = u8::try_from((raw >> FLAG_SHIFT) & 3).ok()?;
        if Bound::from_bits(bits).is_some() {
            Some(Self(raw))
        } else {
            None
        }
    }

    /// Used for retrieving the packed 64-bit representation.
    ///
    /// # Returns
    ///
    /// Raw payload bits exactly as stored in a table entry.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Used for retrieving the packed remaining search depth in plies.
    ///
    /// # Returns
    ///
    /// The 14-bit depth field as an unsigned ply count.
    #[must_use]
    pub fn depth(self) -> u16 {
        u16::try_from((self.0 >> DEPTH_SHIFT) & ((1_u64 << DEPTH_BITS) - 1)).unwrap_or(0)
    }

    /// Used for retrieving the packed signed search score.
    ///
    /// The upper 32 payload bits are reinterpreted bitwise as a signed
    /// integer.
    ///
    /// # Returns
    ///
    /// Signed 32-bit score in its stored position-relative frame.
    #[must_use]
    pub fn score(self) -> i32 {
        let bits = u32::try_from(self.0 >> SCORE_SHIFT).unwrap_or(0);
        i32::from_ne_bytes(bits.to_ne_bytes())
    }

    /// Used for retrieving the stored bound classification.
    ///
    /// # Returns
    ///
    /// The decoded [`Bound`]; the reserved bit pattern falls back to
    /// [`Bound::Exact`].
    #[must_use]
    pub fn bound(self) -> Bound {
        let bits = u8::try_from((self.0 >> FLAG_SHIFT) & 3).unwrap_or(0);
        Bound::from_bits(bits).unwrap_or(Bound::Exact)
    }

    /// Used for retrieving the stored move bit pattern, including
    /// [`NO_MOVE`].
    ///
    /// # Returns
    ///
    /// The low 16 payload bits without validity filtering.
    #[must_use]
    pub fn best_move_raw(self) -> u16 {
        u16::try_from(self.0 & u64::from(u16::MAX)).unwrap_or(NO_MOVE)
    }

    /// Used for retrieving a valid stored move, if present.
    ///
    /// [`NO_MOVE`] and malformed raw encodings both produce `None`.
    ///
    /// # Returns
    ///
    /// The decoded move-ordering hint, or `None` when absent or malformed.
    #[must_use]
    pub fn best_move(self) -> Option<Move> {
        let raw = self.best_move_raw();
        if raw == NO_MOVE {
            None
        } else {
            Move::from_raw(raw).ok()
        }
    }
}

/// Used for the bucket count above which clearing is worth spreading across
/// threads.
///
/// A competition-sized table is tens of gibibytes, and a single-threaded fill
/// of that much memory takes long enough for a host to time out the
/// `ucinewgame` that triggered it.
const PARALLEL_CLEAR_THRESHOLD: usize = 1 << 22;

/// Used for resetting a large bucket slice with every available core.
///
/// Below [`PARALLEL_CLEAR_THRESHOLD`] the slice is filled directly, so small
/// tables and tests keep their existing single-threaded behaviour exactly.
///
/// # Arguments
///
/// * `entries` - bucket slice to reset
/// * `value` - value written into every bucket
fn clear_in_parallel<T: Copy + Send>(entries: &mut [T], value: T) {
    let workers = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    if workers <= 1 || entries.len() < PARALLEL_CLEAR_THRESHOLD {
        entries.fill(value);
        return;
    }
    let chunk = entries.len().div_ceil(workers);
    std::thread::scope(|scope| {
        for slice in entries.chunks_mut(chunk) {
            spawn_scoped_or_run(scope, "janus-tt-clear", move || slice.fill(value));
        }
    });
}

/// One direct-mapped bucket protected by an XOR key/data consistency check.
///
/// A default entry is unoccupied; the explicit flag keeps the all-zero key
/// and payload representable as real data.
#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    /// Used for validating probes: full key XOR packed data, so recombining
    /// with the payload must reproduce the probing key.
    key_xor_data: u64,
    /// Used for holding the packed [`TtPayload`] bits.
    data: u64,
    /// Used for replacement decisions and occupancy telemetry: the
    /// iterative-deepening age that stored the entry.
    generation: u16,
    /// Used for explicit occupancy so key and payload zero remain
    /// representable.
    occupied: bool,
}

/// Validated table hit.
///
/// Produced only after the XOR consistency check matched the probing key and
/// the payload's bound field decoded successfully.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TtHit {
    /// Used for carrying the packed move, bound, depth, and score.
    pub payload: TtPayload,
    /// Used for reporting the iterative-deepening generation that stored the
    /// entry.
    pub generation: u16,
}

/// Single-threaded direct-mapped transposition table.
///
/// Exactly one entry maps to each bucket. A full-key XOR check prevents a
/// colliding key from becoming a false hit. Within one generation, a shallower
/// entry cannot replace a deeper entry; a new generation may replace it.
#[derive(Debug)]
pub struct TranspositionTable {
    /// Used for storing the power-of-two direct-mapped bucket allocation.
    entries: Box<[Entry]>,
    /// Used for mixed-key indexing; always `entries.len() - 1`.
    mask: usize,
}

impl TranspositionTable {
    /// Used for creating a table with a fixed direct-mapped allocation.
    ///
    /// # Arguments
    ///
    /// * `entry_count` - number of buckets; must be a nonzero power of two
    ///
    /// # Returns
    ///
    /// An empty table with `entry_count` buckets.
    ///
    /// # Panics
    ///
    /// Panics when `entry_count` is zero or not a power of two.
    #[must_use]
    pub fn new(entry_count: usize) -> Self {
        Self::try_new(entry_count).expect("transposition allocation succeeds")
    }

    /// Used for allocating a table without aborting when memory runs out.
    ///
    /// A competition host may request a hash far larger than the machine can
    /// provide. The infallible allocator would abort the process, which
    /// forfeits the game, so callers that accept an operator-supplied size use
    /// this and shrink their request instead.
    ///
    /// # Arguments
    ///
    /// * `entry_count` - number of buckets; must be a nonzero power of two
    ///
    /// # Returns
    ///
    /// `Some(table)` when the allocation succeeded, `None` when it failed.
    ///
    /// # Panics
    ///
    /// Panics when `entry_count` is zero or not a power of two.
    #[must_use]
    pub fn try_new(entry_count: usize) -> Option<Self> {
        assert!(
            entry_count.is_power_of_two(),
            "TT size must be a power of two"
        );
        let mut entries: Vec<Entry> = Vec::new();
        entries.try_reserve_exact(entry_count).ok()?;
        entries.resize(entry_count, Entry::default());
        Some(Self {
            entries: entries.into_boxed_slice(),
            mask: entry_count - 1,
        })
    }

    /// Used for retrieving the number of direct-mapped buckets.
    ///
    /// # Returns
    ///
    /// Total bucket count of the allocation.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Used for checking whether the table has no buckets.
    ///
    /// # Returns
    ///
    /// `true` when the allocation contains zero buckets.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Used for discarding all entries while retaining allocated storage.
    ///
    /// Every bucket is reset to its unoccupied default value.
    pub fn clear(&mut self) {
        clear_in_parallel(&mut self.entries, Entry::default());
    }

    /// Used for estimating current-generation occupancy in per mille for UCI
    /// telemetry.
    ///
    /// At most the first 1,000 buckets are sampled, keeping the cost bounded
    /// independently of the configured hash size. Tables smaller than the
    /// sample are scaled to the same `0..=1000` range.
    ///
    /// # Arguments
    ///
    /// * `generation` - iterative-deepening age counted as current
    ///
    /// # Returns
    ///
    /// Estimated occupancy in the range `0..=1000`.
    #[must_use]
    pub fn hashfull_per_mille(&self, generation: u16) -> u16 {
        let sample_size = self.entries.len().min(HASHFULL_SAMPLE_BUCKETS);
        let current = self.entries[..sample_size]
            .iter()
            .filter(|entry| entry.occupied && entry.generation == generation)
            .count();
        u16::try_from(current * 1_000 / sample_size).unwrap_or(1_000)
    }

    /// Used for looking up `key`, treating any inconsistent XOR pair as a
    /// miss.
    ///
    /// The returned score remains in its stored position-relative frame; the
    /// alpha-beta caller is responsible for mate-distance conversion.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit position hash key
    ///
    /// # Returns
    ///
    /// A validated hit, or `None` for an empty, mismatched, or undecodable
    /// bucket.
    #[must_use]
    pub fn probe(&self, key: u64) -> Option<TtHit> {
        let entry = self.entries[self.index(key)];
        if !entry.occupied || entry.key_xor_data ^ entry.data != key {
            return None;
        }
        Some(TtHit {
            payload: TtPayload::from_raw(entry.data)?,
            generation: entry.generation,
        })
    }

    /// Used for retrieving only the table's validated move-ordering hint.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit position hash key
    ///
    /// # Returns
    ///
    /// The stored move of a validated hit, or `None`.
    #[must_use]
    pub fn best_move(&self, key: u64) -> Option<Move> {
        self.probe(key).and_then(|hit| hit.payload.best_move())
    }

    /// Used for storing an entry using depth-preserving same-generation
    /// replacement.
    ///
    /// A different generation always replaces the colliding bucket. Within the
    /// same generation, an existing strictly deeper payload is retained.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit position hash key
    /// * `payload` - packed move, bound, depth, and score
    /// * `generation` - iterative-deepening age of the storing search
    pub fn store(&mut self, key: u64, payload: TtPayload, generation: u16) {
        let index = self.index(key);
        let current = self.entries[index];
        if current.occupied
            && current.generation == generation
            && TtPayload::from_raw(current.data).is_some_and(|old| old.depth() > payload.depth())
        {
            return;
        }
        let data = payload.raw();
        self.entries[index] = Entry {
            key_xor_data: key ^ data,
            data,
            generation,
            occupied: true,
        };
    }

    /// Used for mixing both halves of `key` and masking it into the table
    /// allocation.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit position hash key
    ///
    /// # Returns
    ///
    /// Bucket index in `0..self.len()`.
    ///
    /// The mixed key is masked with the full bucket mask rather than being
    /// truncated to 32 bits first. Truncation was equivalent for tables up to
    /// 2^32 buckets but silently capped every larger allocation onto the first
    /// four gibi-entries, which a competition-sized hash exceeds.
    #[inline]
    fn index(&self, key: u64) -> usize {
        let mixed = key ^ (key >> 32);
        let mask = u64::try_from(self.mask).expect("a bucket mask fits u64");
        usize::try_from(mixed & mask).expect("a masked bucket index fits usize")
    }
}

impl Default for TranspositionTable {
    /// Used for creating the standard table with 2^20 direct-mapped buckets.
    ///
    /// # Returns
    ///
    /// A cleared table containing `1 << 20` entries.
    fn default() -> Self {
        Self::new(1 << 20)
    }
}

/// One atomically published direct-mapped bucket.
///
/// Even sequence values denote a stable snapshot and odd values denote an
/// in-progress writer. Writers serialize through a compare-and-exchange on
/// `sequence`; readers accept data only when equal even sequence samples
/// surround all payload loads.
#[derive(Debug, Default)]
struct SharedEntry {
    /// Used for the per-bucket sequence and writer-lock word.
    sequence: AtomicU64,
    /// Used for validating probes: full position key XOR packed payload.
    key_xor_data: AtomicU64,
    /// Used for holding the packed [`TtPayload`] bits.
    data: AtomicU64,
    /// Used for holding the low 16 generation bits plus [`SHARED_OCCUPIED`].
    metadata: AtomicU32,
}

impl SharedEntry {
    /// Used for acquiring the writer lock without unbounded search-path
    /// waiting.
    ///
    /// At most [`SHARED_STORE_ATTEMPTS`] acquisition attempts are made; an
    /// odd (locked) sequence or a lost compare-and-exchange race spins and
    /// retries.
    ///
    /// # Returns
    ///
    /// The even sequence value observed before locking, or `None` when the
    /// bucket stayed contended.
    fn try_lock(&self) -> Option<u64> {
        for _ in 0..SHARED_STORE_ATTEMPTS {
            let sequence = self.sequence.load(Ordering::Acquire);
            if sequence & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            if self
                .sequence
                .compare_exchange_weak(
                    sequence,
                    sequence.wrapping_add(1),
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return Some(sequence);
            }
            std::hint::spin_loop();
        }
        None
    }

    /// Used for acquiring the writer lock for an administrative operation
    /// such as clear.
    ///
    /// Retries [`Self::try_lock`] indefinitely, yielding the thread between
    /// bounded rounds of failed attempts.
    ///
    /// # Returns
    ///
    /// The even sequence value observed before locking.
    fn lock(&self) -> u64 {
        loop {
            if let Some(sequence) = self.try_lock() {
                return sequence;
            }
            std::thread::yield_now();
        }
    }

    /// Used for publishing all writes performed while holding `sequence`'s
    /// odd successor.
    ///
    /// Release-stores the next even value, `sequence + 2`, so readers that
    /// observe it also observe the protected payload writes.
    ///
    /// # Arguments
    ///
    /// * `sequence` - even value returned by the matching lock acquisition
    fn unlock(&self, sequence: u64) {
        self.sequence
            .store(sequence.wrapping_add(2), Ordering::Release);
    }
}

/// Atomic direct-mapped table shared by Lazy-SMP search workers.
///
/// The table contains one entry per bucket. A writer first changes the bucket's
/// sequence from even to odd with compare-and-exchange, writes the key, payload,
/// and generation, then release-publishes the next even sequence. A reader
/// samples the sequence before and after its loads and treats contention as a
/// miss. This is a search cache, so bounded misses are preferable to blocking.
///
/// Generation ownership is also shared: a parallel-search coordinator calls
/// [`Self::advance_generation`] once, and every worker stores with the returned
/// generation. [`Self::clear`] safely resets both entries and the generation.
#[derive(Debug)]
pub struct SharedTranspositionTable {
    /// Used for storing the power-of-two atomic bucket allocation.
    entries: Box<[SharedEntry]>,
    /// Used for mixed-key indexing; always `entries.len() - 1`.
    mask: usize,
    /// Used for the search generation shared by every worker using this
    /// allocation.
    generation: AtomicU32,
}

impl SharedTranspositionTable {
    /// Used for creating a shared table with a fixed atomic allocation.
    ///
    /// # Arguments
    ///
    /// * `entry_count` - number of buckets; must be a nonzero power of two
    ///
    /// # Returns
    ///
    /// An empty shared table with `entry_count` buckets and generation zero.
    ///
    /// # Panics
    ///
    /// Panics when `entry_count` is zero or not a power of two.
    #[must_use]
    pub fn new(entry_count: usize) -> Self {
        Self::try_new(entry_count).expect("shared transposition allocation succeeds")
    }

    /// Used for allocating a shared table without aborting when memory runs
    /// out.
    ///
    /// # Arguments
    ///
    /// * `entry_count` - number of buckets; must be a nonzero power of two
    ///
    /// # Returns
    ///
    /// `Some(table)` when the allocation succeeded, `None` when it failed.
    ///
    /// # Panics
    ///
    /// Panics when `entry_count` is zero or not a power of two.
    #[must_use]
    pub fn try_new(entry_count: usize) -> Option<Self> {
        assert!(
            entry_count.is_power_of_two(),
            "TT size must be a power of two"
        );
        let mut entries: Vec<SharedEntry> = Vec::new();
        entries.try_reserve_exact(entry_count).ok()?;
        entries.resize_with(entry_count, SharedEntry::default);
        Some(Self {
            entries: entries.into_boxed_slice(),
            mask: entry_count - 1,
            generation: AtomicU32::new(0),
        })
    }

    /// Used for retrieving the number of direct-mapped buckets.
    ///
    /// # Returns
    ///
    /// Total bucket count of the allocation.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Used for checking whether the table has no buckets.
    ///
    /// # Returns
    ///
    /// `true` when the allocation contains zero buckets.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Used for advancing and returning the shared nonzero search generation.
    ///
    /// A parallel coordinator must call this exactly once before starting its
    /// workers. Wrapping skips zero, which is reserved as the cleared age.
    ///
    /// # Returns
    ///
    /// The newly assigned generation in `1..=u16::MAX`.
    #[must_use]
    pub fn advance_generation(&self) -> u16 {
        loop {
            let current = self.generation.load(Ordering::Acquire);
            let current_u16 = u16::try_from(current).unwrap_or(0);
            let mut next = current_u16.wrapping_add(1);
            if next == 0 {
                next = 1;
            }
            if self
                .generation
                .compare_exchange_weak(
                    current,
                    u32::from(next),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return next;
            }
        }
    }

    /// Used for reading the generation most recently assigned by the
    /// coordinator.
    ///
    /// # Returns
    ///
    /// The current shared generation; zero denotes a cleared table.
    #[must_use]
    pub fn generation(&self) -> u16 {
        u16::try_from(self.generation.load(Ordering::Acquire)).unwrap_or(0)
    }

    /// Used for discarding all entries and resetting the shared generation to
    /// zero.
    ///
    /// UCI invokes this only while workers are idle. Bucket locking still makes
    /// the operation memory-safe if an external caller clears concurrently.
    pub fn clear(&self) {
        let workers = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        if workers <= 1 || self.entries.len() < PARALLEL_CLEAR_THRESHOLD {
            Self::clear_slice(&self.entries);
        } else {
            let chunk = self.entries.len().div_ceil(workers);
            std::thread::scope(|scope| {
                for slice in self.entries.chunks(chunk) {
                    spawn_scoped_or_run(scope, "janus-shared-tt-clear", move || {
                        Self::clear_slice(slice);
                    });
                }
            });
        }
        self.generation.store(0, Ordering::Release);
    }

    /// Used for resetting one disjoint span of shared buckets.
    ///
    /// Each bucket carries its own sequence lock, so disjoint spans are
    /// independent and may be reset concurrently. A competition-sized shared
    /// table takes tens of seconds to reset one bucket at a time, which is
    /// long enough for a host to time out the `ucinewgame` that triggered it.
    ///
    /// # Arguments
    ///
    /// * `entries` - disjoint span of buckets to reset
    fn clear_slice(entries: &[SharedEntry]) {
        for entry in entries {
            let sequence = entry.lock();
            entry.key_xor_data.store(0, Ordering::Release);
            entry.data.store(0, Ordering::Release);
            entry.metadata.store(0, Ordering::Release);
            entry.unlock(sequence);
        }
    }

    /// Used for estimating current-generation occupancy in per mille for UCI
    /// telemetry.
    ///
    /// Each sampled bucket is read through the same consistency protocol as a
    /// probe. Contended buckets count as empty, keeping telemetry bounded.
    ///
    /// # Arguments
    ///
    /// * `generation` - shared search generation counted as current
    ///
    /// # Returns
    ///
    /// Estimated occupancy in the range `0..=1000`.
    #[must_use]
    pub fn hashfull_per_mille(&self, generation: u16) -> u16 {
        let sample_size = self.entries.len().min(HASHFULL_SAMPLE_BUCKETS);
        let current = self.entries[..sample_size]
            .iter()
            .filter(|entry| Self::snapshot(entry).is_some_and(|(_, _, age)| age == generation))
            .count();
        u16::try_from(current * 1_000 / sample_size).unwrap_or(1_000)
    }

    /// Used for looking up `key`, rejecting contended, torn, and colliding
    /// snapshots.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit position hash key
    ///
    /// # Returns
    ///
    /// A validated hit, or `None` for an empty, contended, mismatched, or
    /// undecodable bucket.
    #[must_use]
    pub fn probe(&self, key: u64) -> Option<TtHit> {
        let (key_xor_data, data, generation) = Self::snapshot(&self.entries[self.index(key)])?;
        if key_xor_data ^ data != key {
            return None;
        }
        Some(TtHit {
            payload: TtPayload::from_raw(data)?,
            generation,
        })
    }

    /// Used for retrieving only the table's validated move-ordering hint.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit position hash key
    ///
    /// # Returns
    ///
    /// The stored move of a validated hit, or `None`.
    #[must_use]
    pub fn best_move(&self, key: u64) -> Option<Move> {
        self.probe(key).and_then(|hit| hit.payload.best_move())
    }

    /// Used for attempting to store an entry with same-generation depth
    /// preference.
    ///
    /// Contention after a bounded number of compare-and-exchange attempts drops
    /// the cache write. This cannot change search correctness because table
    /// entries are optional hints and validated bounds.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit position hash key
    /// * `payload` - packed move, bound, depth, and score
    /// * `generation` - shared search generation of the storing worker
    pub fn store(&self, key: u64, payload: TtPayload, generation: u16) {
        let entry = &self.entries[self.index(key)];
        let Some(sequence) = entry.try_lock() else {
            return;
        };
        let metadata = entry.metadata.load(Ordering::Relaxed);
        let occupied = metadata & SHARED_OCCUPIED != 0;
        let old_generation = u16::try_from(metadata & u32::from(u16::MAX)).unwrap_or(0);
        let preserve = occupied
            && old_generation == generation
            && TtPayload::from_raw(entry.data.load(Ordering::Relaxed))
                .is_some_and(|old| old.depth() > payload.depth());
        if !preserve {
            let data = payload.raw();
            entry.key_xor_data.store(key ^ data, Ordering::Release);
            entry.data.store(data, Ordering::Release);
            entry
                .metadata
                .store(SHARED_OCCUPIED | u32::from(generation), Ordering::Release);
        }
        entry.unlock(sequence);
    }

    /// Used for reading one stable occupied bucket snapshot or reporting a
    /// bounded miss.
    ///
    /// Equal even sequence samples must surround the three payload loads;
    /// contended and unoccupied buckets both return `None`.
    ///
    /// # Arguments
    ///
    /// * `entry` - shared bucket to sample
    ///
    /// # Returns
    ///
    /// `(key_xor_data, data, generation)` for a stable occupied bucket, or
    /// `None`.
    fn snapshot(entry: &SharedEntry) -> Option<(u64, u64, u16)> {
        let before = entry.sequence.load(Ordering::Acquire);
        if before & 1 != 0 {
            return None;
        }
        // Acquiring each payload word makes any newly observed word publish the
        // writer's earlier odd sequence. The final sequence sample must then
        // observe that odd value or its later even successor, so a mixed
        // snapshot cannot validate even under the language memory model.
        let key_xor_data = entry.key_xor_data.load(Ordering::Acquire);
        let data = entry.data.load(Ordering::Acquire);
        let metadata = entry.metadata.load(Ordering::Acquire);
        let after = entry.sequence.load(Ordering::Acquire);
        if before != after || after & 1 != 0 || metadata & SHARED_OCCUPIED == 0 {
            return None;
        }
        let generation = u16::try_from(metadata & u32::from(u16::MAX)).ok()?;
        Some((key_xor_data, data, generation))
    }

    /// Used for mixing both halves of `key` and masking it into the table
    /// allocation.
    ///
    /// # Arguments
    ///
    /// * `key` - full 64-bit position hash key
    ///
    /// # Returns
    ///
    /// Bucket index in `0..self.len()`.
    ///
    /// The mixed key is masked with the full bucket mask rather than being
    /// truncated to 32 bits first. Truncation was equivalent for tables up to
    /// 2^32 buckets but silently capped every larger allocation onto the first
    /// four gibi-entries, which a competition-sized hash exceeds.
    #[inline]
    fn index(&self, key: u64) -> usize {
        let mixed = key ^ (key >> 32);
        let mask = u64::try_from(self.mask).expect("a bucket mask fits u64");
        usize::try_from(mixed & mask).expect("a masked bucket index fits usize")
    }
}

