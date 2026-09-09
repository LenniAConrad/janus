//! Deterministic per-worker cache for compressed tablebase blocks.
//!
//! Each alpha-beta worker owns one [`BlockCache`], so cache state never
//! crosses workers and identical probe sequences always produce identical
//! hits, loads, and evictions. Replacement is exact least-recently-used
//! over an intrusive doubly linked list; the map is only ever used for
//! point lookups, never iterated, so `HashMap`'s unspecified iteration
//! order cannot influence behavior.

use super::SyzygyError;
use std::collections::HashMap;

/// Sentinel index marking the absence of a neighbor in the LRU list.
const NO_SLOT: usize = usize::MAX;

/// Default per-worker cache budget in bytes.
///
/// Sixteen mebibytes comfortably holds the working set of blocks touched by
/// endgame probing while bounding per-worker memory alongside the
/// transposition table.
pub(crate) const DEFAULT_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// Identity of one cached block.
///
/// The file identifier is assigned by the table registry (one per physical
/// WDL or DTZ file) and the block index counts fixed-size compressed blocks
/// inside that file.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BlockKey {
    /// Used for identifying the physical table file.
    pub file: u32,
    /// Used for identifying the block within the file.
    pub block: u32,
}

/// One occupied cache slot linked into the LRU order.
struct Slot {
    /// Used for removing the map entry when this slot is evicted.
    key: BlockKey,
    /// Used for storing the owned block bytes.
    data: Box<[u8]>,
    /// Used for linking toward the more recently used neighbor.
    prev: usize,
    /// Used for linking toward the less recently used neighbor.
    next: usize,
}

/// Deterministic least-recently-used block cache budgeted by bytes.
///
/// Loads happen through [`BlockCache::get_or_load`]; on a miss the caller's
/// loader supplies the block and older blocks are evicted until the byte
/// budget holds again (the newest block always stays resident even if it
/// exceeds the budget on its own).
pub(crate) struct BlockCache {
    /// Used for bounding the total cached bytes.
    budget: usize,
    /// Used for tracking the currently cached bytes.
    used: usize,
    /// Used for point lookups from block identity to slot index.
    map: HashMap<BlockKey, usize>,
    /// Used for slab storage of slots; freed indices are recycled in LIFO
    /// order to keep allocation deterministic.
    slots: Vec<Option<Slot>>,
    /// Used for recycling slot indices after eviction.
    free: Vec<usize>,
    /// Used for marking the most recently used slot.
    head: usize,
    /// Used for marking the least recently used slot.
    tail: usize,
}

impl BlockCache {
    /// Used for creating an empty cache with the given byte budget.
    ///
    /// # Arguments
    ///
    /// * `budget` - maximum total bytes of cached block data
    ///
    /// # Returns
    ///
    /// An empty cache.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            used: 0,
            map: HashMap::new(),
            slots: Vec::new(),
            free: Vec::new(),
            head: NO_SLOT,
            tail: NO_SLOT,
        }
    }

    /// Used for fetching a block, loading and caching it on a miss.
    ///
    /// A hit moves the block to the most recently used position. On a miss
    /// the loader runs exactly once and the least recently used blocks are
    /// evicted until the budget holds; the loaded block itself is never
    /// evicted by its own insertion.
    ///
    /// # Arguments
    ///
    /// * `key` - identity of the requested block
    /// * `load` - fallible producer of the block bytes on a miss
    ///
    /// # Returns
    ///
    /// Borrowed block bytes, valid until the next cache call.
    ///
    /// # Errors
    ///
    /// Propagates the loader's [`SyzygyError`]; the cache is unchanged when
    /// the loader fails.
    pub fn get_or_load<F>(&mut self, key: BlockKey, load: F) -> Result<&[u8], SyzygyError>
    where
        F: FnOnce() -> Result<Vec<u8>, SyzygyError>,
    {
        if let Some(&index) = self.map.get(&key) {
            self.unlink(index);
            self.push_front(index);
            let slot = self.slots[index].as_ref().expect("cached slot occupied");
            return Ok(&slot.data);
        }

        let data = load()?.into_boxed_slice();
        self.used = self.used.saturating_add(data.len());
        let index = if let Some(recycled) = self.free.pop() {
            recycled
        } else {
            self.slots.push(None);
            self.slots.len() - 1
        };
        self.slots[index] = Some(Slot {
            key,
            data,
            prev: NO_SLOT,
            next: NO_SLOT,
        });
        self.map.insert(key, index);
        self.push_front(index);
        while self.used > self.budget && self.tail != index && self.tail != NO_SLOT {
            self.evict_tail();
        }
        let slot = self.slots[index].as_ref().expect("inserted slot occupied");
        Ok(&slot.data)
    }

    /// Used for detaching a slot from the LRU list without freeing it.
    ///
    /// # Arguments
    ///
    /// * `index` - slot index currently linked into the list
    fn unlink(&mut self, index: usize) {
        let (prev, next) = {
            let slot = self.slots[index].as_ref().expect("linked slot occupied");
            (slot.prev, slot.next)
        };
        if prev == NO_SLOT {
            self.head = next;
        } else {
            self.slots[prev]
                .as_mut()
                .expect("linked slot occupied")
                .next = next;
        }
        if next == NO_SLOT {
            self.tail = prev;
        } else {
            self.slots[next]
                .as_mut()
                .expect("linked slot occupied")
                .prev = prev;
        }
    }

    /// Used for linking a detached slot in as the most recently used.
    ///
    /// # Arguments
    ///
    /// * `index` - slot index not currently linked into the list
    fn push_front(&mut self, index: usize) {
        let old_head = self.head;
        {
            let slot = self.slots[index].as_mut().expect("slot occupied");
            slot.prev = NO_SLOT;
            slot.next = old_head;
        }
        if old_head != NO_SLOT {
            self.slots[old_head]
                .as_mut()
                .expect("linked slot occupied")
                .prev = index;
        }
        self.head = index;
        if self.tail == NO_SLOT {
            self.tail = index;
        }
    }

    /// Used for evicting the least recently used block.
    ///
    /// The freed slot index is recycled and the byte accounting shrinks by
    /// the evicted block's length.
    fn evict_tail(&mut self) {
        let index = self.tail;
        if index == NO_SLOT {
            return;
        }
        self.unlink(index);
        let slot = self.slots[index].take().expect("tail slot occupied");
        self.map.remove(&slot.key);
        self.used = self.used.saturating_sub(slot.data.len());
        self.free.push(index);
    }
}
