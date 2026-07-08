//! Pre-allocated memory arena for transform intermediate buffers.
//!
//! Uses `Rc`-based interior mutability so `acquire` takes `&self` and
//! multiple guards can be live simultaneously.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArenaError {
    #[error("Arena exhausted: requested {requested} bytes but only {available} available")]
    Exhausted { requested: usize, available: usize },
    #[error("All buffers in use")]
    AllBusy,
}

struct PoolSlot {
    data: RefCell<Option<Vec<f32>>>,
    capacity_elements: usize,
    in_use: Cell<bool>,
    /// True once this slot has been leased at least once and returned.
    ever_returned: Cell<bool>,
}

impl PoolSlot {
    fn new(capacity_elements: usize) -> Rc<Self> {
        Rc::new(Self {
            data: RefCell::new(Some(vec![0.0f32; capacity_elements])),
            capacity_elements,
            in_use: Cell::new(false),
            ever_returned: Cell::new(false),
        })
    }
    fn size_bytes(&self) -> usize {
        self.capacity_elements * 4
    }
}

pub struct ArenaBuffer {
    pub(crate) data: Vec<f32>,
    pub(crate) capacity_elements: usize,
    pub(crate) in_use: bool,
}

pub struct ArenaGuard {
    data: Vec<f32>,
    elements: usize,
    slot: Rc<PoolSlot>,
    used_bytes: Rc<Cell<usize>>,
}

impl std::fmt::Debug for ArenaGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArenaGuard")
            .field("elements", &self.elements)
            .finish()
    }
}

impl std::ops::Deref for ArenaGuard {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.data[..self.elements]
    }
}
impl std::ops::DerefMut for ArenaGuard {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.data[..self.elements]
    }
}

impl Drop for ArenaGuard {
    fn drop(&mut self) {
        let mut data = std::mem::take(&mut self.data);
        data.resize(self.slot.capacity_elements, 0.0);
        *self.slot.data.borrow_mut() = Some(data);
        self.slot.in_use.set(false);
        // Mark that this slot has been returned at least once — future
        // acquisitions of this slot count as "reuse".
        self.slot.ever_returned.set(true);
        let released = self.elements * 4;
        self.used_bytes
            .set(self.used_bytes.get().saturating_sub(released));
    }
}

#[derive(Debug, Clone)]
pub struct ArenaStats {
    pub total_capacity_bytes: usize,
    pub used_bytes: usize,
    pub peak_used_bytes: usize,
    pub alloc_count: u64,
    pub reuse_count: u64,
    pub reuse_ratio: f64,
    pub num_buffers: usize,
    pub free_buffers: usize,
}

pub struct TransformArena {
    slots: RefCell<Vec<Rc<PoolSlot>>>,
    total_capacity_bytes: usize,
    used_bytes: Rc<Cell<usize>>,
    peak_used_bytes: Cell<usize>,
    alloc_count: Cell<u64>,
    reuse_count: Cell<u64>,
}

impl TransformArena {
    const INITIAL_BUFFERS: usize = 4;

    pub fn new(total_capacity_bytes: usize) -> Self {
        let mut slots: Vec<Rc<PoolSlot>> = Vec::with_capacity(Self::INITIAL_BUFFERS);
        if total_capacity_bytes > 0 {
            let bytes_per_slot = total_capacity_bytes / Self::INITIAL_BUFFERS;
            let elems = (bytes_per_slot / 4).max(1);
            for _ in 0..Self::INITIAL_BUFFERS {
                slots.push(PoolSlot::new(elems));
            }
        }
        Self {
            slots: RefCell::new(slots),
            total_capacity_bytes,
            used_bytes: Rc::new(Cell::new(0)),
            peak_used_bytes: Cell::new(0),
            alloc_count: Cell::new(0),
            reuse_count: Cell::new(0),
        }
    }

    pub fn acquire(&self, elements: usize) -> Result<ArenaGuard, ArenaError> {
        self.alloc_count.set(self.alloc_count.get() + 1);
        let requested_bytes = elements * 4;

        let reuse_slot: Option<Rc<PoolSlot>> = self
            .slots
            .borrow()
            .iter()
            .find(|s| !s.in_use.get() && s.capacity_elements >= elements)
            .cloned();

        if let Some(slot) = reuse_slot {
            // Only count as a reuse if the slot was previously leased and returned.
            if slot.ever_returned.get() {
                self.reuse_count.set(self.reuse_count.get() + 1);
            }
            slot.in_use.set(true);
            let mut data = slot
                .data
                .borrow_mut()
                .take()
                .expect("free slot must have data");
            data.resize(elements, 0.0);
            self.bump_used(requested_bytes);
            return Ok(ArenaGuard {
                data,
                elements,
                slot,
                used_bytes: Rc::clone(&self.used_bytes),
            });
        }

        let current: usize = self.slots.borrow().iter().map(|s| s.size_bytes()).sum();
        if self.total_capacity_bytes == 0 || current + requested_bytes > self.total_capacity_bytes {
            return Err(ArenaError::Exhausted {
                requested: requested_bytes,
                available: self.total_capacity_bytes.saturating_sub(current),
            });
        }

        let slot = PoolSlot::new(elements);
        slot.in_use.set(true);
        let data = slot
            .data
            .borrow_mut()
            .take()
            .expect("new slot must have data");
        self.slots.borrow_mut().push(Rc::clone(&slot));
        self.bump_used(requested_bytes);
        Ok(ArenaGuard {
            data,
            elements,
            slot,
            used_bytes: Rc::clone(&self.used_bytes),
        })
    }

    fn bump_used(&self, delta: usize) {
        let new = self.used_bytes.get() + delta;
        self.used_bytes.set(new);
        if new > self.peak_used_bytes.get() {
            self.peak_used_bytes.set(new);
        }
    }

    pub fn stats(&self) -> ArenaStats {
        let slots = self.slots.borrow();
        let ac = self.alloc_count.get();
        let rc = self.reuse_count.get();
        ArenaStats {
            total_capacity_bytes: self.total_capacity_bytes,
            used_bytes: self.used_bytes.get(),
            peak_used_bytes: self.peak_used_bytes.get(),
            alloc_count: ac,
            reuse_count: rc,
            reuse_ratio: if ac == 0 { 0.0 } else { rc as f64 / ac as f64 },
            num_buffers: slots.len(),
            free_buffers: slots.iter().filter(|s| !s.in_use.get()).count(),
        }
    }

    pub fn reset(&self) {
        for slot in self.slots.borrow().iter() {
            slot.in_use.set(false);
            let mut g = slot.data.borrow_mut();
            if g.is_none() {
                *g = Some(vec![0.0f32; slot.capacity_elements]);
            }
        }
        self.used_bytes.set(0);
    }

    #[inline]
    pub fn capacity_bytes(&self) -> usize {
        self.total_capacity_bytes
    }
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.used_bytes.get()
    }
    #[inline]
    pub fn reuse_ratio(&self) -> f64 {
        let ac = self.alloc_count.get();
        if ac == 0 {
            0.0
        } else {
            self.reuse_count.get() as f64 / ac as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const ARENA_8M: usize = 8 * 1024 * 1024;

    #[test]
    fn test_arena_acquire_and_use() {
        let arena = TransformArena::new(ARENA_8M);
        {
            let mut buf = arena.acquire(100).unwrap();
            for x in buf.iter_mut() {
                *x = 1.0;
            }
            assert!(buf.iter().all(|&x| x == 1.0));
        }
        assert_eq!(arena.used_bytes(), 0);
    }

    #[test]
    fn test_arena_acquire_multiple() {
        let arena = TransformArena::new(ARENA_8M);
        let mut a = arena.acquire(128).unwrap();
        let mut b = arena.acquire(256).unwrap();
        for x in a.iter_mut() {
            *x = 1.0;
        }
        for x in b.iter_mut() {
            *x = 2.0;
        }
        assert!(a.iter().all(|&x| x == 1.0));
        assert!(b.iter().all(|&x| x == 2.0));
        drop(a);
        drop(b);
        assert_eq!(arena.used_bytes(), 0);
    }

    #[test]
    fn test_arena_reuse_after_drop() {
        let arena = TransformArena::new(ARENA_8M);
        {
            let _b = arena.acquire(64).unwrap();
        }
        assert_eq!(arena.stats().reuse_count, 0);
        {
            let _b = arena.acquire(64).unwrap();
        }
        assert_eq!(arena.stats().reuse_count, 1);
    }

    #[test]
    fn test_arena_stats_after_allocs() {
        let arena = TransformArena::new(ARENA_8M);
        // Keep all 3 guards alive simultaneously so 3 different slots are used.
        {
            let _b1 = arena.acquire(100).unwrap();
            let _b2 = arena.acquire(200).unwrap();
            let _b3 = arena.acquire(300).unwrap();
        } // all 3 drop here → 3 slots have ever_returned=true
          // Second wave also acquires 3 simultaneously — each should reuse.
        {
            let _b4 = arena.acquire(100).unwrap();
            let _b5 = arena.acquire(200).unwrap();
            let _b6 = arena.acquire(300).unwrap();
        }
        let s = arena.stats();
        assert_eq!(s.alloc_count, 6);
        assert_eq!(s.reuse_count, 3, "second wave should all be reuses");
        assert_eq!(s.used_bytes, 0, "all guards dropped");
    }

    #[test]
    fn test_arena_capacity() {
        let arena = TransformArena::new(4 * 1024 * 1024);
        assert_eq!(arena.capacity_bytes(), 4 * 1024 * 1024);
    }

    #[test]
    fn test_arena_exhausted_error() {
        let arena = TransformArena::new(64);
        assert!(matches!(
            arena.acquire(32),
            Err(ArenaError::Exhausted { .. })
        ));
    }

    #[test]
    fn test_arena_reset_frees_all() {
        let arena = TransformArena::new(ARENA_8M);
        for _ in 0..4 {
            let _b = arena.acquire(512).unwrap();
        }
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        for _ in 0..4 {
            arena.acquire(512).unwrap();
        }
    }

    #[test]
    fn test_arena_peak_usage() {
        let arena = TransformArena::new(ARENA_8M);
        let _b1 = arena.acquire(1000).unwrap();
        let _b2 = arena.acquire(2000).unwrap();
        let expected = 3000 * 4;
        assert!(arena.stats().peak_used_bytes >= expected);
        drop(_b1);
        drop(_b2);
        assert_eq!(arena.stats().used_bytes, 0);
        assert!(arena.stats().peak_used_bytes >= expected);
    }

    #[test]
    fn test_arena_reuse_ratio() {
        let arena = TransformArena::new(ARENA_8M);
        {
            let _b = arena.acquire(512).unwrap();
        }
        for _ in 0..99 {
            let _b = arena.acquire(512).unwrap();
        }
        assert!(arena.stats().reuse_ratio > 0.95);
    }

    #[test]
    fn test_arena_zero_capacity() {
        let arena = TransformArena::new(0);
        assert!(arena.acquire(1).is_err());
    }

    #[test]
    fn test_arena_guard_debug() {
        let arena = TransformArena::new(ARENA_8M);
        let g = arena.acquire(32).unwrap();
        assert!(format!("{g:?}").contains("ArenaGuard"));
    }

    #[test]
    fn test_arena_multiple_simultaneous_guards() {
        let arena = TransformArena::new(ARENA_8M);
        let _g1 = arena.acquire(100).unwrap();
        let _g2 = arena.acquire(200).unwrap();
        let _g3 = arena.acquire(300).unwrap();
        assert_eq!(arena.stats().used_bytes, (100 + 200 + 300) * 4);
    }
}
