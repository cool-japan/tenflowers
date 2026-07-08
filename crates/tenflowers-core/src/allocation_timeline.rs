/// GPU Allocation Timeline — chronological allocation history with ring-buffer semantics.
///
/// This module records every `alloc` and `dealloc` event with a monotonically-increasing
/// sequence number and a running cumulative-usage counter. The timeline operates as a
/// ring buffer: once the capacity is reached the oldest events are dropped.

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// AllocationEvent
// ---------------------------------------------------------------------------

/// A single allocation or deallocation event captured by the timeline.
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    /// Unique, monotonically-increasing event identifier.
    pub event_id: u64,
    /// Name of the operation that triggered the event (e.g. `"matmul"`, `"conv2d"`).
    pub operation_name: String,
    /// Size of the allocation or deallocation in bytes.
    pub size_bytes: usize,
    /// `true` if this is an allocation; `false` if it is a deallocation.
    pub is_alloc: bool,
    /// Monotonically-increasing sequence counter (same as `event_id` in this impl).
    pub sequence_number: u64,
    /// Running total of bytes currently in use **after** this event was applied.
    pub cumulative_usage: usize,
}

// ---------------------------------------------------------------------------
// AllocationTimeline
// ---------------------------------------------------------------------------

/// Chronological history of GPU memory allocation and deallocation events.
///
/// Internally this is a ring buffer: once [`capacity`] events have been recorded
/// the oldest entries are discarded.  Recording can be paused via [`disable`] and
/// resumed via [`enable`].
pub struct AllocationTimeline {
    events: Mutex<Vec<AllocationEvent>>,
    event_counter: AtomicU64,
    /// Maximum number of events retained (ring-buffer capacity).
    capacity: usize,
    enabled: AtomicBool,
    /// Running cumulative bytes currently in use.
    cumulative_usage: AtomicUsize,
}

impl AllocationTimeline {
    /// Create a new timeline that retains at most `capacity` events.
    pub fn new(capacity: usize) -> Self {
        Self {
            events: Mutex::new(Vec::with_capacity(capacity.min(1024))),
            event_counter: AtomicU64::new(0),
            capacity,
            enabled: AtomicBool::new(true),
            cumulative_usage: AtomicUsize::new(0),
        }
    }

    /// Record a memory allocation of `size` bytes attributed to `op_name`.
    ///
    /// Does nothing when the timeline is disabled.
    pub fn record_alloc(&self, op_name: &str, size: usize) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let seq = self.event_counter.fetch_add(1, Ordering::Relaxed);
        let cumulative = self.cumulative_usage.fetch_add(size, Ordering::Relaxed) + size;

        let event = AllocationEvent {
            event_id: seq,
            operation_name: op_name.to_string(),
            size_bytes: size,
            is_alloc: true,
            sequence_number: seq,
            cumulative_usage: cumulative,
        };
        self.push_event(event);
    }

    /// Record a memory deallocation of `size` bytes attributed to `op_name`.
    ///
    /// Does nothing when the timeline is disabled.
    pub fn record_dealloc(&self, op_name: &str, size: usize) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let seq = self.event_counter.fetch_add(1, Ordering::Relaxed);
        // Saturating subtraction — dealloc should never underflow in correct usage.
        let prev = self.cumulative_usage.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
            Some(v.saturating_sub(size))
        });
        let cumulative = match prev {
            Ok(v) => v.saturating_sub(size),
            Err(v) => v.saturating_sub(size),
        };

        let event = AllocationEvent {
            event_id: seq,
            operation_name: op_name.to_string(),
            size_bytes: size,
            is_alloc: false,
            sequence_number: seq,
            cumulative_usage: cumulative,
        };
        self.push_event(event);
    }

    /// Internal helper — appends an event and enforces ring-buffer capacity.
    fn push_event(&self, event: AllocationEvent) {
        let mut guard = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return, // poisoned lock, skip recording
        };
        if self.capacity == 0 {
            return;
        }
        if guard.len() >= self.capacity {
            // Remove oldest entry to maintain ring-buffer semantics.
            guard.remove(0);
        }
        guard.push(event);
    }

    /// Return a snapshot of all retained events in chronological order.
    pub fn get_events(&self) -> Vec<AllocationEvent> {
        match self.events.lock() {
            Ok(g) => g.clone(),
            Err(_) => Vec::new(), // poisoned lock, return empty snapshot
        }
    }

    /// Return the `n` most-recent events in chronological order.
    pub fn get_recent_events(&self, n: usize) -> Vec<AllocationEvent> {
        let guard = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(), // poisoned lock, return empty
        };
        let len = guard.len();
        if n >= len {
            guard.clone()
        } else {
            guard[len - n..].to_vec()
        }
    }

    /// Enable event recording (default state after construction).
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable event recording.  All subsequent `record_*` calls are no-ops.
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Clear all retained events and reset the cumulative-usage counter.
    pub fn clear(&self) {
        let mut guard = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return, // poisoned lock, skip clear
        };
        guard.clear();
        self.cumulative_usage.store(0, Ordering::Relaxed);
        self.event_counter.store(0, Ordering::Relaxed);
    }

    /// Return the highest `cumulative_usage` value ever observed.
    pub fn peak_usage(&self) -> usize {
        let guard = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return 0, // poisoned lock, return safe default
        };
        guard.iter().map(|e| e.cumulative_usage).max().unwrap_or(0)
    }

    /// Return the sum of all allocation sizes recorded (not net usage).
    pub fn total_allocated_bytes(&self) -> usize {
        let guard = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return 0, // poisoned lock, return safe default
        };
        guard
            .iter()
            .filter(|e| e.is_alloc)
            .map(|e| e.size_bytes)
            .sum()
    }

    /// Return the sum of all deallocation sizes recorded.
    pub fn total_deallocated_bytes(&self) -> usize {
        let guard = match self.events.lock() {
            Ok(g) => g,
            Err(_) => return 0, // poisoned lock, return safe default
        };
        guard
            .iter()
            .filter(|e| !e.is_alloc)
            .map(|e| e.size_bytes)
            .sum()
    }

    /// Return `true` if recording is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Global instance
// ---------------------------------------------------------------------------

/// Process-wide allocation timeline with a ring-buffer capacity of 4 096 events.
pub static GLOBAL_TIMELINE: Lazy<AllocationTimeline> =
    Lazy::new(|| AllocationTimeline::new(4096));

/// Obtain a reference to the global [`AllocationTimeline`].
pub fn get_allocation_timeline() -> &'static AllocationTimeline {
    &GLOBAL_TIMELINE
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> AllocationTimeline {
        AllocationTimeline::new(10)
    }

    #[test]
    fn test_allocation_timeline_record_alloc() {
        let tl = fresh();
        tl.record_alloc("matmul", 100);
        tl.record_alloc("conv2d", 200);
        tl.record_alloc("relu", 50);
        let events = tl.get_events();
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(|e| e.is_alloc));
    }

    #[test]
    fn test_allocation_timeline_record_dealloc() {
        let tl = fresh();
        tl.record_alloc("matmul", 100);
        tl.record_dealloc("matmul", 100);
        let events = tl.get_events();
        assert_eq!(events.len(), 2);
        assert!(!events[1].is_alloc);
        assert_eq!(events[1].size_bytes, 100);
    }

    #[test]
    fn test_allocation_timeline_peak_usage() {
        let tl = fresh();
        tl.record_alloc("op_a", 100);
        tl.record_alloc("op_b", 200);
        tl.record_dealloc("op_a", 100);
        // peak should be 300 (after second alloc, before dealloc)
        assert_eq!(tl.peak_usage(), 300);
    }

    #[test]
    fn test_allocation_timeline_ring_buffer() {
        let tl = AllocationTimeline::new(3);
        for i in 0..6_u64 {
            tl.record_alloc("op", i as usize * 10 + 10);
        }
        let events = tl.get_events();
        // Only the last 3 events should be retained
        assert_eq!(events.len(), 3);
        // The oldest retained event should have the 4th alloc's seq number
        assert_eq!(events[0].sequence_number, 3);
    }

    #[test]
    fn test_allocation_timeline_enable_disable() {
        let tl = fresh();
        tl.disable();
        tl.record_alloc("ignored", 999);
        assert_eq!(tl.get_events().len(), 0, "disabled timeline must not record");
        tl.enable();
        tl.record_alloc("recorded", 10);
        assert_eq!(tl.get_events().len(), 1);
    }

    #[test]
    fn test_allocation_timeline_recent_events() {
        let tl = fresh();
        tl.record_alloc("a", 10);
        tl.record_alloc("b", 20);
        tl.record_alloc("c", 30);
        tl.record_alloc("d", 40);
        let recent = tl.get_recent_events(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].operation_name, "c");
        assert_eq!(recent[1].operation_name, "d");
    }

    #[test]
    fn test_allocation_timeline_totals() {
        let tl = fresh();
        tl.record_alloc("x", 500);
        tl.record_alloc("y", 300);
        tl.record_dealloc("x", 500);
        assert_eq!(tl.total_allocated_bytes(), 800);
        assert_eq!(tl.total_deallocated_bytes(), 500);
    }

    #[test]
    fn test_allocation_timeline_clear() {
        let tl = fresh();
        tl.record_alloc("op", 100);
        tl.clear();
        assert_eq!(tl.get_events().len(), 0);
        assert_eq!(tl.peak_usage(), 0);
    }
}
