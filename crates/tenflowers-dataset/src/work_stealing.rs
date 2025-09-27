//! Work-stealing queue for efficient multi-threaded data loading
//!
//! This module provides a work-stealing queue implementation that allows
//! worker threads to steal work from other threads when they run out of tasks,
//! leading to better load balancing and higher throughput.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// A work-stealing queue that allows multiple workers to efficiently share work.
/// Workers can push/pop from their own queue and steal from others when idle.
pub struct WorkStealingQueue<T> {
    /// Local queues for each worker thread
    worker_queues: Vec<Arc<Mutex<VecDeque<T>>>>,
    /// Number of worker threads
    num_workers: usize,
    /// Atomic counter for round-robin work distribution
    next_worker: AtomicUsize,
    /// Signal for shutdown
    shutdown: Arc<AtomicBool>,
    /// Condition variable for blocking when no work is available
    work_available: Arc<(Mutex<bool>, Condvar)>,
    /// Total number of tasks in the system
    total_tasks: AtomicUsize,
}

impl<T> WorkStealingQueue<T> {
    /// Create a new work-stealing queue with the specified number of workers
    pub fn new(num_workers: usize) -> Self {
        let mut worker_queues = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            worker_queues.push(Arc::new(Mutex::new(VecDeque::new())));
        }

        Self {
            worker_queues,
            num_workers,
            next_worker: AtomicUsize::new(0),
            shutdown: Arc::new(AtomicBool::new(false)),
            work_available: Arc::new((Mutex::new(false), Condvar::new())),
            total_tasks: AtomicUsize::new(0),
        }
    }

    /// Push work to the queue. Uses round-robin distribution to balance load.
    pub fn push(&self, item: T) {
        let worker_id = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.num_workers;

        {
            let mut queue = self.worker_queues[worker_id].lock().unwrap();
            queue.push_back(item);
        }

        // Update task count and notify waiting workers
        self.total_tasks.fetch_add(1, Ordering::Relaxed);
        let (lock, cvar) = &*self.work_available;
        {
            let mut available = lock.lock().unwrap();
            *available = true;
        }
        cvar.notify_all();
    }

    /// Pop work from a specific worker's queue (used by the worker itself)
    pub fn pop(&self, worker_id: usize) -> Option<T> {
        if worker_id >= self.num_workers {
            return None;
        }

        let mut queue = self.worker_queues[worker_id].lock().unwrap();
        let item = queue.pop_front();
        if item.is_some() {
            self.total_tasks.fetch_sub(1, Ordering::Relaxed);
        }
        item
    }

    /// Steal work from other workers when this worker has no work
    pub fn steal(&self, worker_id: usize) -> Option<T> {
        if worker_id >= self.num_workers {
            return None;
        }

        // Try to steal from other workers, starting from a random offset
        let start_offset = (worker_id + 1) % self.num_workers;

        for i in 0..self.num_workers - 1 {
            let target_worker = (start_offset + i) % self.num_workers;
            if target_worker == worker_id {
                continue; // Skip self
            }

            let mut queue = self.worker_queues[target_worker].lock().unwrap();
            // Steal from the back to minimize contention with the owner
            if let Some(item) = queue.pop_back() {
                self.total_tasks.fetch_sub(1, Ordering::Relaxed);
                return Some(item);
            }
        }

        None
    }

    /// Try to get work for a specific worker (pop from own queue or steal from others)
    pub fn get_work(&self, worker_id: usize) -> Option<T> {
        // First try to pop from own queue
        if let Some(item) = self.pop(worker_id) {
            return Some(item);
        }

        // If no local work, try to steal from others
        self.steal(worker_id)
    }

    /// Wait for work to become available (blocking operation)
    pub fn wait_for_work(&self, worker_id: usize, timeout_ms: Option<u64>) -> Option<T> {
        // First check if work is immediately available
        if let Some(item) = self.get_work(worker_id) {
            return Some(item);
        }

        // If shutdown is signaled, return immediately
        if self.shutdown.load(Ordering::Relaxed) {
            return None;
        }

        // Wait for work to become available
        let (lock, cvar) = &*self.work_available;
        let mut available = lock.lock().unwrap();

        loop {
            // Check for shutdown signal
            if self.shutdown.load(Ordering::Relaxed) {
                return None;
            }

            // Try to get work again
            drop(available); // Release lock temporarily
            if let Some(item) = self.get_work(worker_id) {
                return Some(item);
            }
            available = lock.lock().unwrap();

            // If still no work and no tasks in the system, we're done
            if self.total_tasks.load(Ordering::Relaxed) == 0 {
                *available = false;
                return None;
            }

            // Wait for notification or timeout
            available = if let Some(timeout) = timeout_ms {
                let (guard, result) = cvar
                    .wait_timeout(available, std::time::Duration::from_millis(timeout))
                    .unwrap();
                if result.timed_out() {
                    return None;
                }
                guard
            } else {
                cvar.wait(available).unwrap()
            };
        }
    }

    /// Signal shutdown to all workers
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let (lock, cvar) = &*self.work_available;
        {
            let mut available = lock.lock().unwrap();
            *available = true;
        }
        cvar.notify_all();
    }

    /// Check if the queue is empty (all workers have no work)
    pub fn is_empty(&self) -> bool {
        self.total_tasks.load(Ordering::Relaxed) == 0
    }

    /// Get the total number of tasks currently in the system
    pub fn total_tasks(&self) -> usize {
        self.total_tasks.load(Ordering::Relaxed)
    }

    /// Get statistics about each worker's queue length
    pub fn queue_lengths(&self) -> Vec<usize> {
        self.worker_queues
            .iter()
            .map(|queue| queue.lock().unwrap().len())
            .collect()
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

impl<T> Clone for WorkStealingQueue<T> {
    fn clone(&self) -> Self {
        Self {
            worker_queues: self.worker_queues.clone(),
            num_workers: self.num_workers,
            next_worker: AtomicUsize::new(self.next_worker.load(Ordering::Relaxed)),
            shutdown: Arc::clone(&self.shutdown),
            work_available: Arc::clone(&self.work_available),
            total_tasks: AtomicUsize::new(self.total_tasks.load(Ordering::Relaxed)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_push_pop() {
        let queue = WorkStealingQueue::new(2);

        queue.push(1);
        queue.push(2);

        assert_eq!(queue.total_tasks(), 2);
        assert_eq!(queue.pop(0), Some(1));
        assert_eq!(queue.total_tasks(), 1);
        assert_eq!(queue.pop(1), Some(2));
        assert_eq!(queue.total_tasks(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_work_stealing() {
        let queue = WorkStealingQueue::new(3);

        // Push work to worker 0
        queue.push(1);
        queue.push(2);
        queue.push(3);

        // Worker 1 should be able to steal work
        assert!(queue.steal(1).is_some());
        assert_eq!(queue.total_tasks(), 2);
    }

    #[test]
    fn test_get_work() {
        let queue = WorkStealingQueue::new(2);

        queue.push(1);
        queue.push(2);

        // Worker 0 should get work (either from own queue or by stealing)
        assert!(queue.get_work(0).is_some());
        assert!(queue.get_work(1).is_some());
        assert!(queue.get_work(0).is_none());
    }

    #[test]
    fn test_concurrent_access() {
        let queue = Arc::new(WorkStealingQueue::new(4));

        // Add work first to ensure it's available when workers start
        for i in 0..100 {
            queue.push(i);
        }

        let mut handles = Vec::new();

        // Spawn workers after work is added
        for worker_id in 0..4 {
            let queue_clone = Arc::clone(&queue);
            let handle = thread::spawn(move || {
                let mut processed = 0;
                // Use wait_for_work to handle the case where work might not be immediately available
                while let Some(_item) = queue_clone.wait_for_work(worker_id, Some(50)) {
                    processed += 1;
                    thread::sleep(Duration::from_millis(1));
                }
                processed
            });
            handles.push(handle);
        }

        // Give workers time to process all work
        thread::sleep(Duration::from_millis(200));
        queue.shutdown();

        // Collect results
        let total_processed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

        assert_eq!(total_processed, 100);
    }

    #[test]
    fn test_queue_lengths() {
        let queue = WorkStealingQueue::new(3);

        // Add items and check distribution
        for i in 0..6 {
            queue.push(i);
        }

        let lengths = queue.queue_lengths();
        assert_eq!(lengths.len(), 3);
        assert_eq!(lengths.iter().sum::<usize>(), 6);
    }
}
