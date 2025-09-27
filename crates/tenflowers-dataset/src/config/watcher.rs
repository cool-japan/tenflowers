//! Configuration file hot-reload watcher
//!
//! This module provides functionality to watch configuration files for changes
//! and automatically reload them when modifications are detected.

use crate::{Result, TensorError};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, SystemTime};

/// Events that can occur during configuration file watching
#[derive(Debug, Clone)]
pub enum WatchEvent {
    /// Configuration file was modified
    Modified(PathBuf),
    /// Error occurred during watching
    Error(String),
}

/// Configuration file watcher for hot-reload functionality
#[derive(Debug)]
pub struct ConfigWatcher {
    /// Path to the configuration file being watched
    file_path: PathBuf,
    /// Last modification time of the file
    last_modified: Option<SystemTime>,
    /// Channel receiver for file system events (when available)
    #[allow(dead_code)]
    receiver: Option<Receiver<WatchEvent>>,
    /// Polling interval for file changes
    poll_interval: Duration,
    /// Last poll time
    last_poll: SystemTime,
}

impl ConfigWatcher {
    /// Create a new configuration file watcher
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();

        // Verify file exists
        if !file_path.exists() {
            return Err(TensorError::invalid_argument(format!(
                "Configuration file does not exist: {}",
                file_path.display()
            )));
        }

        // Get initial modification time
        let last_modified = std::fs::metadata(&file_path)
            .map_err(|e| {
                TensorError::invalid_argument(format!(
                    "Failed to get file metadata for {}: {}",
                    file_path.display(),
                    e
                ))
            })?
            .modified()
            .ok();

        Ok(Self {
            file_path,
            last_modified,
            receiver: None,
            poll_interval: Duration::from_secs(1),
            last_poll: SystemTime::now(),
        })
    }

    /// Create a watcher with custom polling interval
    pub fn with_poll_interval<P: AsRef<Path>>(path: P, interval: Duration) -> Result<Self> {
        let mut watcher = Self::new(path)?;
        watcher.poll_interval = interval;
        Ok(watcher)
    }

    /// Get the path being watched
    pub fn path(&self) -> &Path {
        &self.file_path
    }

    /// Get the polling interval
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Set the polling interval
    pub fn set_poll_interval(&mut self, interval: Duration) {
        self.poll_interval = interval;
    }

    /// Check for file changes (polling-based implementation)
    pub fn check_changes(&mut self) -> Result<Option<WatchEvent>> {
        let now = SystemTime::now();

        // Check if enough time has passed since last poll
        if now
            .duration_since(self.last_poll)
            .unwrap_or(Duration::from_secs(0))
            < self.poll_interval
        {
            return Ok(None);
        }

        self.last_poll = now;

        // Check if file still exists
        if !self.file_path.exists() {
            return Ok(Some(WatchEvent::Error(format!(
                "Configuration file no longer exists: {}",
                self.file_path.display()
            ))));
        }

        // Get current modification time
        let current_modified = std::fs::metadata(&self.file_path)
            .map_err(|e| {
                TensorError::invalid_argument(format!(
                    "Failed to get file metadata for {}: {}",
                    self.file_path.display(),
                    e
                ))
            })?
            .modified()
            .ok();

        // Compare with last known modification time
        if current_modified != self.last_modified {
            self.last_modified = current_modified;
            return Ok(Some(WatchEvent::Modified(self.file_path.clone())));
        }

        // Check for events from file system watcher (if available)
        if let Some(ref receiver) = self.receiver {
            match receiver.try_recv() {
                Ok(event) => return Ok(Some(event)),
                Err(TryRecvError::Empty) => {} // No events
                Err(TryRecvError::Disconnected) => {
                    return Ok(Some(WatchEvent::Error(
                        "File watcher disconnected".to_string(),
                    )));
                }
            }
        }

        Ok(None)
    }

    /// Start watching with native file system events (if available)
    pub fn start_native_watching(&mut self) -> Result<()> {
        // This would use a file system watcher like notify in a real implementation
        // For now, we'll use polling-based watching as a fallback
        self.start_polling_watching()
    }

    /// Start polling-based watching
    pub fn start_polling_watching(&mut self) -> Result<()> {
        // Polling is already implemented in check_changes()
        // This method is for consistency with the API
        Ok(())
    }

    /// Stop watching
    pub fn stop_watching(&mut self) {
        self.receiver = None;
    }

    /// Check if the watcher is currently active
    pub fn is_watching(&self) -> bool {
        self.receiver.is_some()
    }

    /// Get file information
    pub fn file_info(&self) -> Result<FileInfo> {
        let metadata = std::fs::metadata(&self.file_path).map_err(|e| {
            TensorError::invalid_argument(format!(
                "Failed to get file metadata for {}: {}",
                self.file_path.display(),
                e
            ))
        })?;

        Ok(FileInfo {
            path: self.file_path.clone(),
            size: metadata.len(),
            modified: metadata.modified().ok(),
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
        })
    }

    /// Wait for the next change event (blocking)
    pub fn wait_for_change(&mut self, timeout: Option<Duration>) -> Result<Option<WatchEvent>> {
        let start_time = SystemTime::now();

        loop {
            if let Some(event) = self.check_changes()? {
                return Ok(Some(event));
            }

            // Check timeout
            if let Some(timeout_duration) = timeout {
                if start_time.elapsed().unwrap_or(Duration::from_secs(0)) >= timeout_duration {
                    return Ok(None);
                }
            }

            // Sleep for a short time to avoid busy waiting
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Get the last modification time
    pub fn last_modified(&self) -> Option<SystemTime> {
        self.last_modified
    }

    /// Force a check for changes regardless of polling interval
    pub fn force_check(&mut self) -> Result<Option<WatchEvent>> {
        let old_poll_time = self.last_poll;
        self.last_poll = SystemTime::UNIX_EPOCH; // Force check
        let result = self.check_changes();
        self.last_poll = old_poll_time;
        result
    }
}

/// File information structure
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// Last modification time
    pub modified: Option<SystemTime>,
    /// Whether this is a regular file
    pub is_file: bool,
    /// Whether this is a directory
    pub is_dir: bool,
}

impl FileInfo {
    /// Get a human-readable description of the file
    pub fn description(&self) -> String {
        let file_type = if self.is_file {
            "file"
        } else if self.is_dir {
            "directory"
        } else {
            "unknown"
        };

        let size_str = if self.size < 1024 {
            format!("{} B", self.size)
        } else if self.size < 1024 * 1024 {
            format!("{:.1} KB", self.size as f64 / 1024.0)
        } else {
            format!("{:.1} MB", self.size as f64 / (1024.0 * 1024.0))
        };

        let modified_str = if let Some(modified) = self.modified {
            format!("{:?}", modified)
        } else {
            "unknown".to_string()
        };

        format!(
            "{} ({}, {}, modified: {})",
            self.path.display(),
            file_type,
            size_str,
            modified_str
        )
    }
}

/// Multi-file watcher for watching multiple configuration files
#[derive(Debug)]
pub struct MultiFileWatcher {
    /// Individual file watchers
    watchers: Vec<ConfigWatcher>,
    /// Global polling interval
    poll_interval: Duration,
}

impl MultiFileWatcher {
    /// Create a new multi-file watcher
    pub fn new() -> Self {
        Self {
            watchers: Vec::new(),
            poll_interval: Duration::from_secs(1),
        }
    }

    /// Add a file to watch
    pub fn add_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let watcher = ConfigWatcher::with_poll_interval(path, self.poll_interval)?;
        self.watchers.push(watcher);
        Ok(())
    }

    /// Remove a file from watching
    pub fn remove_file<P: AsRef<Path>>(&mut self, path: P) -> bool {
        let path = path.as_ref();
        if let Some(pos) = self.watchers.iter().position(|w| w.path() == path) {
            self.watchers.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check for changes in any watched file
    pub fn check_changes(&mut self) -> Result<Vec<WatchEvent>> {
        let mut events = Vec::new();

        for watcher in &mut self.watchers {
            if let Some(event) = watcher.check_changes()? {
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Get the number of files being watched
    pub fn file_count(&self) -> usize {
        self.watchers.len()
    }

    /// Get paths of all watched files
    pub fn watched_paths(&self) -> Vec<&Path> {
        self.watchers.iter().map(|w| w.path()).collect()
    }

    /// Set polling interval for all watchers
    pub fn set_poll_interval(&mut self, interval: Duration) {
        self.poll_interval = interval;
        for watcher in &mut self.watchers {
            watcher.set_poll_interval(interval);
        }
    }

    /// Get file information for all watched files
    pub fn file_infos(&self) -> Result<Vec<FileInfo>> {
        self.watchers
            .iter()
            .map(|w| w.file_info())
            .collect::<Result<Vec<_>>>()
    }
}

impl Default for MultiFileWatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_watcher_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let watcher = ConfigWatcher::new(temp_file.path()).unwrap();
        assert_eq!(watcher.path(), temp_file.path());
        assert_eq!(watcher.poll_interval(), Duration::from_secs(1));
    }

    #[test]
    fn test_nonexistent_file() {
        let result = ConfigWatcher::new("/nonexistent/path/config.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_poll_interval() {
        let temp_file = NamedTempFile::new().unwrap();
        let interval = Duration::from_millis(500);
        let watcher = ConfigWatcher::with_poll_interval(temp_file.path(), interval).unwrap();
        assert_eq!(watcher.poll_interval(), interval);
    }

    #[test]
    fn test_file_change_detection() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let mut watcher =
            ConfigWatcher::with_poll_interval(temp_file.path(), Duration::from_millis(10)).unwrap();

        // Initial check should return no changes
        let initial_check = watcher.check_changes().unwrap();
        assert!(initial_check.is_none());

        // Wait a bit and modify the file
        std::thread::sleep(Duration::from_millis(20));
        writeln!(temp_file, "new content").unwrap();
        temp_file.flush().unwrap();

        // Wait for polling interval
        std::thread::sleep(Duration::from_millis(20));

        // Should detect change
        let change_check = watcher.force_check().unwrap();
        match change_check {
            Some(WatchEvent::Modified(path)) => {
                assert_eq!(path, temp_file.path());
            }
            _ => panic!("Expected Modified event"),
        }
    }

    #[test]
    fn test_file_info() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test content").unwrap();
        temp_file.flush().unwrap();

        let watcher = ConfigWatcher::new(temp_file.path()).unwrap();
        let file_info = watcher.file_info().unwrap();

        assert_eq!(file_info.path, temp_file.path());
        assert!(file_info.is_file);
        assert!(!file_info.is_dir);
        assert!(file_info.size > 0);
        assert!(file_info.modified.is_some());

        let description = file_info.description();
        assert!(description.contains("file"));
        assert!(description.contains("B")); // Size in bytes
    }

    #[test]
    fn test_wait_for_change_timeout() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut watcher =
            ConfigWatcher::with_poll_interval(temp_file.path(), Duration::from_millis(10)).unwrap();

        let start_time = SystemTime::now();
        let result = watcher
            .wait_for_change(Some(Duration::from_millis(50)))
            .unwrap();
        let elapsed = start_time.elapsed().unwrap();

        assert!(result.is_none()); // Should timeout
        assert!(elapsed >= Duration::from_millis(50));
        assert!(elapsed < Duration::from_millis(200)); // Should not take too long
    }

    #[test]
    fn test_multi_file_watcher() {
        let temp_file1 = NamedTempFile::new().unwrap();
        let temp_file2 = NamedTempFile::new().unwrap();

        let mut multi_watcher = MultiFileWatcher::new();
        assert_eq!(multi_watcher.file_count(), 0);

        multi_watcher.add_file(temp_file1.path()).unwrap();
        multi_watcher.add_file(temp_file2.path()).unwrap();
        assert_eq!(multi_watcher.file_count(), 2);

        let watched_paths = multi_watcher.watched_paths();
        assert!(watched_paths.contains(&temp_file1.path()));
        assert!(watched_paths.contains(&temp_file2.path()));

        // Remove one file
        let removed = multi_watcher.remove_file(temp_file1.path());
        assert!(removed);
        assert_eq!(multi_watcher.file_count(), 1);

        // Try to remove non-existent file
        let not_removed = multi_watcher.remove_file("/nonexistent/path");
        assert!(!not_removed);
    }

    #[test]
    fn test_multi_file_watcher_changes() {
        let mut temp_file1 = NamedTempFile::new().unwrap();
        let mut temp_file2 = NamedTempFile::new().unwrap();

        let mut multi_watcher = MultiFileWatcher::new();
        multi_watcher.set_poll_interval(Duration::from_millis(10));
        multi_watcher.add_file(temp_file1.path()).unwrap();
        multi_watcher.add_file(temp_file2.path()).unwrap();

        // Initial check should return no changes
        let initial_changes = multi_watcher.check_changes().unwrap();
        assert!(initial_changes.is_empty());

        // Modify both files
        std::thread::sleep(Duration::from_millis(20));
        writeln!(temp_file1, "content1").unwrap();
        temp_file1.flush().unwrap();
        writeln!(temp_file2, "content2").unwrap();
        temp_file2.flush().unwrap();

        // Wait for polling interval
        std::thread::sleep(Duration::from_millis(20));

        // Should detect changes in both files
        let changes = multi_watcher.check_changes().unwrap();
        assert_eq!(changes.len(), 2);

        for change in changes {
            match change {
                WatchEvent::Modified(path) => {
                    assert!(path == temp_file1.path() || path == temp_file2.path());
                }
                _ => panic!("Expected Modified event"),
            }
        }
    }

    #[test]
    fn test_file_info_descriptions() {
        let temp_file = NamedTempFile::new().unwrap();
        let watcher = ConfigWatcher::new(temp_file.path()).unwrap();
        let file_info = watcher.file_info().unwrap();

        let description = file_info.description();
        assert!(description.contains(temp_file.path().to_str().unwrap()));
        assert!(description.contains("file"));
        assert!(
            description.contains("B") || description.contains("KB") || description.contains("MB")
        );
        assert!(description.contains("modified:"));
    }

    #[test]
    fn test_watcher_state_management() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut watcher = ConfigWatcher::new(temp_file.path()).unwrap();

        assert!(!watcher.is_watching());

        watcher.start_polling_watching().unwrap();
        // Polling watching doesn't change the is_watching state in this implementation

        watcher.stop_watching();
        assert!(!watcher.is_watching());
    }
}
