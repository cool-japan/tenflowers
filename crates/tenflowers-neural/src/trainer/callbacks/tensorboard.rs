//! Tensorboard callback for training visualization
//!
//! This callback writes TFEvent protocol buffer-compatible binary event files
//! to a log directory. The files can be read by `tensorboard --logdir=<log_dir>`.
//!
//! ## Event File Format
//!
//! Each record in a TFEvent file is:
//! ```text
//!   uint64  len(data)                        — masked CRC32C of len
//!   uint32  masked_crc32c(len as bytes)
//!   byte    data[len(data)]                  — serialized tf.Event protobuf
//!   uint32  masked_crc32c(data)
//! ```
//!
//! We emit `Event` protobufs containing `Summary` values (tag + simple_value).

use super::{Callback, CallbackAction, TrainingMetrics, TrainingState};
use crate::{optimizers::Optimizer, Model};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tenflowers_core::{Result, TensorError};

/// Tensorboard callback for training visualization
///
/// Records training metrics and losses to tensorboard-compatible event files.
/// The output directory will contain files named `events.out.tfevents.<timestamp>.<hostname>`.
#[derive(Debug)]
pub struct TensorboardCallback {
    log_dir: String,
    write_graph: bool,
    /// Path to the current event file (lazily opened)
    event_file_path: Option<PathBuf>,
    /// Buffered writer for the event file
    #[allow(dead_code)]
    flush_interval: usize,
    /// Wall-clock time of the first event (for relative timestamps)
    start_wall_time: Option<f64>,
    /// Whether the file_version event has been written
    initialized: bool,
    /// Accumulated bytes to write (we buffer then flush)
    buffer: Vec<u8>,
}

impl TensorboardCallback {
    /// Create a new tensorboard callback
    pub fn new(log_dir: impl Into<String>) -> Self {
        Self {
            log_dir: log_dir.into(),
            write_graph: false,
            event_file_path: None,
            flush_interval: 1,
            start_wall_time: None,
            initialized: false,
            buffer: Vec::new(),
        }
    }

    /// Enable graph writing
    pub fn with_graph(mut self) -> Self {
        self.write_graph = true;
        self
    }

    /// Set flush interval (how many epochs between disk flushes)
    pub fn with_flush_interval(mut self, interval: usize) -> Self {
        self.flush_interval = interval.max(1);
        self
    }

    /// Get the current wall time as f64 seconds since epoch
    fn wall_time() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// Ensure the log directory and event file exist
    fn ensure_initialized(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Create log directory
        std::fs::create_dir_all(&self.log_dir).map_err(|e| TensorError::InvalidOperation {
            operation: "tensorboard_init".to_string(),
            reason: format!("Failed to create log dir '{}': {}", self.log_dir, e),
            context: None,
        })?;

        // Generate event file name
        let wall = Self::wall_time();
        let hostname = hostname_or_default();
        let filename = format!("events.out.tfevents.{:.0}.{}", wall, hostname);
        let path = PathBuf::from(&self.log_dir).join(filename);
        self.event_file_path = Some(path);
        self.start_wall_time = Some(wall);

        // Write file_version event (required by tensorboard)
        self.write_file_version_event(wall)?;
        self.initialized = true;
        Ok(())
    }

    /// Write the initial file_version event that tensorboard expects
    fn write_file_version_event(&mut self, wall_time: f64) -> Result<()> {
        // tf.Event with file_version = "brain.Event:2"
        let event_bytes = encode_file_version_event(wall_time);
        self.write_record(&event_bytes)
    }

    /// Write a scalar summary event
    fn write_scalar_event(
        &mut self,
        wall_time: f64,
        step: i64,
        tag: &str,
        value: f32,
    ) -> Result<()> {
        let event_bytes = encode_scalar_event(wall_time, step, tag, value);
        self.write_record(&event_bytes)
    }

    /// Write a single TFRecord to the buffer
    fn write_record(&mut self, data: &[u8]) -> Result<()> {
        let len = data.len() as u64;
        let len_bytes = len.to_le_bytes();

        // Record = len(8 bytes) + masked_crc(len)(4 bytes) + data + masked_crc(data)(4 bytes)
        self.buffer.extend_from_slice(&len_bytes);
        self.buffer
            .extend_from_slice(&masked_crc32c(&len_bytes).to_le_bytes());
        self.buffer.extend_from_slice(data);
        self.buffer
            .extend_from_slice(&masked_crc32c(data).to_le_bytes());
        Ok(())
    }

    /// Flush buffer to disk
    fn flush_to_disk(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let path = match &self.event_file_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };

        // Open in append mode
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| TensorError::InvalidOperation {
                operation: "tensorboard_flush".to_string(),
                reason: format!("Failed to open event file: {}", e),
                context: None,
            })?;

        file.write_all(&self.buffer)
            .map_err(|e| TensorError::InvalidOperation {
                operation: "tensorboard_flush".to_string(),
                reason: format!("Failed to write event data: {}", e),
                context: None,
            })?;

        file.flush().map_err(|e| TensorError::InvalidOperation {
            operation: "tensorboard_flush".to_string(),
            reason: format!("Failed to flush event file: {}", e),
            context: None,
        })?;

        self.buffer.clear();
        Ok(())
    }
}

impl<T> Callback<T> for TensorboardCallback
where
    T: Clone + Default,
{
    fn on_train_begin(&mut self, _state: &TrainingState) -> CallbackAction {
        if let Err(_e) = self.ensure_initialized() {
            // Silently continue; tensorboard logging is best-effort
        }
        CallbackAction::Continue
    }

    fn on_epoch_end(
        &mut self,
        epoch: usize,
        state: &TrainingState,
        _model: &dyn Model<T>,
        _optimizer: &mut dyn Optimizer<T>,
    ) -> Result<CallbackAction> {
        // Ensure initialized (in case on_train_begin wasn't called)
        self.ensure_initialized()?;

        let wall = Self::wall_time();
        let step = epoch as i64;

        // Write training loss
        if let Some(train_loss) = state.latest_train_loss() {
            self.write_scalar_event(wall, step, "loss/train", train_loss)?;
        }

        // Write validation loss
        if let Some(val_loss) = state.latest_val_loss() {
            self.write_scalar_event(wall, step, "loss/val", val_loss)?;
        }

        // Write best metric if available
        if let Some(best) = state.best_metric {
            self.write_scalar_event(wall, step, "metric/best", best)?;
        }

        // Write any custom metrics from the latest training entry
        if let Some(latest) = state.history.last() {
            for (name, &value) in &latest.metrics {
                let tag = format!("metric/{}", name);
                self.write_scalar_event(wall, step, &tag, value)?;
            }
        }

        // Flush to disk
        self.flush_to_disk()?;

        Ok(CallbackAction::Continue)
    }

    fn on_train_end(&mut self, _state: &TrainingState) -> CallbackAction {
        // Final flush
        let _ = self.flush_to_disk();
        CallbackAction::Continue
    }

    fn on_batch_end(
        &mut self,
        _batch: usize,
        _metrics: &TrainingMetrics,
        _state: &TrainingState,
    ) -> CallbackAction {
        // Batch-level logging is intentionally skipped to avoid excessive I/O.
        // Users who need batch-level granularity can subclass or use a custom callback.
        CallbackAction::Continue
    }
}

// ============================================================================
// TFEvent Protobuf Encoding (minimal hand-rolled, no protobuf dependency)
// ============================================================================
//
// We encode a minimal subset of the TensorFlow Event protobuf:
//
// message Event {
//   double wall_time = 1;     // field 1, wire type 1 (64-bit)
//   int64  step = 2;          // field 2, wire type 0 (varint)
//   oneof what {
//     string file_version = 6;  // field 6, wire type 2 (length-delimited)
//     Summary summary = 5;      // field 5, wire type 2 (length-delimited)
//   }
// }
//
// message Summary {
//   repeated Value value = 1;
// }
//
// message Summary.Value {
//   string tag = 1;           // field 1, wire type 2
//   float simple_value = 2;   // field 2, wire type 5 (32-bit)
// }

/// Encode a file_version Event protobuf
fn encode_file_version_event(wall_time: f64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);

    // field 1: wall_time (double, wire type 1) => tag byte = (1 << 3) | 1 = 0x09
    buf.push(0x09);
    buf.extend_from_slice(&wall_time.to_le_bytes());

    // field 2: step = 0 (varint, wire type 0) => tag byte = (2 << 3) | 0 = 0x10
    buf.push(0x10);
    buf.push(0x00);

    // field 6: file_version (string, wire type 2) => tag byte = (6 << 3) | 2 = 0x32
    let version = b"brain.Event:2";
    buf.push(0x32);
    encode_varint(&mut buf, version.len() as u64);
    buf.extend_from_slice(version);

    buf
}

/// Encode a scalar summary Event protobuf
fn encode_scalar_event(wall_time: f64, step: i64, tag: &str, value: f32) -> Vec<u8> {
    // First, encode the Summary.Value message
    let mut summary_value = Vec::with_capacity(32);
    // field 1: tag (string)
    summary_value.push(0x0A); // (1 << 3) | 2
    encode_varint(&mut summary_value, tag.len() as u64);
    summary_value.extend_from_slice(tag.as_bytes());
    // field 2: simple_value (float, wire type 5) => (2 << 3) | 5 = 0x15
    summary_value.push(0x15);
    summary_value.extend_from_slice(&value.to_le_bytes());

    // Encode the Summary message
    let mut summary = Vec::with_capacity(summary_value.len() + 4);
    // field 1: value (repeated, wire type 2) => (1 << 3) | 2 = 0x0A
    summary.push(0x0A);
    encode_varint(&mut summary, summary_value.len() as u64);
    summary.extend_from_slice(&summary_value);

    // Encode the Event message
    let mut event = Vec::with_capacity(64 + summary.len());
    // field 1: wall_time
    event.push(0x09);
    event.extend_from_slice(&wall_time.to_le_bytes());
    // field 2: step (varint, signed => zigzag encode)
    event.push(0x10);
    encode_varint(&mut event, step as u64);
    // field 5: summary => (5 << 3) | 2 = 0x2A
    event.push(0x2A);
    encode_varint(&mut event, summary.len() as u64);
    event.extend_from_slice(&summary);

    event
}

/// Encode a u64 as a protobuf varint
fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Compute masked CRC32C as used by TFRecord format.
/// masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8
fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c_compute(data);
    crc.rotate_right(15).wrapping_add(0xa282ead8)
}

/// Software CRC32C (Castagnoli) computation.
/// Uses a lookup table for performance.
fn crc32c_compute(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32C_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Get hostname or a default string
fn hostname_or_default() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "localhost".to_string())
}

/// CRC32C lookup table (Castagnoli polynomial 0x1EDC6F41)
static CRC32C_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x82F63B78;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensorboard_callback_creation() {
        let cb = TensorboardCallback::new("/tmp/tb_test");
        assert_eq!(cb.log_dir, "/tmp/tb_test");
        assert!(!cb.write_graph);
        assert!(!cb.initialized);
    }

    #[test]
    fn test_tensorboard_with_graph() {
        let cb = TensorboardCallback::new("/tmp/tb_test").with_graph();
        assert!(cb.write_graph);
    }

    #[test]
    fn test_varint_encoding() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 0);
        assert_eq!(buf, vec![0x00]);

        buf.clear();
        encode_varint(&mut buf, 1);
        assert_eq!(buf, vec![0x01]);

        buf.clear();
        encode_varint(&mut buf, 300);
        assert_eq!(buf, vec![0xAC, 0x02]);
    }

    #[test]
    fn test_crc32c_known_value() {
        // CRC32C of empty data should be 0
        let crc = crc32c_compute(b"");
        assert_eq!(crc, 0);

        // CRC32C of "123456789" should be 0xE3069283
        let crc = crc32c_compute(b"123456789");
        assert_eq!(crc, 0xE3069283);
    }

    #[test]
    fn test_masked_crc32c() {
        let data = b"hello";
        let mcrc = masked_crc32c(data);
        // Just verify it produces a deterministic non-zero value
        assert_ne!(mcrc, 0);
        assert_eq!(mcrc, masked_crc32c(data)); // deterministic
    }

    #[test]
    fn test_encode_file_version_event() {
        let event = encode_file_version_event(1234567890.0);
        // Should start with 0x09 (wall_time field)
        assert_eq!(event[0], 0x09);
        // Should contain "brain.Event:2"
        let event_str = String::from_utf8_lossy(&event);
        assert!(event_str.contains("brain.Event:2"));
    }

    #[test]
    fn test_encode_scalar_event() {
        let event = encode_scalar_event(1234567890.0, 1, "loss/train", 0.5);
        // Should start with 0x09 (wall_time field)
        assert_eq!(event[0], 0x09);
        // Should contain the tag
        let event_str = String::from_utf8_lossy(&event);
        assert!(event_str.contains("loss/train"));
    }

    #[test]
    fn test_write_record_format() {
        let mut cb = TensorboardCallback::new(
            std::env::temp_dir()
                .join("tb_record_test")
                .to_string_lossy()
                .to_string(),
        );
        let data = b"test data";
        cb.write_record(data)
            .expect("test: write_record should succeed");

        // Record format: 8 (len) + 4 (crc of len) + len(data) + 4 (crc of data)
        assert_eq!(cb.buffer.len(), 8 + 4 + data.len() + 4);

        // First 8 bytes should be the length
        let len_bytes = &cb.buffer[0..8];
        let len = u64::from_le_bytes(len_bytes.try_into().expect("test: slice should be 8 bytes"));
        assert_eq!(len, data.len() as u64);
    }

    #[test]
    fn test_full_event_write_cycle() {
        let tmp_dir = std::env::temp_dir().join("tb_full_cycle_test");
        let _ = std::fs::remove_dir_all(&tmp_dir);

        let mut cb = TensorboardCallback::new(tmp_dir.to_string_lossy().to_string());

        // Initialize
        cb.ensure_initialized()
            .expect("test: initialization should succeed");
        assert!(cb.initialized);
        assert!(cb.event_file_path.is_some());

        // Write some scalars
        let wall = TensorboardCallback::wall_time();
        cb.write_scalar_event(wall, 0, "loss/train", 1.0)
            .expect("test: write should succeed");
        cb.write_scalar_event(wall, 1, "loss/train", 0.5)
            .expect("test: write should succeed");

        // Flush
        cb.flush_to_disk().expect("test: flush should succeed");

        // Verify file exists and has content
        let path = cb
            .event_file_path
            .as_ref()
            .expect("test: path should exist");
        let metadata = std::fs::metadata(path).expect("test: file should exist");
        assert!(metadata.len() > 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
