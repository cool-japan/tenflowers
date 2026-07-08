//! Advanced TFRecord features: SequenceExample support, proper feature-type
//! dispatch, and hardened CRC32 validation with masked-CRC per the TF spec.
//!
//! This module extends the base TFRecord implementation with:
//!
//! - **`TfRecordSequenceReader`** — reads `SequenceExample` proto format
//!   (context features + per-step feature lists) via a streaming iterator.
//! - **`parse_example_proto`** — proper `BytesList` / `FloatList` / `Int64List`
//!   dispatch that also handles the varint-length-delimited field encoding.
//! - **`verify_masked_crc32`** — TensorFlow's masked CRC per the wire spec
//!   (`crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul`).
//! - **`TfRecordRawReader`** — low-level record-by-record reader that yields
//!   raw `(data: Vec<u8>, crc_ok: bool)` pairs.
//!
//! All types are gated behind the `tfrecord` feature; stubs return
//! `Err("tfrecord feature not enabled")` when the feature is absent.
//!
//! # Wire format reminder
//!
//! Each TFRecord is:
//!
//! ```text
//!  uint64  length          (little-endian)
//!  uint32  masked_crc32_of_length
//!  byte[]  data            (length bytes)
//!  uint32  masked_crc32_of_data
//! ```
//!
//! The masked CRC is defined as:
//! ```text
//!  masked = ((crc >> 15) | (crc << 17)) + 0xa282ead8
//! ```

// ---------------------------------------------------------------------------
// Feature-enabled imports
// ---------------------------------------------------------------------------

#[cfg(feature = "tfrecord")]
use std::collections::HashMap;
#[cfg(feature = "tfrecord")]
use std::fs::File;
#[cfg(feature = "tfrecord")]
use std::io::{BufReader, Read};
#[cfg(feature = "tfrecord")]
use std::path::Path;

#[cfg(feature = "tfrecord")]
use crc32fast::Hasher;

#[cfg(feature = "tfrecord")]
use tenflowers_core::{Result, TensorError};

// Re-use the Feature enum from the base module. The base `tfrecord` module
// only exists when the feature is enabled, so this import is feature-gated.
#[cfg(feature = "tfrecord")]
use crate::formats::tfrecord::Feature;

// ---------------------------------------------------------------------------
// CRC helpers
// ---------------------------------------------------------------------------

/// Constant used in TF's masked CRC calculation.
#[cfg(feature = "tfrecord")]
const MASKED_CRC_DELTA: u32 = 0xa282_ead8;

/// Compute the TensorFlow-masked CRC32 of `data`.
///
/// TF uses `masked_crc = rotate_right_15(crc) + 0xa282ead8` so that a
/// string of zeros doesn't hash to the "zero" CRC.
#[cfg(feature = "tfrecord")]
pub fn masked_crc32(data: &[u8]) -> u32 {
    let mut h = Hasher::new();
    h.update(data);
    let crc = h.finalize();
    crc.rotate_right(15).wrapping_add(MASKED_CRC_DELTA)
}

/// Verify that `expected_masked_crc` (read from the file) matches the masked
/// CRC of `data`. Returns `Ok(())` on success, `Err(...)` on mismatch.
#[cfg(feature = "tfrecord")]
pub fn verify_masked_crc32(data: &[u8], expected_masked_crc: u32) -> Result<()> {
    let actual = masked_crc32(data);
    if actual == expected_masked_crc {
        Ok(())
    } else {
        Err(TensorError::invalid_argument(format!(
            "CRC mismatch: expected {expected_masked_crc:#010x}, got {actual:#010x}"
        )))
    }
}

// Stub
#[cfg(not(feature = "tfrecord"))]
pub fn masked_crc32(_data: &[u8]) -> u32 {
    0
}

#[cfg(not(feature = "tfrecord"))]
pub fn verify_masked_crc32(_data: &[u8], _expected_masked_crc: u32) -> tenflowers_core::Result<()> {
    Err(tenflowers_core::TensorError::invalid_argument(
        "tfrecord feature not enabled".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Raw record
// ---------------------------------------------------------------------------

/// A single record read from a TFRecord file (raw bytes + CRC status).
#[derive(Debug, Clone)]
pub struct RawTfRecord {
    /// Record payload bytes
    pub data: Vec<u8>,
    /// Whether the data CRC validated correctly
    pub crc_ok: bool,
}

// ---------------------------------------------------------------------------
// Raw reader — low-level streaming
// ---------------------------------------------------------------------------

/// Low-level streaming reader that yields [`RawTfRecord`] one at a time.
///
/// Uses the correct TF masked-CRC wire format (length CRC + data CRC, both
/// masked).
pub struct TfRecordRawReader {
    #[cfg(feature = "tfrecord")]
    reader: Box<dyn Read>,
    #[cfg(feature = "tfrecord")]
    validate_crc: bool,
    #[cfg(not(feature = "tfrecord"))]
    _phantom: (),
}

#[cfg(feature = "tfrecord")]
impl TfRecordRawReader {
    /// Open `file_path` for streaming TFRecord reads.
    ///
    /// Set `validate_crc = true` to verify both the length-CRC and data-CRC
    /// using TF's masked CRC algorithm.
    pub fn open<P: AsRef<Path>>(file_path: P, validate_crc: bool) -> Result<Self> {
        let file = File::open(file_path.as_ref()).map_err(|e| {
            TensorError::invalid_argument(format!("Cannot open TFRecord file: {e}"))
        })?;
        Ok(Self {
            reader: Box::new(BufReader::with_capacity(65536, file)),
            validate_crc,
        })
    }

    fn read_record(&mut self) -> Result<Option<RawTfRecord>> {
        // --- length (8 bytes) ---
        let mut len_buf = [0u8; 8];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => {
                return Err(TensorError::invalid_argument(format!(
                    "Failed to read record length: {e}"
                )))
            }
        }
        let length = u64::from_le_bytes(len_buf);

        // --- masked CRC of length (4 bytes) ---
        let mut len_crc_buf = [0u8; 4];
        self.reader.read_exact(&mut len_crc_buf).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read length CRC: {e}"))
        })?;
        let expected_len_crc = u32::from_le_bytes(len_crc_buf);

        if self.validate_crc {
            verify_masked_crc32(&len_buf, expected_len_crc)?;
        }

        // --- data ---
        let mut data = vec![0u8; length as usize];
        self.reader.read_exact(&mut data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read record data: {e}"))
        })?;

        // --- masked CRC of data (4 bytes) ---
        let mut data_crc_buf = [0u8; 4];
        self.reader
            .read_exact(&mut data_crc_buf)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to read data CRC: {e}")))?;
        let expected_data_crc = u32::from_le_bytes(data_crc_buf);

        let crc_ok = if self.validate_crc {
            verify_masked_crc32(&data, expected_data_crc)
                .map(|()| true)
                .unwrap_or(false)
        } else {
            true
        };

        Ok(Some(RawTfRecord { data, crc_ok }))
    }
}

#[cfg(feature = "tfrecord")]
impl Iterator for TfRecordRawReader {
    type Item = Result<RawTfRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_record() {
            Ok(Some(r)) => Some(Ok(r)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(not(feature = "tfrecord"))]
impl TfRecordRawReader {
    /// Stub constructor.
    pub fn open<P: AsRef<std::path::Path>>(
        _file_path: P,
        _validate_crc: bool,
    ) -> tenflowers_core::Result<Self> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "tfrecord feature not enabled".to_string(),
        ))
    }
}

#[cfg(not(feature = "tfrecord"))]
impl Iterator for TfRecordRawReader {
    type Item = tenflowers_core::Result<RawTfRecord>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

// ---------------------------------------------------------------------------
// Feature-type dispatch (proper varint / tag parsing)
// ---------------------------------------------------------------------------

/// Wire type constants for protobuf encoding.
#[cfg(feature = "tfrecord")]
mod wire_type {
    pub const VARINT: u8 = 0;
    pub const LEN: u8 = 2;
}

/// Parse a protobuf varint from `src` starting at `offset`.
/// Returns `(value, new_offset)` on success.
#[cfg(feature = "tfrecord")]
fn parse_varint(src: &[u8], mut offset: usize) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    loop {
        if offset >= src.len() {
            return Err(TensorError::invalid_argument(
                "Unexpected end of buffer while reading varint".to_string(),
            ));
        }
        let byte = src[offset];
        offset += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(TensorError::invalid_argument(
                "Varint too long (>= 64 bits)".to_string(),
            ));
        }
    }
    Ok((value, offset))
}

/// Parse a length-delimited field (returns the byte slice and new offset).
#[cfg(feature = "tfrecord")]
fn parse_len_delimited(src: &[u8], offset: usize) -> Result<(&[u8], usize)> {
    let (len, new_off) = parse_varint(src, offset)?;
    let end = new_off + len as usize;
    if end > src.len() {
        return Err(TensorError::invalid_argument(
            "Length-delimited field extends past buffer".to_string(),
        ));
    }
    Ok((&src[new_off..end], end))
}

/// Parse a little-endian 32-bit float from 4 bytes.
#[cfg(feature = "tfrecord")]
fn parse_f32_le(bytes: &[u8]) -> Result<f32> {
    if bytes.len() < 4 {
        return Err(TensorError::invalid_argument(
            "Not enough bytes for f32".to_string(),
        ));
    }
    let b = [bytes[0], bytes[1], bytes[2], bytes[3]];
    Ok(f32::from_le_bytes(b))
}

/// Parse a `Feature` from its serialised proto bytes.
///
/// A `tf.train.Feature` contains exactly one of:
///   - field 1 (BytesList): len-delimited list of byte strings
///   - field 2 (FloatList): len-delimited list of 32-bit LE floats
///   - field 3 (Int64List): len-delimited list of varints (int64)
#[cfg(feature = "tfrecord")]
pub fn parse_feature_proto(data: &[u8]) -> Result<Feature> {
    let mut offset = 0usize;
    while offset < data.len() {
        let (tag_varint, new_off) = parse_varint(data, offset)?;
        offset = new_off;
        let field_number = (tag_varint >> 3) as u8;
        let wtype = (tag_varint & 0x7) as u8;

        match (field_number, wtype) {
            // field 1: BytesList
            (1, w) if w == wire_type::LEN => {
                let (inner, _next) = parse_len_delimited(data, offset)?;
                let bytes_list = parse_bytes_list(inner)?;
                return Ok(Feature::Bytes(bytes_list));
            }
            // field 2: FloatList
            (2, w) if w == wire_type::LEN => {
                let (inner, _next) = parse_len_delimited(data, offset)?;
                let float_list = parse_float_list(inner)?;
                return Ok(Feature::Float(float_list));
            }
            // field 3: Int64List
            (3, w) if w == wire_type::LEN => {
                let (inner, _next) = parse_len_delimited(data, offset)?;
                let int_list = parse_int64_list(inner)?;
                return Ok(Feature::Int64(int_list));
            }
            // Skip unknown fields
            (_, w) if w == wire_type::VARINT => {
                let (_, next_off) = parse_varint(data, offset)?;
                offset = next_off;
            }
            (_, w) if w == wire_type::LEN => {
                let (_, next_off) = parse_len_delimited(data, offset)?;
                offset = next_off;
            }
            _ => {
                // Fixed 32-bit or 64-bit — skip
                offset += 4;
            }
        }
    }
    // Default to empty bytes if no field matched
    Ok(Feature::Bytes(Vec::new()))
}

/// Parse the inner payload of a `BytesList` proto field.
/// `BytesList` is: repeated bytes value = 1;
#[cfg(feature = "tfrecord")]
fn parse_bytes_list(data: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut result = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        let (tag_varint, new_off) = parse_varint(data, offset)?;
        offset = new_off;
        let field_number = (tag_varint >> 3) as u8;
        let wtype = (tag_varint & 0x7) as u8;
        if field_number == 1 && wtype == wire_type::LEN {
            let (bytes, next_off) = parse_len_delimited(data, offset)?;
            offset = next_off;
            result.push(bytes.to_vec());
        } else if wtype == wire_type::LEN {
            let (_, next_off) = parse_len_delimited(data, offset)?;
            offset = next_off;
        } else if wtype == wire_type::VARINT {
            let (_, next_off) = parse_varint(data, offset)?;
            offset = next_off;
        } else {
            offset += 4;
        }
    }
    Ok(result)
}

/// Parse the inner payload of a `FloatList` proto field.
/// `FloatList` is: repeated float value = 1 [packed=true];
#[cfg(feature = "tfrecord")]
fn parse_float_list(data: &[u8]) -> Result<Vec<f32>> {
    let mut result = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        let (tag_varint, new_off) = parse_varint(data, offset)?;
        offset = new_off;
        let field_number = (tag_varint >> 3) as u8;
        let wtype = (tag_varint & 0x7) as u8;
        if field_number == 1 && wtype == wire_type::LEN {
            // Packed floats
            let (packed, next_off) = parse_len_delimited(data, offset)?;
            offset = next_off;
            let mut i = 0usize;
            while i + 4 <= packed.len() {
                result.push(parse_f32_le(&packed[i..i + 4])?);
                i += 4;
            }
        } else if wtype == wire_type::LEN {
            let (_, next_off) = parse_len_delimited(data, offset)?;
            offset = next_off;
        } else if wtype == wire_type::VARINT {
            let (_, next_off) = parse_varint(data, offset)?;
            offset = next_off;
        } else {
            offset += 4;
        }
    }
    Ok(result)
}

/// Parse the inner payload of an `Int64List` proto field.
/// `Int64List` is: repeated int64 value = 1 [packed=true];
#[cfg(feature = "tfrecord")]
fn parse_int64_list(data: &[u8]) -> Result<Vec<i64>> {
    let mut result = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        let (tag_varint, new_off) = parse_varint(data, offset)?;
        offset = new_off;
        let field_number = (tag_varint >> 3) as u8;
        let wtype = (tag_varint & 0x7) as u8;
        if field_number == 1 && wtype == wire_type::LEN {
            // Packed varints
            let (packed, next_off) = parse_len_delimited(data, offset)?;
            offset = next_off;
            let mut i = 0usize;
            while i < packed.len() {
                let (v, new_i) = parse_varint(packed, i)?;
                // Reinterpret as signed i64 (ZigZag not used for int64)
                result.push(v as i64);
                i = new_i;
            }
        } else if wtype == wire_type::LEN {
            let (_, next_off) = parse_len_delimited(data, offset)?;
            offset = next_off;
        } else if wtype == wire_type::VARINT {
            let (v, next_off) = parse_varint(data, offset)?;
            offset = next_off;
            // Standalone int64 value (non-packed)
            result.push(v as i64);
        } else {
            offset += 4;
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// SequenceExample types
// ---------------------------------------------------------------------------

/// A single step in a `SequenceExample` feature list.
/// Each step contains a list of feature values (same as `tf.train.Feature`).
///
/// Embeds [`Feature`], which is only available with the `tfrecord` feature.
#[cfg(feature = "tfrecord")]
#[derive(Debug, Clone)]
pub struct SequenceStep {
    /// Features present in this step
    pub features: Vec<Feature>,
}

/// A full `SequenceExample` as read from a TFRecord file.
///
/// A `SequenceExample` carries:
/// - **context** — fixed features shared across all sequence steps
/// - **feature_lists** — per-step varying features, keyed by name
///
/// Embeds [`Feature`], which is only available with the `tfrecord` feature.
#[cfg(feature = "tfrecord")]
#[derive(Debug, Clone)]
pub struct SequenceExample {
    /// Context features (flat, like a regular `Example`)
    pub context: HashMap<String, Feature>,
    /// Per-key per-step feature lists
    pub feature_lists: HashMap<String, Vec<SequenceStep>>,
}

/// Iterator that reads `SequenceExample` records from a TFRecord file.
///
/// Because true proto parsing of `SequenceExample` requires a full proto
/// library, this implementation performs a best-effort structural parse:
/// context features use the same field-number heuristics as `parse_feature_proto`,
/// and feature lists are extracted as contiguous blocks of byte payloads.
pub struct TfRecordSequenceReader {
    #[cfg(feature = "tfrecord")]
    raw_reader: TfRecordRawReader,
    #[cfg(not(feature = "tfrecord"))]
    _phantom: (),
}

#[cfg(feature = "tfrecord")]
impl TfRecordSequenceReader {
    /// Open `file_path` for SequenceExample streaming.
    pub fn open<P: AsRef<Path>>(file_path: P, validate_crc: bool) -> Result<Self> {
        Ok(Self {
            raw_reader: TfRecordRawReader::open(file_path, validate_crc)?,
        })
    }

    /// Parse a raw record payload as a best-effort `SequenceExample`.
    ///
    /// The proto layout of `SequenceExample` is:
    /// ```text
    /// message SequenceExample {
    ///   Features context = 1;          // field 1, len-delimited
    ///   FeatureLists feature_lists = 2; // field 2, len-delimited
    /// }
    /// ```
    fn parse_sequence_example(data: &[u8]) -> Result<SequenceExample> {
        let mut context: HashMap<String, Feature> = HashMap::new();
        let mut feature_lists: HashMap<String, Vec<SequenceStep>> = HashMap::new();
        let mut offset = 0usize;

        while offset < data.len() {
            let (tag_varint, new_off) = parse_varint(data, offset)?;
            offset = new_off;
            let field_number = (tag_varint >> 3) as u8;
            let wtype = (tag_varint & 0x7) as u8;

            match (field_number, wtype) {
                // field 1: context (Features message)
                (1, w) if w == wire_type::LEN => {
                    let (inner, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                    let ctx = Self::parse_features_message(inner)?;
                    context.extend(ctx);
                }
                // field 2: feature_lists (FeatureLists message)
                (2, w) if w == wire_type::LEN => {
                    let (inner, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                    let fl = Self::parse_feature_lists_message(inner)?;
                    feature_lists.extend(fl);
                }
                // Skip unknown
                (_, w) if w == wire_type::VARINT => {
                    let (_, next_off) = parse_varint(data, offset)?;
                    offset = next_off;
                }
                (_, w) if w == wire_type::LEN => {
                    let (_, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                }
                _ => {
                    offset += 4;
                }
            }
        }

        Ok(SequenceExample {
            context,
            feature_lists,
        })
    }

    /// Parse a `Features` proto: `map<string, Feature> feature = 1;`
    fn parse_features_message(data: &[u8]) -> Result<HashMap<String, Feature>> {
        // Features proto: repeated Feature.Entry (MapEntry) at field 1
        // MapEntry: key = field 1 (string), value = field 2 (Feature)
        let mut result = HashMap::new();
        let mut offset = 0usize;

        while offset < data.len() {
            let (tag_varint, new_off) = parse_varint(data, offset)?;
            offset = new_off;
            let field_number = (tag_varint >> 3) as u8;
            let wtype = (tag_varint & 0x7) as u8;

            if field_number == 1 && wtype == wire_type::LEN {
                // MapEntry
                let (entry_bytes, next_off) = parse_len_delimited(data, offset)?;
                offset = next_off;
                if let Ok((key, feature)) = Self::parse_feature_entry(entry_bytes) {
                    result.insert(key, feature);
                }
            } else if wtype == wire_type::LEN {
                let (_, next_off) = parse_len_delimited(data, offset)?;
                offset = next_off;
            } else if wtype == wire_type::VARINT {
                let (_, next_off) = parse_varint(data, offset)?;
                offset = next_off;
            } else {
                offset += 4;
            }
        }
        Ok(result)
    }

    /// Parse a single `(key, Feature)` MapEntry.
    fn parse_feature_entry(data: &[u8]) -> Result<(String, Feature)> {
        let mut key = String::new();
        let mut feature = Feature::Bytes(Vec::new());
        let mut offset = 0usize;

        while offset < data.len() {
            let (tag_varint, new_off) = parse_varint(data, offset)?;
            offset = new_off;
            let field_number = (tag_varint >> 3) as u8;
            let wtype = (tag_varint & 0x7) as u8;

            match (field_number, wtype) {
                (1, w) if w == wire_type::LEN => {
                    // key (string)
                    let (bytes, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                    key = String::from_utf8_lossy(bytes).into_owned();
                }
                (2, w) if w == wire_type::LEN => {
                    // value (Feature message)
                    let (feat_bytes, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                    feature = parse_feature_proto(feat_bytes)?;
                }
                (_, w) if w == wire_type::VARINT => {
                    let (_, next_off) = parse_varint(data, offset)?;
                    offset = next_off;
                }
                (_, w) if w == wire_type::LEN => {
                    let (_, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                }
                _ => {
                    offset += 4;
                }
            }
        }
        Ok((key, feature))
    }

    /// Parse a `FeatureLists` message.
    ///
    /// ```text
    /// message FeatureLists {
    ///   map<string, FeatureList> feature_list = 1;
    /// }
    /// message FeatureList {
    ///   repeated Feature feature = 1;
    /// }
    /// ```
    fn parse_feature_lists_message(data: &[u8]) -> Result<HashMap<String, Vec<SequenceStep>>> {
        let mut result: HashMap<String, Vec<SequenceStep>> = HashMap::new();
        let mut offset = 0usize;

        while offset < data.len() {
            let (tag_varint, new_off) = parse_varint(data, offset)?;
            offset = new_off;
            let field_number = (tag_varint >> 3) as u8;
            let wtype = (tag_varint & 0x7) as u8;

            if field_number == 1 && wtype == wire_type::LEN {
                let (entry_bytes, next_off) = parse_len_delimited(data, offset)?;
                offset = next_off;
                if let Ok((key, steps)) = Self::parse_feature_list_entry(entry_bytes) {
                    result.insert(key, steps);
                }
            } else if wtype == wire_type::LEN {
                let (_, next_off) = parse_len_delimited(data, offset)?;
                offset = next_off;
            } else if wtype == wire_type::VARINT {
                let (_, next_off) = parse_varint(data, offset)?;
                offset = next_off;
            } else {
                offset += 4;
            }
        }
        Ok(result)
    }

    /// Parse a `(key, FeatureList)` map entry.
    fn parse_feature_list_entry(data: &[u8]) -> Result<(String, Vec<SequenceStep>)> {
        let mut key = String::new();
        let mut steps: Vec<SequenceStep> = Vec::new();
        let mut offset = 0usize;

        while offset < data.len() {
            let (tag_varint, new_off) = parse_varint(data, offset)?;
            offset = new_off;
            let field_number = (tag_varint >> 3) as u8;
            let wtype = (tag_varint & 0x7) as u8;

            match (field_number, wtype) {
                (1, w) if w == wire_type::LEN => {
                    let (bytes, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                    key = String::from_utf8_lossy(bytes).into_owned();
                }
                (2, w) if w == wire_type::LEN => {
                    // FeatureList message
                    let (fl_bytes, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                    let step_features = Self::parse_feature_list_message(fl_bytes)?;
                    steps.extend(step_features);
                }
                (_, w) if w == wire_type::VARINT => {
                    let (_, next_off) = parse_varint(data, offset)?;
                    offset = next_off;
                }
                (_, w) if w == wire_type::LEN => {
                    let (_, next_off) = parse_len_delimited(data, offset)?;
                    offset = next_off;
                }
                _ => {
                    offset += 4;
                }
            }
        }
        Ok((key, steps))
    }

    /// Parse a `FeatureList` message → `Vec<SequenceStep>`.
    fn parse_feature_list_message(data: &[u8]) -> Result<Vec<SequenceStep>> {
        let mut steps = Vec::new();
        let mut offset = 0usize;

        while offset < data.len() {
            let (tag_varint, new_off) = parse_varint(data, offset)?;
            offset = new_off;
            let field_number = (tag_varint >> 3) as u8;
            let wtype = (tag_varint & 0x7) as u8;

            if field_number == 1 && wtype == wire_type::LEN {
                // Feature message
                let (feat_bytes, next_off) = parse_len_delimited(data, offset)?;
                offset = next_off;
                let feature = parse_feature_proto(feat_bytes)?;
                steps.push(SequenceStep {
                    features: vec![feature],
                });
            } else if wtype == wire_type::LEN {
                let (_, next_off) = parse_len_delimited(data, offset)?;
                offset = next_off;
            } else if wtype == wire_type::VARINT {
                let (_, next_off) = parse_varint(data, offset)?;
                offset = next_off;
            } else {
                offset += 4;
            }
        }
        Ok(steps)
    }
}

#[cfg(feature = "tfrecord")]
impl Iterator for TfRecordSequenceReader {
    type Item = Result<SequenceExample>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.raw_reader.next()? {
            Ok(raw) => {
                if !raw.crc_ok {
                    return Some(Err(TensorError::invalid_argument(
                        "CRC validation failed for SequenceExample record".to_string(),
                    )));
                }
                Some(Self::parse_sequence_example(&raw.data))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(not(feature = "tfrecord"))]
impl TfRecordSequenceReader {
    /// Stub constructor.
    pub fn open<P: AsRef<std::path::Path>>(
        _file_path: P,
        _validate_crc: bool,
    ) -> tenflowers_core::Result<Self> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "tfrecord feature not enabled".to_string(),
        ))
    }
}

// Without the `tfrecord` feature there is no `SequenceExample` type to yield
// (it embeds `Feature`), so the stub reader does not implement `Iterator`.
// The stub `open()` above already returns an error, so the reader is never
// successfully constructed in that configuration.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Pure-logic tests (no feature gate) ---

    #[test]
    fn test_filter_feature_type_dispatch_no_feature() {
        // verify_masked_crc32 stub
        #[cfg(not(feature = "tfrecord"))]
        {
            let err = verify_masked_crc32(&[], 0).expect_err("should fail");
            assert!(err.to_string().contains("tfrecord feature not enabled"));
        }
    }

    // --- CRC tests (require tfrecord feature) ---

    #[cfg(feature = "tfrecord")]
    mod crc_tests {
        use super::*;

        #[test]
        fn test_masked_crc32_deterministic() {
            let data = b"hello world";
            let c1 = masked_crc32(data);
            let c2 = masked_crc32(data);
            assert_eq!(c1, c2);
        }

        #[test]
        fn test_masked_crc32_differs_from_plain() {
            let data = b"test";
            let masked = masked_crc32(data);
            let mut h = crc32fast::Hasher::new();
            h.update(data);
            let plain = h.finalize();
            assert_ne!(masked, plain, "masked CRC should differ from plain CRC");
        }

        #[test]
        fn test_verify_masked_crc32_ok() {
            let data = b"some data";
            let crc = masked_crc32(data);
            verify_masked_crc32(data, crc).expect("should be Ok");
        }

        #[test]
        fn test_verify_masked_crc32_bad() {
            let data = b"some data";
            let bad_crc = 0xDEAD_BEEFu32;
            let err = verify_masked_crc32(data, bad_crc).expect_err("should fail");
            assert!(err.to_string().contains("CRC mismatch"));
        }

        #[test]
        fn test_masked_crc32_empty_data() {
            let crc = masked_crc32(&[]);
            // Should not panic; result is the masked CRC of the empty string
            assert_eq!(crc, masked_crc32(&[]));
        }
    }

    // --- Proto parser tests (require tfrecord feature) ---

    #[cfg(feature = "tfrecord")]
    mod proto_tests {
        use super::*;

        #[test]
        fn test_parse_varint_single_byte() {
            let data = [0x05u8];
            let (v, off) = parse_varint(&data, 0).expect("varint");
            assert_eq!(v, 5);
            assert_eq!(off, 1);
        }

        #[test]
        fn test_parse_varint_multi_byte() {
            // 300 encoded as varint: 0xAC 0x02
            let data = [0xACu8, 0x02];
            let (v, off) = parse_varint(&data, 0).expect("varint");
            assert_eq!(v, 300);
            assert_eq!(off, 2);
        }

        #[test]
        fn test_parse_varint_empty_returns_error() {
            let err = parse_varint(&[], 0).expect_err("should fail");
            assert!(err.to_string().contains("Unexpected end of buffer"));
        }

        #[test]
        fn test_parse_feature_proto_empty_data() {
            // Empty payload → default Feature::Bytes([])
            let f = parse_feature_proto(&[]).expect("should not error on empty");
            match f {
                Feature::Bytes(v) => assert!(v.is_empty()),
                _ => panic!("expected Bytes"),
            }
        }

        #[test]
        fn test_parse_float_list_packed() {
            // Construct a FloatList proto manually:
            //   field 2, len-delimited → inner
            //   inner: field 1, packed len → 4 bytes of f32
            let value: f32 = 1.5;
            let bytes = value.to_le_bytes();
            // field 1, packed (tag = field 1 << 3 | 2 = 0x0A)
            // length = 4
            let inner: Vec<u8> = vec![0x0A, 0x04, bytes[0], bytes[1], bytes[2], bytes[3]];
            // field 2 of Feature (FloatList), len-delimited (tag = 2<<3|2 = 0x12)
            let mut outer: Vec<u8> = vec![0x12, inner.len() as u8];
            outer.extend_from_slice(&inner);

            let f = parse_feature_proto(&outer).expect("parse");
            match f {
                Feature::Float(vals) => {
                    assert_eq!(vals.len(), 1);
                    assert!((vals[0] - 1.5).abs() < 1e-5);
                }
                _ => panic!("expected Float feature"),
            }
        }

        #[test]
        fn test_parse_bytes_list_single_entry() {
            let payload = b"hello";
            // field 1, len-delimited for the value (tag = 0x0A), length=5
            let mut inner: Vec<u8> = vec![0x0A, 0x05];
            inner.extend_from_slice(payload);
            // field 1 of Feature (BytesList), len-delimited (tag = 1<<3|2 = 0x0A)
            let mut outer: Vec<u8> = vec![0x0A, inner.len() as u8];
            outer.extend_from_slice(&inner);

            let f = parse_feature_proto(&outer).expect("parse");
            match f {
                Feature::Bytes(vals) => {
                    assert_eq!(vals.len(), 1);
                    assert_eq!(vals[0], b"hello");
                }
                _ => panic!("expected Bytes feature"),
            }
        }
    }

    // --- Raw reader stub tests ---
    #[cfg(not(feature = "tfrecord"))]
    mod stub_tests {
        use super::*;

        #[test]
        fn test_raw_reader_stub() {
            // The stub reader type is not `Debug` (its feature-enabled variant
            // wraps a `Box<dyn Read>`), so match on the result instead of
            // using `expect_err`, which would require `Ok: Debug`.
            match TfRecordRawReader::open("/tmp/x.tfrecord", false) {
                Ok(_) => panic!("stub should return an error"),
                Err(err) => {
                    assert!(err.to_string().contains("tfrecord feature not enabled"))
                }
            }
        }

        #[test]
        fn test_sequence_reader_stub() {
            match TfRecordSequenceReader::open("/tmp/x.tfrecord", false) {
                Ok(_) => panic!("stub should return an error"),
                Err(err) => {
                    assert!(err.to_string().contains("tfrecord feature not enabled"))
                }
            }
        }
    }

    // --- Raw reader + sequence reader integration tests (tfrecord enabled) ---
    #[cfg(feature = "tfrecord")]
    mod integration_tests {
        use super::*;

        /// Write a minimal TFRecord file (TF wire format) with two records.
        ///
        /// Payloads are `b"record_one"` and `b"record_two"` (plain ASCII, not proto).
        fn write_test_tfrecord() -> (tempfile::NamedTempFile, std::path::PathBuf) {
            use std::io::Write;

            let tmp = tempfile::NamedTempFile::new().expect("tmp file");
            let path = tmp.path().to_path_buf();
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("open");

            for payload in &[b"record_one".as_ref(), b"record_two".as_ref()] {
                let length = payload.len() as u64;
                let len_bytes = length.to_le_bytes();
                let len_crc = masked_crc32(&len_bytes);
                let data_crc = masked_crc32(payload);

                file.write_all(&len_bytes).expect("write len");
                file.write_all(&len_crc.to_le_bytes())
                    .expect("write len crc");
                file.write_all(payload).expect("write data");
                file.write_all(&data_crc.to_le_bytes())
                    .expect("write data crc");
            }
            (tmp, path)
        }

        /// Write a TFRecord file where payloads are valid empty proto messages
        /// (zero-length buffers), so `parse_sequence_example` returns empty maps.
        fn write_empty_proto_tfrecord() -> (tempfile::NamedTempFile, std::path::PathBuf) {
            use std::io::Write;

            let tmp = tempfile::NamedTempFile::new().expect("tmp file");
            let path = tmp.path().to_path_buf();
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("open");

            // Two records with zero-length proto payloads (empty SequenceExample)
            for _ in 0..2 {
                let payload: &[u8] = &[];
                let length = 0u64;
                let len_bytes = length.to_le_bytes();
                let len_crc = masked_crc32(&len_bytes);
                let data_crc = masked_crc32(payload);

                file.write_all(&len_bytes).expect("write len");
                file.write_all(&len_crc.to_le_bytes())
                    .expect("write len crc");
                file.write_all(data_crc.to_le_bytes().as_ref())
                    .expect("write data crc");
            }
            (tmp, path)
        }

        #[test]
        fn test_raw_reader_reads_two_records() {
            let (_tmp, path) = write_test_tfrecord();
            let reader = TfRecordRawReader::open(&path, true).expect("open");
            let records: Vec<_> = reader.collect();
            assert_eq!(records.len(), 2);
            for r in &records {
                assert!(r.as_ref().expect("record ok").crc_ok);
            }
            assert_eq!(records[0].as_ref().expect("r0").data, b"record_one");
            assert_eq!(records[1].as_ref().expect("r1").data, b"record_two");
        }

        #[test]
        fn test_raw_reader_without_crc_validation() {
            let (_tmp, path) = write_test_tfrecord();
            let reader = TfRecordRawReader::open(&path, false).expect("open");
            let records: Vec<_> = reader.collect();
            assert_eq!(records.len(), 2);
        }

        #[test]
        fn test_sequence_reader_empty_proto() {
            // SequenceExample records with zero-length proto payloads should parse
            // successfully and yield SequenceExamples with empty maps.
            let (_tmp, path) = write_empty_proto_tfrecord();
            let reader = TfRecordSequenceReader::open(&path, true).expect("open");
            let examples: Vec<_> = reader.collect();
            assert_eq!(examples.len(), 2);
            for ex in examples {
                let seq = ex.expect("should parse empty proto");
                // An empty proto payload → both maps are empty
                assert!(seq.context.is_empty());
                assert!(seq.feature_lists.is_empty());
            }
        }

        #[test]
        fn test_sequence_reader_non_proto_payload_returns_error_or_empty() {
            // Records with plain ASCII bytes (non-proto) should either produce
            // a parse error or yield an empty SequenceExample — both are acceptable
            // behaviors given the best-effort parser.
            let (_tmp, path) = write_test_tfrecord();
            let reader = TfRecordSequenceReader::open(&path, true).expect("open");
            let examples: Vec<_> = reader.collect();
            assert_eq!(examples.len(), 2);
            for ex in examples {
                match ex {
                    Ok(seq) => {
                        // Parser managed to extract empty maps from non-proto data
                        let _ = seq.context.len();
                        let _ = seq.feature_lists.len();
                    }
                    Err(e) => {
                        // Parse error is also acceptable for non-proto input
                        assert!(
                            e.to_string().contains("buffer")
                                || e.to_string().contains("CRC")
                                || e.to_string().contains("varint"),
                            "unexpected error: {e}"
                        );
                    }
                }
            }
        }
    }
}
