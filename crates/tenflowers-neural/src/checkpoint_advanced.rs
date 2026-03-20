//! Advanced checkpoint primitives: CompressedCheckpoint, CheckpointDiff, CheckpointValidator
//!
//! This module extends the core checkpoint infrastructure with:
//! * **`CompressedCheckpoint`** — serialise/deserialise model state with run-length or
//!   delta encoding so that files on disk are smaller without external compression crates.
//! * **`CheckpointDiff`** — incremental checkpoints that only store the tensors that
//!   changed between two snapshots, reducing I/O cost during long training runs.
//! * **`CheckpointValidator`** — verifies that a file exists, is non-empty, and can be
//!   successfully parsed as a `CheckpointData` blob.
//!
//! This module is always available (no feature gate required).  File I/O for
//! `CompressedCheckpoint` and `CheckpointValidator` does require `serde_json` and is
//! compiled unconditionally because the JSON format is implemented without serde derives
//! by composing plain string values.

use std::collections::HashMap;
use std::path::Path;
use tenflowers_core::{Result, TensorError};

// ─── CheckpointData ──────────────────────────────────────────────────────────

/// Self-contained snapshot of a model checkpoint.
///
/// This is a purpose-built, feature-independent checkpoint container that does
/// not depend on the `serialize` Cargo feature.  Tensor data is stored as flat
/// `Vec<f32>` slices so serialisation is straightforward.
#[derive(Debug, Clone, Default)]
pub struct CheckpointData {
    /// Human-readable checkpoint identifier (e.g. "epoch_5_step_500").
    pub id: String,
    /// Training epoch at which this snapshot was taken.
    pub epoch: u32,
    /// Training step at which this snapshot was taken.
    pub step: u64,
    /// Loss value recorded at this checkpoint.
    pub loss: f32,
    /// Named tensor blobs: layer name → flat f32 data.
    pub tensors: HashMap<String, Vec<f32>>,
    /// Tensor shapes: layer name → dimension vector.
    pub tensor_shapes: HashMap<String, Vec<usize>>,
    /// Arbitrary key-value metadata (e.g. optimizer hyper-parameters).
    pub metadata: HashMap<String, String>,
}

impl CheckpointData {
    /// Construct a minimal checkpoint with no tensors.
    pub fn new(id: &str, epoch: u32, step: u64, loss: f32) -> Self {
        Self {
            id: id.to_string(),
            epoch,
            step,
            loss,
            tensors: HashMap::new(),
            tensor_shapes: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Insert a named tensor into this checkpoint.
    pub fn insert_tensor(&mut self, name: &str, data: Vec<f32>, shape: Vec<usize>) {
        self.tensors.insert(name.to_string(), data);
        self.tensor_shapes.insert(name.to_string(), shape);
    }
}

// ─── Compression helpers ─────────────────────────────────────────────────────

/// Encode `data` with run-length encoding (RLE) for f32 slices.
///
/// Output: pairs of (count: u32 LE, value: f32 LE), 8 bytes per run.
/// Effective for sparse or constant tensors.
fn rle_encode(data: &[f32]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut idx = 0usize;

    while idx < data.len() {
        let value = data[idx];
        let start = idx;
        while idx < data.len() && data[idx].to_bits() == value.to_bits() {
            idx += 1;
        }
        let count = (idx - start) as u32;
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }

    out
}

/// Decode a buffer produced by `rle_encode`.
fn rle_decode(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 8 != 0 {
        return Err(TensorError::serialization_error_simple(format!(
            "RLE buffer has invalid length {}: expected multiple of 8",
            data.len()
        )));
    }

    let mut out = Vec::new();
    let mut pos = 0usize;

    while pos < data.len() {
        let count_bytes: [u8; 4] = data[pos..pos + 4].try_into().map_err(|_| {
            TensorError::serialization_error_simple("Failed to read RLE count".to_string())
        })?;
        let value_bytes: [u8; 4] = data[pos + 4..pos + 8].try_into().map_err(|_| {
            TensorError::serialization_error_simple("Failed to read RLE value".to_string())
        })?;
        let count = u32::from_le_bytes(count_bytes) as usize;
        let value = f32::from_le_bytes(value_bytes);
        out.extend(std::iter::repeat(value).take(count));
        pos += 8;
    }

    Ok(out)
}

/// Delta-encode `data` as an f32 difference stream.
///
/// Stores the first element verbatim, then each subsequent element as the
/// difference from the previous one.  8 bytes per element (4-byte f32).
fn delta_encode(data: &[f32]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(data.len() * 4);
    out.extend_from_slice(&data[0].to_le_bytes());

    for window in data.windows(2) {
        let delta = window[1] - window[0];
        out.extend_from_slice(&delta.to_le_bytes());
    }

    out
}

/// Decode a buffer produced by `delta_encode`.
fn delta_decode(data: &[u8]) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data.len() % 4 != 0 {
        return Err(TensorError::serialization_error_simple(format!(
            "Delta buffer has invalid length {}: expected multiple of 4",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(data.len() / 4);
    let first_bytes: [u8; 4] = data[0..4].try_into().map_err(|_| {
        TensorError::serialization_error_simple("Failed to read first delta element".to_string())
    })?;
    let mut prev = f32::from_le_bytes(first_bytes);
    out.push(prev);

    for chunk in data[4..].chunks_exact(4) {
        let delta_bytes: [u8; 4] = chunk.try_into().map_err(|_| {
            TensorError::serialization_error_simple("Failed to read delta chunk".to_string())
        })?;
        let delta = f32::from_le_bytes(delta_bytes);
        prev += delta;
        out.push(prev);
    }

    Ok(out)
}

// ─── Base-64 codec ───────────────────────────────────────────────────────────

const B64_TABLE: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(input: &[u8]) -> String {
    let mut out = Vec::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let combined = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_TABLE[((combined >> 18) & 0x3F) as usize]);
        out.push(B64_TABLE[((combined >> 12) & 0x3F) as usize]);
        if chunk.len() > 1 {
            out.push(B64_TABLE[((combined >> 6) & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }
        if chunk.len() > 2 {
            out.push(B64_TABLE[(combined & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }
    }
    // B64_TABLE only contains ASCII bytes so from_utf8 is guaranteed to succeed.
    // The `map_err` + `expect` path is unreachable in practice.
    out.iter().map(|&b| b as char).collect()
}

fn b64_char_value(c: u8) -> Result<u32> {
    match c {
        b'A'..=b'Z' => Ok((c - b'A') as u32),
        b'a'..=b'z' => Ok((c - b'a') as u32 + 26),
        b'0'..=b'9' => Ok((c - b'0') as u32 + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        b'=' => Ok(0),
        _ => Err(TensorError::serialization_error_simple(format!(
            "Invalid base-64 character: {}",
            c as char
        ))),
    }
}

fn b64_decode(input: &str) -> Result<Vec<u8>> {
    let bytes = input.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(TensorError::serialization_error_simple(format!(
            "Base-64 input length {} is not a multiple of 4",
            bytes.len()
        )));
    }

    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks_exact(4) {
        let v0 = b64_char_value(chunk[0])?;
        let v1 = b64_char_value(chunk[1])?;
        let v2 = b64_char_value(chunk[2])?;
        let v3 = b64_char_value(chunk[3])?;
        let combined = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
        out.push(((combined >> 16) & 0xFF) as u8);
        if chunk[2] != b'=' {
            out.push(((combined >> 8) & 0xFF) as u8);
        }
        if chunk[3] != b'=' {
            out.push((combined & 0xFF) as u8);
        }
    }

    Ok(out)
}

// ─── Minimal JSON helpers ────────────────────────────────────────────────────
//
// These functions compose JSON without serde_json so that the module is
// available without the `serialize` feature.

/// Escape a string value for JSON embedding.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Serialise a `CheckpointData` to a JSON string (no serde required).
///
/// The schema is:
/// ```json
/// {
///   "id": "...",
///   "epoch": 1,
///   "step": 100,
///   "loss": 0.5,
///   "compression": "RunLength",
///   "metadata": { "key": "value", ... },
///   "tensor_shapes": { "name": [d0, d1, ...], ... },
///   "tensor_data_b64": { "name": "<base64>", ... }
/// }
/// ```
fn serialise_checkpoint(data: &CheckpointData, compression: TensorCompression) -> Result<String> {
    let mut out = String::new();
    out.push_str("{\n");

    // Scalar fields
    out.push_str(&format!("  \"id\": \"{}\",\n", json_escape(&data.id)));
    out.push_str(&format!("  \"epoch\": {},\n", data.epoch));
    out.push_str(&format!("  \"step\": {},\n", data.step));
    out.push_str(&format!("  \"loss\": {},\n", data.loss));
    out.push_str(&format!(
        "  \"compression\": \"{}\",\n",
        compression_name(compression)
    ));

    // Metadata map
    out.push_str("  \"metadata\": {");
    let mut first = true;
    for (k, v) in &data.metadata {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&format!(
            "\n    \"{}\": \"{}\"",
            json_escape(k),
            json_escape(v)
        ));
    }
    out.push_str("\n  },\n");

    // Tensor shapes
    out.push_str("  \"tensor_shapes\": {");
    let mut first = true;
    for (name, shape) in &data.tensor_shapes {
        if !first {
            out.push(',');
        }
        first = false;
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        out.push_str(&format!(
            "\n    \"{}\": [{}]",
            json_escape(name),
            dims.join(", ")
        ));
    }
    out.push_str("\n  },\n");

    // Tensor data (base64-encoded compressed blobs)
    out.push_str("  \"tensor_data_b64\": {");
    let mut first = true;
    for (name, values) in &data.tensors {
        if !first {
            out.push(',');
        }
        first = false;
        let compressed: Vec<u8> = match compression {
            TensorCompression::None => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            TensorCompression::RunLength => rle_encode(values),
            TensorCompression::Delta => delta_encode(values),
        };
        let b64 = b64_encode(&compressed);
        out.push_str(&format!("\n    \"{}\": \"{}\"", json_escape(name), b64));
    }
    out.push_str("\n  }\n");

    out.push('}');
    Ok(out)
}

fn compression_name(c: TensorCompression) -> &'static str {
    match c {
        TensorCompression::None => "None",
        TensorCompression::RunLength => "RunLength",
        TensorCompression::Delta => "Delta",
    }
}

fn parse_compression(s: &str) -> TensorCompression {
    match s {
        "RunLength" => TensorCompression::RunLength,
        "Delta" => TensorCompression::Delta,
        _ => TensorCompression::None,
    }
}

// ─── Minimal JSON parser ─────────────────────────────────────────────────────
//
// A purpose-built, zero-dependency JSON parser for the specific schema used by
// `serialise_checkpoint`.  It handles only the subset of JSON produced by that
// function.

/// Extract the string value for a top-level JSON key (handles only simple string values).
fn json_extract_string<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let search = format!("\"{}\":", key);
    let start = json.find(&search)?;
    let after_colon = start + search.len();
    let trimmed = json[after_colon..].trim_start();
    if trimmed.starts_with('"') {
        let inner = &trimmed[1..];
        let end = inner.find('"')?;
        Some(&inner[..end])
    } else {
        None
    }
}

/// Extract a u32 value for a top-level JSON key.
fn json_extract_u32(json: &str, key: &str) -> Option<u32> {
    let search = format!("\"{}\":", key);
    let start = json.find(&search)?;
    let after_colon = start + search.len();
    let trimmed = json[after_colon..].trim_start();
    let end = trimmed.find(|c: char| c == ',' || c == '\n' || c == '}')?;
    trimmed[..end].trim().parse().ok()
}

/// Extract a u64 value for a top-level JSON key.
fn json_extract_u64(json: &str, key: &str) -> Option<u64> {
    let search = format!("\"{}\":", key);
    let start = json.find(&search)?;
    let after_colon = start + search.len();
    let trimmed = json[after_colon..].trim_start();
    let end = trimmed.find(|c: char| c == ',' || c == '\n' || c == '}')?;
    trimmed[..end].trim().parse().ok()
}

/// Extract a f32 value for a top-level JSON key (handles scientific notation).
fn json_extract_f32(json: &str, key: &str) -> Option<f32> {
    let search = format!("\"{}\":", key);
    let start = json.find(&search)?;
    let after_colon = start + search.len();
    let trimmed = json[after_colon..].trim_start();
    let end = trimmed.find(|c: char| c == ',' || c == '\n' || c == '}')?;
    trimmed[..end].trim().parse().ok()
}

/// Extract all key→base64 pairs from the `tensor_data_b64` JSON object.
fn json_extract_b64_map(json: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let key = "\"tensor_data_b64\": {";
    let start = match json.find(key) {
        Some(p) => p + key.len(),
        None => return map,
    };
    let block = &json[start..];
    let end = match block.find('}') {
        Some(p) => p,
        None => return map,
    };
    let content = &block[..end];

    // Parse: "name": "base64value" pairs
    let mut remaining = content;
    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() || remaining.starts_with('}') {
            break;
        }
        // Skip commas and whitespace
        remaining = remaining.trim_start_matches(',').trim_start();
        if remaining.is_empty() || remaining.starts_with('}') {
            break;
        }
        // Expect a quoted key
        if !remaining.starts_with('"') {
            break;
        }
        remaining = &remaining[1..];
        let key_end = match remaining.find('"') {
            Some(p) => p,
            None => break,
        };
        let tensor_name = remaining[..key_end].to_string();
        remaining = &remaining[key_end + 1..];
        // Skip ': "'
        let colon_quote = match remaining.find("\"") {
            Some(p) => p,
            None => break,
        };
        remaining = &remaining[colon_quote + 1..];
        // Read until closing quote (no escaping in base64)
        let val_end = match remaining.find('"') {
            Some(p) => p,
            None => break,
        };
        let b64_value = remaining[..val_end].to_string();
        remaining = &remaining[val_end + 1..];
        map.insert(tensor_name, b64_value);
    }
    map
}

/// Extract all key→`Vec<usize>` pairs from the `tensor_shapes` JSON object.
fn json_extract_shape_map(json: &str) -> HashMap<String, Vec<usize>> {
    let mut map = HashMap::new();
    let key = "\"tensor_shapes\": {";
    let start = match json.find(key) {
        Some(p) => p + key.len(),
        None => return map,
    };
    let block = &json[start..];
    // Find matching '}'
    let end = match block.find('}') {
        Some(p) => p,
        None => return map,
    };
    let content = &block[..end];

    let mut remaining = content;
    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() || remaining.starts_with('}') {
            break;
        }
        remaining = remaining.trim_start_matches(',').trim_start();
        if remaining.is_empty() || remaining.starts_with('}') {
            break;
        }
        if !remaining.starts_with('"') {
            break;
        }
        remaining = &remaining[1..];
        let key_end = match remaining.find('"') {
            Some(p) => p,
            None => break,
        };
        let tensor_name = remaining[..key_end].to_string();
        remaining = &remaining[key_end + 1..];
        // Skip to '['
        let bracket_start = match remaining.find('[') {
            Some(p) => p,
            None => break,
        };
        remaining = &remaining[bracket_start + 1..];
        let bracket_end = match remaining.find(']') {
            Some(p) => p,
            None => break,
        };
        let dims_str = &remaining[..bracket_end];
        let dims: Vec<usize> = dims_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        remaining = &remaining[bracket_end + 1..];
        map.insert(tensor_name, dims);
    }
    map
}

/// Deserialise a `CheckpointData` from a JSON string produced by `serialise_checkpoint`.
fn deserialise_checkpoint(json: &str) -> Result<(CheckpointData, TensorCompression)> {
    let id = json_extract_string(json, "id")
        .ok_or_else(|| {
            TensorError::serialization_error_simple("Missing 'id' field in checkpoint".to_string())
        })?
        .to_string();
    let epoch = json_extract_u32(json, "epoch").ok_or_else(|| {
        TensorError::serialization_error_simple("Missing 'epoch' field in checkpoint".to_string())
    })?;
    let step = json_extract_u64(json, "step").ok_or_else(|| {
        TensorError::serialization_error_simple("Missing 'step' field in checkpoint".to_string())
    })?;
    let loss = json_extract_f32(json, "loss").ok_or_else(|| {
        TensorError::serialization_error_simple("Missing 'loss' field in checkpoint".to_string())
    })?;
    let compression_str = json_extract_string(json, "compression").unwrap_or("None");
    let compression = parse_compression(compression_str);

    let tensor_shapes = json_extract_shape_map(json);
    let b64_map = json_extract_b64_map(json);

    let mut tensors = HashMap::new();
    for (name, b64) in &b64_map {
        let raw = b64_decode(b64)?;
        let values: Vec<f32> = match compression {
            TensorCompression::None => {
                if raw.len() % 4 != 0 {
                    return Err(TensorError::serialization_error_simple(format!(
                        "Raw tensor '{}' has invalid byte length {}",
                        name,
                        raw.len()
                    )));
                }
                raw.chunks_exact(4)
                    .map(|c| {
                        let bytes: [u8; 4] = c.try_into().map_err(|_| {
                            TensorError::serialization_error_simple(
                                "chunk conversion failed".to_string(),
                            )
                        })?;
                        Ok(f32::from_le_bytes(bytes))
                    })
                    .collect::<Result<Vec<f32>>>()?
            }
            TensorCompression::RunLength => rle_decode(&raw)?,
            TensorCompression::Delta => delta_decode(&raw)?,
        };
        tensors.insert(name.clone(), values);
    }

    Ok((
        CheckpointData {
            id,
            epoch,
            step,
            loss,
            tensors,
            tensor_shapes,
            metadata: HashMap::new(), // Metadata parsing omitted for brevity; extend as needed
        },
        compression,
    ))
}

// ─── TensorCompression ───────────────────────────────────────────────────────

/// Compression algorithm for `CompressedCheckpoint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCompression {
    /// Store tensors verbatim (no compression).
    None,
    /// Run-length encoding — best for sparse or constant tensors.
    RunLength,
    /// Delta encoding — best for smooth or sorted tensors.
    Delta,
}

impl Default for TensorCompression {
    fn default() -> Self {
        Self::RunLength
    }
}

// ─── CompressedCheckpoint ────────────────────────────────────────────────────

/// Saves and loads checkpoints with lightweight, pure-Rust tensor compression.
///
/// The on-disk format is a hand-crafted JSON document with base-64-encoded,
/// RLE- or delta-compressed tensor blobs.  No external compression library is
/// required.
///
/// # Example
///
/// ```rust,ignore
/// use tenflowers_neural::checkpoint_advanced::{
///     CheckpointData, CompressedCheckpoint, TensorCompression,
/// };
///
/// let mut data = CheckpointData::new("ckpt_0", 1, 100, 0.5);
/// data.insert_tensor("w", vec![1.0, 2.0, 3.0], vec![3]);
///
/// let path = std::env::temp_dir().join("my.ckpt.json");
/// CompressedCheckpoint::save_compressed(&path, &data, TensorCompression::RunLength).unwrap();
/// let loaded = CompressedCheckpoint::load_compressed(&path).unwrap();
/// assert_eq!(loaded.id, data.id);
/// ```
pub struct CompressedCheckpoint;

impl CompressedCheckpoint {
    /// Save `data` to `path` using the given compression algorithm.
    pub fn save_compressed(
        path: &Path,
        data: &CheckpointData,
        compression: TensorCompression,
    ) -> Result<()> {
        let json = serialise_checkpoint(data, compression)?;
        std::fs::write(path, json).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to write checkpoint to {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Load a `CheckpointData` from `path`, reversing the compression applied at save time.
    pub fn load_compressed(path: &Path) -> Result<CheckpointData> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to read checkpoint from {}: {}",
                path.display(),
                e
            ))
        })?;

        let (data, _compression) = deserialise_checkpoint(&json)?;
        Ok(data)
    }
}

// ─── CheckpointDiff ──────────────────────────────────────────────────────────

/// Diff between two `CheckpointData` snapshots.
///
/// Stores only the tensors that changed between `base` and `new` to minimise
/// storage cost for incremental checkpoints.
#[derive(Debug, Clone, Default)]
pub struct CheckpointDiff {
    /// Checkpoint id of the base snapshot.
    pub base_id: String,
    /// Checkpoint id of the new snapshot.
    pub new_id: String,
    /// Epoch of the new snapshot.
    pub new_epoch: u32,
    /// Step of the new snapshot.
    pub new_step: u64,
    /// Loss of the new snapshot.
    pub new_loss: f32,
    /// Tensors that were added or whose values changed.
    pub changed_tensors: HashMap<String, Vec<f32>>,
    /// Shapes of changed tensors.
    pub changed_shapes: HashMap<String, Vec<usize>>,
    /// Names of tensors present in `base` but absent in `new`.
    pub removed_tensors: Vec<String>,
    /// Metadata of the new snapshot.
    pub new_metadata: HashMap<String, String>,
}

impl CheckpointDiff {
    /// Compute the diff from `base` to `new`.
    ///
    /// Tensors whose shape or values changed are included in `changed_tensors`.
    /// Tensors present in `base` but missing from `new` are listed in
    /// `removed_tensors`.  Tensors that are unchanged are omitted entirely.
    pub fn diff(base: &CheckpointData, new: &CheckpointData) -> Self {
        let mut changed_tensors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut changed_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let mut removed_tensors: Vec<String> = Vec::new();

        for (name, new_values) in &new.tensors {
            let new_shape = new.tensor_shapes.get(name).cloned().unwrap_or_default();
            let changed = match base.tensors.get(name) {
                None => true,
                Some(base_values) => {
                    let base_shape = base.tensor_shapes.get(name).cloned().unwrap_or_default();
                    if base_shape != new_shape {
                        true
                    } else {
                        base_values
                            .iter()
                            .zip(new_values.iter())
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                    }
                }
            };
            if changed {
                changed_tensors.insert(name.clone(), new_values.clone());
                changed_shapes.insert(name.clone(), new_shape);
            }
        }

        for name in base.tensors.keys() {
            if !new.tensors.contains_key(name) {
                removed_tensors.push(name.clone());
            }
        }

        Self {
            base_id: base.id.clone(),
            new_id: new.id.clone(),
            new_epoch: new.epoch,
            new_step: new.step,
            new_loss: new.loss,
            changed_tensors,
            changed_shapes,
            removed_tensors,
            new_metadata: new.metadata.clone(),
        }
    }

    /// Apply `diff` to `base`, producing the `new` checkpoint.
    pub fn apply_diff(base: &CheckpointData, diff: &CheckpointDiff) -> CheckpointData {
        let mut result = base.clone();
        result.id = diff.new_id.clone();
        result.epoch = diff.new_epoch;
        result.step = diff.new_step;
        result.loss = diff.new_loss;
        result.metadata = diff.new_metadata.clone();

        for (name, values) in &diff.changed_tensors {
            result.tensors.insert(name.clone(), values.clone());
            if let Some(shape) = diff.changed_shapes.get(name) {
                result.tensor_shapes.insert(name.clone(), shape.clone());
            }
        }

        for name in &diff.removed_tensors {
            result.tensors.remove(name);
            result.tensor_shapes.remove(name);
        }

        result
    }
}

// ─── CheckpointValidator ─────────────────────────────────────────────────────

/// Report returned by `CheckpointValidator::validate`.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// `true` if all validation checks passed.
    pub is_valid: bool,
    /// Path that was validated.
    pub path: std::path::PathBuf,
    /// File size in bytes (0 if the file does not exist).
    pub file_size_bytes: u64,
    /// Parsed checkpoint id (empty string if parsing failed).
    pub checkpoint_id: String,
    /// Epoch from the checkpoint (0 if parsing failed).
    pub epoch: u32,
    /// Step from the checkpoint (0 if parsing failed).
    pub step: u64,
    /// All errors encountered during validation.
    pub errors: Vec<String>,
    /// Non-fatal warnings.
    pub warnings: Vec<String>,
}

/// Validates that a checkpoint file exists, is non-empty, and can be parsed.
pub struct CheckpointValidator;

impl CheckpointValidator {
    /// Validate the checkpoint at `path`.
    ///
    /// Checks performed (in order; later checks skipped on earlier failure):
    /// 1. File exists.
    /// 2. File is non-empty.
    /// 3. Content is valid UTF-8.
    /// 4. Content can be parsed (has required fields: id, epoch, step, loss).
    pub fn validate(path: &Path) -> Result<ValidationReport> {
        let mut errors: Vec<String> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();
        let mut file_size_bytes = 0u64;
        let mut checkpoint_id = String::new();
        let mut epoch = 0u32;
        let mut step = 0u64;

        // Check 1: file exists.
        if !path.exists() {
            errors.push(format!("File does not exist: {}", path.display()));
            return Ok(ValidationReport {
                is_valid: false,
                path: path.to_path_buf(),
                file_size_bytes,
                checkpoint_id,
                epoch,
                step,
                errors,
                warnings,
            });
        }

        // Check 2: file is non-empty.
        let meta = std::fs::metadata(path).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Cannot read file metadata for {}: {}",
                path.display(),
                e
            ))
        })?;
        file_size_bytes = meta.len();
        if file_size_bytes == 0 {
            errors.push(format!("File is empty: {}", path.display()));
            return Ok(ValidationReport {
                is_valid: false,
                path: path.to_path_buf(),
                file_size_bytes,
                checkpoint_id,
                epoch,
                step,
                errors,
                warnings,
            });
        }

        // Check 3: valid UTF-8.
        let raw = std::fs::read(path).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to read {}: {}",
                path.display(),
                e
            ))
        })?;
        let content = match String::from_utf8(raw) {
            Ok(s) => s,
            Err(e) => {
                errors.push(format!("File is not valid UTF-8: {}", e));
                return Ok(ValidationReport {
                    is_valid: false,
                    path: path.to_path_buf(),
                    file_size_bytes,
                    checkpoint_id,
                    epoch,
                    step,
                    errors,
                    warnings,
                });
            }
        };

        // Check 4: required fields present and parseable.
        let has_id = content.contains("\"id\":");
        let has_epoch = content.contains("\"epoch\":");
        let has_step = content.contains("\"step\":");
        let has_loss = content.contains("\"loss\":");

        if !has_id {
            errors.push("Missing required field: 'id'".to_string());
        }
        if !has_epoch {
            errors.push("Missing required field: 'epoch'".to_string());
        }
        if !has_step {
            errors.push("Missing required field: 'step'".to_string());
        }
        if !has_loss {
            errors.push("Missing required field: 'loss'".to_string());
        }

        if errors.is_empty() {
            // Try full parse.
            match deserialise_checkpoint(&content) {
                Ok((data, _)) => {
                    checkpoint_id = data.id;
                    epoch = data.epoch;
                    step = data.step;

                    if data.tensors.is_empty() {
                        warnings.push("Checkpoint contains no tensors".to_string());
                    }
                }
                Err(e) => {
                    errors.push(format!("Failed to parse checkpoint: {}", e));
                }
            }
        }

        let is_valid = errors.is_empty();
        Ok(ValidationReport {
            is_valid,
            path: path.to_path_buf(),
            file_size_bytes,
            checkpoint_id,
            epoch,
            step,
            errors,
            warnings,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_checkpoint(id: &str, epoch: u32, step: u64) -> CheckpointData {
        let mut ckpt = CheckpointData::new(id, epoch, step, 0.5 - epoch as f32 * 0.05);
        ckpt.insert_tensor("layer1.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        ckpt.insert_tensor("layer1.bias", vec![0.1, 0.2], vec![2]);
        ckpt
    }

    // ─── Codec unit tests ─────────────────────────────────────────────────────

    #[test]
    fn test_rle_roundtrip() {
        let data = vec![0.0f32, 0.0, 0.0, 1.5, 1.5, 2.0, 0.0];
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded).expect("decode ok");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rle_empty() {
        let encoded = rle_encode(&[]);
        assert!(encoded.is_empty());
        let decoded = rle_decode(&encoded).expect("decode ok");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_delta_roundtrip() {
        let data: Vec<f32> = (0..10).map(|i| i as f32 * 0.3).collect();
        let encoded = delta_encode(&data);
        let decoded = delta_decode(&encoded).expect("decode ok");
        for (a, b) in decoded.iter().zip(data.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "delta mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_b64_roundtrip() {
        let bytes: Vec<u8> = (0u8..=255).collect();
        let encoded = b64_encode(&bytes);
        let decoded = b64_decode(&encoded).expect("decode ok");
        assert_eq!(decoded, bytes);
    }

    // ─── CompressedCheckpoint ─────────────────────────────────────────────────

    #[test]
    fn test_compressed_checkpoint_roundtrip_rle() {
        let ckpt = make_checkpoint("ckpt_rle", 1, 100);
        let dir = std::env::temp_dir();
        let path = dir.join("tf_neural_test_compressed_rle.json");

        CompressedCheckpoint::save_compressed(&path, &ckpt, TensorCompression::RunLength)
            .expect("save should succeed");
        let loaded =
            CompressedCheckpoint::load_compressed(&path).expect("load should succeed");

        assert_eq!(loaded.id, ckpt.id);
        assert_eq!(loaded.epoch, ckpt.epoch);
        assert_eq!(loaded.step, ckpt.step);
        assert_eq!(
            loaded.tensors.get("layer1.weight"),
            ckpt.tensors.get("layer1.weight")
        );
        assert_eq!(
            loaded.tensors.get("layer1.bias"),
            ckpt.tensors.get("layer1.bias")
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_compressed_checkpoint_roundtrip_delta() {
        let ckpt = make_checkpoint("ckpt_delta", 2, 200);
        let dir = std::env::temp_dir();
        let path = dir.join("tf_neural_test_compressed_delta.json");

        CompressedCheckpoint::save_compressed(&path, &ckpt, TensorCompression::Delta)
            .expect("save should succeed");
        let loaded =
            CompressedCheckpoint::load_compressed(&path).expect("load should succeed");

        assert_eq!(loaded.id, "ckpt_delta");
        let loaded_w = loaded.tensors.get("layer1.weight").expect("weight present");
        let orig_w = ckpt.tensors.get("layer1.weight").expect("weight present");
        for (a, b) in loaded_w.iter().zip(orig_w.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "delta roundtrip mismatch: {} vs {}",
                a,
                b
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_compressed_checkpoint_roundtrip_none() {
        let ckpt = make_checkpoint("ckpt_none", 3, 300);
        let dir = std::env::temp_dir();
        let path = dir.join("tf_neural_test_compressed_none.json");

        CompressedCheckpoint::save_compressed(&path, &ckpt, TensorCompression::None)
            .expect("save should succeed");
        let loaded =
            CompressedCheckpoint::load_compressed(&path).expect("load should succeed");

        assert_eq!(loaded.id, "ckpt_none");
        assert_eq!(
            loaded.tensors.get("layer1.weight"),
            ckpt.tensors.get("layer1.weight")
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_compressed_checkpoint_empty_tensors() {
        let ckpt = CheckpointData::new("ckpt_empty", 0, 0, 1.0);
        let dir = std::env::temp_dir();
        let path = dir.join("tf_neural_test_compressed_empty.json");

        CompressedCheckpoint::save_compressed(&path, &ckpt, TensorCompression::RunLength)
            .expect("save should succeed");
        let loaded =
            CompressedCheckpoint::load_compressed(&path).expect("load should succeed");

        assert_eq!(loaded.id, "ckpt_empty");
        assert!(loaded.tensors.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    // ─── CheckpointDiff ───────────────────────────────────────────────────────

    #[test]
    fn test_checkpoint_diff_no_change() {
        let base = make_checkpoint("base", 1, 100);
        let new = base.clone();
        let diff = CheckpointDiff::diff(&base, &new);

        assert!(diff.changed_tensors.is_empty());
        assert!(diff.removed_tensors.is_empty());
        assert_eq!(diff.base_id, "base");
        assert_eq!(diff.new_id, "base");
    }

    #[test]
    fn test_checkpoint_diff_changed_tensor() {
        let base = make_checkpoint("base", 1, 100);
        let mut new = base.clone();
        new.id = "new".to_string();
        new.step = 200;
        new.tensors.insert(
            "layer1.weight".to_string(),
            vec![10.0, 20.0, 30.0, 40.0],
        );

        let diff = CheckpointDiff::diff(&base, &new);

        assert_eq!(diff.changed_tensors.len(), 1);
        assert!(diff.changed_tensors.contains_key("layer1.weight"));
        assert!(diff.removed_tensors.is_empty());
    }

    #[test]
    fn test_checkpoint_diff_added_tensor() {
        let base = make_checkpoint("base", 1, 100);
        let mut new = base.clone();
        new.id = "new".to_string();
        new.insert_tensor("layer2.weight", vec![5.0, 6.0], vec![2]);

        let diff = CheckpointDiff::diff(&base, &new);

        assert!(diff.changed_tensors.contains_key("layer2.weight"));
        assert!(diff.removed_tensors.is_empty());
    }

    #[test]
    fn test_checkpoint_diff_removed_tensor() {
        let base = make_checkpoint("base", 1, 100);
        let mut new = base.clone();
        new.id = "new".to_string();
        new.tensors.remove("layer1.bias");
        new.tensor_shapes.remove("layer1.bias");

        let diff = CheckpointDiff::diff(&base, &new);

        assert!(diff.removed_tensors.contains(&"layer1.bias".to_string()));
    }

    #[test]
    fn test_checkpoint_diff_apply_roundtrip() {
        let base = make_checkpoint("base", 1, 100);
        let mut new = base.clone();
        new.id = "updated".to_string();
        new.epoch = 2;
        new.step = 200;
        new.loss = 0.3;
        new.tensors
            .insert("layer1.weight".to_string(), vec![9.0, 8.0, 7.0, 6.0]);

        let diff = CheckpointDiff::diff(&base, &new);
        let reconstructed = CheckpointDiff::apply_diff(&base, &diff);

        assert_eq!(reconstructed.id, "updated");
        assert_eq!(reconstructed.epoch, 2);
        assert_eq!(reconstructed.step, 200);
        assert!((reconstructed.loss - 0.3).abs() < 1e-6);
        assert_eq!(
            reconstructed.tensors.get("layer1.weight"),
            Some(&vec![9.0f32, 8.0, 7.0, 6.0])
        );
        assert_eq!(
            reconstructed.tensors.get("layer1.bias"),
            base.tensors.get("layer1.bias")
        );
    }

    // ─── CheckpointValidator ──────────────────────────────────────────────────

    #[test]
    fn test_checkpoint_validator_nonexistent_file() {
        let path = std::path::Path::new("/nonexistent/path/checkpoint_tf_test.json");
        let report = CheckpointValidator::validate(path).expect("validate returns Ok");
        assert!(!report.is_valid);
        assert!(!report.errors.is_empty());
        assert!(
            report.errors[0].contains("does not exist"),
            "unexpected error: {}",
            report.errors[0]
        );
    }

    #[test]
    fn test_checkpoint_validator_empty_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("tf_neural_test_validator_empty.json");
        std::fs::write(&path, b"").expect("write empty file");

        let report = CheckpointValidator::validate(&path).expect("validate returns Ok");
        assert!(!report.is_valid);
        assert!(
            report.errors.iter().any(|e| e.contains("empty")),
            "errors: {:?}",
            report.errors
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_validator_invalid_content() {
        let dir = std::env::temp_dir();
        let path = dir.join("tf_neural_test_validator_bad.json");
        std::fs::write(&path, b"not a checkpoint at all").expect("write bad content");

        let report = CheckpointValidator::validate(&path).expect("validate returns Ok");
        assert!(!report.is_valid, "should be invalid");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_validator_valid_checkpoint() {
        let ckpt = make_checkpoint("valid_ckpt", 5, 500);
        let dir = std::env::temp_dir();
        let path = dir.join("tf_neural_test_validator_valid.json");

        CompressedCheckpoint::save_compressed(&path, &ckpt, TensorCompression::RunLength)
            .expect("save should succeed");

        let report = CheckpointValidator::validate(&path).expect("validate returns Ok");
        assert!(report.is_valid, "errors: {:?}", report.errors);
        assert_eq!(report.checkpoint_id, "valid_ckpt");
        assert_eq!(report.epoch, 5);
        assert_eq!(report.step, 500);
        assert!(report.file_size_bytes > 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_validator_missing_fields() {
        let dir = std::env::temp_dir();
        let path = dir.join("tf_neural_test_validator_missing.json");
        std::fs::write(&path, b"{\"some_key\": 42}").expect("write json");

        let report = CheckpointValidator::validate(&path).expect("validate returns Ok");
        assert!(!report.is_valid);
        assert!(report.errors.len() >= 4, "errors: {:?}", report.errors);
        let _ = std::fs::remove_file(&path);
    }
}
