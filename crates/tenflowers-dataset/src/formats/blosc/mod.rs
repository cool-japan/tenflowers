//! Blosc chunk decoder for Zarr arrays.
//!
//! [Blosc](https://github.com/Blosc/c-blosc) is a "meta-compressor": each
//! compressed chunk is a small container format wrapping one of several
//! interchangeable inner codecs (`blosclz`, `lz4`/`lz4hc`, `snappy`, `zlib`,
//! `zstd`), plus an optional byte-shuffle or bit-shuffle pre-filter applied
//! before compression. This module implements the container format and all
//! five inner codecs so that Zarr chunks written with `"compressor":
//! {"id": "blosc", ...}` can be decoded for real.
//!
//! The container format (verified directly against the upstream C-Blosc
//! source -- `blosc/blosc.c`, `blosc/blosc.h`, `README_CHUNK_FORMAT.rst` --
//! rather than assumed) is:
//!
//! ```text
//! offset  size  field
//! 0       1     version
//! 1       1     versionlz
//! 2       1     flags (+ compressor id in bits 5-7)
//! 3       1     typesize
//! 4       4     nbytes     (u32 LE, uncompressed size)
//! 8       4     blocksize  (u32 LE, nominal per-block uncompressed size)
//! 12      4     cbytes     (u32 LE, total compressed size, header included)
//! 16      ..    either:
//!                 - raw `nbytes` bytes, if flags bit 1 (memcpy'ed) is set, or
//!                 - a table of `ceil(nbytes/blocksize)` LE `u32` block
//!                   start offsets, followed by the blocks themselves.
//! ```
//!
//! Each (non-memcpy'ed) block is itself split into `1` or `typesize`
//! "splits" (sub-streams), each individually compressed and each prefixed by
//! its own `i32` LE compressed length. Splitting only happens when the
//! "don't split" flag bit is clear *and* `typesize <= 16` *and*
//! `blocksize / typesize >= 128` *and* the block isn't the shorter, final
//! "leftover" block -- see `decode_block`.
//!
//! This implementation was validated against real chunks produced by the
//! reference C-Blosc library (via its Python bindings) covering blosclz,
//! lz4, and zlib as inner codecs (snappy was unavailable in the local
//! verification build; see the `snappy` tests for how that codec was
//! validated instead), both shuffle filters, single- and multi-block
//! chunks, blocks with a non-power-of-two "leftover" remainder, and the
//! whole-chunk memcpy fallback.
//!
//! zstd is fully implemented and its container-level handling here was
//! verified the same way the other codecs were, but the underlying
//! `oxiarc-zstd` 0.3.3 crate has its own pre-existing decoding bug (an
//! arithmetic-underflow panic in its Huffman-literals bitstream padding
//! calculation, plus at least one non-panicking parse error) that surfaces
//! on some short/highly-compressible zstd frames -- see
//! `tests::KNOWN_OXIARC_ZSTD_AFFECTED_VECTORS` for the detailed writeup and
//! `call_third_party_codec` below for how that failure mode is contained
//! (converted into a clean [`TensorError`] rather than an unwinding panic).

mod blosclz;
mod filters;
#[cfg(test)]
mod tests;

use tenflowers_core::{Result, TensorError};

/// Size in bytes of the fixed blosc chunk header.
const HEADER_LEN: usize = 16;

/// Flag bit: a byte-shuffle filter was applied before compression.
const FLAG_BYTE_SHUFFLE: u8 = 0x01;
/// Flag bit: the chunk body is a raw, uncompressed copy (no bstarts table,
/// no per-block splits -- just `nbytes` raw bytes starting at offset 16).
const FLAG_MEMCPYED: u8 = 0x02;
/// Flag bit: a bit-shuffle filter was applied before compression.
const FLAG_BIT_SHUFFLE: u8 = 0x04;
/// Flag bit: blocks were *not* split into `typesize` sub-streams.
const FLAG_NO_SPLIT: u8 = 0x10;
/// Bits 5-7 of the flags byte encode which inner codec was used.
const FORMAT_CODE_MASK: u8 = 0xE0;
const FORMAT_CODE_SHIFT: u32 = 5;

/// A block is only split into `typesize` sub-streams when `typesize` is no
/// larger than this.
const MAX_SPLITS: usize = 16;
/// ... and only when each resulting sub-stream would be at least this many
/// bytes (`blocksize / typesize >= MIN_BUFFERSIZE_FOR_SPLIT`).
const MIN_BUFFERSIZE_FOR_SPLIT: usize = 128;

/// Build a [`TensorError`] for malformed/truncated blosc input.
///
/// Centralized so every call site produces a consistently-prefixed message;
/// used throughout this module and its submodules.
fn corrupt(context: &str) -> TensorError {
    TensorError::invalid_argument(format!("corrupt blosc chunk: {context}"))
}

/// The parsed fixed-size blosc chunk header.
#[derive(Debug, Clone, Copy)]
struct BloscHeader {
    flags: u8,
    typesize: u8,
    nbytes: u32,
    blocksize: u32,
    cbytes: u32,
}

impl BloscHeader {
    /// Parse the 16-byte header from the start of `data`.
    ///
    /// `version`/`versionlz` are intentionally not validated against a
    /// specific expected value: being lenient there is strictly safer than
    /// rejecting otherwise-decodable chunks written by a newer/older but
    /// wire-compatible blosc release.
    fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_LEN {
            return Err(corrupt(&format!(
                "need at least {HEADER_LEN} header bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            flags: data[2],
            typesize: data[3],
            nbytes: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
            blocksize: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            cbytes: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
        })
    }

    fn is_memcpyed(&self) -> bool {
        self.flags & FLAG_MEMCPYED != 0
    }

    fn is_byte_shuffled(&self) -> bool {
        self.flags & FLAG_BYTE_SHUFFLE != 0
    }

    fn is_bit_shuffled(&self) -> bool {
        self.flags & FLAG_BIT_SHUFFLE != 0
    }

    fn no_split(&self) -> bool {
        self.flags & FLAG_NO_SPLIT != 0
    }

    fn format_code(&self) -> u8 {
        (self.flags & FORMAT_CODE_MASK) >> FORMAT_CODE_SHIFT
    }
}

/// The inner compression codec used for a chunk's blocks/splits, as encoded
/// in bits 5-7 of the header's flags byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InnerCodec {
    BloscLz,
    Lz4,
    Snappy,
    Zlib,
    Zstd,
}

impl InnerCodec {
    fn from_format_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::BloscLz),
            1 => Ok(Self::Lz4),
            2 => Ok(Self::Snappy),
            3 => Ok(Self::Zlib),
            4 => Ok(Self::Zstd),
            other => Err(corrupt(&format!(
                "unknown blosc inner codec id {other} (expected 0..=4 for \
                 blosclz/lz4/snappy/zlib/zstd)"
            ))),
        }
    }
}

/// Decompress one split (a single compressed sub-stream within a block)
/// using the chunk's declared inner codec. `expected_len` is used as an
/// output-size bound; the caller separately verifies the returned length.
///
/// Calls into the four third-party codecs are wrapped with
/// [`call_third_party_codec`] so that a panic inside one of them (as
/// opposed to a clean [`Result::Err`]) can never unwind out of this crate;
/// `blosclz::decompress` is our own carefully bounds-checked code and is
/// intentionally left unwrapped so a latent bug there would still surface
/// as a normal test panic during development.
fn decode_split(codec: InnerCodec, data: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    match codec {
        InnerCodec::BloscLz => blosclz::decompress(data, expected_len),
        InnerCodec::Lz4 => call_third_party_codec("lz4", || {
            oxiarc_lz4::block::decompress_block(data, expected_len)
                .map_err(|e| corrupt(&format!("lz4 split decompression failed: {e}")))
        }),
        InnerCodec::Snappy => call_third_party_codec("snappy", || {
            oxiarc_snappy::decompress(data)
                .map_err(|e| corrupt(&format!("snappy split decompression failed: {e}")))
        }),
        InnerCodec::Zlib => call_third_party_codec("zlib", || {
            oxiarc_deflate::zlib::zlib_decompress(data)
                .map_err(|e| corrupt(&format!("zlib split decompression failed: {e}")))
        }),
        InnerCodec::Zstd => call_third_party_codec("zstd", || {
            oxiarc_archive::zstd::decompress(data)
                .map_err(|e| corrupt(&format!("zstd split decompression failed: {e}")))
        }),
    }
}

/// Invoke a third-party codec's decompressor, converting any panic it
/// raises into a clean [`TensorError`] instead of letting it unwind through
/// this crate.
///
/// This crate's own code never panics on untrusted input by construction,
/// but not every dependency has the same guarantee -- for example
/// `oxiarc-zstd` 0.3.3 has a known arithmetic-overflow panic
/// (`huffman.rs`, Huffman bitstream padding calculation) on certain short
/// Huffman-coded literal streams, which real (non-malformed) zstd frames
/// can legitimately produce. Catching that here keeps this decoder's "never
/// crash on real input" guarantee intact even when a dependency's does not.
fn call_third_party_codec(
    codec_name: &str,
    f: impl FnOnce() -> Result<Vec<u8>> + std::panic::UnwindSafe,
) -> Result<Vec<u8>> {
    std::panic::catch_unwind(f).unwrap_or_else(|payload| {
        let detail = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_string());
        Err(corrupt(&format!(
            "{codec_name} decoder panicked internally: {detail}"
        )))
    })
}

/// Decode one block (`bsize` uncompressed bytes) starting at `src_offset`
/// within `compressed`. Mirrors the reference `blosc_d` function: reads
/// `nsplits` length-prefixed splits, concatenates their decompressed bytes,
/// then applies whichever inverse filter (byte-unshuffle / bit-unshuffle /
/// none) the header requests.
///
/// (Many parameters: the workspace lint config already sets
/// `clippy::too_many_arguments = "allow"`, so no local `#[allow(...)]` is
/// needed here.)
fn decode_block(
    compressed: &[u8],
    src_offset: usize,
    bsize: usize,
    is_leftover_block: bool,
    typesize: usize,
    no_split: bool,
    byte_shuffle_flag: bool,
    bit_shuffle_flag: bool,
    codec: InnerCodec,
) -> Result<Vec<u8>> {
    if typesize == 0 {
        return Err(corrupt("header has typesize == 0"));
    }

    // Mirrors blosc_d's nsplits derivation exactly, including that it uses
    // this block's actual size (bsize), not the chunk's nominal blocksize.
    let nsplits: usize = if !no_split
        && typesize <= MAX_SPLITS
        && (bsize / typesize) >= MIN_BUFFERSIZE_FOR_SPLIT
        && !is_leftover_block
    {
        typesize
    } else {
        1
    };
    let neblock = bsize / nsplits;

    let mut shuffled = vec![0u8; bsize];
    let mut offset = src_offset;
    let mut filled = 0usize;

    for _ in 0..nsplits {
        let len_field_end = offset
            .checked_add(4)
            .ok_or_else(|| corrupt("split length prefix overflow"))?;
        if len_field_end > compressed.len() {
            return Err(corrupt("truncated split length prefix"));
        }
        let cbytes_raw = i32::from_le_bytes([
            compressed[offset],
            compressed[offset + 1],
            compressed[offset + 2],
            compressed[offset + 3],
        ]);
        if cbytes_raw < 0 {
            return Err(corrupt("split has a negative compressed length"));
        }
        let cbytes = cbytes_raw as usize;
        let data_start = len_field_end;
        let data_end = data_start
            .checked_add(cbytes)
            .ok_or_else(|| corrupt("split length overflow"))?;
        if data_end > compressed.len() {
            return Err(corrupt("truncated split data"));
        }
        let split_compressed = &compressed[data_start..data_end];

        let decoded = if cbytes == neblock {
            // Per-split memcpy fallback: the compressor gave up on this
            // split and stored it raw (its "compressed" length exactly
            // equals the uncompressed split size).
            split_compressed.to_vec()
        } else {
            decode_split(codec, split_compressed, neblock)?
        };
        if decoded.len() != neblock {
            return Err(corrupt(&format!(
                "split decoded to {} bytes, expected {neblock}",
                decoded.len()
            )));
        }

        let dst_end = (filled + neblock).min(bsize);
        let copy_len = dst_end - filled;
        shuffled[filled..dst_end].copy_from_slice(&decoded[..copy_len]);
        filled += neblock;
        offset = data_end;
    }

    let doshuffle = byte_shuffle_flag && typesize > 1;
    let dobitshuffle = bit_shuffle_flag && bsize >= typesize;

    if doshuffle {
        Ok(filters::unshuffle(typesize, &shuffled))
    } else if dobitshuffle {
        filters::bitunshuffle(typesize, &shuffled)
    } else {
        Ok(shuffled)
    }
}

/// Decompress a blosc-compressed Zarr chunk.
///
/// Returns the exact `nbytes` (as declared in the chunk header) of
/// decompressed bytes, or an error if `compressed` is truncated, internally
/// inconsistent, or uses an unsupported inner codec format code. Never
/// silently returns partially-decoded or still-compressed data.
pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>> {
    let header = BloscHeader::parse(compressed)?;

    if u64::from(header.cbytes) > compressed.len() as u64 {
        return Err(corrupt(&format!(
            "header declares cbytes={} but only {} bytes are available",
            header.cbytes,
            compressed.len()
        )));
    }

    let nbytes = header.nbytes as usize;
    if nbytes == 0 {
        return Ok(Vec::new());
    }

    if header.is_memcpyed() {
        let end = HEADER_LEN
            .checked_add(nbytes)
            .ok_or_else(|| corrupt("nbytes overflow"))?;
        if compressed.len() < end {
            return Err(corrupt(&format!(
                "memcpy'ed chunk truncated: need {end} bytes, have {}",
                compressed.len()
            )));
        }
        return Ok(compressed[HEADER_LEN..end].to_vec());
    }

    if header.blocksize == 0 {
        return Err(corrupt("blocksize == 0"));
    }
    let blocksize = header.blocksize as usize;
    let typesize = header.typesize as usize;
    let codec = InnerCodec::from_format_code(header.format_code())?;

    // ceil(nbytes / blocksize), computed without relying on a
    // post-1.73 stdlib API so this keeps compiling under the workspace's
    // declared MSRV.
    let nblocks = nbytes / blocksize + usize::from(nbytes % blocksize != 0);

    let bstarts_bytes = nblocks
        .checked_mul(4)
        .ok_or_else(|| corrupt("block count overflow"))?;
    let bstarts_end = HEADER_LEN
        .checked_add(bstarts_bytes)
        .ok_or_else(|| corrupt("block count overflow"))?;
    if compressed.len() < bstarts_end {
        return Err(corrupt(&format!(
            "block-starts table truncated: need {bstarts_end} bytes, have {}",
            compressed.len()
        )));
    }

    let mut bstarts = Vec::with_capacity(nblocks);
    for i in 0..nblocks {
        let off = HEADER_LEN + i * 4;
        let v = u32::from_le_bytes([
            compressed[off],
            compressed[off + 1],
            compressed[off + 2],
            compressed[off + 3],
        ]);
        bstarts.push(v as usize);
    }

    let byte_shuffle_flag = header.is_byte_shuffled();
    let bit_shuffle_flag = header.is_bit_shuffled();
    let no_split = header.no_split();

    let mut output = vec![0u8; nbytes];
    for (block_idx, &src_offset) in bstarts.iter().enumerate() {
        let block_start = block_idx * blocksize;
        let bsize = if block_idx == nblocks - 1 {
            nbytes - block_start
        } else {
            blocksize
        };
        let is_leftover_block = bsize != blocksize;

        if src_offset > compressed.len() {
            return Err(corrupt("block-start offset points outside the chunk"));
        }

        let decoded_block = decode_block(
            compressed,
            src_offset,
            bsize,
            is_leftover_block,
            typesize,
            no_split,
            byte_shuffle_flag,
            bit_shuffle_flag,
            codec,
        )?;
        if decoded_block.len() != bsize {
            return Err(corrupt(&format!(
                "block {block_idx} decoded to {} bytes, expected {bsize}",
                decoded_block.len()
            )));
        }
        output[block_start..block_start + bsize].copy_from_slice(&decoded_block);
    }

    Ok(output)
}
