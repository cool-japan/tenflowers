//! Inverse byte-shuffle and bit-shuffle filters.
//!
//! Blosc can apply a "shuffle" byte filter before compressing a block: for
//! `typesize`-byte elements, it transposes the block so that all byte-0's of
//! every element come first, then all byte-1's, and so on. This tends to
//! group similar-magnitude bytes together (e.g. all the exponent bytes of an
//! array of `f32`s), which downstream LZ/entropy coders can exploit far
//! better than the naturally-interleaved layout.
//!
//! "Bitshuffle" is the bit-level analogue: it transposes individual *bits*
//! within elements rather than whole bytes.
//!
//! Both routines here faithfully reproduce the *generic* (non-SIMD)
//! reference implementations from `blosc/shuffle-generic.h` and
//! `blosc/bitshuffle-generic.c`. The generic versions are what the reference
//! library itself falls back to on platforms without SSE2/AVX2, and they are
//! bit-for-bit equivalent to the vectorized versions used for the bulk of a
//! block on other platforms -- there is exactly one on-disk bitshuffle
//! format, only the implementation used to produce/consume it varies.
//!
//! Endianness: the reference bitshuffle implementation branches on host
//! endianness (`TRANS_BIT_8X8` vs `TRANS_BIT_8X8_BE`) purely to keep the
//! on-disk bit layout identical across little- and big-endian hosts. This
//! port only implements the little-endian path (`TRANS_BIT_8X8`), which
//! covers every realistic deployment target for this project (x86_64,
//! aarch64, wasm32).

use tenflowers_core::Result;

use super::corrupt;

/// Invert a byte-shuffle filter.
///
/// `shuffled` holds one block's worth of bytes (length == the block's
/// uncompressed size) and `typesize` is the element size used to shuffle it.
/// Only called when `typesize > 1` (shuffling a 1-byte-wide type is a no-op
/// in the reference implementation, so callers gate on that beforehand).
///
/// Trailing bytes that don't form a complete element (`blocksize % typesize`
/// of them) were left un-shuffled by the encoder and are copied back as-is,
/// matching `unshuffle_generic_inline`.
pub(crate) fn unshuffle(typesize: usize, shuffled: &[u8]) -> Vec<u8> {
    let blocksize = shuffled.len();
    // typesize > 1 is guaranteed by the caller; guard defensively anyway so
    // this function can never panic even if that invariant is violated.
    if typesize == 0 {
        return shuffled.to_vec();
    }
    let neblock_quot = blocksize / typesize;
    let neblock_rem = blocksize % typesize;
    let mut dest = vec![0u8; blocksize];

    for i in 0..neblock_quot {
        for j in 0..typesize {
            dest[i * typesize + j] = shuffled[j * neblock_quot + i];
        }
    }

    let leftover_start = blocksize - neblock_rem;
    dest[leftover_start..].copy_from_slice(&shuffled[leftover_start..]);
    dest
}

/// Invert a bit-shuffle filter on one block's worth of bytes.
///
/// Mirrors the reference dispatcher `blosc_internal_bitunshuffle`: bit
/// transposition only applies to the leading run of *complete* 8-element
/// groups (`size = blocksize / typesize` must itself be a multiple of 8);
/// everything else (including the whole block, when `size` isn't a multiple
/// of 8) is left untouched.
pub(crate) fn bitunshuffle(typesize: usize, data: &[u8]) -> Result<Vec<u8>> {
    if typesize == 0 {
        return Err(corrupt("bitshuffle block has typesize == 0"));
    }
    let blocksize = data.len();
    let size = blocksize / typesize;

    if size % 8 != 0 {
        return Ok(data.to_vec());
    }

    let core_len = size * typesize;
    let mut out = if size > 0 {
        let tmp = trans_byte_bitrow(&data[..core_len], size, typesize);
        shuffle_bit_eightelem(&tmp, size, typesize)
    } else {
        Vec::new()
    };
    out.extend_from_slice(&data[core_len..]);
    Ok(out)
}

/// Transpose an 8x8 bit matrix packed into a single `u64`, treating it as
/// eight rows of 8 bits each (little-endian byte order). This operation is
/// its own inverse (transposing twice returns the original matrix), which is
/// what lets the same primitive serve both the forward-shuffle and
/// inverse-shuffle directions in the reference implementation -- only the
/// surrounding byte-level permutation stages differ between the two.
///
/// Direct port of the `TRANS_BIT_8X8` macro from `bitshuffle-generic.h`.
#[inline]
fn trans_bit_8x8(mut x: u64) -> u64 {
    let mut t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AAu64;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCCu64;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0u64;
    x ^= t ^ (t << 28);
    x
}

/// Port of `bshuf_trans_byte_bitrow_scal`: a pure byte-level (not bit-level)
/// re-permutation from "one bit-row per output byte" layout into "grouped by
/// 8-element chunks" layout. `size` is the element count (multiple of 8),
/// `elem_size` is the element byte width (`typesize`).
fn trans_byte_bitrow(input: &[u8], size: usize, elem_size: usize) -> Vec<u8> {
    let nbyte_row = size / 8;
    let mut out = vec![0u8; size * elem_size];
    for jj in 0..elem_size {
        for ii in 0..nbyte_row {
            for kk in 0..8usize {
                out[ii * 8 * elem_size + jj * 8 + kk] = input[(jj * 8 + kk) * nbyte_row + ii];
            }
        }
    }
    out
}

/// Port of `blosc_internal_bshuf_shuffle_bit_eightelem_scal` (little-endian
/// path only; see module docs). Applies [`trans_bit_8x8`] to each
/// consecutive 8-byte word and scatters the resulting bytes to their final
/// positions.
fn shuffle_bit_eightelem(input: &[u8], size: usize, elem_size: usize) -> Vec<u8> {
    let nbyte = elem_size * size;
    let mut out = vec![0u8; nbyte];
    if elem_size == 0 {
        return out;
    }

    let mut jj = 0usize;
    while jj < 8 * elem_size {
        let stride = 8 * elem_size;
        let mut ii = 0usize;
        while ii + stride - 1 < nbyte {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&input[ii + jj..ii + jj + 8]);
            let mut x = trans_bit_8x8(u64::from_le_bytes(buf));
            for kk in 0..8usize {
                let out_index = ii + jj / 8 + kk * elem_size;
                out[out_index] = (x & 0xFF) as u8;
                x >>= 8;
            }
            ii += stride;
        }
        jj += 8;
    }
    out
}
