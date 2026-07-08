//! BloscLZ decompressor.
//!
//! BloscLZ is blosc's own default inner codec (format code `0`). It is a
//! small LZ77-style scheme: each "opcode" byte (`ctrl`) either starts a
//! literal run (`ctrl < 32`, run length `ctrl + 1`) or a back-reference
//! match (`ctrl >= 32`), possibly followed by extra length/offset bytes.
//!
//! This is a faithful, line-by-line port of the reference `blosclz_decompress`
//! routine from the upstream C-Blosc project (`blosc/blosclz.c`), translated
//! to bounds-checked, panic-free Rust. Every pointer-arithmetic step in the
//! original has a corresponding checked index computation here so that
//! malformed input produces a [`TensorError`] instead of an out-of-bounds
//! access or an arithmetic overflow.
//!
//! Reference layout (verified against the upstream source, not guessed):
//! - A `ctrl` byte `< 32` is a literal run of `ctrl + 1` raw bytes.
//! - A `ctrl` byte `>= 32` is a match:
//!   - `len = (ctrl >> 5) - 1`, extended via extra `0xFF`-terminated bytes
//!     when `len == 6` (i.e. the 3-bit field was maxed out).
//!   - `ofs = (ctrl & 31) << 8`, combined with the next byte (`code`) to
//!     form a near back-reference distance of `ofs + code + 1`.
//!   - `len` is finally biased by `+3` (minimum match length is 3).
//!   - If `code == 255` and `ofs` was the maximum (`31 << 8`), two more
//!     bytes encode a "far" 16-bit distance of `ofs_far + MAX_DISTANCE`.

use tenflowers_core::Result;

use super::corrupt;

/// Maximum "near" back-reference distance before switching to the 16-bit
/// "far" distance encoding. Matches `MAX_DISTANCE` in the reference decoder.
const MAX_DISTANCE: i64 = 8191;

/// Decompress a single BloscLZ-compressed split.
///
/// `max_output` is the exact number of bytes the split is expected to
/// decompress to (blosc always decompresses a split to a known, fixed
/// size); it is used purely as a safety bound while decoding; the returned
/// vector length reflects however many bytes were actually produced; the
/// caller is responsible for checking that it matches the expected length.
pub(crate) fn decompress(input: &[u8], max_output: usize) -> Result<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = vec![0u8; max_output];
    let ip_limit = input.len();
    let mut ip: usize = 1;
    let mut op: usize = 0;

    let mut ctrl = u32::from(input[0]) & 31;

    loop {
        if ctrl >= 32 {
            // --- match ---
            let mut len: i64 = i64::from(ctrl >> 5) - 1;
            let mut ofs: i64 = i64::from(ctrl & 31) << 8;

            if len == 6 {
                loop {
                    if ip + 1 >= ip_limit {
                        return Err(corrupt("blosclz: truncated extended match length"));
                    }
                    let code = input[ip];
                    ip += 1;
                    len += i64::from(code);
                    if code != 255 {
                        break;
                    }
                }
            } else if ip + 1 >= ip_limit {
                return Err(corrupt("blosclz: truncated match token"));
            }

            let code = input[ip];
            ip += 1;
            len += 3;
            let mut ref_pos: i64 = op as i64 - ofs - i64::from(code);

            if code == 255 && ofs == (31i64 << 8) {
                if ip + 1 >= ip_limit {
                    return Err(corrupt("blosclz: truncated far-distance bytes"));
                }
                let hi = input[ip];
                ip += 1;
                let lo = input[ip];
                ip += 1;
                ofs = (i64::from(hi) << 8) + i64::from(lo);
                ref_pos = op as i64 - ofs - MAX_DISTANCE;
            }

            let len_usize =
                usize::try_from(len).map_err(|_| corrupt("blosclz: invalid match length"))?;
            let op_end = op
                .checked_add(len_usize)
                .ok_or_else(|| corrupt("blosclz: match length overflow"))?;
            if op_end > max_output {
                return Err(corrupt("blosclz: match exceeds output bound"));
            }
            if ref_pos < 1 {
                return Err(corrupt(
                    "blosclz: back-reference points before start of output",
                ));
            }

            if ip >= ip_limit {
                // Matches the reference decoder exactly: if the input is
                // exhausted right here, the match's copy is *not* performed
                // and decoding stops. A well-formed encoder never emits a
                // match as the very last token (it always reserves a
                // trailing literal-run placeholder byte), so this only
                // matters for malformed/truncated streams.
                break;
            }
            ctrl = u32::from(input[ip]);
            ip += 1;

            ref_pos -= 1;
            let ref_idx =
                usize::try_from(ref_pos).map_err(|_| corrupt("blosclz: invalid back-reference"))?;
            if ref_idx >= op {
                return Err(corrupt(
                    "blosclz: back-reference does not precede current output position",
                ));
            }
            // Sequential byte-by-byte copy: this intentionally also covers
            // the "run" case (distance == 1, i.e. ref_idx == op - 1) since
            // reading a just-written byte reproduces run-length repetition
            // without a special case, matching `memset`/`wild_copy` in the
            // reference decoder for any amount of source/destination
            // overlap.
            for k in 0..len_usize {
                output[op + k] = output[ref_idx + k];
            }
            op = op_end;
        } else {
            // --- literal run ---
            let lit_len = (ctrl + 1) as usize;
            let op_end = op
                .checked_add(lit_len)
                .ok_or_else(|| corrupt("blosclz: literal length overflow"))?;
            if op_end > max_output {
                return Err(corrupt("blosclz: literal run exceeds output bound"));
            }
            let ip_end = ip
                .checked_add(lit_len)
                .ok_or_else(|| corrupt("blosclz: literal length overflow"))?;
            if ip_end > ip_limit {
                return Err(corrupt("blosclz: truncated literal run"));
            }
            output[op..op_end].copy_from_slice(&input[ip..ip_end]);
            op = op_end;
            ip = ip_end;

            if ip >= ip_limit {
                break;
            }
            ctrl = u32::from(input[ip]);
            ip += 1;
        }
    }

    output.truncate(op);
    Ok(output)
}
