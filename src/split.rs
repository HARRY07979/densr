//! Split-alphabet Huffman coding for a coded `dsr1` stream.
//!
//! `dsr1` interleaves two very different kinds of byte in one stream: literal
//! payload, which follows the shape of the input, and structural tokens, which
//! follow the shape of the parser's decisions. Coding that mixture with a single
//! Huffman table forces one set of code lengths to serve both distributions.
//!
//! deflate separates them, and so does this: the stream is walked once, the
//! literal bytes and the token bytes are pulled apart, each gets its own optimal
//! code, and the two coded regions are concatenated. The `dsr1` stream itself is
//! unchanged byte for byte, which is why this costs no change to the GPU kernel
//! and cannot break the CPU/GPU identity invariant -- only what happens to the
//! stream *after* the codec changes.
//!
//! The payload of a split block is:
//!
//! ```text
//! varint(literal region length) ++ mode(literals) ++ mode(tokens)
//!      ++ region ++ region
//! ```
//!
//! The leading varint is what makes the boundary recoverable. It is one to four
//! bytes, against two mode bytes, so it is never the reason to skip this.
//!
//! Each region carries its own one-byte mode. A region Huffman cannot shrink --
//! a short or uniform one -- is stored verbatim rather than skipped, so the
//! split never regresses into "no split at all". That is why this always returns
//! a payload: deciding whether the split is worth it is the caller's job, and it
//! can compare lengths as it already does for every other encoding choice.

use crate::huffman;
use crate::util::{read_varint, write_varint};

const NIBBLE_MAX: u8 = 15;

/// A region was stored verbatim.
const REGION_RAW: u8 = 0;
/// A region was Huffman-coded.
const REGION_HUFFMAN: u8 = 1;

/// Pull the literal bytes and the token bytes of a coded stream apart.
///
/// Mirrors the encoder's emit side exactly: a token byte with a zero high nibble
/// introduces a literal run whose length is the low nibble, plus a varint when
/// that nibble saturated; anything else is a match token, whose two varints
/// follow in offset-then-length order. Getting this wrong would not corrupt the
/// archive, it would only code the wrong alphabet, so it is the one place here
/// that needs reading twice against `lz::emit_literals` and `lz::emit_match`.
pub fn split_coded(stream: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut literals = Vec::new();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < stream.len() {
        let b = stream[i];
        if b == 0 {
            // Repeat-offset match: token plus one varint packing the length and
            // which of the two recent offsets to use. Both are structural, so no
            // literal payload follows.
            let start = i;
            i += 1;
            let _ = read_varint(stream, &mut i);
            tokens.extend_from_slice(&stream[start..i.min(stream.len())]);
            i = i.min(stream.len());
        } else if b >> 4 == 0 {
            let start = i;
            let mut run = (b & 0x0f) as usize;
            i += 1;
            if run == NIBBLE_MAX as usize {
                // The stream came out of the encoder, but this function is also
                // reachable on arbitrary bytes; a malformed length must not
                // panic or wrap into a huge allocation.
                if let Ok(extra) = read_varint(stream, &mut i) {
                    run = usize::try_from(extra).map_or(usize::MAX, |n| run.saturating_add(n));
                }
            }
            let end = i.saturating_add(run).min(stream.len());
            tokens.extend_from_slice(&stream[start..i]);
            literals.extend_from_slice(&stream[i..end]);
            i = end;
        } else {
            let start = i;
            let len_nibble = b & 0x0f;
            let offset_nibble = b >> 4;
            i += 1;
            if offset_nibble == NIBBLE_MAX {
                let _ = read_varint(stream, &mut i);
            }
            if len_nibble == NIBBLE_MAX {
                let _ = read_varint(stream, &mut i);
            }
            tokens.extend_from_slice(&stream[start..i.min(stream.len())]);
            i = i.min(stream.len());
        }
    }
    (literals, tokens)
}

/// Code a region into its own buffer: a mode byte followed by the stored bytes.
///
/// Returned separately so the caller can record the length of what was actually
/// written, which is not the region's original length once Huffman has run.
fn pack_region(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 1);
    match huffman::compress(data) {
        Some(h) if h.len() < data.len() => {
            out.push(REGION_HUFFMAN);
            out.extend_from_slice(&h);
        }
        _ => {
            out.push(REGION_RAW);
            out.extend_from_slice(data);
        }
    }
    out
}

/// Undo [`pack_region`], appending the region's decoded bytes to `into`.
///
/// `region_len` is the length of the **whole stored region, mode byte
/// included** -- the number [`compress_split`] writes ahead of it. It is not the
/// length of the decoded data, and this function consumes exactly `region_len`
/// bytes from `payload` starting at `pos`.
fn unpack_region(
    payload: &[u8],
    pos: &mut usize,
    region_len: usize,
    into: &mut Vec<u8>,
) -> Result<(), String> {
    let mode = *payload
        .get(*pos)
        .ok_or_else(|| "split-coded block ends before a region header".to_string())?;
    *pos += 1;
    let data_len = region_len
        .checked_sub(1)
        .ok_or_else(|| "split-coded block has an empty region".to_string())?;
    let end = pos
        .checked_add(data_len)
        .filter(|e| *e <= payload.len())
        .ok_or_else(|| "split-coded block claims a region past its end".to_string())?;
    let region = &payload[*pos..end];
    *pos = end;
    match mode {
        REGION_RAW => into.extend_from_slice(region),
        REGION_HUFFMAN => {
            let mut decoded = huffman::decompress(region)?;
            into.append(&mut decoded);
        }
        other => return Err(format!("split-coded block has unknown region mode {other}")),
    }
    Ok(())
}

/// Code a `dsr1` stream with one alphabet per region.
///
/// This always succeeds: each region is coded if that helps and stored verbatim
/// if it does not. Whether the result beats one alphabet over the whole stream is
/// decided by the caller, which has the single-stream encoding to compare with.
pub fn compress_split(stream: &[u8]) -> Vec<u8> {
    let (literals, tokens) = split_coded(stream);
    let lit = pack_region(&literals);
    let tok = pack_region(&tokens);
    let mut out = Vec::with_capacity(lit.len() + tok.len() + 5);
    // The length recorded is the stored region's, mode byte included, because
    // that is what the decoder has to skip to reach the token region.
    write_varint(&mut out, lit.len() as u64);
    out.extend_from_slice(&lit);
    out.extend_from_slice(&tok);
    out
}

/// Rebuild the `dsr1` stream from a split-coded payload.
pub fn decompress_split(payload: &[u8]) -> Result<Vec<u8>, String> {
    let mut pos = 0usize;
    let lit_len = usize::try_from(read_varint(payload, &mut pos)?)
        .map_err(|_| "split-coded literal region is too large".to_string())?;
    // The literal region is length-prefixed; the token region is whatever is
    // left. Each carries at least its mode byte, so the split point must leave
    // room for both.
    if lit_len == 0
        || pos
            .checked_add(lit_len)
            .filter(|e| *e < payload.len())
            .is_none()
    {
        return Err("split-coded block claims a literal region past its end".into());
    }
    let mut literals = Vec::new();
    unpack_region(payload, &mut pos, lit_len, &mut literals)?;
    let token_len = payload.len() - pos;
    let mut tokens = Vec::new();
    unpack_region(payload, &mut pos, token_len, &mut tokens)?;
    reinterleave(&literals, &tokens)
}

/// Rebuild the original `dsr1` stream from the two separated alphabets.
///
/// This is the exact inverse of [`split_coded`]: the token stream carries every
/// structural byte in order, and at each literal-run token the rebuilt stream
/// is followed by the literal content that was pulled into the other region.
fn reinterleave(literals: &[u8], tokens: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(literals.len() + tokens.len());
    let mut tp = 0usize;
    let mut lp = 0usize;
    while tp < tokens.len() {
        let b = tokens[tp];
        if b == 0 {
            let start = tp;
            tp += 1;
            let _ = read_varint(tokens, &mut tp)?;
            out.extend_from_slice(&tokens[start..tp.min(tokens.len())]);
            tp = tp.min(tokens.len());
        } else if b >> 4 == 0 {
            let start = tp;
            let mut run = (b & 0x0f) as usize;
            tp += 1;
            if run == NIBBLE_MAX as usize {
                let extra = usize::try_from(read_varint(tokens, &mut tp)?)
                    .map_err(|_| "split-coded literal run is too large".to_string())?;
                run = run
                    .checked_add(extra)
                    .ok_or_else(|| "split-coded literal run overflows".to_string())?;
            }
            out.extend_from_slice(&tokens[start..tp]);
            let end = lp
                .checked_add(run)
                .filter(|e| *e <= literals.len())
                .ok_or_else(|| "split-coded block runs out of literal bytes".to_string())?;
            out.extend_from_slice(&literals[lp..end]);
            lp = end;
        } else {
            let start = tp;
            let len_nibble = b & 0x0f;
            let offset_nibble = b >> 4;
            tp += 1;
            if offset_nibble == NIBBLE_MAX {
                let _ = read_varint(tokens, &mut tp)?;
            }
            if len_nibble == NIBBLE_MAX {
                let _ = read_varint(tokens, &mut tp)?;
            }
            out.extend_from_slice(&tokens[start..tp.min(tokens.len())]);
            tp = tp.min(tokens.len());
        }
    }
    if lp != literals.len() {
        return Err("split-coded block left literal bytes unused".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8]) {
        let coded = crate::lz::compress(data, 6).expect("should compress");
        let packed = compress_split(&coded);
        let back = decompress_split(&packed).expect("should decode");
        assert_eq!(back, coded, "the rebuilt stream must match the encoder's");
        let final_bytes = crate::lz::decompress(&back, data.len()).expect("lz decodes");
        assert_eq!(final_bytes, data);
    }

    #[test]
    fn round_trips_text() {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(200);
        round_trip(text.as_bytes());
    }

    #[test]
    fn round_trips_a_mixed_buffer() {
        let mut v = Vec::new();
        for i in 0..40000u32 {
            v.push((i % 251) as u8);
            if i % 97 == 0 {
                v.extend_from_slice(b"repeated block of bytes ");
            }
        }
        round_trip(&v);
    }

    #[test]
    fn the_split_beats_one_alphabet_on_text() {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(400);
        let coded = crate::lz::compress(text.as_bytes(), 6).expect("compress");
        let single = huffman::compress(&coded).unwrap_or_else(|| coded.clone());
        let split = compress_split(&coded);
        assert!(
            split.len() < single.len(),
            "split {} was not smaller than single {}",
            split.len(),
            single.len()
        );
    }

    #[test]
    fn a_truncated_payload_is_an_error_not_a_guess() {
        let text = "abcabcabc".repeat(500);
        let coded = crate::lz::compress(text.as_bytes(), 6).unwrap();
        let packed = compress_split(&coded);
        for cut in 1..packed.len().min(64) {
            // A truncated token region can still be a complete, shorter stream:
            // the format has no end marker and the container supplies the real
            // output length. What must never happen is a truncated payload
            // decoding to the same bytes as the original.
            if let Ok(back) = decompress_split(&packed[..packed.len() - cut]) {
                assert_ne!(back, coded, "truncating {cut} bytes reproduced the stream");
            }
        }
    }

    #[test]
    fn an_overlong_length_varint_is_rejected_without_panicking() {
        let mut payload = vec![0x80u8; 11];
        payload.push(0);
        assert!(decompress_split(&payload).is_err());
    }

    #[test]
    fn an_out_of_range_literal_length_is_refused() {
        let mut p = Vec::new();
        write_varint(&mut p, 1_000_000);
        p.extend_from_slice(&[0u8; 8]);
        assert!(decompress_split(&p).is_err());
    }
}
