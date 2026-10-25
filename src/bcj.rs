//! x86/x86-64 branch-conversion filter (BCJ, a.k.a. Bra86).
//!
//! # Why
//!
//! Machine code stores most branch targets as a distance from the instruction
//! itself. Two byte-identical functions living at different addresses therefore
//! store *different* relative offsets, and LZ cannot match them. Converting
//! those offsets to absolute makes the addresses repeat, so the same code at two
//! addresses becomes the same bytes and compresses to a match.
//!
//! 7-Zip applies BCJ/BCJ2 to `.exe`/`.dll` by default; xz ships BCJ filters; the
//! Linux kernel compresses itself with BCJ. The technique is old and settled, and
//! this is the textbook one-instruction-loop version of it.
//!
//! # What it does
//!
//! Three instruction encodings carry a 32-bit displacement:
//!
//! | opcode            | instruction             | displacement at |
//! |-------------------|-------------------------|-----------------|
//! | `E8`              | `CALL rel32`            | `+1`            |
//! | `E9`              | `JMP rel32`             | `+1`            |
//! | `0F 80..8F`       | `Jcc rel32` (near)      | `+2`            |
//!
//! Encode adds the displacement's own address to the value; decode subtracts it.
//! Absolute target = relative target + (address of the *next* instruction), which
//! is why the two spellings of that address appear above.
//!
//! # Why it is safe
//!
//! Both directions walk the buffer the same way and **skip past every converted
//! 4-byte field**, so both see identical opcode positions and the transform is
//! exactly invertible. Because no opcode byte is ever rewritten, the byte stream
//! is bit-identical after a round trip.
//!
//! Nothing validates that the targets are plausible addresses. The filter is a
//! blind reinterpretation; "rejecting" implausible targets is how a filter starts
//! corrupting files, and on x86 a displacement is not required to be an offset
//! into the file in the first place.
//!
//! # Per-block, not streaming
//!
//! `encode` and `decode` are called on the same block with the same arguments,
//! which is what makes them cancel exactly. xz's stronger variant threads a
//! running instruction pointer across buffers to catch matches that straddle a
//! boundary. That buys a little ratio and costs the property that a block is
//! self-contained, which this container needs: every block carries its own
//! checksums and must decode from its record alone.

/// Convert relative branch displacements to absolute, in place, or back again.
///
/// `encode = true` is the write path, run before compression. `false` is the read
/// path, run after decompression. Both are self-inverse on the same buffer.
///
/// Trailing bytes that cannot hold a complete instruction are left alone, which
/// is what lets a block end anywhere.
pub fn x86(data: &mut [u8], encode: bool) {
    let len = data.len();
    let mut i = 0usize;

    while i + 5 <= len {
        let b = data[i];
        if b == 0xE8 || b == 0xE9 {
            // CALL rel32 / JMP rel32: displacement starts at i+1, and the
            // instruction is 5 bytes so the next one is at i+5.
            let v = i32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]);
            let v = if encode {
                v.wrapping_add((i + 5) as i32)
            } else {
                v.wrapping_sub((i + 5) as i32)
            };
            data[i + 1..i + 5].copy_from_slice(&v.to_le_bytes());
            i += 5;
        } else if b == 0x0F && i + 6 <= len && (data[i + 1] & 0xF0) == 0x80 {
            // Jcc rel32 near: displacement starts at i+2, instruction is 6 bytes.
            // The `i + 6 <= len` guard is separate from the loop's `i + 5 <= len`
            // because this encoding is one byte longer; without it a buffer whose
            // last byte is `0F 80` would be read past the end.
            let v = i32::from_le_bytes([data[i + 2], data[i + 3], data[i + 4], data[i + 5]]);
            let v = if encode {
                v.wrapping_add((i + 6) as i32)
            } else {
                v.wrapping_sub((i + 6) as i32)
            };
            data[i + 2..i + 6].copy_from_slice(&v.to_le_bytes());
            i += 6;
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round trip must be the identity for every length, including the ones that
    /// end mid-instruction. Off-by-one in the length guard shows up here.
    #[test]
    fn round_trips_at_every_length() {
        let mut s = 0x0123_4567_89AB_CDEFu64;
        for len in 0..600usize {
            let mut data = Vec::with_capacity(len);
            while data.len() < len {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                data.push((s >> 33) as u8);
            }
            let original = data.clone();
            x86(&mut data, true);
            x86(&mut data, false);
            assert_eq!(data, original, "length {len} did not round trip");
        }
    }

    #[test]
    fn round_trips_buffers_that_end_mid_instruction() {
        // Place opcodes right at the end so the trailing bytes are a partial
        // displacement.
        for len in 0..16usize {
            for opcode in [0xE8u8, 0xE9, 0x0F] {
                let mut data = vec![0u8; len];
                if len > 0 {
                    data[len - 1] = opcode;
                }
                let original = data.clone();
                x86(&mut data, true);
                x86(&mut data, false);
                assert_eq!(data, original, "len {len} opcode {opcode:#04x}");
            }
        }
    }

    #[test]
    fn encoding_actually_changes_the_displacement() {
        let mut data = vec![0xE8, 0x10, 0x20, 0x30, 0x40, 0xAA, 0xBB];
        let before = data.clone();
        x86(&mut data, true);
        assert_ne!(data, before, "filter must rewrite the displacement");
        // The opcode byte must survive untouched: that is what makes the walk
        // reversible.
        assert_eq!(data[0], 0xE8);
        // Little-endian so the low byte carries the added address.
        assert_eq!(data[1], 0x10u8.wrapping_add(5));
    }

    #[test]
    fn jcc_uses_a_two_byte_displacement_offset() {
        let mut data = vec![0x0F, 0x85, 0x00, 0x01, 0x00, 0x00, 0xAA];
        x86(&mut data, true);
        assert_eq!(data[0], 0x0F);
        assert_eq!(data[1], 0x85);
        // 0x0100 + 6 = 0x0106
        assert_eq!(data[2], 0x06);
        assert_eq!(data[3], 0x01);
    }

    /// Guards against writing the same code on both paths, which would pass a
    /// round-trip test while corrupting every archive.
    #[test]
    fn applying_the_filter_twice_is_not_the_identity() {
        let mut data = vec![0xE8, 0x10, 0x20, 0x30, 0x40];
        x86(&mut data, true);
        let once = data.clone();
        x86(&mut data, true);
        assert_ne!(
            data, once,
            "two encodes must differ; encode and decode are not the same code"
        );
    }

    /// The opcode set is narrow. `E7`, for instance, is a different instruction
    /// and must be left alone.
    #[test]
    fn leaves_other_opcodes_alone() {
        for opcode in [0x00u8, 0x01, 0x50, 0x88, 0xE7, 0xEA, 0xF0] {
            let mut data = vec![opcode, 0x11, 0x22, 0x33, 0x44, 0x55];
            let original = data.clone();
            x86(&mut data, true);
            assert_eq!(data, original, "opcode {opcode:#04x} must not be touched");
        }
    }

    /// `0F 8x` only when the low nibble group is 8; `0F 90` and `0F 00` are not
    /// near Jcc.
    #[test]
    fn two_byte_opcodes_must_be_jcc() {
        let mut data = vec![0x0F, 0x90, 0x11, 0x22, 0x33, 0x44];
        let original = data.clone();
        x86(&mut data, true);
        assert_eq!(data, original, "0F 90 is SETO, not Jcc");
    }

    #[test]
    fn real_pe_bytes_round_trip() {
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(path) = std::env::var("DENSR_TEST_PE") {
            candidates.push(path.into());
        }
        candidates.push("mingw64/bin/gcc.exe".into());
        // The test binary itself is a real PE on Windows, so the test still
        // runs on a clean checkout with no corpus present.
        #[cfg(windows)]
        if let Ok(exe) = std::env::current_exe() {
            candidates.push(exe);
        }
        let mut found = None;
        for c in candidates.iter() {
            if c.exists() {
                found = Some(std::fs::read(c).expect("read pe"));
                break;
            }
        }
        let Some(mut data) = found else {
            eprintln!("skipping: no PE fixture found (set DENSR_TEST_PE)");
            return;
        };
        let original_len = data.len();
        let original = data.clone();
        x86(&mut data, true);
        assert_ne!(data, original, "the filter must change real PE bytes");
        x86(&mut data, false);
        assert_eq!(data.len(), original_len);
        assert_eq!(data, original, "real PE bytes must round trip");
    }
}
