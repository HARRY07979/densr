//! Small helpers with no home of their own: integer encoding, byte order and
//! human-readable sizes.
//!
//! Everything here is used by both the writer and the reader, which is why it
//! sits below them in the dependency graph.

use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Integers on disk
// ---------------------------------------------------------------------------

/// Append `v` as LEB128: seven bits per byte, low group first, high bit set on
/// every byte but the last.
///
/// Small values cost one byte, and the values in a stream of offsets are
/// overwhelmingly small. A fixed-width field would pay the full width on every
/// one of them.
pub fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Read a LEB128 value.
///
/// Rejects a value whose tenth byte still carries bits: a `u64` cannot hold more,
/// so a longer encoding is malformed rather than merely large. Without this a
/// corrupt stream could shift the rest of the parse by an arbitrary amount.
pub fn read_varint(input: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut result = 0u64;
    let mut shift = 0u32;
    for _ in 0..10 {
        let b = read_u8(input, pos)?;
        let low = (b & 0x7F) as u64;
        if shift == 63 && low > 1 {
            return Err("varint overflows 64 bits".into());
        }
        result |= low << shift;
        if b & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
    Err("varint is longer than 10 bytes".into())
}

#[inline]
pub fn read_u8(input: &[u8], pos: &mut usize) -> Result<u8, String> {
    if *pos >= input.len() {
        return Err(format!(
            "unexpected end of data at offset {pos} of {}",
            input.len()
        ));
    }
    let b = input[*pos];
    *pos += 1;
    Ok(b)
}

/// Read a little-endian `u16`.
pub fn read_u16(input: &[u8], pos: &mut usize) -> Result<u16, String> {
    let b = slice(input, pos, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

/// Read a little-endian `u32`.
pub fn read_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
    let b = slice(input, pos, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Read a little-endian `u64`.
pub fn read_u64(input: &[u8], pos: &mut usize) -> Result<u64, String> {
    let b = slice(input, pos, 8)?;
    Ok(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

pub fn read_u128(input: &[u8], pos: &mut usize) -> Result<u128, String> {
    let b = slice(input, pos, 16)?;
    let mut buf = [0u8; 16];
    buf.copy_from_slice(b);
    Ok(u128::from_le_bytes(buf))
}

/// Borrow `n` bytes, advancing the cursor.
pub fn slice<'a>(input: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let end = pos
        .checked_add(n)
        .ok_or_else(|| format!("length {n} overflows"))?;
    if end > input.len() {
        return Err(format!(
            "want {n} bytes at offset {pos} but only {} remain",
            input.len() - *pos
        ));
    }
    let out = &input[*pos..end];
    *pos = end;
    Ok(out)
}

/// Seconds since the Unix epoch, saturating at zero if the clock is before it.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

/// A byte count in the largest unit that leaves a useful number.
pub fn human(bytes: f64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = bytes;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{:.0} {}", bytes, UNITS[0])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// A duration, in the largest unit that leaves two useful digits.
pub fn human_time(seconds: f64) -> String {
    if seconds < 1.0 {
        return format!("{:.0} ms", seconds * 1000.0);
    }
    if seconds < 60.0 {
        return format!("{seconds:.1} s");
    }
    let m = (seconds / 60.0).floor();
    let s = seconds - m * 60.0;
    if m < 60.0 {
        return format!("{m:.0} m {s:.0} s");
    }
    let h = (m / 60.0).floor();
    format!("{h:.0} h {:.0} m", m - h * 60.0)
}

/// An unambiguous modification time: UTC, ISO 8601.
///
/// Local time with a bare offset is ambiguous across daylight-saving changes,
/// and an archive that travels between machines should not depend on the
/// reader's locale or timezone at all.
pub fn format_time(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let secs_of_day = unix_secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60,
    )
}

/// Days since 1970-01-01 to a civil date.
///
/// Howard Hinnant's algorithm: shift the year so that the leap day lands at the
/// end of the cycle, then invert the day count. Chosen over a table of month
/// lengths because it is short enough to verify by eye and has no allocation.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrips_at_every_boundary() {
        for v in [
            0u64,
            1,
            126,
            127,
            128,
            129,
            16_382,
            16_383,
            16_384,
            1 << 31,
            (1 << 63) - 1,
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_varint(&buf, &mut pos).unwrap(), v, "value {v}");
            assert_eq!(pos, buf.len(), "value {v} left trailing bytes");
        }
    }

    #[test]
    fn varint_uses_the_fewest_bytes() {
        let mut buf = Vec::new();
        write_varint(&mut buf, 300);
        assert_eq!(buf, vec![0xAC, 0x02]);
        let mut pos = 0;
        assert_eq!(read_varint(&buf, &mut pos).unwrap(), 300);
    }

    #[test]
    fn varint_rejects_overlong_and_unterminated_input() {
        // Eleven continuation bytes cannot describe a u64.
        let mut pos = 0;
        assert!(read_varint(&[0xFF; 11], &mut pos).is_err());
        // A run that never terminates.
        let mut pos = 0;
        assert!(read_varint(&[0x80; 10], &mut pos).is_err());
        // Empty input.
        let mut pos = 0;
        assert!(read_varint(&[], &mut pos).is_err());
        // A tenth byte carrying more than the single remaining bit.
        let mut pos = 0;
        assert!(read_varint(&[0x80; 9 | 0x02], &mut pos).is_err());
    }

    #[test]
    fn fixed_width_readers_reject_truncation() {
        for n in 0..8usize {
            let buf = vec![1u8; n];
            let mut pos = 0;
            assert!(read_u64(&buf, &mut pos).is_err(), "{n} bytes");
        }
        let buf = vec![0x12u8, 0x34, 0x56, 0x78];
        let mut pos = 0;
        assert_eq!(read_u32(&buf, &mut pos).unwrap(), 0x7856_3412);
        assert_eq!(pos, 4);
    }

    #[test]
    fn human_picks_a_readable_unit() {
        assert_eq!(human(0.0), "0 B");
        assert_eq!(human(999.0), "999 B");
        assert_eq!(human(1024.0), "1.00 KiB");
        assert_eq!(human(1536.0), "1.50 KiB");
        assert_eq!(human(1024.0 * 1024.0), "1.00 MiB");
        assert_eq!(human(2.0 * 1024.0 * 1024.0 * 1024.0), "2.00 GiB");
    }

    #[test]
    fn human_time_picks_a_readable_unit() {
        assert_eq!(human_time(0.25), "250 ms");
        assert_eq!(human_time(5.0), "5.0 s");
        assert_eq!(human_time(90.0), "1 m 30 s");
        assert_eq!(human_time(3700.0), "1 h 1 m");
    }

    #[test]
    fn civil_dates_are_correct() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(1), (1970, 1, 2));
        // January has 31 days, so day 31 is 1 February.
        assert_eq!(civil_from_days(31), (1970, 2, 1));
        // A non-leap year.
        assert_eq!(civil_from_days(365), (1971, 1, 1));
        assert_eq!(civil_from_days(59), (1970, 3, 1));
        // 2024 is a leap year: 19723 days from the epoch is 2024-01-01, and 59
        // days later is 29 February.
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(19_783), (2024, 3, 1));
        // 2100 is not a leap year. A rule that adds a day every fourth year gets
        // this wrong; the century rule does not.
        assert_eq!(civil_from_days(47_482), (2100, 1, 1));
        assert_eq!(civil_from_days(47_540), (2100, 2, 28));
        assert_eq!(civil_from_days(47_541), (2100, 3, 1));
    }

    #[test]
    fn format_time_renders_utc() {
        assert_eq!(format_time(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(format_time(1_000_000_000), "2001-09-09 01:46:40 UTC");
    }
}
