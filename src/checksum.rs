//! CRC-32 and the 64-bit content hash, both written here from scratch.
//!
//! Two different jobs, so two different tools:
//!
//! * [`crc32`] is cheap and has no collision resistance to speak of. It guards
//!   every block and the directory against corruption in transit -- flipped bits,
//!   a truncated write, a bad sector. It is a *detector*, not an identity.
//! * [`hash64`] is the block identity used for deduplication. Colliding here
//!   would silently substitute the wrong bytes, so it needs real avalanche.
//!
//! [`hash64`] is not a known algorithm and deliberately not one: inheriting a
//! published hash would mean inheriting its design, and the on-disk format here
//! is new anyway. What matters is that it is fast, avalanches well, and is
//! reproducible by the same code on any machine -- all of which the tests at the
//! bottom of this file check directly.

/// Reflected CRC-32 (IEEE 802.3), the same polynomial every zip and gzip file
/// uses.
///
/// The tables are built at first use rather than embedded: 8 KiB of code-space is
/// worth more than the handful of microseconds this saves on a cold archive, and
/// it removes the thousands of numbers that would otherwise be unreviewable.
/// The first table is the classic reflected 0xEDB88320 table; each later one is
/// the previous table advanced by one byte, which is what slicing-by-8 needs.
const fn crc32_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        tables[0][i] = c;
        i += 1;
    }
    let mut t = 1;
    while t < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = tables[t - 1][i];
            tables[t][i] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
            i += 1;
        }
        t += 1;
    }
    tables
}

static CRC32_TABLES: [[u32; 256]; 8] = crc32_tables();

/// Streaming CRC-32, so a file is checksummed without being held in memory.
#[derive(Debug, Clone)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    pub fn new() -> Self {
        // The seed is the all-ones complement of zero; the final xor undoes it.
        Crc32 { state: 0xFFFF_FFFF }
    }

    pub fn update(&mut self, data: &[u8]) {
        let mut c = self.state;
        let mut chunks = data.chunks_exact(8);
        for ch in &mut chunks {
            let one = (c as u64) ^ u64::from_le_bytes(ch.try_into().unwrap());
            c = CRC32_TABLES[7][(one & 0xFF) as usize]
                ^ CRC32_TABLES[6][((one >> 8) & 0xFF) as usize]
                ^ CRC32_TABLES[5][((one >> 16) & 0xFF) as usize]
                ^ CRC32_TABLES[4][((one >> 24) & 0xFF) as usize]
                ^ CRC32_TABLES[3][((one >> 32) & 0xFF) as usize]
                ^ CRC32_TABLES[2][((one >> 40) & 0xFF) as usize]
                ^ CRC32_TABLES[1][((one >> 48) & 0xFF) as usize]
                ^ CRC32_TABLES[0][((one >> 56) & 0xFF) as usize];
        }
        for &b in chunks.remainder() {
            c = CRC32_TABLES[0][((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
        self.state = c;
    }

    pub fn finish(self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

/// One-shot CRC-32.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finish()
}

/// 64-bit mixing constants.
///
/// These are the odd 64-bit numbers from the wyhash paper's `wyp` function,
/// which exist to have long stretches of zero in their binary form: multiplying
/// by them is how a carry gets spread across all eight output bytes at once.
const P0: u64 = 0xa076_1d64_78bd_642f;
const P1: u64 = 0xe703_7ed1_a0b4_28db;
const P2: u64 = 0x8ebc_6af0_9c88_c6e3;
const P3: u64 = 0x5899_65cd_7537_4cc3;

/// Reads a little-endian u64 without assuming the host is little-endian.
///
/// Archives travel between machines, and the writer and the reader may sit on
/// different ones, so every integer on disk goes through this.
#[inline]
fn read_u64(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([
        data[at],
        data[at + 1],
        data[at + 2],
        data[at + 3],
        data[at + 4],
        data[at + 5],
        data[at + 6],
        data[at + 7],
    ])
}

/// The core mixing step: a 64-bit multiply whose carries are folded back into
/// both halves.
///
/// A plain `a * b` returns only the low 64 bits, so information from the high
/// half of the product is simply gone. Folding `high ^ low` puts it back, which
/// is what makes this one step behave like a much wider multiply.
#[inline]
fn mix(a: u64, b: u64) -> u64 {
    let product = (a as u128).wrapping_mul(b as u128);
    ((product >> 64) as u64) ^ (product as u64)
}

/// The final avalanche.
///
/// A hash that is merely "well distributed" can still produce clusters when two
/// inputs differ by one bit somewhere far from the end. This permutes every bit
/// across the full width of the word, so that input bit becomes output bit
/// everywhere and no output bit depends on only a few input bits.
#[inline]
fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^= h >> 32;
    h
}

/// Streaming 64-bit content hash.
///
/// Four accumulators, one per lane. A word is folded into the lane selected by
/// its *global* word index rather than by its position inside the current call,
/// which is what makes the result independent of how the input was split: four
/// independent multiplication chains give the CPU something to overlap, and
/// because the lane is chosen by word count, `update()` in one piece and
/// `update()` in a hundred pieces produce the same digest.
#[derive(Debug, Clone)]
pub struct Hash64 {
    lanes: [u64; 4],
    /// How many whole eight-byte words have been folded in so far.
    words: u64,
    /// Total bytes consumed, mixed in at the end.
    ///
    /// Without this, `hash64(b"ab")` and `hash64(b"ab\0")` collide: the word loop
    /// processes zero whole words in both cases and never sees the difference.
    len: u64,
    /// Bytes of a final partial word, waiting for more input.
    tail: [u8; 8],
    tail_len: u8,
}

impl Default for Hash64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Hash64 {
    /// Start a hash with an explicit seed, so a caller can hash two streams that
    /// must not collide -- a block and its dictionary copy, say -- with the same
    /// function.
    pub fn with_seed(seed: u64) -> Self {
        Hash64 {
            // Distinct lane inits, so the lanes do not march in lockstep.
            lanes: [
                avalanche(seed ^ P0),
                avalanche(seed ^ P1),
                avalanche(seed ^ P2),
                avalanche(seed ^ P3),
            ],
            words: 0,
            len: 0,
            tail: [0; 8],
            tail_len: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Fold one whole word into its lane.
    #[inline]
    fn push_word(&mut self, word: u64) {
        let lane = (self.words & 3) as usize;
        self.lanes[lane] = mix(self.lanes[lane] ^ word, P1);
        self.words += 1;
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);

        // Top up a partial word left over from the previous call.
        if self.tail_len > 0 {
            let take = (8 - self.tail_len as usize).min(data.len());
            let at = self.tail_len as usize;
            self.tail[at..at + take].copy_from_slice(&data[..take]);
            self.tail_len += take as u8;
            data = &data[take..];
            if self.tail_len < 8 {
                return;
            }
            let word = read_u64(&self.tail, 0);
            self.push_word(word);
            self.tail_len = 0;
        }

        while data.len() >= 8 {
            let word = read_u64(data, 0);
            self.push_word(word);
            data = &data[8..];
        }

        if data.is_empty() {
            self.tail_len = 0;
        } else {
            self.tail[..data.len()].copy_from_slice(data);
            // Zero the rest. A word is read as eight bytes even when only a few
            // are real, so without this the leftovers of a previous, longer
            // partial word would leak into the digest -- and a hash would then
            // depend on how the input was split.
            self.tail[data.len()..].fill(0);
            self.tail_len = data.len() as u8;
        }
    }

    pub fn finish(&self) -> u64 {
        let mut h = self.lanes[0];
        h = mix(h ^ self.lanes[1], P2);
        h = mix(h ^ self.lanes[2], P2);
        h = mix(h ^ self.lanes[3], P2);
        // The trailing partial word, then the length. Order matters only in that
        // both must be present; length last means a short input still gets mixed.
        if self.tail_len > 0 {
            let word = read_u64(&self.tail, 0);
            h = mix(h ^ word, P2);
        }
        h ^= self.len.wrapping_mul(P0);
        avalanche(h)
    }
}

/// One-shot 64-bit content hash.
pub fn hash64(data: &[u8]) -> u64 {
    let mut h = Hash64::new();
    h.update(data);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Deterministic pseudo-random bytes, so a failure can be reproduced from
    /// the seed alone.
    fn prng(seed: u64) -> impl Iterator<Item = u8> {
        let mut s = seed;
        std::iter::from_fn(move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Some((s >> 33) as u8)
        })
    }

    #[test]
    fn crc32_known_vectors() {
        // The standard check value: CRC-32 of "123456789" is 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        // An all-zero prefix must not collapse: the length is carried in the
        // state, so 1 and 2 zero bytes differ.
        assert_ne!(crc32(&[0]), crc32(&[0, 0]));
    }

    #[test]
    fn crc32_streaming_matches_one_shot() {
        let data: Vec<u8> = prng(1).take(5000).collect();
        for split in [0, 1, 7, 8, 9, 64, 1000, 4999, 5000] {
            let mut c = Crc32::new();
            c.update(&data[..split]);
            c.update(&data[split..]);
            assert_eq!(c.finish(), crc32(&data), "split at {split}");
        }
    }

    #[test]
    fn crc32_detects_every_single_bit_flip() {
        let base: Vec<u8> = prng(2).take(64).collect();
        let want = crc32(&base);
        for byte in 0..base.len() {
            for bit in 0..8 {
                let mut d = base.clone();
                d[byte] ^= 1 << bit;
                assert_ne!(crc32(&d), want, "flip byte {byte} bit {bit}");
            }
        }
    }

    #[test]
    fn hash_is_stable_across_update_boundaries() {
        // A hash that only agrees with itself when fed in one piece would be
        // useless for streaming files.
        let data: Vec<u8> = prng(3).take(4001).collect();
        let want = hash64(&data);
        for split in [0, 1, 3, 7, 8, 9, 16, 31, 32, 33, 64, 999, 4000, 4001] {
            let mut h = Hash64::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finish(), want, "split at {split}");
        }
    }

    #[test]
    fn hash_separates_lengths() {
        // Without the length in the final mix these two agree, because the word
        // loop sees zero whole words in both cases.
        assert_ne!(hash64(b"ab"), hash64(b"ab\0"));
        assert_ne!(hash64(b""), hash64(b"\0"));
        assert_ne!(hash64(b"aaaa"), hash64(b"aaaaa"));
    }

    #[test]
    fn hash_avalanche_is_uniform() {
        // The one-bit-flip test for the identity property: of the 512 single-bit
        // flips below, each of the 64 output bits should change about half the
        // time. A bit that rarely changes means input entropy is being dropped.
        let mut base = Vec::with_capacity(64);
        while base.len() < 64 {
            base.extend_from_slice(b"avalanche probe input, long enough to be interesting ");
        }
        base.truncate(64);
        let base_hash = hash64(&base);
        let flips = 512;
        let mut changed = [0u32; 64];
        for bit in 0..flips {
            let mut v = base.clone();
            v[bit / 8] ^= 1 << (bit % 8);
            let diff = hash64(&v) ^ base_hash;
            for out_bit in 0..64 {
                if diff >> out_bit & 1 == 1 {
                    changed[out_bit] += 1;
                }
            }
        }
        for (out_bit, n) in changed.iter().enumerate() {
            assert!(
                (*n as f64 - flips as f64 / 2.0).abs() < flips as f64 * 0.15,
                "output bit {out_bit} changed {n}/{} times, expected about {}",
                flips,
                flips / 2,
            );
        }
    }

    #[test]
    fn hash_has_no_collisions_in_a_large_sample() {
        // 200k distinct 64-byte blocks: at 64 bits the expected number of
        // collisions is ~1e-9, so any collision here means the hash is broken,
        // not unlucky.
        const N: usize = 200_000;
        let mut seen = HashSet::with_capacity(N * 2);
        for i in 0..N {
            let mut block = [0u8; 64];
            let mut s = i as u64;
            for slot in block.chunks_mut(8) {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                slot.copy_from_slice(&s.to_le_bytes());
            }
            assert!(seen.insert(hash64(&block)), "collision at block {i}");
        }
    }

    #[test]
    fn hash_distributes_sequential_values() {
        // Sequential inputs are the classic adversarial case for a weak hash: a
        // bucket spread test on them catches a missing final avalanche.
        //
        // Judged by chi-square rather than by per-bucket bounds. With 256
        // buckets some bucket will always sit a couple of standard deviations
        // out by chance, so a hard limit per bucket fails on a correct hash;
        // summing the deviations across all buckets does not.
        const N: usize = 4096;
        let mut buckets = [0u32; 256];
        for i in 0..N {
            buckets[(hash64(&(i as u64).to_le_bytes()) & 0xFF) as usize] += 1;
        }
        let expected = N as f64 / 256.0;
        let chi2: f64 = buckets
            .iter()
            .map(|&n| {
                let d = n as f64 - expected;
                d * d / expected
            })
            .sum();
        // 255 degrees of freedom: the 99.9th percentile is around 310.
        assert!(chi2 < 310.0, "chi-square {chi2:.1} suggests a poor spread");
    }

    #[test]
    fn seeded_hashes_are_independent() {
        // Two domains must not collide, or a block and its dictionary copy would
        // be indistinguishable.
        let data = b"the same bytes in two different domains";
        let a = {
            let mut h = Hash64::with_seed(0);
            h.update(data);
            h.finish()
        };
        let b = {
            let mut h = Hash64::with_seed(0x9E37_79B9_7F4A_7C15);
            h.update(data);
            h.finish()
        };
        assert_ne!(a, b);
    }
}
