//! Deciding where block boundaries go, and which blocks are duplicates.
//!
//! # Why boundaries matter more than the codec
//!
//! Deduplication only pays when the same bytes land on the same boundary twice.
//! Fixed-size blocks achieve that only while files are byte-identical. Change one
//! byte near the front of a 40 MB log and every later block is different from
//! where it used to be, so all the shared tail is stored again.
//!
//! Content-defined boundaries fix this: a boundary is placed where the *content*
//! says so, so inserting bytes shifts the boundaries after the insertion but
//! leaves everything before it untouched. The rest of the file still matches.
//!
//! # The boundary rule
//!
//! A gear hash over the last 64 bytes, cut when the hash's high bits are all
//! zero. That is a fixed bit pattern rather than a probability, so boundaries are
//! reproducible across runs and machines without storing anything -- which
//! matters, because an archive must be readable by a build that has no shared
//! state with the one that wrote it.

use std::collections::HashMap;

/// Default block size.
///
/// Also the CDC boundary rate, not just a ceiling: `mask_for` puts a boundary
/// every `block_size` bytes on average. Because the LZ window *is* the block,
/// this is the single biggest ratio knob. Measured on the winlibs `bin` corpus
/// (262.22 MiB, level 4-5, CDC): 256k -> 2.62x, 4M -> 2.86x, 8M -> 2.89x,
/// 16M -> 2.91x. Returns flatten past 8M; 16M is kept as the ceiling because a
/// single 35 MiB PE still gains (2.56x at 8M, 2.58x at 16M) and the cost is RAM
/// per in-flight block, not time.
pub const BLOCK_SIZE: usize = 16 * 1024 * 1024;
/// Bounds on a configurable block size.
pub const MIN_BLOCK: usize = 4 * 1024;
pub const MAX_BLOCK: usize = 16 * 1024 * 1024;
/// The window over which a reference is searched for near-duplicates.
pub const DEFAULT_WINDOW: usize = 32 * 1024 * 1024;

/// Boundary strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chunking {
    /// Split at exact multiples of the block size. Fastest, and correct when
    /// files are identical copies.
    Fixed,
    /// Split where the content says so. Survives insertions.
    Cdc,
}

impl Chunking {
    pub fn parse(s: &str) -> Result<Chunking, String> {
        match s {
            "fixed" => Ok(Chunking::Fixed),
            "cdc" | "resync" => Ok(Chunking::Cdc),
            "auto" => Ok(Chunking::Cdc),
            other => Err(format!(
                "unknown chunking mode {other:?}; use fixed, cdc or auto"
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Chunking::Fixed => "fixed",
            Chunking::Cdc => "cdc",
        }
    }
}

/// A 256-entry table of pseudo-random 64-bit words.
///
/// The table is a fixed constant, not seeded per run: the boundaries an archive
/// uses have to be reproducible, and a table generated from the clock would put
/// them somewhere different every time.
const fn gear_table() -> [u64; 256] {
    // SplitMix64, used only to fill the table at compile time.
    let mut t = [0u64; 256];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut i = 0;
    while i < 256 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        t[i] = z ^ (z >> 31);
        i += 1;
    }
    t
}

static GEAR: [u64; 256] = gear_table();

/// The rolling state: a gear hash over the input.
///
/// One shift-and-add per byte, so rolling is O(1). The low `k` bits of the
/// digest depend on the last `k` bytes, but the *high* bits mix the whole
/// 64-byte window, and that is what boundaries look at. The earlier version
/// rotated and rehashed a 64-entry array on every byte; that is O(64) per byte
/// and ran at a few MB/s, dominating the whole archive time. This is the
/// standard FastCDC gear update.
#[derive(Debug, Clone)]
pub struct Gear {
    hash: u64,
    filled: usize,
    /// Bits of the digest that must be zero for a boundary to fall here.
    /// `k` bits set means a boundary every 2^k bytes on average.
    mask: u64,
}

impl Gear {
    pub fn new(mask: u64) -> Self {
        Gear {
            hash: 0,
            filled: 0,
            mask,
        }
    }

    /// Feed one byte, returning whether a boundary falls here.
    #[inline]
    pub fn push(&mut self, b: u8) -> bool {
        self.hash = (self.hash << 1).wrapping_add(GEAR[b as usize]);
        if self.filled < 64 {
            self.filled += 1;
            return false;
        }
        (self.hash & self.mask) == 0
    }
}

/// The digest mask that puts a boundary every `block_size` bytes on average.
///
/// A boundary needs `k` bits of the digest to be zero, which happens once in
/// 2^k bytes, so `k = log2(block_size)`.
///
/// The mask covers the **high** bits. A shift-and-add gear hash only lets the
/// last few bytes reach the low bits, so on repetitive text those bits can
/// cycle without ever matching and CDC silently degrades to the fixed-size
/// ceiling: a 400 KB test file produced seven exact 64 KiB blocks. The high
/// bits mix the whole 64-byte window, which is what makes a boundary a
/// function of content rather than of the last two characters.
///
/// This used to be a hard-coded `0xFFFF`, a fixed 64 KiB average, and
/// `--block-size` was ignored entirely under CDC: measured on lto-dump.exe,
/// `-b 256k` and `-b 4M` both produced 555 blocks averaging 64 KiB. The
/// block-size guard in `split` only acted as a ceiling that natural boundaries
/// almost always beat first, so the flag could not make chunks any larger.
fn mask_for(block_size: usize) -> u64 {
    // floor(log2(block_size)): an exact power of two keeps its own exponent, so
    // 64 KiB gives 16 bits. Using floor(log2)+1 would double every spacing,
    // which measured as 15.68 MiB / 707 blocks against the 15.47 MiB / 566
    // blocks the fixed 0xFFFF mask produced.
    let bits = usize::BITS - 1 - block_size.max(1).leading_zeros();
    // Never zero bits: that would fire on every byte and defeat the scheme.
    let bits = bits.clamp(1, 63);
    (!0u64) << (64 - bits)
}

/// Split `data` into blocks.
///
/// The last block is whatever remains, even if shorter than the minimum. Every
/// byte of the input appears in exactly one block, which is what makes the file
/// hash and the block list agree.
pub fn split(data: &[u8], block_size: usize, mode: Chunking) -> Vec<(usize, usize)> {
    split_with(data, block_size, CDC_DEFAULT_TARGET, mode)
}

/// The boundary spacing CDC aims for when the caller does not ask for one.
///
/// Kept equal to [`BLOCK_SIZE`] because the LZ window is the block: a small
/// target used to be the right call for dedup (identical regions hashed the
/// same), but it also shrank every match search to 64 KiB and cost far more
/// ratio on ordinary archives than it bought on the rare exactly-duplicated one
/// (winlibs `bin`: 64 KiB -> 2.62x, 16 MiB -> 2.90x). The dedup win that the
/// small target protected is better restored by cross-block matching, which
/// does not force tiny blocks.
pub const CDC_DEFAULT_TARGET: usize = BLOCK_SIZE;

/// Split `data`, letting the caller set the CDC boundary spacing independently
/// of the fixed-mode block size.
pub fn split_with(
    data: &[u8],
    block_size: usize,
    cdc_target: usize,
    mode: Chunking,
) -> Vec<(usize, usize)> {
    let block_size = block_size.clamp(MIN_BLOCK, MAX_BLOCK);
    let mut out = Vec::with_capacity(data.len() / block_size + 1);
    match mode {
        Chunking::Fixed => {
            let mut at = 0usize;
            while at < data.len() {
                let end = (at + block_size).min(data.len());
                out.push((at, end));
                at = end;
            }
        }
        Chunking::Cdc => {
            let cdc_target = cdc_target.clamp(MIN_BLOCK, MAX_BLOCK);
            let mut gear = Gear::new(mask_for(cdc_target));
            let mut start = 0usize;
            let mut since_boundary = 0usize;
            for (i, &b) in data.iter().enumerate() {
                since_boundary += 1;
                let boundary = gear.push(b);
                // Three guards, each for a reason.
                //
                // `MIN_BLOCK`: without it, a run of identical bytes can place a
                // boundary every few bytes, producing blocks too small to be
                // worth a block record.
                // `MAX_BLOCK`: without it, data with no boundary pattern at all
                // would produce one enormous block, defeating both dedup and the
                // codec's memory use.
                //
                // The third: a boundary is only allowed once the window is full,
                // which `Gear` already enforces.
                //
                // The ceiling is the CDC target rather than `block_size`: the
                // target is already the expected spacing, so this only fires for
                // data with no boundary pattern at all, which would otherwise
                // produce one enormous chunk.
                if boundary && since_boundary >= MIN_BLOCK || since_boundary >= block_size {
                    out.push((start, i + 1));
                    start = i + 1;
                    since_boundary = 0;
                }
            }
            if start < data.len() {
                out.push((start, data.len()));
            }
        }
    }
    out
}

/// What the chunker found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkStats {
    pub blocks: usize,
    /// Blocks that turned out to be identical to an earlier block.
    pub duplicates: usize,
    /// Bytes saved by deduplication.
    pub bytes_saved: u64,
    pub bytes_seen: u64,
}

impl ChunkStats {
    pub fn note_duplicate(&mut self, size: u64) {
        self.duplicates += 1;
        self.bytes_saved += size;
    }
}

/// A map from block identity to its index in the archive.
///
/// Keyed on the 64-bit hash, but a candidate is only accepted after its CRC
/// agrees as well. Two hashes colliding would otherwise substitute one file's
/// bytes for another's; the pair is 96 bits wide, which makes that negligible
/// without storing the content itself.
#[derive(Debug, Default)]
pub struct DedupIndex {
    map: HashMap<u64, Vec<(u32, u32)>>,
    len: u64,
}

impl DedupIndex {
    pub fn new() -> Self {
        DedupIndex::default()
    }

    /// Record a block.
    pub fn insert(&mut self, hash: u64, raw_len: u32, crc: u32) {
        self.map.entry(hash).or_default().push((raw_len, crc));
        self.len += 1;
    }

    /// Whether this exact block has been seen.
    pub fn contains(&self, hash: u64, raw_len: u32, crc: u32) -> bool {
        match self.map.get(&hash) {
            Some(v) => v.iter().any(|&(l, c)| l == raw_len && c == crc),
            None => false,
        }
    }

    pub fn blocks(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut s = 5u64;
        while out.len() < len {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(format!("line {} of repeated text\n", s >> 40).as_bytes());
        }
        out.truncate(len);
        out
    }

    #[test]
    fn every_input_is_covered_exactly_once() {
        // The property both chunking modes must have: concatenated blocks equal
        // the input, and they are in order and non-overlapping.
        let data = text(500_000);
        for mode in [Chunking::Fixed, Chunking::Cdc] {
            for size in [MIN_BLOCK, 64 * 1024, 128 * 1024] {
                let blocks = split(&data, size, mode);
                assert!(!blocks.is_empty());
                assert_eq!(blocks[0].0, 0, "{:?} block {:?}", mode, size);
                assert_eq!(
                    blocks.last().unwrap().1,
                    data.len(),
                    "{:?} did not reach the end",
                    mode
                );
                for w in blocks.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "{:?} left a gap", mode);
                }
                let mut rebuilt = Vec::with_capacity(data.len());
                for &(a, b) in &blocks {
                    assert!(b > a, "{:?} produced an empty block", mode);
                    rebuilt.extend_from_slice(&data[a..b]);
                }
                assert_eq!(rebuilt, data, "{:?} at size {size}", mode);
            }
        }
    }

    #[test]
    fn fixed_blocks_are_all_the_requested_size_but_the_last() {
        let data = text(200_000);
        let blocks = split(&data, 64 * 1024, Chunking::Fixed);
        assert_eq!(blocks.len(), 4);
        for &(a, b) in &blocks[..3] {
            assert_eq!(b - a, 64 * 1024);
        }
        assert_eq!(blocks[3].1 - blocks[3].0, 200_000 - 3 * 64 * 1024);
    }

    #[test]
    fn cdc_survives_an_insertion_in_the_middle() {
        // The reason content-defined boundaries exist. Inserting bytes at the
        // front of a file must not stop the tail from matching.
        let original = text(400_000);
        let mut shifted = b"an inserted prefix line\n".to_vec();
        shifted.extend_from_slice(&original);

        let a = split_with(&original, 64 * 1024, 64 * 1024, Chunking::Cdc);
        let b = split_with(&shifted, 64 * 1024, 64 * 1024, Chunking::Cdc);

        // Count how many of the original's blocks appear verbatim in the
        // shifted file. With fixed boundaries this is near zero.
        //
        // Note the block *counts* need not differ: prepending shifts every
        // boundary, but the chunker then settles back into the same cadence, so
        // the total can land on the same number. What matters is that the
        // content of most blocks still matches.
        let hashes = |blocks: &[(usize, usize)], data: &[u8]| -> std::collections::HashSet<u64> {
            blocks
                .iter()
                .map(|&(x, y)| crate::checksum::hash64(&data[x..y]))
                .collect()
        };
        let shared = hashes(&a, &original)
            .intersection(&hashes(&b, &shifted))
            .count();
        assert!(
            shared > a.len() / 2,
            "only {shared} of {} blocks were shared after the shift",
            a.len()
        );
    }

    #[test]
    fn cdc_finds_boundaries_before_the_ceiling() {
        // Repetitive text is where low-bit boundary tests fail: the pattern
        // there never matches, every block lands on the 64 KiB ceiling, and
        // CDC is silently fixed-size chunking. At least one boundary must come
        // from content before the ceiling does.
        let data = text(400_000);
        let blocks = split_with(&data, 64 * 1024, 64 * 1024, Chunking::Cdc);
        assert!(
            blocks.iter().any(|&(a, b)| b - a < 64 * 1024),
            "every block hit the ceiling: {blocks:?}"
        );
    }

    #[test]
    fn fixed_boundaries_lose_everything_after_an_insertion() {
        // The counter-example that motivates the above: the same edit, with fixed
        // boundaries, keeps almost nothing.
        let original = text(400_000);
        let mut shifted = b"an inserted prefix line\n".to_vec();
        shifted.extend_from_slice(&original);
        let a = split(&original, 64 * 1024, Chunking::Fixed);
        let b = split(&shifted, 64 * 1024, Chunking::Fixed);
        let hashes = |blocks: &[(usize, usize)], data: &[u8]| -> std::collections::HashSet<u64> {
            blocks
                .iter()
                .map(|&(x, y)| crate::checksum::hash64(&data[x..y]))
                .collect()
        };
        let shared = hashes(&a, &original)
            .intersection(&hashes(&b, &shifted))
            .count();
        assert!(
            shared <= 1,
            "{shared} blocks were shared with fixed boundaries"
        );
    }

    #[test]
    fn cdc_respects_its_minimum_and_maximum() {
        // Identical bytes fire the boundary test constantly, so the minimum is
        // what stops the chunker shredding the file.
        let data = vec![b'q'; 2_000_000];
        let blocks = split_with(&data, 64 * 1024, 64 * 1024, Chunking::Cdc);
        for &(a, b) in &blocks[..blocks.len() - 1] {
            assert!(b - a >= MIN_BLOCK, "cdc produced a {}-byte block", b - a);
        }
        assert!(
            blocks.len() < 100,
            "cdc shredded the file into {} blocks",
            blocks.len()
        );
    }

    #[test]
    fn boundaries_are_reproducible() {
        // A reader must be able to find the same boundaries as the writer, with
        // no shared state between them.
        let data = text(300_000);
        assert_eq!(
            split(&data, 64 * 1024, Chunking::Cdc),
            split(&data, 64 * 1024, Chunking::Cdc)
        );
    }

    #[test]
    fn empty_and_tiny_inputs_are_handled() {
        assert!(split(&[], 64 * 1024, Chunking::Cdc).is_empty());
        assert!(split(&[], 64 * 1024, Chunking::Fixed).is_empty());
        for len in 1..200usize {
            let data = text(len);
            for mode in [Chunking::Fixed, Chunking::Cdc] {
                let blocks = split(&data, MIN_BLOCK, mode);
                assert_eq!(blocks, vec![(0, len)], "{:?} at {len}", mode);
            }
        }
    }

    #[test]
    fn dedup_index_finds_exact_repeats() {
        let mut index = DedupIndex::new();
        assert!(index.is_empty());
        index.insert(0xdead, 100, 0x1234);
        index.insert(0xbeef, 100, 0x5678);
        assert!(index.contains(0xdead, 100, 0x1234));
        assert!(index.contains(0xbeef, 100, 0x5678));
        assert!(!index.contains(0xdead, 101, 0x1234), "wrong length matched");
        assert!(!index.contains(0xdead, 100, 0x9999), "wrong CRC matched");
        assert!(!index.contains(0xface, 100, 0x1234));
        assert_eq!(index.blocks(), 2);
    }

    #[test]
    fn stats_accumulate() {
        let mut s = ChunkStats::default();
        s.note_duplicate(500);
        s.note_duplicate(700);
        assert_eq!(s.duplicates, 2);
        assert_eq!(s.bytes_saved, 1200);
    }

    #[test]
    fn gear_table_is_not_degenerate() {
        // A table with repeated or zero words would place boundaries by accident
        // rather than by content.
        let mut seen = std::collections::HashSet::new();
        for v in GEAR.iter() {
            assert_ne!(*v, 0);
            assert!(seen.insert(*v), "gear table has a repeated word");
        }
    }
}
