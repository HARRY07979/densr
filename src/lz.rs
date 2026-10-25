//! `dsr1`: the compression codec, written here from scratch.
//!
//! # What kind of compressor this is
//!
//! An LZ77 variant in the LZ4/snappy family: a hash table finds byte sequences
//! that already appeared earlier in the block, and each one is replaced by a
//! back-reference. Every block compresses and decompresses on its own, which is
//! what makes block-level deduplication meaningful -- see [`crate::chunking`].
//!
//! The target is deliberately "balanced": roughly an order of magnitude faster
//! than a general-purpose compressor, at a compression ratio in the same
//! neighbourhood. This codec does not try to reach zstd's ratio, because the
//! ratio there comes from entropy coding and optimal parsing, both of which cost
//! far more time than an archiver should spend.
//!
//! # The format
//!
//! A coded block is a flat sequence of *segments*. There is no end-of-block
//! marker and no back-reference may cross the end of the input, so decoding
//! stops exactly when the expected output length is reached.
//!
//! ```text
//! segment := literal-run | match
//!
//! literal-run := 0x00 | u8[0x0F], u8[len] bytes
//!     The whole token byte is the literal length, 0-15. A value of 15 means a
//!     varint holding length - 15 follows. Every literal-run token has a high
//!     nibble of zero.
//!
//! match := 0x10 | u8[0xF0], varint offset, u8[match_len - MIN_MATCH] bytes
//!     The high nibble is the low four bits of the offset, the low nibble is the
//!     match length minus MIN_MATCH. A nibble of 15 means a varint holding the
//!     overflow follows, for the offset or the length.
//!
//! varint := u8..     LEB128, little-endian base 128, 7 bits per byte.
//! ```
//!
//! ## Why the two segment kinds cannot be confused
//!
//! A literal length and a match length both want to be four bits, so something
//! has to break the tie. It is the high nibble: a match's offset is never zero,
//! so the high nibble of a match token is always at least one, and a literal
//! run's is always zero. The two token ranges are therefore disjoint and the
//! decoder never has to guess.
//!
//! Spending a nibble of every literal token on that distinction is cheap -- a
//! literal costs one byte plus its content, so the marker is at most a
//! percent or two of the output -- and it is what lets the match token carry
//! both a length and offset bits.
//!
//! ## Why offsets are LEB128 rather than fixed 16-bit
//!
//! LZ4's 16-bit offset caps the window at 64 KiB, which is why long-range
//! matches are out of reach for it. A varint costs one byte for offsets under
//! 128, two under 16 KiB, and four for anything up to 4 GiB, so a large window
//! costs almost nothing on the small offsets that dominate real text and buys
//! long-range matches when the data warrants it.
//!
//! # Level
//!
//! Level selects search effort, not a different algorithm: how many hash chains
//! are walked per position, how deep the search may go, and how far ahead a
//! pending match may be extended by a better one. Level 0 stores the block
//! uncompressed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::util::{read_varint, write_varint};

/// The shortest run worth encoding as a reference.
///
/// Below this, a back-reference plus its offset and length bytes costs more than
/// the literals it replaces. Four is the usual figure: it pays for itself almost
/// immediately on real data.
pub const MIN_MATCH: usize = 4;

/// The longest run the 4-bit length nibble can express on its own.
const NIBBLE_MAX: usize = 15;

/// Longest match the search will consider. Matches longer than this are
/// representable -- the varint extension has no upper bound short of the block
/// size -- but finding them exactly is not worth the search cost.
pub const MAX_MATCH: usize = 512;

/// A match this long ends the chain walk early.
///
/// The chain is walked to find the *best* match at a position, but each step is
/// a random jump into a chain array several times the size of L2, so every step
/// is likely a cache miss. Once a match this long is in hand, the extra length a
/// few more candidates could add is not worth another handful of misses: this is
/// the classic "nice length" cutoff. It is the same at every level, so raising
/// `max_chain` still lets higher levels search further for a *longer* match --
/// the cutoff is a floor on quality, not a cap on effort.
const NICE_MATCH: usize = 64;

/// How far back a reference may point, capped by the format so that an offset
/// always fits in the varint extension of the token stream.
///
/// 4 GiB is the natural limit of the 32-bit space the container allocates for
/// offsets; in practice the caller's window setting is far smaller.
pub const MAX_OFFSET: usize = u32::MAX as usize;

/// Compress `data` at the given effort level.
///
/// A level of 0 (or any level that fails to beat the input) returns `None`,
/// leaving the caller to store the block verbatim. Deciding that here means the
/// writer never has to compare sizes itself.
pub fn compress(data: &[u8], level: u8) -> Option<Vec<u8>> {
    if data.len() < MIN_MATCH * 2 {
        return None;
    }
    let effort = Effort::for_level(level);
    let mut out = Vec::with_capacity(data.len() / 2 + 32);
    compress_into(data, effort, &mut out)?;
    Some(out)
}

/// Compress `data` with a shared dictionary in front of it.
///
/// The dictionary is not part of the output: it is seeded into the match table
/// so a match may point back into it, and the decoder is expected to have the
/// same bytes available in its window. Offsets are therefore measured from the
/// current position across the dictionary boundary, exactly as they are inside
/// one block, which is why no new token and no format change is needed here.
///
/// Returns `None` when the coded form would not be smaller than `data`.
pub fn compress_with_dict(data: &[u8], dict: &[u8], level: u8) -> Option<Vec<u8>> {
    if data.len() < MIN_MATCH * 2 {
        return None;
    }
    let seed = DictSeed::new(Arc::new(dict.to_vec()));
    compress_with_dict_seed(data, &seed, level)
}

/// A dictionary index that can be shared by every block of one archive.
///
/// The hash table and chain heads describing the dictionary depend only on the
/// dictionary bytes and the table's slot count, not on the block being coded.
/// Building them once per archive and copying them per block removes an
/// O(dictionary length) rescan that otherwise dominates archives made of many
/// small files.
///
/// The slot count follows the same rule as [`MatchFinder`] and depends on the
/// total (dictionary + block) length, so one table is cached per slot count.
/// There are only a handful of distinct counts (powers of two, clamped), and
/// each entry is immutable once built.
pub struct DictSeed {
    dict: Arc<Vec<u8>>,
    tables: Mutex<HashMap<usize, Arc<SeedTable>>>,
}

struct SeedTable {
    head: Vec<u32>,
    chain: Vec<u32>,
    /// Positions indexed here. The last `MIN_MATCH - 1` positions of the
    /// dictionary are *not* included because their four-byte key straddles the
    /// dictionary boundary into the block, so only the block can finish them.
    indexed: usize,
}

impl DictSeed {
    pub fn new(dict: Arc<Vec<u8>>) -> Self {
        DictSeed {
            dict,
            tables: Mutex::new(HashMap::new()),
        }
    }

    /// The dictionary bytes this seed indexes.
    pub fn dict(&self) -> &[u8] {
        &self.dict
    }

    fn table(&self, total_len: usize) -> Arc<SeedTable> {
        let slots = table_slots(total_len);
        let mut tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = tables.get(&slots) {
            return Arc::clone(t);
        }

        let dict = &self.dict;
        // A position is indexable on its own only if its whole four-byte key is
        // inside the dictionary. The few that straddle the boundary are indexed
        // per block, in order, exactly where the old per-block scan did it.
        let indexed = dict.len().saturating_sub(MIN_MATCH - 1);
        let mask = slots - 1;
        let mut head = vec![NONE; slots];
        let mut chain = vec![NONE; indexed];
        for (i, link) in chain.iter_mut().enumerate() {
            let slot = hash_key(key_at(dict, i)) & mask;
            *link = head[slot];
            head[slot] = i as u32;
        }
        let t = Arc::new(SeedTable {
            head,
            chain,
            indexed,
        });
        tables.insert(slots, Arc::clone(&t));
        t
    }
}

/// Window length whose slot count a dictionary-backed block should use.
///
/// The dictionary is part of the search window, so its bytes get slots too:
/// that keeps dictionary chains short and measurable on `include` (3.21 vs
/// 3.37 MiB with a capped table). The per-block head copy this costs is only
/// material on trees of tiny files, where the win in ratio is larger than the
/// win in time and the block count makes the copy visible.
#[inline]
fn seeded_table_len(dict_len: usize, raw_len: usize) -> usize {
    dict_len + raw_len
}

/// Slot count for a match table covering `total_len` bytes.
///
/// One slot per four bytes keeps chains short; the power-of-two clamp bounds
/// memory on large blocks and the minimum keeps tiny ones from colliding.
fn table_slots(total_len: usize) -> usize {
    (total_len / 4).next_power_of_two().clamp(1 << 10, 1 << 21)
}

/// [`compress_with_dict`], reusing a [`DictSeed`] built once per archive.
pub fn compress_with_dict_seed(data: &[u8], seed: &DictSeed, level: u8) -> Option<Vec<u8>> {
    if data.len() < MIN_MATCH * 2 {
        return None;
    }
    let effort = Effort::for_level(level);
    let mut out = Vec::with_capacity(data.len() / 2 + 32);
    compress_seeded(data, seed.dict(), Some(seed), effort, &mut out)?;
    Some(out)
}

/// Compress into a caller-provided buffer, returning `None` if the result would
/// not be smaller than the input.
pub fn compress_into(data: &[u8], effort: Effort, out: &mut Vec<u8>) -> Option<()> {
    compress_seeded(data, &[], None, effort, out)
}

/// [`compress_into`], with a shared dictionary seeded in front of the data.
pub fn compress_into_with_dict(
    data: &[u8],
    dict: &[u8],
    effort: Effort,
    out: &mut Vec<u8>,
) -> Option<()> {
    compress_seeded(data, dict, None, effort, out)
}

/// The one parse loop, optionally seeded with a dictionary.
///
/// The dictionary is prepended to a scratch buffer and the finder is seeded with
/// it, so the parse loop itself is unchanged: offsets that reach into the
/// dictionary fall out of the existing `pos - c` arithmetic, and the literal and
/// match emitters see one contiguous buffer.
fn compress_seeded(
    data: &[u8],
    dict: &[u8],
    seed: Option<&DictSeed>,
    effort: Effort,
    out: &mut Vec<u8>,
) -> Option<()> {
    let start = out.len();
    if effort.max_chain == 0 || data.len() < MIN_MATCH * 2 {
        return None;
    }
    let mut buf = Vec::with_capacity(dict.len() + data.len());
    buf.extend_from_slice(dict);
    buf.extend_from_slice(data);
    let data = buf.as_slice();
    // The size guard is against the *block*, not the scratch buffer: the
    // dictionary is not stored in this block's output.
    let raw_len = data.len() - dict.len();

    // Build or borrow the dictionary's chain index before any position of the
    // block is examined, so a first-position match can already reach into it.
    // The base table is immutable and shared; only the block's own positions,
    // the head overlay and at most three boundary-straddling dictionary links
    // are per block.
    let table = seed.map(|seed| {
        debug_assert_eq!(seed.dict(), dict);
        // A large block amortises a large table; a small one does not. Using the
        // dictionary length in the slot count gives the dictionary a sparse
        // table, but for a 4 KiB block in front of a 1 MiB dictionary that means
        // copying a megabyte of head slots per block. Cap small blocks and let
        // large ones (where the copy is negligible) keep the extra slots.
        let table_len = seeded_table_len(dict.len(), raw_len);
        seed.table(table_len)
    });
    let mut finder = match table.as_deref() {
        Some(t) => MatchFinder::new_seeded(data, effort, t, dict.len()),
        None if dict.is_empty() => MatchFinder::new(data, effort),
        // A fresh scan over a dictionary must use the same capped slot count as
        // the seeded path, or the two parses could choose different
        // equal-length matches and stop being byte identical. The cap avoids a
        // large table (and the per-block copy that comes with it) when a 1 MiB
        // dictionary fronts a small block.
        None => MatchFinder::with_slots(
            data,
            effort,
            table_slots(seeded_table_len(dict.len(), raw_len)),
        ),
    };
    let indexed = table.as_ref().map_or(0, |t| t.indexed);
    for i in indexed..dict.len() {
        finder.index(i);
    }
    finder.indexed = dict.len();
    // The parse starts *after* the dictionary: those bytes are history, not
    // output, so emitting them as literals would store the dictionary twice.
    finder.cursor = dict.len();
    let mut anchor = dict.len();
    // Lazy parsing: a match at `pos` is held as pending while the next
    // position is examined. If the next position has a strictly better match
    // (more bytes saved), the pending one is abandoned in its favour --
    // the classic fix for greedy-parse wobble. Same token format, so the
    // decoder and the container format are untouched.
    let mut pending: Option<(usize, Match)> = None;
    // The offsets of the last two emitted matches, so a match repeating either
    // can omit the offset entirely: token `0x00` plus one varint packing the
    // length and which slot was used.
    let mut last_offset = 0usize;
    let mut prev_offset = 0usize;

    while let Some(pos) = finder.next() {
        // Search BEFORE this position is indexed: the chain must not start by
        // pointing at the position being searched (see MatchFinder::next).
        let found = if pos < anchor {
            // A position inside a reference already emitted cannot start
            // another one.
            None
        } else {
            finder.find(pos).filter(|m| {
                // A match must pay for itself: it costs a token plus an offset
                // varint, and every literal it removes costs one byte.
                let cost = 1 + varint_len(m.offset as u64) + extension_len(m.len);
                cost < m.len
            })
        };

        // Index everything decided so far. The interior of a reference is
        // included deliberately: those bytes are gone from the output, but a
        // later match still has to be able to point at them, and skipping them
        // would cost most of the ratio on repetitive input.
        finder.index_through(usize::MAX);

        if pos < anchor {
            // Unreachable: the anchor only advances by consuming `pending`,
            // so a live pending match always starts at or after it.
            debug_assert!(pending.is_none());
            continue;
        }

        if !effort.lazy {
            let Some(m) = found else {
                continue;
            };
            // Not worth it: the anchor stays put, so this byte simply becomes
            // part of the next literal run -- emitting it as a one-byte run
            // would spend two bytes to store one.
            emit_literals(out, data, anchor, pos);
            emit_match(out, m, &mut last_offset, &mut prev_offset);
            anchor = pos + m.len;
            continue;
        }

        match (pending.take(), found) {
            (None, None) => {}
            (None, Some(m)) => {
                pending = Some((pos, m));
            }
            (Some((ppos, pm)), None) => {
                // No challenger: the pending match stands.
                emit_literals(out, data, anchor, ppos);
                emit_match(out, pm, &mut last_offset, &mut prev_offset);
                anchor = ppos + pm.len;
            }
            (Some((ppos, pm)), Some(m)) if pos == ppos + 1 => {
                if saving(&m) > saving(&pm) {
                    // The next match saves strictly more: abandon pending.
                    // Its first byte becomes literal material via the anchor.
                    pending = Some((pos, m));
                } else {
                    emit_literals(out, data, anchor, ppos);
                    emit_match(out, pm, &mut last_offset, &mut prev_offset);
                    anchor = ppos + pm.len;
                    // `m` starts inside the emitted match and is void.
                }
            }
            (Some((ppos, pm)), Some(_)) => {
                // Non-consecutive (can only happen right after an anchor jump):
                // keep the earlier pending match.
                emit_literals(out, data, anchor, ppos);
                emit_match(out, pm, &mut last_offset, &mut prev_offset);
                anchor = ppos + pm.len;
            }
        }
    }

    if let Some((ppos, pm)) = pending {
        emit_literals(out, data, anchor, ppos);
        emit_match(out, pm, &mut last_offset, &mut prev_offset);
        anchor = ppos + pm.len;
    }

    emit_literals(out, data, anchor, data.len());

    // The caller's buffer may already have held a prefix; only judge what we
    // added.
    if out.len() - start >= raw_len {
        out.truncate(start);
        return None;
    }
    Some(())
}

/// Decompress exactly `expected_len` bytes from `input`.
///
/// `expected_len` is part of the block record, not of the payload: the format has
/// no end-of-block marker, so the decoder is told how much to produce and stops
/// there. That check is also what stops a corrupt or hostile block from
/// allocating without bound.
pub fn decompress(input: &[u8], expected_len: usize) -> Result<Vec<u8>, String> {
    decompress_windowed(input, &[], expected_len)
}

/// [`decompress`], with the shared dictionary the block was encoded against.
///
/// The dictionary is loaded into the output window first and stripped off at the
/// end, so a match pointing into it is an ordinary back-reference: the offset
/// bounds check, the overlapping-run copy and the "no end marker" length check
/// all behave exactly as they do inside one block. The dictionary bytes must be
/// the ones the encoder seeded -- they are the same bytes the archive stores,
/// and the caller checks them against the recorded hash.
pub fn decompress_with_dict(
    input: &[u8],
    dict: &[u8],
    expected_len: usize,
) -> Result<Vec<u8>, String> {
    decompress_windowed(input, dict, expected_len)
}

/// Copy `len` bytes ending `offset` back from the current end of `out`.
///
/// Overlapping copies -- offset smaller than length -- are legal and common:
/// that is how runs are encoded. The run is the `offset` bytes before it,
/// repeated. Copy one period, then double by copying what has already been
/// produced. `extend_from_within` appends inside the same allocation, so unlike
/// a resize-then-copy it neither zero-fills the destination nor reallocates on
/// every step.
fn copy_match(
    out: &mut Vec<u8>,
    offset: usize,
    len: usize,
    expected_len: usize,
) -> Result<(), String> {
    if offset == 0 {
        return Err("match with zero offset".into());
    }
    if offset > out.len() {
        return Err(format!(
            "match offset {offset} reaches before the start of the block"
        ));
    }
    if len > expected_len - out.len() {
        return Err(format!(
            "match of {len} bytes overruns the {expected_len}-byte block"
        ));
    }
    let base = out.len();
    let first = len.min(offset);
    out.extend_from_within(base - offset..base - offset + first);
    let mut produced = first;
    while produced < len {
        let n = (len - produced).min(produced);
        out.extend_from_within(base..base + n);
        produced += n;
    }
    Ok(())
}

/// Read a varint that extends a four-bit nibble, rejecting values that cannot
/// be represented as a host `usize` or that overflow when added to `base`.
///
/// A hostile stream can carry up to `u64::MAX` here. Doing the addition in
/// release-mode wrapping arithmetic would turn an enormous length into a tiny
/// one and decode the wrong bytes; in debug it would panic.
fn extended_value(input: &[u8], pos: &mut usize, base: usize, what: &str) -> Result<usize, String> {
    let value = read_varint(input, pos)?;
    let value = usize::try_from(value).map_err(|_| format!("{what} does not fit in memory"))?;
    base.checked_add(value)
        .ok_or_else(|| format!("{what} overflows"))
}

fn decompress_windowed(input: &[u8], dict: &[u8], expected_len: usize) -> Result<Vec<u8>, String> {
    let dict_len = dict.len();
    // The loop below measures progress against `out.len()`, so the window's
    // budget is the dictionary plus the block.
    let window_len = dict_len
        .checked_add(expected_len)
        .ok_or_else(|| "block length overflows".to_string())?;
    let mut out = Vec::with_capacity(window_len.min(64 << 20));
    out.extend_from_slice(dict);
    let expected_len = window_len;
    let mut pos = 0usize;
    let mut last_offset = 0usize;
    let mut prev_offset = 0usize;

    while out.len() < expected_len {
        let token = read_byte(input, &mut pos)?;
        let nibble = (token >> 4) as usize;
        let low = (token & 0x0F) as usize;

        if nibble == 0 && low == 0 {
            // Repeat-offset match: one of the previous two offsets, with the
            // length and the slot packed into a single varint. A literal token
            // always has a non-zero length nibble, so the zero byte is
            // unambiguous.
            let code = usize::try_from(read_varint(input, &mut pos)?)
                .map_err(|_| "repeat-offset code does not fit in memory".to_string())?;
            let which = code & 1;
            let len = MIN_MATCH
                .checked_add(code >> 1)
                .ok_or_else(|| "repeat-offset match length overflows".to_string())?;
            let offset = if which == 0 { last_offset } else { prev_offset };
            if offset == 0 {
                return Err("repeat-offset match before any offset was set".into());
            }
            copy_match(&mut out, offset, len, expected_len)?;
            if which == 1 {
                std::mem::swap(&mut last_offset, &mut prev_offset);
            } else {
                prev_offset = last_offset;
            }
        } else if nibble == 0 {
            // A literal run. The whole token is its length, which can never be
            // zero: an empty run would cost a byte to say nothing. A nibble of
            // fifteen always means an overflow varint follows.
            let len = if low == NIBBLE_MAX {
                extended_value(input, &mut pos, NIBBLE_MAX, "literal run length")?
            } else {
                low
            };
            if len == 0 {
                return Err("zero-length literal run".into());
            }
            if len > expected_len - out.len() {
                return Err(format!(
                    "literal run of {len} bytes overruns the {expected_len}-byte block"
                ));
            }
            let end = pos
                .checked_add(len)
                .ok_or_else(|| "literal run length overflows".to_string())?;
            if end > input.len() {
                return Err("literal run runs past the end of the block".into());
            }
            out.extend_from_slice(&input[pos..end]);
            pos = end;
        } else {
            // A match. The offset's low four bits ride in the high nibble, the
            // rest follows as a varint when the nibble overflowed; likewise for
            // the length in the low nibble.
            let offset = if nibble == NIBBLE_MAX {
                extended_value(input, &mut pos, NIBBLE_MAX, "match offset")?
            } else {
                nibble
            };
            let len_matched = if low == NIBBLE_MAX {
                extended_value(input, &mut pos, NIBBLE_MAX, "match length")?
            } else {
                low
            };
            let len = len_matched
                .checked_add(MIN_MATCH)
                .ok_or_else(|| "match length overflows".to_string())?;
            copy_match(&mut out, offset, len, expected_len)?;
            prev_offset = last_offset;
            last_offset = offset;
        }
    }

    if pos != input.len() {
        return Err(format!(
            "{} trailing bytes after a {expected_len}-byte block",
            input.len() - pos
        ));
    }
    if dict_len == 0 {
        return Ok(out);
    }
    if out.len() < dict_len {
        return Err("block is shorter than its dictionary prefix".into());
    }
    // Strip the dictionary prefix in place: `split_off` allocated and copied
    // the whole block again, which is measurable on 16 MiB blocks.
    let total = out.len();
    out.copy_within(dict_len.., 0);
    out.truncate(total - dict_len);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Match finding
// ---------------------------------------------------------------------------

/// A located repetition: `len` bytes starting `offset` back from `pos`.
#[derive(Debug, Clone, Copy)]
struct Match {
    len: usize,
    offset: usize,
}

impl Match {
    /// The position just past this match, given the position it starts at.
    #[allow(dead_code)]
    fn end_from(&self, pos: usize) -> usize {
        pos + self.len
    }
}

/// How hard to search, resolved from a level number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Effort {
    /// Hash chains to walk per position. 0 disables matching entirely.
    pub max_chain: usize,
    /// Cap on a single match length, bounding both the tail-compare cost and the
    /// longest reference the encoder will emit.
    pub max_match: usize,
    /// Cap on how far back a reference may point.
    pub window: usize,
    /// Whether to use lazy parsing (hold a match one step to see if the next
    /// position has a strictly better one). Costs extra search time.
    pub lazy: bool,
}

impl Effort {
    pub fn for_level(level: u8) -> Effort {
        // A log-ish curve: the first few levels buy most of the ratio for a
        // fraction of the time, and past level 6 each step costs noticeably more
        // than it returns.
        let level = level.min(12);
        let max_chain = match level {
            0 => 0,
            1 => 4,
            2 => 8,
            3 => 16,
            4 => 32,
            5 => 64,
            6 => 128,
            7 => 256,
            8 => 512,
            9 => 1024,
            10 => 2048,
            11 => 4096,
            _ => 8192,
        };
        Effort {
            max_chain,
            max_match: if level <= 2 {
                // A long match costs a long tail comparison at every position, and
                // on short-to-medium text the extra length rarely repays it.
                128
            } else {
                MAX_MATCH
            },
            // A match can never point further back than the block it lives in,
            // because every block is compressed independently and starts with an
            // empty history. A window wider than the largest allowed block can
            // therefore never be reached, so setting it to MAX_BLOCK costs
            // nothing and removes a second, invisible cap on match distance.
            //
            // This was 1 MiB, which silently clamped every block larger than
            // that: a `-b 4M` run could never emit a match pointing further back
            // than 1 MiB whatever the block size said.
            window: crate::chunking::MAX_BLOCK,
            // Lazy parsing roughly doubles match searches; level 1-3 stay
            // greedy as the fast path, level 4+ pays for the better parse.
            lazy: level >= 4,
        }
    }
}

/// A hash table of recent positions plus the chain that links each position to
/// the previous one with the same hash.
struct MatchFinder<'a> {
    data: &'a [u8],
    /// Dictionary chain links, shared by every block of the archive. Positions
    /// `0..boundary_start` are looked up here.
    base_chain: &'a [u32],
    /// The few dictionary positions whose four-byte key crosses into the block.
    /// Their link depends on the block's first bytes, so it lives here per block.
    boundary: [u32; MIN_MATCH - 1],
    boundary_start: usize,
    /// First position stored in `local_chain`; equal to the dictionary length.
    local_start: usize,
    /// Next-earlier position sharing this hash, or `NONE`, for the block itself.
    local_chain: Vec<u32>,
    /// Most recent position for this hash, or `NONE`.
    head: Vec<u32>,
    effort: Effort,
    /// The next position to examine.
    cursor: usize,
    /// The next position not yet inserted into the table. Kept separate from
    /// `cursor` because a search must happen *before* its own position is
    /// inserted, or the chain would start by pointing at the position being
    /// searched.
    indexed: usize,
}

const NONE: u32 = u32::MAX;

/// Ask the CPU to start fetching `chain[idx]` into cache now.
///
/// The chain walk is a linked list through an array several times the size of
/// L2, so each `chain[c]` load is a likely cache miss. The next pointer is
/// known one step ahead, so issuing a prefetch for it before the current
/// candidate is compared overlaps that miss with the comparison instead of
/// paying for it serially. A no-op off x86_64.
#[inline(always)]
fn prefetch(chain: &[u32], idx: u32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        if idx != NONE {
            _mm_prefetch(chain.as_ptr().add(idx as usize).cast::<i8>(), _MM_HINT_T0);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (chain, idx);
    }
}

/// Read four bytes as a key.
///
/// A four-byte window catches the repeats that matter in practice and costs one
/// load. Reading eight would find more long anchors but doubles the table's
/// collision rate, which costs more than the extra anchors return.
#[inline]
fn key_at(data: &[u8], pos: usize) -> usize {
    let b = &data[pos..pos + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
}

#[inline]
fn hash_key(key: usize) -> usize {
    // Knuth multiplicative hashing. The high bits of the product depend on every
    // input bit, so they are taken from the high end; masking off the low bits
    // instead would throw away exactly the bits that vary.
    key.wrapping_mul(0x9E37_79B1) >> 16
}

impl<'a> MatchFinder<'a> {
    fn new(data: &'a [u8], effort: Effort) -> Self {
        // One slot per four input bytes keeps the table sparse, which is what
        // keeps the chains short.
        MatchFinder::with_slots(data, effort, table_slots(data.len()))
    }

    fn with_slots(data: &'a [u8], effort: Effort, slots: usize) -> Self {
        MatchFinder {
            data,
            base_chain: &[],
            boundary: [NONE; MIN_MATCH - 1],
            boundary_start: 0,
            local_start: 0,
            // One chain link per input position, not per table slot: two positions
            // hashing to the same slot still need distinct links, or one would
            // overwrite the other's history.
            local_chain: vec![NONE; data.len()],
            head: vec![NONE; slots],
            effort,
            cursor: 0,
            indexed: 0,
        }
    }

    /// A finder over a block preceded by a shared dictionary table.
    ///
    /// The dictionary's chain is borrowed, not copied: only the block's own
    /// positions get a local array. On a tree of many small files the old
    /// per-block copy of the dictionary chain was several times the block size
    /// and dominated encoding time.
    fn new_seeded(data: &'a [u8], effort: Effort, table: &'a SeedTable, dict_len: usize) -> Self {
        let boundary_start = table.indexed;
        debug_assert!(dict_len >= boundary_start);
        debug_assert!(dict_len - boundary_start < MIN_MATCH);
        MatchFinder {
            data,
            base_chain: &table.chain,
            boundary: [NONE; MIN_MATCH - 1],
            boundary_start,
            local_start: dict_len,
            local_chain: vec![NONE; data.len() - dict_len],
            head: table.head.clone(),
            effort,
            cursor: dict_len,
            indexed: 0,
        }
    }

    fn table_mask(&self) -> usize {
        self.head.len() - 1
    }

    #[inline(always)]
    fn chain_at(&self, pos: usize) -> u32 {
        if pos < self.local_start {
            if pos < self.boundary_start {
                self.base_chain[pos]
            } else {
                self.boundary[pos - self.boundary_start]
            }
        } else {
            self.local_chain[pos - self.local_start]
        }
    }

    #[inline(always)]
    fn set_chain(&mut self, pos: usize, link: u32) {
        if pos < self.local_start {
            // Only the dictionary positions whose key straddles the block
            // boundary are ever (re)indexed here.
            self.boundary[pos - self.boundary_start] = link;
        } else {
            self.local_chain[pos - self.local_start] = link;
        }
    }

    #[inline(always)]
    fn prefetch_chain(&self, idx: u32) {
        if idx == NONE {
            return;
        }
        let c = idx as usize;
        if c < self.local_start {
            if c < self.boundary_start {
                prefetch(self.base_chain, idx);
            }
        } else {
            prefetch(&self.local_chain, (c - self.local_start) as u32);
        }
    }

    /// The next position that should be examined for a match.
    fn next(&mut self) -> Option<usize> {
        if self.cursor + MIN_MATCH > self.data.len() {
            return None;
        }
        let pos = self.cursor;
        self.cursor += 1;
        Some(pos)
    }

    /// Insert positions up to `upto` into the table.
    ///
    /// Bounded by the scan cursor: a position that has not been examined yet must
    /// not be in the table, or a later search would find it as its own match.
    fn index_through(&mut self, upto: usize) {
        let limit = upto.min(self.cursor);
        while self.indexed < limit {
            self.index(self.indexed);
            self.indexed += 1;
        }
    }

    #[inline]
    fn index(&mut self, pos: usize) {
        if pos + MIN_MATCH > self.data.len() {
            return;
        }
        let slot = hash_key(key_at(self.data, pos)) & self.table_mask();
        let link = self.head[slot];
        self.set_chain(pos, link);
        self.head[slot] = pos as u32;
    }

    /// Search for the best match at `pos`.
    fn find(&mut self, pos: usize) -> Option<Match> {
        if pos + MIN_MATCH > self.data.len() {
            return None;
        }
        let data = self.data;
        let max_len = self.effort.max_match.min(data.len() - pos);
        if max_len < MIN_MATCH {
            return None;
        }
        let floor = pos.saturating_sub(self.effort.window);
        let mut best: Option<Match> = None;
        let mut candidate = self.head[hash_key(key_at(data, pos)) & self.table_mask()];
        let mut chain = 0usize;

        while candidate != NONE && chain < self.effort.max_chain {
            let c = candidate as usize;
            chain += 1;
            if c >= pos {
                // Cannot happen: only earlier positions are indexed. Checked
                // anyway, because trusting it would be an infinite loop.
                break;
            }
            candidate = self.chain_at(c);
            self.prefetch_chain(candidate);
            if c < floor {
                // The chain runs backwards through positions, so once it leaves
                // the window the rest of it is older still.
                break;
            }

            // The bucket is keyed by a *hash* of the anchor word, so a candidate
            // only guarantees the same hash, not the same bytes: two four-byte
            // keys can collide and differ in their very first byte. Verify the
            // whole anchor word before trusting anything -- assuming byte 0 was
            // equal because the keys hashed alike silently emitted matches whose
            // first byte did not actually match, and the block failed its CRC.
            if u32::from_le_bytes(data[c..c + 4].try_into().unwrap())
                != u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap())
            {
                continue;
            }
            let mut len = MIN_MATCH;
            // Whole words while they fit, then the last few bytes one at a
            // time: the tail is at most seven bytes, so that loop is short.
            while len + 8 <= max_len {
                let a = u64::from_le_bytes(data[c + len..c + len + 8].try_into().unwrap());
                let b = u64::from_le_bytes(data[pos + len..pos + len + 8].try_into().unwrap());
                if a != b {
                    break;
                }
                len += 8;
            }
            while len < max_len && data[c + len] == data[pos + len] {
                len += 1;
            }
            if len < MIN_MATCH {
                continue;
            }
            let offset = pos - c;
            if offset > MAX_OFFSET {
                continue;
            }

            let better = match best {
                None => true,
                // Prefer the longer match, then the nearer one: at equal length a
                // smaller offset costs fewer varint bytes.
                Some(b) => len > b.len || (len == b.len && offset < b.offset),
            };
            if better {
                best = Some(Match { len, offset });
                // Stop at this level's own length cap, or once the match is
                // already long enough that searching further is not worth the
                // cache misses. Continuing at the cap provably cannot improve.
                if len >= max_len || len >= NICE_MATCH {
                    break;
                }
            }
        }

        best
    }
}

// ---------------------------------------------------------------------------
// Emitting
// ---------------------------------------------------------------------------

fn emit_literals(out: &mut Vec<u8>, data: &[u8], from: usize, to: usize) {
    let len = to.saturating_sub(from);
    if len == 0 {
        return;
    }
    let (nibble, extra) = split_nibble(len);
    out.push(nibble as u8);
    if let Some(extra) = extra {
        write_varint(out, extra as u64);
    }
    out.extend_from_slice(&data[from..to]);
}

fn emit_match(out: &mut Vec<u8>, m: Match, last_offset: &mut usize, prev_offset: &mut usize) {
    debug_assert!(m.len >= MIN_MATCH);
    debug_assert!(m.offset >= 1);
    if m.offset == *last_offset || (m.offset == *prev_offset && *prev_offset != 0) {
        // Repeat-offset form: a zero token selects the most recent slot (bit 0
        // clear) or the one before it (bit 0 set), and the same varint carries
        // the length. It cannot collide with a literal token, whose length
        // nibble is never zero.
        let which = u64::from(m.offset != *last_offset);
        if which == 1 {
            std::mem::swap(last_offset, prev_offset);
        } else {
            *prev_offset = *last_offset;
        }
        out.push(0);
        write_varint(out, ((m.len - MIN_MATCH) as u64) * 2 + which);
        return;
    }
    *prev_offset = *last_offset;
    *last_offset = m.offset;
    let (len_nibble, len_extra) = split_nibble(m.len - MIN_MATCH);
    let (offset_nibble, offset_extra) = split_nibble(m.offset);
    // The high nibble is never zero, which is what keeps match tokens distinct
    // from literal tokens.
    out.push(((offset_nibble as u8) << 4) | len_nibble as u8);
    if let Some(extra) = offset_extra {
        write_varint(out, extra as u64);
    }
    if let Some(extra) = len_extra {
        write_varint(out, extra as u64);
    }
}

/// Split a value across a four-bit token nibble and a varint overflow.
///
/// A nibble of exactly [`NIBBLE_MAX`] *always* means "an overflow varint
/// follows", never the value fifteen. The alternative -- using fifteen for both
/// meanings -- looks tidier and is subtly broken: the decoder cannot tell
/// whether to read a varint for the value 15, and whichever way it guesses, some
/// block fails to parse. One meaning per code point, no overlap.
///
/// The overflow is therefore `Some` whenever the nibble saturated, even when it
/// saturates to zero. Skipping a zero overflow would leave the decoder waiting
/// for a varint that never came, and every following field would be read at the
/// wrong offset.
#[inline]
fn split_nibble(value: usize) -> (usize, Option<usize>) {
    if value < NIBBLE_MAX {
        (value, None)
    } else {
        (NIBBLE_MAX, Some(value - NIBBLE_MAX))
    }
}

/// Bytes a match saves versus emitting its bytes as literals.
fn saving(m: &Match) -> usize {
    m.len
        .saturating_sub(1 + varint_len(m.offset as u64) + extension_len(m.len))
}

/// Bytes a match token costs, including both possible varints.
fn varint_len(v: u64) -> usize {
    let mut n = 1;
    let mut v = v >> 7;
    while v != 0 {
        n += 1;
        v >>= 7;
    }
    n
}

/// Extra bytes a match costs beyond its token: the varint overflow, which exists
/// whenever a nibble saturated.
fn extension_len(len: usize) -> usize {
    match split_nibble(len - MIN_MATCH) {
        (_, None) => 0,
        (_, Some(extra)) => varint_len(extra as u64),
    }
}

#[inline]
fn read_byte(input: &[u8], pos: &mut usize) -> Result<u8, String> {
    if *pos >= input.len() {
        return Err("unexpected end of compressed block".into());
    }
    let b = input[*pos];
    *pos += 1;
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::hash64;

    /// Deterministic bytes, so any failure is reproducible from the seed.
    fn prng(seed: u64) -> impl Iterator<Item = u8> {
        let mut s = seed;
        std::iter::from_fn(move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Some((s >> 33) as u8)
        })
    }

    /// Text-like data: small alphabet, repeated words. This is what the ratio
    /// claims are about, so it is what most tests use.
    fn prose(len: usize) -> Vec<u8> {
        const WORDS: [&str; 12] = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "then", "runs",
            "away",
        ];
        let mut s = 7u64;
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = WORDS[((s >> 33) as usize) % WORDS.len()];
            out.extend_from_slice(w.as_bytes());
            out.push(b' ');
        }
        out.truncate(len);
        out
    }

    /// Round-trip and assert the block did not come back changed.
    fn roundtrip(data: &[u8], level: u8) -> Option<(usize, usize)> {
        let packed = compress(data, level)?;
        let back = decompress(&packed, data.len()).expect("decompress");
        assert_eq!(
            hash64(&back),
            hash64(data),
            "level {level}: round trip changed the data"
        );
        Some((packed.len(), data.len()))
    }

    #[test]
    fn roundtrips_incompressible_data_via_the_raw_path() {
        // Random bytes have no matches, so the codec must decline to store them
        // and the writer keeps them verbatim.
        let data: Vec<u8> = prng(1).take(50_000).collect();
        assert!(
            compress(&data, 6).is_none(),
            "random data must not be coded"
        );
    }

    #[test]
    fn roundtrips_every_level_on_prose() {
        let data = prose(200_000);
        // Level 0 means "store", so it is not in this loop; see
        // `level_zero_declines_to_compress`.
        for level in 1..=12 {
            let (packed, raw) = roundtrip(&data, level).expect("prose must compress");
            let ratio = raw as f64 / packed as f64;
            assert!(ratio > 1.8, "level {level}: ratio {ratio:.2} is too poor");
            println!("level {level:2}: {packed:>7} / {raw:>7} = {ratio:.2}x");
        }
    }

    #[test]
    fn level_zero_declines_to_compress() {
        // Level 0 is the "store" level: the caller keeps the block verbatim, so
        // the codec must decline rather than expand the data.
        let data = prose(50_000);
        assert!(compress(&data, 0).is_none());
    }

    #[test]
    fn higher_levels_find_at_least_as_good_a_match() {
        // More search must never *lose* a match at the position it examines: a
        // longer chain can only turn up a longer or nearer candidate. That is the
        // property level tuning actually rests on.
        //
        // What is deliberately not asserted is that the final output shrinks
        // monotonically with level. Greedy parsing commits to whichever match it
        // finds at each position, so finding a longer match early can leave less
        // room for the matches behind it -- the classic greedy-parse wobble. The
        // best level is still clearly better than the cheapest one, which is what
        // this test checks.
        let data = prose(300_000);
        let fastest = compress(&data, 1).expect("level 1");
        let mut best = usize::MAX;
        for level in 1..=10 {
            let packed = compress(&data, level).expect("compress");
            assert!(
                packed.len() <= fastest.len() * 101 / 100 + 512,
                "level {level} is far worse than level 1: {} vs {}",
                packed.len(),
                fastest.len(),
            );
            best = best.min(packed.len());
        }
        assert!(
            best < fastest.len(),
            "the best level ({best}) did not beat level 1 ({})",
            fastest.len(),
        );
    }

    #[test]
    fn roundtrips_every_size_around_the_boundaries() {
        // Sizes near MIN_MATCH, near the nibble limits, and near a varint edge
        // are where off-by-one errors live.
        let data = prose(2000);
        for len in 0..600usize {
            let slice = &data[..len];
            if let Some(packed) = compress(slice, 5) {
                let back = decompress(&packed, len).expect("decompress");
                assert_eq!(back, slice, "length {len} did not round trip");
            }
        }
    }

    #[test]
    fn roundtrips_at_every_nibble_boundary() {
        // Force literal runs and match lengths that straddle 15 and 19, the two
        // places the token nibbles overflow into a varint.
        for repeat in 1..40usize {
            let mut data = Vec::new();
            for i in 0..repeat {
                data.extend(std::iter::repeat_n(b'A' + (i % 26) as u8, 15 + i));
                data.extend_from_slice(b" padding to break the run up ");
            }
            for level in [1u8, 5, 9] {
                if let Some(packed) = compress(&data, level) {
                    assert_eq!(decompress(&packed, data.len()).unwrap(), data);
                }
            }
        }
    }

    #[test]
    fn roundtrips_highly_repetitive_data() {
        // One byte repeated: matches at every offset, including overlapping
        // copies that reference bytes still being produced.
        for len in [1usize, 100, 5000, 100_000] {
            let data = vec![b'z'; len];
            for level in [1u8, 4, 9] {
                if let Some(packed) = compress(&data, level) {
                    assert_eq!(decompress(&packed, len).unwrap(), data, "len {len}");
                }
            }
        }
    }

    #[test]
    fn roundtrips_structured_binary() {
        // A plausible binary file: a repeating record structure with varying
        // fields, which is what Smart Bypass lets through.
        let mut data = Vec::new();
        let mut s = 99u64;
        while data.len() < 200_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            data.extend_from_slice(b"RECORD\x00\x01\x02");
            data.extend_from_slice(&(s >> 40).to_le_bytes());
            data.extend_from_slice(&[0u8; 12]);
            data.extend_from_slice(b"payload payload payload");
        }
        if let Some(packed) = compress(&data, 6) {
            assert_eq!(decompress(&packed, data.len()).unwrap(), data);
        }
    }

    #[test]
    fn roundtrips_a_large_file_at_every_level() {
        // The buffer must not depend on the caller pre-reserving.
        let data = prose(1_000_000);
        for level in [1u8, 3, 6, 9] {
            let packed = compress(&data, level).expect("compress");
            let back = decompress(&packed, data.len()).expect("decompress");
            assert_eq!(back, data, "level {level}");
        }
    }

    #[test]
    fn reusable_dictionary_seed_is_byte_identical_to_a_fresh_scan() {
        // The seed caches the dictionary's hash table per slot count. Every
        // block must produce exactly the bytes the old per-block rescan did,
        // including the few dictionary positions whose four-byte key straddles
        // the dictionary/block boundary and blocks small enough to change the
        // table size.
        let dict = prose(300_000);
        let seed = DictSeed::new(Arc::new(dict.clone()));
        for dict_len in [1usize, 3, 4, 5, 64, 4096, 65_537, 300_000] {
            let dict = &dict[..dict_len];
            let fresh;
            let seed = if dict_len == 300_000 {
                &seed
            } else {
                // A fresh seed for each slice; the cache itself is exercised
                // across the block sizes below.
                fresh = DictSeed::new(Arc::new(dict.to_vec()));
                &fresh
            };
            for data_len in [8usize, 9, 10, 100, 4096, 65_536, 200_001] {
                for level in [1u8, 4, 7] {
                    let data = prose(data_len);
                    let effort = Effort::for_level(level);
                    let mut old = Vec::new();
                    let a = compress_into_with_dict(&data, dict, effort, &mut old);
                    let mut new = Vec::new();
                    let b = compress_seeded(&data, dict, Some(seed), effort, &mut new);
                    assert_eq!(a.is_some(), b.is_some(), "dict {dict_len} data {data_len}");
                    assert_eq!(old, new, "dict {dict_len} data {data_len} level {level}");
                }
            }
        }
    }

    #[test]
    fn randomised_roundtrips() {
        // The fuzz: structured data with random splices, random truncation and
        // random levels. Structure gives the match finder something to find;
        // splicing gives it boundaries it cannot predict.
        let mut s = 0xDEAD_BEEFu64;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as usize
        };
        for case in 0..400 {
            let len = 64 + next() % 30_000;
            let mut data = prose(len);
            // Splice in noise so no two cases share a prefix structure.
            let splices = next() % 8;
            for _ in 0..splices {
                let at = next() % data.len().max(1);
                let noise_len = 1 + next() % 64;
                let noise: Vec<u8> = prng(next() as u64).take(noise_len).collect();
                let end = (at + noise.len()).min(data.len());
                data.splice(at..end, noise);
            }
            let level = (next() % 12) as u8;
            if let Some(packed) = compress(&data, level) {
                let back = decompress(&packed, data.len())
                    .unwrap_or_else(|e| panic!("case {case} level {level}: {e}"));
                assert_eq!(
                    hash64(&back),
                    hash64(&data),
                    "case {case} level {level}: data changed"
                );
            }
        }
    }

    #[test]
    fn decoder_rejects_truncated_and_corrupt_input() {
        let data = prose(20_000);
        let packed = compress(&data, 6).expect("compress");

        // Truncation at every length must be an error, never a panic.
        for cut in [0, 1, 5, 100, packed.len() / 2, packed.len() - 1] {
            let r = decompress(&packed[..cut], data.len());
            assert!(r.is_err(), "truncated to {cut} bytes should not decode");
        }

        // A wrong expected length must not be quietly accepted.
        assert!(decompress(&packed, data.len() - 1).is_err());
        assert!(decompress(&packed, data.len() + 1).is_err());

        // Trailing garbage must be rejected rather than ignored.
        let mut extra = packed.clone();
        extra.push(0);
        assert!(decompress(&extra, data.len()).is_err());

        // A match pointing before the start of the block must be refused. The
        // first byte of a coded stream is a literal token, so forging one that
        // reads as a match is enough.
        assert!(decompress(&[0x10, 0x01], 16).is_err());
    }

    #[test]
    fn decoder_never_speeds_past_the_expected_length() {
        // A hostile block must not be able to make the decoder allocate far more
        // than the length it was promised.
        let mut packed = Vec::new();
        // Token 0xF0: offset nibble 15, length nibble 0 -> a long match.
        packed.push(0xF0);
        write_varint(&mut packed, 1 << 40);
        assert!(decompress(&packed, 1024).is_err());
    }

    #[test]
    fn token_encoding_agrees_with_the_decoder_at_every_boundary() {
        // Encode one literal run of every length and one match of every length and
        // offset, then decode each one on its own. This is the test that catches a
        // nibble meaning two different things -- the case where an encoder emits
        // the value 15 with no overflow and a decoder insists on reading one.
        for len in 1..300usize {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let mut packed = Vec::new();
            emit_literals(&mut packed, &data, 0, len);
            assert_eq!(
                decompress(&packed, len).unwrap_or_else(|e| panic!("literal {len}: {e}")),
                data
            );
        }
        for len in MIN_MATCH..200usize {
            for offset in [1usize, 2, 14, 15, 16, 127, 128, 129, 1000, 70_000] {
                let data: Vec<u8> = (0..offset).map(|i| (i % 97) as u8).collect();
                let total = offset + len;
                let mut packed = Vec::new();
                emit_literals(&mut packed, &data, 0, offset);
                let mut last_offset = 0usize;
                let mut prev_offset = 0usize;
                emit_match(
                    &mut packed,
                    Match { len, offset },
                    &mut last_offset,
                    &mut prev_offset,
                );
                let mut expected = data.clone();
                for i in 0..len {
                    let b = expected[i % offset];
                    expected.push(b);
                }
                assert_eq!(expected.len(), total);
                assert_eq!(
                    decompress(&packed, total)
                        .unwrap_or_else(|e| { panic!("match len {len} offset {offset}: {e}") }),
                    expected,
                    "match len {len} offset {offset}"
                );
            }
        }
    }

    #[test]
    fn a_nibble_of_fifteen_always_means_an_overflow_follows() {
        // Fifteen is an escape code, never a value. Pinned here because the
        // tempting shortcut -- treat 15 as the value 15 when nothing follows --
        // is invisible until a block fails to parse.
        for value in [0usize, 13, 14, 15, 16, 30, 31, 200] {
            let (nibble, extra) = split_nibble(value);
            if value < NIBBLE_MAX {
                assert_eq!((nibble, extra), (value, None));
            } else {
                assert_eq!(nibble, NIBBLE_MAX);
                assert_eq!(
                    extra,
                    Some(value - NIBBLE_MAX),
                    "a saturated nibble must carry its overflow, including a zero one"
                );
            }
        }
    }

    #[test]
    fn varint_lengths_match_the_encoding() {
        for v in [0u64, 1, 127, 128, 16_383, 16_384, 1 << 20, 1 << 35] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            assert_eq!(buf.len(), varint_len(v), "varint length wrong for {v}");
            let mut pos = 0;
            assert_eq!(read_varint(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
        }
    }

    /// A block that repeats its dictionary must decode byte-identically with the
    /// dictionary in the window. If the encoder and decoder disagree about where
    /// the window starts, this is what fails.
    #[test]
    fn dictionary_seeded_blocks_round_trip() {
        let dict: Vec<u8> = prng(0xA5A5).take(8192).collect();
        for level in [1u8, 3, 6, 9] {
            let mut data = dict.clone();
            data.extend(prng(0x1234).take(4096));
            data.extend_from_slice(&dict);
            let coded = compress_with_dict(&data, &dict, level).expect("should compress");
            let back = decompress_with_dict(&coded, &dict, data.len()).expect("should decode");
            assert_eq!(back, data, "level {level} did not round trip");
        }
    }

    /// The dictionary is history, not content: it must not appear in the output,
    /// and the block's own size guard must ignore it too.
    #[test]
    fn dictionary_is_not_part_of_the_block() {
        let dict = b"a repetitive dictionary body that appears nowhere else".repeat(40);
        let data = dict.clone();
        let coded = compress_with_dict(&data, &dict, 6).expect("should compress");
        assert!(
            coded.len() < data.len(),
            "a pure copy should collapse, got {} from {}",
            coded.len(),
            data.len()
        );
        let back = decompress_with_dict(&coded, &dict, data.len()).unwrap();
        assert_eq!(back.len(), data.len());
    }

    /// A match may point into the dictionary, so a block whose content is only in
    /// the dictionary is the case the seeding exists for.
    #[test]
    fn a_match_can_reach_into_the_dictionary() {
        let dict = b"common symbol table content ".repeat(200);
        let data = b"common symbol table content ".repeat(20);
        let coded = compress_with_dict(&data, &dict, 6).expect("should compress");
        assert!(coded.len() < data.len() / 2, "dictionary was not used");
        let back = decompress_with_dict(&coded, &dict, data.len()).unwrap();
        assert_eq!(back, data);
    }

    /// Blocks with no dictionary must keep working exactly as before.
    #[test]
    fn an_empty_dictionary_is_the_old_path() {
        let data = b"the same sentence again, so both paths have something to match ".repeat(700);
        let plain = compress(&data, 6).expect("plain should compress");
        let with_empty = compress_with_dict(&data, &[], 6).expect("should compress");
        assert_eq!(
            plain, with_empty,
            "an empty dictionary changed the encoding"
        );
        let back = decompress_with_dict(&plain, &[], data.len()).unwrap();
        assert_eq!(back, data);
    }

    /// A corrupt length or offset must be an error, not a wrapping add that
    /// turns `u64::MAX` into a plausible small value (or a debug-build panic).
    #[test]
    fn hostile_varints_are_rejected_without_overflowing() {
        let mut huge = Vec::new();
        crate::util::write_varint(&mut huge, u64::MAX);

        let mut literal = vec![0x0F];
        literal.extend_from_slice(&huge);
        assert!(decompress(&literal, 1).is_err());

        let mut offset = vec![0xF0];
        offset.extend_from_slice(&huge);
        assert!(decompress(&offset, 4).is_err());

        let mut len = vec![0x1F];
        len.extend_from_slice(&huge);
        assert!(decompress(&len, 4).is_err());
    }
}
