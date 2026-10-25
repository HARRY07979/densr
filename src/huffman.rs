//! Canonical Huffman coding for one block's already-coded bytes.
//!
//! The LZ codec emits a skewed byte stream: small token nibbles, small varint
//! bytes and (on text) repeated literal bytes dominate, while high bytes are
//! rare. A per-block canonical Huffman code turns that skew into ~10-30% fewer
//! bytes with no change to the LZ format itself: the decoder Huffman-decodes
//! first, then parses the result exactly as before.
//!
//! # Layout of an encoded payload
//!
//! ```text
//! u32  decoded_len      (little-endian, length of the coded stream)
//! u8   n_symbols        (count of (symbol, length) pairs; 0 means empty input)
//! u8[n_symbols] symbol
//! u8[n_symbols] length  (code length in bits for that symbol, 1..32)
//! u8[] bitstream        (MSB-first canonical codes, zero-padded to a byte)
//! ```
//!
//! Only symbols that occur are stored, so the table costs two bytes per live
//! symbol. A block whose Huffman form would not beat its input is declined by
//! returning `None`, leaving the caller to keep the plain coded form.

/// Longest code the encoder will emit. Byte alphabets practically never exceed
/// ~16 bits; 32 keeps the bit buffer a plain `u64` with room to spare.
const MAX_BITS: usize = 32;

/// Encode `data`, returning `None` when the Huffman form is not smaller
/// (table included) or the input is empty.
pub fn compress(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    let mut freq = [0u64; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    let lengths = code_lengths(&freq);
    // A canonical code stored in a `u32` cannot represent a code longer than
    // `MAX_BITS`. `code_lengths` may return such a length on an adversarial
    // frequency set (a very deep Huffman tree), and clamping it afterwards would
    // over-subscribe the code space and produce codes that do not fit their
    // stated length. Huffman is optional, so decline instead of emitting an
    // invalid table; the caller keeps the plain LZ stream.
    if lengths.iter().any(|&l| l as usize > MAX_BITS) {
        return None;
    }
    // Bit cost of the stream plus the table, in bytes rounded up.
    let mut bits = 0u64;
    let mut live = 0usize;
    for s in 0..256 {
        if freq[s] > 0 {
            live += 1;
            bits += freq[s] * lengths[s] as u64;
        }
    }
    let table_bytes = 4 + 1 + live * 2;
    let total = table_bytes + ((bits + 7) / 8) as usize;
    if total >= data.len() {
        return None;
    }

    let codes = canonical_codes(&lengths);
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.push(live as u8);
    for s in 0..256 {
        if freq[s] > 0 {
            out.push(s as u8);
        }
    }
    for s in 0..256 {
        if freq[s] > 0 {
            out.push(lengths[s]);
        }
    }
    // Bitstream, MSB-first.
    let mut acc = 0u64;
    let mut acc_bits = 0u32;
    for &b in data {
        let (code, len) = codes[b as usize];
        acc = (acc << len) | code as u64;
        acc_bits += len as u32;
        while acc_bits >= 8 {
            acc_bits -= 8;
            out.push((acc >> acc_bits) as u8);
        }
    }
    if acc_bits > 0 {
        out.push((acc << (8 - acc_bits)) as u8);
    }
    debug_assert_eq!(out.len(), total);
    Some(out)
}

/// Decode a payload produced by [`compress`].
pub fn decompress(input: &[u8]) -> Result<Vec<u8>, String> {
    if input.len() < 5 {
        return Err("huffman block too short for a header".into());
    }
    let decoded_len = u32::from_le_bytes([input[0], input[1], input[2], input[3]]) as usize;
    // A full 256-symbol alphabet stores 0 (256 does not fit in a byte);
    // the encoder never emits an empty table, so 0 unambiguously means 256.
    let live = if input[4] == 0 {
        256
    } else {
        input[4] as usize
    };
    if decoded_len == 0 {
        return Err("huffman block with empty header".into());
    }
    let head = 5 + live * 2;
    if input.len() < head {
        return Err("huffman block shorter than its table".into());
    }
    let mut lengths = [0u8; 256];
    for i in 0..live {
        let sym = input[5 + i] as usize;
        let len = input[5 + live + i];
        if len == 0 || len as usize > MAX_BITS {
            return Err(format!("bad huffman code length {len}"));
        }
        if lengths[sym] != 0 {
            return Err("duplicate huffman symbol".into());
        }
        lengths[sym] = len;
    }
    // Reject an over-subscribed table before it can drive the lookup tables.
    // Without this check, 256 symbols all claiming length 1 makes the LUT fill
    // write past its end, so a crafted archive aborts the process instead of
    // reporting a bad block. An incomplete code is legal (not every code needed
    // by the input has to be used), but the Kraft sum may never exceed one.
    {
        let mut used = 0u64;
        for l in 1..=MAX_BITS {
            let n = lengths.iter().filter(|&&x| x as usize == l).count() as u64;
            used += n << (MAX_BITS - l);
        }
        if used > 1u64 << MAX_BITS {
            return Err("over-subscribed huffman table".into());
        }
    }
    // Canonical decode in O(1) per bit.
    //
    // The previous version scanned all 256 symbols after every bit to find one
    // matching (length, code). That is ~256 comparisons per bit, so decoding a
    // megabyte of coded stream cost hundreds of millions of comparisons and
    // dominated `verify`. Canonical codes are assigned in order, so the symbols
    // of a given length form one contiguous numeric run and the symbol can be
    // found by arithmetic instead of by search:
    //
    //   first_code[l]  smallest code of length l
    //   count[l]       how many symbols have length l
    //   first_index[l] where those symbols start in `ordered`
    //
    // A bit is consumed until `code` lands inside the run for its length.
    let mut ordered: [u8; 256] = [0; 256];
    let mut count = [0u32; MAX_BITS + 1];
    let mut first_code = [0u32; MAX_BITS + 2];
    let mut first_index = [0u32; MAX_BITS + 2];
    {
        let mut n = 0usize;
        for l in 1..=MAX_BITS {
            first_index[l] = n as u32;
            first_code[l] = 0u32;
            for s in 0..256 {
                if lengths[s] as usize == l {
                    ordered[n] = s as u8;
                    n += 1;
                }
            }
            count[l] = (n as u32) - first_index[l];
        }
        // Canonical codes are sequential, and each length level is one bit
        // deeper than the last, so the next level starts at the current end
        // shifted up by one: first_code[l+1] = (first_code[l] + count[l]) << 1.
        // The running value is kept in `u64`: after the last level it can reach
        // 2^32, which does not fit the `u32` the codes themselves use.
        let mut code = 0u64;
        for l in 1..=MAX_BITS {
            first_code[l] = code as u32;
            code = (code + count[l] as u64) << 1;
        }
    }

    // Fast path: a lookup table indexed by the next LUT_BITS bits. Codes up to
    // LUT_BITS long decode with one array read instead of one loop per bit.
    // Shorter codes fill every suffix below them, so the trailing bits that
    // follow a short code (including zero padding at the end of the stream)
    // cannot change the entry. Longer codes leave a zero entry and fall back
    // to the canonical walk; they are rare by construction, since Huffman
    // hands long codes to the least frequent symbols.
    const LUT_BITS: u32 = 12;
    let lut_len = 1usize << LUT_BITS;
    let mut lut = vec![0u16; lut_len];
    for l in 1..=LUT_BITS as usize {
        let n = count[l] as usize;
        if n == 0 {
            continue;
        }
        let shift = LUT_BITS as usize - l;
        let reps = 1usize << shift;
        for i in 0..n {
            let sym = ordered[first_index[l] as usize + i];
            let code = first_code[l] as usize + i;
            let base = code << shift;
            let entry = ((l as u16) << 8) | sym as u16;
            lut[base..base + reps].fill(entry);
        }
    }

    let mut bits = BitReader::new(&input[head..]);
    let mut out = Vec::with_capacity(decoded_len.min(64 << 20));
    while out.len() < decoded_len {
        let entry = lut[bits.peek(LUT_BITS) as usize];
        if entry != 0 {
            let l = (entry >> 8) as u32;
            if bits.remaining() < l as u64 {
                return Err("huffman bitstream ends early".into());
            }
            bits.skip(l);
            out.push((entry & 0xFF) as u8);
            continue;
        }
        // Long code: keep the LUT_BITS-bit prefix and extend it one bit at a
        // time until it lands in the run assigned to its length.
        if bits.remaining() <= LUT_BITS as u64 {
            return Err("huffman bitstream ends early".into());
        }
        let mut code = bits.peek(LUT_BITS);
        bits.skip(LUT_BITS);
        let mut len = LUT_BITS;
        loop {
            if bits.remaining() == 0 {
                return Err("huffman bitstream ends early".into());
            }
            code = (code << 1) | bits.peek(1);
            bits.skip(1);
            len += 1;
            if len > MAX_BITS as u32 {
                return Err("huffman code length exceeded".into());
            }
            let l = len as usize;
            let n = count[l];
            if n > 0 && code >= first_code[l] && code - first_code[l] < n {
                out.push(ordered[(first_index[l] + (code - first_code[l])) as usize]);
                break;
            }
        }
    }
    Ok(out)
}

/// MSB-first bit reader with a 64-bit staging buffer.
///
/// `buf` holds the next `bits` bits in its low `bits` positions, so `peek` is
/// one shift and one mask. Bytes are appended eight at a time while at most 56
/// bits are buffered, leaving room to shift the whole buffer left without
/// dropping the bits still to be consumed. At the end of the input `peek`
/// zero-pads, which matches the encoder's zero padding; the decoder separately
/// checks that a code's bits were actually present.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits: u32,
    consumed: u64,
    total: u64,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            buf: 0,
            bits: 0,
            consumed: 0,
            total: data.len() as u64 * 8,
        }
    }

    #[inline]
    fn refill(&mut self) {
        while self.bits <= 56 && self.pos < self.data.len() {
            self.buf = (self.buf << 8) | self.data[self.pos] as u64;
            self.bits += 8;
            self.pos += 1;
        }
    }

    #[inline]
    fn peek(&mut self, n: u32) -> u32 {
        self.refill();
        let mask = (1u64 << n) - 1;
        let v = if self.bits >= n {
            (self.buf >> (self.bits - n)) & mask
        } else {
            (self.buf << (n - self.bits)) & mask
        };
        v as u32
    }

    #[inline]
    fn skip(&mut self, n: u32) {
        self.bits -= n;
        self.consumed += n as u64;
    }

    #[inline]
    fn remaining(&self) -> u64 {
        self.total - self.consumed
    }
}

/// Huffman code lengths from frequencies.
///
/// A simple two-queue Huffman build on sorted frequencies. Lengths may exceed
/// [`MAX_BITS`] on adversarial distributions; the caller declines to use the
/// result rather than clamping it into an invalid code (see [`compress`]).
fn code_lengths(freq: &[u64; 256]) -> [u8; 256] {
    // Nodes: (frequency, lowest symbol, left, right). Leaves are symbols.
    #[derive(Clone, Copy)]
    struct Node {
        freq: u64,
        min_sym: usize,
    }
    // Active (symbol, frequency) pairs, sorted by (frequency, symbol).
    let mut active: Vec<Node> = (0..256)
        .filter(|&s| freq[s] > 0)
        .map(|s| Node {
            freq: freq[s],
            min_sym: s,
        })
        .collect();
    active.sort_by_key(|n| (n.freq, n.min_sym));

    let mut lengths = [0u8; 256];
    if active.len() == 1 {
        // One symbol: a single 1-bit code.
        lengths[active[0].min_sym] = 1;
        return lengths;
    }

    // Depth of each leaf = number of times its subtree is merged.
    // Track membership as a bitmask over the sorted order is overkill;
    // instead keep explicit member lists (at most 256 symbols total).
    let members: Vec<Vec<usize>> = active.iter().map(|n| vec![n.min_sym]).collect();
    let freqs: Vec<(u64, usize)> = active
        .iter()
        .enumerate()
        .map(|(i, n)| (n.freq, i))
        .collect();
    // Two queues: initial sorted leaves, then merged nodes (non-decreasing).
    let mut leaf = 0usize; // next leaf index in freqs order
    let mut merged: Vec<(u64, Vec<usize>)> = Vec::new();
    let mut merged_head = 0usize;
    // Remaining count.
    let mut remaining = freqs.len();
    // Helper to pop the smallest of (next leaf, next merged).
    macro_rules! pop_min {
        () => {{
            let take_leaf = if leaf < freqs.len() {
                if merged_head < merged.len() {
                    freqs[leaf].0 < merged[merged_head].0
                } else {
                    true
                }
            } else {
                false
            };
            if take_leaf {
                let (f, i) = freqs[leaf];
                leaf += 1;
                (f, members[i].clone())
            } else {
                let (f, m) = merged[merged_head].clone();
                merged_head += 1;
                (f, m)
            }
        }};
    }
    while remaining > 1 {
        let (f1, m1) = pop_min!();
        let (f2, m2) = pop_min!();
        for &s in m1.iter().chain(m2.iter()) {
            lengths[s] += 1;
        }
        let mut both = m1;
        both.extend_from_slice(&m2);
        merged.push((f1 + f2, both));
        remaining -= 1;
    }
    // Lengths are not clamped here. Clamping changes the tree's shape without
    // restoring the Kraft equality, which can over-subscribe the code space.
    // `compress` checks the result against `MAX_BITS` and declines instead.
    lengths
}

/// Canonical codes from lengths: symbols sorted by (length, symbol) get
/// sequential codes. Returns (code, length) per symbol.
fn canonical_codes(lengths: &[u8; 256]) -> [(u32, u8); 256] {
    let mut order: Vec<usize> = (0..256).filter(|&s| lengths[s] > 0).collect();
    order.sort_by_key(|&s| (lengths[s], s));
    let mut codes = [(0u32, 0u8); 256];
    let mut code = 0u32;
    let mut prev_len = 0u8;
    for &s in &order {
        let len = lengths[s];
        code <<= (len - prev_len) as u32;
        codes[s] = (code, len);
        code += 1;
        prev_len = len;
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        match compress(data) {
            Some(packed) => {
                assert!(packed.len() < data.len());
                let back = decompress(&packed).expect("decompress");
                assert_eq!(back, data);
            }
            None => {
                // Declined: only legal when incompressible.
                let mut freq = [0u64; 256];
                for &b in data {
                    freq[b as usize] += 1;
                }
                let live = freq.iter().filter(|&&f| f > 0).count();
                // A rough bound: skewed data must compress.
                if data.len() > 4096 && live < 200 {
                    panic!("skewed {}-byte input declined", data.len());
                }
            }
        }
    }

    #[test]
    fn roundtrips_skewed_data() {
        let mut data = Vec::new();
        for i in 0..50_000 {
            data.push(if i % 10 == 0 { 0xFF } else { (i % 4) as u8 });
        }
        roundtrip(&data);
    }

    #[test]
    fn roundtrips_text_like_data() {
        let words = ["the ", "quick ", "brown ", "fox ", "jumps "];
        let mut data = Vec::new();
        for i in 0..20_000 {
            data.extend_from_slice(words[i % words.len()].as_bytes());
        }
        roundtrip(&data);
    }

    #[test]
    fn roundtrips_single_symbol() {
        roundtrip(&vec![0xABu8; 10_000]);
    }

    #[test]
    fn declines_random_data() {
        let mut s = 0x12345678u64;
        let data: Vec<u8> = (0..20_000)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect();
        assert!(compress(&data).is_none(), "random data must decline");
    }

    #[test]
    fn declines_empty_and_tiny() {
        assert!(compress(&[]).is_none());
        assert!(compress(&[1, 2, 3]).is_none());
    }

    #[test]
    fn decoder_rejects_garbage() {
        assert!(decompress(&[]).is_err());
        assert!(decompress(&[1, 2, 3, 4]).is_err());
        // Valid header shape, truncated table.
        assert!(decompress(&[10, 0, 0, 0, 5, 65]).is_err());
        // Round-trip then corrupt the tail.
        let data = vec![7u8; 5000];
        let mut packed = compress(&data).expect("compress");
        packed.pop();
        assert!(decompress(&packed).is_err());
    }

    #[test]
    fn decodes_codes_longer_than_the_lut() {
        // A complete tree that reaches 13 bits: one symbol for each length
        // 1..=12 and two at 13. The 13-bit symbols miss the lookup table and
        // must take the canonical fallback.
        let lens = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13];
        let mut lengths = [0u8; 256];
        for (s, &l) in lens.iter().enumerate() {
            lengths[s] = l;
        }
        let codes = canonical_codes(&lengths);
        let symbols = [0u8, 13, 1, 12, 2, 13, 3, 12, 0, 11, 13];
        let mut acc = 0u64;
        let mut nbits = 0u32;
        let mut body = Vec::new();
        for &s in &symbols {
            let (code, len) = codes[s as usize];
            acc = (acc << len) | code as u64;
            nbits += len as u32;
            while nbits >= 8 {
                nbits -= 8;
                body.push((acc >> nbits) as u8);
            }
        }
        if nbits > 0 {
            body.push((acc << (8 - nbits)) as u8);
        }

        let mut payload = Vec::new();
        payload.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
        payload.push(lens.len() as u8);
        for s in 0..lens.len() {
            payload.push(s as u8);
        }
        payload.extend_from_slice(&lens);
        payload.extend_from_slice(&body);

        assert_eq!(decompress(&payload).expect("decode"), symbols);
    }

    #[test]
    fn rejects_an_over_subscribed_table_without_panicking() {
        // 256 symbols all claiming a one-bit code: the Kraft sum is 128, and a
        // decoder that trusted it would index the LUT far past its end.
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.push(0); // 0 means a full 256-symbol alphabet
        payload.extend(0u8..=255);
        payload.extend_from_slice(&[1u8; 256]);
        payload.push(0); // one byte of bitstream, enough for the decoded length
        assert!(decompress(&payload).is_err());
    }

    #[test]
    fn deep_frequencies_are_reported_rather_than_clamped() {
        // Fibonacci frequencies produce a maximally deep tree; with 50 leaves it
        // exceeds MAX_BITS. The encoder must decline, not silently clamp.
        let mut freq = [0u64; 256];
        let (mut a, mut b) = (1u64, 1u64);
        for f in freq.iter_mut().take(50) {
            *f = a;
            let next = a + b;
            a = b;
            b = next;
        }
        let lengths = code_lengths(&freq);
        assert!(lengths.iter().any(|&l| l as usize > MAX_BITS));
    }

    #[test]
    fn canonical_codes_are_prefix_free() {
        // Exhaustive: no code is a prefix of another at these lengths.
        let mut freq = [0u64; 256];
        for s in 0..256 {
            freq[s] = 1 + (s as u64 * 37 % 100);
        }
        let lengths = code_lengths(&freq);
        let codes = canonical_codes(&lengths);
        for a in 0..256 {
            for b in 0..256 {
                if a == b || lengths[a] == 0 || lengths[b] == 0 {
                    continue;
                }
                let (ca, la) = codes[a];
                let (cb, lb) = codes[b];
                if la <= lb {
                    assert_ne!(ca, cb >> (lb - la), "code {a} is a prefix of code {b}");
                }
            }
        }
    }
}
