//! The container: how an archive is laid out on disk.
//!
//! ```text
//! +----------------------------------+ 0
//! | header, 64 bytes                 |
//! +----------------------------------+ 64
//! | block frames, back to back       |
//! |   32-byte record + payload       |
//! +----------------------------------+ dir_offset
//! | directory: one record per entry  |
//! |   varint-encoded, CRC per entry  |
//! +----------------------------------+ dir_offset + dir_len
//! | footer, 40 bytes                 |
//! +----------------------------------+ end of file
//! ```
//!
//! # Why the footer duplicates what the header says
//!
//! A reader that has just opened a file should be able to find the directory
//! without seeking from the start, and a reader holding only the tail -- the usual
//! case when an archive is handed over rather than opened by name -- should be
//! able to find everything else. So both ends carry the directory's position,
//! and each is checked against the other. A mismatch is a hard error rather than
//! a guess, because a wrong guess means extracting the wrong bytes.
//!
//! # Why integers are little-endian and varints where they can be
//!
//! Archives move between machines, so every fixed-width integer is little-endian
//! rather than host-endian. Directory fields are varints because most of them are
//! small -- a mode, a timestamp, a block count -- and the directory is paid for
//! once per file, so a few bytes each is real money on a tree of a hundred
//! thousand files.
//!
//! # Integrity
//!
//! Every block record carries three independent checks: a 64-bit hash of the
//! *uncompressed* bytes, a CRC of those same bytes, and a CRC of the stored
//! payload. The first two are what make deduplication safe; the third catches a
//! disk that corrupts the archive we just wrote. The directory has a CRC per entry
//! and one over the whole blob.

use crate::checksum::{crc32, Crc32};
use crate::lz;
use crate::magic::FileClass;
use crate::util::{read_u32, read_u64, read_u8, read_varint, write_varint};

/// Format version. Bumped whenever a record layout changes.
///
/// The first public release is `1.0.0-beta.1` and carries format **v1**. The
/// version and the magic both move together, so a reader from a different
/// format generation rejects the archive at the header instead of failing later
/// on a block checksum. Beta archives make no compatibility promise: an older
/// beta may not be readable by this build, and that is intentional.
pub const FORMAT_VERSION: u32 = 1;

/// Start of an archive.
pub const MAGIC: [u8; 8] = *b"D\x01.dnsr\x00";

/// Start of the trailing footer.
pub const FOOTER_MAGIC: [u8; 8] = *b"D\x01END\x00\x00\x00";

pub const HEADER_SIZE: usize = 64;
pub const FOOTER_SIZE: usize = 40;
/// Fixed part of a block record, before its payload.
pub const BLOCK_RECORD_SIZE: usize = 32;

// Header flags.
pub const FLAG_METADATA: u32 = 1 << 0;
pub const FLAG_CHECKSUMS: u32 = 1 << 1;
pub const FLAG_DEDUP: u32 = 1 << 2;
pub const FLAG_RESYNC: u32 = 1 << 3;
pub const FLAG_HARDLINKS: u32 = 1 << 4;
pub const FLAG_CHUNKING_CDC: u32 = 1 << 5;

/// How a block's payload is to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Stored verbatim: the data was incompressible, or already compressed.
    Stored,
    /// An `dsr1` stream -- see [`crate::lz`].
    Coded,
    /// An `dsr1` stream Huffman-coded per block -- see [`crate::huffman`].
    /// Huffman-decode first, then parse exactly like [`BlockKind::Coded`].
    /// Old readers reject this kind; new readers still read kinds 0 and 1.
    CodedHuff,
    /// An executable's bytes run through the branch filter before coding:
    /// `bcj(x86)` then `dsr1`. Decoding reverses that: `dsr1` then the
    /// inverse filter.
    CodedX86,
    /// The same, but the `dsr1` stream is also Huffman-coded. Decoding reverses
    /// it: Huffman, then `dsr1`, then the inverse filter.
    ///
    /// Huffman is a separate kind rather than a flag because it is applied only
    /// when it pays for its own table. A reader that assumed Huffman here would
    /// try to decode a raw `dsr1` stream as a Huffman table and fail.
    CodedX86Huff,
    /// The archive's shared dictionary: a sample of the archive's own bytes,
    /// stored once before the payload blocks and used as history by every block
    /// whose record sets [`BLOCK_FLAG_USES_DICT`]. Its payload is the dictionary
    /// verbatim -- it is compressed data, so coding it again buys nothing.
    Dictionary,
    /// A `dsr1` stream whose literals and tokens were Huffman-coded separately
    /// -- see [`crate::split`]. Decoding joins the two regions back and hands the
    /// result to the same `dsr1` decoder as [`BlockKind::CodedHuff`].
    CodedHuffSplit,
    /// [`BlockKind::CodedX86Huff`] with the two alphabets separated, so decoding
    /// reverses: join the regions, Huffman-decode, then `dsr1`, then the inverse
    /// branch filter.
    CodedX86HuffSplit,
}

/// A block's coded form was produced against the archive's shared dictionary.
///
/// The flag lives in the block record's reserved field, so an archive that uses a
/// dictionary keeps the same record size and the same header. A reader that does
/// not know the flag must fail on it rather than decode without the dictionary:
/// the offsets in such a block reach past the block's own first byte.
pub const BLOCK_FLAG_USES_DICT: u32 = 1 << 0;

impl BlockKind {
    pub fn id(self) -> u8 {
        match self {
            BlockKind::Stored => 0,
            BlockKind::Coded => 1,
            BlockKind::CodedHuff => 2,
            BlockKind::CodedX86 => 3,
            BlockKind::CodedX86Huff => 4,
            BlockKind::Dictionary => 5,
            BlockKind::CodedHuffSplit => 6,
            BlockKind::CodedX86HuffSplit => 7,
        }
    }
    pub fn from_id(id: u8) -> Result<BlockKind, String> {
        match id {
            0 => Ok(BlockKind::Stored),
            1 => Ok(BlockKind::Coded),
            2 => Ok(BlockKind::CodedHuff),
            3 => Ok(BlockKind::CodedX86),
            4 => Ok(BlockKind::CodedX86Huff),
            5 => Ok(BlockKind::Dictionary),
            6 => Ok(BlockKind::CodedHuffSplit),
            7 => Ok(BlockKind::CodedX86HuffSplit),
            other => Err(format!("unknown block kind {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Dictionary payload
// ---------------------------------------------------------------------------
//
// A dictionary block stores the shared dictionary the archive was coded
// against. Its bytes are ordinary input (a sample of the archive's own
// content), so they compress: storing them verbatim wastes the very space the
// dictionary is meant to save. The first byte says how the rest is encoded, and
// the rest is decoded with the same stages a normal block uses. The stage list
// mirrors [`BlockKind`] minus the branch filter: the dictionary is a byte
// sample, not a whole executable, and its samples are already filtered when
// they come from one.

const DICT_VERBATIM: u8 = 0;
const DICT_LZ: u8 = 1;
const DICT_LZ_HUFF: u8 = 2;
const DICT_LZ_HUFF_SPLIT: u8 = 3;

/// Encode a dictionary for storage, picking the smallest of the four forms.
///
/// Returns `[method][body]`. The caller writes the result as the payload of a
/// [`BlockKind::Dictionary`] block whose `raw_len` is `dict.len()`.
pub fn encode_dictionary(dict: &[u8], level: u8) -> Vec<u8> {
    let mut method = DICT_VERBATIM;
    let mut body = dict.to_vec();

    if let Some(c) = lz::compress(dict, level) {
        if c.len() < body.len() {
            method = DICT_LZ;
            body = c;
        }
    }
    if method == DICT_LZ {
        let coded = body.clone();
        if let Some(h) = crate::huffman::compress(&coded) {
            if h.len() < body.len() {
                method = DICT_LZ_HUFF;
                body = h;
            }
        }
        if method == DICT_LZ_HUFF {
            let s = crate::split::compress_split(&coded);
            if s.len() < body.len() {
                method = DICT_LZ_HUFF_SPLIT;
                body = s;
            }
        }
    }

    let mut out = Vec::with_capacity(body.len() + 1);
    out.push(method);
    out.extend_from_slice(&body);
    out
}

/// Decode a [`encode_dictionary`] payload back to the dictionary bytes.
pub fn decode_dictionary(payload: &[u8], raw_len: usize) -> Result<Vec<u8>, String> {
    let (&method, body) = payload
        .split_first()
        .ok_or_else(|| "dictionary payload is empty".to_string())?;
    match method {
        DICT_VERBATIM => {
            if body.len() != raw_len {
                return Err(format!(
                    "stored dictionary is {} bytes, record says {raw_len}",
                    body.len()
                ));
            }
            Ok(body.to_vec())
        }
        DICT_LZ => lz::decompress(body, raw_len),
        DICT_LZ_HUFF => {
            let coded = crate::huffman::decompress(body)?;
            lz::decompress(&coded, raw_len)
        }
        DICT_LZ_HUFF_SPLIT => {
            let coded = crate::split::decompress_split(body)?;
            lz::decompress(&coded, raw_len)
        }
        other => Err(format!("unknown dictionary method {other}")),
    }
}

/// What a directory entry refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    File,
    Directory,
    /// A symbolic link; the target is the entry's metadata.
    Symlink,
    /// A second name for a file already in the archive; the target is the
    /// metadata.
    Hardlink,
}

impl EntryKind {
    pub fn id(self) -> u8 {
        match self {
            EntryKind::File => 0,
            EntryKind::Directory => 1,
            EntryKind::Symlink => 2,
            EntryKind::Hardlink => 3,
        }
    }
    pub fn from_id(id: u8) -> Result<EntryKind, String> {
        match id {
            0 => Ok(EntryKind::File),
            1 => Ok(EntryKind::Directory),
            2 => Ok(EntryKind::Symlink),
            3 => Ok(EntryKind::Hardlink),
            other => Err(format!("unknown entry kind {other}")),
        }
    }
    pub fn is_file(self) -> bool {
        matches!(self, EntryKind::File)
    }
    /// Whether the entry's block list is meaningful.
    pub fn has_blocks(self) -> bool {
        matches!(self, EntryKind::File)
    }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub flags: u32,
    /// The level the payload blocks were produced at. Recorded so that a future
    /// writer can decide whether recompressing is worth it.
    pub level: u8,
    pub block_size: u32,
    pub created: u64,
    pub payload_len: u64,
    pub dir_offset: u64,
    pub dir_len: u32,
    pub dir_crc: u32,
}

impl Header {
    pub fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..8].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&self.version.to_le_bytes());
        b[12..16].copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
        b[16..20].copy_from_slice(&self.flags.to_le_bytes());
        b[20..24].copy_from_slice(&(self.level as u32).to_le_bytes());
        b[24..28].copy_from_slice(&self.block_size.to_le_bytes());
        b[28..32].copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
        b[32..40].copy_from_slice(&self.created.to_le_bytes());
        b[40..48].copy_from_slice(&self.payload_len.to_le_bytes());
        b[48..56].copy_from_slice(&self.dir_offset.to_le_bytes());
        b[56..60].copy_from_slice(&self.dir_len.to_le_bytes());
        b[60..64].copy_from_slice(&self.dir_crc.to_le_bytes());
        b
    }

    pub fn decode(buf: &[u8]) -> Result<Header, String> {
        if buf.len() < HEADER_SIZE {
            return Err(format!(
                "need {HEADER_SIZE} bytes for a header, got {}",
                buf.len()
            ));
        }
        if buf[0..8] != MAGIC {
            return Err("not a Densr archive, or a different format version".into());
        }
        let version = read_u32(buf, &mut 8)?;
        if version != FORMAT_VERSION {
            return Err(format!(
                "archive is format version {version}, this build reads version {FORMAT_VERSION}"
            ));
        }
        let declared = read_u32(buf, &mut 12)?;
        if declared as usize != HEADER_SIZE {
            return Err(format!("header claims to be {declared} bytes"));
        }
        let flags = read_u32(buf, &mut 16)?;
        let level = read_u32(buf, &mut 20)? as u8;
        let block_size = read_u32(buf, &mut 24)?;
        let _reserved = read_u32(buf, &mut 28)?;
        Ok(Header {
            version,
            flags,
            level,
            block_size,
            created: read_u64(buf, &mut 32)?,
            payload_len: read_u64(buf, &mut 40)?,
            dir_offset: read_u64(buf, &mut 48)?,
            dir_len: read_u32(buf, &mut 56)?,
            dir_crc: read_u32(buf, &mut 60)?,
        })
    }
}

impl Default for Header {
    fn default() -> Self {
        Header {
            version: FORMAT_VERSION,
            flags: FLAG_METADATA | FLAG_CHECKSUMS | FLAG_DEDUP,
            level: 4,
            block_size: 64 * 1024,
            created: crate::util::now_unix(),
            payload_len: 0,
            dir_offset: 0,
            dir_len: 0,
            dir_crc: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Block records
// ---------------------------------------------------------------------------

/// One block of file content, as stored in the payload area.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRec {
    /// Bytes of the original data this block expands to.
    pub raw_len: u32,
    pub kind: BlockKind,
    /// [`crate::checksum::hash64`] of the original bytes. Also the dedup key.
    pub hash: u64,
    /// CRC of the original bytes, so a hash collision cannot substitute the wrong
    /// content.
    pub raw_crc: u32,
    /// CRC of the payload as stored.
    pub payload_crc: u32,
    /// Reserved on disk; only [`BLOCK_FLAG_USES_DICT`] is defined.
    pub flags: u32,
}

impl BlockRec {
    pub fn uses_dict(&self) -> bool {
        self.flags & BLOCK_FLAG_USES_DICT != 0
    }
}

impl BlockRec {
    /// Build a record for a block about to be written.
    pub fn new(raw: &[u8], kind: BlockKind, payload: &[u8]) -> BlockRec {
        BlockRec {
            raw_len: raw.len() as u32,
            kind,
            hash: crate::checksum::hash64(raw),
            raw_crc: crc32(raw),
            payload_crc: crc32(payload),
            flags: 0,
        }
    }

    /// Build a record from hashes the caller already computed.
    ///
    /// The writer hashes each block on a worker thread, so rebuilding
    /// them here would hash every block twice.
    pub fn with_hashes(
        raw_len: u32,
        hash: u64,
        raw_crc: u32,
        kind: BlockKind,
        payload: &[u8],
    ) -> BlockRec {
        BlockRec {
            raw_len,
            kind,
            hash,
            raw_crc,
            payload_crc: crc32(payload),
            flags: 0,
        }
    }

    /// [`BlockRec::with_hashes`] with the record's flags set, which is how a
    /// block coded against the shared dictionary records that fact on disk.
    pub fn with_flags(mut self, flags: u32) -> BlockRec {
        self.flags = flags;
        self
    }

    /// The record on disk, without the payload.
    pub fn encode(&self, payload_len: u32) -> [u8; BLOCK_RECORD_SIZE] {
        let mut b = [0u8; BLOCK_RECORD_SIZE];
        b[0..4].copy_from_slice(&payload_len.to_le_bytes());
        b[4..8].copy_from_slice(&self.raw_len.to_le_bytes());
        b[8] = self.kind.id();
        b[12..16].copy_from_slice(&self.flags.to_le_bytes());
        b[16..24].copy_from_slice(&self.hash.to_le_bytes());
        b[24..28].copy_from_slice(&self.raw_crc.to_le_bytes());
        b[28..32].copy_from_slice(&self.payload_crc.to_le_bytes());
        b
    }

    pub fn decode(buf: &[u8]) -> Result<(BlockRec, u32), String> {
        if buf.len() < BLOCK_RECORD_SIZE {
            return Err(format!(
                "need {BLOCK_RECORD_SIZE} bytes for a block record, got {}",
                buf.len()
            ));
        }
        let payload_len = read_u32(buf, &mut 0)?;
        let raw_len = read_u32(buf, &mut 4)?;
        let kind = BlockKind::from_id(read_u8(buf, &mut 8)?)?;
        let flags = read_u32(buf, &mut 12)?;
        if flags & !BLOCK_FLAG_USES_DICT != 0 {
            return Err(format!("unknown block flags {flags:#x}"));
        }
        Ok((
            BlockRec {
                raw_len,
                kind,
                flags,
                hash: read_u64(buf, &mut 16)?,
                raw_crc: read_u32(buf, &mut 24)?,
                payload_crc: read_u32(buf, &mut 28)?,
            },
            payload_len,
        ))
    }

    /// Expand a block's payload back to its original bytes.
    ///
    /// The length is taken from the record rather than trusted from the stream,
    /// which is what bounds the work: a corrupt length cannot make this allocate
    /// without limit, and `lz::decompress` refuses to produce more than it was
    /// promised.
    pub fn expand(&self, payload: &[u8]) -> Result<Vec<u8>, String> {
        self.expand_with_dict(payload, &[])
    }

    /// [`BlockRec::expand`], with the archive's shared dictionary available.
    ///
    /// A block that records [`BLOCK_FLAG_USES_DICT`] must be given the dictionary
    /// it was coded against; its offsets point past the block's first byte, so
    /// decoding it without the dictionary would either fail or produce garbage.
    /// The reverse is not true and is checked rather than assumed.
    pub fn expand_with_dict(&self, payload: &[u8], dict: &[u8]) -> Result<Vec<u8>, String> {
        if self.uses_dict() && dict.is_empty() {
            return Err("block was coded against a shared dictionary that is missing".into());
        }
        if crc32(payload) != self.payload_crc {
            return Err("stored payload failed its CRC".into());
        }
        let raw = match self.kind {
            BlockKind::Stored => payload.to_vec(),
            // The dictionary itself is stored with a one-byte method prefix and
            // coded with the same stages a normal block uses -- see
            // [`encode_dictionary`]. It is decoded without a dictionary of its
            // own; it is the base of the chain.
            BlockKind::Dictionary => decode_dictionary(payload, self.raw_len as usize)?,
            BlockKind::Coded => lz::decompress_with_dict(payload, dict, self.raw_len as usize)?,
            BlockKind::CodedHuff => {
                let coded = crate::huffman::decompress(payload)?;
                lz::decompress_with_dict(&coded, dict, self.raw_len as usize)?
            }
            BlockKind::CodedX86 | BlockKind::CodedX86Huff => {
                // Reverse of the write path. Huffman only when the kind says so.
                let raw = if self.kind == BlockKind::CodedX86Huff {
                    let coded = crate::huffman::decompress(payload)?;
                    let mut r = lz::decompress_with_dict(&coded, dict, self.raw_len as usize)?;
                    crate::bcj::x86(&mut r, false);
                    r
                } else {
                    let mut r = lz::decompress_with_dict(payload, dict, self.raw_len as usize)?;
                    crate::bcj::x86(&mut r, false);
                    r
                };
                raw
            }
            // The split kinds differ only in how the Huffman stage stored the
            // stream. Joining the two alphabet regions produces exactly the
            // `dsr1` stream the plain kinds would have carried, so the decode
            // path below them is the same one.
            BlockKind::CodedHuffSplit => {
                let coded = crate::split::decompress_split(payload)?;
                lz::decompress_with_dict(&coded, dict, self.raw_len as usize)?
            }
            BlockKind::CodedX86HuffSplit => {
                let coded = crate::split::decompress_split(payload)?;
                let mut r = lz::decompress_with_dict(&coded, dict, self.raw_len as usize)?;
                crate::bcj::x86(&mut r, false);
                r
            }
        };
        if raw.len() != self.raw_len as usize {
            return Err(format!(
                "block produced {} bytes, record says {}",
                raw.len(),
                self.raw_len
            ));
        }
        if crc32(&raw) != self.raw_crc {
            return Err("decompressed block failed its CRC".into());
        }
        if crate::checksum::hash64(&raw) != self.hash {
            return Err("decompressed block failed its hash".into());
        }
        Ok(raw)
    }
}

// ---------------------------------------------------------------------------
// Directory
// ---------------------------------------------------------------------------

/// One file, directory, symlink or hardlink in the archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryRec {
    /// Path with forward slashes, relative to the archive root, no leading `./`.
    pub path: String,
    pub kind: EntryKind,
    pub class: FileClass,
    /// Unix mode, or zero when metadata was not stored.
    pub mode: u32,
    /// Modification time in Unix seconds, or zero when not stored.
    pub mtime: u64,
    /// Uncompressed size. For a hardlink this is the target's size.
    pub size: u64,
    /// [`crate::checksum::hash64`] of the whole file, or zero when not stored.
    pub content_hash: u64,
    /// Which blocks, in order. A file repeated in one archive references the same
    /// blocks as the first copy.
    pub blocks: Vec<u32>,
    /// Symlink target, or hardlink target path.
    pub meta: Vec<u8>,
}

impl EntryRec {
    pub fn encode(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(32 + self.path.len() + self.meta.len());
        write_varint(&mut body, self.path.len() as u64);
        body.push(self.kind.id());
        body.push(self.class.id());
        write_varint(&mut body, self.mode as u64);
        write_varint(&mut body, self.mtime);
        write_varint(&mut body, self.size);
        write_varint(&mut body, self.content_hash);
        write_varint(&mut body, self.blocks.len() as u64);
        for b in &self.blocks {
            write_varint(&mut body, *b as u64);
        }
        write_varint(&mut body, self.meta.len() as u64);

        let mut out = Vec::with_capacity(body.len() + self.path.len() + self.meta.len() + 4);
        out.extend_from_slice(&body);
        out.extend_from_slice(self.path.as_bytes());
        out.extend_from_slice(&self.meta);
        // The CRC covers the fields and the variable-length parts together: a
        // name of the wrong length would otherwise silently shift the parse.
        let sum = crc32(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        out
    }

    pub fn decode(buf: &[u8]) -> Result<EntryRec, String> {
        if buf.len() < 4 {
            return Err("directory entry is too short to hold a CRC".into());
        }
        let split = buf.len() - 4;
        let body = &buf[..split];
        if crc32(body)
            != u32::from_le_bytes([buf[split], buf[split + 1], buf[split + 2], buf[split + 3]])
        {
            return Err("directory entry failed its CRC".into());
        }
        let mut pos = 0usize;
        let path_len = read_varint(body, &mut pos)? as usize;
        let kind = EntryKind::from_id(read_u8(body, &mut pos)?)?;
        let class = FileClass::from_id(read_u8(body, &mut pos)?)?;
        let mode = read_varint(body, &mut pos)? as u32;
        let mtime = read_varint(body, &mut pos)?;
        let size = read_varint(body, &mut pos)?;
        let content_hash = read_varint(body, &mut pos)?;
        let n_blocks = read_varint(body, &mut pos)? as usize;
        let mut blocks = Vec::with_capacity(n_blocks.min(4096));
        for _ in 0..n_blocks {
            let id = read_varint(body, &mut pos)?;
            if id > u32::MAX as u64 {
                return Err(format!("block id {id} does not fit in 32 bits"));
            }
            blocks.push(id as u32);
        }
        let meta_len = read_varint(body, &mut pos)? as usize;

        let path_bytes = crate::util::slice(body, &mut pos, path_len)?;
        let name = String::from_utf8(path_bytes.to_vec())
            .map_err(|_| "directory entry has a non-UTF-8 path".to_string())?;
        let meta = crate::util::slice(body, &mut pos, meta_len)?.to_vec();
        if pos != body.len() {
            return Err(format!(
                "directory entry has {} unread bytes",
                body.len() - pos
            ));
        }
        Ok(EntryRec {
            path: name,
            kind,
            class,
            mode,
            mtime,
            size,
            content_hash,
            blocks,
            meta,
        })
    }

    /// A one-line description for `list`.
    pub fn describe(&self, raw_len: u64) -> String {
        let kind = match self.kind {
            EntryKind::File => "-",
            EntryKind::Directory => "d",
            EntryKind::Symlink => "l",
            EntryKind::Hardlink => "h",
        };
        let extra = if self.kind.has_blocks() {
            format!(
                "{:>10}  {} blk  {}",
                crate::util::human(raw_len as f64),
                self.blocks.len(),
                self.class.label()
            )
        } else if let Ok(target) = std::str::from_utf8(&self.meta) {
            format!("{:>10}  -> {}", crate::util::human(0.0), target)
        } else {
            format!("{:>10}", crate::util::human(0.0))
        };
        format!("{kind} {extra}  {}", self.path)
    }
}

/// The set of entries, serialised as one blob.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Directory {
    pub entries: Vec<EntryRec>,
}

impl Directory {
    /// Encode and checksum. The returned CRC covers the whole blob.
    pub fn encode(&self) -> (Vec<u8>, u32) {
        let mut out = Vec::new();
        for e in &self.entries {
            out.extend_from_slice(&e.encode());
        }
        let crc = crc32(&out);
        (out, crc)
    }

    pub fn decode(buf: &[u8]) -> Result<Directory, String> {
        let mut entries = Vec::new();
        let mut pos = 0usize;
        while pos < buf.len() {
            // Each record ends with a four-byte CRC, so a truncated tail is
            // detectable: find the record's own end by trial rather than by a
            // length prefix, which would be a second thing to keep in sync.
            let rec = decode_one(buf, &mut pos)?;
            entries.push(rec);
        }
        Ok(Directory { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn get(&self, path: &str) -> Option<&EntryRec> {
        self.entries.iter().find(|e| e.path == path)
    }
}

/// Decode one entry, advancing `pos` past it.
fn decode_one(buf: &[u8], pos: &mut usize) -> Result<EntryRec, String> {
    // Walk candidate end offsets: the record is `body || path || meta || crc`, so
    // each plausible split is tried and the CRC decides.
    for split in (*pos + 4)..=buf.len() {
        let candidate = &buf[*pos..split];
        if let Ok(e) = EntryRec::decode(candidate) {
            *pos = split;
            return Ok(e);
        }
    }
    Err(format!(
        "no valid directory entry ends at offset {pos}..{}",
        buf.len()
    ))
}

// ---------------------------------------------------------------------------
// Footer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footer {
    pub dir_offset: u64,
    pub dir_len: u32,
    pub dir_crc: u32,
    pub archive_len: u64,
}

impl Footer {
    pub fn encode(&self) -> [u8; FOOTER_SIZE] {
        let mut b = [0u8; FOOTER_SIZE];
        b[0..8].copy_from_slice(&FOOTER_MAGIC);
        b[8..16].copy_from_slice(&self.dir_offset.to_le_bytes());
        b[16..20].copy_from_slice(&self.dir_len.to_le_bytes());
        b[20..24].copy_from_slice(&self.dir_crc.to_le_bytes());
        b[24..28].copy_from_slice(&(FOOTER_SIZE as u32).to_le_bytes());
        b[28..36].copy_from_slice(&self.archive_len.to_le_bytes());
        b
    }

    pub fn decode(buf: &[u8]) -> Result<Footer, String> {
        if buf.len() < FOOTER_SIZE {
            return Err(format!(
                "need {FOOTER_SIZE} bytes for a footer, got {}",
                buf.len()
            ));
        }
        if buf[0..8] != FOOTER_MAGIC {
            return Err("archive does not end with a Densr footer".into());
        }
        let dir_offset = read_u64(buf, &mut 8)?;
        let dir_len = read_u32(buf, &mut 16)?;
        let dir_crc = read_u32(buf, &mut 20)?;
        let declared = read_u32(buf, &mut 24)?;
        if declared as usize != FOOTER_SIZE {
            return Err(format!("footer claims to be {declared} bytes"));
        }
        Ok(Footer {
            dir_offset,
            dir_len,
            dir_crc,
            archive_len: read_u64(buf, &mut 28)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// A finished archive held in memory, ready to be written.
#[derive(Debug, Clone)]
pub struct BuiltArchive {
    pub header: Header,
    pub payload: Vec<u8>,
    pub directory: Directory,
}

impl BuiltArchive {
    /// Lay out header, payload, directory and footer, and return the bytes.
    ///
    /// The header is written twice: once as a placeholder so the payload can be
    /// laid out at a known offset, then again with the directory's real position.
    pub fn into_bytes(self) -> Vec<u8> {
        let BuiltArchive {
            mut header,
            payload,
            directory,
        } = self;

        let (dir_blob, dir_crc) = directory.encode();
        let dir_offset = (HEADER_SIZE + payload.len()) as u64;
        header.payload_len = payload.len() as u64;
        header.dir_offset = dir_offset;
        header.dir_len = dir_blob.len() as u32;
        header.dir_crc = dir_crc;

        let archive_len = (dir_offset as usize) + dir_blob.len() + FOOTER_SIZE;
        let footer = Footer {
            dir_offset,
            dir_len: dir_blob.len() as u32,
            dir_crc,
            archive_len: archive_len as u64,
        };

        let mut out = Vec::with_capacity(archive_len);
        out.extend_from_slice(&header.encode());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&dir_blob);
        out.extend_from_slice(&footer.encode());
        debug_assert_eq!(out.len(), archive_len);
        out
    }

    /// Check that the directory's CRC is consistent, used after reading.
    pub fn verify_directory(dir_blob: &[u8], expected: u32) -> Result<(), String> {
        let mut c = Crc32::new();
        c.update(dir_blob);
        let actual = c.finish();
        if actual != expected {
            return Err(format!(
                "directory checksum {actual:#010x} does not match the recorded {expected:#010x}"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry() -> EntryRec {
        EntryRec {
            path: "src/main.rs".into(),
            kind: EntryKind::File,
            class: FileClass::Code,
            mode: 0o644,
            mtime: 1_700_000_000,
            size: 4096,
            content_hash: 0x0123_4567_89ab_cdef,
            blocks: vec![0, 1, 2, 700],
            meta: Vec::new(),
        }
    }

    #[test]
    fn header_roundtrips() {
        let h = Header {
            level: 7,
            block_size: 128 * 1024,
            payload_len: 999_999,
            dir_offset: 1_234_567,
            dir_len: 4321,
            dir_crc: 0xdead_beef,
            ..Header::default()
        };
        let bytes = h.encode();
        assert_eq!(bytes.len(), HEADER_SIZE);
        assert_eq!(&bytes[0..8], &MAGIC);
        assert_eq!(Header::decode(&bytes).unwrap(), h);
    }

    #[test]
    fn header_rejects_a_foreign_file() {
        let mut bytes = Header::default().encode();
        bytes[1] = b'X';
        assert!(Header::decode(&bytes).is_err());
        assert!(Header::decode(&[0u8; 10]).is_err());
    }

    #[test]
    fn header_rejects_a_different_version() {
        let mut bytes = Header::default().encode();
        bytes[8] = FORMAT_VERSION.wrapping_add(1) as u8;
        let err = Header::decode(&bytes).unwrap_err();
        assert!(err.contains("version"), "{err}");
    }

    #[test]
    fn block_record_roundtrips_and_expands() {
        let raw: Vec<u8> = b"the quick brown fox jumps over the lazy dog".repeat(20);
        let level = 4u8;
        let payload = lz::compress(&raw, level).expect("should compress");
        let rec = BlockRec::new(&raw, BlockKind::Coded, &payload);
        let encoded = rec.encode(payload.len() as u32);

        let (back, payload_len) = BlockRec::decode(&encoded).unwrap();
        assert_eq!(back, rec);
        assert_eq!(payload_len as usize, payload.len());
        assert_eq!(back.expand(&payload).unwrap(), raw);
    }

    #[test]
    fn block_record_expands_stored_data() {
        let raw = b"already compressed bytes".to_vec();
        let rec = BlockRec::new(&raw, BlockKind::Stored, &raw);
        assert_eq!(rec.expand(&raw).unwrap(), raw);
    }

    #[test]
    fn block_record_notices_corruption() {
        let raw: Vec<u8> = b"some compressible text ".repeat(50);
        let payload = lz::compress(&raw, 3).expect("compress");
        let rec = BlockRec::new(&raw, BlockKind::Coded, &payload);

        // Flipping a payload bit must be caught before anything tries to decode.
        let mut bad = payload.clone();
        let mid = bad.len() / 2;
        bad[mid] ^= 0x01;
        assert!(rec.expand(&bad).is_err(), "corrupt payload was accepted");
        // So must a record whose hash does not describe the data.
        let mut lying = rec.clone();
        lying.hash ^= 1;
        assert!(lying.expand(&payload).is_err(), "wrong hash was accepted");
        let mut lying = rec.clone();
        lying.raw_crc ^= 1;
        assert!(lying.expand(&payload).is_err(), "wrong CRC was accepted");
    }

    #[test]
    fn entry_roundtrips() {
        let e = sample_entry();
        let bytes = e.encode();
        assert_eq!(EntryRec::decode(&bytes).unwrap(), e);
    }

    #[test]
    fn entry_roundtrips_at_every_field_boundary() {
        // Zero and maximum values, because varint encoding is where a field is
        // most likely to be read at the wrong width.
        for (path_len, blocks, meta_len) in [
            (0usize, 0usize, 0usize),
            (1, 0, 0),
            (200, 300, 10),
            (70_000, 1, 1),
        ] {
            let e = EntryRec {
                path: "x".repeat(path_len),
                kind: EntryKind::File,
                class: FileClass::Data,
                mode: if path_len % 2 == 0 { 0 } else { 0o7777 },
                mtime: if path_len % 2 == 0 { 0 } else { u64::MAX },
                size: if path_len % 2 == 0 { 0 } else { u64::MAX },
                content_hash: if path_len % 2 == 0 { 0 } else { u64::MAX },
                blocks: (0..blocks).map(|i| i as u32).collect(),
                meta: vec![b'm'; meta_len],
            };
            let bytes = e.encode();
            assert_eq!(EntryRec::decode(&bytes).unwrap(), e, "path {path_len}");
        }
    }

    #[test]
    fn entry_rejects_corruption() {
        let mut bytes = sample_entry().encode();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        assert!(
            EntryRec::decode(&bytes).is_err(),
            "corrupt entry was accepted"
        );
        // Truncation must fail rather than decode to something shorter.
        let full = sample_entry().encode();
        assert!(EntryRec::decode(&full[..full.len() - 1]).is_err());
    }

    #[test]
    fn directory_roundtrips() {
        let dir = Directory {
            entries: (0..50)
                .map(|i| EntryRec {
                    path: format!("dir{}/file{i}.txt", i % 7),
                    blocks: vec![i as u32; i % 4],
                    size: i as u64 * 1000,
                    ..sample_entry()
                })
                .collect(),
        };
        let (blob, crc) = dir.encode();
        BuiltArchive::verify_directory(&blob, crc).unwrap();
        assert!(BuiltArchive::verify_directory(&blob, crc ^ 1).is_err());
        assert_eq!(Directory::decode(&blob).unwrap(), dir);
    }

    #[test]
    fn directory_rejects_a_corrupted_blob() {
        let dir = Directory {
            entries: (0..20)
                .map(|i| EntryRec {
                    path: format!("f{i}"),
                    ..sample_entry()
                })
                .collect(),
        };
        let (mut blob, crc) = dir.encode();
        blob[30] ^= 0xFF;
        assert!(BuiltArchive::verify_directory(&blob, crc).is_err());
        // And a truncated blob must not decode to a shorter valid directory.
        blob.truncate(blob.len() - 2);
        assert!(Directory::decode(&blob).is_err());
    }

    #[test]
    fn footer_roundtrips() {
        let f = Footer {
            dir_offset: 12345,
            dir_len: 678,
            dir_crc: 0xabcd_ef01,
            archive_len: 987_654,
        };
        let bytes = f.encode();
        assert_eq!(bytes.len(), FOOTER_SIZE);
        assert_eq!(Footer::decode(&bytes).unwrap(), f);
        let mut bad = bytes;
        bad[2] = b'Z';
        assert!(Footer::decode(&bad).is_err());
    }

    #[test]
    fn archive_assembly_places_everything_where_the_records_say() {
        let payload: Vec<u8> = vec![0xAB; 100];
        let archive = BuiltArchive {
            header: Header::default(),
            payload: payload.clone(),
            directory: Directory {
                entries: vec![sample_entry()],
            },
        };
        let bytes = archive.into_bytes();

        let header = Header::decode(&bytes).unwrap();
        let footer = Footer::decode(&bytes[bytes.len() - FOOTER_SIZE..]).unwrap();
        assert_eq!(header.dir_offset, footer.dir_offset);
        assert_eq!(header.dir_len, footer.dir_len);
        assert_eq!(header.dir_crc, footer.dir_crc);
        assert_eq!(footer.archive_len, bytes.len() as u64);
        assert_eq!(
            &bytes[HEADER_SIZE..HEADER_SIZE + payload.len()],
            &payload[..],
            "payload moved"
        );
        let dir_blob = &bytes
            [header.dir_offset as usize..(header.dir_offset + header.dir_len as u64) as usize];
        BuiltArchive::verify_directory(dir_blob, header.dir_crc).unwrap();
        assert_eq!(Directory::decode(dir_blob).unwrap().len(), 1);
    }

    #[test]
    fn entry_kind_ids_round_trip() {
        for k in [
            EntryKind::File,
            EntryKind::Directory,
            EntryKind::Symlink,
            EntryKind::Hardlink,
        ] {
            assert_eq!(EntryKind::from_id(k.id()).unwrap(), k);
        }
        assert!(EntryKind::from_id(200).is_err());
        assert!(EntryKind::Directory.has_blocks() == false);
        assert!(EntryKind::File.has_blocks());
    }

    #[test]
    fn format_time_is_used_for_display() {
        assert_eq!(crate::util::format_time(0), "1970-01-01 00:00:00 UTC");
    }

    #[test]
    fn dictionary_payload_round_trips_in_every_shape() {
        let text: Vec<u8> = b"function prologue mov rbp rsp push rbx ".repeat(400);
        let noise: Vec<u8> = (0..8192u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        let empty: Vec<u8> = Vec::new();
        for (name, data) in [("text", &text), ("noise", &noise), ("empty", &empty)] {
            let payload = encode_dictionary(data, 4);
            assert!(!payload.is_empty(), "{name}: no payload");
            let back = decode_dictionary(&payload, data.len()).unwrap();
            assert_eq!(&back, data, "{name} did not round-trip");
        }
    }

    #[test]
    fn a_dictionary_block_decodes_its_payload() {
        let dict: Vec<u8> = b"shared dictionary body ".repeat(300);
        let payload = encode_dictionary(&dict, 4);
        let rec = BlockRec::with_hashes(
            dict.len() as u32,
            crate::checksum::hash64(&dict),
            crate::checksum::crc32(&dict),
            BlockKind::Dictionary,
            &payload,
        );
        // The dictionary is the base of the chain: it decodes without one.
        assert_eq!(rec.expand_with_dict(&payload, &[]).unwrap(), dict);
    }
}
