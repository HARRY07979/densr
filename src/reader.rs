//! Reading an archive: listing, testing, extracting.
//!
//! # What the reader trusts
//!
//! The layout is validated against the header and the footer, but the *contents*
//! are validated per block. Each record carries a CRC of the block as it was
//! before compression and a CRC of the payload as stored, and both are checked.
//!
//! The reason to check both is that they fail differently. The payload CRC
//! catches corruption on disk or in transit. The raw CRC catches a codec that
//! decoded to something of the right length but the wrong content -- a flipped
//! bit that turned one literal into a different literal, with the payload itself
//! perfectly intact.
//!
//! Nothing is trusted far enough to matter: block indices are bounds-checked
//! before use, sizes are cross-checked against the blocks they point at, and
//! every archive path is re-checked on extraction.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::checksum::{crc32, hash64};
use crate::format::{
    BlockKind, BlockRec, Directory, EntryKind, EntryRec, Footer, Header, BLOCK_RECORD_SIZE,
    FOOTER_SIZE, HEADER_SIZE,
};

/// One entry as it is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub path: String,
    pub kind: EntryKind,
    pub class: crate::magic::FileClass,
    pub mode: u32,
    pub mtime: u64,
    pub size: u64,
    pub content_hash: u64,
    pub blocks: Vec<u32>,
    pub meta: Vec<u8>,
}

impl Entry {
    fn from_rec(r: &EntryRec) -> Entry {
        Entry {
            path: r.path.clone(),
            kind: r.kind,
            class: r.class,
            mode: r.mode,
            mtime: r.mtime,
            size: r.size,
            content_hash: r.content_hash,
            blocks: r.blocks.clone(),
            meta: r.meta.clone(),
        }
    }

    pub fn is_file(&self) -> bool {
        self.kind == EntryKind::File
    }
}

/// A file's content, read out of the archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryData {
    pub entry: Entry,
    pub data: Vec<u8>,
}

/// Where a block sits in the file, found by scanning once at open time.
#[derive(Debug, Clone, Copy)]
struct BlockLoc {
    offset: u64,
    payload_len: u32,
}

/// An open archive.
///
/// The block index is built by a single sequential pass at open time. That costs
/// one read of the record area and makes every later access a seek, which is the
/// right trade for an archive that will be read more than once.
pub struct Archive {
    file: File,
    path: PathBuf,
    header: Header,
    footer: Footer,
    directory: Directory,
    blocks: HashMap<u32, BlockLoc>,
    /// Each block's identity, so an entry's blocks can be checked for
    /// consistency without decoding them.
    sizes: HashMap<u32, u32>,
    /// The archive's shared dictionary, loaded from block 0 the first time a
    /// block that needs it is read. Not decoded at open: an archive that is only
    /// listed or tested cheaply never pays for it.
    dict: Option<Vec<u8>>,
    /// The block index of the dictionary, once its record has been seen.
    dict_block: Option<u32>,
}

impl Archive {
    /// Open and validate the structure of an archive.
    pub fn open(path: &Path) -> Result<Archive, String> {
        let mut file =
            File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();

        // The header sits at a fixed offset, so it can be read before anything
        // else is known.
        let mut head = [0u8; HEADER_SIZE];
        file.seek(SeekFrom::Start(0))
            .and_then(|_| file.read_exact(&mut head))
            .map_err(|e| format!("{} is too short to hold a header: {e}", path.display()))?;
        let header = Header::decode(&head)?;

        if file_len < (HEADER_SIZE + FOOTER_SIZE) as u64 {
            return Err(format!(
                "{} is {file_len} bytes, too short to be an archive",
                path.display()
            ));
        }

        // The footer says where the directory is. It is checked against the
        // header, because either could be wrong and the two have to agree.
        let mut foot = [0u8; FOOTER_SIZE];
        file.seek(SeekFrom::Start(file_len - FOOTER_SIZE as u64))
            .and_then(|_| file.read_exact(&mut foot))
            .map_err(|e| format!("{}: cannot read the footer: {e}", path.display()))?;
        let footer = Footer::decode(&foot)?;

        if footer.archive_len != file_len {
            return Err(format!(
                "{}: the footer says the archive is {} bytes but the file is {file_len}",
                path.display(),
                footer.archive_len
            ));
        }
        if header.dir_offset != footer.dir_offset
            || header.dir_len != footer.dir_len
            || header.dir_crc != footer.dir_crc
        {
            return Err(format!(
                "{}: the header and the footer disagree about the directory",
                path.display()
            ));
        }
        let dir_end = header
            .dir_offset
            .checked_add(header.dir_len as u64)
            .ok_or_else(|| format!("{}: the directory offset overflows", path.display()))?;
        if dir_end > file_len - FOOTER_SIZE as u64 {
            return Err(format!(
                "{}: the directory runs past the end of the archive",
                path.display()
            ));
        }

        let mut dir_blob = vec![0u8; header.dir_len as usize];
        file.seek(SeekFrom::Start(header.dir_offset))
            .and_then(|_| file.read_exact(&mut dir_blob))
            .map_err(|e| format!("{}: cannot read the directory: {e}", path.display()))?;
        crate::format::BuiltArchive::verify_directory(&dir_blob, header.dir_crc)?;
        let directory = Directory::decode(&dir_blob)?;

        let mut ar = Archive {
            file,
            path: path.to_path_buf(),
            header,
            footer,
            directory,
            blocks: HashMap::new(),
            sizes: HashMap::new(),
            dict: None,
            dict_block: None,
        };
        ar.scan_blocks()?;
        Ok(ar)
    }

    /// Walk the block records once and note where each block lives.
    fn scan_blocks(&mut self) -> Result<(), String> {
        let end = self.header.dir_offset;
        let mut at = HEADER_SIZE as u64;
        let mut next: u32 = 0;
        let mut rec = [0u8; BLOCK_RECORD_SIZE];

        while at < end {
            if at + BLOCK_RECORD_SIZE as u64 > end {
                return Err(format!(
                    "{}: a block record at {at} runs into the directory",
                    self.path.display()
                ));
            }
            self.file
                .seek(SeekFrom::Start(at))
                .and_then(|_| self.file.read_exact(&mut rec))
                .map_err(|e| format!("{}: block {next} at {at}: {e}", self.path.display()))?;
            let (info, payload_len) = BlockRec::decode(&rec)
                .map_err(|e| format!("{}: block {next} at {at}: {e}", self.path.display()))?;

            let payload_at = at + BLOCK_RECORD_SIZE as u64;
            if payload_at + payload_len as u64 > end {
                return Err(format!(
                    "{}: block {next}'s payload runs past the directory",
                    self.path.display()
                ));
            }
            if next == u32::MAX {
                return Err(format!("{}: too many blocks", self.path.display()));
            }
            if info.kind == BlockKind::Dictionary {
                if self.dict_block.is_some() {
                    return Err(format!(
                        "{}: a second shared dictionary block at {at}",
                        self.path.display()
                    ));
                }
                self.dict_block = Some(next);
            }
            self.sizes.insert(next, info.raw_len);
            self.blocks.insert(
                next,
                BlockLoc {
                    offset: payload_at,
                    payload_len,
                },
            );
            at = payload_at + payload_len as u64;
            next += 1;
        }

        if at != end {
            return Err(format!(
                "{}: the blocks end at {at} but the directory starts at {end}",
                self.path.display()
            ));
        }

        let mut seen: HashSet<&str> = HashSet::new();
        for e in self.directory.entries.iter() {
            if !seen.insert(e.path.as_str()) {
                return Err(format!(
                    "{}: {:?} appears twice in the directory",
                    self.path.display(),
                    e.path
                ));
            }
            // Checked here as well as on extraction. An archive can arrive from
            // anywhere, so the writer's promise is not evidence.
            safe_relative(&e.path).map_err(|why| format!("{}: {why}", self.path.display()))?;

            let mut total: u64 = 0;
            for &b in &e.blocks {
                let raw_len = *self.sizes.get(&b).ok_or_else(|| {
                    format!(
                        "{}: {:?} refers to block {b}, but only {next} exist",
                        self.path.display(),
                        e.path
                    )
                })?;
                total += raw_len as u64;
            }
            if e.kind.has_blocks() && total != e.size {
                return Err(format!(
                    "{}: {:?} claims {} bytes but its blocks hold {total}",
                    self.path.display(),
                    e.path,
                    e.size
                ));
            }
        }
        Ok(())
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
    pub fn footer(&self) -> &Footer {
        &self.footer
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn entries(&self) -> Vec<Entry> {
        self.directory.entries.iter().map(Entry::from_rec).collect()
    }
    pub fn get(&self, path: &str) -> Option<Entry> {
        self.directory.get(path).map(Entry::from_rec)
    }
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Count the stored blocks per `BlockKind`.
    ///
    /// Reads only the fixed-size records, never a payload, so this is cheap
    /// enough to run inside `info`. It is the only way to tell from the outside
    /// which transformations a write actually applied -- a block filter that
    /// silently stopped matching its gate would otherwise look identical.
    pub fn block_kind_counts(&mut self) -> Result<crate::writer::KindCounts, String> {
        let mut counts = crate::writer::KindCounts::default();
        let locs: Vec<(u64, u32)> = self
            .blocks
            .values()
            .map(|l| (l.offset, l.payload_len))
            .collect();
        for (offset, payload_len) in locs {
            let mut rec = [0u8; BLOCK_RECORD_SIZE];
            self.file
                .seek(SeekFrom::Start(offset - BLOCK_RECORD_SIZE as u64))
                .and_then(|_| self.file.read_exact(&mut rec))
                .map_err(|e| format!("cannot read a block record: {e}"))?;
            let (info, _) = BlockRec::decode(&rec)?;
            counts.record(info.kind, info.raw_len, payload_len as usize);
        }
        Ok(counts)
    }

    fn read_payload(&mut self, index: u32) -> Result<(BlockRec, Vec<u8>), String> {
        let loc = *self
            .blocks
            .get(&index)
            .ok_or_else(|| format!("block {index} does not exist"))?;
        let mut payload = vec![0u8; loc.payload_len as usize];
        self.file
            .seek(SeekFrom::Start(loc.offset))
            .and_then(|_| self.file.read_exact(&mut payload))
            .map_err(|e| format!("block {index}: {e}"))?;

        let mut rec = [0u8; BLOCK_RECORD_SIZE];
        self.file
            .seek(SeekFrom::Start(loc.offset - BLOCK_RECORD_SIZE as u64))
            .and_then(|_| self.file.read_exact(&mut rec))
            .map_err(|e| format!("block {index}: cannot read its record: {e}"))?;
        let (info, _) = BlockRec::decode(&rec)?;
        Ok((info, payload))
    }

    /// The archive's shared dictionary, read and verified on first use.
    ///
    /// A block that points into the dictionary cannot be decoded without it, so a
    /// missing or damaged dictionary block is an error rather than a fallback.
    fn dict(&mut self) -> Result<&[u8], String> {
        if self.dict.is_none() {
            let index = self.dict_block.ok_or_else(|| {
                format!(
                    "{}: a block needs the shared dictionary, which is missing",
                    self.path.display()
                )
            })?;
            let (info, payload) = self.read_payload(index)?;
            let bytes = info.expand_with_dict(&payload, &[]).map_err(|e| {
                format!(
                    "{}: shared dictionary block {index}: {e}",
                    self.path.display()
                )
            })?;
            self.dict = Some(bytes);
        }
        Ok(self.dict.as_deref().unwrap_or(&[]))
    }

    /// Read and verify one block, returning its original bytes.
    pub fn block(&mut self, index: u32) -> Result<Vec<u8>, String> {
        let (info, payload) = self.read_payload(index)?;
        if crc32(&payload) != info.payload_crc {
            return Err(format!(
                "block {index}: payload CRC mismatch -- the stored bytes are damaged"
            ));
        }
        let empty: Vec<u8> = Vec::new();
        let dict: &[u8] = if info.uses_dict() {
            self.dict()?
        } else {
            &empty
        };
        let raw = info
            .expand_with_dict(&payload, dict)
            .map_err(|e| format!("block {index}: {e}"))?;
        if raw.len() as u32 != info.raw_len {
            return Err(format!(
                "block {index}: decoded to {} bytes but the record says {}",
                raw.len(),
                info.raw_len
            ));
        }
        if crc32(&raw) != info.raw_crc {
            return Err(format!(
                "block {index}: content CRC mismatch -- it decoded to the wrong bytes"
            ));
        }
        if hash64(&raw) != info.hash {
            return Err(format!(
                "block {index}: hash mismatch -- the content does not match its identity"
            ));
        }
        Ok(raw)
    }

    /// Read one file, verifying every block and the whole-file hash.
    pub fn read(&mut self, path: &str) -> Result<EntryData, String> {
        let rec = self
            .directory
            .get(path)
            .ok_or_else(|| format!("{path:?} is not in the archive"))?
            .clone();
        if rec.kind != EntryKind::File {
            return Err(format!("{path:?} is not a file"));
        }
        let mut data = Vec::with_capacity(rec.size as usize);
        for &b in &rec.blocks {
            data.extend_from_slice(&self.block(b)?);
        }
        if data.len() as u64 != rec.size {
            return Err(format!(
                "{path:?}: read {} bytes but the directory says {}",
                data.len(),
                rec.size
            ));
        }
        // The per-file hash catches a block list that reassembles into the right
        // length but the wrong bytes -- for instance two files' blocks swapped.
        if rec.content_hash != 0 && hash64(&data) != rec.content_hash {
            return Err(format!("{path:?}: the reassembled file fails its hash"));
        }
        Ok(EntryData {
            entry: Entry::from_rec(&rec),
            data,
        })
    }

    /// Check every block and every file without writing anything.
    ///
    /// This reads the archive the way a reader would, so it answers "is this
    /// archive intact" rather than "are the numbers in the header consistent".
    pub fn test(&mut self) -> Result<Report, String> {
        let mut r = Report::default();
        for e in self.directory.entries.clone() {
            r.entries += 1;
            match e.kind {
                EntryKind::File => {
                    let d = self.read(&e.path)?;
                    r.files += 1;
                    r.blocks += d.entry.blocks.len();
                    r.logical += d.data.len() as u64;
                }
                EntryKind::Symlink => {
                    if e.meta.is_empty() {
                        return Err(format!("{}: a symlink with no target", e.path));
                    }
                    let t = String::from_utf8_lossy(&e.meta).into_owned();
                    safe_relative(&t).map_err(|why| format!("{}: {why}", e.path))?;
                    r.symlinks += 1;
                }
                EntryKind::Directory => r.dirs += 1,
                EntryKind::Hardlink => {
                    if e.meta.is_empty() {
                        return Err(format!("{}: a hardlink with no target", e.path));
                    }
                    let target = String::from_utf8_lossy(&e.meta).into_owned();
                    safe_relative(&target).map_err(|why| format!("{}: {why}", e.path))?;
                    if self.directory.get(&target).is_none() {
                        return Err(format!(
                            "{}: hardlink target {target:?} is not in the archive",
                            e.path
                        ));
                    }
                }
            }
        }
        Ok(r)
    }

    /// Extract everything under `dest`.
    ///
    /// Payloads are read once, in ascending file order. Workers then decode each
    /// file's blocks and write it immediately, so decoding overlaps disk writes
    /// and no worker ever reopens or seeks the archive. Verification is
    /// unchanged: `expand_with_dict` checks payload CRC, decoded length, raw CRC
    /// and hash per block, and every file still checks its whole-file hash
    /// unless a one-block file's verified block hash already is that hash.
    pub fn extract_to(&mut self, dest: &Path) -> io::Result<u64> {
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};

        fs::create_dir_all(dest)?;
        // Resolve once: if `dest` is a symlink to somewhere else, every path
        // built from it must still land inside.
        let root = dest.canonicalize().unwrap_or_else(|_| dest.to_path_buf());
        let entries = self.directory.entries.clone();
        let mut files: Vec<(Entry, PathBuf)> = Vec::new();
        let mut hardlinks: Vec<(Entry, PathBuf, PathBuf)> = Vec::new();

        for e in &entries {
            let rel = safe_relative(&e.path).map_err(io::Error::other)?;
            let target = root.join(&rel);
            if !target.starts_with(&root) {
                return Err(io::Error::other(format!(
                    "{}: resolves outside the extraction root",
                    e.path
                )));
            }
            match e.kind {
                EntryKind::Directory => {
                    fs::create_dir_all(&target)?;
                }
                EntryKind::File => {
                    if let Some(p) = target.parent() {
                        fs::create_dir_all(p)?;
                    }
                    files.push((Entry::from_rec(e), target));
                }
                EntryKind::Symlink => {
                    let t = String::from_utf8_lossy(&e.meta).into_owned();
                    let link = safe_relative(&t).map_err(io::Error::other)?;
                    #[cfg(unix)]
                    {
                        let _ = fs::remove_file(&target);
                        std::os::unix::fs::symlink(&link, &target)?;
                    }
                    #[cfg(not(unix))]
                    {
                        // Windows wants the target type spelled out, and creating
                        // a symlink needs a privilege most shells do not have.
                        // Writing a file that names the target keeps the
                        // information instead of failing the whole extraction.
                        let _ = fs::remove_file(&target);
                        fs::write(&target, link.to_string_lossy().as_bytes())?;
                    }
                }
                EntryKind::Hardlink => {
                    let t = String::from_utf8_lossy(&e.meta).into_owned();
                    let rel = safe_relative(&t).map_err(io::Error::other)?;
                    let source = root.join(&rel);
                    if !source.starts_with(&root) {
                        return Err(io::Error::other(format!(
                            "{}: hardlink target resolves outside the extraction root",
                            e.path
                        )));
                    }
                    if let Some(p) = target.parent() {
                        fs::create_dir_all(p)?;
                    }
                    hardlinks.push((Entry::from_rec(e), target, source));
                }
            }
        }

        if files.is_empty() && hardlinks.is_empty() {
            return Ok(0);
        }

        let threads = std::env::var("DENSR_EXTRACT_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
            .clamp(1, 8);

        // The dictionary is a normal block but must be available before any
        // other block can be decoded, so verify it here first.
        let dict_block = self.dict_block;
        let dict = match dict_block {
            Some(index) => {
                let (info, payload) = self.read_payload(index).map_err(io::Error::other)?;
                if crc32(&payload) != info.payload_crc {
                    return Err(io::Error::other(format!(
                        "block {index}: payload CRC mismatch -- the stored bytes are damaged"
                    )));
                }
                let raw = info
                    .expand_with_dict(&payload, &[])
                    .map_err(io::Error::other)?;
                if raw.len() as u32 != info.raw_len
                    || crc32(&raw) != info.raw_crc
                    || hash64(&raw) != info.hash
                {
                    return Err(io::Error::other(format!(
                        "dictionary block {index}: failed verification"
                    )));
                }
                raw
            }
            None => Vec::new(),
        };

        // Read every payload in offset order, before any worker competes for the
        // handle. This is the only pass that touches block data on disk.
        let mut order: Vec<u32> = self.blocks.keys().copied().collect();
        order.sort_unstable_by_key(|i| self.blocks[i].offset);
        let mut jobs = Vec::with_capacity(order.len());
        for index in order {
            let (info, payload) = self.read_payload(index).map_err(io::Error::other)?;
            jobs.push((index, info, payload));
        }

        // Move the payloads into an immutable map. Workers decode each file's
        // blocks and write it immediately, so decoding overlaps disk writes
        // instead of forming a separate phase.
        let blocks: Arc<HashMap<u32, (BlockRec, Vec<u8>)>> = Arc::new(
            jobs.into_iter()
                .map(|(index, info, payload)| (index, (info, payload)))
                .collect(),
        );
        let next = AtomicUsize::new(0);
        let written = AtomicU64::new(0);
        let failure: Mutex<Option<String>> = Mutex::new(None);
        let workers = threads.min(files.len());

        std::thread::scope(|scope| {
            for _ in 0..workers {
                let files = &files;
                let blocks = &blocks;
                let dict = &dict;
                let next = &next;
                let written = &written;
                let failure = &failure;
                scope.spawn(move || loop {
                    if failure.lock().unwrap().is_some() {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((entry, target)) = files.get(i) else {
                        return;
                    };
                    let result = (|| -> Result<(), String> {
                        let mut data;
                        let mut single_block_hash = 0u64;
                        if entry.blocks.len() == 1 {
                            // One block is the whole file: keep its decoded
                            // buffer instead of copying it, and its verified block
                            // hash is the file hash too.
                            let b = entry.blocks[0];
                            let (info, payload) = blocks
                                .get(&b)
                                .ok_or_else(|| format!("{}: block {b} is missing", entry.path))?;
                            let history: &[u8] = if Some(b) == dict_block {
                                &[]
                            } else if info.uses_dict() {
                                dict
                            } else {
                                &[]
                            };
                            data = info
                                .expand_with_dict(payload, history)
                                .map_err(|e| format!("block {b}: {e}"))?;
                            single_block_hash = info.hash;
                        } else {
                            data = Vec::with_capacity(entry.size as usize);
                            for &b in &entry.blocks {
                                let (info, payload) = blocks.get(&b).ok_or_else(|| {
                                    format!("{}: block {b} is missing", entry.path)
                                })?;
                                // `expand_with_dict` performs the payload CRC,
                                // raw CRC, length and hash checks itself.
                                // Repeating them here only doubled verification
                                // CPU.
                                let history: &[u8] = if Some(b) == dict_block {
                                    &[]
                                } else if info.uses_dict() {
                                    dict
                                } else {
                                    &[]
                                };
                                let raw = info
                                    .expand_with_dict(payload, history)
                                    .map_err(|e| format!("block {b}: {e}"))?;
                                data.extend_from_slice(&raw);
                            }
                        }
                        if data.len() as u64 != entry.size {
                            return Err(format!(
                                "{}: read {} bytes but the directory says {}",
                                entry.path,
                                data.len(),
                                entry.size
                            ));
                        }
                        if entry.content_hash != 0
                            && entry.content_hash != single_block_hash
                            && hash64(&data) != entry.content_hash
                        {
                            return Err(format!(
                                "{}: the reassembled file fails its hash",
                                entry.path
                            ));
                        }
                        fs::write(target, &data).map_err(|e| e.to_string())?;
                        written.fetch_add(data.len() as u64, Ordering::Relaxed);
                        apply_metadata(target, entry);
                        Ok(())
                    })();
                    if let Err(e) = result {
                        *failure.lock().unwrap() = Some(e);
                        return;
                    }
                });
            }
        });

        if let Some(e) = failure.into_inner().unwrap() {
            return Err(io::Error::other(e));
        }

        // Hardlinks are created last: their target may be any file entry, and
        // the workers above are what write those files. A target that is itself
        // a later hardlink resolves on the next pass; a cycle makes no progress
        // and is reported instead of silently leaving entries out.
        let mut pending = hardlinks;
        while !pending.is_empty() {
            let mut deferred = Vec::new();
            let mut made_progress = false;
            for (entry, dest, source) in pending {
                match fs::symlink_metadata(&source) {
                    Ok(meta) if meta.file_type().is_file() => {}
                    Ok(_) => {
                        return Err(io::Error::other(format!(
                            "{}: hardlink target is not a regular file",
                            entry.path
                        )));
                    }
                    Err(_) => {
                        deferred.push((entry, dest, source));
                        continue;
                    }
                }
                let _ = fs::remove_file(&dest);
                if let Err(link_err) = fs::hard_link(&source, &dest) {
                    // Hard links need both names on one volume. Falling back to
                    // a copy preserves the content (though not the inode link)
                    // instead of failing an otherwise valid extraction.
                    fs::copy(&source, &dest).map_err(|copy_err| {
                        io::Error::other(format!(
                            "{}: cannot create hardlink ({link_err}) or copy ({copy_err})",
                            entry.path
                        ))
                    })?;
                }
                apply_metadata(&dest, &entry);
                made_progress = true;
            }
            if !made_progress {
                let names: Vec<&str> = deferred.iter().map(|(e, _, _)| e.path.as_str()).collect();
                return Err(io::Error::other(format!(
                    "hardlink targets cannot be resolved: {}",
                    names.join(", ")
                )));
            }
            pending = deferred;
        }

        Ok(written.into_inner())
    }

    /// A one-line description of the archive's shape.
    pub fn summary(&self) -> String {
        let mut logical = 0u64;
        let mut refs = 0usize;
        let mut files = 0usize;
        for e in &self.directory.entries {
            if e.kind == EntryKind::File {
                files += 1;
                logical += e.size;
                refs += e.blocks.len();
            }
        }
        let stored = refs.saturating_sub(self.dedup_refs());
        format!(
            "{files} files, {} of content in {} of archive, {:.2}x, \
             {refs} block references into {stored} stored blocks",
            crate::util::human(logical as f64),
            crate::util::human(self.footer.archive_len as f64),
            logical as f64 / self.footer.archive_len.max(1) as f64,
        )
    }

    /// How many block references point at an earlier block.
    fn dedup_refs(&self) -> usize {
        let mut first_seen: HashSet<u32> = HashSet::new();
        let mut repeats = 0usize;
        for e in &self.directory.entries {
            for &b in &e.blocks {
                if !first_seen.insert(b) {
                    repeats += 1;
                }
            }
        }
        repeats
    }
}

impl std::fmt::Debug for Archive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Archive")
            .field("path", &self.path)
            .field("entries", &self.directory.len())
            .field("blocks", &self.blocks.len())
            .field("archive_len", &self.footer.archive_len)
            .finish()
    }
}

/// What a full `test` pass found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub entries: usize,
    pub files: usize,
    pub dirs: usize,
    pub symlinks: usize,
    pub blocks: usize,
    pub logical: u64,
}

/// Restore what was recorded, where the platform makes that possible.
///
/// A failure here is not worth reporting: the bytes are already on disk and
/// correct, and a read-only filesystem or a platform without the call should not
/// turn a successful extraction into an error.
fn apply_metadata(path: &Path, e: &Entry) {
    #[cfg(unix)]
    if e.mode != 0 {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(e.mode));
    }
    #[cfg(not(unix))]
    {
        // mtime and the unix mode have no portable equivalent here, and setting
        // one without the other would be misleading. The values stay in the
        // directory, so a tool that can set them still can.
        let _ = (path, e.mtime);
    }
}

/// Validate an archive path and return it as a relative path.
///
/// The rules are the ones that matter for extraction safety: no absolute paths,
/// no drive letters, no `..`, no empty components, and no name Windows would
/// silently mangle. A name that fails is refused rather than cleaned, because
/// cleaning is a guess about what the writer meant.
pub fn safe_relative(p: &str) -> Result<PathBuf, String> {
    if p.is_empty() {
        return Err("empty path".into());
    }
    if p.starts_with('/') || p.starts_with('\\') {
        return Err(format!("refusing absolute path {p:?}"));
    }
    // `C:foo` is drive-relative on Windows and absolute once joined to a base.
    let b = p.as_bytes();
    if b.len() >= 2 && b[1] == b':' && b[0].is_ascii_alphabetic() {
        return Err(format!("refusing drive-lettered path {p:?}"));
    }
    let mut out = PathBuf::new();
    let mut n = 0usize;
    for part in p.split(['/', '\\']) {
        if part.is_empty() {
            return Err(format!("{p:?} has an empty component"));
        }
        if part == "." || part == ".." {
            return Err(format!(
                "refusing {p:?}: a {part:?} component can escape the extraction root"
            ));
        }
        // A name ending in a space or a dot cannot be created on Windows, and
        // one that gets silently trimmed there could collide with another entry.
        if part.ends_with(' ') || part.ends_with('.') {
            return Err(format!("refusing {p:?}: {part:?} is not a portable name"));
        }
        out.push(part);
        n += 1;
    }
    if n == 0 {
        return Err(format!("refusing empty path {p:?}"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{write_archive, WriteOptions};

    fn prefix(dir: &Path) -> String {
        dir.file_name().unwrap().to_string_lossy().into_owned()
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("densr-reader-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, rel: &str, data: &[u8]) {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, data).unwrap();
    }

    fn sample_tree(dir: &Path) -> Vec<u8> {
        let mut v = b"repeated line of text for compression\n".repeat(500);
        v.extend_from_slice(b"a distinct tail\n");
        write(dir, "a.txt", &v);
        write(dir, "sub/b.txt", &v);
        write(dir, "sub/deep/c.bin", &vec![9u8; 9000]);
        v
    }

    fn build(dir: &Path) -> (PathBuf, Vec<u8>) {
        let body = sample_tree(dir);
        let out = dir.join("out.dnsr");
        write_archive(&out, &[dir.to_path_buf()], &WriteOptions::default()).unwrap();
        (out, body)
    }

    #[test]
    fn unsafe_paths_are_refused() {
        for p in [
            "",
            "/etc/passwd",
            "\\windows\\system32",
            "..",
            "../escape",
            "a/../../b",
            "a/./b",
            "C:/windows",
            "c:relative",
            "trailing ",
            "trailing.",
            "a//b",
        ] {
            assert!(safe_relative(p).is_err(), "{p:?} was allowed");
        }
        for p in ["a", "a/b", "a/b/c.txt", "dir.d/file"] {
            assert_eq!(safe_relative(p).unwrap(), PathBuf::from(p));
        }
    }

    #[test]
    fn every_truncation_is_rejected_without_panicking() {
        let dir = tmp("truncated");
        let (out, _) = build(&dir);
        let good = fs::read(&out).unwrap();
        for cut in [
            0usize,
            1,
            8,
            32,
            HEADER_SIZE,
            good.len() / 2,
            good.len() - 1,
        ] {
            let p = dir.join(format!("cut{cut}.dnsr"));
            fs::write(&p, &good[..cut]).unwrap();
            assert!(
                Archive::open(&p).is_err(),
                "a {cut}-byte prefix was accepted"
            );
        }
    }

    #[test]
    fn a_wrong_magic_is_rejected() {
        let dir = tmp("magic");
        let (out, _) = build(&dir);
        let mut bytes = fs::read(&out).unwrap();
        bytes[0] = b'X';
        let p = dir.join("bad.dnsr");
        fs::write(&p, &bytes).unwrap();
        assert!(Archive::open(&p).is_err());
    }

    #[test]
    fn the_footer_must_agree_with_the_header() {
        let dir = tmp("footer");
        let (out, _) = build(&dir);
        let mut bytes = fs::read(&out).unwrap();
        let n = bytes.len();
        bytes[n - FOOTER_SIZE] ^= 0xFF;
        let p = dir.join("bad.dnsr");
        fs::write(&p, &bytes).unwrap();
        assert!(Archive::open(&p).is_err());

        // And a footer whose length field lies about the file size.
        let mut bytes = fs::read(&out).unwrap();
        let n = bytes.len();
        bytes[n - 12..n - 4].copy_from_slice(&1u64.to_le_bytes());
        let p = dir.join("bad2.dnsr");
        fs::write(&p, &bytes).unwrap();
        assert!(Archive::open(&p).is_err());
    }

    #[test]
    fn a_block_whose_payload_overruns_the_directory_is_refused() {
        let dir = tmp("overrun");
        let (out, _) = build(&dir);
        let mut bytes = fs::read(&out).unwrap();
        // `payload_len` is the first field of a block record, so a maximum value
        // there claims a payload far larger than the archive.
        bytes[HEADER_SIZE..HEADER_SIZE + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let p = dir.join("bad.dnsr");
        fs::write(&p, &bytes).unwrap();
        assert!(
            Archive::open(&p).is_err(),
            "an overrunning payload was accepted"
        );
    }

    #[test]
    fn extraction_reproduces_the_tree() {
        let dir = tmp("extract");
        let (out, body) = build(&dir);
        let dest = dir.join("restored");
        let mut ar = Archive::open(&out).unwrap();
        ar.extract_to(&dest).unwrap();
        // Entries are stored under the root's own name, so the tree reappears one
        // level down.
        let top = dest.join(prefix(&dir));
        assert_eq!(fs::read(top.join("a.txt")).unwrap(), body);
        assert_eq!(fs::read(top.join("sub/b.txt")).unwrap(), body);
        assert_eq!(
            fs::read(top.join("sub/deep/c.bin")).unwrap(),
            vec![9u8; 9000]
        );
        assert!(top.join("sub/deep").is_dir());
    }

    #[test]
    fn hardlink_entries_are_restored_after_their_targets() {
        let dir = tmp("hardlink");
        write(&dir, "a.txt", b"hardlink payload\n");
        write(&dir, "b.txt", b"hardlink payload\n");
        let out = dir.join("out.dnsr");
        write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();

        // Rewrite `b.txt` as a hardlink to `a.txt`, which the writer does not
        // emit today but the format defines.
        let orig = fs::read(&out).unwrap();
        let header = Header::decode(&orig[..HEADER_SIZE]).unwrap();
        let at = header.dir_offset as usize;
        let mut recs = Directory::decode(&orig[at..at + header.dir_len as usize]).unwrap();
        let a_path = recs
            .entries
            .iter()
            .find(|e| e.path.ends_with("/a.txt"))
            .unwrap()
            .path
            .clone();
        let b = recs
            .entries
            .iter_mut()
            .find(|e| e.path.ends_with("/b.txt"))
            .unwrap();
        b.kind = EntryKind::Hardlink;
        b.meta = a_path.as_bytes().to_vec();
        b.blocks.clear();
        b.content_hash = 0;

        let (blob, crc) = recs.encode();
        let head = Header {
            dir_len: blob.len() as u32,
            dir_crc: crc,
            ..header
        };
        let footer = Footer {
            dir_offset: head.dir_offset,
            dir_len: head.dir_len,
            dir_crc: crc,
            archive_len: head.dir_offset + blob.len() as u64 + FOOTER_SIZE as u64,
        };
        let mut bytes = orig[..head.dir_offset as usize].to_vec();
        bytes[0..HEADER_SIZE].copy_from_slice(&head.encode());
        bytes.extend_from_slice(&blob);
        bytes.extend_from_slice(&footer.encode());
        let p = dir.join("hardlink.dnsr");
        fs::write(&p, &bytes).unwrap();

        let mut ar = Archive::open(&p).unwrap();
        ar.test().unwrap();
        let dest = dir.join("restored");
        ar.extract_to(&dest).unwrap();
        let top = dest.join(prefix(&dir));
        assert_eq!(fs::read(top.join("a.txt")).unwrap(), b"hardlink payload\n");
        assert_eq!(fs::read(top.join("b.txt")).unwrap(), b"hardlink payload\n");
    }

    #[test]
    fn a_single_flipped_bit_is_caught() {
        let dir = tmp("bitflip");
        let (out, _) = build(&dir);
        let bytes = fs::read(&out).unwrap();
        let header = Header::decode(&bytes[..HEADER_SIZE]).unwrap();

        // Walk the payload area and flip a bit in each block's middle, checking
        // that every one of them is detected.
        let mut checked = 0;
        for offset in (HEADER_SIZE + 40)..(header.dir_offset as usize) {
            let mut b = bytes.clone();
            b[offset] ^= 0x01;
            let p = dir.join("flip.dnsr");
            fs::write(&p, &b).unwrap();
            let mut ar = Archive::open(&p).unwrap();
            let r = ar.test();
            assert!(r.is_err(), "a flip at {offset} went unnoticed");
            checked += 1;
            if checked >= 40 {
                break;
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn a_directory_naming_a_traversal_is_refused_at_open() {
        // The reader must not rely on the writer having been careful: an archive
        // can arrive from anywhere. This one is assembled by hand.
        let dir = tmp("evil");
        let rec = EntryRec {
            path: "../../escape.txt".into(),
            kind: EntryKind::File,
            class: crate::magic::FileClass::Prose,
            mode: 0,
            mtime: 0,
            size: 0,
            content_hash: 0,
            blocks: vec![],
            meta: vec![],
        };
        let (blob, crc) = Directory { entries: vec![rec] }.encode();
        let header = Header {
            version: crate::format::FORMAT_VERSION,
            flags: 0,
            level: 7,
            block_size: 65536,
            created: 0,
            payload_len: 0,
            dir_offset: HEADER_SIZE as u64,
            dir_len: blob.len() as u32,
            dir_crc: crc,
        };
        let footer = Footer {
            dir_offset: header.dir_offset,
            dir_len: header.dir_len,
            dir_crc: crc,
            archive_len: header.dir_offset + blob.len() as u64 + FOOTER_SIZE as u64,
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&header.encode());
        bytes.extend_from_slice(&blob);
        bytes.extend_from_slice(&footer.encode());
        assert_eq!(&bytes[..8], &crate::format::MAGIC[..]);
        let p = dir.join("evil.dnsr");
        fs::write(&p, &bytes).unwrap();

        let e = Archive::open(&p).unwrap_err();
        assert!(e.contains("escape") || e.contains("refusing"), "{e}");
    }

    #[test]
    fn an_entry_whose_size_disagrees_with_its_blocks_is_refused() {
        // The directory is rewritten by hand so that one entry claims a size its
        // block list cannot produce.
        let dir = tmp("sizemismatch");
        let (out, _) = build(&dir);
        let orig = fs::read(&out).unwrap();
        let header = Header::decode(&orig[..HEADER_SIZE]).unwrap();
        let at = header.dir_offset as usize;
        let dir_recs = Directory::decode(&orig[at..at + header.dir_len as usize]).unwrap();

        let mut broken = dir_recs.clone();
        for e in broken.entries.iter_mut() {
            if e.kind == EntryKind::File {
                e.size += 7;
                break;
            }
        }
        assert!(broken != dir_recs);
        let (blob, crc) = broken.encode();
        let head = Header {
            dir_len: blob.len() as u32,
            dir_crc: crc,
            ..header
        };
        let footer = Footer {
            dir_offset: head.dir_offset,
            dir_len: head.dir_len,
            dir_crc: crc,
            archive_len: head.dir_offset + blob.len() as u64 + FOOTER_SIZE as u64,
        };
        // Keep the payload, replace the header, then append the new
        // directory and footer. Starting from the header alone would silently
        // drop every block and produce an archive of the wrong length.
        let mut bytes = orig[..head.dir_offset as usize].to_vec();
        bytes[0..HEADER_SIZE].copy_from_slice(&head.encode());
        bytes.extend_from_slice(&blob);
        bytes.extend_from_slice(&footer.encode());
        let p = dir.join("broken.dnsr");
        fs::write(&p, &bytes).unwrap();

        let e = Archive::open(&p).unwrap_err();
        assert!(e.contains("bytes"), "{e}");
    }

    #[test]
    fn a_block_index_past_the_end_is_refused() {
        let dir = tmp("badindex");
        let (out, _) = build(&dir);
        let orig = fs::read(&out).unwrap();
        let header = Header::decode(&orig[..HEADER_SIZE]).unwrap();
        let at = header.dir_offset as usize;
        let mut recs = Directory::decode(&orig[at..at + header.dir_len as usize]).unwrap();
        for e in recs.entries.iter_mut() {
            if e.kind == EntryKind::File {
                e.blocks = vec![9999];
                break;
            }
        }
        let (blob, crc) = recs.encode();
        let head = Header {
            dir_len: blob.len() as u32,
            dir_crc: crc,
            ..header
        };
        let footer = Footer {
            dir_offset: head.dir_offset,
            dir_len: head.dir_len,
            dir_crc: crc,
            archive_len: head.dir_offset + blob.len() as u64 + FOOTER_SIZE as u64,
        };
        // Keep the payload, replace the header, then append the new
        // directory and footer. Starting from the header alone would silently
        // drop every block and produce an archive of the wrong length.
        let mut bytes = orig[..head.dir_offset as usize].to_vec();
        bytes[0..HEADER_SIZE].copy_from_slice(&head.encode());
        bytes.extend_from_slice(&blob);
        bytes.extend_from_slice(&footer.encode());
        let p = dir.join("badindex.dnsr");
        fs::write(&p, &bytes).unwrap();

        let e = Archive::open(&p).unwrap_err();
        assert!(e.contains("block"), "{e}");
    }

    #[test]
    fn test_reports_what_it_checked() {
        let dir = tmp("report");
        let (out, _) = build(&dir);
        let mut ar = Archive::open(&out).unwrap();
        let r = ar.test().unwrap();
        assert_eq!(r.files, 3);
        assert_eq!(r.entries, 3 + 3, "three files and three directories");
        assert!(r.blocks > 0);
        assert!(r.logical > 0);
        assert!(ar.summary().contains("files"));
    }

    #[test]
    fn a_missing_file_is_an_error_not_an_empty_result() {
        let dir = tmp("missing");
        write(&dir, "a.txt", b"x\n");
        let out = dir.join("out.dnsr");
        write_archive(&out, &[dir.to_path_buf()], &WriteOptions::default()).unwrap();
        let mut ar = Archive::open(&out).unwrap();
        assert!(ar.read("nope.txt").is_err());
        assert!(ar.get("nope.txt").is_none());
        assert!(
            ar.read("out.dnsr").is_err(),
            "the archive itself is not a member"
        );
    }
}
