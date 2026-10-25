//! Smart Bypass: deciding what to compress at all.
//!
//! # The problem
//!
//! Running an LZ matcher over a JPEG costs the same CPU per byte as running it
//! over source code, and returns roughly nothing: the entropy is already gone.
//! On a mixed directory that is most of the wall-clock time, spent on files that
//! cannot shrink.
//!
//! # The rule
//!
//! Read a small header, match it against a table of known signatures, and if the
//! payload is already compressed then store it verbatim. Everything else is
//! compressed normally.
//!
//! Signatures are checked before extensions on purpose. A `.zip` full of JPEGs is
//! a zip, and a `.jpg` that is really a PNG is a PNG; trusting the name gets both
//! wrong, and names are the part of a file most likely to have been renamed.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// How much of a file to look at before deciding.
pub const HEADER_PROBE: usize = 1024;

/// What a file is, in the only sense the writer cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileClass {
    /// Source code: identifiers, punctuation, indentation.
    Code,
    /// Markup and structured text: tags, entities, heavy whitespace.
    Markup,
    /// Tabular or record-structured data: fixed fields, few distinct bytes.
    Data,
    /// Ordinary running prose.
    Prose,
    /// Already compressed, or not worth compressing: images, audio, video,
    /// archives, executables.
    Binary,
}

impl FileClass {
    pub fn label(self) -> &'static str {
        match self {
            FileClass::Code => "code",
            FileClass::Markup => "markup",
            FileClass::Data => "data",
            FileClass::Prose => "prose",
            FileClass::Binary => "binary",
        }
    }

    pub fn from_id(id: u8) -> Result<FileClass, String> {
        Ok(match id {
            0 => FileClass::Code,
            1 => FileClass::Markup,
            2 => FileClass::Data,
            3 => FileClass::Prose,
            4 => FileClass::Binary,
            other => return Err(format!("unknown file class {other}")),
        })
    }

    pub fn id(self) -> u8 {
        self as u8
    }

    /// Whether this class is worth handing to the codec.
    ///
    /// Always true: every block is tried through the codec and stored if it
    /// does not shrink (see `compress_block` in writer.rs). Classification is
    /// kept for reporting only; it no longer gates compression.
    pub fn is_compressible(self) -> bool {
        let _ = self;
        true
    }
}

/// The container family a signature belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Image,
    Audio,
    Video,
    Archive,
    Executable,
    Document,
    Database,
    Font,
    Other,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::Image => "image",
            Format::Audio => "audio",
            Format::Video => "video",
            Format::Archive => "archive",
            Format::Executable => "executable",
            Format::Document => "document",
            Format::Database => "database",
            Format::Font => "font",
            Format::Other => "other",
        }
    }
}

/// A signature: bytes at a fixed offset.
pub struct Signature {
    pub offset: usize,
    pub magic: &'static [u8],
    pub format: Format,
    /// When set, the byte at `mask_offset` must be one of `mask_values` too.
    /// Needed where the fixed part of the magic is short and ambiguous.
    pub mask_offset: usize,
    pub mask_values: &'static [u8],
}

const EMPTY: &[u8] = &[];

/// Signatures of formats whose data is already entropy-coded.
///
/// Every entry here is a promise that compressing the payload further is a waste
/// of time. An entry that is wrong in that direction costs only speed; an entry
/// missing costs speed too, so the table errs toward including formats.
pub static SIGNATURES: &[Signature] = &[
    // -- Images ---------------------------------------------------------
    Signature {
        offset: 0,
        magic: b"\xff\xd8\xff",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x89PNG\r\n\x1a\n",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"GIF87a",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"GIF89a",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"BM",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    // RIFF containers: the real type is a fourcc at offset 8, checked below.
    Signature {
        offset: 0,
        magic: b"II*\x00",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"MM\x00*",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x00\x00\x01\x00",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x76\x2f\x31\x01",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    // JPEG 2000 puts its signature in a codestream box 4 bytes in.
    Signature {
        offset: 4,
        magic: b"jP  ",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 4,
        magic: b"jP\x1a",
        format: Format::Image,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x1f\x8b",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"BZh",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\xfd7zXZ\x00",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"7z\xbc\xaf\x27\x1c",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x28\xb5\x2f\xfd",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"Rar!\x1a\x07",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x04\x22\x4d\x18",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x50\x4b\x03\x04",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x50\x4b\x05\x06",
        format: Format::Archive,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    // -- Audio and video -----------------------------------------------
    Signature {
        offset: 0,
        magic: b"ID3",
        format: Format::Audio,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"OggS",
        format: Format::Audio,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"fLaC",
        format: Format::Audio,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b".snd",
        format: Format::Audio,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"FORM",
        format: Format::Audio,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    // Matroska and WebM share a container; the doctype at offset 4 says which.
    Signature {
        offset: 0,
        magic: b"\x1a\x45\xdf\xa3",
        format: Format::Video,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 4,
        magic: b"ftyp",
        format: Format::Video,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x00\x00\x01\xba",
        format: Format::Video,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x00\x00\x01\xb3",
        format: Format::Video,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"FLV\x01",
        format: Format::Video,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    // -- Executables ----------------------------------------------------
    Signature {
        offset: 0,
        magic: b"MZ",
        format: Format::Executable,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x7fELF",
        format: Format::Executable,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\xca\xfe\xba\xbe",
        format: Format::Executable,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\xcf\xfa\xed\xfe",
        format: Format::Executable,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\xce\xfa\xed\xfe",
        format: Format::Executable,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    // Mach-O: 32- and 64-bit, both endiannesses.
    Signature {
        offset: 0,
        magic: b"\xfe\xed\xfa\xce",
        format: Format::Executable,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\xfe\xed\xfa\xcf",
        format: Format::Executable,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    // -- Documents, databases, fonts ------------------------------------
    Signature {
        offset: 0,
        magic: b"%PDF-",
        format: Format::Document,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"SQLite format 3\x00",
        format: Format::Database,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"\x00\x01\x00\x00\x00",
        format: Format::Font,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"OTTO",
        format: Format::Font,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"wOFF",
        format: Format::Font,
        mask_offset: 0,
        mask_values: EMPTY,
    },
    Signature {
        offset: 0,
        magic: b"wOF2",
        format: Format::Font,
        mask_offset: 0,
        mask_values: EMPTY,
    },
];

/// Extension to class, for files whose header is uninformative.
///
/// This is the fallback, not the primary signal: it exists for text formats that
/// have no signature at all, where the extension is the only evidence there is.
pub static EXTENSION_CLASSES: &[(&str, FileClass)] = &[
    // Source code.
    ("rs", FileClass::Code),
    ("py", FileClass::Code),
    ("js", FileClass::Code),
    ("ts", FileClass::Code),
    ("tsx", FileClass::Code),
    ("jsx", FileClass::Code),
    ("c", FileClass::Code),
    ("h", FileClass::Code),
    ("cc", FileClass::Code),
    ("cpp", FileClass::Code),
    ("cxx", FileClass::Code),
    ("hpp", FileClass::Code),
    ("go", FileClass::Code),
    ("java", FileClass::Code),
    ("kt", FileClass::Code),
    ("cs", FileClass::Code),
    ("swift", FileClass::Code),
    ("rb", FileClass::Code),
    ("php", FileClass::Code),
    ("pl", FileClass::Code),
    ("lua", FileClass::Code),
    ("sh", FileClass::Code),
    ("bash", FileClass::Code),
    ("zsh", FileClass::Code),
    ("ps1", FileClass::Code),
    ("bat", FileClass::Code),
    ("cmd", FileClass::Code),
    ("sql", FileClass::Code),
    ("r", FileClass::Code),
    ("m", FileClass::Code),
    ("scala", FileClass::Code),
    ("hs", FileClass::Code),
    ("ml", FileClass::Code),
    ("erl", FileClass::Code),
    ("ex", FileClass::Code),
    ("exs", FileClass::Code),
    ("clj", FileClass::Code),
    ("vim", FileClass::Code),
    ("asm", FileClass::Code),
    ("s", FileClass::Code),
    ("f90", FileClass::Code),
    ("jl", FileClass::Code),
    // Build and config files that are really code.
    ("mk", FileClass::Code),
    ("cmake", FileClass::Code),
    ("gradle", FileClass::Code),
    // Markup.
    ("html", FileClass::Markup),
    ("htm", FileClass::Markup),
    ("xhtml", FileClass::Markup),
    ("xml", FileClass::Markup),
    ("xsl", FileClass::Markup),
    ("svg", FileClass::Markup),
    ("css", FileClass::Markup),
    ("scss", FileClass::Markup),
    ("sass", FileClass::Markup),
    ("less", FileClass::Markup),
    ("vue", FileClass::Markup),
    ("svelte", FileClass::Markup),
    // Structured data.
    ("json", FileClass::Data),
    ("jsonl", FileClass::Data),
    ("ndjson", FileClass::Data),
    ("csv", FileClass::Data),
    ("tsv", FileClass::Data),
    ("yaml", FileClass::Data),
    ("yml", FileClass::Data),
    ("toml", FileClass::Data),
    ("ini", FileClass::Data),
    ("cfg", FileClass::Data),
    ("conf", FileClass::Data),
    ("properties", FileClass::Data),
    ("plist", FileClass::Data),
    ("proto", FileClass::Data),
    ("parquet", FileClass::Data),
    // Prose.
    ("md", FileClass::Prose),
    ("markdown", FileClass::Prose),
    ("rst", FileClass::Prose),
    ("txt", FileClass::Prose),
    ("text", FileClass::Prose),
    ("adoc", FileClass::Prose),
    ("org", FileClass::Prose),
    ("tex", FileClass::Prose),
    ("log", FileClass::Prose),
    ("nfo", FileClass::Prose),
    ("me", FileClass::Prose),
    // Prose-like text that is really code, disambiguated by the extension.
    ("gitignore", FileClass::Code),
    ("gitattributes", FileClass::Code),
    ("dockerignore", FileClass::Code),
    ("editorconfig", FileClass::Code),
];

/// Extensions that mean "binary" even though there is no signature, mostly
/// because the file is empty or too short to probe.
pub static BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "avif", "heic", "heif", "tiff", "tif",
    "svgz", "jp2", "j2k", "psd", "ai", "eps", "xcf", "raw", "dng", "cr2", "nef", "mp3", "flac",
    "wav", "ogg", "oga", "m4a", "aac", "wma", "opus", "aiff", "mid", "mp4", "mkv", "webm", "avi",
    "mov", "wmv", "flv", "m4v", "mpg", "mpeg", "3gp", "zip", "7z", "rar", "gz", "bz2", "xz", "lz",
    "lzma", "zst", "zstd", "br", "tar", "cab", "iso", "dmg", "msi", "deb", "rpm", "apk", "jar",
    "war", "whl", "egg", "exe", "dll", "so", "dylib", "obj", "lib", "a", "o", "pdb", "pyc", "pyd",
    "class", "woff", "woff2", "ttf", "otf", "eot", "mp3_", "sqlite", "db", "mdb", "accdb", "dat",
    "idx",
];

/// Split a path's extension, without the dot.
///
/// Dotfiles are not extensions: `.gitignore` has no extension, and treating it
/// as one would classify every dotfile as binary.
pub fn split_ext(name: &str) -> &str {
    if let Some(rest) = name.strip_prefix('.') {
        if rest.is_empty() {
            return "";
        }
    }
    match name.rfind('.') {
        // A leading dot is the whole name, so `Cargo.toml`'s extension is
        // `toml` but `.gitignore`'s is none.
        Some(0) => "",
        Some(i) => {
            let ext = &name[i + 1..];
            if ext.is_empty() {
                ""
            } else {
                ext
            }
        }
        None => "",
    }
}

/// Match a header against the signature table.
pub fn classify_header(header: &[u8]) -> Option<Format> {
    for sig in SIGNATURES {
        let end = sig.offset + sig.magic.len();
        if header.len() < end {
            continue;
        }
        if &header[sig.offset..end] == sig.magic {
            if !sig.mask_values.is_empty() {
                let b = header.get(sig.mask_offset).copied().unwrap_or(0);
                if !sig.mask_values.contains(&b) {
                    continue;
                }
            }
            return Some(sig.format);
        }
    }
    // RIFF is a container: the payload type is a fourcc at offset 8, and an
    // uncompressed WAV inside it really is compressible.
    if header.len() >= 12 && &header[0..4] == b"RIFF" {
        match &header[8..12] {
            b"WAVE" => return Some(Format::Audio),
            b"WEBP" => return Some(Format::Image),
            b"AVI " => return Some(Format::Video),
            _ => {}
        }
        // An unknown RIFF payload: assume the container holds something already
        // compressed, which is the common case.
        return Some(Format::Other);
    }
    // Matroska: the doctype string starts at offset 8.
    if header.len() >= 16 && &header[0..4] == b"\x1a\x45\xdf\xa3" {
        return Some(Format::Video);
    }
    None
}

/// Decide a text class from the bytes alone, for files with no signature.
///
/// The signal is the shape of the text, not the words: code has long runs of
/// punctuation and few distinct characters, prose has spaces and a wider
/// alphabet, markup is dominated by angle brackets and indentation.
pub fn classify_text(data: &[u8]) -> FileClass {
    if data.is_empty() {
        return FileClass::Data;
    }
    let probe = &data[..data.len().min(HEADER_PROBE)];
    let mut printable = 0usize;
    let mut nul = 0usize;
    let mut high = 0usize;
    let mut punctuation = 0usize;
    let mut spaces = 0usize;
    let mut newlines = 0usize;
    let mut distinct = [false; 256];
    let mut distinct_count = 0usize;

    for &b in probe {
        match b {
            0 => nul += 1,
            0x20..=0x7E => {
                printable += 1;
                if b.is_ascii_punctuation() {
                    punctuation += 1;
                }
                if b == b' ' {
                    spaces += 1;
                }
            }
            b'\n' | b'\r' | b'\t' => {
                // Whitespace, and text rather than binary.
                printable += 1;
                if b == b'\n' {
                    newlines += 1;
                }
            }
            0x80..=0xFF => high += 1,
            _ => {}
        }
        if !distinct[b as usize] {
            distinct[b as usize] = true;
            distinct_count += 1;
        }
    }

    let total = probe.len();
    // A NUL byte, or a high proportion of non-text bytes, settles it.
    if nul > 0 || (printable + high) * 10 < total * 9 {
        return FileClass::Binary;
    }

    let punct_ratio = punctuation as f64 / printable.max(1) as f64;
    let space_ratio = spaces as f64 / printable.max(1) as f64;
    let distinct_ratio = distinct_count as f64 / printable.max(1) as f64;

    // Angle brackets and entities in bulk: markup.
    let angle = count_byte(probe, b'<') + count_byte(probe, b'>');
    let angle_ratio = angle as f64 / total as f64;
    if angle_ratio > 0.01 {
        return FileClass::Markup;
    }
    // Punctuation with hardly any spaces: a record format such as CSV, rather
    // than prose or code. This is tested before code on purpose -- code is
    // punctuation-heavy too, and without the ordering it swallows every
    // comma-separated file.
    if space_ratio < 0.05 && punct_ratio > 0.08 {
        return FileClass::Data;
    }
    // Heavy punctuation with a narrow alphabet: code.
    if punct_ratio > 0.13 && distinct_ratio < 0.55 {
        return FileClass::Code;
    }
    // Narrow alphabet, spaces present: tabular data such as CSV.
    if distinct_ratio < 0.35 && newlines > 0 {
        return FileClass::Data;
    }
    if space_ratio > 0.08 {
        return FileClass::Prose;
    }
    FileClass::Code
}

fn count_byte(data: &[u8], b: u8) -> usize {
    data.iter().filter(|&&x| x == b).count()
}

/// Classify from a header and an extension together.
pub fn classify_bytes(header: &[u8], ext: &str) -> FileClass {
    if let Some(format) = classify_header(header) {
        return match format {
            // A PDF, an office document and a font all compress well despite being
            // "document" formats, so the class comes from the content, not the
            // container.
            Format::Document | Format::Database => {
                if looks_binary(header) {
                    FileClass::Binary
                } else {
                    classify_text(header)
                }
            }
            _ => FileClass::Binary,
        };
    }

    // A NUL in the first kilobyte settles it, whatever the file is called. A
    // `.txt` full of NULs is not prose, and the extension is the least reliable
    // thing available.
    if looks_binary(header) {
        return FileClass::Binary;
    }

    let ext = ext.to_ascii_lowercase();
    for (name, class) in EXTENSION_CLASSES {
        if *name == ext {
            return *class;
        }
    }
    if BINARY_EXTENSIONS.contains(&ext.as_str()) {
        return FileClass::Binary;
    }
    // Dotfiles: configuration, not binary. A file with no name worth guessing at
    // is still text if it reads as text.
    if header.is_empty() {
        return FileClass::Data;
    }
    classify_text(header)
}

fn looks_binary(header: &[u8]) -> bool {
    header.contains(&0)
}

/// The lower-case extension of a path's file name.
pub fn path_ext(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .map(|n| split_ext(&n).to_ascii_lowercase())
        .unwrap_or_default()
}

/// Read a file's header and classify it.
pub fn classify_path(path: &Path) -> Result<FileClass, String> {
    classify_path_with_probe(path, HEADER_PROBE)
}

/// Classify with an explicit probe size, so tests can be cheap.
pub fn classify_path_with_probe(path: &Path, probe: usize) -> Result<FileClass, String> {
    let ext = path_ext(path);
    // Only the header is read. A 4 GB video classified by its first kilobyte
    // must not be loaded into memory to find out it is a video.
    let mut header = vec![0u8; probe.max(32)];
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut filled = 0usize;
    while filled < header.len() {
        match f.read(&mut header[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
    }
    header.truncate(filled);
    Ok(classify_bytes(&header, &ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_split_correctly() {
        assert_eq!(split_ext("main.rs"), "rs");
        assert_eq!(split_ext("a.b.c"), "c");
        assert_eq!(split_ext("noext"), "");
        assert_eq!(split_ext("trailing."), "");
        // A leading dot is the whole name, not a separator.
        assert_eq!(split_ext(".gitignore"), "");
        assert_eq!(split_ext(".env.local"), "local");
        assert_eq!(split_ext(""), "");
    }

    #[test]
    fn signatures_win_over_extensions() {
        // A JPEG named `.txt` is still a JPEG.
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0];
        jpeg.extend_from_slice(b"JFIF");
        assert_eq!(classify_bytes(&jpeg, "txt"), FileClass::Binary);
        // A PNG named `.rs`.
        let png = b"\x89PNG\r\n\x1a\nrest";
        assert_eq!(classify_bytes(png, "rs"), FileClass::Binary);
        // A ZIP named `.png`.
        let zip = b"PK\x03\x04rest";
        assert_eq!(classify_bytes(zip, "png"), FileClass::Binary);
    }

    #[test]
    fn every_signature_offset_fits_in_the_probe() {
        for sig in SIGNATURES {
            assert!(
                sig.offset + sig.magic.len() <= HEADER_PROBE,
                "{:?} needs {} bytes of header",
                sig.format,
                sig.offset + sig.magic.len()
            );
        }
    }

    #[test]
    fn signature_table_has_no_duplicates_that_would_mask_each_other() {
        // Two entries with the same offset and magic are a copy-paste error; the
        // first would always win and the second would be dead code.
        for (i, a) in SIGNATURES.iter().enumerate() {
            for b in &SIGNATURES[i + 1..] {
                if a.offset == b.offset && a.magic == b.magic {
                    // Four-byte duplicates are a real problem; shorter ones can
                    // legitimately be prefixes ordered longest-first.
                    assert!(
                        a.magic.len() < 4 || b.magic.len() < 4 || a.magic != b.magic,
                        "duplicate signature at offset {}",
                        a.offset
                    );
                }
            }
        }
    }

    #[test]
    fn riff_dispatches_on_its_payload_type() {
        let mut wave = b"RIFF".to_vec();
        wave.extend_from_slice(&[0, 0, 0, 0]);
        wave.extend_from_slice(b"WAVEfmt ");
        assert_eq!(classify_header(&wave), Some(Format::Audio));
        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&[0, 0, 0, 0]);
        webp.extend_from_slice(b"WEBPVP8 ");
        assert_eq!(classify_header(&webp), Some(Format::Image));
    }

    #[test]
    fn jpeg2000_signature_at_offset_four() {
        // The codestream box type sits after the JP2 container's length field.
        let mut jp2 = vec![0x00, 0x00, 0x00, 0x0c];
        jp2.extend_from_slice(b"jP  ");
        jp2.extend_from_slice(&[0; 32]);
        assert_eq!(classify_header(&jp2), Some(Format::Image));
        assert_eq!(classify_bytes(&jp2, "jp2"), FileClass::Binary);
    }

    #[test]
    fn nul_bytes_make_it_binary() {
        assert_eq!(
            classify_bytes(b"text\x00more text here", "txt"),
            FileClass::Binary
        );
    }

    #[test]
    fn text_shapes_are_told_apart() {
        let code = b"fn main() {\n    let x: u32 = 42;\n    println!(\"{}\", x);\n}\n";
        assert_eq!(classify_text(code), FileClass::Code);
        let markup = b"<html><body><div class=\"a\">hi</div></body></html>";
        assert_eq!(classify_text(markup), FileClass::Markup);
        let data = b"1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15\n";
        assert_eq!(classify_text(data), FileClass::Data);
    }

    #[test]
    fn extensions_carry_classes_when_there_is_no_signature() {
        assert_eq!(classify_bytes(b"anything at all", "md"), FileClass::Prose);
        assert_eq!(classify_bytes(b"anything at all", "rs"), FileClass::Code);
        assert_eq!(classify_bytes(b"anything at all", "json"), FileClass::Data);
        assert_eq!(
            classify_bytes(b"anything at all", "html"),
            FileClass::Markup
        );
    }

    #[test]
    fn class_ids_round_trip() {
        for class in [
            FileClass::Code,
            FileClass::Markup,
            FileClass::Data,
            FileClass::Prose,
            FileClass::Binary,
        ] {
            assert_eq!(FileClass::from_id(class.id()).unwrap(), class);
        }
        assert!(FileClass::from_id(99).is_err());
    }

    #[test]
    fn all_classes_are_compressed() {
        for class in [
            FileClass::Code,
            FileClass::Markup,
            FileClass::Data,
            FileClass::Prose,
            FileClass::Binary,
        ] {
            assert!(
                class.is_compressible(),
                "{} should be compressed",
                class.label()
            );
        }
    }
}
