//! The `densr` command-line tool.
//!
//! Five verbs, and the argument handling is written here rather than pulled in:
//! the crate has no dependencies, and a dependency that only prints help text
//! costs more than it saves.
//!
//! # Exit codes
//!
//! `0` success, `1` the operation failed, `2` the command line was wrong. The
//! third exists so a script can tell "your archive is broken" from "you typed the
//! flags wrong" without parsing messages.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use densr::chunking::{Chunking, MAX_BLOCK, MIN_BLOCK};
use densr::format::EntryKind;
use densr::reader::Archive;
use densr::writer::{write_archive, WriteOptions};

const USAGE: &str = "\
densr -- compress intelligently instead of compressing blindly

USAGE:
    densr <COMMAND> [OPTIONS] [ARGS]

COMMANDS:
    add <archive> <path>...   Write files and directories into an archive
    extract <archive> [dir]   Write everything back out (default: current dir)
    list <archive>            List what an archive holds
    info <archive>            Show the header and a size summary
    test <archive>            Check every block against its checksums

OPTIONS for add:
    -l, --level <0-12>        Effort: higher searches harder and squeezes
                             tighter (default 4)
    -b, --block-size <size>   Block size, e.g. 262144 or 256k (default 16M,
                             clamped to 4k..16M)
         --fixed               Split at fixed offsets instead of by content
         --threads <n>         Compress with n workers (default: CPU count)
          --gpu                 Also compress blocks on the GPU (OpenCL, falls
                                back to CPU-only if unavailable)
          --gpu-share <0-100>   Percent of blocks offloaded to the GPU. Omit it
                                and the share is chosen for you: densr measures
                                the GPU against the CPU workers and offloads only
                                what the GPU can absorb without becoming the
                                bottleneck. Archives too small to be worth
                                opening a context for stay on the CPU entirely.
                                CPU and GPU run in parallel.
          --gpu-batch <1-32>    Blocks per GPU kernel launch (default 32)
          --dict-size <size>    Shared dictionary budget, sampled from this
                                archive and stored compressed inside it (default
                                1M). 0 disables it.
          --no-dict             Do not build a shared dictionary
         --no-metadata         Do not record mtime and permissions
         --no-verify           Skip decoding each block back to check it
    -q, --quiet               Print nothing on success
    -h, --help                Show this
    -V, --version             Show the version

EXIT CODES:
    0  success
    1  the operation failed (corrupt archive, unreadable file)
    2  the command line was wrong
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Fail::Usage(m)) => {
            eprintln!("error: {m}");
            eprintln!("\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Fail::Error(m)) => {
            eprintln!("error: {m}");
            ExitCode::FAILURE
        }
    }
}

/// Why a command stopped, kept apart so the exit code can reflect which.
#[derive(Debug)]
enum Fail {
    /// The command line did not make sense.
    Usage(String),
    /// The command made sense but could not be carried out.
    Error(String),
}

impl From<String> for Fail {
    fn from(m: String) -> Fail {
        Fail::Error(m)
    }
}

fn usage(m: impl Into<String>) -> Result<(), Fail> {
    Err(Fail::Usage(m.into()))
}

fn run(args: &[String]) -> Result<(), Fail> {
    // `--flag=value` is expanded into two tokens, so the parser below only ever
    // has to understand the spaced form.
    let expanded = expand_equals(args);
    let args = &expanded[..];

    let Some(cmd) = args.first() else {
        print!("{USAGE}");
        return Ok(());
    };

    // Help and version are accepted anywhere, so `densr add -h` works.
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(());
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("densr {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let rest = &args[1..];
    match cmd.as_str() {
        "add" => cmd_add(rest),
        "extract" => cmd_extract(rest),
        "list" => cmd_list(rest),
        "info" => cmd_info(rest),
        "test" => cmd_test(rest),
        other => usage(format!("unknown command {other:?}")),
    }
}

// -- add ---------------------------------------------------------------------

fn cmd_add(args: &[String]) -> Result<(), Fail> {
    let mut opts = WriteOptions::default();
    let mut quiet = false;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        // Everything after `--` is a path, even if it looks like a flag.
        if a == "--" {
            positional.extend(args[i + 1..].iter().cloned());
            break;
        }
        match a {
            "-l" | "--level" => {
                let v = next(args, &mut i, "--level")?;
                opts.level = v
                    .parse::<u8>()
                    .map_err(|_| Fail::Usage(format!("--level wants a number, got {v:?}")))?;
                if opts.level > 12 {
                    return usage(format!("--level {} is out of range; max is 12", opts.level));
                }
            }
            "-b" | "--block-size" => {
                let v = next(args, &mut i, "--block-size")?;
                opts.block_size = parse_size(&v)?;
                // Asking for a block size is also asking for CDC boundaries at
                // that spacing. Left unset, CDC keeps its own smaller default,
                // which measurably protects dedup.
                opts.cdc_target = opts.block_size;
            }
            "--fixed" => opts.chunking = Chunking::Fixed,
            "--cdc" => opts.chunking = Chunking::Cdc,
            "--threads" => {
                let v = next(args, &mut i, "--threads")?;
                opts.threads = v
                    .parse::<usize>()
                    .map_err(|_| Fail::Usage(format!("--threads wants a number, got {v:?}")))?;
                if opts.threads == 0 {
                    return usage("--threads must be at least 1");
                }
            }
            "--gpu" => opts.gpu = true,
            "--gpu-share" => {
                let v = next(args, &mut i, "--gpu-share")?;
                let pct: usize = v
                    .parse()
                    .map_err(|_| Fail::Usage(format!("--gpu-share wants a number, got {v:?}")))?;
                if pct > 100 {
                    return usage("--gpu-share must be 0..100");
                }
                opts.gpu_share = Some(pct as u8);
            }
            "--gpu-batch" => {
                let v = next(args, &mut i, "--gpu-batch")?;
                opts.gpu_batch = v
                    .parse()
                    .map_err(|_| Fail::Usage(format!("--gpu-batch wants a number, got {v:?}")))?;
                if opts.gpu_batch == 0 || opts.gpu_batch > 32 {
                    return usage("--gpu-batch must be 1..32");
                }
            }
            "--no-dict" => opts.dict_size = 0,
            "--dict-size" => {
                let v = next(args, &mut i, "--dict-size")?;
                opts.dict_size = parse_size(&v)?;
            }
            "--no-metadata" => opts.metadata = false,
            "--no-verify" => opts.verify = false,
            "-q" | "--quiet" => quiet = true,
            other if other.starts_with('-') && other != "-" => {
                return usage(format!("unknown option {other:?} for `add`"));
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }

    if positional.len() < 2 {
        return usage("`add` needs an archive and at least one path");
    }
    let archive = PathBuf::from(&positional[0]);
    let roots: Vec<PathBuf> = positional[1..].iter().map(PathBuf::from).collect();

    for r in &roots {
        if !r.exists() {
            return Err(Fail::Error(format!(
                "{}: no such file or directory",
                r.display()
            )));
        }
    }
    if archive.exists() && archive.is_dir() {
        return Err(Fail::Error(format!(
            "{}: is a directory, not an archive",
            archive.display()
        )));
    }

    // Reported before the work starts, because a run over a large tree takes
    // long enough that silence looks like a hang.
    if !quiet {
        let gpu_tag = if opts.gpu { ", gpu" } else { "" };
        eprintln!(
            "densr: {} -> {} ({}, level {}, {}-byte blocks, {} threads{})",
            roots
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(" "),
            archive.display(),
            opts.chunking.name(),
            opts.level,
            opts.block_size.clamp(MIN_BLOCK, MAX_BLOCK),
            opts.threads,
            gpu_tag,
        );
    }

    let started = std::time::Instant::now();
    let report = write_archive(&archive, &roots, &opts).map_err(|e| Fail::Error(e.to_string()))?;
    let elapsed = started.elapsed();

    if !quiet {
        eprintln!(
            "densr: done in {:.1}s, {}",
            elapsed.as_secs_f64(),
            report.summary()
        );
    }
    Ok(())
}

/// Read the value that follows a flag, advancing past it.
fn next(args: &[String], i: &mut usize, flag: &str) -> Result<String, Fail> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| Fail::Usage(format!("{flag} needs a value")))
}

/// Turn `--flag=value` into `--flag`, `value`.
///
/// Only for long flags, and only the first `=` splits, so a value may itself
/// contain one. A bare `--flag=` becomes an empty value, which the flag's own
/// parser then rejects with a message about that flag rather than a generic one.
fn expand_equals(args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        if a.starts_with("--") && !a.starts_with("---") {
            if let Some((name, value)) = a.split_once('=') {
                out.push(name.to_string());
                out.push(value.to_string());
                continue;
            }
        }
        out.push(a.clone());
    }
    out
}

/// Parse a byte count: `65536`, `64k`, `4M`, `2mb`, `2MiB`.
///
/// The suffix is optional and a trailing `b` is tolerated, because `64kb` and
/// `2MiB` are what people type even when the tool documents `64k`.
fn parse_size(s: &str) -> Result<usize, Fail> {
    let mut t = s.trim().to_ascii_lowercase();
    // "KiB" and "kb" both mean the same thing here; a bare "b" means bytes.
    if t.ends_with("ib") && t.len() > 2 {
        t.truncate(t.len() - 2);
    } else if t.len() > 1 && t.ends_with('b') {
        t.pop();
    }
    let (digits, mult) = match t.as_bytes().last() {
        Some(b'k') => (&t[..t.len() - 1], 1024usize),
        Some(b'm') => (&t[..t.len() - 1], 1024 * 1024),
        Some(b'g') => (&t[..t.len() - 1], 1024 * 1024 * 1024),
        _ => (t.as_str(), 1),
    };
    let n = digits.trim().parse::<usize>().map_err(|_| {
        Fail::Usage(format!(
            "a size wants a number like 65536 or 64k, got {s:?}"
        ))
    })?;
    Ok(n.saturating_mul(mult))
}

// -- extract -----------------------------------------------------------------

fn cmd_extract(args: &[String]) -> Result<(), Fail> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    if positional.is_empty() {
        return usage("`extract` needs an archive");
    }
    if positional.len() > 2 {
        return usage("`extract` takes an archive and an optional directory");
    }
    let mut ar = Archive::open(Path::new(positional[0].as_str())).map_err(Fail::Error)?;
    let dest = positional
        .get(1)
        .map(|s| PathBuf::from(s.as_str()))
        .unwrap_or_else(|| PathBuf::from("."));

    // Refuse before writing anything, rather than halfway through.
    let written = ar
        .extract_to(&dest)
        .map_err(|e| Fail::Error(e.to_string()))?;
    println!(
        "extracted {} to {}",
        densr::util::human(written as f64),
        dest.display()
    );
    Ok(())
}

// -- list --------------------------------------------------------------------

fn cmd_list(args: &[String]) -> Result<(), Fail> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    if positional.len() != 1 {
        return usage("`list` needs exactly one archive");
    }
    let ar = Archive::open(Path::new(positional[0].as_str())).map_err(Fail::Error)?;
    let entries = ar.entries();
    let mut total = 0u64;
    for e in &entries {
        total += e.size;
        let kind = match e.kind {
            EntryKind::Directory => 'd',
            EntryKind::Symlink => 'l',
            _ => '-',
        };
        println!(
            "{kind} {:>10}  {:<6}  {}",
            if e.kind == EntryKind::Directory {
                "-".to_string()
            } else {
                densr::util::human(e.size as f64)
            },
            e.class.label(),
            e.path
        );
    }
    println!(
        "--\n{} entries, {} of content",
        entries.len(),
        densr::util::human(total as f64)
    );
    Ok(())
}

// -- info --------------------------------------------------------------------

fn cmd_info(args: &[String]) -> Result<(), Fail> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    if positional.len() != 1 {
        return usage("`info` needs exactly one archive");
    }
    let mut ar = Archive::open(Path::new(positional[0].as_str())).map_err(Fail::Error)?;
    // Read before borrowing the header, since counting needs `&mut` for seeking.
    let kind_counts = ar.block_kind_counts();
    let h = ar.header();

    println!("archive     {}", ar.path().display());
    println!(
        "version     v{} (format v{})",
        env!("CARGO_PKG_VERSION"),
        h.version
    );
    println!("created     {}", densr::util::format_time(h.created));
    println!("level       {}", h.level);
    println!("block size  {}", h.block_size);
    println!("blocks      {}", ar.block_count());
    match kind_counts {
        Ok(counts) => {
            let parts: Vec<String> = counts
                .lines()
                .iter()
                .map(|(n, c)| format!("{n}={c}"))
                .collect();
            if !parts.is_empty() {
                println!("block kinds {}", parts.join("  "));
            }
        }
        Err(e) => println!("block kinds unreadable: {e}"),
    }
    println!(
        "checksums   {}",
        [
            ("crc", densr::format::FLAG_CHECKSUMS),
            ("dedup", densr::format::FLAG_DEDUP),
            ("resync", densr::format::FLAG_RESYNC),
            ("metadata", densr::format::FLAG_METADATA),
        ]
        .iter()
        .map(|(n, f)| format!("{n}{}", if h.has(*f) { "=yes" } else { "=no" }))
        .collect::<Vec<_>>()
        .join("  ")
    );
    println!("{}", ar.summary());
    Ok(())
}

// -- test --------------------------------------------------------------------

fn cmd_test(args: &[String]) -> Result<(), Fail> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    if positional.len() != 1 {
        return usage("`test` needs exactly one archive");
    }
    let mut ar = Archive::open(Path::new(positional[0].as_str())).map_err(Fail::Error)?;
    let r = ar.test().map_err(Fail::Error)?;
    println!(
        "ok: {} entries ({} files, {} dirs, {} symlinks), {} block references, {} of content",
        r.entries,
        r.files,
        r.dirs,
        r.symlinks,
        r.blocks,
        densr::util::human(r.logical as f64)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse_plain_and_suffixed() {
        assert_eq!(parse_size("65536").unwrap(), 65536);
        assert_eq!(parse_size("64k").unwrap(), 65536);
        assert_eq!(parse_size("4M").unwrap(), 4 * 1024 * 1024);
        assert_eq!(parse_size(" 2mb ").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_size("2MiB").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_size("1G").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size("4b").unwrap(), 4, "a bare b still means bytes");
    }

    #[test]
    fn bad_sizes_are_usage_errors() {
        // Reported as a usage error so the exit code tells the caller it was the
        // command line, not the archive.
        for s in ["", "k", "64x", "-1", "many"] {
            match parse_size(s) {
                Err(Fail::Usage(_)) => {}
                other => panic!("{s:?} gave {other:?}"),
            }
        }
    }

    #[test]
    fn equals_form_is_expanded() {
        assert_eq!(
            expand_equals(&["--level=3".into(), "--block-size=64k".into()]),
            vec![
                "--level".to_string(),
                "3".to_string(),
                "--block-size".to_string(),
                "64k".to_string()
            ]
        );
        // A value may contain an equals sign; only the first one splits.
        assert_eq!(
            expand_equals(&["--block-size=a=b".into()]),
            vec!["--block-size".to_string(), "a=b".to_string()]
        );
        // Short flags and bare words are untouched.
        assert_eq!(
            expand_equals(&["-l".into(), "3".into(), "file=dnsr".into()]),
            vec!["-l".to_string(), "3".to_string(), "file=dnsr".to_string()]
        );
    }

    #[test]
    fn help_and_version_short_circuit() {
        for args in [
            vec!["--help".to_string()],
            vec!["add".to_string(), "-h".to_string()],
            vec!["-V".to_string()],
        ] {
            assert!(run(&args).is_ok());
        }
    }

    #[test]
    fn unknown_commands_and_missing_arguments_are_usage_errors() {
        for args in [
            vec!["frobnicate".to_string()],
            vec!["add".to_string()],
            vec!["add".to_string(), "only-archive.dnsr".to_string()],
            vec!["list".to_string()],
            vec!["extract".to_string()],
            vec![
                "add".to_string(),
                "--nonsense".to_string(),
                "a".to_string(),
                "b".to_string(),
            ],
            vec!["add".to_string(), "-l".to_string()],
            vec![
                "add".to_string(),
                "--level".to_string(),
                "99".to_string(),
                "a".to_string(),
                "b".to_string(),
            ],
        ] {
            match run(&args) {
                Err(Fail::Usage(_)) => {}
                other => panic!("{args:?} gave {other:?}"),
            }
        }
    }

    #[test]
    fn add_flag_parsing_accepts_the_documented_forms() {
        let mk = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        // None of these should be a usage error; the paths simply do not exist,
        // which is an ordinary error.
        for args in [
            mk(&["add", "out.dnsr", "-l", "9", "x"]),
            mk(&["add", "out.dnsr", "--level", "0", "x"]),
            mk(&["add", "out.dnsr", "-b", "8k", "x"]),
            mk(&["add", "out.dnsr", "--block-size=1M", "x"]),
            mk(&["add", "out.dnsr", "--level=3", "x"]),
            mk(&["add", "--fixed", "-q", "out.dnsr", "x"]),
            mk(&["add", "out.dnsr", "--no-metadata", "--no-verify", "x"]),
            mk(&["add", "out.dnsr", "--", "-weird-name"]),
        ] {
            match run(&args) {
                Err(Fail::Usage(m)) => panic!("{args:?} rejected: {m}"),
                Err(Fail::Error(_)) | Ok(()) => {}
            }
        }
    }

    #[test]
    fn level_zero_is_allowed_and_thirteen_is_not() {
        let mk = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        // Level 0 means "store without coding", which is a real mode.
        assert!(!matches!(
            run(&mk(&["add", "o.dnsr", "-l", "0", "missing"])),
            Err(Fail::Usage(_))
        ));
        assert!(matches!(
            run(&mk(&["add", "o.dnsr", "-l", "13", "missing"])),
            Err(Fail::Usage(_))
        ));
    }
}
