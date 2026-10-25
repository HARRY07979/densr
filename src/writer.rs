//! Turning a directory tree into an archive.
//!
//! # The order of decisions
//!
//! For each block, in this order:
//!
//! 1. **Bypass.** An mp3 or a png does not get better by being run through the
//!    codec -- at best it comes out the same size, and in practice a little
//!    larger. Storing it verbatim also keeps it bit-identical, which matters for
//!    anything downstream that checks its own hashes.
//! 2. **Deduplicate.** If this exact block is already in the archive, point at it
//!    and store nothing.
//! 3. **Compress.** Otherwise hand it to the codec, and if the result is not
//!    actually smaller, store the original instead. A block that would grow is
//!    stored as `Stored` and costs one flag byte.
//!
//! # Memory
//!
//! The writer appends to the output as it goes rather than assembling the whole
//! archive in memory, so the peak footprint is one file plus the directory. The
//! directory grows with the input, but it is a few dozen bytes per block index,
//! not per block.
//!
//! Because the header holds the payload length and the directory position, and
//! neither is known until the end, it is written first as a placeholder and
//! rewritten once the numbers are final. Everything after it -- payload,
//! directory, footer -- is already in the right place and does not move.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::checksum::{crc32, hash64};
use crate::chunking::{self, Chunking, BLOCK_SIZE, CDC_DEFAULT_TARGET};
use crate::format::{
    encode_dictionary, BlockKind, BlockRec, Directory, EntryKind, EntryRec, Footer, Header,
    BLOCK_FLAG_USES_DICT, BLOCK_RECORD_SIZE, FLAG_CHECKSUMS, FLAG_CHUNKING_CDC, FLAG_DEDUP,
    FLAG_METADATA, FLAG_RESYNC, FOOTER_SIZE, FORMAT_VERSION, HEADER_SIZE,
};
use crate::magic::{classify_bytes, FileClass};

/// How much of a file to read when deciding what it is.
const HEADER_PROBE: usize = 4096;

/// Whether the branch filter should run on a file with this header.
///
/// Executables are the only case: the filter rewrites relative branch
/// displacements into absolute ones, which helps code but hurts the arbitrary
/// data (symbol tables, strings) inside an `ar` archive -- measured on `lib`, it
/// made the archive larger, so object containers are left alone.
fn wants_bcj(head: &[u8]) -> bool {
    crate::magic::classify_header(head) == Some(crate::magic::Format::Executable)
}

// --- Per-archive dictionary ------------------------------------------------
//
// Every block is otherwise a fresh LZ problem with no history, so a corpus of
// small or similar files pays for the same headers, tables and startup code
// over and over. A dictionary sampled from this archive's own content is seeded
// into every block's match table and stored once, compressed, in the archive.
// The result is self-contained: nothing outside the file is needed to read it.
//
// The samples are taken with the same transform the blocks get, so a block from
// an executable can match a dictionary sample from an executable. The cost is
// one pass over the input to collect them, plus the dictionary's own (coded)
// payload. Both are bounded and paid once.

/// Below this much input a dictionary's own payload is likely to cost more than
/// the matches it creates, so none is built.
const MIN_INPUT_FOR_DICT: u64 = 512 * 1024;

/// A dictionary smaller than this is not worth a block record or the flag on
/// every block that uses it.
const MIN_DICT: usize = 16 * 1024;

/// How much of a single file may go into the dictionary. The per-file share is
/// `dict_size / file_count` bound by these.
const DICT_SAMPLE_MIN: usize = 4 * 1024;
const DICT_SAMPLE_MAX: usize = 256 * 1024;

// --- Smart GPU offload -----------------------------------------------------
//
// Measured on the reference integrated GPU, the OpenCL path costs 112.4 s
// per 100 MiB against 26.7 s on the CPU: 4.2x slower. On an i3-10105 / UHD 630
// the same thing is worse -- 1.47 s per 256 KiB block against 0.107 s on the
// CPU, about 13.7x. An iGPU shares system RAM and has no dedicated bandwidth,
// so that is the expected shape of the result, not a bug in the kernel.
//
// The conclusion that follows is that a fixed share is the wrong knob. At 25%
// offload on the UHD 630 a 206-block corpus took 75.4 s against 22.2 s on the
// CPU alone, because one GPU worker cannot absorb even a quarter of the load.
// The GPU becomes the critical path and the whole run waits on it.
//
// So the default measures instead of guessing. The CPU workers' per-block cost
// and the GPU's are both observed during the run, and the share is set to the
// fraction the GPU can take while still running *behind* the CPUs. On this
// hardware that lands near zero and the GPU is effectively left alone; on a
// discrete part it lands high. Either way `--gpu` stops being a trap.

/// Blocks that must be submitted before `auto` will consider the GPU at all.
/// Below this the archive is small enough that opening a context and compiling
/// the kernel cannot pay for itself, so it is never opened.
const AUTO_MIN_BLOCKS: usize = 256;

/// Blocks routed to the GPU purely to measure it, and the number of CPU blocks
/// measured before that. Kept small on purpose: the probe is a cost paid
/// unconditionally, and a GPU that turns out to be slower should be discovered
/// as cheaply as possible.
const AUTO_PROBE_BLOCKS: usize = 8;

/// Shrink the computed share slightly so the GPU stays off the critical path
/// even when run-to-run timing noise pushes it to the edge.
const AUTO_SAFETY: f64 = 0.9;

/// How to write an archive.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    pub level: u8,
    pub block_size: usize,
    pub chunking: Chunking,
    /// Record mtime and the unix mode in the directory.
    pub metadata: bool,
    /// Decode every block back and compare it, catching codec bugs at write time
    /// rather than at extract time.
    pub verify: bool,
    /// How many worker threads compress blocks. One is allowed and
    /// simply means no parallelism.
    pub threads: usize,
    /// Also compress blocks on the GPU via OpenCL (falls back to CPU-only on error).
    pub gpu: bool,
    /// Percent of blocks (0-100) routed to the GPU queue when `gpu` is on.
    /// `None` means *auto*: measure the GPU against the CPU workers and offload
    /// only as much as the GPU can absorb without becoming the bottleneck.
    /// An explicit value pins the share and skips the measurement.
    ///
    /// Routing never changes the archive bytes either way, so `auto` is a speed
    /// decision only.
    pub gpu_share: Option<u8>,
    /// Max blocks per GPU kernel launch.
    pub gpu_batch: usize,
    /// Where CDC aims to put boundaries. Separate from `block_size` because the
    /// two want different answers: the codec prefers bigger chunks, dedup
    /// prefers smaller ones. See [`crate::chunking::CDC_DEFAULT_TARGET`].
    pub cdc_target: usize,
    /// Upper bound on the per-archive shared dictionary, in bytes. `0` disables
    /// it. The dictionary is sampled from this archive's own content, coded, and
    /// stored in the archive, so the archive stays self-contained.
    pub dict_size: usize,
}

impl Default for WriteOptions {
    fn default() -> Self {
        WriteOptions {
            level: 4,
            block_size: BLOCK_SIZE,
            chunking: Chunking::Cdc,
            metadata: true,
            verify: true,
            threads: default_threads(),
            gpu: false,
            gpu_share: None,
            gpu_batch: 32,
            cdc_target: CDC_DEFAULT_TARGET,
            dict_size: 1024 * 1024,
        }
    }
}

/// How many workers to use when the caller did not say.
fn default_threads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// What the writer did, for `--verbose` and the summary line.
#[derive(Debug, Clone, Default)]
pub struct WriteReport {
    pub files: usize,
    pub dirs: usize,
    pub symlinks: usize,
    /// Block records actually written.
    pub blocks: usize,
    /// Block references that point at an earlier block instead of storing data.
    pub dedup_blocks: usize,
    /// Blocks stored verbatim because the codec was skipped or did not help.
    pub bypass_blocks: usize,
    pub coded_blocks: usize,
    pub input_bytes: u64,
    pub payload_bytes: u64,
    pub archive_bytes: u64,
    pub bytes_saved: u64,
    /// Blocks successfully compressed by the GPU (hybrid path only).
    pub gpu_blocks: usize,
    /// The offload share in force for this run, when `--gpu` was on. `Some(0)`
    /// means the GPU was available but never worth using; `None` means no GPU
    /// was asked for. Auto mode picks the number, an explicit `--gpu-share`
    /// pins it.
    pub gpu_share: Option<usize>,
    /// Blocks tagged per `BlockKind`, so `add` can report exactly which
    /// transformations were applied and how often.
    pub kinds: KindCounts,
    /// Nanoseconds spent per stage, summed over every block. CPU worker time
    /// only; GPU codec time is tracked separately by the offload statistics.
    pub ns_bcj: u64,
    pub ns_lz: u64,
    pub ns_huff: u64,
    /// Total bytes the `dsr1` codec emitted for blocks that were written, before
    /// Huffman. Together with the archive's payload size this is what the
    /// entropy stage is actually worth.
    pub coded_bytes: u64,
    /// Payload bytes written for the same blocks, after Huffman.
    pub huffman_payload_bytes: u64,
    /// Payload bytes the shared dictionary itself cost, after coding. Zero when
    /// no dictionary was written.
    pub dict_bytes: u64,
}

/// How many blocks landed in each `BlockKind`, and what they cost.
///
/// Counted rather than inferred: the only way to tell whether the branch filter
/// actually ran is to look at what the writer chose. The byte totals answer the
/// other question -- whether a stage that runs is a stage that pays -- which a
/// timing alone cannot.
#[derive(Debug, Clone, Copy, Default)]
pub struct KindStats {
    pub blocks: usize,
    /// Bytes the block expands to, before coding.
    pub raw: u64,
    /// Bytes actually written, excluding the fixed record header.
    pub payload: u64,
}

impl KindStats {
    /// How many bytes coding saved, as a fraction of the original.
    pub fn gain(&self) -> f64 {
        if self.raw == 0 {
            return 0.0;
        }
        1.0 - self.payload as f64 / self.raw as f64
    }
}

/// The per-kind tally for a whole write.
#[derive(Debug, Clone, Copy, Default)]
pub struct KindCounts {
    pub stored: KindStats,
    pub coded: KindStats,
    pub coded_huff: KindStats,
    pub coded_x86: KindStats,
    pub coded_x86_huff: KindStats,
    pub dictionary: KindStats,
    pub coded_huff_split: KindStats,
    pub coded_x86_huff_split: KindStats,
}

impl KindCounts {
    pub fn record(&mut self, kind: BlockKind, raw_len: u32, payload_len: usize) {
        let s = match kind {
            BlockKind::Stored => &mut self.stored,
            BlockKind::Coded => &mut self.coded,
            BlockKind::CodedHuff => &mut self.coded_huff,
            BlockKind::CodedX86 => &mut self.coded_x86,
            BlockKind::CodedX86Huff => &mut self.coded_x86_huff,
            BlockKind::Dictionary => &mut self.dictionary,
            BlockKind::CodedHuffSplit => &mut self.coded_huff_split,
            BlockKind::CodedX86HuffSplit => &mut self.coded_x86_huff_split,
        };
        s.blocks += 1;
        s.raw += raw_len as u64;
        s.payload += payload_len as u64;
    }

    pub fn get(&self, kind: BlockKind) -> KindStats {
        match kind {
            BlockKind::Stored => self.stored,
            BlockKind::Coded => self.coded,
            BlockKind::CodedHuff => self.coded_huff,
            BlockKind::CodedX86 => self.coded_x86,
            BlockKind::CodedX86Huff => self.coded_x86_huff,
            BlockKind::Dictionary => self.dictionary,
            BlockKind::CodedHuffSplit => self.coded_huff_split,
            BlockKind::CodedX86HuffSplit => self.coded_x86_huff_split,
        }
    }

    /// Blocks that went through the x86 branch filter.
    pub fn filtered_blocks(&self) -> usize {
        self.coded_x86.blocks + self.coded_x86_huff.blocks
    }

    /// Every kind that was used, paired with a label for it.
    pub fn all(&self) -> [(&'static str, BlockKind, KindStats); 8] {
        [
            ("stored", BlockKind::Stored, self.stored),
            ("coded", BlockKind::Coded, self.coded),
            ("coded+huffman", BlockKind::CodedHuff, self.coded_huff),
            ("coded+x86", BlockKind::CodedX86, self.coded_x86),
            (
                "coded+x86+huffman",
                BlockKind::CodedX86Huff,
                self.coded_x86_huff,
            ),
            ("dictionary", BlockKind::Dictionary, self.dictionary),
            (
                "coded+huffman+split",
                BlockKind::CodedHuffSplit,
                self.coded_huff_split,
            ),
            (
                "coded+x86+huffman+split",
                BlockKind::CodedX86HuffSplit,
                self.coded_x86_huff_split,
            ),
        ]
    }

    /// One `name=count` entry per kind that was actually used.
    pub fn lines(&self) -> Vec<(&'static str, usize)> {
        self.all()
            .into_iter()
            .filter(|(_, _, s)| s.blocks > 0)
            .map(|(n, _, s)| (n, s.blocks))
            .collect()
    }
}

impl WriteReport {
    pub fn ratio(&self) -> f64 {
        if self.input_bytes == 0 {
            return 1.0;
        }
        self.input_bytes as f64 / self.archive_bytes.max(1) as f64
    }

    pub fn summary(&self) -> String {
        let gpu_note = match self.gpu_share {
            None => String::new(),
            Some(share) => {
                let what = if self.gpu_blocks > 0 {
                    format!("{} coded on GPU", self.gpu_blocks)
                } else {
                    "left unused".to_string()
                };
                format!("\ngpu: {what}, {share}% offload")
            }
        };
        let kinds: Vec<String> = self
            .kinds
            .lines()
            .iter()
            .map(|(n, c)| format!("{n}={c}"))
            .collect();
        let stages = self.stage_note();
        format!(
            "{} files, {} dirs, {} symlinks\n\
             {} blocks written: {} coded, {} stored, {} references reused\n\
             block kinds: {}\n\
             stages: {}\n\
             {}\n\
             {} in, {} out, {:.2}x{}",
            self.files,
            self.dirs,
            self.symlinks,
            self.blocks,
            self.coded_blocks,
            self.bypass_blocks,
            self.dedup_blocks,
            kinds.join("  "),
            stages,
            self.gain_table(),
            crate::util::human(self.input_bytes as f64),
            crate::util::human(self.archive_bytes as f64),
            self.ratio(),
            gpu_note,
        )
    }

    /// Where the time went, per pipeline stage.
    ///
    /// CPU worker time only. With `--gpu` the LZ figure reads zero because the
    /// codec ran on the GPU and its time is in the offload statistics instead.
    pub fn stage_note(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        format!(
            "bcj {:.0}ms, lz {:.0}ms, huffman {:.0}ms",
            ms(self.ns_bcj),
            ms(self.ns_lz),
            ms(self.ns_huff)
        )
    }

    /// A per-kind byte breakdown: what each stage cost, and what it saved.
    ///
    /// The entropy line is the important one. A stage that runs but saves nothing
    /// is pure cost, and this is the only place that shows which of the two it is.
    pub fn gain_table(&self) -> String {
        use crate::util::human;
        let mut rows: Vec<String> = Vec::new();
        let mut raw_total = 0u64;
        let mut payload_total = 0u64;
        let mut block_total = 0usize;
        for (name, _, s) in self.kinds.all() {
            if s.blocks == 0 {
                continue;
            }
            raw_total += s.raw;
            payload_total += s.payload;
            block_total += s.blocks;
            rows.push(format!(
                "  {name:<20} {:>4} blocks  {:>9} -> {:>9}  {:>6.2}x",
                s.blocks,
                human(s.raw as f64),
                human(s.payload as f64),
                if s.payload == 0 {
                    0.0
                } else {
                    s.raw as f64 / s.payload as f64
                }
            ));
        }
        let entropy = if self.coded_bytes > 0 {
            let saved = self.coded_bytes.saturating_sub(self.huffman_payload_bytes);
            format!(
                "  entropy (huffman) saved {} of {} lz bytes, {:.1}%",
                human(saved as f64),
                human(self.coded_bytes as f64),
                100.0 * (1.0 - self.huffman_payload_bytes as f64 / self.coded_bytes.max(1) as f64)
            )
        } else {
            "  entropy (huffman) had nothing to work on".to_string()
        };
        format!(
            "block breakdown:\n{}\n{}\n  total              {:>4} blocks  {:>9} -> {:>9}  {:>6.2}x",
            rows.join("\n"),
            entropy,
            block_total,
            human(raw_total as f64),
            human(payload_total as f64),
            if payload_total == 0 {
                0.0
            } else {
                raw_total as f64 / payload_total as f64
            }
        )
    }

    /// Seconds spent in each stage, for callers that want to compare runs.
    pub fn stage_seconds(&self) -> [f64; 3] {
        [
            self.ns_bcj as f64 / 1e9,
            self.ns_lz as f64 / 1e9,
            self.ns_huff as f64 / 1e9,
        ]
    }
}

// ---------------------------------------------------------------------------
// Parallel block compression
// ---------------------------------------------------------------------------

/// A block handed to a worker: its bytes, and whether the codec may help.
struct BlockJob {
    raw: Vec<u8>,
    /// Whether this block came from an executable, so the branch filter applies.
    /// Decided from the file's magic bytes, never its name.
    executable: bool,
}

/// What a worker learned about a block.
///
/// The hashes travel back with the payload so the writer thread, which
/// owns the dedup table and the output file, never hashes a block twice.
struct BlockDone {
    raw_len: u32,
    hash: u64,
    raw_crc: u32,
    kind: BlockKind,
    payload: Vec<u8>,
    /// Whether this block was coded by the GPU (hybrid path only).
    via_gpu: bool,
    /// Nanoseconds spent in each stage, for the per-stage breakdown. These are
    /// statistics gathered on worker threads and read by the writer thread; they
    /// cannot change a byte of the archive.
    ns_bcj: u32,
    ns_lz: u32,
    ns_huff: u32,
    /// Bytes the `dsr1` codec produced, before Huffman. Zero when the block was
    /// stored. Keeping it is what makes the entropy stage's contribution
    /// measurable instead of assumed: the difference against the final payload
    /// is exactly what Huffman saved.
    coded_len: u32,
    /// This block's offsets reach into the archive's shared dictionary, so its
    /// record sets [`BLOCK_FLAG_USES_DICT`] and a reader must load the
    /// dictionary before decoding it.
    uses_dict: bool,
}

/// A queued block plus the channel its result travels back on.
type Job = (BlockJob, Sender<Result<BlockDone, String>>);

/// A pool of workers that hash and compress blocks.
///
/// Blocks are independent, and a block's coded form is a pure function
/// of its bytes and the level, so compression parallelises without
/// changing the result: the writer thread keeps every decision that
/// touches order -- deduplication, block ids, the byte stream -- and
/// the archive comes out byte-for-byte identical to a single-threaded
/// write.
/// What the smart offload policy measures during a run. Every field is only
/// ever added to, by whichever worker thread made the observation, and every
/// read is `Relaxed` because the numbers are statistics: a slightly stale read
/// only costs a slightly wrong share, and a wrong share cannot change the bytes
/// written.
#[derive(Default)]
struct Stats {
    cpu_blocks: AtomicU64,
    cpu_nanos: AtomicU64,
    gpu_blocks: AtomicU64,
    gpu_nanos: AtomicU64,
}

impl Stats {
    fn record_cpu(&self, elapsed: std::time::Duration) {
        self.cpu_blocks.fetch_add(1, Ordering::Relaxed);
        self.cpu_nanos
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    fn record_gpu(&self, blocks: usize, elapsed: std::time::Duration) {
        self.gpu_blocks.fetch_add(blocks as u64, Ordering::Relaxed);
        self.gpu_nanos
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }
}

struct BlockPool {
    cpu_jobs: Option<SyncSender<Job>>,
    gpu_jobs: Option<SyncSender<Job>>,
    workers: Vec<JoinHandle<()>>,
    /// Round-robin dispatcher: the next submit with `dispatch % 100 < share`
    /// goes to the GPU queue, the rest to the CPU queue.
    dispatch: Arc<AtomicUsize>,
    stats: Arc<Stats>,
    /// Blocks actually routed to the GPU, sent or completed. Stops the probe
    /// from turning into a burst.
    gpu_routed: AtomicUsize,
    /// An explicit `--gpu-share`, which skips measurement entirely. `None` is
    /// auto mode.
    pinned_share: Option<usize>,
    /// The share auto mode settled on, and whether it has settled. Set once the
    /// GPU has completed a full probe batch so the ratio stops drifting
    /// mid-archive.
    settled_share: AtomicUsize,
    auto_settled: AtomicBool,
    /// CPU worker count. `cpu_blocks` and `cpu_nanos` are summed over workers,
    /// so throughput needs this to turn summed worker-time into wall-clock
    /// throughput.
    threads: usize,
}

impl BlockPool {
    fn new(
        threads: usize,
        level: u8,
        verify: bool,
        use_gpu: bool,
        gpu_share: Option<u8>,
        gpu_batch: usize,
        seed: Option<Arc<crate::lz::DictSeed>>,
    ) -> BlockPool {
        let stats = Arc::new(Stats::default());

        // A few blocks in flight per worker keeps every core busy
        // without holding more than a handful of blocks in memory.
        // The queue is deeper than that on purpose: a worker that has to
        // wait for room is a core idling, and the writer thread is busy
        // hashing, deduplicating and writing what came back.
        let (cpu_tx, cpu_rx) = sync_channel::<Job>((threads * 8).max(16));
        let cpu_rx = Arc::new(Mutex::new(cpu_rx));
        let mut workers = Vec::with_capacity(threads + 1);

        // CPU workers.
        for _ in 0..threads {
            let rx = Arc::clone(&cpu_rx);
            let stats = Arc::clone(&stats);
            let seed = seed.clone();
            workers.push(thread::spawn(move || {
                loop {
                    // Drop the receiver lock before compressing. Keeping the
                    // guard in the `while let` scrutinee would hold it for the
                    // whole body and serialize every worker.
                    let next = rx.lock().unwrap().recv();
                    let Ok((job, done)) = next else { break };
                    let t0 = std::time::Instant::now();
                    let out = compress_block(job, level, verify, seed.as_deref());
                    stats.record_cpu(t0.elapsed());
                    let _ = done.send(out);
                }
            }));
        }

        // GPU worker (hybrid path only) with its own queue: it batches jobs
        // via try_recv to fill a GPU kernel launch, falls back to CPU
        // compression on any OpenCL error. CPU and GPU workers run in
        // parallel on their own queues, so both stay fed.
        let gpu_jobs = if use_gpu {
            let (gpu_tx, gpu_rx) = sync_channel::<Job>((gpu_batch * 8).max(64));
            let stats = Arc::clone(&stats);
            workers.push(thread::spawn(move || {
                gpu_worker(gpu_rx, level, verify, gpu_batch, stats);
            }));
            Some(gpu_tx)
        } else {
            None
        };

        BlockPool {
            cpu_jobs: Some(cpu_tx),
            gpu_jobs,
            workers,
            dispatch: Arc::new(AtomicUsize::new(0)),
            stats,
            gpu_routed: AtomicUsize::new(0),
            pinned_share: gpu_share.map(|p| p.min(100) as usize),
            settled_share: AtomicUsize::new(0),
            auto_settled: AtomicBool::new(gpu_share.is_some()),
            threads,
        }
    }

    /// Blocks per second the CPU workers can absorb, from what they have done so
    /// far. `cpu_nanos` is summed across workers, so it is divided by the
    /// worker count to get elapsed wall time.
    fn cpu_blocks_per_sec(&self) -> f64 {
        let blocks = self.stats.cpu_blocks.load(Ordering::Relaxed) as f64;
        let nanos = self.stats.cpu_nanos.load(Ordering::Relaxed) as f64;
        if blocks <= 0.0 || nanos <= 0.0 {
            return 0.0;
        }
        blocks * self.threads as f64 / (nanos / 1e9)
    }

    /// Blocks per second the single GPU worker can absorb.
    fn gpu_blocks_per_sec(&self) -> f64 {
        let blocks = self.stats.gpu_blocks.load(Ordering::Relaxed) as f64;
        let nanos = self.stats.gpu_nanos.load(Ordering::Relaxed) as f64;
        if blocks <= 0.0 || nanos <= 0.0 {
            return 0.0;
        }
        blocks / (nanos / 1e9)
    }

    /// The share to route the next block with.
    ///
    /// In auto mode this walks the run in three stages: nothing goes to the GPU
    /// until the archive is big enough to be worth opening a context, the CPU
    /// workers get a probe batch first so there is something to compare against,
    /// and then the share is whatever keeps the GPU finishing *behind* the CPUs.
    fn current_share(&self) -> usize {
        if let Some(p) = self.pinned_share {
            return p;
        }
        if self.auto_settled.load(Ordering::Relaxed) {
            return self.settled_share.load(Ordering::Relaxed);
        }
        // Too little work to open a context for. This is what keeps `--gpu` off
        // small archives entirely: no block is ever routed, so the worker never
        // gets past its initial receive and OpenCL is never touched.
        if self.dispatch.load(Ordering::Relaxed) < AUTO_MIN_BLOCKS {
            return 0;
        }
        // Measure the CPU side first, so the comparison has a denominator.
        if self.stats.cpu_blocks.load(Ordering::Relaxed) < AUTO_PROBE_BLOCKS as u64 {
            return 0;
        }
        let cpu = self.cpu_blocks_per_sec();
        if cpu <= 0.0 {
            return 0;
        }
        let gpu = self.gpu_blocks_per_sec();
        if gpu <= 0.0 {
            // Nothing timed on the GPU yet. Route exactly one small probe: long
            // enough to time a kernel launch, short enough that finding out the
            // GPU is slow costs a handful of blocks rather than a third of the
            // archive.
            if self.gpu_routed.load(Ordering::Relaxed) < AUTO_PROBE_BLOCKS {
                return 100;
            }
            // The probe is in flight and has not reported yet. Wait for it rather
            // than guessing at a rate.
            return 0;
        }

        let measured = self.stats.gpu_blocks.load(Ordering::Relaxed) >= AUTO_PROBE_BLOCKS as u64;

        // The writer collects block results in submission order, so a block on
        // the GPU holds up every later block no matter how many CPU workers are
        // idle. Offloading therefore only pays when the GPU finishes blocks
        // *faster* than the CPU pool does. When it does not, the arithmetic below
        // would still pick a small non-zero share and still lose, because the
        // probe has already been paid for and no share earns it back. Stop at
        // zero instead.
        let share = if gpu <= cpu {
            0
        } else {
            // Of the work on offer, the share the GPU can finish before the CPUs
            // do. Above this the GPU becomes the critical path.
            ((100.0 * AUTO_SAFETY * gpu / (cpu + gpu)).round() as usize).min(100)
        };

        // One probe batch is enough to know the rate; freeze it so the ratio
        // cannot drift part-way through an archive.
        if measured {
            self.settled_share.store(share, Ordering::Relaxed);
            self.auto_settled.store(true, Ordering::Relaxed);
        }
        share
    }

    /// The share in force when the archive finished, for the report.
    fn effective_share(&self) -> usize {
        if let Some(p) = self.pinned_share {
            return p;
        }
        if self.auto_settled.load(Ordering::Relaxed) {
            return self.settled_share.load(Ordering::Relaxed);
        }
        // Never settled: the archive was too small to be worth opening a
        // context for, so nothing was ever routed.
        0
    }

    /// Queue a block and return the channel its result will arrive on.
    fn submit(
        &self,
        raw: Vec<u8>,
        executable: bool,
    ) -> io::Result<Receiver<Result<BlockDone, String>>> {
        let (done_tx, done_rx) = channel::<Result<BlockDone, String>>();
        let job = (BlockJob { raw, executable }, done_tx);
        // Route `share`% of blocks to the GPU queue, the rest to CPUs.
        // When the GPU queue is full, spill over to the CPU queue so the
        // writer never stalls.
        let share = self.current_share();
        let to_gpu =
            self.gpu_jobs.is_some() && self.dispatch.fetch_add(1, Ordering::Relaxed) % 100 < share;
        if to_gpu {
            self.gpu_routed.fetch_add(1, Ordering::Relaxed);
            let tx = self.gpu_jobs.as_ref().expect("gpu queue alive");
            match tx.try_send(job) {
                Ok(()) => return Ok(done_rx),
                Err(std::sync::mpsc::TrySendError::Full((j, d))) => {
                    let _ = self
                        .cpu_jobs
                        .as_ref()
                        .expect("the pool is alive while the archive is being written")
                        .send((j, d));
                    return Ok(done_rx);
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    return Err(io::Error::other("the worker pool closed"));
                }
            }
        }
        self.cpu_jobs
            .as_ref()
            .expect("the pool is alive while the archive is being written")
            .send(job)
            .map_err(|e| io::Error::other(format!("the worker pool closed: {e}")))?;
        Ok(done_rx)
    }
}

impl Drop for BlockPool {
    fn drop(&mut self) {
        // Dropping the last senders tells every worker that no more
        // work is coming, so they exit and are joined here.
        self.cpu_jobs.take();
        self.gpu_jobs.take();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// GPU worker: batches jobs from the shared queue, compresses them via OpenCL,
/// and falls back to CPU compression on any error. Never panics; always sends
/// a result back on every job's channel.
fn gpu_worker(rx: Receiver<Job>, level: u8, verify: bool, gpu_batch: usize, stats: Arc<Stats>) {
    use crate::gpu::GpuBackend;

    let max_batch = gpu_batch.clamp(1, 32);
    // Opened on the first job rather than at thread start. If the pool never
    // routes a block this way -- auto mode on an archive too small to be worth
    // opening a context for, or any run pinned to a share of 0 -- this thread
    // stays blocked in the recv below and the OpenCL runtime is never loaded, no
    // context is created and no kernel is compiled. That is the entire cost of
    // asking for the GPU on a small job, and it is now zero.
    let mut gpu: Option<crate::gpu::Gpu> = None;

    loop {
        // First job: blocking recv so the thread idles when there is no work.
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => break,
        };
        let batch_start = std::time::Instant::now();
        let mut batch = vec![first];

        // Fill the batch via try_recv: many small tasks per kernel launch.
        {
            let mut bytes = batch[0].0.raw.len();
            while batch.len() < max_batch {
                match rx.try_recv() {
                    Ok(j) => {
                        bytes += j.0.raw.len();
                        batch.push(j);
                        if bytes >= 8 * 1024 * 1024 {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        // A job has arrived, so a context is worth having now.
        if gpu.is_none() {
            match crate::gpu::Gpu::open() {
                Ok(g) => {
                    eprintln!(
                        "densr: using GPU: {} ({})",
                        g.device_name(),
                        g.backend_name()
                    );
                    gpu = Some(g);
                }
                Err(e) => {
                    eprintln!("densr: GPU unavailable ({e}), compressing on CPU only");
                    // Behave as a plain CPU worker for the rest of the run.
                    for (job, done) in batch {
                        let t0 = std::time::Instant::now();
                        let out = compress_block(job, level, verify, None);
                        stats.record_cpu(t0.elapsed());
                        let _ = done.send(out);
                    }
                    while let Ok((job, done)) = rx.recv() {
                        let t0 = std::time::Instant::now();
                        let out = compress_block(job, level, verify, None);
                        stats.record_cpu(t0.elapsed());
                        let _ = done.send(out);
                    }
                    return;
                }
            }
        }

        // Compress the batch on the GPU. Executable blocks are branch-filtered
        // first, on the host, so the kernel sees exactly the bytes the CPU
        // codec would see. If the two paths ever disagree, the archives differ,
        // which is the intended check on that invariant.
        let mut bcj_ns: Vec<u32> = Vec::with_capacity(batch.len());
        let results = {
            let owned: Vec<Vec<u8>> = batch
                .iter()
                .map(|(j, _)| {
                    if j.executable {
                        let mut w = j.raw.clone();
                        let t = std::time::Instant::now();
                        crate::bcj::x86(&mut w, true);
                        bcj_ns.push(t.elapsed().as_nanos().min(u32::MAX as u128) as u32);
                        w
                    } else {
                        bcj_ns.push(0);
                        j.raw.clone()
                    }
                })
                .collect();
            let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
            gpu.as_mut()
                .expect("the context is opened before the first launch")
                .compress_batch(&refs, level)
        };
        stats.record_gpu(batch.len(), batch_start.elapsed());

        match results {
            Ok(coded) => {
                for (((job, done_tx), compressed), ns_bcj) in
                    batch.into_iter().zip(coded).zip(bcj_ns.into_iter())
                {
                    let f = job.executable;
                    // The codec itself ran on the GPU; the stage timings for it
                    // are left at zero here and counted by `Stats` instead.
                    let _ =
                        done_tx.send(make_done(job, compressed, verify, true, f, ns_bcj, 0, None));
                }
            }
            Err(e) => {
                eprintln!("densr: GPU compression failed ({e}), falling back to CPU");
                for (job, done_tx) in batch {
                    let _ = done_tx.send(compress_block(job, level, verify, None));
                }
            }
        }
    }
}

/// Hash, checksum and code one block on the CPU.
fn compress_block(
    job: BlockJob,
    level: u8,
    verify: bool,
    seed: Option<&crate::lz::DictSeed>,
) -> Result<BlockDone, String> {
    let dict = seed.map(crate::lz::DictSeed::dict);
    // The branch filter runs before the codec, on executables only. It rewrites
    // relative branch displacements into absolute ones so repeated code holds
    // repeated addresses and LZ can match it. A filtered copy is compressed;
    // `job.raw` itself is untouched because every hash and the stored length
    // must describe the original bytes.
    let filtered = job.executable;
    let mut ns_bcj = 0u32;
    let ns_lz;
    let compressed = if filtered {
        let mut work = job.raw.clone();
        let t = std::time::Instant::now();
        crate::bcj::x86(&mut work, true);
        ns_bcj = t.elapsed().as_nanos().min(u32::MAX as u128) as u32;
        let t = std::time::Instant::now();
        let c = match seed {
            Some(s) => crate::lz::compress_with_dict_seed(&work, s, level),
            None => crate::lz::compress(&work, level),
        };
        ns_lz = t.elapsed().as_nanos().min(u32::MAX as u128) as u32;
        c
    } else {
        let t = std::time::Instant::now();
        let c = match seed {
            Some(s) => crate::lz::compress_with_dict_seed(&job.raw, s, level),
            None => crate::lz::compress(&job.raw, level),
        };
        ns_lz = t.elapsed().as_nanos().min(u32::MAX as u128) as u32;
        c
    };
    make_done(
        job, compressed, verify, false, filtered, ns_bcj, ns_lz, dict,
    )
}

/// Assemble a `BlockDone` from a raw block and an optional pre-computed
/// compression result (from either the CPU or the GPU).
///
/// `filtered` records that the bytes given to the codec had the branch filter
/// applied, so the block decodes as Huffman, then `dsr1`, then the inverse
/// filter. It changes the tag only, never the payload layout.
fn make_done(
    job: BlockJob,
    compressed: Option<Vec<u8>>,
    verify: bool,
    via_gpu: bool,
    filtered: bool,
    ns_bcj: u32,
    ns_lz: u32,
    dict: Option<&[u8]>,
) -> Result<BlockDone, String> {
    let raw_len = job.raw.len() as u32;
    let hash = hash64(&job.raw);
    let raw_crc = crc32(&job.raw);
    // Four-way choice, smallest wins. `filtered` says the bytes handed to the
    // codec already went through the branch filter, which only changes the tag:
    // the payload shape is identical either way. The comparison is against the
    // ORIGINAL raw length, because that is what the record promises to expand to.
    let mut ns_huff = 0u32;
    let mut coded_len = 0u32;
    let (kind, payload) = match compressed {
        Some(c) if c.len() < job.raw.len() => {
            coded_len = c.len() as u32;
            // Huffman only pays for its table on blocks of a decent size.
            const HUFF_MIN: usize = 1024;
            let coded = if filtered {
                BlockKind::CodedX86
            } else {
                BlockKind::Coded
            };
            if c.len() >= HUFF_MIN {
                let t = std::time::Instant::now();
                let attempt = crate::huffman::compress(&c);
                ns_huff = t.elapsed().as_nanos().min(u32::MAX as u128) as u32;
                match attempt {
                    Some(h) if h.len() < c.len() => {
                        let k = if filtered {
                            BlockKind::CodedX86Huff
                        } else {
                            BlockKind::CodedHuff
                        };
                        // Two alphabets beat one on almost every real block, but
                        // the second table is a fixed cost. Take whichever
                        // encoding is actually smaller, rather than assuming.
                        match crate::split::compress_split(&c) {
                            s if s.len() < h.len() => {
                                let sk = if filtered {
                                    BlockKind::CodedX86HuffSplit
                                } else {
                                    BlockKind::CodedHuffSplit
                                };
                                (sk, s)
                            }
                            _ => (k, h),
                        }
                    }
                    _ => (coded, c),
                }
            } else {
                (coded, c)
            }
        }
        _ => (BlockKind::Stored, job.raw),
    };
    let done = BlockDone {
        raw_len,
        hash,
        raw_crc,
        kind,
        payload,
        via_gpu,
        ns_bcj,
        ns_lz,
        ns_huff,
        coded_len,
        // A stored block references no dictionary: it is the original bytes.
        uses_dict: dict.is_some() && kind != BlockKind::Stored,
    };
    if verify {
        // `expand_with_dict` checks the payload CRC, the length, the raw CRC
        // and the content hash, so an Ok result means the block came
        // back intact -- including when it reaches into the dictionary.
        let mut rec = BlockRec::with_hashes(raw_len, hash, raw_crc, kind, &done.payload);
        if done.uses_dict {
            rec = rec.with_flags(BLOCK_FLAG_USES_DICT);
        }
        rec.expand_with_dict(&done.payload, dict.unwrap_or(&[]))?;
    }
    Ok(done)
}

/// One archive entry whose blocks may still be compressing.
///
/// Files are submitted ahead of the commit point so workers stay busy across
/// file boundaries. Many-small-file trees otherwise serialize on the fact that
/// a file has only one block: the writer would wait for it before reading the
/// next file, leaving every other worker idle.
enum PendingEntry {
    Ready(EntryRec),
    Blocks(EntryRec, VecDeque<Receiver<Result<BlockDone, String>>>),
}

/// Commit the oldest pending entry.
///
/// Entries are consumed strictly in input order, so dedup ids, block ids and the
/// payload stream are identical to a one-file-at-a-time writer even though
/// compression ran ahead.
fn finish_entry(
    queue: &mut VecDeque<PendingEntry>,
    entries: &mut Vec<EntryRec>,
    report: &mut WriteReport,
    seen: &mut HashMap<(u64, u32, u32), u32>,
    f: &mut File,
    pos: &mut u64,
) -> io::Result<()> {
    match queue.pop_front() {
        Some(PendingEntry::Ready(entry)) => entries.push(entry),
        Some(PendingEntry::Blocks(mut entry, mut pending)) => {
            while let Some(done) = pending.pop_front() {
                let d = done
                    .recv()
                    .map_err(|e| io::Error::other(format!("a worker died: {e}")))?
                    .map_err(io::Error::other)?;

                if d.via_gpu {
                    report.gpu_blocks += 1;
                }

                // Counted before the dedup check: a block that turns out to be a duplicate
                // still cost a worker its time, and leaving that out would make
                // the stage breakdown depend on how repetitive the input was.
                report.ns_bcj += d.ns_bcj as u64;
                report.ns_lz += d.ns_lz as u64;
                report.ns_huff += d.ns_huff as u64;

                if let Some(&idx) = seen.get(&(d.hash, d.raw_len, d.raw_crc)) {
                    entry.blocks.push(idx);
                    report.dedup_blocks += 1;
                    report.bytes_saved += d.raw_len as u64;
                    continue;
                }

                if d.kind != BlockKind::Stored {
                    report.coded_blocks += 1;
                } else {
                    report.bypass_blocks += 1;
                }
                report.kinds.record(d.kind, d.raw_len, d.payload.len());
                report.coded_bytes += d.coded_len as u64;
                report.huffman_payload_bytes += d.payload.len() as u64;

                let mut rec =
                    BlockRec::with_hashes(d.raw_len, d.hash, d.raw_crc, d.kind, &d.payload);
                if d.uses_dict {
                    rec = rec.with_flags(BLOCK_FLAG_USES_DICT);
                }

                f.write_all(&rec.encode(d.payload.len() as u32))?;
                f.write_all(&d.payload)?;
                *pos += (BLOCK_RECORD_SIZE + d.payload.len()) as u64;

                let idx = report.blocks as u32;
                report.blocks += 1;
                entry.blocks.push(idx);
                seen.insert((d.hash, d.raw_len, d.raw_crc), idx);
            }
            entries.push(entry);
        }
        None => {}
    }
    Ok(())
}

/// Write `roots` into `out_path`.
///
/// Archive paths are relative and never contain a `.` or `..` component. A
/// traversal is refused at write time rather than normalised away: an archive
/// that has to be sanitised on read is an archive someone will eventually
/// extract into the wrong directory.
pub fn write_archive(
    out_path: &Path,
    roots: &[PathBuf],
    opts: &WriteOptions,
) -> io::Result<WriteReport> {
    let mut items: BTreeMap<String, PathBuf> = BTreeMap::new();
    for root in roots {
        let name = root_name(root);
        collect(root, &name, &out_path, &mut items)?;
    }

    let block_size = opts
        .block_size
        .clamp(chunking::MIN_BLOCK, chunking::MAX_BLOCK);

    // Build the dictionary before the first block is written, because every
    // block is coded against it. This reads the head of each file once more than
    // the main pass does; the cost is bounded by the sampling rule.
    let dict = if opts.dict_size == 0 {
        None
    } else {
        build_dictionary(&items, opts.dict_size)?.map(Arc::new)
    };
    // One match-table index over the dictionary, reused by every block. Building
    // it per block was O(dictionary length) and dominated many-small-file trees.
    let seed = dict
        .as_ref()
        .map(|d| Arc::new(crate::lz::DictSeed::new(Arc::clone(d))));
    // The OpenCL kernel has no dictionary support, and routing only some blocks
    // through a path that ignores the dictionary would make the archives differ
    // from a CPU-only run for the same input. Keep dictionary archives on the CPU.
    let use_gpu = opts.gpu && dict.is_none();
    if opts.gpu && dict.is_some() {
        eprintln!("densr: shared dictionary in use, compressing on CPU only");
    }

    let mut f = File::create(out_path)?;

    let flags = FLAG_METADATA
        | FLAG_CHECKSUMS
        | FLAG_DEDUP
        | if opts.chunking == Chunking::Cdc {
            FLAG_RESYNC | FLAG_CHUNKING_CDC
        } else {
            0
        };
    let mut header = Header {
        version: FORMAT_VERSION,
        flags,
        level: opts.level,
        block_size: block_size as u32,
        created: crate::util::now_unix(),
        payload_len: 0,
        dir_offset: 0,
        dir_len: 0,
        dir_crc: 0,
    };
    f.write_all(&header.encode())?;
    let mut pos = HEADER_SIZE as u64;

    let mut report = WriteReport::default();
    let mut seen: HashMap<(u64, u32, u32), u32> = HashMap::new();
    let mut entries: Vec<EntryRec> = Vec::with_capacity(items.len());

    // The dictionary is the archive's first block. Nothing references it, but a
    // reader has to find it before it can decode any block flagged
    // `BLOCK_FLAG_USES_DICT`; writing it first keeps that order natural.
    if let Some(d) = &dict {
        let payload = encode_dictionary(d, opts.level);
        let rec = BlockRec::with_hashes(
            d.len() as u32,
            hash64(d),
            crc32(d),
            BlockKind::Dictionary,
            &payload,
        );
        f.write_all(&rec.encode(payload.len() as u32))?;
        f.write_all(&payload)?;
        pos += (BLOCK_RECORD_SIZE + payload.len()) as u64;
        report.blocks = 1;
        report
            .kinds
            .record(BlockKind::Dictionary, d.len() as u32, payload.len());
        report.dict_bytes = payload.len() as u64;
    }

    let pool = BlockPool::new(
        opts.threads.max(1),
        opts.level,
        opts.verify,
        use_gpu,
        opts.gpu_share,
        opts.gpu_batch.max(1),
        seed.clone(),
    );

    // Let compression run ahead across file boundaries: many-small-file trees
    // otherwise leave every worker but one idle while the writer waits for a
    // single-block file. The depth bounds how many entries are in flight; the
    // pool's bounded job queue bounds the bytes held by unfinished work.
    let mut pipeline: VecDeque<PendingEntry> = VecDeque::new();
    let pipeline_depth = (opts.threads.max(1) * 8).clamp(16, 128);

    for (arc, real) in &items {
        let meta = fs::symlink_metadata(real)?;
        let kind = if meta.is_symlink() {
            EntryKind::Symlink
        } else if meta.is_dir() {
            EntryKind::Directory
        } else if meta.is_file() {
            EntryKind::File
        } else {
            // Sockets, fifos and devices have no portable serialisation here,
            // and archiving one is almost always a mistake.
            continue;
        };

        let mut entry = EntryRec {
            path: arc.clone(),
            kind,
            class: FileClass::Binary,
            mode: 0,
            mtime: 0,
            size: 0,
            content_hash: 0,
            blocks: Vec::new(),
            meta: Vec::new(),
        };

        if opts.metadata && kind != EntryKind::Symlink {
            entry.mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                entry.mode = meta.permissions().mode() & 0o7777;
            }
        }

        match kind {
            EntryKind::Directory => {
                report.dirs += 1;
                pipeline.push_back(PendingEntry::Ready(entry));
            }
            EntryKind::Symlink => {
                // The link is stored as itself. Following it would either loop or
                // quietly copy something outside the tree the user did not name.
                let target = fs::read_link(real)?;
                // The reader refuses to restore a link whose target is absolute
                // or contains `..`, because creating it before other entries
                // could redirect their writes outside the extraction root. Do
                // the same check here so `add` cannot produce an archive that
                // `test` or `extract` then rejects.
                let target_text = target.to_string_lossy();
                crate::reader::safe_relative(&target_text).map_err(|why| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("{:?}: symlink target: {why}", entry.path),
                    )
                })?;
                entry.meta = target_text.into_owned().into_bytes();
                entry.size = entry.meta.len() as u64;
                report.symlinks += 1;
                pipeline.push_back(PendingEntry::Ready(entry));
            }
            EntryKind::File => {
                let data = read_file(real)?;
                entry.size = data.len() as u64;
                entry.class = classify_file(real, &data);
                entry.content_hash = hash64(&data);
                report.files += 1;
                report.input_bytes += data.len() as u64;

                // Compression is the expensive part and blocks are
                // independent, so the blocks go to the pool while this
                // thread keeps the decisions that must stay in order:
                // deduplication, block ids and the byte stream.
                // Branch filtering is gated on the header, never the file
                // name: renaming an .exe to .dat must not change the output.
                let executable = wants_bcj(&data[..data.len().min(HEADER_PROBE)]);
                let mut pending: VecDeque<Receiver<Result<BlockDone, String>>> = VecDeque::new();
                for (start, end) in
                    chunking::split_with(&data, block_size, opts.cdc_target, opts.chunking)
                {
                    pending.push_back(pool.submit(data[start..end].to_vec(), executable)?);
                }
                pipeline.push_back(PendingEntry::Blocks(entry, pending));
            }
            _ => {}
        }

        while pipeline.len() >= pipeline_depth {
            finish_entry(
                &mut pipeline,
                &mut entries,
                &mut report,
                &mut seen,
                &mut f,
                &mut pos,
            )?;
        }
    }

    while !pipeline.is_empty() {
        finish_entry(
            &mut pipeline,
            &mut entries,
            &mut report,
            &mut seen,
            &mut f,
            &mut pos,
        )?;
    }

    let directory = Directory { entries };
    let (dir_blob, dir_crc) = directory.encode();
    header.payload_len = pos - HEADER_SIZE as u64;
    header.dir_offset = pos;
    header.dir_len = dir_blob.len() as u32;
    header.dir_crc = dir_crc;

    f.write_all(&dir_blob)?;
    f.write_all(
        &Footer {
            dir_offset: header.dir_offset,
            dir_len: header.dir_len,
            dir_crc,
            archive_len: pos + dir_blob.len() as u64 + FOOTER_SIZE as u64,
        }
        .encode(),
    )?;

    // The header went out before those numbers were known.
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&header.encode())?;
    f.flush()?;

    report.payload_bytes = header.payload_len;
    report.archive_bytes = header.dir_offset + dir_blob.len() as u64 + FOOTER_SIZE as u64;
    report.gpu_share = opts.gpu.then(|| pool.effective_share());
    Ok(report)
}

/// Classify from the data already in hand, falling back to the extension alone.
fn classify_file(path: &Path, data: &[u8]) -> FileClass {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    classify_bytes(&data[..data.len().min(HEADER_PROBE)], &ext)
}

/// Build the per-archive shared dictionary.
///
/// Deterministic: the same tree at the same options produces the same bytes.
/// Only the head of each file is read, and the per-file share is bounded, so the
/// extra pass is cheap. A sample from an executable is branch-filtered the same
/// way its first block will be, so the bytes live in the same domain as the data
/// they are meant to match.
fn build_dictionary(items: &BTreeMap<String, PathBuf>, cap: usize) -> io::Result<Option<Vec<u8>>> {
    if cap < MIN_DICT {
        return Ok(None);
    }

    // Count files and total size without reading content: a small tree gets no
    // dictionary at all, because its own payload would cost more than it saves.
    let mut files: Vec<&PathBuf> = Vec::new();
    let mut total: u64 = 0;
    for p in items.values() {
        // `metadata` follows symlinks. A symlink is archived as a link, not as
        // its target, so sampling through it would put bytes into the dictionary
        // that the archive never stores -- and could read a file the user did
        // not name.
        let Ok(meta) = fs::symlink_metadata(p) else {
            continue;
        };
        if meta.file_type().is_file() {
            files.push(p);
            total += meta.len();
        }
    }
    if files.is_empty() || total < MIN_INPUT_FOR_DICT {
        return Ok(None);
    }

    // Spread the budget across every file, so a tree of many small files is
    // represented instead of only the first few.
    let per = (cap / files.len()).clamp(DICT_SAMPLE_MIN, DICT_SAMPLE_MAX);
    let mut dict: Vec<u8> = Vec::with_capacity(cap.min(total as usize));
    let mut seen: HashSet<u64> = HashSet::new();

    for p in files {
        if dict.len() >= cap {
            break;
        }
        let want = per.min(cap - dict.len());
        let mut sample = read_head(p, want)?;
        if sample.len() < 16 {
            continue;
        }
        let executable = wants_bcj(&sample[..sample.len().min(HEADER_PROBE)]);
        if executable {
            crate::bcj::x86(&mut sample, true);
        }
        // Identical heads (two copies of a file, say) add nothing after the
        // first; skipping them lets the budget reach more distinct content.
        if !seen.insert(hash64(&sample)) {
            continue;
        }
        let take = sample.len().min(cap - dict.len());
        dict.extend_from_slice(&sample[..take]);
    }

    if dict.len() < MIN_DICT {
        return Ok(None);
    }
    Ok(Some(dict))
}

/// Read at most `max` bytes from the start of a file.
fn read_head(path: &Path, max: usize) -> io::Result<Vec<u8>> {
    let f = File::open(path)?;
    let mut v = Vec::with_capacity(max.min(64 * 1024));
    f.take(max as u64).read_to_end(&mut v)?;
    Ok(v)
}

fn root_name(root: &Path) -> String {
    root.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "root".to_string())
}

fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let mut v = Vec::with_capacity(f.metadata().map(|m| m.len() as usize).unwrap_or(0));
    f.read_to_end(&mut v)?;
    Ok(v)
}

/// Walk `root`, giving every entry an archive path under `arc`.
fn collect(
    root: &Path,
    arc: &str,
    skip: &Path,
    out: &mut BTreeMap<String, PathBuf>,
) -> io::Result<()> {
    let arc = check_path(arc)?;
    // The archive is usually written inside the directory it is archiving. If it
    // were picked up, it would contain a copy of itself: the walk would never
    // finish, and the result would depend on how far it had got.
    if same_file(root, skip) {
        return Ok(());
    }
    // A `BTreeMap` keyed by path gives the sorted order and the duplicate check
    // for free: inserting the same path twice would otherwise write it twice.
    if out.insert(arc.clone(), root.to_path_buf()).is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{arc:?} appears twice in the input"),
        ));
    }

    let meta = fs::symlink_metadata(root)?;
    if meta.is_symlink() || !meta.is_dir() {
        return Ok(());
    }

    let mut kids: Vec<PathBuf> = fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    // Sorted, so archiving the same tree twice gives the same bytes.
    kids.sort();
    for kid in kids {
        let name = kid.file_name().map(|s| s.to_string_lossy().into_owned());
        let Some(name) = name.filter(|n| !n.is_empty() && n != "." && n != "..") else {
            continue;
        };
        collect(&kid, &format!("{arc}/{name}"), skip, out)?;
    }
    Ok(())
}

/// Whether two paths are the same file, falling back to a path comparison.
fn same_file(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// Reject any path that must not appear inside an archive.
fn check_path(p: &str) -> io::Result<String> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
    if p.is_empty() {
        return Err(bad("empty path".into()));
    }
    if p.starts_with('/') || p.contains('\\') || p.contains(':') {
        return Err(bad(format!(
            "refusing {p:?}: archive paths are relative and use forward slashes"
        )));
    }
    for part in p.split('/') {
        if part.is_empty() {
            return Err(bad(format!("{p:?} has an empty component")));
        }
        if part == "." || part == ".." {
            return Err(bad(format!(
                "refusing {p:?}: a {part:?} component can escape the extraction root"
            )));
        }
    }
    Ok(p.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::Archive;
    use std::collections::HashSet;

    /// Archive paths are relative to the root's own name.
    fn prefix(dir: &Path) -> String {
        dir.file_name().unwrap().to_string_lossy().into_owned()
    }
    fn key(dir: &Path, rel: &str) -> String {
        format!("{}/{rel}", prefix(dir))
    }
    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("densr-writer-{name}"));
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

    fn tree() -> Vec<u8> {
        let mut v = b"the quick brown fox jumps over the lazy dog\n".repeat(400);
        v.extend_from_slice(&vec![7u8; 5000]);
        v
    }

    #[test]
    fn a_written_archive_reads_back_byte_for_byte() {
        let dir = tmp("roundtrip");
        let body = tree();
        write(&dir, "a.txt", &body);
        write(&dir, "sub/b.bin", &vec![0xAB; 3000]);
        fs::create_dir_all(dir.join("empty")).unwrap();

        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert_eq!(rep.files, 2);
        assert_eq!(rep.dirs, 3, "the root, sub/ and empty/");
        assert!(rep.archive_bytes > 0);

        let mut ar = Archive::open(&out).unwrap();
        let got = ar.entries();
        assert_eq!(got.len(), 5, "two files, the root, sub/ and empty/");
        assert!(got.iter().all(|e| e.path.starts_with(&prefix(&dir))));
        let a = ar.read(&key(&dir, "a.txt")).unwrap();
        assert_eq!(a.data, body, "content differed");
        assert_eq!(a.entry.size, body.len() as u64);
    }

    #[test]
    fn identical_blocks_are_stored_once() {
        // Four copies of the same 300 KB file must cost one copy.
        let dir = tmp("dedup");
        let body = tree();
        for n in 0..4 {
            write(&dir, &format!("copy{n}.bin"), &body);
        }
        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert_eq!(rep.files, 4);
        assert!(
            rep.dedup_blocks >= 3,
            "expected the repeats to be reused, got {}",
            rep.dedup_blocks
        );
        assert!(rep.bytes_saved > 0);

        let mut ar = Archive::open(&out).unwrap();
        for n in 0..4 {
            assert_eq!(
                ar.read(&key(&dir, &format!("copy{n}.bin"))).unwrap().data,
                body
            );
        }
    }

    #[test]
    fn compressible_data_actually_shrinks() {
        let dir = tmp("ratio");
        write(&dir, "text.txt", &tree());
        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert!(
            rep.ratio() > 2.0,
            "repeated text only reached {:.2}x",
            rep.ratio()
        );
    }

    #[test]
    fn incompressible_data_is_stored_verbatim() {
        // Random bytes cannot be compressed, so the codec must decline rather
        // than make the file bigger.
        let dir = tmp("random");
        // A simple LCG, so the test needs no dependency.
        let mut s = 0x12345678u32;
        let mut noise = Vec::with_capacity(200_000);
        while noise.len() < 200_000 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            noise.push((s >> 24) as u8);
        }
        write(&dir, "noise.bin", &noise);
        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert!(
            rep.coded_blocks == 0,
            "noise was handed to the codec anyway"
        );
        assert!(rep.ratio() > 0.95, "noise grew to {:.3}x", rep.ratio());
        let mut ar = Archive::open(&out).unwrap();
        assert_eq!(ar.read(&key(&dir, "noise.bin")).unwrap().data, noise);
    }

    #[test]
    #[cfg(unix)]
    fn symlinks_are_stored_as_links_and_not_followed() {
        let dir = tmp("links");
        write(&dir, "real.txt", b"content\n");
        #[cfg(unix)]
        std::os::unix::fs::symlink("real.txt", dir.join("link.txt")).unwrap();
        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert_eq!(rep.symlinks, 1);
        let mut ar = Archive::open(&out).unwrap();
        let all = ar.entries();
        let link = all.iter().find(|e| e.path.ends_with("link.txt")).unwrap();
        assert_eq!(link.kind, EntryKind::Symlink);
        assert_eq!(link.meta, b"real.txt");
        // The link must not have been turned into a second copy of the file.
        assert_eq!(ar.read("link.txt").is_err(), true);
    }

    #[test]
    #[cfg(unix)]
    fn unsafe_symlink_targets_are_refused_at_write_time() {
        for (i, target) in ["/etc/passwd", "../escape"].into_iter().enumerate() {
            let dir = tmp(&format!("badlink{i}"));
            write(&dir, "real.txt", b"x");
            std::os::unix::fs::symlink(target, dir.join("link")).unwrap();
            let out = dir.join("out.dnsr");
            let e = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidInput, "{target}");
        }
    }

    #[test]
    fn traversal_and_absolute_paths_are_refused() {
        let _dir = tmp("badpaths");
        for p in ["../escape", "/abs/path", "a/../b", "a/./b", ""] {
            let e = check_path(p).unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidInput, "{p:?} was allowed");
        }
        for p in ["ok", "a/b/c.txt", "dir.with.dots/file"] {
            assert_eq!(check_path(p).unwrap(), p);
        }
    }

    #[test]
    fn names_are_checked_before_anything_is_written() {
        let dir = tmp("badroot");
        write(&dir, "f.txt", b"x");
        let mut link = dir.clone();
        link.push("link");
        #[cfg(unix)]
        std::os::unix::fs::symlink("../elsewhere", &link).unwrap();
        let out = dir.join("out.dnsr");
        let r = write_archive(&out, &[link], &WriteOptions::default());
        assert!(r.is_err(), "a symlink named .. was archived");
    }

    #[test]
    fn every_block_index_is_distinct_and_in_range() {
        let dir = tmp("indices");
        write(&dir, "a.txt", &tree());
        write(&dir, "b.txt", &tree());
        let out = dir.join("out.dnsr");
        write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        let ar = Archive::open(&out).unwrap();
        let total = ar.block_count();
        let mut seen = HashSet::new();
        for e in ar.entries() {
            for &b in &e.blocks {
                assert!((b as usize) < total, "block {b} is out of range");
                seen.insert(b);
            }
        }
        // Two identical files: every index is used at least once.
        assert_eq!(
            seen.len(),
            total,
            "some blocks are stored but never referenced"
        );
    }

    #[test]
    fn an_empty_tree_still_produces_a_valid_archive() {
        let dir = tmp("emptytree");
        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert_eq!(rep.files, 0);
        let mut ar = Archive::open(&out).unwrap();
        assert!(
            ar.entries().iter().all(|e| !e.is_file()),
            "an empty tree should hold directories only"
        );
        assert!(ar.test().is_ok());
    }

    #[test]
    fn a_single_file_root_is_named_after_itself() {
        let dir = tmp("singlefile");
        write(&dir, "only.txt", b"hello\n");
        let out = dir.join("out.dnsr");
        write_archive(&out, &[dir.join("only.txt")], &WriteOptions::default()).unwrap();
        let ar = Archive::open(&out).unwrap();
        assert_eq!(ar.entries().len(), 1);
        assert_eq!(
            ar.entries()[0].path,
            "only.txt",
            "a lone file is named after itself"
        );
    }

    #[test]
    fn verification_catches_a_corrupt_block() {
        let dir = tmp("corrupt");
        write(&dir, "a.txt", &tree());
        let out = dir.join("out.dnsr");
        write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();

        // Flip a byte inside the payload area, past the header and the record.
        let mut bytes = fs::read(&out).unwrap();
        let victim = bytes.len() / 2;
        bytes[victim] ^= 0xFF;
        let bad = dir.join("bad.dnsr");
        fs::write(&bad, &bytes).unwrap();

        let mut ar = Archive::open(&bad).unwrap();
        let e = ar.test().unwrap_err();
        assert!(
            e.contains("crc") || e.contains("CRC"),
            "unhelpful error: {e}"
        );
    }

    #[test]
    fn write_options_clamp_the_block_size() {
        let dir = tmp("clamp");
        write(&dir, "a.txt", &tree());
        let out = dir.join("out.dnsr");
        let opts = WriteOptions {
            block_size: 10,
            ..Default::default()
        };
        write_archive(&out, &[dir.clone()], &opts).unwrap();
        let mut ar = Archive::open(&out).unwrap();
        assert!(ar.header().block_size as usize >= chunking::MIN_BLOCK);
        assert_eq!(ar.read(&key(&dir, "a.txt")).unwrap().data, tree());
    }

    #[test]
    fn pe_magic_is_compressed_not_stored() {
        use crate::magic::classify_bytes;
        // MZ header must still classify as Binary but now be compressible,
        // and the archive must actually code its blocks.
        let mut mz = b"MZ".to_vec();
        mz.extend(vec![0u8; 4094]);
        assert_eq!(classify_bytes(&mz, "exe"), FileClass::Binary);
        assert!(FileClass::Binary.is_compressible());

        let dir = tmp("pe");
        // PE-like payload with enough redundancy to be compressible.
        let mut body = b"MZ".to_vec();
        body.extend_from_slice(&b"repeated-text-block-0123456789\n".repeat(4000));
        body.extend(vec![0u8; 1024]);
        write(&dir, "prog.exe", &body);
        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert!(
            rep.coded_blocks > 0,
            "MZ file was stored without trying the codec: {rep:?}"
        );
        let mut ar = Archive::open(&out).unwrap();
        assert_eq!(ar.read(&key(&dir, "prog.exe")).unwrap().data, body);
    }

    #[test]
    fn a_tree_large_enough_gets_a_shared_dictionary_and_roundtrips() {
        let dir = tmp("dict");
        // Above MIN_INPUT_FOR_DICT, with shared repeated content so matches into
        // the dictionary actually pay.
        let mut body = Vec::new();
        for i in 0..64 {
            body.extend_from_slice(
                format!(
                    "section {i}: the quick brown fox jumps over the lazy dog; mov rax, rbx; push rbp\n"
                )
                .as_bytes(),
            );
        }
        body.resize(600 * 1024, b'x');
        write(&dir, "a.bin", &body);
        let mut body2 = body.clone();
        body2.extend_from_slice(b"tail");
        write(&dir, "b.bin", &body2);

        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert_eq!(
            rep.kinds.dictionary.blocks, 1,
            "a large tree should carry one dictionary block: {rep:?}"
        );
        assert!(rep.dict_bytes > 0, "the dictionary payload was empty");

        let dest = dir.join("restored");
        let mut ar = Archive::open(&out).unwrap();
        assert!(ar.test().unwrap().files >= 2);
        ar.extract_to(&dest).unwrap();
        let top = dest.join(prefix(&dir));
        assert_eq!(fs::read(top.join("a.bin")).unwrap(), body);
        assert_eq!(fs::read(top.join("b.bin")).unwrap(), body2);
    }

    #[test]
    fn a_small_tree_gets_no_dictionary() {
        let dir = tmp("nodict");
        write(&dir, "a.txt", &tree());
        let out = dir.join("out.dnsr");
        let rep = write_archive(&out, &[dir.clone()], &WriteOptions::default()).unwrap();
        assert_eq!(rep.kinds.dictionary.blocks, 0, "{rep:?}");
        assert_eq!(rep.dict_bytes, 0);
    }
}
