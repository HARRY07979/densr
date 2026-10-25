# Densr

Densr is a self-contained archiver written in Rust. It has its own container
format (`.dnsr`), its own LZ77 codec, its own hashing and checksums, and its own
argument parser. There are **no third-party dependencies**: every byte on disk
and every line of the CLI is defined in this repository.

**Status:** `v1.0.0-beta.1`, archive format **v1**. This is a beta: the format
may still change, and archives written by an older beta are not guaranteed to
open in a newer build.

## Install

Prebuilt packages are attached to the [v1.0-beta release](https://github.com/HARRY07979/densr/releases/tag/v1.0-beta):

| Platform | Asset | Notes |
| :--- | :--- | :--- |
| Windows (x86-64) | `setup.exe` | NSIS installer |
| Debian / Ubuntu (amd64) | `setup.deb` | needs `libc6 >= 2.34`, installs `/usr/bin/densr` |
| Fedora / RHEL | `setup.rpm` | RPM package |

The Windows build links the C runtime statically, so it needs no MSVC or MinGW
runtime on the target machine; OpenCL is loaded at runtime.

## Build from source

```bash
git clone https://github.com/HARRY07979/densr.git
cd densr
cargo build --release
```

See [Building](#building) for platform details.

## Quick start

```bash
# Archive a directory (creates or replaces backup.dnsr)
densr add backup.dnsr ./project

# Inspect it
densr list backup.dnsr
densr info backup.dnsr

# Verify every block against its checksums
densr test backup.dnsr

# Restore into ./restored
densr extract backup.dnsr ./restored
```

`add` takes one or more paths, so several trees can go into one archive:

```bash
densr add photos.dnsr ./photos ./videos/clips
```

## Commands

```
densr <COMMAND> [OPTIONS] [ARGS]
```

| Command | Syntax | What it does |
| :--- | :--- | :--- |
| `add` | `densr add <archive> <path>...` | Write files and directories into an archive |
| `extract` | `densr extract <archive> [dir]` | Write everything back out (default: current directory) |
| `list` | `densr list <archive>` | List entries with size and detected file class |
| `info` | `densr info <archive>` | Show header fields, block kinds and a size summary |
| `test` | `densr test <archive>` | Decode every block and check its checksums |

`-h` / `--help` and `-V` / `--version` work anywhere, e.g. `densr add -h`.

## `add` options

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `-l, --level <0-12>` | `4` | Compression effort. `0` stores blocks uncompressed; higher levels search harder. |
| `-b, --block-size <size>` | `16M` | Block size (and CDC boundary target), clamped to `4k`..`16M`. Accepts `262144`, `256k`, `4M`, `2MiB`. |
| `--cdc` | enabled | Place boundaries by content (gear hash over the last 64 bytes). |
| `--fixed` | | Place boundaries at fixed offsets. |
| `--threads <n>` | CPU count | Number of compression workers. |
| `--gpu` | off | Also compress blocks on the GPU (OpenCL). Silently stays on CPU if unavailable. |
| `--gpu-share <0-100>` | auto | Percentage of blocks offloaded to the GPU. Omitted, Densr measures the GPU against the CPU workers and offloads only what it can absorb. |
| `--gpu-batch <1-32>` | `32` | Blocks per GPU kernel launch. |
| `--dict-size <size>` | `1M` | Shared dictionary budget, sampled from the archive itself and stored (compressed) inside it. `0` disables it. |
| `--no-dict` | | Same as `--dict-size 0`. |
| `--no-metadata` | | Do not record modification times and Unix permissions. |
| `--no-verify` | | Skip decoding each block back after writing it. |
| `-q, --quiet` | | Print nothing on success. |

`--flag=value` is accepted as well as `--flag value`, and everything after `--`
is treated as a path.

## Exit codes

| Code | Meaning |
| :--- | :--- |
| `0` | Success |
| `1` | The operation failed (corrupt archive, unreadable file) |
| `2` | The command line was wrong |

Exit code `2` exists so a script can tell "your archive is broken" from "you
typed the flags wrong" without parsing messages.

## How it works

### Chunking and deduplication

By default boundaries are content-defined: a gear hash over the preceding 64
bytes is cut when its high bits are all zero. Inserting bytes near the front of a
file therefore shifts only the boundaries after the insertion, and the shared
tail still deduplicates. `--fixed` switches to exact block-size offsets, which is
fastest when files are identical copies.

Each block is identified by a 64-bit content hash. A block that has been written
before is stored once; later entries reference it. The hash and a CRC-32 of the
same bytes travel with every block, so deduplication can be checked rather than
trusted.

### Compression pipeline

The codec (`dsr1`) is an LZ77 variant in the LZ4/snappy family. Every block is
coded independently, which is what makes block-level deduplication meaningful.

On top of the raw LZ stream, the writer picks the smallest representation per
block, among:

- literals and matches with LEB128 offsets (window up to 4 GiB),
- canonical Huffman coding of the coded byte stream,
- split-alphabet Huffman, with separate tables for literal bytes and structural
  tokens,
- the x86 branch-conversion filter for executable blocks, turning relative call
  and jump targets into absolute ones so repeated code matches,
- a verbatim "stored" form for incompressible data.

Files that are already compressed (JPEG, PNG, ZIP, video, ...) are detected by
their signatures, not their extensions, and stored without wasting CPU on them.

A 1 MiB shared dictionary, sampled from the archive itself, is stored compressed
inside the archive so it stays self-contained.

### Integrity

- 64-bit content hash of every block's uncompressed bytes (block identity),
- CRC-32 of those uncompressed bytes,
- CRC-32 of the stored payload,
- CRC per directory entry plus one over the whole directory,
- optional decode-and-compare of every block at write time (`--no-verify` turns
  it off; it is on by default).

### GPU acceleration

`--gpu` enables an OpenCL kernel that is a byte-exact port of the CPU codec:
the archive bytes are identical with or without it, only the speed differs.
OpenCL is loaded at runtime (`LoadLibrary`), so a machine without a GPU driver
still runs the same binary. CPU and GPU compress in parallel.

GPU support is currently **Windows-only**: on other platforms the loader is a
stub that always reports "no GPU", and compression stays on the CPU.

### Path safety

Archive entry names are validated before extraction. Absolute paths,
drive-lettered paths, `..` components that escape the destination root, and
non-portable names are refused rather than cleaned up.

## Archive format (v1)

```text
+--------------------------------------------------+ 0
| header, 64 bytes                                 |
+--------------------------------------------------+ 64
| block frames, back to back                       |
|   32-byte record + payload                       |
+--------------------------------------------------+ dir_offset
| directory: one varint record per entry,          |
|   CRC per entry                                  |
+--------------------------------------------------+ dir_offset + dir_len
| footer, 40 bytes                                 |
+--------------------------------------------------+ EOF
```

- Header magic: `D\x01.dnsr\0`; footer magic: `D\x01END\0\0\0`.
- Every fixed-width integer is little-endian; directory fields use varints.
- The footer repeats the directory's position so an archive can be opened from
  either end, and the two copies are checked against each other.
- Format version is `1`, carried in both the header and the magic.

## Performance

Block size is the biggest ratio knob, because the LZ window *is* the block.
Measured on the winlibs `bin` corpus (262.22 MiB, level 4-5, CDC):

| Block size | Ratio |
| :--- | :--- |
| 256 KiB | 2.62x |
| 4 MiB | 2.86x |
| 8 MiB | 2.89x |
| 16 MiB | 2.91x |

Returns flatten past 8 MiB; 16 MiB is kept as the default because large
executables still gain, and the cost is RAM per in-flight block rather than time.

## Building

Requirements: a stable Rust toolchain (edition 2021). There are no C
dependencies and nothing to install on `PATH`.

### Windows

`.cargo/config.toml` pins the `x86_64-pc-windows-gnu` target and links with
Rust's bundled `rust-lld` and self-contained CRT objects. That means no Visual
Studio Build Tools and no MinGW installation are needed:

```bash
rustup target add x86_64-pc-windows-gnu
cargo build --release
```

The binary is written to `target/x86_64-pc-windows-gnu/release/densr.exe`.

### Linux / other platforms

The same config sets a default `target`, so name the target explicitly:

```bash
cargo build --release --target x86_64-unknown-linux-gnu
```

Release builds use `opt-level = 3`, fat LTO, one codegen unit, `panic = "abort"`
and stripped symbols.

## Testing

```bash
cargo test
```

The tests are unit tests inside the modules under `src/` (format round trips,
checksum vectors, chunking boundaries, unsafe-path rejection, CLI argument
parsing). There is currently no separate `tests/` integration suite.

## Project layout

```text
src/
  main.rs       CLI: argument parsing, five commands, exit codes
  lib.rs        module list (the crate is also usable as a library)
  format.rs     container layout: header, block records, directory, footer
  writer.rs     archive writer: chunking, dedup, worker/GPU scheduling
  reader.rs     archive reader: open, list, verify, extract
  lz.rs         dsr1 LZ77 codec
  huffman.rs    canonical Huffman coding of a coded block
  split.rs      split-alphabet Huffman (literals vs tokens)
  bcj.rs        x86/x86-64 branch-conversion filter
  magic.rs      file classification and compression bypass
  chunking.rs   fixed and content-defined chunking, dedup index
  checksum.rs   CRC-32 and the 64-bit block hash
  gpu.rs        optional OpenCL backend (runtime-loaded)
  util.rs       byte I/O, varints, size formatting, time formatting
packaging/
  build-linux.sh   builds setup.deb and setup.rpm from Cargo.toml
  densr.spec       RPM spec used by build-linux.sh
```

## Limitations

- Beta format: no compatibility promise between beta builds.
- No encryption or password support.
- `add` creates a new archive; an existing file at the archive path is replaced.
- GPU acceleration is Windows-only for now.
- Extraction refuses unsafe entry names by design.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

You may use, modify and redistribute Densr, including commercially, provided
that you keep the copyright, patent, trademark and attribution notices, state
changes you made, and respect the trademark restrictions.
