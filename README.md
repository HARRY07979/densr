# Densr

Densr is a self-contained, high-performance archiver built from scratch in Rust. It utilizes a custom container format and custom codecs, explicitly avoiding any external dependencies to ensure full control over its on-disk representation and performance characteristics.

## Features

- **Self-Contained:** Zero dependencies. Every component—from hashing and checksums to compression codecs and container formatting—is implemented natively within the project.
- **Intelligent Compression:** Designed to compress data intelligently.
- **Customizable Chunking:** Supports both fixed-size and content-defined chunking (CDC) for optimal data deduplication.
- **GPU Acceleration:** Optional GPU-accelerated block compression (OpenCL) that can run in parallel with CPU workers.
- **High Performance:** Multi-threaded compression, LTO-optimized release builds, and performance-focused design.
- **Verification:** Built-in tools to verify archive integrity block-by-block.

## CLI Usage

Densr provides a straightforward command-line interface:

```bash
densr <COMMAND> [OPTIONS] [ARGS]
```

### Commands

| Command | Description |
| :--- | :--- |
| `add` | Write files and directories into an archive |
| `extract` | Extract contents from an archive |
| `list` | List contents of an archive |
| `info` | Show archive header and size summary |
| `test` | Verify archive integrity (checksum validation) |

### Key Options (`add` command)

- `-l, --level <0-12>`: Compression effort (default: 4).
- `-b, --block-size <size>`: Block size (e.g., `256k`, `16M`).
- `--fixed` / `--cdc`: Choose between fixed or content-defined chunking.
- `--threads <n>`: Number of compression workers.
- `--gpu`: Enable GPU-accelerated compression.
- `--dict-size <size>`: Shared dictionary budget (default: 1M).

For more info, run:

```bash
densr
```

## Project Structure

Densr is structured to keep all implementation logic separate from CLI concerns:

- `src/`: Core implementation logic.
    - `format.rs`: Container format definitions.
    - `writer.rs` / `reader.rs`: Archive reading and writing logic.
    - `gpu.rs`: GPU acceleration implementation.
    - `lz.rs`, `huffman.rs`, `bcj.rs`: Compression codecs.
    - `chunking.rs`: Chunking strategies.
    - `checksum.rs`: Integrity verification.
- `examples/`: Example implementations and benchmarks.
- `tests/`: Integration tests for the CLI.

## Building

Densr is a standard Rust project. Build it using Cargo:

```bash
cargo build --release
```

The project is configured for maximum performance in release builds (LTO, strip, abort-on-panic).

## License

This project is licensed under the **Apache License 2.0**.

### Summary

The Apache 2.0 License is a permissive license that allows you to freely use, modify, and distribute this software, including for commercial purposes, provided that you:

- **Include Copyright & License:** Retain all original copyright, patent, trademark, and attribution notices alongside a copy of the Apache 2.0 License.
- **State Changes:** Clearly state if you have modified any existing files.
- **Patent & Trademark Protection:** Receive explicit patent rights granted by contributors while strictly prohibiting unauthorized use of project trademarks or logos.