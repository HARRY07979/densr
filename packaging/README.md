# Packaging

Release artifacts are built from a clean checkout, tagged commit.

## Linux (`.deb`, `.rpm`)

Build inside Debian/Ubuntu:

```bash
sudo apt-get install -y cargo dpkg-dev rpm
bash packaging/build-linux.sh
```

Outputs:

- `target/package/setup.deb` — amd64, installs `/usr/bin/densr`,
  `/usr/share/doc/densr/{README.md,copyright,LICENSE}`
- `target/package/setup.rpm` — x86_64, installs `/usr/bin/densr` plus
  `%doc` README and `%license` LICENSE

Versions are derived from `Cargo.toml`:

| Cargo version | Debian version | RPM version-release |
| :--- | :--- | :--- |
| `1.0.0-beta.1` | `1.0.0~beta.1-2` | `1.0.0-0.2.beta.1` |

`PACKAGE_REV` in the script is bumped when the same upstream version is
repackaged. The Debian `~` guarantees the beta sorts before a final `1.0.0`,
and both formats get the Apache-2.0 license from `LICENSE`.

## Windows (`setup.exe`)

The installer is an NSIS build of the Cargo-produced binary:

```powershell
cargo build --release
# target\x86_64-pc-windows-gnu\release\densr.exe
```

The original NSIS script is not in the repository yet, so a rebuild of
`setup.exe` is currently done by hand.
