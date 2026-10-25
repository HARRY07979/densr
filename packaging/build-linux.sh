#!/usr/bin/env bash
# Build the Linux release packages: target/package/setup.deb and setup.rpm.
#
# Run from anywhere; paths are resolved relative to this script:
#
#   bash packaging/build-linux.sh
#
# Requirements (Debian/Ubuntu): cargo, dpkg-deb, rpmbuild.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
TARGET="x86_64-unknown-linux-gnu"
OUT="$ROOT/target/package"

# --- versions ---------------------------------------------------------------
VERSION="$(sed -n 's/^version = "\([^"]*\)".*/\1/p' Cargo.toml | head -n 1)"
UPSTREAM="${VERSION%%-*}"          # 1.0.0-beta.1 -> 1.0.0
PRE="${VERSION#"$UPSTREAM"}"       # 1.0.0-beta.1 -> -beta.1
PACKAGE_REV=2                      # bump when repackaging the same upstream version

DEB_PRE=""
if [ -n "$PRE" ]; then
    DEB_PRE="~${PRE#-}"
    DEB_PRE="${DEB_PRE//-/.}"      # ~beta.1
fi
DEB_VERSION="${UPSTREAM}${DEB_PRE}-${PACKAGE_REV}"

RPM_PRE=""
if [ -n "$PRE" ]; then
    RPM_PRE=".${PRE#-}"
    RPM_PRE="${RPM_PRE//-/.}"      # .beta.1
fi
RPM_RELEASE="0.${PACKAGE_REV}${RPM_PRE}"

echo "densr $VERSION -> deb $DEB_VERSION, rpm $UPSTREAM-$RPM_RELEASE"

# --- build ------------------------------------------------------------------
cargo build --release --target "$TARGET" --bin densr
BIN="target/$TARGET/release/densr"

rm -rf "$OUT"
mkdir -p "$OUT"

# dpkg-deb and rpmbuild both reject world-writable directories, and files under
# /mnt/c or /mnt/f are 0777 by default. Stage in a real Linux filesystem and
# copy only the finished packages into the repository.
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/densr-package.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT

# --- .deb -------------------------------------------------------------------
STAGE="$BUILD_ROOT/deb"
install -Dm755 "$BIN" "$STAGE/usr/bin/densr"
install -Dm644 README.md "$STAGE/usr/share/doc/densr/README.md"
install -Dm644 LICENSE "$STAGE/usr/share/doc/densr/LICENSE"

# Debian copyright format: the full license text is indented by one space and
# blank lines are " ." so the parser can find the end of the paragraph.
{
    echo "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/"
    echo "Upstream-Name: densr"
    echo "Source: https://github.com/HARRY07979/densr"
    echo
    echo "Files: *"
    echo "Copyright: 2026 HARRY07979"
    echo "License: Apache-2.0"
    sed -e 's/^$/./' -e 's/^/ /' LICENSE
} > "$STAGE/usr/share/doc/densr/copyright"

INSTALLED_SIZE="$(du -sk "$STAGE" | cut -f1)"
mkdir -p "$STAGE/DEBIAN"
cat > "$STAGE/DEBIAN/control" <<EOF
Package: densr
Version: $DEB_VERSION
Architecture: amd64
Priority: optional
Section: utils
Installed-Size: $INSTALLED_SIZE
Depends: libc6 (>= 2.34)
Homepage: https://github.com/HARRY07979/densr
Maintainer: HARRY07979 <HARRY07979@users.noreply.github.com>
Description: self-contained archiver with its own container format and codec
 Densr stores files in a custom .dnsr container using its own LZ77 codec,
 content-defined chunking, block-level deduplication, per-block checksums and
 optional OpenCL acceleration. It has no third-party dependencies.
EOF

dpkg-deb --root-owner-group --build "$STAGE" "$OUT/setup.deb" >/dev/null

# --- .rpm -------------------------------------------------------------------
SRC="$BUILD_ROOT/rpm-src"
mkdir -p "$SRC"
install -m755 "$BIN" "$SRC/densr"
install -m644 README.md LICENSE "$SRC/"

mkdir -p "$BUILD_ROOT/rpmbuild"
sed -e "s|@VERSION@|$UPSTREAM|g" \
    -e "s|@RELEASE@|$RPM_RELEASE|g" \
    -e "s|@DATE@|$(date '+%a %b %d %Y')|g" \
    packaging/densr.spec > "$BUILD_ROOT/densr.spec"

rpmbuild -bb "$BUILD_ROOT/densr.spec" \
    --define "_topdir $BUILD_ROOT/rpmbuild" \
    --define "_sourcedir $SRC" \
    --define "_specdir $BUILD_ROOT" \
    --define "_rpmdir $BUILD_ROOT/rpm" \
    --define "_srcrpmdir $BUILD_ROOT/rpmbuild/SRPMS" \
    --define "_builddir $BUILD_ROOT/rpmbuild/BUILD" \
    --define "_buildrootdir $BUILD_ROOT/rpmbuild/BUILDROOT" \
    >/dev/null

cp "$BUILD_ROOT"/rpm/x86_64/*.rpm "$OUT/setup.rpm"

echo
ls -l "$OUT/setup.deb" "$OUT/setup.rpm"
