# RPM spec for Densr. Built by packaging/build-linux.sh, which substitutes
# @VERSION@, @RELEASE@ and @DATE@. The binary is built by Cargo first, so this
# spec only installs it; there is nothing to compile here.

Name:           densr
Version:        @VERSION@
Release:        @RELEASE@
Summary:        Self-contained archiver with its own container format and codec
License:        Apache-2.0
URL:            https://github.com/HARRY07979/densr
Source0:        densr
Source1:        README.md
Source2:        LICENSE
BuildArch:      x86_64

%global debug_package %{nil}
# Keep docs under /usr/share/doc/densr and /usr/share/licenses/densr, matching
# the Debian layout instead of the default versioned directory.
%global _docdir_fmt %{name}

%description
Densr stores files in a custom .dnsr container using its own LZ77 codec,
content-defined chunking, block-level deduplication, per-block checksums and
optional OpenCL acceleration. It has no third-party dependencies.

%prep
cp -p %{SOURCE1} README.md
cp -p %{SOURCE2} LICENSE

%build

%install
install -Dm755 %{SOURCE0} %{buildroot}%{_bindir}/densr

%files
%{_bindir}/densr
%doc README.md
%license LICENSE

%changelog
* @DATE@ HARRY07979 <HARRY07979@users.noreply.github.com> - @VERSION@-@RELEASE@
- Rebuild with Apache-2.0 license metadata and the current README
