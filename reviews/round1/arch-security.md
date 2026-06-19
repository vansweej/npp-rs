# Architectural Security Review — npp-rs

**Date:** 2026-06-19  
**Reviewer:** Security-Conscious System Architect  
**Scope:** Full codebase — `npp-sys` (FFI bindings layer) and `npp` (safe Rust wrapper library)

---

## Executive Summary

`npp-rs` is a Rust library that wraps NVIDIA's NPP (NVIDIA Performance Primitives) GPU image-processing library. The attack surface is dominated by a deep `unsafe` FFI boundary that feeds raw GPU device pointers directly into unverified C functions. There is no authentication or authorization layer (the library is a local computation library, not a service), so classic auth/authz risks do not apply. The meaningful security risks are:

1. **Memory unsafety through incorrect pointer arithmetic on GPU device memory** — incorrect bounds checks allow out-of-bounds GPU memory access that is completely invisible to Rust's borrow checker.
2. **`debug_assert!` preconditions are stripped in release builds** — all layout invariants protecting FFI calls are silently absent in production code.
3. **Implicit, unverifiable trust placed in an ancient, pinned CUDA 10.2 toolchain** — downloaded over plain HTTP from NVIDIA with no integrity verification.
4. **Unsafe code escaping its intended boundary** — `unsafe` blocks appear directly inside safe public API methods with no documented safety invariants.
5. **Uncontrolled file-system writes in the `Persistable` trait** — callers supply a filename that is appended to the system temp directory without sanitization.
6. **Supply-chain risk in `npp-sys`** — bindings are re-generated at build time from whatever CUDA headers exist on the host; the published crate on crates.io (`npp-sys = "0.0.1"`) is consumed by `npp` without a workspace-path override.

---

## Trust Boundary Map

```
┌──────────────────────────────────────────────────────────────────┐
│  Caller (Rust, safe)                                             │
│    CudaImage::resize / bgra_to_rgb / sub_image / save           │
└───────────────────────────────┬──────────────────────────────────┘
                                │  [Trust boundary — Rust safe/unsafe]
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  npp crate — "safe" wrapper layer                                │
│    image.rs · resize_ops.rs · swap_channel_ops.rs               │
│                                                                  │
│    ⚠ Raw pointer arithmetic on DeviceBuffer                     │
│    ⚠ debug_assert! layout checks (release: DISABLED)            │
│    ⚠ Rc<RefCell<DeviceBuffer<T>>> — not Send/Sync               │
└───────────────────────────────┬──────────────────────────────────┘
                                │  [Trust boundary — Rust/C FFI]
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  npp-sys crate — raw bindgen bindings                            │
│    nppiResize_8u_C3R · nppiSwapChannels_8u_C4C3R                │
│    nppiMalloc_* · nppiFree                                       │
└───────────────────────────────┬──────────────────────────────────┘
                                │  [Trust boundary — userspace/kernel]
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  NVIDIA NPP / CUDA Runtime — closed-source kernel driver        │
└──────────────────────────────────────────────────────────────────┘
```

Every layer below the safe/unsafe boundary is **fully trusted without verification**.

---

## High-Risk Findings

### RISK-01 — Out-of-Bounds GPU Memory Access via Unsafe Pointer Offset  
**Severity: HIGH**  
**Files:** `npp/src/image.rs:170-176`, `npp/src/resize_ops.rs:61-73`, `npp/src/swap_channel_ops.rs:19-32`

Every NPP FFI call constructs a raw GPU device pointer using:
```rust
di.image_buf.borrow_mut().as_device_ptr().offset(di.layout.img_index as isize)
```

`img_index` is a `usize` from `CudaLayout`, computed from caller-supplied `x`, `y`, `w`, `h` in `sub_image()`. The `in_bounds` check in `sub_image` is the only guard:

```rust
fn in_bounds(&self, x: u32, y: u32) -> bool {
    let (ix, iy, iw, ih) = self.bounds();
    x >= ix && x <= ix + iw && y >= iy && y <= iy + ih   // ← inclusive boundary
}
```

**Vulnerability:** The check is `x <= ix + iw` (inclusive), but valid pixel indices are `0..width-1`. A sub-image at `(width, height, 0, 0)` passes the check, and `get_index(width, height)` produces an offset one full row past the end of the allocation. When this is passed to `DeviceBuffer::offset()`, CUDA reads/writes memory outside the allocated region. GPU memory is not subject to Rust's borrow checker or CUDA's MMU-level isolation between allocations by default.

**Blast Radius:** An attacker or buggy caller can read or corrupt arbitrary GPU device memory belonging to the same process, including other images, neural network weights, or model parameters loaded on the GPU.

---

### RISK-02 — Safety Invariants Are Only Enforced in Debug Builds  
**Severity: HIGH**  
**Files:** `npp/src/resize_ops.rs:38-39`, `npp/src/swap_channel_ops.rs:9-13`

All precondition checks guarding FFI call correctness are `debug_assert!`:

```rust
debug_assert!(src.layout.channel_stride == 1 && dst.layout.channel_stride == 1);
debug_assert!(src.layout.width_stride == 3 && dst.layout.width_stride == 3);
```

`debug_assert!` compiles to nothing in `--release` builds. There is no runtime enforcement that:
- The source has exactly 3 channels before calling the C3R (3-channel) NPP variant.
- Source and destination images have the same dimensions before `bgra_to_rgb`.
- Stride values are within the range `nppiResize` expects.

**Vulnerability:** In release mode, a 1-channel or 4-channel image passed to `CudaImage::resize` will invoke `nppiResize_8u_C3R` with a mismatched stride, causing the NPP C library to read/write a different number of bytes per row than were allocated. This is a classic out-of-bounds memory write at the C level, with no Rust safety net.

**Blast Radius:** Mismatched strides silently corrupt adjacent GPU memory. If a neural network inference pipeline shares GPU context, its weight buffers are within reach.

---

### RISK-03 — Build-Time Supply-Chain Trust: HTTP Download with No Integrity Check  
**Severity: HIGH**  
**Files:** `ci/install_server.sh`

The CI provisioning script downloads ~25 CUDA 10.2 `.deb` packages over **plain HTTP** from `developer.download.nvidia.com`:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-npp-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-npp-dev-10-2_10.2.89-1_amd64.deb
```

The URL scheme is HTTPS, but there is **no `--check-certificate` enforcement, no SHA256 verification, and no GPG signature check** before `dpkg -i` runs with `sudo`. Any attacker with a position between the runner and NVIDIA's CDN (BGP hijack, CDN compromise, or a stale cache) can substitute a malicious package, which will be installed as root and become part of the static library linked into every binary produced by this CI pipeline.

**Blast Radius:** Full compromise of the build environment and any artifact produced by CI. All published crate versions built through this pipeline must be considered potentially tainted.

---

### RISK-04 — Unsafe Code Exposed Directly Through Safe Public API  
**Severity: MEDIUM-HIGH**  
**Files:** `npp/src/image.rs:156-190`, `npp/src/resize_ops.rs:32-93`, `npp/src/swap_channel_ops.rs:7-49`

Rust's safety contract requires that `unsafe` code in a non-`unsafe fn` function must uphold all invariants by the surrounding code alone. The public API functions (`CudaImage::resize`, `CudaImage::bgra_to_rgb`, `RgbImage::try_from(&CudaImage)`) are all safe functions that contain `unsafe` blocks relying on preconditions that are either:
- Only checked with `debug_assert!` (see RISK-02), or
- Not checked at all (e.g., the `DeviceSlice::from_raw_parts` in `try_from`).

```rust
// image.rs:175-176 — no bounds validation before constructing a DeviceSlice
let slice = unsafe {
    DeviceSlice::from_raw_parts(ptr, di.layout.width as usize * di.layout.width_stride)
};
```

The length passed to `from_raw_parts` is computed from `CudaLayout` fields. If `CudaLayout` was created via `From<SampleLayout>` with caller-controlled data, this length is completely unvalidated against the actual size of `image_buf`.

**Blast Radius:** Read-over-the-end of a GPU DeviceBuffer during host-to-device copy operations. Undefined behavior at the rustacuda level; actual behavior depends on CUDA driver version and allocation alignment.

---

### RISK-05 — Uncontrolled File-System Write via `Persistable::save`  
**Severity: MEDIUM**  
**Files:** `npp/src/image.rs:192-201`

```rust
impl Persistable for CudaImage<u8> {
    fn save(&self, filename: &str) -> Result<(), image::ImageError> {
        let mut tmp_dir = temp_dir();
        tmp_dir.push(filename);          // ← caller controls path component
        tmp_dir.set_extension("png");
        let path = tmp_dir.to_str().unwrap();
        ...
    }
}
```

`push()` on a `PathBuf` replaces the entire path if the argument is an absolute path (e.g., `/etc/passwd`) or navigates upward if it contains `..`. A caller passing `"../../etc/cron.d/evil"` will write a PNG file to a path outside of `temp_dir`.

Additionally, `tmp_dir.to_str().unwrap()` will panic if the path contains non-UTF-8 bytes on platforms where that is possible.

**Blast Radius:** Arbitrary file write to any path accessible to the process. While constrained by OS-level permissions, in containerized or privileged build/inference environments, this could overwrite configuration or executable files.

---

### RISK-06 — Implicit Trust in Crates.io npp-sys vs. Local Path  
**Severity: MEDIUM**  
**Files:** `npp/Cargo.toml:13-14`

```toml
# npp-sys = { path = "../npp-sys" }   ← commented out
npp-sys = "0.0.1"                     ← active: resolved from crates.io
```

The `npp` crate consumes `npp-sys` from **crates.io** rather than the workspace-local crate. The local `npp-sys` at `../npp-sys` is the source of truth for the bindings, but any divergence between the two (version drift, crates.io upload delay, or a typosquatting/dependency-confusion attack against the `npp-sys` package name) means the published API contract is not verified against the local build.

**Blast Radius:** A compromised or outdated `npp-sys` on crates.io introduces incorrect FFI declarations. Since the bindings are `unsafe`, a mismatched function signature (e.g., wrong argument count or type) compiles without error and produces undefined behavior at runtime.

---

### RISK-07 — CUDA Device 0 Always Selected Without Validation  
**Severity: MEDIUM**  
**Files:** `npp/src/cuda.rs:6`

```rust
let device = Device::get_device(0)?;
```

Device index 0 is unconditionally selected. In a multi-tenant or multi-GPU environment, this creates two problems:

1. **Data Isolation:** There is no enforcement that device 0 belongs to or is isolated for the calling workload. In shared inference servers, multiple processes sharing device 0 share the same CUDA context namespace. CUDA contexts provide process-level isolation, but NPP operates in the default stream unless told otherwise — all operations are serialized and potentially observable via timing channels.
2. **No Validation of `get_device` Result:** The `?` propagates `CudaError`, but callers in tests and benchmarks use `.unwrap()` on the context, meaning any multi-GPU or headless environment that fails to enumerate device 0 will panic rather than fail gracefully.

**Blast Radius:** In shared-GPU deployments, timing side-channels on the default stream could allow one workload to infer properties of another's computation.

---

## Additional Observations (Low Severity)

| ID | Location | Observation |
|----|----------|-------------|
| OBS-01 | `npp-sys/build.rs:7` | `CUDA_INSTALL_DIR` falls back to `/usr/local/cuda` silently; builds succeed against a mismatched CUDA version with no warning, producing bindings that may not match linked libraries. |
| OBS-02 | `npp/src/image.rs:187` | `CudaError::UnknownError` is used as a catch-all for image errors. Error information is destroyed at the boundary; callers cannot distinguish allocation failure from dimension mismatch. |
| OBS-03 | `npp-sys/src/lib.rs:1-3` | `#![allow(non_upper_case_globals/non_camel_case_types/non_snake_case)]` suppresses all naming warnings for the entire bindgen output. Any naming collision with Rust keywords or future std symbols will be silently accepted. |
| OBS-04 | `npp/benches/*.rs` | Each benchmark function calls `rustacuda::init()` independently. Multiple inits in one process is explicitly unsupported by the CUDA runtime; benchmarks may behave non-deterministically depending on run order. |
| OBS-05 | `build.yml` | CI targets `ubuntu-18.04`, which reached end-of-life in April 2023. The runner image no longer receives security patches. |

---

## Risk Summary Table

| ID | Description | Severity | Confidentiality | Integrity |
|----|-------------|----------|-----------------|-----------|
| RISK-01 | OOB GPU memory via inclusive `in_bounds` | **HIGH** | Yes — read adjacent GPU memory | Yes — corrupt adjacent buffers |
| RISK-02 | `debug_assert!` guards absent in release | **HIGH** | — | Yes — corrupt GPU memory on stride mismatch |
| RISK-03 | HTTP + no integrity check in CI install | **HIGH** | Yes — CI artifact exfil | Yes — backdoored static libs |
| RISK-04 | Unsafe in safe public API, unverified preconditions | **MEDIUM-HIGH** | Yes | Yes |
| RISK-05 | Path traversal in `Persistable::save` | **MEDIUM** | — | Yes — arbitrary file write |
| RISK-06 | npp-sys from crates.io, not workspace path | **MEDIUM** | — | Yes — FFI mismatch / UB |
| RISK-07 | Device 0 always selected, default stream | **MEDIUM** | Timing side-channel | — |

---

## Prioritized Remediation Order

1. **RISK-02 first:** Replace `debug_assert!` with `assert!` or explicit `Result`-returning validation. This is the most likely path to silent memory corruption in a release binary and requires the least code change.
2. **RISK-01:** Fix the `in_bounds` boundary check (change `<=` to `<` for both `x` and `y` against `iw`/`ih`). Add a separate check that `x + w <= self.layout.width` and `y + h <= self.layout.height`.
3. **RISK-03:** Add SHA256 checksums for every `.deb` download in `install_server.sh`. Verify with `sha256sum --check` before `dpkg -i`. Separately, update CI to a supported Ubuntu LTS.
4. **RISK-05:** Sanitize `filename` in `Persistable::save`: reject absolute paths and any component containing `..` or path separators.
5. **RISK-06:** Re-enable `npp-sys = { path = "../npp-sys" }` in the workspace and remove the crates.io dependency to guarantee local and published bindings match.
6. **RISK-04:** Document every `unsafe` block with a `// SAFETY:` comment that explicitly states the invariants being upheld and where they are verified. Add runtime checks where `debug_assert!` was the only guard.
