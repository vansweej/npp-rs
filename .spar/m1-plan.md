# Feature: Milestone 1 — Build again on latest CUDA via Nix + cudarc

> **MANDATORY READING before working any phase:**
> Read `reviews/final-report.md` IN FULL. This plan cites findings by ID (C1, C2, …,
> CT1, …) but the report holds the evidence, the cross-panel reasoning, and the
> Round-2 retractions that changed severities (CT2, CT6).
>
> Also read `reviews/M1-build-again-plan.md` — its Phase 1/3/5 infrastructure
> shape is retained but its Phase 2 is superseded by this plan.
>
> Also read `.spar/brief.md` — the design decisions from the June-19 review
> session that this plan implements.

## Scope statement

Restore buildability on the latest nixpkgs CUDA, replace the dead
`rustacuda`/static-link/CUDA-10.2 stack with **cudarc 0.9 + shared NPP linking +
bindgen allowlist**. Beyond the mechanical port, this milestone also delivers:

- The `NppPixelType` alphabet (~9 types, all constructible)
- Capability-trait architecture (`Resize`, `SwapChannels`)
- Hand-written `u8` and `f32` `Resize` exemplars
- `Vec<T>` round-trip (no `image` crate in core)
- One golden-image correctness test (NearestNeighbor)

**Out of scope (M2+):** macro codegen (F1), alphabet ops beyond `u8`/`f32` (F2),
image-rs (F3) and graphynx (F4) boundaries, convert ops (F5), full golden suite
(F6), C2 `Result`-hardening beyond the type seal (F7), streams/exec-context (F8),
`npps*` signals (F9), IPP bindings (F10), benchmark port (parked).

## Cross-cutting rules (apply everywhere)

- `status >= 0` is success: positive NPP codes are warnings, not errors (fixes C1/NEW-01).
- `NppError` wraps the raw signed `NppStatus` (`i32`). No `NotImplemented` variant.
  cudarc errors are `#[from] cudarc::driver::DriverError` (not `CudaResultError` — that type does not exist).
- No `unwrap()`/`expect()` in non-test code. Use `Result<T, NppError>`.
- Document the C7 context-lifetime invariant (`Arc<CudaDevice>` outlives all
  `CudaImage`s) at every relevant `pub` item.
- `npp-sys/src/bindings.rs` is generated into `OUT_DIR` — never edit or commit it.
- `npp/` directory is crate `npp-rs`. Use `cargo ... -p npp-rs` for it,
  `cargo ... -p npp-sys` for the FFI crate.

---

## Phase 1: Nix dev shell + npp-sys build rewrite

The Nix dev shell provides CUDA toolkit, NPP libraries, libclang for bindgen, and
Rust toolchain in a pinned, reproducible environment. The `build.rs` rewrite consumes
these Nix-provided paths and switches from static to shared NPP linking.

Commit message: `feat: add Nix dev shell and rewrite npp-sys build for shared NPP linking`

### Step 1: Create `flake.nix` at `/home/vansweej/Work/npp-rs/flake.nix`

Create a Nix flake modelled on the graphynx reference (inlined in
`reviews/M1-build-again-plan.md` Appendix A.1) with a **lean** CUDA toolchain
(this crate has no PTX kernels — no nvcc/nvrtc/gcc13 needed).

**Inputs:**
- `nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"`
- `rust-overlay = { url = "github:oxalica/rust-overlay"; inputs.nixpkgs.follows = "nixpkgs"; }`
- `flake-utils.url = "github:numtide/flake-utils"`

**Outputs config:**
- `pkgs = import nixpkgs { inherit system overlays; config.allowUnfree = true; }`
- `rustVersion = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;`
- `cudaPackages = pkgs.cudaPackages;`

**`devShells.default = pkgs.mkShell { name = "npp-rs"; … }` buildInputs:**
- `rustVersion` (toolchain from rust-toolchain.toml)
- `rust-analyzer`
- `cudaPackages.cuda_cudart` (CUDA runtime headers + stub libs)
- `cudaPackages.libnpp` (NPP shared libs + headers)
- `llvmPackages.libclang` (for bindgen)
- `stdenv.cc.cc.lib` (C++ stdlib needed by CUDA headers)
- `pkg-config`

**Do NOT include:** `cuda_nvcc`, `cuda_nvrtc`, `gcc13`, `alsa-lib`, `cargo-deny`,
`cargo-tarpaulin`, `cargo-edit`, `cargo-watch`, `cargo-outdated`, `cargo-update`.

**shellHook** must set:
- `export CUDA_PATH="${cudaPackages.cuda_cudart}"`
- `export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"`
- `export BINDGEN_EXTRA_CLANG_ARGS="-I${cudaPackages.cuda_cudart}/include -I${cudaPackages.libnpp}/include"`
- `export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${cudaPackages.cuda_cudart}/lib:${cudaPackages.libnpp}/lib:$LD_LIBRARY_PATH"`
- The `.nvidia-libs/` symlink loop (exactly as Appendix A.1: create `mkdir -p .nvidia-libs`, symlink `libcuda.so.1`, `libnvidia-ptxjitcompiler.so.1`, `libnvidia-nvvm.so.4` from `/usr/lib/x86_64-linux-gnu`).
- The `NIX_LDFLAGS` `-rpath …/outputs/out` scrub (exact sed from Appendix A.1).
- `export RUSTFLAGS="-C link-arg=-Wl,-rpath,$PWD/.nvidia-libs -C link-arg=-Wl,-rpath,${pkgs.glibc}/lib $RUSTFLAGS"`

### Step 2: Create `rust-toolchain.toml` at `/home/vansweej/Work/npp-rs/rust-toolchain.toml`

```toml
[toolchain]
channel = "1.94.1"
```

Add a comment above: `# Pinned — update deliberately with nix flake update + cargo test`.

### Step 3: Rewrite `/home/vansweej/Work/npp-rs/npp-sys/build.rs`

Replace the entire file. **Spec:**
- Delete every `#[cfg(target_os = "windows")]` function and branch. Linux only.
- `fn cuda_path() -> String`: read `std::env::var("CUDA_PATH")` first, fall back to
  `option_env!("CUDA_INSTALL_DIR")` (set in `$HOME/.cargo/config.toml` for
  non-Nix builds), then `"/usr/local/cuda"`.
- `fn cuda_include_path() -> String { format!("{}/include", cuda_path()) }`.
- `fn cuda_configuration()`: `println!("cargo:rustc-link-search=native={}/lib", cuda_path())`
  and the same for `lib64`.
- `fn cuda_link_libs()`: emit **shared** NPP libs only:
  `cargo:rustc-link-lib=shared=nppc` through `shared=nppitc` (exactly the 10 libs
  from `nppc/nppial/nppicc/nppidei/nppif/nppig/nppim/nppist/nppisu/nppitc`),
  plus `cargo:rustc-link-lib=shared=cudart`.
  **Remove** all `*_static` names, `cudart_static`, `culibos`, `dylib=stdc++`.
- In `main()`: call `cuda_configuration()` then `cuda_link_libs()`, emit
  `cargo:rerun-if-changed=wrapper.h`, emit `cargo:rerun-if-env-changed=CUDA_PATH`.
- Bindgen builder:
  ```rust
  let bindings = bindgen::Builder::default()
      .header("wrapper.h")
      .clang_args(&["-I", &cuda_include_path()])
      .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
      .generate_comments(false)
      .allowlist_function("nppi.*")
      .allowlist_type("Nppi.*")
      .allowlist_var("Nppi.*")
      .generate()
      .expect("Unable to generate bindings");
  ```
  Write to `OUT_DIR/bindings.rs` as today.

Wrapper header (`npp-sys/wrapper.h`): leave unchanged (includes `nppi.h`, comments out `npps.h`).

### Step 4: Update crate manifests for path deps

In `/home/vansweej/Work/npp-rs/npp-sys/Cargo.toml`:
- Update `[build-dependencies] bindgen = "0.69"` (latest 0.69.x release).
- Keep `build = "build.rs"` (harmless, explicit).

In `/home/vansweej/Work/npp-rs/npp/Cargo.toml`:
- Change `npp-sys = "0.0.1"` to `npp-sys = { path = "../npp-sys" }`.
- Keep the old line as a commented-out release fallback:
  `# npp-sys = "0.0.1"  # release fallback`

Do NOT touch any other dependencies yet (Phase 2 swaps those).

### Step 5: Update `.gitignore` at `/home/vansweej/Work/npp-rs/.gitignore`

Create the file if it doesn't exist. Ensure it contains:
```
/target
Cargo.lock
.nvidia-libs/
result
result-*
# Generated bindings are in OUT_DIR
```

Add a comment explaining `Cargo.lock` is gitignored per project convention (AGENTS.md).

### Step 6: Verify Phase 1 builds

Verification guidance (no file change — executor runs this):
```bash
nix develop . --command cargo build -p npp-sys
```
Must succeed — `npp-sys` compiles against the local CUDA + NPP libraries.
The `npp-rs` crate will NOT compile yet (still depends on `rustacuda` from
crates.io).

---

## Phase 2: Replace rustacuda with cudarc, port device integration, capability-trait architecture, and golden test

**CRITICAL: implement steps in order. Step 1 (the FFI bridge) gates everything.
Do not proceed past it until the documented pattern compiles.**

Commit message:
`feat: port to cudarc 0.9 with NppPixelType capability traits and Vec<T> round-trip`

### Step 1: Document the cudarc `CudaSlice` → raw NPP pointer bridge

Create `/home/vansweej/Work/npp-rs/docs/spike-cudarc-ptr-bridge.md`.

Write a short note establishing the exact, authoritative pattern every NPP call
site uses to obtain raw device pointers from a `CudaSlice<T>`. This is the single
source of truth for Steps 3, 6, 7, and 8.

**Authoritative pattern (cudarc 0.9.x):**

```rust
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};

// ── Allocation ──────────────────────────────────────────────────────────
// alloc_zeros is SAFE in cudarc 0.9 — no `unsafe` needed.
// Corrects the stale plan's incorrect `unsafe` label.
let device: Arc<CudaDevice> = CudaDevice::new(0)?;
let buf: CudaSlice<u8> = device.alloc_zeros::<u8>(num_elements)?;

// ── Host upload ─────────────────────────────────────────────────────────
let uploaded: CudaSlice<u8> = device.htod_sync_copy(&host_data)?;

// ── Raw pointer from CudaSlice (for NPP extern "C" calls) ──────────────
// Use the `DevicePtr` / `DevicePtrMut` traits from cudarc:
use cudarc::driver::DevicePtr as _;    // provides .device_ptr() -> CUdeviceptr (u64)
use cudarc::driver::DevicePtrMut as _; // provides .device_ptr_mut() -> CUdeviceptr

// For a const (read-only) pointer at element offset `img_index`:
let cu_ptr: u64 = DevicePtr::device_ptr(&buf);
let src_ptr: *const u8 = (cu_ptr + (img_index as u64) * size_of::<u8>()) as *const u8;

// For a mutable pointer:
let cu_ptr_mut: u64 = DevicePtrMut::device_ptr_mut(&mut buf);
let dst_ptr: *mut u8 = (cu_ptr_mut + (img_index as u64) * size_of::<u8>()) as *mut u8;

// IMPORTANT: the CudaSlice must NOT be dropped for the duration of the NPP
// call. Derive the pointer and call NPP in the same statement scope.

// ── Borrowed sub-views (for CudaImageView) ──────────────────────────────
// cudarc CudaSlice provides `.slice(range)` / `.slice_mut(range)` which
// return borrowed slice references. These reference types implement DevicePtr
// / DevicePtrMut too, so raw-pointer extraction follows the same pattern:
let sub: &CudaSlice<u8> = &buf.slice(start..end);  // CHECK: actual return type
//     may be CudaSliceRef<'_, u8> — if so, use it via Deref<Target=CudaSlice<u8>>.
let sub_ptr: *const u8 = {
    let cu = DevicePtr::device_ptr(sub);
    cu as *const u8
};

// ── NPP call (CudaSlice must outlive the raw pointer) ───────────────────
let status = unsafe {
    nppiResize_8u_C3R(src_ptr, nStep_bytes, src_size, src_rect,
                      dst_ptr, dst_step_bytes, dst_size, dst_rect, mode)
};
check_status(status)?;

// ── Host read-back (replaces old set_len pattern — C5 fixed) ────────────
let host_vec: Vec<u8> = device.dtoh_sync_copy(&buf)?;

// ── Error type ─────────────────────────────────────────────────────────
// cudarc errors are cudarc::driver::DriverError (NOT CudaResultError,
// which does not exist in 0.9).
```

**If the pinned cudarc 0.9.x minor version uses different names** for the
borrowed-slice types or the trait function signatures, update the file
accordingly. The pattern inside the `use` + `DevicePtr::device_ptr` +
`(cu_ptr + offset)` + `as *const T` sequence is structurally stable across
cudarc 0.9.x.

Document this pattern in the spike note. Cite this file from every
`pub(crate) fn device_ptr(&self)` method on `CudaImageView`.

### Step 2: Create `/home/vansweej/Work/npp-rs/npp/src/error.rs`

Create the file. **Exact contents:**

```rust
use thiserror::Error;

/// Raw NPP status code. Positive values are warnings, zero is success,
/// negative values are errors.
pub type NppStatus = i32;

/// Errors that can occur during NPP operations.
#[derive(Error, Debug)]
pub enum NppError {
    /// NPP library returned a negative (error) status code.
    #[error("NPP returned error status {0}")]
    Npp(NppStatus),

    /// CUDA driver-level error (allocation, copy, context).
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    /// Invalid argument passed to a function (precondition failure).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Check an NPP status code and return `Ok(())` on success (zero or positive
/// warning code) or `Err(NppError::Npp(status))` on error (negative code).
///
/// # Notes
///
/// Positive `NppStatus` values are *warnings* and do not indicate failure.
/// This is a deliberate fix for the original crate's `status == 0` check which
/// incorrectly treated positive warning codes as hard errors (C1/NEW-01).
pub fn check_status(status: NppStatus) -> Result<(), NppError> {
    if status >= 0 {
        Ok(())
    } else {
        Err(NppError::Npp(status))
    }
}
```

### Step 3: Create the `NppPixelType` marker trait and new `CudaImage<T>` + `CudaImageView<'a, T>`

This step replaces the entire `/home/vansweej/Work/npp-rs/npp/src/image.rs`.

**Requirements for this one file (write it in full):**

**Imports:**
```rust
use crate::layout::CudaLayout;
use crate::error::NppError;
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, CudaViewMut};
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::sync::Arc;
```

**Marker trait (sealed):**
```rust
/// Marker trait for NPP primitive pixel element types. Every concrete NPP
/// element type implements this trait. The trait is sealed — it cannot be
/// implemented outside this module.
///
/// Types implementing `NppPixelType` are:
///
/// | NPP name | Rust type | Bits |
/// |----------|-----------|------|
/// | `8u`     | `u8`      | 8    |
/// | `8s`     | `i8`      | 8    |
/// | `16u`    | `u16`     | 16   |
/// | `16s`    | `i16`     | 16   |
/// | `32u`    | `u32`     | 32   |
/// | `32s`    | `i32`     | 32   |
/// | `32f`    | `f32`     | 32   |
/// | `64f`    | `f64`     | 64   |
/// | `16f`    | `half::f16` | 16 |  (requires `half` crate)
///
/// All types are constructible (allocatable + zeroable) in M1. Operation
/// capability is expressed by separate traits (e.g. `Resize`, `SwapChannels`);
/// an unsupported `(type, op)` pair simply has no trait impl, making it a
/// compile-time error rather than a runtime `NotImplemented`.
pub trait NppPixelType: cudarc::driver::DeviceRepr + Copy + private::Sealed {
    const BITS: u8;
}

mod private {
    pub trait Sealed {}
}

impl private::Sealed for u8 {}
impl NppPixelType for u8 { const BITS: u8 = 8; }

impl private::Sealed for i8 {}
impl NppPixelType for i8 { const BITS: u8 = 8; }

impl private::Sealed for u16 {}
impl NppPixelType for u16 { const BITS: u8 = 16; }

impl private::Sealed for i16 {}
impl NppPixelType for i16 { const BITS: u8 = 16; }

impl private::Sealed for u32 {}
impl NppPixelType for u32 { const BITS: u8 = 32; }

impl private::Sealed for i32 {}
impl NppPixelType for i32 { const BITS: u8 = 32; }

impl private::Sealed for f32 {}
impl NppPixelType for f32 { const BITS: u8 = 32; }

impl private::Sealed for f64 {}
impl NppPixelType for f64 { const BITS: u8 = 64; }
```

If you add `half = "2"` to `npp/Cargo.toml` in Step 9, add the 16f impl here too:
```rust
// Requires `half = "2"` in Cargo.toml
impl private::Sealed for half::f16 {}
impl NppPixelType for half::f16 { const BITS: u8 = 16; }
```
Otherwise add a `// TODO(M2): 16f via half crate` note.

**`CudaImage<T>` struct:**
```rust
/// A GPU-resident N-dimensional image buffer backed by a contiguous NPP-compatible
/// device allocation. `T` is the element type (see `NppPixelType`).
///
/// # Device lifetime invariant (C7)
///
/// The `Arc<CudaDevice>` stored in every `CudaImage` must outlive all operations
/// on the image. Dropping the device while a `CudaImage` is live results in
/// `cuMemFree` against a destroyed context. This is enforced for free by cudarc's
/// internal `Arc<CudaDevice>` reference on every `CudaSlice`.
///
/// # Thread safety
///
/// `CudaImage<T>` is `Send + Sync` because `CudaSlice<T>` is `Send + Sync` in
/// cudarc 0.9. However, CUDA contexts are thread-bound; safe cross-thread usage
/// requires explicit context management (deferred to M2 — see F8).
#[derive(Debug)]
pub struct CudaImage<T: NppPixelType> {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) buf: CudaSlice<T>,
    pub(crate) layout: CudaLayout,
}
```

**`impl<T: NppPixelType> CudaImage<T>` methods:**
- `pub fn new(device: Arc<CudaDevice>, channels: u8, width: u32, height: u32) -> Result<Self, NppError>`:
  Allocate `device.alloc_zeros::<T>(width as usize * height as usize * channels as usize)?`,
  build `CudaLayout::row_major_packed(channels, width, height)`.
- `pub fn from_host(device: Arc<CudaDevice>, channels: u8, width: u32, height: u32, data: &[T]) -> Result<Self, NppError>`:
  Validate `data.len() == width*height*channels` (return `Err(NppError::InvalidArgument(…))`),
  allocate + `device.htod_sync_copy(data)?`, build layout.
- `dimensions`, `width`, `height`, `channels` — delegate to `self.layout`.
- `get_index(x, y) -> usize`: `y * height_stride + x * width_stride`.
- `get_start_point(&self) -> (u32, u32)` — port logic from old code, but divide
  `img_index` by `layout.channels as usize` for x (corrected from old code that
  used `channels` where channels already includes the element-size factor — in the
  new layout, `channels` is pure channel count and `width_stride` is `channels`).
- `bounds(&self) -> (0, 0, width, height)`.
- `in_bounds(&self, x, y) -> bool`: use strict `<` on far corner: `x < self.layout.width && y < self.layout.height`. (Fix CT6 off-by-one — simple inclusive check is fine if using both corners in `sub_image`.)
- `sub_image(&self, x, y, w, h) -> Result<CudaImageView<'_, T>, NppError>`:
  - Validates **both** corners: if `x + w > self.layout.width || y + h > self.layout.height` → `InvalidArgument`.
  - Computes `img_index = self.get_index(x, y)`.
  - The element range for the sub-slice: `start = self.layout.img_index + img_index`,
    `len = h as usize * self.layout.height_stride` (the whole sub-rect as a contiguous sub-slice;
    actual row-by-row stride handling for ROI reads is M2).
  - Takes `self.buf.slice(start..start + len)` → `CudaView<'_, T>`.
  - Builds `CudaLayout { channels, channel_stride: 1, width: w, width_stride: self.layout.width_stride, height: h, height_stride: self.layout.height_stride, img_index: 0 }` — the view's `img_index` is 0 because the slice starts at the correct offset.
  - Returns `CudaImageView { device: self.device.clone(), view, layout }`.
- `sub_image_mut(&mut self, …) -> Result<CudaImageViewMut<'_, T>, NppError>` — same logic
  but with `slice_mut(range)` and `CudaViewMut`.

**`CudaImageView<'a, T>` and `CudaImageViewMut<'a, T>`:**
```rust
#[derive(Debug)]
pub struct CudaImageView<'a, T: NppPixelType> {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) view: CudaView<'a, T>,
    pub(crate) layout: CudaLayout,
}

pub struct CudaImageViewMut<'a, T: NppPixelType> {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) view: CudaViewMut<'a, T>,
    pub(crate) layout: CudaLayout,
}
```

Implement `width`, `height`, `channels`, `dimensions`, `get_start_point` on both,
delegating to `self.layout`. Add a `pub(crate)` method to extract the raw device
pointer at the view's base offset (used by resize/swap ops):
```rust
impl<'a, T: NppPixelType> CudaImageView<'a, T> {
    pub(crate) fn device_ptr(&self) -> *const T {
        // Use the spike-confirmed pattern to get raw pointer from CudaView
        self.view.as_device_ptr()
    }
}
impl<'a, T: NppPixelType> CudaImageViewMut<'a, T> {
    pub(crate) fn device_ptr_mut(&mut self) -> *mut T {
        self.view.as_device_ptr_mut()
    }
}
```
(Adjust method names to match cudarc 0.9's view traits.)

**`Vec<T>` round-trip:**
```rust
impl<T: NppPixelType> TryFrom<&CudaImage<T>> for Vec<T> {
    type Error = NppError;
    fn try_from(img: &CudaImage<T>) -> Result<Self, Self::Error> {
        let host: Vec<T> = img.device.dtoh_sync_copy(&img.buf)?;
        Ok(host)
    }
}
```
This copies the *entire* device buffer (contiguous allocation). Sub-image
stride-correct read-back is M2. Unit tests and the golden test work on full images.

Do **NOT** port `Persistable`, the `RgbImage`/`RgbaImage`/`Bgra` `TryFrom` impls,
`CopyFromImage`, or `CopyToImage` — they are removed in M1.

### Step 4: Update `/home/vansweej/Work/npp-rs/npp/src/layout.rs` — remove `image` dep

- Remove `use image::flat::SampleLayout;` and `use std::convert::From;` (keep only
  the `From` import if needed — `From` is in the prelude on 2021 edition, so the
  explicit `use` is unnecessary).
- Remove the `impl From<SampleLayout> for CudaLayout` block and its associated test
  `test_from_sample_layout`.
- Keep `CudaLayout{..}`, `row_major_packed(..)`, `test_row_major_packed`.
- Add `///` doc to `CudaLayout` struct: "Describes the memory layout of a packed
  NPP-compatible image buffer."

### Step 5: Define the `Resize` and `SwapChannels` capability traits

Replace `/home/vansweej/Work/npp-rs/npp/src/imageops.rs` entirely:

```rust
use crate::error::NppError;

/// Interpolation methods supported by NPP resize operations.
///
/// Note: `Lanczos` is not supported for 16f channel types (NPP restriction).
#[derive(Debug, Clone, Copy)]
pub enum ResizeInterpolation {
    NearestNeighbor,
    Linear,
    Cubic,
    Super,
    Lanczos,
}

/// Capability trait for NPP resize operations.
///
/// Implemented only for pixel types that NPP supports for resize.
/// Unsupported `(type, op)` pairs simply have no impl — calling them is
/// a compile-time error.
///
/// # Precondition
///
/// `src` and `dst` must refer to **non-overlapping** device buffers.
/// Passing overlapping ROIs (e.g. two sub-views of the same parent image)
/// to resize is undefined behavior in NPP.
pub trait Resize: Sized {
    fn resize(&self, dst: &mut Self, inter: ResizeInterpolation) -> Result<(), NppError>;
}

/// Capability trait for 4-channel BGRA → 3-channel RGB channel reordering.
///
/// Same impl/non-impl model as `Resize`. Current M1 scope: `CudaImage<u8>` only.
///
/// # Precondition
///
/// `src` and `dst` must refer to **non-overlapping** device buffers.
pub trait SwapChannels: Sized {
    fn bgra_to_rgb(&self, dst: &mut Self) -> Result<(), NppError>;
}
```

### Step 6: Implement `Resize for CudaImage<u8>` — `/home/vansweej/Work/npp-rs/npp/src/resize_ops.rs`

Replace the entire file. Must use the spike-confirmed pointer bridge (Step 1).

```rust
use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::{Resize, ResizeInterpolation};
use npp_sys::{
    nppiResize_8u_C3R,
    NppiInterpolationMode_NPPI_INTER_CUBIC,
    NppiInterpolationMode_NPPI_INTER_LANCZOS,
    NppiInterpolationMode_NPPI_INTER_LINEAR,
    NppiInterpolationMode_NPPI_INTER_NN,
    NppiInterpolationMode_NPPI_INTER_SUPER,
    NppiRect, NppiSize,
};

fn interpolation_mode(inter: ResizeInterpolation) -> i32 {
    match inter {
        ResizeInterpolation::NearestNeighbor => NppiInterpolationMode_NPPI_INTER_NN as i32,
        ResizeInterpolation::Linear => NppiInterpolationMode_NPPI_INTER_LINEAR as i32,
        ResizeInterpolation::Cubic => NppiInterpolationMode_NPPI_INTER_CUBIC as i32,
        ResizeInterpolation::Super => NppiInterpolationMode_NPPI_INTER_SUPER as i32,
        ResizeInterpolation::Lanczos => NppiInterpolationMode_NPPI_INTER_LANCZOS as i32,
    }
}

/// Resize for `CudaImage<u8>` over `nppiResize_8u_C3R` (3-channel packed RGB).
///
/// The raw pointer for both src and dst is offset by `layout.img_index` so
/// this impl works correctly on sub-images created via `CudaImage::sub_image`
/// (whose `layout.img_index` carries the parent's offset). Because the
/// `CudaImage` stores a full `CudaSlice<T>` whose base is the allocation
/// origin, and `img_index` is the element offset to the first pixel of the
/// (possibly sub-)image, we apply the offset uniformly — it is zero for
/// full-size images.
///
/// # Precondition
///
/// `self` and `dst` must refer to **non-overlapping** device buffers.
/// Passing overlapping ROIs is undefined behaviour in NPP (C4).
impl Resize for CudaImage<u8> {
    fn resize(&self, dst: &mut Self, inter: ResizeInterpolation) -> Result<(), NppError> {
        let (src_w, src_h) = (self.width(), self.height());
        let (dst_w, dst_h) = (dst.width(), dst.height());

        let src_size = NppiSize { width: src_w as i32, height: src_h as i32 };
        let dst_size = NppiSize { width: dst_w as i32, height: dst_h as i32 };
        let src_rect = NppiRect { x: 0, y: 0, width: src_w as i32, height: src_h as i32 };
        let dst_rect = NppiRect { x: 0, y: 0, width: dst_w as i32, height: dst_h as i32 };

        // ── Raw pointers via DevicePtr/DevicePtrMut trait ──────────────
        // Both the const and mutable CUdeviceptrs are offset by img_index
        // so sub-images (which have img_index != 0) address the correct
        // device memory. The height_stride is inherited from the parent
        // layout and is already correct.
        let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
        let src_ptr = (src_base + self.layout.img_index as u64) as *const u8;
        let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
        let dst_ptr = (dst_base + dst.layout.img_index as u64) as *mut u8;

        let status = unsafe {
            nppiResize_8u_C3R(
                src_ptr,
                self.layout.height_stride as i32,
                src_size,
                src_rect,
                dst_ptr,
                dst.layout.height_stride as i32,
                dst_size,
                dst_rect,
                interpolation_mode(inter),
            )
        };
        check_status(status)
    }
}

### Step 7: Implement `SwapChannels for CudaImage<u8>` — `/home/vansweej/Work/npp-rs/npp/src/swap_channel_ops.rs`

Replace the entire file with the `SwapChannels` impl for `nppiSwapChannels_8u_C4C3R`.

```rust
use crate::error::{check_status, NppError};
use crate::image::CudaImage;
use crate::imageops::SwapChannels;
use npp_sys::{nppiSwapChannels_8u_C4C3R, NppiSize};
use std::os::raw::c_int;

/// BGRA→RGB channel reorder for `CudaImage<u8>` over `nppiSwapChannels_8u_C4C3R`.
///
/// Takes a 4-channel (BGRA) source and writes a 3-channel (RGB) destination.
/// The raw pointer offset logic mirrors the `Resize` impl — sub-images are
/// handled correctly via `layout.img_index`.
///
/// # Precondition
///
/// - `src` and `dst` must refer to **non-overlapping** device buffers.
/// - `src` must have 4 channels (BGRA), `dst` must have 3 channels (RGB).
/// - Dimensions (width, height) of src and dst must match (enforced at
///   runtime via `InvalidArgument` — survives `--release`).
///
/// # Errors
///
/// Returns `InvalidArgument` if src and dst dimensions disagree. Returns
/// `Npp` (containing the raw NPP status code) if the NPP call fails.
impl SwapChannels for CudaImage<u8> {
    fn bgra_to_rgb(&self, dst: &mut Self) -> Result<(), NppError> {
        // Dimension agreement check (survives --release — C2 hardening)
        if self.width() != dst.width() || self.height() != dst.height() {
            return Err(NppError::InvalidArgument(
                "src and dst dimensions must match for bgra_to_rgb".into(),
            ));
        }

        let nppi_size = NppiSize {
            width: dst.width() as i32,
            height: dst.height() as i32,
        };

        let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
        let src_ptr = (src_base + self.layout.img_index as u64) as *const u8;
        let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
        let dst_ptr = (dst_base + dst.layout.img_index as u64) as *mut u8;

        let order: [c_int; 3] = [2, 1, 0];
        let status = unsafe {
            nppiSwapChannels_8u_C4C3R(
                src_ptr,
                self.layout.height_stride as i32,
                dst_ptr,
                dst.layout.height_stride as i32,
                nppi_size,
                &order[0],
            )
        };
        check_status(status)
    }
}
```

### Step 8: Implement `Resize for CudaImage<f32>` — same file, second impl

Append to `/home/vansweej/Work/npp-rs/npp/src/resize_ops.rs`. This is the
second hand-written exemplar that validates the trait architecture against a
different element type (brief mandatory step 4; user-confirmed C3/3-channel).

> **`nStep` unit warning:** NPP's `nStep` parameter is always in **bytes**.
> For `CudaSlice<f32>`, the layout's `height_stride` counts **elements** of `T`.
> Multiply by `size_of::<f32>()` to get the byte step required by NPP. This is
> the exact class of element-size-vs-channel-count bug that finding C11 warned
> about — the new layout model separates element size (carried by `CudaSlice<T>`)
> from channel count (carried by `CudaLayout`), and the conversion here is
> explicit and auditable.

```rust
use npp_sys::nppiResize_32f_C3R;

/// Resize for `CudaImage<f32>` over `nppiResize_32f_C3R` (3-channel packed float).
///
/// # nStep unit conversion
///
/// NPP's `nStep` is in **bytes**. `layout.height_stride` stores the per-row
/// element count; we multiply by `size_of::<f32>()` to produce the byte step.
/// This is the correct pattern for all non-`u8` types — for `u8` the two
/// coincide, but for wider types the explicit conversion prevents the C11
/// class of bug.
///
/// # Precondition
///
/// `self` and `dst` must refer to **non-overlapping** device buffers.
impl Resize for CudaImage<f32> {
    fn resize(&self, dst: &mut Self, inter: ResizeInterpolation) -> Result<(), NppError> {
        let (src_w, src_h) = (self.width(), self.height());
        let (dst_w, dst_h) = (dst.width(), dst.height());

        let src_size = NppiSize { width: src_w as i32, height: src_h as i32 };
        let dst_size = NppiSize { width: dst_w as i32, height: dst_h as i32 };
        let src_rect = NppiRect { x: 0, y: 0, width: src_w as i32, height: src_h as i32 };
        let dst_rect = NppiRect { x: 0, y: 0, width: dst_w as i32, height: dst_h as i32 };

        // nStep is in BYTES. height_stride counts f32 elements. Convert.
        let src_step_bytes = (self.layout.height_stride * std::mem::size_of::<f32>()) as i32;
        let dst_step_bytes = (dst.layout.height_stride * std::mem::size_of::<f32>()) as i32;

        // Raw pointer offset via DevicePtr/DevicePtrMut (handles sub-images).
        let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
        let src_ptr = (src_base + self.layout.img_index as u64) as *const f32;
        let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
        let dst_ptr = (dst_base + dst.layout.img_index as u64) as *mut f32;

        let status = unsafe {
            nppiResize_32f_C3R(
                src_ptr as *const u8,
                src_step_bytes,
                src_size,
                src_rect,
                dst_ptr as *mut u8,
                dst_step_bytes,
                dst_size,
                dst_rect,
                interpolation_mode(inter),
            )
        };
        check_status(status)
    }
}
```

### Step 9: Update `npp/Cargo.toml`, `npp/src/cuda.rs`, and `npp/src/lib.rs`

**`/home/vansweej/Work/npp-rs/npp/Cargo.toml`** — replace [dependencies] section:
- Remove `image = "0.23.13"`, `rustacuda = "0.1"`, `rustacuda_core = "0.1"`, `rustacuda_derive = "0.1"`.
- Add `cudarc = { version = "0.9", default-features = false, features = ["driver", "std"] }`.
- Add `thiserror = "2"`.
- Optionally add `half = "2"` if you included `half::f16` in the `NppPixelType` alphabet (Step 3).
- Under `[dev-dependencies]`: keep `pretty_assertions = "1.0.0"`; remove `criterion` and `cuda-runtime-sys`.
- Add `autobenches = false` to `[package]` (benches are deferred).
- Remove the `[[bench]]` block entirely.
- Add:
  ```toml
  [features]
  default = []
  # Enable to run GPU-dependent tests (manual gate — no GPU lane in CI).
  gpu = []
  ```

**`/home/vansweej/Work/npp-rs/npp/src/cuda.rs`** — replace entirely:
```rust
//! CUDA device initialization using `cudarc`.

use cudarc::driver::CudaDevice;
use std::sync::Arc;

use crate::error::NppError;

/// Initialize a CUDA device at the given ordinal and return a shared handle.
///
/// # Context-lifetime invariant (C7)
///
/// The returned `Arc<CudaDevice>` must be kept alive for the duration of any
/// `CudaImage` created from it. Dropping the device while buffers are live
/// results in `cuMemFree` against a destroyed context. cudarc's internal
/// `Arc<CudaDevice>` reference on every `CudaSlice` prevents this for the
/// common case.
pub fn initialize_cuda_device(ordinal: usize) -> Result<Arc<CudaDevice>, NppError> {
    let dev = CudaDevice::new(ordinal)?;
    Ok(dev)
}

/// Convenience wrapper: initialize device 0 (the default GPU).
pub fn default_cuda_device() -> Result<Arc<CudaDevice>, NppError> {
    initialize_cuda_device(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(feature = "gpu"), ignore)]
    #[test]
    fn test_cuda_initialize() {
        let result = default_cuda_device();
        assert!(result.is_ok());
    }
}
```

**`/home/vansweej/Work/npp-rs/npp/src/lib.rs`** — replace entirely:
```rust
//! Safe Rust bindings for NVIDIA NPP image operations.
//!
//! Built on:
//! - `cudarc` for CUDA device management (`Arc<CudaDevice>`, `CudaSlice<T>`)
//! - `npp-sys` for generated FFI bindings to the NPP image domain (`nppi*` symbols)
//!
//! The core type is [`CudaImage<T>`], where `T: NppPixelType` covers the full NPP
//! primitive alphabet (~9 types). Operation capability is expressed via traits
//! (e.g. [`Resize`], [`SwapChannels`]): an unsupported `(type, op)` pair simply
//! has no trait impl, making it a compile-time error.
//!
//! Round-trip to host memory uses `TryFrom<&CudaImage<T>> for Vec<T>`.
//! There is no `image` crate dependency in the core.
//!
//! # Test tiers
//!
//! - `cargo test` — pure-logic unit tests (no GPU required, no device init).
//! - `cargo test --features gpu` — device-dependent integration tests (requires
//!   an NVIDIA GPU + driver). This is a manual gate; CI has no GPU lane.
//!- CudaError -> CUDA driver error
//! # Safety
//!
//! The [`CudaImage`] constructor requires an `Arc<CudaDevice>` handle; the device
//! must outlive all images created from it (C7). Raw-pointer extraction at the
//! FFI boundary follows the pattern documented in `docs/spike-cudarc-ptr-bridge.md`.

#![deny(missing_docs)]

pub mod cuda;
pub mod error;
pub mod image;
pub mod imageops;
pub mod layout;
pub mod resize_ops;
pub mod swap_channel_ops;

#[cfg(test)]
mod raw_tests;
```

### Step 10: Exclude benchmarks from M1 build

The files in `/home/vansweej/Work/npp-rs/npp/benches/` use `rustacuda`, `image`,
and `cuda-runtime-sys` which are no longer available. They are deferred to a
dedicated benchmarking session.

Since `npp/Cargo.toml` no longer has `[[bench]]` and has `autobenches = false`
(Step 9), cargo will not build them. Leave the bench source files in place
(do not delete) as reference for the future session.

Add a brief note to `docs/roadmap.md` below the F6 entry:
```
## F6.1 — Benchmark port
The five `npp/benches/*.rs` files from the original crate use `rustacuda`,
`image-rs`, and `cuda-runtime-sys` — none of which exist in the M1 dependency
set. They are parked (not built) and must be reimplemented as full benchmarks
asserting both timing and output content. Depends on M1's new API.
```

### Step 11: Add the single golden-image correctness test

Create `/home/vansweej/Work/npp-rs/npp/tests/golden_resize.rs`:

```rust
//! Golden-image correctness test for `Resize` on `CudaImage<u8>`.
//!
//! This is the **one** M1 test that proves the cudarc port produces correct
//! pixels, not just correct geometry (C12 minimum mitigation).
//!
//! # Manual procedure to pin the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_resize`
//! 2. The test will print the captured output and panic ("golden reference not yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below (or commit the binary file).
//! 4. Re-run to confirm the assertion passes.
//!
//! Uses `NearestNeighbor` interpolation so expected output is bit-exact across
//! NPP versions (no floating-point rounding variance).

#![cfg(feature = "gpu")]

use npp_rs::error::NppError;
use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::cuda::default_cuda_device;
use std::convert::TryFrom;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

const SRC_W: u32 = 12;
const SRC_H: u32 = 8;
const DST_W: u32 = 6;
const DST_H: u32 = 4;

/// Input: procedurally generated 3-channel u8 gradient (12x8).
fn make_input() -> Vec<u8> {
    let mut data = Vec::with_capacity((SRC_W * SRC_H * 3) as usize);
    for y in 0..SRC_H {
        for x in 0..SRC_W {
            data.push((x * 21) as u8);  // R: x-gradient
            data.push((y * 32) as u8);  // G: y-gradient
            data.push(128);             // B: constant
        }
    }
    data
}

/// Golden output for NearestNeighbor 12x8 → 6x4.
/// Generated by running this test on a known-good setup and committing the result.
/// Replace with actual captured bytes on first GPU run.
const EXPECTED: &[u8] = &[
    // TODO: pin on first GPU run — 6*4*3 = 72 bytes
    // Temporary placeholder — this will fail:
    0; 72
];

#[test]
fn test_golden_resize_u8_nn() {
    let device: Arc<CudaDevice> = default_cuda_device().expect("CUDA device init");
    let src = CudaImage::from_host(
        device.clone(),
        3, SRC_W, SRC_H,
        &make_input(),
    ).expect("src allocation");

    let mut dst = CudaImage::<u8>::new(
        device.clone(),
        3, DST_W, DST_H,
    ).expect("dst allocation");

    src.resize(&mut dst, ResizeInterpolation::NearestNeighbor)
        .expect("resize");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    if EXPECTED.len() != (DST_W * DST_H * 3) as usize || EXPECTED.iter().all(|&b| b == 0) {
        // Golden reference not yet pinned
        eprintln!("=== Golden reference NOT pinned ===");
        eprintln!("Captured output ({} bytes): {:?}", output.len(), output);
        panic!("golden reference not yet pinned — commit the captured bytes");
    }

    assert_eq!(output, EXPECTED, "pixel mismatch in NearestNeighbor resize");
}
```

### Step 12: Verify Phase 2 builds

```bash
nix develop . --command cargo build
```
Must compile the entire workspace with zero `rustacuda`/`image` references remaining.
Also verify `cargo build --features gpu` compiles (it adds no deps — just the feature flag).

**Do not run tests yet** (Phase 3 gates the device tests; the golden test in `tests/`
is already `#![cfg(feature = "gpu")]` and won't run under plain `cargo test`).

---

## Phase 3: Test tiering with the `gpu` feature gate

Separate GPU-dependent tests from pure-logic tests. After this phase,
`cargo test` runs only pure-logic tests; `cargo test --features gpu` also
runs device tests.

Commit message: `test: add gpu feature gate and annotate device-dependent tests`

### Step 1: Add `gpu` feature to `npp/Cargo.toml`

Already done in Phase 2 Step 9 (part of the `[features]` section). Verify it exists:
```toml
[features]
default = []
gpu = []
```

### Step 2: Gate every device-touching unit test

Apply `#[cfg_attr(not(feature = "gpu"), ignore)]` to each `#[test]` that
initializes a device or allocates a `CudaImage`. Files/positions to update:

- `/home/vansweej/Work/npp-rs/npp/src/cuda.rs` — line with `fn test_cuda_initialize`:
  add `#[cfg_attr(not(feature = "gpu"), ignore)]` above `#[test]`.
- `/home/vansweej/Work/npp-rs/npp/src/image.rs` — each surviving unit test
  (rewritten test suite: `test_new`, `test_sub_image`, `test_get_index`,
  `test_get_start_point`; remove tests that depended on `image` crate or
  `RgbImage`/`Persistable`). Add the gate attribute to each.
- `/home/vansweej/Work/npp-rs/npp/src/resize_ops.rs` — each `#[test]` that calls
  resize. Add the attribute.
- `/home/vansweej/Work/npp-rs/npp/src/swap_channel_ops.rs` — `test_bgra_to_rgb`.
- `/home/vansweej/Work/npp-rs/npp/src/raw_tests.rs` — `test_allocations`.

Pure-logic tests (e.g. `layout.rs::test_row_major_packed`, `error.rs` unit tests
if any) must **not** be gated.

### Step 3: Verify `raw_tests` module is single-gated

In `/home/vansweej/Work/npp-rs/npp/src/lib.rs`, confirm `mod raw_tests;` is
already gated with `#[cfg(test)]` (done in Phase 2 Step 9). Inside
`raw_tests.rs`, there is an inner `#[cfg(test)] mod tests { ... }`. This is
redundant but harmless; leave it in place — the outer `#[cfg(test)]` ensures
the module is never compiled in non-test builds.

### Step 4: Verify Phase 3

```bash
nix develop . --command cargo test --no-default-features
```
Must pass, running only pure-logic tests. Must NOT try to initialize a CUDA
device (importing `default_cuda_device` is fine as long as it's not called).

```bash
nix develop . --command cargo test --features gpu 2>/dev/null || echo "(non-GPU host expected)"
```
On a non-GPU host, the device-init tests will report "ignored" and the two
compile-only build steps (`cargo build --features gpu`) must pass. On a GPU
host, the suite runs; the golden test will require the manual pinning step
(Phase 2 Step 11).

---

## Phase 4: Documentation verification + narrative guides

All `///` and `//!` doc comments were baked into the source code during
Phase 2 (`error.rs`, `image.rs`, `imageops.rs`, `resize_ops.rs`,
`swap_channel_ops.rs`, `cuda.rs`, `lib.rs`, `layout.rs`). This phase adds the
two remaining doc targets (the npp-sys crate-level header and narrative docs/)
and then **verifies** that `#[deny(missing_docs)]` passes.

Commit message: `docs: add npp-sys crate-level docs and docs/ narrative guides`

### Step 1: Add crate-level `//!` doc to `npp-sys/src/lib.rs`

Edit `/home/vansweej/Work/npp-rs/npp-sys/src/lib.rs`. Insert above the `#![allow(...)]` lines:

```rust
//! Generated FFI bindings to the NVIDIA NPP image-processing domain.
//!
//! Auto-generated by `bindgen` from `wrapper.h`. Scoped to `nppi*` symbols only
//! (NPP image domain; `npps*` signal symbols are excluded).
//! The bindings are generated into `OUT_DIR` and are never committed.
//!
//! # Safety
//!
//! These are raw `extern "C"` bindings. Use the `npp-rs` crate for safe wrappers.
```

### Step 2: Write the `docs/` narrative guides

Create three files under `/home/vansweej/Work/npp-rs/docs/`:

**`docs/getting-started.md`:**
- Prerequisites: Nix flakes, NVIDIA driver + GPU, this repo cloned.
- Build: `nix develop . --command cargo build`
- Test tiers:
  - Unit: `nix develop . --command cargo test --no-default-features` (no GPU needed)
  - GPU: `nix develop . --command cargo test --features gpu` (requires GPU — manual gate)
  - Golden test: see `npp/tests/golden_resize.rs` for the 2-step manual pinning procedure.
- Lint: `nix develop . --command cargo clippy -- -D warnings`
  and `nix develop . --command cargo fmt --check`
- Doc generation: `nix develop . --command cargo doc --no-deps -p npp-rs -p npp-sys`

**`docs/architecture.md`:**
- Two-crate workspace: `npp-sys` (bindgen FFI) and `npp` (safe wrapper, crate `npp-rs`).
- Device management via `cudarc`: `Arc<CudaDevice>` → `CudaSlice<T>` + borrowed views.
- The `NppPixelType` alphabet and capability-trait model.
- `Vec<T>` round-trip (no `image` crate in core).
- Pointer bridge: cudarc `as_device_ptr*` → raw NPP `extern "C"` call (refer to
  `docs/spike-cudarc-ptr-bridge.md` for the exact pattern).
- Context-lifetime invariant (C7): device outlives images.
- Deferred boundaries (image-rs, graphynx) and roadmap (link `docs/roadmap.md`).

**`docs/npp-bindings.md`:**
- The bindgen allowlist: `nppi.*` functions/types, `npps.*` excluded.
- Which NPP functions are wrapped in M1: `nppiResize_8u_C3R`, `nppiResize_32f_C3R`,
  `nppiSwapChannels_8u_C4C3R`.
- How to add a new NPP primitive: add bindgen allowlist entry, define a capability
  trait or extend an existing one, implement the trait for the concrete type(s),
  follow the pointer-bridge pattern from `docs/spike-cudarc-ptr-bridge.md`.
- Test requirement: every new FFI call needs at minimum a geometry assertion test
  and, for correctness-critical paths, a golden-image test (see F6 in roadmap).

### Step 4: Verify docs

```bash
nix develop . --command cargo doc --no-deps -p npp-rs -p npp-sys
```
Must build with no broken intra-doc links and no `missing_docs` warnings.
(The `#![deny(missing_docs)]` in `npp/src/lib.rs` makes this a compile error.)

---

## Phase 5: README, CI workflow, and cleanup

Commit message: `docs: rewrite README, add Nix-based CI workflow, and remove obsolete install script`

### Step 1: Rewrite `/home/vansweej/Work/npp-rs/README.md`

Replace the entire file with Nix-first, cudarc-based, Linux-only README:

```markdown
# npp-rs
![build](https://github.com/vansweej/npp-rs/actions/workflows/build.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/npp-rs)](https://crates.io/crates/npp-rs)

Safe Rust bindings for NVIDIA NPP **image** operations (not signal ops).

Built on `cudarc` for CUDA device management and `npp-sys` (bindgen) for NPP FFI.
Features a `NppPixelType` alphabet (~9 types) with capability-trait dispatch:
unsupported `(type, op)` pairs are compile-time errors. Host round-trip via
`Vec<T>` — no `image` crate in the core.

## Dependencies

- Nix with flakes (enter the shell: `nix develop .`)
- An NVIDIA GPU with the proprietary driver installed
- CUDA toolkit + NPP libraries — provided by the Nix dev shell

## Quick start

```bash
# Enter the dev shell
nix develop .

# Build all crates
cargo build

# Run unit tests (no GPU required)
cargo test --no-default-features

# Run GPU-dependent tests (requires hardware)
cargo test --features gpu

# Lint
cargo clippy -- -D warnings
cargo fmt --check

# Documentation
cargo doc --no-deps -p npp-rs -p npp-sys
```

## Project structure

| Directory | Cargo package | Purpose                                    |
|-----------|---------------|--------------------------------------------|
| `npp-sys/` | `npp-sys`     | Bindgen FFI to NPP image domain (`nppi*`)  |
| `npp/`     | `npp-rs`      | Safe wrapper with `CudaImage`, traits, ops |

## License

MIT

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the post-M1 plan (macro codegen,
image-rs boundary, signal ops, IPP bindings, full golden test suite, etc.).
```

### Step 2: Create `/home/vansweej/Work/npp-rs/.github/workflows/build.yml`

Create the file (it does not exist yet — this is not a rewrite):

```yaml
name: Build

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always

jobs:
  fmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v27
      - run: nix develop . --command cargo fmt --check

  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v27
      - run: nix develop . --command cargo clippy -- -D warnings

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v27
      - run: nix develop . --command cargo build

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v27
      - run: nix develop . --command cargo test --no-default-features
```

Note: no GPU lane. The GPU suite (`--features gpu`) is a documented manual gate.
On GPU-less CI runners it would fail and that's correct behavior.

### Step 3: Remove `ci/install_server.sh`

Delete `/home/vansweej/Work/npp-rs/ci/install_server.sh` and the `ci/` directory
(which contains only that script). The Nix dev shell replaces the dpkg bootstrap.

### Step 4: Add `.envrc` and bump crate versions

Create `/home/vansweej/Work/npp-rs/.envrc` with exactly:
```
use flake
```
This lets `direnv` users auto-enter the Nix dev shell on `cd`.

In `/home/vansweej/Work/npp-rs/npp/Cargo.toml`: bump `version = "0.1.0-dev"`.
In `/home/vansweej/Work/npp-rs/npp-sys/Cargo.toml`: bump `version = "0.1.0-dev"`.
(This signals the breaking cudarc port per semver — no external dependents exist,
but the version change records the break.)

### Step 5: Verify Phase 5

```bash
nix develop . --command cargo build          # still compiles
nix develop . --command cargo fmt --check    # no formatting issues
nix develop . --command cargo clippy -- -D warnings  # clean
```
And verify README renders correctly (if you have a markdown viewer).

---

## Definition of Done (Milestone 1)

1. `nix develop --command cargo build` succeeds against `cudaPackages.libnpp` on
   pinned `nixos-unstable`.
2. `cargo test --no-default-features` passes, running zero GPU code.
3. `cargo test --features gpu` compiles (runs on a GPU host, manual).
4. `cargo clippy -- -D warnings` and `cargo fmt --check` are clean.
5. `cargo doc --no-deps -p npp-rs -p npp-sys` builds; every `pub` item documented;
   `docs/` populated.
6. README + CI workflow written; `ci/install_server.sh` removed; no Windows paths.
7. No `rustacuda*` anywhere; `npp-sys` consumed via local path; generated bindings
   scoped to NPP image domain (`nppi*` symbols).
8. `CudaImage<T>` exists with the full `NppPixelType` alphabet (all 9 types
   constructible), `Vec<T>` round-trip, and the `Resize`/`SwapChannels` trait
   architecture.
9. Hand-written `Resize` impls for `u8` (over `nppiResize_8u_C3R`) and `f32`
   (over `nppiResize_32f_C3R`) exist.
10. One golden-image correctness test for `u8` NearestNeighbor resize exists
   (with a documented 2-step manual pinning procedure).
