# Feature: Milestone 1 — Build again on latest CUDA via Nix

> **MANDATORY READING for every thinking agent (planner, sparrer, implementer):**
> Read `reviews/final-report.md` IN FULL before working any phase. This plan
> cites findings by ID (C1, C2, …, CT1, …) but the report holds the evidence,
> the cross-panel reasoning, and the Round-2 retractions that changed severities
> (CT2, CT6). Acting on these task summaries instead of the source will produce
> wrong decisions. A per-phase findings map is in the appendix.

Goal: restore buildability on the latest nixpkgs CUDA, establish a reproducible
Nix dev shell, full public-API rustdoc + `docs/`, rewritten README + CI workflow,
and a unit/GPU test split (GPU tests manual-only). Deeper code-soundness hardening
is deferred to Milestone 2.

**Reference implementation (same author):** `github.com/vansweej/graphynx`.
Appendix A inlines its `flake.nix`, `rust-toolchain.toml`, and `backends-cuda/build.rs`
verbatim for agents without repo access.

**Resolved decisions (not to re-litigate):**

| Decision | Resolution |
|----------|-----------|
| CUDA major | Latest in nixpkgs — `nixos-unstable`, `pkgs.cudaPackages` |
| CUDA crate | `cudarc = { version = "0.9", default-features = false, features = ["driver","std"] }` — replaces `rustacuda*` |
| GPU tests | Manual only — feature-gate behind `gpu`; plain `cargo test` skips them |
| NPP linking | Shared (dynamic) — remove all `static=` names |
| Platform | Linux only — delete all Windows paths and README prose |
| NPP nixpkgs attr | `cudaPackages.libnpp` — confirm on pinned `nixos-unstable` |
| Binding philosophy | Safe, idiomatic Rust. Rewrite (not faithfully port) most existing code; treat current files as behavioral reference |
| Future scope (context) | IPP bindings as siblings later; NPP image ops (`nppi*`) only, signal (`npps*`) deferred; image-rs/imageproc integration shapes type design |

---

## Phase 1: Nix dev shell + build script rewrite

The Nix dev shell provides the CUDA toolkit, NPP libraries, libclang for bindgen,
and Rust toolchain in a pinned, reproducible environment. The `build.rs` rewrite
consumes these Nix-provided paths and switches from static to shared NPP linking.

Commit message: `feat: add Nix dev shell and rewrite npp-sys build for shared NPP linking`

### Step 1: Create `flake.nix` at workspace root

Model on `graphynx/flake.nix` (Appendix A.1) with these deltas:

- **buildInputs changes:**
  - ADD `cudaPackages.libnpp` (provides NPP shared libs + headers)
  - ADD `llvmPackages.libclang` or `clang` (bindgen needs libclang to parse `npp.h`)
  - REMOVE `cudaPackages.cuda_nvrtc` / `NVRTC_PATH` unless a transitive dep needs it
    (npp-rs has no PTX kernels; nvrtc is optional overhead)
  - REMOVE `cudaPackages.cuda_nvcc`, `gcc13`, `NVCC_HOST_COMPILER` unless
    bindgen/clang choke on CUDA headers with the default gcc (verify during the
    spike; keep gcc13 only if needed)
  - REMOVE `cargo-tarpaulin`, `cargo-deny`, `cargo-outdated` etc. unless desired
    (crate-specific; graphynx's extras are opinionated)
- **shellHook additions:**
  - Set `LIBCLANG_PATH` to the libclang from `buildInputs`
  - Set `BINDGEN_EXTRA_CLANG_ARGS` pointing at `cuda_cudart/include` and
    `libnpp/include` so bindgen finds NPP headers
  - Set `CUDA_PATH` to `cudaPackages.cuda_cudart` (already done in graphynx)
- **Keep verbatim from graphynx:** `nixos-unstable` pin, `rust-overlay`, `allowUnfree`,
  `.nvidia-libs/` driver-symlink rpath trick, `NIX_LDFLAGS` scrub.
- **Config:** `config.allowUnfree = true` (CUDA requires it).

### Step 2: Create `rust-toolchain.toml` at workspace root

Pin an exact stable channel. Copy graphynx's `rust-toolchain.toml` (Appendix A.3)
verbatim, or update to a newer version if desired. The toolchain version must be
compatible with `cudarc 0.9` and `bindgen 0.58+`.

```toml
[toolchain]
channel = "1.94.1"
```

### Step 3: Rewrite `npp-sys/build.rs`

Replace the entire file to match these requirements:

- **Linux only** — delete all `#[cfg(target_os = "windows")]` branches
  (functions `cuda_include_path` win variant at line 21-29, `cuda_configuration`
  win variant at line 31-39, `cuda_link_libs` win variant at line 56-69).
- **Nix-provided paths** — read include/lib directories from env var `CUDA_PATH`
  (set by the flake shellHook) instead of hardcoded `/usr/local/cuda` (line 7).
  Keep the `option_env!("CUDA_INSTALL_DIR")` fallback for non-Nix local builds.
- **Shared (dynamic) linking** — replace `rustc-link-lib=static=*_static` with
  `rustc-link-lib=shared=nppc`, `shared=nppial`, `shared=nppicc`, etc.
  Remove `shared=cudart_static` → `shared=cudart`.
  Remove `shared=culibos` and `shared=stdc++` (not needed with shared libs).
- **Bindgen allowlist** — add `.allowlist_function("nppi.*")` and
  `.allowlist_type("Nppi.*")` and `.allowlist_var("Nppi.*")` to the
  `bindgen::Builder` so the generated surface is bounded to the NPP image domain
  (`nppi*` symbols only; `npps*` signal symbols excluded).
- **Bindgen config** — keep `.generate_comments(false)` (the report found generated
  doc comments wasteful). Add `.clang_args(&["-I", &npp_include_path()])` where
  `npp_include_path()` reads from `CUDA_PATH/include` like `cuda_include_path`.
- **Wrapper header** (`npp-sys/wrapper.h`) stays the same (includes `nppi.h`,
  comments out `npps.h`). No changes needed.

### Step 4: Update `npp/Cargo.toml` — repoint sys dependency

- Change line 14 from `npp-sys = "0.0.1"` (crates.io) to
  `npp-sys = { path = "../npp-sys" }`. Keep the crates.io version as a comment
  above for release workflow.
- Keep `npp-sys = "0.0.1"` as a commented-out fallback.

### Step 5: Update `npp-sys/Cargo.toml` — update metadata

- Remove `build = "build.rs"` if it exists (not needed with 2021 edition; harmless,
  but clean up).
- Drop `categories = ["external-ffi-bindings", "computer-vision"]` if desired.
  Keep the description accurate.

### Step 6: Verify Phase 1

Run inside the Nix dev shell:
```bash
nix develop . --command cargo build -p npp-sys
```

This must succeed — `npp-sys` compiles against the local CUDA + NPP libraries.
The `npp` crate (safe wrapper) will NOT compile yet (still depends on `rustacuda`
from crates.io).

---

## Phase 2: Replace rustacuda with cudarc and port CUDA integration

Remove all `rustacuda*` dependencies, introduce `cudarc`, define the new error
type, and port every site that touches CUDA device buffers/contexts. This is the
core code rewrite. The open boundary decision for C11/C3/C4 (seal generic `T` /
redesign ownership model) must be settled by the sparrer with the author BEFORE
Step 5 of this phase.

Commit message: `feat: replace rustacuda with cudarc 0.9, introduce NppError, and port device integration`

### Step 1: Update `npp/Cargo.toml` — swap CUDA deps and add thiserror

- REMOVE (lines 16-18): `rustacuda`, `rustacuda_core`, `rustacuda_derive`
- ADD: `cudarc = { version = "0.9", default-features = false, features = ["driver", "std"] }`
- ADD: `thiserror = "2"` (for the new error type)
- UPDATE dev-dep `cuda-runtime-sys` (line 23): remove if nothing uses it;
  keep only if cudarc transitively needs it (test-compile to verify).
- ADD: `[features]` section with `gpu = ["cudarc/driver"]` (will be populated
  in Phase 3; add the skeleton now so Cargo.toml is consistent).
- If the `image` crate is upgraded (optional, not required): bump `image` from
  `0.23.13` to a modern version. Otherwise leave pinned.

### Step 2: Create `npp/src/error.rs` — NppError enum

- Define with `thiserror`:
  ```rust
  #[derive(Error, Debug)]
  pub enum NppError {
      #[error("NPP returned status {0}")]
      Npp(NppStatus),  // wrap the raw i32 from npp-sys

      #[error("CUDA error: {0}")]
      Cuda(#[from] cudarc::driver::result::CudaResultError),  // or wherever cudarc's error lives

      #[error("Image error: {0}")]
      Image(#[from] image::ImageError),

      #[error("Invalid argument: {0}")]
      InvalidArgument(String),
  }
  ```
- `NppStatus` — type alias or newtype for the raw `i32` from `npp_sys`. DO NOT
  enumerate all NPP status codes (CUDA-version-specific per CT4). Just wrap the
  `i32` raw value.
- Add a helper `fn check_status(status: i32) -> Result<(), NppError>` that:
  - Returns `Ok(())` if `status >= 0` (fixes the C1/NEW-01 bug: positive NPP
    warning codes are NOT errors — the old `status == 0` check treated them as
    hard failures).
  - Returns `Err(NppError::Npp(status))` if `status < 0`.
- The `InvalidArgument` variant is for validation errors that will replace
  `debug_assert!` (C2 — planned for M2, but include the variant now for forward
  compatibility).
- Export `NppError` from `npp/src/lib.rs` as `pub mod error;`.

### Step 3: Port `npp/src/cuda.rs` from rustacuda to cudarc

Replace the entire module:

- **Before** (old):
  ```rust
  pub fn initialize_cuda_device() -> Result<Context, CudaError> {
      rustacuda::init(rustacuda::CudaFlags::empty())?;
      let device = Device::get_device(0)?;
      Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
  }
  ```
- **After** (new):
  ```rust
  use cudarc::driver::CudaDevice;
  use std::sync::Arc;
  use crate::error::NppError;

  /// Initialize a CUDA device and return a handle.
  /// The returned `Arc<CudaDevice>` must be kept alive for the duration of
  /// any `CudaImage` that was created from it; dropping the device while
  /// buffers are live results in `cuMemFree` against a destroyed context.
  pub fn initialize_cuda_device(ordinal: usize) -> Result<Arc<CudaDevice>, NppError> {
      let dev = CudaDevice::new(ordinal)?;
      Ok(dev)
  }

  /// Helper: return the global default device, equivalent to the old
  /// hardcoded `Device::get_device(0)`.
  pub fn default_cuda_device() -> Result<Arc<CudaDevice>, NppError> {
      initialize_cuda_device(0)
  }
  ```
- The old `#[cfg(test)] mod tests` test `test_cuda_initialize` must be
  feature-gated (Phase 3). For now, just gate it with `#[cfg(feature = "gpu")]`.
- Add `// SAFETY:` note to the context-lifetime invariant (report C7) so it's
  documented at the source.
- Update `npp/src/lib.rs` accordingly — `pub mod cuda;` stays; add `pub mod error;`.

### Step 4: SPIKE — confirm cudarc CudaSlice → raw device pointer bridge for NPP FFI

This is the one non-graphynx-adaptable integration point and must be resolved
BEFORE Step 5. Do not proceed to Step 5 until this is settled.

**Task:** Write a compile-check or small test program (outside the crate or in
a scratch file) that:

1. Allocates a `CudaSlice<u8>` on device via cudarc.
2. Extracts a raw `*const u8` (for src) and `*mut u8` (for dst) at a given byte
   offset (simulating the `img_index` offset used in `sub_image`).
3. Confirms the resulting raw pointer can be passed to an `extern "C"` function
   (like `nppiResize_8u_C3R`) without the `CudaSlice` being dropped mid-call.
4. Documents the exact API used (likely `CudaDevice::alloc::<u8>` → `CudaSlice<u8>`,
   then via `cudarc::driver::DevicePtr` or `DevicePtrMut` traits to get the
   underlying `CUdeviceptr`, then cast to `*const u8`/`*mut u8`).

**Result:** a short markdown note (`docs/spike-cudarc-ptr-bridge.md` or appended
to this plan) that records the confirmed API + any lifetime/keep-alive requirements,
so every call site follows the same pattern.

**Implementation note:** The offset arithmetic must replace the old pattern:
```rust
// OLD (rustacuda):
src.image_buf.borrow_mut().as_device_ptr().offset(img_index as isize).as_raw()
```
with something like:
```rust
// NEW (cudarc — exact API TBD from the spike):
let base: CUdeviceptr = device_ptr(&src.image_buf);
(base + img_index as u64) as *const u8
```

### Step 5: Port `npp/src/image.rs` — CudaImage buffer, ownership, and TryFrom impls

**BEFORE THIS STEP:** the sparrer must have resolved the open boundary decision
(C11 generic `T` / C3/C4 ownership redistribution) with the author. The scope of
this step changes depending on the outcome:

- **If C11/C3/C4 are pulled into M1:** redesign `CudaImage` to remove the generic
  `T` (seal to `u8` or define an `NppPixelType` marker), replace `Rc<RefCell<...>>`
  with a cudarc-compatible ownership model (`Arc<CudaSlice<u8>>` or owned `CudaSlice
  + view`), and remove the `RefCell` entirely. Honor the image-rs type-design
  constraints (see appendix B of this plan).
- **If C11/C3/C4 are deferred to M2:** preserve the generic `T` and `Rc<RefCell<...>>`
  shape as closely as possible, but replace `DeviceBuffer` internals with cudarc's
  `CudaSlice<u8>`. This is a mechanical port, not a redesign.

**Regardless of the decision, the following changes are always required:**

- Replace `use rustacuda::memory::*` with cudarc equivalents.
- Replace `DeviceBuffer::zeroed(img_size_bytes)?` with `device.alloc::<u8>(img_size_bytes)?`
  (or whichever cudarc API the spike confirms).
- Replace `DeviceBuffer::from_slice(...)` with cudarc `htod_copy_sync(...)` or
  `CudaDevice::htod_sync(...)`.
- Replace the read-back loop (lines 160-181, `CudaImage<u8>` → `RgbImage TryFrom`) —
  use cudarc `dtod_copy_sync` or `device.dtoh_sync(...)` for each row. Keep the
  stride-correct row-by-row loop semantics. Do NOT fix the C5 `set_len` pattern here
  (that's M2); preserve it as-is.
- Replace `Rc<RefCell<...>>` internal type. If C3/C4 are deferred, use a simple
  `pub(in crate) CudaSlice<u8>` owned directly by `CudaImage` (which makes it
  `!Clone` — sub_image will need a different aliasing strategy; document the
  limitation). If C3/C4 are in M1, use the model the sparrer agrees.

### Step 6: Port `npp/src/resize_ops.rs` — raw pointer extraction and NPP call

- Replace `use rustacuda::error::*` → `use crate::error::NppError`.
- Replace the two raw-pointer extraction blocks (lines 61-74):
  - Remove `borrow_mut()` and `as_device_ptr().offset().as_raw()`.
  - Use the spike-confirmed cudarc raw-pointer bridge.
- Replace the `status == 0` check with `check_status(status)?` from error.rs.
- Update function signature's return type: `Result<(), CudaError>` → `Result<(), NppError>`.

### Step 7: Port `npp/src/swap_channel_ops.rs` — raw pointer extraction and NPP call

- Same pattern as Step 6 but for `nppiSwapChannels_8u_C4C3R`.
- Replace rustacuda pointer extraction + status check with cudarc bridge + `check_status`.
- Update return type to `Result<(), NppError>`.
- Update the trait bound in `npp/src/imageops.rs` (`SwapChannels<T>`) to use
  `NppError` instead of `CudaError`.

### Step 8: Port `npp/src/imageops.rs` — trait error type

- Change the `SwapChannels` trait signature from `Result<(), CudaError>` to
  `Result<(), NppError>`.
- Ensure all trait impls match the new return type.

### Step 9: Verify Phase 2 compiles

```bash
nix develop . --command cargo build -p npp-rs
```

This should now compile the entire workspace. It may produce clippy/fmt warnings —
those are addressed in later phases. A successful `cargo build` is the only gate.

---

## Phase 3: Test tiering

Separate GPU-dependent tests from pure unit tests using a feature gate. After this
phase, `cargo test` runs only logic tests; `cargo test --features gpu` runs the
device integration suite.

Commit message: `test: add gpu feature gate and annotate device-dependent tests`

### Step 1: Add `gpu` feature to `npp/Cargo.toml`

Add or complete the `[features]` section:
```toml
[features]
default = []
gpu = []
```

(The `gpu` feature does NOT depend on `cudarc/driver` — cudarc is always linked
because it's a required dependency, not optional. The feature only gates test
execution.)

### Step 2: Feature-gate all GPU-dependent tests

Every test that calls `initialize_cuda_device()` or touches a CUDA device must
be gated. Apply `#[cfg_attr(not(feature = "gpu"), ignore)]` to each `#[test]`
function:

- `npp/src/cuda.rs`: `test_cuda_initialize`
- `npp/src/image.rs`: `test_new`, `test_try_from_dynamic_image`,
  `test_try_from_cudaimage_image`, `test_get_index`, `test_in_bounds`,
  `test_sub_image`, `test_sub_image2`, `test_get_start_point`
- `npp/src/resize_ops.rs`: `test_resize1`, `test_resize2`, `test_resize3`,
  `test_resize4`
- `npp/src/swap_channel_ops.rs`: `test_bgra_to_rgb`
- `npp/src/raw_tests.rs`: `test_allocations`

Alternatively, move all device tests into `npp/tests/` (integration test
directory) and feature-gate the entire module. The `#[cfg_attr(not(feature = "gpu"), ignore)]`
approach is simpler and keeps tests close to the code they test.

### Step 3: Gate `mod raw_tests` in `npp/src/lib.rs`

Change line 5 from `mod raw_tests;` to:
```rust
#[cfg(test)]
mod raw_tests;
```

This already happens to be the case (the module itself is `#[cfg(test)]`), but
confirm it's not `pub` and cannot be reached from non-test code. In the original
code, `raw_tests.rs:1` has `#[cfg(test)] mod tests { ... }` — the module is doubly
gated, which is redundant. Consolidate to a single gate at the `mod` declaration.

### Step 4: Verify Phase 3

```bash
nix develop . --command cargo test --no-default-features
```
- Must pass with zero tests run or only pure-logic tests (layout.rs).
- Must NOT try to initialize a CUDA device.

```bash
nix develop . --command cargo test --features gpu
```
- Must compile. On a machine without a GPU, all GPU tests will fail with
  a device-init error (acceptable — manual run contract).

---

## Phase 4: Documentation

Rustdoc on every public item, crate-level docs, and a `docs/` directory with
narrative guides.

Commit message: `docs: add crate-level docs, rustdoc on all public items, and docs/ directory`

### Step 1: Create `docs/` directory with narrative guides

Mirror the structure graphynx uses. At minimum:

- **`docs/getting-started.md`**: Prerequisites (Nix + NVIDIA driver), build
  (`nix develop --command cargo build`), test (unit vs `--features gpu`),
  lint (`clippy`, `fmt`), generate docs (`cargo doc --no-deps`).
- **`docs/architecture.md`**: Overview of the workspace — `npp-sys` (bindgen FFI)
  and `npp` (safe wrapper). The cudarc device/ownership model. The NPP image-domain
  scope. The future IPP/signal/image-rs integration vision.
- **`docs/npp-bindings.md`**: How the bindgen allowlist works, which NPP functions
  are wrapped, how to add a new NPP primitive.
- Cross-link from README.

### Step 2: Add crate-level `//!` docs to `npp-sys/src/lib.rs`

Currently the file is a bare:
```rust
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

Add `//!` at the top describing this as generated FFI bindings to NVIDIA NPP
(image-processing domain), auto-generated by bindgen from `wrapper.h`. State
that the bindings are scoped to the NPP image domain (`nppi*` symbols only).

### Step 3: Add crate-level `//!` docs to `npp/src/lib.rs`

Currently:
```rust
pub mod cuda;
pub mod image;
pub mod imageops;
pub mod layout;
pub mod resize_ops;
pub mod swap_channel_ops;
```

Add `//!` describing the crate as safe Rust bindings for NVIDIA NPP image
operations, built on `cudarc` for device management and `npp-sys` for the NPP FFI
layer. State the scope (image operations only; signal ops deferred), the test
tiers (unit vs `--features gpu`), and the minimum CUDA version.

### Step 4: Add `///` docs to every public item

Inventory of items requiring documentation:

- **`npp/src/cuda.rs`:** `initialize_cuda_device(ordinal)` — document the
  context-lifetime invariant (the returned `Arc<CudaDevice>` must outlive all
  buffers created from it). `default_cuda_device()`.
- **`npp/src/error.rs`:** `NppError` enum — document each variant; `check_status`
  function.
- **`npp/src/image.rs`:**
  - `CudaImage` struct — document what it represents, its ownership model,
    thread-safety (or lack thereof).
  - Traits: `CopyFromImage`, `CopyToImage` (note: these are stubs — document
    as "reserved for future use"), `Persistable` (document the current temp-dir
    behavior truthfully as a known limitation per report C9).
  - Methods: `new`, `dimensions`, `width`, `height`, `channels`, `get_start_point`,
    `sub_image`.
  - `TryFrom<&RgbImage>`, `TryFrom<&RgbaImage>`, `TryFrom<&ImageBuffer<Bgra<u8>,Vec<u8>>>`,
    `TryFrom<&CudaImage<u8>> for RgbImage`.
  - `Persistable` impl on `CudaImage<u8>`.
- **`npp/src/layout.rs`:** `CudaLayout` struct (each field is already documented;
  add doc on the struct itself). `row_major_packed` — document the panic
  precondition. `From<SampleLayout>` impl.
- **`npp/src/imageops.rs`:** `SwapChannels` trait.
- **`npp/src/resize_ops.rs`:** `ResizeInterpolation` enum + each variant;
  `CudaImage<u8>::resize` — document the src/dst non-aliasing precondition
  (report C4).
- **`npp/src/swap_channel_ops.rs`:** `SwapChannels` impl `bgra_to_rgb`.

### Step 5: Generate and verify docs

```bash
nix develop . --command cargo doc --no-deps -p npp-rs -p npp-sys --open
```

Verify:
- No broken intra-doc links.
- Every `pub` item is documented (the Rust compiler warns on undocumented public
  items — check for `missing_docs` warnings).
- The `docs/` directory is populated with guides and can be navigated from
  README.md.

---

## Phase 5: README and CI

Rewrite the README for the Nix-first workflow, replace the CI workflow with a
Nix-based one, and clean up obsolete infrastructure.

Commit message: `docs: rewrite README, update CI workflow, remove obsolete install script`

### Step 1: Rewrite `README.md`

Use graphynx's README as the template (shows the right structure). Sections:

- **Title + badges** — fix the `build` badge URL (it points at `build.yml` that's
  about to be rewritten).
- **What is npp-rs?** — safe Rust bindings to NVIDIA NPP image operations. Mention
  the cudarc-based device management. State scope (image ops; signal to follow).
  Mention the future IPP and image-rs/imageproc vision briefly.
- **Dependencies** — Nix with flakes, NVIDIA GPU + driver, the chosen CUDA version.
- **Build** — single command: `nix develop --command cargo build`.
- **Test** — explain the two tiers: `nix develop --command cargo test` (unit, no GPU)
  and `nix develop --command cargo test --features gpu` (integration, requires GPU).
- **Lint** — `nix develop --command cargo clippy` and `nix develop --command cargo fmt --check`.
- **Documentation** — `nix develop --command cargo doc --no-deps` or link to `docs/`.
- **Project structure** — brief (workspace, npp-sys, npp).
- **License** — MIT.
- Drop all Windows instructions (D5).

### Step 2: Rewrite `.github/workflows/build.yml`

Completely replace the current file (19 lines, `ubuntu-18.04`, `cargo build` only):

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

Note: NO GPU lane. The GPU suite is a documented manual gate per D3. The
assumed `cachix/install-nix-action` version is placeholder; use the latest
available. If the CI runner has no NVIDIA driver, the Nix shell will enter but
`cargo test --features gpu` would fail — that's correct behavior (manual gate).

### Step 3: Delete `ci/install_server.sh`

The entire directory `ci/` (or just the script). The Nix dev shell replaces the
dpkg bootstrap for all CUDA/NPP dependencies. Keep `ci/` only if it contains
other utilities; otherwise delete the directory.

### Step 4: Clean up `npp/Cargo.toml` and `npp-sys/Cargo.toml` metadata

- Remove Windows from the description if it mentions cross-platform support.
- Update `authors`, `edition` if needed (2021 is already set — correct).
- Remove `readme = "README.md"` from the package if it's the workspace default
  (harmless, but clean).
- Bump `version` from `0.0.1` to `0.1.0-dev` or `0.0.2` to reflect the breaking
  nature of the cudarc port (per semver, replacing the CUDA crate is a breaking
  change for dependent crates if any exist — none do, but signal the change).

### Step 5: Add `.envrc` (optional but recommended for Nix convenience)

At workspace root, create:
```
use flake
```

This lets `direnv` users auto-enter the Nix shell on `cd`. Optional — the
`nix develop . --command` pattern is always available.

---

## Definition of Done (Milestone 1)

1. `nix develop --command cargo build` succeeds against `cudaPackages.libnpp` on
   pinned `nixos-unstable`.
2. `cargo test` (no features) passes and runs zero GPU code.
3. `cargo test --features gpu` compiles (runs on a GPU host, manual).
4. `cargo clippy -D warnings` and `cargo fmt --check` clean.
5. `cargo doc --no-deps` builds; every public item documented; `docs/` populated.
6. README + workflow rewritten; `ci/install_server.sh` removed; no Windows paths.
7. No `rustacuda*` anywhere; `npp-sys` consumed via local path; generated bindings
   scoped to NPP image domain (`nppi` symbols).

## Explicitly OUT OF SCOPE (Milestone 2)

- C2 — replace `debug_assert!` with `Result`-returning validation and seal the format
- C5 — replace `Vec::with_capacity` + `set_len` with zeroed/`MaybeUninit` + stride fix
- C8 — stream/execution-context model (CUDA streams, async ops)
- C11 — (if deferred by the open decision) seal/remove generic `T`
- C12 — golden-image correctness tests
- IPP bindings (`ipp-sys`/`ipp`)
- NPP signal ops (`npps*`)
- `image` crate upgrade from `0.23.13` to modern
- Broadening NPP coverage or adding pixel formats

---

## Appendix A — graphynx reference files (inlined verbatim)

These are fetched from `github.com/vansweej/graphynx` `main` branch. They are
the same author's Rust+CUDA+cudarc project and serve as the template for the
Nix dev shell, toolchain pinning, and `build.rs` link-search pattern.

### A.1 — `graphynx/flake.nix`

```nix
{
  # ============================================================================
  # Reproducible Rust + CUDA development environment
  #
  # All build tools (Rust toolchain, CUDA toolkit, cargo utilities) are pinned
  # via flake.lock to exact nixpkgs and rust-overlay commits. This means the
  # environment can be reproduced exactly, even years from now, with:
  #
  #   nix develop
  #
  # The one unavoidable system dependency is the NVIDIA kernel driver interface
  # (libcuda.so.1 and friends). These libraries MUST match the kernel module
  # version installed on the host and cannot be packaged in Nix. Everything
  # else — headers, nvcc, nvrtc, the Rust toolchain — is fully Nix-managed.
  #
  # To update all pinned inputs:
  #
  #   nix flake update
  # ============================================================================

  description = "Reproducible Rust + CUDA development environment";

  inputs = {
    # nixos-unstable is used for up-to-date CUDA packages. The exact commit is
    # pinned in flake.lock — the "unstable" in the name refers to the NixOS
    # release channel, not to the reproducibility of this flake.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # rust-overlay provides nightly/stable/beta Rust toolchains via rustup
    # toolchain files. Follows nixpkgs to avoid duplicate glibc versions.
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ rust-overlay.overlays.default ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true; # required for CUDA packages
        };

        rustVersion = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        cudaPackages = pkgs.cudaPackages;

        # ── System NVIDIA driver libs ─────────────────────────────────────────
        # These three libraries are the userspace interface to the NVIDIA kernel
        # module. They must come from the host OS because:
        #
        #   1. They must exactly match the kernel module version.
        #   2. The kernel module is managed by the OS, not Nix.
        #
        # We symlink only these specific files into .nvidia-libs/ rather than
        # putting /usr/lib/x86_64-linux-gnu on LD_LIBRARY_PATH or rpath
        # directly. That directory also contains the system glibc, which
        # conflicts with the newer Nix glibc used by the shell, causing
        # segfaults in unrelated tools (e.g. rust-analyzer, which).
        #
        # NOTE: This path is Debian/Ubuntu/Pop!_OS specific. On other distros
        # the NVIDIA driver libs may be in a different location, e.g.:
        #   Fedora/RHEL: /usr/lib64
        #   Arch:        /usr/lib
        systemNvidiaLibDir = "/usr/lib/x86_64-linux-gnu";
        systemNvidiaLibs = [
          "libcuda.so.1"                   # CUDA driver API
          "libnvidia-ptxjitcompiler.so.1"  # PTX JIT compiler
          "libnvidia-nvvm.so.4"            # NVVM IR compiler
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          name = "rusty-cuda";

          buildInputs = with pkgs; [
            # Rust toolchain — version pinned via rust-toolchain.toml
            rustVersion

            # Cargo utilities
            cargo-deny
            cargo-edit
            cargo-tarpaulin
            cargo-watch
            cargo-outdated
            cargo-update

            # LSP — nvim rust-analyzer will use this binary from PATH
            rust-analyzer

            # CUDA toolkit components (all pinned via flake.lock)
            cudaPackages.cuda_cudart  # CUDA runtime headers + stub libs
            cudaPackages.cuda_nvrtc   # NVRTC runtime compilation library
            cudaPackages.cuda_nvcc    # CUDA compiler driver

            # C++ standard library needed by Rust's cc crate and CUDA code
            stdenv.cc.cc.lib

            # gcc-13 is used as the nvcc host compiler. gcc-14+ triggers a
            # stack overflow when processing CUDA headers because Nix injects
            # -fstack-protector-strong and -fstack-clash-protection via
            # NIX_HARDENING_ENABLE, which overwhelms the CUDA header template
            # instantiation depth.
            gcc13

            # Audio capture — required by cpal on Linux (live-audio feature)
            alsa-lib
            pkg-config
          ];

          shellHook = ''
            export CUDA_PATH="${cudaPackages.cuda_cudart}"
            export NVRTC_PATH="${cudaPackages.cuda_nvrtc}"

            # Expose Nix CUDA and C++ runtime libs for dynamic linking at runtime.
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$CUDA_PATH/lib:$NVRTC_PATH/lib:$LD_LIBRARY_PATH"

            # LIBRARY_PATH is used by gcc-wrapper for link-time -L search paths.
            # Do NOT put stub directories here: gcc-wrapper records every
            # LIBRARY_PATH entry as an rpath in the output binary, which would
            # embed stub-only paths that must never be loaded at runtime.
            # Stubs are provided via cargo:rustc-link-search in build.rs instead.

            # NIX_LDFLAGS is set by mkShell and includes a -rpath $out/lib entry
            # where $out resolves to outputs/out inside the project directory.
            # That directory never exists and would be a dangling rpath entry in
            # every compiled binary. We remove only the two tokens "-rpath <out>"
            # while keeping all -L flags (which crtbeginS.o and friends need).
            export NIX_LDFLAGS="$(echo "$NIX_LDFLAGS" | sed 's|-rpath [^ ]*outputs/out[^ ]*||g')"

            # Create .nvidia-libs/ containing symlinks to the host NVIDIA driver
            # libs. This directory is added as an rpath in compiled binaries so
            # they find the real driver at runtime. See comment above for why
            # we symlink rather than using the system lib dir directly.
            mkdir -p .nvidia-libs
            for lib in ${pkgs.lib.concatStringsSep " " systemNvidiaLibs}; do
              src="${systemNvidiaLibDir}/$lib"
              if [ -f "$src" ]; then
                ln -sf "$src" ".nvidia-libs/$lib"
              fi
            done

            # Embed .nvidia-libs as rpath so compiled binaries resolve the
            # real NVIDIA driver libs at runtime without LD_PRELOAD or
            # LD_LIBRARY_PATH modifications. The Nix glibc rpath is also
            # embedded so that cargo-tarpaulin, which overrides RUSTFLAGS when
            # building test binaries, cannot accidentally cause the system
            # glibc (2.35 on Ubuntu/Pop!_OS) to be loaded instead of the Nix
            # glibc (2.42). The other half of the tarpaulin fix is in build.rs:
            # we must NOT emit cargo:rustc-link-search for /usr/lib/x86_64-linux-gnu
            # because tarpaulin converts those search paths into LD_LIBRARY_PATH
            # entries for test binaries, which would expose the system glibc.
            export RUSTFLAGS="-C link-arg=-Wl,-rpath,$PWD/.nvidia-libs -C link-arg=-Wl,-rpath,${pkgs.glibc}/lib $RUSTFLAGS"

            # Point nvcc at gcc-13 as its host compiler (see buildInputs above).
            export NVCC_HOST_COMPILER="${pkgs.gcc13}/bin/gcc"
            export PATH="${cudaPackages.cuda_nvcc}/bin:$PATH"
          '';
        };
      }
    );
}
```

### A.2 — `graphynx/backends-cuda/build.rs`

```rust
fn main() {
    // Provide the CUDA stub libs to the linker so it can resolve -lcuda and
    // -lnvrtc at link time without the real driver being present.
    //
    // CUDA_PATH and NVRTC_PATH are set by the Nix shell hook (flake.nix).
    //
    // The stubs directory will appear in the binary's rpath (Rust records any
    // -L native= path that contains .so files). This is harmless at runtime:
    // the real libcuda.so.1 is found first via .nvidia-libs/ (which appears
    // earlier in the rpath), and the stubs dir only contains libcuda.so (no
    // .1 suffix), so the dynamic linker ignores it.
    //
    // We do NOT add /usr/lib/x86_64-linux-gnu here. That would cause tarpaulin
    // to inject it into LD_LIBRARY_PATH for test binaries, loading the system
    // glibc (2.35) instead of the Nix glibc (2.42) and causing:
    //   symbol lookup error: /usr/lib/x86_64-linux-gnu/libc.so.6:
    //   undefined symbol: __nptl_change_stack_perm, version GLIBC_PRIVATE
    //
    // At runtime, the real libcuda.so.1 is resolved via the rpath pointing to
    // .nvidia-libs/, which is set in RUSTFLAGS from the shell hook.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib/stubs");
    }
    if let Ok(nvrtc_path) = std::env::var("NVRTC_PATH") {
        println!("cargo:rustc-link-search=native={nvrtc_path}/lib/stubs");
    }
}
```

### A.3 — `graphynx/rust-toolchain.toml`

```toml
[toolchain]
# Pinned to an exact version to prevent silent upgrades from breaking CUDA
# build scripts or cudarc API compatibility. Update deliberately with:
#   1. Change the version here
#   2. Run: nix flake update
#   3. Test: cargo build && cargo test && cargo tarpaulin
channel = "1.94.1"
```

---

## Appendix B — Type-design constraints for image-rs/imageproc future integration

These are binding constraints on any type/signature work in this milestone
(especially Phase 2 Step 5 if C11/C3/C4 are pulled in). Derived from the
long-term intent to integrate with `github.com/image-rs/image` and
`github.com/image-rs/imageproc`, confirmed by the author.

1. **`image`-crate types stay at the INTEROP BOUNDARY** (`From`/`TryFrom` edges),
   not woven into `CudaImage`'s core. This lets the crate drop into `image` or
   `imageproc` later without re-architecture. The current design already does this
   (`image.rs:117-190`) — preserve this property.

2. **If a pixel type marker is defined (e.g. `NppPixelType` for C11):** design it
   so it can later MAP ONTO image-rs's `Pixel`/`Primitive` trait family. Do NOT
   invent a parallel pixel taxonomy that will fight image-rs at integration time.

3. **`CudaLayout` ↔ `SampleLayout`:** today only `From<SampleLayout>` exists
   (one-way, `layout.rs:54`). Do not foreclose the round-trip. If the type is
   refactored, consider implementing `From<CudaLayout> for SampleLayout` or
   keeping the field structure compatible.

4. **Version watch-item (not an M1 change):** the integration target is modern
   image-rs (0.25+), whose pixel/color API differs from the pinned `image = "0.23.13"`.
   Do NOT design the C11 marker or the error type against 0.23's color model
   specifically. Upgrading `image` remains M2.

---

## Appendix C — Per-phase findings map (`reviews/final-report.md`)

Each agent MUST read the full report at least once, but these are the findings
most relevant to each phase:

- **Phase 1 (Nix shell + build.rs):** C6 (CI/registry deps), C10 (EOL stack),
  CT5 (packed-vs-pitched stride alignment for NPP build config).
- **Phase 2 (cudarc port):** C1 (error collapse), C3/C4 (ownership + RefCell),
  C7 (context lifetime), C10 (EOL stack), C11 (generic T). MUST read CT2 and
  CT6 retractions to avoid re-introducing phantom fixes.
- **Phase 3 (test tiering):** C6 (CI gaps), C12 (no correctness tests).
  Read CT1 (set_len dormant-vs-immediate debate) for severity calibration.
- **Phase 4 (docs):** C9 (save footgun to document truthfully), C7 (context
  invariant to document).
- **Phase 5 (README + CI):** C6 (CI pipeline), C10 (EOL stack).
- **Cross-cutting (type design):** C3/C4, C11 + Appendix B of this plan.
