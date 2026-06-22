# Feature: F8 (Session 1) — Stream/execution-context: FFI + abstraction

**Definition of done (Session 1):**
- `npp-sys` allowlist widened to include `NppStreamContext`, `Npp.*`, `NPP_.*`;
  `cargo build -p npp-sys` regenerates `bindings.rs` with these symbols.
- A compile-only smoke test (`stream_context_symbols.rs`) confirms all expected
  `_Ctx` symbol names exist and have the right arity.
- `StreamContext` struct exists in `npp/src/stream.rs` with `new()`, `device()`,
  `raw_ctx()`, and `synchronize()`.
- `stream_context_for(ordinal)` convenience constructor in `npp/src/cuda.rs`.
- Phase-0 findings (0.1–0.4) are recorded as a comment block in the smoke test.
- `cargo build`, `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`,
  and `cargo doc --no-deps -p npp-rs -p npp-sys` all pass on CPU.
- `cargo test` runs zero GPU code (no device initialisation).

**Execution-mode legend:**
- 🧑‍🔬 — human-gated GPU (require device)
- 🐚 — human-gated Nix shell (no GPU)
- 🤖 — codegen-able / pure-Rust offline
- 🧭 — fact-finding checkpoint

---

## Phase 0 — Pin the FFI surface

**Commit message:** *(none — no code committed in this phase)*

Phase 0 is a **fact-finding checkpoint** that runs Phase 1 first (widening
the allowlist), then inspects the generated `bindings.rs` to record four
specific facts. These facts parameterise Session 2 and are the entire reason
for the two-session split.

**Order dependency:** Phase 1 must complete first so the generated
`bindings.rs` contains the `_Ctx` symbols and `NppStreamContext`. Phase 0
then inspects the output.

### Step 0.1 — Run Phase 1 (widened allowlist)

Build `npp-sys` with the widened allowlist from Phase 1.

```bash
nix develop . --command cargo build -p npp-sys
```

This regenerates `npp-sys/src/bindings.rs` into `OUT_DIR`. The file is
gitignored — we inspect it in its build output location.

### Step 0.2 — Confirm `_Ctx` symbol existence and arity

**Fact to record:** do all four expected `_Ctx` symbols exist with the right
arity? The "right arity" means: same parameters as the non-`_Ctx` variant
PLUS a trailing `NppStreamContext` parameter (either by value or by pointer).

The four symbols to check:

1. `nppiResize_8u_C1R_Ctx`
2. `nppiSwapChannels_8u_C4C3R_Ctx`
3. `nppiMean_8u_C1R_Ctx`
4. `nppiMeanGetBufferHostSize_8u_C1R_Ctx`

**Procedure:** Find the generated `bindings.rs` (under
`target/debug/build/npp-sys-*/out/` or wherever `OUT_DIR` points), then
`grep` for each symbol and compare its parameter list to the non-`_Ctx`
version. Record the trailing `NppStreamContext` parameter style (is it
`nppStreamCtx: NppStreamContext` by value, or pointer?).

**Especially:**
- For `nppiSwapChannels_8u_C4C3R_Ctx` — if this symbol **does not exist**
  at all, STOP and escalate to the human. The conversion-op `_Ctx` coverage
  is the likeliest gap and would reshape S2 Phase 4 significantly.

**Record finding 0.2** as a comment in the Phase-1 smoke test file:
```rust
// Phase 0.2: _Ctx symbol arity (recorded YYYY-MM-DD)
// nppiResize_8u_C1R_Ctx: <exists/absent>, params = NPP non-ctx args + <trailing NppStreamContext>
// nppiSwapChannels_8u_C4C3R_Ctx: <exists/absent>, params = ...
// nppiMean_8u_C1R_Ctx: <exists/absent>, params = ...
// nppiMeanGetBufferHostSize_8u_C1R_Ctx: <exists/absent>, params = ...
// ESCALATION NOTE: (present/absent for SwapChannels C4C3R)
```

### Step 0.3a — Mean buffer-size out-pointer type

**Fact to record:** what is the Rust type of the `hpBufferSize` parameter
in `nppiMeanGetBufferHostSize_8u_C1R_Ctx`? Is it `*mut usize` (NPP `size_t`
in the Rust layout, which is `usize`) or `*mut i32` (NPP `int`)?

NPP 12.4 changed several `GetBufferHostSize` functions from `int*` to
`size_t*`. This distinction matters for S2 Phase 4 because passing a
`*mut i32` where `*mut usize` is expected (or vice versa) is an ABI
mismatch that can corrupt adjacent memory.

**Procedure:** `grep` for `nppiMeanGetBufferHostSize_8u_C1R` in the new
`bindings.rs` and look at the `hpBufferSize` parameter type.

**Record finding 0.3a:** as a comment in the smoke test.

### Step 0.3b — Is `NppStreamContext` `Copy`?

**Fact to record:** does bindgen's generated `NppStreamContext` derive
`Copy`? Look for:
```rust
#[repr(C)]
#[derive(Copy, Clone)]
pub struct NppStreamContext { ... }
```
vs just `Clone` or neither.

This determines whether `StreamContext::raw_ctx()` can return by value
(`Copy`) or must use `ptr::read` (non-`Copy`, safe because the struct is
fully populated by the constructor and never mutably borrowed afterwards).

**Procedure:** `grep` for `pub struct NppStreamContext` in `bindings.rs` and
read the `#[derive(...)]` line preceding it.

**Record finding 0.3b:** as a comment in the smoke test.

### Step 0.3c — What is `npp_sys::cudaStream_t`?

**Fact to record:** what is `cudaStream_t` in `npp-sys`'s bindings? It's
defined in the CUDA driver API bindings, typically:
```rust
pub type cudaStream_t = *mut CUstream_st;
```
or similar opaque pointer type.

This determines the cast expression for `StreamContext::raw_ctx()`:
`hStream = stream.stream as npp_sys::cudaStream_t` (where `stream.stream`
is `cudarc::driver_sys::CUstream`, also an opaque `*mut`).

**Procedure:** `grep` for `cudaStream_t` and `CUstream_st` in `bindings.rs`.

**Record finding 0.3c:** as a comment in the smoke test.

---

## Phase 1 — Widen `npp-sys` allowlist

**Commit message:** `feat(npp-sys): widen allowlist with NppStreamContext and
NPP core types`

Every `_Ctx` symbol uses `NppStreamContext`; some use other `Nppi*` and `Npp*`
types. The current allowlist (`nppi.*`/`Nppi.*`) is too narrow — it captures
operation types but misses the core context and status types.

### Step 1.1 — Widen `build.rs` allowlist

**File:** `npp-sys/build.rs`

Current allowlist (approximately):
```rust
.allowlist_function("nppi.*")
.allowlist_type("Nppi.*")
```

Change to:
```rust
.allowlist_function("nppi.*")
.allowlist_type("Nppi.*")
.allowlist_type("NppStreamContext")
.allowlist_type("Npp.*")
.allowlist_var("NPP_.*")
```

**Do NOT add:**
- `nppGetStreamContext` / `nppSetStream` (deprecated for CUDA 13.0)
- `npps.*` (signal domain, explicitly out of scope per roadmap F9)
- `npp.*` (the `npp` prefix matches utility functions we don't need yet)

The `Npp.*` glob is deliberately broad — it covers `NppStatus`, `NppBool`,
`NppLibraryVersion`, and any other core types referenced by `Nppi*` or
`NppStreamContext` fields. If bindgen pulls in types that can't compile,
narrow to exactly what's needed (but the glob is safe for these C-compatible
types).

### Step 1.2 — Rebuild and verify

```bash
nix develop . --command cargo build -p npp-sys
```

No errors expected. If `Npp.*` pulls in something that fails to compile (e.g.
a VLA or flexible array member), narrow the glob.

### Step 1.3 — Create compile-only `_Ctx` smoke test

**New file:** `npp-sys/tests/stream_context_symbols.rs`

This is a **non-GPU, compile-only test** that coerces each known `_Ctx`
symbol to a fully-typed function pointer. The coercion fails at compile time
if the symbol has the wrong arity or parameter types — this is the strongest
compile-time check available without calling the FFI.

```rust
//! Compile-time arity check for _Ctx symbols.
//!
//! Phase-0 findings recorded below.

// Phase 0.2: _Ctx symbol arity (recorded YYYY-MM-DD)
// ... (fact block per 0.2–0.3c)

use npp_sys::*;
use std::ffi::c_int;

// Resize: SRC+STEP, SRC_SIZE, SRC_RECT, DST+STEP, DST_SIZE, DST_RECT, INTERP
// Non-_Ctx: 9 params + _Ctx → 10 params with trailing NppStreamContext
#[test]
fn resize_ctx_symbol_exists() {
    let _: unsafe extern "C" fn(
        *const u8, c_int, NppiSize, NppiRect,
        *mut u8, c_int, NppiSize, NppiRect, c_int,
        NppStreamContext,
    ) -> NppStatus = nppiResize_8u_C1R_Ctx;
}

// SwapChannels C4C3R: SRC+STEP, DST+STEP, SIZE, CHANNEL_ORDER
// Non-_Ctx: 6 params + _Ctx → 7 params
#[test]
fn swap_channels_ctx_symbol_exists() {
    let _: unsafe extern "C" fn(
        *const u8, c_int,
        *mut u8, c_int,
        NppiSize, *const c_int,
        NppStreamContext,
    ) -> NppStatus = nppiSwapChannels_8u_C4C3R_Ctx;
}

// Mean: SRC+STEP, SIZE, SCRATCH_BUF, OUT_SCALAR
// Non-_Ctx: 5 params + _Ctx → 6 params
#[test]
fn mean_ctx_symbol_exists() {
    let _: unsafe extern "C" fn(
        *const u8, c_int, NppiSize,
        *mut u8, *mut f64,
        NppStreamContext,
    ) -> NppStatus = nppiMean_8u_C1R_Ctx;
}

// MeanGetBufferHostSize: SIZE, OUT_SCALAR
// Non-_Ctx: 2 params + _Ctx → 3 params
// NOTE: hpBufferSize type depends on Phase 0.3a (*mut usize or *mut c_int)
#[test]
fn mean_buffer_size_ctx_symbol_exists() {
    let _: unsafe extern "C" fn(
        NppiSize, *mut usize,
        NppStreamContext,
    ) -> NppStatus = nppiMeanGetBufferHostSize_8u_C1R_Ctx;
}

// Verify NppStreamContext itself is available
#[test]
fn npp_stream_context_is_accessible() {
    // Just assert the struct has a known size; no init needed
    assert_eq!(
        std::mem::size_of::<NppStreamContext>(),
        176,  // 8 + 4 + 4 + 4 + 4 + 8 + 4 + 4 + 4 + 4 + 8*16 = expected size
        "NppStreamContext size changed; verify struct layout"
    );
}
```

**IMPORTANT:** The exact function-pointer types above are **tentative** —
they must be adjusted after Phase 0.2 confirms the actual arities. The
`// Phase 0.2:` comment block is updated in place with the confirmed types.

The `SwapChannels` test **must be gated behind a `#[cfg(any())]` or
commented out** if Phase 0.2 finds the symbol doesn't exist, with an
escalation note. (The preferred approach is to keep it compiled and let it
fail loudly so the gap is visible.)

### Step 1.4 — Verify all builds pass

```bash
nix develop . --command cargo build -p npp-sys && \
nix develop . --command cargo test -p npp-sys && \
nix develop . --command cargo clippy -- -D warnings && \
nix develop . --command cargo fmt --check
```

**Acceptance:** `cargo test -p npp-sys` reports "3 passed; 0 failed"
(the three `_Ctx` symbol tests, unless SwapChannels is absent).

---

## Phase 2 — `StreamContext` abstraction

**Commit message:** `feat(npp-rs): add StreamContext abstraction with
NppStreamContext population and synchronize`

### Step 2.1 — Create `npp/src/stream.rs`

**New file:** `npp/src/stream.rs`

Contains the `StreamContext` struct and population logic:

```rust
//! Application-managed NPP stream context.
//!
//! # Ordering correctness (key to the async contract)
//!
//! NPP `_Ctx` functions are enqueued asynchronously on the stream stored
//! in this context. **Correctness relies on implicit ordering guarantees**
//! from `CudaDevice::fork_default_stream()`:
//!
//! 1. A host-to-device copy on the **default stream** completes first.
//! 2. `fork_default_stream()` creates a new stream and inserts an implicit
//!    wait so all previously-enqueued default-stream work is visible.
//! 3. The NPP operation is enqueued on the **forked stream**.
//! 4. Read-back (`TryFrom<&CudaImage>`) performs a synchronous DtoH copy
//!    on the **forked stream**, which blocks until the NPP op completes.
//!
//! This means: `htod_copy → NPP_op → dtoh_readback` is ordered BY
//! CONSTRUCTION, not by accident. See C8 rationale in the review report.
//!
//! # Safety
//!
//! `NppStreamContext` is populated by querying the CUDA device with
//! `cuDeviceGetAttribute` (via `cudarc::driver::result::device::get_attribute`).
//! The `hStream` field is a cross-crate pointer cast from
//! `cudarc::driver_sys::CUstream` to `npp_sys::cudaStream_t` — both are
//! opaque `*mut` to the same underlying driver object, so the cast is
//! semantically valid despite crossing FFI-crate boundaries.
//!
//! The originating [`cudaDeviceGetAttribute`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html)
//! driver call is **unsafe**:
//! - It dereferences a raw `CUdevice` handle.
//! - It must only be called while the CUDA driver is initialised and the
//!   device handle is valid (both invariants upheld by `cudarc::CudaDevice`).

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaStream, DriverError};
use npp_sys::{NppStreamContext, NppStatus};

/// A CUDA device handle with an associated stream and a populated
/// [`NppStreamContext`] for use with NPP `_Ctx` functions.
///
/// Every [`StreamContext`] owns its device reference and stream. The device
/// handle **must outlive** all [`CudaImage`](crate::CudaImage) buffers
/// created from it (finding C7).
pub struct StreamContext {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    raw: NppStreamContext,
}

impl StreamContext {
    /// Create a new stream context on the given device.
    ///
    /// Forks a new default stream and populates an [`NppStreamContext`] from
    /// device attributes.
    ///
    /// # Errors
    ///
    /// Returns [`DriverError`] if any `cuDeviceGetAttribute` call fails,
    /// or if stream creation fails.
    ///
    /// # Panics
    ///
    /// Panics if the device ordinal does not match the expected device
    /// (defensive; should never happen if `device` is well-typed).
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, DriverError> {
        // 1. Fork a new stream (if this is the forked stream, it inserts
        //    an implicit wait that sees all prior default-stream work).
        let stream = device.fork_default_stream()?;

        // 2. Populate NppStreamContext.
        let raw = populate_stream_context(&device, &stream)?;

        Ok(Self { device, stream, raw })
    }

    /// Reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Reference to the CUDA stream.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// The populated [`NppStreamContext`] for passing to NPP `_Ctx` functions.
    ///
    /// # Safety
    ///
    /// The caller must not modify the returned struct. It is a populated
    /// snapshot of device attributes and a stream handle; mutating it
    /// (e.g. changing `hStream`) will cause NPP to enqueue operations
    /// on a different stream than expected, breaking the ordering contract.
    #[doc(hidden)]  // pub(crate) until CudaImage wiring in S2
    pub fn raw_ctx(&self) -> NppStreamContext {
        // Phase 0.3b determines whether NppStreamContext is Copy.
        // If Copy: just return self.raw (implicit copy).
        // If not Copy: self.raw (move) is fine because the struct is
        // fully populated and we never mutate it.
        self.raw
    }

    /// Block until all operations enqueued on this stream complete.
    ///
    /// This is the primary sync point for async NPP operations.
    /// After calling `synchronize()`, results can be read back safely.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.device.wait_for(&self.stream)
    }
}

#[cfg(not(tarpaulin_include))]
fn populate_stream_context(
    device: &CudaDevice,
    stream: &CudaStream,
) -> Result<NppStreamContext, DriverError> {
    use cudarc::driver::result::device::get_attribute;
    use cudarc::driver_sys::CUdevice_attribute::*;

    let cu_dev = *device.cu_device();
    let ordinal = device.ordinal();

    // Helper: query an attribute, converting DriverError into our error type
    macro_rules! attr {
        ($attr:ident) => {
            // SAFETY: device is guaranteed alive by the CudaDevice Arc
            get_attribute(cu_dev, $attr).map_err(|e| {
                DriverError::from(e)
            })?
        };
    }

    let hStream: npp_sys::cudaStream_t = stream.stream as npp_sys::cudaStream_t;

    Ok(NppStreamContext {
        hStream,
        nCudaDeviceId: ordinal as i32,
        nMultiProcessorCount: attr!(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        nMaxThreadsPerMultiProcessor: attr!(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
        nMaxThreadsPerBlock: attr!(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        nSharedMemPerBlock: attr!(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK) as usize,
        nCudaDevAttrComputeCapabilityMajor: attr!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        nCudaDevAttrComputeCapabilityMinor: attr!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        nStreamFlags: 0,  // fork_default_stream creates a standard stream
        reserved: [0i64; 16],
    })
}
```

**Edge-case behaviour:**
- `nStreamFlags = 0`: `fork_default_stream` creates a standard (non-blocking
  with respect to the default stream) stream, so `CU_STREAM_NON_BLOCKING` is
  the implicit default. A `TODO(perf)` comment should note that explicit
  `CU_STREAM_NON_BLOCKING` is a no-op here but may be relevant if we ever
  allow configurable stream creation flags.
- Device attribute queries can theoretically fail if the device handle
  is stale. In practice, the `Arc<CudaDevice>` guarantees liveness. If a
  query fails, the error propagates through `DriverError`.
- The `reserved` field: NPP requires this to be zeroed. The struct literal
  ensures this as long as `#[derive(Zeroable)]` or manual zeroing confirms
  it. If `NppStreamContext` is not `Zeroable` (rare for repr(C) structs),
  initialise with `unsafe { std::mem::zeroed() }` and then populate
  every field.

### Step 2.2 — Add `StreamContext` constructor to `cuda.rs`

**File:** `npp/src/cuda.rs`

Add a convenience function:

```rust
/// Create a [`StreamContext`] for the device at the given ordinal.
///
/// # Panics
///
/// Panics if the device cannot be initialised (e.g. no CUDA-capable GPU
/// at that ordinal).
///
/// # Example
///
/// ```rust,ignore
/// use npp_rs::stream_context_for;
/// let ctx = stream_context_for(0).expect("no GPU at ordinal 0");
/// ```
#[cfg(not(tarpaulin_include))]
pub fn stream_context_for(ordinal: usize) -> Result<Arc<StreamContext>, DriverError> {
    let device = Arc::new(CudaDevice::new(ordinal)?);
    StreamContext::new(device)
}
```

The return type is `Arc<StreamContext>` to match the pattern used by
`CudaImage` in S2 (which will store `Arc<StreamContext>`).

### Step 2.3 — Wire into module tree

**File:** `npp/src/lib.rs`

Add:
```rust
pub mod stream;
pub use stream::StreamContext;

// In the cuda module or alongside it:
// pub use stream::stream_context_for?  -- or keep it in cuda.rs
```

Decision: `stream_context_for` can live in `cuda.rs` (which already has
`CudaDevice` initialisation helpers) or in `stream.rs`. Keep it in
`stream.rs` for locality, and re-export from `lib.rs`.

### Step 2.4 — Verify all builds pass

```bash
nix develop . --command cargo build && \
nix develop . --command cargo test && \
nix develop . --command cargo clippy -- -D warnings && \
nix develop . --command cargo fmt --check && \
nix develop . --command cargo doc --no-deps -p npp-rs -p npp-sys
```

**Acceptance:**
- `cargo build` succeeds.
- `cargo test` passes (no GPU initialisation — `StreamContext::new` is not
  called by any test in this session because all tests run on CPU only).
- `cargo tarpaulin` still works (all CUDA-touching functions are annotated
  `#[cfg(not(tarpaulin_include))]`).

### Step 2.5 — Write the ordering-correctness design note

**In `docs/`** (new file or appended to an existing arch doc).

Add a note at `docs/stream-context.md` that explains the async contract,
the fork_default_stream implicit-wait guarantee, and the
construct-by-construction ordering of `htod → NPP op → dtoh`. This is the
documentation anchor that readers of `StreamContext`'s doc comments are
pointed to. Cross-link from the roadmap.

```markdown
# Stream-context ordering model

[StreamContext](../npp/src/stream.rs) guarantees **ordered execution without
explicit synchronisation** between:

1. Host-to-device copies (performed on the default stream).
2. NPP operations (enqueued on a forked stream).
3. Device-to-host read-backs (synchronous DtoH copy on the forked stream).

The guarantee rests on `CudaDevice::fork_default_stream()`:
the forked stream inserts an implicit wait that makes all prior default-stream
work visible before the first NPP op executes. See `npp/src/stream.rs` for the
full doc comment.

## Sync points

- `StreamContext::synchronize()` — blocks until all enqueued work completes.
- `TryFrom<&CudaImage>` read-back — synchronous DtoH copy (blocks on the
  forked stream).
```

**Verify:** `cargo doc --no-deps -p npp-rs -p npp-sys` renders the new doc
cross-references.

---

## Cross-cutting risks (Session 1)

| ID | Risk | Mitigation |
|----|------|------------|
| S1-R1 | `Npp.*` glob pulls in an un-compilable type (VLA, flex array member) | If build fails, narrow glob to explicit allowlist. Unlikely for the core NPP types. |
| S1-R2 | `NppStreamContext` field layout/size doesn't match expected 176 bytes | Phase 0.3b records the actual size; adjust `raw_ctx()` return strategy accordingly. |
| S1-R3 | `get_attribute` requires active CUDA context and fails if called CPU-only | `populate_stream_context` is `#[cfg(not(tarpaulin_include))]` but can run CPU-side as long as a CUDA driver is loaded. The function is only called from `StreamContext::new` which requires a real device; no CPU test calls it. |
| S1-R4 | `fork_default_stream` not available in older cudarc? | Already confirmed in cudarc 0.9. If the version in nixpkgs is newer, the API is stable. |
| S1-R5 | `cudaDeviceGetAttribute` not imported in `cudarc::driver::result::device` | Already confirmed in the research phase. If the function path differs, adjust imports. |
