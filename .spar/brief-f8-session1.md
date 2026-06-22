# Brief: F8 (Session 1 of 2) — Stream/execution-context: FFI + abstraction

## Feature / context

F8 on the roadmap (`docs/roadmap.md:249–269`): a first-class stream/execution-
context model for `npp-rs`. Closes **C8** (zero stream concept; correctness is
emergent default-stream ordering) and **C7** (hardcoded `Device::get_device(0)`;
context-lifetime unenforced invariant). The review report recommends bundling
these with configurable device selection into a single owned execution-context
type.

## Session split rationale

F8 is split into **two batch sessions** because the work touches both the FFI
boundary and the runtime API, and the concrete `_Ctx` FFI surface cannot be
inspected until the allowlist is widened.

| Session | What | Validated how | Gate |
|---------|------|---------------|------|
| **S1** | FFI allowlist widening + `StreamContext` abstraction | CPU-only: `cargo build`, `cargo test`, `cargo clippy`, `cargo doc` | No GPU needed |
| **S2** | Wire `Arc<StreamContext>` onto `CudaImage`; pivot 3 ops to `_Ctx`; regenerate codegen; async tests; docs | GPU run on real device | **Mandatory pre-merge GPU validation** |

After S1 lands, the S2 brief is **regenerated** with concrete Phase-0 facts
baked in (real arities, buffer-size type, `Copy` answer, `cudaStream_t` cast)
— so S2 is a plan-with-facts, not a plan-with-assumptions.

## The decision that drives everything (binding)

**Pivot fully to NPP's application-managed `_Ctx` API.**

This **REVERSES** the F2 `.spar/brief.md` decision O4 ("emit non-`_Ctx` only"),
which was made for codegen simplicity before the NPP deprecation timeline was
fully understood.

**Rationale (verified from NPP official documentation and headers):**
- NPP ≥12.5 **mandates** the application-managed stream-context interface;
  the legacy `nppSetStream`/`nppGetStream` API now returns
  `NPP_STREAM_CONTEXT_ERROR` when used (they were single-context globals that
  don't compose with modern CUDA multi-stream usage).
- **Non-`_Ctx` functions are REMOVED in CUDA Toolkit 13.0.** The project
  targets "latest CUDA in nixpkgs," so this is a dated cliff, not a
  hypothetical future concern.
- `nppGetStreamContext()` itself is **deprecated for CUDA 13.0** — the context
  must be **application-populated** from `cudaDeviceGetAttribute` queries, not
  queried from the driver. This means the `StreamContext` helper must be a
  full field-by-field population, not a thin query wrapper.

## Locked decisions (from sparring)

These four were confirmed in the architecture sparring session and are frozen
for both S1 and S2:

1. **_Ctx-exclusive (single code path).** All op implementations call only
   the `_Ctx` variant. No `#[cfg]` gating between `_Ctx`/non-`_Ctx`. The
   non-`_Ctx` path is dead code once S2 lands.
2. **Stream stored on `CudaImage` as `Arc<StreamContext>`.** Op signatures
   keep their current shape — the stream context is accessed via
   `self.ctx()`. No new parameter threading through public API.
3. **Async-by-default + explicit sync points.** NPP `_Ctx` ops are enqueued
   on the stream without blocking. Synchronisation happens at well-defined
   points: `TryFrom<&CudaImage>` read-back (DTOH copy) and
   `StreamContext::synchronize()` (explicit fence). Every doc comment
   on `_Ctx`-wired ops states this contract.
4. **Wire all 3 existing ops (Resize, SwapChannels, Mean) in Session 2.**
   No new ops are added in F8; the work is purely mechanical pivoting of
   existing generated code.

## Verified facts (cudarc 0.9, web-confirmed)

- `CudaDevice::fork_default_stream()` -> `CudaStream { pub stream: CUstream, … }`,
  which is `!Send + !Sync`. Sync via `dev.wait_for(&stream)` /
  `dev.synchronize()`.
- `cudarc::driver::result::device::get_attribute(dev: CUdevice,
  attrib: CUdevice_attribute) -> Result<i32, DriverError>` — this is `unsafe`.
  Returns `i32` for all attributes except `nSharedMemPerBlock` (cast to
  `usize`).
- `device.cu_device()` -> `&CUdevice`; `device.ordinal()` -> `usize`.
- `fork_default_stream` inserts an implicit wait so the forked stream
  sees prior default-stream work. This is the **ordering-correctness basis**
  for F8's async contract: host-to-device copy (default stream) → NPP op
  (forked stream, implicitly ordered by the fork wait) → read-back via
  `DtoH` sync copy on the same stream — no races *by construction*.

## `NppStreamContext` fields to populate (from `nppdefs.h`)

| Field | Source | Notes |
|-------|--------|-------|
| `hStream` | `stream.stream` as `cudaStream_t` | Cross-FFI-crate opaque pointer cast |
| `nCudaDeviceId` | `device.ordinal() as i32` | |
| `nMultiProcessorCount` | `get_attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)` | |
| `nMaxThreadsPerMultiProcessor` | `get_attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)` | |
| `nMaxThreadsPerBlock` | `get_attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)` | |
| `nSharedMemPerBlock` | `get_attribute(…CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK) as usize` | i32 → usize |
| `nCudaDevAttrComputeCapabilityMajor` | `get_attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)` | |
| `nCudaDevAttrComputeCapabilityMinor` | `get_attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)` | |
| `nStreamFlags` | `0` | fork_default_stream creates a standard (non-blocking, non-sync) stream |
| `reserved` | `[0i64; 16]` | Zero-initialised; "do not touch" per NPP docs |

## Session 1 scope — Phases 0–2

### Phase 0 — Pin the FFI surface (no code written; gates S2 parameters)

Run Phase 1 first to widen the allowlist, rebuild `npp-sys`, then inspect the
generated `bindings.rs` to record **four concrete facts** that parameterise
Session 2:

- **0.1  `_Ctx` arity:** confirm `nppiResize_8u_C1R_Ctx`,
  `nppiSwapChannels_8u_C4C3R_Ctx`, `nppiMean_8u_C1R_Ctx`,
  `nppiMeanGetBufferHostSize_8u_C1R_Ctx` all exist with the expected arity
  (base args + trailing `NppStreamContext`). **If
  `nppiSwapChannels_8u_C4C3R_Ctx` is absent → STOP & escalate** — conversion-op
  `_Ctx` coverage is the likeliest gap, and it would reshape S2 Phase 4.
- **0.2  Mean buffer-size out-pointer type:** `*mut usize` (NPP `size_t`) or
  `*mut i32` (NPP `int`)? The NPP 12.4 change (int→size_t) must be resolved.
- **0.3  `NppStreamContext` auto-traits:** does bindgen derive `Copy` on
  `NppStreamContext`? If yes, `raw_ctx()` returns by value; if no,
  use `ptr::read` to move out of the zeroed skeleton (safe because the
  skeleton is fully initialised before any `raw_ctx()` call).
- **0.4  `cudaStream_t` definition in `npp-sys`:** confirm that
  `npp_sys::cudaStream_t` is a `*mut` to an opaque type, and that a
  raw-pointer transmute from `cudarc::driver_sys::CUstream` (also a `*mut`
  opaque) is the correct cross-crate cast.

### Phase 1 — Widen `npp-sys` allowlist

`npp-sys/build.rs`: add `allowlist_type("NppStreamContext")`,
`allowlist_type("Npp.*")`, `allowlist_var("NPP_.*")`. Exclude
`nppGetStreamContext`, `nppSetStream`, and all `npps*` symbols. Phase-0
findings recorded as a comment block in the new compile-test.

### Phase 2 — `StreamContext` abstraction

New file `npp/src/stream.rs`: a population helper that zeroes an
`NppStreamContext` and fills the 8 device-attribute fields via
`get_attribute`; `nStreamFlags = 0` with a `TODO(perf)` noting that
`CU_STREAM_NON_BLOCKING` is the default for `fork_default_stream`.

`pub struct StreamContext { device: Arc<CudaDevice>, stream: CudaStream,
raw: NppStreamContext }` with constructor, accessors (`device()`, `raw_ctx()`,
`synchronize()`), and the ordering-correctness doc paragraph.

`npp/src/cuda.rs`: add `stream_context_for(ordinal)` convenience constructor.

All CUDA-touching functions `#[cfg(not(tarpaulin_include))]`.

## Risks (ordered by severity)

1. **No GPU in CI** — but GPU run is a **mandatory pre-merge gate**
   (confirmed by the user). S2 correctness is validated before landing.
   S1 is CPU-complete and does not require GPU.
2. **`SwapChannels` `_Ctx` may not exist** — Phase 0.1 stop-and-escalate
   (control-flow is "report facts back to the human," not "guess and proceed").
3. **Mean buffer-size int/size_t ABI mismatch** — pinned by Phase 0.2, applied
   in S2 Phase 4. The `*mut c_int` vs `*mut usize` distinction is a real
   memory-safety issue if wrong, and it's invisible to the Rust type system.
4. **`NppStreamContext` not `Copy`** — `raw_ctx()` uses `ptr::read` (safe
   because the struct is fully populated by the constructor and never
   borrowed mutably afterwards). Phase 0.3 determines which path.
5. **Cross-crate `cudaStream_t` cast friction** — pre-emptively documented
   in Phase 0.4; the cast is an unchecked pointer transmute between two
   opaque `*mut` types from different bindgen invocations. Semantically
   equivalent, but unsightly and a maintenance hazard if cudarc changes
   `CUstream`'s definition.

## Session 2 hand-off

After S1 merges, regenerate the S2 brief WITH the concrete Phase-0 findings
baked in: real arities, buffer-size type, `Copy` answer, `cudaStream_t` cast
expression. S2 = Phases 3–7. Phases 4+5 land together (green only when both
in). GPU run before merge to `main`.
