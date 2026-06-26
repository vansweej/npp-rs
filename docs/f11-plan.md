# Feature: F11 — Upgrade cudarc from 0.9.15 to 0.19.x

## Preconditions (verified before Phase 1)

- `nix develop . --command cargo build -p npp-rs` succeeds on the current `main`.
- `nix develop . --command cargo test -p npp-rs` succeeds (pure-logic tests, no GPU).
- `nix develop . --command cargo clippy -- -D warnings` passes.
- `nix develop . --command cargo fmt --check` passes.
- `nix develop . --command cargo tarpaulin` passes with ≥90 % coverage.
- `nix develop . --command cargo test --features gpu -p npp-rs` passes (all golden + ROI tests green on a GPU host).
- cudarc `=0.19.8` (or latest `0.19.x`) is published on crates.io; its `cuda-12090` feature flag is confirmed in the crate's `Cargo.toml` by inspecting the [cudarc repo](https://github.com/vosen/cudarc) or `crates.io` metadata.
- The Nix dev shell (`CUDA_PATH` from `flake.nix:82`) provides CUDA 12.9 headers and libs — this agrees with the `cuda-12090` feature value.
- The Nix `shellHook` (`flake.nix:96`) exposes `libcuda.so` via `LD_LIBRARY_PATH` and `.nvidia-libs/` — the driver-version agreement is a host precondition, not checked here.

## Execution split (CRITICAL for the driver)

- **Pipeline-completable on any host (no GPU needed):** Phase 0 (pre-flight verification), Phase 1 (manifest bump), Phase 3 (core types — compiles but streams can't be used without GPU), Phase 4 (macros — compiles but can't be tested without GPU), Phase 5 (test port — compiles but ROI tests can't run), Phase 7 (documentation), Phase 8 (final gate — pure-logic tests, clippy, fmt, tarpaulin).
- **Requires CUDA host + real GPU (mark `pending-manual-GPU`, NOT done):** Phase 2 (SPIKE — the NppStreamContext bridge verification requires `golden_chained_ctx.rs` to pass on GPU). Phase 6 (golden suite re-pin — every golden test must be re-run on GPU to capture new byte output and verify byte-identity guards still pass). These phases cannot complete without a GPU; the plan's pipeline runner must not falsely report success.

## Design summary (read once before execution)

**Goal:** Upgrade cudarc from `=0.9` to `=0.19.x` (pinned, `cuda-12090` feature), rewriting the core type layer, all seven op macros, and the ROI test probes to use the new `DevicePtr::device_ptr(&self, stream)` → `(CUdeviceptr, SyncOnDrop)` signature. The NppStreamContext bridge must produce byte-identical golden output to the 0.9.15 baseline (Phase 2 spike is the oracle).

**Why standalone (F11, not bundled with F8.2):** cudarc 0.9.15's context-wide `synchronize` makes fence-absent tests mechanically infeasible (`docs/stream-context.md` finding, confirmed by F8 authors). 0.19.x exposes per-stream synchronize, per-stream copies (`CudaStream::clone_htod`/`clone_dtoh`), and `CudaStream::fork`/`join`/`wait` — all required for F8.2's async multi-stream chaining. The upgrade is a mechanical prerequisite that must land independently before any F8.2 work begins.

**CUDA 13 deferred:** cudarc 0.19.x supports `-F cuda-12090` through `-F cuda-1330` from the same crate version. This upgrade pins `cuda-12090` (matches the Nix shell's CUDA 12.9). A CUDA 13 upgrade is a separate later feature (one-flag-flip + golden re-pin).

**Key API changes (0.9.15 → 0.19.8):**

| Aspect | 0.9.15 | 0.19.8 |
|--------|--------|--------|
| `DevicePtr::device_ptr` | `(&self) -> &CUdeviceptr` (no stream) | `(&self, &CudaStream) -> (CUdeviceptr, SyncOnDrop<'a>)` |
| `DevicePtrMut::device_ptr_mut` | `(&mut self) -> &CUdeviceptr` (no stream) | `(&mut self, &CudaStream) -> (CUdeviceptr, SyncOnDrop<'a>)` |
| Host-to-device copy | `device.htod_sync_copy(data)` | `stream.clone_htod::<T>(data)?` (per-stream) |
| Device-to-host copy | `device.dtoh_sync_copy(buf)` | `stream.clone_dtoh::<T>(buf)?` (per-stream) |
| Device-to-host copy-into | `device.dtoh_sync_copy_into(src, dst)` | Use `stream.sync_copy_htod` or manual path |
| Stream creation | `device.fork_default_stream()` | `device.fork_default_stream()` (same name, new signature) |
| Stream synchronize | `result::stream::synchronize(stream.stream)` (unsafe) | `CudaStream::synchronize()` (safe method) |
| `CudaStream` thread safety | `!Send + !Sync` | `Send + Sync` (must re-impose `!Send + !Sync` via `PhantomData<*const ()>`) |
| `CudaDevice::new` | `CudaDevice::new(ordinal)` | `CudaDevice::new(ordinal)` (same name, may return new error type) |
| Device attribute | `device.attribute(attr)` | `device.attribute(attr)` (same name, may return `CudaResult` wrapper) |
| Event create/record/elapsed | `result::event::create(flags)`, `result::event::record(...)` | Same patterns but may live under different module paths |

**The `SyncOnDrop` binding guard is load-bearing:** In 0.19.x, `device_ptr` returns a `SyncOnDrop` that must live across the NPP FFI call. If the guard drops before NPP reads the buffer, the kernel races with the internal synchronize. This is a runtime-only bug (no compile error) — only GPU golden tests catch it. Every macro and ROI test must thread the guard to the same `let status = unsafe { ... }` scope.

**`CudaStream` becomes `Send + Sync`:** Without the `PhantomData<*const ()>` marker, `CudaImage` silently becomes `Send + Sync`, erasing the C7 thread-affinity invariant. The marker is re-imposed in `StreamContext` (match the 0.9.15 behaviour) and the rationale is documented in `stream.rs`.

**Sequence discipline:** F11 lands before F8.2. No F8.2 work is bundled. Phase 2 spike gates the entire upgrade — if the NppStreamContext bridge on 0.19.x does not produce byte-identical goldens, STOP and reassess (do not proceed to Phase 3).

### Key file map

| File | Role | Upgrade change |
|------|------|---------------|
| `npp/Cargo.toml:16` | cudarc dependency pin | `=0.9` → `=0.19.8`, add `cuda-12090` |
| `npp/src/cuda.rs` | `initialize_cuda_device` + C7 doc | `CudaDevice::new` return type update; context-lifetime doc |
| `npp/src/stream.rs` | `StreamContext`, `Event`, `populate_stream_context` | `CudaStream` type + Send/Sync; `synchronize()` API; `Result` error types; `Event` creation paths; `device_ptr(stream)` bridge |
| `npp/src/image.rs` | `CudaImage`, `CudaImageView`, `TryFrom` readbacks | `from_host`: `htod_sync_copy` → `stream.clone_htod`; `TryFrom`: `dtoh_sync_copy` → `stream.clone_dtoh`; `device_ptr` signature in view types |
| `npp/src/resize_macros.rs` | Resize engine + trait impl | `DevicePtr::device_ptr(&self.buf)` → `DevicePtr::device_ptr(&self.buf, stream)` + bind guard |
| `npp/src/swap_channels_macros.rs` | SwapChannels engine + trait impl | Same change |
| `npp/src/convert_macros.rs` | ConvertTo trait impl | Same change |
| `npp/src/convert_round_macros.rs` | ConvertRounded trait impl | Same change |
| `npp/src/convert_round_scaled_macros.rs` | ConvertRoundedScaled trait impl | Same change |
| `npp/src/mean_macros.rs` | Mean trait impl | Same change for src/scratch/out buffer extraction; `dtoh_sync_copy` → `stream.clone_dtoh` |
| `npp/src/normalize_macros.rs` | Normalize trait impl | Same change for MulC dst pointer |
| `npp/src/resize_roi_tests.rs` | In-crate ROI golden test | Manual `device_ptr`/`device_ptr_mut` extraction on lines 60, 62 |
| `npp/src/swap_channels_roi_tests.rs` | In-crate ROI golden test | Manual `device_ptr`/`device_ptr_mut` extraction on lines 66, 68 |
| `npp/tests/probe_resize_caps.rs` | GPU probe harness | Probe API paths (still calls raw FFI, but device init paths change) |
| `npp-sys/tests/stream_context_symbols.rs` | ABI tripwire | Re-verify after spike |
| `docs/roadmap.md` | Roadmap | Add F11 entry, CUDA 13 deferred entry, update F8.2 note |
| `docs/stream-context.md` | Stream model doc | Update synchronize semantics + Send/Sync reconciliation |

---

## Phase 0: Pre-flight verification (no code changes)

Commit message: `chore: verify Nix shell CUDA version and cudarc 0.19.x target agreement before F11 bump`

### Step 0.1 — Verify Nix shell CUDA version

Confirm that the CUDA toolkit provided by the Nix dev shell is version 12.9:

```bash
nix develop . --command bash -c 'echo "CUDA_PATH=$CUDA_PATH"; ls "$CUDA_PATH/include/cuda.h" 2>/dev/null && grep "define CUDA_VERSION" "$CUDA_PATH/include/cuda.h" || echo "cuda.h not found"'
nix develop . --command nvidia-smi 2>/dev/null | head -3 || echo "No NVIDIA driver (expected on non-GPU host)"
```

Record the reported CUDA version. It must be 12.9 to match the `cuda-12090` feature flag. If the Nix shell provides a different version, STOP — either update the flake inputs or adjust the feature flag, but do not proceed with a mismatch.

### Step 0.2 — Verify cudarc 0.19.x feature list

Fetch cudarc 0.19.x metadata and confirm `cuda-12090` is a valid feature:

```bash
nix develop . --command cargo search cudarc 2>/dev/null | head -5
# Or inspect locally:
nix develop . --command bash -c '
  mkdir -p /tmp/cudarc-check
  cd /tmp/cudarc-check
  cargo init --name cudarc-check 2>/dev/null
  cargo add cudarc@0.19.8 --dry-run 2>&1 | head -20
' 2>/dev/null || echo "Cannot verify remotely — check crates.io/cudarc/0.19.8 manually"
```

If `cuda-12090` is not a valid feature string (some versions use `cuda-12_0` format or similar), adjust the Cargo.toml pin accordingly. Record the exact feature string used.

### Step 0.3 — Confirm clean baseline

```bash
nix develop . --command cargo build -p npp-rs
nix develop . --command cargo test -p npp-rs
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo fmt --check
nix develop . --command cargo tarpaulin
```

All pass clean. Record the current golden test pass state on GPU (if available) for the Phase 6 delta comparison.

---

## Phase 1: Bump manifest pin

Commit message: `feat: bump cudarc from =0.9 to =0.19.8 with cuda-12090 feature`

### Step 1.1 — Edit `npp/Cargo.toml`

In `/home/vansweej/Work/npp-rs/npp/Cargo.toml`, line 16:

```toml
# Before:
cudarc = { version = "0.9", default-features = false, features = ["driver", "std"] }

# After:
cudarc = { version = "=0.19.8", default-features = false, features = ["driver", "std", "cuda-12090"] }
```

The `=` prefix pins the exact version (no range). `cuda-12090` explicitly selects CUDA 12.9 toolkit compatibility. This pin is the conscious upgrade decision — it must be changed deliberately on the next upgrade.

### Step 1.2 — Verify compilation breaks as expected

```bash
nix develop . --command cargo build -p npp-rs 2>&1 | head -40
```

Expected: compilation fails with a mapped set of errors. Each error should be one of:
- `DevicePtr::device_ptr(&self.buf)` — now expects a stream argument, but none is given.
- `device.htod_sync_copy(data)` — method deprecated or removed in favour of per-stream copy.
- `device.dtoh_sync_copy(&buf)` — same.
- `result::stream::synchronize(stream.stream)` — path/module changed.
- `result::event::create(...)` — path/module changed.
- `CudaDevice::new(ordinal)` — may return a different error type.
- `CudaDevice::attribute(...)` — may return a different wrapper.
- `CudaStream` field `stream` visibility or name change.

DO NOT try to fix any of these in Phase 1. The purpose is only to confirm the manifest change produces the expected class of compile errors. Record the error count and taxonomy for reference in Phase 3.

### Step 1.3 — Verify

```bash
nix develop . --command cargo clippy -- -D warnings 2>&1 | head -10
nix develop . --command cargo fmt --check
```

These may also fail due to the breakage — that is expected. The only goal is to confirm the manifest bump is mechanically correct (the file is valid TOML, the dep resolves).

---

## Phase 2: NppStreamContext bridge spike (🔑 GO/STOP gate)

Commit message: **(N/A — scratch branch, discarded after GO/STOP decision)**

**THIS IS A SPIKE.** Create a throwaway scratch branch from `main`. Port **only** the `NppStreamContext` bridge — `cuda.rs` init → `StreamContext::new` → `populate_stream_context` → `hStream` cast. Run `golden_chained_ctx.rs` on a GPU host. The spike branch is never merged — it is discarded after the GO/STOP decision.

### Step 2.1 — Create scratch branch

```bash
git checkout -b spike/cudarc-0.19-bridge
```

Apply **only** the Phase 1 manifest change (`Cargo.toml` bump). Do NOT port any macros or tests yet.

### Step 2.2 — Port `cuda.rs`

Edit `/home/vansweej/Work/npp-rs/npp/src/cuda.rs`:

Update imports and constructor to match cudarc 0.19.x API. The 0.19.x `CudaDevice::new` returns `Result<Arc<CudaDevice>, ...>` (the new error type). The function signature stays the same but the error path may change.

```rust
// Approximate 0.19.x pattern (verify against actual 0.19.8 API):
use cudarc::driver::CudaDevice;
use std::sync::Arc;

pub fn initialize_cuda_device(ordinal: usize) -> Result<Arc<CudaDevice>, NppError> {
    let dev = CudaDevice::new(ordinal)?;
    Ok(dev)
}
```

The error coercion (`?`) may need updating if `CudaDevice::new` returns a different error type. The C7 doc comment is unchanged.

### Step 2.3 — Port `stream.rs`

Edit `/home/vansweej/Work/npp-rs/npp/src/stream.rs`:

1. **Update imports** — reflect any module path changes in cudarc 0.19.x. `CudaStream` may now live in `cudarc::driver::stream::CudaStream` (or similar). `sys` module structure may differ.

2. **`StreamContext` struct** — the `stream: CudaStream` field is now `Send + Sync`. Add `_nosync: PhantomData<*const ()>` to the struct:

   ```rust
   pub struct StreamContext {
       device: Arc<CudaDevice>,
       stream: CudaStream,
       raw: NppStreamContext,
       // Re-impose !Send + !Sync. CudaStream became Send + Sync in 0.19.x,
       // but CUDA streams are thread-bound (C7). Without this marker,
       // CudaImage silently becomes Send + Sync.
       _nosync: PhantomData<*const ()>,
   }
   ```

3. **`StreamContext::new`** — `device.fork_default_stream()` may return `Result<CudaStream, ...>` or `CudaStream` directly. Adjust the `?` accordingly. The `populate_stream_context` call stays but the arguments may need adjusting.

4. **`populate_stream_context`** — the `stream.stream` field may not be directly accessible. Check whether cudarc 0.19.x exposes a method like `stream.as_inner()` or `stream.handle()`. The cross-crate opaque pointer cast (`stream.stream as npp_sys::cudaStream_t`) is the riskiest line — this must compile on 0.19.x with the same semantic cast.

   If `stream.stream` is no longer a public field, use the accessor:
   ```rust
   let h_stream: npp_sys::cudaStream_t = stream.handle() as npp_sys::cudaStream_t;
   // or similar — verify against 0.19.8 API
   ```

5. **`StreamContext::synchronize`** — replace the `unsafe { result::stream::synchronize(stream.stream) }` with the safe method on `CudaStream`:
   ```rust
   pub fn synchronize(&self) -> Result<(), DriverError> {
       self.stream.synchronize()
   }
   ```

6. **`StreamContext::device_fence`** — `self.device.wait_for(&self.stream)` — check if the method signature changed. It may now accept `&CudaStream` differently.

7. **`record_event`/`elapsed`/`Event`** — update event creation path. If `result::event::create` moved to a different module path, update the call. The `stream.stream` field access for recording also needs updating.

8. **Update `stream_context_for`** — no signature change, but the `Arc::new(StreamContext::new(device)?)` chain must compile with the updated constructor.

**Do NOT update `image.rs` or any macros.** The spike only validates the bridge — those come in Phase 3+.

### Step 2.4 — Port `image.rs` minimally (bridge deps only)

Edit `/home/vansweej/Work/npp-rs/npp/src/image.rs` only enough to make `CudaImage` constructible:

1. **`CudaImage::new`** — `ctx.device().alloc_zeros::<T>(num_elements)` — check if `alloc_zeros` signature changed on `CudaDevice`. It may need a stream argument in 0.19.x.

2. **`CudaImage::from_host`** — `ctx.device().htod_sync_copy(data)` — this MUST be ported to a per-stream copy in 0.19.x:
   ```rust
   let buf = ctx.stream().clone_htod::<T>(data)?;
   // Verify method name against 0.19.8 — may be sync_copy_htod or clone_htod
   ```

3. **`TryFrom<&CudaImage>`** — `ctx.device().dtoh_sync_copy(&img.buf)` — port to per-stream copy. The synchronize fence stays.

4. **`TryFrom<&CudaImageView>`** — `ctx.device().dtoh_sync_copy_into(&v.view, &mut host)` — port to per-stream.

**Do NOT touch `CudaImageView::device_ptr()`** or the macro files. The spike does not need to call any NPP op.

### Step 2.5 — Compile-check the spike

```bash
nix develop . --command cargo build -p npp-rs 2>&1
```

Target: only `golden_chained_ctx.rs` uses the bridge. Expected: the bridge compiles but the resize/swap_channels symbols that `golden_chained_ctx.rs` calls through the macros may not compile (macro files haven't been ported yet).

If the bridge itself does not compile, fix it — this is the entire point of the spike. Record the exact API shape that works.

### Step 2.6 — Run `golden_chained_ctx.rs` on GPU (GO/STOP)

```bash
nix develop . --command cargo test --features gpu --test golden_chained_ctx -p npp-rs 2>&1
```

If the macro-level code does not compile yet (Step 2.5 expected), this test cannot run. The spike oracle is **byte-identical golden output** — the test input/output is already pinned at line 70 of `golden_chained_ctx.rs`. The result must be:

- **GO (proceed):** The test produces the EXACT same bytes as the pinned `EXPECTED` at line 70 (the 12-byte slice `[0, 0, 255, 255, 0, 0, 0, 0, 255, 255, 0, 0]`). If the bridge is correct, the output bytes are unchanged from 0.9.15 because NPP behaviour is unchanged — only the Rust plumbing changed.
- **STOP (reassess):** Any value drift in the output bytes, any ABI mismatch (segfault, `CUDA_ERROR_INVALID_HANDLE`), or any `NppStreamContext` field that the 0.19.x `CudaStream` handle does not correctly populate. This indicates the bridge's pointer cast or attribute query produces a semantically different context. STOP and investigate before any further work.

### Step 2.7 — Record spike finding

Regardless of GO or STOP, record:

- Exact cudarc version (e.g. `0.19.8`)
- Exact `CudaStream` handle accessor path (e.g. `stream.handle()`, `stream.as_inner()`, or `stream.stream` if still a public field)
- Exact `DevicePtr`/`DevicePtrMut` import paths for Phase 4
- Exact per-stream copy method names (`clone_htod`, `sync_copy_htod`, etc.)
- Exact per-stream readback method names
- `CudaDevice::alloc_zeros` signature (does it need a stream now?)
- Any error type changes
- The duck-typed API shape for `device.attribute()` (does it return `i32` still, or `Result<i32, ...>`?)

### Step 2.8 — Discard scratch branch

```bash
git checkout main
git branch -D spike/cudarc-0.19-bridge
```

The spike branch is gone. The finding record (Step 2.7) is carried forward into Phase 3 as the authoritative API map.

---

## Phase 3: Port core types — cuda.rs, stream.rs, image.rs

Commit message: `refactor: port cuda.rs, stream.rs, image.rs to cudarc 0.19.x API`

Apply the Phase 1 manifest bump (line 16) plus the Phase 2 spike findings. This phase produces compilable core types.

### Step 3.1 — Port `cuda.rs`

Apply the changes verified in Phase 2 Step 2.2. Update the `CudaDevice::new` call to match 0.19.x return type. Update error coercion if the error type changed. The `#[cfg(not(tarpaulin_include))]` annotation stays.

Add a doc comment referencing the explicit CUDA pin:

```rust
/// NOTE: The cuda-12090 feature flag in Cargo.toml selects CUDA 12.9
/// specifically (Phase 0 verification). See docs/f11-plan.md.
```

### Step 3.2 — Port `stream.rs`

Apply the changes verified in Phase 2 Step 2.3:

1. Add `PhantomData<*const ()>` to `StreamContext` for `!Send + !Sync`.
2. Update `populate_stream_context` — replace `stream.stream` direct field access with the 0.19.x accessor (e.g. `stream.handle()`).
3. Replace `unsafe { result::stream::synchronize(stream.stream) }` with `self.stream.synchronize()`.
4. Update event creation paths — `result::event::create(...)` may be at a different module path. Use the spike finding.
5. Update `Event` struct — `stream: sys::CUstream` field may need a different accessor path.
6. Update `Event::record` and `Event::drop` for the new module paths.
7. Update `device_fence` — verify `self.device.wait_for(&self.stream)` still compiles.
8. Update doc comments referencing the old 0.9.15 NULL-stream readback path to reflect 0.19.x per-stream copies.

### Step 3.3 — Port `image.rs` constructors and readbacks

Apply the changes verified in Phase 2 Step 2.4:

1. **`CudaImage::new`** — update `alloc_zeros` if the signature changed. If it now requires a stream, pass `ctx.stream()`.
2. **`CudaImage::from_host`** — replace `ctx.device().htod_sync_copy(data)` with the 0.19.x per-stream copy method:
   ```rust
   let buf = ctx.stream().clone_htod::<T>(data)?;
   // exact method name from spike finding (Step 2.7)
   ```
3. **`TryFrom<&CudaImage<T>> for Vec<T>`** — replace `ctx.device().dtoh_sync_copy(&img.buf)`:
   ```rust
   let host: Vec<T> = ctx.stream().clone_dtoh::<T>(&img.buf)?;
   // exact method name from spike finding
   ```
   The synchronize fence (Step 1) stays — it is the load-bearing barrier before the per-stream readback.
4. **`TryFrom<&CudaImageView<'a, T>> for Vec<T>`** — replace `ctx.device().dtoh_sync_copy_into(&v.view, &mut host)` with per-stream equivalent:
   ```rust
   v.ctx.synchronize()?;
   let len = cudarc::driver::DeviceSlice::len(&v.view);
   let mut host: Vec<T> = vec![unsafe { std::mem::zeroed::<T>() }; len];
   v.ctx.stream().clone_dtoh_into(&v.view, &mut host)?;
   // exact method name from spike finding
   ```

**Do NOT touch `CudaImageView::device_ptr()` or `CudaImageViewMut::device_ptr_mut()`** yet — those use `DevicePtr::device_ptr(&self.view)` and need a stream argument. They are ported in Phase 4 alongside the macros.

### Step 3.4 — Verify core types compile

```bash
nix develop . --command cargo build -p npp-rs 2>&1
```

Expected: `cuda.rs`, `stream.rs`, and `image.rs` compile. The macro-generated files and their includes fail because the macro files haven't been ported yet. The `CudaImageView::device_ptr()` calls in the ROI tests may also fail (they use the old `DevicePtr::device_ptr(&self.view)` signature).

Target error: only the macro layer and ROI tests. If any core type file fails, fix it before proceeding.

```bash
nix develop . --command cargo clippy -- -D warnings 2>&1 | grep -v "generated.rs" | head -20
nix develop . --command cargo fmt --check
```

### Step 3.5 — Verify pure-logic tests pass

```bash
nix develop . --command cargo test -p npp-rs 2>&1
```

Pure-logic tests (`validate_dims`, `npp_stream_context_size`, `npp_stream_context_is_copy`, `raw_ctx_returns_by_value`, `test_ms_to_duration_conversion`) must pass.

---

## Phase 4: Port all seven `*_macros.rs` to `device_ptr(stream)` + `SyncOnDrop` form, plus view types

Commit message: `refactor: port all op macros to device_ptr(stream, stream) form with SyncOnDrop guard`

Update all seven macro files plus `image.rs` view types to use the new `DevicePtr::device_ptr(&self, &CudaStream) -> (CUdeviceptr, SyncOnDrop)` signature. The `SyncOnDrop` guard MUST outlive the NPP FFI call.

### Step 4.1 — Port `resize_macros.rs`

In `/home/vansweej/Work/npp-rs/npp/src/resize_macros.rs`, lines 143–155:

The current code (the `impl Resize for CudaImage<$rust_ty>` block at line 138+):

```rust
let src_ptr = *cudarc::driver::DevicePtr::device_ptr(&self.buf) as *const $rust_ty;
let dst_ptr = *cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut $rust_ty;
```

Port to 0.19.x:

```rust
let src_stream = &self.ctx.stream();
let (src_ptr_u64, _src_guard) = cudarc::driver::DevicePtr::device_ptr(&self.buf, src_stream);
let src_ptr = src_ptr_u64 as *const $rust_ty;

let dst_stream = &self.ctx.stream();
let (dst_ptr_u64, _dst_guard) = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf, dst_stream);
let dst_ptr = dst_ptr_u64 as *mut $rust_ty;
```

The `_src_guard` and `_dst_guard` (`SyncOnDrop`) variables must NOT be dropped before the `$fn_name(...)` call — they keep the bindings alive. The call site is:

```rust
$fn_name(
    src_ptr, src_step_bytes, self.width(), self.height(), self.channels(),
    dst_ptr, dst_step_bytes, dst.width(), dst.height(),
    inter, self.ctx.raw_ctx(),
)
```

The `_src_guard` and `_dst_guard` are alive through this call because they are declared in the same block. This is the load-bearing pattern.

If the method signature differs from what is described here (verified in Phase 2 spike), use the exact API shape found in Step 2.7.

**Do NOT change** the engine function signature (`$fn_name` parameters) — only the trait impl's pointer extraction.

### Step 4.2 — Port `swap_channels_macros.rs`

Same pattern as Step 4.1. The `impl_swap_channels_for!` macro's trait impl block (lines 129–145 approximately):

```rust
// Before:
let src_ptr = *cudarc::driver::DevicePtr::device_ptr(&self.buf) as *const $rust_ty;
let dst_ptr = *cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut $rust_ty;

// After:
let src_stream = &self.ctx.stream();
let (src_ptr_u64, _src_guard) = cudarc::driver::DevicePtr::device_ptr(&self.buf, src_stream);
let src_ptr = src_ptr_u64 as *const $rust_ty;

let dst_stream = &self.ctx.stream();
let (dst_ptr_u64, _dst_guard) = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf, dst_stream);
let dst_ptr = dst_ptr_u64 as *mut $rust_ty;
```

### Step 4.3 — Port `convert_macros.rs`

In `/home/vansweej/Work/npp-rs/npp/src/convert_macros.rs`, lines 97–101:

```rust
// Before:
let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
let src_ptr = *src_base as *const $src_ty;

let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
let dst_ptr = *dst_base as *mut $dst_ty;

// After:
let src_stream = &self.ctx.stream();
let (src_ptr_u64, _src_guard) = cudarc::driver::DevicePtr::device_ptr(&self.buf, src_stream);
let src_ptr = src_ptr_u64 as *const $src_ty;

let dst_stream = &self.ctx.stream();
let (dst_ptr_u64, _dst_guard) = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf, dst_stream);
let dst_ptr = dst_ptr_u64 as *mut $dst_ty;
```

### Step 4.4 — Port `convert_round_macros.rs`

Same pattern — the `impl_convert_rounded_for!` macro has the same `device_ptr`/`device_ptr_mut` extraction with `_src_guard`/`_dst_guard`.

### Step 4.5 — Port `convert_round_scaled_macros.rs`

In `/home/vansweej/Work/npp-rs/npp/src/convert_round_scaled_macros.rs`, lines 122–127:

Same pattern as Step 4.3. The `impl_convert_rounded_scaled_for!` macro's pointer extraction at lines 123–127 gets the `(stream)` argument + `SyncOnDrop` guard.

### Step 4.6 — Port `mean_macros.rs`

In `/home/vansweej/Work/npp-rs/npp/src/mean_macros.rs`, there are **four** pointer extractions to port:

1. **Source buffer** (line 97–98):
   ```rust
   let src_base = cudarc::driver::DevicePtr::device_ptr(&self.buf);
   let src_ptr = *src_base as *const $rust_ty;
   ```
2. **Scratch buffer** (line 100–102):
   ```rust
   let base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut scratch_buf);
   ```
3. **Output buffer** (line 104–106):
   ```rust
   let base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut out_buf);
   ```
4. **DtoH readback** (line 131):
   ```rust
   let result: Vec<f64> = self.ctx.device().dtoh_sync_copy(&out_buf)?;
   ```
   → per-stream:
   ```rust
   let result: Vec<f64> = self.ctx.stream().clone_dtoh::<f64>(&out_buf)?;
   ```

All three `SyncOnDrop` guards (src, scratch, out) must live until after the NPP mean call (Step 4 at line 109). The readback guard is not needed because `clone_dtoh` is an immediate synchronous copy.

### Step 4.7 — Port `normalize_macros.rs`

In `/home/vansweej/Work/npp-rs/npp/src/normalize_macros.rs`, the MulC pointer extraction at lines 110–111:

```rust
let dst_base = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf);
let dst_ptr = *dst_base as *mut f32;
```

Port to:

```rust
let dst_stream = &dst.ctx.stream();
let (dst_ptr_u64, _dst_guard) = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf, dst_stream);
let dst_ptr = dst_ptr_u64 as *mut f32;
```

The `_dst_guard` must live across the `match dst.channels()` FFI dispatch (lines 120–162).

### Step 4.8 — Port `CudaImageView::device_ptr()` and `CudaImageViewMut::device_ptr_mut()`

In `/home/vansweej/Work/npp-rs/npp/src/image.rs`, lines 403–406 and 459–462:

**`CudaImageView::device_ptr`** (line 403):

```rust
// Before:
pub(crate) fn device_ptr(&self) -> *const T {
    let cu_ptr = cudarc::driver::DevicePtr::device_ptr(&self.view);
    *cu_ptr as *const T
}

// After:
pub(crate) fn device_ptr(&self) -> *const T {
    let stream = self.ctx.stream();
    let (cu_ptr, _guard) = cudarc::driver::DevicePtr::device_ptr(&self.view, stream);
    cu_ptr as *const T
}
```

**IMPORTANT:** The `_guard` lives in this function scope. It is alive through the NPP call **only if the caller captures the return value and passes it immediately to NPP**. This works because `device_ptr()` is called just-in-time in the ROI tests (lines `resize_roi_tests.rs:60` and `swap_channels_roi_tests.rs:66`), and the guard lives until the end of the enclosing function. Verify this in Phase 5.

**`CudaImageViewMut::device_ptr_mut`** (line 459):

```rust
// Before:
pub(crate) fn device_ptr_mut(&mut self) -> *mut T {
    let cu_ptr = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut self.view);
    *cu_ptr as *mut T
}

// After:
pub(crate) fn device_ptr_mut(&mut self) -> *mut T {
    let stream = self.ctx.stream();
    let (cu_ptr, _guard) = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut self.view, stream);
    cu_ptr as *mut T
}
```

### Step 4.9 — Verify compilation

```bash
nix develop . --command cargo build -p npp-rs 2>&1
```

Expected: all files compile. The `generated.rs` include files may still cause issues if the generated invocation format changed (but the macro signatures themselves didn't change — only the impl body inside the macro changed, so generated invocations are unchanged).

If any macro has a syntax error, fix it. Do NOT proceed if there are compile errors.

```bash
nix develop . --command cargo clippy -- -D warnings 2>&1
nix develop . --command cargo fmt --check
nix develop . --command cargo test -p npp-rs
```

Pure-logic tests pass. The `golden_*` tests may fail to compile if they reference the old `stream_context_for` path or old import paths — those are fixed in Phase 5.

---

## Phase 5: Port ROI tests and probe

Commit message: `test: update ROI tests and probe_resize_caps for cudarc 0.19.x`

Port the `resize_roi_tests.rs`, `swap_channels_roi_tests.rs`, and `probe_resize_caps.rs` to the new API. ROI tests need `DevicePtrMut::device_ptr_mut` on `dst.buf` (not via a view) to use the new signature.

### Step 5.1 — Port `resize_roi_tests.rs`

In `/home/vansweej/Work/npp-rs/npp/src/resize_roi_tests.rs`, lines 60–63:

The `src_ptr` extraction (line 60) calls `view.device_ptr()` which was already ported in Phase 4.8 — no change needed there.

The `dst_ptr` extraction (line 62):
```rust
let dst_ptr = *cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut u8;
```

Port to:
```rust
let stream = ctx.stream();
let (dst_ptr_u64, _guard) = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf, stream);
let dst_ptr = dst_ptr_u64 as *mut u8;
```

The `_guard` lives across the `resize_into_8u(...)` call on line 65.

### Step 5.2 — Port `swap_channels_roi_tests.rs`

In `/home/vansweej/Work/npp-rs/npp/src/swap_channels_roi_tests.rs`, lines 66–68:

Same pattern as Step 5.1. `view.device_ptr()` on line 66 is already ported. `dst_ptr` on line 68:

```rust
// Before:
let dst_ptr = *cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf) as *mut u8;

// After:
let stream = ctx.stream();
let (dst_ptr_u64, _guard) = cudarc::driver::DevicePtrMut::device_ptr_mut(&mut dst.buf, stream);
let dst_ptr = dst_ptr_u64 as *mut u8;
```

### Step 5.3 — Port `probe_resize_caps.rs`

Examine `/home/vansweej/Work/npp-rs/npp/tests/probe_resize_caps.rs`. This test initialises a device and calls raw NPP FFI functions through `npp_sys` to probe type-mode support. It likely uses `initialize_cuda_device` or `CudaDevice::new` directly.

Port device init to the 0.19.x API if needed. The probe itself does not use `DevicePtr`/`DevicePtrMut` — it calls NPP status functions with NULL pointers to probe error codes. No `SyncOnDrop` guards needed for NULL-pointer probes.

If the probe uses `alloc_zeros` to create test buffers, update that call to match the new signature (stream argument if required).

### Step 5.4 — Port `golden_chained_ctx.rs` if needed

The chained golden test at `/home/vansweej/Work/npp-rs/npp/tests/golden_chained_ctx.rs` may need import path updates. Specifically:
- `stream_context_for` is unchanged (core type, already ported in Phase 3)
- `CudaImage::from_host` was already ported in Phase 3
- `TryFrom<&CudaImage>` was already ported in Phase 3
- `bgra_to_rgb` and `resize` go through the macros ported in Phase 4

Likely no changes needed — the Phase 2 spike already validated this end-to-end.

### Step 5.5 — Verify compilation

```bash
nix develop . --command cargo build -p npp-rs 2>&1
nix develop . --command cargo clippy -- -D warnings 2>&1
nix develop . --command cargo fmt --check
nix develop . --command cargo test -p npp-rs
```

All pure-logic tests pass. All ROI tests and probe tests compile (they are gated behind `#[cfg(feature = "gpu")]` or `#[cfg(all(test, feature = "gpu"))]` and won't run without the feature).

---

## Phase 6: GPU golden suite re-pin (manual GPU lane)

Commit message: `test: re-pin golden tests and verify byte-identity on cudarc 0.19.x`

**THIS PHASE REQUIRES A GPU HOST.** Every golden test, ROI test, and probe test must be run on a GPU. Byte-identity must be verified against the 0.9.15 baseline. Any value delta is a stop-and-investigate.

### Step 6.1 — Re-pin `golden_chained_ctx.rs`

```bash
nix develop . --command cargo test --features gpu --test golden_chained_ctx -p npp-rs 2>&1
```

This test's `EXPECTED` is already pinned (the 12-byte slice at line 70). The test must pass without changes. If it fails with a byte mismatch, it means the NppStreamContext bridge produces different device output on 0.19.x — STOP and investigate. Record the delta.

### Step 6.2 — Run all golden tests

```bash
nix develop . --command cargo test --features gpu -p npp-rs --test golden_resize 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_resize_16s 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_resize_16u 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_resize_32f 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_convert 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_convert_16u32f 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_convert_8u16u 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_convert_round_32f8u 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_convert_round_chained 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_convert_round_scaled 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_swap_channels 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_mean 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_normalize 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_normalize_16u32f 2>&1
nix develop . --command cargo test --features gpu -p npp-rs --test golden_normalize_16s32f 2>&1
```

Each test must pass with the existing pinned `EXPECTED` values. If any test fails with a byte mismatch, the `SyncOnDrop` guard lifetime in that macro is wrong, or the per-stream copy semantics differ from 0.9.15's synchronous copies. STOP and investigate — do NOT just re-pin the bytes unless you have confirmed the guard lifetimes are correct.

**IMPORTANT:** Byte-identical golden output is the evidence that the `SyncOnDrop` guards are correct. If all goldens match the 0.9.15 baseline, the port is sound. If ANY golden drifts, the runtime behaviour changed and the guard pattern must be re-audited.

### Step 6.3 — Run ROI tests

```bash
nix develop . --command cargo test --features gpu -p npp-rs resize_roi_tests 2>&1
nix develop . --command cargo test --features gpu -p npp-rs swap_channels_roi_tests 2>&1
```

Must pass with pinned `EXPECTED` values. Same stop-and-investigate rule as Step 6.2.

### Step 6.4 — Run probe test

```bash
nix develop . --command cargo test --features gpu -p npp-rs --test probe_resize_caps 2>&1
```

This test probes device capabilities and does not have pinned output. It must succeed (no crash, no `CUDA_ERROR`). The probed capability matrix should match the 0.9.15 probe result (same CUDA version).

### Step 6.5 — Run full GPU test suite

```bash
nix develop . --command cargo test --features gpu -p npp-rs 2>&1
```

All GPU-gated tests pass.

### Step 6.6 — Verify npp-sys ABI tripwire

```bash
nix develop . --command cargo test --features gpu -p npp-sys 2>&1
```

The `stream_context_symbols.rs` ABI tripwire in `npp-sys/tests/` must pass — it verifies that `NppStreamContext` struct layout and symbol availability match expectations.

### Step 6.7 — Run smoke benches (timing only, no correctness assert)

```bash
nix develop . --command cargo bench --features gpu -p npp-rs 2>&1 | head -80
```

Benches must execute (warm-up + timing iterations) without crashing. Timing results are not compared — this is a smoke test that the bench infrastructure works on 0.19.x.

---

## Phase 7: Documentation reconciliation

Commit message: `docs: update roadmap, stream-context doc, and AGENTS.md for F11`

### Step 7.1 — Add F11 entry to `docs/roadmap.md`

Insert a new F11 section after the existing features (after F8.2, before F9). Use the same format as the roadmap entries:

```markdown
## F11 — cudarc 0.9.15 → 0.19.x upgrade *(complete)*

**What:** Upgrade the cudarc dependency from `=0.9` to `=0.19.8` with the
`cuda-12090` feature flag (CUDA 12.9). This is a mechanical prerequisite for
F8.2 (async multi-stream chaining), which requires 0.19.x's per-stream
synchronize, per-stream copies (`clone_htod`/`clone_dtoh`), and
`CudaStream::fork`/`join`/`wait` primitives. The upgrade is standalone — CUDA
13 is deferred.

**Scope of changes:**

| Layer | Change |
|-------|--------|
| `Cargo.toml` | `=0.9` → `=0.19.8`, add `cuda-12090` |
| `cuda.rs` | `CudaDevice::new` error type update, doc fix |
| `stream.rs` | `CudaStream` API (safe `synchronize()`, `handle()` accessor), `!Send + !Sync` marker via `PhantomData<*const ()>`, `Event` creation paths |
| `image.rs` | `from_host`: `htod_sync_copy` → `stream.clone_htod`; `TryFrom` readbacks: `dtoh_sync_copy` → `stream.clone_dtoh` |
| `*_macros.rs` (7 files) | `DevicePtr::device_ptr(&self, stream)` → `(CUdeviceptr, SyncOnDrop)` with load-bearing guard lifetime |
| View types | `CudaImageView::device_ptr` / `CudaImageViewMut::device_ptr_mut` same `(stream)` signature |
| ROI tests | `copy_channels_roi_tests.rs`, `resize_roi_tests.rs` — manual pointer extractions ported |
| `probe_resize_caps.rs` | Device init paths updated |

**Key design decisions:**
- **Pinned version** (`=0.19.8`), not a range — no lockfile (gitignored), so pinning is the only reproducibility mechanism.
- **Explicit CUDA feature** (`cuda-12090`) — never `cuda-version-from-build-system`. CUDA version selection is a conscious decision.
- **`!Send + !Sync` re-imposed** on `StreamContext` via `PhantomData<*const ()>` — cudarc 0.19.x `CudaStream` became `Send + Sync`, which would silently relax the C7 thread-affinity invariant without the marker.
- **`SyncOnDrop` guards are load-bearing** — the guard returned by `device_ptr(stream)` must live across the NPP FFI call. This is a runtime-only correctness condition; only GPU golden tests catch it.

**Dependencies:** M1 (clean cudarc-based build), F8 core (the `StreamContext` and `_Ctx` pivot that this upgrade ports).

---

## F11.1 — CUDA 13 upgrade *(deferred)*

**What:** Flip the feature flag from `cuda-12090` to `cuda-1300` (or whatever
the exact feature string is), update the Nix shell to provide CUDA 13, and
re-pin all goldens. Expected to be a one-flag-flip plus golden re-pin.

**Why deferred:** Not blocked — cudarc 0.19.x supports `cuda-1300` through
`cuda-1330` from the same crate version. The deferral is intentional: CUDA 12.9
is the current nixpkgs version and the upgrade has no user-facing benefit for
the MTCNN consumer.
```

Also update the roadmap's "Suggested rough sequencing" diagram to include F11:

```
F8 (streams/execution context) [DONE]
...
└─> F8.2 (async multi-stream chaining) [blocked on F11]
F11 (cudarc upgrade) [DONE]
├─> F11.1 (CUDA 13) [deferred]
└─> F8.2 (async multi-stream chaining) [blocked on F11]
```

### Step 7.2 — Update `docs/stream-context.md`

The stream model document must be updated for 0.19.x semantics:

1. **Readback path** — replace all references to "NULL-stream DtoH copy via `dtoh_sync_copy`" with "per-stream DtoH copy via `stream.clone_dtoh`". The fence (`synchronize()`) before readback is still required, but the copy now happens on the same stream as the NPP ops (not the NULL stream).

2. **`!Send + !Sync` justification** — add a section noting that `CudaStream` became `Send + Sync` in cudarc 0.19.x, and that `StreamContext` explicitly re-imposes `!Send + !Sync` via `PhantomData<*const ()>` to preserve the C7 thread-affinity invariant. Reference the `PhantomData` type and explain why CUDA streams are thread-bound.

3. **Update the "NULL stream" section** — in cudarc 0.9, the only DtoH API was NULL-stream-only. On 0.19.x, `clone_dtoh` operates on the caller's stream, so the host-fenced NULL-stream rationale no longer applies. The fence is still required (it synchronises the stream before the per-stream readback), but the two-path concern is eliminated.

4. **Update the F8.2 deferred note** — cross-reference F11 as the prerequisite that unblocks F8.2. State that per-stream copies (`clone_htod`/`clone_dtoh`) and per-stream synchronize are prerequisites for multi-stream chaining, and both are now available.

### Step 7.3 — Update `docs/spike-cudarc-ptr-bridge.md`

This document (the authoritative pointer-bridge pattern) must be updated for 0.19.x:

1. Replace the 0.9.x `DevicePtr::device_ptr(&buf)` pattern with the 0.19.x `device_ptr(&buf, &stream)` → `(CUdeviceptr, SyncOnDrop)` form.
2. Add a warning about `SyncOnDrop` guard lifetime — the guard must outlive the FFI call.
3. Update the host upload/readback examples to use per-stream copies.
4. Update the error type section (cudarc 0.19.x may use different error types).
5. Update the example to show the stream being passed explicitly.

### Step 7.4 — Update `npp/src/lib.rs` doc comments

Update any public-API doc comments in `/home/vansweej/Work/npp-rs/npp/src/lib.rs` that reference the old cudarc version or the old stream model:

- Line referencing `Arc<CudaDevice>` — update if the type path changed.
- Any doc claiming "readback uses the NULL stream" — update to "per-stream readback".
- Any doc referencing the cudarc version — update to `0.19.x`.

### Step 7.5 — Update F8.2 entry in roadmap

The current F8.2 entry says "**Dependencies:** F8 (core)." Update to:

```markdown
**Dependencies:** F8 (core stream abstraction, `_Ctx` pivot) and **F11**
(cudarc 0.19.x, without which per-stream copies and per-stream synchronize
are unavailable). F8.2 is blocked until F11 lands.
```

---

## Phase 8: Final verification gate

Commit message: `chore: final verification — build, test, clippy, fmt, doc, tarpaulin`

### Step 8.1 — Full build sweep

```bash
nix develop . --command cargo build
nix develop . --command cargo build --features gpu -p npp-rs
```

Both succeed. The non-GPU build must not initialise any CUDA device.

### Step 8.2 — Pure-logic test sweep

```bash
nix develop . --command cargo test
nix develop . --command cargo test -p npp-sys
```

All pure-logic tests pass. Zero GPU code runs.

### Step 8.3 — Clippy + fmt

```bash
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo clippy --features gpu -p npp-rs -- -D warnings
nix develop . --command cargo fmt --check
```

Both clippy invocations pass (without and with `gpu` feature — the `gpu` feature compiles ROI test modules that are otherwise unchecked).

### Step 8.4 — Doc check

```bash
nix develop . --command cargo doc --no-deps -p npp-rs -p npp-sys
```

No broken intra-doc links. This is a non-blocking check (doc warnings are acceptable).

### Step 8.5 — Coverage

```bash
nix develop . --command cargo tarpaulin
```

≥90 % coverage. GPU-only functions are annotated `#[cfg(not(tarpaulin_include))]` and excluded. New pure-logic code (if any was added) counts toward coverage.

---

## Consolidated risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | `SyncOnDrop` guard dropped before NPP FFI call — runtime data race, no compile error | Medium | Silent wrong pixel output, intermittent garbage | Phase 2 spike catches this (non-identical golden = STOP); Phase 6 re-confirms with full golden suite; guard pattern is load-bearing and explicitly documented |
| R2 | `CudaStream::handle()` or accessor does not exist in 0.19.8 — ABI cast broken | Medium | Bridge does not compile, or compiles with wrong handle | Phase 2 spike is the oracle — the exact accessor path is recorded in Step 2.7 and carried forward to Phase 3 |
| R3 | `CudaDevice::alloc_zeros` requires a stream argument in 0.19.x | Medium | `CudaImage::new` fails to compile | Phase 2 spike validates this; Phase 3 ports the call accordingly |
| R4 | per-stream copy methods (`clone_htod`, `clone_dtoh`) have different names or return types than expected | High | Phase 3 and 4 compile failures | Phase 2 spike records the exact method signatures; all macro copies follow that single pattern |
| R5 | `CudaStream::synchronize()` returns a different error type than `result::stream::synchronize` | Low | Error type mismatch in `StreamContext::synchronize` | Phase 2 spike records the exact return type; the `Result<(), DriverError>` signature is adjusted accordingly |
| R6 | Golden test output drifts between 0.9.15 and 0.19.x due to NPP behaviour change (not bridge bug) | Low | False STOP in Phase 6 | Extremely unlikely — the NPP library version hasn't changed, only the Rust plumbing. If drift occurs, re-run goldens on the 0.9.15 baseline on the same GPU to confirm NPP behaviour is consistent. If NPP is identical, the bridge is the culprit |
| R7 | `tarpaulin` coverage drops because new `#[cfg(not(tarpaulin_include))]` annotations are needed | Low | Phase 8 fails | GPU-only code (`stream.rs`, `image.rs` constructors, all macros) are already excluded. New code in this upgrade is a rewrite of existing GPU-only code — no new coverage issue expected |
| R8 | `CudaStream` becomes `Send + Sync` — without re-imposing `!Send + !Sync`, `CudaImage` silently becomes `Send + Sync` | Low (if the marker is missed) | Thread-safety invariant silently relaxed (C7 violation) | Step 3.2 explicitly adds `PhantomData<*const ()>`; Phase 7 updates docs; code review catches the absence. This is a compile-time invariant, not a runtime bug |
| R9 | `alloc_zeros` may not zero memory on 0.19.x (semantic change) | Very low | Uninitialized data in output buffers | `ValidAsZeroBits` trait is still in the `NppPixelType` bound; golden tests catch uninitialized output because pinned expected values would mismatch. However, if ALL pixels happen to be zero-initialized, this could slip through. Mitigation: `alloc_zeros` contract is unchanged — it allocates zeroed memory in both versions |

## Dependencies

- **Phase 0:** Clean baseline on `main` (build, test, clippy, fmt, tarpaulin all pass; golden suite green on GPU).
- **Phase 1:** Phase 0 (baseline to confirm pre-existing state) + CUDA 12.9 in Nix shell + cudarc 0.19.x published.
- **Phase 2 (spike):** Phase 1 (manifest bump). Requires a GPU host. The spike is a throwaway branch — it can start from the Phase 1 state on a scratch branch, but the F11 branch unwinds to the same manifest bump.
- **Phase 3:** Phase 1 (manifest bump) + Phase 2 findings (spike API map). The core types must compile before the macros can use them.
- **Phase 4:** Phase 3 (core types ported and compiling). The macros use the stream reference and guard pattern that depends on the ported `StreamContext`.
- **Phase 5:** Phase 4 (macros and view types ported). ROI tests use the ported macros and view types.
- **Phase 6:** Phase 5 (everything compiles). Requires a GPU host. This is the re-pin gate.
- **Phase 7:** All previous phases complete (docs reflect the shipped state).
- **Phase 8:** Phase 7 (docs reconciled). No new code — final verification.
