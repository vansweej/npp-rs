# Feature: F8 — Stream / execution-context model (C8 + C7)

## Session 2: Wire `Arc<StreamContext>` onto images, pivot ops to `_Ctx`

Overall F8 spans three sessions:
1. **Session 1 (done):** `StreamContext` abstraction landed, `npp-sys` allowlist
   widened, `_Ctx` ABI facts pinned (`stream_context_symbols.rs`), readback
   redesign validated against cudarc 0.9.15.
2. **Session 2 (this plan):** Wire `Arc<StreamContext>` onto `CudaImage`, pivot
   Resize/SwapChannels/Mean to `_Ctx` variants with host-fenced NULL-stream
   readback.
3. **Session 3 (future):** Upsize tests/benches, optional async chaining.

### Constraint notes

- **Session-end-green only** — intermediate commits may be red. Only Phase 6
  must be green + GPU-validated. The Mean readback (`self.device.dtoh_sync_copy`
  at `mean_macros.rs:120`) is not pre-moved into Phase 2 (Finding A skipped).
- **Two-op smoke test (C1)** — not a fence regression guard. cudarc 0.9's
  context-wide `synchronize()` makes fence-absent tests mechanically
  infeasible. The C1 test is reshaped to `Op1→Op2→single readback`.
- **Target NPP 12.4.x** — 13.x migration is future work with its own plan.
  The ABI-tripwire note in `stream_context_symbols.rs` gates that migration.

---

## Phase 0: `StreamContext` abstraction (reference — already landed in Session 1)

Commit message: `feat(stream): add StreamContext with host-blocking synchronize`

### Context

Session 1 already landed `npp/src/stream.rs` with `StreamContext` struct,
`raw_ctx()`, `synchronize()`, and `populate_stream_context`. However the
`synchronize()` implementation uses `device.wait_for(&self.stream)` which is
**device-side only** (it orders operations from the device's perspective but
does not block the host). This must be fixed to a true host-blocking
`stream.synchronize()` for the readback fence to be correct.

### Step 0.1: Fix `synchronize()` to be host-blocking

**File:** `npp/src/stream.rs`

Replace the body of `synchronize()`:

```rust
// BEFORE (device-side only — does not block host):
pub fn synchronize(&self) -> Result<(), DriverError> {
    self.device.wait_for(&self.stream)
}
```

```rust
// AFTER (host-blocking):
pub fn synchronize(&self) -> Result<(), DriverError> {
    self.stream.synchronize()
}
```

Add doc comment clarifying:
- This **blocks the host** until all work on the forked stream completes.
- This is the fence that makes the host-side readback safe.
- `device_fence()` (`wait_for`) is retained as a separate method for callers
  that only need device-side ordering (future async chaining).

### Step 0.2: Verify module docs are false (prerequisite for Phase 6 rewrite)

**File:** `npp/src/stream.rs` (lines 3-19)

Note for Phase 6: items 3-4 of the numbered list currently claim "read-back
… on the forked stream … ordered by construction." This is **incorrect**
per cudarc 0.9.15 source (§brief). Documented here so Phase 6 knows what
to rewrite.

### Step 0.3: Validation

- `nix develop . --command cargo build -p npp-rs` compiles.
- `nix develop . --command cargo clippy -p npp-rs -- -D warnings` passes.
- `nix develop . --command cargo test -p npp-rs` passes (no GPU needed).

---

## Phase 1: `npp-sys` allowlist + ABI pin

Commit message: `feat(npp-sys): widen bindgen allowlist for _Ctx symbols; pin ABI`

### Step 1.0: Widen bindgen allowlist

**File:** `npp-sys/build.rs` (lines 68-73)

The current allowlist patterns (`nppi.*`, `Nppi.*`, `NppStreamContext`, etc.)
should already capture `_Ctx` symbols since they match `nppi.*`. Verify by
checking the generated bindings for `nppiResize_8u_C1R_Ctx` etc. No code
change expected unless a pattern is missing.

**Verification:**
```bash
grep -c 'nppiResize_8u_C1R_Ctx' "$(nix develop . --command bash -c 'find target -name bindings.rs -print -quit')"
```
Expect ≥1 match. If zero, add explicit `allowlist_function("nppi.*_Ctx")`.

### Step 1.1: Compile-time ABI pinning

**File:** `npp-sys/tests/stream_context_symbols.rs`

This file already exists from Session 1 with pins for the four symbols we need.
No code change needed in this phase.

**Verification:**
```bash
nix develop . --command cargo test -p npp-sys -- stream_context_symbols
```
All 7 tests pass (4 symbol coercions + 3 type-property tests).

---

## Phase 2: Wire `StreamContext` into `CudaImage`

Commit message: `feat(image): integrate StreamContext into CudaImage; !Send+!Sync`

This phase is a structural refactor. The module may not compile in isolation
because macro-generated impls in `mean_macros.rs` reference `self.device`.
This is acceptable under session-end-green.

### Step 2.1: Rename `CudaImage.device` → `CudaImage.ctx`

**File:** `npp/src/image.rs`

Change the struct field:

```rust
// BEFORE:
pub struct CudaImage<T: NppPixelType> {
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) buf: CudaSlice<T>,
    pub(crate) layout: CudaLayout,
}

// AFTER:
pub struct CudaImage<T: NppPixelType> {
    pub(crate) ctx: Arc<StreamContext>,
    pub(crate) buf: CudaSlice<T>,
    pub(crate) layout: CudaLayout,
}
```

Add import: `use crate::stream::StreamContext;`

### Step 2.2: Update `CudaImage::new()` to accept `Arc<StreamContext>`

**File:** `npp/src/image.rs` (lines 117-132)

```rust
// BEFORE:
pub fn new(
    device: Arc<CudaDevice>,
    channels: u8,
    width: u32,
    height: u32,
) -> Result<Self, NppError> {
    let num_elements = (width as usize) * (height as usize) * (channels as usize);
    let buf = device.alloc_zeros::<T>(num_elements)?;
    let layout = CudaLayout::row_major_packed(channels, width, height);
    Ok(CudaImage {
        device,
        buf,
        layout,
    })
}

// AFTER:
pub fn new(
    ctx: Arc<StreamContext>,
    channels: u8,
    width: u32,
    height: u32,
) -> Result<Self, NppError> {
    let num_elements = (width as usize) * (height as usize) * (channels as usize);
    let buf = ctx.device().alloc_zeros::<T>(num_elements)?;
    let layout = CudaLayout::row_major_packed(channels, width, height);
    Ok(CudaImage {
        ctx,
        buf,
        layout,
    })
}
```

### Step 2.3: Update `CudaImage::from_host()` to accept `Arc<StreamContext>`

**File:** `npp/src/image.rs` (lines 148-174)

Same pattern: parameter `device: Arc<CudaDevice>` → `ctx: Arc<StreamContext>`,
replace `device.htod_sync_copy(data)` with `ctx.device().htod_sync_copy(data)`,
store `ctx` instead of `device`.

### Step 2.4: Hybrid readback in `TryFrom<&CudaImage<T>> for Vec<T>`

**File:** `npp/src/image.rs` (lines 440-449)

This is the most critical change. Replace the simple `dtoh_sync_copy` with the
host-fenced pattern:

```rust
// BEFORE:
impl<T: NppPixelType> TryFrom<&CudaImage<T>> for Vec<T> {
    type Error = NppError;

    fn try_from(img: &CudaImage<T>) -> Result<Self, Self::Error> {
        let host: Vec<T> = img.device.dtoh_sync_copy(&img.buf)?;
        Ok(host)
    }
}

// AFTER:
impl<T: NppPixelType> TryFrom<&CudaImage<T>> for Vec<T> {
    type Error = NppError;

    /// Read-back device pixels to host memory.
    ///
    /// # Ordering contract
    ///
    /// NPP `_Ctx` operations are enqueued on the forked stream. This readback
    /// performs:
    ///
    /// 1. A host-blocking `synchronize()` on the forked stream (the "fence"),
    ///    ensuring all prior `_Ctx` work is complete.
    /// 2. A synchronous DtoH copy on the **NULL stream** via
    ///    `dtoh_sync_copy` (cudarc 0.9 does not expose a stream-targeted
    ///    DtoH API — the copy always goes to the NULL stream).
    ///
    /// The fence in step 1 makes step 2 safe. Without it, the NULL-stream copy
    /// could race with the forked-stream NPP work.
    fn try_from(img: &CudaImage<T>) -> Result<Self, Self::Error> {
        // Step 1: host fence — block until all forked-stream work is done
        img.ctx.synchronize()?;
        // Step 2: NULL-stream DtoH copy (cudarc's only DtoH path)
        let host: Vec<T> = img.ctx.device().dtoh_sync_copy(&img.buf)?;
        Ok(host)
    }
}
```

Add doc comment linking to `docs/stream-context.md` for the full rationale.

### Step 2.5: Remove `Send + Sync` from `CudaImage`

**File:** `npp/src/image.rs` (lines 92-96)

Update the thread-safety doc section:

```rust
// BEFORE:
/// # Thread safety
///
/// `CudaImage<T>` is `Send + Sync` because `CudaSlice<T>` is `Send + Sync` in
/// cudarc 0.9. However, CUDA contexts are thread-bound; safe cross-thread usage
/// requires explicit context management (deferred to M2 — see F8).

// AFTER:
/// # Thread safety
///
/// `CudaImage<T>` is `!Send + !Sync` because [`StreamContext`] contains a
/// `CudaStream` which is thread-bound (CUDA streams cannot safely be used
/// from multiple threads). Images created from the same `StreamContext` share
/// the same forked stream; cross-thread usage requires external
/// synchronisation.
```

No explicit `impl !Send` or `impl !Sync` needed — `Arc<StreamContext>` is
`!Send + !Sync` because `CudaStream` is `!Send + !Sync`, and the compiler
infers this through field composition.

### Step 2.6: Update `CudaImageView` and `CudaImageViewMut`

**File:** `npp/src/image.rs`

Both view types have `device: Arc<CudaDevice>` fields. Rename to `ctx` and
change type to `Arc<StreamContext>`. Update their `sub_image`/`sub_image_mut`
constructors in `CudaImage`.

For `sub_image` (line 270):
```rust
// BEFORE:
Ok(CudaImageView {
    device: self.device.clone(),
    view,
    layout,
})

// AFTER:
Ok(CudaImageView {
    ctx: self.ctx.clone(),
    view,
    layout,
})
```

For `sub_image_mut` (line 319): same pattern.

### Step 2.7: Add `device()` accessor on `CudaImage`

**File:** `npp/src/image.rs`

Add a convenience method so that code holding a `CudaImage` can reach the
underlying device without going through `self.ctx.device()`:

```rust
/// Reference to the underlying CUDA device (shared from the StreamContext).
pub fn device(&self) -> &Arc<CudaDevice> {
    self.ctx.device()
}
```

This is how macro code (e.g. `mean_macros.rs` lines 80, 84) will access the
device after the rename — `self.device` becomes `self.ctx.device()`, which
requires the `device()` accessor or direct field access.

### Step 2.8: Re-export `StreamContext` from crate root for convenience

**File:** `npp/src/lib.rs`

```rust
pub use stream::{StreamContext, stream_context_for};
```

### Step 2.9: Module-level `#[allow(...)]` for `Arc<StreamContext>` lint

The `Arc<StreamContext>` field on `CudaImage` will trigger
`clippy::arc_with_non_send_sync`. Since `StreamContext` is intentionally
`!Send + !Sync` (thread-bound CUDA stream), add:

```rust
// At module level in image.rs:
#![allow(clippy::arc_with_non_send_sync)]
```

### Validation (Phase 2)

- `nix develop . --command cargo build -p npp-rs` — may fail because
  `mean_macros.rs`, `resize_macros.rs`, `swap_channels_macros.rs`,
  `mean_generated.rs`, etc. still reference `self.device`. This is expected
  under session-end-green. Errors should be limited to `self.device` references.
- Verify no `impl Send`/`impl Sync` blanket for `CudaImage` exists accidentally.

---

## Phase 3: Pivot Resize, SwapChannels, Mean to `_Ctx` variants

Commit message: `feat(ops): pivot Resize, SwapChannels, Mean to _Ctx variants`

### Step 3.1: Replace non-`_Ctx` symbols with `_Ctx` in Resize macro

**File:** `npp/src/resize_macros.rs`

The `impl_resize_for!` macro currently calls base symbols (e.g.
`nppiResize_8u_C1R`). Change each channel-dispatch arm to the `_Ctx` variant
and add a trailing `NppStreamContext` argument obtained from `self.ctx.raw_ctx()`.

```rust
// BEFORE (inside the `match self.channels()` arms):
$ch => $sym(
    src_ptr as *const _,
    src_step_bytes,
    src_size,
    src_rect,
    dst_ptr as *mut _,
    dst_step_bytes,
    dst_size,
    dst_rect,
    $crate::resize_ops::interpolation_mode(inter),
),

// AFTER:
$ch => $sym(
    src_ptr as *const _,
    src_step_bytes,
    src_size,
    src_rect,
    dst_ptr as *mut _,
    dst_step_bytes,
    dst_size,
    dst_rect,
    $crate::resize_ops::interpolation_mode(inter),
    // Trailing NppStreamContext (by value, Copy):
    self.ctx.raw_ctx(),
),
```

Also add the import guard: the macro already imports from `npp_sys` but may
not have `NppStreamContext` — it's obtained via `raw_ctx()` which returns
`NppStreamContext`, so the type must be visible. The macro expansions use
`self.ctx` which is `Arc<StreamContext>`. Ensure the macro file imports
`crate::stream::StreamContext` if not already present (it's accessed through
the call to `raw_ctx()` which is defined on `StreamContext`).

### Step 3.2: SwapChannels macro

**File:** `npp/src/swap_channels_macros.rs`

Same pattern — swap the symbol (e.g. `nppiSwapChannels_8u_C4C3R` →
`nppiSwapChannels_8u_C4C3R_Ctx`) and add trailing context arg:

```rust
$ch => $sym(
    src_ptr as *const _,
    src_step_bytes,
    dst_ptr as *mut _,
    dst_step_bytes,
    nppi_size,
    &order[0],
    self.ctx.raw_ctx(),  // NEW: trailing _Ctx parameter
),
```

### Step 3.3: Mean macro

**File:** `npp/src/mean_macros.rs`

Two symbols to swap: the `_Ctx` variant for the mean computation and for the
buffer-size query. The Mean macro has a special structure because it calls
two NPP symbols.

**Buffer-size query (line 65):**
```rust
// BEFORE:
$ch => $buffer_sym(nppi_size, &mut buffer_size as *mut usize),

// AFTER:
$ch => $buffer_sym(nppi_size, &mut buffer_size as *mut usize, self.ctx.raw_ctx()),
```

**Mean computation (lines 106-112):**
```rust
// BEFORE:
$ch => $mean_sym(
    src_ptr as *const _,
    src_step_bytes,
    nppi_size,
    scratch_ptr as *mut u8,
    out_ptr as *mut f64,
),

// AFTER:
$ch => $mean_sym(
    src_ptr as *const _,
    src_step_bytes,
    nppi_size,
    scratch_ptr as *mut u8,
    out_ptr as *mut f64,
    self.ctx.raw_ctx(),
),
```

### Step 3.4: Update generated files (resize_generated.rs, swap_channels_generated.rs, mean_generated.rs)

These files reference the old non-`_Ctx` symbols by path. The symbols are
changed in the macros, not in the generated files — the generated files
provide the symbol paths (e.g. `npp_sys::nppiResize_8u_C1R`), and the
macro body adds the trailing ctx arg and maps to the `_Ctx` variant.

**Two approaches:**

**Approach A (preferred — macro-internal mapping):** Keep the generated files
unchanged. The macro receives the base symbol and appends `_Ctx` internally.
This avoids regenerating the codegen output and makes the `_Ctx` pivot an
implementation detail of the macro.

**Approach B (update generated files):** Change the generated files to pass
`_Ctx` symbols and update the macro to accept both base and ctx variants.

**Decision:** Use Approach A. The macro appends `_Ctx` by path manipulation
is not possible in Rust macros (can't concat identifiers in `path` fragments).
Instead, change the generated files to pass `_Ctx` symbols and update the macro
to expect `_Ctx` symbols:

In `mean_generated.rs`:
```rust
// BEFORE:
1 => (npp_sys::nppiMean_8u_C1R, npp_sys::nppiMeanGetBufferHostSize_8u_C1R),

// AFTER:
1 => (npp_sys::nppiMean_8u_C1R_Ctx, npp_sys::nppiMeanGetBufferHostSize_8u_C1R_Ctx),
```

Same for resize and swap_channels generated files.

### Step 3.5: Update macro doc comments

Update the doc comments in each macro file to note that `_Ctx` variants are
used and that the `NppStreamContext` is obtained from `self.ctx.raw_ctx()`.

### Validation (Phase 3)

- `nix develop . --command cargo build -p npp-rs` — should now compile.
  Previous `self.device` references in macros resolve via `self.ctx.device()`
  (the accessor added in Phase 2.7).
- `nix develop . --command cargo clippy -p npp-rs -- -D warnings` passes.
- `nix develop . --command cargo test -p npp-rs` passes (no GPU).
- GPU test: `nix develop . --command cargo test --features gpu -p npp-rs` —
  golden tests may fail (need re-pin after `_Ctx` pivot — see Phase 5).

---

## Phase 4: Verify `synchronize()` host-blocking semantics

Commit message: `fix(stream): make synchronize() host-blocking (F8 B2)`

### Step 4.1: Confirm `stream.synchronize()` is host-blocking

**File:** `npp/src/stream.rs` — already changed in Phase 0.1 to use
`self.stream.synchronize()`.

Verify against cudarc source:
- `CudaStream::synchronize()` calls `cuStreamSynchronize` which is a
  host-blocking call — it does not return until all work on the stream
  has completed.
- `CudaDevice::wait_for` calls `cuEventSynchronize` / `cuStreamWaitEvent`
  which only orders operations on the device side — it does not block the
  host.

### Step 4.2: Keep `device_fence()` for device-side ordering

The `wait_for` pattern is still useful for callers that want to ensure
device-side ordering without host blocking. Keep `device_fence()` as a
separate method.

**Add to `StreamContext`:**
```rust
/// Device-side fence: ensure all prior work on this stream is visible
/// to subsequent work on other streams on the same device, without
/// blocking the host.
///
/// This is useful for synchronising streams within the same device
/// context without a host round-trip.
#[cfg(not(tarpaulin_include))]
pub fn device_fence(&self) -> Result<(), DriverError> {
    self.device.wait_for(&self.stream)
}
```

### Validation (Phase 4)

- `nix develop . --command cargo build -p npp-rs` compiles.
- `nix develop . --command cargo clippy -p npp-rs -- -D warnings` passes.
- `nix develop . --command cargo test -p npp-rs` passes (no GPU).

---

## Phase 5: C1 two-op smoke test (reshaped)

Commit message: `test(gpu): add reshaped C1 two-op smoke test`

### Step 5.1: Create the two-op test

**File:** New file: `npp/tests/golden_chained_ctx.rs`

This test verifies that two `_Ctx` operations chained on the same
`StreamContext` produce correct pixels without intermediate host
synchronisation. The test structure:

```
Operation 1: BGRA→RGB (4ch→3ch) on the forked stream
Operation 2: Resize NearestNeighbor (3ch) on Op 1's output — no intervening readback
Single readback: try_from at the end with fence + NULL-stream copy
```

```rust
use npp_rs::prelude::*;  // or direct imports
use npp_rs::test_helpers::assert_golden;

#[cfg_attr(not(feature = "gpu"), ignore)]
#[test]
fn chained_bgr_to_rgb_then_resize_produces_correct_pixels() {
    // ── Setup ──
    let ctx = stream_context_for(0).expect("GPU at ordinal 0");
    let (w, h) = (4u32, 4u32);

    // Known BGRA input: 4x4 checkerboard-like pattern
    // Each pixel: B, G, R, A (4 bytes)
    let bgra_data: Vec<u8> = vec![
        255, 0, 0, 255,   0, 255, 0, 255,   0, 0, 255, 255,   255, 255, 0, 255,
        0, 255, 255, 255,  255, 0, 255, 255,   128, 128, 128, 255, 64, 64, 64, 255,
        255, 0, 0, 255,   0, 255, 0, 255,   0, 0, 255, 255,   255, 255, 0, 255,
        0, 255, 255, 255,  255, 0, 255, 255,   128, 128, 128, 255, 64, 64, 64, 255,
    ];

    let src = CudaImage::<u8>::from_host(ctx.clone(), 4, w, h, &bgra_data)
        .expect("upload BGRA");

    // ── Op 1: BGRA→RGB (4ch → 3ch) ──
    let mut rgb = CudaImage::<u8>::new(ctx.clone(), 3, w, h)
        .expect("allocate RGB");
    src.bgra_to_rgb(&mut rgb).expect("BGRA→RGB");

    // ── Op 2: Resize (downsample 4x4 → 2x2, NearestNeighbor) ──
    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, w / 2, h / 2)
        .expect("allocate dst");
    rgb.resize(&mut dst, ResizeInterpolation::NearestNeighbor)
        .expect("resize");

    // ── Single readback (fence + NULL-stream copy) ──
    let result: Vec<u8> = Vec::try_from(&dst).expect("readback");

    // ── Golden ──
    // EXPECTED must be pinned by running once and copying the printed output.
    // 2x2 RGB = 4 pixels × 3 channels = 12 bytes.
    const EXPECTED: &[u8] = &[];  // EMPTY → prints output for pinning
    assert_golden(&result, EXPECTED, "chained_bgr_to_rgb_then_resize");
}
```

### Step 5.2: Pin the golden reference

1. Run: `nix develop . --command cargo test --features gpu -p npp-rs -- golden_chained_ctx`
2. Copy the printed byte literal into `EXPECTED`.
3. Re-run to confirm.

### Step 5.3: Re-pin existing golden tests after `_Ctx` pivot

The `_Ctx` pivot should be semantically identical (same computation, same
results) but the output may differ due to internal NPP differences. Run all
golden tests and re-pin any that fail:

1. `nix develop . --command cargo test --features gpu -p npp-rs -- golden_`
2. For each failure, copy the printed output into the corresponding
   `EXPECTED` constant.
3. Re-run all golden tests to confirm all pass.

### Step 5.4: Add doc comment clarifying what C1 does and does not test

In the test file, add module-level docs:

```rust
//! # C1 chained-op smoke test
//!
//! This test verifies that two `_Ctx` operations (SwapChannels → Resize)
//! chained on the same [`StreamContext`] produce correct pixels through a
//! shared execution context, with a single host-fenced readback at the end.
//!
//! ## What it tests
//!
//! - Intra-stream op chaining: Op2 reads Op1's device output without
//!   intervening host sync.
//! - The [`StreamContext::synchronize()`] fence before readback.
//! - Correct pixel output through the pipeline.
//!
//! ## What it does NOT test
//!
//! - Fence-absent behaviour (not discriminable under cudarc 0.9's
//!   context-wide synchronize — see `docs/stream-context.md`).
//! - Cross-stream re-entry.
//! - The device-fence (`wait_for`) path.
```

### Validation (Phase 5)

- `nix develop . --command cargo build -p npp-rs` compiles.
- `nix develop . --command cargo test --features gpu -p npp-rs` — all golden
  tests pass.

---

## Phase 6: Documentation rewrite + ABI-tripwire note

Commit message: `docs: correct stream-context ordering narrative + add ABI gate note`

### Step 6.1: Rewrite `docs/stream-context.md` (REQUIRED — Risk 1)

**File:** `docs/stream-context.md`

Currently claims ordering is "by construction" (lines 8-19). Rewrite the
entire document with the honest three-part model.

**Rewrite §"Ordered execution without explicit synchronisation" (lines 8-19):**

Replace with:

```markdown
## Execution model

`StreamContext` provides three execution stages with two distinct ordering
guarantees:

1. **Host-to-device copies** are performed on the CUDA default stream
   (e.g. `CudaDevice::htod_sync_copy_into`).
2. **NPP operations** are enqueued on the `StreamContext`'s forked stream
   via the `_Ctx` API.
3. **Device-to-host read-backs** (`TryFrom<&CudaImage>`) execute on the
   **NULL stream** via cudarc's `dtoh_sync_copy`.

## Ordering guarantees

### Stage 1 → Stage 2: Ordered by forked-stream creation wait

`CudaDevice::fork_default_stream()` creates a new stream and inserts an
implicit wait that makes all prior default-stream work visible before the
first operation on the forked stream executes. This means **HTOD copy → NPP
op** is ordered **by construction** — no explicit synchronisation needed
between upload and computation.

### Stage 2 → Stage 3: NOT ordered — requires explicit host fence

`TryFrom<&CudaImage>` performs a synchronous DtoH copy on the **NULL
stream** via `dtoh_sync_copy`. cudarc 0.9 does not expose a stream-targeted
DtoH API — the copy always goes to the NULL stream. The forked stream
(NPP ops) and the NULL stream (readback) are genuinely unordered unless
an explicit barrier is inserted.

The barrier is the host-blocking `synchronize()` call (via
`cuStreamSynchronize`) on the forked stream **before** the NULL-stream
copy. This is implemented in `TryFrom<&CudaImage>`:

```rust
img.ctx.synchronize()?;                    // host fence
ctx.device().dtoh_sync_copy(&img.buf)?;    // NULL-stream copy
```

### Retraction

Earlier revisions of this document stated that the readback was ordered
"by construction" on the forked stream. This was incorrect. The fence in
`TryFrom<&CudaImage>` is **load-bearing** — removing it would create a
data race between NPP work on the forked stream and the NULL-stream DtoH
copy.
```

**Rewrite §"Sync points" (lines 33-41):**

```markdown
## Sync points

- [`StreamContext::synchronize()`](../npp/src/stream.rs) — Host-blocking
  fence. Calls `cuStreamSynchronize` on the forked stream. Guarantees all
  prior `_Ctx` work is complete before the host continues.
- [`TryFrom<&CudaImage>`](../npp/src/image.rs) — Performs a host fence
  via `synchronize()` followed by a NULL-stream DtoH copy. This is the
  recommended readback path for single-step or chained pipelines.
- [`StreamContext::device_fence()`](../npp/src/stream.rs) — Device-side
  only. Calls `cuEventSynchronize` (via `CudaDevice::wait_for`). Orders
  work between streams on the same device without blocking the host. Not
  sufficient for host readback safety.
```

**Revise §"Why this closes C8" (lines 43-56) — keep the same factual
content, but note that the fence specifically addresses the cross-stream
race.**

**§"StreamContext is `!Send + !Sync`"** — already correct, keep.

### Step 6.2: Rewrite `npp/src/stream.rs` module doc (REQUIRED — Risk 1)

**File:** `npp/src/stream.rs` (lines 1-19)

Replace the incorrect numbered list (items 3-4 claim readback on forked
stream) with the honest description:

```rust
//! Application-managed NPP stream context abstraction.
//!
//! # Execution model
//!
//! NPP `_Ctx` functions are enqueued asynchronously on the stream stored
//! in this context. The ordering contract has two parts:
//!
//! 1. **Upload → NPP op: ordered by construction.**
//!    `CudaDevice::fork_default_stream()` creates a new stream with an
//!    implicit wait so all prior default-stream work is visible. A
//!    host-to-device copy on the default stream completes before the NPP
//!    op on the forked stream begins.
//!
//! 2. **NPP op → readback: requires explicit host fence.**
//!    `TryFrom<&CudaImage>` performs its DtoH copy on the NULL stream
//!    (cudarc 0.9's only DtoH API — `dtoh_sync_copy` does not accept a
//!    caller-supplied stream). The forked stream and NULL stream are
//!    unordered. The host-blocking `synchronize()` call inserted *before*
//!    the NULL-stream copy is the load-bearing barrier that makes this safe.
//!
//! **Earlier revisions** claimed the readback was on the forked stream and
//! ordered by construction. That was incorrect — see `docs/stream-context.md`
//! for the full rationale.
```

### Step 6.3: Add ABI-tripwire forward-facing header (REQUIRED — Risk 4)

**File:** `npp-sys/tests/stream_context_symbols.rs`

Add a header block at the top of the file, before the existing Phase 0
findings comment:

```rust
//! # ABI migration gate (e.g. CUDA 12.x → 13.x)
//!
//! If any coercion test below fails after a CUDA/NPP version bump:
//!
//! 1. The NPP signature changed — do NOT just edit the type to compile.
//! 2. Inspect the regenerated `bindings.rs`; identify the ABI change
//!    (e.g. `int` → `size_t`, field added/removed from `NppStreamContext`).
//! 3. Update the matching macro in `npp/src/*_macros.rs` and the
//!    `raw_ctx()` return type / method signature if needed.
//! 4. Re-pin ALL affected golden tests (C12 discipline).
//! 5. Refresh the Phase 0 findings table in this file.
```

### Step 6.4: Final validation and golden re-pin

1. `nix develop . --command cargo build -p npp-rs` compiles.
2. `nix develop . --command cargo clippy -p npp-rs -- -D warnings` passes.
3. `nix develop . --command cargo test -p npp-rs` passes (no GPU).
4. `nix develop . --command cargo test --features gpu -p npp-rs` — ALL
   golden tests pass, including the new C1 chained-op test.
5. `nix develop . --command cargo fmt --check` passes.

---

## Rollback plan

If any GPU test cannot be fixed by re-pinning (indicating a semantic change
from the `_Ctx` pivot, not a byte-identical computation):

1. Revert the `_Ctx` symbol swap in the macros (Phase 3), keeping the
   `StreamContext` wiring (Phase 2) — this creates a valid intermediate where
   the context exists but ops use the default stream.
2. Investigate the NPP behaviour difference between `_Ctx` and non-`_Ctx`
   variants for the specific type/interpolation combination.
3. File a bug or constraint note in the roadmap.

---

## Appendix: Reshaped C1 test (Finding B)

The original plan etched a C1 test with two readbacks (Op1 → Readback1 →
Op2 → Readback2). This session reshapes it to:

```
ctx = StreamContext::new(device)?
img_bgra = CudaImage::<u8>::from_host(ctx.clone(), 4, w, h, bgra_data)?
img_rgb  = CudaImage::<u8>::new(ctx.clone(), 3, w, h)?
img_dst  = CudaImage::<u8>::new(ctx.clone(), 3, w/2, h/2)?

img_bgra.bgra_to_rgb(&mut img_rgb)?          // Op1 on forked stream
img_rgb.resize(&mut img_dst, NearestNeighbor)? // Op2 on forked stream (no sync)

result: Vec<u8> = Vec::try_from(&img_dst)?     // single readback (fence + NULL-stream copy)
assert_golden(&result, EXPECTED);
```

**Why:** Removing the intermediate readback tests real intra-stream op
chaining (Op2 reads Op1's device output without host round-trip). The fence
is still exercised once at the end.

## Appendix: Omitted Finding A (Mean readback location)

`mean_macros.rs:120` currently reads `self.device.dtoh_sync_copy(&out_buf)?`.
Under session-end-green this does not need to be pre-fixed in Phase 2; it
will naturally resolve in Phase 3 (after the field rename) as
`self.ctx.device().dtoh_sync_copy(&out_buf)?` via the `device()` accessor
added in Phase 2.7.
