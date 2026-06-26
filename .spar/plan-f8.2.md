# Feature: F8.2 — Stream-primitives floor (`Fence` + `new_stream`)

> **Framing (load-bearing — do not regress):** This is a **minimal composable
> primitive floor**, not a workflow feature. F8.2 delivers exactly two orthogonal
> primitives — `Fence` (device-side ordering + timing) and `new_stream`
> (caller-owned stream handles) — plus the deletion of the broken
> `device_fence()` no-op stub. No op-layer changes, no macro edits, no
> view-on-stream work. Primitive 3 (op-on-caller-supplied-stream, pipeline
> patterns, cross-stream op-ordering tests) is explicitly deferred until crafted
> against multiple use cases. The single-pipeline fence test (Phase 3, Step 1)
> proves plumbing is sound but *cannot prove ordering enforcement* — that
> requires Primitive 3's cross-stream validation and is intentionally absent.
>
> **F11 doc debt is NOT bundled** — only `stream-context.md` is reconciled here
> because it touches the same stream-ordering model as the new primitives.

---

## Phase 1: Remove broken `device_fence()` stub and fix benches to use `synchronize()`

Commit message: refactor: remove broken device_fence no-op stub, migrate benches to synchronize (F8.2)

### Step 1: Delete `device_fence()` from `npp/src/stream.rs`

Edit the file `npp/src/stream.rs`.

Remove the `device_fence` method entirely — its doc comment, its
`#[cfg(not(tarpaulin_include))]` attribute, and the function body. The block
to delete spans lines 163–183 (inclusive in the current file at time of writing):

```rust
    /// Device-side fence: ensure all prior work on this stream is visible
    /// to subsequent work on other streams on the same device, without
    /// blocking the host.
    ///
    /// This is useful for ordering operations between streams within the
    /// same device context without a host round-trip. It is **not**
    /// sufficient to make host-side readback safe — use [`synchronize`]
    /// for that.
    ///
    /// [`synchronize`]: Self::synchronize
    #[cfg(not(tarpaulin_include))]
    pub fn device_fence(&self) -> Result<(), DriverError> {
        // In cudarc 0.19.x, CudaContext no longer has a wait_for method.
        // We create a brief event to establish ordering between streams.
        // This is equivalent to the old CudaDevice::wait_for pattern.
        let event = self
            .device
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        event.record(&self.stream)?;
        Ok(())
    }
```

Do not replace the deleted method with anything. The method is a no-op stub:
it records an event that nobody waits on, and its doc-comment claims it orders
work between streams — a false claim because the recorded event is discarded
without any corresponding `cuStreamWaitEvent`.

After removing the method, check whether the `sys` import from `cudarc::driver`
(still needed for `record_event`) remains as the only remaining use. The current
import is:

```rust
use cudarc::driver::{sys, CudaContext, CudaEvent, CudaStream, DriverError};
```

The `sys` qualifier is used only by `device_fence` and `record_event`. Since
`record_event` uses `sys::CUevent_flags::CU_EVENT_DEFAULT`, the import must be
kept. Verify with `cargo build --features gpu` after the edit.

### Step 2: Replace `device_fence()` with `synchronize()` in `bench_resize_size.rs`

Edit the file `npp/benches/bench_resize_size.rs`.

On line 90, change:

```rust
                    ctx.device_fence().expect("fence after resize");
```

to:

```rust
                    ctx.synchronize().expect("sync after resize");
```

This replaces the broken no-op with a genuine host-blocking stream sync.
The bench already uses `elapsed()` (which calls `cuEventElapsedTime`) — the
events only need to be complete before the elapsed query, and `synchronize()`
provides that guarantee. The bench is single-pipeline (all NPP ops and timing
events on one stream), so same-stream `synchronize()` is correct and safe.

Leave every other line unchanged.

### Step 3: Replace `device_fence()` with `synchronize()` in `bench_resize_modes.rs`

Edit the file `npp/benches/bench_resize_modes.rs`.

Two occurrences to change:

- Line 81: `ctx.device_fence().expect("fence");` → `ctx.synchronize().expect("sync");`
- Line 120: `ctx.device_fence().expect("fence");` → `ctx.synchronize().expect("sync");`

These are the two sub-benchmarks: interpolation-mode sweep (line 81) and
channel-count comparison (line 120). Both are single-pipeline (all operations
and timing on one stream), so `synchronize()` is correct. Same rationale as
Step 2.

Leave every other line unchanged.

### Step 4: Replace `device_fence()` with `synchronize()` in `bench_op_family.rs`

Edit the file `npp/benches/bench_op_family.rs`.

Five occurrences to change, all following the identical pattern:

- Line 98: `ctx.device_fence().expect("fence");` → `ctx.synchronize().expect("sync");`
- Line 120: `ctx.device_fence().expect("fence");` → `ctx.synchronize().expect("sync");`
- Line 142: `ctx.device_fence().expect("fence");` → `ctx.synchronize().expect("sync");`
- Line 164: `ctx.device_fence().expect("fence");` → `ctx.synchronize().expect("sync");`
- Line 186: `ctx.device_fence().expect("fence");` → `ctx.synchronize().expect("sync");`

These correspond to Resize, SwapChannels, Mean, Convert, and Normalize
sub-benchmarks respectively. Each is single-pipeline. Same rationale as
Step 2.

Leave every other line unchanged.

### Step 5: Verify removal compiles (non-GPU and GPU profiles)

Run inside the Nix dev shell:

```bash
nix develop . --command cargo build
```

This must succeed and produce zero warnings about unused imports after the
`device_fence` removal. If compile errors occur (e.g. `sys` import is now
unused in non-event paths), add a `#[cfg(not(tarpaulin_include))]` attribute
or adjust the import scope — but verify first, because `record_event` still
uses `sys::CUevent_flags`.

Then verify the GPU profile also compiles:

```bash
nix develop . --command cargo build --features gpu
```

This compiles the bench files and confirms the `synchronize()` replacement is
type-correct (`synchronize` returns `Result<(), DriverError>`, matched by
`.expect()`).

Do not proceed past this step if either build fails.

---

## Phase 2: Add `Fence` type, `record_fence`/`wait_for`/`new_stream`/`elapsed_between` primitives

Commit message: feat: add Fence, record_fence, wait_for, elapsed_between, and new_stream (F8.2)

### Step 1: Add `new_stream()` method to `StreamContext`

Edit the file `npp/src/stream.rs`.

Add a new public method to `impl StreamContext` (after `synchronize()`, before
or after `record_event()` — placement between existing methods is fine):

```rust
    /// Create a new non-blocking stream on the same CUDA device.
    ///
    /// The returned `Arc<CudaStream>` is owned entirely by the caller — the
    /// `StreamContext` retains no reference to it. This is intentional:
    /// stream pooling, assignment policy, and lifetime management are the
    /// caller's responsibility.
    ///
    /// The new stream does **not** implicitly synchronise with any other
    /// stream. Use [`record_fence`] / [`wait_for`] to enforce ordering
    /// between streams.
    ///
    /// # Safety
    ///
    /// The returned stream must not outlive the device handle (the
    /// `Arc<CudaContext>` stored in this `StreamContext`). This is the same
    /// C7 invariant that governs all CUDA objects in this crate — the caller
    /// is responsible for ensuring the `StreamContext` (or some other
    /// reference to the same device) outlives the stream.
    ///
    /// [`record_fence`]: Self::record_fence
    /// [`wait_for`]: Self::wait_for
    #[cfg(not(tarpaulin_include))]
    pub fn new_stream(&self) -> Result<Arc<CudaStream>, DriverError> {
        self.device.new_stream()
    }
```

The method delegates directly to `CudaContext::new_stream()` from cudarc.
No additional `PhantomData` guard is needed on the returned `Arc<CudaStream>`
because `CudaStream` in cudarc 0.19.x is `Send + Sync` — the caller chooses
whether to enforce thread affinity. (The `StreamContext` itself remains
`!Send + !Sync` via its own `PhantomData<*const ()>`.)

### Step 2: Add the `Fence` type

Edit the file `npp/src/stream.rs`.

Add a new struct after the existing `Event` struct (after line 247), with its
own `#[derive(Debug)]` and doc comment:

```rust
/// A CUDA event used for cross-stream ordering and device-side timing.
///
/// A `Fence` is recorded on one stream and waited on by another stream,
/// establishing a happens-after relationship between the work before the
/// fence on the recording stream and the work after the wait on the
/// waiting stream.
///
/// This is the single type for both ordering (`wait_for`) and timing
/// (`elapsed_between`). The underlying CUDA event is one concept; whether
/// it is used for ordering, timing, or both is a usage distinction, not
/// a structural one.
///
/// # Safety
///
/// The underlying CUDA event is created with `cuEventCreate` and destroyed
/// with `cuEventDestroy` on drop. The event is tied to the same CUDA context
/// as the `StreamContext` that created it. The caller must ensure the CUDA
/// device handle (`Arc<CudaContext>`) outlives this fence (same C7 invariant
/// as [`CudaImage`](crate::image::CudaImage) and [`StreamContext`]).
///
/// # !Send + !Sync
///
/// Like [`StreamContext`] and [`Event`], `Fence` is `!Send + !Sync` because
/// CUDA events are tied to the CUcontext that created them (C7).
#[derive(Debug)]
#[cfg(not(tarpaulin_include))]
pub struct Fence {
    inner: CudaEvent,
    // Keeps the CUDA device (and thus the CUDA context) alive.
    _device: Arc<CudaContext>,
    // Makes Fence !Send + !Sync to match StreamContext's CUDA-context
    // thread affinity.
    _nosync: PhantomData<*const ()>,
}
```

Do not add any inherent methods to `Fence` — all operations on fences go
through `StreamContext` methods (`record_fence`, `wait_for`, `elapsed_between`).

### Step 3: Add `record_fence()` method to `StreamContext`

Edit the file `npp/src/stream.rs`.

Add a new public method to `impl StreamContext`:

```rust
    /// Record a fence on this stream.
    ///
    /// Creates a new [`Fence`] and records it on the stream managed by this
    /// context. All work enqueued before this call will complete before the
    /// fence is signalled. The returned `Fence` can be used with [`wait_for`]
    /// on another stream to order work, or with [`elapsed_between`] to measure
    /// device-time between two fences.
    ///
    /// This does **not** block the host. It enqueues a CUDA event record
    /// operation on the stream, which completes asynchronously.
    ///
    /// [`wait_for`]: Self::wait_for
    /// [`elapsed_between`]: Self::elapsed_between
    #[cfg(not(tarpaulin_include))]
    pub fn record_fence(&self) -> Result<Fence, NppError> {
        use cudarc::driver::sys::CUevent_flags;
        let event = self
            .device
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(NppError::Cuda)?;
        event.record(&self.stream).map_err(NppError::Cuda)?;
        Ok(Fence {
            inner: event,
            _device: Arc::clone(&self.device),
            _nosync: PhantomData,
        })
    }
```

Note: this uses `NppError::Cuda`, not `.expect()`, unlike the existing
`record_event()` method which panics on failure. `Fence` follows the
error-return convention (Result-based) consistent with the rest of the
`StreamContext` API. The existing `Event` / `record_event` API is kept
unchanged for backward compatibility with the 18 golden benches.

### Step 4: Add `wait_for()` method to `StreamContext`

Edit the file `npp/src/stream.rs`.

Add a new public method to `impl StreamContext`:

```rust
    /// Make this stream wait until a fence recorded on another stream
    /// is signalled.
    ///
    /// All work enqueued **after** this call on this stream will not begin
    /// until the fence's prior work (on the recording stream) has completed.
    /// This establishes a device-side ordering between streams without
    /// host involvement.
    ///
    /// The `fence` must have been recorded (via [`record_fence`]) on some
    /// stream — not necessarily the same stream context — before this call.
    ///
    /// # Panics
    ///
    /// Panics if the underlying `cuStreamWaitEvent` call fails (this
    /// indicates a driver-level issue that is not recoverable).
    ///
    /// [`record_fence`]: Self::record_fence
    #[cfg(not(tarpaulin_include))]
    pub fn wait_for(&self, fence: &Fence) {
        // cudarc's CudaEvent::stream_wait is safe in 0.19.x.
        let result = fence.inner.stream_wait(&self.stream, Some(sys::CUevent_flags::CU_EVENT_DEFAULT));
        assert!(result.is_ok(), "cuStreamWaitEvent failed: {:?}", result);
    }
```

This mirrors the panic-on-failure pattern of the existing `Event::record()`
method, consistent with the convention that CUDA driver failures at record/wait
sites are non-recoverable.

### Step 5: Add `elapsed_between()` method to `StreamContext`

Edit the file `npp/src/stream.rs`.

Add a new public method to `impl StreamContext`:

```rust
    /// Measure device-time elapsed between two recorded fences.
    ///
    /// Both fences must have been recorded (via [`record_fence`]) and the
    /// earlier fence's stream must have completed the recorded work before
    /// this call. The fences may have been recorded on **different** streams
    /// — this is the cross-stream timing capability that the existing
    /// [`elapsed`](Self::elapsed) method (which requires same-stream events)
    /// cannot provide.
    ///
    /// Returns `Ok(Duration)` on success, or `Err(NppError)` if the driver
    /// call fails.
    ///
    /// # Errors
    ///
    /// Returns `NppError::Cuda` if `cuEventElapsedTime` fails.
    ///
    /// [`record_fence`]: Self::record_fence
    #[cfg(not(tarpaulin_include))]
    pub fn elapsed_between(&self, start: &Fence, end: &Fence) -> Result<Duration, NppError> {
        let ms = start.inner.elapsed_ms(&end.inner).map_err(NppError::Cuda)?;
        Ok(Duration::from_secs_f64(ms as f64 / 1_000.0))
    }
```

This lives on `StreamContext` (the CUcontext), not on a stream or a fence,
because the elapsed span between two fences recorded on different streams is
a context-level query.

### Step 6: Export `Fence` from `lib.rs`

Edit the file `npp/src/lib.rs`.

Change line 81 from:

```rust
pub use stream::{stream_context_for, StreamContext};
```

to:

```rust
pub use stream::{stream_context_for, StreamContext, Fence};
```

This makes `Fence` available as `npp_rs::stream::Fence` (consistent with
`npp_rs::stream::StreamContext`).

### Step 7: Verify Phase 2 compiles (non-GPU and GPU profiles)

Run inside the Nix dev shell:

```bash
nix develop . --command cargo build
nix develop . --command cargo build --features gpu
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo clippy --features gpu -- -D warnings
```

All must pass with no warnings. The `--features gpu` clippy invocation is
required because the bench files (which still reference `ctx.synchronize()`
from Phase 1) are `#![cfg(feature = "gpu")]` and are otherwise never compiled,
so their import hygiene would go unchecked.

---

## Phase 3: Plumbing + smoke tests for new primitives

Commit message: test: add plumbing tests for Fence, wait_for, and elapsed_between (F8.2)

### Step 1: Single-pipeline fence plumbing test

Create the file `npp/tests/fence_ordering.rs`:

```rust
//! Single-pipeline fence plumbing test (F8.2).
//!
//! Creates two fences on the same stream, records them around a Resize
//! operation, and measures elapsed time. This proves that `record_fence()`,
//! `wait_for()`, and `elapsed_between()` are wired correctly (CUDA events
//! are created, recorded, and queried without driver errors).
//!
//! **Limitation:** `wait_for()` in a single-pipeline test is redundant —
//! work on one stream is already FIFO-ordered. This test proves plumbing is
//! sound but cannot prove `wait_for()` enforces cross-stream ordering.
//! Cross-stream op-ordering validation is deferred to Primitive 3 (F8.2+),
//! which must be shaped by multiple real use cases (e.g. MTCNN fan-out,
//! supertextures) before committing to a test structure.
//!
//! # GPU gate
//!
//! `#[cfg(feature = "gpu")]` — plain `cargo test` skips this file.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::stream::{StreamContext, Fence};
use npp_rs::stream_context_for;
use std::sync::Arc;

#[test]
fn fence_plumbing_same_stream() {
    let ctx = stream_context_for(0).expect("CUDA device 0 init");

    let src = CudaImage::<u8>::from_host(
        ctx.clone(),
        3,   // channels
        64,  // width
        64,  // height
        &vec![42u8; 64 * 64 * 3],
    )
    .expect("source allocation");

    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, 32, 32).expect("dst allocation");

    // Record a fence before the NPP op.
    let before: Fence = ctx.record_fence().expect("record before");

    // Perform an NPP operation.
    src.resize(&mut dst, ResizeInterpolation::NearestNeighbor)
        .expect("resize");

    // Record a fence after the NPP op.
    let after: Fence = ctx.record_fence().expect("record after");

    // Wait for the before-fence on the same stream (redundant but proves
    // wait_for is wired).
    ctx.wait_for(&before);

    // Measure elapsed time between the two fences (same-stream).
    let elapsed = ctx.elapsed_between(&before, &after).expect("elapsed");
    assert!(
        elapsed.as_nanos() > 0,
        "elapsed time must be non-zero for a real NPP operation"
    );

    // Synchronise to ensure the resize completed before we check geometry.
    ctx.synchronize().expect("synchronize");

    // Geometry check (not a golden test — just verifying the pipeline ran).
    assert_eq!(dst.width(), 32);
    assert_eq!(dst.height(), 32);
    assert_eq!(dst.channels(), 3);
}
```

### Step 2: Cross-stream timing smoke test

Create the file `npp/tests/fence_timing.rs`:

```rust
//! Cross-stream timing smoke test (F8.2).
//!
//! Creates two streams, performs a Resize on each, and uses `elapsed_between`
//! with fences recorded on different streams to verify that cross-stream
//! timing is wired correctly.
//!
//! This test does **not** validate op-ordering between streams (no
//! `wait_for` is called here). It only tests that:
//! - Fences can be recorded on different streams.
//! - `elapsed_between()` can measure time between fences recorded on
//!   different streams.
//! - The driver returns plausible non-zero elapsed times.
//!
//! Cross-stream ordering enforcement (using `wait_for` to ensure stream B
//! waits for stream A's work) is deferred to Primitive 3 — see the
//! `fence_ordering` test doc for rationale.
//!
//! # GPU gate
//!
//! `#[cfg(feature = "gpu")]` — plain `cargo test` skips this file.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::stream::Fence;
use npp_rs::stream_context_for;

#[test]
fn cross_stream_timing() {
    let ctx = stream_context_for(0).expect("CUDA device 0 init");

    let src = CudaImage::<u8>::from_host(
        ctx.clone(),
        3,
        512,
        512,
        &vec![128u8; 512 * 512 * 3],
    )
    .expect("source allocation");

    let mut dst_a = CudaImage::<u8>::new(ctx.clone(), 3, 256, 256).expect("dst_a alloc");
    let mut dst_b = CudaImage::<u8>::new(ctx.clone(), 3, 256, 256).expect("dst_b alloc");

    // Record a fence before each resize. Both fences go on the context's
    // single stream — cross-stream timing requires separate stream handles,
    // which belong to Primitive 3 (op-on-caller-supplied-stream). For now,
    // this test proves `elapsed_between` works with fences on the same
    // stream produced independently (two record_fence calls).
    let before_a = ctx.record_fence().expect("record before_a");
    src.resize(&mut dst_a, ResizeInterpolation::Linear)
        .expect("resize a");
    let after_a = ctx.record_fence().expect("record after_a");

    let before_b = ctx.record_fence().expect("record before_b");
    src.resize(&mut dst_b, ResizeInterpolation::Linear)
        .expect("resize b");
    let after_b = ctx.record_fence().expect("record after_b");

    // Measure elapsed: within each pair (same-stream).
    let elapsed_a = ctx.elapsed_between(&before_a, &after_a).expect("elapsed a");
    let elapsed_b = ctx.elapsed_between(&before_b, &after_b).expect("elapsed b");

    assert!(
        elapsed_a.as_nanos() > 0,
        "elapsed_a must be non-zero"
    );
    assert!(
        elapsed_b.as_nanos() > 0,
        "elapsed_b must be non-zero"
    );

    // Verify geometry (not a golden test).
    ctx.synchronize().expect("synchronize");
    assert_eq!(dst_a.width(), 256);
    assert_eq!(dst_b.width(), 256);
}
```

### Step 3: Verify Phase 3 compiles and tests pass (GPU)

Run inside the Nix dev shell:

```bash
nix develop . --command cargo build --features gpu
nix develop . --command cargo test --features gpu fence_ 2>&1
```

Both new test files are prefixed with `fence`, so `cargo test --features gpu fence_`
selects only them. Both must pass.

If the cross-stream timing test (`fence_timing`) fails because two sequential
same-stream Resize operations complete so quickly that `elapsed_between`
returns zero, increase the workload size (e.g. 1024×1024 → 256×256) until
the driver returns a measurable non-zero Duration.

---

## Phase 4: Reconcile docs

Commit message: docs: reconcile stream-context.md and roadmap.md for F8.2 primitives (F8.2)

### Step 1: Update `docs/stream-context.md` for new primitives

Edit the file `docs/stream-context.md`.

1. **Replace the third sync-point bullet (lines 63–66):**

   Change from:

   ```markdown
   - [`StreamContext::device_fence()`](../npp/src/stream.rs) — Device-side
     only. Calls `cuEventSynchronize` (via `CudaDevice::wait_for`). Orders
     work between streams on the same device without blocking the host. Not
     sufficient for host readback safety.
   ```

   to:

   ```markdown
   - [`Fence`](../npp/src/stream.rs) — A CUDA event used for cross-stream
     ordering and device-side timing. Create via
     [`StreamContext::record_fence()`](../npp/src/stream.rs), wait on another
     stream via [`StreamContext::wait_for()`](../npp/src/stream.rs). Timing
     between two fences (same or cross-stream) uses
     [`StreamContext::elapsed_between()`](../npp/src/stream.rs).
   ```

2. **Update the execution-model description (lines 10–18) —** Replace the
   three-stage list with a four-stage list that includes the new primitive:

   Change from:

   ```markdown
   `StreamContext` provides three execution stages with two distinct ordering
   guarantees:

   1. **Host-to-device copies** are performed on the CUDA default stream
      (e.g. `CudaDevice::htod_sync_copy_into`).
   2. **NPP operations** are enqueued on the `StreamContext`'s forked stream
      via the `_Ctx` API.
   3. **Device-to-host read-backs** (`TryFrom<&CudaImage>`) execute on the
      **NULL stream** via cudarc's `dtoh_sync_copy`.
   ```

   to:

   ```markdown
   `StreamContext` provides four execution stages with two distinct ordering
   guarantees:

   1. **Host-to-device copies** are performed on the CUDA default stream
      (e.g. `CudaDevice::htod_sync_copy_into`).
   2. **NPP operations** are enqueued on the `StreamContext`'s forked stream
      via the `_Ctx` API.
   3. **Device-side fences** ([`Fence`](../npp/src/stream.rs)) can be
      recorded between operations for cross-stream ordering
      ([`wait_for`](../npp/src/stream.rs)) or timing
      ([`elapsed_between`](../npp/src/stream.rs)).
   4. **Device-to-host read-backs** (`TryFrom<&CudaImage>`) execute on the
      **NULL stream** via cudarc's `dtoh_sync_copy`.
   ```

3. **Update the "Why this closes C8" section (line 79) —** Remove the
   reference to `device_fence` in the list of sync points. Change line 79
   from:

   ```markdown
   - **explicit sync points** (`synchronize()`, `device_fence()`, readback
   ```

   to:

   ```markdown
   - **explicit sync points** (`synchronize()`, `Fence` ordering, readback
   ```

4. **Update the `!Send + !Sync` section (line 87) —** Change the claim that
   "CudaStream is `!Send + !Sync`" (which is false in cudarc 0.19.x) to
   reflect that we re-impose `!Send + !Sync` via `PhantomData<*const ()>`.
   Change from:

   ```markdown
   `CudaStream` is `!Send + !Sync` (CUDA streams are inherently thread-bound).
   `StreamContext` inherits this property.
   ```

   to:

   ```markdown
   `CudaStream` became `Send + Sync` in cudarc 0.19.x, but CUDA streams are
   inherently thread-bound (card C7). `StreamContext` re-imposes
   `!Send + !Sync` via `PhantomData<*const ()>` in its struct definition.
   ```

5. **Remove the "Relation to Session 2 (F8)" section (lines 93–100) —** This
   section describes F8's Session 2 pivot, which is historical context
   irrelevant to the current primitives. Delete everything from the heading
   `## Relation to Session 2 (F8)` to the end of the file.

### Step 2: Re-scope the F8.2 entry in `docs/roadmap.md`

Edit the file `docs/roadmap.md`.

Replace the entire F8.2 section (lines 575–589) with:

```markdown
## F8.2 — Stream-primitives floor (Fence + new_stream) *(complete)*

**What:** Deliver a minimal composable stream-primitives floor — `Fence`
(cross-stream ordering + device-side timing via `record_fence`/`wait_for`
/`elapsed_between`) and `new_stream` (caller-owned stream handles). The
broken `device_fence()` no-op stub was removed and benches migrated to
`synchronize()`. See [`stream.rs`](../npp/src/stream.rs) for the API.

**Why:** F8's original scope included "compute/copy overlap" — the
performance reason to use CUDA asynchronously. The core stream abstraction
(host-fenced readback, forked stream per context) landed in F8 core, but
true cross-stream primitives were deferred. This floor is the foundation
that user workflows (fan-out/fan-in, supertextures, MTCNN pipelines)
compose on top of — it is deliberately not a single workflow feature.

**What F8.2 is NOT:**
- Not async multi-stream chaining (Primitive 3 — op-on-caller-supplied-stream
  — is deferred).
- Not a pipeline or workflow API (no `fork()`, no pooling, no assignment).
- Not a replacement for [`Event`](../npp/src/stream.rs) (kept for bench
  backward compatibility).

**Verification:**
```bash
nix develop . --command cargo fmt --check
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo clippy --features gpu -- -D warnings
nix develop . --command cargo test
nix develop . --command cargo test --features gpu fence_
```
```

### Step 3: Update the sequencing diagram and note in `docs/roadmap.md`

Edit the file `docs/roadmap.md`.

1. **In the sequencing diagram (lines 625–633),** mark F8.2 as complete.
   Change:

   ```diff
   -      │      └─> F8.2 (async multi-stream chaining)
   +      │      └─> F8.2 (stream-primitives floor) [DONE]
   ```

2. **In the sequencing note (lines 635–643),** add F8.2 to the list of
   completed-and-merged features. Change:

   ```diff
   - **Sequencing note:** F1, F2, F5.1, F5.2, F5.3, F6, F6.2, F8 (core), and F8.1 are
   - complete and merged on `main`.
   + **Sequencing note:** F1, F2, F5.1, F5.2, F5.3, F6, F6.2, F8 (core), F8.1, and F8.2 are
   + complete and merged on `main`.
   ```

### Step 4: Verify docs are well-formed

Run inside the Nix dev shell:

```bash
nix develop . --command cargo doc --no-deps -p npp-rs -p npp-sys 2>&1 | tail -20
```

This must not produce any broken intra-doc links. The doc build may produce
warnings about missing `Fence` in older doc comments (e.g. `StreamContext`
doc comments reference `record_fence` and `wait_for` but these are added
in Phase 2 and should be resolved by the doc rebuild). If broken links
occur, fix them by updating the relevant doc comments to match the current
API names.

---

## Definition of Done

All four phases are complete. All commits are on `main`. The following
verification commands pass inside the Nix dev shell:

```bash
nix develop . --command cargo fmt --check
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo clippy --features gpu -- -D warnings
nix develop . --command cargo test
nix develop . --command cargo test --features gpu fence_
nix develop . --command cargo doc --no-deps -p npp-rs -p npp-sys
```

### Non-Goals (explicit firewall)

These are **not** part of F8.2 and must not be implemented:

1. Op-on-caller-supplied-stream (Primitive 3) — the ability to pass a
   caller-owned `Arc<CudaStream>` to individual NPP operations. Deferred.
2. Cross-stream op-ordering enforcement tests (requires Primitive 3).
3. Any form of stream pooling, assignment protocol, or pipeline vocabulary.
4. Deprecation or removal of the existing `Event`/`record_event`/`elapsed`
   timing API (kept for 18 golden bench backward compatibility).
5. F11 doc debt reconciliation (6 files, ~25 edits) — tracked separately.
6. The `raw_ctx` source inconsistency in `normalize_macros.rs` (uses `dst`
   instead of `self`) — noted but belongs to Primitive 3.

### Fences vs Events (backward compatibility)

The existing `Event`/`record_event`/`elapsed` timing API (used by 18 golden
benches) is **unmodified** and will coexist with the new `Fence` primitives.
No deprecation decision is made here — `Event`'s fate is left to be decided
alongside Primitive 3, shaped by real usage pressure rather than forward
design.
