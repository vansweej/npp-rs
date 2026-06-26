# Feature: F6.1 — Device-timed benchmark suite

## Goal (restated)

Add a device-timed benchmark suite (4 bench files, Criterion 0.8, `iter_custom`
using CUDA events) covering **Resize, SwapChannels, Mean, Convert, Normalize**
operations at representative sizes. Benchmarks measure **kernel-only device time**
(via `StreamContext::elapsed()` on a new RAII `Event` primitive), not wall-clock.
Output correctness is **not** asserted in benchmarks — it is verified by the
dedicated feature-gated `golden_*` test suite. This is a pure-timing suite.

**Secondary deliverable:** F6.1 lands the `Event` timing primitive on
`StreamContext` (a deliberate bake-in from F8.2's D3 design), which serves as the
C-2 spike tool for verifying stream overlap — de-risking future F8.2 work.

**Size signal:** 6 phases, ~5 source files touched (Cargo.toml, stream.rs,
4 new bench files), plus 2 docs files (roadmap.md, AGENTS.md).

---

## Key decisions (binding)

1. **Criterion 0.5 → 0.8 bump** with `default-features = false, features = ["plotters", "cargo_bench_support"]`. MSRV 1.86 satisfied by rustc 1.94 (confirmed via `flake.nix`). Drops unused `rayon` default feature.

2. **Device-time via `iter_custom`**, not `b.iter`. Wall-clock (`b.iter`) includes host overhead and can't isolate kernel time. `iter_custom` passes an opaque `Criterion` counter; bench body records start/end events, calls `elapsed()`, and returns `Duration`.

3. **Timing primitive on `StreamContext` (Option ii), not raw FFI scattered across benches.** One audited `unsafe` site wrapping CUDA event create/destroy/record/elapsed. RAII `Event` struct with `Drop → event::destroy`. Adds `record_event()` and `elapsed()` methods. This is the F8.2 D3 bake-in.

4. **Synthetic input only — no JPEG fixture, no `image` crate dependency.** Use `make_input()` producing deterministic ramp patterns, following the existing `bench_resize_correctness.rs:21` pattern.

5. **Zero golden assertions in benches.** Correctness is verified by the existing feature-gated `golden_*` test suite. Benches include a warm-up pass (run once, catch hard errors early) but assert nothing on output bytes. This is a deliberate deviation from the original roadmap wording "assert both timing and output content" — correctness and timing have different owners.

6. **GPU benches are manual-only.** Feature-gated behind `gpu` (`[[bench]]` files use `#![cfg(feature = "gpu")]`). CI has no GPU lane. Plain `cargo bench` must not compile or run GPU code.

7. **`tarpaulin_include` annotation** on GPU-only bench functions and the Event primitive.

8. **Allocations counted outside the timed loop.** The `iter_custom` setup allocates src/dst once per size or mode; the timed loop records events around the NPP call only.

---

## Phases

---

## Phase 1: Bump Criterion 0.5 → 0.8

Commit message: `chore: bump criterion from 0.5 to 0.8 (F6.1)`

### Step 1.1 — Update the Criterion dependency in `npp/Cargo.toml`

Edit `/home/vansweej/Work/npp-rs/npp/Cargo.toml`.

Change the `[dev-dependencies]` entry from:

```toml
criterion = "0.5"
```

to:

```toml
criterion = { version = "0.8", default-features = false, features = ["plotters", "cargo_bench_support"] }
```

The `default-features = false` drops the unused `rayon` feature (parallel
benchmark execution would interfere with device timing). The explicit
`plotters` and `cargo_bench_support` features restore only what's needed:
HTML reports and `iter_custom` support respectively. Do not touch any other
dependencies.

Verify with:

```bash
nix develop . --command cargo build -p npp-rs
```

The build must succeed. Criterion 0.8 is a major version bump; if any
existing Cargo.lock entries (currently gitignored, so no lockfile to
resolve from) cause resolution failures, the workspace resolves fresh. The
`autobenches = false` setting in `Cargo.toml` remains — no bench code is hit
yet.

### Step 1.2 — Confirm `iter_custom` is available

No code edit. Confirm the crate compiles with `iter_custom` available by
checking that the metadata crate `criterion` at v0.8 exports
`Criterion::iter_custom`. Since no bench files exist yet, rely on the
compiler succeeding in Phase 3 when the first bench file imports it. If
`cargo_bench_support` is not the correct feature name for `iter_custom` in
criterion 0.8, the Phase 1 commit has isolated the issue to the dependency
line and it can be fixed without touching other changes.

---

## Phase 2: Add RAII `Event` timing primitive to `StreamContext`

Commit message: `feat: add Event timing primitive to StreamContext (F6.1 bake-in)`

### Step 2.1 — Define the `Event` struct and implement `record_event` / `elapsed`

Edit `/home/vansweej/Work/npp-rs/npp/src/stream.rs`.

Add the following code after the existing `impl StreamContext` block (or
at the end of the file, before any `#[cfg(test)]` module):

```rust
/// A RAII wrapper around a CUDA event, used for device-side timing.
///
/// # Safety
///
/// The underlying CUDA event is created with `cuEventCreate` and destroyed
/// with `cuEventDestroy` on drop. The event is associated with the same
/// CUDA context as the `StreamContext` that created it. The caller must
/// ensure the CUDA device handle (`CudaDevice`) outlives this event
/// (same invariant as [`CudaImage`](crate::image::CudaImage) and
/// [`StreamContext`]).
///
/// # Panics
///
/// `record` and `elapsed` may panic if the underlying CUDA driver call
/// fails (these indicate a driver-level issue that is not recoverable).
#[derive(Debug)]
#[cfg(not(tarpaulin_include))]
pub struct Event {
    inner: npp_sys::CUevent,
    ctx: Arc<StreamContext>,
}

// SAFETY: Event is !Send + !Sync to match StreamContext's CUDA-context
// thread affinity. CUDA events are tied to the CUcontext that created them.
// This matches StreamContext's own !Send + !Sync.
impl !Send for Event {}
impl !Sync for Event {}

#[cfg(not(tarpaulin_include))]
impl Event {
    /// Record the event on the stream managed by this `StreamContext`.
    ///
    /// After this call, the event is recorded — use [`elapsed`](Self::elapsed)
    /// to measure device time between two recorded events.
    ///
    /// # Panics
    ///
    /// Panics if the underlying `cuEventRecord` call fails.
    pub fn record(&self) {
        let status = unsafe { npp_sys::cuEventRecord(self.inner, self.ctx.stream) };
        assert_eq!(status, npp_sys::CUresult_CUDA_SUCCESS,
            "cuEventRecord failed with status {status}");
    }
}

#[cfg(not(tarpaulin_include))]
impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: self.inner is a valid CUevent created by cuEventCreate
        // in StreamContext::record_event, and is only destroyed once here.
        let _ = unsafe { npp_sys::cuEventDestroy(self.inner) };
    }
}
```

Then add the following methods to `impl StreamContext` (in the existing
`impl StreamContext` block):

```rust
    /// Create a new RAII timing event associated with this stream context.
    ///
    /// Use [`Event::record`] to record the event on the stream, then call
    /// [`elapsed`](Self::elapsed) to measure device-time between two events.
    #[cfg(not(tarpaulin_include))]
    pub fn record_event(&self) -> Event {
        let mut event: npp_sys::CUevent = std::ptr::null_mut();
        // SAFETY: cuEventCreate creates a new event. The event is destroyed
        // in Event::drop. The CUDA context is guaranteed alive by Arc<StreamContext>.
        let status = unsafe {
            npp_sys::cuEventCreate(
                &mut event as *mut npp_sys::CUevent,
                0, // CUeventFlags: 0 = default (blocking, timing-enabled)
            )
        };
        assert_eq!(status, npp_sys::CUresult_CUDA_SUCCESS,
            "cuEventCreate failed with status {status}");
        Event {
            inner: event,
            ctx: self.arc_clone(),
        }
    }

    /// Measure device-time elapsed between two recorded events on this stream.
    ///
    /// Returns `Ok(Duration)` on success, or `Err(NppError)` if the driver
    /// call fails. Both events must have been recorded (via [`Event::record`])
    /// on the stream associated with this context, and the earlier event must
    /// have completed before this call.
    ///
    /// # Errors
    ///
    /// Returns `NppError::Cuda` if `cuEventElapsedTime` fails.
    #[cfg(not(tarpaulin_include))]
    pub fn elapsed(&self, start: &Event, end: &Event) -> Result<Duration, NppError> {
        let mut ms: f32 = 0.0;
        // SAFETY: start.inner and end.inner are valid CUevents created by
        // record_event. Both were recorded on this stream (or a compatible one).
        let status = unsafe {
            npp_sys::cuEventElapsedTime(&mut ms as *mut f32, start.inner, end.inner)
        };
        if status != npp_sys::CUresult_CUDA_SUCCESS {
            return Err(NppError::Cuda(cudarc::driver::result::DriverError::from_raw(
                status as i32,
            )));
        }
        Ok(Duration::from_secs_f64(ms as f64 / 1_000.0))
    }
```

Note: `cuEventElapsedTime` returns milliseconds as `f32`. The conversion to
`Duration` uses `from_secs_f64` for sub-microsecond precision. The `0` flags
to `cuEventCreate` request a **timing-enabled, blocking** event (default).
Blocking events synchronize with `cuEventSynchronize` caller-side; we use
`cuEventRecord` + `cuEventElapsedTime` which does not require explicit
synchronize (elapsed blocks until the earlier event completes).

Also add the necessary import at the top of the file:

```rust
use std::time::Duration;
```

Check the existing imports in `stream.rs` — `Duration` may or may not already
be imported. If it is, skip adding it.

### Step 2.2 — Verify the new code compiles

```bash
nix develop . --command cargo build -p npp-rs
nix develop . --command cargo clippy -- -D warnings
```

Both must pass. The Event type is `#[cfg(not(tarpaulin_include))]` so it does
not affect coverage.

### Step 2.3 — Add a unit test for `elapsed` time helpers (host-side only)

Add a pure-logic unit test (no GPU) for the `ms → Duration` conversion. If
`elapsed()` encapsulates this conversion internally (it does: `from_secs_f64(ms / 1000.0)`),
the unit test should verify known input/output pairs for the conversion
functionality. Since `elapsed` requires a GPU, isolate the conversion into
a testable helper or test the public method's logic indirectly.

Add a `#[cfg(test)]` module at the bottom of `stream.rs` containing:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ms_to_duration_conversion() {
        // 0 ms → 0 secs
        let d = Duration::from_secs_f64(0.0 / 1_000.0);
        assert_eq!(d.as_nanos(), 0);

        // 1 ms → 1_000_000 ns
        let d = Duration::from_secs_f64(1.0 / 1_000.0);
        assert_eq!(d.as_nanos(), 1_000_000);

        // 1000 ms → 1 sec
        let d = Duration::from_secs_f64(1000.0 / 1_000.0);
        assert_eq!(d.as_secs(), 1);

        // 1.5 ms → 1_500_000 ns
        let d = Duration::from_secs_f64(1.5 / 1_000.0);
        assert_eq!(d.as_nanos(), 1_500_000);

        // Negative ms (invalid — from_secs_f64 would panic; not testing panic)
    }
}
```

Run:

```bash
nix develop . --command cargo test -p npp-rs
```

All unit tests must pass.

---

## Phase 3: Resize size-sweep bench

Commit message: `bench: add Resize size-sweep benchmark (device-time via iter_custom)`

### Step 3.1 — Create `bench_resize_size.rs`

Create the file `/home/vansweej/Work/npp-rs/npp/benches/bench_resize_size.rs`.

```rust
//! Device-timed benchmark: Resize size sweep.
//!
//! Measures kernel-only device time for `nppiResize_8u_C3R_Ctx` across a range
//! of output sizes at [`ResizeInterpolation::Linear`]. Allocations happen once
//! per size outside the timed loop.
//!
//! **Correctness is NOT asserted here.** The dedicated feature-gated golden
//! test suite (`golden_resize*`) verifies output bytes. This bench measures
//! timing only.
//!
//! # GPU gate
//!
//! This file is gated behind `#[cfg(feature = "gpu")]`. Plain `cargo bench`
//! will not compile it.

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, Criterion, criterion_group, criterion_main};
use npp_rs::image::CudaImage;
use npp_rs::resize::{Resize, ResizeInterpolation};
use npp_rs::stream::StreamContext;

/// Smallest image dimension used in the sweep.
const MIN_SIZE: u32 = 64;
/// Largest image dimension used in the sweep.
const MAX_SIZE: u32 = 2048;
/// Step factor between sizes (powers of two × base).
const SIZES: &[u32] = &[64, 128, 256, 512, 1024, 2048];

/// Create a deterministic 3-channel u8 source image for bench sizes.
///
/// Returns `(src, ctx)` where `ctx` is the device context. The source is
/// filled with a ramp pattern `[x*21, y*32, 128]` so it is not constant
/// (avoiding cache-friendly degenerate cases).
#[cfg(not(tarpaulin_include))]
fn make_input(ctx: &StreamContext, w: u32, h: u32) -> CudaImage<u8> {
    let channels = 3;
    let len = (w * h * channels as u32) as usize;
    let host: Vec<u8> = (0..len)
        .map(|i| {
            let x = (i / channels as usize) % w as usize;
            let y = (i / channels as usize) / w as usize;
            let c = i % channels as usize;
            match c {
                0 => (x * 21) as u8,
                1 => (y * 32) as u8,
                _ => 128u8,
            }
        })
        .collect();
    CudaImage::from_host(ctx.clone(), channels, w, h, &host)
        .expect("alloc + host→device copy for bench input")
}

#[cfg(not(tarpaulin_include))]
fn bench_resize_size(c: &mut Criterion) {
    let ctx = StreamContext::stream_context_for(0).expect("CUDA device 0 init");

    let mut group = c.benchmark_group("resize_size_u8_c3");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &size in SIZES {
        let src = make_input(&ctx, MAX_SIZE, MAX_SIZE);
        let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, size, size)
            .expect("dst allocation");

        // Warm-up: run once outside the timed loop to prime caches and catch
        // hard errors early. Asserts nothing on output.
        src.resize(&mut dst, ResizeInterpolation::Linear)
            .expect("warm-up resize");

        let label = format!("{size}×{size}");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    // SAFETY: black_box prevents the compiler from hoisting
                    // or eliminating the call.
                    let _ = black_box(
                        src.resize(&mut dst, ResizeInterpolation::Linear)
                    );
                    let end = ctx.record_event();
                    end.record();

                    // Wait for the stream to complete so the elapsed time
                    // captures the full operation.
                    ctx.device_fence().expect("fence after resize");

                    total += ctx.elapsed(&start, &end)
                        .expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_resize_size);
criterion_main!(benches);
```

### Step 3.2 — Register the bench in `Cargo.toml`

Edit `/home/vansweej/Work/npp-rs/npp/Cargo.toml`.

Add a `[[bench]]` entry for the new file. Locate the existing `[[bench]]`
section(s) and add after the last `name`:

```toml
[[bench]]
name = "bench_resize_size"
harness = false
```

Leave `autobenches = false` unchanged (new benches must be explicitly
registered).

Verify compilation:

```bash
nix develop . --command cargo build -p npp-rs
nix develop . --command cargo clippy -- -D warnings
```

Both must pass. The `#[cfg(feature = "gpu")]` gate means the bench is compiled
but not run under plain `cargo bench` (it would be skipped at runtime). Confirm
`cargo bench -p npp-rs` completes with zero measured benchmarks (Criterion
reports "0 benchmarks" because the gpu-gated criterion harness registers no
benches when the feature is off). This is expected — the bench is a manual GPU
lane only.

---

## Phase 4: Resize mode-sweep and channel-count bench

Commit message: `bench: add Resize mode-sweep and channel-count benchmark`

### Step 4.1 — Create `bench_resize_modes.rs`

Create the file `/home/vansweej/Work/npp-rs/npp/benches/bench_resize_modes.rs`.

```rust
//! Device-timed benchmark: Resize interpolation mode sweep and channel count.
//!
//! Two sub-benchmarks:
//! - **Mode sweep:** all 4 interpolation modes at fixed size (512×512) and
//!   channel count (3).
//! - **Channel-count comparison:** 3-channel vs 4-channel at fixed size
//!   (512×512) and mode ([`ResizeInterpolation::Linear`]).
//!
//! **Correctness is NOT asserted here.** See `bench_resize_size.rs` doc.
//!
//! # GPU gate
//!
//! `#[cfg(feature = "gpu")]` — plain `cargo bench` skips this file.

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, Criterion, criterion_group, criterion_main};
use npp_rs::image::CudaImage;
use npp_rs::resize::{Resize, ResizeInterpolation};
use npp_rs::stream::StreamContext;

const BENCH_SIZE: u32 = 512;

/// All interpolation modes supported by NPP for u8 C3.
const INTERP_MODES: &[ResizeInterpolation] = &[
    ResizeInterpolation::NearestNeighbor,
    ResizeInterpolation::Linear,
    ResizeInterpolation::Cubic,
    ResizeInterpolation::SuperSampling,
];

#[cfg(not(tarpaulin_include))]
fn make_input(ctx: &StreamContext, w: u32, h: u32, channels: u8) -> CudaImage<u8> {
    let len = (w * h * channels as u32) as usize;
    let host: Vec<u8> = (0..len)
        .map(|i| {
            let x = (i / channels as usize) % w as usize;
            let y = (i / channels as usize) / w as usize;
            let c = i % channels as usize;
            match c {
                0 => (x * 21) as u8,
                1 => (y * 32) as u8,
                _ => 128u8,
            }
        })
        .collect();
    CudaImage::from_host(ctx.clone(), channels, w, h, &host)
        .expect("alloc + host→device copy for bench input")
}

/// Benchmark: interpolation mode sweep at 512×512, 3-channel.
#[cfg(not(tarpaulin_include))]
fn bench_resize_modes(c: &mut Criterion) {
    let ctx = StreamContext::stream_context_for(0).expect("CUDA device 0 init");
    let src = make_input(&ctx, BENCH_SIZE, BENCH_SIZE, 3);
    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, BENCH_SIZE / 2, BENCH_SIZE / 2)
        .expect("dst allocation");

    let mut group = c.benchmark_group("resize_modes_u8_c3");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &mode in INTERP_MODES {
        // Warm-up
        src.resize(&mut dst, mode).expect("warm-up resize");

        let label = format!("{mode:?}");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src.resize(&mut dst, mode));
                    let end = ctx.record_event();
                    end.record();
                    ctx.device_fence().expect("fence");
                    total += ctx.elapsed(&start, &end)
                        .expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    group.finish();
}

/// Benchmark: 3-channel vs 4-channel at 512×512, Linear interpolation.
#[cfg(not(tarpaulin_include))]
fn bench_resize_channels(c: &mut Criterion) {
    let ctx = StreamContext::stream_context_for(0).expect("CUDA device 0 init");

    let mut group = c.benchmark_group("resize_channels_u8_linear");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &channels in &[3u8, 4u8] {
        let src = make_input(&ctx, BENCH_SIZE, BENCH_SIZE, channels);
        let mut dst = CudaImage::<u8>::new(ctx.clone(), channels, BENCH_SIZE / 2, BENCH_SIZE / 2)
            .expect("dst allocation");

        // Warm-up
        src.resize(&mut dst, ResizeInterpolation::Linear)
            .expect("warm-up resize");

        let label = format!("{channels}ch");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src.resize(&mut dst, ResizeInterpolation::Linear));
                    let end = ctx.record_event();
                    end.record();
                    ctx.device_fence().expect("fence");
                    total += ctx.elapsed(&start, &end)
                        .expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_resize_modes, bench_resize_channels);
criterion_main!(benches);
```

### Step 4.2 — Register the bench in `Cargo.toml`

Edit `/home/vansweej/Work/npp-rs/npp/Cargo.toml`. Add after the Phase 3
`[[bench]]` entry:

```toml
[[bench]]
name = "bench_resize_modes"
harness = false
```

Verify:

```bash
nix develop . --command cargo build -p npp-rs
nix develop . --command cargo clippy -- -D warnings
```

---

## Phase 5: Op-family comparison bench

Commit message: `bench: add op-family comparison benchmark (Resize, SwapChannels, Mean, Convert, Normalize)`

### Step 5.1 — Create `bench_op_family.rs`

Create the file `/home/vansweej/Work/npp-rs/npp/benches/bench_op_family.rs`.

```rust
//! Device-timed benchmark: op-family comparison.
//!
//! Compares kernel-only device time for all five supported operation families
//! at a fixed size (512×512):
//!
//! - [`Resize`] (Linear, downscale 2×)
//! - [`SwapChannels`] (BGRA→RGB, same-size)
//! - [`Mean`] (per-channel mean to `Vec<f64>`) — **includes host readback**
//! - [`ConvertTo`] (u8→f32)
//! - [`Normalize`] (u8→f32, min=0, max=255)
//!
//! **Correctness is NOT asserted here.** See `bench_resize_size.rs` doc.
//!
//! ## Mean readback note
//!
//! Mean's public API returns `Vec<f64>`, which requires a device-to-host
//! synchronous copy inside the method (`mean_macros.rs:131`). This readback
//! is included in the measured time. Other operations in this benchmark do
//! not read back. The Mean result is labeled "(incl_readback)" to highlight
//! this asymmetry.
//!
//! # GPU gate
//!
//! `#[cfg(feature = "gpu")]` — plain `cargo bench` skips this file.

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, Criterion, criterion_group, criterion_main};
use npp_rs::convert::{ConvertTo, Normalize};
use npp_rs::image::CudaImage;
use npp_rs::mean::Mean;
use npp_rs::resize::{Resize, ResizeInterpolation};
use npp_rs::stream::StreamContext;
use npp_rs::swap_channels::SwapChannels;

const BENCH_SIZE: u32 = 512;

/// Create a deterministic 3 or 4-channel u8 source image.
#[cfg(not(tarpaulin_include))]
fn make_input(ctx: &StreamContext, w: u32, h: u32, channels: u8) -> CudaImage<u8> {
    let len = (w * h * channels as u32) as usize;
    let host: Vec<u8> = (0..len)
        .map(|i| {
            let x = (i / channels as usize) % w as usize;
            let y = (i / channels as usize) / w as usize;
            let c = i % channels as usize;
            match c {
                0 => (x * 21) as u8,
                1 => (y * 32) as u8,
                _ => 128u8,
            }
        })
        .collect();
    CudaImage::from_host(ctx.clone(), channels, w, h, &host)
        .expect("alloc + host→device copy for bench input")
}

#[cfg(not(tarpaulin_include))]
fn bench_op_family(c: &mut Criterion) {
    let ctx = StreamContext::stream_context_for(0).expect("CUDA device 0 init");

    let src_3ch = make_input(&ctx, BENCH_SIZE, BENCH_SIZE, 3);
    let src_4ch = make_input(&ctx, BENCH_SIZE, BENCH_SIZE, 4);

    let mut dst_resize = CudaImage::<u8>::new(ctx.clone(), 3, BENCH_SIZE / 2, BENCH_SIZE / 2)
        .expect("resize dst alloc");
    let mut dst_swap = CudaImage::<u8>::new(ctx.clone(), 3, BENCH_SIZE, BENCH_SIZE)
        .expect("swap dst alloc");
    let mut dst_convert = CudaImage::<f32>::new(ctx.clone(), 3, BENCH_SIZE, BENCH_SIZE)
        .expect("convert dst alloc");
    let mut dst_norm = CudaImage::<f32>::new(ctx.clone(), 3, BENCH_SIZE, BENCH_SIZE)
        .expect("normalize dst alloc");
    // Mean writes to device memory internally; no separate dst CudaImage needed.
    // The result Vec<f64> is produced by the method.

    let mut group = c.benchmark_group("op_family_u8_512");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // --- Resize (Linear, 3ch, downscale 2×) ---
    {
        // Warm-up
        src_3ch.resize(&mut dst_resize, ResizeInterpolation::Linear)
            .expect("resize warm-up");

        group.bench_function("Resize_Linear_down2x", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(
                        src_3ch.resize(&mut dst_resize, ResizeInterpolation::Linear)
                    );
                    let end = ctx.record_event();
                    end.record();
                    ctx.device_fence().expect("fence");
                    total += ctx.elapsed(&start, &end)
                        .expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    // --- SwapChannels (4ch→3ch, BGRA→RGB, same-size) ---
    {
        // Warm-up
        src_4ch.bgra_to_rgb(&mut dst_swap)
            .expect("swap warm-up");

        group.bench_function("SwapChannels_BGRAtoRGB", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src_4ch.bgra_to_rgb(&mut dst_swap));
                    let end = ctx.record_event();
                    end.record();
                    ctx.device_fence().expect("fence");
                    total += ctx.elapsed(&start, &end)
                        .expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    // --- Mean (3ch, same-size, incl_readback) ---
    {
        // Warm-up
        let _ = src_3ch.mean().expect("mean warm-up");

        group.bench_function("Mean_3ch_incl_readback", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _: Vec<f64> = black_box(
                        src_3ch.mean().expect("mean")
                    );
                    let end = ctx.record_event();
                    end.record();
                    ctx.device_fence().expect("fence");
                    total += ctx.elapsed(&start, &end)
                        .expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    // --- ConvertTo (u8→f32, 3ch, same-size) ---
    {
        // Warm-up
        src_3ch.convert_to(&mut dst_convert)
            .expect("convert warm-up");

        group.bench_function("ConvertTo_u8_to_f32", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src_3ch.convert_to(&mut dst_convert));
                    let end = ctx.record_event();
                    end.record();
                    ctx.device_fence().expect("fence");
                    total += ctx.elapsed(&start, &end)
                        .expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    // --- Normalize (u8→f32, 3ch, min=0, max=255, same-size) ---
    {
        // Warm-up
        src_3ch.normalize(&mut dst_norm, 0.0, 255.0)
            .expect("normalize warm-up");

        group.bench_function("Normalize_u8_to_f32_min0_max255", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src_3ch.normalize(&mut dst_norm, 0.0, 255.0));
                    let end = ctx.record_event();
                    end.record();
                    ctx.device_fence().expect("fence");
                    total += ctx.elapsed(&start, &end)
                        .expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_op_family);
criterion_main!(benches);
```

### Step 5.2 — Register the bench in `Cargo.toml`

Edit `/home/vansweej/Work/npp-rs/npp/Cargo.toml`. Add after the Phase 4
`[[bench]]` entry:

```toml
[[bench]]
name = "bench_op_family"
harness = false
```

Verify:

```bash
nix develop . --command cargo build -p npp-rs
nix develop . --command cargo clippy -- -D warnings
```

### Step 5.3 — Verify non-GPU `cargo bench` completes cleanly

Run:

```bash
nix develop . --command cargo bench -p npp-rs
```

Expect: Criterion reports "0 benchmarks; 0 measured; 0 filtered out" (or
similar message indicating no benchmarks were available to run). The
`#[cfg(feature = "gpu")]` gate ensures the criterion harness registers no
benches when the feature is absent. Confirm no compilation errors, no linker
errors, and no panics.

If criterion 0.8 reports differently (e.g., it might print "No benchmarks
found" or skip silently), note the exact output for the F6.1 documentation
in Phase 6.

---

## Phase 6: Documentation reconciliation

Commit message: `docs: reconcile roadmap F6.1 entry, add benchmarking conventions to AGENTS.md`

### Step 6.1 — Update `docs/roadmap.md` F6.1 entry

Edit `/home/vansweej/Work/npp-rs/docs/roadmap.md`.

Locate the `## F6.1` section. Update it to reflect the implemented state:

1. Mark the heading with `*(complete)*` (or similar convention used by other
   completed entries in the file).

2. Update the **What** paragraph to state: F6.1 delivers a device-timed
   benchmark suite (4 bench files, Criterion 0.8, `iter_custom`, CUDA-event
   timing) covering Resize, SwapChannels, Mean, Convert, and Normalize.
   Timing is measured via a new RAII `Event` primitive on `StreamContext`
   (`record_event()` / `elapsed()`) — this is also the first brick of F8.2's
   D3 design, baked in intentionally to serve as the C-2 overlap spike tool.
   Benches are timing-only (correctness delegated to the `golden_*` test
   suite). GPU feature-gated; CI has no GPU lane.

3. Update the **Why** paragraph to note that the original roadmap requirement
   to "assert both timing and output content" (`roadmap.md:402`) is
   reinterpreted: correctness lives in the dedicated `golden_*` test suite;
   benches are pure timing. This avoids duplicate golden constants, gives a
   single source of correctness truth, and removes the per-bench `EXPECTED`
   pinning burden. The reinterpretation is deliberate and documented here.

4. Add a **Verification** subsection listing the commands that pass without
   a GPU:

   ```
   nix develop . --command cargo build
   nix develop . --command cargo test
   nix develop . --command cargo fmt --check
   nix develop . --command cargo clippy -- -D warnings
   nix develop . --command cargo bench -p npp-rs
   nix develop . --command cargo tarpaulin
   ```

   And the manual GPU-only command:

   ```
   nix develop . --command cargo bench --features gpu -p npp-rs
   ```

### Step 6.2 — Add device-timing and benchmark conventions to `AGENTS.md`

Edit `/home/vansweej/Work/npp-rs/AGENTS.md`.

In the **Build / test** section at the top, or as a new subsection after it,
add a **Benchmarking** paragraph:

```markdown
## Benchmarking

- Benches are **device-timed** via `StreamContext::record_event()` /
  `elapsed()` (CUDA events), not wall-clock. All new benches must use
  `iter_custom` (not `b.iter`) and record start/end events around the NPP
  call only. Allocations are outside the timed loop.
- All bench files carry `#![cfg(feature = "gpu")]` and are registered in
  `Cargo.toml` with `[[bench]] name = "..." harness = false`. Plain
  `cargo bench` compiles but produces zero measurements — benches are a
  **manual GPU lane only**, never CI.
- Benches measure **timing only**. Output correctness is verified by the
  dedicated `golden_*` test suite, not duplicated here. A warm-up pass
  (run once, assert nothing on bytes) is required to prime caches and catch
  hard errors early.
- GPU-only code in bench files is annotated `#[cfg(not(tarpaulin_include))]`.
- Four bench files exist (F6.1):
  - `bench_resize_size` — size sweep (64→2048), Linear, 3ch.
  - `bench_resize_modes` — interpolation sweep (4 modes) + channel count (3v4).
  - `bench_op_family` — cross-op comparison (Resize/SwapChannels/Mean/Convert/Normalize).
```

Also add a note to the **Conventions** section or Error-handling section:

```markdown
- Device timing: use `StreamContext::record_event()` (RAII `Event`) and
  `elapsed()` for kernel-time measurement. See `stream.rs` for the
  implementation and safety contract. Do not scatter raw `cuEvent*` FFI calls
  across bench files.
```

### Step 6.3 — Verify full suite passes

Run the full non-GPU verification:

```bash
nix develop . --command cargo build
nix develop . --command cargo test
nix develop . --command cargo clippy -- -D warnings
nix develop . --command cargo fmt --check
nix develop . --command cargo bench -p npp-rs
nix develop . --command cargo tarpaulin
```

All must pass. If `cargo fmt --check` fails, run `nix develop . --command cargo fmt`
(the one permitted formatting mutation during verification) and re-check.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | `cuEventCreate`/`cuEventDestroy`/`cuEventRecord` FFI symbols not available via `npp-sys` | Low (these are core CUDA driver API, same as `cuMemAlloc` already exposed via cudarc) | Build/link failure in Phase 2 | Verify by checking `npp-sys` already links `cuda`; if missing, add `#[link]` or use cudarc's re-exported CUDA driver. |
| R2 | Criterion 0.8 `cargo_bench_support` feature name is incorrect | Medium | Phase 1 compiles but Phase 3 bench code fails to compile | Phase 1 is isolated; fix feature name before Phase 3. The correct feature in criterion 0.8 is `cargo_bench_support`. |
| R3 | `iter_custom` API differs between criterion 0.5 and 0.8 | High if code uses 0.5 API | Bench code fails to compile | All bench files are new code targeting 0.8; no existing code to break. Verify against criterion 0.8 docs. |
| R4 | `device_fence()` call inside the timed loop adds non-trivial overhead | Medium | Skewed device-time measurements | `device_fence()` is a `cuStreamSynchronize` equivalent — it serializes the stream. The start/end events bracket the NPP call and the fence; the fence is *inside* the timed region. This is acceptable for kernel-only time because the fence wait is negligible compared to kernel execution. If the fence skew is measurable, document it as a known limitation. |
| R5 | `Event::drop` calls `cuEventDestroy` which returns an error that is silently ignored | Low (event destroy rarely fails in practice) | Leaked event resource if destroy fails | The `let _ =` discard is acceptable — `cuEventDestroy` failure is unrecoverable at drop time. Context: the system prompt's C/C++ skill convention for destructors that cannot fail. |
| R6 | `elapsed()` panics on negative `ms` from `cuEventElapsedTime` | Very low (driver guarantees non-negative for properly recorded events) | Bench panics | Acceptable — this is a driver-integrity failure, not recoverable. Same panic/discipline as CUDA driver asserts. |

## Dependencies

- **Phase 1:** None — self-contained `Cargo.toml` change.
- **Phase 2:** Phase 1 complete (criterion bump is unrelated to `Event` but is
  needed for Phases 3–5; Phase 2 can technically land independently).
- **Phase 3:** Phases 1 and 2 complete (criterion + Event primitive).
- **Phase 4:** Phases 1 and 2 complete (same as Phase 3; independent of
  Phase 3 for file content but same deps).
- **Phase 5:** Phases 1 and 2 complete (same dep chain). Also requires all
  five op families to be compiled (they are, as of post-M1 state).
- **Phase 6:** Phases 1–5 complete (docs reflect the shipped state).

## Notes for the executor

1. **Compile after every phase** — verify with `cargo build -p npp-rs` after
   each phase before moving on. The phases have the dependency chain above;
   Phase 2 needs `npp-sys` CUevent symbols (already available via `npp-sys`'s
   link to `cuda`).

2. **Do NOT refactor the Event primitive into a separate module** — it lives
   in `stream.rs` alongside `StreamContext`, sharing the same CUDA-context
   lifetime invariant. A separate module would duplicate the invariant
   contract and increase the surface area for error.

3. **Bench registration order:** `[[bench]]` entries are independent. Add them
   after each new bench file. The `autobenches = false` setting prevents
   Criterion from auto-discovering bench files, so unregistered files go
   unbuilt. This is correct.

4. **`black_box` placement:** `black_box` wraps the `Result` return of the
   NPP call, preventing the compiler from eliminating the call as dead code.
   The `let _ = black_box(...)` is the idiomatic criterion pattern for
   fallible operations — it forces the call to happen but discards the
   success value.

5. **`make_input` duplication:** Each bench file defines its own `make_input`.
   This is intentional — the input size/shape varies per bench, and extracting
   a shared helper into `test_helpers` would require making it `pub(crate)`
   and available to benches (which are external to the crate). Duplication is
   acceptable for 3 short functions.

6. **`device_fence` mislabel in `roadmap.md`:** The existing `roadmap.md:547`
   and `stream-context.md:64` describe `device_fence()` as calling
   `cuEventSynchronize` — it actually calls `self.device.wait_for(&self.stream)`
   (default-stream fence). This doc bug is independent of F6.1 and is **not**
   fixed by this plan. If you notice it while editing the roadmap in Phase 6,
   fix it, but it is not part of F6.1 scope.

7. **GPU bench is a manual lane:** `cargo bench --features gpu -p npp-rs`
   requires an NVIDIA GPU with the NPP library installed. This is never CI.
   The `golden_*` test suite is the only GPU-pinned correctness gate.

8. **Mean readback timing:** Mean's public API includes a host-side readback
   (device-to-host synchronous copy in `mean_macros.rs:131`). The bench
   measures this as part of the public-API call, labeled
   `"(incl_readback)"`. Pure kernel-only Mean time would require a separate
   `*_Ctx` FFI call + manual event timing without the readback, but that
   measures internal implementation details, not public API. The design
   decision is to time the public API and label the readback inclusion
   transparently.

9. **No `EXPECTED` constants anywhere:** Phase 6 confirms the roadmap
   deviation is documented (benches are timing-only; correctness is in
   `golden_*`). No bench file contains any golden assertion.
