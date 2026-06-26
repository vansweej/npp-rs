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

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, Criterion, criterion_group, criterion_main};
use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::stream::StreamContext;
use npp_rs::stream_context_for;

/// Largest image dimension used in the sweep (source size).
const MAX_SIZE: u32 = 2048;

/// Step factor between sizes (powers of two × base).
const SIZES: &[u32] = &[64, 128, 256, 512, 1024, 2048];

/// Create a deterministic 3-channel u8 source image for bench sizes.
///
/// Returns `(src, ctx)` where `ctx` is the device context. The source is
/// filled with a ramp pattern `[x*21, y*32, 128]` so it is not constant
/// (avoiding cache-friendly degenerate cases).
#[cfg(not(tarpaulin_include))]
fn make_input(ctx: &Arc<StreamContext>, w: u32, h: u32) -> CudaImage<u8> {
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
    let ctx = stream_context_for(0).expect("CUDA device 0 init");

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
