//! Device-timed benchmark: op-family comparison.
//!
//! Compares kernel-only device time for all five supported operation families
//! at a fixed size (512×512):
//!
//! - [`Resize`] (Linear, downscale 2×)
//! - [`SwapChannels`] (BGRA→RGB, same-size)
//! - [`Mean`] (per-channel mean to `Vec<f64>`) — **includes host readback**
//! - [`Convert`] (u8→f32)
//! - [`Normalize`] (u8→f32)
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

use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use npp_rs::image::CudaImage;
use npp_rs::imageops::{ConvertTo, Mean, Normalize, Resize, ResizeInterpolation, SwapChannels};
use npp_rs::stream::StreamContext;
use npp_rs::stream_context_for;
use std::hint::black_box;

const BENCH_SIZE: u32 = 512;

/// Create a deterministic 3 or 4-channel u8 source image.
#[cfg(not(tarpaulin_include))]
fn make_input(ctx: &Arc<StreamContext>, w: u32, h: u32, channels: u8) -> CudaImage<u8> {
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
    let ctx = stream_context_for(0).expect("CUDA device 0 init");

    let src_3ch = make_input(&ctx, BENCH_SIZE, BENCH_SIZE, 3);
    let src_4ch = make_input(&ctx, BENCH_SIZE, BENCH_SIZE, 4);

    let mut dst_resize = CudaImage::<u8>::new(ctx.clone(), 3, BENCH_SIZE / 2, BENCH_SIZE / 2)
        .expect("resize dst alloc");
    let mut dst_swap =
        CudaImage::<u8>::new(ctx.clone(), 3, BENCH_SIZE, BENCH_SIZE).expect("swap dst alloc");
    let mut dst_convert =
        CudaImage::<f32>::new(ctx.clone(), 3, BENCH_SIZE, BENCH_SIZE).expect("convert dst alloc");
    let mut dst_norm =
        CudaImage::<f32>::new(ctx.clone(), 3, BENCH_SIZE, BENCH_SIZE).expect("normalize dst alloc");
    // Mean writes to device memory internally; no separate dst CudaImage needed.
    // The result Vec<f64> is produced by the method.

    let mut group = c.benchmark_group("op_family_u8_512");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // --- Resize (Linear, 3ch, downscale 2×) ---
    {
        // Warm-up
        src_3ch
            .resize(&mut dst_resize, ResizeInterpolation::Linear)
            .expect("resize warm-up");

        group.bench_function("Resize_Linear_down2x", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src_3ch.resize(&mut dst_resize, ResizeInterpolation::Linear));
                    let end = ctx.record_event();
                    end.record();
                    ctx.synchronize().expect("sync");
                    total += ctx.elapsed(&start, &end).expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    // --- SwapChannels (4ch→3ch, BGRA→RGB, same-size) ---
    {
        // Warm-up
        src_4ch.bgra_to_rgb(&mut dst_swap).expect("swap warm-up");

        group.bench_function("SwapChannels_BGRAtoRGB", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src_4ch.bgra_to_rgb(&mut dst_swap));
                    let end = ctx.record_event();
                    end.record();
                    ctx.synchronize().expect("sync");
                    total += ctx.elapsed(&start, &end).expect("cuEventElapsedTime");
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
                    let _: Vec<f64> = black_box(src_3ch.mean().expect("mean"));
                    let end = ctx.record_event();
                    end.record();
                    ctx.synchronize().expect("sync");
                    total += ctx.elapsed(&start, &end).expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    // --- ConvertTo (u8→f32, 3ch, same-size) ---
    {
        // Warm-up
        src_3ch.convert(&mut dst_convert).expect("convert warm-up");

        group.bench_function("Convert_u8_to_f32", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src_3ch.convert(&mut dst_convert));
                    let end = ctx.record_event();
                    end.record();
                    ctx.synchronize().expect("sync");
                    total += ctx.elapsed(&start, &end).expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    // --- Normalize (u8→f32, 3ch, same-size) ---
    {
        // Warm-up
        src_3ch.normalize(&mut dst_norm).expect("normalize warm-up");

        group.bench_function("Normalize_u8_to_f32", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = ctx.record_event();
                    start.record();
                    let _ = black_box(src_3ch.normalize(&mut dst_norm));
                    let end = ctx.record_event();
                    end.record();
                    ctx.synchronize().expect("sync");
                    total += ctx.elapsed(&start, &end).expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_op_family);
criterion_main!(benches);
