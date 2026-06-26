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

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::stream::StreamContext;
use npp_rs::stream_context_for;

const BENCH_SIZE: u32 = 512;

/// All interpolation modes supported by NPP for u8 C3.
const INTERP_MODES: &[ResizeInterpolation] = &[
    ResizeInterpolation::NearestNeighbor,
    ResizeInterpolation::Linear,
    ResizeInterpolation::Cubic,
    ResizeInterpolation::Super,
];

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

/// Benchmark: interpolation mode sweep at 512×512, 3-channel.
#[cfg(not(tarpaulin_include))]
fn bench_resize_modes(c: &mut Criterion) {
    let ctx = stream_context_for(0).expect("CUDA device 0 init");
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
                    ctx.synchronize().expect("sync");
                    total += ctx.elapsed(&start, &end).expect("cuEventElapsedTime");
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
    let ctx = stream_context_for(0).expect("CUDA device 0 init");

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
                    ctx.synchronize().expect("sync");
                    total += ctx.elapsed(&start, &end).expect("cuEventElapsedTime");
                }
                total
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_resize_modes, bench_resize_channels);
criterion_main!(benches);
