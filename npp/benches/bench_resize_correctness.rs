//! Correctness-asserting benchmark for Resize.
//!
//! Every iteration asserts that the device output matches a pinned golden.
//! If a CUDA or NPP version change shifts output, this panics immediately —
//! the benchmark is a correctness gate first, a timing measurement second.

#![cfg(feature = "gpu")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const SRC_W: u32 = 32;
const SRC_H: u32 = 24;
const DST_W: u32 = 16;
const DST_H: u32 = 12;

fn make_input() -> Vec<u8> {
    let mut data = Vec::with_capacity((SRC_W * SRC_H * 3) as usize);
    for y in 0..SRC_H {
        for x in 0..SRC_W {
            data.push((x * 7) as u8);
            data.push((y * 11) as u8);
            data.push(128);
        }
    }
    data
}

// Generated on NVIDIA GPU, Linear interpolation (bit-exact for this input).
const EXPECTED: &[u8] = &[
    0, 0, 128, 14, 0, 128, 28, 0, 128, 42, 0, 128, 56, 0, 128, 70, 0, 128, 84, 0, 128, 98, 0, 128,
    112, 0, 128, 126, 0, 128, 140, 0, 128, 154, 0, 128, 168, 0, 128, 182, 0, 128, 196, 0, 128, 210,
    0, 128, 0, 22, 128, 14, 22, 128, 28, 22, 128, 42, 22, 128, 56, 22, 128, 70, 22, 128, 84, 22,
    128, 98, 22, 128, 112, 22, 128, 126, 22, 128, 140, 22, 128, 154, 22, 128, 168, 22, 128, 182,
    22, 128, 196, 22, 128, 210, 22, 128, 0, 44, 128, 14, 44, 128, 28, 44, 128, 42, 44, 128, 56, 44,
    128, 70, 44, 128, 84, 44, 128, 98, 44, 128, 112, 44, 128, 126, 44, 128, 140, 44, 128, 154, 44,
    128, 168, 44, 128, 182, 44, 128, 196, 44, 128, 210, 44, 128, 0, 66, 128, 14, 66, 128, 28, 66,
    128, 42, 66, 128, 56, 66, 128, 70, 66, 128, 84, 66, 128, 98, 66, 128, 112, 66, 128, 126, 66,
    128, 140, 66, 128, 154, 66, 128, 168, 66, 128, 182, 66, 128, 196, 66, 128, 210, 66, 128, 0, 88,
    128, 14, 88, 128, 28, 88, 128, 42, 88, 128, 56, 88, 128, 70, 88, 128, 84, 88, 128, 98, 88, 128,
    112, 88, 128, 126, 88, 128, 140, 88, 128, 154, 88, 128, 168, 88, 128, 182, 88, 128, 196, 88,
    128, 210, 88, 128, 0, 110, 128, 14, 110, 128, 28, 110, 128, 42, 110, 128, 56, 110, 128, 70,
    110, 128, 84, 110, 128, 98, 110, 128, 112, 110, 128, 126, 110, 128, 140, 110, 128, 154, 110,
    128, 168, 110, 128, 182, 110, 128, 196, 110, 128, 210, 110, 128, 0, 132, 128, 14, 132, 128, 28,
    132, 128, 42, 132, 128, 56, 132, 128, 70, 132, 128, 84, 132, 128, 98, 132, 128, 112, 132, 128,
    126, 132, 128, 140, 132, 128, 154, 132, 128, 168, 132, 128, 182, 132, 128, 196, 132, 128, 210,
    132, 128, 0, 154, 128, 14, 154, 128, 28, 154, 128, 42, 154, 128, 56, 154, 128, 70, 154, 128,
    84, 154, 128, 98, 154, 128, 112, 154, 128, 126, 154, 128, 140, 154, 128, 154, 154, 128, 168,
    154, 128, 182, 154, 128, 196, 154, 128, 210, 154, 128, 0, 176, 128, 14, 176, 128, 28, 176, 128,
    42, 176, 128, 56, 176, 128, 70, 176, 128, 84, 176, 128, 98, 176, 128, 112, 176, 128, 126, 176,
    128, 140, 176, 128, 154, 176, 128, 168, 176, 128, 182, 176, 128, 196, 176, 128, 210, 176, 128,
    0, 198, 128, 14, 198, 128, 28, 198, 128, 42, 198, 128, 56, 198, 128, 70, 198, 128, 84, 198,
    128, 98, 198, 128, 112, 198, 128, 126, 198, 128, 140, 198, 128, 154, 198, 128, 168, 198, 128,
    182, 198, 128, 196, 198, 128, 210, 198, 128, 0, 220, 128, 14, 220, 128, 28, 220, 128, 42, 220,
    128, 56, 220, 128, 70, 220, 128, 84, 220, 128, 98, 220, 128, 112, 220, 128, 126, 220, 128, 140,
    220, 128, 154, 220, 128, 168, 220, 128, 182, 220, 128, 196, 220, 128, 210, 220, 128, 0, 242,
    128, 14, 242, 128, 28, 242, 128, 42, 242, 128, 56, 242, 128, 70, 242, 128, 84, 242, 128, 98,
    242, 128, 112, 242, 128, 126, 242, 128, 140, 242, 128, 154, 242, 128, 168, 242, 128, 182, 242,
    128, 196, 242, 128, 210, 242, 128,
];

fn bench_resize_correctness(c: &mut Criterion) {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let src =
        CudaImage::from_host(ctx.clone(), 3, SRC_W, SRC_H, &make_input()).expect("src allocation");
    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, DST_W, DST_H).expect("dst allocation");

    c.bench_with_input(
        BenchmarkId::new(
            "resize_correctness",
            format!("{SRC_W}x{SRC_H}->{DST_W}x{DST_H}"),
        ),
        &(),
        |b, _| {
            b.iter(|| {
                src.resize(&mut dst, ResizeInterpolation::Linear)
                    .expect("resize");
                let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");
                assert_golden(&output, EXPECTED, "bench resize correctness");
            })
        },
    );
}

criterion_group!(benches, bench_resize_correctness);
criterion_main!(benches);
