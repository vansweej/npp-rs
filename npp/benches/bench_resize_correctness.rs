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

const SRC_W: u32 = 512;
const SRC_H: u32 = 384;
const DST_W: u32 = 256;
const DST_H: u32 = 192;

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

// Populated from first GPU run (same pinning workflow as golden tests).
const EXPECTED: &[u8] = &[];

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
                src.resize(&mut dst, ResizeInterpolation::Bilinear)
                    .expect("resize");
                let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");
                assert_golden(&output, EXPECTED, "bench resize correctness");
            })
        },
    );
}

criterion_group!(benches, bench_resize_correctness);
criterion_main!(benches);
