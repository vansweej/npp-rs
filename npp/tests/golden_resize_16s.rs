//! Golden-image correctness test for `Resize` on `CudaImage<i16>`.
//!
//! See `golden_resize.rs` for the pinned-reference procedure.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const SRC_W: u32 = 12;
const SRC_H: u32 = 8;
const DST_W: u32 = 6;
const DST_H: u32 = 4;

/// Input: procedurally generated 3-channel i16 gradient (12x8).
fn make_input() -> Vec<i16> {
    let mut data = Vec::with_capacity((SRC_W * SRC_H * 3) as usize);
    for y in 0..SRC_H {
        for x in 0..SRC_W {
            data.push((x as i16) * 21); // X-gradient
            data.push((y as i16) * 32); // Y-gradient
            data.push(128); // Constant
        }
    }
    data
}

/// Golden output for NearestNeighbor 12x8 → 6x4.
/// Generated on NVIDIA GPU, NearestNeighbor interpolation (bit-exact).
const EXPECTED: &[i16] = &[
    0, 0, 128, 42, 0, 128, 84, 0, 128, 126, 0, 128, 168, 0, 128, 210, 0, 128, 0, 64, 128, 42, 64,
    128, 84, 64, 128, 126, 64, 128, 168, 64, 128, 210, 64, 128, 0, 128, 128, 42, 128, 128, 84, 128,
    128, 126, 128, 128, 168, 128, 128, 210, 128, 128, 0, 192, 128, 42, 192, 128, 84, 192, 128, 126,
    192, 128, 168, 192, 128, 210, 192, 128,
];

#[test]
fn test_golden_resize_16s_nn() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let src =
        CudaImage::from_host(ctx.clone(), 3, SRC_W, SRC_H, &make_input()).expect("src allocation");
    let mut dst = CudaImage::<i16>::new(ctx.clone(), 3, DST_W, DST_H).expect("dst allocation");
    src.resize(&mut dst, ResizeInterpolation::NearestNeighbor)
        .expect("resize");
    let output: Vec<i16> = Vec::try_from(&dst).expect("read-back");
    assert_golden(&output, EXPECTED, "NearestNeighbor resize i16 C3");
}
