//! Golden-image correctness test for `Resize` on `CudaImage<f32>`.
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

/// Input: procedurally generated 3-channel f32 gradient (12x8).
fn make_input() -> Vec<f32> {
    let mut data = Vec::with_capacity((SRC_W * SRC_H * 3) as usize);
    for y in 0..SRC_H {
        for x in 0..SRC_W {
            data.push((x as f32) * 21.0); // X-gradient
            data.push((y as f32) * 32.0); // Y-gradient
            data.push(128.0); // Constant
        }
    }
    data
}

/// Golden output for NearestNeighbor 12x8 → 6x4.
/// GPU-captured bytes — empty until pinned.
const EXPECTED: &[f32] = &[];

#[test]
fn test_golden_resize_32f_nn() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let src =
        CudaImage::from_host(ctx.clone(), 3, SRC_W, SRC_H, &make_input()).expect("src allocation");
    let mut dst = CudaImage::<f32>::new(ctx.clone(), 3, DST_W, DST_H).expect("dst allocation");
    src.resize(&mut dst, ResizeInterpolation::NearestNeighbor)
        .expect("resize");
    let output: Vec<f32> = Vec::try_from(&dst).expect("read-back");
    assert_golden(&output, EXPECTED, "NearestNeighbor resize f32 C3");
}
