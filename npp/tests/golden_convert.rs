//! Golden-image correctness test for `ConvertTo` on `CudaImage<u8>` → `CudaImage<f32>`.
//!
//! Proves that the hand-written `convert` produces correct pixels,
//! not just correct geometry (C12 minimum mitigation).
//!
//! # Manual procedure to pin the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_convert`
//! 2. The test will print the captured output and panic ("golden reference not yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below.
//! 4. Re-run to confirm the assertion passes.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::ConvertTo;
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 12;
const H: u32 = 8;

/// Input: procedurally generated 3-channel RGB gradient (12x8).
///
/// Channel layout per pixel: [R, G, B]
fn make_input() -> Vec<u8> {
    let mut data = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            data.push((x * 21) as u8); // R: x-gradient
            data.push((y * 32) as u8); // G: y-gradient
            data.push(128); // B: constant
        }
    }
    data
}

/// Golden output for u8→f32 conversion (C3).
/// Generated on NVIDIA GPU. FP-exact (deterministic conversion).
///
/// NOTE: Set this to `&[]` initially, run on GPU, copy the printed bytes here.
const EXPECTED: &[f32] = &[];

#[test]
fn test_golden_convert_8u32f_c3() {
    let ctx = stream_context_for(0).expect("CUDA device init");

    // 3-channel source (u8)
    let src = CudaImage::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");

    // 3-channel destination (f32)
    let mut dst = CudaImage::<f32>::new(ctx.clone(), 3, W, H).expect("dst allocation");

    src.convert(&mut dst).expect("convert");

    let output: Vec<f32> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED, "convert_8u32f_c3");
}
