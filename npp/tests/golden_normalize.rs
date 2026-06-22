//! Golden-image correctness test for `Normalize` on `CudaImage<u8>` → `CudaImage<f32>`.
//!
//! Proves that the hand-written `normalize` produces correct pixels,
//! not just correct geometry (C12 minimum mitigation).
//!
//! # Manual procedure to pin the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_normalize`
//! 2. The test will print the captured output and panic ("golden reference not yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below.
//! 4. Re-run to confirm the assertion passes.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::Normalize;
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 12;
const H: u32 = 8;

/// Input: procedurally generated 3-channel RGB with test values (12x8).
///
/// Channel layout per pixel: [R, G, B]
/// Uses values 0, 128, 255 to produce exact normalized values:
/// - 0 → 0.0
/// - 128 → 128/255 ≈ 0.50196078...
/// - 255 → 1.0
fn make_input() -> Vec<u8> {
    let mut data = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            // Cycle through test values: 0, 128, 255
            let val = match (x + y) % 3 {
                0 => 0,
                1 => 128,
                _ => 255,
            };
            data.push(val); // R
            data.push(val); // G
            data.push(val); // B
        }
    }
    data
}

/// Golden output for u8→f32 normalization (C3).
/// Generated on NVIDIA GPU. FP-exact (deterministic normalization).
const EXPECTED: &[f32] = &[
    0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608,
    0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.5019608, 0.5019608,
    0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608,
    0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608,
    0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608,
    0.5019608, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608,
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.5019608,
    0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608,
    0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608,
    0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608,
    0.5019608, 0.5019608, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608,
    0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0,
    0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608,
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.5019608, 0.5019608, 0.5019608, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
];

#[test]
fn test_golden_normalize_8u32f_c3() {
    let ctx = stream_context_for(0).expect("CUDA device init");

    // 3-channel source (u8)
    let src = CudaImage::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");

    // 3-channel destination (f32)
    let mut dst = CudaImage::<f32>::new(ctx.clone(), 3, W, H).expect("dst allocation");

    src.normalize(&mut dst).expect("normalize");

    let output: Vec<f32> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED, "normalize_8u32f_c3");
}
