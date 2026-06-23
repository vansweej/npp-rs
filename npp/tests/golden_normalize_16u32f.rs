//! Golden-image correctness test for `Normalize` on `CudaImage<u16>` → `CudaImage<f32>`.
//!
//! # To update the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_normalize_16u32f`
//! 2. The test will print the captured output and panic ("golden reference not yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below.
//! 4. Re-run to confirm the assertion passes.
//!
//! # Numerical note
//!
//! The normalized output is `(x as f32) * (1.0 / 65535.0)` (multiply-by-reciprocal,
//! not `(x as f32) / 65535.0`). These can differ in the last ULP. Always pin
//! `EXPECTED` from an actual GPU run — do not fabricate by hand.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::Normalize;
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 12;
const H: u32 = 8;

fn make_input() -> Vec<u16> {
    let mut data = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            // Values 0, 32768, 65535 produce exact normalized outputs 0.0, ~0.5, 1.0
            let val: u16 = match (x + y) % 3 {
                0 => 0,
                1 => 32768,
                _ => 65535,
            };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Golden output for u16→f32 normalization (C3).
/// Pinned on NVIDIA GPU on 2026-06-24.
const EXPECTED: &[f32] = &[
    0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076,
    0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.5000076, 0.5000076,
    0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076,
    0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076,
    0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076,
    0.5000076, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076,
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.5000076,
    0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076,
    0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076,
    0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076,
    0.5000076, 0.5000076, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076,
    0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0,
    0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076,
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.5000076, 0.5000076, 0.5000076, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
];

#[test]
fn test_golden_normalize_16u32f_c3() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let src = CudaImage::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");
    let mut dst = CudaImage::<f32>::new(ctx.clone(), 3, W, H).expect("dst allocation");
    src.normalize(&mut dst).expect("normalize");
    let output: Vec<f32> = Vec::try_from(&dst).expect("read-back");
    assert_golden(&output, EXPECTED, "normalize_16u32f_c3");
}
