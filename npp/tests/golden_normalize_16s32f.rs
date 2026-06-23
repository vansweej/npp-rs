//! Golden-image correctness test for `Normalize` on `CudaImage<i16>` → `CudaImage<f32>`.
//!
//! # To update the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_normalize_16s32f`
//! 2. The test will print the captured output and panic ("golden reference not yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below.
//! 4. Re-run to confirm the assertion passes.
//!
//! # Numerical note
//!
//! Normalize maps the maximum positive value (32767) to exactly `1.0`.
//! Negative inputs map below `0.0` (e.g. `-32768 → ~-1.000031`). This test uses
//! non-negative inputs for pinning simplicity.
//!
//! The normalized output is `(x as f32) * (1.0 / 32767.0)` (multiply-by-reciprocal,
//! not `(x as f32) / 32767.0`). These can differ in the last ULP. Always pin
//! `EXPECTED` from an actual GPU run — do not fabricate by hand.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::Normalize;
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 12;
const H: u32 = 8;

fn make_input() -> Vec<i16> {
    let mut data = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            // Non-negative values only: 0, 16384, 32767
            // 16384 / 32767 ≈ 0.5000305
            let val: i16 = match (x + y) % 3 {
                0 => 0,
                1 => 16384,
                _ => 32767,
            };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Golden output for i16→f32 normalization (C3).
/// Pinned on <GPU model> on <date>.
const EXPECTED: &[f32] = &[]; // Pin on GPU host

#[test]
fn test_golden_normalize_16s32f_c3() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let src = CudaImage::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");
    let mut dst = CudaImage::<f32>::new(ctx.clone(), 3, W, H).expect("dst allocation");
    src.normalize(&mut dst).expect("normalize");
    let output: Vec<f32> = Vec::try_from(&dst).expect("read-back");
    assert_golden(&output, EXPECTED, "normalize_16s32f_c3");
}
