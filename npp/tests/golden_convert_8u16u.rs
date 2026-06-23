//! Golden-image correctness test for `ConvertTo` on `CudaImage<u8>` → `CudaImage<u16>`.
//!
//! Proves that the generated `convert` for 8u→16u (integer widening) produces correct pixels,
//! not just correct geometry (C12 minimum mitigation). Exercises the dual-type dst_step_bytes
//! path (size_of::<u16>() differs from size_of::<u8>()) with a non-float destination.
//!
//! # To update the golden reference
//!
//! If the conversion logic changes, run the test with `EXPECTED` temporarily
//! set to `&[]` — the `assert_golden` helper will panic and print the actual
//! GPU output. Copy that output into `EXPECTED` above.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::ConvertTo;
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 12;
const H: u32 = 8;

/// Input: procedurally generated 3-channel gradient (12x8).
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

/// Golden output for u8→u16 conversion (C3).
///
/// 12×8 × 3‑channel gradient: R = x*21, G = y*32, B = 128.
/// Pinned 2025-06-23 on A100.
///
/// Layout: row‑major, pixel‑interleaved [R, G, B] x 12 per row.
const EXPECTED: &[u16] = &[
    // Row 0 (y=0, G=0)
    0, 0, 128, 21, 0, 128, 42, 0, 128, 63, 0, 128, 84, 0, 128, 105, 0, 128, 126, 0, 128, 147, 0,
    128, 168, 0, 128, 189, 0, 128, 210, 0, 128, 231, 0, 128, // Row 1 (y=1, G=32)
    0, 32, 128, 21, 32, 128, 42, 32, 128, 63, 32, 128, 84, 32, 128, 105, 32, 128, 126, 32, 128, 147,
    32, 128, 168, 32, 128, 189, 32, 128, 210, 32, 128, 231, 32, 128, // Row 2 (y=2, G=64)
    0, 64, 128, 21, 64, 128, 42, 64, 128, 63, 64, 128, 84, 64, 128, 105, 64, 128, 126, 64, 128, 147,
    64, 128, 168, 64, 128, 189, 64, 128, 210, 64, 128, 231, 64, 128, // Row 3 (y=3, G=96)
    0, 96, 128, 21, 96, 128, 42, 96, 128, 63, 96, 128, 84, 96, 128, 105, 96, 128, 126, 96, 128, 147,
    96, 128, 168, 96, 128, 189, 96, 128, 210, 96, 128, 231, 96, 128, // Row 4 (y=4, G=128)
    0, 128, 128, 21, 128, 128, 42, 128, 128, 63, 128, 128, 84, 128, 128, 105, 128, 128, 126, 128,
    128, 147, 128, 128, 168, 128, 128, 189, 128, 128, 210, 128, 128, 231, 128, 128,
    // Row 5 (y=5, G=160)
    0, 160, 128, 21, 160, 128, 42, 160, 128, 63, 160, 128, 84, 160, 128, 105, 160, 128, 126, 160,
    128, 147, 160, 128, 168, 160, 128, 189, 160, 128, 210, 160, 128, 231, 160, 128,
    // Row 6 (y=6, G=192)
    0, 192, 128, 21, 192, 128, 42, 192, 128, 63, 192, 128, 84, 192, 128, 105, 192, 128, 126, 192,
    128, 147, 192, 128, 168, 192, 128, 189, 192, 128, 210, 192, 128, 231, 192, 128,
    // Row 7 (y=7, G=224)
    0, 224, 128, 21, 224, 128, 42, 224, 128, 63, 224, 128, 84, 224, 128, 105, 224, 128, 126, 224,
    128, 147, 224, 128, 168, 224, 128, 189, 224, 128, 210, 224, 128, 231, 224, 128,
];

#[test]
fn test_golden_convert_8u16u_c3() {
    let ctx = stream_context_for(0).expect("CUDA device init");

    // 3-channel source (u8)
    let src = CudaImage::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");

    // 3-channel destination (u16)
    let mut dst = CudaImage::<u16>::new(ctx.clone(), 3, W, H).expect("dst allocation");

    src.convert(&mut dst).expect("convert");

    let output: Vec<u16> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED, "convert_8u16u_c3");
}
