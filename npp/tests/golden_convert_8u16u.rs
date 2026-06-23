//! Golden-image correctness test for `ConvertTo` on `CudaImage<u8>` → `CudaImage<u16>`.
//!
//! Proves that the generated `convert` for 8u→16u (integer widening) produces correct pixels,
//! not just correct geometry (C12 minimum mitigation). Exercises the dual-type dst_step_bytes
//! path (size_of::<u16>() differs from size_of::<u8>()) with a non-float destination.
//!
//! # Manual procedure to pin the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_convert_8u16u`
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
/// Not yet pinned — run on a GPU host to generate.
const EXPECTED: &[u16] = &[
    // PLACEHOLDER — golden reference not yet pinned.
    // Replace with actual GPU-generated values.
    0,
];

#[test]
fn test_golden_convert_8u16u_c3() {
    let ctx = stream_context_for(0).expect("CUDA device init");

    // 3-channel source (u8)
    let src =
        CudaImage::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");

    // 3-channel destination (u16)
    let mut dst = CudaImage::<u16>::new(ctx.clone(), 3, W, H).expect("dst allocation");

    src.convert(&mut dst).expect("convert");

    let output: Vec<u16> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED, "convert_8u16u_c3");
}
