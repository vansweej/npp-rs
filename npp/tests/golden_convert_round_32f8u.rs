//! Golden-image correctness test for `ConvertRounded` on `CudaImage<f32>` → `CudaImage<u8>`.
//!
//! Chosen fractional values where all three rounding modes produce **different** u8 outputs:
//! 0.5 (Near→0, Fin→1, Zero→0), 2.5 (Near→2, Fin→3, Zero→2),
//! 1.9 (Near/Fin→2, Zero→1), 5.0 (all→5, mode-invariant anchor),
//! 0.4 (all→0), 3.5 (Near→4, Fin→4, Zero→3).
//!
//! Three tests share one `make_input` — one per `RoundMode`. Each has its own
//! EXPECTED, pinned from a real GPU run. Identical pins across modes would
//! indicate the `mode` parameter is not plumbed.
//!
//! # To update the golden reference
//!
//! Run the test with `EXPECTED` temporarily set to `&[]` — the `assert_golden`
//! helper will panic and print the actual GPU output. Copy that output into
//! `EXPECTED` for the relevant mode.
//!
//! ```bash
//! nix develop . --command cargo test --features gpu --test golden_convert_round_32f8u
//! ```

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{ConvertRounded, RoundMode};
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 6;
const H: u32 = 1;

/// Input: 1×6 row of 3-channel f32 values chosen so Nearest/Financial/Zero diverge.
///
/// Channel layout per pixel: [R, G, B]. All three channels use the same values
/// to make the pattern easy to reason about.
fn make_input() -> Vec<f32> {
    vec![
        0.5, 2.5, 1.9, 5.0, 0.4, 3.5, 0.5, 2.5, 1.9, 5.0, 0.4, 3.5, 0.5, 2.5, 1.9, 5.0, 0.4, 3.5,
    ]
}

/// Golden output for `RoundMode::Nearest`, pinned from GPU.
///
/// Pinned: (not yet pinned — run on GPU host and fill in).
const EXPECTED_NEAR: &[u8] = &[];

/// Golden output for `RoundMode::Financial`, pinned from GPU.
///
/// Pinned: (not yet pinned — run on GPU host and fill in).
const EXPECTED_FIN: &[u8] = &[];

/// Golden output for `RoundMode::Zero`, pinned from GPU.
///
/// Pinned: (not yet pinned — run on GPU host and fill in).
const EXPECTED_ZERO: &[u8] = &[];

#[test]
fn test_golden_convert_round_32f8u_nearest() {
    let ctx = stream_context_for(0).expect("CUDA device init");

    let src =
        CudaImage::<f32>::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");

    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, W, H).expect("dst allocation");

    src.convert_rounded(&mut dst, RoundMode::Nearest)
        .expect("convert_rounded Nearest");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED_NEAR, "convert_round_32f8u_nearest");
}

#[test]
fn test_golden_convert_round_32f8u_financial() {
    let ctx = stream_context_for(0).expect("CUDA device init");

    let src =
        CudaImage::<f32>::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");

    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, W, H).expect("dst allocation");

    src.convert_rounded(&mut dst, RoundMode::Financial)
        .expect("convert_rounded Financial");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED_FIN, "convert_round_32f8u_financial");
}

#[test]
fn test_golden_convert_round_32f8u_zero() {
    let ctx = stream_context_for(0).expect("CUDA device init");

    let src =
        CudaImage::<f32>::from_host(ctx.clone(), 3, W, H, &make_input()).expect("src allocation");

    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, W, H).expect("dst allocation");

    src.convert_rounded(&mut dst, RoundMode::Zero)
        .expect("convert_rounded Zero");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    assert_golden(&output, EXPECTED_ZERO, "convert_round_32f8u_zero");
}
