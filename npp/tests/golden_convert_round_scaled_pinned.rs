//! Golden-image correctness test for `ConvertRoundedScaled` on `CudaImage<f32>` → `CudaImage<u8>`.
//!
//! Single-channel, `scale_factor = 0` (identity scaling, `2^0 = 1`).
//! Chosen fractional values where all three rounding modes produce **different** u8 outputs:
//! 0.5 (Near→0, Fin→1, Zero→0), 2.5 (Near→2, Fin→3, Zero→2),
//! 1.9 (Near/Fin→2, Zero→1), 5.0 (all→5, mode-invariant anchor),
//! 0.4 (all→0), 3.5 (Near→4, Fin→4, Zero→3).
//!
//! Three tests share one `make_input` — one per `RoundMode`. Each has its own
//! EXPECTED, pinned from a real GPU run.
//!
//! # To update the golden reference
//!
//! Run the test with `EXPECTED` temporarily set to `&[]` — the `assert_golden`
//! helper will panic and print the actual GPU output. Copy that output into
//! `EXPECTED` for the relevant mode.
//!
//! ```bash
//! nix develop . --command cargo test --features gpu --test golden_convert_round_scaled_pinned
//! ```

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{ConvertRoundedScaled, RoundMode};
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 6;
const H: u32 = 1;

/// Input: 1×6 row of single-channel f32 values chosen so Nearest/Financial/Zero diverge.
fn make_input() -> Vec<f32> {
    vec![0.5, 2.5, 1.9, 5.0, 0.4, 3.5]
}

/// Golden output for `RoundMode::Nearest`, pinned from GPU.
///
/// Pinned: 2026-06-25 — 0.5→0 (tie→even), 2.5→2 (tie→even), 1.9→2 (std),
///         5.0→5 (exact), 0.4→0 (trunc), 3.5→4 (tie→even).
const EXPECTED_NEAR: &[u8] = &[0, 2, 2, 5, 0, 4];

/// Golden output for `RoundMode::Financial`, pinned from GPU.
///
/// Pinned: 2026-06-25 — 0.5→1 (half-up), 2.5→3 (half-up), 1.9→2 (std),
///         5.0→5 (exact), 0.4→0 (trunc), 3.5→4 (half-up→4 not 3).
const EXPECTED_FIN: &[u8] = &[1, 3, 2, 5, 0, 4];

/// Golden output for `RoundMode::Zero`, pinned from GPU.
///
/// Pinned: 2026-06-25 — 0.5→0 (trunc), 2.5→2 (trunc), 1.9→1 (trunc),
///         5.0→5 (exact), 0.4→0 (trunc), 3.5→3 (trunc).
const EXPECTED_ZERO: &[u8] = &[0, 2, 1, 5, 0, 3];

const SCALE: i32 = 0;

#[test]
fn test_golden_convert_round_scaled_f32_to_u8_nearest() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let ch = 1u8;

    let src =
        CudaImage::<f32>::from_host(ctx.clone(), ch, W, H, &make_input()).expect("src allocation");

    let mut dst = CudaImage::<u8>::new(ctx.clone(), ch, W, H).expect("dst allocation");

    src.convert_rounded_scaled(&mut dst, RoundMode::Nearest, SCALE)
        .expect("convert_rounded_scaled Nearest");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    assert_golden(
        &output,
        EXPECTED_NEAR,
        "convert_round_scaled_f32_to_u8_nearest",
    );
}

#[test]
fn test_golden_convert_round_scaled_f32_to_u8_financial() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let ch = 1u8;

    let src =
        CudaImage::<f32>::from_host(ctx.clone(), ch, W, H, &make_input()).expect("src allocation");

    let mut dst = CudaImage::<u8>::new(ctx.clone(), ch, W, H).expect("dst allocation");

    src.convert_rounded_scaled(&mut dst, RoundMode::Financial, SCALE)
        .expect("convert_rounded_scaled Financial");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    assert_golden(
        &output,
        EXPECTED_FIN,
        "convert_round_scaled_f32_to_u8_financial",
    );
}

#[test]
fn test_golden_convert_round_scaled_f32_to_u8_zero() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let ch = 1u8;

    let src =
        CudaImage::<f32>::from_host(ctx.clone(), ch, W, H, &make_input()).expect("src allocation");

    let mut dst = CudaImage::<u8>::new(ctx.clone(), ch, W, H).expect("dst allocation");

    src.convert_rounded_scaled(&mut dst, RoundMode::Zero, SCALE)
        .expect("convert_rounded_scaled Zero");

    let output: Vec<u8> = Vec::try_from(&dst).expect("read-back");

    assert_golden(
        &output,
        EXPECTED_ZERO,
        "convert_round_scaled_f32_to_u8_zero",
    );
}
