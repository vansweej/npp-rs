//! `_Ctx` chained-operation smoke test for `ConvertRounded`.
//!
//! Chains `Resize` (`_Ctx`) → `convert_rounded` (`_Ctx`) on a single
//! [`StreamContext`] with a single host-fenced readback at the end. This is a
//! **`_Ctx`-plumbing regression guard** only — it verifies the stream-context
//! flows through both ops without mismatch. Round-mode semantics are tested
//! separately by `golden_convert_round_32f8u.rs`.
//!
//! The intermediate (resize output) `f32` values depend on interpolation and
//! are **not** the clean `0.5/2.5/…` pattern from the standalone golden, so
//! the chained `EXPECTED` is **not** hand-reasoned — it is purely pinned from
//! GPU output.
//!
//! # To update the golden reference
//!
//! Run on a GPU host:
//! ```bash
//! nix develop . --command cargo test --features gpu --test golden_convert_round_chained
//! ```
//! Then copy the printed output into `EXPECTED` below.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{ConvertRounded, Resize, ResizeInterpolation, RoundMode};
use npp_rs::stream::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

const W: u32 = 8;
const H: u32 = 6;

/// Input: 8×6 f32 gradient (1-channel, grayscale-like).
fn make_input() -> Vec<f32> {
    let mut data = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            data.push((x as f32 + y as f32) * 0.3);
        }
    }
    data
}

const EXPECTED: &[u8] = &[];

#[test]
fn chained_resize_then_convert_rounded_produces_correct_pixels() {
    // ── Setup ──
    let ctx = stream_context_for(0).expect("GPU at ordinal 0");
    let ch = 1u8;

    let src = CudaImage::<f32>::from_host(ctx.clone(), ch, W, H, &make_input())
        .expect("upload f32 gradient");

    // ── Op 1: Resize f32 (downsample 8×6 → 4×3, Linear) ──
    let mut resized =
        CudaImage::<f32>::new(ctx.clone(), ch, W / 2, H / 2).expect("allocate resized");
    src.resize(&mut resized, ResizeInterpolation::Linear)
        .expect("resize f32");

    // ── Op 2: ConvertRounded f32→u8 (Nearest) ──
    let mut dst = CudaImage::<u8>::new(ctx.clone(), ch, W / 2, H / 2).expect("allocate dst");
    resized
        .convert_rounded(&mut dst, RoundMode::Nearest)
        .expect("convert_rounded f32→u8");

    // ── Single readback ──
    let result: Vec<u8> = Vec::try_from(&dst).expect("readback");

    // ── Golden ──
    assert_golden(&result, EXPECTED, "chained_resize_then_convert_rounded");
}
