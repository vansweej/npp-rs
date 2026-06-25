//! Shape-check test for `ConvertRoundedScaled`.
//!
//! Verifies that `convert_rounded_scaled` propagates geometry correctly for a
//! single-channel `u8 → i8` conversion with `RoundMode::Nearest, scale_factor = 0`.
//! This is a geometry-only smoke test — pixel correctness is verified by
//! `golden_convert_round_scaled_pinned.rs`.
//!
//! # To run
//!
//! ```bash
//! nix develop . --command cargo test --features gpu --test golden_convert_round_scaled
//! ```

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{ConvertRoundedScaled, RoundMode};
use npp_rs::stream::stream_context_for;

const SRC_W: u32 = 8;
const SRC_H: u32 = 4;

#[test]
fn test_convert_rounded_scaled_u8_to_i8_shape() {
    let ctx = stream_context_for(0).expect("CUDA device init");
    let ch = 1u8;

    let src_data: Vec<u8> = (0..SRC_W * SRC_H).map(|i| (i * 37) as u8).collect();
    let src = CudaImage::<u8>::from_host(ctx.clone(), ch, SRC_W, SRC_H, &src_data)
        .expect("src allocation");
    let mut dst = CudaImage::<i8>::new(ctx.clone(), ch, SRC_W, SRC_H).expect("dst allocation");

    src.convert_rounded_scaled(&mut dst, RoundMode::Nearest, 0)
        .expect("convert_rounded_scaled u8→i8");

    assert_eq!(dst.width(), SRC_W);
    assert_eq!(dst.height(), SRC_H);
    assert_eq!(dst.channels(), ch);
}
