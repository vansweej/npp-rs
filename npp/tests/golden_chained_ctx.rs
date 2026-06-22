//! # C1 chained-op smoke test
//!
//! This test verifies that two `_Ctx` operations (SwapChannels → Resize)
//! chained on the same [`StreamContext`] produce correct pixels through a
//! shared execution context, with a single host-fenced readback at the end.
//!
//! ## What it tests
//!
//! - Intra-stream op chaining: Op2 reads Op1's device output without
//!   intervening host sync.
//! - The [`StreamContext::synchronize()`] fence before readback.
//! - Correct pixel output through the pipeline.
//!
//! ## What it does NOT test
//!
//! - Fence-absent behaviour (not discriminable under cudarc 0.9's
//!   context-wide synchronize — see `docs/stream-context.md`).
//! - Cross-stream re-entry.
//! - The device-fence (`wait_for`) path.
//!
//! # Manual procedure to pin the golden reference
//!
//! 1. Run on a GPU host:
//!    `nix develop . --command cargo test --features gpu --test golden_chained_ctx`
//! 2. The test will print the captured output and panic ("golden reference not
//!    yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below.
//! 4. Re-run to confirm.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation, SwapChannels};
use npp_rs::stream_context_for;
use npp_rs::test_helpers::assert_golden;
use std::convert::TryFrom;

#[test]
fn chained_bgr_to_rgb_then_resize_produces_correct_pixels() {
    // ── Setup ──
    let ctx = stream_context_for(0).expect("GPU at ordinal 0");
    let (w, h) = (4u32, 4u32);

    // Known BGRA input: 4x4 checkerboard-like pattern
    // Each pixel: B, G, R, A (4 bytes)
    let bgra_data: Vec<u8> = vec![
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 0,
        255, 255, 128, 128, 128, 255, 64, 64, 64, 255, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255,
        255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 0, 255, 255, 128, 128, 128, 255, 64, 64, 64,
        255,
    ];

    let src = CudaImage::<u8>::from_host(ctx.clone(), 4, w, h, &bgra_data).expect("upload BGRA");

    // ── Op 1: BGRA→RGB (4ch → 3ch) ──
    let mut rgb = CudaImage::<u8>::new(ctx.clone(), 3, w, h).expect("allocate RGB");
    src.bgra_to_rgb(&mut rgb).expect("BGRA→RGB");

    // ── Op 2: Resize (downsample 4x4 → 2x2, NearestNeighbor) ──
    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, w / 2, h / 2).expect("allocate dst");
    rgb.resize(&mut dst, ResizeInterpolation::NearestNeighbor)
        .expect("resize");

    // ── Single readback (fence + NULL-stream copy) ──
    let result: Vec<u8> = Vec::try_from(&dst).expect("readback");

    // ── Golden ──
    // EXPECTED must be pinned by running once and copying the printed output.
    // 2x2 RGB = 4 pixels × 3 channels = 12 bytes.
    const EXPECTED: &[u8] = &[0, 0, 255, 255, 0, 0, 0, 0, 255, 255, 0, 0];
    assert_golden(&result, EXPECTED, "chained_bgr_to_rgb_then_resize");
}
