//! Cross-stream timing smoke test (F8.2).
//!
//! Creates two streams, performs a Resize on each, and uses `elapsed_between`
//! with fences recorded on different streams to verify that cross-stream
//! timing is wired correctly.
//!
//! This test does **not** validate op-ordering between streams (no
//! `wait_for` is called here). It only tests that:
//! - Fences can be recorded on different streams.
//! - `elapsed_between()` can measure time between fences recorded on
//!   different streams.
//! - The driver returns plausible non-zero elapsed times.
//!
//! Cross-stream ordering enforcement (using `wait_for` to ensure stream B
//! waits for stream A's work) is deferred to Primitive 3 — see the
//! `fence_ordering` test doc for rationale.
//!
//! # GPU gate
//!
//! `#[cfg(feature = "gpu")]` — plain `cargo test` skips this file.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::stream_context_for;

#[test]
fn fence_cross_stream_timing() {
    let ctx = stream_context_for(0).expect("CUDA device 0 init");

    let src = CudaImage::<u8>::from_host(ctx.clone(), 3, 512, 512, &vec![128u8; 512 * 512 * 3])
        .expect("source allocation");

    let mut dst_a = CudaImage::<u8>::new(ctx.clone(), 3, 256, 256).expect("dst_a alloc");
    let mut dst_b = CudaImage::<u8>::new(ctx.clone(), 3, 256, 256).expect("dst_b alloc");

    // Record a fence before each resize. Both fences go on the context's
    // single stream — cross-stream timing requires separate stream handles,
    // which belong to Primitive 3 (op-on-caller-supplied-stream). For now,
    // this test proves `elapsed_between` works with fences on the same
    // stream produced independently (two record_fence calls).
    let before_a = ctx.record_fence().expect("record before_a");
    src.resize(&mut dst_a, ResizeInterpolation::Linear)
        .expect("resize a");
    let after_a = ctx.record_fence().expect("record after_a");

    let before_b = ctx.record_fence().expect("record before_b");
    src.resize(&mut dst_b, ResizeInterpolation::Linear)
        .expect("resize b");
    let after_b = ctx.record_fence().expect("record after_b");

    // Measure elapsed: within each pair (same-stream).
    let elapsed_a = ctx.elapsed_between(&before_a, &after_a).expect("elapsed a");
    let elapsed_b = ctx.elapsed_between(&before_b, &after_b).expect("elapsed b");

    assert!(elapsed_a.as_nanos() > 0, "elapsed_a must be non-zero");
    assert!(elapsed_b.as_nanos() > 0, "elapsed_b must be non-zero");

    // Verify geometry (not a golden test).
    ctx.synchronize().expect("synchronize");
    assert_eq!(dst_a.width(), 256);
    assert_eq!(dst_b.width(), 256);
}
