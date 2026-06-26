//! Single-pipeline fence plumbing test (F8.2).
//!
//! Creates two fences on the same stream, records them around a Resize
//! operation, and measures elapsed time. This proves that `record_fence()`,
//! `wait_for()`, and `elapsed_between()` are wired correctly (CUDA events
//! are created, recorded, and queried without driver errors).
//!
//! **Limitation:** `wait_for()` in a single-pipeline test is redundant —
//! work on one stream is already FIFO-ordered. This test proves plumbing is
//! sound but cannot prove `wait_for()` enforces cross-stream ordering.
//! Cross-stream op-ordering validation is deferred to Primitive 3 (F8.2+),
//! which must be shaped by multiple real use cases (e.g. MTCNN fan-out,
//! supertextures) before committing to a test structure.
//!
//! # GPU gate
//!
//! `#[cfg(feature = "gpu")]` — plain `cargo test` skips this file.

#![cfg(feature = "gpu")]

use npp_rs::image::CudaImage;
use npp_rs::imageops::{Resize, ResizeInterpolation};
use npp_rs::stream::Fence;
use npp_rs::stream_context_for;

#[test]
fn fence_plumbing_same_stream() {
    let ctx = stream_context_for(0).expect("CUDA device 0 init");

    let src = CudaImage::<u8>::from_host(
        ctx.clone(),
        3,  // channels
        64, // width
        64, // height
        &vec![42u8; 64 * 64 * 3],
    )
    .expect("source allocation");

    let mut dst = CudaImage::<u8>::new(ctx.clone(), 3, 32, 32).expect("dst allocation");

    // Record a fence before the NPP op.
    let before: Fence = ctx.record_fence().expect("record before");

    // Perform an NPP operation.
    src.resize(&mut dst, ResizeInterpolation::NearestNeighbor)
        .expect("resize");

    // Record a fence after the NPP op.
    let after: Fence = ctx.record_fence().expect("record after");

    // Wait for the before-fence on the same stream (redundant but proves
    // wait_for is wired).
    ctx.wait_for(&before);

    // Measure elapsed time between the two fences (same-stream).
    let elapsed = ctx.elapsed_between(&before, &after).expect("elapsed");
    assert!(
        elapsed.as_nanos() > 0,
        "elapsed time must be non-zero for a real NPP operation"
    );

    // Synchronise to ensure the resize completed before we check geometry.
    ctx.synchronize().expect("synchronize");

    // Geometry check (not a golden test — just verifying the pipeline ran).
    assert_eq!(dst.width(), 32);
    assert_eq!(dst.height(), 32);
    assert_eq!(dst.channels(), 3);
}
